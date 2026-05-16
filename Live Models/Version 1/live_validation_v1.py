import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from entsoe import EntsoePandasClient
from dotenv import load_dotenv
import pytz
from alerts import send_pipeline_alert

load_dotenv()

DUTCH_TZ = "Europe/Amsterdam"
LIVE_DIR = "voltcast_ipi_live_v1"
REPORT_DIR = "voltcast_ipi_reports_v1"
os.makedirs(REPORT_DIR, exist_ok=True)

def calculate_metrics():
    print("Starting VoltCast Daily Validation...")
    
    # 1. Define the Target Date (We validate YESTERDAY's delivery)
    tz = pytz.timezone(DUTCH_TZ)
    yesterday = datetime.now(tz) - timedelta(days=1)
    '''target_date_str = yesterday.strftime('%Y%m%d')
    target_date_iso = yesterday.strftime('%Y-%m-%d')'''
    target_date_str = "20260517"
    target_date_iso = "2026-05-17"

    print(f"Target Delivery Date: {target_date_iso}")

    # 2. Load the Predictions from your LIVE directory JSON
    json_path = os.path.join(LIVE_DIR, f"predictions_{target_date_str}.json")
    
    if not os.path.exists(json_path):
        msg = f"Could not find prediction JSON for {target_date_str}. Skipping validation."
        print(f"[ERROR] {msg}")
        send_pipeline_alert("VALIDATION", "Validation Aborted: Missing JSON", msg)
        return

    with open(json_path, 'r') as f:
        report_data = json.load(f)
    
    # Extract hourly predictions from the specific JSON structure
    try:
        pred_data = report_data['voltcast_ipi']['ptus']
        df_pred = pd.DataFrame(pred_data)
        # Extract the hour from the local timestamp
        df_pred['timestamp_local'] = pd.to_datetime(df_pred['timestamp_local'])
        df_pred['hour'] = df_pred['timestamp_local'].dt.hour
        pred_col = 'forecast_p50'
    except KeyError as e:
        msg = f"Unexpected JSON structure. Missing key: {e}"
        print(f"[ERROR] {msg}")
        send_pipeline_alert("VALIDATION", "Validation Aborted: JSON Structure Error", msg)
        return
    
    # 3. Fetch the Actual EPEX Prices from ENTSO-E
    api_key = os.environ.get('VOLTCAST_ENTSOE_KEY')
    if not api_key:
        print("[ERROR] ENTSOE_TOKEN not found in .env")
        send_pipeline_alert("VALIDATION", "Validation Aborted: Missing API Key", "ENTSOE_TOKEN missing.")
        return
        
    client = EntsoePandasClient(api_key=api_key)
    
    # ENTSO-E requires timezone-aware timestamps for the query
    start = pd.Timestamp(target_date_iso, tz=DUTCH_TZ)
    end = start + pd.Timedelta(days=1)
    
    print("Fetching actual Day-Ahead prices from ENTSO-E (NL)...")
    try:
        actual_series = client.query_day_ahead_prices('NL', start=start, end=end)
        
        # ── THE FIX: Force strict Dutch Timezone before extracting the hour ──
        if actual_series.index.tz is None:
            actual_series.index = actual_series.index.tz_localize('UTC').tz_convert(DUTCH_TZ)
        else:
            actual_series.index = actual_series.index.tz_convert(DUTCH_TZ)
            
        # Isolate just the 24 hours of the target day
        actual_series = actual_series[actual_series.index.date == start.date()]
        df_actual = actual_series.to_frame(name='actual_price')
        df_actual['hour'] = df_actual.index.hour
        
    except Exception as e:
        msg = f"Failed to fetch actual prices from ENTSO-E: {e}"
        print(f"[ERROR] {msg}")
        send_pipeline_alert("VALIDATION", "Validation Aborted: ENTSO-E API Error", msg)
        return

# 4. Merge Predictions and Actuals
    # ── THE FIX: Force both datasets into exactly 24 hourly averages before merging ──
    df_pred_hourly = df_pred.groupby('hour', as_index=False)[pred_col].mean()
    df_actual_hourly = df_actual.groupby('hour', as_index=False)['actual_price'].mean()
    
    df = pd.merge(df_pred_hourly, df_actual_hourly, on='hour', how='inner')
    
    print("\n[DEBUG] Clean Hourly Alignment Check (First 8 Hours):")
    print(df[['hour', pred_col, 'actual_price']].head(8))
    print("-" * 40)
    
    if len(df) == 0:
        msg = "Merge failed: Could not align prediction hours with actual hours."
        print(f"[ERROR] {msg}")
        send_pipeline_alert("VALIDATION", "Validation Aborted: Data Alignment Error", msg)
        return
    
    # 5. Calculate Advanced Metrics
    df['error'] = df[pred_col] - df['actual_price']
    df['abs_error'] = df['error'].abs()
    
    mae = df['abs_error'].mean()
    mbe = df['error'].mean() 
    
    spike_threshold = 100
    df_spikes = df[df['actual_price'] > spike_threshold]
    spike_mae = df_spikes['abs_error'].mean() if not df_spikes.empty else 0.0
    
    actual_top_4 = df.nlargest(4, 'actual_price')['hour'].tolist()
    pred_top_4 = df.nlargest(4, pred_col)['hour'].tolist()
    hits = len(set(actual_top_4).intersection(set(pred_top_4)))
    buy_avoid_precision = (hits / 4.0) * 100
    
    df['actual_diff'] = df['actual_price'].diff()
    df['pred_diff'] = df[pred_col].diff()
    dir_acc = (np.sign(df['actual_diff']) == np.sign(df['pred_diff'])).mean() * 100

    # 6. Save to CSV Ledger
    ledger_path = os.path.join(REPORT_DIR, "performance_ledger.csv")
    new_row = pd.DataFrame([{
        'delivery_date': target_date_iso,
        'mae': round(mae, 2),
        'mbe': round(mbe, 2),
        'spike_mae': round(spike_mae, 2),
        'directional_acc_pct': round(dir_acc, 1),
        'peak_precision_pct': round(buy_avoid_precision, 1)
    }])
    
    if os.path.exists(ledger_path):
        new_row.to_csv(ledger_path, mode='a', header=False, index=False)
    else:
        new_row.to_csv(ledger_path, index=False)
        
    print(f"Ledger updated: {ledger_path}")

    # 7. Send the Email Alert
    email_body = (
        f"VoltCast Daily Performance Validation for Delivery Date: {target_date_iso}\n\n"
        f"Overall MAE:           {mae:.2f} EUR/MWh\n"
        f"Mean Bias Error (MBE): {mbe:.2f} EUR/MWh\n"
        f"Spike MAE (>€100):     {spike_mae:.2f} EUR/MWh\n"
        f"Directional Accuracy:  {dir_acc:.1f}%\n"
        f"Peak Precision:        {buy_avoid_precision:.1f}% (Top 4 Hour Identification)\n\n"
        f"The updated performance ledger CSV is attached."
    )
    
    send_pipeline_alert(
        stage="VALIDATION",
        subject=f"Validation OK: {target_date_iso} (MAE: {mae:.2f})",
        body=email_body,
        attachments=[ledger_path]
    )
    print("Validation email successfully dispatched.")

if __name__ == "__main__":
    calculate_metrics()