import os
import sys
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
MORNING_DIR = "voltcast_ipi_live_v1"
EVENING_DIR = "voltcast_ipi_live_v1_evening"
REPORT_DIR = "voltcast_ipi_reports_v1"
os.makedirs(REPORT_DIR, exist_ok=True)

def load_predictions(directory, date_str):
    """Helper function to load and format predictions from a specific sandbox."""
    json_path = os.path.join(directory, f"predictions_{date_str}.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r') as f:
        report_data = json.load(f)
    try:
        df = pd.DataFrame(report_data['voltcast_ipi']['ptus'])
        df['timestamp_local'] = pd.to_datetime(df['timestamp_local'])
        df['hour'] = df['timestamp_local'].dt.hour
        # Aggregate to strictly 24 hours just like before
        return df.groupby('hour', as_index=False)['forecast_p50'].mean()
    except KeyError:
        return None

def calculate_metrics():
    print("Starting VoltCast Dual-Basis Validation...")
    tz = pytz.timezone(DUTCH_TZ)
    yesterday = datetime.now(tz) - timedelta(days=1)
    target_date_str = yesterday.strftime('%Y%m%d')
    target_date_iso = yesterday.strftime('%Y-%m-%d')
    
    # 1. Load Morning and Evening Predictions
    df_morning = load_predictions(MORNING_DIR, target_date_str)
    df_evening = load_predictions(EVENING_DIR, target_date_str)

    if df_morning is None:
        send_pipeline_alert("VALIDATION", "Validation Aborted", f"Missing Morning JSON for {target_date_str}")
        return

    # Rename columns to keep them distinct
    df_morning.rename(columns={'forecast_p50': 'pred_morning'}, inplace=True)
    if df_evening is not None:
        df_evening.rename(columns={'forecast_p50': 'pred_evening'}, inplace=True)

    # 2. Fetch Actuals
    api_key = os.environ.get('ENTSOE_TOKEN')
    client = EntsoePandasClient(api_key=api_key)
    start = pd.Timestamp(target_date_iso, tz=DUTCH_TZ)
    end = start + pd.Timedelta(days=1)
    
    try:
        actual_series = client.query_day_ahead_prices('NL', start=start, end=end)
        if actual_series.index.tz is None:
            actual_series.index = actual_series.index.tz_localize('UTC').tz_convert(DUTCH_TZ)
        else:
            actual_series.index = actual_series.index.tz_convert(DUTCH_TZ)
        actual_series = actual_series[actual_series.index.date == start.date()]
        df_actual = actual_series.to_frame(name='actual_price')
        df_actual['hour'] = df_actual.index.hour
        df_actual_hourly = df_actual.groupby('hour', as_index=False)['actual_price'].mean()
    except Exception as e:
        send_pipeline_alert("VALIDATION", "Validation Aborted: ENTSO-E Error", str(e))
        return

    # 3. Merge Everything
    df = pd.merge(df_morning, df_actual_hourly, on='hour', how='inner')
    if df_evening is not None:
        df = pd.merge(df, df_evening, on='hour', how='left')

    # 4. Calculate Diagnostics
    df['err_morning'] = df['pred_morning'] - df['actual_price']
    mae_morn = df['err_morning'].abs().mean()
    mbe_morn = df['err_morning'].mean()
    
    dir_acc_morn = (np.sign(df['actual_price'].diff()) == np.sign(df['pred_morning'].diff())).mean() * 100

    mae_eve, mbe_eve, dir_acc_eve = None, None, None
    if df_evening is not None and 'pred_evening' in df.columns:
        df['err_evening'] = df['pred_evening'] - df['actual_price']
        mae_eve = df['err_evening'].abs().mean()
        mbe_eve = df['err_evening'].mean()
        dir_acc_eve = (np.sign(df['actual_price'].diff()) == np.sign(df['pred_evening'].diff())).mean() * 100

    # 5. Log and Alert
    print(f"\n--- PERFORMANCE: {target_date_iso} ---")
    print(f"Morning MAE (Production): {mae_morn:.2f}  |  MBE: {mbe_morn:.2f}  |  DirAcc: {dir_acc_morn:.1f}%")
    if mae_eve is not None:
        print(f"Evening MAE (Diagnostic): {mae_eve:.2f}  |  MBE: {mbe_eve:.2f}  |  DirAcc: {dir_acc_eve:.1f}%")
        data_drift = mae_morn - mae_eve
        print(f"Data Drift (Morning MAE - Evening MAE): {data_drift:.2f} EUR")

    # Save to CSV
    ledger_path = os.path.join(REPORT_DIR, "performance_ledger.csv")
    new_row = pd.DataFrame([{
        'delivery_date': target_date_iso,
        'mae_morning': round(mae_morn, 2),
        'mae_evening': round(mae_eve, 2) if mae_eve else np.nan,
        'mbe_morning': round(mbe_morn, 2),
        'mbe_evening': round(mbe_eve, 2) if mbe_eve else np.nan,
        'dir_acc_morn': round(dir_acc_morn, 1),
        'dir_acc_eve': round(dir_acc_eve, 1) if dir_acc_eve else np.nan
    }])
    new_row.to_csv(ledger_path, mode='a', header=not os.path.exists(ledger_path), index=False)

    email_body = (
        f"VoltCast Dual-Basis Validation: {target_date_iso}\n\n"
        f"--- PRODUCTION RUN (10:10 AM) ---\n"
        f"MAE: {mae_morn:.2f} EUR/MWh\n"
        f"MBE: {mbe_morn:.2f} EUR/MWh\n"
        f"Directional Accuracy: {dir_acc_morn:.1f}%\n\n"
    )
    if mae_eve is not None:
        email_body += (
            f"--- DIAGNOSTIC RUN (18:00 PM) ---\n"
            f"MAE: {mae_eve:.2f} EUR/MWh\n"
            f"MBE: {mbe_eve:.2f} EUR/MWh\n"
            f"Directional Accuracy: {dir_acc_eve:.1f}%\n\n"
            f"DATA DRIFT IMPACT: {mae_morn - mae_eve:.2f} EUR/MWh\n"
            f"(Positive number = afternoon weather updates improved accuracy)"
        )

    send_pipeline_alert("VALIDATION", f"Validation OK: {target_date_iso} (MAE: {mae_morn:.2f})", email_body, [ledger_path])

if __name__ == "__main__":
    calculate_metrics()