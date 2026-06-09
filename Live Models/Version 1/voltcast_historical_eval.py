"""
================================================================================
 VOLTCAST IPI — HISTORICAL PERFORMANCE EVALUATOR
 Purpose: Backtest EPEX DA prices vs VoltCast predictions (May 17 -> Today)
================================================================================
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from entsoe import EntsoePandasClient
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
DUTCH_TZ = "Europe/Amsterdam"
MORNING_DIR = "voltcast_ipi_live_v1"
START_DATE = "2026-05-17"
REPORT_DIR = "voltcast_ipi_reports_v1"
os.makedirs(REPORT_DIR, exist_ok=True)

def load_historical_predictions(start_date_str, end_date_str):
    """Sweeps the live directory for stored prediction JSONs."""
    print(f"🔍 Sweeping directory {MORNING_DIR} for predictions...")
    
    date_range = pd.date_range(start=start_date_str, end=end_date_str, freq='D')
    all_preds = []

    for target_date in date_range:
        date_str = target_date.strftime('%Y%m%d')
        json_path = os.path.join(MORNING_DIR, f"predictions_{date_str}.json")
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                report_data = json.load(f)
            try:
                df = pd.DataFrame(report_data['voltcast_ipi']['ptus'])
                df['timestamp_local'] = pd.to_datetime(df['timestamp_local'])
                df['delivery_date'] = df['timestamp_local'].dt.date
                df['hour'] = df['timestamp_local'].dt.hour
                
                # Use corrected p50 if available
                target_col = 'forecast_p50_corr' if 'forecast_p50_corr' in df.columns else 'forecast_p50'
                
                df_hourly = df.groupby(['delivery_date', 'hour'], as_index=False)[target_col].mean()
                df_hourly.rename(columns={target_col: 'predicted_price'}, inplace=True)
                all_preds.append(df_hourly)
            except KeyError:
                print(f"  ⚠️ Malformed JSON for {date_str}")
        else:
            print(f"  ⚠️ Missing prediction file for {date_str}")

    if not all_preds:
        return pd.DataFrame()
        
    return pd.concat(all_preds, ignore_index=True)

def fetch_historical_actuals(start_date_str, end_date_str):
    """Fetches the ground-truth EPEX DA prices from ENTSO-E."""
    print(f"📡 Fetching actual EPEX prices from ENTSO-E...")
    client = EntsoePandasClient(api_key=os.environ.get('VOLTCAST_ENTSOE_KEY'))
    
    start_ts = pd.Timestamp(start_date_str, tz=DUTCH_TZ)
    # Add 1 day to ensure we capture the end date fully
    end_ts = pd.Timestamp(end_date_str, tz=DUTCH_TZ) + pd.Timedelta(days=1)
    
    try:
        actuals = client.query_day_ahead_prices('NL', start=start_ts, end=end_ts)
        # Ensure timezone consistency
        if actuals.index.tz is None:
            actuals.index = actuals.index.tz_localize('UTC').tz_convert(DUTCH_TZ)
        else:
            actuals.index = actuals.index.tz_convert(DUTCH_TZ)
            
        df_actual = actuals.to_frame(name='actual_price')
        df_actual['delivery_date'] = df_actual.index.date
        df_actual['hour'] = df_actual.index.hour
        
        # Aggregate 15-min PTUs to Hourly
        return df_actual.groupby(['delivery_date', 'hour'], as_index=False)['actual_price'].mean()
    except Exception as e:
        print(f"❌ ENTSO-E Fetch Error: {e}")
        return pd.DataFrame()

def run_historical_backtest():
    tz = pytz.timezone(DUTCH_TZ)
    today_str = datetime.now(tz).strftime('%Y-%m-%d')
    
    print("=" * 60)
    print(f" VOLTCAST HISTORICAL BACKTEST: {START_DATE} to {today_str}")
    print("=" * 60)

    # 1. Gather Data
    df_preds = load_historical_predictions(START_DATE, today_str)
    if df_preds.empty:
        print("❌ No prediction files found to backtest against.")
        return

    df_actuals = fetch_historical_actuals(START_DATE, today_str)
    if df_actuals.empty:
        return

    # 2. Merge and Compute Errors
    df_eval = pd.merge(df_preds, df_actuals, on=['delivery_date', 'hour'], how='inner')
    df_eval['absolute_error'] = (df_eval['predicted_price'] - df_eval['actual_price']).abs()
    df_eval['bias'] = df_eval['predicted_price'] - df_eval['actual_price']
    
    # 3. Aggregate Daily Metrics
    daily_metrics = df_eval.groupby('delivery_date').agg(
        MAE=('absolute_error', 'mean'),
        MBE=('bias', 'mean'),
        Max_Error=('absolute_error', 'max'),
        Avg_Actual_Price=('actual_price', 'mean')
    ).reset_index()

    # 4. Save and Display
    eval_path = os.path.join(REPORT_DIR, f"historical_eval_{START_DATE}_to_today.csv")
    daily_metrics.to_csv(eval_path, index=False)
    
    overall_mae = df_eval['absolute_error'].mean()
    overall_mbe = df_eval['bias'].mean()
    
    print("\n--- DAILY PERFORMANCE SUMMARY ---")
    print(daily_metrics.to_string(index=False, float_format="%.2f"))
    print("-" * 60)
    print(f"🏆 OVERALL MAE : €{overall_mae:.2f}")
    print(f"📉 OVERALL MBE : €{overall_mbe:.2f}")
    print(f"💾 Report saved to: {eval_path}")

if __name__ == "__main__":
    run_historical_backtest()