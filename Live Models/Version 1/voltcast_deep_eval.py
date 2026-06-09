"""
================================================================================
 VOLTCAST IPI — DEEP HOURLY PERFORMANCE EVALUATOR
 Purpose: Backtest EPEX DA prices vs VoltCast with Weather & Fuel Context
================================================================================
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
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
                
                # Prioritize corrected p50 if available
                target_col = 'forecast_p50_corr' if 'forecast_p50_corr' in df.columns else 'forecast_p50'
                
                df_hourly = df.groupby(['delivery_date', 'hour'], as_index=False)[target_col].mean()
                df_hourly.rename(columns={target_col: 'predicted_price'}, inplace=True)
                all_preds.append(df_hourly)
            except KeyError:
                print(f"  ⚠️ Malformed JSON for {date_str}")

    return pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

def fetch_historical_actuals(start_date_str, end_date_str):
    print(f"📡 Fetching actual EPEX prices from ENTSO-E...")
    client = EntsoePandasClient(api_key=os.environ.get('VOLTCAST_ENTSOE_KEY'))
    
    start_ts = pd.Timestamp(start_date_str, tz=DUTCH_TZ)
    end_ts = pd.Timestamp(end_date_str, tz=DUTCH_TZ) + pd.Timedelta(days=1)
    
    try:
        actuals = client.query_day_ahead_prices('NL', start=start_ts, end=end_ts)
        if actuals.index.tz is None:
            actuals.index = actuals.index.tz_localize('UTC').tz_convert(DUTCH_TZ)
        else:
            actuals.index = actuals.index.tz_convert(DUTCH_TZ)
            
        df_actual = actuals.to_frame(name='actual_price')
        df_actual['delivery_date'] = df_actual.index.date
        df_actual['hour'] = df_actual.index.hour
        
        return df_actual.groupby(['delivery_date', 'hour'], as_index=False)['actual_price'].mean()
    except Exception as e:
        print(f"❌ ENTSO-E Fetch Error: {e}")
        return pd.DataFrame()

def load_historical_context(start_date_str, end_date_str):
    """Pulls Weather, Load, and Gas from the saved Live Master parquets."""
    print(f"🌦️ Sweeping directory {MORNING_DIR} for physics and fuel context...")
    date_range = pd.date_range(start=start_date_str, end=end_date_str, freq='D')
    all_context = []

    for target_date in date_range:
        # The fetch runs on D-1 to forecast for D
        run_date = target_date - pd.Timedelta(days=1)
        date_str = run_date.strftime('%Y%m%d')
        parquet_path = os.path.join(MORNING_DIR, f"live_master_{date_str}.parquet")
        
        if os.path.exists(parquet_path):
            try:
                df_master = pd.read_parquet(parquet_path)
                
                delivery_start = pd.Timestamp(target_date, tz=DUTCH_TZ).tz_convert("UTC")
                delivery_end = delivery_start + pd.DateOffset(days=1)
                df_d1 = df_master[(df_master.index >= delivery_start) & (df_master.index < delivery_end)].copy()
                
                if df_d1.empty: continue

                # Map raw columns to clean names
                cols_map = {
                    'TTF_Gas_EURMWh': 'TTF_Gas',
                    'EUA_Carbon_EUR': 'EUA_Carbon',
                    'NL_TSO_Load_Forecast_MW': 'TSO_Load_FC',
                    'temperature_2m_Amsterdam': 'Temp_Ams',
                    'shortwave_radiation_Amsterdam': 'Solar_Rad_Ams',
                    'wind_speed_100m_Amsterdam': 'Wind_Spd_Ams',
                    'NL_Thermal_Outage_MW': 'Thermal_Outages'
                }
                
                extract = pd.DataFrame(index=df_d1.index)
                for raw_col, clean_name in cols_map.items():
                    if raw_col in df_d1.columns:
                        extract[clean_name] = df_d1[raw_col]
                    else:
                        extract[clean_name] = np.nan
                        
                extract.index = extract.index.tz_convert(DUTCH_TZ)
                extract['delivery_date'] = extract.index.date
                extract['hour'] = extract.index.hour
                
                df_hourly = extract.groupby(['delivery_date', 'hour'], as_index=False).mean()
                all_context.append(df_hourly)
                
            except Exception as e:
                pass
                
    return pd.concat(all_context, ignore_index=True) if all_context else pd.DataFrame()

def run_deep_backtest():
    tz = pytz.timezone(DUTCH_TZ)
    today_str = datetime.now(tz).strftime('%Y-%m-%d')
    
    print("=" * 70)
    print(f" VOLTCAST DEEP HOURLY BACKTEST: {START_DATE} to {today_str}")
    print("=" * 70)

    df_preds = load_historical_predictions(START_DATE, today_str)
    df_actuals = fetch_historical_actuals(START_DATE, today_str)
    df_context = load_historical_context(START_DATE, today_str)

    if df_preds.empty or df_actuals.empty:
        print("❌ Missing predictions or actuals. Cannot proceed.")
        return

    # Merge core data
    df_eval = pd.merge(df_preds, df_actuals, on=['delivery_date', 'hour'], how='inner')
    
    # Merge physics context safely
    if not df_context.empty:
        df_eval = pd.merge(df_eval, df_context, on=['delivery_date', 'hour'], how='left')

    df_eval['error'] = df_eval['predicted_price'] - df_eval['actual_price']
    df_eval['abs_error'] = df_eval['error'].abs()
    
    csv_path = os.path.join(REPORT_DIR, f"hourly_deep_eval_{START_DATE}_to_{today_str}.csv")
    df_eval.to_csv(csv_path, index=False)

    # ----- ANALYTICS TERMINAL -----
    print(f"\n📊 HOURLY REPORT GENERATED ({len(df_eval)} hours)")
    print(f"💾 Saved to: {csv_path}\n")
    
    # 1. Split Performance by Market Regime
    spikes = df_eval[df_eval['actual_price'] >= 100]
    negatives = df_eval[df_eval['actual_price'] < 0]
    normal = df_eval[(df_eval['actual_price'] >= 0) & (df_eval['actual_price'] < 100)]
    
    print("--- PERFORMANCE BY MARKET REGIME ---")
    print(f"🔥 SPIKES (>= €100) : {len(spikes):>3} hrs | MAE: €{spikes['abs_error'].mean():.2f} | MBE: €{spikes['error'].mean():.2f}")
    print(f"❄️ NEGATIVE (< €0)  : {len(negatives):>3} hrs | MAE: €{negatives['abs_error'].mean():.2f} | MBE: €{negatives['error'].mean():.2f}")
    print(f"🟢 NORMAL (€0-100)  : {len(normal):>3} hrs | MAE: €{normal['abs_error'].mean():.2f} | MBE: €{normal['error'].mean():.2f}\n")

    # 2. Top 5 Worst Hours Breakdown
    print("--- 🚨 TOP 5 WORST MISSES (Investigation Required) ---")
    worst = df_eval.nlargest(5, 'abs_error')
    for _, row in worst.iterrows():
        dt = f"{row['delivery_date']} H{row['hour']:02d}"
        print(f"\n❌ {dt} | Actual: €{row['actual_price']:.1f} | Pred: €{row['predicted_price']:.1f} | Err: €{row['error']:.1f}")
        
        if not df_context.empty:
            gas = row.get('TTF_Gas', np.nan)
            wind = row.get('Wind_Spd_Ams', np.nan)
            solar = row.get('Solar_Rad_Ams', np.nan)
            load = row.get('TSO_Load_FC', np.nan)
            print(f"   ↳ Context: Load {load:.0f}MW | Wind {wind:.1f}m/s | Solar {solar:.0f}W/m² | Gas €{gas:.1f}")

if __name__ == "__main__":
    run_deep_backtest()