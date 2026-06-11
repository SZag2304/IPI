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
MORNING_DIR = "voltcast_ipi_live_v2"
EVENING_DIR = "voltcast_ipi_live_v2_evening"
REPORT_DIR = "voltcast_ipi_reports_v2"
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
        # Check if this JSON has the new dual-tracking variables
        if 'forecast_p50_raw' in df.columns:
            return df.groupby('hour', as_index=False)[['forecast_p50_raw', 'forecast_p50']].mean()
        else:
            # Fallback for older JSONs
            df_fallback = df.groupby('hour', as_index=False)['forecast_p50'].mean()
            df_fallback['forecast_p50_raw'] = df_fallback['forecast_p50'] 
            return df_fallback
    except KeyError:
        return None

def calculate_metrics(target_date: str = None):
    """target_date: optional 'YYYY-MM-DD' (v2: enables backfilling missed days);
    defaults to yesterday — the last settled delivery day."""
    print("Starting VoltCast v2 Challenger Validation...")
    tz = pytz.timezone(DUTCH_TZ)
    target = (datetime.strptime(target_date, "%Y-%m-%d").replace(tzinfo=tz)
              if target_date else datetime.now(tz) - timedelta(days=1))
    target_date_str = target.strftime('%Y%m%d')
    target_date_iso = target.strftime('%Y-%m-%d')
    
    # 1. Load Morning and Evening Predictions
    df_morning = load_predictions(MORNING_DIR, target_date_str)
    df_evening = load_predictions(EVENING_DIR, target_date_str)

    if df_morning is None:
        send_pipeline_alert("VALIDATION", "Validation Aborted", f"Missing Morning JSON for {target_date_str}")
        return

    # Rename columns to keep them distinct
    df_morning.rename(columns={'forecast_p50_raw': 'pred_morn_raw', 'forecast_p50': 'pred_morn_corr'}, inplace=True)
    if df_evening is not None:
        df_evening.rename(columns={'forecast_p50_raw': 'pred_eve_raw', 'forecast_p50': 'pred_eve_corr'}, inplace=True)

    # 2. Fetch Actuals
    api_key = os.environ.get('VOLTCAST_ENTSOE_KEY')
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

    # 4. Calculate Diagnostics: MORNING
    # Morning Raw
    df['err_morn_raw'] = df['pred_morn_raw'] - df['actual_price']
    mae_morn_raw = df['err_morn_raw'].abs().mean()
    mbe_morn_raw = df['err_morn_raw'].mean()
    
    # Morning Corrected
    df['err_morn_corr'] = df['pred_morn_corr'] - df['actual_price']
    mae_morn_corr = df['err_morn_corr'].abs().mean()
    mbe_morn_corr = df['err_morn_corr'].mean()

    # Morning Shape (Constant scalar shift doesn't change shape, calculate once)
    dir_acc_morn = (np.sign(df['actual_price'].diff()) == np.sign(df['pred_morn_corr'].diff())).mean() * 100
    actual_top_4 = df.nlargest(4, 'actual_price')['hour'].tolist()
    pred_top_4_morn = df.nlargest(4, 'pred_morn_corr')['hour'].tolist()
    hits_morn = len(set(actual_top_4).intersection(set(pred_top_4_morn)))
    peak_prec_morn = (hits_morn / 4.0) * 100

    # Morning Spikes
    spike_threshold = 100
    df_spikes = df[df['actual_price'] > spike_threshold]
    spike_mae_morn_raw = df_spikes['err_morn_raw'].abs().mean() if not df_spikes.empty else 0.0
    spike_mae_morn_corr = df_spikes['err_morn_corr'].abs().mean() if not df_spikes.empty else 0.0

# 5. Calculate Diagnostics: EVENING
    mae_eve_raw, mbe_eve_raw, spike_mae_eve_raw = None, None, None
    mae_eve_corr, mbe_eve_corr, spike_mae_eve_corr = None, None, None
    dir_acc_eve, peak_prec_eve = None, None
    
    if df_evening is not None and 'pred_eve_corr' in df.columns:
        # Evening Raw
        df['err_eve_raw'] = df['pred_eve_raw'] - df['actual_price']
        mae_eve_raw = df['err_eve_raw'].abs().mean()
        mbe_eve_raw = df['err_eve_raw'].mean()

        # Evening Corrected
        df['err_eve_corr'] = df['pred_eve_corr'] - df['actual_price']
        mae_eve_corr = df['err_eve_corr'].abs().mean()
        mbe_eve_corr = df['err_eve_corr'].mean()

        # Evening Shape
        dir_acc_eve = (np.sign(df['actual_price'].diff()) == np.sign(df['pred_eve_corr'].diff())).mean() * 100
        pred_top_4_eve = df.nlargest(4, 'pred_eve_corr')['hour'].tolist()
        hits_eve = len(set(actual_top_4).intersection(set(pred_top_4_eve)))
        peak_prec_eve = (hits_eve / 4.0) * 100

        # ── THE FIX: Refresh the snapshot to include the new evening error columns ──
        df_spikes = df[df['actual_price'] > spike_threshold]

        # Evening Spikes
        spike_mae_eve_raw = df_spikes['err_eve_raw'].abs().mean() if not df_spikes.empty else 0.0
        spike_mae_eve_corr = df_spikes['err_eve_corr'].abs().mean() if not df_spikes.empty else 0.0

    # 6. Console Output
    print(f"\n--- PERFORMANCE: {target_date_iso} ---")
    print("\n[ PRODUCTION RUN - 09:15 AM ]")
    print(f"Shape -> DirAcc: {dir_acc_morn:.1f}% | Peak Precision: {peak_prec_morn:.1f}%")
    print(f" RAW        -> MAE: {mae_morn_raw:.2f} | MBE: {mbe_morn_raw:.2f} | Spike MAE: {spike_mae_morn_raw:.2f}")
    print(f" CORRECTED  -> MAE: {mae_morn_corr:.2f} | MBE: {mbe_morn_corr:.2f} | Spike MAE: {spike_mae_morn_corr:.2f}")

    if df_evening is not None:
        print("\n[ DIAGNOSTIC RUN - 18:30 PM ]")
        print(f"Shape -> DirAcc: {dir_acc_eve:.1f}% | Peak Precision: {peak_prec_eve:.1f}%")
        print(f" RAW        -> MAE: {mae_eve_raw:.2f} | MBE: {mbe_eve_raw:.2f} | Spike MAE: {spike_mae_eve_raw:.2f}")
        print(f" CORRECTED  -> MAE: {mae_eve_corr:.2f} | MBE: {mbe_eve_corr:.2f} | Spike MAE: {spike_mae_eve_corr:.2f}")

    # 7. Save to CSV
    ledger_path = os.path.join(REPORT_DIR, "performance_ledger.csv")
    new_row = pd.DataFrame([{
        'delivery_date': target_date_iso,
        'dir_acc_morn': round(dir_acc_morn, 1),
        'peak_prec_morn': round(peak_prec_morn, 1),
        'mae_morn_raw': round(mae_morn_raw, 2),
        'mbe_morn_raw': round(mbe_morn_raw, 2),
        'spike_morn_raw': round(spike_mae_morn_raw, 2),
        'mae_morn_corr': round(mae_morn_corr, 2),
        'mbe_morn_corr': round(mbe_morn_corr, 2),
        'spike_morn_corr': round(spike_mae_morn_corr, 2),
        'dir_acc_eve': round(dir_acc_eve, 1) if df_evening is not None else np.nan,
        'peak_prec_eve': round(peak_prec_eve, 1) if df_evening is not None else np.nan,
        'mae_eve_raw': round(mae_eve_raw, 2) if df_evening is not None else np.nan,
        'mbe_eve_raw': round(mbe_eve_raw, 2) if df_evening is not None else np.nan,
        'mae_eve_corr': round(mae_eve_corr, 2) if df_evening is not None else np.nan,
        'mbe_eve_corr': round(mbe_eve_corr, 2) if df_evening is not None else np.nan
    }])
    new_row.to_csv(ledger_path, mode='a', header=not os.path.exists(ledger_path), index=False)

    # 8. Email Alert Formatting
    email_body = (
        f"VoltCast 2x2 Master Validation: {target_date_iso}\n\n"
        f"=== PRODUCTION RUN (09:15 AM) ===\n"
        f"Directional Accuracy: {dir_acc_morn:.1f}%\n"
        f"Peak Precision:       {peak_prec_morn:.1f}%\n"
        f"• RAW MODEL:\n"
        f"  MAE: {mae_morn_raw:.2f} | MBE: {mbe_morn_raw:.2f} | Spike MAE: {spike_mae_morn_raw:.2f}\n"
        f"• CORRECTED MODEL (adaptive intercept):\n"
        f"  MAE: {mae_morn_corr:.2f} | MBE: {mbe_morn_corr:.2f} | Spike MAE: {spike_mae_morn_corr:.2f}\n\n"
    )

    if df_evening is not None:
        email_body += (
            f"=== DIAGNOSTIC RUN (18:30 PM) ===\n"
            f"Directional Accuracy: {dir_acc_eve:.1f}%\n"
            f"Peak Precision:       {peak_prec_eve:.1f}%\n"
            f"• RAW MODEL:\n"
            f"  MAE: {mae_eve_raw:.2f} | MBE: {mbe_eve_raw:.2f} | Spike MAE: {spike_mae_eve_raw:.2f}\n"
            f"• CORRECTED MODEL (adaptive intercept):\n"
            f"  MAE: {mae_eve_corr:.2f} | MBE: {mbe_eve_corr:.2f} | Spike MAE: {spike_mae_eve_corr:.2f}\n"
        )

    send_pipeline_alert(
        "VALIDATION", 
        f"Validation OK: {target_date_iso} (Corr. MAE: {mae_morn_corr:.2f})", 
        email_body, 
        [ledger_path]
    )

if __name__ == "__main__":
    calculate_metrics(sys.argv[1] if len(sys.argv) > 1 else None)