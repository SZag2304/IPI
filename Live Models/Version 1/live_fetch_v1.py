"""
================================================================================
 VOLTCAST IPI — LIVE DATA FETCH PIPELINE
 Purpose: Fetch all signals for D+1 day-ahead price forecast
 Schedule: Daily cron at 10:10 CET (before EPEX DA auction at 12:00, after TSO load forecast at 10:00)
 
 Key differences from historical fetch:
   - Fetches a rolling 35-day window (sufficient for all lag features)
   - Validates D+1 forecast data completeness before exit
   - Sends structured alerts on failure
   - Applies TTF/EUA lag at fetch time for live inference
   - Writes a fetch_status.json for downstream scripts to check
================================================================================
"""

import os
import sys
import json
import time
import logging
import warnings
import requests
import smtplib
import traceback
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
from email.mime.text import MIMEText
from entsoe import EntsoePandasClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import config

# This single line finds your .env file and loads the variables into the system environment
load_dotenv()

warnings.filterwarnings("ignore")

# ================================================================================
# 0. CONFIGURATION
# ================================================================================

# --- API Keys from environment (never hardcode in source) ---
NED_API_KEY    = os.environ.get("VOLTCAST_NED_KEY")
ENTSOE_API_KEY = os.environ.get("VOLTCAST_ENTSOE_KEY")

# Fail immediately if keys are missing — do not silently produce empty data
if not NED_API_KEY or not ENTSOE_API_KEY:
    raise EnvironmentError(
        "VOLTCAST_NED_KEY and VOLTCAST_ENTSOE_KEY must be set as environment variables. "
        "Run: export VOLTCAST_NED_KEY='your_key' before executing this script."
    )

# --- Alert configuration (set via environment) ---
ALERT_EMAIL_TO   = os.environ.get("VOLTCAST_ALERT_EMAIL", "")
ALERT_EMAIL_FROM = os.environ.get("VOLTCAST_ALERT_FROM", "")
ALERT_SMTP_HOST  = os.environ.get("VOLTCAST_SMTP_HOST", "localhost")
ALERT_SMTP_PORT  = int(os.environ.get("VOLTCAST_SMTP_PORT", "25"))
ALERT_SMTP_USER  = os.environ.get("VOLTCAST_SMTP_USER", "")   # <--- ADDED THIS
ALERT_SMTP_PASS  = os.environ.get("VOLTCAST_SMTP_PASS", "")   # <--- ADDED THIS

# --- Timezone ---
DUTCH_TZ = "Europe/Amsterdam"

# --- Rolling window: 35 days back + 2 days forward (covers all lag_672 features + D+1) ---
# The model's longest lag is price_lag1344 = 14 days. 35 days provides safe margin.
LIVE_LOOKBACK_DAYS = 35
LIVE_FORWARD_DAYS  = 2   # D+1 forecast target day

now_dutch = pd.Timestamp.now(tz=DUTCH_TZ)
DELIVERY_DATE = (now_dutch + pd.DateOffset(days=1)).date()  # Tomorrow

START_LOCAL = pd.Timestamp(now_dutch.date() - pd.DateOffset(days=LIVE_LOOKBACK_DAYS), tz=DUTCH_TZ)
END_LOCAL   = pd.Timestamp(now_dutch.date() + pd.DateOffset(days=LIVE_FORWARD_DAYS),  tz=DUTCH_TZ)

START_UTC = START_LOCAL.tz_convert("UTC")
END_UTC   = END_LOCAL.tz_convert("UTC")

MASTER_INDEX = pd.date_range(
    start=START_UTC, end=END_UTC, freq="15min", inclusive="left"
)

# ── NEW: HOURLY MASTER INDEX FOR WEATHER FETCHES ──
MASTER_INDEX_HOURLY = pd.date_range(
    start=START_UTC, end=END_UTC, freq="h", inclusive="left"
)

# NOTE: END_LOCAL = today + 2 days.
# At 10:10 CET, D+1 (tomorrow) ENTSO-E forecasts are not available.
# The model targets D+1 delivery rows only (see validate_delivery_day_coverage).

# --- Directories ---
CACHE_DIR   = "voltcast_ipi_cache_v1"
LIVE_DIR    = "voltcast_ipi_live_v1"
LOG_DIR     = "voltcast_ipi_logs_v1"
for d in [CACHE_DIR, LIVE_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Output paths ---
RUN_DATE_STR  = now_dutch.strftime("%Y%m%d")
LIVE_PARQUET  = os.path.join(LIVE_DIR, f"live_master_{RUN_DATE_STR}.parquet")
STATUS_FILE   = os.path.join(LIVE_DIR, "fetch_status.json")
LOG_FILE      = os.path.join(LOG_DIR,  f"fetch_{RUN_DATE_STR}.log")

# ================================================================================
# 1. LOGGING
# ================================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
log = logging.getLogger("VoltCast.Live")

# ================================================================================
# 2. ALERT SYSTEM
# ================================================================================

class FetchStatus:
    """
    Tracks the status of each data source fetch.
    Written to fetch_status.json for downstream scripts to read.
    """
    def __init__(self):
        self.run_date     = RUN_DATE_STR
        self.delivery_date = str(DELIVERY_DATE)
        self.run_time     = now_dutch.isoformat()
        self.sources      = {}
        self.alerts       = []
        self.overall_pass = False

    def record(self, source: str, status: str, rows: int = 0,
               message: str = "", critical: bool = True):
        self.sources[source] = {
            "status":   str(status),          
            "rows":     int(rows),       # <--- Forces Python native integer
            "message":  str(message),    # <--- Forces Python native string
            "critical": bool(critical),  # <--- DESTROYS the numpy.bool_ error
        }
        if status == "FAILED" and critical:
            self.alerts.append(f"CRITICAL: {source} — {message}")
        elif status == "PARTIAL":
            self.alerts.append(f"WARNING: {source} — {message}")
        log.info(f"  [{status}] {source}: {rows:,} rows — {message}")

    def finalize(self):
        critical_failures = [
            s for s, v in self.sources.items()
            if v["status"] == "FAILED" and v["critical"]
        ]
        self.overall_pass = len(critical_failures) == 0
        return self.overall_pass

    def save(self):
        with open(STATUS_FILE, "w") as f:
            json.dump({
                "run_date":      self.run_date,
                "delivery_date": self.delivery_date,
                "run_time":      self.run_time,
                "overall_pass":  self.overall_pass,
                "sources":       self.sources,
                "alerts":        self.alerts,
            }, f, indent=2)
        log.info(f"  Status written: {STATUS_FILE}")


def send_alert(status: FetchStatus):
    """
    Sends email alert if critical failures exist.
    Requires VOLTCAST_ALERT_EMAIL environment variable.
    Falls back to log-only if SMTP is not configured.
    """
    if not status.alerts:
        return

    subject = f"[VoltCast IPI] Fetch {'FAILED' if not status.overall_pass else 'WARNING'} — {RUN_DATE_STR}"
    body    = f"VoltCast IPI Live Fetch Report\n"
    body   += f"Delivery date: {DELIVERY_DATE}\n"
    body   += f"Run time: {status.run_time}\n\n"
    body   += "ALERTS:\n" + "\n".join(f"  • {a}" for a in status.alerts)
    body   += "\n\nSource Summary:\n"
    for src, v in status.sources.items():
        icon = "✓" if v["status"] == "OK" else ("⚠" if v["status"] == "PARTIAL" else "✗")
        body += f"  {icon} {src}: {v['status']} ({v['rows']:,} rows)\n"

    log.warning(f"\n{'='*60}\nFETCH ALERTS:\n{body}\n{'='*60}")

    if not ALERT_EMAIL_TO:
        return  # No email configured — log-only mode

    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"]    = ALERT_EMAIL_FROM
        msg["To"]      = ALERT_EMAIL_TO

        with smtplib.SMTP(ALERT_SMTP_HOST, ALERT_SMTP_PORT) as smtp:
            if ALERT_SMTP_USER and ALERT_SMTP_PASS:
                smtp.starttls() # <--- CRITICAL: Encrypts the connection
                smtp.login(ALERT_SMTP_USER, ALERT_SMTP_PASS) # <--- CRITICAL: Logs into the bot account
            smtp.sendmail(ALERT_EMAIL_FROM, [ALERT_EMAIL_TO], msg.as_string())
            
        log.info(f"  Alert email sent to {ALERT_EMAIL_TO}")
    except Exception as e:
        log.error(f"  Alert email failed: {e}")


# ================================================================================
# 3. INTEGRITY VALIDATORS
# ================================================================================

# Sanity bounds for each signal — flag if outside these ranges
SIGNAL_BOUNDS = {
    "DA_Price_NL_EURMWh":         (-600, 1000),   # EUR/MWh — EPEX clearing range
    "DA_Price_DE_EURMWh":         (-600, 1000),
    "NL_TSO_Load_Forecast_MW":    (5000, 25000),   # NL grid load MW
    "TTF_Gas_EURMWh":             (2, 400),        # EUR/MWh gas — crisis range capped
    "EUA_Carbon_EUR":             (10, 150),       # EUR/tonne carbon
    "NL_Actual_Load_MW":          (5000, 25000),
}

def validate_signal(series, col, status, critical=True):
    """
    Validates a single signal series:
    1. NaN rate check
    2. Bounds check (physical plausibility)
    3. Flatline check (stuck sensor / API serving stale data)
    Returns the series unchanged — validation is non-destructive.
    """
    if series.empty or series.isna().all():
        status.record(col, "FAILED", 0, "All values NaN", critical)
        return series

    nan_pct = series.isna().mean() * 100
    valid_n = int(series.notna().sum())
    issues  = []

    if nan_pct > 30:
        issues.append(f"High NaN: {nan_pct:.1f}%")
    elif nan_pct > 5:
        issues.append(f"Elevated NaN: {nan_pct:.1f}%")

    if col in SIGNAL_BOUNDS:
        lo, hi = SIGNAL_BOUNDS[col]
        oob = ((series.dropna() < lo) | (series.dropna() > hi)).mean() * 100
        if oob > 1.0:
            issues.append(f"{oob:.1f}% out of bounds [{lo},{hi}]")

    non_zero = series[series != 0].dropna()
    if len(non_zero) > 24:
        if (non_zero.rolling(24).std() < 0.01).mean() * 100 > 20:
            issues.append("Flatline detected")

    if issues:
        final_status = "FAILED" if nan_pct > 30 and critical else "PARTIAL"
        status.record(col, final_status, valid_n, " | ".join(issues),
                      critical=(nan_pct > 30 and critical))
    else:
        status.record(col, "OK", valid_n, f"NaN={nan_pct:.1f}%", critical)

    return series
'''
    # 3. Flatline detection (same value for >6 consecutive hours = 24 PTUs)
    # Excludes zeros (legitimate for solar at night)
    non_zero = series[series != 0].dropna()
    if len(non_zero) > 24:
        rolling_std = non_zero.rolling(24).std()
        flatline_pct = (rolling_std < 0.01).mean() * 100
        if flatline_pct > 20:
            status.record(col, "PARTIAL", int(valid_n),
                          f"Flatline detected: {flatline_pct:.1f}% of windows have std<0.01",
                          critical=False)
            return series

    if nan_pct <= 5:
        status.record(col, "OK", int(valid_n), f"NaN={nan_pct:.1f}%", critical)

    return series'''


def validate_delivery_day_coverage(df: pd.DataFrame, status: FetchStatus):
    """
    Critical check: verifies D+1 delivery day has complete coverage
    for all D+1 forecast signals. Without these, the model cannot run.
    """
    delivery_start = pd.Timestamp(DELIVERY_DATE, tz=DUTCH_TZ).tz_convert("UTC")
    delivery_end   = (pd.Timestamp(DELIVERY_DATE, tz=DUTCH_TZ) + pd.DateOffset(days=1)).tz_convert("UTC")
    delivery_mask  = (df.index >= delivery_start) & (df.index < delivery_end)

    expected_ptus = 96  # 24h × 4 per hour

    # NEW Physics Bridge Validation List
    D1_REQUIRED_SIGNALS = [
        "NL_TSO_Load_Forecast_MW",      # Still critical (available at 10:00)
        "temperature_2m_Amsterdam", 
        "shortwave_radiation_Amsterdam", 
        "wind_speed_100m_Amsterdam",
        "shortwave_radiation_Bremen",   # Critical for German Wind/Solar proxy
        "shortwave_radiation_Stuttgart"
    ]

    log.info(f"\n  [D+1 Coverage Check] Delivery date: {DELIVERY_DATE}")

    all_ok = True
    for col in D1_REQUIRED_SIGNALS:
        if col not in df.columns:
            log.error(f"  [D+1 MISSING] {col} — column not in dataframe")
            status.alerts.append(f"CRITICAL: D+1 column {col} missing — model cannot run")
            all_ok = False
            continue

        delivery_series = df.loc[delivery_mask, col]
        present_ptus    = delivery_series.notna().sum()

        if present_ptus < 80:  # Allow up to 16 missing PTUs (partial publication)
            log.error(f"  [D+1 INCOMPLETE] {col}: {present_ptus}/{expected_ptus} PTUs")
            status.alerts.append(
                f"CRITICAL: {col} has only {present_ptus}/{expected_ptus} PTUs for {DELIVERY_DATE} "
                f"NL TSO load forecast (TenneT) publishes ~10:00 CET. "
                f"This script must run at 10:10 CET or later."
                )
            status.record(
                f"D1_Coverage_{col}", "FAILED", int(present_ptus),
                f"Only {present_ptus}/{expected_ptus} D+1 PTUs available",
                critical=True
                )
            all_ok = False
        else:
            log.info(f"  [D+1 OK] {col}: {present_ptus}/{expected_ptus} PTUs")

    return all_ok


# ================================================================================
# 4. NED.nl LIVE FETCHER
# ================================================================================

NL_WEATHER_STATIONS = {
    "Amsterdam":  {"lat": 52.3740, "lon": 4.8897, "weight": 0.22},
    "Rotterdam":  {"lat": 51.9225, "lon": 4.4792, "weight": 0.22},
    "Utrecht":    {"lat": 52.0908, "lon": 5.1222, "weight": 0.18},
    "Eindhoven":  {"lat": 51.4408, "lon": 5.4778, "weight": 0.18},
    "Maastricht": {"lat": 50.8483, "lon": 5.6889, "weight": 0.08},
    "Deventer":   {"lat": 52.2550, "lon": 6.1639, "weight": 0.06},
    "Friesland":  {"lat": 53.2012, "lon": 5.7999, "weight": 0.06},
}

DE_WEATHER_STATIONS = {
    "Hamburg":  {"lat": 53.5511, "lon": 9.9937},
    "Bremen":   {"lat": 53.0793, "lon": 8.8017},
    "Kiel":     {"lat": 54.3233, "lon": 10.1394},
    "Munich":   {"lat": 48.1351, "lon": 11.5820},
    "Stuttgart":{"lat": 48.7758, "lon": 9.1829},
    "Freiburg": {"lat": 47.9990, "lon": 7.8421},
}

NEIGHBOR_PRICE_ZONES = {
    "DE_LU": "DA_Price_DE_EURMWh",
    "BE":    "DA_Price_BE_EURMWh",
    "SE_3":  "DA_Price_SE3_EURMWh",
    "FR":    "DA_Price_FR_EURMWh",
}

NED_STREAMS = [
    ("NL_Solar_MW", 2, 1),
    ("NL_Wind_MW",  1, 1),
    ("NL_Load_MW",  0, 1),
]


def _ned_fetch_single_day(api_key: str, date_str: str, type_id: int,
                          activity_id: int, label: str,
                          max_retries: int = 4) -> pd.DataFrame:
    """
    Identical to historical fetch — NED API is unchanged in live mode.
    One thread per calendar day.
    """
    url     = "https://api.ned.nl/v1/utilizations"
    headers = {"X-AUTH-TOKEN": api_key, "accept": "application/ld+json"}

    curr   = pd.Timestamp(date_str)
    nxt    = curr + pd.DateOffset(days=1)
    params = {
        "point": 0, "type": type_id, "granularity": 4,
        "granularitytimezone": 1, "classification": 2, "activity": activity_id,
        "validfrom[after]":           curr.strftime("%Y-%m-%d"),
        "validfrom[strictly_before]": nxt.strftime("%Y-%m-%d"),
    }

    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, params=params,
                             allow_redirects=False, timeout=30)
            if r.status_code == 200:
                members = r.json().get("hydra:member", [])
                if not members:
                    return pd.DataFrame()
                df = pd.DataFrame(members)
                df["time"]  = pd.to_datetime(df["validfrom"], utc=True)
                # NED API returns kWh per 15-min interval summed across 250 reporting points.
                # Division by 250.0 converts to average MW per PTU.
                # Verified against ENTSO-E actual load: typical NL load 10,000-14,000 MW.
                df[label]   = pd.to_numeric(df["volume"], errors="coerce") / 250.0
                return df.set_index("time")[[label]]
            elif r.status_code == 401:
                log.warning(f"NED 401 on {date_str} — API key may have expired")
                return pd.DataFrame()
            elif r.status_code == 429:
                time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            log.warning(f"NED request failed {date_str} attempt {attempt}: {e}")
            time.sleep(2)

    return pd.DataFrame()


def fetch_ned_live(status: FetchStatus) -> pd.DataFrame:
    """
    Fetches NED actuals for the rolling 35-day window.
    NED publishes actuals with ~1-2 hour delay, so today's last few hours
    may be missing — this is expected and handled by forward-fill in features.
    """
    days = pd.date_range(
        start=START_LOCAL.date(), end=now_dutch.date(),
        freq="D", inclusive="both"
    )
    # Note: NED does NOT publish future data. Do not request tomorrow.

    frames = []
    for label, type_id, activity_id in NED_STREAMS:
        all_days = []
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(
                    _ned_fetch_single_day, NED_API_KEY,
                    day.strftime("%Y-%m-%d"), type_id, activity_id, label
                ): day for day in days
            }
            for future in as_completed(futures):
                result = future.result()
                if not result.empty:
                    all_days.append(result)

        if not all_days:
            df_stream = pd.DataFrame(index=MASTER_INDEX, columns=[label])
            status.record(f"NED_{label}", "FAILED", 0, "No data retrieved", critical=False)
        else:
            df_stream = pd.concat(all_days).sort_index()
            df_stream = df_stream[~df_stream.index.duplicated(keep="first")]
            df_stream = df_stream.resample("15min").mean().reindex(MASTER_INDEX).ffill(limit=8)
            # NED actuals are not available for tomorrow — fill future rows with NaN (correct)
            tomorrow_start = pd.Timestamp(DELIVERY_DATE, tz="UTC")
            df_stream.loc[df_stream.index >= tomorrow_start, label] = np.nan
            validate_signal(df_stream[label], f"NED_{label}", status, critical=False)

        frames.append(df_stream)

    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=MASTER_INDEX)


# ================================================================================
# 5. ENTSO-E LIVE FETCHERS
# ================================================================================

def _entsoe_client():
    return EntsoePandasClient(api_key=ENTSOE_API_KEY)


def _snap(obj, freq="15min"):
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return obj.resample(freq).ffill().reindex(MASTER_INDEX)


def fetch_entsoe_nl_da_price_live(status: FetchStatus) -> pd.DataFrame:
    """
    Fetches NL DA prices for the lookback window.
    Tomorrow's price (delivery day) is NOT available at fetch time —
    that is what the model predicts. This column will be NaN for D+1.
    """
    client = _entsoe_client()
    df = pd.DataFrame(index=MASTER_INDEX)
    for attempt in range(4):
        try:
            prices = client.query_day_ahead_prices("NL", start=START_UTC, end=END_UTC)
            df["DA_Price_NL_EURMWh"] = _snap(prices)
            # Verify the target is NaN for delivery day (no lookahead)
            tomorrow_start = pd.Timestamp(DELIVERY_DATE, tz="UTC")
            future_values = df.loc[df.index >= tomorrow_start, "DA_Price_NL_EURMWh"].notna().sum()
            if future_values >= 80:
                log.info(f"  D+1 NL DA price CONFIRMED: {future_values} PTUs available ({DELIVERY_DATE})")
                status.record("DA_Price_NL_D1_Confirmed", "OK", int(future_values),
                            f"D+1 auction cleared — {future_values}/96 PTUs known", critical=False)
            else:
                log.info(f"  D+1 NL DA price: {future_values} PTUs (may not have cleared yet)")

            validate_signal(df["DA_Price_NL_EURMWh"], "DA_Price_NL", status, critical=False)
            break

        except Exception as e:
            if attempt < 3:
                # If we haven't reached the last attempt, wait and try again
                # Exponential backoff: sleeps for 5s, 10s, 20s
                time.sleep(2 ** attempt * 5)
            else:
                log.error(f"  NL DA price failed: {e}")
                df["DA_Price_NL_EURMWh"] = np.nan
                status.record("DA_Price_NL", "FAILED", 0, str(e), critical=False)
            
    return df


def fetch_entsoe_neighbor_da_prices_live(status: FetchStatus) -> pd.DataFrame:
    client = _entsoe_client()
    df = pd.DataFrame(index=MASTER_INDEX)
    for zone, col in NEIGHBOR_PRICE_ZONES.items():
        for attempt in range(4):
            try:
                prices = client.query_day_ahead_prices(zone, start=START_UTC, end=END_UTC)
                df[col] = _snap(prices)
                validate_signal(df[col], f"DA_{zone}", status, critical=False)
                break

            except Exception as e:
                if attempt < 3:
                    time.sleep(2 ** attempt * 5)
                else:
                    log.warning(f"  DA price {zone} failed: {e}")
                    df[col] = np.nan
                    status.record(f"DA_{zone}", "FAILED", 0, str(e), critical=False)
    return df


def fetch_entsoe_load_live(status: FetchStatus) -> pd.DataFrame:
    """
    Fetches NL actual load (lagged use only) and NL TSO D+1 load forecast.
    The TSO D+1 forecast is the CRITICAL signal — must have D+1 data.
    """
    client = _entsoe_client()
    df = pd.DataFrame(index=MASTER_INDEX)

    # Actual load — not available for D+1, used lagged
    for attempt in range(4):
        try:
            actual = client.query_load("NL", start=START_UTC, end=END_UTC)
            df["NL_Actual_Load_MW"] = _snap(actual)
            validate_signal(df["NL_Actual_Load_MW"], "NL_Actual_Load", status, critical=False)
            break

        except Exception as e:
            if attempt < 3:
                time.sleep(2 ** attempt * 5)
            else:
                log.warning(f"  NL actual load failed: {e}")
                status.record("NL_Actual_Load", "FAILED", 0, str(e), critical=False)

    # TSO D+1 forecast — CRITICAL: must cover delivery day
    for attempt in range(4):
        try:
            forecast = client.query_load_forecast("NL", start=START_UTC, end=END_UTC)
            df["NL_TSO_Load_Forecast_MW"] = _snap(forecast)
            validate_signal(df["NL_TSO_Load_Forecast_MW"], "NL_TSO_Load_Forecast", status, critical=True)
            break

        except Exception as e:
            if attempt < 3:
                time.sleep(2 ** attempt * 5)
            else:
                log.error(f"  NL TSO load forecast failed: {e}")
                df["NL_TSO_Load_Forecast_MW"] = np.nan
                status.record("NL_TSO_Load_Forecast", "FAILED", 0, str(e), critical=True)

    # Neighbor load forecasts
    for country in ["DE_LU", "BE"]:
        for attempt in range(4):
            try:
                fc = client.query_load_forecast(country, start=START_UTC, end=END_UTC)
                df[f"Load_Forecast_{country}_MW"] = _snap(fc)
                validate_signal(df[f"Load_Forecast_{country}_MW"],
                                f"Load_Forecast_{country}", status, critical=False)
                break
            except Exception as e:
                if attempt < 3:
                    time.sleep(2 ** attempt * 5)
                else:
                    log.warning(f"  Load forecast {country} failed: {e}")
                    df[f"Load_Forecast_{country}_MW"] = np.nan
                    status.record(f"Load_Forecast_{country}", "FAILED", 0, str(e), critical=False)

    return df


def fetch_entsoe_crossborder_flows_live(status: FetchStatus) -> pd.DataFrame:
    client = _entsoe_client()
    df = pd.DataFrame(index=MASTER_INDEX)
    flow_pairs = [
        ("NL", "DE_LU", "Flow_NL_DE_MW"),
        ("NL", "BE",    "Flow_NL_BE_MW"),
        ("DE_LU", "NL", "Flow_DE_NL_MW"),
    ]
    for from_z, to_z, col in flow_pairs:
        for attempt in range(4):
            try:
                flows = client.query_scheduled_exchanges(from_z, to_z,
                                                        start=START_UTC, end=END_UTC)
                df[col] = _snap(flows)
                validate_signal(df[col], f"Flow_{from_z}_{to_z}", status, critical=False)
                break

            except Exception as e:
                if attempt < 3:
                    time.sleep(2 ** attempt * 5)
                else:
                    log.warning(f"  Flow {from_z}→{to_z} failed: {e}")
                    status.record(f"Flow_{from_z}_{to_z}", "FAILED", 0, str(e), critical=False)

    if "Flow_NL_DE_MW" in df and "Flow_NL_BE_MW" in df:
        df["NL_Net_Export_MW"] = df["Flow_NL_DE_MW"] + df["Flow_NL_BE_MW"]

    return df


def fetch_entsoe_imbalance_live(status: FetchStatus) -> pd.DataFrame:
    client = _entsoe_client()
    df = pd.DataFrame(index=MASTER_INDEX)
    for attempt in range(4):
        try:
            imb = client.query_imbalance_prices("NL", start=START_UTC, end=END_UTC)
            col = "Price for consumption" if isinstance(imb, pd.DataFrame) and \
                "Price for consumption" in imb.columns else (imb.columns[0] if isinstance(imb, pd.DataFrame) else None)
            if col:
                df["NL_Imbalance_Price_EURMWh"] = _snap(imb[col])
            else:
                df["NL_Imbalance_Price_EURMWh"] = _snap(imb)
            validate_signal(df["NL_Imbalance_Price_EURMWh"], "NL_Imbalance", status, critical=False)
            break

        except Exception as e:
            if attempt < 3:
                time.sleep(2 ** attempt * 5)
            else:
                log.warning(f"  NL imbalance failed: {e}")
                status.record("NL_Imbalance", "FAILED", 0, str(e), critical=False)
    return df


def fetch_entsoe_thermal_outages_live(status: FetchStatus) -> pd.DataFrame:
    """
    Fetches REMIT outages using 1-month chunking.
    In live mode, only the last 35 days are needed.
    """
    client = _entsoe_client()
    df = pd.DataFrame(index=MASTER_INDEX)
    zones = {"NL": "NL_Thermal_Outage_MW", "FR": "FR_Nuclear_Outage_MW"}

    for zone, col_name in zones.items():
        df[col_name] = 0.0
        chunks = []
        curr = START_UTC
        while curr < END_UTC:
            nxt = min(curr + pd.DateOffset(months=1), END_UTC)
            for attempt in range(4):
                try:
                    outages = client.query_unavailability_of_generation_units(
                        zone, start=curr, end=nxt, docstatus=None)
                    if outages is not None and not outages.empty:
                        chunks.append(outages)
                    break  # Success (empty or not) — move to next month
                except Exception as e:
                    if "404" in str(e):
                        break  # 404 = no outages this month — expected, don't retry
                    if attempt < 3:
                        time.sleep(2 ** attempt * 5)
                    else:
                        log.warning(f"  REMIT {zone} chunk {curr.date()}: {e}")
            curr = nxt
            time.sleep(1)

        if chunks:
            all_outages = pd.concat(chunks).drop_duplicates()
            if "nominal_power" in all_outages.columns and "avail_qty" in all_outages.columns:
                all_outages["nominal_power"] = pd.to_numeric(all_outages["nominal_power"], errors="coerce")
                all_outages["avail_qty"]     = pd.to_numeric(all_outages["avail_qty"], errors="coerce")
                if zone == "FR" and "plant_type" in all_outages.columns:
                    all_outages = all_outages[
                        all_outages["plant_type"].astype(str).str.contains("Nuclear", case=False, na=False)
                    ]
                all_outages["offline_mw"] = (all_outages["nominal_power"] - all_outages["avail_qty"]).clip(lower=0)
                temp_series = pd.Series(0.0, index=MASTER_INDEX)
                for _, row in all_outages.iterrows():
                    o_start = pd.Timestamp(row["start"])
                    o_start = o_start.tz_localize("UTC") if o_start.tz is None else o_start.tz_convert("UTC")
                    o_end   = pd.Timestamp(row["end"])
                    o_end   = o_end.tz_localize("UTC") if o_end.tz is None else o_end.tz_convert("UTC")
                    mask    = (temp_series.index >= o_start) & (temp_series.index < o_end)
                    temp_series.loc[mask] += row["offline_mw"]
                df[col_name] = temp_series
                status.record(f"REMIT_{zone}", "OK", len(all_outages),
                              f"{len(all_outages)} outage events", critical=False)
                
        else:
            status.record(f"REMIT_{zone}", "OK", 0,
                        "No outages reported — normal", critical=False)

    return df


def fetch_entsoe_actual_generation_live(status: FetchStatus) -> pd.DataFrame:
    """
    Fetches NL gas and coal actual generation for the lookback window.
    Used as lagged features only — raw columns dropped before ML matrix.
    """
    client = _entsoe_client()
    df = pd.DataFrame(index=MASTER_INDEX)

    def get_actuals(gen_df, fuel_type):
        if fuel_type in gen_df.columns.get_level_values(0):
            data = gen_df[fuel_type]
            if isinstance(data, pd.DataFrame):
                return data["Actual Aggregated"] if "Actual Aggregated" in data.columns else data.iloc[:, 0]
            return data
        return None

    chunks = []
    curr   = START_UTC
    while curr < END_UTC:
        nxt = min(curr + pd.DateOffset(months=1), END_UTC)
        for attempt in range(4):
            try:
                gen_chunk = client.query_generation("NL", start=curr, end=nxt)
                chunks.append(gen_chunk)
                break
            except Exception as e:
                if "503" in str(e) or "504" in str(e):
                    time.sleep(2 ** attempt * 5)
                else:
                    break
        curr = nxt
        time.sleep(1)

    if chunks:
        master_gen = pd.concat(chunks)
        master_gen = master_gen[~master_gen.index.duplicated(keep="first")]
        gas_series  = get_actuals(master_gen, "Fossil Gas")
        coal_series = get_actuals(master_gen, "Fossil Hard coal")
        if gas_series is not None:
            df["NL_Fossil_Gas_Actual_MW"] = _snap(gas_series)
            status.record("NL_Gas_Actual", "OK",
                          int(df["NL_Fossil_Gas_Actual_MW"].notna().sum()), "", critical=False)
        if coal_series is not None:
            df["NL_Coal_Actual_MW"] = _snap(coal_series)
            status.record("NL_Coal_Actual", "OK",
                    int(df["NL_Coal_Actual_MW"].notna().sum()), "", critical=False)
        else:
            status.record("NL_Coal_Actual", "PARTIAL", 0,
                    "Coal generation not available in response", critical=False)
    else:
        status.record("NL_Gas_Actual", "FAILED", 0, "No generation data", critical=False)

    return df


# ================================================================================
# 6. WEATHER LIVE FETCHER
# ================================================================================

WEATHER_VARIABLES = [
    "temperature_2m", "apparent_temperature",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "wind_speed_100m", "wind_direction_100m",
    "cloud_cover", "relative_humidity_2m", "precipitation",
    "weather_code"  # <--- FIX: Added missing feature required by the model
]

def fetch_weather_live(status: FetchStatus) -> pd.DataFrame:
    # Combine both dictionaries
    all_stations = {**NL_WEATHER_STATIONS, **DE_WEATHER_STATIONS}
    cities = list(all_stations.keys())
    lats   = [str(all_stations[c]["lat"]) for c in cities]
    lons   = [str(all_stations[c]["lon"]) for c in cities]

    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":   ",".join(lats),
                "longitude":  ",".join(lons),
                "hourly":     ",".join(WEATHER_VARIABLES),
                "timezone":   "UTC",
                "wind_speed_unit": "ms",
                "past_days": LIVE_LOOKBACK_DAYS,      
                "forecast_days": LIVE_FORWARD_DAYS + 1 
            },
            timeout=60,
        )
        resp = r.json()
        city_frames = []
        
        if isinstance(resp, list):
            for i, city in enumerate(cities):
                if "hourly" not in resp[i]:
                    continue
                df_city = pd.DataFrame(resp[i]["hourly"])
                df_city["time"] = pd.to_datetime(df_city["time"]).dt.tz_localize("UTC")
                df_city.set_index("time", inplace=True)
                
                # ── THE FIX: Force strict alignment before appending ──
                # This guarantees every city has the exact same rows. Missing data 
                # becomes NaN in its correct time slot, rather than shifting the whole column.
                df_city = df_city.reindex(MASTER_INDEX_HOURLY)
                
                # Rename columns to match model expectations (e.g., temperature_2m_Amsterdam)
                df_city.columns = [f"{col}_{city}" for col in df_city.columns]

                city_frames.append(df_city)
                
        if city_frames:
            # Concatenate all cities horizontally
            df_weather = pd.concat(city_frames, axis=1)
            # Resample from hourly to 15-min PTUs using forward-fill
            df_weather = df_weather.resample("15min").ffill().reindex(MASTER_INDEX)
            
            status.record("Weather_Forecast", "OK", len(df_weather), "", critical=False)
            log.info(f"  Weather: {df_weather.shape[1]} cols, {df_weather.notna().mean().mean()*100:.1f}% fill")
            return df_weather
        else:
            status.record("Weather_Forecast", "FAILED", 0, "API returned empty list", critical=True)
            return pd.DataFrame(index=MASTER_INDEX)

    except Exception as e:
        log.error(f"  Weather forecast API failed: {e}")
        status.record("Weather_Forecast", "FAILED", 0, str(e), critical=True)
        return pd.DataFrame(index=MASTER_INDEX)


# ================================================================================
# 7. MACRO LIVE FETCHER — TTF Gas, EUA Carbon
# ================================================================================

def fetch_macro_live(status: FetchStatus) -> pd.DataFrame:
    """
    CRITICAL LIVE PRODUCTION NOTE (10:10 CET cron — OPTION 3):

    The DA auction for delivery day D clears at 12:00 CET on day D-1.
    Our live script executes at 10:10 CET on day D-1.
    ICE (where TTF and EUA trade) closes at ~17:00 CET, so at 10:10 AM:
    - Today's (D-1) closing price is NOT yet available
    - Yesterday's (D-2) closing price is the most recent finalized data

    After ffill, all delivery-day PTUs hold the D-2 close. The feature
    engineering script applies shift(96) for training-parity, which
    yields ttf_spot = D-2 close at delivery day D.

    KNOWN LAG MISMATCH:
    - Training: ttf_spot at delivery day T = T-1 close (24h prior)
    - Live:     ttf_spot at delivery day D = D-2 close (48h prior)

    Impact: ~EUR 1-3/MWh distribution shift on TTF and ~EUR 0.5/EUA tonne.
    Expected to add ~EUR 0.5-1.5/MWh to live MAE vs holdout MAE of 14.79.

    This is logged as a known production lag. Resolution in next iteration:
    either switch to intraday TTF feed (ICE Endex Connect) or remove shift(96)
    in live mode and retrain.
    """
    df = pd.DataFrame(index=MASTER_INDEX)
    yf_s = START_UTC.strftime("%Y-%m-%d")
    yf_e = END_UTC.strftime("%Y-%m-%d")

    # TTF Gas
    try:
        raw = yf.download("TTF=F", start=yf_s, end=yf_e, progress=False, auto_adjust=True)
        if not raw.empty:
            c = raw["Close"].squeeze() if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
            c.index = pd.to_datetime(c.index)
            c.index = c.index.tz_localize("UTC") if c.index.tz is None else c.index.tz_convert("UTC")
            df["TTF_Gas_EURMWh"] = c.reindex(MASTER_INDEX).ffill().bfill()
            mean_ttf = float(df["TTF_Gas_EURMWh"].dropna().mean())
            log.info(f"  TTF Gas: mean EUR{mean_ttf:.2f}/MWh")
            if mean_ttf < 5 or mean_ttf > 300:
                status.record("TTF_Gas", "PARTIAL", int(df["TTF_Gas_EURMWh"].notna().sum()),
                              f"Mean EUR{mean_ttf:.2f}/MWh — outside expected range [5, 300]",
                              critical=False)
            else:
                status.record("TTF_Gas", "OK", int(df["TTF_Gas_EURMWh"].notna().sum()),
                              f"Mean EUR{mean_ttf:.2f}/MWh", critical=False)
    except Exception as e:
        log.error(f"  TTF Gas failed: {e}")
        status.record("TTF_Gas", "FAILED", 0, str(e), critical=False)

    # EUA Carbon
    eua_fetched = False
    for ticker in ["CO2.L", "CARB.PA", "XCO2.PA"]:
        try:
            raw = yf.download(ticker, start=yf_s, end=yf_e, progress=False, auto_adjust=True)
            if not raw.empty:
                c = raw["Close"].squeeze() if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
                mean_price = float(c.mean())
                if 45.0 <= mean_price <= 120.0:
                    c.index = pd.to_datetime(c.index)
                    c.index = c.index.tz_localize("UTC") if c.index.tz is None else c.index.tz_convert("UTC")
                    df["EUA_Carbon_EUR"] = c.reindex(MASTER_INDEX).ffill().bfill()
                    status.record("EUA_Carbon", "OK", int(df["EUA_Carbon_EUR"].notna().sum()),
                                  f"Mean EUR{mean_price:.2f}/t via {ticker}", critical=False)
                    eua_fetched = True
                    break
        except Exception:
            continue

    if not eua_fetched:
        EUA_CURVE = {"2026-01": 65.0, "2026-04": 64.0, "2026-05": 68.5, "2026-06": 70.0, "2026-07": 71.0,}

        latest_curve_month = max(pd.to_datetime(list(EUA_CURVE.keys())))
        months_stale = (pd.Timestamp.now() - latest_curve_month).days / 30
        if months_stale > 1:
            status.alerts.append(
                f"WARNING: EUA_CURVE hardcoded fallback is {months_stale:.0f} months stale — "
                f"update EUA_CURVE in voltcast_fetch_live.py"
            )

        monthly = pd.Series(
            list(EUA_CURVE.values()),
            index=pd.to_datetime(list(EUA_CURVE.keys())).tz_localize("UTC")
        ).sort_index()
        df["EUA_Carbon_EUR"] = monthly.reindex(MASTER_INDEX, method="ffill").ffill().bfill()
        status.record("EUA_Carbon", "PARTIAL", int(df["EUA_Carbon_EUR"].notna().sum()),
                      "Using hardcoded EEX curve — update EUA_CURVE monthly", critical=False)

    # Derived: CCGT marginal cost
    if "TTF_Gas_EURMWh" in df.columns and "EUA_Carbon_EUR" in df.columns:
        df["CCGT_Marginal_Cost_EUR"] = (
            df["TTF_Gas_EURMWh"] / 0.56 + df["EUA_Carbon_EUR"] * 0.36
        )

    return df


# ================================================================================
# 8. MASTER LIVE PIPELINE
# ================================================================================

def run_live_fetch() -> int:
    """
    Main entry point. Returns exit code:
      0 = all critical sources OK, downstream can proceed
      1 = one or more critical failures, downstream should abort
    """
    log.info("=" * 65)
    log.info(f"  VOLTCAST IPI — LIVE DATA FETCH")
    log.info(f"  Delivery date : {DELIVERY_DATE}")
    log.info(f"  Fetch window  : {START_LOCAL.date()} to {END_LOCAL.date()}")
    log.info(f"  Run time      : {now_dutch.strftime('%Y-%m-%d %H:%M %Z')}")
    log.info("=" * 65)

    status = FetchStatus()
    frames = []

    # --- Step 1: NED Actuals ---
    log.info("\n[1/6] NED.nl — Solar, Wind, Load actuals")
    try:
        df_ned = fetch_ned_live(status)
        frames.append(df_ned)
        log.info(f"  NED shape: {df_ned.shape}")
    except Exception as e:
        log.error(f"  NED fetch crashed: {traceback.format_exc()}")
        status.record("NED_ALL", "FAILED", 0, str(e), critical=False)

    # --- Step 2: ENTSO-E ---
    log.info("\n[2/6] ENTSO-E — Prices, Forecasts, Flows, Outages")
    entsoe_fetchers = [
        ("NL_DA_Price",    fetch_entsoe_nl_da_price_live),
        ("Neighbor_Prices",fetch_entsoe_neighbor_da_prices_live),
        ("Load",           fetch_entsoe_load_live),
        ("Flows",          fetch_entsoe_crossborder_flows_live),
        ("Imbalance",      fetch_entsoe_imbalance_live),
        ("REMIT",          fetch_entsoe_thermal_outages_live),
        ("Gen_Actuals",    fetch_entsoe_actual_generation_live),
    ]
    for name, fn in entsoe_fetchers:
        try:
            df_part = fn(status)
            frames.append(df_part)
        except Exception as e:
            log.error(f"  ENTSO-E {name} crashed: {e}")
            status.record(f"ENTSOE_{name}", "FAILED", 0, str(e), critical=True)

    # --- Step 3: Weather ---
    log.info("\n[3/6] Open-Meteo — Weather (7 NL cities)")
    try:
        df_weather = fetch_weather_live(status)
        frames.append(df_weather)
        log.info(f"  Weather shape: {df_weather.shape}")
    except Exception as e:
        log.error(f"  Weather fetch crashed: {e}")
        status.record("Weather_ALL", "FAILED", 0, str(e), critical=True)

    # --- Step 4: Macro ---
    log.info("\n[4/6] Yahoo Finance — TTF Gas, EUA Carbon")
    try:
        df_macro = fetch_macro_live(status)
        frames.append(df_macro)
        log.info(f"  Macro shape: {df_macro.shape}")
    except Exception as e:
        log.error(f"  Macro fetch crashed: {e}")
        status.record("Macro_ALL", "FAILED", 0, str(e), critical=False)

    # --- Step 5: Merge ---
    log.info("\n[5/6] Merging all sources")
    df_master = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=MASTER_INDEX)
    df_master = df_master.reindex(MASTER_INDEX)
    df_master = df_master.loc[:, ~df_master.columns.duplicated(keep="first")]

    outage_cols = [c for c in df_master.columns if "Outage_MW" in c]
    df_master[outage_cols] = df_master[outage_cols].fillna(0.0)
    df_master = df_master.ffill(limit=4)

    # --- Step 6: D+1 Coverage Validation ---
    log.info("\n[6/6] Validating D+1 delivery day coverage")
    d1_ok = validate_delivery_day_coverage(df_master, status)
    if not d1_ok:
        log.error("  D+1 coverage FAILED — model cannot produce valid forecast")

    # --- Finalize ---
    overall_pass = status.finalize()

    log.info(f"\n  Master shape: {df_master.shape[0]:,} rows × {df_master.shape[1]} columns")
    log.info(f"  Overall status: {'PASS' if overall_pass else 'FAIL'}")

    # Save data and status
    df_master.to_parquet(LIVE_PARQUET)
    log.info(f"  Saved: {LIVE_PARQUET}")

    status.save()
    send_alert(status)

    # Print final summary
    log.info("\n" + "=" * 65)
    log.info("  FETCH SUMMARY")
    log.info("=" * 65)
    for src, v in status.sources.items():
        icon = "✓" if v["status"] == "OK" else ("⚠" if v["status"] == "PARTIAL" else "✗")
        log.info(f"  {icon} {src:<35} {v['status']:<8} {v['rows']:>8,} rows")

    if status.alerts:
        log.warning(f"\n  {len(status.alerts)} alert(s):")
        for a in status.alerts:
            log.warning(f"    • {a}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    exit_code = run_live_fetch()
    sys.exit(exit_code)