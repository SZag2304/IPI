"""
================================================================================
 VOLTCAST IPI — MASTER DATA FETCH PIPELINE
 Target: Netherlands EPEX Day-Ahead Price Forecasting
 Product: Industrial Procurement Intelligence
================================================================================
 Data Sources:
   [1] NED.nl          — NL Solar, Wind, Actual Load (15-min actuals)
   [2] ENTSO-E         — DA Prices, Load Forecasts, Generation Forecasts,
                         Cross-border Flows, Thermal Outages, Imbalance Prices
   [3] Open-Meteo      — High-resolution weather for 7 NL cities
   [4] Yahoo Finance   — TTF Gas, EUA Carbon (macro price floor signals)

 Architecture:
   - Every fetcher is independent and testable
   - Every source is cached to parquet — re-runs are instant
   - Every source is forced onto the same 15-min UTC master index
   - Integrity report runs automatically after merge
================================================================================
"""

import os
import time
import logging
import warnings
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from entsoe import EntsoePandasClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# This single line finds your .env file and loads the variables into the system environment
load_dotenv()

warnings.filterwarnings("ignore")

# ================================================================================
# 0. LOGGING — structured output, saved to file and console
# ================================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("voltcast_fetch_v2.log", mode="w"),
    ],
)
log = logging.getLogger("VoltCast")


# ================================================================================
# 1. CONFIGURATION — change ONLY this block
# ================================================================================

# --- API Keys (replace with your actual keys) ---
NED_API_KEY    = os.environ.get("VOLTCAST_NED_KEY")
ENTSOE_API_KEY = os.environ.get("VOLTCAST_ENTSOE_KEY")

# --- Time Range (Dutch local time → converted to UTC internally) ---
DUTCH_TZ         = "Europe/Amsterdam"
START_LOCAL      = pd.Timestamp("2023-01-01 00:00:00", tz=DUTCH_TZ)
END_LOCAL        = pd.Timestamp("2026-04-01 00:00:00", tz=DUTCH_TZ)

# --- Internal UTC backbone ---
START_UTC        = START_LOCAL.tz_convert("UTC")
END_UTC          = END_LOCAL.tz_convert("UTC")

# --- Master 15-min UTC grid (every fetcher aligns to this) ---
MASTER_INDEX     = pd.date_range(
    start=START_UTC, end=END_UTC, freq="15min", inclusive="left"
)

# --- Cache directory (all .parquet files land here) ---
CACHE_DIR        = "voltcast_ipi_cache_v2"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- NL weather stations with population weights ---
# Weights sum to 1.0 → used later for national-level aggregation
NL_WEATHER_STATIONS = {
    "Amsterdam":  {"lat": 52.3740, "lon": 4.8897, "weight": 0.22},
    "Rotterdam":  {"lat": 51.9225, "lon": 4.4792, "weight": 0.22},
    "Utrecht":    {"lat": 52.0908, "lon": 5.1222, "weight": 0.18},
    "Eindhoven":  {"lat": 51.4408, "lon": 5.4778, "weight": 0.18},
    "Maastricht": {"lat": 50.8483, "lon": 5.6889, "weight": 0.08},
    "Deventer":   {"lat": 52.2550, "lon": 6.1639, "weight": 0.06},
    "Friesland":  {"lat": 53.2012, "lon": 5.7999, "weight": 0.06},  # Leeuwarden representative
}

# --- DE weather stations (Wind in North, Solar in South) ---
DE_WEATHER_STATIONS = {
    "Hamburg":  {"lat": 53.5511, "lon": 9.9937},
    "Bremen":   {"lat": 53.0793, "lon": 8.8017},
    "Kiel":     {"lat": 54.3233, "lon": 10.1394},
    "Munich":   {"lat": 48.1351, "lon": 11.5820},
    "Stuttgart":{"lat": 48.7758, "lon": 9.1829},
    "Freiburg": {"lat": 47.9990, "lon": 7.8421},
}

# --- ENTSO-E zones to fetch neighbor prices from ---
NEIGHBOR_PRICE_ZONES = {
    "DE_LU": "DA_Price_DE_EURMWh",   # Germany — highest coupling with NL
    "BE":    "DA_Price_BE_EURMWh",   # Belgium  — direct border neighbor
    "SE_3":  "DA_Price_SE3_EURMWh",  # Sweden 3 — Nordic hydro proxy
    "FR":    "DA_Price_FR_EURMWh",   # France   — interconnected via BE
    "NO_2":  "DA_Price_NO2_EURMWh",  # Norway 2 — NorNed hydro anchor (v2, forensic P4)
}

# --- NED API: (label, type_id, activity_id) ---
NED_STREAMS = [
    ("NL_Solar_MW",  2, 1),   # Solar generation actuals
    ("NL_Wind_MW",   1, 1),   # Wind generation actuals
    ("NL_Load_MW",   0, 1),   # Actual consumption (NED source)
]


# ================================================================================
# 2. CACHE MANAGER — prevents redundant API calls across runs
# ================================================================================

def cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.parquet")


def load_cache(name: str) -> pd.DataFrame | None:
    path = cache_path(name)
    if os.path.exists(path):
        log.info(f"[CACHE HIT]  {name}")
        return pd.read_parquet(path)
    return None


def save_cache(df: pd.DataFrame, name: str):
    if not df.empty:
        df.to_parquet(cache_path(name))
        log.info(f"[CACHED]     {name}  →  {df.shape[0]:,} rows, {df.shape[1]} cols")


def fetch_or_cache(name: str, fetch_fn, *args, force_refresh=False):
    """
    Returns cached data if available, otherwise runs fetch_fn.
    Set force_refresh=True to bypass cache and re-fetch.
    """
    if not force_refresh:
        cached = load_cache(name)
        if cached is not None:
            return cached

    log.info(f"[FETCHING]   {name} ...")
    df = fetch_fn(*args)
    save_cache(df, name)
    return df


# ================================================================================
# 3. NED.nl FETCHER — Solar, Wind, Load actuals (15-min, multithreaded)
# ================================================================================

def _ned_fetch_single_day(api_key: str, date_str: str, type_id: int,
                          activity_id: int, label: str, max_retries: int = 8) -> pd.DataFrame:
    """
    Fetches one calendar day from NED. Designed to run in a thread pool.
    Uses local date strings to be leap-year safe (avoids UTC offset traps).
    """
    url     = "https://api.ned.nl/v1/utilizations"
    headers = {"X-AUTH-TOKEN": api_key, "accept": "application/ld+json"}

    curr = pd.Timestamp(date_str)
    nxt  = curr + pd.DateOffset(days=1)

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
                df[label]   = pd.to_numeric(df["volume"], errors="coerce") / 250.0
                return df.set_index("time")[[label]]

            elif r.status_code == 401:
                log.warning(f"NED auth error on {date_str} — check API key")
                return pd.DataFrame()

            elif r.status_code == 429:
                wait = 2 ** attempt
                time.sleep(wait)
                continue

            else:
                # For 500, 502, 503 errors
                time.sleep(2)
                continue

        except requests.exceptions.RequestException:
            sleep_time = 2 ** attempt
            time.sleep(sleep_time)

    log.error(f"NED FAILED: {label} on {date_str} after {max_retries} retries")
    return pd.DataFrame()


def fetch_ned_stream(api_key: str, label: str, type_id: int,
                     activity_id: int, max_workers: int = 4) -> pd.DataFrame:
    """
    Fetches a full NED stream by dispatching one thread per calendar day.
    Covers the full START_LOCAL → END_LOCAL range.
    """
    days = pd.date_range(
        start=START_LOCAL.date(), end=END_LOCAL.date(),
        freq="D", inclusive="left"
    )

    all_days = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _ned_fetch_single_day, api_key,
                day.strftime("%Y-%m-%d"), type_id, activity_id, label
            ): day
            for day in days
        }

        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if not result.empty:
                all_days.append(result)
            if done % 150 == 0:
                log.info(f"  NED {label}: {done}/{len(days)} days complete")

    if not all_days:
        log.error(f"NED: no data retrieved for {label}")
        return pd.DataFrame(index=MASTER_INDEX)

    df = pd.concat(all_days).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    # Align to master index, forward-fill sub-gaps
    df = df.resample("15min").mean().reindex(MASTER_INDEX).ffill()
    return df


def fetch_all_ned(api_key: str) -> pd.DataFrame:
    """Fetches all NED streams and merges into one DataFrame."""
    if not api_key:
        log.error("NED_API_KEY not set — skipping NED fetch, returning empty frame")
        return pd.DataFrame(index=MASTER_INDEX)
    frames = []
    for label, type_id, activity_id in NED_STREAMS:
        df_stream = fetch_or_cache(
            f"ned_{label.lower()}",
            fetch_ned_stream,
            api_key, label, type_id, activity_id
        )
        frames.append(df_stream)

    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=MASTER_INDEX)


# ================================================================================
# 4. ENTSO-E FETCHER — modular, one function per signal class
# ================================================================================

def _entsoe_client() -> EntsoePandasClient:
    return EntsoePandasClient(api_key=ENTSOE_API_KEY)


def _snap(obj, freq="15min"):
    """Resamples any ENTSO-E series or DataFrame to the 15-min master index."""
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return obj.resample(freq).ffill().reindex(MASTER_INDEX)


# ── 4A. DA PRICES ─────────────────────────────────────────────────────────────

def fetch_entsoe_nl_da_price() -> pd.DataFrame:
    """
    NL EPEX day-ahead price — the TARGET VARIABLE for IPI.
    Hourly from ENTSO-E, upsampled to 15-min with ffill.
    """
    client = _entsoe_client()
    df = pd.DataFrame(index=MASTER_INDEX)
    try:
        prices = client.query_day_ahead_prices("NL", start=START_UTC, end=END_UTC)
        df["DA_Price_NL_EURMWh"] = _snap(prices)
        log.info(f"  NL DA price: {df['DA_Price_NL_EURMWh'].notna().sum():,} valid rows")
    except Exception as e:
        log.error(f"  NL DA price fetch failed: {e}")
        df["DA_Price_NL_EURMWh"] = np.nan
    return df


def fetch_entsoe_neighbor_da_prices() -> pd.DataFrame:
    """
    DA prices for DE, BE, SE3, FR.
    DE price is the single strongest correlate with NL price (~75-85%).
    Used as lagged features only (lag_96 at minimum) to avoid leakage.
    """
    client = _entsoe_client()
    df = pd.DataFrame(index=MASTER_INDEX)

    for zone, col in NEIGHBOR_PRICE_ZONES.items():
        try:
            prices = client.query_day_ahead_prices(zone, start=START_UTC, end=END_UTC)
            df[col] = _snap(prices)
            log.info(f"  DA price {zone}: {df[col].notna().sum():,} valid rows")
        except Exception as e:
            log.warning(f"  DA price {zone} failed: {e}")
            df[col] = np.nan

    return df


# ── 4B. LOAD DATA ──────────────────────────────────────────────────────────────

def fetch_entsoe_load_data() -> pd.DataFrame:
    """
    NL actual load + NL TSO load forecast (D+1 from TenneT).
    Also fetches DE and BE load forecasts as grid pressure signals.
    TSO forecast is a legal D+1 feature — zero leakage.
    Actual load must be used as lag_96+ only.
    """
    client = _entsoe_client()
    df = pd.DataFrame(index=MASTER_INDEX)

    # NL actual load (ENTSO-E source — use lag_96 in ML)
    try:
        actual = client.query_load("NL", start=START_UTC, end=END_UTC)
        df["NL_Actual_Load_MW"] = _snap(actual)
        log.info(f"  NL actual load: {df['NL_Actual_Load_MW'].notna().sum():,} rows")
    except Exception as e:
        log.warning(f"  NL actual load failed: {e}")

    # NL TSO day-ahead load forecast (zero leakage — published D-1 by TenneT)
    try:
        forecast = client.query_load_forecast("NL", start=START_UTC, end=END_UTC)
        df["NL_TSO_Load_Forecast_MW"] = _snap(forecast)
        log.info(f"  NL TSO load forecast: {df['NL_TSO_Load_Forecast_MW'].notna().sum():,} rows")
    except Exception as e:
        log.warning(f"  NL TSO load forecast failed: {e}")

    # Neighbor load forecasts (grid pressure on interconnectors)
    for country in ["DE_LU", "BE"]:
        try:
            fc = client.query_load_forecast(country, start=START_UTC, end=END_UTC)
            df[f"Load_Forecast_{country}_MW"] = _snap(fc)
            log.info(f"  Load forecast {country}: {df[f'Load_Forecast_{country}_MW'].notna().sum():,} rows")
        except Exception as e:
            log.warning(f"  Load forecast {country} failed: {e}")

    return df


'''# ── 4C. GENERATION FORECASTS ──────────────────────────────────────────────────

def fetch_entsoe_generation_forecasts() -> pd.DataFrame:
    """
    Wind + solar generation forecasts for NL and DE.
    These are D+1 forecasts published by ENTSO-E — what the market actually
    uses to price the day-ahead auction. Critical for renewable suppression signal.
    """
    client = _entsoe_client()
    df = pd.DataFrame(index=MASTER_INDEX)

    # NL wind + solar forecast
    try:
        nl_ren = client.query_wind_and_solar_forecast("NL", start=START_UTC, end=END_UTC)
        if not nl_ren.empty:
            wind_cols  = [c for c in nl_ren.columns if "Wind"  in str(c)]
            solar_cols = [c for c in nl_ren.columns if "Solar" in str(c)]
            if wind_cols:
                df["NL_Wind_Forecast_MW"]  = _snap(nl_ren[wind_cols].sum(axis=1))
            if solar_cols:
                df["NL_Solar_Forecast_MW"] = _snap(nl_ren[solar_cols].sum(axis=1))
            
            # 🌟 FIX B2: Safe addition handling missing columns
            _w = df["NL_Wind_Forecast_MW"] if "NL_Wind_Forecast_MW" in df.columns else 0
            _s = df["NL_Solar_Forecast_MW"] if "NL_Solar_Forecast_MW" in df.columns else 0
            if not (isinstance(_w, int) and isinstance(_s, int)):
                df["NL_Renewables_Forecast_MW"] = (
                    (_w if not isinstance(_w, int) else 0) + 
                    (_s if not isinstance(_s, int) else 0)
                )
        log.info("  NL renewables forecast: OK")
    except Exception as e:
        log.warning(f"  NL renewables forecast failed: {e}")

    # DE wind + solar forecast (German surplus is NL's strongest price suppressor)
    try:
        de_ren = client.query_wind_and_solar_forecast("DE_LU", start=START_UTC, end=END_UTC)
        if not de_ren.empty:
            df["DE_Renewables_Forecast_MW"] = _snap(de_ren.sum(axis=1))
        log.info("  DE renewables forecast: OK")
    except Exception as e:
        log.warning(f"  DE renewables forecast failed: {e}")

    return df'''


# ── 4D. CROSS-BORDER FLOWS ────────────────────────────────────────────────────

def fetch_entsoe_crossborder_flows() -> pd.DataFrame:
    """
    Scheduled net cross-border exchanges for NL↔DE and NL↔BE.
    Positive = NL exporting. Negative = NL importing cheap power.
    Congestion on these interconnectors drives NL-DE price decoupling events
    which are the highest price spike risk for industrial buyers.
    """
    client = _entsoe_client()
    df = pd.DataFrame(index=MASTER_INDEX)

    flow_pairs = [
        ("NL", "DE_LU", "Flow_NL_DE_MW"),
        ("NL", "BE",    "Flow_NL_BE_MW"),
        ("DE_LU", "NL", "Flow_DE_NL_MW"),   # Reverse direction for validation
        ("NL", "NO_2",  "Flow_NL_NO_MW"),   # v2 (forensic P4): NorNed export leg
        ("NO_2", "NL",  "Flow_NO_NL_MW"),   # v2 (forensic P4): NorNed import leg
    ]

    for from_z, to_z, col in flow_pairs:
        try:
            flows = client.query_scheduled_exchanges(from_z, to_z,
                                                     start=START_UTC, end=END_UTC)
            df[col] = _snap(flows)
            log.info(f"  Flow {from_z}→{to_z}: {df[col].notna().sum():,} rows")
        except Exception as e:
            log.warning(f"  Flow {from_z}→{to_z} failed: {e}")

    # Net interconnector pressure: positive = NL is a net exporter (price will be lower)
    if "Flow_NL_DE_MW" in df and "Flow_NL_BE_MW" in df:
        df["NL_Net_Export_MW"] = df["Flow_NL_DE_MW"] + df["Flow_NL_BE_MW"]

    return df


# ── 4E. IMBALANCE PRICES ─────────────────────────────────────────────────────

def fetch_entsoe_imbalance_prices() -> pd.DataFrame:
    """
    NL imbalance settlement price (15-min PTU resolution).
    NOT a D+1 feature — must be used as lag_96+ only in ML training.
    Included here for historical pattern analysis and lag feature creation.
    """
    client = _entsoe_client()
    df = pd.DataFrame(index=MASTER_INDEX)

    try:
        imb = client.query_imbalance_prices("NL", start=START_UTC, end=END_UTC)
        # ENTSO-E returns a DataFrame with multiple price directions
        if isinstance(imb, pd.DataFrame):
            # 'Price for consumption' is the most relevant for industrial buyers
            col = "Price for consumption" if "Price for consumption" in imb.columns \
                  else imb.columns[0]
            df["NL_Imbalance_Price_EURMWh"] = _snap(imb[col])
        else:
            df["NL_Imbalance_Price_EURMWh"] = _snap(imb)
        log.info(f"  NL imbalance price: {df['NL_Imbalance_Price_EURMWh'].notna().sum():,} rows")
    except Exception as e:
        log.warning(f"  NL imbalance price failed: {e}")

    return df


# ── 4F. THERMAL OUTAGES (REMIT) ───────────────────────────────────────────────

'''def fetch_entsoe_thermal_outages() -> pd.DataFrame:
    """
    Generation unit unavailability (REMIT).
    Fetches NL (Local Scarcity) and FR (European Nuclear Floor)
    and correctly maps start/end intervals to the 15-min master grid.
    """
    client = _entsoe_client()
    df = pd.DataFrame(index=MASTER_INDEX)
    
    zones = {"NL": "NL_Thermal_Outage_MW", "FR": "FR_Nuclear_Outage_MW"}

    for zone, col_name in zones.items():
        df[col_name] = 0.0 # Initialize with zeros
        try:
            outages = client.query_unavailability_of_generation_units(
                zone, start=START_UTC, end=END_UTC, docstatus=None
            )

            if outages is not None and not outages.empty:
                # Parse ENTSO-E columns
                if "nominal_power" in outages.columns and "avail_qty" in outages.columns:
                    
                    outages["nominal_power"] = pd.to_numeric(outages["nominal_power"], errors="coerce")
                    outages["avail_qty"]     = pd.to_numeric(outages["avail_qty"], errors="coerce")
                    
                    if zone == "FR" and "plant_type" in outages.columns:
                        outages = outages[outages["plant_type"].astype(str).str.contains("Nuclear", case=False, na=False)]

                    outages["offline_mw"] = (outages["nominal_power"] - outages["avail_qty"]).clip(lower=0)
                    
                    # 🌟 THE FIX: Map Start/End intervals to the 15-min Master Index
                    temp_series = pd.Series(0.0, index=MASTER_INDEX)
                    
                    for _, row in outages.iterrows():
                        # Ensure timestamps are localized correctly to match UTC Master Index
                        o_start = pd.Timestamp(row['start'])
                        if o_start.tz is None: o_start = o_start.tz_localize('UTC')
                        else: o_start = o_start.tz_convert('UTC')
                        
                        o_end = pd.Timestamp(row['end'])
                        if o_end.tz is None: o_end = o_end.tz_localize('UTC')
                        else: o_end = o_end.tz_convert('UTC')
                        
                        # Add this outage MW to all 15-min buckets it overlaps
                        mask = (temp_series.index >= o_start) & (temp_series.index < o_end)
                        temp_series.loc[mask] += row['offline_mw']

                    df[col_name] = temp_series
                    log.info(f"  REMIT outages {zone}: OK")
                else:
                    log.warning(f"  REMIT {zone}: Missing nominal_power or avail_qty columns.")
            else:
                log.info(f"  REMIT {zone}: No outages reported.")
        except Exception as e:
            log.warning(f"  REMIT outages {zone} failed: {e}")

    return df'''

def fetch_entsoe_thermal_outages() -> pd.DataFrame:
    """
    Generation unit unavailability (REMIT).
    Fetches NL (Local Scarcity) and FR (European Nuclear Floor)
    Uses 1-month chunking to bypass ENTSO-E's 200-document silent limit.
    """
    client = _entsoe_client()
    df = pd.DataFrame(index=MASTER_INDEX)
    
    zones = {"NL": "NL_Thermal_Outage_MW", "FR": "FR_Nuclear_Outage_MW"}

    for zone, col_name in zones.items():
        df[col_name] = 0.0 # Initialize with zeros
        chunks = []
        curr = START_UTC
        
        # 1. Chunking Loop (1 month at a time)
        while curr < END_UTC:
            nxt = min(curr + pd.DateOffset(months=1), END_UTC)
            try:
                outages = client.query_unavailability_of_generation_units(
                    zone, start=curr, end=nxt, docstatus=None
                )
                if outages is not None and not outages.empty:
                    chunks.append(outages)
            except Exception as e:
                # ENTSO-E often throws 404 if literally zero outages occurred that month. We ignore it.
                if "404" not in str(e):
                    log.warning(f"  REMIT {zone} chunk {curr.date()} failed: {e}")
            
            curr = nxt
            time.sleep(1) # Be a good API citizen

        # 2. Merge and Deduplicate
        if chunks:
            all_outages = pd.concat(chunks)
            
            # CRITICAL: Drop duplicates to prevent double-counting outages that span across two chunked months
            all_outages = all_outages.drop_duplicates()
            
            # 3. Parse ENTSO-E columns
            if "nominal_power" in all_outages.columns and "avail_qty" in all_outages.columns:
                
                all_outages["nominal_power"] = pd.to_numeric(all_outages["nominal_power"], errors="coerce")
                all_outages["avail_qty"]     = pd.to_numeric(all_outages["avail_qty"], errors="coerce")
                
                if zone == "FR" and "plant_type" in all_outages.columns:
                    all_outages = all_outages[all_outages["plant_type"].astype(str).str.contains("Nuclear", case=False, na=False)]

                all_outages["offline_mw"] = (all_outages["nominal_power"] - all_outages["avail_qty"]).clip(lower=0)
                
                # 4. Map Start/End intervals to the 15-min Master Index
                temp_series = pd.Series(0.0, index=MASTER_INDEX)
                for _, row in all_outages.iterrows():
                    o_start = pd.Timestamp(row['start'])
                    o_start = o_start.tz_localize('UTC') if o_start.tz is None else o_start.tz_convert('UTC')
                    o_end = pd.Timestamp(row['end'])
                    o_end = o_end.tz_localize('UTC') if o_end.tz is None else o_end.tz_convert('UTC')
                    # Clip to master index range before masking — prevents full-scan on out-of-range rows
                    if o_end < MASTER_INDEX[0] or o_start > MASTER_INDEX[-1]:
                        continue
                    mask = (temp_series.index >= o_start) & (temp_series.index < o_end)
                    temp_series.loc[mask] += row['offline_mw']

                df[col_name] = temp_series
                log.info(f"  REMIT outages {zone}: OK ({len(all_outages)} unique events)")
            else:
                log.warning(f"  REMIT {zone}: Missing nominal_power or avail_qty columns.")
        else:
            log.info(f"  REMIT {zone}: No outages reported across any chunk.")

    log.info(f"DEBUG: Master Index TZ: {MASTER_INDEX.tz}")

    return df


# ── 4G. ACTUAL GENERATION BY TYPE (The Gas Calibration Signal) ─────────────────

def fetch_entsoe_actual_generation() -> pd.DataFrame:
    """
    Fetches actual physical generation by plant type.
    Uses chunking (1 month at a time) to prevent ENTSO-E 503 Server Errors.
    """
    client = _entsoe_client()
    df = pd.DataFrame(index=MASTER_INDEX)
    
    # 1. Helper to safely extract a 1D series from entsoe-py's MultiIndex
    def get_actuals(gen_df, fuel_type):
        if fuel_type in gen_df.columns.get_level_values(0):
            data = gen_df[fuel_type]
            if isinstance(data, pd.DataFrame):
                if 'Actual Aggregated' in data.columns:
                    return data['Actual Aggregated']
                return data.iloc[:, 0]
            return data
        return None

    # 2. Setup Chunking (1 month intervals)
    chunks = []
    curr = START_UTC
    
    while curr < END_UTC:
        nxt = min(curr + pd.DateOffset(months=1), END_UTC)
        log.info(f"    Generation chunk: {curr.date()} → {nxt.date()}")
        
        # 3. Retry logic for temporary server hiccups
        for attempt in range(4):
            try:
                gen_chunk = client.query_generation("NL", start=curr, end=nxt)
                chunks.append(gen_chunk)
                break # Success, exit retry loop
            except Exception as e:
                if "503" in str(e) or "504" in str(e) or "Timeout" in str(e):
                    wait_time = 2 ** attempt * 5
                    log.warning(f"    ENTSO-E busy. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    log.warning(f"    Chunk failed for {curr.date()}: {e}")
                    break # Not a timeout, move to next chunk
                    
        curr = nxt
        time.sleep(1) # Be a good API citizen

    # 4. Merge all successful chunks
    if not chunks:
        log.error("  NL actual generation failed entirely.")
        return df
        
    master_gen = pd.concat(chunks)
    master_gen = master_gen[~master_gen.index.duplicated(keep='first')]

    # 5. Extract specific fuels safely
    gas_series = get_actuals(master_gen, "Fossil Gas")
    if gas_series is not None:
        df["NL_Fossil_Gas_Actual_MW"] = _snap(gas_series)
        
    coal_series = get_actuals(master_gen, "Fossil Hard coal")
    if coal_series is not None:
        df["NL_Coal_Actual_MW"] = _snap(coal_series)
        
    log.info(f"  NL actual generation (Gas/Coal): OK")

    return df

# ── 4H. WRAPPER: FETCH ALL ENTSO-E ───────────────────────────────────────────


def fetch_entsoe_transmission_outages() -> pd.DataFrame:
    """
    v2 (forensic P6): interconnector (transmission) unavailability under REMIT
    for the NL borders. NTC reductions decouple NL from its neighbours and
    shift the spread regime. Forward-published -> legitimately usable unlagged.
    """
    client = _entsoe_client()
    df = pd.DataFrame(index=MASTER_INDEX)
    total = pd.Series(0.0, index=MASTER_INDEX)
    fn = getattr(client, "query_unavailability_transmission", None) or \
         getattr(client, "query_unavailability_of_transmission", None)
    if fn is None:
        log.warning("  entsoe-py lacks the transmission-unavailability endpoint - IC_Outage_MW = 0")
        df["IC_Outage_MW"] = total
        return df

    for fz, tz_ in [("NL", "DE_LU"), ("NL", "BE"), ("NL", "NO_2"), ("NL", "GB")]:
        curr = START_UTC
        while curr < END_UTC:
            nxt = min(curr + pd.DateOffset(months=1), END_UTC)
            try:
                r = fn(fz, tz_, start=curr, end=nxt)
                if r is not None and not r.empty and {"start", "end"} <= set(r.columns):
                    qty = pd.to_numeric(r.get("nominal_power", pd.Series(dtype=float)),
                                        errors="coerce").fillna(500.0)
                    for (_, row), q in zip(r.iterrows(), qty):
                        t0 = pd.Timestamp(row["start"]); t0 = t0.tz_localize("UTC") if t0.tz is None else t0.tz_convert("UTC")
                        t1 = pd.Timestamp(row["end"]);   t1 = t1.tz_localize("UTC") if t1.tz is None else t1.tz_convert("UTC")
                        total.loc[(total.index >= t0) & (total.index < t1)] += q
            except Exception as e:
                # Silently ignore the month if there are no outages
                if type(e).__name__ == "NoMatchingDataError":
                    continue
                
                # Log any other actual errors
                if "404" not in str(e):
                    log.warning(f"  TX REMIT {fz}->{tz_} chunk {curr.date()}: {e}")
            curr = nxt
            time.sleep(1)

    df["IC_Outage_MW"] = total
    log.info(f"  IC outage: max {total.max():,.0f} MW | nonzero PTUs {(total > 0).sum():,}")
    return df


def fetch_all_entsoe() -> pd.DataFrame:
    """
    Orchestrates all ENTSO-E fetchers with individual caching.
    Each sub-fetch is cached independently — if one fails on re-run,
    only that sub-fetch is retried, not the entire pipeline.
    """
    fetchers = [
        ("entsoe_nl_da_price",          fetch_entsoe_nl_da_price),
        ("entsoe_neighbor_da_prices",   fetch_entsoe_neighbor_da_prices),
        ("entsoe_load_data",            fetch_entsoe_load_data),
        #("entsoe_generation_forecasts", fetch_entsoe_generation_forecasts),
        ("entsoe_crossborder_flows",    fetch_entsoe_crossborder_flows),
        ("entsoe_imbalance_prices",     fetch_entsoe_imbalance_prices),
        ("entsoe_thermal_outages",      fetch_entsoe_thermal_outages), # 👈 MUST BE HERE
        ("entsoe_transmission_outages", fetch_entsoe_transmission_outages), # v2 (forensic P6)
        ("entsoe_actual_generation",    fetch_entsoe_actual_generation), # 🌟 NEW INCLUSION
    ]

    frames = []
    for cache_name, fn in fetchers:
        df_part = fetch_or_cache(cache_name, fn)
        frames.append(df_part)

    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=MASTER_INDEX)

# ================================================================================
# 5. OPEN-METEO WEATHER FETCHER — bulk, all variables, 7 NL cities
# ================================================================================

# Variables that matter for DA price forecasting
# Note: precipitation & weather_code added vs your original — covers demand spikes
WEATHER_VARIABLES = [
    "temperature_2m",
    "apparent_temperature",
    "shortwave_radiation",
    "direct_radiation",      # NEW: Direct sunlight (hits max panel efficiency)
    "diffuse_radiation",     # NEW: Scattered light (maintains baseload solar on cloudy days)
    "wind_speed_100m",
    "wind_direction_100m",   # Direction matters: SW wind = offshore wind power
    "cloud_cover",
    "relative_humidity_2m",
    "precipitation",         # Rain/snow spikes heating demand unexpectedly
    "weather_code",          # WMO code: extreme weather classification
]


def fetch_weather_chunk(cities: list, lats: list, lons: list,
                        chunk_start: str, chunk_end: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   ",".join(lats),
        "longitude":  ",".join(lons),
        "start_date": chunk_start,
        "end_date":   chunk_end,
        "hourly":     ",".join(WEATHER_VARIABLES),
        "timezone":   "UTC",
        "wind_speed_unit": "ms",
    }

    for attempt in range(5): # Increased attempts slightly
        try:
            r = requests.get(url, params=params, timeout=60)
            resp = r.json()

            if isinstance(resp, dict) and "error" in resp:
                error_msg = resp.get("reason", "")
                # If we hit the rate limit, wait a full minute
                if "limit exceeded" in error_msg.lower():
                    log.warning(f"  [RATE LIMIT] Waiting 60s to reset Open-Meteo bucket...")
                    time.sleep(62) # 62s to be safe
                    continue # Retry this chunk
                
                log.warning(f"Open-Meteo error: {error_msg}")
                return pd.DataFrame()

            city_frames = []
            for i, city in enumerate(cities):
                city_data = resp[i] if isinstance(resp, list) else resp
                if "hourly" not in city_data: continue
                df_city = pd.DataFrame(city_data["hourly"])
                df_city["time"] = pd.to_datetime(df_city["time"]).dt.tz_localize("UTC")
                df_city.set_index("time", inplace=True)
                df_city.columns = [f"{col}_{city}" for col in df_city.columns]
                city_frames.append(df_city)

            if city_frames: return pd.concat(city_frames, axis=1)
            return pd.DataFrame()

        except Exception as e:
            log.warning(f"Open-Meteo attempt {attempt+1} failed: {e}")
            time.sleep(5) # Standard retry delay

    return pd.DataFrame()


def fetch_all_weather() -> pd.DataFrame:
    """
    Fetches weather for all NL cities in 6-month chunks to respect Open-Meteo limits.
    Upsamples from hourly to 15-min with forward-fill, then aligns to master index.
    """
    cities = list(NL_WEATHER_STATIONS.keys())
    lats   = [str(NL_WEATHER_STATIONS[c]["lat"]) for c in cities]
    lons   = [str(NL_WEATHER_STATIONS[c]["lon"]) for c in cities]

    chunks = []
    curr = START_UTC

    while curr < END_UTC:
        nxt         = min(curr + pd.DateOffset(months=6), END_UTC)
        chunk_start = (curr - pd.DateOffset(days=1)).strftime("%Y-%m-%d")  # 1-day pad for TZ
        chunk_end   = nxt.strftime("%Y-%m-%d")

        log.info(f"  Weather chunk: {chunk_start} → {chunk_end}")
        df_chunk = fetch_weather_chunk(cities, lats, lons, chunk_start, chunk_end)

        if not df_chunk.empty:
            chunks.append(df_chunk)

        curr = nxt
        time.sleep(1.5)  # Be a good API citizen

    if not chunks:
        log.error("Open-Meteo: no data retrieved")
        return pd.DataFrame(index=MASTER_INDEX)

    df_weather = pd.concat(chunks, axis=0)
    df_weather = df_weather[~df_weather.index.duplicated(keep="first")]

    # Upsample hourly → 15-min, align to master grid
    df_weather = df_weather.resample("15min").ffill().reindex(MASTER_INDEX)

    log.info(f"  Weather complete: {df_weather.shape[1]} columns, "
             f"{df_weather.notna().mean().mean()*100:.1f}% fill rate")
    return df_weather

def fetch_de_weather() -> pd.DataFrame:
    """
    Fetches weather for strategic German cities using the exact same logic as NL.
    Saved to an independent cache to prevent breaking existing NL data.
    """
    cities = list(DE_WEATHER_STATIONS.keys())
    lats   = [str(DE_WEATHER_STATIONS[c]["lat"]) for c in cities]
    lons   = [str(DE_WEATHER_STATIONS[c]["lon"]) for c in cities]

    chunks = []
    curr = START_UTC

    while curr < END_UTC:
        nxt         = min(curr + pd.DateOffset(months=6), END_UTC)
        chunk_start = (curr - pd.DateOffset(days=1)).strftime("%Y-%m-%d")  # 1-day pad for TZ
        chunk_end   = nxt.strftime("%Y-%m-%d")

        log.info(f"  DE Weather chunk: {chunk_start} → {chunk_end}")
        df_chunk = fetch_weather_chunk(cities, lats, lons, chunk_start, chunk_end)

        if not df_chunk.empty:
            chunks.append(df_chunk)

        curr = nxt
        time.sleep(1.5)  # Be a good API citizen

    if not chunks:
        log.error("Open-Meteo DE: no data retrieved")
        return pd.DataFrame(index=MASTER_INDEX)

    df_weather = pd.concat(chunks, axis=0)
    df_weather = df_weather[~df_weather.index.duplicated(keep="first")]

    # Upsample hourly → 15-min, align to master grid
    df_weather = df_weather.resample("15min").ffill().reindex(MASTER_INDEX)

    log.info(f"  DE Weather complete: {df_weather.shape[1]} columns, "
             f"{df_weather.notna().mean().mean()*100:.1f}% fill rate")
    return df_weather


# ================================================================================
# 6. MACRO FETCHER — TTF Gas, EUA Carbon (Yahoo Finance)
# ================================================================================

def fetch_eua_carbon_price(yf_start: str, yf_end: str) -> pd.Series:
    # H1 Fix: Drop ICLN, add CARB.PA and XCO2.PA
    for ticker in ["CO2.L", "CARB.PA", "XCO2.PA"]:
        try:
            raw = yf.download(ticker, start=yf_start, end=yf_end, progress=False, auto_adjust=True)
            if not raw.empty:
                close = raw["Close"].squeeze() if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
                mean_price = float(close.mean())
                if 45.0 <= mean_price <= 100.0:
                    close.index = pd.to_datetime(close.index)
                    close.index = close.index.tz_localize("UTC") if close.index.tz is None else close.index.tz_convert("UTC")
                    return close.reindex(MASTER_INDEX).ffill().bfill()
        except: pass

    EUA_REF = 62.14 # Jan 2024
    try:
        raw_ref = yf.download("CARB.PA", start="2024-01-10", end="2024-01-20", progress=False, auto_adjust=True)
        if not raw_ref.empty:
            ref_close = raw_ref["Close"].squeeze() if isinstance(raw_ref.columns, pd.MultiIndex) else raw_ref["Close"]
            scale_factor = EUA_REF / float(ref_close.mean())
            if 1.5 <= scale_factor <= 5.0:
                raw_full = yf.download("CARB.PA", start=yf_start, end=yf_end, progress=False, auto_adjust=True)
                if not raw_full.empty:
                    close = (raw_full["Close"].squeeze() if isinstance(raw_full.columns, pd.MultiIndex) else raw_full["Close"]) * scale_factor
                    close.index = pd.to_datetime(close.index)
                    close.index = close.index.tz_localize("UTC") if close.index.tz is None else close.index.tz_convert("UTC")
                    return close.reindex(MASTER_INDEX).ffill().bfill()
    except: pass

    log.warning("Using hardcoded EEX curve.")
    EUA_CURVE = {"2026-01": 65.0, "2026-04": 64.0, "2026-05": 68.5} # Update monthly!
    monthly = pd.Series(list(EUA_CURVE.values()), index=pd.to_datetime(list(EUA_CURVE.keys()))).sort_index()
    monthly.index = monthly.index.tz_localize("UTC")
    return monthly.reindex(MASTER_INDEX, method="ffill").ffill().bfill()

def fetch_macro_prices() -> pd.DataFrame:
    df = pd.DataFrame(index=MASTER_INDEX)
    yf_s, yf_e = START_UTC.strftime("%Y-%m-%d"), END_UTC.strftime("%Y-%m-%d")

    try:
        raw = yf.download("TTF=F", start=yf_s, end=yf_e, progress=False, auto_adjust=True)
        if not raw.empty:
            c = raw["Close"].squeeze() if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
            c.index = pd.to_datetime(c.index).tz_localize("UTC") if c.index.tz is None else pd.to_datetime(c.index).tz_convert("UTC")
            df["TTF_Gas_EURMWh"] = c.reindex(MASTER_INDEX).ffill().bfill()
    except: pass

    df["EUA_Carbon_EUR"] = fetch_eua_carbon_price(yf_s, yf_e)

    if "TTF_Gas_EURMWh" in df.columns and "EUA_Carbon_EUR" in df.columns:
        df["CCGT_Marginal_Cost_EUR"] = (df["TTF_Gas_EURMWh"] / 0.56) + (df["EUA_Carbon_EUR"] * 0.36)

    return df


# ================================================================================
# 7. DATA INTEGRITY REPORT
# ================================================================================

def run_integrity_report(df: pd.DataFrame, target_col: str = "DA_Price_NL_EURMWh"):
    """
    Runs after the master merge. Prints a structured quality report.
    Flags: missing timestamps, high NaN rates, suspicious correlations,
    and potential leakage columns.
    """
    print("\n" + "=" * 65)
    print("  VOLTCAST IPI — DATA INTEGRITY REPORT")
    print("=" * 65)
    print(f"  Shape     : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Time range: {df.index.min()} → {df.index.max()}")
    print(f"  Frequency : 15-minute UTC\n")

    # ── Check 1: timestamp continuity ──
    expected = pd.date_range(df.index.min(), df.index.max(), freq="15min")
    missing_ts = expected.difference(df.index)
    if missing_ts.empty:
        print("  [✓] Timestamp continuity: PASS — no missing intervals")
    else:
        print(f"  [✗] Timestamp continuity: FAIL — {len(missing_ts):,} missing intervals")
        print(f"       First missing: {missing_ts[0]}")

    # ── Check 2: NaN rates ──
    print("\n  NaN rates per column:")
    nan_rates = (df.isna().mean() * 100).sort_values(ascending=False)
    high_nan  = nan_rates[nan_rates > 5.0]
    ok_nan    = nan_rates[nan_rates <= 5.0]

    if not high_nan.empty:
        print(f"  {'Column':<40} {'NaN %':>7}")
        print("  " + "-" * 50)
        for col, pct in high_nan.items():
            flag = "⚠️ " if pct > 20 else "  "
            print(f"  {flag}{col:<38} {pct:>6.1f}%")

    print(f"\n  {len(ok_nan)} columns have < 5% NaN (OK)")

    # ── Check 3: Target variable ──
    if target_col in df.columns:
        t = df[target_col].dropna()
        print(f"\n  Target: {target_col}")
        print(f"    Min   : €{t.min():.2f}/MWh")
        print(f"    Max   : €{t.max():.2f}/MWh")
        print(f"    Mean  : €{t.mean():.2f}/MWh")
        print(f"    Zeros : {(t == 0).sum():,}  (zero-price hours — expected in solar oversupply)")
        print(f"    Neg   : {(t < 0).sum():,}  (negative-price hours — expected in high wind)")
    else:
        print(f"\n  [✗] Target column '{target_col}' NOT FOUND")

    # ── Check 4: Leakage warning ──
    print("\n  Leakage audit — checking concurrent actuals vs target:")
    LEAKAGE_RISK_COLS = [
        "NL_Actual_Load_MW",        # Must use lag_96+ in ML
        "NL_Imbalance_Price_EURMWh", # Settles after DA gate closure
        "NL_Load_MW",               # NED actual — lag_96+ only
        "NL_Solar_MW",              # NED actual — lag_96+ only
        "NL_Wind_MW",               # NED actual — lag_96+ only
    ]

    for col in LEAKAGE_RISK_COLS:
        if col in df.columns:
            print(f"  [!] {col} — concurrent actual. MUST lag_96 before ML training.")

    # ── Check 5: Feature completeness score ──
    CRITICAL_FEATURES = [
        "DA_Price_NL_EURMWh", "DA_Price_DE_EURMWh", "DA_Price_BE_EURMWh",
        "NL_TSO_Load_Forecast_MW", "TTF_Gas_EURMWh",
        "EUA_Carbon_EUR", "CCGT_Marginal_Cost_EUR",
        "NL_Net_Export_MW", "NL_Thermal_Outage_MW",
        "temperature_2m_Amsterdam",
    ]
    present  = [c for c in CRITICAL_FEATURES if c in df.columns]
    missing  = [c for c in CRITICAL_FEATURES if c not in df.columns]
    score    = len(present) / len(CRITICAL_FEATURES) * 100

    print(f"\n  Feature completeness: {score:.0f}%  ({len(present)}/{len(CRITICAL_FEATURES)} critical features)")
    if missing:
        print(f"  Missing critical: {', '.join(missing)}")
    
    # ── Check 6: Macro Scaling Audit ──
    print("\n  Macro Data Scaling Audit:")
    if "EUA_Carbon_EUR" in df.columns:
        mean_carbon = df["EUA_Carbon_EUR"].mean()
        print(f"    EUA Carbon Mean : €{mean_carbon:.2f}")
        if mean_carbon < 50.0:
            print("    [!] WARNING: Carbon price seems too low. Check ETF multiplier.")
        elif mean_carbon > 120.0:
            print("    [!] WARNING: Carbon price seems too high.")
        else:
            print("    [✓] Carbon scale: PASS")
            
    if "TTF_Gas_EURMWh" in df.columns:
        mean_gas = df["TTF_Gas_EURMWh"].mean()
        print(f"    TTF Gas Mean    : €{mean_gas:.2f}")

    print("\n" + "=" * 65)
    print("  REPORT COMPLETE")
    print("=" * 65 + "\n")


# ================================================================================
# 8. MASTER PIPELINE — entry point
# ================================================================================

if __name__ == "__main__":

    log.info("=" * 65)
    log.info("  VOLTCAST IPI — MASTER DATA FETCH PIPELINE")
    log.info("=" * 65)

    # ── STEP 1: NED actuals ──────────────────────────────────────────
    log.info("\n[STEP 1/5]  NED.nl — Solar, Wind, Load actuals")
    df_ned = fetch_or_cache("ned_all", fetch_all_ned, NED_API_KEY)
    log.info(f"  NED shape: {df_ned.shape}")

    # ── STEP 2: ENTSO-E all signals ──────────────────────────────────
    log.info("\n[STEP 2/5]  ENTSO-E — Prices, Forecasts, Flows, Outages")
    df_entsoe = fetch_or_cache("entsoe_all", fetch_all_entsoe)
    log.info(f"  ENTSO-E shape: {df_entsoe.shape}")

    # ── STEP 3: Open-Meteo weather ───────────────────────────────────
    log.info("\n[STEP 3/5]  Open-Meteo — Weather (7 NL cities)")
    df_weather = fetch_or_cache("weather_all", fetch_all_weather)
    log.info(f"  NL Weather shape: {df_weather.shape}")

    # --- MANDATORY COOL-OFF ---
    log.info("  Pausing 30s to respect Open-Meteo minutely limits...")
    time.sleep(30)

    log.info("\n[STEP 3B/5] Open-Meteo — Weather (6 DE cities)")
    df_weather_de = fetch_or_cache("weather_de", fetch_de_weather)
    log.info(f"  DE Weather shape: {df_weather_de.shape}")

    # ── STEP 4: Macro (TTF Gas, EUA Carbon) ─────────────────────────
    log.info("\n[STEP 4/5]  Yahoo Finance — TTF Gas, EUA Carbon")
    df_macro = fetch_or_cache("macro_all", fetch_macro_prices)
    log.info(f"  Macro shape: {df_macro.shape}")

    # ── STEP 5: Master merge ──
    log.info("\n[STEP 5/5]  Master merge → VoltCast_IPI_Master_v2.parquet")

    # 1. Join all dataframes
    df_master = pd.concat([df_ned, df_entsoe, df_weather, df_weather_de, df_macro], axis=1)

    # 2. Reindex to the full requested timeline
    df_master = df_master.reindex(MASTER_INDEX)

    # 3. THE CRITICAL FIX: Thermal outages MUST be 0.0, not NaN
    outage_cols = [c for c in df_master.columns if "Outage_MW" in c]
    df_master[outage_cols] = df_master[outage_cols].fillna(0.0)

    # 4. Forward-fill other sensors (Weather/Prices/Forecasts)
    # We don't fill these with 0 because a NaN temperature isn't 0 degrees!
    df_master = df_master.ffill(limit=4)
    
    # Save
    output_path = "VoltCast_IPI_Master_v2.parquet"
    df_master.to_parquet(output_path)

    log.info(f"\n  Saved: {output_path}")
    log.info(f"  Final shape: {df_master.shape[0]:,} rows × {df_master.shape[1]} columns")

    # ── INTEGRITY REPORT ─────────────────────────────────────────────
    run_integrity_report(df_master)

    log.info("\n" + "=" * 65)