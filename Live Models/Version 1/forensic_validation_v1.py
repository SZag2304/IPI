"""
================================================================================
 VOLTCAST IPI — FORENSIC VALIDATION & MISSING-FEATURE ATTRIBUTION (v1)
 Purpose : Post-mortem analysis of live forecast performance vs realised EPEX
           NL prices for the window 2026-05-17 -> today, at HOURLY resolution.
           Quantifies WHERE the model errs (hour/day/regime), WHY it errs
           (driver attribution via correlation screen + residual error model),
           and WHICH data/features are missing (8 explicit probes, incl.
           sources the live pipeline does NOT fetch: NorNed, BritNed, DE wind,
           transmission REMIT, ENTSO-E RES forecasts as benchmark).
 Outputs : voltcast_ipi_forensics_v1/
             forensic_master_hourly.parquet / .csv   (analysis frame)
             daily_scorecard.csv                     (per-delivery-day metrics)
             driver_correlations.csv                 (error vs driver screen)
             error_model_importances.csv             (residual XGB importances)
             charts/*.png                            (5 report figures)
             FORENSIC_REPORT.md                      (PhD-ready technical report)
 Run     : python forensic_validation_v1.py [--start 2026-05-17] [--force-refresh]
 Notes   : Read-only analysis. Does not touch the live cron pipeline.
           Conventions match live_fetch_v1.py (env keys, Dutch TZ, parquet cache).
================================================================================
"""

import os
import sys
import json
import glob
import time
import logging
import warnings
import argparse
import requests
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# ================================================================================
# 0. CONFIGURATION
# ================================================================================

DUTCH_TZ = "Europe/Amsterdam"

parser = argparse.ArgumentParser()
parser.add_argument("--start", default="2026-05-17", help="Analysis start date (local), default 2026-05-17")
parser.add_argument("--end",   default=None,         help="Analysis end date (local), default = yesterday (last settled delivery day)")
parser.add_argument("--force-refresh", action="store_true", help="Ignore cached parquets and refetch all sources")
parser.add_argument("--pred-dir", default="voltcast_ipi_live_v1", help="Directory holding predictions_*.json artefacts")
ARGS = parser.parse_args()

now_dutch   = pd.Timestamp.now(tz=DUTCH_TZ)
START_LOCAL = pd.Timestamp(ARGS.start, tz=DUTCH_TZ)
# Default end = yesterday: the last delivery day with a settled EPEX price.
END_LOCAL   = (pd.Timestamp(ARGS.end, tz=DUTCH_TZ) if ARGS.end
               else pd.Timestamp(now_dutch.date(), tz=DUTCH_TZ))
START_UTC   = START_LOCAL.tz_convert("UTC")
END_UTC     = END_LOCAL.tz_convert("UTC")          # exclusive
HOURLY_IDX  = pd.date_range(START_UTC, END_UTC, freq="h", inclusive="left")

FORENSIC_DIR = "voltcast_ipi_forensics_v1"
CACHE_DIR    = os.path.join(FORENSIC_DIR, "cache")
CHART_DIR    = os.path.join(FORENSIC_DIR, "charts")
for d in (FORENSIC_DIR, CACHE_DIR, CHART_DIR):
    os.makedirs(d, exist_ok=True)

PRED_DIRS = [ARGS.pred_dir, ARGS.pred_dir + "_evening"]   # morning is primary; evening optional

ENTSOE_API_KEY = os.environ.get("VOLTCAST_ENTSOE_KEY")
if not ENTSOE_API_KEY:
    raise EnvironmentError("VOLTCAST_ENTSOE_KEY must be set (same .env as live pipeline).")

SPIKE_THRESHOLD_EUR = 120.0
PEAK_TOP_N          = 4

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S",
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(os.path.join(FORENSIC_DIR, "forensic.log"), mode="w")])
log = logging.getLogger("VoltCast.Forensic")

# Station sets — identical to live_fetch_v1.py for parity
NL_STATIONS = {
    "Amsterdam":  (52.3740, 4.8897, 0.22), "Rotterdam": (51.9225, 4.4792, 0.22),
    "Utrecht":    (52.0908, 5.1222, 0.18), "Eindhoven": (51.4408, 5.4778, 0.18),
    "Maastricht": (50.8483, 5.6889, 0.08), "Deventer":  (52.2550, 6.1639, 0.06),
    "Friesland":  (53.2012, 5.7999, 0.06),  # Leeuwarden proxy
}
DE_STATIONS = {
    "Hamburg": (53.5511, 9.9937), "Bremen": (53.0793, 8.8017), "Kiel": (54.3233, 10.1394),
    "Munich":  (48.1351, 11.5820), "Stuttgart": (48.7758, 9.1829), "Freiburg": (47.9990, 7.8421),
}
WEATHER_VARS = ["temperature_2m", "apparent_temperature", "shortwave_radiation",
                "direct_radiation", "diffuse_radiation", "wind_speed_100m",
                "cloud_cover", "relative_humidity_2m", "precipitation"]


# ================================================================================
# 1. CACHED FETCH HELPERS
# ================================================================================

def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}_{START_LOCAL.date()}_{END_LOCAL.date()}.parquet")

def cached(name: str):
    """Decorator: parquet-cache a fetcher returning a DataFrame on HOURLY_IDX."""
    def deco(fn):
        def wrapper(*a, **k):
            p = _cache_path(name)
            if os.path.exists(p) and not ARGS.force_refresh:
                log.info(f"  [cache] {name}")
                return pd.read_parquet(p)
            df = fn(*a, **k)
            if df is not None and not df.empty:
                df.to_parquet(p)
            return df if df is not None else pd.DataFrame(index=HOURLY_IDX)
        return wrapper
    return deco

def to_hourly(obj) -> pd.Series:
    """Resample any series/frame-col to hourly mean on HOURLY_IDX (UTC)."""
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    obj.index = pd.to_datetime(obj.index)
    if obj.index.tz is None:
        obj.index = obj.index.tz_localize("UTC")
    else:
        obj.index = obj.index.tz_convert("UTC")
    return obj.resample("h").mean().reindex(HOURLY_IDX)

def _retry(fn, label, tries=4, base=5):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            if attempt < tries - 1 and "404" not in str(e):
                time.sleep(base * (2 ** attempt))
            else:
                log.warning(f"  {label} failed: {e}")
                return None


# ================================================================================
# 2. ENTSO-E ACTUALS (post-mortem: contemporaneous data is legitimate here)
# ================================================================================

@cached("entsoe_actuals")
def fetch_entsoe_actuals() -> pd.DataFrame:
    from entsoe import EntsoePandasClient
    client = EntsoePandasClient(api_key=ENTSOE_API_KEY)
    df = pd.DataFrame(index=HOURLY_IDX)
    s, e = START_UTC, END_UTC

    # --- DA prices: NL target + neighbours + NO_2 (NOT in live pipeline -> probe P4) ---
    for zone, col in [("NL", "actual_price"), ("DE_LU", "de_price"), ("BE", "be_price"),
                      ("FR", "fr_price"), ("SE_3", "se3_price"), ("NO_2", "no2_price")]:
        r = _retry(lambda z=zone: client.query_day_ahead_prices(z, start=s, end=e), f"DA {zone}")
        if r is not None:
            df[col] = to_hourly(r)

    # --- Load: actual + TSO D+1 forecast (-> TSO bias probe P8) ---
    r = _retry(lambda: client.query_load("NL", start=s, end=e), "NL load")
    if r is not None:
        df["nl_actual_load"] = to_hourly(r)
    r = _retry(lambda: client.query_load_forecast("NL", start=s, end=e), "NL load fc")
    if r is not None:
        df["nl_tso_load_fc"] = to_hourly(r)

    # --- Actual renewable generation (-> Physics Bridge fidelity probe P1) ---
    r = _retry(lambda: client.query_generation("NL", start=s, end=e), "NL generation")
    if r is not None and not r.empty:
        def col_of(gdf, name):
            if name in gdf.columns.get_level_values(0):
                sub = gdf[name]
                return sub["Actual Aggregated"] if isinstance(sub, pd.DataFrame) and "Actual Aggregated" in sub else (sub.iloc[:, 0] if isinstance(sub, pd.DataFrame) else sub)
            return None
        won, wof = col_of(r, "Wind Onshore"), col_of(r, "Wind Offshore")
        sol      = col_of(r, "Solar")
        gas      = col_of(r, "Fossil Gas")
        if won is not None or wof is not None:
            wind = (to_hourly(won) if won is not None else 0)
            wind = wind + (to_hourly(wof) if wof is not None else 0)
            df["nl_wind_actual_mw"] = wind
        if sol is not None:
            df["nl_solar_actual_mw"] = to_hourly(sol)
        if gas is not None:
            df["nl_gas_actual_mw"] = to_hourly(gas)

    # --- ENTSO-E D+1 RES forecasts NL & DE (the feed you REMOVED -> benchmark, probes P1/P2) ---
    for zone, pre in [("NL", "nl"), ("DE_LU", "de")]:
        r = _retry(lambda z=zone: client.query_wind_and_solar_forecast(z, start=s, end=e), f"RES fc {zone}")
        if r is not None and not r.empty:
            cols = {c: c for c in r.columns}
            if "Solar" in cols:
                df[f"{pre}_solar_fc_mw"] = to_hourly(r["Solar"])
            wind_cols = [c for c in r.columns if "Wind" in str(c)]
            if wind_cols:
                df[f"{pre}_wind_fc_mw"] = to_hourly(r[wind_cols].sum(axis=1))

    # --- Flows: existing pair + NorNed + BritNed (latter two NOT in live pipeline) ---
    for fz, tz_, col in [("NL", "DE_LU", "flow_nl_de"), ("NL", "BE", "flow_nl_be"),
                         ("NL", "NO_2", "flow_nl_no"), ("NO_2", "NL", "flow_no_nl"),
                         ("NL", "GB", "flow_nl_gb"),   ("GB", "NL", "flow_gb_nl")]:
        r = _retry(lambda a=fz, b=tz_: client.query_scheduled_exchanges(a, b, start=s, end=e), f"flow {fz}->{tz_}", tries=2)
        if r is not None:
            df[col] = to_hourly(r)

    # --- Imbalance ---
    r = _retry(lambda: client.query_imbalance_prices("NL", start=s, end=e), "NL imbalance", tries=2)
    if r is not None:
        try:
            ser = r["Price for consumption"] if isinstance(r, pd.DataFrame) and "Price for consumption" in r.columns else r
            df["nl_imbalance_price"] = to_hourly(ser)
        except Exception:
            pass

    # --- REMIT generation outages NL + FR nuclear (parity with live) ---
    for zone, col in [("NL", "nl_thermal_outage_mw"), ("FR", "fr_nuclear_outage_mw")]:
        r = _retry(lambda z=zone: client.query_unavailability_of_generation_units(z, start=s, end=e, docstatus=None),
                   f"REMIT gen {zone}", tries=2)
        ser = pd.Series(0.0, index=HOURLY_IDX)
        if r is not None and not r.empty and {"nominal_power", "avail_qty", "start", "end"} <= set(r.columns):
            r = r.copy()
            if zone == "FR" and "plant_type" in r.columns:
                r = r[r["plant_type"].astype(str).str.contains("Nuclear", case=False, na=False)]
            r["nominal_power"] = pd.to_numeric(r["nominal_power"], errors="coerce")
            r["avail_qty"]     = pd.to_numeric(r["avail_qty"], errors="coerce")
            r["off"] = (r["nominal_power"] - r["avail_qty"]).clip(lower=0)
            for _, row in r.iterrows():
                t0 = pd.Timestamp(row["start"]); t0 = t0.tz_localize("UTC") if t0.tz is None else t0.tz_convert("UTC")
                t1 = pd.Timestamp(row["end"]);   t1 = t1.tz_localize("UTC") if t1.tz is None else t1.tz_convert("UTC")
                ser.loc[(ser.index >= t0) & (ser.index < t1)] += row["off"]
        df[col] = ser

    # --- Transmission (interconnector) unavailability — NOT in live pipeline (probe P6) ---
    tx_total = pd.Series(0.0, index=HOURLY_IDX); tx_found = False
    for fz, tz_ in [("NL", "DE_LU"), ("NL", "BE"), ("NL", "NO_2"), ("NL", "GB")]:
        fn = getattr(client, "query_unavailability_transmission", None) or \
             getattr(client, "query_unavailability_of_transmission", None)
        if fn is None:
            break
        r = _retry(lambda a=fz, b=tz_: fn(a, b, start=s, end=e), f"REMIT tx {fz}->{tz_}", tries=2)
        if r is not None and not r.empty and {"start", "end"} <= set(r.columns):
            tx_found = True
            qty = pd.to_numeric(r.get("nominal_power", pd.Series(500.0, index=r.index)), errors="coerce").fillna(500.0)
            for (_, row), q in zip(r.iterrows(), qty):
                t0 = pd.Timestamp(row["start"]); t0 = t0.tz_localize("UTC") if t0.tz is None else t0.tz_convert("UTC")
                t1 = pd.Timestamp(row["end"]);   t1 = t1.tz_localize("UTC") if t1.tz is None else t1.tz_convert("UTC")
                tx_total.loc[(tx_total.index >= t0) & (tx_total.index < t1)] += q
    if tx_found:
        df["interconnector_outage_mw"] = tx_total

    log.info(f"  ENTSO-E actuals: {df.shape[1]} columns")
    return df


# ================================================================================
# 3. WEATHER (Open-Meteo, single past_days call; reconstruct Physics Bridge)
# ================================================================================

@cached("weather")
def fetch_weather() -> pd.DataFrame:
    past_days = min(92, (now_dutch.normalize() - START_LOCAL.normalize()).days + 1)
    allst = {**{k: v[:2] for k, v in NL_STATIONS.items()}, **DE_STATIONS}
    cities = list(allst.keys())
    r = requests.get("https://api.open-meteo.com/v1/forecast",
                     params={"latitude": ",".join(str(allst[c][0]) for c in cities),
                             "longitude": ",".join(str(allst[c][1]) for c in cities),
                             "hourly": ",".join(WEATHER_VARS),
                             "timezone": "UTC", "wind_speed_unit": "ms",
                             "past_days": past_days, "forecast_days": 1},
                     timeout=90)
    resp = r.json()
    if not isinstance(resp, list):
        resp = [resp]
    frames = []
    for i, city in enumerate(cities):
        if i >= len(resp) or "hourly" not in resp[i]:
            continue
        d = pd.DataFrame(resp[i]["hourly"])
        d["time"] = pd.to_datetime(d["time"]).dt.tz_localize("UTC")
        d = d.set_index("time")
        d.columns = [f"{c}_{city}" for c in d.columns]
        frames.append(d)
    if not frames:
        return pd.DataFrame(index=HOURLY_IDX)
    full_idx = pd.date_range(min(f.index.min() for f in frames),
                             max(f.index.max() for f in frames), freq="h", tz="UTC")
    df = pd.concat([f.reindex(full_idx) for f in frames], axis=1).reindex(HOURLY_IDX)
    log.info(f"  Weather: {df.shape[1]} columns")
    return df


def derive_physics_bridge(df_w: pd.DataFrame) -> pd.DataFrame:
    """Recompute NL Physics Bridge proxies with TRAINING constants for parity,
    plus the DE wind aggregate that the live bridge does NOT include (probe P2)."""
    out = pd.DataFrame(index=HOURLY_IDX)
    tw = sum(w for *_, w in NL_STATIONS.values())
    for var in ["temperature_2m", "apparent_temperature", "shortwave_radiation",
                "direct_radiation", "diffuse_radiation", "wind_speed_100m", "cloud_cover"]:
        cols = [(f"{var}_{c}", w) for c, (_, _, w) in NL_STATIONS.items() if f"{var}_{c}" in df_w.columns]
        if cols:
            out[f"nat_{var}"] = sum(df_w[c] * (w / tw) for c, w in cols)
    if "nat_wind_speed_100m" in out:
        out["wind_power_proxy_cubed"] = out["nat_wind_speed_100m"].clip(0, 25) ** 3
    if "nat_shortwave_radiation" in out:
        out["solar_proxy"] = out["nat_shortwave_radiation"]
    # DE aggregates (simple mean — not in live bridge)
    de_ws = [f"wind_speed_100m_{c}" for c in DE_STATIONS if f"wind_speed_100m_{c}" in df_w.columns]
    de_rd = [f"shortwave_radiation_{c}" for c in DE_STATIONS if f"shortwave_radiation_{c}" in df_w.columns]
    if de_ws:
        out["de_wind_speed_mean"] = df_w[de_ws].mean(axis=1)
        out["de_wind_proxy_cubed"] = out["de_wind_speed_mean"].clip(0, 25) ** 3
    if de_rd:
        out["de_radiation_mean"] = df_w[de_rd].mean(axis=1)
    return out


# ================================================================================
# 4. MACRO (Yahoo: TTF + EUA chain — same fallback logic as live)
# ================================================================================

@cached("macro")
def fetch_macro() -> pd.DataFrame:
    import yfinance as yf
    df = pd.DataFrame(index=HOURLY_IDX)
    ys = (START_UTC - pd.Timedelta(days=10)).strftime("%Y-%m-%d")   # buffer for ffill + deltas
    ye = (END_UTC + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        raw = yf.download("TTF=F", start=ys, end=ye, progress=False, auto_adjust=True)
        if not raw.empty:
            c = raw["Close"].squeeze() if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
            c.index = pd.to_datetime(c.index).tz_localize("UTC")
            df["ttf_close"] = c.reindex(HOURLY_IDX, method="ffill").ffill().bfill()
            daily = c.resample("D").last().ffill()
            df["ttf_delta_1d"] = (daily - daily.shift(1)).reindex(HOURLY_IDX, method="ffill")
    except Exception as e:
        log.warning(f"  TTF failed: {e}")
    for ticker in ["CO2.L", "CARB.PA", "XCO2.PA"]:
        try:
            raw = yf.download(ticker, start=ys, end=ye, progress=False, auto_adjust=True)
            if raw.empty:
                continue
            c = raw["Close"].squeeze() if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
            if 40 <= float(c.mean()) <= 120:
                c.index = pd.to_datetime(c.index).tz_localize("UTC")
                df["eua_close"] = c.reindex(HOURLY_IDX, method="ffill").ffill().bfill()
                break
        except Exception:
            continue
    if "ttf_close" in df and "eua_close" in df:
        df["mc_ccgt"] = df["ttf_close"] / 0.56 + df["eua_close"] * 0.36
    return df


# ================================================================================
# 5. VOLTCAST PREDICTION ARTEFACTS
# ================================================================================

_P10 = ["forecast_p10", "p10", "price_p10", "lower"]
_P90 = ["forecast_p90", "p90", "price_p90", "upper"]
_SIG = ["signal_strength", "signal", "regime_class", "regime"]

def _pick(d: pd.DataFrame, names):
    for n in names:
        if n in d.columns:
            return n
    return None

def load_predictions() -> pd.DataFrame:
    """Hourly frame of forecasts across the whole window.
    Handles BOTH filename conventions (named by delivery date or by run date):
    timestamps inside the JSON are authoritative."""
    rows = []
    for d in PRED_DIRS:
        suffix = "_eve" if d.endswith("_evening") else ""
        for path in sorted(glob.glob(os.path.join(d, "predictions_*.json"))):
            try:
                with open(path) as f:
                    ptus = pd.DataFrame(json.load(f)["voltcast_ipi"]["ptus"])
            except Exception:
                continue
            if "timestamp_local" not in ptus.columns or "forecast_p50" not in ptus.columns:
                continue
            ts = pd.to_datetime(ptus["timestamp_local"])
            ts = ts.dt.tz_localize(DUTCH_TZ, ambiguous="NaT", nonexistent="NaT") if ts.dt.tz is None else ts
            ptus.index = ts.dt.tz_convert("UTC")
            keep = pd.DataFrame(index=ptus.index)
            keep[f"p50_corr{suffix}"] = pd.to_numeric(ptus["forecast_p50"], errors="coerce")
            keep[f"p50_raw{suffix}"]  = pd.to_numeric(ptus.get("forecast_p50_raw", ptus["forecast_p50"]), errors="coerce")
            c10, c90, csig = _pick(ptus, _P10), _pick(ptus, _P90), _pick(ptus, _SIG)
            if c10:  keep[f"p10{suffix}"] = pd.to_numeric(ptus[c10], errors="coerce")
            if c90:  keep[f"p90{suffix}"] = pd.to_numeric(ptus[c90], errors="coerce")
            if csig: keep[f"signal{suffix}"] = ptus[csig].astype(str)
            rows.append(keep)
    if not rows:
        raise FileNotFoundError(f"No predictions_*.json found in {PRED_DIRS} — run dir correct?")
    allp = pd.concat(rows).sort_index()
    allp = allp[~allp.index.duplicated(keep="last")]      # later file wins per timestamp
    num = allp.select_dtypes(include=[np.number]).resample("h").mean()
    if "signal" in allp.columns:
        num["signal"] = allp["signal"].resample("h").agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
    out = num.reindex(HOURLY_IDX)
    cov_days = out["p50_corr"].notna().resample("D").max().sum()
    log.info(f"  Predictions: {int(cov_days)} delivery days covered in window")
    return out


# ================================================================================
# 6. MASTER FRAME + SCORECARDS
# ================================================================================

def build_master() -> pd.DataFrame:
    log.info("[1/5] Fetching actuals & drivers")
    parts = [fetch_entsoe_actuals()]
    w = fetch_weather()
    parts += [derive_physics_bridge(w), fetch_macro()]
    log.info("[2/5] Loading VoltCast prediction artefacts")
    parts.append(load_predictions())
    m = pd.concat(parts, axis=1).loc[:, lambda d: ~d.columns.duplicated()]

    # Derived analysis columns
    if {"p50_corr", "actual_price"} <= set(m.columns):
        m["err"]     = m["p50_corr"] - m["actual_price"]
        m["abs_err"] = m["err"].abs()
        if "p50_raw" in m:
            m["err_raw"] = m["p50_raw"] - m["actual_price"]
    if {"nl_tso_load_fc", "nl_actual_load"} <= set(m.columns):
        m["tso_load_bias"] = m["nl_tso_load_fc"] - m["nl_actual_load"]
    if {"nl_tso_load_fc", "wind_power_proxy_cubed", "solar_proxy"} <= set(m.columns):
        m["weather_scarcity_index"] = (m["nl_tso_load_fc"] / 20000.0
                                       - (m["wind_power_proxy_cubed"] / (12 ** 3) * 0.4
                                          + m["solar_proxy"] / 800.0 * 0.6))
    if {"flow_nl_no", "flow_no_nl"} <= set(m.columns):
        m["norned_net_import"] = m["flow_no_nl"].fillna(0) - m["flow_nl_no"].fillna(0)
    loc = m.index.tz_convert(DUTCH_TZ)
    m["hour_local"], m["dow"], m["date_local"] = loc.hour, loc.dayofweek, loc.date
    m["is_evening_ramp"] = ((m["hour_local"] >= 17) & (m["hour_local"] <= 21)).astype(int)
    return m


def daily_scorecard(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, g in m.dropna(subset=["actual_price", "p50_corr"]).groupby("date_local"):
        if len(g) < 20:
            continue
        diff_ok = (np.sign(g["actual_price"].diff()) == np.sign(g["p50_corr"].diff())).mean() * 100
        top_a = set(g.nlargest(PEAK_TOP_N, "actual_price")["hour_local"])
        top_p = set(g.nlargest(PEAK_TOP_N, "p50_corr")["hour_local"])
        spk = g[g["actual_price"] > SPIKE_THRESHOLD_EUR]
        cov = np.nan
        if {"p10", "p90"} <= set(g.columns) and g["p10"].notna().any():
            cov = ((g["actual_price"] >= g["p10"]) & (g["actual_price"] <= g["p90"])).mean() * 100
        rows.append({"delivery_date": d, "n_hours": len(g),
                     "avg_actual": round(g["actual_price"].mean(), 2),
                     "mae": round(g["abs_err"].mean(), 2),
                     "mbe": round(g["err"].mean(), 2),
                     "mae_raw": round(g["err_raw"].abs().mean(), 2) if "err_raw" in g else np.nan,
                     "dir_acc_pct": round(diff_ok, 1),
                     "peak_precision_pct": round(len(top_a & top_p) / PEAK_TOP_N * 100, 1),
                     "neg_hours_actual": int((g["actual_price"] < 0).sum()),
                     "spike_hours_actual": int(len(spk)),
                     "spike_mae": round(spk["abs_err"].mean(), 2) if len(spk) else np.nan,
                     "coverage_p10_p90_pct": round(cov, 1) if cov == cov else np.nan})
    sc = pd.DataFrame(rows)
    if not sc.empty:
        thr = sc["mae"].mean() + 2 * sc["mae"].std()
        sc["flag_investigate"] = np.where(sc["mae"] > thr, "YES — check news/regulatory/asset events", "")
    return sc


# ================================================================================
# 7. ATTRIBUTION — correlation screen + residual error model
# ================================================================================

DRIVER_COLS = ["de_price", "be_price", "fr_price", "se3_price", "no2_price",
               "nl_actual_load", "nl_tso_load_fc", "tso_load_bias",
               "nl_wind_actual_mw", "nl_solar_actual_mw", "nl_gas_actual_mw",
               "nl_wind_fc_mw", "nl_solar_fc_mw", "de_wind_fc_mw", "de_solar_fc_mw",
               "wind_power_proxy_cubed", "solar_proxy", "weather_scarcity_index",
               "de_wind_proxy_cubed", "de_radiation_mean",
               "nat_temperature_2m", "nat_cloud_cover",
               "ttf_close", "ttf_delta_1d", "eua_close", "mc_ccgt",
               "nl_thermal_outage_mw", "fr_nuclear_outage_mw", "interconnector_outage_mw",
               "flow_nl_de", "flow_nl_be", "norned_net_import",
               "nl_imbalance_price", "hour_local", "dow"]

def attribution(m: pd.DataFrame):
    drivers = [c for c in DRIVER_COLS if c in m.columns and m[c].notna().sum() > 50]
    base = m.dropna(subset=["err"])
    corr = pd.DataFrame({
        "pearson_err":  [base["err"].corr(base[c]) for c in drivers],
        "spearman_abs_err": [base["abs_err"].corr(base[c], method="spearman") for c in drivers],
        "n": [int(base[c].notna().sum()) for c in drivers],
    }, index=drivers).sort_values("spearman_abs_err", key=np.abs, ascending=False)
    corr.to_csv(os.path.join(FORENSIC_DIR, "driver_correlations.csv"))

    # Residual error model: drivers with high importance = signal your model is missing
    imp = None
    X = base[drivers].astype(float)
    y = base["err"].astype(float)
    mask = X.notna().mean(axis=1) > 0.6
    X, y = X[mask].fillna(X.median()), y[mask]
    if len(X) >= 150:
        try:
            from xgboost import XGBRegressor
            mdl = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, random_state=42)
            mdl.fit(X, y)
            imp = pd.Series(mdl.feature_importances_, index=drivers, name="importance")
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            mdl = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
            mdl.fit(X, y)
            imp = pd.Series(mdl.feature_importances_, index=drivers, name="importance")
        imp = imp.sort_values(ascending=False)
        imp.to_csv(os.path.join(FORENSIC_DIR, "error_model_importances.csv"))
    return corr, imp


# ================================================================================
# 8. MISSING-FEATURE PROBES (P1–P8)
# ================================================================================

def run_probes(m: pd.DataFrame) -> list:
    P, base = [], m.dropna(subset=["err"])

    def add(tag, title, stat, verdict):
        P.append({"tag": tag, "title": title, "stat": stat, "verdict": verdict})

    # P1 — Physics Bridge fidelity vs realised NL renewables
    if {"wind_power_proxy_cubed", "nl_wind_actual_mw"} <= set(m.columns):
        r_w = m["wind_power_proxy_cubed"].corr(m["nl_wind_actual_mw"])
        r_s = m["solar_proxy"].corr(m["nl_solar_actual_mw"]) if {"solar_proxy", "nl_solar_actual_mw"} <= set(m.columns) else np.nan
        add("P1", "Physics Bridge fidelity (proxy vs actual NL generation)",
            f"wind r={r_w:.3f} (R²={r_w**2:.2f}); solar r={r_s:.3f}",
            "Proxy adequate" if min(r_w, r_s if r_s == r_s else 1) >= 0.85
            else "GAP: proxy R² below 0.72 — blend ENTSO-E RES forecast at a later checkpoint or add ICON-EU ensemble members")
    # P2 — German wind absent from bridge
    if "de_wind_proxy_cubed" in base:
        r = base["abs_err"].corr(base["de_wind_proxy_cubed"], method="spearman")
        add("P2", "German wind (not in Physics Bridge)",
            f"spearman(|err|, DE wind proxy) = {r:.3f}",
            "GAP: |err| rises with DE wind — add DE national wind proxy (population/turbine-weighted) to Block C" if abs(r) > 0.12 else "No strong evidence in window")
    # P3 — TTF intraday lag (known issue 8.3)
    if "ttf_delta_1d" in base:
        # FIX: Use labels=False so pandas dynamically handles reduced bin counts
        terc = pd.qcut(base["ttf_delta_1d"].abs(), 3, labels=False, duplicates="drop")
        g = base.groupby(terc)["abs_err"].mean().round(2)
        spread = float(g.iloc[-1] - g.iloc[0]) if len(g) >= 2 else np.nan
        
        # Format the dict so the report is still easy to read
        bin_dict = {f"Bin_{int(k)}": v for k, v in g.to_dict().items()}
        
        add("P3", "TTF 48h live lag cost", f"MAE by |ΔTTF| severity: {bin_dict} (spread €{spread:.2f})",
            f"CONFIRMED: gas-move days cost ≈ €{spread:.2f}/MWh extra MAE — prioritise intraday TTF (ICE Endex)" if spread == spread and spread > 1.5 else "Lag cost small in this window")
    # P4 — NorNed / Nordic hydro (not fetched in live)
    if "no2_price" in base:
        r1 = base["err"].corr(base["no2_price"])
        r2 = base["err"].corr(base["norned_net_import"]) if "norned_net_import" in base else np.nan
        add("P4", "NorNed & NO2 price (not in live pipeline)",
            f"corr(err, NO2 price)={r1:.3f}; corr(err, NorNed net import)={r2:.3f}",
            "GAP: Nordic hydro pressure leaks into residuals — add NO2 price lag + NorNed flow to Block E" if max(abs(r1), abs(r2) if r2 == r2 else 0) > 0.15 else "Minor in this window")
    # P5 — Negative-price regime (May–June solar weekends)
    neg = base[base["actual_price"] < 0]
    if len(neg):
        pred_neg = (neg["p50_corr"] < 0).mean() * 100
        add("P5", "Negative-price hours", f"{len(neg)} hours; MAE €{neg['abs_err'].mean():.2f} vs overall €{base['abs_err'].mean():.2f}; model predicted <0 in {pred_neg:.0f}%",
            "GAP: negative-price recall weak — add solar-curtailment / must-run CHP floor features" if pred_neg < 50 else "Negative regime handled acceptably")
    else:
        add("P5", "Negative-price hours", "0 hours in window", "n/a")
    # P6 — Interconnector (transmission REMIT) outages
    if "interconnector_outage_mw" in base and base["interconnector_outage_mw"].max() > 0:
        out_h = base[base["interconnector_outage_mw"] > 0]
        add("P6", "Interconnector outages (transmission REMIT — not fetched live)",
            f"{len(out_h)} affected hours; MAE €{out_h['abs_err'].mean():.2f} vs €{base['abs_err'].mean():.2f}",
            "GAP: add query_unavailability_transmission to live_fetch (NL borders)" if len(out_h) > 10 and out_h["abs_err"].mean() > 1.2 * base["abs_err"].mean() else "Limited impact in window")
    else:
        add("P6", "Interconnector outages", "No transmission-REMIT data retrievable / none in window",
            "Add fetch anyway — structural risk for coupling features")
    # P7 — Evening duck ramp
    ev, rest = base[base["is_evening_ramp"] == 1], base[base["is_evening_ramp"] == 0]
    if len(ev) and len(rest):
        add("P7", "Evening ramp (17–21 local)", f"MAE €{ev['abs_err'].mean():.2f} vs €{rest['abs_err'].mean():.2f}; MBE {ev['err'].mean():+.2f}",
            "GAP: ramp under/over-forecast — sharpen duck_curve_ramp_stress with solar-decline x load-rise at 15-min granularity, add sunset-hour encoding" if ev["abs_err"].mean() > 1.25 * rest["abs_err"].mean() else "Ramp errors in line with base")
    # P8 — TSO load forecast bias propagation
    if "tso_load_bias" in base:
        r = base["err"].corr(base["tso_load_bias"])
        add("P8", "TenneT load-forecast bias propagation", f"corr(err, TSO bias) = {r:.3f}",
            "GAP: model inherits TenneT bias — add a learned TSO-bias correction (rolling 14d) feature" if abs(r) > 0.2 else "TSO bias not a dominant residual driver")
    return P


# ================================================================================
# 9. CHARTS
# ================================================================================

def make_charts(m: pd.DataFrame, sc: pd.DataFrame, corr: pd.DataFrame, imp):
    b = m.dropna(subset=["actual_price"])
    # 1. Time series with band
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(b.index, b["actual_price"], lw=0.9, label="EPEX NL actual", color="black")
    if "p50_corr" in b:
        ax.plot(b.index, b["p50_corr"], lw=0.9, label="VoltCast P50 (corrected)", color="tab:blue")
    if {"p10", "p90"} <= set(b.columns):
        ax.fill_between(b.index, b["p10"], b["p90"], alpha=0.18, color="tab:blue", label="P10–P90")
    ax.set_ylabel("€/MWh"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Forecast vs realised day-ahead price — hourly")
    fig.tight_layout(); fig.savefig(os.path.join(CHART_DIR, "01_timeseries.png"), dpi=150); plt.close(fig)
    # 2. Daily MAE
    if not sc.empty:
        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.bar(sc["delivery_date"].astype(str), sc["mae"], color="tab:blue")
        thr = sc["mae"].mean() + 2 * sc["mae"].std()
        ax.axhline(thr, ls="--", color="red", lw=1, label="mean + 2σ (investigate)")
        ax.set_ylabel("MAE €/MWh"); ax.tick_params(axis="x", rotation=90, labelsize=7); ax.legend(fontsize=8)
        ax.set_title("Daily MAE — corrected model")
        fig.tight_layout(); fig.savefig(os.path.join(CHART_DIR, "02_daily_mae.png"), dpi=150); plt.close(fig)
    # 3. Error heatmap hour × date
    e = m.dropna(subset=["abs_err"])
    if len(e):
        piv = e.pivot_table(index="hour_local", columns="date_local", values="abs_err", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(12, 4.5))
        im = ax.imshow(piv.values, aspect="auto", cmap="Reds", origin="lower")
        ax.set_yticks(range(0, 24, 3)); ax.set_ylabel("Hour (local)")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([str(c) for c in piv.columns], rotation=90, fontsize=6)
        fig.colorbar(im, label="|err| €/MWh"); ax.set_title("Absolute error heatmap — hour × delivery day")
        fig.tight_layout(); fig.savefig(os.path.join(CHART_DIR, "03_error_heatmap.png"), dpi=150); plt.close(fig)
    # 4. Attribution bars
    src = imp.head(12) if imp is not None else corr["spearman_abs_err"].abs().head(12)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    src.iloc[::-1].plot.barh(ax=ax, color="tab:orange")
    ax.set_title("Top residual-error drivers " + ("(XGB importance)" if imp is not None else "(|spearman|)"))
    fig.tight_layout(); fig.savefig(os.path.join(CHART_DIR, "04_attribution.png"), dpi=150); plt.close(fig)
    # 5. Proxy fidelity
    if {"wind_power_proxy_cubed", "nl_wind_actual_mw"} <= set(m.columns):
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(m["wind_power_proxy_cubed"], m["nl_wind_actual_mw"], s=4, alpha=0.4)
        ax.set_xlabel("Cubic wind proxy (m³/s³)"); ax.set_ylabel("NL actual wind MW")
        ax.set_title("P1 — Physics Bridge wind fidelity")
        fig.tight_layout(); fig.savefig(os.path.join(CHART_DIR, "05_proxy_fidelity.png"), dpi=150); plt.close(fig)


# ================================================================================
# 10. MARKDOWN REPORT
# ================================================================================

def md_table(df: pd.DataFrame, max_rows=40) -> str:
    d = df.head(max_rows).copy()
    cols = list(d.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and v != v) else str(v) for v in r) + " |")
    return "\n".join(lines)

STATIC_GAP_LIST = """
| Candidate source | Why it matters for NL DA | Access | Effort |
|---|---|---|---|
| NorNed flows + NO2 price | 700 MW Nordic hydro link; reservoir-driven price pull | ENTSO-E (free) | Low |
| BritNed flows (+GB proxy price) | 1,000 MW GB link; GB scarcity imports volatility | ENTSO-E / Elexon | Low–Med |
| DE national wind in Physics Bridge | DE wind sets coupled NL price floor; bridge currently uses NL wind + DE radiation only | Open-Meteo (already fetched, unused) | Low |
| Transmission REMIT (interconnector outages) | NTC cuts decouple NL from neighbours → spread regime shifts | ENTSO-E | Low |
| Intraday TTF (ICE Endex Connect) | Closes the documented 48h-vs-24h gas lag (issue 8.3) | Paid feed | Med |
| EU gas storage (AGSI+) + LNG sendout (ALSI) | Structural gas-regime anchor, weekly cadence | GIE API (free) | Low |
| Norwegian reservoir levels (NVE weekly) | Drives NO2 price → NorNed pressure | NVE open data | Low |
| BE nuclear availability | Only FR nuclear fetched today; BE units swing BE price | ENTSO-E REMIT | Low |
| DE public holidays | Coupled-market demand shifts not in NL calendar block | `holidays` pkg | Trivial |
| EUA auction calendar (EEX) | Auction days move carbon → mc_ccgt | EEX calendar | Low |
| Solar curtailment / must-run CHP floor | Negative-price depth in May–June | Derived / NED | Med |
"""

def write_report(m, sc, corr, imp, probes):
    b = m.dropna(subset=["err"])
    cov = np.nan
    if {"p10", "p90"} <= set(b.columns) and b["p10"].notna().any():
        cov = ((b["actual_price"] >= b["p10"]) & (b["actual_price"] <= b["p90"])).mean() * 100
    dir_acc = (np.sign(b["actual_price"].diff()) == np.sign(b["p50_corr"].diff())).mean() * 100
    spk = b[b["actual_price"] > SPIKE_THRESHOLD_EUR]
    head = {
        "Window": f"{START_LOCAL.date()} → {(END_LOCAL - pd.Timedelta(days=1)).date()}",
        "Scored hours": len(b),
        "Avg actual price €/MWh": round(b["actual_price"].mean(), 2),
        "MAE (corrected) €/MWh": round(b["abs_err"].mean(), 2),
        "MAE (raw) €/MWh": round(b["err_raw"].abs().mean(), 2) if "err_raw" in b else "n/a",
        "MBE €/MWh": round(b["err"].mean(), 2),
        "RMSE €/MWh": round(np.sqrt((b["err"] ** 2).mean()), 2),
        "Direction accuracy %": round(dir_acc, 1),
        "P10–P90 coverage % (target 80)": round(cov, 1) if cov == cov else "p10/p90 not in JSON",
        "Spike hours (>€120) / spike MAE": f"{len(spk)} / €{spk['abs_err'].mean():.2f}" if len(spk) else "0 / n/a",
        "Holdout reference MAE": "€14.79/MWh (thesis §6.3)",
    }
    worst = sc.sort_values("mae", ascending=False).head(6)[["delivery_date", "mae", "mbe", "avg_actual", "spike_hours_actual"]].copy()
    worst["event_annotation (fill manually: ACM/ACER news, REMIT, EEX auctions, weather anomaly)"] = ""

    r = []
    r.append(f"# VoltCast IPI — Live Forensic Validation Report\n*Generated {now_dutch.strftime('%Y-%m-%d %H:%M %Z')} · window {head['Window']} · hourly resolution*\n")
    r.append("## 1. Process & Method\nDaily 10:45 CET D+1 forecasts (raw and bias-corrected P50, conformal P10/P90, regime signals) were recovered from production `predictions_*.json` artefacts and scored against realised EPEX SPOT NL day-ahead clearing prices at hourly resolution. Independently re-fetched contemporaneous drivers (weather across 13 stations, realised NL renewable generation, ENTSO-E RES forecasts, TTF/EUA closes, REMIT outages, cross-border flows including NorNed/BritNed, and TSO load forecasts vs actuals) were merged onto the same UTC-hourly index. Residuals were attributed via (a) a correlation screen and (b) a gradient-boosted *error model* trained to predict the signed residual from candidate drivers — drivers carrying importance in this model represent information the production model does not capture. Eight pre-registered probes (P1–P8) test specific missing-feature hypotheses. Contemporaneous data is legitimate here: this is attribution, not forecasting.\n")
    r.append("## 2. Headline Performance (live, not backtest)\n" + md_table(pd.DataFrame(head.items(), columns=["Metric", "Value"])) + "\n")
    r.append("## 3. Daily Scorecard\n" + md_table(sc) + "\n\n![Daily MAE](charts/02_daily_mae.png)\n")
    r.append("## 4. Temporal Error Structure\n![Timeseries](charts/01_timeseries.png)\n\n![Heatmap](charts/03_error_heatmap.png)\n")
    r.append("## 5. Driver Attribution\nTop drivers of the residual (full tables in `driver_correlations.csv`, `error_model_importances.csv`):\n\n"
             + md_table(corr.head(12).round(3).reset_index().rename(columns={"index": "driver"})) + "\n\n![Attribution](charts/04_attribution.png)\n")
    r.append("## 6. Missing-Feature Probes (P1–P8)\n" + md_table(pd.DataFrame(probes)) + "\n\n![Proxy fidelity](charts/05_proxy_fidelity.png)\n")
    r.append("## 7. Candidate Data Sources Not Yet Integrated\n" + STATIC_GAP_LIST + "\n")
    r.append("## 8. Event / Regulatory Annotation (manual)\nWorst-error days for manual cross-referencing against ACER/ACM announcements, EEX EUA auction calendar, REMIT messages, and TenneT system notices:\n\n" + md_table(worst) + "\n")
    r.append("## 9. Future Scope\n1. Integrate the low-effort gaps confirmed by probes (DE wind proxy, NorNed/NO2, transmission REMIT, DE holidays) and re-run this forensic after 15 further trading days. 2. Close the TTF lag via intraday feed if P3 confirms ≥€1.5/MWh cost. 3. Promote the residual error model into a production drift monitor (thesis §12.5). 4. Replace bias-table point estimates with hierarchical shrinkage (thesis §8.6). 5. Extend scoring to signal economics: realised €/MWh captured by BUY windows vs baseload — the broker-facing number.\n")
    r.append("## 10. Reproducibility\n`python forensic_validation_v1.py --start 2026-05-17` · sources cached under `voltcast_ipi_forensics_v1/cache/` · master frame persisted as `forensic_master_hourly.parquet`.\n")
    with open(os.path.join(FORENSIC_DIR, "FORENSIC_REPORT.md"), "w") as f:
        f.write("\n".join(r))
    log.info("  Report written: FORENSIC_REPORT.md")


# ================================================================================
# 11. MAIN
# ================================================================================

def main() -> int:
    log.info("=" * 65)
    log.info("  VOLTCAST IPI — FORENSIC VALIDATION")
    log.info(f"  Window: {START_LOCAL.date()} → {END_LOCAL.date()} (exclusive)")
    log.info("=" * 65)
    m = build_master()
    m.to_parquet(os.path.join(FORENSIC_DIR, "forensic_master_hourly.parquet"))
    m.to_csv(os.path.join(FORENSIC_DIR, "forensic_master_hourly.csv"))
    if "err" not in m.columns or m["err"].notna().sum() < 24:
        log.error("Insufficient overlapping forecast/actual hours — check pred-dir and JSON schema.")
        return 1
    log.info("[3/5] Scorecards")
    sc = daily_scorecard(m)
    sc.to_csv(os.path.join(FORENSIC_DIR, "daily_scorecard.csv"), index=False)
    log.info("[4/5] Attribution + probes")
    corr, imp = attribution(m)
    probes = run_probes(m)
    for p in probes:
        log.info(f"  [{p['tag']}] {p['title']} — {p['verdict']}")
    log.info("[5/5] Charts + report")
    make_charts(m, sc, corr, imp)
    write_report(m, sc, corr, imp, probes)
    log.info(f"\n  Done. Open {FORENSIC_DIR}/FORENSIC_REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())