"""
================================================================================
 VOLTCAST IPI — LIVE PREDICTION PIPELINE (10:45 AM BID-READY)
 Purpose: Generate D+1 day-ahead price forecast and buy/avoid signals
 Schedule: Daily cron at 10:45 AM Dutch time (runs after features at 10:30 AM)
           Must complete before EPEX gate closure at 12:00 CET.

 Inputs  (all from new_voltcast_cache/ — training artefacts):
   model_point.json          XGBoost point forecast
   model_point_lgb.txt       LightGBM point forecast
   model_p10.json            Quantile P10 (raw, not used directly)
   model_p90.json            Quantile P90 (raw, not used directly)
   model_classifier.json     XGBoost regime classifier
   model_classifier_lgb.txt  LightGBM regime classifier
   feature_names.txt         Exact ordered list
   season_hour_bias.json     Season × DOW × Hour bias correction table
   conformal_params.json     P10/P90 conformal offsets
   classifier_thresholds.json  Cheap/Expensive probability thresholds

 Inputs  (from new_voltcast_live/ — today's data):
   live_features_{date}.parquet   96-row D+1 feature matrix
   features_status.json           Gate: did feature engineering pass?

 Outputs (written to new_voltcast_live/):
   predictions_{date}.parquet     Full prediction record (internal use)
   predictions_{date}.json        Client-facing D+1 forecast
   forecast_{date}_summary.json   Day-level summary for dashboard
   prediction_status.json         Gate for downstream consumers (API, alerts)
================================================================================
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import config
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ================================================================================
# 0. LOGGING & CONFIGURATION
# ================================================================================

DUTCH_TZ     = "Europe/Amsterdam"
now_dutch    = pd.Timestamp.now(tz=DUTCH_TZ)
RUN_DATE_STR = now_dutch.strftime("%Y%m%d")
PRED_DATE_STR = (now_dutch + timedelta(days=1)).strftime("%Y%m%d")

# UPDATED DIRECTORIES
# --- Directories ---
# --- DUAL-BASIS DIAGNOSTIC SETUP ---
# If 'evening' is passed in the terminal, divert all data to the evening sandbox
RUN_MODE = "evening" if len(sys.argv) > 1 and sys.argv[1] == "evening" else "morning"
LIVE_DIR = "voltcast_ipi_live_v1_evening" if RUN_MODE == "evening" else "voltcast_ipi_live_v1"
os.makedirs(LIVE_DIR, exist_ok=True)
LOG_DIR     = "voltcast_ipi_logs_v1"
#LIVE_DIR    = "voltcast_ipi_live_v1"
CACHE_DIR   = "voltcast_ipi_cache_v1"

for d in [LOG_DIR, LIVE_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(LOG_DIR, f"predict_{RUN_DATE_STR}.log"), mode="w"
        ),
    ],
)
log = logging.getLogger("VoltCast.Predict")

# Output paths
PREDICTIONS_PARQUET = os.path.join(LIVE_DIR, f"predictions_{PRED_DATE_STR}.parquet")
PREDICTIONS_JSON    = os.path.join(LIVE_DIR, f"predictions_{PRED_DATE_STR}.json")
PRED_STATUS_FILE    = os.path.join(LIVE_DIR, "prediction_status.json")


# ================================================================================
# 1. GUARDS — CHECK UPSTREAM STATUS FILES
# ================================================================================

def check_features_status() -> tuple[bool, dict]:
    feat_status_path = os.path.join(LIVE_DIR, "features_status.json")
    if not os.path.exists(feat_status_path):
        log.error("features_status.json not found — feature script may not have run")
        return False, {}

    with open(feat_status_path) as f:
        status = json.load(f)

    if not status.get("overall_pass", False):
        log.error(f"Feature engineering did not pass: {status.get('alerts', [])}")
        return False, status

    if status.get("run_date") != RUN_DATE_STR:
        log.warning(f"features_status.json is from {status.get('run_date')}, not today")

    log.info(f"  Features status: PASS — {status['n_rows']} rows, {status['n_features']} features")
    return True, status


def write_prediction_status(pass_flag: bool, delivery_date: str,
                             alerts: list, summary: dict = None):
    out = {
        "run_date":       RUN_DATE_STR,
        "delivery_date":  delivery_date,
        "run_time":       now_dutch.isoformat(),
        "overall_pass":   pass_flag,
        "alerts":         alerts,
        "summary":        summary or {},
    }
    with open(PRED_STATUS_FILE, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"  Prediction status written: {PRED_STATUS_FILE}")


# ================================================================================
# 2. MODEL & ARTEFACT LOADER
# ================================================================================

def load_artefacts() -> dict:
    """
    Loads all training artefacts required for live inference.
    Fails fast with a descriptive error if any critical file is missing.
    """
    required = {
        "model_point":     os.path.join(CACHE_DIR, "model_point.json"),
        "model_point_lgb": os.path.join(CACHE_DIR, "model_point_lgb.txt"),
        "model_cls":       os.path.join(CACHE_DIR, "model_classifier.json"),
        "model_cls_lgb":   os.path.join(CACHE_DIR, "model_classifier_lgb.txt"),
        "feature_names":   os.path.join(CACHE_DIR, "feature_names.txt"),
        "bias_table":      os.path.join(CACHE_DIR, "season_hour_bias.json"),
        "conformal":       os.path.join(CACHE_DIR, "conformal_params.json"),
        "thresholds":      os.path.join(CACHE_DIR, "classifier_thresholds.json"),
    }

    missing = [k for k, p in required.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing artefacts: {missing}. "
            "Re-run training pipeline with T1/T2 fixes applied."
        )

    artefacts = {}

    # Point forecast models
    artefacts["model_xgb"] = xgb.XGBRegressor()
    artefacts["model_xgb"].load_model(required["model_point"])

    artefacts["model_lgb"] = lgb.Booster(model_file=required["model_point_lgb"])

    # Classifier models
    artefacts["model_xgb_cls"] = xgb.XGBClassifier()
    artefacts["model_xgb_cls"].load_model(required["model_cls"])

    artefacts["model_lgb_cls"] = lgb.Booster(model_file=required["model_cls_lgb"])

    # Feature names (enforces exact feature order)
    with open(required["feature_names"]) as f:
        artefacts["feature_names"] = [l.strip() for l in f if l.strip()]
    log.info(f"  Loaded {len(artefacts['feature_names'])} feature names")

    # Bias table: Season_DOW_Hour → correction value
    with open(required["bias_table"]) as f:
        artefacts["bias_table"] = json.load(f)
    log.info(f"  Loaded bias table: {len(artefacts['bias_table'])} entries")

    # Conformal params
    with open(required["conformal"]) as f:
        artefacts["conformal"] = json.load(f)
    log.info(f"  Conformal: P10={artefacts['conformal']['p10_offset']:.2f}  "
             f"P90={artefacts['conformal']['p90_offset']:.2f}  "
             f"Coverage={artefacts['conformal']['coverage']*100:.1f}%")

    # Classifier thresholds
    with open(required["thresholds"]) as f:
        artefacts["thresholds"] = json.load(f)
    log.info(f"  Thresholds: Cheap≥{artefacts['thresholds']['cheap_threshold']:.2f}  "
             f"Expensive≥{artefacts['thresholds']['expensive_threshold']:.2f}")

    return artefacts


# ================================================================================
# 3. INFERENCE ENGINE
# ================================================================================

SEASON_MAP = {1:"Win",2:"Win",3:"Spr",4:"Spr",5:"Spr",
              6:"Sum",7:"Sum",8:"Sum",9:"Aut",10:"Aut",11:"Aut",12:"Win"}

BLEND_W_XGB = 0.50
BLEND_W_LGB = 0.50

# ── LEVEL 1 DEFENSE PROTOCOL: STATIC BIAS TRACKER ──
# This scalar is added to every hour of the final prediction
STATIC_BIAS_OFFSET = 11.88

def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))

def symlog_inv(x):
    return np.sign(x) * (np.expm1(np.abs(x)))


def run_point_forecast(X: pd.DataFrame, artefacts: dict) -> tuple:
    """
    Runs the XGB + LGB ensemble point forecast and applies bias correction.
    Returns corrected P50 values in EUR/MWh.
    """
    # Step 1: Ensemble prediction in symlog space → invert to EUR/MWh
    pred_xgb = symlog_inv(artefacts["model_xgb"].predict(X))
    pred_lgb = symlog_inv(artefacts["model_lgb"].predict(X))
    pred_raw = BLEND_W_XGB * pred_xgb + BLEND_W_LGB * pred_lgb

    # Step 2: Season × DOW × Hour bias correction (Vectorized)
    bias_table = artefacts["bias_table"]
    
    # ── THE VECTORIZED LOOKUP ──
    # 1. Map numerical months to string seasons
    seasons = X.index.month.map(SEASON_MAP)
    
    # 2. Construct the exact dictionary key array: e.g., "Win_0_12"
    keys = seasons + "_" + X.index.dayofweek.astype(str) + "_" + X.index.hour.astype(str)
    
    # 3. Map keys to the bias dict, fill missing with 0.0, and clip to ±15 EUR/MWh
    corrections = keys.map(bias_table).fillna(0.0).values
    corrections = np.clip(corrections, -15.0, 15.0)

    # Apply the vectorized corrections array to the raw predictions
    pred_p50_raw = pred_raw + corrections

    # 2. The Corrected Prediction (Level 1 Defense Protocol)
    pred_p50_corrected = pred_p50_raw + STATIC_BIAS_OFFSET
    
    log.info(f"  Point forecast (RAW): EUR{pred_p50_raw.mean():.2f}/MWh mean  "
             f"(range EUR{pred_p50_raw.min():.1f}–EUR{pred_p50_raw.max():.1f})")
    log.info(f"  Point forecast (Corrected + EUR{STATIC_BIAS_OFFSET}): EUR{pred_p50_corrected.mean():.2f}/MWh mean  "
             f"(range EUR{pred_p50_corrected.min():.1f}–EUR{pred_p50_corrected.max():.1f})")
    
    return pred_p50_raw, pred_p50_corrected


def run_conformal_intervals(pred_p50: np.ndarray, artefacts: dict) -> tuple:
    """
    Applies asymmetric conformal offsets to produce P10/P90 intervals.
    Offsets are derived from all CV folds (out-of-sample residuals).
    """
    p10_offset = artefacts["conformal"]["p10_offset"]
    p90_offset = artefacts["conformal"]["p90_offset"]

    pred_p10 = pred_p50 + p10_offset
    pred_p90 = pred_p50 + p90_offset

    # Enforce physical floor (prices can go to -500 EUR/MWh in EPEX)
    pred_p10 = np.clip(pred_p10, -600.0, None)

    coverage = artefacts["conformal"]["coverage"]
    log.info(f"  Conformal interval: P10={pred_p10.mean():.2f}  "
             f"P90={pred_p90.mean():.2f}  (target coverage={coverage*100:.0f}%)")
    return pred_p10, pred_p90


def run_classifier(X: pd.DataFrame, artefacts: dict) -> tuple:
    """
    Runs the XGB + LGB ensemble regime classifier.
    Applies calibrated thresholds from training to assign Cheap/Normal/Expensive.

    Returns:
        pred_regime: int array (0=Cheap, 1=Normal, 2=Expensive) per PTU
        proba:       float32 array (n, 3) — probability per class
    """
    # Ensemble probabilities
    proba_xgb = artefacts["model_xgb_cls"].predict_proba(X)
    _lgb_raw = artefacts["model_lgb_cls"].predict(X)
    
    # lgb.Booster returns (n*3,) in older versions and (n,3) in newer — handle both
    if _lgb_raw.ndim == 1:
        proba_lgb = _lgb_raw.reshape(len(X), 3)
    else:
        proba_lgb = _lgb_raw
        
    proba     = BLEND_W_XGB * proba_xgb + BLEND_W_LGB * proba_lgb

    cheap_thresh = artefacts["thresholds"]["cheap_threshold"]
    exp_thresh   = artefacts["thresholds"]["expensive_threshold"]

    # Apply calibrated thresholds (base state = Normal)
    regime = np.ones(len(X), dtype=int)
    regime[proba[:, 0] >= cheap_thresh] = 0
    regime[proba[:, 2] >= exp_thresh]   = 2

    # Conflict resolution
    conflict = (proba[:, 0] >= cheap_thresh) & (proba[:, 2] >= exp_thresh)
    if conflict.sum() > 0:
        regime[conflict] = np.argmax(proba[conflict], axis=1)

    regime_counts = {0: int((regime==0).sum()), 1: int((regime==1).sum()), 2: int((regime==2).sum())}
    log.info(f"  Regime: Cheap={regime_counts[0]} PTUs  "
             f"Normal={regime_counts[1]}  Expensive={regime_counts[2]}")

    return regime, proba


# ================================================================================
# 4. CLIENT OUTPUT BUILDER
# ================================================================================

def build_ptus_output(X: pd.DataFrame, pred_p50_raw: np.ndarray, pred_p50_corrected: np.ndarray,
                       pred_p10: np.ndarray, pred_p90: np.ndarray,
                       regime: np.ndarray, proba: np.ndarray,
                       delivery_date: str) -> list:
    """
    Builds the per-PTU forecast list for the client JSON.
    Each PTU includes: timestamp, P50, P10, P90, regime, and raw probabilities.
    """
    regime_labels = {0: "Cheap", 1: "Normal", 2: "Expensive"}
    ptus = []

    for i, ts in enumerate(X.index):
        ts_dutch = ts.tz_convert("Europe/Amsterdam")

        p_cheap = round(float(proba[i, 0]), 3)
        p_norm  = round(float(proba[i, 1]), 3)
        p_exp   = round(float(proba[i, 2]), 3)
        
        # ── BROKER-READY STRATIFICATION LOGIC ──
        if p_cheap >= config.PROB_STRONG_SIGNAL:
            signal_strength = "Strong BUY"
        elif p_cheap >= config.PROB_MODERATE_SIGNAL:
            signal_strength = "Moderate BUY"
        elif p_exp >= config.PROB_STRONG_SIGNAL:
            signal_strength = "Strong AVOID"
        elif p_exp >= config.PROB_MODERATE_SIGNAL:
            signal_strength = "Moderate AVOID"
        else:
            signal_strength = "NEUTRAL"

        ptus.append({
            "ptu":              i + 1,
            "timestamp_utc":    ts.isoformat(),
            "timestamp_local":  ts_dutch.strftime("%Y-%m-%d %H:%M"),
            "hour_local":       int(ts_dutch.hour),
            
            # ── DUAL TRACKING FIELDS ──
            "forecast_p50_raw": round(float(pred_p50_raw[i]), 2),
            "forecast_p50":     round(float(pred_p50_corrected[i]), 2), # Client facing
            
            "forecast_p10":     round(float(pred_p10[i]), 2),
            "forecast_p90":     round(float(pred_p90[i]), 2),
            "interval_width":   round(float(pred_p90[i] - pred_p10[i]), 2),
            "regime":           regime_labels[int(regime[i])],
            "regime_code":      int(regime[i]),
            "prob_cheap":       p_cheap,
            "prob_normal":      p_norm,
            "prob_expensive":   p_exp,
            "signal_strength":  signal_strength,  # The new Stratified Signal
            "buy_signal":       bool(regime[i] == 0),
            "avoid_signal":     bool(regime[i] == 2),
        })

    return ptus


def build_day_summary(ptus: list, delivery_date: str,
                       artefacts: dict) -> dict:
    """
    Builds the day-level summary for the client dashboard.
    This is the primary client-facing output for Tier 3 procurement decisions.
    """
    p50s    = [p["forecast_p50"] for p in ptus]
    
    # Stratified Signal Counts
    strong_buy_ptus   = [p for p in ptus if p["signal_strength"] == "Strong BUY"]
    mod_buy_ptus      = [p for p in ptus if p["signal_strength"] == "Moderate BUY"]
    strong_avoid_ptus = [p for p in ptus if p["signal_strength"] == "Strong AVOID"]
    mod_avoid_ptus    = [p for p in ptus if p["signal_strength"] == "Moderate AVOID"]

    # Day verdict (Based on Strong + Moderate combined)
    total_buys = len(strong_buy_ptus) + len(mod_buy_ptus)
    total_avoids = len(strong_avoid_ptus) + len(mod_avoid_ptus)
    
    cheap_fraction = total_buys / len(ptus)
    expensive_fraction = total_avoids / len(ptus)
    
    if cheap_fraction >= config.VERDICT_CHEAP_FRAC_MIN and cheap_fraction > expensive_fraction:
        day_verdict = "BUY"
    elif expensive_fraction >= config.VERDICT_EXP_FRAC_MIN and expensive_fraction > cheap_fraction:
        day_verdict = "AVOID"
    else:
        day_verdict = "NEUTRAL"

    # Consecutive windows (Updated to rely on signal_strength strings, not regime_codes)
    buy_windows = _find_consecutive_windows(ptus, 
        target_signals=("Strong BUY", "Moderate BUY"), 
        min_ptus=config.MIN_PTUS_BUY_WINDOW, label="cheap"
    )
    
    avoid_windows = _find_consecutive_windows(ptus, 
        target_signals=("Strong AVOID", "Moderate AVOID"), 
        min_ptus=config.MIN_PTUS_AVOID_WINDOW, label="expensive"
    )

    # Price outlook
    daily_avg  = float(np.mean(p50s))
    daily_min  = float(np.min(p50s))
    daily_max  = float(np.max(p50s))
    peak_ptu   = ptus[int(np.argmax(p50s))]
    valley_ptu = ptus[int(np.argmin(p50s))]

    # Risk flags
    spike_risk    = any(p["forecast_p90"] > config.SPIKE_RISK_EUR for p in ptus)
    neg_risk      = any(p["forecast_p10"] < config.NEG_RISK_EUR   for p in ptus)
    high_vol_ptus = [p for p in ptus if p["interval_width"] > 60.0]
    avg_interval  = float(np.mean([p["interval_width"] for p in ptus]))

    # Confidence calculation
    dominant_probs = [max(p["prob_cheap"], p["prob_normal"], p["prob_expensive"]) for p in ptus]
    high_conf_fraction = float(np.mean([x > 0.60 for x in dominant_probs]))

    summary = {
        "schema_version":   config.SCHEMA_VERSION,    # ── THE NEW SCHEMA GUARD ──
        "delivery_date":    delivery_date,
        "generated_at":     now_dutch.strftime("%Y-%m-%d %H:%M CET"),
        "model_version":    "VoltCast IPI v2.1 — Physics Bridge (Corrected)",

        # ── PRIMARY CLIENT SIGNAL ──────────────────────────────────────
        "day_verdict": day_verdict,
        "day_verdict_rationale": _day_verdict_rationale(
            day_verdict, cheap_fraction, expensive_fraction,
            daily_avg, daily_min, daily_max
        ),

        # ── PRICE OUTLOOK ──────────────────────────────────────────────
        "price_outlook": {
            "expected_avg_eur_mwh":  round(daily_avg, 2),
            "expected_low_eur_mwh":  round(daily_min, 2),
            "expected_high_eur_mwh": round(daily_max, 2),
            "peak_hour_local":       peak_ptu["timestamp_local"],
            "peak_forecast_eur_mwh": round(peak_ptu["forecast_p50"], 2),
            "valley_hour_local":     valley_ptu["timestamp_local"],
            "valley_forecast_eur_mwh": round(valley_ptu["forecast_p50"], 2),
        },

        # ── STRATIFIED PROCUREMENT SIGNALS ─────────────────────────────
        "signal_counts": {
            "strong_buy_ptus":     len(strong_buy_ptus),
            "moderate_buy_ptus":   len(mod_buy_ptus),
            "strong_avoid_ptus":   len(strong_avoid_ptus),
            "moderate_avoid_ptus": len(mod_avoid_ptus),
            "neutral_ptus":        len(ptus) - (total_buys + total_avoids),
        },
        "buy_windows":   buy_windows,
        "avoid_windows": avoid_windows,

        # ── RISK FLAGS ─────────────────────────────────────────────────
        "risk_flags": {
            "spike_risk":             spike_risk,
            "spike_risk_note":        "P90 forecast exceeds EUR120/MWh in ≥1 PTU" if spike_risk else None,
            "negative_price_risk":    neg_risk,
            "negative_price_note":    "P10 forecast below EUR0/MWh in ≥1 PTU" if neg_risk else None,
            "high_volatility_ptus":   len(high_vol_ptus),
            "avg_interval_width_eur": round(avg_interval, 2),
        },

        # ── MODEL CONFIDENCE ───────────────────────────────────────────
        "model_confidence": {
            "high_confidence_fraction": round(high_conf_fraction, 3),
            "conformal_coverage_pct":   round(artefacts["conformal"]["coverage"] * 100, 1)
        },
    }
    return summary


def _find_consecutive_windows(ptus: list, target_signals: tuple, # regime_code: int,
                               min_ptus: int, label: str) -> list:
    windows = []
    current_window = []

    for ptu in ptus:
        # Check if the PTU's signal matches our targets (e.g., "Strong BUY" or "Moderate BUY")
        if ptu["signal_strength"] in target_signals:
            current_window.append(ptu)
        else:
            if len(current_window) >= min_ptus:
                _append_window(windows, current_window, label)
            current_window = []

    '''for ptu in ptus:
        if ptu["regime_code"] == regime_code:
            current_window.append(ptu)
        else:
            if len(current_window) >= min_ptus:
                _append_window(windows, current_window, label)
            current_window = []'''

    if len(current_window) >= min_ptus:
        _append_window(windows, current_window, label)

    return windows


def _append_window(windows: list, ptus_in_window: list, label: str):
    p50s = [p["forecast_p50"] for p in ptus_in_window]
    windows.append({
        "start_local":     ptus_in_window[0]["timestamp_local"],
        "end_local":       ptus_in_window[-1]["timestamp_local"],
        "duration_hours":  round(len(ptus_in_window) / 4, 1),
        "avg_forecast_eur_mwh": round(float(np.mean(p50s)), 2),
        "min_forecast_eur_mwh": round(float(np.min(p50s)), 2),
        "max_forecast_eur_mwh": round(float(np.max(p50s)), 2),
        "ptu_count":       len(ptus_in_window),
        "type":            label,
    })


def _day_verdict_rationale(verdict: str, cheap_frac: float,
                            exp_frac: float, avg: float,
                            low: float, high: float) -> str:
    if verdict == "BUY":
        return (
            f"{cheap_frac*100:.0f}% of PTUs are BUY-tier (Strong or Moderate). " #of delivery day PTUs forecast as Cheap
            f"Expected average EUR{avg:.1f}/MWh with lows near EUR{low:.1f}/MWh. "
            "Schedule flexible load during identified buy windows."
        )
    elif verdict == "AVOID":
        return (
            f"{exp_frac*100:.0f}% of PTUs are AVOID-tier (Strong or Moderate). " #of delivery day PTUs forecast as Expensive
            f"Expected peaks near EUR{high:.1f}/MWh. "
            "Defer flexible load or pre-purchase on spot if possible."
        )
    else:
        return (
            f"Mixed outlook. Expected average EUR{avg:.1f}/MWh "
            f"(range EUR{low:.1f}–EUR{high:.1f}/MWh). "
            "Selective load scheduling recommended — see buy/avoid windows."
        )

# ================================================================================
# 5. WEEKLY SIGNAL BUILDER
# ================================================================================

def build_weekly_signal(delivery_date: str) -> dict:
    """
    Builds a weekly procurement card for the ISO calendar week (Mon-Sun)
    containing the delivery date of today's D+1 prediction.
    """
    delivery_ts    = pd.Timestamp(delivery_date)
    days_to_monday = delivery_ts.dayofweek          # 0=Mon, 6=Sun
    week_monday    = delivery_ts - pd.DateOffset(days=days_to_monday)
    week_sunday    = week_monday + pd.DateOffset(days=6)
    today          = now_dutch.date()

    weekly = {
        "generated_at": now_dutch.strftime("%Y-%m-%d %H:%M CET"),
        "week_start":   str(week_monday.date()),
        "week_end":     str(week_sunday.date()),
        "days":         [],
    }

    for day_offset in range(7):    # 0 = Monday through 6 = Sunday
        target_date  = (week_monday + pd.DateOffset(days=day_offset)).date()
        summary_path = os.path.join(
            LIVE_DIR,
            f"forecast_{target_date.strftime('%Y%m%d')}_summary.json"
        )

        if os.path.exists(summary_path):
            with open(summary_path) as f:
                day_summary = json.load(f)

            # ── NEW: SCHEMA ENFORCEMENT ──
            schema = day_summary.get("schema_version", "legacy")
            
            if schema == "1.0":
                # Parse normally for our current version
                weekly["days"].append({
                    "date":          str(target_date),
                    "day_name":      pd.Timestamp(target_date).strftime("%A"),
                    "verdict":       day_summary["day_verdict"],
                    "avg_eur_mwh":   day_summary["price_outlook"]["expected_avg_eur_mwh"],
                    "range_low":     day_summary["price_outlook"]["expected_low_eur_mwh"],
                    "range_high":    day_summary["price_outlook"]["expected_high_eur_mwh"],
                    "buy_windows":   day_summary.get("buy_windows", []),
                    "spike_risk":    day_summary["risk_flags"]["spike_risk"],
                    "confidence":    day_summary["model_confidence"]["high_confidence_fraction"],
                    "confirmed":     day_summary.get("d1_price_confirmed", False),
                    "is_past":       target_date < today,
                })
            else:
                # Graceful fallback for old or unknown schemas
                log.warning(f"  Schema mismatch for {target_date} (found '{schema}'). Skipping detail parsing.")
                weekly["days"].append({
                    "date":     str(target_date),
                    "day_name": pd.Timestamp(target_date).strftime("%A"),
                    "verdict":  "UNAVAILABLE",
                    "note":     f"Data format incompatible (schema: {schema})",
                    "is_past":  target_date < today,
                })
        else:
            # Handle days that haven't been forecasted yet
            if target_date < today:
                verdict = "UNAVAILABLE"
                note    = "Pipeline did not run for this date"
            else:
                verdict = "PENDING"
                note    = "Forecast not yet generated — will arrive this morning"

            weekly["days"].append({
                "date":     str(target_date),
                "day_name": pd.Timestamp(target_date).strftime("%A"),
                "verdict":  verdict,
                "note":     note,
            })

    # Week-level summary
    buy_days    = [d for d in weekly["days"] if d.get("verdict") == "BUY"]
    avoid_days  = [d for d in weekly["days"] if d.get("verdict") == "AVOID"]
    filled_days = [d for d in weekly["days"] if d.get("verdict") not in ["PENDING", "UNAVAILABLE"]]

    weekly["week_summary"] = {
        "week":          f"{week_monday.strftime('%d %b')} – {week_sunday.strftime('%d %b %Y')}",
        "buy_days":      [d["day_name"] for d in buy_days],
        "avoid_days":    [d["day_name"] for d in avoid_days],
        "days_filled":   len(filled_days),
        "days_pending":  7 - len(filled_days),
        "card_status":   "COMPLETE" if len(filled_days) == 7 else f"{len(filled_days)}/7 days available",
        "narrative":     _weekly_narrative(buy_days, avoid_days),
    }

    weekly_path = os.path.join(LIVE_DIR, f"weekly_signal_{RUN_DATE_STR}.json")
    with open(weekly_path, "w") as f:
        json.dump(weekly, f, indent=2)

    log.info(f"  Weekly signal saved: {weekly_path}")
    log.info(f"  Week: {week_monday.date()} → {week_sunday.date()}")
    log.info(f"  Card: {len(filled_days)}/7 days filled  |  "
             f"Buy={[d['day_name'] for d in buy_days]}  Avoid={[d['day_name'] for d in avoid_days]}")

    return weekly


def _weekly_narrative(buy_days: list, avoid_days: list) -> str:
    if not buy_days and not avoid_days:
        return "Neutral week. No strong procurement timing signal."
    
    parts = []
    if buy_days:
        names = ", ".join(d["day_name"] for d in buy_days)
        avg   = round(sum(d.get("avg_eur_mwh", 0) for d in buy_days) / len(buy_days), 1)
        parts.append(f"BUY window: {names} (avg EUR{avg}/MWh).")
    if avoid_days:
        names = ", ".join(d["day_name"] for d in avoid_days)
        parts.append(f"Avoid: {names}.")
    return " ".join(parts)

def verify_output_quality(ptus: list, delivery_date: str, alerts: list) -> bool:
    """
    Final quality gate to ensure the output is technically and physically 
    sound before delivery to broker systems. Includes exact DST-aware PTU counts.
    """
    log.info("\n[VERIFICATION] Auditing output quality...")
    pass_quality = True

    # 1. Strict Count Check (DST-Aware)
    # Calculate exactly how many 15-min intervals exist in this specific Dutch day
    ts_start = pd.Timestamp(f"{delivery_date} 00:00:00", tz="Europe/Amsterdam")
    ts_end   = ts_start + pd.DateOffset(days=1)
    
    # Total seconds in the day / 900 seconds (15 mins) = Expected PTUs
    expected_ptus = int((ts_end - ts_start).total_seconds() / 900)

    n_ptus = len(ptus)
    if n_ptus != expected_ptus:
        msg = f"CRITICAL: Output contains {n_ptus} PTUs, expected {expected_ptus} for delivery date {delivery_date}."
        log.error(f"  {msg}")
        alerts.append(msg)
        pass_quality = False
    else:
        log.info(f"  PTU Count Check: {n_ptus}/{expected_ptus} (PASS - DST adjusted if applicable)")

    # 2. Probability Integrity (Range & Sum check)
    prob_errors = 0
    sum_errors = 0
    for p in ptus:
        probs = [p["prob_cheap"], p["prob_normal"], p["prob_expensive"]]
        
        # Check if any probability is mathematically impossible
        if any(x < 0.0 or x > 1.0 for x in probs):
            prob_errors += 1
            
        # Check if they sum to 1.0 (with small float tolerance)
        if abs(sum(probs) - 1.0) > 0.005:
            sum_errors += 1

    if prob_errors > 0:
        msg = f"CRITICAL: Found {prob_errors} PTUs with probabilities outside [0, 1]."
        log.error(f"  {msg}")
        alerts.append(msg)
        pass_quality = False

    if sum_errors > 0:
        msg = f"CRITICAL: Found {sum_errors} PTUs where probabilities do not sum to 1.0."
        log.error(f"  {msg}")
        alerts.append(msg)
        pass_quality = False

    # 3. Forecast Monotonicity (P10 <= P50 <= P90)
    mono_errors = 0
    for p in ptus:
        if not (p["forecast_p10"] <= p["forecast_p50"] <= p["forecast_p90"]):
            mono_errors += 1
    
    if mono_errors > 0:
        log.warning(f"  WARNING: {mono_errors} PTUs violate forecast monotonicity (P10 <= P50 <= P90).")

    if pass_quality:
        log.info("  Probability Integrity: Validated (Sum ~ 1.0, Range [0,1])")
        log.info("  Final Output Quality: PASS")
    
    return pass_quality

# ================================================================================
# 6. MASTER LIVE PIPELINE
# ================================================================================

def run_live_predict() -> int:
    """
    Returns exit code:
      0 = predictions generated, client output written
      1 = critical failure, no forecast produced
    """
    log.info("=" * 65)
    log.info("  VOLTCAST IPI — LIVE PREDICTION (10:45 AM)")
    log.info(f"  Run time: {now_dutch.strftime('%Y-%m-%d %H:%M %Z')}")
    log.info("=" * 65)

    alerts = []

    # --- Guard: features passed ---
    feat_ok, feat_info = check_features_status()
    if not feat_ok:
        write_prediction_status(False, "unknown", ["Feature engineering did not pass"])
        from alerts import send_pipeline_alert
        send_pipeline_alert("FEATURE", "Fetch did not pass — features skipped",
                            f"Delivery {feat_info.get('delivery_date')}\n"
                            f"Alerts: {feat_info.get('alerts', [])}")
        return 1

    delivery_date    = feat_info["delivery_date"]
    delivery_date_str = delivery_date.replace("-", "")
    summary_json_path = os.path.join(LIVE_DIR, f"forecast_{delivery_date_str}_summary.json")

    features_parquet = os.path.join(LIVE_DIR, f"live_features_{feat_info['run_date']}.parquet")

    if not os.path.exists(features_parquet):
        msg = f"Live features parquet not found: {features_parquet}"
        log.error(msg)
        write_prediction_status(False, delivery_date, [msg])
        return 1

    # --- Load artefacts ---
    log.info("\n[ARTEFACTS] Loading model artefacts...")
    try:
        artefacts = load_artefacts()
    except FileNotFoundError as e:
        log.error(str(e))
        write_prediction_status(False, delivery_date, [str(e)])
        return 1

    # --- Load features ---
    log.info(f"\n[FEATURES] Loading {features_parquet}...")
    X = pd.read_parquet(features_parquet)
    log.info(f"  Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")

    # Confirm feature count matches training
    expected_n = len(artefacts["feature_names"])
    if X.shape[1] != expected_n:
        msg = f"Feature count mismatch: got {X.shape[1]}, expected {expected_n}"
        log.error(f"  {msg}")
        alerts.append(f"CRITICAL: {msg}")
        write_prediction_status(False, delivery_date, alerts)
        return 1

    if X.shape[0] < 96:
        alerts.append(f"WARNING: Only {X.shape[0]}/96 PTUs in feature matrix")
        log.warning(f"  Incomplete: {X.shape[0]}/96 PTUs")

    # --- Run inference ---
    log.info("\n[INFERENCE] Running models...")

    # Point forecast + bias correction
    log.info("  Running point forecast (XGB + LGB + bias)...")
    pred_p50_raw, pred_p50_corrected = run_point_forecast(X, artefacts)

    # Conformal intervals
    log.info("  Computing conformal intervals...")
    pred_p10, pred_p90 = run_conformal_intervals(pred_p50_corrected, artefacts)

    # Regime classifier
    log.info("  Running regime classifier (XGB + LGB + thresholds)...")
    regime, proba = run_classifier(X, artefacts)

    # --- Build output ---
    log.info("\n[OUTPUT] Building client forecast...")

    # Per-PTU output
    ptus = build_ptus_output(X, pred_p50_raw, pred_p50_corrected, pred_p10, pred_p90, regime, proba, delivery_date)

    # Day summary (with d1_confirmed injected)
    summary = build_day_summary(ptus, delivery_date, artefacts)

    # ── NEW: FINAL QUALITY VERIFICATION ──
    quality_pass = verify_output_quality(ptus, delivery_date, alerts)
    
    # If it fails quality, we set overall_pass to False
    if not quality_pass:
        log.error("  CRITICAL: Output quality check failed. Review alerts.")

    # --- Save parquet (internal — full numerical record) ---
    df_pred = pd.DataFrame({
        "timestamp_utc":  X.index,
        "pred_p50_raw":   pred_p50_raw,
        "pred_p50":       pred_p50_corrected,
        "pred_p10":       pred_p10,
        "pred_p90":       pred_p90,
        "regime":         regime,
        "prob_cheap":     proba[:, 0],
        "prob_normal":    proba[:, 1],
        "prob_expensive": proba[:, 2],
    }).set_index("timestamp_utc")
    df_pred.to_parquet(PREDICTIONS_PARQUET)

    # --- Save client JSON (full PTU list) ---
    client_output = {
        "voltcast_ipi": {
            "version":       "2.1",
            "delivery_date": delivery_date,
            "generated_at":  now_dutch.strftime("%Y-%m-%d %H:%M CET"),
            "market":        "EPEX SPOT NL Day-Ahead",
            "resolution":    "15-minute PTU",
            
            # ── B2B BROKER METADATA ──
            "signal_definitions": {
                "Strong BUY":     "High conviction period (Prob > 0.70). Recommended for automated asset dispatch.",
                "Moderate BUY":   "Directional advantage (Prob > 0.55). Recommended for manual review or flexible shifting.",
                "Strong AVOID":   "High conviction expensive period (Prob > 0.70). Curtailment highly recommended.",
                "Moderate AVOID": "Directional expensive (Prob > 0.55). Monitor carefully.",
                "NEUTRAL":        "Price expected to track daily average. No strong timing advantage."
            },
            "disclaimer": "VoltCast IPI forecasts are generated via machine learning models for informational purposes only. Energy markets are inherently volatile; users assume all risk for trading or automated dispatch decisions made using this data. This forecast does not constitute financial advice.",
            
            "summary":       summary,
            "ptus":          ptus,
        }
    }
    with open(PREDICTIONS_JSON, "w") as f:
        json.dump(client_output, f, indent=2)

    # --- Save day summary separately ---
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # --- Build Weekly Signal Card ---
    log.info("\n[WEEKLY] Building weekly signal card...")
    weekly = build_weekly_signal(delivery_date)
    log.info(f"  Buy days: {weekly['week_summary']['buy_days']}")
    log.info(f"  Avoid days: {weekly['week_summary']['avoid_days']}")

    # --- Final log ---
    log.info("\n" + "=" * 65)
    log.info("  FORECAST COMPLETE")
    log.info("=" * 65)
    log.info(f"  Delivery date    : {delivery_date}")
    log.info(f"  Day verdict      : {summary['day_verdict']}")
    log.info(f"  Expected avg     : EUR{summary['price_outlook']['expected_avg_eur_mwh']:.2f}/MWh")
    log.info(f"  Expected range   : EUR{summary['price_outlook']['expected_low_eur_mwh']:.1f} – "
             f"EUR{summary['price_outlook']['expected_high_eur_mwh']:.1f}/MWh")
    log.info(f"  Buy windows      : {len(summary['buy_windows'])}")
    log.info(f"  Avoid windows    : {len(summary['avoid_windows'])}")
    log.info(f"  Spike risk       : {'YES' if summary['risk_flags']['spike_risk'] else 'No'}")
    log.info(f"  Negative risk    : {'YES' if summary['risk_flags']['negative_price_risk'] else 'No'}")
    log.info(f"  Confidence       : {summary['model_confidence']['high_confidence_fraction']*100:.0f}% "
             f"of PTUs high-confidence")
    log.info(f"\n  Saved: {PREDICTIONS_JSON}")
    log.info(f"  Saved: {PREDICTIONS_PARQUET}")
    log.info(f"  Saved: {summary_json_path}")

    overall_pass = not any("CRITICAL" in a for a in alerts) and quality_pass
    write_prediction_status(overall_pass, delivery_date, alerts, summary)

    if not overall_pass:
        from alerts import send_pipeline_alert
        send_pipeline_alert("PREDICT", "Prediction pipeline failed", 
                            f"Delivery {delivery_date}\nAlerts: {alerts}")
    # (ADD THIS SUCCESS BLOCK)
    else:
        from alerts import send_pipeline_alert
        send_pipeline_alert("MODEL", f"Model Inference OK: {delivery_date}",
                            f"Successfully generated DA price predictions.\nVerdict: {summary['day_verdict']}\nExp Avg: EUR {summary['price_outlook']['expected_avg_eur_mwh']}/MWh")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(run_live_predict())