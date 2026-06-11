# config_v2.py
"""
VoltCast IPI v2 - Central Configuration (FORENSIC ABLATION BUILD)
v2 runs as a CHALLENGER in parallel with v1: all directories and artefacts are
separate, so the v1 champion cron is never touched. Every v2 change maps to a
finding of the May-June 2026 forensic validation (see CHANGELOG_V2.md):
  P1  solar-limb reconstruction      P4  Nordic channel (NO2/NorNed)
  P3  gas-lag train/serve parity     P5  negative-price features
  P6  transmission REMIT             BIAS adaptive rolling intercept
"""
import os

# --- DIRECTORIES (v2 sandbox — fully parallel to v1) ---
LIVE_DIR    = "voltcast_ipi_live_v2"
CACHE_DIR   = "voltcast_ipi_cache_v2"
REPORT_DIR  = "voltcast_ipi_reports_v2"
LOG_DIR     = "voltcast_ipi_logs_v2"

# --- BUSINESS LOGIC & THRESHOLDS (unchanged vs v1 — not part of the ablation) ---
SCHEMA_VERSION = "2.0"
VERDICT_CHEAP_FRAC_MIN = 0.25      # >= 25% cheap PTUs triggers BUY day
VERDICT_EXP_FRAC_MIN   = 0.25      # >= 25% exp PTUs triggers AVOID day
SPIKE_RISK_EUR         = 120.0     # P90 > this value triggers spike alert
NEG_RISK_EUR           = 0.0       # P10 < this value triggers negative alert
SPIKE_THRESHOLD_EUR    = 120.0     # Spike threshold value

# --- CONSECUTIVE WINDOW RULES ---
MIN_PTUS_BUY_WINDOW   = 8  # 2 hours
MIN_PTUS_AVOID_WINDOW = 4  # 1 hour

# --- STRATIFICATION PROBABILITIES ---
PROB_STRONG_SIGNAL   = 0.70
PROB_MODERATE_SIGNAL = 0.55

# ==============================================================================
# P3 — GAS LAG TRAIN/SERVE PARITY
# ==============================================================================
# Forensic finding P3: live MAE is ~EUR 2.65/MWh worse on gas-volatile days
# because training saw the D-1 TTF close (shift 96) while live, fetching at
# 10:10 CET before ICE settles, can only ever see the D-2 close (effective
# shift 192). v2 closes the skew WITHOUT a paid intraday feed by training on
# exactly what live will have: the D-2 close.
#   TRAINING : raw daily-ffilled series  -> shift(192) = D-2 close at delivery
#   LIVE     : series already 1 day stale -> shift(96) = D-2 close at delivery
# Both therefore present the SAME information to the model.
GAS_LAG_PTU_TRAINING = 192
GAS_LAG_PTU_LIVE     = 96

# ==============================================================================
# BIAS — ADAPTIVE ROLLING INTERCEPT (replaces the hardcoded +12 of v1)
# ==============================================================================
# Forensic finding (report section 3.5/6.3): the static Season x DOW x Hour
# bias table failed under regime shift (-16 EUR/MWh MBE in 4 days) and was
# patched mid-pilot with a manual constant. v2 decomposes the correction:
#   SHAPE : hierarchically-shrunk Season x DOW x Hour table, built on
#           DE-MEANED CV residuals at training time (level removed).
#   LEVEL : rolling intercept learned LIVE from the last N delivery days'
#           own raw predictions vs realised DA prices.
ADAPTIVE_BIAS_WINDOW_DAYS = 10    # rolling window of settled delivery days
ADAPTIVE_BIAS_MIN_DAYS    = 4     # below this, fall back to bootstrap offset
ADAPTIVE_BIAS_CLIP_EUR    = 20.0  # hard clip on the learned intercept
ADAPTIVE_BIAS_BOOTSTRAP   = 12.0  # fallback during the first challenger days
                                  # (the empirically validated v1 constant)

# --- BIAS SHAPE TABLE (training-side hierarchical shrinkage) ---
BIAS_SHRINKAGE_K   = 10.0   # pseudo-count toward season x hour parent cell
BIAS_TABLE_CLIP    = 15.0   # unchanged clip on the shape correction

# --- P5: negative-hour emphasis in point-model sample weights (v1 -> v2) ---
NEG_WEIGHT_XGB = 6.0   # was 4.0
NEG_WEIGHT_LGB = 3.0   # was 2.0
