# config.py
"""
VoltCast IPI - Central Configuration
Holds all static thresholds, file paths, and business logic constants.
"""
import os

# --- DIRECTORIES ---
LIVE_DIR    = "voltcast_ipi_live_v1"
CACHE_DIR   = "voltcast_ipi_cache_v1"
REPORT_DIR  = "voltcast_ipi_reports_v1"
LOG_DIR     = "voltcast_ipi_logs_v1"

# --- BUSINESS LOGIC & THRESHOLDS ---
SCHEMA_VERSION = "1.0"
VERDICT_CHEAP_FRAC_MIN = 0.25      # >= 25% cheap PTUs triggers BUY day
VERDICT_EXP_FRAC_MIN   = 0.25      # >= 25% exp PTUs triggers AVOID day
SPIKE_RISK_EUR         = 120.0     # P90 > this value triggers spike alert
NEG_RISK_EUR           = 0.0       # P10 < this value triggers negative alert
SPIKE_THRESHOLD_EUR = 120.0        # Spike Threshold value

# --- CONSECUTIVE WINDOW RULES ---
MIN_PTUS_BUY_WINDOW   = 8  # 2 hours
MIN_PTUS_AVOID_WINDOW = 4  # 1 hour

# --- STRATIFICATION PROBABILITIES ---
PROB_STRONG_SIGNAL   = 0.70
PROB_MODERATE_SIGNAL = 0.55

# --- BIAS CORRECTION PROTOCOL ---
# Level 1 Defense: Manual offset applied to final predictions to counter structural baseline drift.
MANUAL_BIAS_OFFSET = 12
# --------------------------------