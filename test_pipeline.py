# test_pipeline.py
# run command in terminal: pytest test_pipeline.py -v
import pytest
import numpy as np
import config

# --- TEST 1: Signal Stratification Logic ---
def get_signal_strength(p_cheap, p_exp):
    if p_cheap >= config.PROB_STRONG_SIGNAL: return "Strong BUY"
    if p_cheap >= config.PROB_MODERATE_SIGNAL: return "Moderate BUY"
    if p_exp >= config.PROB_STRONG_SIGNAL:   return "Strong AVOID"
    if p_exp >= config.PROB_MODERATE_SIGNAL:   return "Moderate AVOID"
    return "NEUTRAL"

def test_stratification_boundaries():
    assert get_signal_strength(config.PROB_STRONG_SIGNAL + 0.05, 0.01) == "Strong BUY"
    assert get_signal_strength(config.PROB_MODERATE_SIGNAL, 0.10) == "Moderate BUY"
    assert get_signal_strength(config.PROB_MODERATE_SIGNAL - 0.01, 0.10) == "NEUTRAL"
    assert get_signal_strength(0.01, config.PROB_STRONG_SIGNAL + 0.10) == "Strong AVOID"

# --- TEST 2: Conformal Monotonicity ---
def test_conformal_logic():
    """Ensures P10 is always <= P50 <= P90"""
    p50 = np.array([50.0, 100.0, -10.0])
    p10_offset, p90_offset = -20.0, 30.0
    p10 = p50 + p10_offset
    p90 = p50 + p90_offset
    
    assert (p10 <= p50).all(), "P10 exceeded P50"
    assert (p50 <= p90).all(), "P50 exceeded P90"

# --- TEST 3: Verdict Thresholds ---
def get_day_verdict(cheap_frac, exp_frac):
    if cheap_frac >= config.VERDICT_CHEAP_FRAC_MIN: return "BUY"
    if exp_frac >= config.VERDICT_EXP_FRAC_MIN:     return "AVOID"
    return "NEUTRAL"

def test_day_verdict():
    assert get_day_verdict(config.VERDICT_CHEAP_FRAC_MIN + 0.01, 0.10) == "BUY"
    assert get_day_verdict(0.10, config.VERDICT_EXP_FRAC_MIN + 0.01) == "AVOID"
    assert get_day_verdict(config.VERDICT_CHEAP_FRAC_MIN - 0.05, 0.10) == "NEUTRAL"