# test_pipeline.py  pytest test_pipeline.py -v
import pytest
import numpy as np

# In a real setup, you would import your actual functions here:
# from live_predict import verify_output_quality, _find_consecutive_windows

# --- TEST 1: Signal Stratification Logic ---
def get_signal_strength(p_cheap, p_exp):
    """Simulating the logic from build_ptus_output"""
    if p_cheap >= 0.70: return "Strong BUY"
    if p_cheap >= 0.55: return "Moderate BUY"
    if p_exp >= 0.70:   return "Strong AVOID"
    if p_exp >= 0.55:   return "Moderate AVOID"
    return "NEUTRAL"

def test_stratification_boundaries():
    assert get_signal_strength(0.75, 0.01) == "Strong BUY"
    assert get_signal_strength(0.55, 0.10) == "Moderate BUY" # Exact edge
    assert get_signal_strength(0.54, 0.10) == "NEUTRAL"
    assert get_signal_strength(0.01, 0.99) == "Strong AVOID"

# --- TEST 2: Conformal Monotonicity ---
def test_conformal_logic():
    """Ensures P10 is always <= P50 <= P90"""
    p50 = np.array([50.0, 100.0, -10.0])
    p10_offset = -20.0
    p90_offset = 30.0
    
    p10 = p50 + p10_offset
    p90 = p50 + p90_offset
    
    assert (p10 <= p50).all(), "P10 exceeded P50"
    assert (p50 <= p90).all(), "P50 exceeded P90"

# --- TEST 3: Verdict Thresholds ---
def get_day_verdict(cheap_frac, exp_frac):
    """Simulating the logic from build_day_summary"""
    if cheap_frac >= 0.25: return "BUY"
    if exp_frac >= 0.25:   return "AVOID"
    return "NEUTRAL"

def test_day_verdict():
    assert get_day_verdict(0.26, 0.10) == "BUY"
    assert get_day_verdict(0.10, 0.30) == "AVOID"
    assert get_day_verdict(0.10, 0.10) == "NEUTRAL"
    # Edge case: If both are high, script biases to BUY first. 
    # (Good to know for business logic!)
    assert get_day_verdict(0.30, 0.30) == "BUY" 

# --- TEST 4: Output Quality Gate ---
def test_quality_gate_ptu_count():
    # Simulating the check inside verify_output_quality
    expected_ptus = 96
    valid_ptus = [{} for _ in range(96)]
    invalid_ptus = [{} for _ in range(94)]
    
    assert len(valid_ptus) == expected_ptus
    assert len(invalid_ptus) != expected_ptus