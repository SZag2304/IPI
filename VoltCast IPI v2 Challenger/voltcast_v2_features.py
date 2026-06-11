"""
================================================================================
 VOLTCAST IPI v2 — SHARED FEATURE ADD-ONS (FORENSIC ABLATION BLOCKS)
 Imported by BOTH training_feature_v2.py and live_feature_v2.py so that the
 new feature code exists exactly once: train/serve parity by construction,
 not by discipline.

 Block N  (P4)   Nordic coupling      — NO2 price + NorNed flows, lagged
 Block S2 (P1)   Solar limb v2        — clear-sky index, diffuse share,
                                        south/north gradient, piecewise wind
 Block W2 (§7.1) German wind proxy    — DE 100m wind, cubed (ablation candidate)
 Block X  (P5)   Negative-price block — renewable surplus, oversupply,
                                        same-hour negative-price climatology
 Block T  (P6)   Transmission REMIT   — interconnector outage MW (forward-
                                        published, legitimately unlagged)
 Block DH (§7.1) German holidays      — coupled-market demand calendar

 LEAKAGE DISCIPLINE: identical to v1. Neighbour prices and flows enter at
 lag-96 minimum. Weather and REMIT publications are exogenous/forward-
 published and may enter unlagged. The target is never touched here.
================================================================================
"""
import numpy as np
import pandas as pd
import logging

log = logging.getLogger("VoltCast.FeaturesV2")

TARGET_COL = "DA_Price_NL_EURMWh"

# Station groups used by the v2 solar/wind blocks (must match fetch scripts)
NL_SOUTH_STATIONS = ["Eindhoven", "Maastricht"]            # PV-dense south
NL_NORTH_STATIONS = ["Amsterdam", "Friesland"]             # maritime north
DE_STATIONS       = ["Hamburg", "Bremen", "Kiel", "Munich", "Stuttgart", "Freiburg"]


# ==============================================================================
# BLOCK N — NORDIC COUPLING (forensic probe P4: corr(err, NO2) = -0.395)
# ==============================================================================
def add_nordic_coupling_features(df: pd.DataFrame) -> pd.DataFrame:
    """NO2 day-ahead price and NorNed scheduled flows, lag-96 discipline.
    Falsifiable prediction (pre-registered): the -0.395 error/NO2 correlation
    measured on the v1 challenger window shrinks materially under v2."""
    out = df.copy()

    if "DA_Price_NO2_EURMWh" in out.columns:
        p = out["DA_Price_NO2_EURMWh"]
        out["no2_price_lag96"]        = p.shift(96).astype("float32")
        out["no2_price_lag672"]       = p.shift(672).astype("float32")
        out["no2_price_roll7d_lag96"] = p.shift(96).rolling(672, min_periods=96).mean().astype("float32")
        if TARGET_COL in out.columns:
            out["spread_nl_no2_lag96"] = (out[TARGET_COL].shift(96) - p.shift(96)).astype("float32")
        if "DA_Price_SE3_EURMWh" in out.columns:
            # Nordic internal spread: NO2 vs SE3 — hydro-vs-thermal pressure
            out["nordic_internal_spread_lag96"] = (
                p.shift(96) - out["DA_Price_SE3_EURMWh"].shift(96)
            ).astype("float32")

    # NorNed scheduled exchanges (700 MW link). Schedules for D+1 publish
    # after the auction -> strictly lagged, like all other flow features.
    if "Flow_NO_NL_MW" in out.columns and "Flow_NL_NO_MW" in out.columns:
        net_import = out["Flow_NO_NL_MW"].fillna(0) - out["Flow_NL_NO_MW"].fillna(0)
        out["norned_net_import_lag96"]   = net_import.shift(96).astype("float32")
        out["norned_roll24h_lag96"]      = net_import.shift(96).rolling(96, min_periods=24).mean().astype("float32")
        out["norned_saturation_lag96"]   = (net_import.shift(96).abs() / 700.0).clip(0, 1).astype("float32")

    n = len([c for c in out.columns if c not in df.columns])
    log.info(f"  [N]  Nordic coupling (P4): {n} features")
    return out


# ==============================================================================
# BLOCK S2 — SOLAR LIMB v2 + PIECEWISE WIND (forensic probe P1: solar R2=0.52)
# ==============================================================================
def add_solar_limb_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuilds the weaker solar limb of the Physics Bridge.
    clear-sky index: radiation forecast relative to a same-hour rolling-14-day
    historical maximum (history enters at lag-96 to stay strictly causal).
    Falsifiable prediction: the solar driver cluster (Spearman |err| ~ 0.20)
    weakens under v2."""
    out = df.copy()

    if "Nat_shortwave_radiation" in out.columns:
        rad = out["Nat_shortwave_radiation"]
        hours = pd.Series(out.index.hour, index=out.index)
        # Same-hour clear-sky proxy from the trailing 14 days (lag-96 history)
        clearsky = (
            rad.shift(96)
               .groupby(hours)
               .transform(lambda s: s.rolling(14, min_periods=5).max())
               .clip(lower=50.0)
        )
        out["clear_sky_index"] = (rad / clearsky).clip(0, 1.5).astype("float32")
        if "is_midday_solar" in out.columns:
            out["csi_midday"] = (out["clear_sky_index"] * out["is_midday_solar"]).astype("float32")

    if "Nat_direct_radiation" in out.columns and "Nat_diffuse_radiation" in out.columns:
        tot = out["Nat_direct_radiation"] + out["Nat_diffuse_radiation"]
        out["diffuse_share"] = (out["Nat_diffuse_radiation"] / tot.clip(lower=1.0)).clip(0, 1).astype("float32")

    # South/north PV geography gradient from per-city columns (computed
    # BEFORE the per-city drop in build_ml_matrix)
    south = [f"shortwave_radiation_{c}" for c in NL_SOUTH_STATIONS if f"shortwave_radiation_{c}" in out.columns]
    north = [f"shortwave_radiation_{c}" for c in NL_NORTH_STATIONS if f"shortwave_radiation_{c}" in out.columns]
    if south and north:
        out["solar_south_north_gradient"] = (
            out[south].mean(axis=1) - out[north].mean(axis=1)
        ).astype("float32")

    # Piecewise wind power curve (P1, Fig. 5: cubic proxy saturates above
    # rated speed). Normalised 0..1: cut-in 3 m/s, rated 12 m/s, cut-out 25.
    if "Nat_wind_speed_100m" in out.columns:
        v = out["Nat_wind_speed_100m"].astype("float32")
        pc = pd.Series(0.0, index=out.index, dtype="float32")
        ramp = ((v - 3.0) / 9.0).clip(0, 1) ** 3
        pc = np.where(v < 3.0, 0.0, np.where(v <= 12.0, ramp, np.where(v <= 25.0, 1.0, 0.0)))
        out["wind_powercurve_v2"]       = pc.astype("float32")
        out["wind_powercurve_v2_lag96"] = pd.Series(pc, index=out.index).shift(96).astype("float32")
        out["wind_pc_ramp_3h"]          = (pd.Series(pc, index=out.index)
                                           - pd.Series(pc, index=out.index).shift(12)).astype("float32")

    n = len([c for c in out.columns if c not in df.columns])
    log.info(f"  [S2] Solar limb v2 + piecewise wind (P1): {n} features")
    return out


# ==============================================================================
# BLOCK W2 — GERMAN WIND PROXY (§7.1 ablation candidate; P2 was unsupported
# on the 24-day window but the data is already fetched and the coupled DE
# price floor is physically wind-driven — included and flagged for ablation)
# ==============================================================================
def add_de_wind_proxy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = [f"wind_speed_100m_{c}" for c in DE_STATIONS if f"wind_speed_100m_{c}" in out.columns]
    if cols:
        v = out[cols].mean(axis=1)
        out["de_wind_proxy_cubed"]       = (v.clip(0, 25) ** 3).astype("float32")
        out["de_wind_proxy_lag96"]       = out["de_wind_proxy_cubed"].shift(96).astype("float32")
        out["de_wind_collapse_proxy"]    = (
            out["de_wind_proxy_cubed"].shift(192) - out["de_wind_proxy_lag96"]
        ).clip(lower=0).astype("float32")
    n = len([c for c in out.columns if c not in df.columns])
    log.info(f"  [W2] German wind proxy (ablation): {n} features")
    return out


# ==============================================================================
# BLOCK X — NEGATIVE-PRICE BLOCK (forensic probe P5: 10% recall on 52 hours)
# ==============================================================================
def add_negative_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Curtailment/oversupply economics absent from v1.
    Falsifiable prediction: negative-hour recall rises well above the 10%
    measured live, without degrading overall MAE."""
    out = df.copy()

    # Renewable surplus: the mirrored, floor-clipped scarcity index
    if "weather_scarcity_index" in out.columns:
        surplus = (-out["weather_scarcity_index"]).clip(lower=0)
        out["renewable_surplus_index"] = surplus.astype("float32")
        if "is_weekend" in out.columns:
            out["surplus_x_weekend"] = (surplus * out["is_weekend"]).astype("float32")
        if "is_midday_solar" in out.columns:
            out["surplus_x_midday"] = (surplus * out["is_midday_solar"]).astype("float32")

    # Oversupply under clear skies on low-demand days — the canonical
    # May/June negative-price configuration
    if all(c in out.columns for c in ["clear_sky_index", "is_midday_solar", "is_weekend", "is_solar_season"]):
        out["oversupply_csi"] = (
            out["clear_sky_index"] * out["is_midday_solar"]
            * out["is_weekend"] * out["is_solar_season"]
        ).astype("float32")

    # Same-hour negative-price climatology over the trailing 14 days (lag-96)
    if TARGET_COL in out.columns:
        p = out[TARGET_COL]
        hours = pd.Series(out.index.hour, index=out.index)
        neg_hist = (p.shift(96) < 0).astype(float)
        out["neg_price_same_hour_14d"] = (
            neg_hist.groupby(hours).transform(lambda s: s.rolling(14, min_periods=5).sum())
        ).astype("float32")

    n = len([c for c in out.columns if c not in df.columns])
    log.info(f"  [X]  Negative-price block (P5): {n} features")
    return out


# ==============================================================================
# BLOCK T — TRANSMISSION REMIT (forensic probe P6: fetch added; probe was
# degenerate on the 24-day window — retained so the next window can test it)
# ==============================================================================
def add_ic_outage_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "IC_Outage_MW" in out.columns:
        ic = out["IC_Outage_MW"].fillna(0.0)
        out["ic_outage_mw"]      = ic.astype("float32")
        out["ic_outage_roll24h"] = ic.rolling(96, min_periods=1).mean().astype("float32")
        if "congestion_flag_lag96" in out.columns:
            out["ic_outage_x_congestion"] = (
                out["ic_outage_mw"] * out["congestion_flag_lag96"]
            ).astype("float32")
    n = len([c for c in out.columns if c not in df.columns])
    log.info(f"  [T]  Transmission REMIT (P6): {n} features")
    return out


# ==============================================================================
# BLOCK DH — GERMAN PUBLIC HOLIDAYS (§7.1, trivial; coupled-market demand)
# ==============================================================================
def add_de_holiday_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    try:
        import holidays as _hol
        de_ph = _hol.Germany(years=list(out.index.year.unique()))
        out["is_de_public_holiday"] = np.array(
            [1.0 if d in de_ph else 0.0 for d in out.index.date], dtype="float32"
        )
        log.info("  [DH] German public holidays: 1 feature")
    except Exception as e:
        log.warning(f"  [DH] German holidays skipped: {e}")
    return out


# ==============================================================================
# WRAPPER — single call site for both pipelines
# ==============================================================================
def apply_v2_feature_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all v2 ablation blocks in fixed order. Call AFTER
    add_regime_features and BEFORE targets/boosters in BOTH pipelines."""
    out = add_nordic_coupling_features(df)
    out = add_solar_limb_v2(out)
    out = add_de_wind_proxy(out)
    out = add_negative_price_features(out)   # needs clear_sky_index -> after S2
    out = add_ic_outage_features(out)
    out = add_de_holiday_flag(out)
    return out


# Raw v2 columns that must be DROPPED from the ML matrix (concurrent data —
# only the lagged derivatives above are legitimate model inputs)
V2_RAW_DROP = [
    "DA_Price_NO2_EURMWh",
    "Flow_NL_NO_MW", "Flow_NO_NL_MW",
    "Flow_NL_GB_MW", "Flow_GB_NL_MW",
    "IC_Outage_MW",   # replaced by named ic_outage_* features
]

# v2 additions to the leakage-audit concurrent list
V2_LEAKAGE_AUDIT = ["DA_Price_NO2_EURMWh", "Flow_NL_NO_MW", "Flow_NO_NL_MW"]
