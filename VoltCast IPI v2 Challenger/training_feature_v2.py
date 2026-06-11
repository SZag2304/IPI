"""
================================================================================
 VOLTCAST IPI — FEATURE ENGINEERING PIPELINE (10:15 AM BID-READY VERSION)
 Target: Netherlands EPEX Day-Ahead Price (EUR/MWh) D+1 Forecast
================================================================================
"""
import numpy as np
import pandas as pd
from dateutil.easter import easter
import holidays
import config_v2 as config
from voltcast_v2_features import apply_v2_feature_blocks, V2_RAW_DROP, V2_LEAKAGE_AUDIT
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import logging
log = logging.getLogger("VoltCast.Features")

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("VoltCast")

TARGET_COL   = "DA_Price_NL_EURMWh"
INPUT_FILE   = "VoltCast_IPI_Master_v2.parquet"
OUTPUT_FILE  = "VoltCast_IPI_Features_v2.parquet"

CCGT_EFFICIENCY  = 0.56
CCGT_EMISSIONS   = 0.36
OCGT_EFFICIENCY  = 0.40
OCGT_EMISSIONS   = 0.50
HEATING_BASE     = 15.5
COOLING_BASE     = 22.0
CHEAP_PERCENTILE     = 25
EXPENSIVE_PERCENTILE = 75
PRICE_SPIKE_THRESHOLD_EUR = 150.0
NEGATIVE_PRICE_THRESHOLD  = 0.0
NL_THERMAL_CAPACITY_MW    = 11_000

NL_STATION_WEIGHTS = {
    "Amsterdam":  0.22, "Rotterdam":  0.22, "Utrecht":    0.18,
    "Eindhoven":  0.18, "Maastricht": 0.08, "Deventer":   0.06, "Friesland":  0.06,
}

# ==============================================================================
# 1. DUTCH SCHOOL HOLIDAY CALENDAR
# ==============================================================================
def build_nl_school_holidays(years):
    holidays_set = set()
    SCHEDULE = {
        2024: [(1,1,1,7),(2,10,2,18),(4,27,5,5),(7,20,9,1),(10,19,10,27),(12,21,1,5)],
        2025: [(1,1,1,5),(2,22,3,2),(4,12,4,27),(7,19,8,31),(10,18,10,26),(12,20,1,4)],
        2026: [(1,1,1,4),(2,14,2,22),(4,4,4,19),(7,18,8,30),(10,17,10,25),(12,19,1,3)],
    }
    for year in years:
        if year in SCHEDULE:
            for ms, ds, me, de in SCHEDULE[year]:
                ye = year+1 if me < ms else year
                for d in pd.date_range(pd.Timestamp(year,ms,ds), pd.Timestamp(ye,me,de), freq="D"):
                    holidays_set.add(d.date())
        else:
            log.warning(f"Using fallback holiday logic for year {year}")
            fallback_windows = [(2,20,2,28), (5,1,5,10), (7,15,8,31), (10,15,10,25), (12,22,1,5)]
            for ms, ds, me, de in fallback_windows:
                ye = year+1 if me < ms else year
                for d in pd.date_range(pd.Timestamp(year,ms,ds), pd.Timestamp(ye,me,de), freq="D"):
                    holidays_set.add(d.date())
    return holidays_set

# ==============================================================================
# 2. PREPROCESSING
# ==============================================================================
def preprocess(df):
    log.info("  [PREP] Cleaning master data...")
    out = df.copy()
    # 10:15 AM FIX: Removed ENTSO-E RES columns from ffill
    ffill_cols = ["DA_Price_NL_EURMWh","NL_TSO_Load_Forecast_MW",
                  "TTF_Gas_EURMWh","EUA_Carbon_EUR",
                  "CCGT_Marginal_Cost_EUR","NL_Net_Export_MW","NL_Thermal_Outage_MW"]
    for c in ffill_cols:
        if c in out.columns: out[c] = out[c].ffill(limit=4)
    weather_pfx = ["temperature_2m_","apparent_temperature_","shortwave_radiation_",
                   "wind_speed_100m_","wind_direction_100m_","cloud_cover_",
                   "relative_humidity_2m_","precipitation_"]
    for c in out.columns:
        if any(c.startswith(p) for p in weather_pfx):
            out[c] = out[c].ffill(limit=8)
    if "DA_Price_BE_EURMWh" in out.columns and "DA_Price_DE_EURMWh" in out.columns:
        mask = out["DA_Price_BE_EURMWh"].isna()
        if mask.sum() > 0:
            spread = (out["DA_Price_BE_EURMWh"] - out["DA_Price_DE_EURMWh"]).median()
            out.loc[mask,"DA_Price_BE_EURMWh"] = out.loc[mask,"DA_Price_DE_EURMWh"] + spread
    num = out.select_dtypes(include=[np.number]).columns
    out[num] = out[num].astype("float32")
    log.info(f"  [PREP] Done. Shape: {out.shape}")
    return out

# ==============================================================================
# 3. NATIONAL WEATHER AGGREGATION
# ==============================================================================
def add_national_weather(df):
    out = df.copy()
    for var in ["temperature_2m", "apparent_temperature", "shortwave_radiation",
                "direct_radiation", "diffuse_radiation", "wind_speed_100m", 
                "cloud_cover", "relative_humidity_2m", "precipitation"]:
        cols = [f"{var}_{c}" for c in NL_STATION_WEIGHTS if f"{var}_{c}" in out.columns]
        if not cols: continue
        ws = [NL_STATION_WEIGHTS[c.split("_")[-1]] for c in cols]
        tw = sum(ws)
        out[f"Nat_{var}"] = sum(out[c]*(w/tw) for c,w in zip(cols,ws)).astype("float32")
        out[f"Nat_{var}_var"] = pd.concat([out[c] for c in cols],axis=1).std(axis=1).astype("float32")
    return out

# ==============================================================================
# 4. TIME & CALENDAR FEATURES
# ==============================================================================
def add_time_calendar_features(df):
    out = df.copy()
    idx = out.index
    years = list(idx.year.unique())
    nl_ph = holidays.Netherlands(years=years)
    nl_sh = build_nl_school_holidays(years)

    out["hour"]    = idx.hour.astype("float32")
    out["dow"]     = idx.dayofweek.astype("float32")
    out["month"]   = idx.month.astype("float32")
    out["week"]    = idx.isocalendar().week.astype("float32")
    out["quarter"] = idx.quarter.astype("float32")

    out["hour_sin"]  = np.sin(2*np.pi*out["hour"]/24).astype("float32")
    out["hour_cos"]  = np.cos(2*np.pi*out["hour"]/24).astype("float32")
    out["dow_sin"]   = np.sin(2*np.pi*out["dow"]/7).astype("float32")
    out["dow_cos"]   = np.cos(2*np.pi*out["dow"]/7).astype("float32")
    out["month_sin"] = np.sin(2*np.pi*(out["month"]-1)/12).astype("float32")
    out["month_cos"] = np.cos(2*np.pi*(out["month"]-1)/12).astype("float32")
    out["week_sin"]  = np.sin(2*np.pi*out["week"]/52).astype("float32")
    out["week_cos"]  = np.cos(2*np.pi*out["week"]/52).astype("float32")

    out["is_weekend"]  = (out["dow"]>=5).astype("float32")
    out["is_monday"]   = (out["dow"]==0).astype("float32")
    out["is_friday"]   = (out["dow"]==4).astype("float32")

    ph = np.array([1 if d in nl_ph else 0 for d in idx.date],"float32")
    sh = np.array([1 if d in nl_sh else 0 for d in idx.date],"float32")
    out["is_public_holiday"]  = ph
    out["is_school_holiday"]  = sh

    ph_s = pd.Series(ph,index=idx)
    sh_s = pd.Series(sh,index=idx)
    out["is_bridge_day"] = (
        ((ph_s.shift(96)==1)&(out["dow"]==4)) | ((ph_s.shift(-96).fillna(0)==1)&(out["dow"]==0))
    ).astype("float32")
    out["is_post_holiday_monday"]  = ((out["is_monday"]==1)&(ph_s.shift(96)==1)).astype("float32")
    out["is_post_school_break"]    = ((out["is_monday"]==1)&(sh_s.shift(96)==1)&(sh_s==0)).astype("float32")

    h = out["hour"]
    out["is_night_baseload"] = h.between(0,5).astype("float32")
    out["is_morning_ramp"]   = h.between(6,9).astype("float32")
    out["is_midday_solar"]   = h.between(10,15).astype("float32")
    out["is_evening_peak"]   = h.between(16,20).astype("float32")
    out["is_late_evening"]   = h.between(21,23).astype("float32")

    out["is_winter"]      = out["month"].isin([12,1,2]).astype("float32")
    out["is_spring"]      = out["month"].isin([3,4,5]).astype("float32")
    out["is_summer"]      = out["month"].isin([6,7,8]).astype("float32")
    out["is_autumn"]      = out["month"].isin([9,10,11]).astype("float32")
    out["is_solar_season"]= out["month"].isin([3,4,5,6,7,8,9]).astype("float32")

    n = len([c for c in out.columns if c not in df.columns])
    log.info(f"  [A] Time & calendar: {n} features")
    return out

# ==============================================================================
# 4.1. SPRING STRUCTURAL FEATURES
# ==============================================================================
def add_spring_structural_features(df):
    out = df.copy()
    idx = out.index.tz_convert("Europe/Amsterdam")

    dst_dates = set()
    for year in idx.year.unique():
        mar31 = pd.Timestamp(year, 3, 31, tz="Europe/Amsterdam")
        last_sun = mar31 - pd.DateOffset(days=mar31.dayofweek + 1 if mar31.dayofweek != 6 else 0)
        for d in pd.date_range(last_sun, periods=10, freq='D'):
            dst_dates.add(d.date())
            
    out['is_dst_transition_week'] = pd.Series(
        [1.0 if d.date() in dst_dates else 0.0 for d in idx], 
        index=out.index, dtype='float32'
    )

    easter_dates = set()
    for year in idx.year.unique():
        e_sunday = easter(year)
        for offset in [-2, -1, 0, 1]:
            easter_dates.add(e_sunday + pd.Timedelta(days=offset))

    out['is_easter_period'] = pd.Series(
        [1.0 if d.date() in easter_dates else 0.0 for d in idx], 
        index=out.index, dtype='float32'
    )

    return out

# ==============================================================================
# 5. DEMAND FEATURES
# ==============================================================================
def add_demand_features(df):
    out = df.copy()

    if "NL_TSO_Load_Forecast_MW" in out.columns:
        fc = out["NL_TSO_Load_Forecast_MW"]
        out["tso_fc_mw"]            = fc
        out["tso_fc_lag96"]         = fc.shift(96).astype("float32")
        out["tso_fc_lag672"]        = fc.shift(672).astype("float32")
        out["tso_fc_delta_day"]     = (fc - fc.shift(96)).astype("float32")
        out["tso_fc_delta_week"]    = (fc - fc.shift(672)).astype("float32")
        daily_avg = fc.resample("D").transform("mean")
        out["tso_fc_intraday_dev"]  = (fc - daily_avg).astype("float32")

    for col in ["NL_Actual_Load_MW","NL_Load_MW"]:
        if col in out.columns:
            out["actual_load_lag96"]   = out[col].shift(96).astype("float32")
            out["actual_load_lag192"]  = out[col].shift(192).astype("float32")
            out["actual_load_lag672"]  = out[col].shift(672).astype("float32")
            out["actual_load_roll24h_lag96"] = out[col].shift(96).rolling(96,min_periods=48).mean().astype("float32")
            break

    for city in NL_STATION_WEIGHTS:
        app = f"apparent_temperature_{city}"
        if app in out.columns:
            out[f"HDD_{city}"] = (HEATING_BASE - out[app]).clip(lower=0).astype("float32")
            out[f"CDD_{city}"] = (out[app] - COOLING_BASE).clip(lower=0).astype("float32")

    hdd_cols = [f"HDD_{c}" for c in NL_STATION_WEIGHTS if f"HDD_{c}" in out.columns]
    cdd_cols = [f"CDD_{c}" for c in NL_STATION_WEIGHTS if f"CDD_{c}" in out.columns]
    if hdd_cols:
        ws = [NL_STATION_WEIGHTS[c.split("_")[1]] for c in hdd_cols]
        out["HDD_national"] = sum(out[c]*w for c,w in zip(hdd_cols,ws)).astype("float32")
    if cdd_cols:
        ws = [NL_STATION_WEIGHTS[c.split("_")[1]] for c in cdd_cols]
        out["CDD_national"] = sum(out[c]*w for c,w in zip(cdd_cols,ws)).astype("float32")

    if "HDD_national" in out.columns:
        out["HDD_3d_inertia"] = out["HDD_national"].ewm(span=288,adjust=False).mean().astype("float32")

    if "Nat_apparent_temperature" in out.columns:
        at = out["Nat_apparent_temperature"]
        out["temp_shock_vs_lastweek"]  = (at - at.shift(672)).astype("float32")
        out["temp_shock_vs_yesterday"] = (at - at.shift(96)).astype("float32")
        out["temp_rate_1h"] = (at - at.shift(4)).astype("float32")
        out["temp_rate_6h"] = (at - at.shift(24)).astype("float32")

    if "is_morning_ramp" in out.columns and "HDD_national" in out.columns:
        out["morning_heating_demand"] = (out["is_morning_ramp"]*out["HDD_national"]).astype("float32")

    if "NL_TSO_Load_Forecast_MW" in out.columns and "actual_load_lag96" in out.columns:
        out["tso_bias_yesterday"] = (out["NL_TSO_Load_Forecast_MW"].shift(96) - out["actual_load_lag96"]).astype("float32")

    for country in ["DE_LU","BE"]:
        col = f"Load_Forecast_{country}_MW"
        if col in out.columns:
            out[f"neighbor_load_fc_{country}"] = out[col].astype("float32")
            out[f"neighbor_load_fc_delta_{country}"] = (out[col]-out[col].shift(96)).astype("float32")

    n = len([c for c in out.columns if c not in df.columns])
    log.info(f"  [B] Demand: {n} features")
    return out

# ==============================================================================
# 6. WEATHER-PHYSICS BRIDGE (10:15 AM REPLACEMENT FOR BLOCK C)
# ==============================================================================
def add_weather_physics_features(df):
    out = df.copy()
    
    # 1. Cubic Wind Power Curve Proxy (P ∝ v^3)
    if "Nat_wind_speed_100m" in out.columns:
        ws = out["Nat_wind_speed_100m"]
        out["wind_power_proxy_cubed"] = (ws.clip(0, 25) ** 3).astype("float32")
        out["wind_proxy_lag96"] = out["wind_power_proxy_cubed"].shift(96).astype("float32")
        out["wind_48h_collapse_proxy"] = (out["wind_power_proxy_cubed"].shift(192) - out["wind_proxy_lag96"]).clip(lower=0).astype("float32")

    # 2. Solar Radiation Proxy
    if "Nat_shortwave_radiation" in out.columns:
        rad = out["Nat_shortwave_radiation"]
        out["solar_proxy"] = rad.astype("float32")
        if "is_midday_solar" in out.columns:
            out["weather_duck_curve_depth"] = (rad * out["is_midday_solar"]).astype("float32")
    
    # [NEW] ADD THESE LINES HERE
    if "Nat_direct_radiation" in out.columns:
        out["direct_radiation_lag96"] = out["Nat_direct_radiation"].shift(96).astype("float32")
    
    if "Nat_diffuse_radiation" in out.columns:
        out["diffuse_radiation_lag96"] = out["Nat_diffuse_radiation"].shift(96).astype("float32")
    # [END NEW]

    # 3. The Pseudo-Scarcity Index (Replaces Residual Load)
    if "NL_TSO_Load_Forecast_MW" in out.columns and "wind_power_proxy_cubed" in out.columns and "solar_proxy" in out.columns:
        load_norm = out["NL_TSO_Load_Forecast_MW"] / 20000.0 # Approx peak NL load
        wind_norm = out["wind_power_proxy_cubed"] / (12**3)  # Approx max useful wind cubed
        solar_norm = out["solar_proxy"] / 800.0              # Approx max radiation
        
        # Scarcity = Load - (Effective Weather Supply)
        out["weather_scarcity_index"] = (load_norm - (wind_norm * 0.4 + solar_norm * 0.6)).astype("float32")
        
        # Scarcity Amplifier (replaces old residual_load multiplier)
        if "mc_ccgt" in out.columns:
            out["scarcity_amplifier_proxy"] = (out["weather_scarcity_index"].clip(lower=0) ** 2 * out["mc_ccgt"]).astype("float32")

    # 4. Duck Curve Neck Stress (Weather version)
    if "solar_proxy" in out.columns and "tso_fc_mw" in out.columns:
        solar_drop = (out["solar_proxy"].shift(4) - out["solar_proxy"]).clip(lower=0)
        load_rise = (out["tso_fc_mw"] - out["tso_fc_mw"].shift(4)).clip(lower=0)
        out["duck_curve_ramp_stress"] = (solar_drop * (load_rise / 1000)).astype("float32")

    n = len([c for c in out.columns if c not in df.columns])
    log.info(f"  [C_NEW] Weather Physics Bridge: {n} features")
    return out

# ==============================================================================
# 7. MACRO / THERMAL FEATURES
# ==============================================================================
def add_macro_thermal_features(df):
    out = df.copy()
    ttf_col = "TTF_Gas_EURMWh"
    eua_col = "EUA_Carbon_EUR"

    # 1. GAS FEATURES
    if ttf_col in out.columns:
        ttf = out[ttf_col]
        # v2 P3 PARITY: train on the D-2 close (shift 192) because that is all
        # the 10:10 CET live fetch can ever see (ICE settles ~17:00). Live keeps
        # shift(96) on its 1-day-stale series -> identical information content.
        ttf_safe = ttf.shift(config.GAS_LAG_PTU_TRAINING).astype("float32")
        out["ttf_spot"]          = ttf_safe
        out["ttf_lag7d"]         = ttf.shift(config.GAS_LAG_PTU_TRAINING + 672).astype("float32") 
        out["ttf_delta_7d"]      = (ttf_safe - out["ttf_lag7d"]).astype("float32")
        out["ttf_delta_1d"]      = (ttf_safe - ttf.shift(config.GAS_LAG_PTU_TRAINING + 96)).astype("float32")
        roll_mean = ttf_safe.rolling(672, min_periods=96).mean()
        roll_std  = ttf_safe.rolling(672, min_periods=96).std().replace(0, np.nan)
        out["ttf_roll7d_mean"]   = roll_mean.astype("float32")
        out["ttf_roll7d_zscore"] = ((ttf_safe - roll_mean) / roll_std).astype("float32")
        out["ttf_regime_high"]   = (out["ttf_roll7d_zscore"] > 1.0).astype("float32")
        out["ttf_regime_low"]    = (out["ttf_roll7d_zscore"] < -1.0).astype("float32")

    # 2. CARBON FEATURES
    if eua_col in out.columns:
        eua = out[eua_col]
        eua_safe = eua.shift(config.GAS_LAG_PTU_TRAINING).astype("float32")  # v2 P3 parity
        out["eua_spot"]          = eua_safe
        out["eua_delta_7d"]      = (eua_safe - eua.shift(config.GAS_LAG_PTU_TRAINING + 672)).astype("float32")
        out["eua_roll7d_mean"]   = eua_safe.rolling(672, min_periods=96).mean().astype("float32")

    # 3. MARGINAL COST
    if "ttf_spot" in out.columns and "eua_spot" in out.columns:
        mc_ccgt = out["ttf_spot"] / CCGT_EFFICIENCY + out["eua_spot"] * CCGT_EMISSIONS
        mc_ocgt = out["ttf_spot"] / OCGT_EFFICIENCY + out["eua_spot"] * OCGT_EMISSIONS
        out["mc_ccgt"]              = mc_ccgt.astype("float32")
        out["mc_ocgt"]              = mc_ocgt.astype("float32")
        out["mc_ccgt_lag7d"]        = mc_ccgt.shift(672).astype("float32")
        out["mc_ccgt_delta_7d"]     = (mc_ccgt - out["mc_ccgt_lag7d"]).astype("float32")
        out["mc_spread_ccgt_ocgt"]  = (mc_ocgt - mc_ccgt).astype("float32")

    # 4. THERMAL OUTAGES & SUPPLY CRUNCH (10:15 FIX: Base on Raw Load, not Residual)
    if "NL_Thermal_Outage_MW" in out.columns:
        to = out["NL_Thermal_Outage_MW"]
        out["thermal_outage_mw"]       = to.astype("float32")
        out["thermal_outage_roll24h"]  = to.rolling(96,min_periods=1).mean().astype("float32")
        if "tso_fc_mw" in out.columns:
            out["supply_crunch_signal"] = (out["tso_fc_mw"] + to).astype("float32")

    if "thermal_outage_mw" in out.columns and "tso_fc_mw" in out.columns:
        avail = NL_THERMAL_CAPACITY_MW - out["thermal_outage_mw"]
        out["supply_margin_mw"]  = (avail - out["tso_fc_mw"]).astype("float32")
        out["supply_margin_pct"] = (out["supply_margin_mw"]/NL_THERMAL_CAPACITY_MW).clip(-0.5,1.5).astype("float32")

    for col, label in [("NL_Fossil_Gas_Actual_MW", "gas_actual"), ("NL_Coal_Actual_MW", "coal_actual")]:
        if col in out.columns:
            out[f"{label}_lag192"]  = out[col].shift(192).astype("float32")
            out[f"{label}_lag768"] = out[col].shift(768).astype("float32")
            out[f"{label}_delta_24h"] = (out[col].shift(192) - out[col].shift(288)).astype("float32")
            out = out.drop(columns=[col])

    n = len([c for c in out.columns if c not in df.columns])
    log.info(f"  [D] Macro/thermal: {n} features")
    return out

# ==============================================================================
# 8. MARKET COUPLING FEATURES
# ==============================================================================
def add_market_coupling_features(df):
    out = df.copy()
    price_cols = {
        "DA_Price_DE_EURMWh":"de","DA_Price_BE_EURMWh":"be",
        "DA_Price_SE3_EURMWh":"se3","DA_Price_FR_EURMWh":"fr",
    }
    
    for col,abbr in price_cols.items():
        if col not in out.columns: continue
        p = out[col]
        out[f"{abbr}_price_lag96"]          = p.shift(96).astype("float32")
        out[f"{abbr}_price_lag192"]         = p.shift(192).astype("float32")
        out[f"{abbr}_price_lag672"]         = p.shift(672).astype("float32")
        out[f"{abbr}_price_roll7d_lag96"]   = p.shift(96).rolling(672,min_periods=96).mean().astype("float32")
        out[f"{abbr}_price_daily_avg_lag96"]= p.resample("D").transform("mean").shift(96).astype("float32")

    if TARGET_COL in out.columns and "DA_Price_DE_EURMWh" in out.columns:
        nl_l = out[TARGET_COL].shift(96)
        de_l = out["DA_Price_DE_EURMWh"].shift(96)
        out["spread_nl_de_lag96"]    = (nl_l-de_l).astype("float32")
        out["spread_nl_de_lag672"]   = (out[TARGET_COL].shift(672)-out["DA_Price_DE_EURMWh"].shift(672)).astype("float32")
        out["spread_nl_de_momentum"] = ((out["spread_nl_de_lag96"]-out["spread_nl_de_lag672"])/7).astype("float32")
        out["congestion_flag_lag96"] = (out["spread_nl_de_lag96"].abs()>10.0).astype("float32")

    for col,nm in [("Flow_NL_DE_MW","flow_nl_de"),("Flow_NL_BE_MW","flow_nl_be"),("NL_Net_Export_MW","nl_net_export")]:
        if col in out.columns:
            out[nm]                 = out[col].astype("float32")
            out[f"{nm}_lag96"]      = out[col].shift(96).astype("float32")
            out[f"{nm}_roll24h"]    = out[col].rolling(96,min_periods=1).mean().astype("float32")

    if "DA_Price_SE3_EURMWh" in out.columns and "DA_Price_DE_EURMWh" in out.columns:
        out["nordic_de_spread_lag96"] = (out["DA_Price_SE3_EURMWh"].shift(96)-out["DA_Price_DE_EURMWh"].shift(96)).astype("float32")

    nuclear_cols = [c for c in ['be_price_lag96', 'fr_price_lag96'] if c in out.columns]
    if nuclear_cols:
        out['neighbor_nuclear_pressure'] = out[nuclear_cols].min(axis=1).astype("float32")
    else:
        out['neighbor_nuclear_pressure'] = out.get('de_price_lag96', 0)

    if "nl_net_export_lag96" in out.columns:
        out['export_saturation_risk'] = (out['nl_net_export_lag96'] / 6000).clip(0, 1) ** 2
        out['export_saturation_risk'] = out['export_saturation_risk'].astype("float32")

    n = len([c for c in out.columns if c not in df.columns])
    log.info(f"  [E] Market coupling: {n} features")
    return out

def add_advanced_physics_features(df):
    out = df.copy()
    THERMAL_CAPACITY_NL = 11000 # MW

    if "NL_Thermal_Outage_MW" in out.columns:
        out["effective_thermal_supply"] = (THERMAL_CAPACITY_NL - out["NL_Thermal_Outage_MW"]).astype("float32")
        
    if "be_price_lag96" in out.columns and "de_price_lag96" in out.columns:
        out["neighbor_spread_momentum"] = (out["be_price_lag96"] - out["de_price_lag96"]).astype("float32")

    return out

def add_regime_stabilizers(df):
    out = df.copy()
    if 'de_price_lag96' in out.columns:
        out['is_de_negative_lag96'] = (out['de_price_lag96'] < 0).astype('float32')
    return out

def add_2026_market_drivers(df):
    out = df.copy()
    if "NL_Imbalance_Price_EURMWh" in out.columns:
        out['imb_momentum_lag96'] = out['NL_Imbalance_Price_EURMWh'].shift(96).rolling(4).mean()
    return out

def add_expensive_regime_boosters(df):
    out = df.copy()
    
    if "de_price_lag96" in out.columns and "be_price_lag96" in out.columns and "price_lag96" in out.columns:
        neighbor_avg = (out["de_price_lag96"] + out["be_price_lag96"]) / 2
        out["neighbor_pull_force"] = (neighbor_avg - out["price_lag96"]).clip(lower=0)

    if "supply_margin_mw" in out.columns:
        out["scarcity_exponential"] = 1 / (out["supply_margin_mw"].clip(lower=100) / 1000)

    if 'supply_margin_mw' in out.columns and 'price_lag96' in out.columns and 'price_lag192' in out.columns:
        margin_norm = (1 / out['supply_margin_mw'].clip(lower=50) * 1000).clip(0, 10)
        price_momentum = (out['price_lag96'] - out['price_lag192']).clip(lower=0)
        out['scarcity_momentum_amplifier'] = (margin_norm * price_momentum / 100).astype('float32')

    return out

# ==============================================================================
# 9. PRICE DYNAMICS FEATURES
# ==============================================================================
def add_price_dynamics_features(df):
    out = df.copy()
    if TARGET_COL not in out.columns:
        log.info("  [F] WARNING: target column missing"); return out
    p = out[TARGET_COL]

    out["price_lag96"]   = p.shift(96).astype("float32")
    out["price_lag192"]  = p.shift(192).astype("float32")
    out["price_lag288"]  = p.shift(288).astype("float32")
    out["price_lag672"]  = p.shift(672).astype("float32")
    out["price_lag1344"] = p.shift(1344).astype("float32")

    daily_avg = p.resample("D").transform("mean")
    out["price_daily_avg_lag1d"] = daily_avg.shift(96).astype("float32")
    out["price_daily_avg_lag2d"] = daily_avg.shift(192).astype("float32")
    out["price_daily_avg_lag7d"] = daily_avg.shift(672).astype("float32")

    out["price_mom_1d_vs_2d"] = (p.shift(96)-p.shift(192)).astype("float32")
    out["price_mom_1w_vs_2w"] = (p.shift(672)-p.shift(1344)).astype("float32")
    out["price_ema_7d"]       = p.shift(96).ewm(span=672,adjust=False).mean().astype("float32")
    out["price_ema_3d"]       = p.shift(96).ewm(span=288,adjust=False).mean().astype("float32")
    out["price_trend_7d"]     = (out["price_ema_3d"]-out["price_ema_7d"]).astype("float32")

    out["price_vol_7d"]  = p.shift(96).rolling(672,min_periods=96).std().astype("float32")
    out["price_vol_30d"] = p.shift(96).rolling(2880,min_periods=672).std().astype("float32")

    out["price_1w_max_lag96"]   = p.shift(96).rolling(672,min_periods=96).max().astype("float32")
    out["price_1w_min_lag96"]   = p.shift(96).rolling(672,min_periods=96).min().astype("float32")
    out["price_1w_range_lag96"] = (out["price_1w_max_lag96"]-out["price_1w_min_lag96"]).astype("float32")

    p_lag = p.shift(96)
    hour_avg = p_lag.groupby(out.index.hour).transform(lambda x: x.rolling(30*24,min_periods=24).mean())
    out["price_hourly_profile_lag"] = hour_avg.astype("float32")
    out["price_intraday_dev_lag96"] = (p_lag - hour_avg).astype("float32")

    out["spike_flag_lag96"]         = (p.shift(96)>PRICE_SPIKE_THRESHOLD_EUR).astype("float32")
    out["neg_price_flag_lag96"]     = (p.shift(96)<NEGATIVE_PRICE_THRESHOLD).astype("float32")
    out["neg_price_count_7d_lag96"] = (p.shift(96)<0).rolling(672,min_periods=96).sum().astype("float32")

    n = len([c for c in out.columns if c not in df.columns])
    log.info(f"  [F] Price dynamics: {n} features")
    return out

# ==============================================================================
# 10. INTERACTION FEATURES
# ==============================================================================
def add_interaction_features(df):
    out = df.copy()
    safe_max = lambda s,w: s.rolling(w,min_periods=96).max().replace(0,np.nan)
    safe_mean= lambda s,w: s.rolling(w,min_periods=96).mean().replace(0,np.nan)

    if "HDD_national" in out.columns and "wind_power_proxy_cubed" in out.columns:
        w_norm = (out["wind_power_proxy_cubed"]/safe_max(out["wind_power_proxy_cubed"],672)).clip(0,1)
        c_norm = (out["HDD_national"]/safe_max(out["HDD_national"],672).fillna(1)).clip(0,1)
        out["cold_lowwind_stress"] = (c_norm*(1-w_norm)).astype("float32")

    if "is_post_school_break" in out.columns and "HDD_national" in out.columns:
        out["rebound_cold_stress"] = (out["is_post_school_break"]*out["HDD_national"]).astype("float32")

    # [NEW] ADD THESE LINES HERE
    if "solar_proxy" in out.columns:
        # Spring interaction: High solar + variable heating demand
        if "is_spring" in out.columns:
            out["spring_solar_interaction"] = (out["is_spring"] * out["solar_proxy"]).astype("float32")
        
        # Summer interaction: Maximum curtailment & negative price risk
        if "is_summer" in out.columns:
            out["summer_solar_interaction"] = (out["is_summer"] * out["solar_proxy"]).astype("float32")
    # [END NEW]

    if "ttf_spot" in out.columns and "thermal_outage_mw" in out.columns:
        ttf_n = (out["ttf_spot"]/safe_mean(out["ttf_spot"],2880)).clip(0,3)
        out_n = (out["thermal_outage_mw"]/safe_max(out["thermal_outage_mw"],2880).fillna(1)).clip(0,1)
        out["gas_outage_interaction"] = (ttf_n*out_n).astype("float32")

    if "is_monday" in out.columns and "is_morning_ramp" in out.columns:
        out["monday_morning_ramp"] = (out["is_monday"]*out["is_morning_ramp"]).astype("float32")

    n = len([c for c in out.columns if c not in df.columns])
    log.info(f"  [G] Interactions: {n} features")
    return out

# ==============================================================================
# 11. REGIME FEATURES
# ==============================================================================
def add_regime_features(df):
    out = df.copy()
    neg = pd.Series(0.0,index=out.index)
    if "is_weekend" in out.columns:           neg += out["is_weekend"]*0.5
    if "is_midday_solar" in out.columns and "is_solar_season" in out.columns:
        neg += out["is_midday_solar"]*out["is_solar_season"]*0.5
    if "neg_price_count_7d_lag96" in out.columns:
        neg += (out["neg_price_count_7d_lag96"]>3).astype(float)
    out["neg_price_risk_score"] = neg.clip(0,4).astype("float32")

    spk = pd.Series(0.0,index=out.index)
    if "supply_margin_pct" in out.columns: spk += (out["supply_margin_pct"]<0.15).astype(float)*2
    if "ttf_regime_high" in out.columns:   spk += out["ttf_regime_high"]
    if "cold_lowwind_stress" in out.columns: spk += (out["cold_lowwind_stress"]>0.5).astype(float)
    if "congestion_flag_lag96" in out.columns: spk += out["congestion_flag_lag96"]
    out["price_spike_risk_score"] = spk.clip(0,4).astype("float32")

    # 10:15 AM FIX: Tightness index is now Scarcity vs TTF
    if "weather_scarcity_index" in out.columns and "mc_ccgt" in out.columns:
        def zscore(s): return ((s-s.rolling(672,min_periods=96).mean())/s.rolling(672,min_periods=96).std().replace(0,1))
        out["market_tightness_index"] = ((zscore(out["weather_scarcity_index"])+zscore(out["mc_ccgt"]))/2).clip(-4,4).astype("float32")

    if "price_vol_7d" in out.columns and "price_vol_30d" in out.columns:
        out["vol_regime_elevated"] = (out["price_vol_7d"]>out["price_vol_30d"]*1.5).astype("float32")

    n = len([c for c in out.columns if c not in df.columns])
    log.info(f"  [H] Regimes: {n} features")
    return out

# ==============================================================================
# 12. TARGETS
# ==============================================================================
def add_targets(df):
    out = df.copy()
    if TARGET_COL not in out.columns:
        log.info("  [TARGET] WARNING: target column missing"); return out
    p = out[TARGET_COL]
    rl = p.rolling(2880,min_periods=672)
    regime = pd.Series(1,index=out.index,dtype="float32")
    regime[p <= rl.quantile(CHEAP_PERCENTILE/100)]     = 0
    regime[p >= rl.quantile(EXPENSIVE_PERCENTILE/100)] = 2
    out["price_regime_label"] = regime
    out["daily_avg_price"]    = p.resample("D").transform("mean").astype("float32")
    out["is_spike_hour"]      = (p>PRICE_SPIKE_THRESHOLD_EUR).astype("float32")
    out["is_negative_hour"]   = (p<NEGATIVE_PRICE_THRESHOLD).astype("float32")
    dist = regime.value_counts().sort_index()
    log.info(f"  [TARGET] Regime: Cheap={dist.get(0,0):,}  Normal={dist.get(1,0):,}  Expensive={dist.get(2,0):,}")
    return out

# ==============================================================================
# 13. LEAKAGE AUDIT
# ==============================================================================
def run_leakage_audit(df):
    log.info("\n" + "="*65)
    log.info("  LEAKAGE AUDIT")
    log.info("="*65)
    CONCURRENT = ["NL_Actual_Load_MW","NL_Load_MW","NL_Solar_MW","NL_Wind_MW",
                  "NL_Imbalance_Price_EURMWh","DA_Price_DE_EURMWh","DA_Price_BE_EURMWh",
                  "DA_Price_SE3_EURMWh","DA_Price_FR_EURMWh"] + V2_LEAKAGE_AUDIT
    issues = 0
    for c in CONCURRENT:
        if c in df.columns:
            log.info(f"  [WARNING] {c} — concurrent, must be lagged or excluded"); issues += 1
    if issues == 0: log.info("  [OK] No concurrent columns in feature set")
    if TARGET_COL in df.columns:
        num = df.select_dtypes(include=[np.number])
        high = [(c,abs(num[TARGET_COL].corr(num[c]))) for c in num.columns
                if c not in [TARGET_COL,"price_regime_label","daily_avg_price","is_spike_hour","is_negative_hour"]]
        high = [(c,r) for c,r in high if r>0.90 and not pd.isna(r)]
        if high:
            log.info("\n  [HIGH CORRELATION > 0.90 — investigate]:")
            for c,r in sorted(high,key=lambda x:-x[1])[:10]:
                log.info(f"    {c:<52} r={r:.3f}")
        else:
            log.info("  [OK] No suspiciously high correlations found")
    log.info("="*65+"\n")

# ==============================================================================
# 14. BUILD ML MATRIX
# ==============================================================================
def build_ml_matrix(df):
    # ── CRON: 10:15 CET — BEFORE ENTSO-E D+1 RENEWABLE FORECASTS ARE PUBLISHED ──
    # ALL ENTSO-E renewable forecasts MUST be dropped here to prevent fatal target leakage.
    DROP = [
        # ── Raw data actuals (real-time, not D+1 forecasts — must be lagged) ──
        "NL_Actual_Load_MW", "NL_Load_MW", "NL_Solar_MW", "NL_Wind_MW",
        "NL_Imbalance_Price_EURMWh",
        "DA_Price_DE_EURMWh", "DA_Price_BE_EURMWh", 
        "DA_Price_SE3_EURMWh", "DA_Price_FR_EURMWh",

        # ── Forecasts (10:15 AM FIX: RENEWABLES NOW DROPPED) ──
        "NL_Wind_Forecast_MW", "NL_Solar_Forecast_MW",
        "NL_Renewables_Forecast_MW",
        "DE_Renewables_Forecast_MW",
        
        # ── Per-city weather (aggregated to Nat_ versions) ──
        *[f"temperature_2m_{c}"       for c in NL_STATION_WEIGHTS],
        *[f"apparent_temperature_{c}" for c in NL_STATION_WEIGHTS],
        *[f"shortwave_radiation_{c}"  for c in NL_STATION_WEIGHTS],
        *[f"wind_speed_100m_{c}"      for c in NL_STATION_WEIGHTS],
        *[f"wind_direction_100m_{c}"  for c in NL_STATION_WEIGHTS],
        *[f"cloud_cover_{c}"          for c in NL_STATION_WEIGHTS],
        *[f"relative_humidity_2m_{c}" for c in NL_STATION_WEIGHTS],
        *[f"precipitation_{c}"        for c in NL_STATION_WEIGHTS],
        *[f"HDD_{c}" for c in NL_STATION_WEIGHTS],
        *[f"CDD_{c}" for c in NL_STATION_WEIGHTS],
        
        # ── Cross-border flow actuals (real-time, not forecasts) ──
        "Flow_NL_DE_MW", "Flow_NL_BE_MW", "NL_Net_Export_MW", "Flow_DE_NL_MW",
        "flow_nl_de", "flow_nl_be", "nl_net_export",
        "flow_nl_de_roll24h", "flow_nl_be_roll24h", "nl_net_export_roll24h",
        
        # ── Neighbor load forecasts (raw — derived features kept) ──
        "Load_Forecast_DE_LU_MW", "Load_Forecast_BE_MW",
        "neighbor_load_fc_DE_LU", "neighbor_load_fc_BE",
        "neighbor_load_fc_delta_DE_LU", "neighbor_load_fc_delta_BE",
        
        # ── Targets ──
        "price_regime_label", "daily_avg_price", "is_spike_hour", "is_negative_hour",
        "spike_flag_lag96",

        # v2: raw Nordic / transmission columns (lagged derivatives kept)
        *V2_RAW_DROP,
    ]
    
    y_price  = df[TARGET_COL].copy()         if TARGET_COL in df.columns            else pd.Series(dtype="float32")
    y_regime = df["price_regime_label"].copy() if "price_regime_label" in df.columns else pd.Series(dtype="float32")
    y_daily  = df["daily_avg_price"].copy()   if "daily_avg_price" in df.columns     else pd.Series(dtype="float32")
    X = df.drop(columns=[c for c in DROP+[TARGET_COL] if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf,-np.inf],np.nan).dropna(axis=1,how="all")
    return X, y_price, y_regime, y_daily

# ==============================================================================
# 15. MASTER PIPELINE
# ==============================================================================
if __name__ == "__main__":
    log.info("="*65)
    log.info("  VOLTCAST IPI — FEATURE ENGINEERING PIPELINE (10:15 AM)")
    log.info("="*65)

    log.info(f"\n[LOAD] Reading {INPUT_FILE}...")
    df_raw = pd.read_parquet(INPUT_FILE)
    log.info(f"  Raw shape: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} cols")

    log.info("\n[PIPELINE] Running feature blocks...")
    df = preprocess(df_raw)
    df = add_national_weather(df)
    df = add_time_calendar_features(df)
    df = add_demand_features(df)
    
    # 10:15 AM FIX: REPLACED BLOCK 6
    df = add_weather_physics_features(df)
    
    df = add_spring_structural_features(df)
    df = add_macro_thermal_features(df)
    df = add_market_coupling_features(df)
    df = add_advanced_physics_features(df)
    df = add_regime_stabilizers(df)
    df = add_2026_market_drivers(df)
    df = add_price_dynamics_features(df)
    df = add_interaction_features(df)
    df = add_regime_features(df)

    # v2 FORENSIC ABLATION BLOCKS (shared module -> train/serve parity)
    df = apply_v2_feature_blocks(df)

    df = add_targets(df)
    df = add_expensive_regime_boosters(df)
    

    run_leakage_audit(df)

    log.info("[MATRIX] Building X, y vectors...")
    X, y_price, y_regime, y_daily = build_ml_matrix(df)

    WARMUP = 1344
    X=X.iloc[WARMUP:]; y_price=y_price.iloc[WARMUP:]
    y_regime=y_regime.iloc[WARMUP:]; y_daily=y_daily.iloc[WARMUP:]
    X = X.dropna(how="all")
    idx = X.index.intersection(y_price.index)
    X=X.loc[idx]; y_price=y_price.loc[idx]; y_regime=y_regime.loc[idx]; y_daily=y_daily.loc[idx]

    df_out = X.copy()
    df_out[TARGET_COL]           = y_price
    df_out["price_regime_label"] = y_regime
    df_out["daily_avg_price"]    = y_daily
    df_out.to_parquet(OUTPUT_FILE)

    nan_pct = X.isna().mean().mean()*100
    log.info("\n"+"="*65)
    log.info("  FEATURE ENGINEERING COMPLETE")
    log.info("="*65)
    log.info(f"  Final X shape   : {X.shape[0]:,} rows x {X.shape[1]} features")
    log.info(f"  NaN rate (mean) : {nan_pct:.2f}%")
    if len(y_price):
        log.info(f"  Price range     : EUR{y_price.min():.1f} to EUR{y_price.max():.1f}/MWh")
        log.info(f"  Avg price       : EUR{y_price.mean():.1f}/MWh")
        log.info(f"  Spike hours     : {(y_price>PRICE_SPIKE_THRESHOLD_EUR).sum():,}")
        log.info(f"  Negative hours  : {(y_price<0).sum():,}")
    log.info(f"\n  Saved: {OUTPUT_FILE}")
    log.info("="*65)

    # 1. Broad search: Find anything in X that looks like a thermal lag
    thermal_keywords = ['thermal', 'gas', 'coal']
    found_lags = [col for col in X.columns if ('lag' in col.lower() or 'delta' in col.lower()) and any(kw in col.lower() for kw in thermal_keywords)]

    log.info(f"--- BROAD SEARCH ---")
    log.info(f"Found {len(found_lags)} thermal-related lag columns in X:")
    for col in found_lags:
        log.info(f"  - {col}")

    # 2. Exact check: Replace these with the actual names of the 6 lags you expect
    expected_lags = [
        "gas_actual_lag192",
        "gas_actual_lag768",
        "gas_actual_delta_24h",
        "coal_actual_lag192",
        "coal_actual_lag768",
        "coal_actual_delta_24h"
    ]

    log.info(f"\n--- EXACT MATCH CHECK ---")
    missing_lags = [col for col in expected_lags if col not in X.columns]
    present_lags = [col for col in expected_lags if col in X.columns]

    if missing_lags:
        log.info(f"❌ WARNING: You are missing these {len(missing_lags)} lags in X:")
        for m in missing_lags:
            log.info(f"  - {m}")
    else:
        log.info(f"✅ SUCCESS: All expected thermal lags are present in X!")