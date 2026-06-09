"""
================================================================================
 VOLTCAST IPI — HISTORICAL BACKFILL MATCHER
 Purpose: Reconstruct master data for post-mortem analysis (May 17 -> Today)
================================================================================
"""
import os
import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from entsoe import EntsoePandasClient

# --- CONFIGURATION ---
START_LOCAL = "2026-05-17 00:00:00"
END_LOCAL   = pd.Timestamp.now(tz="Europe/Amsterdam").strftime("%Y-%m-%d 23:45:00")

ENTSOE_KEY  = os.environ.get("VOLTCAST_ENTSOE_KEY", "YOUR_KEY_HERE")
DUTCH_TZ    = "Europe/Amsterdam"

START_UTC   = pd.Timestamp(START_LOCAL, tz=DUTCH_TZ).tz_convert("UTC")
END_UTC     = pd.Timestamp(END_LOCAL, tz=DUTCH_TZ).tz_convert("UTC")
MASTER_IDX  = pd.date_range(start=START_UTC, end=END_UTC, freq="15min", inclusive="left")

NL_STATIONS = {
    "Amsterdam": {"lat": 52.37, "lon": 4.89}, "Rotterdam": {"lat": 51.92, "lon": 4.48},
    "Utrecht": {"lat": 52.09, "lon": 5.12}, "Eindhoven": {"lat": 51.44, "lon": 5.48},
    "Maastricht": {"lat": 50.85, "lon": 5.69}, "Deventer": {"lat": 52.26, "lon": 6.16},
    "Friesland": {"lat": 53.20, "lon": 5.80}
}

logging.basicConfig(level=logging.INFO, format="%(message)s")

def _snap(series: pd.Series) -> pd.Series:
    return series.resample("15min").ffill().reindex(MASTER_IDX)

# --- 1. ENTSO-E DATA ---
def fetch_entsoe_backfill():
    logging.info("📡 Fetching ENTSO-E Backfill...")
    client = EntsoePandasClient(api_key=ENTSOE_KEY)
    df = pd.DataFrame(index=MASTER_IDX)

    try:
        # 1. ACTUAL TARGET PRICES (For comparison)
        prices = client.query_day_ahead_prices("NL", start=START_UTC, end=END_UTC)
        df["DA_Price_NL_EURMWh"] = _snap(prices)

        # Neighbor Prices
        for z, col in {"DE_LU": "DA_Price_DE_EURMWh", "BE": "DA_Price_BE_EURMWh"}.items():
            df[col] = _snap(client.query_day_ahead_prices(z, start=START_UTC, end=END_UTC))

        # 2. TSO LOAD FORECASTS (What the model knew D-1)
        df["NL_TSO_Load_Forecast_MW"] = _snap(client.query_load_forecast("NL", start=START_UTC, end=END_UTC))
        
        # 3. ACTUAL LOAD (For lag features)
        df["NL_Actual_Load_MW"] = _snap(client.query_load("NL", start=START_UTC, end=END_UTC))

        # 4. CROSS-BORDER FLOWS
        df["Flow_NL_DE_MW"] = _snap(client.query_scheduled_exchanges("NL", "DE_LU", start=START_UTC, end=END_UTC))
        df["Flow_NL_BE_MW"] = _snap(client.query_scheduled_exchanges("NL", "BE", start=START_UTC, end=END_UTC))
        df["NL_Net_Export_MW"] = df["Flow_NL_DE_MW"] + df["Flow_NL_BE_MW"]

        # 5. RENEWABLE FORECASTS
        ren_fc = client.query_wind_and_solar_forecast("NL", start=START_UTC, end=END_UTC)
        if not ren_fc.empty:
            df["NL_Renewables_Forecast_MW"] = _snap(ren_fc.sum(axis=1))

    except Exception as e:
        logging.error(f"❌ ENTSO-E Error: {e}")

    return df

# --- 2. OPEN-METEO DATA ---
def fetch_weather_backfill():
    logging.info("🌤️ Fetching Open-Meteo Backfill...")
    vars_str = "temperature_2m,apparent_temperature,shortwave_radiation,direct_radiation,diffuse_radiation,wind_speed_100m,cloud_cover,relative_humidity_2m,precipitation"
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    frames = []
    for city, coords in NL_STATIONS.items():
        params = {
            "latitude": coords["lat"], "longitude": coords["lon"],
            "start_date": START_UTC.strftime("%Y-%m-%d"),
            "end_date": END_UTC.strftime("%Y-%m-%d"),
            "hourly": vars_str, "timezone": "UTC"
        }
        r = requests.get(url, params=params).json()
        if "hourly" in r:
            df_c = pd.DataFrame(r["hourly"])
            df_c["time"] = pd.to_datetime(df_c["time"]).dt.tz_localize("UTC")
            df_c.set_index("time", inplace=True)
            df_c.columns = [f"{c}_{city}" for c in df_c.columns]
            frames.append(df_c)
            time.sleep(1)

    if frames:
        df_w = pd.concat(frames, axis=1).resample("15min").ffill().reindex(MASTER_IDX)
        return df_w
    return pd.DataFrame(index=MASTER_IDX)

# --- 3. MACRO FUEL DATA ---
def fetch_macro_backfill():
    logging.info("🛢️ Fetching Macro Fuel Backfill...")
    df = pd.DataFrame(index=MASTER_IDX)
    start_str = START_UTC.strftime("%Y-%m-%d")
    end_str = (END_UTC + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
    
    tickers = {"TTF=F": "TTF_Gas_EURMWh", "KEUA": "EUA_Carbon_EUR"}
    for t, col in tickers.items():
        raw = yf.download(t, start=start_str, end=end_str, progress=False)
        if not raw.empty:
            close = raw["Close"].squeeze()
            close.index = pd.to_datetime(close.index).tz_localize("UTC")
            df[col] = close.reindex(MASTER_IDX).ffill().bfill()
            
    if "TTF_Gas_EURMWh" in df and "EUA_Carbon_EUR" in df:
        df["CCGT_Marginal_Cost_EUR"] = (df["TTF_Gas_EURMWh"]/0.56) + (df["EUA_Carbon_EUR"]*0.36)
        
    return df

if __name__ == "__main__":
    df_e = fetch_entsoe_backfill()
    df_w = fetch_weather_backfill()
    df_m = fetch_macro_backfill()

    master = pd.concat([df_e, df_w, df_m], axis=1).ffill(limit=4)
    filename = f"backfill_master_{START_UTC.strftime('%Y%m%d')}_{END_UTC.strftime('%Y%m%d')}.parquet"
    master.to_parquet(filename)
    logging.info(f"✅ Backfill complete. Saved to {filename} ({len(master)} rows)")