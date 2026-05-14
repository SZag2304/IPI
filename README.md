# ⚡ VoltCast IPI (Industrial Procurement Intelligence)

**An autonomous day-ahead price forecasting and procurement signal platform for the Netherlands EPEX Spot Market.**

## 📖 Overview
VoltCast IPI is a production-grade machine learning pipeline designed to optimize energy procurement for **Tier 3 Flexible Loads** (e.g., greenhouses, cold storage, water treatment facilities, and heavy manufacturing). 

Operating autonomously via a daily cron sequence, VoltCast generates a D+1 price forecast at 10:45 AM CET—a full 75 minutes before the EPEX spot market gate closes. By predicting price spikes and "duck curve" negative price windows, the system translates raw market data into actionable, confidence-stratified `BUY` and `AVOID` signals for energy brokers and automated dispatch systems.

## 🎯 Target Audience
* **Energy Brokers:** To route high-confidence signals to non-technical clients or automate API-driven trading.
* **Tier 3 Industrials:** Facilities with >10 GWh annual consumption and 5%–25% operational flexibility.

## ⚙️ Core Architecture & Methodology

VoltCast completely decouples from delayed Transmission System Operator (TSO) renewable publications using a proprietary data architecture and a robust ensemble model.

### 1. The "Physics Bridge"
Rather than relying on lagging institutional forecasts, VoltCast reconstructs the European grid's physical supply side using:
* High-resolution meteorology (Open-Meteo) across NL and DE interconnectors.
* Hard macroeconomic anchors (TTF Gas, EUA Carbon).
* Real-time thermal generation baselines and calculated efficiencies.

### 2. Tier 2 Ensemble Model
The prediction engine utilizes a highly tuned, walk-forward cross-validated ensemble:
* **Algorithms:** XGBoost (Histogram) and LightGBM (Leaf-wise).
* **Target Transformation:** Symmetric Logarithmic (Symlog) transformation to handle heavy-tailed distribution and negative spot prices.
* **Conformal Prediction:** Outputs strict P10 (lower bound) and P90 (upper bound) risk intervals.
* **Stratified Classification:** Out-of-sample calibrated precision thresholds to classify PTUs into distinct `Cheap`, `Normal`, and `Expensive` regimes.

## 📊 Business Value & ROI Projection
Based on rigorous 6-month out-of-sample holdout backtesting:
* **Theoretical Maximum:** €47.66/MWh savings vs. naive baseload procurement.
* **Realistic Industrial Yield:** Assuming a standard 10 GWh/yr client with a 15% flexibility factor, VoltCast delivers defensible savings of **~€71,490 annually**.

## 🚀 Pipeline Workflow
The pipeline is designed for zero-touch execution via a Linux `cron` schedule. 

1. `live_fetch` (10:30 CET): Pulls and standardizes actuals, forecasts, and macro data to a strict 15-minute UTC master index.
2. `live_feature` (10:35 CET): Executes the Physics Bridge engineering and data imputation.
3. `live_predict` (10:45 CET): Ingests cached model artifacts, generates the D+1 forecast, and applies B2B confidence stratification.
4. `generate_report` (10:50 CET): Compiles a glossy, human-readable PDF Executive Summary and JSON payload for API ingestion.

## 🔒 License & Copyright

**Copyright (c) 2026 VoltCast.**
**All Rights Reserved.**

This repository, including all associated documentation, models, and algorithms, contains the confidential and proprietary intellectual property of VoltCast. Unauthorized copying, distribution, modification, or use of these files, via any medium, is strictly prohibited.