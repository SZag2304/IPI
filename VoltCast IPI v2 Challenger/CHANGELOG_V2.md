# VoltCast IPI v2 — Changelog & Deployment Runbook
**Forensic ablation build · derived from the 17 May – 9 June 2026 live forensic validation**

v2 is a **challenger**: it runs in fully separate directories (`*_v2`) alongside the untouched v1 champion. Nothing in v1's cron, cache, or artefacts is modified. Every v2 change maps to a forensic finding and carries a pre-registered, falsifiable prediction — re-running `forensic_validation_v1.py --pred-dir voltcast_ipi_live_v2` after ~15 challenger days tests them.

## 1. Change registry (finding → change → falsifiable prediction)

| # | Forensic finding | v2 change | Where | Falsifiable prediction |
|---|---|---|---|---|
| 1 | **P4** corr(err, NO2) = −0.395; NorNed absent | Fetch NO2 DA price + NorNed scheduled flows; lagged Block N features (`no2_price_lag96/672`, `spread_nl_no2_lag96`, `norned_net_import_lag96`, saturation) | both fetches; `voltcast_v2_features.py` | err/NO2 correlation shrinks toward 0 on the next window |
| 2 | **P1** solar proxy R² = 0.52 (wind 0.70); solar cluster tops \|err\| correlations | Solar limb v2: clear-sky index (same-hour 14-day max, lag-96 history), diffuse share, south/north PV gradient; piecewise wind power curve (cut-in 3 / rated 12 / cut-out 25 m/s) alongside the cubic | `voltcast_v2_features.py` | solar-cluster Spearman \|err\| (~0.20) weakens; midday error band (Fig. 3) fades |
| 3 | **P3** 48 h gas lag costs €2.65/MWh on gas-volatile days | Train/serve parity: training now uses the **D-2 close** (`shift(192)` via `GAS_LAG_PTU_TRAINING`) — exactly the information live has at 10:10 CET. No paid feed needed. EUA aligned identically | `training_feature_v2.py`; parity note in `live_feature_v2.py` | MAE spread across \|ΔTTF\| terciles compresses below ~€1/MWh |
| 4 | **P5** negative-price recall 10% (52 hours, MBE +17.5) | Block X: renewable surplus index, surplus×weekend/midday, oversupply-under-clear-sky, same-hour 14-day negative-price climatology; negative-hour sample weights raised (XGB 4→6, LGB 2→3) | `voltcast_v2_features.py`; `training_model_v2.py` | negative-hour recall rises well above 10% without overall MAE degradation |
| 5 | **§3.5/6.3** static bias table failed under regime shift; manual +12 patched mid-pilot | Correction decomposed: **shape** = de-meaned Season×DoW×Hour table with hierarchical shrinkage toward Season×Hour parents (K=10, also closes thesis issue 8.6 sparse cells); **level** = adaptive rolling intercept learned daily from the challenger's own settled raw-vs-actual history (10-day window, ≥4 days, clip ±€20, bootstrap +12). Audited via `adaptive_bias_status.json` | `training_model_v2.py`; `live_model_v2.py`; `config_v2.py` | corrected-stream MBE stays near 0 across regime moves with **no manual intervention** |
| 6 | **P6** transmission REMIT never fetched (probe degenerate) | `IC_Outage_MW` fetched for NL–DE/BE/NO2/GB borders (forward-published → legitimately unlagged); features `ic_outage_mw`, roll24h, ×congestion | both fetches; `voltcast_v2_features.py` | P6 becomes testable on the next window |
| 7 | **§7.1** DE wind absent from Bridge (P2 unsupported but data already fetched) | `de_wind_proxy_cubed` + lag + collapse from 6 DE stations — explicitly flagged an ablation candidate | `voltcast_v2_features.py` | retained only if walk-forward CV gain > 0 |
| 8 | **§7.1** DE holidays (coupled-market demand) | `is_de_public_holiday` via `holidays.Germany` | `voltcast_v2_features.py` | minor; calendar-block hygiene |
| 9 | **Audit C1** dead Physics-Bridge-era branches | `add_2026_market_drivers` reduced to the surviving `imb_momentum_lag96`, exact training parity | `live_feature_v2.py` | n/a (hygiene) |
| 10 | **Audit I4** duplicate business-value block | superseded block deleted | `training_model_v2.py` | n/a (hygiene) |

Engineering invariants: the shared module `voltcast_v2_features.py` is imported by **both** feature scripts, so train/serve parity for all new features holds by construction; raw NO2/NorNed/IC columns are added to the DROP list and the leakage audit (lag-96 discipline unchanged); the regime-label scheme, thresholds, conformal method, CV design, holdout length, and the **training data window (Jan 2023 – Apr 2026)** are deliberately unchanged so the v1→v2 delta is attributable to the registered changes only. Extending the training window to include May 2026 is a separate, later step — do not bundle it.

## 2. Retrain & deployment runbook

```
# 0. Place all v2 files + config_v2.py + voltcast_v2_features.py + alerts.py
#    in the project root, next to the v1 scripts. Same .env applies.

# 1. Retrain (order matters; ~same runtime as v1):
python training_fetch_v2.py        # -> VoltCast_IPI_Master_v2.parquet + cache_v2/
python training_feature_v2.py      # -> VoltCast_IPI_Features_v2.parquet
python training_model_v2.py        # -> models, feature_names.txt, season_hour_bias.json,
                                   #    bias_meta.json, conformal, thresholds in cache_v2/

# 2. Verify before scheduling:
#    - training log: walk-forward CV + holdout MAE vs v1's 14.79 reference
#    - cache_v2/bias_meta.json: global level removed, n_cells, K
#    - leakage audit: zero V2 raw columns in the final matrix

# 3. Challenger cron (parallel to the untouched v1 lines):
10:12  python live_fetch_v2.py        # 2 min after v1 to spread API load
10:32  python live_feature_v2.py
10:47  python live_model_v2.py        # adaptive offset bootstraps at +12 for days 1-3
11:02  python live_report_v2.py
07:30  python live_validation_v2.py   # next morning; supports backfill: ... 2026-06-15

# 4. After 15 challenger trading days — the decision gate:
python forensic_validation_v1.py --start <challenger_day_1> --pred-dir voltcast_ipi_live_v2
#    Compare probe-by-probe vs the v1 forensic report. Promote v2 only if the
#    pre-registered predictions in section 1 are met; otherwise the probe table
#    tells you which block to revert. Rollback = delete the v2 cron lines.
```

## 3. Operational notes

The adaptive intercept needs settled history from the **v2 sandbox itself**; for its first 3 days it applies the bootstrap +12 (the empirically validated v1 constant) and logs `fallback_used: true` in `adaptive_bias_status.json` — expected, not an error. The GB legs of the transmission/flow fetches may 404 (post-Brexit publication gaps); they are non-critical and degrade gracefully. `live_validation_v2.py` now accepts an optional `YYYY-MM-DD` argument so missed validation days can be backfilled — use it to keep the challenger ledger complete (forensic limitation 6.2). For the PhD narrative, this entire file is your pre-registration document: it was written before the challenger produced a single forecast.
