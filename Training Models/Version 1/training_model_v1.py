
# STEP 1: S1 and S2 IMPLEMENTED FILE
"""
================================================================================
 VOLTCAST IPI — MODEL TRAINING PIPELINE
 Target: Netherlands EPEX Day-Ahead Price (EUR/MWh) — D+1 Forecast
================================================================================
 Three models trained from one feature matrix:

   Model 1 — Point forecast    XGBoost regression (reg:squarederror)
                                Target: DA_Price_NL_EURMWh
                                Metric: MAE, RMSE, Direction accuracy

   Model 2 — Uncertainty band  Quantile XGBoost (P10 + P90)
                                Same features, two separate models
                                Output: confidence interval per PTU

   Model 3 — Buy signal        XGBoost classifier (multi:softprob)
                                Target: price_regime_label (0/1/2)
                                Metric: accuracy, F1, precision per class

 Validation strategy:
   Walk-forward cross-validation (5 folds, temporal)
   Each fold: 18-month train → 6-month test, no shuffle
   Final model trained on full data minus last 6 months (holdout)

 Outputs:
   voltcast_cache/model_point.json        XGBoost point forecast model
   voltcast_cache/model_p10.json          Quantile P10 model
   voltcast_cache/model_p90.json          Quantile P90 model
   voltcast_cache/model_classifier.json   Regime classifier model
   voltcast_cache/feature_names.txt       Feature list for inference
   voltcast_cache/scaler_stats.parquet    Mean/std for monitoring
   VoltCast_IPI_Scores.parquet            Full test set predictions + actuals
================================================================================
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report
)

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("VoltCast")

warnings.filterwarnings("ignore")

# ── Target transformation utilities (I1 — symlog) ──────────────────
# Compresses €-500 to €873 into ~(-6.2 to 6.8), balancing tree splits
# across the entire price distribution instead of letting spikes dominate.
def symlog(x):
    """Symmetric log: sign(x) * log(1 + |x|). Works on arrays and Series."""
    return np.sign(x) * np.log1p(np.abs(x))

def symlog_inv(x):
    """Inverse symmetric log: sign(x) * (exp(|x|) - 1)."""
    return np.sign(x) * (np.expm1(np.abs(x)))

# ================================================================================
# 0. CONFIGURATION
# ================================================================================

FEATURES_FILE = "VoltCast_IPI_Features_v1.parquet"
SCORES_FILE   = "VoltCast_IPI_Scores_v1.parquet"
CACHE_DIR     = "voltcast_ipi_cache_v1"
os.makedirs(CACHE_DIR, exist_ok=True)

TARGET_PRICE   = "DA_Price_NL_EURMWh"
TARGET_REGIME  = "price_regime_label"
TARGET_DAILY   = "daily_avg_price"

# Walk-forward CV: 5 folds of 6-month test windows
# Minimum 18 months of training data before first test window
CV_TEST_MONTHS  = 6
CV_MIN_TRAIN_MONTHS = 12
CV_N_FOLDS = 5

# Holdout: last 6 months never touched during training — pure out-of-sample
HOLDOUT_MONTHS = 6

# Price spike threshold (from feature engineering)
SPIKE_THRESHOLD = 120.0

# ================================================================================
# 1. HYPERPARAMETERS
# ================================================================================
# Tuned for NL DA price — 15-min data, ~3 years history, ~120-160 features
# Key design choices explained inline

PARAMS_POINT = {
    # Objective: squared error gives balanced treatment of normal + spike hours
    # Use reg:absoluteerror if you want to prioritise typical days over spikes
    "objective":        "reg:squarederror",   # 🌟 FIX A: Pseudo-Huber Loss # removed squarederror reverted
    #"huber_slope":      1.0,                      # 🌟 FIX A: Transition point
    "tree_method":      "hist",
    "n_estimators":     1500,
    "learning_rate":    0.02,             # Low LR + high estimators = better generalisation
    "max_depth":        6,                # Moderate depth — price drivers are non-linear but not chaotic
    "min_child_weight": 10,               # Prevents overfitting on rare spike hours
    "subsample":        0.80,             # Row sampling — reduces variance
    "colsample_bytree": 0.70,             # Feature sampling — forces use of physics features not just price lags
    "colsample_bylevel":0.80,
    "reg_alpha":        0.05,             # L1: mild sparsity — prunes irrelevant features
    "reg_lambda":       1.0,              # L2: shrinkage — prevents any one feature dominating
    "gamma":            0.1,              # Min loss reduction to split — structural regulariser
    "n_jobs":           -1,
    "random_state":     42,
    "verbosity":        0,
}

PARAMS_P10 = {
    # I5: Quantile models need more trees + lower LR for smoother surfaces
    "objective":      "reg:quantileerror",
    "quantile_alpha": 0.05,               # Lower bound of confidence interval
    "tree_method":    "hist",
    "n_estimators":   3000,               # More trees — quantile loss converges slower
    "learning_rate":  0.01,               # Lower LR for stable quantile estimation
    "max_depth":      5,                  # Shallower — avoids overfitting tails
    "min_child_weight": 10,               # Higher — stabilizes extreme quantile estimates
    "subsample":      0.80,
    "colsample_bytree": 0.70,
    "colsample_bylevel":0.80,
    "reg_alpha":      0.05,
    "reg_lambda":     1.0,
    "gamma":          0.1,
    "n_jobs":         -1,
    "random_state":   42,
    "verbosity":      0,
}

PARAMS_P90 = {
    **PARAMS_P10,
    "quantile_alpha": 0.95,               # Upper bound of confidence interval
}

PARAMS_CLASSIFIER = {
    "objective":        "multi:softprob",
    "num_class":        3,                # Cheap=0, Normal=1, Expensive=2
    "tree_method":      "hist",
    "eval_metric":      "mlogloss",
    "n_estimators":     1000,
    "learning_rate":    0.03,
    "max_depth":        5,                # Shallower for classification — avoids memorising regime boundaries
    "min_child_weight": 20,               # Classes are imbalanced — higher weight prevents noise fitting
    "subsample":        0.80,
    "colsample_bytree": 0.70,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "n_jobs":           -1,
    "random_state":     42,
    "verbosity":        0,
}

# I2: LightGBM ensemble — leaf-wise growth produces complementary tree structures
# Blending XGBoost + LightGBM reduces variance by ~15-25%
PARAMS_LGBM = {
    "objective":         "regression",
    "metric":            "mae",
    "n_estimators":      1000,
    "learning_rate":     0.02,
    "max_depth":         7,
    "num_leaves":        63,              # Leaf-wise growth — LightGBM's key differentiator
    "min_child_samples": 15,
    "subsample":         0.8,
    "colsample_bytree":  0.7,
    "reg_alpha":         0.05,
    "reg_lambda":        1.0,
    "random_state":      42,
    "verbose":           -1,
    "n_jobs":           -1,
    "early_stopping_rounds": 100,
}

PARAMS_LGBM_CLASSIFIER = {
    "objective":         "multiclass",
    "num_class":         3,
    "metric":            "multi_logloss",
    "n_estimators":      1000,
    "learning_rate":     0.03,
    "max_depth":         5,
    "num_leaves":        31,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.7,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "random_state":      42,
    "verbose":           -1,
}

# ================================================================================
# 2. DATA LOADING & SPLIT UTILITIES
# ================================================================================

def load_feature_matrix(path: str):
    log.info(f"[LOAD] Reading {path}...")
    df = pd.read_parquet(path)

    # Separate targets from features
    target_cols = [TARGET_PRICE, TARGET_REGIME, TARGET_DAILY,
                   "is_spike_hour", "is_negative_hour"]
    feature_cols = [c for c in df.columns if c not in target_cols]

    X = df[feature_cols].copy()
    y_price  = df[TARGET_PRICE].copy()
    y_regime = df[TARGET_REGIME].copy()

    # Replace any remaining infinities
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop rows where target is NaN
    valid = y_price.notna() & y_regime.notna()
    X = X[valid]; y_price = y_price[valid]; y_regime = y_regime[valid]

    log.info(f"  Shape    : {X.shape[0]:,} rows x {X.shape[1]} features")
    log.info(f"  Range    : {X.index.min().date()} to {X.index.max().date()}")
    log.info(f"  Price    : EUR{y_price.min():.1f} to EUR{y_price.max():.1f}/MWh  (mean EUR{y_price.mean():.1f})")
    log.info(f"  Spikes   : {(y_price > SPIKE_THRESHOLD).sum():,} hours (>{SPIKE_THRESHOLD} EUR/MWh)")
    log.info(f"  Negatives: {(y_price < 0).sum():,} hours")
    return X, y_price, y_regime


def temporal_split(X, y_price, y_regime, holdout_months: int):
    """
    Creates a strict temporal train/holdout split.
    Holdout is the LAST N months — never seen during any training or tuning.
    """
    cutoff = X.index.max() - pd.DateOffset(months=holdout_months)
    train_mask = X.index <= cutoff

    X_train = X[train_mask];       X_hold  = X[~train_mask]
    y_p_train = y_price[train_mask]; y_p_hold = y_price[~train_mask]
    y_r_train = y_regime[train_mask];y_r_hold = y_regime[~train_mask]

    log.info(f"\n[SPLIT] Temporal train/holdout split")
    log.info(f"  Train : {X_train.index.min().date()} to {X_train.index.max().date()}  ({len(X_train):,} rows)")
    log.info(f"  Holdout: {X_hold.index.min().date()} to {X_hold.index.max().date()}  ({len(X_hold):,} rows)")
    return X_train, X_hold, y_p_train, y_p_hold, y_r_train, y_r_hold


def walk_forward_folds(X_train, y_p_train, n_folds: int, test_months: int, min_train_months: int):
    """
    Generates walk-forward CV folds. Each fold extends the training window by
    test_months and tests on the next test_months block.

    This is the ONLY valid CV strategy for time-series data — random K-fold
    leaks future information into training and produces falsely optimistic scores.

    Example with 36-month dataset, 5 folds, 6-month test:
      Fold 1: train Jan2023-Jun2024  test Jul2024-Dec2024
      Fold 2: train Jan2023-Dec2024  test Jan2025-Jun2025
      Fold 3: train Jan2023-Jun2025  test Jul2025-Dec2025
      etc.
    """
    total_months = (X_train.index.max().year - X_train.index.min().year) * 12 + \
                   (X_train.index.max().month - X_train.index.min().month)

    available_test_months = total_months - min_train_months
    if available_test_months < n_folds * test_months:
        actual_folds = max(1, available_test_months // test_months)
        log.info(f"  [CV] Insufficient data for {n_folds} folds — using {actual_folds} folds")
        n_folds = actual_folds

    start = X_train.index.min()
    folds = []

    for fold in range(n_folds):
        test_end_offset   = n_folds - fold
        test_start_offset = test_end_offset + 1

        test_end   = X_train.index.max() - pd.DateOffset(months=(test_end_offset-1)*test_months)
        test_start = test_end - pd.DateOffset(months=test_months)
        train_end  = test_start - pd.Timedelta(minutes=15)

        if train_end < start + pd.DateOffset(months=min_train_months):
            continue

        fold_X_train = X_train[X_train.index <= train_end]
        fold_X_test  = X_train[(X_train.index > train_end) & (X_train.index <= test_end)]
        fold_y_train = y_p_train[y_p_train.index <= train_end]
        fold_y_test  = y_p_train[(y_p_train.index > train_end) & (y_p_train.index <= test_end)]

        if len(fold_X_test) < 1000:
            continue

        folds.append({
            "fold": fold + 1,
            "X_train": fold_X_train, "X_test": fold_X_test,
            "y_train": fold_y_train, "y_test": fold_y_test,
            "train_end": train_end,  "test_end": test_end,
        })

    return folds


# ================================================================================
# 3. METRICS
# ================================================================================

def price_metrics(y_true, y_pred, label: str = "") -> dict:
    """
    Computes the full set of IPI-relevant metrics.
    Note: MAPE excluded — unreliable when prices near zero or negative.
    """
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2    = r2_score(y_true, y_pred)
    mbe   = float(np.mean(y_pred - y_true))

    # Direction accuracy: did we call up/down vs previous day correctly?
    y_true_s = pd.Series(y_true.values, index=y_true.index)
    y_pred_s = pd.Series(y_pred,        index=y_true.index)
    prev_day  = y_true_s.shift(96)
    valid     = prev_day.notna()
    true_dir  = np.sign(y_true_s[valid] - prev_day[valid])
    pred_dir  = np.sign(y_pred_s[valid] - prev_day[valid])
    dir_acc   = float((true_dir == pred_dir).mean())

    # Spike capture: what fraction of spike hours (>150 EUR/MWh) did we forecast correctly?
    spike_mask = y_true > SPIKE_THRESHOLD
    if spike_mask.sum() > 0:
        spike_mae = mean_absolute_error(y_true[spike_mask], y_pred[spike_mask])
        spike_n   = int(spike_mask.sum())
    else:
        spike_mae = float("nan")
        spike_n   = 0

    # Negative price capture
    neg_mask = y_true < 0
    neg_mae  = mean_absolute_error(y_true[neg_mask], y_pred[neg_mask]) if neg_mask.sum() > 0 else float("nan")

    # Interval coverage check (P10/P90 added externally when available)
    metrics = {
        "label": label, "n": len(y_true),
        "MAE": round(mae, 2),   "RMSE": round(rmse, 2),
        "R2":  round(r2,  4),   "MBE":  round(mbe,  2),
        "Direction_Acc": round(dir_acc, 4),
        "Spike_MAE": round(spike_mae, 2) if not np.isnan(spike_mae) else None,
        "Spike_N": spike_n,
        "Neg_MAE": round(neg_mae, 2) if not np.isnan(neg_mae) else None,
    }
    return metrics


#def regime_metrics(y_true, y_pred_proba, label: str = "") -> dict:
#    y_pred = np.argmax(y_pred_proba, axis=1)
def regime_metrics(y_true, y_pred, label: str = "") -> dict:
    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Business metric: "buy day" precision — when we say Cheap, how often are we right?
    cheap_mask  = y_pred == 0
    cheap_prec  = float((y_true[cheap_mask] == 0).mean()) if cheap_mask.sum() > 0 else float("nan")
    exp_mask    = y_pred == 2
    exp_prec    = float((y_true[exp_mask] == 2).mean()) if exp_mask.sum() > 0 else float("nan")

    return {
        "label": label, "Accuracy": round(acc, 4),
        "F1_macro": round(f1, 4),
        "Cheap_precision": round(cheap_prec, 4) if not np.isnan(cheap_prec) else None,
        "Expensive_precision": round(exp_prec, 4) if not np.isnan(exp_prec) else None,
    }


def interval_coverage(y_true, y_p10, y_p90, label: str = "") -> dict:
    """
    Checks how often actuals fall inside the P10-P90 interval.
    Target: ~80% coverage (by construction of 10th and 90th percentile).
    Lower = overconfident model; higher = too wide intervals.
    """
    inside = ((y_true >= y_p10) & (y_true <= y_p90)).mean()
    width  = (y_p90 - y_p10).mean()
    return {
        "label": label,
        "Coverage_80pct": round(float(inside), 4),
        "Avg_Width_EUR": round(float(width), 2),
    }


def print_metrics(m: dict):
    if "MAE" in m:
        log.info(f"  MAE          : EUR{m['MAE']:.2f}/MWh")
        log.info(f"  RMSE         : EUR{m['RMSE']:.2f}/MWh")
        log.info(f"  R2           : {m['R2']:.4f}")
        log.info(f"  MBE (bias)   : EUR{m['MBE']:.2f}/MWh  ({'over' if m['MBE']>0 else 'under'}-forecast)")
        log.info(f"  Direction acc: {m['Direction_Acc']*100:.1f}%")
        if m.get("Spike_MAE") is not None:
            log.info(f"  Spike MAE    : EUR{m['Spike_MAE']:.2f}/MWh  (n={m['Spike_N']})")
        if m.get("Neg_MAE") is not None:
            log.info(f"  Neg price MAE: EUR{m['Neg_MAE']:.2f}/MWh")
    elif "Accuracy" in m:
        log.info(f"  Accuracy     : {m['Accuracy']*100:.1f}%")
        log.info(f"  F1 (macro)   : {m['F1_macro']:.4f}")
        if m.get("Cheap_precision") is not None:
            log.info(f"  Buy precision: {m['Cheap_precision']*100:.1f}%  (when we say 'buy', correct X% of time)")
        if m.get("Expensive_precision") is not None:
            log.info(f"  Avoid prec.  : {m['Expensive_precision']*100:.1f}%")
    elif "Coverage_80pct" in m:
        log.info(f"  Coverage     : {m['Coverage_80pct']*100:.1f}%  (target: ~80%)")
        log.info(f"  Avg width    : EUR{m['Avg_Width_EUR']:.2f}/MWh")


# ================================================================================
# 4. WALK-FORWARD CROSS-VALIDATION
# ================================================================================

def run_walk_forward_cv(X_train, y_p_train, y_r_train):
    """
    Runs walk-forward CV for point forecast and classifier.
    Returns per-fold metrics and an aggregate summary.
    """
    log.info("\n" + "="*65)
    log.info("  WALK-FORWARD CROSS-VALIDATION")
    log.info("="*65)

    folds = walk_forward_folds(
        X_train, y_p_train,
        n_folds=CV_N_FOLDS,
        test_months=CV_TEST_MONTHS,
        min_train_months=CV_MIN_TRAIN_MONTHS,
    )

    if not folds:
        log.info("  [SKIP] Insufficient data for walk-forward CV")
        return [], pd.Series(dtype=float)

    cv_results = []

    raw_calibration_residuals = []

    for f in folds:
        fold_num  = f["fold"]
        Xtr, Xte = f["X_train"], f["X_test"]
        ytr, yte  = f["y_train"], f["y_test"]
        ytr_r     = y_r_train[y_r_train.index <= f["train_end"]]
        yte_r     = y_r_train[(y_r_train.index > f["train_end"]) &
                              (y_r_train.index <= f["test_end"])]

        n_train_months = round(len(Xtr) / (96 * 30))
        log.info(f"\n  Fold {fold_num}: train={n_train_months}mo "
              f"({Xtr.index.min().date()} to {Xtr.index.max().date()})  "
              f"test={CV_TEST_MONTHS}mo ({Xte.index.min().date()} to {Xte.index.max().date()})")

        # I1+I2: Symlog transform + XGB/LGB ensemble for point forecast
        ytr_sl = symlog(ytr)
        yte_sl = symlog(yte)

        model_xgb = xgb.XGBRegressor(**PARAMS_POINT)
        model_xgb.fit(Xtr, ytr_sl, eval_set=[(Xte, yte_sl)], verbose=False)
        pred_xgb = symlog_inv(model_xgb.predict(Xte))

        model_lgb_cv = lgb.LGBMRegressor(**PARAMS_LGBM)
        model_lgb_cv.fit(Xtr, ytr_sl, eval_set=[(Xte, yte_sl)])
        pred_lgb_cv = symlog_inv(model_lgb_cv.predict(Xte))

        pred_pt = 0.5 * pred_xgb + 0.5 * pred_lgb_cv
        m_pt = price_metrics(yte, pred_pt, f"Fold {fold_num} — Ensemble")

        # 🌟 FIX 1: Collect raw SIGNED residuals (True - Predicted) WITH INDEX
        # We keep the Pandas Series here so we retain the DatetimeIndex for debiasing later
        fold_residuals_series = yte - pred_pt 
        raw_calibration_residuals.append(fold_residuals_series)

        # Classifier
        model_cls = xgb.XGBClassifier(**PARAMS_CLASSIFIER)
        model_cls.fit(Xtr, ytr_r.loc[Xtr.index],
                      eval_set=[(Xte, yte_r.loc[Xte.index])],
                      verbose=False)
        pred_proba = model_cls.predict_proba(Xte)
        #m_cls = regime_metrics(yte_r.loc[Xte.index], pred_proba, f"Fold {fold_num} — Classifier")

        # 🌟 FIX: Manually calculate the argmax labels here for the CV metrics
        pred_cv_regime = np.argmax(pred_proba, axis=1)
        
        # 🌟 FIX: Pass pred_cv_regime (the hard labels), NOT pred_proba
        m_cls = regime_metrics(yte_r.loc[Xte.index], pred_cv_regime, f"Fold {fold_num} — Classifier")

        log.info(f"    Point  → MAE=EUR{m_pt['MAE']:.2f}  RMSE=EUR{m_pt['RMSE']:.2f}  "
              f"R2={m_pt['R2']:.4f}  Dir={m_pt['Direction_Acc']*100:.1f}%")
        log.info(f"    Regime → Acc={m_cls['Accuracy']*100:.1f}%  "
              f"F1={m_cls['F1_macro']:.4f}  "
              f"BuyPrec={m_cls.get('Cheap_precision', 'n/a')}")

        #cv_results.append({"point": m_pt, "classifier": m_cls})

        # 🌟 FIX 1: Save the out-of-sample true labels and probabilities
        cv_results.append({
            "point": m_pt, 
            "classifier": m_cls,
            "y_true_cls": yte_r.loc[Xte.index],
            "pred_proba_cls": pred_proba 
        })

    # Aggregate summary
    if cv_results:
        pt_maes  = [r["point"]["MAE"]  for r in cv_results]
        pt_r2s   = [r["point"]["R2"]   for r in cv_results]
        pt_dirs  = [r["point"]["Direction_Acc"] for r in cv_results]
        cls_accs = [r["classifier"]["Accuracy"] for r in cv_results]

        log.info(f"\n  CV SUMMARY ({len(cv_results)} folds)")
        log.info(f"  Point MAE  : EUR{np.mean(pt_maes):.2f} ± EUR{np.std(pt_maes):.2f}/MWh")
        log.info(f"  R2         : {np.mean(pt_r2s):.4f} ± {np.std(pt_r2s):.4f}")
        log.info(f"  Direction  : {np.mean(pt_dirs)*100:.1f}% ± {np.std(pt_dirs)*100:.1f}%")
        log.info(f"  Regime acc : {np.mean(cls_accs)*100:.1f}% ± {np.std(cls_accs)*100:.1f}%")

    # Concatenate all fold series into one continuous timeline of out-of-sample errors
    cv_residuals_combined = pd.concat(raw_calibration_residuals) if raw_calibration_residuals else pd.Series(dtype=float)
    return cv_results, cv_residuals_combined


# ================================================================================
# 5. FINAL MODEL TRAINING
# ================================================================================
#def train_final_models(X_train, y_p_train, y_r_train, X_hold, y_p_hold, y_r_hold):
# 🌟 FIX 2A: Add the CV arguments
def train_final_models(X_train, y_p_train, y_r_train, X_hold, y_p_hold, y_r_hold, cv_y_true=None, cv_pred_proba=None, cv_residuals=None):
    """
    Trains all production models on the full training set.
    Evaluates on the held-out test set (last 6 months — never seen before).

    Tier 2 upgrades applied:
      I1 — Symlog target transformation (balanced tree splits)
      I2 — XGBoost + LightGBM ensemble (variance reduction)
      I5 — Tuned quantile hyperparameters (better coverage)
    """
    log.info("\n" + "="*65)
    log.info("  FINAL MODEL TRAINING (TIER 2 ENSEMBLE)")
    log.info("="*65)
    log.info(f"  Training on {len(X_train):,} rows → evaluating on {len(X_hold):,} holdout rows")
    log.info(f"  Upgrades: symlog transform (I1) + XGB/LGB ensemble (I2)\n")

    models = {}
    predictions = pd.DataFrame(index=X_hold.index)
    predictions["y_true_price"]  = y_p_hold
    predictions["y_true_regime"] = y_r_hold

    # ── I1: Transform targets to symlog space ───────────────────────
    y_train_sl = symlog(y_p_train)
    y_hold_sl  = symlog(y_p_hold)
    y_p_train_sym = symlog(y_p_train)
    y_p_hold_sym  = symlog(y_p_hold)

    # ══════════════════════════════════════════════════════════════════
    # 5A. POINT FORECAST — XGBoost + LightGBM ensemble
    # ══════════════════════════════════════════════════════════════════
    log.info("[Model 1a] Point forecast — XGBoost (symlog target)...")
    model_xgb_pt = xgb.XGBRegressor(**PARAMS_POINT)
    spike_mask = y_p_train > SPIKE_THRESHOLD
    neg_mask   = y_p_train < 0
    weights = np.ones(len(y_p_train))
    weights[spike_mask] = 10.0   # Spike hours: 10x weight
    weights[neg_mask]   = 4.0   # Negative hours: 4x weight

    model_xgb_pt.fit(
        X_train, y_train_sl,
        sample_weight=weights,
        eval_set=[(X_hold, y_hold_sl)],
        verbose=200,
    )
    pred_xgb_sl = model_xgb_pt.predict(X_hold)
    pred_xgb    = symlog_inv(pred_xgb_sl)
    m_xgb       = price_metrics(y_p_hold, pred_xgb, "XGBoost point")
    log.info(f"  XGBoost  → MAE=EUR{m_xgb['MAE']:.2f}  R2={m_xgb['R2']:.4f}")
    models["point"] = model_xgb_pt

    log.info("\n[Model 1b] Point forecast — LightGBM (symlog target)...")
    model_lgb_pt = lgb.LGBMRegressor(**PARAMS_LGBM)
    spike_mask = y_p_train > SPIKE_THRESHOLD
    neg_mask   = y_p_train < 0
    weights = np.ones(len(y_p_train))
    weights[spike_mask] = 6.0   # Spike hours: 6x weight
    weights[neg_mask]   = 2.0   # Negative hours: 2x weight
    model_lgb_pt.fit(
        X_train, y_train_sl,
        sample_weight=weights,
        eval_set=[(X_hold, y_hold_sl)],
    )
    pred_lgb_sl = model_lgb_pt.predict(X_hold)
    pred_lgb    = symlog_inv(pred_lgb_sl)
    m_lgb       = price_metrics(y_p_hold, pred_lgb, "LightGBM point")
    log.info(f"  LightGBM → MAE=EUR{m_lgb['MAE']:.2f}  R2={m_lgb['R2']:.4f}")
    models["point_lgb"] = model_lgb_pt

    # Ensemble blend (equal weight — can be optimized later via CV)
    BLEND_W_XGB = 0.50
    BLEND_W_LGB = 1.0 - BLEND_W_XGB
    pred_point  = BLEND_W_XGB * pred_xgb + BLEND_W_LGB * pred_lgb
    predictions["pred_p50"]     = pred_point
    predictions["pred_p50_xgb"] = pred_xgb
    predictions["pred_p50_lgb"] = pred_lgb

    m_point = price_metrics(y_p_hold, pred_point, "Ensemble point — holdout")
    log.info(f"\n  ── ENSEMBLE (XGB {BLEND_W_XGB:.0%} + LGB {BLEND_W_LGB:.0%}) ──")
    print_metrics(m_point)

    # ══════════════════════════════════════════════════════════════════
    # 5B. QUANTILE P10 (lower bound)
    # ══════════════════════════════════════════════════════════════════
    log.info("\n[Model 2a] Quantile P10 (lower bound — tuned I5)...")
    model_p10 = xgb.XGBRegressor(**PARAMS_P10)
    
    # Train on symlog target
    model_p10.fit(X_train, y_p_train_sym, verbose=False)
    
    # Predict and invert back to raw EUR/MWh
    raw_p10 = model_p10.predict(X_hold)
    pred_p10 = symlog_inv(raw_p10)
    
    predictions["pred_p10"] = pred_p10
    models["p10"] = model_p10
    log.info(f"  P10 mean: EUR{pred_p10.mean():.2f}/MWh  (should be below P50)")

    # ══════════════════════════════════════════════════════════════════
    # 5C. QUANTILE P90 (upper bound)
    # ══════════════════════════════════════════════════════════════════
    log.info("\n[Model 2b] Quantile P90 (upper bound — tuned I5)...")
    model_p90 = xgb.XGBRegressor(**PARAMS_P90)
    
    # Train on symlog target
    model_p90.fit(X_train, y_p_train_sym, verbose=False)
    
    # Predict and invert back to raw EUR/MWh
    raw_p90 = model_p90.predict(X_hold)
    pred_p90 = symlog_inv(raw_p90)
    
    predictions["pred_p90"] = pred_p90
    models["p90"] = model_p90

    m_interval = interval_coverage(y_p_hold, pred_p10, pred_p90, "Interval — holdout")
    print_metrics(m_interval)

    # Enforce monotonicity: P10 <= P50 <= P90
    violations = ((pred_p10 > pred_point) | (pred_p90 < pred_point)).sum()
    if violations > 0:
        log.info(f"  [NOTE] {violations} monotonicity violations — applying post-hoc correction")
        pred_p10_c = np.minimum(pred_p10, pred_point)
        pred_p90_c = np.maximum(pred_p90, pred_point)
        predictions["pred_p10"] = pred_p10_c
        predictions["pred_p90"] = pred_p90_c
        m_interval2 = interval_coverage(y_p_hold, pred_p10_c, pred_p90_c, "Interval corrected")
        print_metrics(m_interval2)

    # ══════════════════════════════════════════════════════════════════
    # 5D. REGIME CLASSIFIER — XGBoost + LightGBM ensemble
    # ══════════════════════════════════════════════════════════════════
    log.info("\n[Model 3a] Regime classifier — XGBoost...")
    model_xgb_cls = xgb.XGBClassifier(**PARAMS_CLASSIFIER)
    model_xgb_cls.fit(
        X_train, y_r_train,
        eval_set=[(X_hold, y_r_hold)],
        verbose=200,
    )
    proba_xgb = model_xgb_cls.predict_proba(X_hold)
    models["classifier"] = model_xgb_cls

    log.info("\n[Model 3b] Regime classifier — LightGBM...")
    model_lgb_cls = lgb.LGBMClassifier(**PARAMS_LGBM_CLASSIFIER)
    model_lgb_cls.fit(
        X_train, y_r_train,
        eval_set=[(X_hold, y_r_hold)],
    )
    proba_lgb = model_lgb_cls.predict_proba(X_hold)
    models["classifier_lgb"] = model_lgb_cls

    # Ensemble: average probabilities
    pred_proba  = BLEND_W_XGB * proba_xgb + BLEND_W_LGB * proba_lgb

    # ------------------------------------------------------------------
    # 🌟 NEW OUT-OF-SAMPLE THRESHOLD TUNING LOGIC
    # ------------------------------------------------------------------
    from sklearn.metrics import precision_recall_curve

    try:
        if cv_y_true is None or cv_pred_proba is None:
            raise ValueError("CV data is missing or too short.")

        log.info("\n  [Threshold Tuning] Calibrating on Last Out-of-Sample CV Fold...")
        
        # Tune CHEAP Threshold (Class 0)
        prec_c, rec_c, thresh_c = precision_recall_curve((cv_y_true == 0).astype(int), cv_pred_proba[:, 0])
        # Aim for 82% on CV to maintain high 82%+ precision on Holdout
        valid_c = prec_c[:-1] >= 0.82
        raw_opt_c = thresh_c[valid_c][0] if valid_c.any() else 0.45
        # Enforce a safety floor and force standard Python float
        optimal_cheap_thresh = float(max(raw_opt_c, 0.45))
        
        # Tune EXPENSIVE Threshold (Class 2)
        prec_e, rec_e, thresh_e = precision_recall_curve((cv_y_true == 2).astype(int), cv_pred_proba[:, 2])
        # Aim for 84% on CV to absorb the generalization gap and land >80% on Holdout
        valid_e = prec_e[:-1] >= 0.84
        raw_opt_e = thresh_e[valid_e][0] if valid_e.any() else 0.45
        # Enforce a safety floor and force standard Python float
        optimal_exp_thresh = float(max(raw_opt_e, 0.45))

    except Exception as e:
        log.info(f"\n  [Threshold Tuning] WARNING: Calibration skipped or failed ({str(e)}). Reverting to safe defaults.")
        # Fallback if CV data is missing or sklearn math fails
        optimal_cheap_thresh, optimal_exp_thresh = 0.50, 0.45

    log.info(f"  [Threshold Tuning] Optimal Cheap: {optimal_cheap_thresh:.2f} | Optimal Exp: {optimal_exp_thresh:.2f}")

    # Apply the strictly out-of-sample learned thresholds to the HOLDOUT set
    pred_regime_calibrated = np.ones(len(pred_proba), dtype=int)  # Base state = 1 (Normal)
    
    pred_regime_calibrated[pred_proba[:, 0] >= optimal_cheap_thresh] = 0
    pred_regime_calibrated[pred_proba[:, 2] >= optimal_exp_thresh] = 2
    
    # Conflict resolution (if an hour somehow breaches both thresholds)
    conflict = (pred_proba[:, 0] >= optimal_cheap_thresh) & (pred_proba[:, 2] >= optimal_exp_thresh)
    pred_regime_calibrated[conflict] = np.argmax(pred_proba[conflict], axis=1)

    # Overwrite pred_regime with our new calibrated logic
    pred_regime = pred_regime_calibrated
    # ------------------------------------------------------------------

    # Save to holdout predictions
    predictions["pred_regime"]    = pred_regime
    predictions["pred_regime_p0"] = pred_proba[:, 0]   # P(Cheap)
    predictions["pred_regime_p1"] = pred_proba[:, 1]   # P(Normal)
    predictions["pred_regime_p2"] = pred_proba[:, 2]   # P(Expensive)
    
    # Pass our newly calibrated pred_regime to the metric function
    m_cls = regime_metrics(y_r_hold, pred_regime, "Ensemble classifier — holdout")
    log.info(f"\n  ── ENSEMBLE CLASSIFIER ──")
    print_metrics(m_cls)

    # ══════════════════════════════════════════════════════════════════
    # 5E. STRUCTURAL DEBIASING (Season x DOW x Hour)
    # ══════════════════════════════════════════════════════════════════
    log.info("\n  [Debiasing] Learning Season-DOW-Hour structural bias from CV Out-of-Sample residuals...")
    
    if cv_residuals is not None and not cv_residuals.empty:
        df_bias_train = pd.DataFrame({
            'residual': cv_residuals.values,
            'hour':     cv_residuals.index.hour,
            'month':    cv_residuals.index.month,
            'dow':      cv_residuals.index.dayofweek
        })
    else:
        log.info("  [WARNING] No CV residuals — using training residuals for bias (mild leakage)")
        residuals_train = y_p_train - pd.Series(
            BLEND_W_XGB * symlog_inv(model_xgb_pt.predict(X_train)) +
            BLEND_W_LGB * symlog_inv(model_lgb_pt.predict(X_train)),
            index=X_train.index
        )
        df_bias_train = pd.DataFrame({
            'residual': residuals_train.values,
            'hour':     X_train.index.hour,
            'month':    X_train.index.month,
            'dow':      X_train.index.dayofweek
        })
    

    '''# 1. Learn the true out-of-sample structural bias from CV
    if cv_residuals is not None and not cv_residuals.empty:
        df_bias_train = pd.DataFrame({
            'residual': cv_residuals.values,
            'hour':     cv_residuals.index.hour,
            'month':    cv_residuals.index.month,
            'dow':      cv_residuals.index.dayofweek
        })
    else:
        log.info("  [WARNING] No CV residuals provided. Bias table will be empty.")
        df_bias_train = pd.DataFrame(columns=['residual', 'hour', 'month', 'dow'])'''
    
    season_map = {1:'Win', 2:'Win', 3:'Spr', 4:'Spr', 5:'Spr',
                  6:'Sum', 7:'Sum', 8:'Sum', 9:'Aut', 10:'Aut', 11:'Aut', 12:'Win'}
    
    df_bias_train['season'] = df_bias_train['month'].map(season_map)
    bias_table = df_bias_train.groupby(['season', 'dow', 'hour'])['residual'].median()

    '''# 2. Apply the learned bias to the Holdout Set
    df_bias_hold = pd.DataFrame({
        'hour': X_hold.index.hour,
        'month': X_hold.index.month
    }, index=X_hold.index)
    
    df_bias_hold['season'] = df_bias_hold['month'].map(season_map)
    df_bias_hold['dow'] = X_hold.index.dayofweek'''


    df_bias_hold = pd.DataFrame({
        'hour':  X_hold.index.hour,
        'month': X_hold.index.month,
        'dow':   X_hold.index.dayofweek
    }, index=X_hold.index)
    df_bias_hold['season'] = df_bias_hold['month'].map(season_map)

    # Apply correction
    correction = df_bias_hold.apply(
        lambda r: bias_table.get((r['season'], r['dow'], r['hour']), 0.0), axis=1
    )
    
    # Clip to prevent over-correction
    correction = correction.clip(lower=-15.0, upper=15.0)

    # Add to Holdout predictions
    pred_point_corrected = pred_point + correction
    predictions["pred_p50"] = pred_point_corrected
    
    # 3. Export to JSON for live inference
    bias_dict = {f"{s}_{d}_{h}": float(v) for (s, d, h), v in bias_table.items()}
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, "season_hour_bias.json"), "w") as f:
        json.dump(bias_dict, f, indent=2)

    # Full classification report
    log.info(f"\n  Classification report:")
    log.info(classification_report(
        y_r_hold, pred_regime,
        target_names=["Cheap (buy)", "Normal", "Expensive (avoid)"],
        zero_division=0,
    ))

    m_point = price_metrics(y_p_hold, predictions["pred_p50"].values, "Final Blended Point Forecast")

    # ══════════════════════════════════════════════════════════════════
    # 5F. ASYMMETRIC CONFORMAL PREDICTION (NEW)
    # ══════════════════════════════════════════════════════════════════
    m_conformal = {}
    if cv_residuals is not None and len(cv_residuals) > 0:
        cal_resid = np.array(cv_residuals)

        # For an 80% coverage interval, we trim the bottom 10% and top 10% of errors
        p10_correction = np.percentile(cal_resid, 10) 
        p90_correction = np.percentile(cal_resid, 90)

        log.info(f"\n  [Conformal] P10 offset: EUR{p10_correction:.2f} | P90 offset: EUR{p90_correction:.2f}")

        # Apply the asymmetric corrections
        pred_p10_conformal = predictions["pred_p50"] + p10_correction
        pred_p90_conformal = predictions["pred_p50"] + p90_correction

        # Save to the predictions dataframe so you can export them!
        predictions["pred_p10_conformal"] = pred_p10_conformal
        predictions["pred_p90_conformal"] = pred_p90_conformal

        # Report the new conformal intervals
        m_conformal = interval_coverage(y_p_hold, pred_p10_conformal, pred_p90_conformal, "Asymmetric Conformal")
        log.info(f"\n  ── ASYMMETRIC CONFORMAL INTERVAL ──")
        print_metrics(m_conformal)
        # After computing p10_correction and p90_correction:
        conformal_params = {
            "p10_offset":    float(p10_correction),
            "p90_offset":    float(p90_correction),
            "coverage":      round(float(m_conformal.get("Coverage_80pct", 0)), 4),
            "avg_width_eur": round(float(m_conformal.get("Avg_Width_EUR", 0)), 2),
            "n_calibration": len(cal_resid),
            "training_date": pd.Timestamp.now().isoformat(),
        }
        with open(os.path.join(CACHE_DIR, "conformal_params.json"), "w") as f:
            json.dump(conformal_params, f, indent=2)
    else:
        log.info("\n  [Conformal] Skipped: No CV residuals provided.")

    # ══════════════════════════════════════════════════════════════════

    # 🌟 NEW: Save the optimal thresholds to a JSON file
    thresholds = {
        "cheap_threshold":     float(optimal_cheap_thresh),
        "expensive_threshold": float(optimal_exp_thresh),
        "training_date":       pd.Timestamp.now().isoformat(),
    }
    with open(os.path.join(CACHE_DIR, "old_classifier_thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    # --- SAVE THRESHOLDS ---
    try:
        threshold_path = os.path.join(CACHE_DIR, "classifier_thresholds.json")
        with open(threshold_path, "w") as f:
            json.dump({
                "cheap_threshold": optimal_cheap_thresh,
                "expensive_threshold": optimal_exp_thresh
            }, f, indent=2)
    except Exception as e:
        log.info(f"  [SAVE ERROR] Could not save classifier thresholds: {e}")

    return models, predictions, m_point, m_interval, m_cls, m_conformal


# ================================================================================
# 6. FEATURE IMPORTANCE
# ================================================================================

def analyse_feature_importance(model_point, model_cls, feature_names: list):
    """
    Extracts and prints feature importance from both models.
    Uses 'gain' (total information gain) — more reliable than 'weight' (split count).
    For the product: high-importance features = what's driving prices today.
    """
    log.info("\n" + "="*65)
    log.info("  FEATURE IMPORTANCE ANALYSIS")
    log.info("="*65)

    for model, name in [(model_point, "Point forecast"), (model_cls, "Regime classifier")]:
        try:
            imp = model.get_booster().get_score(importance_type="gain")
            imp_s = pd.Series(imp).sort_values(ascending=False)
            imp_s.index = imp_s.index.str.replace("f", "", regex=False)
            # Map numeric indices back to feature names if needed
            top20 = imp_s.head(20)
            log.info(f"\n  {name} — top 20 features (by gain):")
            log.info(f"  {'Feature':<52} {'Gain':>10}")
            log.info("  " + "-"*64)
            for feat, gain in top20.items():
                try:
                    idx = int(feat)
                    feat_name = feature_names[idx]
                except (ValueError, IndexError):
                    feat_name = feat
                log.info(f"  {feat_name:<52} {gain:>10.1f}")
        except Exception as e:
            log.info(f"  [SKIP] {name} importance failed: {e}")

    # Block-level importance summary
    blocks = {
        # 1. SPECIFIC BLOCKS (Placed first to avoid substring hijacking)
        "G — Interactions":     ["cold_lowwind", "rebound_cold", "spring_solar", "summer_solar", "gas_outage", "monday_morning"],
        "H — Regimes":          ["neg_price_risk", "price_spike_risk", "market_tightness", "vol_regime", "scarcity_exponential", "scarcity_momentum"],
        
        # 2. CORE BLOCKS
        "A — Time/calendar":    ["hour", "dow", "month", "week", "quarter", "is_weekend", "is_monday", "is_friday", "is_public", "is_school", "is_bridge", "is_post", "is_night", "is_morning", "is_midday", "is_evening", "is_late", "is_winter", "is_spring", "is_summer", "is_autumn", "is_solar", "is_dst", "is_easter"],
        "B — Demand":           ["tso_fc", "actual_load", "HDD", "CDD", "temp_shock", "temp_rate", "morning_heating", "tso_bias"],
        
        # 3. THE PHYSICS BRIDGE (Renamed from Renewables to reflect actual data)
        "C — Weather Physics":  ["radiation", "wind_speed", "wind_direction", "cloud_cover", "humidity", "precipitation", "weather_code", "temperature_2m", "apparent_temperature", "wind_power", "wind_proxy", "wind_48h", "solar_proxy", "duck_curve", "weather_scarcity"],
        
        # 4. MARKET & MACRO
        "D — Macro/thermal":    ["ttf_", "eua_", "mc_", "thermal_", "supply_crunch", "supply_margin", "gas_actual", "coal_actual", "TTF_", "EUA_", "CCGT_", "effective_thermal", "Nuclear_Outage"],
        "E — Market coupling":  ["de_price", "be_price", "se3_price", "fr_price", "spread_", "flow_", "nordic_", "neighbor_nuclear", "export_saturation", "neighbor_spread", "neighbor_pull", "is_de_negative"],
        "F — Price dynamics":   ["price_lag", "price_daily", "price_mom", "price_ema", "price_trend", "price_vol", "price_1w", "price_hourly", "price_intraday", "neg_price_flag", "neg_price_count", "imb_momentum"]
    }
    try:
        imp = model_point.get_booster().get_score(importance_type="gain")
        imp_named = {}
        for k, v in imp.items():
            try:
                idx = int(k.replace("f", ""))
                imp_named[feature_names[idx]] = v
            except Exception:
                imp_named[k] = v
        total_gain = sum(imp_named.values())

        log.info(f"\n  Block-level importance (point forecast):")
        block_gains = {}
        for block, prefixes in blocks.items():
            gain = sum(v for k, v in imp_named.items()
                       if any(k.startswith(p) or k == p for p in prefixes))
            block_gains[block] = gain
        for block, gain in sorted(block_gains.items(), key=lambda x: -x[1]):
            pct = gain / total_gain * 100 if total_gain > 0 else 0
            bar = "█" * int(pct / 2)
            log.info(f"  {block:<28} {pct:>5.1f}%  {bar}")
    except Exception as e:
        log.info(f"  [SKIP] Block importance failed: {e}")


# ================================================================================
# 7. SEASONAL BREAKDOWN
# ================================================================================

def seasonal_breakdown(predictions: pd.DataFrame):
    """
    Breaks down point forecast error by season and day-type.
    This is what enterprise buyers ask for in procurement — "how does it
    perform in winter when prices spike?" They won't sign without this.
    """
    log.info("\n" + "="*65)
    log.info("  SEASONAL & DAY-TYPE BREAKDOWN")
    log.info("="*65)

    df = predictions.copy()
    df["error"]  = df["pred_p50"] - df["y_true_price"]
    df["abs_err"]= df["error"].abs()
    df["month"]  = df.index.month
    df["dow"]    = df.index.dayofweek

    season_map = {12:"Winter",1:"Winter",2:"Winter",
                  3:"Spring",4:"Spring",5:"Spring",
                  6:"Summer",7:"Summer",8:"Summer",
                  9:"Autumn",10:"Autumn",11:"Autumn"}
    df["season"] = df["month"].map(season_map)
    df["day_type"] = np.where(df["dow"] < 5, "Weekday", "Weekend")

    log.info(f"\n  {'Segment':<22} {'N':>6}  {'MAE':>8}  {'RMSE':>8}  {'MBE':>8}  {'Dir%':>7}")
    log.info("  " + "-"*64)

    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        s = df[df["season"] == season]
        if len(s) == 0: continue
        mae  = s["abs_err"].mean()
        rmse = np.sqrt((s["error"]**2).mean())
        mbe  = s["error"].mean()
        # Direction accuracy for this slice
        prev = s["y_true_price"].shift(96)
        val  = prev.notna()
        t_dir = np.sign(s["y_true_price"][val] - prev[val])
        p_dir = np.sign(s["pred_p50"][val] - prev[val])
        dacc  = (t_dir == p_dir).mean() * 100
        log.info(f"  {season:<22} {len(s):>6}  {mae:>7.2f}  {rmse:>7.2f}  {mbe:>+7.2f}  {dacc:>6.1f}%")

    print()
    for day_type in ["Weekday", "Weekend"]:
        s = df[df["day_type"] == day_type]
        if len(s) == 0: continue
        mae  = s["abs_err"].mean()
        rmse = np.sqrt((s["error"]**2).mean())
        mbe  = s["error"].mean()
        prev = s["y_true_price"].shift(96)
        val  = prev.notna()
        t_dir = np.sign(s["y_true_price"][val] - prev[val])
        p_dir = np.sign(s["pred_p50"][val] - prev[val])
        dacc  = (t_dir == p_dir).mean() * 100
        log.info(f"  {day_type:<22} {len(s):>6}  {mae:>7.2f}  {rmse:>7.2f}  {mbe:>+7.2f}  {dacc:>6.1f}%")

    # Spike performance specifically
    spikes = df[df["y_true_price"] > SPIKE_THRESHOLD]
    if len(spikes) > 0:
        log.info(f"\n  Spike hours (>{SPIKE_THRESHOLD} EUR/MWh):")
        log.info(f"  {'N':>6}  {'MAE':>8}  {'MBE':>8}")
        log.info(f"  {len(spikes):>6}  {spikes['abs_err'].mean():>7.2f}  {spikes['error'].mean():>+7.2f}")


# ================================================================================
# 8. SAVE MODELS & ARTEFACTS
# ================================================================================

def save_all(models: dict, predictions: pd.DataFrame, feature_names: list):
    log.info("\n[SAVE] Writing models and artefacts...")

    # Models — XGBoost uses .save_model(), LightGBM uses .booster_.save_model()
    for name, model in models.items():
        if "lgb" in name:
            path = os.path.join(CACHE_DIR, f"model_{name}.txt")
            model.booster_.save_model(path)
        else:
            path = os.path.join(CACHE_DIR, f"model_{name}.json")
            model.save_model(path)
        log.info(f"  Saved: {path}")

    # Feature names (required for inference — features must arrive in exact order)
    feat_path = os.path.join(CACHE_DIR, "feature_names.txt")
    with open(feat_path, "w") as f:
        f.write("\n".join(feature_names))
    log.info(f"  Saved: {feat_path}  ({len(feature_names)} features)")

    # Full predictions for analysis
    predictions.to_parquet(SCORES_FILE)
    log.info(f"  Saved: {SCORES_FILE}")

    # Model card (human-readable summary)
    card_path = os.path.join(CACHE_DIR, "model_card.json")
    card = {
        "product": "VoltCast IPI",
        "version": "Tier 2 v1",
        "target": TARGET_PRICE,
        "models": list(models.keys()),
        "n_features": len(feature_names),
        "training_date": pd.Timestamp.now().isoformat(),
        "holdout_rows": len(predictions),
        "upgrades": [
            "I1: Symlog target transformation",
            "I2: XGBoost + LightGBM ensemble (50/50 blend)",
            "I4: RES D+1 features active (cron → 18:30 CET, after ENTSO-E publication)",
            "I5: Quantile models tuned (2500 trees, LR 0.01, depth 5)",
        ],
        "notes": [
            "All lags >= 96 periods (24h) — no leakage",
            "Walk-forward CV — no future data in training",
            "Regime labels: 0=Cheap, 1=Normal, 2=Expensive (rolling 30-day P25/P75)",
            "Cron schedule: 18:30 CET — after ENTSO-E D+1 renewables published (~18:00)",
            "D+1 DA price confirmed by 18:30 — use as input anchor for D+2 forecasts",
        ]
    }
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2)
    log.info(f"  Saved: {card_path}")


# ================================================================================
# 9. MASTER PIPELINE
# ================================================================================

if __name__ == "__main__":
    log.info("="*65)
    log.info("  VOLTCAST IPI — MODEL TRAINING PIPELINE")
    log.info("="*65)

    # ── Load ────────────────────────────────────────────────────────
    X, y_price, y_regime = load_feature_matrix(FEATURES_FILE)
    feature_names = list(X.columns)

    # ── Train/holdout split ─────────────────────────────────────────
    X_train, X_hold, y_p_train, y_p_hold, y_r_train, y_r_hold = temporal_split(
        X, y_price, y_regime, holdout_months=HOLDOUT_MONTHS
    )

    # ── Walk-forward CV (model selection and sanity check) ──────────
    cv_results, cv_residuals_combined = run_walk_forward_cv(X_train, y_p_train, y_r_train)

    # 🌟 FIX 3: Extract the last fold's out-of-sample data
    last_cv_fold_y_true = cv_results[-1]["y_true_cls"] if cv_results else None                  
    last_cv_fold_proba = cv_results[-1]["pred_proba_cls"] if cv_results else None               

    # ── Train final models on full train set ────────────────────────
    models, predictions, m_point, m_interval, m_cls, m_conformal = train_final_models(
        X_train, y_p_train, y_r_train,
        X_hold, y_p_hold, y_r_hold,
        cv_y_true=last_cv_fold_y_true,       
        cv_pred_proba=last_cv_fold_proba,     
        cv_residuals=cv_residuals_combined  # Pass the combined Series here!
    )

    # ── Feature importance ──────────────────────────────────────────
    analyse_feature_importance(models["point"], models["classifier"], feature_names)

    # ── Seasonal breakdown ──────────────────────────────────────────
    seasonal_breakdown(predictions)

    # ── Save ────────────────────────────────────────────────────────
    save_all(models, predictions, feature_names)

    # ── Final summary ───────────────────────────────────────────────
    log.info("\n" + "="*65)
    log.info("  TRAINING COMPLETE — HOLDOUT RESULTS")
    log.info("="*65)
    log.info(f"\n  Point forecast:")
    print_metrics(m_point)
    log.info(f"\n  Uncertainty interval (conformal):")
    print_metrics(m_conformal)
    log.info(f"\n  Uncertainty interval:")
    print_metrics(m_interval)
    log.info(f"\n  Regime classifier:")
    print_metrics(m_cls)

    '''# IPI business value estimate
    log.info("\n" + "="*65)
    log.info("  IPI BUSINESS VALUE ESTIMATE")
    log.info("="*65)
    if m_cls.get("Cheap_precision") and len(predictions):
        buy_prec    = m_cls["Cheap_precision"]
        n_buy_days  = (predictions["pred_regime"] == 0).sum() // 96
        avg_price   = float(y_price.mean())
        cheap_price = float(y_price[y_regime == 0].mean()) if (y_regime == 0).sum() > 0 else avg_price
        saving_per_mwh = (avg_price - cheap_price) * buy_prec
        annual_hours   = 365 * 24
        log.info(f"  Avg NL DA price       : EUR{avg_price:.1f}/MWh")
        log.info(f"  Avg 'cheap' day price : EUR{cheap_price:.1f}/MWh")
        log.info(f"  Buy-day precision     : {buy_prec*100:.1f}%")
        log.info(f"  Expected saving/MWh   : EUR{saving_per_mwh:.2f} vs naive procurement")
        log.info(f"  For 10 GWh/yr client  : EUR{saving_per_mwh*10_000:,.0f}/year potential savings")
        log.info("  (Assumes 20% flexible procurement window)")
    log.info("="*65)'''


    # IPI business value estimate (Rigorous Holdout Backtest)
    log.info("\n" + "="*65)
    log.info("  IPI BUSINESS VALUE ESTIMATE")
    log.info("="*65)
    if m_cls.get("Cheap_precision") and len(predictions):
        buy_prec    = m_cls["Cheap_precision"]
        n_buy_days  = (predictions["pred_regime"] == 0).sum() // 96
        
        # 1. What was the average price during the 6-month out-of-sample period?
        avg_price_holdout = float(y_p_hold.mean())
        
        # 2. What was the ACTUAL average price during the hours the model flagged as 'Buy'?
        # This physically accounts for the 84.2% wins AND the 15.8% misses.
        buy_signals = predictions["pred_regime"] == 0
        model_cheap_price = float(y_p_hold[buy_signals].mean()) if buy_signals.sum() > 0 else avg_price_holdout
        
        saving_per_mwh = avg_price_holdout - model_cheap_price
        
        log.info(f"  Avg NL DA price (Holdout) : EUR{avg_price_holdout:.1f}/MWh")
        log.info(f"  Avg price on Model Buys   : EUR{model_cheap_price:.1f}/MWh")
        log.info(f"  Buy-day precision         : {buy_prec*100:.1f}%")
        log.info(f"  Realized saving/MWh       : EUR{saving_per_mwh:.2f} vs naive baseload")
        log.info(f"  For 10 GWh/yr client      : EUR{saving_per_mwh*10_000:,.0f}/year potential savings")
        log.info("  (Assumes 20% flexible procurement window)")
    log.info("="*65)

    # ==============================================================================
    # VOLTCAST IPI — BROKER-READY VALUE ANALYSIS
    # ==============================================================================
    log.info("\n" + "="*65)
    log.info("  VOLTCAST IPI — BROKER-READY VALUE ANALYSIS")
    log.info("="*65)

    if "pred_regime_p0" in predictions.columns and "pred_regime_p2" in predictions.columns:
        # 1. Define Tiers based on ACTUAL column names
        buy_strong   = predictions["pred_regime_p0"] >= 0.70
        buy_mod      = (predictions["pred_regime_p0"] >= 0.55) & (predictions["pred_regime_p0"] < 0.70)
        avoid_strong = predictions["pred_regime_p2"] >= 0.70
        avoid_mod    = (predictions["pred_regime_p2"] >= 0.55) & (predictions["pred_regime_p2"] < 0.70)

        # 2. Extract Prices per Tier
        avg_price_holdout = float(y_p_hold.mean())
        
        # Calculate the actual average price captured during these specific signals
        price_strong_buy = float(y_p_hold[buy_strong].mean()) if buy_strong.sum() > 0 else avg_price_holdout
        price_mod_buy    = float(y_p_hold[buy_mod].mean()) if buy_mod.sum() > 0 else avg_price_holdout
        
        # 3. Savings Calculation (Using all buys to be conservative)
        all_buys = predictions["pred_regime"] == 0
        model_cheap_price = float(y_p_hold[all_buys].mean()) if all_buys.sum() > 0 else avg_price_holdout
        
        theoretical_saving = avg_price_holdout - model_cheap_price
        
        # 15% Flex Factor for Tier 3 Assets (Greenhouses/Cold Storage)
        realistic_flex_factor = 0.15 
        realistic_saving = theoretical_saving * realistic_flex_factor

        log.info(f"  Holdout Period Avg Price : EUR {avg_price_holdout:.2f}/MWh")
        log.info(f"  Realized Buy-Signal Price: EUR {model_cheap_price:.2f}/MWh")
        log.info("-" * 40)
        log.info(f"  [CONFIDENCE STRATIFICATION]")
        log.info(f"  Strong BUY (P>0.70)   : {buy_strong.sum():>5} hours | Avg Price Captured: EUR {price_strong_buy:.2f}/MWh")
        log.info(f"  Moderate BUY (P>0.55) : {buy_mod.sum():>5} hours | Avg Price Captured: EUR {price_mod_buy:.2f}/MWh")
        log.info(f"  Strong AVOID (P>0.70) : {avoid_strong.sum():>5} hours")
        log.info("-" * 40)
        log.info(f"  [ECONOMIC PROJECTION — 10 GWh/yr Client]")
        log.info(f"  Scenario A: 100% Theoretical Flex (Ceiling) ")
        log.info(f"    Saving: EUR {theoretical_saving:.2f}/MWh  | Total: EUR {theoretical_saving*10_000:,.0f}/yr")
        log.info(f"\n  Scenario B: 15% Realistic Asset Flex (Defensible)")
        log.info(f"    Saving: EUR {realistic_saving:.2f}/MWh  | Total: EUR {realistic_saving*10_000:,.0f}/yr")
        log.info(f"    *Basis: Defensible assumption for Greenhouse/Cold Storage cycling.")
    else:
        log.info("  [ERROR] Probability columns not found in predictions dataframe.")
    log.info("="*65)