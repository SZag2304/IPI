#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
 VOLTCAST IPI v2.1 — FAIR HOLDOUT RE-EVALUATION OF THE V2 CHALLENGER
═══════════════════════════════════════════════════════════════════════
WHY THIS SCRIPT EXISTS
  The v2 bias redesign is SHAPE-ONLY: the +4.10 EUR/MWh level component
  was deliberately removed from the Season×DoW×Hour table and delegated
  to the live adaptive rolling intercept (live_model_v2). The v2 holdout
  evaluation in training_model_v2.py applies the shape table but NEVER
  simulates the intercept — so the reported holdout (MAE 15.46,
  MBE −4.57) systematically understates what the production stack will
  actually deliver. This script re-runs the holdout with the intercept
  simulated exactly per the live spec (10-day window, ≥4 settled days,
  clip ±20) and prints a 3-way decision table:

      V1 champion (reference)  vs  V2 as-reported  vs  V2 fair

USAGE
  python holdout_fair_eval_v2.py \
      --features VoltCast_IPI_Features_v2.parquet \
      --cache    voltcast_ipi_cache_v2 \
      --holdout-start 2025-10-01

  All artefact paths are auto-discovered inside --cache. Every load is
  defensive: missing artefacts produce a clear message, never a silent
  skip. Results are persisted to <cache>/fair_holdout_eval.json.

NO TRAINING ARTEFACT IS MODIFIED. Read-only evaluation.
═══════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

SPIKE_THRESHOLD = 120.0
SHAPE_CLIP_EUR = 15.0          # same clip as training bias application
V1_REFERENCE = {               # from training_model_v1 holdout log (2026-06-11)
    "MAE": 14.79, "RMSE": 20.93, "R2": 0.7357, "MBE": -0.97,
    "Dir%": 77.8, "SpikeMAE": 19.55, "NegMAE": 7.67, "BuyPrec%": 81.3,
}

TARGET_CANDIDATES = ["DA_Price_NL_EURMWh", "NL_DA_Price_EURMWh", "target",
                     "y", "price", "DA_Price_EURMWh"]
REGIME_CANDIDATES = ["regime", "target_regime", "y_regime", "regime_class",
                     "Regime", "regime_label"]
THRESH_FILE_CANDIDATES = ["optimal_thresholds.json", "thresholds.json",
                          "classifier_thresholds.json", "model_card.json"]


# ─────────────────────────────────────────────────────────────────────
#  Symlog inverse (must match training transform)
# ─────────────────────────────────────────────────────────────────────
def inv_symlog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.expm1(np.abs(x))


# ─────────────────────────────────────────────────────────────────────
#  Season / calendar helpers (holdout breakdown in the v2 log confirms
#  meteorological mapping: autumn=Sep–Nov, winter=Dec–Feb, spring=Mar–May)
# ─────────────────────────────────────────────────────────────────────
SEASON_NAME = {12: "winter", 1: "winter", 2: "winter",
               3: "spring", 4: "spring", 5: "spring",
               6: "summer", 7: "summer", 8: "summer",
               9: "autumn", 10: "autumn", 11: "autumn"}
SEASON_INT = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}


# ─────────────────────────────────────────────────────────────────────
#  Bias table parser — tolerant to flat ("winter|2|17", "winter_2_17",
#  "('winter', 2, 17)") or nested {"winter": {"2": {"17": v}}} layouts.
# ─────────────────────────────────────────────────────────────────────
def parse_bias_table(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    table = {}

    def norm_season(s):
        s = str(s).strip().strip("'\"").lower()
        return s

    if isinstance(raw, dict) and raw and isinstance(next(iter(raw.values())), dict):
        for s, dows in raw.items():
            for d, hours in dows.items():
                for h, v in hours.items():
                    table[(norm_season(s), int(float(d)), int(float(h)))] = float(v)
    else:
        for k, v in raw.items():
            parts = re.split(r"[|,_;]+", str(k).strip("()' \""))
            parts = [p.strip().strip("'\"") for p in parts if p.strip()]
            if len(parts) < 3:
                continue
            season = norm_season("_".join(parts[:-2]))
            table[(season, int(float(parts[-2])), int(float(parts[-1])))] = float(v)

    if not table:
        raise ValueError(f"Could not parse any (season, dow, hour) cells from {path}. "
                         f"Sample keys: {list(raw)[:3]}")
    return table


def shape_correction(index: pd.DatetimeIndex, table: dict) -> np.ndarray:
    corr, hits = np.zeros(len(index)), 0
    months, dows, hours = index.month, index.dayofweek, index.hour
    for i in range(len(index)):
        sname = SEASON_NAME[months[i]]
        for key in ((sname, dows[i], hours[i]),
                    (str(SEASON_INT[sname]), dows[i], hours[i])):
            if key in table:
                corr[i] = np.clip(table[key], -SHAPE_CLIP_EUR, SHAPE_CLIP_EUR)
                hits += 1
                break
    hit_rate = hits / len(index)
    if hit_rate < 0.5:
        print(f"  [WARN] Bias-table hit rate only {hit_rate:.0%} — key format mismatch? "
              f"Sample table keys: {list(table)[:3]}. Shape correction may be incomplete.")
    else:
        print(f"  Shape table applied: hit rate {hit_rate:.1%}, "
              f"mean |corr| EUR{np.abs(corr).mean():.2f}/MWh")
    return corr


# ─────────────────────────────────────────────────────────────────────
#  Adaptive intercept simulation — EXACT live_model_v2 spec:
#  intercept(day d) = clip( mean(actual − pred_shape) over the last
#  `window_days` settled days, ±clip_eur ); bootstrap until min_days.
#  Walk-forward: day d only ever sees days < d. Zero look-ahead.
# ─────────────────────────────────────────────────────────────────────
def simulate_adaptive_intercept(pred_shape, actual, index,
                                window_days=10, min_days=4,
                                clip_eur=20.0, bootstrap=4.10):
    days = pd.DatetimeIndex(index.normalize())
    uniq = days.unique().sort_values()
    daily_err, intercepts = {}, {}
    out = np.zeros(len(pred_shape))
    for d in uniq:
        hist = [daily_err[h] for h in uniq[uniq < d][-window_days:] if h in daily_err]
        ic = float(np.clip(np.mean(hist), -clip_eur, clip_eur)) if len(hist) >= min_days \
             else float(bootstrap)
        mask = days == d
        out[mask] = ic
        intercepts[d] = ic
        daily_err[d] = float((actual[mask] - pred_shape[mask]).mean())
    print(f"  Adaptive intercept: bootstrap {bootstrap:+.2f} → "
          f"range [{min(intercepts.values()):+.2f}, {max(intercepts.values()):+.2f}], "
          f"final-30d mean {np.mean(list(intercepts.values())[-30:]):+.2f} EUR/MWh")
    return pred_shape + out, len([d for d in uniq if d not in list(uniq[:min_days])])


# ─────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────
def metric_row(pred, actual, actual_lag96):
    err = pred - actual
    valid = ~np.isnan(actual_lag96)
    dir_acc = float(np.mean(
        np.sign(pred[valid] - actual_lag96[valid]) ==
        np.sign(actual[valid] - actual_lag96[valid]))) * 100
    spike, neg = actual > SPIKE_THRESHOLD, actual < 0
    ss_res, ss_tot = np.sum(err ** 2), np.sum((actual - actual.mean()) ** 2)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1 - ss_res / ss_tot),
        "MBE": float(np.mean(err)),
        "Dir%": dir_acc,
        "SpikeMAE": float(np.mean(np.abs(err[spike]))) if spike.any() else float("nan"),
        "SpikeMBE": float(np.mean(err[spike])) if spike.any() else float("nan"),
        "NegMAE": float(np.mean(np.abs(err[neg]))) if neg.any() else float("nan"),
        "n": int(len(pred)),
    }


def print_table(rows: dict, cols):
    head = f"  {'Variant':38s}" + "".join(f"{c:>10s}" for c in cols)
    print(head + "\n  " + "─" * (len(head) - 2))
    for name, r in rows.items():
        line = f"  {name:38s}"
        for c in cols:
            v = r.get(c, float("nan"))
            line += f"{v:>10.2f}" if isinstance(v, float) else f"{v:>10}"
        print(line)


# ─────────────────────────────────────────────────────────────────────
#  Artefact loaders
# ─────────────────────────────────────────────────────────────────────
def must_exist(path, what):
    if not os.path.exists(path):
        sys.exit(f"[FATAL] {what} not found: {path}")
    return path


def detect_column(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            print(f"  {what} column: '{c}'")
            return c
    pat = [c for c in df.columns if re.search(r"DA_Price_NL", c)] if what == "Target" else []
    if pat:
        print(f"  {what} column (pattern match): '{pat[0]}'")
        return pat[0]
    return None


def load_thresholds(cache):
    for fn in THRESH_FILE_CANDIDATES:
        p = os.path.join(cache, fn)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            d = json.load(f)
        flat = json.dumps(d).lower()
        if "cheap" in flat:
            def find(obj, key):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if key in k.lower() and isinstance(v, (int, float)):
                            return float(v)
                        r = find(v, key)
                        if r is not None:
                            return r
                return None
            c, e = find(d, "cheap"), find(d, "exp")
            if c is not None and e is not None:
                print(f"  Calibrated thresholds from {fn}: cheap={c:.2f} exp={e:.2f}")
                return c, e
    print("  [WARN] No persisted thresholds found — using log values 0.45 / 0.56")
    return 0.45, 0.56


# ═════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="VoltCast_IPI_Features_v2.parquet")
    ap.add_argument("--cache", default="voltcast_ipi_cache_v2")
    ap.add_argument("--holdout-start", default="2025-10-01")
    ap.add_argument("--bootstrap", type=float, default=None,
                    help="Intercept bootstrap; default = global_level_removed "
                         "from bias_meta.json (information available at train "
                         "time), fallback +4.10")
    ap.add_argument("--window-days", type=int, default=10)
    ap.add_argument("--min-days", type=int, default=4)
    ap.add_argument("--clip", type=float, default=20.0)
    args = ap.parse_args()

    print("=" * 65)
    print("  VOLTCAST IPI v2.1 — FAIR HOLDOUT RE-EVALUATION")
    print("=" * 65)

    # ── Load features & target ──────────────────────────────────────
    df = pd.read_parquet(must_exist(args.features, "Features parquet"))
    if not isinstance(df.index, pd.DatetimeIndex):
        sys.exit("[FATAL] Features parquet has no DatetimeIndex.")
    tgt = detect_column(df, TARGET_CANDIDATES, "Target")
    if tgt is None:
        sys.exit(f"[FATAL] No target column found. Tried: {TARGET_CANDIDATES}. "
                 f"Pass the real name by editing TARGET_CANDIDATES.")

    feat_path = must_exist(os.path.join(args.cache, "feature_names.txt"), "feature_names.txt")
    feat_names = [l.strip() for l in open(feat_path) if l.strip()]
    missing = [c for c in feat_names if c not in df.columns]
    if missing:
        sys.exit(f"[FATAL] {len(missing)} model features missing from parquet, "
                 f"e.g. {missing[:5]}")

    actual_lag96_full = df[tgt].shift(96)          # lag computed BEFORE slicing
    hold = df[df.index >= pd.Timestamp(args.holdout_start)]
    print(f"  Holdout: {hold.index.min()} → {hold.index.max()}  "
          f"({len(hold):,} rows; expected 17,472)")
    X = hold[feat_names].astype(float)
    y = hold[tgt].to_numpy(dtype=float)
    y_lag96 = actual_lag96_full.loc[hold.index].to_numpy(dtype=float)

    # ── Point forecast: XGB + LGB 50/50, symlog inverse ─────────────
    if xgb is None or lgb is None:
        sys.exit("[FATAL] xgboost and lightgbm must be installed.")
    bx = xgb.Booster(); bx.load_model(must_exist(
        os.path.join(args.cache, "model_point.json"), "model_point.json"))
    bl = lgb.Booster(model_file=must_exist(
        os.path.join(args.cache, "model_point_lgb.txt"), "model_point_lgb.txt"))
    p_xgb = inv_symlog(bx.predict(xgb.DMatrix(X, feature_names=feat_names)))
    p_lgb = inv_symlog(bl.predict(X.to_numpy()))
    pred_raw = 0.5 * p_xgb + 0.5 * p_lgb

    # ── Shape-only bias table ────────────────────────────────────────
    bias_path = must_exist(os.path.join(args.cache, "season_hour_bias.json"),
                           "season_hour_bias.json")
    table = parse_bias_table(bias_path)
    pred_shape = pred_raw + shape_correction(hold.index, table)

    # ── Bootstrap from bias_meta if present ──────────────────────────
    bootstrap = args.bootstrap
    meta_path = os.path.join(args.cache, "bias_meta.json")
    if bootstrap is None:
        if os.path.exists(meta_path):
            bootstrap = float(json.load(open(meta_path)).get("global_level_removed", 4.10))
            print(f"  Bootstrap intercept from bias_meta.json: {bootstrap:+.2f} EUR/MWh")
        else:
            bootstrap = 4.10
            print("  [WARN] bias_meta.json absent — bootstrap defaulted to +4.10 "
                  "(level removed per v2 training log)")

    # ── Fair prediction: shape + simulated adaptive intercept ────────
    pred_fair, _ = simulate_adaptive_intercept(
        pred_shape, y, hold.index, window_days=args.window_days,
        min_days=args.min_days, clip_eur=args.clip, bootstrap=bootstrap)

    # ── Point decision table ─────────────────────────────────────────
    rows = {
        "V1 champion (reported, reference)": V1_REFERENCE,
        "V2 raw ensemble (no bias layer)":   metric_row(pred_raw, y, y_lag96),
        "V2 shape-only (= reported eval)":   metric_row(pred_shape, y, y_lag96),
        "V2 FAIR (shape + live intercept)":  metric_row(pred_fair, y, y_lag96),
    }
    print("\n" + "=" * 65 + "\n  POINT FORECAST — 3-WAY DECISION TABLE\n" + "=" * 65)
    print_table(rows, ["MAE", "RMSE", "R2", "MBE", "Dir%", "SpikeMAE", "NegMAE"])
    print("\n  Sanity check: 'V2 shape-only' must reproduce the training log "
          "(MAE 15.46, MBE −4.57).\n  If it does not, the bias-key parser missed "
          "— inspect the WARN above before trusting the FAIR row.")

    # ── Seasonal breakdown under FAIR eval ───────────────────────────
    print("\n  FAIR seasonal breakdown (v2-reported in brackets):")
    ref = {"winter": (13.51, -4.57), "spring": (22.49, -8.89), "autumn": (14.78, -2.37)}
    for s in ["autumn", "winter", "spring"]:
        m = np.array([SEASON_NAME[mn] == s for mn in hold.index.month])
        if m.any():
            r = metric_row(pred_fair[m], y[m], y_lag96[m])
            print(f"    {s:8s} n={m.sum():5d}  MAE={r['MAE']:6.2f} ({ref[s][0]:.2f})"
                  f"  MBE={r['MBE']:+6.2f} ({ref[s][1]:+.2f})")

    # ── Classifier: evaluate at calibrated thresholds + diagnostics ──
    out = {"point": {k: rows[k] for k in rows if k != "V1 champion (reported, reference)"}}
    reg_col = detect_column(hold, REGIME_CANDIDATES, "Regime")
    clf_x = os.path.join(args.cache, "model_classifier.json")
    clf_l = os.path.join(args.cache, "model_classifier_lgb.txt")
    if reg_col and os.path.exists(clf_x) and os.path.exists(clf_l):
        cx = xgb.Booster(); cx.load_model(clf_x)
        cl = lgb.Booster(model_file=clf_l)
        px = cx.predict(xgb.DMatrix(X, feature_names=feat_names))
        pl = cl.predict(X.to_numpy())
        if px.ndim == 1:
            print("  [WARN] XGB classifier returned class ids, not probabilities — "
                  "classifier section skipped.")
        else:
            proba = 0.5 * px + 0.5 * pl
            yr = hold[reg_col]
            if yr.dtype == object:
                yr = yr.str.lower().map({"cheap": 0, "normal": 1, "expensive": 2})
            yr = yr.to_numpy()
            thr_c, thr_e = load_thresholds(args.cache)

            def classify(p, tc, te):
                pred = np.ones(len(p), dtype=int)
                pred[(p[:, 0] >= tc) & (p[:, 0] >= p[:, 2])] = 0
                pred[(p[:, 2] >= te) & (p[:, 2] > p[:, 0])] = 2
                return pred

            pc = classify(proba, thr_c, thr_e)
            for cid, nm in [(0, "Buy  (Cheap)"), (2, "Avoid(Expensive)")]:
                sel = pc == cid
                prec = (yr[sel] == cid).mean() * 100 if sel.any() else float("nan")
                print(f"\n  {nm} @ calibrated thr: precision {prec:5.1f}%  "
                      f"(n={sel.sum()})")
            print("\n  DIAGNOSTIC ONLY (holdout precision vs cheap threshold — "
                  "re-tune on the CV fold, never here):")
            print(f"    {'thr':>5s} {'prec%':>7s} {'hours':>7s} {'avg EUR/MWh on buys':>20s}")
            for t in np.arange(0.45, 0.81, 0.05):
                sel = (proba[:, 0] >= t) & (proba[:, 0] >= proba[:, 2])
                if sel.sum() < 50:
                    break
                prec = (yr[sel] == 0).mean() * 100
                print(f"    {t:5.2f} {prec:7.1f} {sel.sum():7d} {y[sel].mean():20.2f}")
            out["classifier"] = {"thr_cheap": thr_c, "thr_exp": thr_e}
    else:
        print("\n  [INFO] Classifier section skipped "
              f"(regime column found: {bool(reg_col)}).")

    with open(os.path.join(args.cache, "fair_holdout_eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {os.path.join(args.cache, 'fair_holdout_eval.json')}")
    print("=" * 65)


if __name__ == "__main__":
    main()