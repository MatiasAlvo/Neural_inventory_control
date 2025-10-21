#!/usr/bin/env python3
"""
Train LGBM quantile models for cumulative (protection-time) demand.

This version ADDS engineered features while preserving your data rules:
- Inclusive lag window up to decision time t
- Future window (t+1..t+h) must be fully uncensored (avail==1)
- No leakage: only lookback + decision-time calendar used

Added features (per row):
  • sales×avail elementwise (LB dims)
  • EWMAs (fast/slow), OLS slope (trend), variance/CV, burstiness (p95-p50)/p50
  • zero-sales counts (all vs in-stock), streaks (in-stock / stockout)
  • availability volatility & switches, tightness flag at t
  • Fourier seasonal terms (periods 52, 26)
  • Week-of-month (numeric + sin/cos) if 'date' exists; fallback to 4-week cycle
  • Optional: include horizon h as a numeric feature (kept even though we train per-h)

Notes:
- Keep an eye on feature dimensionality if LB is large; trees usually handle it fine.
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def parse_periods(two_ints):
    if two_ints is None or len(two_ints) != 2:
        return None
    a, b = int(two_ints[0]), int(two_ints[1])
    if b < a:
        raise ValueError(f"period end {b} < start {a}")
    return (a, b)

def row_belongs_to_split(t, h, split, LB):
    if split is None:
        return False
    start, end = split
    if t < start or t > end:
        return False
    # require t+h <= end (so target window is fully inside split)
    if (t + h) > end:
        return False
    return True

def compute_week_of_month_from_dates(date_series):
    """
    date_series: pandas Series of datetime64[ns] aligned with t indices.
    Returns:
      wom: int array (1..5)
      wom_sin, wom_cos: cyclical encodings
    """
    wom = []
    for dt in date_series:
        wom.append(int(1 + (dt.day - 1) // 7))  # 1..5
    wom = np.array(wom, dtype=np.int32)
    P = 5.0
    wom_sin = np.sin(2*np.pi*(wom-1)/P).astype(np.float32)
    wom_cos = np.cos(2*np.pi*(wom-1)/P).astype(np.float32)
    return wom, wom_sin, wom_cos

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--lookback_weeks", type=int, default=8)
    p.add_argument("--horizons", type=int, nargs="+", default=[1,2,4,8])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    p.add_argument("--use_time_product_features", type=lambda s: s.lower() in ["1","true","yes","y"], default=False)

    # explicit splits
    p.add_argument("--split_by_period", type=lambda s: s.lower() in ["1","true","yes","y"], default=False,
                   help="If true, use explicit train/dev/test periods instead of val_fraction")
    p.add_argument("--train_periods", type=int, nargs=2, default=None, help="e.g., 0 120 (inclusive indices)")
    p.add_argument("--dev_periods", type=int, nargs=2, default=None, help="e.g., 118 157 (inclusive indices)")
    p.add_argument("--test_periods", type=int, nargs=2, default=None, help="optional, for metadata/reporting")

    # fallback when not using split_by_period
    p.add_argument("--val_fraction", type=float, default=0.2)

    # LGBM params
    p.add_argument("--num_threads", type=int, default=8)
    p.add_argument("--num_leaves", type=int, default=63)
    p.add_argument("--max_trees", type=int, default=400)
    p.add_argument("--early_stopping", type=int, default=75)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--random_state", type=int, default=42)

    # engineering options
    p.add_argument("--include_h_as_feature", type=lambda s: s.lower() in ["1","true","yes","y"], default=True,
                   help="Include horizon h as a numeric feature (still trains per-h models).")

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    # lazy imports
    import torch
    try:
        import lightgbm as lgb
        HAS_LGBM = True
    except Exception as e:
        HAS_LGBM = False
        LGBM_IMPORT_ERROR = str(e)

    from sklearn.preprocessing import OrdinalEncoder

    # -----------------------------
    # Load data
    # -----------------------------
    sales = torch.load(Path(args.data_dir) / "sales.pt")
    stock = torch.load(Path(args.data_dir) / "stock.pt")
    product_df = pd.read_csv(Path(args.data_dir) / "product_features.csv")
    date_df = pd.read_csv(Path(args.data_dir) / "date_features.csv")

    tp_path = Path(args.data_dir) / "time_product_features.pt"
    has_tp = tp_path.exists()
    time_product_features = torch.load(tp_path) if has_tp else None

    N, _, T_sales = sales.shape
    _, _, T_stock = stock.shape
    T_date = len(date_df)

    maxH = max(args.horizons)

    # Align timeline (date_df may have one extra row; we cap at T_sales)
    T_base = min(T_sales, T_date)
    T_eff = min(T_base, T_stock - maxH)
    if T_eff <= args.lookback_weeks + maxH - 1:
        raise ValueError(f"Not enough timeline: T_eff={T_eff}, lookback={args.lookback_weeks}, maxH={maxH}")

    sales_np = sales.numpy()[:, 0, :T_eff + maxH]
    stock_np = stock.numpy()[:, 0, :T_eff + maxH]
    date_df_eff = date_df.iloc[:T_eff].reset_index(drop=True)  # stops at last observed demand date

    # Try to parse dates (for true week-of-month); keep a fallback flag
    has_true_wom = False
    if "date" in date_df_eff.columns:
        try:
            dates = pd.to_datetime(date_df_eff["date"])
            wom_idx, wom_sin, wom_cos = compute_week_of_month_from_dates(dates)
            has_true_wom = True
        except Exception:
            has_true_wom = False
    if not has_true_wom:
        # Fallback: 4-week cycle approximation
        t_arr = np.arange(T_eff, dtype=np.int32)
        wom_idx = (t_arr % 4) + 1
        P = 4.0
        wom_sin = np.sin(2*np.pi*(wom_idx-1)/P).astype(np.float32)
        wom_cos = np.cos(2*np.pi*(wom_idx-1)/P).astype(np.float32)

    # Optional time-product features
    if has_tp and args.use_time_product_features:
        tp = time_product_features.numpy()[:, :, 0, :T_eff + maxH]  # [F_tp, N, T_eff+maxH]
    else:
        tp = None

    # Static categorical features
    assert len(product_df) == N, "product_features.csv must have N rows"
    cat_cols = list(product_df.columns)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    static_cats_enc = pd.DataFrame(enc.fit_transform(product_df[cat_cols]), columns=[f"cat_{c}" for c in cat_cols])

    # Time features (drop 'date' column from generic time features; we'll add wom separately)
    time_feats_df = date_df_eff.drop(columns=[c for c in ["date"] if c in date_df_eff.columns]).reset_index(drop=True)
    time_feat_cols = list(time_feats_df.columns)

    LB = args.lookback_weeks
    rows = []

    # Parse explicit splits (inclusive indices), if requested
    train_split = parse_periods(args.train_periods) if args.split_by_period else None
    dev_split   = parse_periods(args.dev_periods)   if args.split_by_period else None
    test_split  = parse_periods(args.test_periods)  if args.split_by_period else None

    # -----------------------------
    # Build supervised rows
    # -----------------------------
    for i in range(N):
        s = sales_np[i]
        a = stock_np[i]

        for t in range(LB - 1, T_eff):  # decision time
            # Inclusive lag window [t-LB+1 : t]
            past_sales = s[t - LB + 1 : t + 1].astype(np.float32)
            past_avail = a[t - LB + 1 : t + 1].astype(np.float32)

            if past_sales.shape[0] != LB or past_avail.shape[0] != LB:
                continue

            eps = 1e-6
            roll_mean_all = float(np.mean(past_sales)) if LB > 0 else 0.0
            # uncensored-only mean
            mask_unc = (past_avail == 1.0)
            if mask_unc.any():
                roll_mean_unc = float(np.sum(past_sales[mask_unc]) / max(1.0, np.sum(mask_unc)))
            else:
                roll_mean_unc = 0.0
            unc_frac = float(np.mean(past_avail)) if LB > 0 else 1.0

            # weeks since last stockout within lookback (LB if none)
            if LB > 0 and np.any(past_avail == 0):
                last_idx = np.where(past_avail == 0)[0].max()
                since_last_so = int(LB - 1 - last_idx)
            else:
                since_last_so = int(LB)

            # ---------- Engineered features from lookback ----------
            ps = past_sales
            pa = past_avail

            # EWMA (fast/slow)
            def ewma_last(x, alpha):
                v = 0.0
                first = True
                for xi in x:
                    if first:
                        v = float(xi)
                        first = False
                    else:
                        v = alpha*float(xi) + (1.0-alpha)*v
                return float(v)

            ewma_fast = ewma_last(ps, 0.5)
            ewma_slow = ewma_last(ps, 0.2)

            # Trend slope (OLS) over indices 0..LB-1
            x_idx = np.arange(LB, dtype=np.float32)
            x_mean = float(x_idx.mean())
            y_mean = float(ps.mean())
            den = float(((x_idx - x_mean)**2).sum() + eps)
            slope = float(((x_idx - x_mean)*(ps - y_mean)).sum() / den)

            # Variability & burstiness (uncensored-only)
            var_all = float(ps.var())
            if mask_unc.any():
                ps_unc = ps[mask_unc]
                mean_unc = float(ps_unc.mean())
                var_unc  = float(ps_unc.var())
                p50_unc  = float(np.percentile(ps_unc, 50))
                p95_unc  = float(np.percentile(ps_unc, 95))
                zeros_all = float((ps == 0).sum())
                zeros_unc = float((ps_unc == 0).sum())
            else:
                mean_unc = 0.0; var_unc = 0.0; p50_unc = 0.0; p95_unc = 0.0
                zeros_all = float((ps == 0).sum())
                zeros_unc = 0.0

            cv_unc = float(np.sqrt(max(var_unc, 0.0)) / (mean_unc + eps))
            burstiness = float((p95_unc - p50_unc) / (p50_unc + eps))

            # Streaks up to t
            streak_in = 0
            for v in pa[::-1]:
                if v == 1.0: streak_in += 1
                else: break
            streak_out = 0
            for v in pa[::-1]:
                if v == 0.0: streak_out += 1
                else: break

            # Availability volatility
            avail_std = float(pa.std())
            avail_switches = float((pa[1:] != pa[:-1]).sum())

            # Tightness at t: in stock AND last sale near mean_unc (simple heuristic)
            tight_flag = float((pa[-1] == 1.0) and (ps[-1] >= 0.8 * (mean_unc if mask_unc.any() else ps.mean())))

            # Interactions: elementwise sales×avail (LB dims)
            sales_x_avail = (ps * pa).astype(np.float32)

            # Optional transforms: log1p of sales lags (helps heavy tails)
            log1p_sales = np.log1p(ps).astype(np.float32)

            # Fourier seasonality using t index (periods 52 & 26)
            week_idx = float(t % 52)
            s1 = np.sin(2*np.pi*week_idx/52.0); c1 = np.cos(2*np.pi*week_idx/52.0)
            s2 = np.sin(2*np.pi*week_idx/26.0); c2 = np.cos(2*np.pi*week_idx/26.0)

            # Week-of-month (numeric + cyclical), aligned at t
            wom_num  = float(wom_idx[t])         # 1..5 (or 1..4 fallback)
            wom_sin_t = float(wom_sin[t])
            wom_cos_t = float(wom_cos[t])

            # ---------- Decision-time features ----------
            time_vec = time_feats_df.iloc[t].values.astype(np.float32)
            static_vec = static_cats_enc.iloc[i].values.astype(np.float32)
            if tp is not None:
                tp_vec = tp[:, i, t].astype(np.float32)
            else:
                tp_vec = np.array([], dtype=np.float32)

            # Pack engineered scalars
            engineered = np.array([
                ewma_fast, ewma_slow, slope,
                var_all, mean_unc, var_unc, cv_unc, burstiness,
                zeros_all, zeros_unc,
                streak_in, streak_out, avail_std, avail_switches,
                tight_flag,
                s1, c1, s2, c2,
                wom_num, wom_sin_t, wom_cos_t
            ], dtype=np.float32)

            # Base vector (order matters; we’ll mirror in feat_names)
            base_prefix = [
                past_sales,                # LB
                past_avail,                # LB
                sales_x_avail,             # LB
                log1p_sales,               # LB
                np.array([roll_mean_all, roll_mean_unc, unc_frac, since_last_so], dtype=np.float32),
                engineered,
                time_vec, static_vec, tp_vec
            ]

            for h in args.horizons:
                # Require full availability in target window (uncensored future)
                future_avail = a[t + 1 : t + 1 + h]
                if future_avail.shape[0] < h:
                    continue
                if not np.all(future_avail == 1.0):
                    continue

                # split tagging
                split_tag = "all"
                if args.split_by_period:
                    if row_belongs_to_split(t, h, train_split, LB):
                        split_tag = "train"
                    elif row_belongs_to_split(t, h, dev_split, LB):
                        split_tag = "dev"
                    elif row_belongs_to_split(t, h, test_split, LB):
                        split_tag = "test"
                    else:
                        continue

                # cumulative future demand for t+1..t+h
                y = float(np.sum(s[t + 1 : t + 1 + h]))

                # Optionally include horizon as a feature
                if args.include_h_as_feature:
                    Xrow = np.concatenate(base_prefix + [np.array([float(h)], dtype=np.float32)])
                else:
                    Xrow = np.concatenate(base_prefix)

                rows.append((i, t, h, Xrow, y, split_tag))

    if len(rows) == 0:
        raise RuntimeError("No training rows constructed. Check availability flags, alignment, and split windows.")

    # Build per-horizon matrices
    rows_by_h = {h: [] for h in args.horizons}
    for rec in rows:
        rows_by_h[rec[2]].append(rec)

    models_info = []
    coverage_report = []
    split_counts = {h: {"train": 0, "dev": 0, "test": 0} for h in args.horizons}

    # ---------- Feature names schema (mirror the construction order) ----------
    def build_feat_names(LB, time_feat_cols, static_cats_enc, tp, include_h):
        names = []
        names += [f"sales_lag_{k}" for k in range(LB, 0, -1)]
        names += [f"avail_lag_{k}" for k in range(LB, 0, -1)]
        names += [f"sales_x_avail_{k}" for k in range(LB, 0, -1)]
        names += [f"log1p_sales_lag_{k}" for k in range(LB, 0, -1)]
        names += ["roll_mean_all", "roll_mean_unc", "unc_frac", "since_last_so"]
        names += [
            "ewma_fast","ewma_slow","trend_slope",
            "var_all","mean_unc","var_unc","cv_unc","burstiness",
            "zeros_all","zeros_unc",
            "streak_in","streak_out","avail_std","avail_switches",
            "tight_flag",
            "fourier_s52","fourier_c52","fourier_s26","fourier_c26",
            "week_of_month_num","week_of_month_sin","week_of_month_cos",
        ]
        names += [f"time_{c}" for c in time_feat_cols]
        names += [f"static_{c}" for c in static_cats_enc.columns]
        if tp is not None:
            names += [f"tp_{k}" for k in range(tp.shape[0])]
        if include_h:
            names += ["horizon"]
        return names

    base_feat_names = build_feat_names(LB, time_feat_cols, static_cats_enc, tp, args.include_h_as_feature)

    for h in args.horizons:
        recs = rows_by_h[h]
        if len(recs) == 0:
            continue

        X = np.stack([r[3] for r in recs], axis=0)
        y = np.array([r[4] for r in recs], dtype=np.float32)
        t_idx = np.array([r[1] for r in recs], dtype=np.int32)
        tags = np.array([r[5] for r in recs], dtype=object)

        # Split masks
        if args.split_by_period:
            train_mask = (tags == "train")
            val_mask   = (tags == "dev")
            split_counts[h]["train"] = int(train_mask.sum())
            split_counts[h]["dev"]   = int(val_mask.sum())
            split_counts[h]["test"]  = int((tags == "test").sum())
            if not val_mask.any():
                t_sorted = np.sort(np.unique(t_idx[train_mask]))
                if len(t_sorted) == 0:
                    raise RuntimeError(f"No train rows for horizon {h}")
                cutoff = t_sorted[int(len(t_sorted) * (1 - args.val_fraction))] if len(t_sorted) > 1 else t_sorted[-1]
                tr = train_mask & (t_idx <= cutoff)
                vl = train_mask & (t_idx > cutoff)
                train_mask, val_mask = tr, vl
        else:
            t_sorted = np.sort(np.unique(t_idx))
            cutoff = t_sorted[int(len(t_sorted) * (1 - args.val_fraction))] if len(t_sorted) > 1 else t_sorted[-1]
            train_mask = t_idx <= cutoff
            val_mask   = t_idx > cutoff

        X_train, y_train = X[train_mask], y[train_mask]
        X_val,   y_val   = X[val_mask],   y[val_mask]

        # Feature names (same for all horizons)
        feat_names = list(base_feat_names)

        if HAS_LGBM:
            import lightgbm as lgb
            lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=feat_names, free_raw_data=False)
            lgb_val   = lgb.Dataset(X_val, label=y_val, reference=lgb_train, free_raw_data=False)
        else:
            raise RuntimeError(f"lightgbm not available: {LGBM_IMPORT_ERROR}")

        for alpha in args.alphas:
            params = {
                "objective": "quantile",
                "alpha": alpha,
                "learning_rate": args.learning_rate,
                "num_leaves": args.num_leaves,
                "min_data_in_leaf": 25,
                "feature_fraction": 0.85,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "metric": "quantile",
                "verbosity": -1,
                "force_col_wise": True,
                "num_threads": args.num_threads,
                "seed": args.random_state,
            }
            booster = lgb.train(
                params,
                lgb_train,
                num_boost_round=args.max_trees,
                valid_sets=[lgb_val],
                valid_names=["valid"],
                callbacks=[lgb.early_stopping(args.early_stopping, verbose=False)],
            )
            model_path = str(out_dir / "models" / f"lgbm_quantile_h{h}_a{alpha}.txt")
            booster.save_model(model_path)
            models_info.append({"h": h, "alpha": float(alpha), "path": model_path})

            if X_val.shape[0] > 0:
                pred = booster.predict(X_val)
                cov = float(np.mean(y_val <= pred))
                coverage_report.append({"h": h, "alpha": float(alpha), "val_rows": int(len(y_val)), "empirical_coverage": cov})

        # Save feature schema (per horizon; same names but keep compatibility)
        schema_path = out_dir / "models" / f"feature_schema_h{h}.json"
        with open(schema_path, "w") as f:
            json.dump({"feat_names": feat_names, "lookback": LB}, f, indent=2)

    # Metadata
    meta_obj = {
        "config": {
            "LOOKBACK_WEEKS": args.lookback_weeks,
            "HORIZONS": args.horizons,
            "ALPHAS": args.alphas,
            "USE_TIME_PRODUCT_FEATURES": args.use_time_product_features,
            "VALIDATION_SPLIT_FRACTION": args.val_fraction,
            "MAX_TREES": args.max_trees,
            "EARLY_STOPPING_ROUNDS": args.early_stopping,
            "NUM_LEAVES": args.num_leaves,
            "LEARNING_RATE": args.learning_rate,
            "N_THREADS": args.num_threads,
            "RANDOM_STATE": args.random_state,
            "SPLIT_BY_PERIOD": bool(args.split_by_period),
            "TRAIN_PERIODS": args.train_periods,
            "DEV_PERIODS": args.dev_periods,
            "TEST_PERIODS": args.test_periods,
            "INCLUDE_H_AS_FEATURE": args.include_h_as_feature,
            "HAS_TRUE_WOM": bool("date" in date_df.columns)
        },
        "N_items": int(N),
        "T_eff": int(T_eff),
        "features_time_cols": list(time_feat_cols),
        "models": models_info,
        "coverage_report": coverage_report,
        "split_counts": split_counts if args.split_by_period else None,
    }
    with open(out_dir / "models" / "metadata.json", "w") as f:
        json.dump(meta_obj, f, indent=2)

    # Forecast service helper
    (out_dir / "models" / "forecast_service.py").write_text(ForecastService_code())

    # Save encoder mapping (static categories)
    enc_path = out_dir / "models" / "static_cats_ordinal_encoder.json"
    try:
        cats = [list(cat) for cat in enc.categories_]
        with open(enc_path, "w") as f:
            json.dump({"columns": cat_cols, "categories": cats}, f, indent=2)
    except Exception:
        pass

    print("Training complete.")
    print("Saved metadata:", str(out_dir / "models" / "metadata.json"))
    print("Saved ForecastService:", str(out_dir / "models" / "forecast_service.py"))
    if len(models_info) > 0:
        print("Example model:", models_info[0])

def ForecastService_code():
    return r"""
import json
import numpy as np
import lightgbm as lgb

class ForecastService:
    \"\"\"Batch LGBM-quantile forecaster for cumulative demand.
    Load with metadata.json, then call predict_quantiles(h, X) to get a matrix of quantile preds.\"\"\"
    def __init__(self, models_meta_path: str):
        with open(models_meta_path, "r") as f:
            meta = json.load(f)
        self.meta = meta
        self.models = {}  # (h, alpha) -> booster
        for m in meta.get("models", []):
            if m.get("path", "").endswith(".txt"):
                booster = lgb.Booster(model_file=m["path"])
                self.models[(m["h"], float(m["alpha"]))] = booster

        # Load feature schema per horizon
        self.schemas = {}
        for h in self.meta["config"]["HORIZONS"]:
            try:
                with open(models_meta_path.replace("metadata.json", f"feature_schema_h{h}.json"), "r") as f:
                    self.schemas[h] = json.load(f)
            except FileNotFoundError:
                pass

    def predict_quantiles(self, h: int, X: np.ndarray, alphas=None):
        if alphas is None:
            alphas = list(self.meta["config"]["ALPHAS"])
        preds = {}
        for a in alphas:
            mdl = self.models.get((h, float(a)))
            if mdl is None:
                continue
            preds[a] = mdl.predict(X)
        alphas_sorted = sorted(preds.keys())
        if len(alphas_sorted) == 0:
            return None, []
        mat = np.stack([preds[a] for a in alphas_sorted], axis=1)
        return mat, alphas_sorted
"""

if __name__ == "__main__":
    main()
