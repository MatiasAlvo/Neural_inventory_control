#!/usr/bin/env python3
"""
Train LGBM quantile models for cumulative (protection-time) demand.

Changes in this version:
- LAG WINDOW IS INCLUSIVE OF t:
  past_sales = s[t-LB+1 : t+1], past_avail = a[t-LB+1 : t+1]
- LOOP STARTS AT t = LB-1 (instead of LB)
- Time features remain at time t (decision-time features).
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
    """
    Decide if a row with decision time t and horizon h belongs to split.
    We require:
      - decision time t within [start, end]
      - target window (t+1 .. t+h) fully within [start, end]
    Note: earliest usable t must be >= LB-1 due to inclusive lookback.
    """
    if split is None:
        return False
    start, end = split
    if t < start or t > end:
        return False
    # require t+h <= end (so target window doesn't cross split boundary)
    if (t + h) > end:
        return False
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--lookback_weeks", type=int, default=8)
    p.add_argument("--horizons", type=int, nargs="+", default=[1,2])
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
    p.add_argument("--max_trees", type=int, default=300)
    p.add_argument("--early_stopping", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--random_state", type=int, default=42)
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
    date_df_eff = date_df.iloc[:T_eff].reset_index(drop=True)  # stops at last observed demand date (e.g., 2024-04-08)

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

    # Time features (drop any 'date' column)
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
        # CHANGED: start at LB-1 so we can include index t in the lag window
        for t in range(LB - 1, T_eff):  # decision time (last observed week)
            # CHANGED: inclusive lag window [t-LB+1 : t+1] -> includes s[t]
            past_sales = s[t - LB + 1 : t + 1]
            past_avail = a[t - LB + 1 : t + 1]

            # defensive: ensure exact length LB
            if past_sales.shape[0] != LB or past_avail.shape[0] != LB:
                continue

            roll_mean_all = float(np.mean(past_sales)) if LB > 0 else 0.0
            roll_mean_unc = float(np.sum(past_sales * past_avail) / max(1, np.sum(past_avail))) if LB > 0 else 0.0
            unc_frac = float(np.mean(past_avail)) if LB > 0 else 1.0

            if LB > 0 and np.any(past_avail == 0):
                last_idx = np.where(past_avail == 0)[0].max()
                since_last_so = int(LB - 1 - last_idx)
            else:
                since_last_so = int(LB)

            # Time features at decision time t (e.g., Apr 8 for predicting Apr 15)
            time_vec = time_feats_df.iloc[t].values.astype(np.float32)
            static_vec = static_cats_enc.iloc[i].values.astype(np.float32)
            if tp is not None:
                tp_vec = tp[:, i, t].astype(np.float32)
            else:
                tp_vec = np.array([], dtype=np.float32)

            base_features = np.concatenate([
                past_sales, past_avail,
                np.array([roll_mean_all, roll_mean_unc, unc_frac, since_last_so], dtype=np.float32),
                time_vec, static_vec, tp_vec
            ])

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
                rows.append((i, t, h, base_features, y, split_tag))

    if len(rows) == 0:
        raise RuntimeError("No training rows constructed. Check availability flags, alignment, and split windows.")

    # Build per-horizon matrices
    rows_by_h = {h: [] for h in args.horizons}
    for rec in rows:
        rows_by_h[rec[2]].append(rec)

    models_info = []
    coverage_report = []
    split_counts = {h: {"train": 0, "dev": 0, "test": 0} for h in args.horizons}

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

        # Feature names (for schema)
        feat_names = []
        # Note: names remain 'lag_k' for compatibility; now they refer to inclusive lags (ending at t)
        feat_names += [f"sales_lag_{k}" for k in range(LB, 0, -1)]
        feat_names += [f"avail_lag_{k}" for k in range(LB, 0, -1)]
        feat_names += ["roll_mean_all", "roll_mean_unc", "unc_frac", "since_last_so"]
        feat_names += [f"time_{c}" for c in time_feat_cols]
        feat_names += [f"static_{c}" for c in static_cats_enc.columns]
        if tp is not None:
            feat_names += [f"tp_{k}" for k in range(tp.shape[0])]

        if HAS_LGBM:
            lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=feat_names, free_raw_data=False)
            lgb_val   = lgb.Dataset(X_val, label=y_val, reference=lgb_train, free_raw_data=False)

        for alpha in args.alphas:
            if not HAS_LGBM:
                raise RuntimeError("lightgbm not available in this environment: please install lightgbm")

            params = {
                "objective": "quantile",
                "alpha": alpha,
                "learning_rate": args.learning_rate,
                "num_leaves": args.num_leaves,
                "min_data_in_leaf": 20,
                "feature_fraction": 0.8,
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

        # Save feature schema
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
