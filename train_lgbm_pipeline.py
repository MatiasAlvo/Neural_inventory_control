#!/usr/bin/env python3
"""
Train LGBM quantile models for cumulative (protection-time) demand.

This version ADDS engineered features while preserving your data rules:
- Exclusive lag window up to decision time t-1 (lags: [t-LB .. t-1])
- Future window (t .. t+h-1) must be fully uncensored (avail==1)
- No leakage: only lookback + decision-time calendar used

New anti-tail-bias options:
  • weeks_to_end feature (distance to series end)
  • Temporal balancing weights (equalize per-week contribution)
  • IPCW weights via a censoring model per horizon

Added features (per row):
  • sales×avail elementwise (LB dims)
  • EWMAs (fast/slow), OLS slope (trend), variance/CV, burstiness (p95-p50)/p50
  • zero-sales counts (all vs in-stock), streaks (in-stock / stockout)
  • availability volatility & switches, tightness flag at t-1
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
    # require t+h-1 <= end (so target window is fully inside split)
    if (t + h - 1) > end:
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
    p.add_argument("--add_weeks_to_end_feature", type=lambda s: s.lower() in ["1","true","yes","y"], default=True,
                   help="Append (T_eff-1 - t) as a feature to de-bias the tail.")

    # anti-tail-bias weighting
    p.add_argument("--temporal_balance_weights", type=lambda s: s.lower() in ["1","true","yes","y"], default=False,
                   help="Reweight rows inversely by #rows in their decision week t.")
    p.add_argument("--ipcw_weights", type=lambda s: s.lower() in ["1","true","yes","y"], default=False,
                   help="Use inverse probability of being uncensored (per-horizon).")
    p.add_argument("--ipcw_cap", type=float, default=50.0, help="Cap for IPCW weights to avoid extremes.")
    p.add_argument("--ipcw_min_p", type=float, default=1e-3, help="Floor for censor prob before inversion.")

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
    rows = []               # uncensored training rows
    rows_by_h = {h: [] for h in args.horizons}

    # For IPCW: we also need a censoring dataset per horizon (ALL rows, censored or not)
    censor_X_by_h = {h: [] for h in args.horizons}
    censor_y_by_h = {h: [] for h in args.horizons}
    censor_it_by_h = {h: [] for h in args.horizons}  # (i,t) keys
    # Parse explicit splits (inclusive indices), if requested
    train_split = parse_periods(args.train_periods) if args.split_by_period else None
    dev_split   = parse_periods(args.dev_periods)   if args.split_by_period else None
    test_split  = parse_periods(args.test_periods)  if args.split_by_period else None

    # -----------------------------
    # Build supervised rows (+ censoring rows)
    # -----------------------------
    for i in range(N):
        s = sales_np[i]
        a = stock_np[i]

        for t in range(LB, T_eff):  # decision time
            # Exclusive lag window [t-LB : t]
            past_sales = s[t - LB: t].astype(np.float32)
            past_avail = a[t - LB: t].astype(np.float32)
            if past_sales.shape[0] != LB or past_avail.shape[0] != LB:
                continue

            eps = 1e-6
            roll_mean_all = float(np.mean(past_sales)) if LB > 0 else 0.0
            mask_unc = (past_avail == 1.0)
            if mask_unc.any():
                roll_mean_unc = float(np.sum(past_sales[mask_unc]) / max(1.0, np.sum(mask_unc)))
            else:
                roll_mean_unc = 0.0
            unc_frac = float(np.mean(past_avail)) if LB > 0 else 1.0

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
                        v = float(xi); first = False
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

            # Streaks up to t-1
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

            # Tightness at t-1
            tight_flag = float((pa[-1] == 1.0) and (ps[-1] >= 0.8 * (mean_unc if mask_unc.any() else ps.mean())))

            # Interactions & transforms
            sales_x_avail = (ps * pa).astype(np.float32)
            log1p_sales   = np.log1p(ps).astype(np.float32)

            # Fourier by t index (decision time only)
            week_idx = float(t % 52)
            s1 = np.sin(2*np.pi*week_idx/52.0); c1 = np.cos(2*np.pi*week_idx/52.0)
            s2 = np.sin(2*np.pi*week_idx/26.0); c2 = np.cos(2*np.pi*week_idx/26.0)

            # Week-of-month encodings at t
            wom_num  = float(wom_idx[t])
            wom_sin_t = float(wom_sin[t])
            wom_cos_t = float(wom_cos[t])

            # Decision-time features
            time_vec = time_feats_df.iloc[t].values.astype(np.float32)
            static_vec = static_cats_enc.iloc[i].values.astype(np.float32)
            if tp is not None:
                tp_vec = tp[:, i, t].astype(np.float32)
            else:
                tp_vec = np.array([], dtype=np.float32)

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
                ps,                         # LB
                pa,                         # LB
                sales_x_avail,              # LB
                log1p_sales,                # LB
                np.array([roll_mean_all, roll_mean_unc, unc_frac, since_last_so], dtype=np.float32),
                engineered,
                time_vec, static_vec, tp_vec
            ]

            # NEW: weeks_to_end feature
            if args.add_weeks_to_end_feature:
                weeks_to_end = np.array([float(T_eff - 1 - t)], dtype=np.float32)

            for h in args.horizons:
                # For censoring model (uses same features, plus optional h, plus weeks_to_end if enabled)
                censor_x_parts = list(base_prefix)
                if args.add_weeks_to_end_feature:
                    censor_x_parts.append(weeks_to_end)
                if args.include_h_as_feature:
                    censor_x_parts.append(np.array([float(h)], dtype=np.float32))
                censor_X = np.concatenate(censor_x_parts)

                # Determine uncensored indicator for this horizon (t .. t+h-1)
                future_avail = a[t: t + h]
                is_unc = (future_avail.shape[0] == h) and np.all(future_avail == 1.0)
                censor_X_by_h[h].append(censor_X)
                censor_y_by_h[h].append(1.0 if is_unc else 0.0)
                censor_it_by_h[h].append((i, t))

                # Only build label row when uncensored
                if not is_unc:
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

                # cumulative future demand for t .. t+h-1
                y = float(np.sum(s[t : t + h]))

                # Final X row
                x_parts = list(base_prefix)
                if args.add_weeks_to_end_feature:
                    x_parts.append(weeks_to_end)
                if args.include_h_as_feature:
                    x_parts.append(np.array([float(h)], dtype=np.float32))
                Xrow = np.concatenate(x_parts)

                rec = (i, t, h, Xrow, y, split_tag)
                rows.append(rec)
                rows_by_h[h].append(rec)

    if len(rows) == 0:
        raise RuntimeError("No training rows constructed. Check availability flags, alignment, and split windows.")

    models_info = []
    coverage_report = []
    split_counts = {h: {"train": 0, "dev": 0, "test": 0} for h in args.horizons}

    # ---------- Feature names schema (mirror the construction order) ----------
    def build_feat_names(LB, time_feat_cols, static_cats_enc, tp, include_h, add_w2e):
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
        if add_w2e:
            names += ["weeks_to_end"]
        if include_h:
            names += ["horizon"]
        return names

    base_feat_names = build_feat_names(LB, time_feat_cols, static_cats_enc, tp,
                                       args.include_h_as_feature, args.add_weeks_to_end_feature)

    # ------------- Optional IPCW: fit censoring models per horizon -------------
    # We will get p_hat[(i,t,h)] = P(uncensored | X), then weight quantile rows by 1/p_hat (capped)
    p_hat_by_h = {}
    if args.ipcw_weights:
        if not HAS_LGBM:
            raise RuntimeError(f"lightgbm not available for IPCW: {LGBM_IMPORT_ERROR}")
        print("[IPCW] Fitting censoring models per horizon...")
        for h in args.horizons:
            Xc = np.stack(censor_X_by_h[h], axis=0).astype(np.float32)
            yc = np.array(censor_y_by_h[h], dtype=np.float32)
            it_keys = censor_it_by_h[h]

            # simple LGBM classifier
            c_params = {
                "objective": "binary",
                "learning_rate": 0.05,
                "num_leaves": 63,
                "min_data_in_leaf": 25,
                "feature_fraction": 0.85,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "metric": "auc",
                "verbosity": -1,
                "force_col_wise": True,
                "num_threads": args.num_threads,
                "seed": args.random_state,
            }
            # No strict split here; we just need probabilities
            dtrain = lgb.Dataset(Xc, label=yc, feature_name=base_feat_names, free_raw_data=False)
            c_booster = lgb.train(c_params, dtrain, num_boost_round=400)

            p_all = c_booster.predict(Xc)  # probabilities
            # map to dict for quick lookup
            p_map = {}
            for k, p_val in zip(it_keys, p_all):
                # k = (i,t)
                p_map[k] = float(p_val)
            p_hat_by_h[h] = p_map
        print("[IPCW] Done.")

    # ---------------- Train quantile models per horizon ----------------
    for h in args.horizons:
        recs = rows_by_h[h]
        if len(recs) == 0:
            continue

        X = np.stack([r[3] for r in recs], axis=0).astype(np.float32)
        y = np.array([r[4] for r in recs], dtype=np.float32)
        t_idx = np.array([r[1] for r in recs], dtype=np.int32)
        tags = np.array([r[5] for r in recs], dtype=object)
        i_idx = np.array([r[0] for r in recs], dtype=np.int32)

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
        t_train          = t_idx[train_mask]
        i_train          = i_idx[train_mask]

        # Build training weights
        train_w = np.ones_like(y_train, dtype=np.float32)

        # Temporal balancing (equalize per-week contribution)
        if args.temporal_balance_weights:
            counts_by_t = np.bincount(t_train, minlength=int(T_eff))
            counts_by_t[counts_by_t == 0] = 1
            w_time = 1.0 / counts_by_t[t_train]
            w_time *= (len(w_time) / w_time.sum())  # normalize to mean 1
            train_w *= w_time.astype(np.float32)

        # IPCW (1 / P(uncensored | X)) per horizon
        if args.ipcw_weights:
            p_map = p_hat_by_h[h]
            p_list = []
            for ii, tt in zip(i_train, t_train):
                p = p_map.get((int(ii), int(tt)), 1.0)
                p_list.append(p)
            p_arr = np.clip(np.array(p_list, dtype=np.float32), args.ipcw_min_p, 1.0)
            w_ipcw = 1.0 / p_arr
            w_ipcw = np.clip(w_ipcw, 1.0, args.ipcw_cap)
            train_w *= w_ipcw

        # Feature names (same for all horizons)
        feat_names = list(base_feat_names)

        if HAS_LGBM:
            import lightgbm as lgb
            lgb_train = lgb.Dataset(X_train, label=y_train, weight=train_w, feature_name=feat_names, free_raw_data=False)
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
            "HAS_TRUE_WOM": bool("date" in date_df.columns),
            "TARGET_STARTS_AT_T": True,
            "LAG_WINDOW_INCLUSIVE_T": False,
            "ADD_WEEKS_TO_END_FEATURE": bool(args.add_weeks_to_end_feature),
            "TEMPORAL_BALANCE_WEIGHTS": bool(args.temporal_balance_weights),
            "IPCW_WEIGHTS": bool(args.ipcw_weights),
            "IPCW_CAP": float(args.ipcw_cap),
            "IPCW_MIN_P": float(args.ipcw_min_p),
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
