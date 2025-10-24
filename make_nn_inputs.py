#!/usr/bin/env python3
"""
Build NN input tensor from trained LGBM quantile forecasters.

This version mirrors the UPDATED training features exactly:
- Exclusive lag window up to decision time t-1  (lags: [t-LB .. t-1])
- Engineered features added: sales×avail (LB), log1p lags (LB),
  EWMA (fast/slow), OLS slope, var/CV, burstiness, zero counts,
  streaks (in-stock / stockout), availability std & switches,
  tightness flag, Fourier (P=52,26), week-of-month numeric + sin/cos
- Optional features to match training flags:
    • "weeks_to_end" if ADD_WEEKS_TO_END_FEATURE=true
    • "horizon"      if INCLUDE_H_AS_FEATURE=true
- Reuses saved OrdinalEncoder mapping for static product features

Outputs:
- forecasts_raw.pt   -> float32 tensor [N, periods, 1, Q*H]
- (optional) forecasts_log1p.pt
- nn_inputs_meta.json

Warm-up handling with exclusive lags:
  usable t starts at LB (since lags need [t-LB .. t-1])
  --warmup_pad_zeros       : fill t < LB with zeros
  --warmup_repeat_first    : copy forecast at t=LB into 0..LB-1

Extra out-of-sample row:
- If date_features.csv has one more row beyond T_eff, we forecast for that
  next decision time using history through t-1.
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def load_boosters(meta_path):
    import lightgbm as lgb
    with open(meta_path, "r") as f:
        meta = json.load(f)
    models = {}
    for m in meta["models"]:
        if m.get("path", "").endswith(".txt"):
            models[(m["h"], float(m["alpha"]))] = lgb.Booster(model_file=m["path"])
    return meta, models

def compute_week_of_month_arrays(date_df_eff):
    """Return wom_idx (1..5 or 1..4 fallback), wom_sin, wom_cos arrays aligned with t."""
    if "date" in date_df_eff.columns:
        try:
            dates = pd.to_datetime(date_df_eff["date"])
            wom = []
            for dt in dates:
                wom.append(int(1 + (dt.day - 1) // 7))  # 1..5
            wom = np.array(wom, dtype=np.int32)
            P = 5.0
            wom_sin = np.sin(2*np.pi*(wom-1)/P).astype(np.float32)
            wom_cos = np.cos(2*np.pi*(wom-1)/P).astype(np.float32)
            return wom, wom_sin, wom_cos
        except Exception:
            pass
    # Fallback: 4-week cycle on index t
    T_eff = len(date_df_eff)
    t_arr = np.arange(T_eff, dtype=np.int32)
    wom = (t_arr % 4) + 1
    P = 4.0
    wom_sin = np.sin(2*np.pi*(wom-1)/P).astype(np.float32)
    wom_cos = np.cos(2*np.pi*(wom-1)/P).astype(np.float32)
    return wom, wom_sin, wom_cos

def load_static_encoder(models_dir, product_df):
    """
    Load the training-time OrdinalEncoder mapping if available.
    If missing/unreadable, fall back to fitting a new encoder.
    """
    enc_path = Path(models_dir) / "static_cats_ordinal_encoder.json"
    from sklearn.preprocessing import OrdinalEncoder
    try:
        if enc_path.exists():
            with open(enc_path, "r") as f:
                blob = json.load(f)
            cols = blob["columns"]
            cats = blob["categories"]
            df = product_df[cols].copy()
            enc = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                categories=cats
            )
            X = enc.fit_transform(df.values)  # fit with fixed categories
            return pd.DataFrame(X, columns=[f"cat_{c}" for c in cols])
    except Exception as e:
        print(f"[WARN] Failed to read saved encoder at {enc_path}: {e}\n"
              f"       Falling back to fitting a new OrdinalEncoder on product_features.csv")

    # Fallback — fit new encoder on current product_df order
    cols_now = list(product_df.columns)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X = enc.fit_transform(product_df[cols_now])
    return pd.DataFrame(X, columns=[f"cat_{c}" for c in cols_now])

def ewma_last(x, alpha):
    v = 0.0
    first = True
    for xi in x:
        xi = float(xi)
        if first:
            v = xi
            first = False
        else:
            v = alpha*xi + (1.0-alpha)*v
    return float(v)

def build_base_features_for_t(i, t, LB, sales_np, stock_np,
                              time_feats_df, static_cats_enc, tp, use_tp,
                              wom_idx, wom_sin, wom_cos,
                              include_w2e, T_eff):
    """
    Rebuild the exact per-(i,t) feature vector used in UPDATED training,
    EXCLUDING the optional horizon feature (added later per-h).

    Uses exclusive lags: past window = [t-LB : t] (last index is t-1).
    Appends weeks_to_end (if enabled) just before the optional horizon,
    matching the trainer's feature order.
    """
    eps = 1e-6
    # Exclusive lag window [t-LB : t]
    ps = sales_np[i, t - LB : t].astype(np.float32)  # [LB]
    pa = stock_np[i, t - LB : t].astype(np.float32)  # [LB]
    if ps.shape[0] != LB or pa.shape[0] != LB:
        raise RuntimeError(f"Lag window mismatch at t={t}: got {ps.shape[0]}, {pa.shape[0]} (LB={LB})")

    # Basic rollups
    roll_mean_all = float(np.mean(ps)) if LB > 0 else 0.0
    mask_unc = (pa == 1.0)
    if mask_unc.any():
        roll_mean_unc = float(np.sum(ps[mask_unc]) / max(1.0, np.sum(mask_unc)))
    else:
        roll_mean_unc = 0.0
    unc_frac = float(np.mean(pa)) if LB > 0 else 1.0
    if LB > 0 and np.any(pa == 0):
        last_idx = np.where(pa == 0)[0].max()
        since_last_so = int(LB - 1 - last_idx)
    else:
        since_last_so = int(LB)

    # Engineered from lookback
    ewma_fast = ewma_last(ps, 0.5)
    ewma_slow = ewma_last(ps, 0.2)
    x_idx = np.arange(LB, dtype=np.float32)
    x_mean = float(x_idx.mean()); y_mean = float(ps.mean())
    den = float(((x_idx - x_mean)**2).sum() + eps)
    slope = float(((x_idx - x_mean)*(ps - y_mean)).sum() / den)

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
        zeros_all = float((ps == 0).sum()); zeros_unc = 0.0

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

    # Tightness at t-1: in stock AND last sale near mean_unc
    tight_flag = float((pa[-1] == 1.0) and (ps[-1] >= 0.8 * (mean_unc if mask_unc.any() else ps.mean())))

    # Vector interactions & transforms
    sales_x_avail = (ps * pa).astype(np.float32)          # [LB]
    log1p_sales   = np.log1p(ps).astype(np.float32)       # [LB]

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
    if use_tp and tp is not None:
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

    parts = [
        ps,                  # sales_lag_*
        pa,                  # avail_lag_*
        sales_x_avail,       # sales_x_avail_*
        log1p_sales,         # log1p_sales_lag_*
        np.array([roll_mean_all, roll_mean_unc, unc_frac, since_last_so], dtype=np.float32),
        engineered,
        time_vec, static_vec, tp_vec
    ]

    # Append weeks_to_end exactly where the trainer put it (before horizon)
    if include_w2e:
        # Use the same T_eff definition as in training and clamp at 0 for any extra row
        weeks_to_end = max(0.0, float(T_eff - 1 - t))
        parts.append(np.array([weeks_to_end], dtype=np.float32))

    base = np.concatenate(parts)
    return base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=str)
    ap.add_argument("--models_meta", required=True, type=str, help="path to models/metadata.json")
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--nonnegative_clip", action="store_true")
    ap.add_argument("--log1p_copy", action="store_true")
    ap.add_argument("--warmup_pad_zeros", action="store_true")
    ap.add_argument("--warmup_repeat_first", action="store_true")
    args = ap.parse_args()

    if args.warmup_pad_zeros and args.warmup_repeat_first:
        raise ValueError("Use only one warm-up option: --warmup_pad_zeros OR --warmup_repeat_first")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load boosters + meta
    meta, boosters = load_boosters(args.models_meta)
    cfg = meta["config"]
    LB         = cfg["LOOKBACK_WEEKS"]
    HORIZONS   = cfg["HORIZONS"]
    ALPHAS     = cfg["ALPHAS"]
    use_tp     = cfg.get("USE_TIME_PRODUCT_FEATURES", False)
    include_h  = cfg.get("INCLUDE_H_AS_FEATURE", False)
    include_w2e = cfg.get("ADD_WEEKS_TO_END_FEATURE", False)

    # Load data
    data_dir = Path(args.data_dir)
    sales = torch.load(data_dir / "sales.pt")        # [N, 1, T_sales]
    stock = torch.load(data_dir / "stock.pt")        # [N, 1, T_stock]
    product_df = pd.read_csv(data_dir / "product_features.csv")
    date_df = pd.read_csv(data_dir / "date_features.csv")

    tp_path = data_dir / "time_product_features.pt"
    has_tp = tp_path.exists()
    time_product_features = torch.load(tp_path) if has_tp else None

    N, _, T_sales = sales.shape
    _, _, T_stock = stock.shape
    T_date = len(date_df)
    maxH = max(HORIZONS)

    # Align timeline consistent with training for in-sample last decision time
    T_base = min(T_sales, T_date)
    T_eff = min(T_base, T_stock - maxH)  # last in-sample decision-time index + 1

    if T_eff < LB:
        raise ValueError(f"Not enough timeline: T_eff={T_eff}, lookback={LB}, maxH={maxH}")

    # We will forecast one extra decision time (t = T_eff) if the calendar has it
    periods = T_eff + 1 if T_date > T_eff else T_eff

    # Truncate/prepare arrays to cover lags up to T_eff and features up to 'periods'
    sales_np = sales.numpy()[:, 0, :T_eff + maxH]
    stock_np = stock.numpy()[:, 0, :T_eff + maxH]
    date_df_eff = date_df.iloc[:periods].reset_index(drop=True)  # include the extra calendar row if present

    # Time-product features
    tp = None
    if has_tp and use_tp:
        tp = time_product_features.numpy()[:, :, 0, :max(periods, T_eff) + maxH]

    # Static encoder mapping (load training-time categories if available)
    models_dir = Path(args.models_meta).parent
    static_cats_enc = load_static_encoder(models_dir, product_df)

    # Time features (drop 'date'), like training, aligned to 'periods'
    time_feats_df = date_df_eff.drop(columns=[c for c in ["date"] if c in date_df_eff.columns]).reset_index(drop=True)

    # Week-of-month arrays aligned with t (length = periods)
    wom_idx, wom_sin, wom_cos = compute_week_of_month_arrays(date_df_eff)

    # Output tensor
    Q = len(ALPHAS)
    H = len(HORIZONS)
    out = np.zeros((N, periods, Q * H), dtype=np.float32)

    # Build forecasts for each decision time t
    for t in range(periods):
        # With exclusive lags, the first usable decision time is t = LB
        if t < LB:
            if args.warmup_pad_zeros:
                out[:, t, :] = 0.0
            continue

        # Precompute base features (without horizon) for all items at t
        base_feats = [build_base_features_for_t(
                          i, t, LB, sales_np, stock_np, time_feats_df, static_cats_enc,
                          tp, use_tp, wom_idx, wom_sin, wom_cos,
                          include_w2e, T_eff)
                      for i in range(N)]
        X_base = np.stack(base_feats, axis=0)  # [N, F_base]

        col_offset = 0
        for h in HORIZONS:
            # Optionally append horizon as a feature
            if include_h:
                h_col = np.full((N, 1), float(h), dtype=np.float32)
                X = np.concatenate([X_base, h_col], axis=1)
            else:
                X = X_base

            preds = []
            for a in ALPHAS:
                booster = boosters.get((h, float(a)))
                if booster is None:
                    raise RuntimeError(f"Missing booster for (h={h}, alpha={a})")
                p = booster.predict(X)
                preds.append(p)
            P = np.stack(preds, axis=1)  # [N, Q] in ALPHAS order
            # Enforce non-crossing
            P.sort(axis=1)
            if args.nonnegative_clip:
                P = np.clip(P, 0.0, None)

            out[:, t, col_offset:col_offset + Q] = P.astype(np.float32)
            col_offset += Q

    # Warm-up repeat-first option: copy forecast at t=LB into 0..LB-1
    if args.warmup_repeat_first and LB > 0:
        first_forecast = out[:, LB, :].copy()
        for tt in range(LB):
            out[:, tt, :] = first_forecast

    # Save tensors
    out_tensor = torch.from_numpy(out).unsqueeze(2)  # [N, periods, 1, Q*H]
    raw_path = Path(args.out_dir) / "forecasts_raw.pt"
    torch.save(out_tensor, raw_path)

    if args.log1p_copy:
        log_tensor = torch.log1p(out_tensor)
        torch.save(log_tensor, Path(args.out_dir) / "forecasts_log1p.pt")

    # Meta
    nn_meta = {
        "shape": [int(out_tensor.shape[0]), int(out_tensor.shape[1]), int(out_tensor.shape[2]), int(out_tensor.shape[3])],
        "alphas": ALPHAS,
        "horizons": HORIZONS,
        "lookback_weeks": LB,
        "periods_out": int(out_tensor.shape[1]),
        "t_start_index": 0,
        "nonnegative_clip": bool(args.nonnegative_clip),
        "log1p_saved": bool(args.log1p_copy),
        "warmup_pad_zeros": bool(args.warmup_pad_zeros),
        "warmup_repeat_first": bool(args.warmup_repeat_first),
        "lag_inclusive": False,
        "include_h_as_feature": bool(include_h),
        "include_weeks_to_end": bool(include_w2e)
    }
    with open(Path(args.out_dir) / "nn_inputs_meta.json", "w") as f:
        json.dump(nn_meta, f, indent=2)

    print("Saved:", raw_path)
    if args.log1p_copy:
        print("Saved:", Path(args.out_dir) / "forecasts_log1p.pt")
    print("Meta:", Path(args.out_dir) / "nn_inputs_meta.json")

if __name__ == "__main__":
    main()
