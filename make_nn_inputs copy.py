#!/usr/bin/env python3
"""
Build NN input tensor from trained LGBM quantile forecasters.

Changes in this version:
- LAG WINDOW IS INCLUSIVE OF t:
  past_sales = sales_np[i, t-LB+1 : t+1]
  past_avail = stock_np[i, t-LB+1 : t+1]
- LOOP STARTS AT t = LB-1 (instead of LB)
- Time features remain at decision time t (e.g., Apr 8 for predicting Apr 15)

Outputs:
- forecasts_raw.pt   -> float32 tensor [N, T_eff, 1, Q*H]
- (optional) forecasts_log1p.pt
- nn_inputs_meta.json

Warm-up handling for the first `lookback_weeks-1` periods:
  --warmup_pad_zeros       : fill with zeros
  --warmup_repeat_first    : repeat the first computed forecast (at t=LB-1)
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
        if m["path"].endswith(".txt"):
            booster = lgb.Booster(model_file=m["path"])
            models[(m["h"], float(m["alpha"]))] = booster
    return meta, models

def build_feature_matrix_for_t(i, t, LB, sales_np, stock_np, time_feats_df, static_cats_enc, tp, use_tp):
    # Inclusive lag window: includes index t
    past_sales = sales_np[i, t - LB + 1 : t + 1]   # shape [LB]
    past_avail = stock_np[i, t - LB + 1 : t + 1]   # shape [LB]

    # Defenses (should always hold)
    if past_sales.shape[0] != LB or past_avail.shape[0] != LB:
        raise RuntimeError(f"Inclusive lag window mismatch at t={t}: got {past_sales.shape[0]} (expected {LB})")

    roll_mean_all = float(np.mean(past_sales)) if LB > 0 else 0.0
    roll_mean_unc = float(np.sum(past_sales * past_avail) / max(1, np.sum(past_avail))) if LB > 0 else 0.0
    unc_frac = float(np.mean(past_avail)) if LB > 0 else 1.0
    if LB > 0 and np.any(past_avail == 0):
        last_idx = np.where(past_avail == 0)[0].max()
        since_last_so = int(LB - 1 - last_idx)
    else:
        since_last_so = int(LB)

    # Decision-time features (e.g., t=Apr 8 for predicting Apr 15)
    time_vec = time_feats_df.iloc[t].values.astype(np.float32)
    static_vec = static_cats_enc.iloc[i].values.astype(np.float32)
    if use_tp and tp is not None:
        tp_vec = tp[:, i, t].astype(np.float32)
    else:
        tp_vec = np.array([], dtype=np.float32)

    feat = np.concatenate([
        past_sales, past_avail,
        np.array([roll_mean_all, roll_mean_unc, unc_frac, since_last_so], dtype=np.float32),
        time_vec, static_vec, tp_vec
    ])
    return feat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=str,
                    help="Folder with sales.pt, stock.pt, product_features.csv, date_features.csv, (optional) time_product_features.pt")
    ap.add_argument("--models_meta", required=True, type=str,
                    help="Path to models/metadata.json from training")
    ap.add_argument("--out_dir", required=True, type=str, help="Output folder for NN inputs")
    ap.add_argument("--nonnegative_clip", action="store_true", help="Clamp predictions to >=0")
    ap.add_argument("--log1p_copy", action="store_true", help="Also save log1p(preds)")
    ap.add_argument("--warmup_pad_zeros", action="store_true", help="Fill warm-up periods 0..LB-2 with zeros")
    ap.add_argument("--warmup_repeat_first", action="store_true", help="Copy forecast at t=LB-1 back into 0..LB-2")
    args = ap.parse_args()

    if args.warmup_pad_zeros and args.warmup_repeat_first:
        raise ValueError("Choose only one warm-up option: --warmup_pad_zeros OR --warmup_repeat_first")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load meta and boosters
    meta, boosters = load_boosters(args.models_meta)
    cfg = meta["config"]
    LB = cfg["LOOKBACK_WEEKS"]
    HORIZONS = cfg["HORIZONS"]
    ALPHAS = cfg["ALPHAS"]
    use_tp = cfg.get("USE_TIME_PRODUCT_FEATURES", False)

    # Load data
    data_dir = Path(args.data_dir)
    sales = torch.load(data_dir / "sales.pt")          # [N, 1, T_sales]
    stock = torch.load(data_dir / "stock.pt")          # [N, 1, T_stock]
    product_df = pd.read_csv(data_dir / "product_features.csv")
    date_df = pd.read_csv(data_dir / "date_features.csv")

    tp_path = data_dir / "time_product_features.pt"
    has_tp = tp_path.exists()
    time_product_features = torch.load(tp_path) if has_tp else None

    N, _, T_sales = sales.shape
    _, _, T_stock = stock.shape
    T_date = len(date_df)
    maxH = max(HORIZONS)

    # Align timeline like training (stop time features at last observed demand date)
    T_base = min(T_sales, T_date)
    T_eff = min(T_base, T_stock - maxH)
    if T_eff < LB:
        raise ValueError(f"Not enough timeline to build features (inclusive lags): T_eff={T_eff}, lookback={LB}")

    sales_np = sales.numpy()[:, 0, :T_eff + maxH]
    stock_np = stock.numpy()[:, 0, :T_eff + maxH]
    date_df_eff = date_df.iloc[:T_eff].reset_index(drop=True)  # e.g., ends at 2024-04-08

    if has_tp and use_tp:
        tp = time_product_features.numpy()[:, :, 0, :T_eff + maxH]  # [F_tp, N, T_eff+maxH]
    else:
        tp = None

    # Ordinal-encode static features (same columns & row order as training)
    from sklearn.preprocessing import OrdinalEncoder
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    static_cats_enc = pd.DataFrame(
        enc.fit_transform(product_df[list(product_df.columns)]),
        columns=[f"cat_{c}" for c in product_df.columns]
    )

    # Time features (drop any 'date' column)
    time_feats_df = date_df_eff.drop(columns=[c for c in ["date"] if c in date_df_eff.columns]).reset_index(drop=True)

    # Output over ALL periods 0..T_eff-1 (warm-up filled per flag)
    periods = T_eff
    Q = len(ALPHAS)
    H = len(HORIZONS)
    out = np.zeros((N, periods, Q * H), dtype=np.float32)

    # Build forecasts
    for t in range(periods):
        # Warm-up now is 0 .. LB-2 because first usable decision time is t=LB-1
        if t < (LB - 1):
            if args.warmup_pad_zeros:
                out[:, t, :] = 0.0
            continue

        # Build batch features for ALL items at decision time t
        feats = [build_feature_matrix_for_t(i, t, LB, sales_np, stock_np, time_feats_df, static_cats_enc, tp, use_tp)
                 for i in range(N)]
        X = np.stack(feats, axis=0)  # [N, F]

        col_offset = 0
        for h in HORIZONS:
            preds = []
            for a in ALPHAS:
                booster = boosters.get((h, float(a)))
                if booster is None:
                    raise RuntimeError(f"No booster found for (h={h}, alpha={a})")
                p = booster.predict(X)
                preds.append(p)
            P = np.stack(preds, axis=1)   # [N, Q] in ALPHAS order
            P.sort(axis=1)                # enforce monotone quantiles
            if args.nonnegative_clip:
                P = np.clip(P, 0.0, None)
            out[:, t, col_offset:col_offset + Q] = P.astype(np.float32)
            col_offset += Q

    # If requested, repeat first computed forecast back into warm-up span
    if args.warmup_repeat_first and LB > 1:
        first_forecast = out[:, LB - 1, :].copy()
        for t in range(LB - 1):
            out[:, t, :] = first_forecast

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
        "periods_out": int(periods),
        "t_start_index": 0,
        "nonnegative_clip": bool(args.nonnegative_clip),
        "log1p_saved": bool(args.log1p_copy),
        "warmup_pad_zeros": bool(args.warmup_pad_zeros),
        "warmup_repeat_first": bool(args.warmup_repeat_first),
        "lag_inclusive": True
    }
    with open(Path(args.out_dir) / "nn_inputs_meta.json", "w") as f:
        json.dump(nn_meta, f, indent=2)

    print("Saved:", raw_path)
    if args.log1p_copy:
        print("Saved:", Path(args.out_dir) / "forecasts_log1p.pt")
    print("Meta:", Path(args.out_dir) / "nn_inputs_meta.json")

if __name__ == "__main__":
    main()
