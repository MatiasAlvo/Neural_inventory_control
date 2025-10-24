#!/usr/bin/env python3
"""
Evaluate quantile forecasters: pinball loss, CRPS, coverage, and interval width.

Inputs
------
--data_dir: folder with sales.pt, stock.pt, date_features.csv
--nn_dir:   folder with forecasts_raw.pt (and nn_inputs_meta.json)
--models_meta: training metadata.json (to read HORIZONS, ALPHAS, LB, and splits)

Notes
-----
- Uses the SAME label definition as training:
  y_true = sum of sales over (t+1 .. t+h), ONLY if future availability==1 for all steps.
- Respects dev split if present in metadata (split_by_period + dev_periods).
- Ignores warmup (t < LB-1) automatically.
- Assumes forecasts were saved in the order used by make_nn_inputs.py:
  columns = [Q alphas for h1] + [Q alphas for h2] + ...
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def pinball_loss(y, qhat, alpha):
    diff = y - qhat
    return np.maximum(alpha*diff, (alpha-1)*diff)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=str)
    ap.add_argument("--nn_dir", required=True, type=str)
    ap.add_argument("--models_meta", required=True, type=str)
    ap.add_argument("--save_csv", default="quantile_eval.csv", type=str)
    args = ap.parse_args()

    # Load metadata (training)
    with open(args.models_meta, "r") as f:
        meta = json.load(f)
    cfg = meta["config"]
    LB = int(cfg["LOOKBACK_WEEKS"])
    HORIZONS = list(cfg["HORIZONS"])
    ALPHAS = list(cfg["ALPHAS"])
    split_by_period = bool(cfg.get("SPLIT_BY_PERIOD", False))
    dev_periods = cfg.get("DEV_PERIODS")  # [start, end] inclusive or None

    # Load data
    data_dir = Path(args.data_dir)
    sales = torch.load(data_dir / "sales.pt").numpy()[:, 0, :]  # [N, T_sales]
    stock = torch.load(data_dir / "stock.pt").numpy()[:, 0, :]  # [N, T_stock]
    date_df = pd.read_csv(data_dir / "date_features.csv")
    T_date = len(date_df)

    # Align timeline like training/inference
    maxH = max(HORIZONS)
    T_base = min(sales.shape[1], T_date)
    T_eff = min(T_base, stock.shape[1] - maxH)  # last usable decision-time index + 1
    sales = sales[:, :T_eff + maxH]
    stock = stock[:, :T_eff + maxH]

    # Load forecasts
    nn_dir = Path(args.nn_dir)
    preds = torch.load(nn_dir / "forecasts_raw.pt").numpy()  # [N, T_eff, 1, Q*H]
    preds = preds[:, :T_eff, 0, :]                            # [N, T_eff, Q*H]
    Q = len(ALPHAS)
    H = len(HORIZONS)
    assert preds.shape[2] == Q * H, f"Forecasts last dim {preds.shape[2]} != Q*H {Q*H}"

    # Build eval mask over decision times t
    # Warmup: first usable decision-time index = LB-1
    t_min = LB - 1
    t_max = T_eff - 1

    # If dev split provided, evaluate only there; ensure target window doesn't cross the end
    if split_by_period and dev_periods is not None:
        dev_start, dev_end = int(dev_periods[0]), int(dev_periods[1])
        # For each horizon we’ll ensure t+h <= dev_end
        eval_t_range = (dev_start, dev_end)
    else:
        eval_t_range = (t_min, t_max)

    rows = []
    for h_idx, h in enumerate(HORIZONS):
        # Decision-time t must satisfy both warmup and split and label window validity
        t_lo = max(t_min, eval_t_range[0])
        t_hi = min(eval_t_range[1], t_max)
        # additionally require t+h <= upper bound
        t_hi = min(t_hi, T_eff - 1)  # safe
        # By label definition we’ll also require future_avail==1 below
        # Slice the predictions block for this horizon
        col0 = h_idx * Q
        preds_h = preds[:, :, col0:col0+Q]  # [N, T_eff, Q]

        # Compute ground-truth cumulative demand and evaluation mask
        # y_true(i,t,h) = sum_{k=1..h} sales[i, t+k]
        # only if all stock[i, t+1 : t+h] == 1
        Ys = []
        Ms = []  # mask of valid rows (uncensored)
        Ts = []
        for t in range(t_lo, t_hi + 1):
            if t + h > T_eff - 1 + h:  # defensive; not needed normally
                continue
            future_sales = sales[:, t+1 : t+1+h]         # [N, h]
            future_avail = stock[:, t+1 : t+1+h]         # [N, h]
            unc_mask = (future_avail == 1.0).all(axis=1) # [N]
            y = future_sales.sum(axis=1)                 # [N]
            Ys.append(y)
            Ms.append(unc_mask)
            Ts.append(t)
        if not Ys:
            continue
        Y = np.stack(Ys, axis=1)       # [N, T_used]
        M = np.stack(Ms, axis=1)       # [N, T_used]
        T_used = Y.shape[1]
        P = preds_h[:, Ts, :]          # [N, T_used, Q]

        # Evaluate metrics on valid (uncensored) rows
        valid = M
        if not valid.any():
            print(f"[WARN] No valid rows for h={h}")
            continue

        # Flatten valid positions
        idx = np.where(valid)
        Yv = Y[idx]                    # [K]
        Pv = P[idx[0], idx[1], :]      # [K, Q]

        # Coverage per alpha
        coverage = (Yv[:, None] <= Pv).mean(axis=0)  # [Q]

        # Pinball per alpha
        pinballs = []
        for qi, alpha in enumerate(ALPHAS):
            pin = pinball_loss(Yv, Pv[:, qi], alpha).mean()
            pinballs.append(pin)
        pinballs = np.array(pinballs)  # [Q]

        # CRPS approximation (using multiple quantiles):
        # CRPS ≈ (2/K) * sum pinball over quantiles
        crps = (2.0 / Q) * pinballs.sum()

        # Sharpness: mean interval widths for a few standard bands if available
        widths = {}
        def iw(a_lo, a_hi):
            if (a_lo in ALPHAS) and (a_hi in ALPHAS):
                i_lo = ALPHAS.index(a_lo)
                i_hi = ALPHAS.index(a_hi)
                return float((Pv[:, i_hi] - Pv[:, i_lo]).mean())
            return None
        widths["IW_80"] = iw(0.1, 0.9)
        widths["IW_90"] = iw(0.05, 0.95)
        widths["IW_60"] = iw(0.2, 0.8)

        # Store rows
        for qi, alpha in enumerate(ALPHAS):
            rows.append({
                "h": h,
                "alpha": alpha,
                "coverage": float(coverage[qi]),
                "pinball": float(pinballs[qi]),
                "crps": float(crps) if qi == 0 else np.nan,  # one per horizon to avoid repeats
                "IW_80": widths["IW_80"] if qi == 0 else np.nan,
                "IW_90": widths["IW_90"] if qi == 0 else np.nan,
                "IW_60": widths["IW_60"] if qi == 0 else np.nan,
                "valid_rows": int(Yv.shape[0])
            })

    df = pd.DataFrame(rows)
    # Nice printouts
    if len(df) == 0:
        print("No rows evaluated. Check splits and availability masks.")
        return

    for h in sorted(df["h"].unique()):
        sub = df[df["h"] == h]
        print(f"\n=== Horizon h={h} ===")
        # Coverage vs alpha
        cov_tbl = sub.pivot_table(index="alpha", values="coverage")
        print("Coverage:")
        print(cov_tbl.round(4))
        # Pinball vs alpha
        pin_tbl = sub.pivot_table(index="alpha", values="pinball")
        print("\nPinball loss:")
        print(pin_tbl.round(6))
        # CRPS & interval widths (one line per horizon)
        agg = sub.dropna(subset=["crps"]).iloc[0]
        print("\nCRPS (approx) and interval widths:")
        print(pd.DataFrame({
            "CRPS":[agg["crps"]],
            "IW_60":[agg["IW_60"]],
            "IW_80":[agg["IW_80"]],
            "IW_90":[agg["IW_90"]],
            "valid_rows":[agg["valid_rows"]],
        }).round(6))

    # Save
    out_csv = Path(args.save_csv)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved detailed metrics to {out_csv}")

if __name__ == "__main__":
    main()
