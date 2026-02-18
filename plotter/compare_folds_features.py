#!/usr/bin/env python3
"""
Compare feature distributions across CV folds as a sanity check.

Example:
  python MVA_training/VBF_new/compare_folds_features.py \
    --data-dir dnn/trained_models/kfold_shuffleFalse_killInf/2022postEE_h-peak_vbf \
    --out-dir  dnn/trained_models/kfold_shuffleFalse_killInf/2022postEE_h-peak_vbf/fold_sanity \
    --folds 0,1,2,3 \
    --split train \
    --weight-col weight \
    --label-col label

Notes:
- Expects files like data_df_train_{fold}.parquet etc.
- Uses ABS(weights) for weighted stats/KS (HEP-stable default).
- Produces one PDF per feature with fold overlays.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def safe_abs_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    w = np.abs(w)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    if np.sum(w) <= 0:
        w = np.ones_like(w, dtype=np.float64)
    return w


def weighted_mean_std(x: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    w = safe_abs_weights(w)
    ws = np.sum(w)
    if ws <= 0:
        return float(np.mean(x)), float(np.std(x))
    mu = float(np.sum(w * x) / ws)
    var = float(np.sum(w * (x - mu) ** 2) / ws)
    return mu, float(math.sqrt(max(var, 0.0)))


def weighted_ks_distance(
    a: np.ndarray, wa: np.ndarray, b: np.ndarray, wb: np.ndarray
) -> float:
    """
    Weighted KS distance between two 1D samples.
    Uses ABS weights and compares step CDFs on the union grid.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    wa = safe_abs_weights(wa)
    wb = safe_abs_weights(wb)

    ma = np.isfinite(a) & np.isfinite(wa)
    mb = np.isfinite(b) & np.isfinite(wb)
    a, wa = a[ma], wa[ma]
    b, wb = b[mb], wb[mb]

    if a.size == 0 or b.size == 0:
        return float("nan")

    sa = np.argsort(a)
    sb = np.argsort(b)
    a_sorted, wa_sorted = a[sa], wa[sa]
    b_sorted, wb_sorted = b[sb], wb[sb]

    cdfa = np.cumsum(wa_sorted) / (np.sum(wa_sorted) + 1e-12)
    cdfb = np.cumsum(wb_sorted) / (np.sum(wb_sorted) + 1e-12)

    grid = np.unique(np.concatenate([a_sorted, b_sorted]))

    def step_cdf(x_sorted, cdf_sorted, x_grid):
        idx = np.searchsorted(x_sorted, x_grid, side="right") - 1
        idx = np.clip(idx, -1, len(cdf_sorted) - 1)
        out = np.zeros_like(x_grid, dtype=np.float64)
        m = idx >= 0
        out[m] = cdf_sorted[idx[m]]
        return out

    Ca = step_cdf(a_sorted, cdfa, grid)
    Cb = step_cdf(b_sorted, cdfb, grid)
    return float(np.max(np.abs(Ca - Cb)))


def robust_range(
    x: np.ndarray, lo_q: float = 0.005, hi_q: float = 0.995
) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (0.0, 1.0)
    lo = float(np.quantile(x, lo_q))
    hi = float(np.quantile(x, hi_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.min(x)), float(np.max(x))
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
    return lo, hi


# -----------------------------
# Core
# -----------------------------
def load_fold_df(data_dir: Path, split: str, fold: int) -> pd.DataFrame:
    p = data_dir / f"data_df_{split}_{fold}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    return pd.read_parquet(p)


def infer_features(
    df: pd.DataFrame, weight_col: str, label_col: str, drop_cols: List[str]
) -> List[str]:
    drop = set(drop_cols + [weight_col, label_col])
    feats = [c for c in df.columns if c not in drop]
    # keep only numeric-ish columns
    out = []
    for c in feats:
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def compare_folds(
    data_dir: Path,
    out_dir: Path,
    folds: List[int],
    split: str,
    weight_col: str,
    label_col: str,
    features: List[str] | None,
    bins: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load all folds
    dfs: Dict[int, pd.DataFrame] = {}
    for f in folds:
        df = load_fold_df(data_dir, split=split, fold=f)
        dfs[f] = df

    # Infer features from first fold if not provided
    if features is None:
        drop_cols = ["event", "process", "process_group", "_fold_id"]
        features = infer_features(dfs[folds[0]], weight_col, label_col, drop_cols)

    # Sanity report
    report_lines = []
    report_lines.append(f"data_dir: {data_dir}")
    report_lines.append(f"split: {split}")
    report_lines.append(f"folds: {folds}")
    report_lines.append(f"n_features: {len(features)}")
    report_lines.append("")

    # Collect per-fold stats per feature
    rows = []
    for f in folds:
        df = dfs[f]
        if weight_col not in df.columns:
            raise KeyError(f"Missing weight_col='{weight_col}' in fold {f}")
        w = df[weight_col].to_numpy(dtype=np.float64, copy=False)

        for feat in features:
            if feat not in df.columns:
                rows.append({"fold": f, "feature": feat, "status": "MISSING_COLUMN"})
                continue

            x = pd.to_numeric(df[feat], errors="coerce").to_numpy(
                dtype=np.float64, copy=False
            )

            n = int(x.size)
            n_nan = int(np.isnan(x).sum())
            n_inf = int(np.isinf(x).sum())
            n_bad = int((~np.isfinite(x)).sum())

            x_finite = x[np.isfinite(x)]
            w_finite = w[np.isfinite(x)]

            if x_finite.size == 0:
                rows.append(
                    {
                        "fold": f,
                        "feature": feat,
                        "status": "ALL_NONFINITE",
                        "n": n,
                        "n_bad": n_bad,
                        "n_nan": n_nan,
                        "n_inf": n_inf,
                    }
                )
                continue

            mu, sig = float(np.mean(x_finite)), float(np.std(x_finite))
            wmu, wsig = weighted_mean_std(x_finite, w_finite)

            rows.append(
                {
                    "fold": f,
                    "feature": feat,
                    "status": "OK" if n_bad == 0 else "HAS_NONFINITE",
                    "n": n,
                    "n_bad": n_bad,
                    "n_nan": n_nan,
                    "n_inf": n_inf,
                    "min": float(np.min(x_finite)),
                    "max": float(np.max(x_finite)),
                    "mean": mu,
                    "std": sig,
                    "wmean_absw": wmu,
                    "wstd_absw": wsig,
                    "sumw": float(np.sum(w)),
                    "sum_absw": float(np.sum(np.abs(w))),
                }
            )

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(out_dir / "feature_summary.csv", index=False)

    # Flag bad features
    bad = df_summary[(df_summary["status"] != "OK") | (df_summary.get("n_bad", 0) > 0)]
    if len(bad) > 0:
        report_lines.append("Features with NONFINITE or missing columns were found:")
        report_lines.append(
            str(bad[["fold", "feature", "status", "n_bad", "n_nan", "n_inf"]].head(50))
        )
        report_lines.append("")

    # Pairwise KS per feature
    ks_rows = []
    for feat in features:
        # compute on pairs
        for i in range(len(folds)):
            for j in range(i + 1, len(folds)):
                fi, fj = folds[i], folds[j]
                dfi, dfj = dfs[fi], dfs[fj]
                if feat not in dfi.columns or feat not in dfj.columns:
                    ks_rows.append(
                        {
                            "feature": feat,
                            "fold_i": fi,
                            "fold_j": fj,
                            "ks_w_abs": float("nan"),
                        }
                    )
                    continue

                xi = pd.to_numeric(dfi[feat], errors="coerce").to_numpy(
                    dtype=np.float64, copy=False
                )
                xj = pd.to_numeric(dfj[feat], errors="coerce").to_numpy(
                    dtype=np.float64, copy=False
                )
                wi = dfi[weight_col].to_numpy(dtype=np.float64, copy=False)
                wj = dfj[weight_col].to_numpy(dtype=np.float64, copy=False)

                ks = weighted_ks_distance(xi, wi, xj, wj)
                ks_rows.append(
                    {"feature": feat, "fold_i": fi, "fold_j": fj, "ks_w_abs": ks}
                )

    df_ks = pd.DataFrame(ks_rows)
    df_ks.to_csv(out_dir / "ks_pairwise.csv", index=False)

    # Plot overlays per feature
    report_lines.append("Plotting feature overlays (one PDF per feature) ...")
    for feat in features:
        # range from all folds combined for stable bins
        all_x = []
        for f in folds:
            if feat in dfs[f].columns:
                x = pd.to_numeric(dfs[f][feat], errors="coerce").to_numpy(
                    dtype=np.float64, copy=False
                )
                all_x.append(x[np.isfinite(x)])
        if not all_x or sum(a.size for a in all_x) == 0:
            continue
        all_x = np.concatenate(all_x, axis=0)
        lo, hi = robust_range(all_x, 0.005, 0.995)

        plt.figure()
        for f in folds:
            df = dfs[f]
            if feat not in df.columns:
                continue
            x = pd.to_numeric(df[feat], errors="coerce").to_numpy(
                dtype=np.float64, copy=False
            )
            w = safe_abs_weights(df[weight_col].to_numpy(dtype=np.float64, copy=False))
            m = np.isfinite(x)
            x, w = x[m], w[m]
            if x.size == 0:
                continue

            h, edges = np.histogram(
                x, bins=bins, range=(lo, hi), weights=w, density=True
            )
            centers = 0.5 * (edges[:-1] + edges[1:])
            plt.step(centers, h, where="mid", label=f"fold {f}")

        plt.xlabel(feat)
        plt.ylabel("density (absw)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{feat}.pdf")
        plt.close()

    # Summarize largest KS features
    if len(df_ks) > 0:
        df_ks_finite = df_ks[np.isfinite(df_ks["ks_w_abs"])]
        if len(df_ks_finite) > 0:
            top = (
                df_ks_finite.groupby("feature")["ks_w_abs"]
                .max()
                .sort_values(ascending=False)
                .head(25)
            )
            report_lines.append("Top 25 features by max pairwise weighted KS (absw):")
            report_lines.append(top.to_string())
            report_lines.append("")
            top.to_csv(out_dir / "top_ks_features.csv", header=True)

    # Write report
    (out_dir / "sanity_report.txt").write_text("\n".join(report_lines))
    print(f"[OK] Wrote: {out_dir}/feature_summary.csv")
    print(f"[OK] Wrote: {out_dir}/ks_pairwise.csv")
    print(f"[OK] Wrote: {out_dir}/sanity_report.txt")
    print(f"[OK] Plots: {out_dir}/plots/*.pdf")


# -----------------------------
# CLI
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Compare DNN input features across folds")
    p.add_argument(
        "--data-dir",
        required=True,
        help="Directory with data_df_{split}_{fold}.parquet",
    )
    p.add_argument(
        "--out-dir", required=True, help="Output directory for sanity products"
    )
    p.add_argument(
        "--folds", default=None, help="Comma list, e.g. 0,1,2,3. If omitted, tries 0..3"
    )
    p.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "evaluation"],
        help="Which split parquet to compare",
    )
    p.add_argument("--weight-col", default="weight", help="Weight column name")
    p.add_argument("--label-col", default="label", help="Label column name")
    p.add_argument(
        "--features-json",
        default=None,
        help="Optional JSON list of features to compare",
    )
    p.add_argument("--bins", type=int, default=60, help="Histogram bins")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    if args.folds is None:
        folds = [0, 1, 2, 3]
    else:
        folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]

    features = None
    if args.features_json is not None:
        features = list(pd.read_json(args.features_json, typ="series"))

    compare_folds(
        data_dir=data_dir,
        out_dir=out_dir,
        folds=folds,
        split=args.split,
        weight_col=args.weight_col,
        label_col=args.label_col,
        features=features,
        bins=args.bins,
    )


if __name__ == "__main__":
    main()
