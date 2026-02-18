#!/usr/bin/env python3
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep

from modules.dask_utils import close_dask_client, get_dask_client

plt.style.use(hep.style.CMS)


# -------------------------
# Config: point to your dir
# -------------------------
BASE = Path(
    "dnn/trained_models/Run3_nanoAODv12_23Jan_JVMFilterJets/run3_h-peak_vbf_28JanV2"
)

TRAIN = BASE / "data_df_train_0.parquet"
VALID = BASE / "data_df_validation_0.parquet"
EVAL = BASE / "data_df_evaluation_0.parquet"
FEAT_PKL = BASE / "training_features.pkl"

OUTDIR = BASE / "feature_plots" / "preprocessing_validation"
OUTDIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Helpers
# -------------------------
DEFAULT_EXCLUDE = {
    "label",
    "process",
    "dataset",
    "year",
    "event",
    "run",
    "luminosityBlock",
    "lumi",
    "fold",
    "wgt_nominal",
    "weight",
    "weights",
}


def _load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _get_features(df: pd.DataFrame, feat_pkl: Path | None) -> list[str]:
    # Prefer training_features.pkl if it exists
    if feat_pkl and feat_pkl.exists():
        with open(feat_pkl, "rb") as f:
            feats = pickle.load(f)
        # keep only those that exist
        feats = [c for c in feats if c in df.columns]
        if feats:
            return feats

    # Fallback: infer numeric columns as features
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in num_cols if c not in DEFAULT_EXCLUDE]
    return feats


def _finite_mask(x, w=None):
    m = np.isfinite(x.to_numpy())
    if w is not None:
        m = m & np.isfinite(w.to_numpy())
    return m


def _robust_range(x_all: np.ndarray, qlo=0.005, qhi=0.995):
    x_all = x_all[np.isfinite(x_all)]
    if x_all.size == 0:
        return None
    lo = np.quantile(x_all, qlo)
    hi = np.quantile(x_all, qhi)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.min(x_all))
        hi = float(np.max(x_all))
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    return float(lo), float(hi)


def _hist_overlay(ax, x, w, bins, label):
    ax.hist(
        x,
        bins=bins,
        weights=w,
        histtype="step",
        density=True,
        linewidth=1.6,
        label=label,
    )


def _extract_arrays(df, feat, wcol):
    w = df[wcol] if (wcol in df.columns) else None
    m = _finite_mask(df[feat], w)
    x = df.loc[m, feat].to_numpy(dtype=np.float64)
    ww = w.loc[m].to_numpy(dtype=np.float64) if w is not None else None
    return x, ww


# -------------------------
# 1) Individual: one PDF per feature
# -------------------------
def plot_feature_individual(
    feat: str, df_tr, df_va, df_ev, wcol="wgt_nominal", nbins=60
) -> bool:
    x_tr, w_tr = _extract_arrays(df_tr, feat, wcol)
    x_va, w_va = _extract_arrays(df_va, feat, wcol)
    x_ev, w_ev = _extract_arrays(df_ev, feat, wcol)

    rng = _robust_range(np.concatenate([x_tr, x_va, x_ev], axis=0))
    if rng is None:
        return False

    lo, hi = rng
    bins = np.linspace(lo, hi, nbins + 1)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    _hist_overlay(ax, x_tr, w_tr, bins, "train")
    _hist_overlay(ax, x_va, w_va, bins, "validation")
    _hist_overlay(ax, x_ev, w_ev, bins, "evaluation")

    ax.set_xlabel(feat)
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, None)

    hep.cms.label(data=False, label="Private Work", com="13.6", ax=ax, fontsize=11)
    ax.legend(frameon=True)
    fig.tight_layout()

    out = OUTDIR / "individual"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{feat}.pdf")
    plt.close(fig)
    return True


# -------------------------
# 2) Overview: many features tiled on ONE PAGE
#    If too many features, it will create multiple pages:
#    all_features_page01.pdf, page02.pdf, ...
# -------------------------
def plot_features_overview(
    feats: list[str],
    df_tr,
    df_va,
    df_ev,
    wcol="wgt_nominal",
    nbins=50,
    nrows=6,
    ncols=4,
):
    per_page = nrows * ncols
    n_pages = math.ceil(len(feats) / per_page)

    out = OUTDIR / "overview_pages"
    out.mkdir(parents=True, exist_ok=True)

    for page in range(n_pages):
        start = page * per_page
        stop = min((page + 1) * per_page, len(feats))
        feats_page = feats[start:stop]

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, 30))
        axes = axes.flatten()

        for ax in axes[len(feats_page) :]:
            ax.axis("off")

        for ax, feat in zip(axes, feats_page):
            x_tr, w_tr = _extract_arrays(df_tr, feat, wcol)
            x_va, w_va = _extract_arrays(df_va, feat, wcol)
            x_ev, w_ev = _extract_arrays(df_ev, feat, wcol)

            rng = _robust_range(np.concatenate([x_tr, x_va, x_ev], axis=0))
            if rng is None:
                ax.axis("off")
                continue

            lo, hi = rng
            bins = np.linspace(lo, hi, nbins + 1)

            _hist_overlay(ax, x_tr, w_tr, bins, "train")
            _hist_overlay(ax, x_va, w_va, bins, "validation")
            _hist_overlay(ax, x_ev, w_ev, bins, "evaluation")

            ax.set_title(feat, fontsize=12)
            ax.set_yscale("log")
            ax.set_ylim(1e-4, None)

        # Single shared legend (cleaner than repeating in every pad)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", frameon=True, fontsize=12)

        # Single CMS label for the whole page
        fig.text(
            0.02, 0.995, "CMS Private Work (13.6 TeV)", ha="left", va="top", fontsize=16
        )

        fig.tight_layout(rect=(0, 0, 0.98, 0.98))
        fig.savefig(out / f"all_features_page{page+1:02d}.pdf")
        plt.close(fig)


# -------------------------
# Main
# -------------------------
def main():
    client = get_dask_client(True)
    df_tr = _load_df(TRAIN)
    df_va = _load_df(VALID)
    df_ev = _load_df(EVAL)

    feats = _get_features(df_tr, FEAT_PKL)
    if not feats:
        raise RuntimeError(
            "No features found (training_features.pkl missing and no numeric columns inferred)."
        )

    print(f"[info] Found {len(feats)} features.")
    print(f"[info] Output dir: {OUTDIR}")

    # A) Individual plots
    ok, fail = 0, 0
    for i, feat in enumerate(feats, 1):
        try:
            if plot_feature_individual(feat, df_tr, df_va, df_ev):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            fail += 1
            print(f"[warn] Failed for {feat}: {e}")
        if i % 25 == 0 or i == len(feats):
            print(f"[progress] {i}/{len(feats)} (ok={ok}, fail={fail})")

    # B) One-page tiled overview (auto-splits into multiple pages if needed)
    plot_features_overview(feats, df_tr, df_va, df_ev, nrows=6, ncols=4)

    print("[done] Individual PDFs:", OUTDIR / "individual")
    print("[done] Overview pages:", OUTDIR / "overview_pages")
    close_dask_client()


if __name__ == "__main__":
    main()
