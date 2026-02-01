import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SUFFIX = ""  # "" or "_AfterImpute"  # optional suffix for output filenames


# -----------------------------------------------------------------------------
# Small internal helpers (shared)
# -----------------------------------------------------------------------------
def _ensure_dir(d: str) -> None:
    Path(d).mkdir(parents=True, exist_ok=True)


def _safe_weights(w: np.ndarray, mode: str = "abs") -> np.ndarray:
    """
    Make weights safe for histogramming/metrics.

    mode:
      - "abs": abs(w)  (HEP-stable default)
      - "clip0": negative -> 0
      - "none": raw weights (not recommended)
    """
    w = np.asarray(w, dtype=np.float64)
    if mode == "abs":
        w = np.abs(w)
    elif mode == "clip0":
        w = np.clip(w, 0.0, None)
    elif mode == "none":
        pass
    else:
        raise ValueError(f"Unknown mode: {mode}")

    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    if np.sum(w) <= 0:
        w = np.ones_like(w, dtype=np.float64)
    return w


def _robust_range(
    x: np.ndarray, w: np.ndarray, qlo: float = 0.001, qhi: float = 0.999
) -> Tuple[float, float]:
    """
    Weighted robust range for histogram limits.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return (0.0, 1.0)
    w = _safe_weights(w, mode="abs")

    # constant feature -> tiny window
    if np.allclose(x, x[0], rtol=0, atol=0):
        v = float(x[0])
        return (v - 1.0, v + 1.0)

    # weighted quantiles via sorting + CDF
    s = np.argsort(x)
    xs = x[s]
    ws = w[s]
    cdf = np.cumsum(ws) / (np.sum(ws) + 1e-12)

    lo = float(np.interp(qlo, cdf, xs))
    hi = float(np.interp(qhi, cdf, xs))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
    return lo, hi


def _page_grid(nshow: int, ncols: int = 3) -> Tuple[int, int]:
    nrows = (nshow + ncols - 1) // ncols
    return nrows, ncols


def _coerce_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


# -----------------------------------------------------------------------------
# 1) Keep: plot_before_after_scaling
# -----------------------------------------------------------------------------
def plot_before_after_scaling(x, w, x_mean, x_std, feature_names, outdir, nshow=51):
    _ensure_dir(outdir)

    x = np.asarray(x, dtype=np.float64)
    w = _safe_weights(np.asarray(w), mode="abs")
    x_mean = np.asarray(x_mean, dtype=np.float64)
    x_std = np.asarray(x_std, dtype=np.float64)

    # protect against tiny std
    x_std = np.where(x_std < 1e-12, 1.0, x_std)
    x_scaled = (x - x_mean) / x_std

    nshow = min(nshow, len(feature_names))
    nrows, ncols = _page_grid(nshow, ncols=3)

    # 1) Linear scale figure
    fig_lin, axes_lin = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True
    )
    axes_lin = np.atleast_1d(axes_lin).ravel()

    # 2) Log scale figure
    fig_log, axes_log = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True
    )
    axes_log = np.atleast_1d(axes_log).ravel()

    for i in range(nshow):
        bins_raw = 50
        bins_scaled = np.linspace(-5, 5, 60)

        # ---------- Linear ----------
        ax = axes_lin[i]
        ax.hist(
            x[:, i],
            bins=bins_raw,
            weights=w,
            density=True,
            histtype="step",
            label="raw",
            linewidth=1.5,
        )
        ax.hist(
            x_scaled[:, i],
            bins=bins_scaled,
            weights=w,
            density=True,
            histtype="step",
            label="scaled",
            linewidth=1.5,
        )
        ax.set_title(feature_names[i])
        ax.legend(fontsize=9)

        mean_raw = np.average(x[:, i], weights=w)
        std_raw = np.sqrt(np.average((x[:, i] - mean_raw) ** 2, weights=w))
        mean_scaled = np.average(x_scaled[:, i], weights=w)
        std_scaled = np.sqrt(np.average((x_scaled[:, i] - mean_scaled) ** 2, weights=w))
        ax.text(
            0.95,
            0.95,
            f"raw: μ={mean_raw:.2f}, σ={std_raw:.2f}\n"
            f"scaled: μ={mean_scaled:.2f}, σ={std_scaled:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
        )

        # ---------- Log ----------
        ax = axes_log[i]
        ax.hist(
            x[:, i],
            bins=bins_raw,
            weights=w,
            density=True,
            histtype="step",
            label="raw",
            linewidth=1.5,
        )
        ax.hist(
            x_scaled[:, i],
            bins=bins_scaled,
            weights=w,
            density=True,
            histtype="step",
            label="scaled",
            linewidth=1.5,
        )
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-6)
        ax.set_title(feature_names[i])
        ax.legend(fontsize=9)

        ax.text(
            0.95,
            0.95,
            f"raw: μ={mean_raw:.2f}, σ={std_raw:.2f}\n"
            f"scaled: μ={mean_scaled:.2f}, σ={std_scaled:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
        )

    # Turn off unused pads
    for ax in axes_lin[nshow:]:
        ax.axis("off")
    for ax in axes_log[nshow:]:
        ax.axis("off")

    fig_lin.suptitle("Scaling validation (linear scale)", fontsize=16)
    fig_log.suptitle("Scaling validation (log scale)", fontsize=16)

    fig_lin.savefig(f"{outdir}/scaling_before_after_linear{SUFFIX}.pdf")
    fig_log.savefig(f"{outdir}/scaling_before_after_log{SUFFIX}.pdf")

    plt.close(fig_lin)
    plt.close(fig_log)


# -----------------------------------------------------------------------------
# 2) Keep: plot_scaled_mean_std
# -----------------------------------------------------------------------------
def plot_scaled_mean_std(x_scaled, w, feature_names, outdir):
    """Sanity check: Check that mean shoudl be around ZERO and
    standard deviation should be around 1. If std is ZERO that means
    its a constant features.

    Args:
        x_scaled (_type_): _description_
        w (_type_): _description_
        feature_names (_type_): _description_
        outdir (_type_): _description_
    """
    _ensure_dir(outdir)

    x_scaled = np.asarray(x_scaled, dtype=np.float64)
    w = _safe_weights(np.asarray(w), mode="abs")

    mu = np.average(x_scaled, axis=0, weights=w)
    var = np.average((x_scaled - mu) ** 2, axis=0, weights=w)
    sigma = np.sqrt(np.maximum(var, 0.0))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(mu, "o")
    ax1.axhline(0, color="k", ls="--")
    ax1.set_ylabel("Weighted mean")

    ax2.plot(sigma, "o")
    ax2.axhline(1, color="k", ls="--")
    ax2.set_ylabel("Weighted std")
    ax2.set_xlabel("Feature index")

    fig.suptitle("Scaled feature statistics", fontsize=14)
    fig.savefig(f"{outdir}/scaling_mean_std{SUFFIX}.pdf")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 3) Keep: plot_corr_before_after
# -----------------------------------------------------------------------------
def plot_corr_before_after(x, x_scaled, outdir):
    """Scaling should not change the correlation between features.

    Correlation should look identical before and after scaling. If not,
    that implies a bug in the scaling code.

    Args:
        x (_type_): _description_
        x_scaled (_type_): _description_
        outdir (_type_): _description_
    """
    _ensure_dir(outdir)

    x = np.asarray(x, dtype=np.float64)
    x_scaled = np.asarray(x_scaled, dtype=np.float64)

    corr_raw = np.corrcoef(x.T)
    corr_scaled = np.corrcoef(x_scaled.T)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(corr_raw, vmin=-1, vmax=1, cmap="coolwarm", ax=axs[0])
    axs[0].set_title("Raw features")

    sns.heatmap(corr_scaled, vmin=-1, vmax=1, cmap="coolwarm", ax=axs[1])
    axs[1].set_title("Scaled features")

    fig.savefig(f"{outdir}/scaling_corr_check{SUFFIX}.pdf")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 4) Keep: plot_scaled_outliers
# -----------------------------------------------------------------------------
def plot_scaled_outliers(x_scaled, feature_names, outdir):
    """Check for outliers in scaled features.
    If max |z| > 10, that indicates a potential outlier in that feature.
    Args:
        x_scaled (_type_): _description_
        feature_names (_type_): _description_
        outdir (_type_): _description_
    """
    _ensure_dir(outdir)

    x_scaled = np.asarray(x_scaled, dtype=np.float64)
    max_abs = np.max(np.abs(x_scaled), axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(max_abs, "o")
    ax.axhline(10, color="r", ls="--", label="|z| = 10")
    ax.set_ylabel("max |z|")
    ax.set_xlabel("Feature index")
    ax.set_yscale("log")
    ax.legend()

    fig.savefig(f"{outdir}/scaling_outliers{SUFFIX}.pdf")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 5) Keep: plot_feature_hists_from_fold_dfs
# -----------------------------------------------------------------------------
def plot_feature_hists_from_fold_dfs(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_eval: pd.DataFrame,
    features: List[str],
    label_col: str,
    weight_col: str,
    outdir: str,
    fold_idx: int,
    bins: int = 60,
    weight_mode: str = "abs",
    max_features_per_pdf: int = 24,
    robust_qlo: float = 0.001,
    robust_qhi: float = 0.999,
    logy: bool = False,
    year_feature_name: Optional[str] = None,
    year_code_map: Optional[Dict[str, int]] = None,
) -> None:
    """
    Make sanity-check plots of ALL features from output fold dataframes.

    PDFs:
      - features_train_fold{fold}.pdf
      - features_val_fold{fold}.pdf
      - features_eval_fold{fold}.pdf
      - features_train_vs_val_fold{fold}.pdf
      - features_train_by_year_fold{fold}.pdf (optional; if year_feature_name numeric)
    """
    _ensure_dir(outdir)

    from matplotlib.backends.backend_pdf import PdfPages

    def _plot_split(df: pd.DataFrame, split_name: str) -> None:
        w_all = _safe_weights(df[weight_col].to_numpy(), mode=weight_mode)
        y_all = df[label_col].to_numpy(dtype=np.int64)

        pdf_path = os.path.join(outdir, f"features_{split_name}_fold{fold_idx}.pdf")
        nfeat = len(features)
        npp = max_features_per_pdf
        n_pages = (nfeat + npp - 1) // npp

        with PdfPages(pdf_path) as pdf:
            for page in range(n_pages):
                i0 = page * npp
                i1 = min(nfeat, (page + 1) * npp)
                feats_page = features[i0:i1]

                nshow = len(feats_page)
                nrows, ncols = _page_grid(nshow, ncols=3)
                fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows))
                axes = np.atleast_1d(axes).ravel()

                for ax, feat in zip(axes, feats_page):
                    x = _coerce_numeric(df[feat])
                    ok = np.isfinite(x)
                    x = x[ok]
                    w = w_all[ok]
                    y = y_all[ok]

                    if x.size == 0:
                        ax.set_title(f"{feat} (empty)")
                        ax.axis("off")
                        continue

                    lo, hi = _robust_range(x, w, qlo=robust_qlo, qhi=robust_qhi)
                    ms = y == 1
                    mb = y == 0

                    if np.any(mb):
                        ax.hist(
                            x[mb],
                            bins=bins,
                            range=(lo, hi),
                            weights=w[mb],
                            density=True,
                            histtype="step",
                            linewidth=1.5,
                            label="bkg",
                        )
                    if np.any(ms):
                        ax.hist(
                            x[ms],
                            bins=bins,
                            range=(lo, hi),
                            weights=w[ms],
                            density=True,
                            histtype="step",
                            linewidth=1.5,
                            label="sig",
                        )

                    ax.set_title(feat, fontsize=10)
                    if logy:
                        ax.set_yscale("log")
                        ax.set_ylim(bottom=1e-8)
                    ax.legend(fontsize=8)

                # turn off unused pads
                for ax in axes[len(feats_page) :]:
                    ax.axis("off")

                fig.suptitle(
                    f"Fold {fold_idx}: {split_name} feature sanity (sig vs bkg)",
                    fontsize=14,
                )
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        print(f"[plot] wrote {pdf_path}")

    def _plot_train_vs_val() -> None:
        """
        Overlay train vs val for each class separately (sig and bkg).
        Very good for quick overtraining smell-test per feature.
        """
        pdf_path = os.path.join(outdir, f"features_train_vs_val_fold{fold_idx}.pdf")

        yt = df_train[label_col].to_numpy(dtype=np.int64)
        yv = df_val[label_col].to_numpy(dtype=np.int64)
        wt = _safe_weights(df_train[weight_col].to_numpy(), mode=weight_mode)
        wv = _safe_weights(df_val[weight_col].to_numpy(), mode=weight_mode)

        nfeat = len(features)
        npp = max_features_per_pdf
        n_pages = (nfeat + npp - 1) // npp

        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(pdf_path) as pdf:
            for page in range(n_pages):
                i0 = page * npp
                i1 = min(nfeat, (page + 1) * npp)
                feats_page = features[i0:i1]

                nshow = len(feats_page)
                nrows, ncols = _page_grid(nshow, ncols=3)
                fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows))
                axes = np.atleast_1d(axes).ravel()

                for ax, feat in zip(axes, feats_page):
                    xt = _coerce_numeric(df_train[feat])
                    xv = _coerce_numeric(df_val[feat])

                    mt_ok = np.isfinite(xt)
                    mv_ok = np.isfinite(xv)

                    xt, wt2, yt2 = xt[mt_ok], wt[mt_ok], yt[mt_ok]
                    xv, wv2, yv2 = xv[mv_ok], wv[mv_ok], yv[mv_ok]

                    xcat = (
                        np.concatenate([xt, xv])
                        if (xt.size + xv.size) > 0
                        else np.array([], dtype=np.float64)
                    )
                    wcat = (
                        np.concatenate([wt2, wv2])
                        if (wt2.size + wv2.size) > 0
                        else np.array([], dtype=np.float64)
                    )
                    lo, hi = _robust_range(xcat, wcat, qlo=robust_qlo, qhi=robust_qhi)

                    for cls, name in [(0, "bkg"), (1, "sig")]:
                        mt = yt2 == cls
                        mv = yv2 == cls
                        if np.any(mt):
                            ax.hist(
                                xt[mt],
                                bins=bins,
                                range=(lo, hi),
                                weights=wt2[mt],
                                density=True,
                                histtype="step",
                                linewidth=1.2,
                                label=f"{name} train",
                            )
                        if np.any(mv):
                            ax.hist(
                                xv[mv],
                                bins=bins,
                                range=(lo, hi),
                                weights=wv2[mv],
                                density=True,
                                histtype="step",
                                linewidth=1.2,
                                label=f"{name} val",
                            )

                    ax.set_title(feat, fontsize=10)
                    if logy:
                        ax.set_yscale("log")
                        ax.set_ylim(bottom=1e-8)
                    ax.legend(fontsize=7)

                for ax in axes[len(feats_page) :]:
                    ax.axis("off")

                fig.suptitle(
                    f"Fold {fold_idx}: Train vs Val overlay (sig/bkg)", fontsize=14
                )
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        print(f"[plot] wrote {pdf_path}")

    def _plot_train_by_year() -> None:
        if year_feature_name is None:
            return
        if year_feature_name not in df_train.columns:
            return

        ycode = _coerce_numeric(df_train[year_feature_name])
        if not np.isfinite(ycode).any():
            return  # not numeric -> skip

        yt = df_train[label_col].to_numpy(dtype=np.int64)
        wt = _safe_weights(df_train[weight_col].to_numpy(), mode=weight_mode)

        codes = sorted(list(set(int(v) for v in np.unique(ycode[np.isfinite(ycode)]))))
        if not codes:
            return

        code_to_name = None
        if year_code_map is not None:
            # user passes {"2022preEE":0, ...} -> invert to {0:"2022preEE",...}
            code_to_name = {int(v): str(k) for k, v in year_code_map.items()}

        pdf_path = os.path.join(outdir, f"features_train_by_year_fold{fold_idx}.pdf")
        nfeat = len(features)
        npp = max_features_per_pdf
        n_pages = (nfeat + npp - 1) // npp

        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(pdf_path) as pdf:
            for page in range(n_pages):
                i0 = page * npp
                i1 = min(nfeat, (page + 1) * npp)
                feats_page = features[i0:i1]

                nshow = len(feats_page)
                nrows, ncols = _page_grid(nshow, ncols=3)
                fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows))
                axes = np.atleast_1d(axes).ravel()

                for ax, feat in zip(axes, feats_page):
                    x = _coerce_numeric(df_train[feat])
                    ok = np.isfinite(x) & np.isfinite(ycode)
                    x = x[ok]
                    w = wt[ok]
                    y = yt[ok]
                    yc = ycode[ok].astype(np.int64)

                    if x.size == 0:
                        ax.set_title(f"{feat} (empty)")
                        ax.axis("off")
                        continue

                    lo, hi = _robust_range(x, w, qlo=robust_qlo, qhi=robust_qhi)

                    # background only for stability
                    mb = y == 0
                    for code in codes:
                        mm = mb & (yc == code)
                        if not np.any(mm):
                            continue
                        lab = (
                            code_to_name[code]
                            if (code_to_name and code in code_to_name)
                            else str(code)
                        )
                        ax.hist(
                            x[mm],
                            bins=bins,
                            range=(lo, hi),
                            weights=w[mm],
                            density=True,
                            histtype="step",
                            linewidth=1.2,
                            label=lab,
                        )

                    ax.set_title(f"{feat} (bkg by year)", fontsize=10)
                    if logy:
                        ax.set_yscale("log")
                        ax.set_ylim(bottom=1e-8)
                    ax.legend(fontsize=7)

                for ax in axes[len(feats_page) :]:
                    ax.axis("off")

                fig.suptitle(f"Fold {fold_idx}: Train background by year", fontsize=14)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        print(f"[plot] wrote {pdf_path}")

    # --- make plots ---
    _plot_split(df_train, "train")
    _plot_split(df_val, "val")
    _plot_split(df_eval, "eval")
    _plot_train_vs_val()
    _plot_train_by_year()


# -----------------------------------------------------------------------------
# 6) Keep: plot_feature_hists_compare_folds
# -----------------------------------------------------------------------------
def plot_feature_hists_compare_folds(
    fold_parquet_dir: str,
    features: List[str],
    label_col: str,
    weight_col: str,
    outdir: str,
    n_folds: int,
    split: str = "train",  # "train" | "validation" | "evaluation" | "val" | "eval"
    cls: int = 0,  # 0=bkg overlay (recommended), 1=sig overlay
    bins: int = 60,
    weight_mode: str = "abs",
    max_features_per_pdf: int = 24,
    robust_qlo: float = 0.001,
    robust_qhi: float = 0.999,
    logy: bool = False,
    normalize_density: bool = True,
) -> None:
    """
    Compare distributions across folds for a given split and class.
    Reads: data_df_{split}_{i}.parquet  (split= train/validation/evaluation)
    """
    _ensure_dir(outdir)

    split_token = split
    if split == "val":
        split_token = "validation"
    if split == "eval":
        split_token = "evaluation"

    # load fold dfs
    dfs: List[pd.DataFrame] = []
    for i in range(n_folds):
        p = Path(fold_parquet_dir) / f"data_df_{split_token}_{i}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing fold parquet: {p}")
        dfs.append(pd.read_parquet(p))

    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = os.path.join(outdir, f"compare_folds_{split_token}_cls{cls}.pdf")
    nfeat = len(features)
    npp = max_features_per_pdf
    n_pages = (nfeat + npp - 1) // npp

    with PdfPages(pdf_path) as pdf:
        for page in range(n_pages):
            i0 = page * npp
            i1 = min(nfeat, (page + 1) * npp)
            feats_page = features[i0:i1]

            nshow = len(feats_page)
            nrows, ncols = _page_grid(nshow, ncols=3)
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows))
            axes = np.atleast_1d(axes).ravel()

            for ax, feat in zip(axes, feats_page):
                # robust range from all folds (selected class)
                xs_all = []
                ws_all = []
                for df in dfs:
                    x = _coerce_numeric(df[feat])
                    y = df[label_col].to_numpy(dtype=np.int64)
                    w = _safe_weights(df[weight_col].to_numpy(), mode=weight_mode)
                    m = np.isfinite(x) & (y == cls)
                    xs_all.append(x[m])
                    ws_all.append(w[m])

                xcat = (
                    np.concatenate(xs_all) if xs_all else np.array([], dtype=np.float64)
                )
                wcat = (
                    np.concatenate(ws_all) if ws_all else np.array([], dtype=np.float64)
                )
                lo, hi = _robust_range(xcat, wcat, qlo=robust_qlo, qhi=robust_qhi)

                # overlay folds
                for fidx, df in enumerate(dfs):
                    x = _coerce_numeric(df[feat])
                    y = df[label_col].to_numpy(dtype=np.int64)
                    w = _safe_weights(df[weight_col].to_numpy(), mode=weight_mode)
                    m = np.isfinite(x) & (y == cls)
                    if not np.any(m):
                        continue
                    ax.hist(
                        x[m],
                        bins=bins,
                        range=(lo, hi),
                        weights=w[m],
                        density=normalize_density,
                        histtype="step",
                        linewidth=1.2,
                        label=f"fold{fidx}",
                    )

                ax.set_title(feat, fontsize=10)
                if logy:
                    ax.set_yscale("log")
                    ax.set_ylim(bottom=1e-8)
                ax.legend(fontsize=7)

            for ax in axes[len(feats_page) :]:
                ax.axis("off")

            cls_name = "bkg" if cls == 0 else "sig"
            fig.suptitle(
                f"Across-fold comparison: {split_token} ({cls_name})", fontsize=14
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[plot] wrote {pdf_path}")
