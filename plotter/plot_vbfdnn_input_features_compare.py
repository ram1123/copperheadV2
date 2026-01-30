#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
from rich import print

from configs.dnn_features import FEATURES
from modules.dask_utils import close_dask_client, get_dask_client

plt.style.use(hep.style.CMS)

PROCESSES = {
    "signal": {"label": "Signal", "color": "black"},
    "dy": {"label": "DY", "color": "tab:blue"},
    "tt_sl": {"label": "TT (SL)", "color": "tab:red"},
    "tt_dl": {"label": "TT (DL)", "color": "tab:orange"},
    "ewk_lljj": {"label": "EWK lljj", "color": "tab:green"},
}


# -------------------------
# Helpers
# -------------------------
def _required_columns() -> list[str]:
    cols = []
    for _feat, cfg in FEATURES.items():
        cols.append(cfg["column"])
    # unique, stable order
    return list(dict.fromkeys(cols))


def _read_all(file_map: dict[str, Path], columns: list[str]) -> dict[str, dd.DataFrame]:
    """
    Read each process parquet set once, with only the needed columns.
    If a dataset is missing some columns, we re-read without 'columns'
    and handle missing columns per-feature (more robust).
    """
    dfs = {}
    for proc, p in file_map.items():
        try:
            dfs[proc] = dd.read_parquet(str(p), columns=columns, engine="pyarrow")
        except Exception as e:
            print(
                f"[yellow][WARN][/yellow] dd.read_parquet(columns=...) failed for {proc}: {e}"
            )
            print(f"[yellow][WARN][/yellow] Re-reading {proc} without column filter.")
            dfs[proc] = dd.read_parquet(str(p), engine="pyarrow")
    return dfs


def _compute_array(dfp: dd.DataFrame, col: str, lo: float, hi: float) -> np.ndarray:
    """
    Compute one column to a NumPy array after dropna + range cut.
    """
    # Dask series
    s = dfp[col].dropna()
    s = s[(s >= lo) & (s <= hi)]
    arr = s.compute()  # pandas Series
    if arr is None:
        return np.array([], dtype=float)
    return arr.to_numpy(dtype=np.float64, copy=False)


def _plot_overlay_hist(ax, arr: np.ndarray, bins: np.ndarray, label: str, color: str):
    ax.hist(
        arr,
        bins=bins,
        histtype="step",
        density=True,
        label=label,
        color=color,
        linewidth=1.5,
    )


def _finalize_axis(ax, title: str, lo: float, hi: float, logy: bool = True):
    ax.set_title(title, fontsize=11)
    ax.set_xlim((lo, hi))
    ax.set_ylabel("Normalized entries")
    if logy:
        ax.set_yscale("log")
        # safe lower bound for density plots
        ax.set_ylim(1e-5, None)


# -------------------------
# Plot: individual PDFs
# -------------------------
def plot_features_individual(
    dfs: dict[str, dd.DataFrame],
    output_dir: Path,
    *,
    logy: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for feat, cfg in FEATURES.items():
        col = cfg["column"]
        lo, hi = cfg["range"]
        nbins = int(cfg.get("bins", 60))
        bins = np.linspace(lo, hi, nbins + 1)

        fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)

        any_drawn = False
        for proc, meta in PROCESSES.items():
            dfp = dfs.get(proc)
            if dfp is None:
                continue

            if col not in dfp.columns:
                print(f"[yellow][WARN][/yellow] {proc} missing column: {col} (skip)")
                continue

            arr = _compute_array(dfp, col, lo, hi)
            if arr.size == 0:
                continue

            _plot_overlay_hist(ax, arr, bins, meta["label"], meta["color"])
            any_drawn = True

        _finalize_axis(ax, cfg.get("title", feat), lo, hi, logy=logy)

        if any_drawn:
            ax.legend(fontsize=16, frameon=True)
        else:
            ax.text(
                0.5,
                0.5,
                "No entries",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        out_path = output_dir / f"{feat.replace(' ', '_')}.pdf"
        fig.savefig(out_path)
        plt.close(fig)


# -------------------------
# Plot: overview pages (fixed grid; multiple pages if needed)
# -------------------------
def plot_features_overview_pages(
    dfs: dict[str, dd.DataFrame],
    output_dir: Path,
    *,
    nrows: int = 7,
    ncols: int = 4,
    logy: bool = True,
    tag: str = "VBF DNN inputs",
):
    output_dir.mkdir(parents=True, exist_ok=True)

    feats = list(FEATURES.items())
    per_page = nrows * ncols
    n_pages = int(np.ceil(len(feats) / per_page))

    for ipage in range(n_pages):
        start = ipage * per_page
        stop = min((ipage + 1) * per_page, len(feats))
        page_feats = feats[start:stop]

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(4.8 * ncols, 3.8 * nrows),
            constrained_layout=True,
        )
        axes = np.array(axes).reshape(-1)

        # Turn off unused pads
        for ax in axes[len(page_feats) :]:
            ax.axis("off")

        for ax, (feat, cfg) in zip(axes, page_feats):
            col = cfg["column"]
            lo, hi = cfg["range"]
            nbins = int(cfg.get("bins", 60))
            bins = np.linspace(lo, hi, nbins + 1)

            any_drawn = False
            for proc, meta in PROCESSES.items():
                dfp = dfs.get(proc)
                if dfp is None:
                    continue

                if col not in dfp.columns:
                    continue

                arr = _compute_array(dfp, col, lo, hi)
                if arr.size == 0:
                    continue

                _plot_overlay_hist(ax, arr, bins, meta["label"], meta["color"])
                any_drawn = True

            _finalize_axis(ax, cfg.get("title", feat), lo, hi, logy=logy)

            # No per-pad legend (too busy). Mark empties.
            if not any_drawn:
                ax.text(
                    0.5,
                    0.5,
                    "No entries",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )

        # One global legend
        handles, labels = [], []
        for proc, meta in PROCESSES.items():
            handles.append(plt.Line2D([], [], color=meta["color"], lw=2))
            labels.append(meta["label"])
        fig.legend(handles, labels, loc="upper right", frameon=True, fontsize=16)
        # Global label/title
        fig.suptitle(f"{tag} — page {ipage+1}/{n_pages}", fontsize=16)
        hep.cms.label(
            data=False,
            label="Private Work",
            com="13.6",
            loc=0,
            ax=axes[0] if len(axes) else None,
            fontsize=11,
        )

        out_pdf = output_dir / f"overview_page{ipage+1:02d}.pdf"
        fig.savefig(out_pdf)
        plt.close(fig)


def main():
    client = get_dask_client(True)
    print("[INFO] Dask client:", client)

    file_map = {
        "signal": Path(
            "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_23Jan_JVMFilterJets/stage1_output/2022postEE/compacted/vbf_powheg_dipole/0/*.parquet"
        ),
        "dy": Path(
            "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_23Jan_JVMFilterJets/stage1_output/2022postEE/compacted/dyTo2L_M-50_incl/0/*.parquet"
        ),
        "tt_sl": Path(
            "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_23Jan_JVMFilterJets/stage1_output/2022postEE/compacted/ttjets_sl/0/*.parquet"
        ),
        "tt_dl": Path(
            "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_23Jan_JVMFilterJets/stage1_output/2022postEE/compacted/ttjets_dl/0/*.parquet"
        ),
        "ewk_lljj": Path(
            "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_23Jan_JVMFilterJets/stage1_output/2022postEE/compacted/ewk_lljj/0/*.parquet"
        ),
    }

    out_dir = (
        Path(
            "dnn/trained_models/Run3_nanoAODv12_23Jan_JVMFilterJets/run3_h-peak_vbf_28JanV2"
        )
        / "input_features_comparison"
    )

    cols = _required_columns()
    print(f"[INFO] Will read {len(cols)} columns (unique) across FEATURES.")
    dfs = _read_all(file_map, columns=cols)

    print("[INFO] Plotting individual feature PDFs...")
    plot_features_individual(dfs, output_dir=out_dir / "individual", logy=True)
    print("[INFO] Saved individual feature plots:", out_dir / "individual")

    print("[INFO] Plotting overview pages (fixed grid, auto-multipage)...")
    plot_features_overview_pages(
        dfs,
        output_dir=out_dir / "overview_pages",
        nrows=6,
        ncols=4,
        logy=True,
        tag="VBF DNN input features: signal vs backgrounds",
    )
    print("[INFO] Saved overview PDFs:", out_dir / "overview_pages")

    close_dask_client()


if __name__ == "__main__":
    main()
