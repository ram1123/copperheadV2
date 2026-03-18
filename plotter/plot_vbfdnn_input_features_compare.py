#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import glob

import awkward as ak
import dask_awkward as dak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from rich import print

from configs.dnn_features import FEATURES
from modules.dask_utils import close_dask_client, get_dask_client
from modules.selection import applyRegionCatCuts

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
    for _, cfg in FEATURES.items():
        cols.append(cfg["column"])
    return list(dict.fromkeys(cols))


def _glob_nonempty(path_pattern: str) -> bool:
    return len(glob.glob(path_pattern)) > 0


def _has_field(arr: dak.Array, name: str) -> bool:
    # safest for dak: check typetracer fields when available
    try:
        return name in ak.fields(arr._meta)  # noqa: SLF001
    except Exception:
        try:
            _ = arr[name]
            return True
        except Exception:
            return False


def _safe_flatten(x: dak.Array) -> dak.Array:
    """
    Flatten if jagged; if already flat scalar, ak.flatten may throw.
    Try/except is the most robust across awkward v2 + dask-awkward typetracer.
    """
    try:
        return ak.flatten(x, axis=None)
    except Exception:
        return x


def _compute_array(arr: dak.Array, col: str, lo: float, hi: float) -> np.ndarray:
    if not _has_field(arr, col):
        return np.array([], dtype=np.float64)

    x = arr[col]

    # Drop None at dask/awkward level
    x = x[~ak.is_none(x)]

    # Flatten if jagged (safe)
    x = _safe_flatten(x)

    # Compute to awkward
    x_ak = x.compute()
    if x_ak is None:
        return np.array([], dtype=np.float64)

    # Ensure flat
    try:
        x_ak = ak.flatten(x_ak, axis=None)
    except Exception:
        pass

    # Convert to numpy
    out = ak.to_numpy(x_ak).astype(np.float64, copy=False)

    # Finite + range cuts (numpy, always available)
    m = np.isfinite(out) & (out >= lo) & (out <= hi)
    return out[m]


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
        ax.set_ylim(1e-5, None)


def _read_all(
    file_map: dict[str, str],
    columns: list[str],
    *,
    category: str,
    region_name: str,
    variation: str,
    persist: bool = True,
) -> dict[str, dak.Array]:
    """
    Read each process parquet set once using dask_awkward,
    apply selection once, then (optionally) persist to cluster memory.

    persist=True is important: it prevents repeating IO/cuts for every feature.
    """
    out: dict[str, dak.Array] = {}

    for proc, pattern in file_map.items():
        if not _glob_nonempty(pattern):
            print(f"[yellow][SKIP][/yellow] {proc}: no files for {pattern}")
            continue

        print(f"[INFO] Reading {proc}: {pattern}")

        # read only needed columns when possible
        try:
            arr = dak.from_parquet(pattern, columns=columns)
        except Exception as e:
            print(
                f"[yellow][WARN][/yellow] from_parquet(columns=...) failed for {proc}: {e}"
            )
            print(f"[yellow][WARN][/yellow] Re-reading {proc} without column filter.")
            arr = dak.from_parquet(pattern)

        # apply selection (must be dak-compatible)
        arr = applyRegionCatCuts(
            arr,
            category=category,
            region_name=region_name,
            process=proc,
            variation=variation,
        )

        # persist to keep results cached across many feature plots
        if persist:
            print(f"[INFO] Persisting {proc} after cuts...")
            arr = arr.persist()

        out[proc] = arr

    return out


# -------------------------
# Plotting
# -------------------------
def plot_features_individual(
    dfs: dict[str, dak.Array],
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
            arr = dfs.get(proc)
            if arr is None:
                continue
            if not _has_field(arr, col):
                # common if some processes miss certain branches
                continue

            vals = _compute_array(arr, col, lo, hi)
            if vals.size == 0:
                continue

            _plot_overlay_hist(ax, vals, bins, meta["label"], meta["color"])
            any_drawn = True

        _finalize_axis(ax, cfg.get("title", feat), lo, hi, logy=logy)

        if any_drawn:
            ax.legend(fontsize=12, frameon=True)
        else:
            ax.text(
                0.5, 0.5, "No entries", ha="center", va="center", transform=ax.transAxes
            )

        out_path = output_dir / f"{feat.replace(' ', '_')}.pdf"
        fig.savefig(out_path)
        plt.close(fig)


def plot_features_overview_pages(
    dfs: dict[str, dak.Array],
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

        for ax in axes[len(page_feats) :]:
            ax.axis("off")

        for ax, (feat, cfg) in zip(axes, page_feats):
            col = cfg["column"]
            lo, hi = cfg["range"]
            nbins = int(cfg.get("bins", 60))
            bins = np.linspace(lo, hi, nbins + 1)

            any_drawn = False
            for proc, meta in PROCESSES.items():
                arr = dfs.get(proc)
                if arr is None or not _has_field(arr, col):
                    continue
                vals = _compute_array(arr, col, lo, hi)
                if vals.size == 0:
                    continue
                _plot_overlay_hist(ax, vals, bins, meta["label"], meta["color"])
                any_drawn = True

            _finalize_axis(ax, cfg.get("title", feat), lo, hi, logy=logy)

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

        # global legend
        handles, labels = [], []
        for _, meta in PROCESSES.items():
            handles.append(plt.Line2D([], [], color=meta["color"], lw=2))
            labels.append(meta["label"])
        fig.legend(handles, labels, loc="upper right", frameon=True, fontsize=12)

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


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--year", default="2022postEE")
    p.add_argument("--tag", default="run3_h-peak_vbf_28JanV2")
    p.add_argument("--category", default="vbf")
    p.add_argument("--region", default="h-peak")
    p.add_argument("--variation", default="nominal")
    p.add_argument(
        "--no-persist", action="store_true", help="Disable dask persist (debug)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    # IMPORTANT: this should create a dask-gateway client in your environment
    client = get_dask_client(True)
    print("[INFO] Dask client:", client)

    # (Optional) helps when gateway starts with 0 workers
    try:
        client.wait_for_workers(1)
    except Exception:
        pass

    base = "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_23Jan_JVMFilterJets/stage1_output"

    file_map = {
        "signal": f"{base}/{args.year}/compacted/vbf_powheg_dipole/0/*.parquet",
        "dy": f"{base}/{args.year}/compacted/dyTo2L_M-50_incl/0/*.parquet",
        "tt_sl": f"{base}/{args.year}/compacted/ttjets_sl/0/*.parquet",
        "tt_dl": f"{base}/{args.year}/compacted/ttjets_dl/0/*.parquet",
        "ewk_lljj": f"{base}/{args.year}/compacted/ewk_lljj/0/*.parquet",
    }

    cols = _required_columns()

    # columns needed by applyRegionCatCuts
    extra_cols = [
        "nBtagLoose_nominal",
        "nBtagMedium_nominal",
        "jj_mass_nominal",
        "jj_dEta_nominal",
        "njets_nominal",
    ]

    read_cols = list(dict.fromkeys(cols + extra_cols))
    print(
        f"[INFO] Will read {len(read_cols)} columns total (features + selection deps)."
    )

    dfs = _read_all(
        file_map,
        columns=read_cols,
        category=args.category,
        region_name=args.region,
        variation=args.variation,
        persist=(not args.no_persist),
    )

    out_dir = (
        Path("dnn/trained_models/Run3_nanoAODv12_23Jan_JVMFilterJets")
        / args.tag
        / "input_features_comparison"
        / f"{args.category}_selection_{args.region}"
    )

    print("[INFO] Plotting individual feature PDFs...")
    plot_features_individual(
        dfs,
        output_dir=out_dir / "individual",
        logy=True,
    )
    print("[INFO] Saved:", out_dir / "individual")

    print("[INFO] Plotting overview pages...")
    plot_features_overview_pages(
        dfs,
        output_dir=out_dir / "overview_pages",
        nrows=7,
        ncols=4,
        logy=True,
        tag="VBF DNN input features: signal vs backgrounds",
    )
    print("[INFO] Saved:", out_dir / "overview_pages")

    close_dask_client()


if __name__ == "__main__":
    main()
