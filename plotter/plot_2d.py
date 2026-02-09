#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import dask_awkward as dak
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from modules import selection
from modules.trials import get_stage1_path
from modules.dask_utils import get_dask_client, close_dask_client


SENTINEL = -999.0


def _to_numpy_clean(x: dak.Array) -> np.ndarray:
    """Compute to numpy, drop None, drop non-finite, drop sentinel."""
    x = ak.fill_none(x, SENTINEL)
    x = x[x != SENTINEL]
    x_ak = x.compute()
    if x_ak is None:
        return np.array([], dtype=np.float64)
    out = ak.to_numpy(x_ak)
    if np.ma.isMaskedArray(out):
        out = out.compressed()
    out = np.asarray(out)
    out = out[np.isfinite(out)]
    return out.astype(np.float64, copy=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--year", default="2022postEE")
    parser.add_argument("-p", "--process", default="dyTo2L_M-50_incl")
    parser.add_argument("--variation", default="nominal")
    parser.add_argument("--do-vbf-filter-study", action="store_true", default=False)
    parser.add_argument("--do-VH-veto", action="store_true", default=False)
    parser.add_argument("--outdir", default="plots_2d")
    parser.add_argument("--ptmax", type=float, default=75.0)
    parser.add_argument("--nbins-pt", type=int, default=60)
    parser.add_argument("--nbins-eta", type=int, default=60)
    parser.add_argument("--use-gateway", action="store_true", default=True)
    parser.add_argument("--cluster-index", type=int, default=0)
    args = parser.parse_args()

    # Fixed request
    region = "h-sidebands"
    category = "nocat"

    outdir = Path(args.outdir) / args.year / (args.process.replace("*", "")) / region / category
    outdir.mkdir(parents=True, exist_ok=True)

    client = get_dask_client(
        use_gateway=args.use_gateway, cluster_index=args.cluster_index
    )
    print("[INFO] Dask client:", client)

    # ------------------------------------------------------------
    # Load stage1 parquet (pattern from your helper)
    # ------------------------------------------------------------
    stage1_path = get_stage1_path()  # default = "current"
    stage1_path = str(Path(stage1_path) / args.year / "f1_0" / args.process  )
    print(f"Using LOAD_PATH template: {stage1_path}")

    # If your helper returns a directory, use a glob:
    # stage1_glob = f"{stage1_path}/*/*.parquet"
    # If it already returns a glob, keep as is:
    stage1_glob = str(stage1_path)
    if stage1_glob.endswith("/"):
        stage1_glob = stage1_glob.rstrip("/")
    if stage1_glob.endswith(".parquet"):
        parquet_pattern = stage1_glob
    else:
        parquet_pattern = f"{stage1_glob}/*/*.parquet"

    print("[INFO] Reading:", parquet_pattern)
    events = dak.from_parquet(parquet_pattern)

    # ------------------------------------------------------------
    # Apply region / category cuts (your exact call)
    # ------------------------------------------------------------
    events = selection.applyRegionCatCuts(
        events,
        category=category,
        region_name=region,
        process=args.process,
        variation=args.variation,
        do_vbf_filter_study=args.do_vbf_filter_study,
        do_VH_veto=args.do_VH_veto,
    )

    # ------------------------------------------------------------
    # Choose jet variables (leading jet)
    # ------------------------------------------------------------
    # Change to jet2_* if you want subleading, or loop if you want both.
    pt_field = "jet1_pt_nominal"
    eta_field = "jet1_eta_nominal"

    if pt_field not in events.fields or eta_field not in events.fields:
        raise KeyError(
            f"Missing fields in parquet. Need '{pt_field}' and '{eta_field}'. "
            f"Available (first 30): {list(events.fields)[:30]}"
        )

    jet_pt = _to_numpy_clean(events[pt_field])
    jet_eta = _to_numpy_clean(events[eta_field])

    # Align lengths if something weird happened (rare, but safe)
    n = min(jet_pt.size, jet_eta.size)
    jet_pt = jet_pt[:n]
    jet_eta = jet_eta[:n]

    if n == 0:
        raise RuntimeError("No events after selections (empty arrays).")

    # ------------------------------------------------------------
    # 2D histogram
    # ------------------------------------------------------------
    pt_bins = np.linspace(25.0, args.ptmax, args.nbins_pt + 1)
    eta_bins = np.linspace(-4.7, 4.7, args.nbins_eta + 1)

    H, xedges, yedges = np.histogram2d(jet_pt, jet_eta, bins=[pt_bins, eta_bins])

    # Linear
    fig, ax = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    m = ax.pcolormesh(
        xedges, yedges, H.T
    )  # transpose because histogram2d returns [x,y]
    fig.colorbar(m, ax=ax, label="Events")
    ax.set_xlabel(r"$p_{T}(\mathrm{jet1})$ [GeV]")
    ax.set_ylabel(r"$\eta(\mathrm{jet1})$")
    ax.set_title(f"{args.process} | {args.year} | {region} | {category}")
    ax.axhline(y=2.5, color='r', linestyle='--')
    ax.axhline(y=-2.5, color='r', linestyle='--')
    ax.axhline(y=3.0, color='r', linestyle='--',)
    ax.axhline(y=-3.0, color='r', linestyle='--')
    # vertical lines at pt = 30 and 50 GeV
    ax.axvline(x=30, color='r', linestyle='--',)
    ax.axvline(x=50, color='r', linestyle='--',)
    ax.legend()

    fig.savefig(outdir / "jet1_pt_vs_eta_h-sidebands_nocat.pdf")
    plt.close(fig)

    # Log-z
    fig, ax = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    # avoid LogNorm issues with zeros by setting vmin=1 for display
    m = ax.pcolormesh(xedges, yedges, H.T, norm=LogNorm(vmin=1, vmax=max(1, H.max())))
    fig.colorbar(m, ax=ax, label="Events (log)")
    ax.set_xlabel(r"$p_{T}(\mathrm{jet1})$ [GeV]")
    ax.set_ylabel(r"$\eta(\mathrm{jet1})$")
    ax.set_title(f"{args.process} | {args.year} | {region} | {category} (log-z)")

    # draw horizontal lines at eta = 2.5 and 3.0. Both sides +ve and -ve
    ax.axhline(y=2.5, color='r', linestyle='--')
    ax.axhline(y=-2.5, color='r', linestyle='--')
    ax.axhline(y=3.0, color='r', linestyle='--',)
    ax.axhline(y=-3.0, color='r', linestyle='--')
    # vertical lines at pt = 30 and 50 GeV
    ax.axvline(x=30, color='r', linestyle='--',)
    ax.axvline(x=50, color='r', linestyle='--',)
    ax.legend()
    fig.savefig(outdir / "jet1_pt_vs_eta_h-sidebands_nocat_logz.pdf")
    plt.close(fig)

    print("[INFO] Saved to:", outdir.resolve())
    close_dask_client()


if __name__ == "__main__":
    main()
