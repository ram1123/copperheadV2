#!/usr/bin/env python3
"""
compare_zpt_variations.py

Quick standalone check of the Z-pT reweighting's up/down systematic impact on
the dimuon pT spectrum, reading directly from stage-1 parquet output.  Parquet
columns remain lazy (virtual) Dask-Awkward arrays and are reduced to histograms
on the workers, so the full event arrays are never materialized on the driver.

Branch names (see src/copperhead_processor.py):
  - wgt_nominal        total event weight, already includes the nominal zpt factor
  - separate_wgt_zpt   the standalone nominal zpt factor that was folded into wgt_nominal
  - zpt_wgt_reco_up/_down   the raw +-1 sigma zpt factor -- only present if
    save_zpt_variations: true was set for the stage-1 run that produced these
    files (default is false everywhere, so this isn't always available).

wgt_nominal is divided by separate_wgt_zpt to recover the weight without any
zpt correction, then re-multiplied by the up/down factor -- same pattern
plotter/validation_plotter_unified.py uses for --remove_zpt_weights.

Usage:
    python scripts/compare_zpt_variations.py \
        --label /work/projects/hmm/$USER/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation \
        --year 2025 \
        --sample dyTo2Mu_M-50_aMCatNLO \
        --out validation/figs/zpt_variation_check/2025_dyTo2Mu_M-50_aMCatNLO_dimuon_pt.pdf
"""
import argparse
import glob

import dask
import dask_awkward as dak
import hist.dask as hda
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pyarrow.parquet as pq


def connect_gateway(cluster_index):
    """Connect to an existing Purdue Kubernetes Dask Gateway cluster."""
    from dask_gateway import Gateway

    gateway = Gateway(
        "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
        proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
    )
    clusters = gateway.list_clusters()
    if not clusters:
        raise RuntimeError(
            "No running Dask Gateway cluster found. Start one first, or omit "
            "--use-gateway to use this session's local Dask scheduler."
        )
    if not 0 <= cluster_index < len(clusters):
        raise ValueError(
            f"--cluster-index {cluster_index} is out of range; "
            f"found {len(clusters)} running cluster(s)"
        )
    client = gateway.connect(clusters[cluster_index].name).get_client()
    print(f"Connected to Dask Gateway cluster {clusters[cluster_index].name}")
    return client


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help=".../copperheadV1clean/<label> directory")
    parser.add_argument("--year", required=True)
    parser.add_argument("--sample", default="dyTo2Mu_M-50_aMCatNLO")
    parser.add_argument("--level", default="compacted", choices=["f1_0", "compacted"], help="stage1_output subdirectory to read")
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--xmax", type=float, default=200.0)
    parser.add_argument("--out", default=None, help="output PDF path (default: ./zpt_variation_<sample>_<year>.pdf)")
    parser.add_argument(
        "--use-gateway",
        action="store_true",
        help="Run parquet reads and histogram reductions on an existing Dask Gateway cluster",
    )
    parser.add_argument(
        "--cluster-index", type=int, default=0,
        help="Running Dask Gateway cluster to use (default: 0)",
    )
    args = parser.parse_args()

    pattern = f"{args.label.rstrip('/')}/stage1_output/{args.year}/{args.level}/{args.sample}/*/*.parquet"
    files = glob.glob(pattern)
    if not files:
        raise SystemExit(f"No parquet files found for pattern: {pattern}")
    print(f"Found {len(files)} files for {args.sample} ({args.year})")

    columns = ["dimuon_pt", "wgt_nominal", "separate_wgt_zpt", "zpt_wgt_reco_up", "zpt_wgt_reco_down"]
    available = set(pq.ParquetFile(files[0]).schema_arrow.names)
    missing = [c for c in columns if c not in available]
    if missing:
        raise SystemExit(
            f"Missing columns in parquet schema: {missing}. zpt_wgt_reco_up/_down only "
            "exist if save_zpt_variations: true was set for the stage-1 run that produced "
            f"this output -- check with pyarrow.parquet.ParquetFile({files[0]!r}).schema_arrow.names"
        )

    client = connect_gateway(args.cluster_index) if args.use_gateway else None

    # These are virtual/lazy arrays. Column projection happens in the parquet
    # reader and only the compact histogram reductions below are returned to
    # this process.
    events = dak.from_parquet(files, columns=columns)
    dimuon_pt = dak.fill_none(events["dimuon_pt"], np.nan)
    wgt_nominal = dak.fill_none(events["wgt_nominal"], 0.0)
    separate_wgt_zpt = dak.fill_none(events["separate_wgt_zpt"], 1.0)
    zpt_up = dak.fill_none(events["zpt_wgt_reco_up"], 1.0)
    zpt_down = dak.fill_none(events["zpt_wgt_reco_down"], 1.0)

    valid = (dimuon_pt != -999.0) & np.isfinite(dimuon_pt)
    dimuon_pt = dimuon_pt[valid]
    wgt_nominal = wgt_nominal[valid]
    separate_wgt_zpt = separate_wgt_zpt[valid]
    zpt_up = zpt_up[valid]
    zpt_down = zpt_down[valid]

    # wgt_nominal already includes the nominal zpt factor -- swap it out for
    # the up/down factor to get the corresponding total-weight variation,
    # keeping every other weight component (PU, muon SF, etc.) unchanged.
    wgt_no_zpt = wgt_nominal / separate_wgt_zpt
    wgt_up = wgt_no_zpt * zpt_up
    wgt_down = wgt_no_zpt * zpt_down

    binning = np.linspace(0, args.xmax, args.bins + 1)
    histograms = []
    for weight in (wgt_nominal, wgt_up, wgt_down):
        histogram = hda.Hist.new.Variable(binning).Weight()
        histogram.fill(dimuon_pt, weight=weight)
        histograms.append(histogram)

    try:
        results = dask.compute(
            *histograms,
            dak.sum(valid),
            dak.sum(wgt_nominal),
            dak.sum(wgt_up),
            dak.sum(wgt_down),
        )
    finally:
        if client is not None:
            client.close()

    h_nom, h_up, h_down = (hist.values() for hist in results[:3])
    entries, sum_nominal, sum_up, sum_down = results[3:]

    plt.style.use(hep.style.CMS)
    fig, (ax_top, ax_ratio) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )

    ax_top.step(binning[:-1], h_nom, where="post", color="black", label="Nominal", linewidth=1.6)
    ax_top.step(binning[:-1], h_up, where="post", color="crimson", linestyle="--", label=r"Z-pT up (+1$\sigma$)", linewidth=1.4)
    ax_top.step(binning[:-1], h_down, where="post", color="royalblue", linestyle="--", label=r"Z-pT down (-1$\sigma$)", linewidth=1.4)
    ax_top.set_ylabel("Weighted events / bin")
    ax_top.legend(frameon=False)
    ax_top.set_title(f"{args.sample}, {args.year} -- Z-pT weight variation on dimuon pT", fontsize=13)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_up = np.where(h_nom > 0, h_up / h_nom, np.nan)
        ratio_down = np.where(h_nom > 0, h_down / h_nom, np.nan)
    ax_ratio.axhline(1.0, color="black", linewidth=1)
    ax_ratio.step(binning[:-1], ratio_up, where="post", color="crimson", linewidth=1.4)
    ax_ratio.step(binning[:-1], ratio_down, where="post", color="royalblue", linewidth=1.4)
    ax_ratio.set_ylabel("Var / Nom")
    ax_ratio.set_xlabel(r"$p_T(\mu\mu)$ [GeV]")
    ax_ratio.set_ylim(0.98, 1.02)

    out_path = args.out or f"zpt_variation_{args.sample}_{args.year}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")

    print(f"\nEntries: {entries}")
    print(f"Sum(wgt_nominal) = {sum_nominal:.1f}")
    print(f"Sum(wgt_up)      = {sum_up:.1f}  (delta = {(sum_up / sum_nominal - 1) * 100:+.2f}%)")
    print(f"Sum(wgt_down)    = {sum_down:.1f}  (delta = {(sum_down / sum_nominal - 1) * 100:+.2f}%)")


if __name__ == "__main__":
    main()
