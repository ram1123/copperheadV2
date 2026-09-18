#!/usr/bin/env python3
"""
compare_zpt_variations.py

Quick standalone check of the Z-pT reweighting's up/down systematic impact on
the dimuon pT spectrum, reading directly from stage-1 parquet output -- no
Dask client, no coffea Runner, just dak.from_parquet + plain histogramming.

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

import awkward as ak
import dask_awkward as dak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help=".../copperheadV1clean/<label> directory")
    parser.add_argument("--year", required=True)
    parser.add_argument("--sample", default="dyTo2Mu_M-50_aMCatNLO")
    parser.add_argument("--level", default="compacted", choices=["f1_0", "compacted"], help="stage1_output subdirectory to read")
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--xmax", type=float, default=200.0)
    parser.add_argument("--out", default=None, help="output PDF path (default: ./zpt_variation_<sample>_<year>.pdf)")
    args = parser.parse_args()

    pattern = f"{args.label.rstrip('/')}/stage1_output/{args.year}/{args.level}/{args.sample}/*/*.parquet"
    files = glob.glob(pattern)
    if not files:
        raise SystemExit(f"No parquet files found for pattern: {pattern}")
    print(f"Found {len(files)} files for {args.sample} ({args.year})")

    columns = ["dimuon_pt", "wgt_nominal", "separate_wgt_zpt", "zpt_wgt_reco_up", "zpt_wgt_reco_down"]
    events = dak.from_parquet(pattern, columns=columns).compute()

    missing = [c for c in columns if c not in events.fields]
    if missing:
        raise SystemExit(
            f"Missing columns in parquet schema: {missing}. zpt_wgt_reco_up/_down only "
            "exist if save_zpt_variations: true was set for the stage-1 run that produced "
            f"this output -- check with pyarrow.parquet.ParquetFile({files[0]!r}).schema_arrow.names"
        )

    dimuon_pt = ak.to_numpy(ak.fill_none(events["dimuon_pt"], -999.0))
    wgt_nominal = ak.to_numpy(ak.fill_none(events["wgt_nominal"], 0.0))
    separate_wgt_zpt = ak.to_numpy(ak.fill_none(events["separate_wgt_zpt"], 1.0))
    zpt_up = ak.to_numpy(ak.fill_none(events["zpt_wgt_reco_up"], 1.0))
    zpt_down = ak.to_numpy(ak.fill_none(events["zpt_wgt_reco_down"], 1.0))

    valid = dimuon_pt != -999.0
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
    h_nom, _ = np.histogram(dimuon_pt, bins=binning, weights=wgt_nominal)
    h_up, _ = np.histogram(dimuon_pt, bins=binning, weights=wgt_up)
    h_down, _ = np.histogram(dimuon_pt, bins=binning, weights=wgt_down)

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
    ax_ratio.set_ylim(0.8, 1.2)

    out_path = args.out or f"zpt_variation_{args.sample}_{args.year}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")

    print(f"\nEntries: {len(dimuon_pt)}")
    print(f"Sum(wgt_nominal) = {wgt_nominal.sum():.1f}")
    print(f"Sum(wgt_up)      = {wgt_up.sum():.1f}  (delta = {(wgt_up.sum() / wgt_nominal.sum() - 1) * 100:+.2f}%)")
    print(f"Sum(wgt_down)    = {wgt_down.sum():.1f}  (delta = {(wgt_down.sum() / wgt_nominal.sum() - 1) * 100:+.2f}%)")


if __name__ == "__main__":
    main()
