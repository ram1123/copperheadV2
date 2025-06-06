#!/usr/bin/env python3
"""
compare_dy_parquet_root.py

Read all Parquet files in multiple directories for two DY samples (each containing
“dimuon_mass” and “wgt_nominal”), build weighted histograms of the dimuon‐mass
distribution using ROOT, and overlay them.

# check with just /0/ directory
time python compare_inclusiveDY_DYvbfFilter.py         --dirs1 /depot/cms/users/shar1172/hmm/copperheadV1clean/May28_NanoV12/stage1_output/2018/f1_0/dy_M-50_MiNNLO/0/ /depot/cms/users/shar1172/hmm/copperheadV1clean/May28_NanoV12/stage1_output/2018/f1_0/dy_M-100To200_MiNNLO/0/         --dirs2 /depot/cms/users/shar1172/hmm/copperheadV1clean/May28_NanoV12/stage1_output/2018/f1_0/dy_VBF_filter/0/               --nbins  110         --xmin 105         --xmax 160         --output compareDY_M50M100_110bins_0.pdf
time python compare_inclusiveDY_DYvbfFilter.py         --dirs1 /depot/cms/users/shar1172/hmm/copperheadV1clean/May28_NanoV12/stage1_output/2018/f1_0/dy_M-50_MiNNLO/ /depot/cms/users/shar1172/hmm/copperheadV1clean/May28_NanoV12/stage1_output/2018/f1_0/dy_M-100To200_MiNNLO/         --dirs2 /depot/cms/users/shar1172/hmm/copperheadV1clean/May28_NanoV12/stage1_output/2018/f1_0/dy_VBF_filter/             --nbins  55         --xmin 105         --xmax 160         --output compareDY_M50M100_55bins.pdf
time python compare_inclusiveDY_DYvbfFilter.py         --dirs1 /depot/cms/users/shar1172/hmm/copperheadV1clean/May28_NanoV12/stage1_output/2018/f1_0/dy_M-50_MiNNLO/ /depot/cms/users/shar1172/hmm/copperheadV1clean/May28_NanoV12/stage1_output/2018/f1_0/dy_M-100To200_MiNNLO/         --dirs2 /depot/cms/users/shar1172/hmm/copperheadV1clean/May28_NanoV12/stage1_output/2018/f1_0/dy_VBF_filter/             --nbins  110         --xmin 105         --xmax 160         --output compareDY_M50M100_110bins.pdf
"""

import os
import glob
import argparse

import dask.dataframe as dd
import numpy as np

# Import ROOT for plotting
import ROOT
from ROOT import TH1D, TCanvas, TLegend, gPad, kBlue, kRed

# dont' show stats box
ROOT.gStyle.SetOptStat(0)

def parse_args():
    p = argparse.ArgumentParser(
        description="Overlay absolute dσ/dm_μμ for two DY samples stored in Parquet (with wgt_nominal), using ROOT"
    )
    p.add_argument(
        "--dirs1",
        required=True,
        nargs="+",
        help="One or more directories containing Parquet files for DY sample 1",
    )
    p.add_argument(
        "--dirs2",
        required=True,
        nargs="+",
        help="One or more directories containing Parquet files for DY sample 2",
    )
    p.add_argument(
        "--lumi",
        default="",
        action="store",
        # type=float,
        # required=True,
        help="Integrated luminosity [pb⁻¹] (e.g. 137 fb⁻¹ → 137000)",
    )
    p.add_argument(
        "--nbins",
        type=int,
        default=60,
        help="Number of bins in m_μμ (default: 60)",
    )
    p.add_argument(
        "--xmin",
        type=float,
        default=60.0,
        help="Lower edge of m_μμ histogram [GeV] (default: 60.0)",
    )
    p.add_argument(
        "--xmax",
        type=float,
        default=120.0,
        help="Upper edge of m_μμ histogram [GeV] (default: 120.0)",
    )
    p.add_argument(
        "--output",
        default="compareDY_abs.pdf",
        help="Filename for the output plot (e.g. compareDY_abs.pdf)",
    )
    return p.parse_args()


def collect_parquet_paths(directories):
    """
    Return a list of all .parquet file paths found in each directory in `directories`.
    """
    paths = []
    for d in directories:
        found = glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True)
        if not found:
            raise FileNotFoundError(f"No .parquet files found in {d!r}")
        paths.extend(found)
    return paths


def compute_weighted_hist_and_sumw2(parquet_paths, bins):
    """
    - Reads all Parquet files in `parquet_paths` into a single Dask DataFrame
      selecting "dimuon_mass" and "wgt_nominal".
    - Computes for each bin:
        * hist_wgt  = Σ wgt_nominal_i         (sum of weights in that bin)
        * hist_w2   = Σ (wgt_nominal_i)^2      (sum of squared weights, for errors)
    Returns:
        hist_wgt : numpy array of shape (len(bins)-1)
        hist_w2  : numpy array of shape (len(bins)-1)
    """
    ddf = dd.read_parquet(parquet_paths, columns=["dimuon_mass", "wgt_nominal"])

    n_bins = len(bins) - 1
    hist_wgt = np.zeros(n_bins, dtype=float)
    hist_w2 = np.zeros(n_bins, dtype=float)

    # Iterate over partitions; each partition is a pandas.DataFrame after compute()
    for delayed_part in ddf.to_delayed():
        df_part = delayed_part.compute()  # pandas.DataFrame
        masses = df_part["dimuon_mass"].to_numpy()
        weights = df_part["wgt_nominal"].fillna(0.0).to_numpy()

        # Weighted histogram of wgt_nominal
        wgt_hist, _ = np.histogram(masses, bins=bins, weights=weights)
        # Weighted histogram of wgt_nominal^2
        w2_hist,  _ = np.histogram(masses, bins=bins, weights=weights**2)

        hist_wgt += wgt_hist
        hist_w2  += w2_hist

    return hist_wgt, hist_w2


def main():
    args = parse_args()

    # 1) Gather all .parquet paths for each sample
    paths1 = collect_parquet_paths(args.dirs1)
    paths2 = collect_parquet_paths(args.dirs2)

    # 2) Define bin edges
    bins = np.linspace(args.xmin, args.xmax, args.nbins + 1)

    # 3) Compute weighted histogram and sum-of-squared-weights for each sample
    hist1_wgt, hist1_w2 = compute_weighted_hist_and_sumw2(paths1, bins)
    hist2_wgt, hist2_w2 = compute_weighted_hist_and_sumw2(paths2, bins)

    # 4) Convert to differential cross-section
    bin_width = (args.xmax - args.xmin) / args.nbins

    # dσ/dm [pb/GeV]
    hist1_pb_per_GeV = hist1_wgt / ( bin_width)
    hist2_pb_per_GeV = hist2_wgt / ( bin_width)

    # Errors [pb/GeV]
    err1 = np.sqrt(hist1_w2) / ( bin_width)
    err2 = np.sqrt(hist2_w2) / ( bin_width)

    # 5) Prepare bin centers
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # 6) Create ROOT histograms (TH1D) and fill
    h1 = TH1D(
        "h1",
        "h1",
        args.nbins,
        args.xmin,
        args.xmax
    )
    h2 = TH1D(
        "h2",
        "Scaled to 0.1477; m_{#mu#mu} [GeV]; d#sigma /dm [pb/GeV]",
        args.nbins,
        args.xmin,
        args.xmax
    )

    for i in range(args.nbins):
        bin_idx = i + 1  # ROOT bins start at 1
        h1.SetBinContent(bin_idx, hist1_pb_per_GeV[i])
        h1.SetBinError(bin_idx, err1[i])
        h2.SetBinContent(bin_idx, hist2_pb_per_GeV[i])
        h2.SetBinError(bin_idx, err2[i])

    # 7) Style and draw with ROOT
    c = TCanvas("c", "DY absolute cross sections", 800, 600)
    logy = False
    if logy:
        gPad.SetLogy()  # Use log scale on y-axis; remove or comment out for linear



    h1.SetLineColor(kBlue)
    h1.SetMinimum(10.0)  # Set minimum for log scale
    h1.SetMaximum(2000.0)  # Set maximum for log scale
    h2.SetMaximum(2000.0)  # Set maximum for log scale

    h2.SetLineColor(kRed)
    # h2.SetLineStyle(2)
    h1.SetTitle("")
    h2.SetTitle("")
    h2.Draw("E1")
    h1.Draw("E1 SAME")

    leg = TLegend(0.60, 0.65, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.AddEntry(h1, "DY M-50 + M100-200", "lep")
    leg.AddEntry(h2, "DY VBF-filter M105-160", "lep")

    # Draw ratio plot
    rp = ROOT.TRatioPlot(h2,h1)
    rp.SetH1DrawOpt('p e0 same')
    rp.SetH2DrawOpt('p e0 same')
    rp.Draw()
    # rp.GetLowerRefGraph().SetMinimum(0.0)
    # rp.GetLowerRefGraph().SetMaximum(1.5)
    leg.Draw()

    # c.Update()
    c.SaveAs((args.output).replace(".pdf", "_log.pdf") if logy else args.output)
    print(f"→ Saved comparison plot to {args.output}")

if __name__ == "__main__":
    main()
