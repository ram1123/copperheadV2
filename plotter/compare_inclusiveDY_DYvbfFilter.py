#!/usr/bin/env python3
"""
compare_dy_parquet_root.py

Read Parquet files for two DY samples, directly fill ROOT TH1D's
(with Sumw2) from “dimuon_mass” and “wgt_nominal”, scale to cross-section/mass,
and overlay them with a ratio panel.


Example usage:
time python compare_inclusiveDY_DYvbfFilter.py         --dirs1 /depot/cms/users/shar1172/hmm/copperheadV1clean/May28_NanoV12/stage1_output/2018/f1_0/dy_M-50_MiNNLO/ /depot/cms/users/shar1172/hmm/copperheadV1clean/May28_NanoV12/stage1_output/2018/f1_0/dy_M-100To200_MiNNLO/         --dirs2 /depot/cms/users/shar1172/hmm/copperheadV1clean/May28_NanoV12/stage1_output/2018/f1_0/dy_VBF_filter/             --nbins  55         --xmin 105         --xmax 160         --output compareDY_M50M100_55bins.pdf
time python compare_inclusiveDY_DYvbfFilter.py         --dirs1 /depot/cms/users/shar1172/hmm/copperheadV1clean/May28_NanoV12/stage1_output/2018/f1_0/dy_M-50_MiNNLO/ /depot/cms/users/shar1172/hmm/copperheadV1clean/May28_NanoV12/stage1_output/2018/f1_0/dy_M-100To200_MiNNLO/         --dirs2 /depot/cms/users/shar1172/hmm/copperheadV1clean/May28_NanoV12/stage1_output/2018/f1_0/dy_VBF_filter/             --nbins  40         --xmin 110         --xmax 150         --output compareDY_M50M100_40bins.pdf
time python compare_inclusiveDY_DYvbfFilter.py         --dirs1 /depot/cms/users/shar1172/hmm/copperheadV1clean/May28_NanoV12/stage1_output/2018/f1_0/dy_M-50_MiNNLO/ /depot/cms/users/shar1172/hmm/copperheadV1clean/May28_NanoV12/stage1_output/2018/f1_0/dy_M-100To200_MiNNLO/         --dirs2 /depot/cms/users/shar1172/hmm/copperheadV1clean/May28_NanoV12/stage1_output/2018/f1_0/dy_VBF_filter/             --nbins  80         --xmin 110         --xmax 150         --output compareDY_M50M100_80bins.pdf
"""

import os
import glob
import argparse

import dask.dataframe as dd

import ROOT
from ROOT import TH1D, TCanvas, TLegend, TRatioPlot, gPad, kBlue, kRed

# dont' show stats box
ROOT.gStyle.SetOptStat(0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay dSigma/dm_MuMu for two DY samples (Parquet -> ROOT TH1D)"
    )
    parser.add_argument(
        "--dirs1",
        required=True,
        nargs="+",
        help="Directories containing Parquet files for DY sample 1",
    )
    parser.add_argument(
        "--dirs2",
        required=True,
        nargs="+",
        help="Directories containing Parquet files for DY sample 2",
    )
    parser.add_argument(
        "--nbins", type=int, default=60, help="Number of bins in m_MuMu"
    )
    parser.add_argument(
        "--xmin", type=float, default=60.0, help="Lower edge of m_MuMu [GeV]"
    )
    parser.add_argument(
        "--xmax", type=float, default=120.0, help="Upper edge of m_MuMu [GeV]"
    )
    parser.add_argument(
        "--output", default="compareDY.pdf", help="Output filename for the plot"
    )
    return parser.parse_args()


def collect_parquet_paths(directories):
    """
    Return a list of all .parquet file paths found under each directory.
    """
    paths = []
    for d in directories:
        found = glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True)
        if not found:
            raise FileNotFoundError(f"No .parquet files found in '{d}'")
        paths.extend(found)
    return paths


def fill_hist_from_parquets(th1, parquet_paths):
    """
    Read all Parquet files into a Dask DataFrame (dimuon_mass, wgt_nominal).
    For each partition, fill the TH1D with (mass, weight). The TH1D
    must have Sumw2() called beforehand, so ROOT accumulates Σw² for errors.
    """
    ddf = dd.read_parquet(parquet_paths, columns=["dimuon_mass", "wgt_nominal"])
    for part in ddf.to_delayed():
        df = part.compute()
        masses = df["dimuon_mass"].to_numpy()
        weights = df["wgt_nominal"].fillna(0.0).to_numpy()
        for m, w in zip(masses, weights):
            th1.Fill(float(m), float(w))


def main():
    args = parse_args()

    # 1) Gather all Parquet file paths for each sample
    paths1 = collect_parquet_paths(args.dirs1)
    paths2 = collect_parquet_paths(args.dirs2)

    # 2) Prepare two ROOT histograms and enable Sumw2() for proper error bars
    h1 = TH1D("DY1", "DY M-50 + M100-200; m_{#mu#mu} [GeV]; d#sigma/dm [pb/GeV]",
              args.nbins, args.xmin, args.xmax)
    h2 = TH1D("DY2", "DY VBF-filter M105-160; m_{#mu#mu} [GeV]; d#sigma/dm [pb/GeV]",
              args.nbins, args.xmin, args.xmax)

    h1.Sumw2()
    h2.Sumw2()

    # 3) Loop over partitions and fill each histogram directly
    fill_hist_from_parquets(h1, paths1)
    fill_hist_from_parquets(h2, paths2)

    # 4) Scale each TH1D by (1 / bin_width) to convert Σw → dσ/dm [pb/GeV]
    bin_width = (args.xmax - args.xmin) / args.nbins
    scale_factor = 1.0 / bin_width
    h1.Scale(scale_factor)
    h2.Scale(scale_factor)

    # 5) Style histograms
    h1.SetLineColor(kBlue)
    h1.SetMarkerColor(kBlue)
    # h1.SetMarkerStyle(20)

    h2.SetLineColor(kRed)
    h2.SetMarkerColor(kRed)
    # h2.SetMarkerStyle(21)

    h1.SetTitle("")  # Title is carried by axis labels
    h1.GetXaxis().SetTitle("m_{#mu#mu} [GeV]")
    h1.GetYaxis().SetTitle("d#sigma/dm [pb/GeV]")

    # 6) Create canvas and draw ratio plot (h2/h1)
    canvas = TCanvas("c", "DY comparison", 800, 600)
    rp = TRatioPlot(h1, h2)
    rp.SetH1DrawOpt("E")  # Draw h1 with error bars
    rp.SetH2DrawOpt("E")  # Draw h2 with error bars
    rp.Draw()

    # 7) Add legend in the upper pad
    rp.GetUpperPad().cd()
    leg = TLegend(0.60, 0.70, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.AddEntry(h1, "DY M-50 + M100-200", "lep")
    leg.AddEntry(h2, "DY VBF-filter M105-160", "lep")
    leg.Draw()

    # Enable log scale on the upper pad:
    # rp.GetUpperPad().SetLogy()

    # 9) Finalize and save
    canvas.Update()
    canvas.SaveAs(args.output)
    print(f"Saved comparison plot to {args.output}")


if __name__ == "__main__":
    main()
