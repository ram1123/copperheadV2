#!/usr/bin/env python3
"""
compare_dy_parquet_root.py

Read Parquet files for three DY samples, directly fill ROOT TH1D's
(with Sumw2) from “dimuon_mass” and “wgt_nominal”, scale to cross-section/mass,
and overlay them.
time python compare_three.py         --dirs1 /depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/stage1_output/2018/f1_0/dy_M-50_MiNNLO/ /depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/stage1_output/2018/f1_0/dy_M-100To200_MiNNLO/  --dirs2  /depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/stage1_output/2018/f1_0/dy_M-50_aMCatNLO/ /depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/stage1_output/2018/f1_0/dy_M-100To200_aMCatNLO/         --dirs3 /depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/stage1_output/2018/f1_0/dy_VBF_filter_NewZWgt/             --nbins  40         --xmin 110         --xmax 150         --output compareDY_M50M100_40bins_three.pdf
time python compare_three.py         --dirs1 /depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/stage1_output/2018/f1_0/dy_M-50_MiNNLO/ /depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/stage1_output/2018/f1_0/dy_M-100To200_MiNNLO/  --dirs2  /depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/stage1_output/2018/f1_0/dy_M-50_aMCatNLO/ /depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/stage1_output/2018/f1_0/dy_M-100To200_aMCatNLO/         --dirs3 /depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/stage1_output/2018/f1_0/dy_VBF_filter_NewZWgt/  --dirs4 /depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/stage1_output/2018/f1_0/dy_ZptWgt_MiNNLO/ --nbins  40         --xmin 110         --xmax 150         --output compareDY_four.pdf
"""

import os
import glob
import argparse

import dask.dataframe as dd

import ROOT
from ROOT import TH1D, TCanvas, TLegend, gPad, kBlue, kRed, kGreen, kMagenta, TRatioPlot

# Disable the stats box
ROOT.gStyle.SetOptStat(0)


def parse_args():
    p = argparse.ArgumentParser(
        description="Overlay dσ/dm_μμ for three DY samples (Parquet -> ROOT TH1D)"
    )
    p.add_argument(
        "--dirs1", required=True, nargs="+",
        help="Parquet directories for DY sample #1"
    )
    p.add_argument(
        "--dirs2", required=True, nargs="+",
        help="Parquet directories for DY sample #2"
    )
    p.add_argument(
        "--dirs3", required=True, nargs="+",
        help="Parquet directories for DY sample #3"
    )
    p.add_argument(
        "--dirs4", required=True, nargs="+",
        help="Parquet directories for DY sample #4"
    )
    p.add_argument(
        "--nbins", type=int, default=60,
        help="Number of m_μμ bins"
    )
    p.add_argument(
        "--xmin", type=float, default=60.0,
        help="Lower edge of m_μμ [GeV]"
    )
    p.add_argument(
        "--xmax", type=float, default=120.0,
        help="Upper edge of m_μμ [GeV]"
    )
    p.add_argument(
        "--output", default="compareDY3.pdf",
        help="Output plot filename"
    )
    return p.parse_args()


def collect_parquet_paths(dirs):
    paths = []
    for d in dirs:
        found = glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True)
        if not found:
            raise FileNotFoundError(f"No .parquet files found in '{d}'")
        paths.extend(found)
    return paths


def fill_hist(th, parquet_paths):
    # assume th.Sumw2() already called
    ddf = dd.read_parquet(parquet_paths, columns=["dimuon_mass", "wgt_nominal", "gjj_mass"])
    ddf = ddf[ddf["gjj_mass"] > 350.0]  # Apply the Mjj > 350 GeV cut
    for part in ddf.to_delayed():
        df = part.compute()
        masses  = df["dimuon_mass"].to_numpy()
        weights = df["wgt_nominal"].fillna(0.0).to_numpy()
        for m, w in zip(masses, weights):
            th.Fill(float(m), float(w))


def main():
    args = parse_args()

    # collect
    paths1 = collect_parquet_paths(args.dirs1)
    paths2 = collect_parquet_paths(args.dirs2)
    paths3 = collect_parquet_paths(args.dirs3)
    paths4 = collect_parquet_paths(args.dirs4)

    # create and configure histograms
    h1 = TH1D("DY1",
              "Sample 1; m_{#mu#mu} [GeV]; d#sigma/dm [pb/GeV]",
              args.nbins, args.xmin, args.xmax)
    h2 = TH1D("DY2",
              "Sample 2; m_{#mu#mu} [GeV]; d#sigma/dm [pb/GeV]",
              args.nbins, args.xmin, args.xmax)
    h3 = TH1D("DY3",
              "Sample 3; m_{#mu#mu} [GeV]; d#sigma/dm [pb/GeV]",
              args.nbins, args.xmin, args.xmax)
    h4 = TH1D("DY4",
              "Sample 4; m_{#mu#mu} [GeV]; d#sigma/dm [pb/GeV]",
              args.nbins, args.xmin, args.xmax)

    for h in (h1, h2, h3, h4):
        h.Sumw2()

    # fill
    fill_hist(h1, paths1)
    fill_hist(h2, paths2)
    fill_hist(h3, paths3)
    fill_hist(h4, paths4)

    # scale to dσ/dm [pb/GeV]
    bin_width = (args.xmax - args.xmin) / args.nbins
    scale_factor = 1.0 / bin_width
    h1.Scale(scale_factor)
    h2.Scale(scale_factor)
    h3.Scale(scale_factor)
    h4.Scale(scale_factor/16000.)

    # style
    h1.SetLineColor(kBlue);  h1.SetMarkerColor(kBlue);  #h1.SetMarkerStyle(20)
    h2.SetLineColor(kRed);   h2.SetMarkerColor(kRed);   #h2.SetMarkerStyle(21)
    h3.SetLineColor(kGreen); h3.SetMarkerColor(kGreen); #h3.SetMarkerStyle(22)
    h4.SetLineColor(kMagenta); h4.SetMarkerColor(kMagenta); #h4.SetMarkerStyle(23)

    h1.SetMaximum(1150.0)
    h2.SetMaximum(1150.0)
    h3.SetMaximum(1150.0)
    h4.SetMaximum(1150.0)

    # Get ratio of h1 and h3 into new histogram h13
    h13 = h1.Clone("h13")
    h13.Divide(h3)
    h13.SetLineColor(kBlue)
    h13.SetMarkerColor(kBlue)

    # draw
    c = TCanvas("c", "DY three-way comparison", 800, 600)
    rp = TRatioPlot(h2, h3)
    rp.SetH1DrawOpt("E")  # Draw h1 with error bars
    rp.SetH2DrawOpt("E")  # Draw h2 with error bars
    rp.Draw()
    rp.GetLowerRefYaxis().SetTitle("IncDY/VBF")

    rp.GetLowerPad().cd()
    h13.Draw("E SAME")

    rp.GetUpperPad().cd()


    h1.Draw("E SAME")
    # h2.Draw("E SAME")
    # h3.Draw("E SAME")
    # h4.Draw("E SAME")

    # legend
    leg = TLegend(0.60, 0.65, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.AddEntry(h1, "Inc. DY MiNNLO", "lep")
    leg.AddEntry(h2, "Inc. DY aMC@NLO", "lep")
    leg.AddEntry(h3, "VBF filter DY", "lep")
    # leg.AddEntry(h4, "New DY", "lep")
    leg.Draw()

    #

    c.Update()
    c.SaveAs(args.output)
    print(f"→ Saved comparison of three DY samples to {args.output}")


if __name__ == "__main__":
    main()
