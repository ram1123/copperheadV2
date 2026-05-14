from __future__ import absolute_import

import HiggsAnalysis.CombinedLimit.util.plotting as plot
import ROOT
import argparse

ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(ROOT.kTRUE)

plot.ModTDRStyle()

parser = argparse.ArgumentParser(description="Arguments parser")
parser.add_argument("--input_file", dest="input_file", help="Input root file from fitDiagnostics", required=True)
parser.add_argument(
    "--shape_type", dest="shape_type", help="Shape directory in input root file from fitDiagnostics (e.g. shapes_fit_b, shapes_fit_s)", required=True
)
parser.add_argument("--region", dest="cards_dir", help="Region of interest (e.g. signal_region, ch1, ch2, etc.)", required=True)
parser.add_argument("--year", dest="year", help="Year of the data taking (e.g. 2016, 2017, 2018)", required=True)
parser.add_argument("--output_dir", dest="output_dir", help="Output directory for plots", required=True)
args = parser.parse_args()


canvas = ROOT.TCanvas()

fin = ROOT.TFile(args.input_file)

first_dir = args.shape_type
second_dir = args.cards_dir

h_bkg = fin.Get(first_dir + "/" + second_dir + "/total_background")
h_sig = fin.Get(first_dir + "/" + second_dir + "/total_signal")
h_dat = fin.Get(first_dir + "/" + second_dir + "/data")  # This is a TGraphAsymmErrors, not a TH1F


h_bkg.SetFillColor(ROOT.TColor.GetColor(100, 192, 232))
h_bkg.Draw("HIST")

h_err = h_bkg.Clone()
h_err.SetFillColorAlpha(12, 0.3)  # Set grey colour (12) and alpha (0.3)
h_err.SetMarkerSize(0)
h_err.Draw("E2SAME")

h_sig.SetLineColor(ROOT.kRed)
h_sig.Draw("HISTSAME")

h_dat.Draw("PSAME")

h_bkg.SetMinimum(0.001)
h_bkg.SetMaximum(h_bkg.GetMaximum() * 1.4)

legend = ROOT.TLegend(0.60, 0.70, 0.90, 0.91, "", "NBNDC")
legend.AddEntry(h_bkg, "Background", "F")
legend.AddEntry(h_sig, "Signal", "L")
legend.AddEntry(h_err, "Background uncertainty", "F")
legend.Draw()


# canvas.SaveAs("plot_%s_%s.root" % (first_dir, second_dir))
# canvas.SaveAs("plot_%s_%s.pdf" % (first_dir, second_dir))

# canvas.SetLogy()
# canvas.SaveAs("plot_%s_%s_log.root" % (first_dir, second_dir))
# canvas.SaveAs("plot_%s_%s_log.pdf" % (first_dir, second_dir))
## --- Ratio plot section ---
# Create pads for main and ratio plot
pad1 = ROOT.TPad("pad1", "pad1", 0, 0.3, 1, 1)
pad2 = ROOT.TPad("pad2", "pad2", 0, 0.0, 1, 0.3)
pad1.SetBottomMargin(0.03)
pad2.SetTopMargin(0.05)
pad2.SetBottomMargin(0.35)
pad2.SetGridy()
canvas.cd()
pad1.Draw()
pad2.Draw()

# Draw main plot in pad1
pad1.cd()
h_bkg.Draw("HIST")
h_err.Draw("E2SAME")
h_sig.Draw("HISTSAME")
h_dat.Draw("PSAME")
h_bkg.SetMinimum(0.001)
h_bkg.SetMaximum(h_bkg.GetMaximum() * 1.4)
legend.Draw()

# Draw ratio plot in pad2
pad2.cd()
nbins = h_bkg.GetNbinsX()
ratio_hist = ROOT.TH1F("ratio_hist", "Data/Background", nbins, h_bkg.GetXaxis().GetXmin(), h_bkg.GetXaxis().GetXmax())
for i in range(1, nbins+1):
    bkg = h_bkg.GetBinContent(i)
    dat = h_dat.GetY()[i-1] if i-1 < h_dat.GetN() else 0
    err = h_dat.GetEYlow()[i-1] if i-1 < h_dat.GetN() else 0
    ratio = dat/bkg if bkg > 0 else 0
    ratio_hist.SetBinContent(i, ratio)
    ratio_hist.SetBinError(i, err/bkg if bkg > 0 else 0)
ratio_hist.SetLineColor(ROOT.kBlack)
ratio_hist.SetMarkerStyle(20)
ratio_hist.SetTitle("")
ratio_hist.GetYaxis().SetTitle("Data / Bkg")
ratio_hist.GetYaxis().SetNdivisions(505)
ratio_hist.GetYaxis().SetTitleSize(0.12)
ratio_hist.GetYaxis().SetTitleOffset(0.4)
ratio_hist.GetYaxis().SetLabelSize(0.10)
ratio_hist.GetXaxis().SetTitleSize(0.12)
ratio_hist.GetXaxis().SetTitleOffset(1.0)
ratio_hist.GetXaxis().SetLabelSize(0.10)
ratio_hist.SetMinimum(0)
ratio_hist.SetMaximum(2)
ratio_hist.Draw("EP")

canvas.cd()
canvas.SaveAs("%s/plot_%s_%s_%s_with_ratio.root" % (args.output_dir, str(args.year), first_dir, second_dir))
canvas.SaveAs("%s/plot_%s_%s_%s_with_ratio.pdf" % (args.output_dir, str(args.year), first_dir, second_dir))

# set pat1 as logY
pad1.SetLogy()
canvas.SaveAs("%s/plot_%s_%s_%s_with_ratio_log.root" % (args.output_dir, str(args.year), first_dir, second_dir))
canvas.SaveAs("%s/plot_%s_%s_%s_with_ratio_log.pdf" % (args.output_dir, str(args.year), first_dir, second_dir))
