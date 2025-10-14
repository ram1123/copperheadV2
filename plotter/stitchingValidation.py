import numpy as np
import dask.dataframe as dd
from pathlib import Path
import ROOT

inFilePath = Path("/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/stage1_output/2018/f1_0/")
file50   = "dy_M-50_aMCatNLO/*/*.parquet"
file100  = "dy_M-100To200_aMCatNLO/*/*.parquet"
fileVBF  = "dy_VBF_filter_NewZWgt/*/*.parquet"

var      = "gjj_mass"
nbins    = 50
xmin, xmax = 0.0, 1500.0

cols = [var]
df50   = dd.read_parquet(str(inFilePath/file50),   columns=cols)
df100  = dd.read_parquet(str(inFilePath/file100),  columns=cols)
dfVBF  = dd.read_parquet(str(inFilePath/fileVBF),  columns=cols)

s50_cut   = df50[df50.gjj_mass <= 350][var].compute().values
s100_cut  = df100[df100.gjj_mass <= 350][var].compute().values
sVBF_cut  = dfVBF[dfVBF.gjj_mass >  350][var].compute().values

s50_nocut  = df50[var].compute().values
s100_nocut = df100[var].compute().values

h50  = ROOT.TH1D("h50",  f"DY-M50 (mJJ≤350);{var};Events", nbins, xmin, xmax)
h100 = ROOT.TH1D("h100", f"DY-M100-200 (mJJ≤350);{var};Events", nbins, xmin, xmax)
hVBF = ROOT.TH1D("hVBF", f"DY_VBF_Filter (mJJ>350);{var};Events", nbins, xmin, xmax)

h50_nc  = ROOT.TH1D("h50_nc",  f"{var};{var};Events", nbins, xmin, xmax)
h100_nc = ROOT.TH1D("h100_nc", f"{var};{var};Events", nbins, xmin, xmax)

# Compute counts per bin for each array in one go:
edges = np.linspace(xmin, xmax, nbins+1)

for vals, hist in [(s50_cut, h50),
                   (s100_cut, h100),
                   (sVBF_cut, hVBF),
                   (s50_nocut, h50_nc),
                   (s100_nocut, h100_nc)]:
    counts, _ = np.histogram(vals, bins=edges)
    # Set bin contents (ROOT bins are 1-based)
    for ibin, c in enumerate(counts, start=1):
        hist.SetBinContent(ibin, c)

# ── STACK HISTOGRAMS ────────────────────────────────────────────────
h50 .SetFillColor(ROOT.kBlue-7)
h100.SetFillColor(ROOT.kGreen-7)
hVBF.SetFillColor(ROOT.kRed-7)

stack = ROOT.THStack("stack","Drell-Yan stitching")
stack.Add(h50)
stack.Add(h100)
stack.Add(hVBF)

hSum = h50_nc.Clone("hSum")
hSum.Add(h100_nc)
hSum.SetLineColor(ROOT.kBlack)
hSum.SetLineWidth(2)

# ── DRAW ────────────────────────────────────────────────────────────────
c = ROOT.TCanvas("c","Stitching Validation",800,600)
ROOT.gStyle.SetOptStat(0)
stack.Draw("HIST")
hSum.Draw("HIST SAME")

leg = ROOT.TLegend(0.65,0.65,0.9,0.9)
leg.AddEntry(h50,  "DY-M50, gjj_mass ≤ 350","f")
leg.AddEntry(h100, "DY-M100-200, gjj_mass ≤ 350","f")
leg.AddEntry(hVBF,  "DY_VBF_Filter, gjj_mass > 350","f")
leg.AddEntry(hSum,  "DY-M50+DY-M100-200 (no cut)","l")
leg.Draw()

c.SetLogy()
c.Draw()
c.SaveAs("stitching_validation.pdf")
