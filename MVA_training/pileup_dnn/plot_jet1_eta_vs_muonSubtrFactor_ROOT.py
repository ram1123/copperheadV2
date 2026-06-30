"""
ROOT RDataFrame version of the jet1 η vs muonSubtrFactor 2D plot.

Parquet files are read via PyArrow, converted to numpy arrays, and fed into
ROOT.RDF.FromNumpy so all downstream operations (Filter, Histo2D) are pure
RDataFrame with lazy evaluation.

Input: stage-1 compacted parquet files from DY→2µ MC (Run 3 NanoAODv15, 2024).
Output: jet1_eta_vs_muonSubtrFactor_ROOT.png (two-panel: full range + zoomed MSF≤0.01)
"""

import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import ROOT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from modules.root_2dColorProfile import set_gradient_style

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
set_gradient_style()


PARQUET_GLOB = (
    "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/"
    "Run3_nanoAODv15_FilterJets_June02_tightPassLepVeto_NoJER/"
    "stage1_output/2024/compacted/dyTo2Mu_M-50_aMCatNLO/*/*.parquet"
)
TRACKER_EDGE = 2.7
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
COLS = ["jet1_eta_nominal", "jet1_muonSubtrFactor_nominal"]

# ── Load with PyArrow → numpy ──────────────────────────────────────────────
files = sorted(glob.glob(PARQUET_GLOB))
print(f"Loading {len(files)} parquet files via PyArrow...")

table = pq.read_table(files, columns=COLS)
df = table.to_pandas().dropna()
print(f"Events with jet1: {len(df):,}")

eta = df["jet1_eta_nominal"].to_numpy(dtype=np.float64)
msf = df["jet1_muonSubtrFactor_nominal"].to_numpy(dtype=np.float64)

# ── Build RDataFrame from numpy arrays ─────────────────────────────────────
rdf = ROOT.RDF.FromNumpy({"jet1_eta_nominal": eta, "jet1_muonSubtrFactor_nominal": msf})

# Lazy filter chains
rdf_pos   = rdf.Filter("jet1_muonSubtrFactor_nominal >= 0")

# Book histograms (lazy — no computation yet)
h_full = rdf_pos.Histo2D(
    ROOT.RDF.TH2DModel("h_full", ";jet1 #eta;muonSubtrFactor", 100, -5, 5, 80, 0, 1),
    "jet1_eta_nominal", "jet1_muonSubtrFactor_nominal",
)

# Trigger lazy evaluation for both histograms together
ROOT.RDF.RunGraphs([h_full])

h_full = h_full.GetValue()

# ── Summary stats via RDataFrame ───────────────────────────────────────────
rdf_in  = rdf.Filter(f"std::abs(jet1_eta_nominal) <= {TRACKER_EDGE}")
rdf_out = rdf.Filter(f"std::abs(jet1_eta_nominal) >  {TRACKER_EDGE}")

mean_in  = rdf_in .Mean("jet1_muonSubtrFactor_nominal")
mean_out = rdf_out.Mean("jet1_muonSubtrFactor_nominal")
count_in  = rdf_in .Count()
count_out = rdf_out.Count()

print(f"\nInside  |η|≤{TRACKER_EDGE}: {count_in.GetValue():,} jets | MSF mean={mean_in.GetValue():.4f}")
print(f"Outside |η|>{TRACKER_EDGE}: {count_out.GetValue():,} jets | MSF mean={mean_out.GetValue():.2e}")

# ── Canvas & drawing ───────────────────────────────────────────────────────
ROOT.gStyle.SetPadRightMargin(0.15)

c = ROOT.TCanvas("c", "", 800, 600)
c.Divide(1, 1)

_keep_alive = []  # prevent ROOT objects from being garbage-collected

def draw_pad(pad, hist, title):
    pad.cd()
    pad.SetLogz()
    # hist.SetTitle(title)
    hist.GetXaxis().SetTitleSize(0.05)
    hist.GetYaxis().SetTitleSize(0.05)
    hist.Draw("COLZ")
    y_lo = hist.GetYaxis().GetXmin()
    y_hi = hist.GetYaxis().GetXmax()
    for x in [-TRACKER_EDGE, TRACKER_EDGE]:
        ln = ROOT.TLine(x, y_lo, x, y_hi)
        ln.SetLineColor(ROOT.kRed)
        ln.SetLineStyle(2)
        ln.SetLineWidth(2)
        ln.Draw()
        _keep_alive.append(ln)

draw_pad(c.cd(1), h_full, "Full range [0, 1]")
# draw_pad(c.cd(2), h_zoom, "Zoomed: MSF #in [0, 0.01]")

c.cd(0)
tex = ROOT.TLatex(
    0.5, 0.975,
    # "jet1 #eta vs muonSubtrFactor    "
    "#font[12]{DY#rightarrow2#mu MC, 2024 (Run 3 NanoAODv15)}",
)
tex.SetNDC()
tex.SetTextAlign(22)
tex.SetTextSize(0.038)
# tex.Draw()
_keep_alive.append(tex)

out_path = os.path.join(OUT_DIR, "jet1_eta_vs_muonSubtrFactor.png")
# out_path = os.path.join(OUT_DIR, "jet1_eta_vs_muonSubtrFactor.pdf")
c.SaveAs(out_path)
print(f"\nSaved: {out_path}")
