#!/usr/bin/env python3
"""
Make "Total (black points)" vs stacked (Real/HS + Fake/PU) plots for a list of jet variables
stored in a flat Parquet file (two leading jets).

- Black markers: total distribution of that variable (all jets passing selection)
- Stacked fills: fake (PU) + real (HS), which should sum to total

Assumes columns like:
  jet1_pt, jet1_eta, jet1_neHEF, ...
  jet2_pt, jet2_eta, jet2_neHEF, ...
and genmatch like:
  jet1_genJetIdx / jet2_genJetIdx   (HS if >=0, PU if <0)
or:
  jet1_isGenMatched / jet2_isGenMatched  (HS if True)

Example:
  python plot_real_fake_stack.py -i input.parquet -o outdir --normalize
"""

import os
import argparse
import numpy as np
import pandas as pd
import ROOT
import sys

ROOT.gROOT.SetBatch(True)


JET_ID_VARIABLES = [
    # --- Jet kinematics ---
    "jet1_pt_nominal", "jet1_eta_nominal",
    "jet2_pt_nominal", "jet2_eta_nominal",
    # "jj_dEta_nominal", "jj_mass_nominal",

    # --- Energy fractions ---
    "jet1_chEmEF_nominal", "jet1_chHEF_nominal", "jet1_neEmEF_nominal", "jet1_neHEF_nominal", "jet1_muEF_nominal",
    "jet2_chEmEF_nominal", "jet2_chHEF_nominal", "jet2_neEmEF_nominal", "jet2_neHEF_nominal", "jet2_muEF_nominal",

    # --- Multiplicities ---
    "jet1_chMultiplicity_nominal", "jet2_chMultiplicity_nominal",
    "jet1_neMultiplicity_nominal", "jet2_neMultiplicity_nominal",

    # --- Constituents ---
    "jet1_nConstituents_nominal", "jet1_nElectrons_nominal", "jet1_nMuons_nominal", "jet1_nSVs_nominal",
    "jet2_nConstituents_nominal", "jet2_nElectrons_nominal", "jet2_nMuons_nominal", "jet2_nSVs_nominal",

    # --- Flavour ---
    "jet1_hadronFlavour_nominal", "jet2_hadronFlavour_nominal",
    "jet1_partonFlavour_nominal", "jet2_partonFlavour_nominal",

    # --- HF noise variables ---
    "jet1_hfcentralEtaStripSize_nominal", "jet1_hfadjacentEtaStripsSize_nominal",
    "jet1_hfsigmaEtaEta_nominal", "jet1_hfsigmaPhiPhi_nominal",
    "jet2_hfcentralEtaStripSize_nominal", "jet2_hfadjacentEtaStripsSize_nominal",
    "jet2_hfsigmaEtaEta_nominal", "jet2_hfsigmaPhiPhi_nominal",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True, help="Input parquet file")
    p.add_argument("-o", "--outdir", default="jetid_real_fake_stack", help="Output directory")
    p.add_argument("--genmatch-suffix", default="hasMatchedGenJet_nominal",
                   help="Gen-match column suffix")
    p.add_argument("--genmatch-mode", choices=["genJetIdx", "bool"], default="genJetIdx",
                   help="How to interpret genmatch column")

    # Basic selection (applied per-jet depending on jet1/jet2)
    p.add_argument("--pt-min", type=float, default=30.0)
    p.add_argument("--abs-eta-max", type=float, default=4.7)

    # Plot options
    p.add_argument("--nbins", type=int, default=40)
    p.add_argument("--normalize", action="store_true", help="Normalize total + stack to unit area")

    # Optional: limit number of rows for quick tests
    p.add_argument("--max-rows", type=int, default=None, help="Read only first N rows")
    return p.parse_args()


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def jet_prefix(varname: str) -> str:
    # "jet1_xxx" -> "jet1_"
    if varname.startswith("jet1_"):
        return "jet1_"
    if varname.startswith("jet2_"):
        return "jet2_"
    # if varname.startswith("jj_"):
    #     return "jj_"        
    raise ValueError(f"Variable does not start with jet1_ or jet2_: {varname}")


def default_range(var: str):
    # You can tweak these defaults anytime.
    if var.endswith("pt"):
        return 0.0, 300.0
    if var.endswith("eta"):
        return -4.7, 4.7
    if var.endswith("phi"):
        return -3.2, 3.2

    # Energy fractions
    if any(var.endswith(s) for s in ["chEmEF", "chHEF", "neEmEF", "neHEF", "muEF"]):
        return 0.0, 1.0

    # Multiplicities / constituents
    if "Multiplicity" in var or "nConstituents" in var or var.endswith(("nElectrons", "nMuons", "nSVs")):
        return 0.0, 80.0

    # Flavour (discrete)
    if var.endswith(("hadronFlavour", "partonFlavour")):
        return -0.5, 25.5  # common hadronFlavour: 0,4,5 ; partonFlavour can be larger in abs

    # HF noise vars (set generous defaults; will auto-adjust if empty)
    if "hfcentralEtaStripSize" in var or "hfadjacentEtaStripsSize" in var:
        return 0.0, 20.0
    if "hfsigmaEtaEta" in var or "hfsigmaPhiPhi" in var:
        return 0.0, 0.05

    return None, None  # fallback -> infer from percentiles


def infer_range(x, fallback=(0.0, 1.0)):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return fallback
    lo = np.percentile(x, 1)
    hi = np.percentile(x, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return fallback
    return float(lo), float(hi)


def make_hist(name, title, nbins, xmin, xmax):
    h = ROOT.TH1F(name, title, nbins, xmin, xmax)
    h.Sumw2()
    h.SetStats(0)
    return h


def fill_hist(h, values):
    for v in values:
        if np.isfinite(v):
            h.Fill(float(v))


def style(h, kind):
    if kind == "real":
        h.SetFillColor(ROOT.kAzure + 1)
        h.SetLineColor(ROOT.kAzure + 2)
    elif kind == "fake":
        h.SetFillColor(ROOT.kOrange - 2)
        h.SetLineColor(ROOT.kOrange + 1)
    elif kind == "total":
        h.SetMarkerStyle(20)
        h.SetMarkerSize(1.0)
        h.SetLineColor(ROOT.kBlack)
    else:
        raise ValueError(kind)


def get_real_fake_masks(df, prefix, genmatch_suffix, genmatch_mode):
    col = prefix + genmatch_suffix
    if col not in df.columns:
        raise KeyError(f"Missing genmatch column: {col}")

    gen = df[col].to_numpy()
    if genmatch_mode == "genJetIdx":
        real = gen == 1
        fake = gen == 0
    else:
        real = gen.astype(bool)
        fake = ~real
    return real, fake


def get_preselection_mask(df, prefix, pt_min, abs_eta_max):
    ptcol = prefix + "pt_nominal"
    etacol = prefix + "eta_nominal"
    if ptcol not in df.columns or etacol not in df.columns:
        raise KeyError(f"Need {ptcol} and {etacol} for selection.")
    pt = df[ptcol].to_numpy(dtype=np.float32)
    eta = df[etacol].to_numpy(dtype=np.float32)
    return (pt >= pt_min) & (np.abs(eta) <= abs_eta_max)


def normalize_hist(h):
    integ = h.Integral()
    if integ > 0:
        h.Scale(1.0 / integ)


def plot_var(df, var, args):
    if var not in df.columns:
        print(f"[WARN] Missing column {var} -> skipping")
        return

    prefix = jet_prefix(var)

    # masks
    pre = get_preselection_mask(df, prefix, args.pt_min, args.abs_eta_max)
    real_mask, fake_mask = get_real_fake_masks(df, prefix, args.genmatch_suffix, args.genmatch_mode)

    total_mask = pre
    real = pre & real_mask
    fake = pre & fake_mask

    x = df[var].to_numpy(dtype=np.float32)

    xmin, xmax = default_range(var)
    if xmin is None or xmax is None:
        xmin2, xmax2 = infer_range(x[total_mask], fallback=(0.0, 1.0))
        xmin = xmin2 if xmin is None else xmin
        xmax = xmax2 if xmax is None else xmax

    # Book hists
    safe_name = var.replace(".", "_")
    h_total = make_hist(f"h_total_{safe_name}", f";{var};Entries", args.nbins, xmin, xmax)
    h_real  = make_hist(f"h_real_{safe_name}",  "", args.nbins, xmin, xmax)
    h_fake  = make_hist(f"h_fake_{safe_name}",  "", args.nbins, xmin, xmax)

    # Fill
    fill_hist(h_total, x[total_mask])
    fill_hist(h_real,  x[real])
    fill_hist(h_fake,  x[fake])

    # Normalize (shapes)
    if args.normalize:
        normalize_hist(h_total)
        # stack needs to match total in normalization too
        s = h_real.Integral() + h_fake.Integral()
        if s > 0:
            h_real.Scale(1.0 / s)
            h_fake.Scale(1.0 / s)

    # Style
    style(h_total, "total")
    style(h_real, "real")
    style(h_fake, "fake")

    # Stack
    stack = ROOT.THStack(f"stk_{safe_name}", f";{var};{'Normalized entries' if args.normalize else 'Entries'}")
    stack.Add(h_fake)   # bottom
    stack.Add(h_real)   # top

    # Draw
    c = ROOT.TCanvas(f"c_{safe_name}", "c", 800, 650)
    c.SetLeftMargin(0.12)
    c.SetBottomMargin(0.12)

    stack.Draw("HIST")
    ymax = max(stack.GetMaximum(), h_total.GetMaximum()) * 1.35
    stack.SetMaximum(ymax)

    h_total.Draw("E SAME")

    # Legend
    leg = ROOT.TLegend(0.60, 0.72, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(h_total, "Total jets", "pe")
    leg.AddEntry(h_real,  "Real (HS)", "f")
    leg.AddEntry(h_fake,  "Fake (PU)", "f")
    leg.Draw()

    # Save
    outpath = os.path.join(args.outdir, f"{var}_total_points_stack_real_fake.pdf")
    c.SaveAs(outpath)
    c.SetLogy(1)
    c.SaveAs(outpath.replace(".pdf","_log.pdf"))
    c.Close()

    # Cleanup ROOT objects
    h_total.Delete()
    h_real.Delete()
    h_fake.Delete()


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    print(f"[INFO] Reading: {args.input}")
    df = pd.read_parquet(args.input)
    if args.max_rows is not None:
        df = df.head(args.max_rows)

    print(f"[INFO] N rows: {len(df)}")

    for var in JET_ID_VARIABLES:
        plot_var(df, var, args)

    print(f"[DONE] Plots saved in: {args.outdir}")


if __name__ == "__main__":
    main()