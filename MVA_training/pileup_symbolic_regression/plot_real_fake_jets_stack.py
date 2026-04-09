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
  python MVA_training/pileup_symbolic_regression/plot_real_fake_jets_stack.py \
    -i "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn25GeV_Apr03_tightPassLepVeto_NoJER_JetIDFix/stage1_output/2022postEE/compacted/dyTo2L_M-50_incl/0/part01*.parquet" \
    -o validation/compare_real_fake/After_JetID_Fix_HEHFcut \
    --apply-cleaning 
"""

import os
import argparse
import glob
import numpy as np
import pandas as pd
import ROOT
import sys

ROOT.gROOT.SetBatch(True)

JET_ID_VARIABLES = [
    # --- Jet kinematics ---
    "jet1_pt_nominal", "jet1_eta_nominal", 
    "jet2_pt_nominal", "jet2_eta_nominal", 
    "jet1_mass_nominal", "jet2_mass_nominal", 
    "jet1_phi_nominal", "jet2_phi_nominal", 
    # "jj_dEta_nominal", "jj_mass_nominal",

    # --- Jet ID / PU ID ---
    # "jet1_puId_nominal", "jet1_jetId_nominal", "jet1_hasMatchedGenJet_nominal",
    # "jet2_puId_nominal", "jet2_jetId_nominal", "jet2_hasMatchedGenJet_nominal",

    # --- Energy fractions ---
    "jet1_chEmEF_nominal", "jet1_chHEF_nominal", "jet1_neEmEF_nominal", "jet1_neHEF_nominal", 
    "jet2_chEmEF_nominal", "jet2_chHEF_nominal", "jet2_neEmEF_nominal", "jet2_neHEF_nominal", 
    "jet1_muEF_nominal", "jet2_muEF_nominal",

    # --- Multiplicities ---
    "jet1_chMultiplicity_nominal", "jet2_chMultiplicity_nominal",
    "jet1_neMultiplicity_nominal", "jet2_neMultiplicity_nominal",

    # --- Constituents / leptons / SVs ---
    "jet1_nConstituents_nominal", "jet2_nConstituents_nominal", 
    # "jet1_nElectrons_nominal", "jet1_nMuons_nominal", "jet1_nSVs_nominal",
    # "jet2_nElectrons_nominal", "jet2_nMuons_nominal", "jet2_nSVs_nominal",

    # --- Object indices ---
    # "jet1_electronIdx1_nominal", "jet1_electronIdx2_nominal",
    # "jet1_muonIdx1_nominal", "jet1_muonIdx2_nominal",
    # "jet1_svIdx1_nominal", "jet1_svIdx2_nominal",
    # "jet1_genJetIdx_nominal",

    # "jet2_electronIdx1_nominal", "jet2_electronIdx2_nominal",
    # "jet2_muonIdx1_nominal", "jet2_muonIdx2_nominal",
    # "jet2_svIdx1_nominal", "jet2_svIdx2_nominal",
    # "jet2_genJetIdx_nominal",

    # --- Flavour / taggers ---
    # "jet1_hadronFlavour_nominal", "jet2_hadronFlavour_nominal",
    # "jet1_partonFlavour_nominal", "jet2_partonFlavour_nominal",
    "jet1_btagDeepFlavQG_nominal", "jet2_btagDeepFlavQG_nominal",
    "jet1_btagPNetQvG_nominal", "jet2_btagPNetQvG_nominal",
    "jet1_btagDeepFlavB_nominal", "jet2_btagDeepFlavB_nominal",
    "jet1_btagDeepFlavCvB_nominal", "jet2_btagDeepFlavCvB_nominal",

    # --- HF noise variables ---
    "jet1_hfcentralEtaStripSize_nominal", "jet1_hfadjacentEtaStripsSize_nominal",
    "jet1_hfsigmaEtaEta_nominal", "jet1_hfsigmaPhiPhi_nominal",
    "jet2_hfcentralEtaStripSize_nominal", "jet2_hfadjacentEtaStripsSize_nominal",
    "jet2_hfsigmaEtaEta_nominal", "jet2_hfsigmaPhiPhi_nominal",

    # --- Raw / muon-subtracted info / geometry ---
    "jet1_area_nominal", "jet2_area_nominal",
    "jet1_rawFactor_nominal", "jet2_rawFactor_nominal",
    "jet1_muonSubtrFactor_nominal", "jet2_muonSubtrFactor_nominal",
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
    p.add_argument("--pt-min", type=float, default=25.0)
    p.add_argument("--abs-eta-max", type=float, default=4.7)

    # Plot options
    p.add_argument("--nbins", type=int, default=100)
    p.add_argument("--normalize", action="store_true", help="Normalize total + stack to unit area")

    # Optional: limit number of rows for quick tests
    p.add_argument("--max-rows", type=int, default=None, help="Read only first N rows")

    p.add_argument(
        "--region",
        default="inclusive",
        choices=[
            "inclusive", "central",
            "HE", "HF",
            "HEpos", "HEneg",
            "HFpos", "HFneg"
        ],
        help="Eta region selection"
    )

    p.add_argument(
        "--apply-cleaning",
        action="store_true",
        help="Apply HE/HF cleaning (same logic as corr script)"
    )    
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
    if "_pt_" in var:
        return 0.0, 200.0, 100
    if "nConstituents" in var:
        return 0.0, 30.0, 30
    if "mass" in var:
        return 0.0, 15.0, 100
    if var.endswith("eta"):
        return -4.7, 4.7, 50
    if var.endswith("phi"):
        return -3.2, 3.2, 50

    # Energy fractions
    if any(var.endswith(s) for s in ["chEmEF", "chHEF", "neEmEF", "neHEF", "muEF"]):
        return 0.0, 1.0, 50

    # Multiplicities / constituents
    if "Multiplicity" in var or "nConstituents" in var or var.endswith(("nElectrons", "nMuons", "nSVs")):
        return 0.0, 10.0, 10

    # Flavour (discrete)
    if var.endswith(("hadronFlavour", "partonFlavour")):
        return -0.5, 25.5, 50  # common hadronFlavour: 0,4,5 ; partonFlavour can be larger in abs

    # HF noise vars (set generous defaults; will auto-adjust if empty)
    if "hfcentralEtaStripSize" in var or "hfadjacentEtaStripsSize" in var:
        return 0.0, 20.0, 50
    if "hfsigmaEtaEta" in var or "hfsigmaPhiPhi" in var:
        return 0.0, 0.05, 50

    return None, None, None  # fallback -> infer from percentiles


def infer_nbins(x, xmin, xmax):
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return 20

    # Freedman–Diaconis rule
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(x) ** (1/3))

    if bin_width <= 0:
        return 40

    nbins = int((xmax - xmin) / bin_width)
    return max(20, min(nbins, 80))


def infer_range(x, fallback=(0.0, 1.0, 100)):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return fallback
    lo = np.percentile(x, 0.001)
    hi = np.percentile(x, 99.99999)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return fallback
    return float(lo), float(hi), infer_nbins(x, float(lo), float(hi))


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


def eta_region_mask(df, prefix, region, abs_eta_max):
    etacol = prefix + "eta_nominal"
    if etacol not in df.columns:
        raise KeyError(f"Need {etacol} for eta-region selection.")

    eta = df[etacol].to_numpy(dtype=np.float32)
    aeta = np.abs(eta)

    if region == "inclusive":
        return aeta <= abs_eta_max
    elif region == "central":
        return aeta < 2.5
    elif region == "HE":
        return (aeta > 2.5) & (aeta <= 3.0)
    elif region == "HF":
        return (aeta > 3.0) & (aeta <= abs_eta_max)
    elif region == "HEpos":
        return (eta > 2.5) & (eta <= 3.0)
    elif region == "HEneg":
        return (eta < -2.5) & (eta >= -3.0)
    elif region == "HFpos":
        return (eta > 3.0) & (eta <= abs_eta_max)
    elif region == "HFneg":
        return (eta < -3.0)
    else:
        raise ValueError(f"Unknown region: {region}")


def region_tag(region):
    return {
        "inclusive": "inclusive",
        "central": "central",
        "HE": "HE",
        "HF": "HF",
    }[region]


def plot_var(df, var, args, region="inclusive"):
    if var not in df.columns:
        print(f"[WARN] Missing column {var} -> skipping")
        return

    prefix = jet_prefix(var)

    # masks
    pre = get_preselection_mask(df, prefix, args.pt_min, args.abs_eta_max)
    reg = eta_region_mask(df, prefix, region, args.abs_eta_max)

    # ---------------------------------------------------------
    # Optional HE/HF cleaning
    # ---------------------------------------------------------
    if args.apply_cleaning:

        nC_col = prefix + "nConstituents_nominal"
        pt_col = prefix + "pt_nominal"
        area_col = prefix + "area_nominal"
        eta_col = prefix + "eta_nominal"
        mass_col = prefix + "mass_nominal"

        nC = df[nC_col].to_numpy(dtype=np.float32)
        pt = df[pt_col].to_numpy(dtype=np.float32)
        area = df[area_col].to_numpy(dtype=np.float32)
        eta = df[eta_col].to_numpy(dtype=np.float32)
        mass = df[mass_col].to_numpy(dtype=np.float32)
        aeta = np.abs(eta)

        # HE region mask (geometry only)
        he_geom = (aeta > 2.5) & (aeta <= 3.0)

        # HF region mask (geometry only)
        hf_geom = (aeta > 3.0)

        # HE cleaning condition
        he_reject = he_geom & (
            ((nC < 4) & (pt < 30.0)) 
            | ((area > 0.56) & (nC < 6))
            # (mass < 3.0)
        )

        # HF cleaning condition
        hf_reject_cut = (
            (nC < 5) & (area > 0.56)
        )
        hf_reject = hf_geom & hf_reject_cut

        # Remove jets failing either
        # reject = he_reject | hf_reject
        reject = hf_reject

        reg = reg & (~reject)

    real_mask, fake_mask = get_real_fake_masks(df, prefix, args.genmatch_suffix, args.genmatch_mode)

    total_mask = pre & reg
    real = total_mask & real_mask
    fake = total_mask & fake_mask

    if total_mask.sum() == 0:
        print(f"[WARN] No entries for {var} in region={region} -> skipping")
        return

    x = df[var].to_numpy(dtype=np.float32)

    xmin, xmax, nbins = default_range(var)
    if xmin is None or xmax is None or nbins is None:
        xmin2, xmax2, nbins = infer_range(x[total_mask], fallback=(0.0, 1.0, 100))
        xmin = xmin2 if xmin is None else xmin
        xmax = xmax2 if xmax is None else xmax

    # Book hists
    safe_name = var.replace(".", "_")
    rtag = region_tag(region)
    ytitle = "Normalized entries" if args.normalize else "Entries"

    h_total = make_hist(f"h_total_{safe_name}_{rtag}", f";{var};{ytitle}", nbins, xmin, xmax)
    h_real  = make_hist(f"h_real_{safe_name}_{rtag}",  "", nbins, xmin, xmax)
    h_fake  = make_hist(f"h_fake_{safe_name}_{rtag}",  "", nbins, xmin, xmax)

    # Fill
    fill_hist(h_total, x[total_mask])
    fill_hist(h_real,  x[real])
    fill_hist(h_fake,  x[fake])

    # Normalize (shapes)
    if args.normalize:
        normalize_hist(h_total)
        s = h_real.Integral() + h_fake.Integral()
        if s > 0:
            h_real.Scale(1.0 / s)
            h_fake.Scale(1.0 / s)

    # Style
    style(h_total, "total")
    style(h_real, "real")
    style(h_fake, "fake")

    # Stack
    stack = ROOT.THStack(
        f"stk_{safe_name}_{rtag}",
        f";{var};{ytitle}"
    )
    stack.Add(h_fake)
    stack.Add(h_real)

    # Ratio hist: Fake / Total
    h_ratio = h_fake.Clone(f"h_ratio_{safe_name}_{rtag}")
    h_ratio.Reset("ICESM")

    for ib in range(1, h_ratio.GetNbinsX() + 1):
        f = h_fake.GetBinContent(ib)
        t = h_total.GetBinContent(ib)

        if t > 0:
            h_ratio.SetBinContent(ib, f / t)
            h_ratio.SetBinError(ib, 0.0)
        else:
            h_ratio.SetBinContent(ib, 0.0)
            h_ratio.SetBinError(ib, 0.0)

    h_ratio.SetTitle("")
    h_ratio.GetYaxis().SetTitle("Fake / Total")
    h_ratio.GetXaxis().SetTitle(var)
    h_ratio.SetMarkerStyle(20)
    h_ratio.SetMarkerSize(0.8)
    h_ratio.SetLineColor(ROOT.kBlack)
    h_ratio.SetMarkerColor(ROOT.kBlack)

    h_ratio.GetYaxis().SetNdivisions(505)
    h_ratio.GetYaxis().SetTitleSize(0.09)
    h_ratio.GetYaxis().SetTitleOffset(0.55)
    h_ratio.GetYaxis().SetLabelSize(0.08)

    h_ratio.GetXaxis().SetTitleSize(0.10)
    h_ratio.GetXaxis().SetTitleOffset(1.0)
    h_ratio.GetXaxis().SetLabelSize(0.08)

    h_ratio.SetMinimum(0.0)
    h_ratio.SetMaximum(1.05)

    # Canvas with 2 pads
    c = ROOT.TCanvas(f"c_{safe_name}_{rtag}", "c", 800, 800)

    pad1 = ROOT.TPad(f"pad1_{safe_name}_{rtag}", "", 0.0, 0.30, 1.0, 1.0)
    pad2 = ROOT.TPad(f"pad2_{safe_name}_{rtag}", "", 0.0, 0.00, 1.0, 0.30)

    pad1.SetLeftMargin(0.12)
    pad1.SetRightMargin(0.04)
    pad1.SetTopMargin(0.06)
    pad1.SetBottomMargin(0.02)

    pad2.SetLeftMargin(0.12)
    pad2.SetRightMargin(0.04)
    pad2.SetTopMargin(0.03)
    pad2.SetBottomMargin(0.32)

    pad1.Draw()
    pad2.Draw()

    # Top pad
    pad1.cd()
    stack.Draw("HIST")
    ymax = max(stack.GetMaximum(), h_total.GetMaximum()) * 1.35
    stack.SetMaximum(ymax)
    stack.GetXaxis().SetLabelSize(0.0)
    stack.GetXaxis().SetTitleSize(0.0)

    h_total.Draw("E SAME")

    leg = ROOT.TLegend(0.58, 0.70, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader(region, "C")
    leg.AddEntry(h_total, "Total jets", "pe")
    leg.AddEntry(h_real,  "Real (HS)", "f")
    leg.AddEntry(h_fake,  "Fake (PU)", "f")
    leg.Draw()

    # Bottom pad
    pad2.cd()
    h_ratio.Draw("E1")

    line = ROOT.TLine(xmin, 0.5, xmax, 0.5)
    line.SetLineStyle(2)
    line.Draw()

    outpath = os.path.join(args.outdir, f"{region}/{var}_total_points_stack_real_fake.pdf")
    print(f"{outpath}")
    c.SaveAs(outpath)

    # Log version
    pad1.cd()
    pad1.SetLogy(1)
    stack.SetMinimum(0.1)
    h_total.SetMinimum(0.1)

    outpath_log = os.path.join(args.outdir, f"{region}/log/{var}_total_points_stack_real_fake_log.pdf")
    c.SaveAs(outpath_log)

    c.Close()

    # Cleanup
    h_total.Delete()
    h_real.Delete()
    h_fake.Delete()
    h_ratio.Delete()


def main():
    args = parse_args()
    ensure_dir(args.outdir)
    ensure_dir(args.outdir + "/log")

    print(f"[INFO] Pattern: {args.input}")
    files = sorted(glob.glob(args.input))
    if len(files) == 0:
        raise RuntimeError(f"No files matched pattern: {args.input}")

    print(f"[INFO] Files found: {len(files)}")
    # df = pd.read_parquet(files)
    all_needed_cols = set()

    for var in JET_ID_VARIABLES:
        prefix = jet_prefix(var)
        all_needed_cols.add(var)
        all_needed_cols.add(prefix + "pt_nominal")
        all_needed_cols.add(prefix + "eta_nominal")
        all_needed_cols.add(prefix + args.genmatch_suffix)

    all_needed_cols = sorted(all_needed_cols)

    print(f"[INFO] Reading {len(all_needed_cols)} columns from {len(files)} files")
    df = pd.read_parquet(files, columns=all_needed_cols)    

    if args.max_rows is not None:
        df = df.head(args.max_rows)

    print(f"[INFO] N rows: {len(df)}")

    regions = [args.region]


    for var in JET_ID_VARIABLES:
        for region in regions:
            ensure_dir(args.outdir + f"/{region}")
            ensure_dir(args.outdir + f"/{region}/log")
            plot_var(df, var, args, region=region)

    print(f"[DONE] Plots saved in: {args.outdir}")


if __name__ == "__main__":
    main()