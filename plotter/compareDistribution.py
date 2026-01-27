#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np

import awkward as ak
import dask
import dask.array as da
import dask_awkward as dak
from distributed import Client

import json

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2(True)

# ------------------------
# Config
# ------------------------
SAMPLES = {
    "ggH": [
        "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_28Nov_HEMVetoFix_NoSyst_V2/stage1_output/2018/compacted/ggh_powhegPS/*/*.parquet"
        # "/path/to/ggH/*.parquet",
        # "/path/to/ggH/part-*.parquet",
    ],
    "VBF": [
        "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_28Nov_HEMVetoFix_NoSyst_V2/stage1_output/2018/compacted/vbf_powheg_dipole/*/*.parquet"
        # "/path/to/VBF/*.parquet",
    ],
    "DY": [
        "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_28Nov_HEMVetoFix_NoSyst_V2/stage1_output/2018/compacted/dy_M-100To200_MiNNLO/*/*.parquet"
        # "/path/to/DY/*.parquet",
    ],
    "EWK": [
        "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_28Nov_HEMVetoFix_NoSyst_V2/stage1_output/2018/compacted/ewk_lljj_mll50_mjj120/*/*.parquet"
        # "/path/to/EWK/*.parquet",
    ],
    "top": [
        "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_28Nov_HEMVetoFix_NoSyst_V2/stage1_output/2018/compacted/ttjets_dl/*/*.parquet"
        # "/path/to/top/*.parquet",
    ],
}

VARIABLES = [
    "mu1_iso", "mu2_iso",
    "mu1_pt_over_mu2_pt", "mu1_eta_over_mu2_eta",
    "mu1_pt_roch", "mu1_pt_fsr", "mu2_pt_roch", "mu2_pt_fsr",
    "nmuons",
    "dimuon_dEta", "dimuon_dPhi", "dimuon_dR",
    "uncalibrated_dimuon_ebe_mass_res",
    "dimuon_pt_over_MET_pt", "dimuon_pt_over_jet1_pt", "dimuon_pt_over_jet2_pt",
    "mu1_pt_raw", "mu2_pt_raw",
    "mu1_dxy", "mu2_dxy", "mu1_dxyErr", "mu2_dxyErr", "mu1_dxybs", "mu2_dxybs",
    "mu1_dz", "mu2_dz", "mu1_dzErr", "mu2_dzErr",
    "mu1_ip3d", "mu2_ip3d", "mu1_sip3d", "mu2_sip3d",
    "mu1_highPurity", "mu2_highPurity", "mu1_inTimeMuon", "mu2_inTimeMuon",
    "mu1_isGlobal", "mu2_isGlobal", "mu1_isPFcand", "mu2_isPFcand",
    "mu1_isStandalone", "mu2_isStandalone", "mu1_isTracker", "mu2_isTracker",
    "mu1_looseId", "mu2_looseId", "mu1_mediumId", "mu2_mediumId",
    "mu1_mediumPromptId", "mu2_mediumPromptId",
    "mu1_tightCharge", "mu2_tightCharge",
    "mu1_pdgId", "mu2_pdgId",
    "mu1_miniIsoId", "mu2_miniIsoId",
    "mu1_miniPFRelIso_all", "mu2_miniPFRelIso_all",
    "mu1_miniPFRelIso_chg", "mu2_miniPFRelIso_chg",
    "mu1_multiIsoId", "mu2_multiIsoId",
    "mu1_pfIsoId", "mu2_pfIsoId",
    "mu1_pfRelIso03_all", "mu2_pfRelIso03_all",
    "mu1_pfRelIso03_chg", "mu2_pfRelIso03_chg",
    "mu1_pfRelIso04_all", "mu2_pfRelIso04_all",
    "mu1_puppiIsoId", "mu2_puppiIsoId",
    "mu1_tkIsoId", "mu2_tkIsoId",
    "mu1_tkRelIso", "mu2_tkRelIso",
    "mu1_nStations", "mu2_nStations",
    "mu1_nTrackerLayers", "mu2_nTrackerLayers",
    "mu1_segmentComp", "mu2_segmentComp",
    "mu1_jetIdx", "mu2_jetIdx",
    "mu1_jetNDauCharged", "mu2_jetNDauCharged",
    "mu1_jetPtRelv2", "mu2_jetPtRelv2",
    "mu1_jetRelIso", "mu2_jetRelIso",
    "mu1_svIdx", "mu2_svIdx",
    "mu12_pt_sum", "mu12_pt_diff", "mu12_pt_absdiff",
    "mu12_pt_prod", "mu12_pt_ratio12", "mu12_pt_ratio21",
    "mu12_pt_min", "mu12_pt_max", "mu12_pt_asym",
    "mu12_eta_sum", "mu12_eta_diff", "mu12_eta_absdiff", "mu12_eta_prod",
    "mu12_absEta_sum", "mu12_absEta_diff", "mu12_absEta_min", "mu12_absEta_max",
    "mu12_iso04_sum", "mu12_iso04_diff", "mu12_iso04_absdiff", "mu12_iso04_prod",
    "mu12_iso04_min", "mu12_iso04_max", "mu12_iso04_asym",
    "mu12_dxy_sum", "mu12_dxy_diff", "mu12_dxy_absdiff",
    "mu12_dz_sum", "mu12_dz_diff", "mu12_dz_absdiff",
    "mu12_sip3d_sum", "mu12_sip3d_diff", "mu12_sip3d_absdiff",
    "mu12_sip3d_prod", "mu12_sip3d_min", "mu12_sip3d_max",
    "mu12_nStations_min", "mu12_nStations_max", "mu12_nStations_sum",
    "mu12_nTrackerLayers_min", "mu12_nTrackerLayers_max", "mu12_nTrackerLayers_sum",
    "mu12_q1q2",
]


PLOT_SETTINGS_JSON = "/depot/cms/users/shar1172/copperheadV2_main/src/lib/histogram/plot_settings_vbfCat_MVA_input.json"

def build_bins_from_plot_settings(plot_settings_path: str, variables: list[str]) -> dict[str, np.ndarray]:
    with open(plot_settings_path, "r") as f:
        cfg = json.load(f)

    bins = {}
    missing = []

    for v in variables:
        if v not in cfg:
            missing.append(v)
            continue

        entry = cfg[v]

        # Priority: non-uniform binning if present
        if "binning_nonuniform" in entry and entry["binning_nonuniform"] is not None:
            edges = np.asarray(entry["binning_nonuniform"], dtype=float)

        # Otherwise: linspace-style [min, max, nbins]
        elif "binning_linspace" in entry and entry["binning_linspace"] is not None:
            xmin, xmax, nbins = entry["binning_linspace"]
            edges = np.linspace(float(xmin), float(xmax), int(nbins))

        else:
            missing.append(v)
            continue

        bins[v] = edges

    if missing:
        print(f"[WARNING] Missing binning for {len(missing)} variables in plot settings JSON:")
        for m in missing:
            print("  -", m)

    return bins

# Usage:
BINS = build_bins_from_plot_settings(PLOT_SETTINGS_JSON, VARIABLES)

COLORS = {
    "ggH": ROOT.kRed + 1,
    "VBF": ROOT.kAzure + 2,
    "DY":  ROOT.kGreen + 2,
    "EWK": ROOT.kMagenta + 1,
    "top": ROOT.kOrange + 7,
}

OUTDIR = "plots/shape_compare_parquet_unweighted"
os.makedirs(OUTDIR, exist_ok=True)

# ------------------------
# Helpers
# ------------------------
def read_parquet_dak(paths, columns):
    """
    Returns dask-awkward Array from parquet (handles list of paths/globs).
    """
    return dak.from_parquet(paths, columns=columns)



def to_flat_da(x, fill_value=np.nan):
    """
    Convert dask-awkward field to flat dask.array safely.
    - flattens jagged arrays
    - fills None (OptionType) -> fill_value
    - returns a 1D dask.array
    """
    # 1) flatten jagged
    try:
        if ak.is_list_type(x.type):
            x = dak.flatten(x)
    except Exception:
        # fallback string check
        if "var *" in str(x.type) or "[]" in str(x.type):
            x = dak.flatten(x)

    # 2) fix OptionType / missing values
    # If column is nullable (?float), fill None -> NaN (or 0)
    try:
        if ak.is_option_type(x.type):
            x = dak.fill_none(x, fill_value)
    except Exception:
        # safe fallback: always try fill_none (no-op if no None)
        x = dak.fill_none(x, fill_value)

    # 3) convert
    x_da = dak.to_dask_array(x)

    # 4) drop NaN / inf if float
    # (for int/bool it's fine; for floats this avoids histogram crashes)
    if np.issubdtype(x_da.dtype, np.floating):
        x_da = x_da[da.isfinite(x_da)]

    return x_da


def make_root_hist(name, edges, counts_np):
    h = ROOT.TH1D(name, name, len(edges) - 1, edges.astype(np.float64))
    for i, c in enumerate(counts_np, start=1):
        h.SetBinContent(i, float(c))
    return h


def normalize_unity(h):
    integ = h.Integral()
    if integ > 0:
        h.Scale(1.0 / integ)


# ------------------------
# Main
# ------------------------
from modules.dask_utils import get_dask_client
def main():
    # ---- Dask client (scale here) ----
    client = get_dask_client()
    print(client)

    # ---- Read parquet per process ----
    arrays = {}
    for proc, paths in SAMPLES.items():
        arrays[proc] = read_parquet_dak(paths, columns=VARIABLES)

    # ---- Build lazy histograms (no weights) ----
    hist_tasks = {}  # (var, proc) -> dask array counts
    for var in VARIABLES:
        edges = BINS[var]
        for proc, arr in arrays.items():
            x = to_flat_da(arr[var])
            # drop non-finite (important for ratios / dxy etc.)
            # x = da.where(da.isfinite(x), x, da.nan)
            # x = x[~da.isnan(x)]
            counts, _ = da.histogram(x, bins=edges)
            hist_tasks[(var, proc)] = counts

    # ---- Compute all in one go (this is where Dask scales) ----
    computed = dask.compute(hist_tasks)[0]  # dict with numpy arrays inside

    # ---- Plot overlays with ROOT ----
    for var in VARIABLES:
        edges = BINS[var]

        hists = {}
        for proc in ["ggH", "VBF", "DY", "EWK", "top"]:
            counts = np.asarray(computed[(var, proc)])
            h = make_root_hist(f"{var}_{proc}", edges, counts)
            normalize_unity(h)
            h.SetLineWidth(3)
            h.SetLineColor(COLORS.get(proc, ROOT.kBlack))
            hists[proc] = h

        c = ROOT.TCanvas(f"c_{var}", var, 900, 800)
        leg = ROOT.TLegend(0.65, 0.70, 0.88, 0.88)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)

        maxy = max(h.GetMaximum() for h in hists.values())
        first = True
        for proc in ["ggH", "VBF", "DY", "EWK", "top"]:
            h = hists[proc]
            h.GetYaxis().SetTitle("Normalized to unity")
            h.SetMaximum(1.3 * maxy)
            h.Draw("HIST" if first else "HIST SAME")
            leg.AddEntry(h, proc, "l")
            first = False

        leg.Draw()
        # c.SaveAs(f"{OUTDIR}/{var}.png")
        c.SaveAs(f"{OUTDIR}/{var}.pdf")

        # log info
        c.SetLogy()
        # c.SaveAs(f"{OUTDIR}/{var}_log.png")
        c.SaveAs(f"{OUTDIR}/{var}_log.pdf")

    client.close()


if __name__ == "__main__":
    main()
