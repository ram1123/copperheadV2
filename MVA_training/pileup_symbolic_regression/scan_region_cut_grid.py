#!/usr/bin/env python3
import os
import glob
import json
import argparse
import itertools
import numpy as np
import pandas as pd
import ROOT

ROOT.gROOT.SetBatch(True)

"""
python MVA_training/pileup_symbolic_regression/scan_region_cut_grid.py \
  -i "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn_Mar19_tightPassLepVeto_NoJER_pySR/stage1_output/2022postEE/compacted/dyTo2L_M-50_incl/0/part*.parquet" \
  -o validation/scan_cut_grid_jet12 \
  --variation nominal \
  --n-jets 2 \
  --pt-min 25 \
  --pt-max 80
"""


# =========================================================
# Helpers
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def infer_existing_files(pattern: str):
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No files matched: {pattern}")
    return files


def col(prefix: str, var: str, variation: str) -> str:
    return f"{prefix}{var}_{variation}"


def make_hist(name, title, nbins, xmin, xmax):
    h = ROOT.TH1F(name, title, nbins, xmin, xmax)
    h.Sumw2()
    h.SetStats(0)
    return h


def fill_hist(h, arr):
    for v in arr:
        if np.isfinite(v):
            h.Fill(float(v))


def normalize_hist(h):
    integ = h.Integral()
    if integ > 0:
        h.Scale(1.0 / integ)


def infer_range(x, fallback=(0.0, 1.0)):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return fallback
    lo = np.percentile(x, 0.5)
    hi = np.percentile(x, 99.5)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return fallback
    return float(lo), float(hi)


# =========================================================
# Build flat jet dataframe
# =========================================================
def build_jet_rows(df_in: pd.DataFrame, prefixes, variation: str, genmatch_mode: str):
    dfs = []

    for jidx, p in enumerate(prefixes):
        need = [
            col(p, "pt", variation),
            col(p, "eta", variation),
            col(p, "mass", variation),
            col(p, "muEF", variation),
            col(p, "rawFactor", variation),
            col(p, "nConstituents", variation),
            col(p, "hasMatchedGenJet", variation),
        ]
        missing = [c for c in need if c not in df_in.columns]
        if missing:
            print(f"[WARN] Skip {p}: missing columns {missing}")
            continue

        pt = pd.to_numeric(df_in[col(p, "pt", variation)], errors="coerce").astype(np.float32).to_numpy()
        eta = pd.to_numeric(df_in[col(p, "eta", variation)], errors="coerce").astype(np.float32).to_numpy()
        mass = pd.to_numeric(df_in[col(p, "mass", variation)], errors="coerce").astype(np.float32).to_numpy()
        muEF = pd.to_numeric(df_in[col(p, "muEF", variation)], errors="coerce").astype(np.float32).to_numpy()
        rawFactor = pd.to_numeric(df_in[col(p, "rawFactor", variation)], errors="coerce").astype(np.float32).to_numpy()
        nconst = pd.to_numeric(df_in[col(p, "nConstituents", variation)], errors="coerce").astype(np.float32).to_numpy()
        gm = pd.to_numeric(df_in[col(p, "hasMatchedGenJet", variation)], errors="coerce").to_numpy()

        exists = np.isfinite(pt) & np.isfinite(eta) & (pt > 0)

        if genmatch_mode == "hasMatchedGenJet":
            real = gm == 1
            fake = gm == 0
        else:
            real = gm.astype(bool)
            fake = ~real

        out = pd.DataFrame({
            "pt": pt,
            "eta": eta,
            "aeta": np.abs(eta),
            "mass": mass,
            "muEF": muEF,
            "rawFactor": rawFactor,
            "nConstituents": nconst,
            "is_real": real,
            "is_fake": fake,
            "prefix": p,
            "jidx": jidx,
        })

        out = out[exists].copy()
        out = out.replace([np.inf, -np.inf], np.nan)
        dfs.append(out)

    if not dfs:
        raise RuntimeError("No valid jet columns found.")

    dfj = pd.concat(dfs, ignore_index=True)
    return dfj


# =========================================================
# Region masks
# =========================================================
def region_mask(dfj: pd.DataFrame, region: str):
    eta = dfj["eta"].to_numpy()
    aeta = np.abs(eta)

    if region == "HEpos":
        return (eta >= 2.5) & (eta < 3.0)
    elif region == "HEneg":
        return (eta <= -2.5) & (eta > -3.0)
    elif region == "HFpos":
        return eta >= 3.0
    elif region == "HFneg":
        return eta <= -3.0
    elif region == "central":
        return aeta < 2.5
    elif region == "inclusive":
        return np.isfinite(eta)
    else:
        raise ValueError(region)


# =========================================================
# Metrics
# =========================================================
def apply_cut(df, thr_nconst=None, thr_mass=None, thr_muef=None, thr_rawFactor=None):
    mask = np.ones(len(df), dtype=bool)

    if thr_nconst is not None:
        mask &= df["nConstituents"].to_numpy() >= thr_nconst
    if thr_mass is not None:
        mask &= df["mass"].to_numpy() >= thr_mass
    if thr_muef is not None:
        mask &= df["muEF"].to_numpy() <= thr_muef
    if thr_rawFactor is not None:
        mask &= df["rawFactor"].to_numpy() >= thr_rawFactor        

    mask &= np.isfinite(df["pt"].to_numpy())
    return mask


def compute_wp_vs_pt(df, pass_mask, pt_bins):
    pt = df["pt"].to_numpy()
    real = df["is_real"].to_numpy().astype(bool)
    fake = df["is_fake"].to_numpy().astype(bool)

    hs_eff = []
    pu_rej = []

    for lo, hi in zip(pt_bins[:-1], pt_bins[1:]):
        m = (pt >= lo) & (pt < hi)

        hs_all = np.sum(m & real)
        hs_pass = np.sum(m & real & pass_mask)

        pu_all = np.sum(m & fake)
        pu_pass = np.sum(m & fake & pass_mask)

        eff = np.nan if hs_all == 0 else hs_pass / hs_all
        rej = np.nan if pu_all == 0 else 1.0 - (pu_pass / pu_all)

        hs_eff.append(eff)
        pu_rej.append(rej)

    return np.array(hs_eff, dtype=float), np.array(pu_rej, dtype=float)


def compute_overall_metrics(df, pass_mask):
    real = df["is_real"].to_numpy().astype(bool)
    fake = df["is_fake"].to_numpy().astype(bool)

    hs_all = np.sum(real)
    hs_pass = np.sum(real & pass_mask)

    pu_all = np.sum(fake)
    pu_pass = np.sum(fake & pass_mask)

    hs_eff = np.nan if hs_all == 0 else hs_pass / hs_all
    pu_rej = np.nan if pu_all == 0 else 1.0 - (pu_pass / pu_all)

    return hs_eff, pu_rej


# =========================================================
# Plots
# =========================================================
def make_tgraph_finite(x, y, name, title, xlab, ylab):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() == 0:
        return None

    g = ROOT.TGraph(int(m.sum()), x[m], y[m])
    g.SetName(name)
    g.SetTitle(f"{title};{xlab};{ylab}")
    g.SetLineWidth(2)
    g.SetMarkerStyle(20)
    g.SetMarkerSize(1.0)
    return g


def plot_wp_vs_pt(df, pass_mask, pt_bins, outdir, tag):
    centers = 0.5 * (np.array(pt_bins[:-1]) + np.array(pt_bins[1:]))

    hs_eff, pu_rej = compute_wp_vs_pt(df, pass_mask, pt_bins)

    c = ROOT.TCanvas(f"c_eff_rej_{tag}", "", 800, 650)

    # --- HS efficiency (left axis)
    g_eff = make_tgraph_finite(
        centers, hs_eff,
        f"g_eff_{tag}",
        "",
        "jet p_{T} [GeV]", "HS efficiency"
    )

    # --- PU rejection (right axis)
    g_rej = make_tgraph_finite(
        centers, pu_rej,
        f"g_rej_{tag}",
        "",
        "", ""
    )

    if g_eff:
        g_eff.SetLineColor(ROOT.kBlue + 1)
        g_eff.SetMarkerColor(ROOT.kBlue + 1)
        g_eff.SetLineWidth(2)

        g_eff.Draw("APL")
        g_eff.GetYaxis().SetRangeUser(0.0, 1.0)
        g_eff.GetYaxis().SetTitle("HS efficiency")

    if g_rej:
        g_rej.SetLineColor(ROOT.kRed + 1)
        g_rej.SetMarkerColor(ROOT.kRed + 1)
        g_rej.SetLineWidth(2)

        g_rej.Draw("PL SAME")

    # --- Right axis
    axis = ROOT.TGaxis(
        ROOT.gPad.GetUxmax(),
        ROOT.gPad.GetUymin(),
        ROOT.gPad.GetUxmax(),
        ROOT.gPad.GetUymax(),
        0.0, 1.0, 510, "+L"
    )
    axis.SetTitle("PU rejection")
    axis.SetLineColor(ROOT.kRed + 1)
    axis.SetLabelColor(ROOT.kRed + 1)
    axis.SetTitleColor(ROOT.kRed + 1)
    axis.Draw()

    # --- Legend
    leg = ROOT.TLegend(0.55, 0.75, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    if g_eff:
        leg.AddEntry(g_eff, "HS efficiency", "lp")
    if g_rej:
        leg.AddEntry(g_rej, "PU rejection", "lp")
    leg.Draw()

    # --- Save
    c.SaveAs(os.path.join(outdir, f"eff_and_rej_vs_pt_{tag}.pdf"))


def plot_pu_rej_vs_hs_eff(df, pass_mask, pt_bins, outdir, tag):
    hs_eff, pu_rej = compute_wp_vs_pt(df, pass_mask, pt_bins)

    # keep only finite points
    m = np.isfinite(hs_eff) & np.isfinite(pu_rej)
    x = hs_eff[m]
    y = pu_rej[m]

    if len(x) == 0:
        print(f"[WARN] No finite ROC-like points for {tag}")
        return

    g = ROOT.TGraph(len(x), np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    g.SetName(f"g_roc_like_{tag}")
    g.SetTitle(f"PU rejection vs HS efficiency [{tag}];HS efficiency;PU rejection")
    g.SetMarkerStyle(20)
    g.SetMarkerSize(1.1)
    g.SetLineWidth(2)

    c = ROOT.TCanvas(f"c_roc_like_{tag}", "", 800, 650)
    g.Draw("APL")
    g.GetXaxis().SetLimits(0.0, 1.0)
    g.GetYaxis().SetRangeUser(0.0, 1.0)

    # optional labels with pT bin ranges
    labels = []
    for i, (lo, hi) in enumerate(zip(pt_bins[:-1], pt_bins[1:])):
        if not m[i]:
            continue
        lab = ROOT.TLatex(float(hs_eff[i]) + 0.01, float(pu_rej[i]) + 0.01, f"{int(lo)}-{int(hi)}")
        lab.SetTextSize(0.025)
        lab.Draw()
        labels.append(lab)

    c.SaveAs(os.path.join(outdir, f"pu_rej_vs_hs_eff_{tag}.pdf"))


def plot_eta_before_after(df, pass_mask, outdir, tag):
    eta = df["eta"].to_numpy()

    h_before = make_hist(f"h_eta_before_{tag}", "Jet #eta before/after cut;jet #eta;Normalized entries", 120, -5, 5)
    h_after  = make_hist(f"h_eta_after_{tag}",  "Jet #eta before/after cut;jet #eta;Normalized entries", 120, -5, 5)

    fill_hist(h_before, eta[np.isfinite(eta)])
    fill_hist(h_after, eta[pass_mask & np.isfinite(eta)])

    # normalize_hist(h_before)
    # normalize_hist(h_after)

    h_before.SetLineColor(ROOT.kBlack)
    h_before.SetLineWidth(2)
    h_after.SetLineColor(ROOT.kRed + 1)
    h_after.SetLineWidth(2)
    h_after.SetLineStyle(2)

    c = ROOT.TCanvas(f"c_eta_{tag}", "", 800, 800)

    pad1 = ROOT.TPad(f"pad1_eta_{tag}", "", 0.0, 0.30, 1.0, 1.0)
    pad2 = ROOT.TPad(f"pad2_eta_{tag}", "", 0.0, 0.00, 1.0, 0.30)
    pad1.SetLeftMargin(0.12)
    pad1.SetBottomMargin(0.02)
    pad2.SetLeftMargin(0.12)
    pad2.SetBottomMargin(0.30)
    pad1.Draw()
    pad2.Draw()

    pad1.cd()
    ymax = max(h_before.GetMaximum(), h_after.GetMaximum()) * 1.25
    h_before.SetMaximum(ymax)
    h_before.Draw("hist")
    h_after.Draw("hist same")

    leg = ROOT.TLegend(0.62, 0.78, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(h_before, "Before cut", "l")
    leg.AddEntry(h_after, "After cut", "l")
    leg.Draw()

    pad2.cd()
    h_ratio = h_after.Clone(f"h_ratio_eta_{tag}")
    h_ratio.Divide(h_before)
    h_ratio.SetTitle("")
    h_ratio.GetYaxis().SetTitle("After/Before")
    h_ratio.GetXaxis().SetTitle("jet #eta")
    h_ratio.GetYaxis().SetRangeUser(0.0, 1.2)
    h_ratio.Draw("hist")

    line = ROOT.TLine(-5, 1.0, 5, 1.0)
    line.SetLineStyle(2)
    line.Draw()

    c.SaveAs(os.path.join(outdir, f"eta_before_after_{tag}.pdf"))


# =========================================================
# Scan
# =========================================================
def run_scan_region(dfj_region, region_name, outdir, pt_bins, nconst_grid, mass_grid, muef_grid, rawFactor_grid):
    ensure_dir(outdir)

    summary = []
    best_by_purej = None

    # 1D scans
    scan_configs = []

    for n in nconst_grid:
        scan_configs.append({"thr_nconst": n, "thr_mass": None, "thr_muef": None, "thr_rawFactor": None, "mode": "nconst"})
    for m in mass_grid:
        scan_configs.append({"thr_nconst": None, "thr_mass": m, "thr_muef": None, "thr_rawFactor": None, "mode": "mass"})
    for u in muef_grid:
        scan_configs.append({"thr_nconst": None, "thr_mass": None, "thr_muef": u, "thr_rawFactor": None, "mode": "muef"})
    for u in rawFactor_grid:
        scan_configs.append({"thr_nconst": None, "thr_mass": None, "thr_muef": None, "thr_rawFactor": u, "mode": "rawFactor"})


    # 2D scans
    for n, m in itertools.product(nconst_grid, mass_grid):
        scan_configs.append({"thr_nconst": n, "thr_mass": m, "thr_muef": None, "thr_rawFactor": None, "mode": "nconst_mass"})
    for n, u in itertools.product(nconst_grid, muef_grid):
        scan_configs.append({"thr_nconst": n, "thr_mass": None, "thr_muef": u, "thr_rawFactor": None, "mode": "nconst_muef"})
    for m, u in itertools.product(mass_grid, muef_grid):
        scan_configs.append({"thr_nconst": None, "thr_mass": m, "thr_muef": u, "thr_rawFactor": None, "mode": "mass_muef"})
    for n, m in itertools.product(nconst_grid, rawFactor_grid):
        scan_configs.append({"thr_nconst": n, "thr_mass": None, "thr_muef": None, "thr_rawFactor": m, "mode": "nconst_mass"})
    for n, u in itertools.product(muef_grid, rawFactor_grid):
        scan_configs.append({"thr_nconst": None, "thr_mass": None, "thr_muef": n, "thr_rawFactor": u, "mode": "nconst_muef"})
    for m, u in itertools.product(mass_grid, rawFactor_grid):
        scan_configs.append({"thr_nconst": None, "thr_mass": m, "thr_muef": None, "thr_rawFactor": u, "mode": "mass_muef"})


    # 3D scans
    for n, m, u in itertools.product(nconst_grid, mass_grid, muef_grid):
        scan_configs.append({"thr_nconst": n, "thr_mass": m, "thr_muef": u, "thr_rawFactor": None, "mode": "nconst_mass_muef"})

    print(f"[INFO] {region_name}: scanning {len(scan_configs)} points")

    for i, cfg in enumerate(scan_configs):
        pass_mask = apply_cut(
            dfj_region,
            thr_nconst=cfg["thr_nconst"],
            thr_mass=cfg["thr_mass"],
            thr_muef=cfg["thr_muef"],
            thr_rawFactor=cfg["thr_rawFactor"],
        )

        hs_eff_all, pu_rej_all = compute_overall_metrics(dfj_region, pass_mask)
        hs_eff_vs_pt, pu_rej_vs_pt = compute_wp_vs_pt(dfj_region, pass_mask, pt_bins)

        rec = {
            "region": region_name,
            "scan_index": i,
            "mode": cfg["mode"],
            "thr_nconst": cfg["thr_nconst"],
            "thr_mass": cfg["thr_mass"],
            "thr_muef": cfg["thr_muef"],
            "thr_rawFactor": cfg["thr_rawFactor"],
            "hs_eff_all": None if np.isnan(hs_eff_all) else float(hs_eff_all),
            "pu_rej_all": None if np.isnan(pu_rej_all) else float(pu_rej_all),
            "hs_eff_vs_pt": [None if np.isnan(x) else float(x) for x in hs_eff_vs_pt],
            "pu_rej_vs_pt": [None if np.isnan(x) else float(x) for x in pu_rej_vs_pt],
        }
        summary.append(rec)

        # choose best by PU rejection with a minimum HS efficiency constraint
        if np.isfinite(hs_eff_all) and np.isfinite(pu_rej_all):
            if hs_eff_all >= 0.80:
                if best_by_purej is None or pu_rej_all > best_by_purej["pu_rej_all"]:
                    best_by_purej = {
                        **rec,
                        "pass_mask": pass_mask.copy(),
                    }

    with open(os.path.join(outdir, f"scan_summary_{region_name}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if best_by_purej is not None:
        tag = (
            f"{region_name}_best_"
            f"{best_by_purej['mode']}_"
            f"n{best_by_purej['thr_nconst']}_"
            f"m{best_by_purej['thr_mass']}_"
            f"u{best_by_purej['thr_muef']}_"
            f"u{best_by_purej['thr_rawFactor']}"
        ).replace("None", "X").replace(".", "p")

        plot_wp_vs_pt(dfj_region, best_by_purej["pass_mask"], pt_bins, outdir, tag)
        plot_eta_before_after(dfj_region, best_by_purej["pass_mask"], outdir, tag)
        plot_pu_rej_vs_hs_eff(dfj_region, best_by_purej["pass_mask"], pt_bins, outdir, tag)

        best_to_save = dict(best_by_purej)
        del best_to_save["pass_mask"]

        with open(os.path.join(outdir, f"best_point_{region_name}.json"), "w") as f:
            json.dump(best_to_save, f, indent=2)

        print(f"[INFO] Best for {region_name}:")
        print(json.dumps(best_to_save, indent=2))
    else:
        print(f"[WARN] No valid best point found for {region_name} with HS eff >= 0.80")


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help='Input parquet or wildcard, e.g. "/path/part*.parquet"')
    ap.add_argument("-o", "--outdir", default="validation/scan_cut_grid")
    ap.add_argument("--variation", default="nominal")
    ap.add_argument("--n-jets", type=int, default=2)
    ap.add_argument("--jet-prefix-base", default="jet")
    ap.add_argument("--genmatch-mode", default="hasMatchedGenJet", choices=["hasMatchedGenJet", "bool"])
    ap.add_argument("--pt-min", type=float, default=25.0)
    ap.add_argument("--pt-max", type=float, default=80.0)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    files = infer_existing_files(args.input)
    print(f"[INFO] Files found: {len(files)}")

    prefixes = [f"{args.jet_prefix_base}{i}_" for i in range(1, max(1, min(args.n_jets, 4)) + 1)]

    needed_cols = []
    for p in prefixes:
        needed_cols.extend([
            col(p, "pt", args.variation),
            col(p, "eta", args.variation),
            col(p, "mass", args.variation),
            col(p, "muEF", args.variation),
            col(p, "rawFactor", args.variation),
            col(p, "nConstituents", args.variation),
            col(p, "hasMatchedGenJet", args.variation),
        ])
    needed_cols = sorted(set(needed_cols))

    df_in = pd.read_parquet(files, columns=needed_cols)
    if args.max_rows is not None:
        df_in = df_in.head(args.max_rows)

    print(f"[INFO] Rows read: {len(df_in)}")

    dfj = build_jet_rows(df_in, prefixes, args.variation, args.genmatch_mode)
    dfj = dfj[(dfj["pt"] >= args.pt_min) & (dfj["pt"] < args.pt_max)].copy()

    pt_bins = [25, 30, 35, 40, 50, 60, 70, 80]

    # Main grids
    nconst_grid = [1, 2, 3, 4, 5, 6, 8, 10]
    mass_grid   = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    muef_grid   = [0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
    rawFactor_grid = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]


    regions = ["HEpos", "HEneg", "HFpos", "HFneg"]

    for region in regions:
        mask = region_mask(dfj, region)
        df_reg = dfj.loc[mask].copy()

        print(f"[INFO] Region {region}: N={len(df_reg)} real={int(df_reg['is_real'].sum())} fake={int(df_reg['is_fake'].sum())}")

        if len(df_reg) < 100:
            print(f"[WARN] Too few jets in {region}, skipping.")
            continue

        run_scan_region(
            df_reg,
            region,
            os.path.join(args.outdir, region),
            pt_bins=pt_bins,
            nconst_grid=nconst_grid,
            mass_grid=mass_grid,
            muef_grid=muef_grid,
            rawFactor_grid=rawFactor_grid,
        )


if __name__ == "__main__":
    main()