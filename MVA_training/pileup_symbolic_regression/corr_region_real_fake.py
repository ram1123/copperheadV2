#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import pandas as pd
import ROOT
from itertools import combinations
from tqdm import tqdm

from modules.root_2dColorProfile import set_gradient_style

ROOT.gROOT.SetBatch(True)

"""
time python MVA_training/pileup_symbolic_regression/corr_region_real_fake.py   \
    -i "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn_Mar19_tightPassLepVeto_NoJER_pySR/stage1_output/2022postEE/compacted/dyTo2L_M-50_incl/0/part*.parquet"   \
    -o validation/corr_jet1_v2   \
    --prefix jet1_

time python MVA_training/pileup_symbolic_regression/corr_region_real_fake.py   \
    -i "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn25GeV_Apr03_tightPassLepVeto_NoJER_JetIDFix/stage1_output/2022postEE/compacted/dyTo2L_M-50_incl/0/part0*.parquet"  \
    -o validation/corr_JetIDFix_jet_   \
    --prefix jet1_  \
    --apply-cleaning
time python MVA_training/pileup_symbolic_regression/corr_region_real_fake.py   \
    -i "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn25GeV_Apr03_tightPassLepVeto_NoJER_JetIDFix/stage1_output/2022postEE/compacted/dyTo2L_M-50_incl/0/part0*.parquet"  \
    -o validation/corr_JetIDFix_jet_   \
    --prefix jet2_  \
    --apply-cleaning

# 08 April 2026
time python MVA_training/pileup_symbolic_regression/corr_region_real_fake.py   \
    -i "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJets_Feb23_tightPassLepVeto_NoJER/stage1_output/2022postEE/compacted/dyTo2L_M-50_incl/0/part0*.parquet"  \
    -o validation/corr_BeforeJetIDFix   \
    --prefix jet1_  
time python MVA_training/pileup_symbolic_regression/corr_region_real_fake.py   \
    -i "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJets_Feb23_tightPassLepVeto_NoJER/stage1_output/2022postEE/compacted/dyTo2L_M-50_incl/0/part0*.parquet"  \
    -o validation/corr_BeforeJetIDFix   \
    --prefix jet2_  

time python MVA_training/pileup_symbolic_regression/corr_region_real_fake.py   \
    -i "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn25GeV_Apr03_tightPassLepVeto_NoJER_JetIDFix/stage1_output/2022postEE/compacted/dyTo2L_M-50_incl/0/part0*.parquet"  \
    -o validation/corr_AfterJetIDFix   \
    --prefix jet1_
time python MVA_training/pileup_symbolic_regression/corr_region_real_fake.py   \
    -i "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn25GeV_Apr03_tightPassLepVeto_NoJER_JetIDFix/stage1_output/2022postEE/compacted/dyTo2L_M-50_incl/0/part0*.parquet"  \
    -o validation/corr_AfterJetIDFix   \
    --prefix jet2_
"""

def infer_existing_files(pattern: str):
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No files matched: {pattern}")
    return files


def load_columns(files, cols):
    return pd.read_parquet(files, columns=cols)


def region_masks(df, eta_col):
    aeta = np.abs(df[eta_col].to_numpy(dtype=np.float32))
    eta = (df[eta_col].to_numpy(dtype=np.float32))
    return {
        # "inclusive": np.isfinite(aeta),
        # "central": (aeta < 2.5),
        # "HE": (aeta >= 2.5) & (aeta < 3.0),
        "HEpos": (eta >= 2.5) & (eta < 3.0),
        "HEneg": (eta <= -2.5) & (eta > -3.0),        
        # "HF": (aeta >= 3.0),
        "HFpos": (eta >= 3.0),
        "HFneg": (eta <= -3.0),
    }

def real_fake_masks(df, genmatch_col, genmatch_mode):
    gen = df[genmatch_col].to_numpy()

    if genmatch_mode == "bool":
        real = gen.astype(bool)
        fake = ~real
    elif genmatch_mode == "hasMatchedGenJet":
        real = (gen == 1)
        fake = (gen == 0)
    else:
        raise ValueError(f"Unknown genmatch_mode: {genmatch_mode}")

    return real, fake


def clean_numeric(df, cols):
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def short_label(colname: str, prefix: str, variation: str):
    s = colname
    if s.startswith(prefix):
        s = s[len(prefix):]
    suffix = f"_{variation}"
    if s.endswith(suffix):
        s = s[: -len(suffix)]
    return s


def draw_corr_heatmap(corr: pd.DataFrame, out_pdf: str, title: str, prefix: str, variation: str):
    labels = [short_label(c, prefix, variation) for c in corr.columns]
    n = len(labels)

    h = ROOT.TH2F("hcorr", f"{title}; ; ", n, 0, n, n, 0, n)

    for ix, xlab in enumerate(labels, start=1):
        h.GetXaxis().SetBinLabel(ix, xlab)
        h.GetYaxis().SetBinLabel(ix, labels[n - ix])

    for i in range(n):
        for j in range(n):
            val = float(corr.iloc[n - 1 - j, i])
            h.SetBinContent(i + 1, j + 1, val)

    c = ROOT.TCanvas("c_corr", "", 900, 800)
    c.SetLeftMargin(0.18)
    c.SetRightMargin(0.15)
    c.SetBottomMargin(0.18)
    c.SetTopMargin(0.08)

    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPaintTextFormat(".2f")

    h.SetMinimum(-1.0)
    h.SetMaximum(1.0)
    h.GetXaxis().LabelsOption("v")
    h.GetXaxis().SetLabelSize(0.035)
    h.GetYaxis().SetLabelSize(0.035)
    h.GetZaxis().SetTitle("Correlation")

    h.Draw("COLZ TEXT")
    c.SaveAs(out_pdf)
    print(f"[Saved] {out_pdf}")


def save_corr_and_heatmap(df_sub, out_base, label, prefix, variation):
    if len(df_sub) < 2:
        print(f"[WARN] Too few rows for {label}: {len(df_sub)}")
        return

    methods = ["pearson", "spearman"]
    for method in methods:
        corr = df_sub.corr(method=method)

        out_csv = f"{out_base}_{method}.csv"
        out_pdf = f"{out_base}_{method}.pdf"

        corr.to_csv(out_csv)
        print(f"[Saved] {out_csv}")

        draw_corr_heatmap(
            corr,
            out_pdf=out_pdf,
            title=f"{label.replace('_', ' ')} ({method})",
            prefix=prefix,
            variation=variation,
        )

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

def infer_range_2d(x, Var, fallback=(0.0, 1.0, 50)):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return fallback
    lo = np.percentile(x, 0.00001)
    hi = np.percentile(x, 99.9999)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return fallback
    if "nConstituents" in Var:
        return int(lo), 10, int(10-int(lo))
    if "mass" in Var: hi = 15.0
    if "pt" in Var: hi = 50.0
    if "rawFactor" in Var: lo, hi = 0.2, 0.5
    return float(lo), float(hi), infer_nbins(x, float(lo), float(hi))


def draw_2d_plot(df_sub, xcol, ycol, out_pdf, title, xVar, yVar, prefix, variation, nbins=50):
    x = pd.to_numeric(df_sub[xcol], errors="coerce").to_numpy(dtype=np.float32)
    y = pd.to_numeric(df_sub[ycol], errors="coerce").to_numpy(dtype=np.float32)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if len(x) < 10:
        print(f"[WARN] Too few points for {title}")
        return

    xmin, xmax, xnbins = infer_range_2d(x, xVar)
    ymin, ymax, ynbins = infer_range_2d(y, yVar)


    print(f"x: {xmin}, {xmax}, {xnbins}")
    print(f"y: {ymin}, {ymax}, {ynbins}")

    def short_label(colname):
        s = colname
        if s.startswith(prefix):
            s = s[len(prefix):]
        suffix = f"_{variation}"
        if s.endswith(suffix):
            s = s[:-len(suffix)]
        return s

    xlab = short_label(xcol)
    ylab = short_label(ycol)

    h2 = ROOT.TH2F(
        "h2",
        f"{title};{xlab};{ylab}",
        xnbins, xmin, xmax,
        ynbins, ymin, ymax
    )
    h2.SetStats(0)

    for xv, yv in zip(x, y):
        h2.Fill(float(xv), float(yv))

    c = ROOT.TCanvas("c2", "", 850, 750)
    c.SetLeftMargin(0.12)
    c.SetRightMargin(0.15)
    c.SetBottomMargin(0.12)
    h2.Draw("COLZ")
    c.SaveAs(out_pdf)
    print(f"[Saved] {out_pdf}")


def draw_2d_ratio_plot(
    df_real,
    df_fake,
    xcol,
    ycol,
    out_pdf,
    title,
    xVar,
    yVar,
    prefix,
    variation,
):
    x_real = pd.to_numeric(df_real[xcol], errors="coerce").to_numpy(dtype=np.float32)
    y_real = pd.to_numeric(df_real[ycol], errors="coerce").to_numpy(dtype=np.float32)

    x_fake = pd.to_numeric(df_fake[xcol], errors="coerce").to_numpy(dtype=np.float32)
    y_fake = pd.to_numeric(df_fake[ycol], errors="coerce").to_numpy(dtype=np.float32)

    m_real = np.isfinite(x_real) & np.isfinite(y_real)
    m_fake = np.isfinite(x_fake) & np.isfinite(y_fake)

    x_real, y_real = x_real[m_real], y_real[m_real]
    x_fake, y_fake = x_fake[m_fake], y_fake[m_fake]

    if len(x_real) + len(x_fake) < 10:
        print(f"[WARN] Too few events for ratio plot: {title}")
        return

    xmin, xmax, xnbins = infer_range_2d(
        np.concatenate([x_real, x_fake]), xVar
    )
    ymin, ymax, ynbins = infer_range_2d(
        np.concatenate([y_real, y_fake]), yVar
    )

    def short_label(colname):
        s = colname
        if s.startswith(prefix):
            s = s[len(prefix):]
        suffix = f"_{variation}"
        if s.endswith(suffix):
            s = s[:-len(suffix)]
        return s

    xlab = short_label(xcol)
    ylab = short_label(ycol)

    h_real = ROOT.TH2F("h_real", "", xnbins, xmin, xmax, ynbins, ymin, ymax)
    h_fake = ROOT.TH2F("h_fake", "", xnbins, xmin, xmax, ynbins, ymin, ymax)

    for xv, yv in zip(x_real, y_real):
        h_real.Fill(float(xv), float(yv))

    for xv, yv in zip(x_fake, y_fake):
        h_fake.Fill(float(xv), float(yv))

    h_ratio = ROOT.TH2F(
        "h_ratio",
        f"{title};{xlab};{ylab}",
        xnbins, xmin, xmax,
        ynbins, ymin, ymax,
    )

    for ix in range(1, xnbins + 1):
        for iy in range(1, ynbins + 1):
            n_real = h_real.GetBinContent(ix, iy)
            n_fake = h_fake.GetBinContent(ix, iy)
            denom = n_real + n_fake
            if denom > 0:
                h_ratio.SetBinContent(ix, iy, n_fake / denom)
            else:
                h_ratio.SetBinContent(ix, iy, 0.0)

    c = ROOT.TCanvas("c_ratio", "", 850, 750)
    c.SetLeftMargin(0.12)
    c.SetRightMargin(0.15)
    c.SetBottomMargin(0.12)

    h_ratio.SetStats(0)
    h_ratio.GetZaxis().SetTitle("Fake / (Real + Fake)")
    h_ratio.SetMinimum(0.0)
    h_ratio.SetMaximum(1.0)

    h_ratio.Draw("COLZ")
    c.SaveAs(out_pdf)

    print(f"[Saved] {out_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help='Parquet file or wildcard, e.g. "/path/part*.parquet"')
    ap.add_argument("-o", "--outdir", default="correlation_outputs")
    ap.add_argument("--prefix", default="jet1_", choices=["jet1_", "jet2_"])
    ap.add_argument("--variation", default="nominal")
    ap.add_argument("--genmatch-suffix", default="hasMatchedGenJet_nominal")
    ap.add_argument("--genmatch-mode", default="hasMatchedGenJet",
                    choices=["hasMatchedGenJet", "bool"])
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument(
        "--apply-cleaning",
        action="store_true",
        help="Apply region-specific cleaning (HE/HF fake rejection)"
    )    
    args = ap.parse_args()

    set_gradient_style()

    os.makedirs(args.outdir, exist_ok=True)

    vars_short = [
        "pt", "eta", "phi", "chEmEF", "neEmEF", "neHEF", "muEF",
        "hfcentralEtaStripSize", "hfadjacentEtaStripsSize",
        "hfsigmaEtaEta", "hfsigmaPhiPhi", "rawFactor", "area", "mass", "nConstituents"
    ]    
    # vars_short = ["muEF", "nConstituents", "pt", "eta"]
    cols = [f"{args.prefix}{v}_{args.variation}" for v in vars_short]

    eta_col = f"{args.prefix}eta_{args.variation}"
    genmatch_col = (
        f"{args.prefix}{args.genmatch_suffix}"
        if not args.genmatch_suffix.startswith(args.prefix)
        else args.genmatch_suffix
    )

    needed = sorted(set(cols + [eta_col, genmatch_col]))

    files = infer_existing_files(args.input)
    print(f"[INFO] Files found: {len(files)}")
    print(f"[INFO] Reading columns: {needed}")

    df = load_columns(files, needed)

    if args.max_rows is not None:
        df = df.head(args.max_rows)

    df_num = clean_numeric(df, cols)

    reg_masks = region_masks(df, eta_col)
    real_mask, fake_mask = real_fake_masks(df, genmatch_col, args.genmatch_mode)

    variables = [
        "pt", "eta", "phi", "chEmEF", "neEmEF", "neHEF", "muEF",
        "hfcentralEtaStripSize", "hfadjacentEtaStripsSize",
        "hfsigmaEtaEta", "hfsigmaPhiPhi", "rawFactor", "area", "mass", "nConstituents"
    ]

    # Generate all unique pairs (combinations of 2)
    pairs = list(combinations(variables, 2))

    print(pairs)
    for i, (var1, var2) in enumerate(pairs):
        print(f'{i:4} ("{var1}", "{var2}"),')
    

    for region_name, reg_mask in tqdm(reg_masks.items(), desc="Processing regions"):
        if args.apply_cleaning:
            print("[INFO] Cleaning ENABLED")
            # ---------------------------------------------------------
            # Apply HE-only cleaning
            # ---------------------------------------------------------
            if region_name in ["HEpos", "HEneg"]:

                nC_col = f"{args.prefix}nConstituents_{args.variation}"
                pt_col = f"{args.prefix}pt_{args.variation}"
                area_col = f"{args.prefix}area_{args.variation}"

                nC = pd.to_numeric(df_num[nC_col], errors="coerce")
                pt = pd.to_numeric(df_num[pt_col], errors="coerce")
                area = pd.to_numeric(df_num[area_col], errors="coerce")

                reject_mask = (nC < 4) & (pt < 40.0)
                reject_mask_area = (area > 0.56)

                he_clean_mask = reg_mask & ~(reject_mask) & (~reject_mask_area)

                print(
                    f"[INFO] HE cleaning {region_name}: "
                    f"{int(reg_mask.sum())} → {int(he_clean_mask.sum())}"
                )

                reg_mask = he_clean_mask
            if region_name in ["HFpos", "HFneg"]:

                nC_col = f"{args.prefix}nConstituents_{args.variation}"
                pt_col = f"{args.prefix}pt_{args.variation}"
                area_col = f"{args.prefix}area_{args.variation}"

                nC = pd.to_numeric(df_num[nC_col], errors="coerce")
                pt = pd.to_numeric(df_num[pt_col], errors="coerce")
                area = pd.to_numeric(df_num[area_col], errors="coerce")

                reject_mask = (nC <= 5) & (area > 0.56)

                hf_clean_mask = reg_mask & ~(reject_mask)

                print(
                    f"[INFO] HE cleaning {region_name}: "
                    f"{int(reg_mask.sum())} → {int(hf_clean_mask.sum())}"
                )

                reg_mask = hf_clean_mask            
        else:
            print("[INFO] Cleaning DISABLED")

        groups = {
            "all": reg_mask,
            "real": reg_mask & real_mask,
            "fake": reg_mask & fake_mask,
        }

        for kind, mask in groups.items():
            df_sel = df_num.loc[mask].dropna()
            tag = f"{args.prefix[:-1]}_{region_name}_{kind}"

            df_real = df_num.loc[groups["real"]].dropna()
            df_fake = df_num.loc[groups["fake"]].dropna()

            for xv, yv in pairs:
                xcol = f"{args.prefix}{xv}_{args.variation}"
                ycol = f"{args.prefix}{yv}_{args.variation}"

                if xcol not in df_num.columns or ycol not in df_num.columns:
                    continue

                # existing real/fake plots already handled above

                out_ratio = os.path.join(
                    args.outdir,
                    f"twod_ratio_{args.prefix[:-1]}_{region_name}_{xv}_vs_{yv}.pdf"
                )

                draw_2d_ratio_plot(
                    df_real=df_real,
                    df_fake=df_fake,
                    xcol=xcol,
                    ycol=ycol,
                    out_pdf=out_ratio,
                    title=f"{args.prefix[:-1]} {region_name}: Fake Fraction ({xv} vs {yv})",
                    xVar=xv,
                    yVar=yv,
                    prefix=args.prefix,
                    variation=args.variation,
                )
                xcol = f"{args.prefix}{xv}_{args.variation}"
                ycol = f"{args.prefix}{yv}_{args.variation}"

                if xcol not in df_sel.columns or ycol not in df_sel.columns:
                    continue

                out_pdf = os.path.join(args.outdir, f"twod_{tag}_{xv}_vs_{yv}.pdf")
                draw_2d_plot(
                    df_sel,
                    xcol=xcol,
                    ycol=ycol,
                    out_pdf=out_pdf,
                    title=f"{tag}: {xv} vs {yv}",
                    xVar=xv,
                    yVar=yv,
                    prefix=args.prefix,
                    variation=args.variation,
                    nbins=50,
                )            

        print(
            f"[INFO] {region_name:9s} | "
            f"all={int(reg_mask.sum()):8d} "
            f"real={int((reg_mask & real_mask).sum()):8d} "
            f"fake={int((reg_mask & fake_mask).sum()):8d}"
        )


if __name__ == "__main__":
    main()