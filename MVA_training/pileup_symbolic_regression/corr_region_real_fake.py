#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import pandas as pd
import ROOT

ROOT.gROOT.SetBatch(True)

"""
time python MVA_training/pileup_symbolic_regression/corr_region_real_fake.py   \
    -i "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn_Mar19_tightPassLepVeto_NoJER_pySR/stage1_output/2022postEE/compacted/dyTo2L_M-50_incl/0/part*.parquet"   \
    -o validation/corr_jet1   \
    --prefix jet1_
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
        "inclusive": np.isfinite(aeta),
        "central": (aeta < 2.5),
        "HE": (aeta >= 2.5) & (aeta < 3.0),
        "HEpos": (eta >= 2.5) & (eta < 3.0),
        "HEneg": (eta <= -2.5) & (eta > -3.0),        
        "HF": (aeta >= 3.0),
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
    lo = np.percentile(x, 0.001)
    hi = np.percentile(x, 99.9999)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return fallback
    if "nConstituents" in Var:
        return 0, 20, 20
    if "pt" in Var: hi = 200.0
    if "mass" in Var: hi = 15.0
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
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    vars_short = ["mass", "muEF", "nConstituents", "pt", "eta", "area"]
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

    for region_name, reg_mask in reg_masks.items():
        groups = {
            "all": reg_mask,
            "real": reg_mask & real_mask,
            "fake": reg_mask & fake_mask,
        }

        for kind, mask in groups.items():
            df_sel = df_num.loc[mask].dropna()
            tag = f"{args.prefix[:-1]}_{region_name}_{kind}"

            # save_corr_and_heatmap(
            #     df_sel,
            #     out_base=os.path.join(args.outdir, f"corr_{tag}"),
            #     label=tag,
            #     prefix=args.prefix,
            #     variation=args.variation,
            # )
            pairs = [
                # ("mass", "nConstituents"),
                # ("mass", "muEF"),
                # ("mass", "pt"),
                # ("nConstituents", "pt"),
                # ("muEF", "pt"),
                # ("muEF", "nConstituents"),
                # ("eta", "nConstituents"),
                # ("eta", "mass"),
                ("area", "nConstituents"),
                ("area", "muEF"),
                ("area", "mass"),
                ("area", "pt"),
                ("area", "eta"),
            ]

            for xv, yv in pairs:
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