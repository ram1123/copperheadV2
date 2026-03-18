#!/usr/bin/env python3
"""
Check the PUscore variable with the SAME style you want:
- black points: total
- stacked fills: fake (PU) + real (HS)

Also saves:
  - PUscore_total_points_stack_real_fake.pdf
  - PUscore_vs_pt_profile_real_fake.pdf   (optional but useful)

Assumes Parquet has (for jet1):
  jet1_pt, jet1_eta, jet1_neEmEF, jet1_neHEF, jet1_chMultiplicity, jet1_neMultiplicity,
  jet1_genJetIdx  (or bool match)

Usage:
  python plot_puscore_check.py -i input.parquet -o outdir --jet jet1
"""

import os
import argparse
import numpy as np
import pandas as pd
import ROOT
ROOT.gROOT.SetBatch(True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--outdir", default="puscore_check")
    p.add_argument("--jet", choices=["jet1", "jet2"], default="jet1")

    p.add_argument("--genmatch-suffix", default="hasMatchedGenJet_nominal")
    p.add_argument("--genmatch-mode", choices=["genJetIdx", "bool"], default="genJetIdx")

    p.add_argument("--pt-min", type=float, default=25.0)
    p.add_argument("--abs-eta-max", type=float, default=4.7)

    # Regions
    p.add_argument("--do-regions", action="store_true",
                   help="Also make barrel/HE/HF separated plots")

    # PUscore thresholds (for drawing vertical lines)
    p.add_argument("--thr-HE", type=float, default=None)
    p.add_argument("--thr-HF", type=float, default=None)

    p.add_argument("--nbins", type=int, default=50)
    p.add_argument("--normalize", action="store_true")

    return p.parse_args()


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def get_masks(df, prefix, genmatch_suffix, genmatch_mode, pt_min, abs_eta_max):
    pt = df[prefix + "pt"].to_numpy(dtype=np.float32)
    eta = df[prefix + "eta"].to_numpy(dtype=np.float32)
    sel = (pt >= pt_min) & (np.abs(eta) <= abs_eta_max)

    gm = df[prefix + genmatch_suffix].to_numpy()
    if genmatch_mode == "genJetIdx":
        real = (gm == 1) & sel
        fake = (gm == 0) & sel
    else:
        gm_bool = gm.astype(bool)
        real = gm_bool & sel
        fake = (~gm_bool) & sel

    total = sel
    return total, real, fake


def compute_puscore(df, prefix):
    # PUscore = charged_frac - 0.7*neEmEF - 0.4*neHEF - 0.2*(neMult/(chMult+1))
    ch = df[prefix + "chMultiplicity"].to_numpy(dtype=np.float32)
    ne = df[prefix + "neMultiplicity"].to_numpy(dtype=np.float32)
    neEmEF = df[prefix + "neEmEF"].to_numpy(dtype=np.float32)
    neHEF = df[prefix + "neHEF"].to_numpy(dtype=np.float32)

    charged_frac = ch / (ch + ne + 1.0)
    ne_over_ch = ne / (ch + 1.0)
    puscore = charged_frac - 0.7 * neEmEF - 0.4 * neHEF - 0.2 * ne_over_ch
    return puscore


def style_total(h):
    h.SetMarkerStyle(20)
    h.SetMarkerSize(1.0)
    h.SetLineColor(ROOT.kBlack)
    h.SetStats(0)


def style_fill(h, kind):
    if kind == "real":
        h.SetFillColor(ROOT.kAzure + 1)
        h.SetLineColor(ROOT.kAzure + 2)
    else:
        h.SetFillColor(ROOT.kOrange - 2)
        h.SetLineColor(ROOT.kOrange + 1)
    h.SetStats(0)


def normalize_stack_and_total(h_total, h_real, h_fake):
    if h_total.Integral() > 0:
        h_total.Scale(1.0 / h_total.Integral())
    s = h_real.Integral() + h_fake.Integral()
    if s > 0:
        h_real.Scale(1.0 / s)
        h_fake.Scale(1.0 / s)


def fill_hist(h, arr):
    for v in arr:
        if np.isfinite(v):
            h.Fill(float(v))


def draw_stack_with_total(puscore, total, real, fake, outpath, nbins, normalize,
                          x_title="PUscore", thr_lines=None):
    # Robust range (clip)
    x = puscore[total]
    x = x[np.isfinite(x)]
    if len(x) == 0:
        print(f"[WARN] No entries for {outpath}")
        return

    lo = float(np.percentile(x, 1))
    hi = float(np.percentile(x, 99))
    if lo == hi:
        lo -= 1.0
        hi += 1.0

    h_total = ROOT.TH1F("h_total", f";{x_title};Entries", nbins, lo, hi)
    h_real  = ROOT.TH1F("h_real",  "", nbins, lo, hi)
    h_fake  = ROOT.TH1F("h_fake",  "", nbins, lo, hi)
    h_total.Sumw2(); h_real.Sumw2(); h_fake.Sumw2()

    fill_hist(h_total, puscore[total])
    fill_hist(h_real,  puscore[real])
    fill_hist(h_fake,  puscore[fake])

    style_total(h_total)
    style_fill(h_real, "real")
    style_fill(h_fake, "fake")

    if normalize:
        normalize_stack_and_total(h_total, h_real, h_fake)

    stack = ROOT.THStack("stack", f";{x_title};{'Normalized entries' if normalize else 'Entries'}")
    stack.Add(h_fake)
    stack.Add(h_real)

    c = ROOT.TCanvas("c", "c", 800, 650)
    c.SetLeftMargin(0.12)
    c.SetBottomMargin(0.12)

    stack.Draw("HIST")
    ymax = max(stack.GetMaximum(), h_total.GetMaximum()) * 1.35
    stack.SetMaximum(ymax)

    h_total.Draw("E SAME")

    # optional vertical threshold lines
    if thr_lines:
        for thr, color, label in thr_lines:
            line = ROOT.TLine(thr, 0.0, thr, ymax)
            line.SetLineColor(color)
            line.SetLineWidth(2)
            line.SetLineStyle(2)
            line.Draw()

    leg = ROOT.TLegend(0.58, 0.72, 0.88, 0.90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(h_total, "Total jets", "pe")
    leg.AddEntry(h_real,  "Real (HS)", "f")
    leg.AddEntry(h_fake,  "Fake (PU)", "f")
    if thr_lines:
        for thr, color, label in thr_lines:
            dummy = ROOT.TLine(0, 0, 0, 0)
            dummy.SetLineColor(color)
            dummy.SetLineStyle(2)
            dummy.SetLineWidth(2)
            leg.AddEntry(dummy, label, "l")
    leg.Draw()

    c.SaveAs(outpath)
    c.Close()

    h_total.Delete(); h_real.Delete(); h_fake.Delete()


def draw_puscore_vs_pt_profile(df, prefix, puscore, total, real, fake, outpath):
    pt = df[prefix + "pt"].to_numpy(dtype=np.float32)

    # binning
    nbins = 20
    xmin, xmax = 25.0, 200.0

    # TProfile
    p_real = ROOT.TProfile("p_real", ";p_{T} [GeV];#LTPUscore#GT", nbins, xmin, xmax)
    p_fake = ROOT.TProfile("p_fake", ";p_{T} [GeV];#LTPUscore#GT", nbins, xmin, xmax)

    for x, y in zip(pt[real], puscore[real]):
        if np.isfinite(x) and np.isfinite(y):
            p_real.Fill(float(x), float(y))
    for x, y in zip(pt[fake], puscore[fake]):
        if np.isfinite(x) and np.isfinite(y):
            p_fake.Fill(float(x), float(y))

    p_real.SetLineColor(ROOT.kAzure + 2)
    p_real.SetMarkerColor(ROOT.kAzure + 2)
    p_real.SetMarkerStyle(20)

    p_fake.SetLineColor(ROOT.kOrange + 1)
    p_fake.SetMarkerColor(ROOT.kOrange + 1)
    p_fake.SetMarkerStyle(21)

    c = ROOT.TCanvas("c2", "c2", 800, 650)
    c.SetLeftMargin(0.12)
    c.SetBottomMargin(0.12)

    p_real.Draw("E")
    p_fake.Draw("E SAME")

    leg = ROOT.TLegend(0.58, 0.78, 0.88, 0.90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(p_real, "Real (HS)", "pe")
    leg.AddEntry(p_fake, "Fake (PU)", "pe")
    leg.Draw()

    c.SaveAs(outpath)
    c.Close()

    p_real.Delete(); p_fake.Delete()


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    df = pd.read_parquet(args.input)

    prefix = args.jet + "_"

    # compute puscore
    for needed in ["pt", "eta", "chMultiplicity", "neMultiplicity", "neEmEF", "neHEF", args.genmatch_suffix]:
        col = prefix + needed
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")

    puscore = compute_puscore(df, prefix)

    total, real, fake = get_masks(
        df, prefix,
        args.genmatch_suffix, args.genmatch_mode,
        args.pt_min, args.abs_eta_max
    )

    # global plot
    thr_lines = []
    if args.thr_HE is not None:
        thr_lines.append((args.thr_HE, ROOT.kRed + 1, "thr_HE"))
    if args.thr_HF is not None:
        thr_lines.append((args.thr_HF, ROOT.kGreen + 2, "thr_HF"))

    out1 = os.path.join(args.outdir, f"{args.jet}_PUscore_total_points_stack_real_fake.pdf")
    draw_stack_with_total(
        puscore, total, real, fake,
        out1, args.nbins, args.normalize,
        x_title=f"{args.jet} PUscore",
        thr_lines=thr_lines if thr_lines else None
    )

    # PUscore vs pt (profile)
    out2 = os.path.join(args.outdir, f"{args.jet}_PUscore_vs_pt_profile_real_fake.pdf")
    draw_puscore_vs_pt_profile(df, prefix, puscore, total, real, fake, out2)

    # regional plots
    if args.do_regions:
        aeta = np.abs(df[prefix + "eta"].to_numpy(dtype=np.float32))
        regions = {
            "barrel": aeta < 2.5,
            "HE": (aeta >= 2.5) & (aeta < 3.0),
            "HF": aeta >= 3.0,
        }
        for rname, rmask in regions.items():
            t = total & rmask
            r = real & rmask
            f = fake & rmask
            out = os.path.join(args.outdir, f"{args.jet}_PUscore_{rname}_stack.pdf")
            draw_stack_with_total(
                puscore, t, r, f,
                out, args.nbins, args.normalize,
                x_title=f"{args.jet} PUscore ({rname})",
                thr_lines=thr_lines if thr_lines else None
            )

    print(f"[DONE] Wrote plots to {args.outdir}")


if __name__ == "__main__":
    main()