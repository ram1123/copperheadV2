#!/usr/bin/env python3
"""
Plot the nominal VBF stage-3 background templates, one panel per process, to show which
bins are negative (before clipping) or that they are at zero (after clipping in
stage3/make_templates.py). One page per (year, SR/SB) file, plus a summary page listing
every negative bin found.

How to run (from the repo root, combine or default pixi env -- needs PyROOT):
    ./run_in_pixi.sh combine python scripts/plot_template_negative_bins.py \
        -i <stage3_templates_dir>/score_<label> -o <output.pdf> [-t "title tag"] [--years 2023,2024]

Example (unclipped backup vs clipped templates of jj_both_fwd25):
    D=/work/projects/hmm/$USER/hmm_ntuples/copperheadV1clean/<label>/stage3_datacards_Oct04_2026_Syst_jj_both_fwd25
    ./run_in_pixi.sh combine python scripts/plot_template_negative_bins.py \
        -i $D/stage3_templates_Oct04_2026_Syst_jj_both_fwd25_unclipped_backup/score_<label> \
        -o validation/stage3_templates/<label>/Oct04_2026_Syst_jj_both_fwd25/templates_nominal_unclipped.pdf \
        -t "jj_both_fwd25 unclipped"

B=/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterEvents_Sep22_tightPassLepVeto_OfficialRecomendation_Systematics_LumiSplit
L=Run3_nanoAODv15_FilterEvents_Sep22_tightPassLepVeto_OfficialRecomendation_Systematics_LumiSplit
D=$B/stage3_datacards_Oct04_2026_Syst_jj_both_fwd25

# before (from the unclipped backup)
./run_in_pixi.sh combine python scripts/plot_template_negative_bins.py \
    -i $D/stage3_templates_Oct04_2026_Syst_jj_both_fwd25_unclipped_backup/score_$L \
    -o $D/score_$L/templates_nominal_unclipped.pdf -t "jj_both_fwd25 unclipped"

# after (current templates)
./run_in_pixi.sh combine python scripts/plot_template_negative_bins.py \
    -i $D/stage3_templates_Oct04_2026_Syst_jj_both_fwd25/score_$L \
    -o $D/score_$L/templates_nominal_clipped.pdf -t "jj_both_fwd25 clipped+floored"

"""
import argparse
import glob
import os
import re

import ROOT

from modules.root_2dColorProfile import set_gradient_style

BKG_PROCESSES = ["DY_matched01J", "DY_matched2J", "EWK", "TT+ST", "VV"]
COLORS = {"DY_matched01J": ROOT.kAzure + 1, "DY_matched2J": ROOT.kBlue + 2, "EWK": ROOT.kViolet,
          "TT+ST": ROOT.kOrange + 1, "VV": ROOT.kGreen + 2, "Total bkg": ROOT.kBlack}


def negative_bins(hist):
    return [(b - 1, hist.GetBinContent(b)) for b in range(1, hist.GetNbinsX() + 1) if hist.GetBinContent(b) < 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input-dir", required=True, help="dir holding vbf_h-peak_<year>.root etc.")
    parser.add_argument("-o", "--output", required=True, help="output multi-page PDF (parent dir is created)")
    parser.add_argument("-t", "--tag", default="", help="text added to every page title")
    parser.add_argument("--years", default="", help="comma-separated years to keep (default: all files)")
    args = parser.parse_args()

    ROOT.gROOT.SetBatch(True)
    set_gradient_style()
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(1)

    files = sorted(glob.glob(os.path.join(args.input_dir, "vbf_h-*_*.root")))
    if not files:
        raise SystemExit(f"No vbf_h-*_*.root templates in {args.input_dir}")

    pages, summary = [], []
    for path in files:
        m = re.match(r"vbf_(h-peak|h-sidebands)_(.+)\.root", os.path.basename(path))
        channel = "SR" if m.group(1) == "h-peak" else "SB"
        year = m.group(2)
        if args.years and year not in args.years.split(","):
            continue
        tfile = ROOT.TFile.Open(path)
        hists = {}
        for proc in BKG_PROCESSES:
            h = tfile.Get(proc)
            if h:
                h = h.Clone(f"{proc}_{year}_{channel}")
                h.SetDirectory(0)
                hists[proc] = h
        tfile.Close()
        if not hists:
            continue
        procs = list(hists.values())
        total = procs[0].Clone(f"total_{year}_{channel}")
        total.SetDirectory(0)
        for h in procs[1:]:
            total.Add(h)
        hists["Total bkg"] = total
        for proc, h in hists.items():
            for b, v in negative_bins(h):
                summary.append(f"{year} {channel} {proc}: bin {b} = {v:.4g}")
        pages.append((year, channel, hists))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    canvas = ROOT.TCanvas("c", "c", 1500, 1000)
    canvas.Print(f"{args.output}[")

    # Summary page
    canvas.Clear()
    text = ROOT.TPaveText(0.03, 0.03, 0.97, 0.97, "NDC")
    text.SetFillColor(0)
    text.SetTextAlign(12)
    text.SetTextFont(42)
    text.AddText(f"Negative nominal bins {args.tag}: {len(summary)} found")
    lines = summary if summary else ["none -- every nominal background bin is >= 0"]
    max_lines = 45
    for line in lines[:max_lines]:
        text.AddText(line)
    if len(lines) > max_lines:
        text.AddText(f"... and {len(lines) - max_lines} more (see the per-file pages)")
    text.SetTextSize(0.018)
    text.Draw()
    canvas.Print(args.output)

    keep = []
    for year, channel, hists in pages:
        canvas.Clear()
        canvas.Divide(3, 2)
        for i, (proc, h) in enumerate(hists.items(), start=1):
            pad = canvas.cd(i)
            pad.SetGrid()
            neg = negative_bins(h)
            h.SetTitle(f"{year} {channel} {proc} {args.tag}"
                       f"{' -- ' + str(len(neg)) + ' negative bin(s)' if neg else ''};DNN score;yield")
            h.SetLineColor(COLORS[proc])
            h.SetLineWidth(2)
            lo, hi = h.GetMinimum(), h.GetMaximum()
            span = max(hi - lo, 1e-6)
            # explicit range so 0 is always visible and negative bins aren't cut off
            h.SetMinimum(min(lo, 0.0) - 0.1 * span)
            h.SetMaximum(hi + 0.15 * span)
            h.Draw("hist e")
            zero = ROOT.TLine(h.GetXaxis().GetXmin(), 0.0, h.GetXaxis().GetXmax(), 0.0)
            zero.SetLineStyle(2)
            zero.SetLineColor(ROOT.kGray + 2)
            zero.Draw()
            marks = ROOT.TGraph()
            for b, v in neg:
                marks.AddPoint(h.GetBinCenter(b + 1), v)
            if neg:
                marks.SetMarkerStyle(20)
                marks.SetMarkerColor(ROOT.kRed)
                marks.SetMarkerSize(1.2)
                marks.Draw("P same")
            keep += [zero, marks]
        canvas.Print(args.output)

    canvas.Print(f"{args.output}]")
    print(f"Wrote {args.output} ({len(pages)} template pages, {len(summary)} negative bins)")


if __name__ == "__main__":
    main()
