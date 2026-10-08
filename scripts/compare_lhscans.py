#!/usr/bin/env python3
"""
Overlay the VBF likelihood scans (run_vbf_lhscan in common_workflow.sh) of several run labels /
postfixes for one year, one panel per jj region (+ the jj-region combination, + the inclusive
"all" phase space -- a separate, standalone result that overlaps the split regions and is never
combined with them), and tabulate each one's 1-sigma interval and its Syst / MCStat / DYNorm /
Stat breakdown (quadrature differences of the four nested scans: with-syst, systs frozen,
systs+MC stat frozen, stat-only). Also writes scenario_table_<year>.csv/.md: one row per run with
each region's expected significance, expected 95% CL limit (from the stats pipeline's
vbf_significance_summary_<postfix>.csv / vbf_expected_limit_summary_<postfix>.csv) and 1-sigma
interval. Missing regions/files show as NA (e.g. no "all" run for a label).

Inputs are the higgsCombine.lhscan<year>_<postfix>.with_syst[.freeze_systs|.freeze_systs_mcstat|
.statonly].MultiDimFit.mH125.root files in each stage3_datacards_<postfix>_<region>/score_<label>/
(stage3_datacards_<postfix>/score_<label>/ for "all", which has no region suffix).
Intervals come from linear interpolation of the -2 Delta ln L grid at 1; a side that does not
cross 1 inside the scanned range is reported as "n/c". The scans are unreliable for r < 0
(signal-only bins, docs/known_issues.md), so a low-side crossing below 0 is flagged.

How to run (repo root, combine pixi env -- needs PyROOT):
    ./run_in_pixi.sh combine python scripts/compare_lhscans.py -y <year> \
        --run "<legend>:<label>:<postfix>" --run "<legend>:<label>:<postfix>" [...] -o <output dir>

Example (2026, with vs without LumiSplit):
    ./run_in_pixi.sh combine python scripts/compare_lhscans.py -y 2026 \
        --run "LumiSplit:Run3_nanoAODv15_FilterEvents_Sep22_tightPassLepVeto_OfficialRecomendation_Systematics_LumiSplit:Oct04_2026" \
        --run "no LumiSplit:Run3_nanoAODv12_FilterEvents_Sep22_tightPassLepVeto_OfficialRecomendation_Systematics:Oct03_2026" \
        -o validation/lhscan_comparison/2026_LumiSplit_vs_noLumiSplit

Example (2024 scenario comparison, nominal-only DNN25 postfix; one --run per scenario):
    ./run_in_pixi.sh combine python scripts/compare_lhscans.py -y 2024 \
        --run "Official:Run3_nanoAODv15_FilterEvents_Sep22_tightPassLepVeto_OfficialRecomendation_Systematics:Oct06_2026_DNN25_NoSyst" \
        --run "LumiSplit:Run3_nanoAODv15_FilterEvents_Sep22_tightPassLepVeto_OfficialRecomendation_Systematics_LumiSplit:Oct06_2026_DNN25_NoSyst" \
        -o validation/scenario_comparison_2024/Oct06_2026_DNN25_NoSyst
"""
import csv
import argparse
import math
import os

import numpy as np
import ROOT

from modules.root_2dColorProfile import set_gradient_style

SAVE_ROOT = f"/work/projects/hmm/{os.environ.get('USER', '')}/hmm_ntuples/copperheadV1clean"
REGIONS = [
    ("jj_both_central", "Both central"),
    ("jj_one_fwd25_one_central", "Central-forward"),
    ("jj_both_fwd25", "Both forward"),
    ("jj_combined_both_central_one_fwd25_one_central_both_fwd25", "3 regions combined"),
    ("all", "Inclusive (all, standalone)"),
]
SCANS = [("with_syst", ""), ("freeze_systs", ".freeze_systs"),
         ("freeze_systs_mcstat", ".freeze_systs_mcstat"), ("statonly", ".statonly")]
BREAKDOWN = [("Syst", "with_syst", "freeze_systs"), ("MCStat", "freeze_systs", "freeze_systs_mcstat"),
             ("DYNorm", "freeze_systs_mcstat", "statonly"), ("Stat", "statonly", None)]
COLORS = [ROOT.kBlack, ROOT.kRed + 1, ROOT.kBlue + 1, ROOT.kGreen + 2, ROOT.kMagenta + 1, ROOT.kOrange + 7,
          ROOT.kCyan + 2]


def read_scan(path):
    """(r, 2*deltaNLL) sorted by r, best-fit row excluded; None if the file is missing."""
    if not os.path.isfile(path):
        return None
    f = ROOT.TFile.Open(path)
    t = f.Get("limit")
    pts = {}
    for i in range(t.GetEntries()):
        t.GetEntry(i)
        if t.GetLeaf("quantileExpected").GetValue() == -1:
            continue
        pts[round(t.GetLeaf("r").GetValue(), 6)] = 2.0 * t.GetLeaf("deltaNLL").GetValue()
    f.Close()
    if len(pts) < 2:  # only a best-fit row: no usable scan
        return None
    r = np.array(sorted(pts))
    return r, np.array([pts[x] for x in r])


def crossings(r, y, level=1.0):
    """Best fit and the r where y crosses `level` below/above it (None if not crossed)."""
    i0 = int(np.argmin(y))
    lo = hi = None
    for i in range(i0, 0, -1):
        if y[i - 1] >= level > y[i] or y[i - 1] > level >= y[i]:
            lo = r[i - 1] + (level - y[i - 1]) * (r[i] - r[i - 1]) / (y[i] - y[i - 1])
            break
    for i in range(i0, len(r) - 1):
        if y[i] < level <= y[i + 1]:
            hi = r[i] + (level - y[i]) * (r[i + 1] - r[i]) / (y[i + 1] - y[i])
            break
    return r[i0], lo, hi


def interval(scan):
    if scan is None:
        return None
    best, lo, hi = crossings(*scan)
    return {"best": best, "up": None if hi is None else hi - best, "down": None if lo is None else best - lo,
            "lo": lo}


def quad_diff(a, b):
    if a is None:
        return None
    if b is None:
        return a
    return math.sqrt(max(a * a - b * b, 0.0))


def fmt(x):
    return "n/c" if x is None else f"{x:.3f}"


def card_dir(label, postfix, region):
    """stage-3 card dir; 'all' carries no region suffix (common_workflow.sh stage3_output_postfix)."""
    suffix = "" if region == "all" else f"_{region}"
    return f"{SAVE_ROOT}/{label}/stage3_datacards_{postfix}{suffix}/score_{label}"


def summary_value(path, year, column):
    """One value from a stats-pipeline summary CSV; 'NA' if the file/row is missing."""
    if not os.path.isfile(path):
        return "NA"
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("year") == year:
                return row.get(column) or "NA"
    return "NA"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-y", "--year", required=True)
    parser.add_argument("--run", action="append", required=True, help="<legend>:<label>:<postfix>")
    parser.add_argument("-o", "--output-dir", required=True)
    args = parser.parse_args()
    runs = [tuple(x.split(":", 2)) for x in args.run]
    if any(len(r) != 3 for r in runs):
        parser.error("--run must be <legend>:<label>:<postfix> (no ':' in the legend)")
    names = [r[0] for r in runs]
    if len(set(names)) != len(names):
        parser.error(f"duplicate --run legends {names}: table rows are keyed by legend")

    ROOT.gROOT.SetBatch(True)
    set_gradient_style()
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(1)
    os.makedirs(args.output_dir, exist_ok=True)

    canvas = ROOT.TCanvas("c", "", 2400, 1200)
    canvas.Divide(3, 2)
    table = {name: {} for name, _, _ in runs}
    keep, report = [], [f"# VBF likelihood scans, year {args.year} (Asimov r=1, blinded)",
                        "# 1-sigma interval of r and its breakdown (quadrature differences of nested scans)", ""]
    for ipad, (region, title) in enumerate(REGIONS, start=1):
        pad = canvas.cd(ipad)
        pad.SetGrid()
        frame = pad.DrawFrame(-0.5, 0.0, 3.5, 6.0, f"{title} ({args.year});r;-2 #Delta ln L")
        keep.append(frame)
        legend = ROOT.TLegend(0.36, max(0.40, 0.89 - 0.045 * len(runs)), 0.89, 0.89)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextSize(0.026 if len(runs) > 3 else 0.032)
        report.append(f"## {title} ({region})")
        for irun, (name, label, postfix) in enumerate(runs):
            d = card_dir(label, postfix, region)
            cell = table[name].setdefault(region, {})
            cell["sig"] = summary_value(f"{d}/vbf_significance_summary_{postfix}.csv", args.year, "significance")
            cell["lim"] = summary_value(f"{d}/vbf_expected_limit_summary_{postfix}.csv", args.year,
                                        "expected_limit_median")
            cell["int"] = "NA"
            base = f"{d}/higgsCombine.lhscan{args.year}_{postfix}.with_syst"
            scans = {k: read_scan(f"{base}{sfx}.MultiDimFit.mH125.root") for k, sfx in SCANS}
            ints = {k: interval(v) for k, v in scans.items()}
            for k, style in (("with_syst", 1), ("statonly", 2)):
                if scans[k] is None:
                    continue
                g = ROOT.TGraph(len(scans[k][0]), scans[k][0].astype(float), scans[k][1].astype(float))
                g.SetLineColor(COLORS[irun % len(COLORS)])
                g.SetLineStyle(style)
                g.SetLineWidth(3 if style == 1 else 2)
                g.Draw("L same")
                legend.AddEntry(g, f"{name}, {'with syst' if k == 'with_syst' else 'stat-only'}", "l")
                keep.append(g)
            tot = ints["with_syst"]
            if tot is not None:
                cell["int"] = f"{tot['best']:.2f} +{fmt(tot['up'])}/-{fmt(tot['down'])}"
            if tot is None:
                report.append(f"   {name}: scan files not found ({base}*)")
                continue
            parts = []
            for bname, a, b in BREAKDOWN:
                ia, ib = ints[a], ints[b] if b else None
                up = quad_diff(ia and ia["up"], ib and ib["up"])
                dn = quad_diff(ia and ia["down"], ib and ib["down"])
                parts.append(f"{bname} +{fmt(up)}/-{fmt(dn)}")
            flag = "  (low side crosses below r=0: unreliable)" if tot["lo"] is not None and tot["lo"] < 0 else ""
            report.append(f"   {name:14s}: r = {tot['best']:.2f} +{fmt(tot['up'])}/-{fmt(tot['down'])}  | "
                          + ", ".join(parts) + flag)
        for level in (1.0, 4.0):
            line = ROOT.TLine(-0.5, level, 3.5, level)
            line.SetLineStyle(3)
            line.SetLineColor(ROOT.kGray + 2)
            line.Draw()
            keep.append(line)
        legend.Draw()
        keep.append(legend)
        report.append("")
    out = os.path.join(args.output_dir, f"lhscan_comparison_{args.year}")
    canvas.SaveAs(f"{out}.pdf")
    canvas.SaveAs(f"{out}.png")
    with open(f"{out}.txt", "w") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"Wrote {out}.pdf/.png/.txt")

    # One row per run: significance / expected limit / 1-sigma interval per region.
    tab = os.path.join(args.output_dir, f"scenario_table_{args.year}")
    cols = [(region, title) for region, title in REGIONS]
    with open(f"{tab}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "label", "postfix"] + [f"{r}_{k}" for r, _ in cols for k in ("significance",
                    "expected_limit", "lhscan_1sigma")])
        for name, label, postfix in runs:
            w.writerow([name, label, postfix] + [table[name].get(r, {}).get(k, "NA") for r, _ in cols
                                                 for k in ("sig", "lim", "int")])
    md = [f"# VBF scenario comparison, {args.year} (expected/Asimov, blinded)", "",
          "Cell: expected significance / expected 95% CL limit on r / r 1-sigma interval (Asimov r=1). "
          "Three different estimators (profile-likelihood q0 with -t -1 --expectSignal=1; AsymptoticLimits "
          "--run blind median; 1D scan grid interpolated at -2 Delta lnL = 1), all with the card's nuisances "
          "(the 'with syst' values), not interchangeable with each other.", "",
          "'all' overlaps the split regions and is a standalone result, never combined with them; "
          "'all' vs '3 regions combined' compares different DNNs/binnings/channel counts, not a pure combination. "
          "For nominal-only (_NoSyst) postfixes the card has no shape systematics, so 'with syst' and the "
          "Syst part of the scan breakdown cover lnN/rate nuisances only. The scan's stat-only also freezes "
          "the DY rateParams, unlike the pipeline's StatOnly significance/limit.", "",
          "| run | " + " | ".join(t for _, t in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, _, _ in runs:
        md.append(f"| {name} | " + " | ".join(
            " / ".join(table[name].get(r, {}).get(k, "NA") for k in ("sig", "lim", "int")) for r, _ in cols) + " |")
    with open(f"{tab}.md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"Wrote {tab}.csv/.md")


if __name__ == "__main__":
    main()
