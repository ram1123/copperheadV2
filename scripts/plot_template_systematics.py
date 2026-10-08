#!/usr/bin/env python3
"""
Validate the stage-3 VBF shape templates systematic by systematic: for every template
file (one per year and SR/SB) it saves one plot per nuisance parameter, with one column
per process that nuisance affects -- nominal/Up/Down shapes on top, Up/nominal and
Down/nominal ratios (with the nominal MC-stat band) below. Each plot is written as PNG
and PDF. A CSV summary lists every (file, process, nuisance) with its yield shifts and
warning flags: missing Up or Down, Up identical to Down but not to nominal, Up and Down
shifting the same way, or a large yield shift.

It is also the mandatory stage-3 -> stats gate (scripts/run_jj_region_scan.sh step
"validate", Snakemake rule stage3_validate): it exits with status 1 if any template has a
hard error -- a non-finite bin, or a negative bin in a non-data template (stage-3 clips
those, so one surviving means the templates are broken) -- and writes them to
<output_dir>/validation_errors.txt. Warnings never fail it; review the CSV/plots.

Templates are read as Combine's "$PROCESS" / "$PROCESS_$SYSTEMATIC{Up,Down}" histograms
in the vbf_h-peak_<year>.root / vbf_h-sidebands_<year>.root files written by
stage3/make_templates.py.

How to run (repo root, combine or default pixi env -- needs PyROOT):
    ./run_in_pixi.sh combine python scripts/plot_template_systematics.py \
        -i <stage3_templates_dir>/score_<label> -o <output_dir> \
        [--years 2023,2024] [--channels SR,SB] [--processes DY_matched2J,EWK] \
        [--nuisance-regex 'LHE|jer'] [--large-shift 0.5]
Output: <output_dir>/<year>_<SR|SB>/<nuisance>.{png,pdf} and <output_dir>/systematics_summary.csv
Keep outputs under validation/stage3_templates/<label>/<postfix>_<region>/ (repo convention).

Example (all systematics of the jj_both_fwd25 templates, 2023 SR only):
    L=Run3_nanoAODv15_FilterEvents_Sep22_tightPassLepVeto_OfficialRecomendation_Systematics_LumiSplit
    D=/work/projects/hmm/$USER/hmm_ntuples/copperheadV1clean/$L/stage3_datacards_Oct04_2026_Syst_jj_both_fwd25
    ./run_in_pixi.sh combine python scripts/plot_template_systematics.py \
        -i $D/stage3_templates_Oct04_2026_Syst_jj_both_fwd25/score_$L \
        -o validation/stage3_templates/$L/Oct04_2026_Syst_jj_both_fwd25/template_systematics \
        --years 2023 --channels SR
"""
import argparse
import csv
from collections import Counter
import glob
import math
import os
import re

import ROOT

from modules.root_2dColorProfile import set_gradient_style


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input-dir", required=True, help="dir holding vbf_h-peak_<year>.root etc.")
    parser.add_argument("-o", "--output-dir", required=True, help="where plots and the CSV summary go")
    parser.add_argument("--years", default="", help="comma-separated years to keep (default: all files)")
    parser.add_argument("--channels", default="SR,SB", help="comma-separated subset of SR,SB")
    parser.add_argument("--processes", default="", help="comma-separated processes to keep (default: all)")
    parser.add_argument("--nuisance-regex", default="", help="only nuisances matching this regex")
    parser.add_argument("--large-shift", type=float, default=0.5,
                        help="flag a variation whose yield moves by more than this fraction (default 0.5)")
    parser.add_argument("--no-fail", action="store_true",
                        help="report hard errors but still exit 0")
    return parser.parse_args()


def split_templates(tfile):
    """Return {process: nominal TH1} and {nuisance: {process: {"Up": TH1, "Down": TH1}}}."""
    hists = {}
    for key in tfile.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom("TH1"):
            obj.SetDirectory(0)
            hists[key.GetName()] = obj
    nominal = {n: h for n, h in hists.items()
               if n != "data_obs" and not n.endswith(("Up", "Down"))}
    # longest match first, so e.g. a process "DY" never claims "DY_matched2J_..."
    procs = sorted(nominal, key=len, reverse=True)
    variations, orphans = {}, []
    for name, hist in hists.items():
        side = "Up" if name.endswith("Up") else "Down" if name.endswith("Down") else None
        if side is None:
            continue
        proc = next((p for p in procs if name.startswith(f"{p}_")), None)
        if proc is None:
            orphans.append(name)
            continue
        nuisance = name[len(proc) + 1:-len(side)]
        variations.setdefault(nuisance, {}).setdefault(proc, {})[side] = hist
    return hists, nominal, variations, orphans


def bin_contents(hist):
    return [hist.GetBinContent(b) for b in range(1, hist.GetNbinsX() + 1)]


def hard_errors(hists, tag):
    """Errors that break or bias the fit: non-finite bins anywhere, negative non-data bins."""
    errors = []
    for name, hist in hists.items():
        values = bin_contents(hist)
        bad = [b for b, v in enumerate(values) if not math.isfinite(v)]
        if bad:
            errors.append(f"{tag} {name}: non-finite bins {bad}")
        neg = [(b, round(v, 6)) for b, v in enumerate(values) if math.isfinite(v) and v < 0]
        if neg and name != "data_obs":
            errors.append(f"{tag} {name}: negative bins {neg}")
    return errors


def rel_shift(var, nom):
    return (var / nom - 1.0) if nom else float("nan")


def style(hist, color, width=2, line_style=1):
    hist.SetLineColor(color)
    hist.SetLineWidth(width)
    hist.SetLineStyle(line_style)
    hist.SetMarkerSize(0)


def draw_nuisance(nuisance, by_proc, nominal, title_prefix, out_base):
    """One canvas per nuisance: a column per process, shapes on top and ratios below."""
    procs = sorted(by_proc)
    ncol = len(procs)
    canvas = ROOT.TCanvas(f"c_{nuisance}", "", 420 * ncol + 60, 720)
    keep = []
    for col, proc in enumerate(procs):
        x_lo = col / ncol
        x_hi = (col + 1) / ncol
        canvas.cd()  # new pads must belong to the canvas, not the previous column's pad
        top = ROOT.TPad(f"top_{col}", "", x_lo, 0.36, x_hi, 0.95)
        bot = ROOT.TPad(f"bot_{col}", "", x_lo, 0.0, x_hi, 0.36)
        top.SetBottomMargin(0.02)
        top.SetLeftMargin(0.16)
        bot.SetTopMargin(0.03)
        bot.SetBottomMargin(0.28)
        bot.SetLeftMargin(0.16)
        for pad in (top, bot):
            pad.SetGrid()
            pad.Draw()
        keep += [top, bot]

        nom = nominal[proc].Clone(f"nom_{nuisance}_{proc}")
        up = by_proc[proc].get("Up")
        down = by_proc[proc].get("Down")
        style(nom, ROOT.kBlack)
        shapes = [h for h in (nom, up, down) if h]
        y_max = max(h.GetMaximum() for h in shapes)
        y_min = min(h.GetMinimum() for h in shapes)
        nom.SetMaximum(y_max * 1.35 if y_max > 0 else 1.0)
        nom.SetMinimum(min(0.0, y_min * 1.2))
        nom.SetTitle(f"{proc};;yield")
        nom.GetYaxis().SetTitleSize(0.055)
        nom.GetYaxis().SetLabelSize(0.045)
        nom.GetXaxis().SetLabelSize(0)

        top.cd()
        nom.Draw("hist e")
        legend = ROOT.TLegend(0.18, 0.66, 0.92, 0.88)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextSize(0.042)
        n_int = nom.Integral()
        legend.AddEntry(nom, f"nominal ({n_int:.4g})", "l")
        if up:
            up = up.Clone(f"up_{nuisance}_{proc}")
            style(up, ROOT.kRed + 1)
            up.Draw("hist same")
            legend.AddEntry(up, f"Up ({100 * rel_shift(up.Integral(), n_int):+.2f}%)", "l")
        else:
            legend.AddEntry(ROOT.nullptr, "Up MISSING", "")
        if down:
            down = down.Clone(f"down_{nuisance}_{proc}")
            style(down, ROOT.kBlue + 1, line_style=2)
            down.Draw("hist same")
            legend.AddEntry(down, f"Down ({100 * rel_shift(down.Integral(), n_int):+.2f}%)", "l")
        else:
            legend.AddEntry(ROOT.nullptr, "Down MISSING", "")
        legend.Draw()
        keep += [nom, up, down, legend]

        # ratios to nominal; empty nominal bins are left at 1 so they don't spike the axis
        bot.cd()
        band = nom.Clone(f"band_{nuisance}_{proc}")
        for b in range(1, band.GetNbinsX() + 1):
            c = nom.GetBinContent(b)
            band.SetBinContent(b, 1.0)
            band.SetBinError(b, nom.GetBinError(b) / c if c > 0 else 0.0)
        band.SetFillColor(ROOT.kGray)
        band.SetLineColor(ROOT.kGray)
        band.SetMarkerSize(0)
        ratios = []
        for var, name in ((up, "up"), (down, "down")):
            if not var:
                continue
            r = var.Clone(f"r{name}_{nuisance}_{proc}")
            for b in range(1, r.GetNbinsX() + 1):
                c = nom.GetBinContent(b)
                r.SetBinContent(b, var.GetBinContent(b) / c if c > 0 else 1.0)
                r.SetBinError(b, 0.0)
            ratios.append(r)
        deviation = max([abs(r.GetBinContent(b) - 1.0) for r in ratios
                         for b in range(1, r.GetNbinsX() + 1)] + [0.05])
        span = min(1.2 * deviation, 2.0)
        band.SetMinimum(1.0 - span)
        band.SetMaximum(1.0 + span)
        band.SetTitle(";DNN score;var / nom")
        for axis in (band.GetXaxis(), band.GetYaxis()):
            axis.SetTitleSize(0.1)
            axis.SetLabelSize(0.085)
        band.GetYaxis().SetTitleOffset(0.75)
        band.GetYaxis().SetNdivisions(505)
        band.Draw("e2")
        unity = ROOT.TLine(band.GetXaxis().GetXmin(), 1.0, band.GetXaxis().GetXmax(), 1.0)
        unity.SetLineStyle(3)
        unity.Draw()
        for r in ratios:
            r.Draw("hist same")
        keep += [band, unity] + ratios

    canvas.cd()
    header = ROOT.TLatex()
    header.SetNDC()
    header.SetTextSize(0.03)
    header.DrawLatex(0.01, 0.965, f"{title_prefix}   nuisance: {nuisance}")
    keep.append(header)
    canvas.SaveAs(f"{out_base}.png")
    canvas.SaveAs(f"{out_base}.pdf")
    canvas.Close()


def main():
    args = parse_args()
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    set_gradient_style()
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(1)
    ROOT.gStyle.SetTitleFontSize(0.06)

    years = {y for y in args.years.split(",") if y}
    channels = {c for c in args.channels.split(",") if c}
    processes = {p for p in args.processes.split(",") if p}
    nuisance_re = re.compile(args.nuisance_regex) if args.nuisance_regex else None

    files = sorted(glob.glob(os.path.join(args.input_dir, "vbf_h-*_*.root")))
    if not files:
        raise SystemExit(f"No vbf_h-*_*.root templates in {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    rows, n_plots, errors, orphan_warnings = [], 0, [], []
    for path in files:
        m = re.match(r"vbf_(h-peak|h-sidebands)_(.+)\.root$", os.path.basename(path))
        if not m:
            continue
        channel = "SR" if m.group(1) == "h-peak" else "SB"
        year = m.group(2)
        if (years and year not in years) or channel not in channels:
            continue
        tfile = ROOT.TFile.Open(path)
        hists, nominal, variations, orphans = split_templates(tfile)
        tfile.Close()
        errors += hard_errors(hists, f"{year} {channel}")
        # a variation without its nominal: the process was dropped (zero/negative yield), so
        # it is not in the datacard and the variation is unused
        orphan_warnings += [f"{year} {channel} {name}" for name in orphans]
        out_dir = os.path.join(args.output_dir, f"{year}_{channel}")
        os.makedirs(out_dir, exist_ok=True)

        for nuisance in sorted(variations):
            if nuisance_re and not nuisance_re.search(nuisance):
                continue
            by_proc = {p: v for p, v in variations[nuisance].items() if not processes or p in processes}
            if not by_proc:
                continue
            for proc, sides in by_proc.items():
                n_int = nominal[proc].Integral()
                up_int = sides["Up"].Integral() if "Up" in sides else float("nan")
                down_int = sides["Down"].Integral() if "Down" in sides else float("nan")
                up_rel, down_rel = rel_shift(up_int, n_int), rel_shift(down_int, n_int)
                flags = [f"missing {s}" for s in ("Up", "Down") if s not in sides]
                if not flags:
                    up_vals, down_vals = bin_contents(sides["Up"]), bin_contents(sides["Down"])
                    if up_vals == down_vals and up_vals != bin_contents(nominal[proc]):
                        flags.append("up==down!=nominal")
                    elif up_rel * down_rel > 0:
                        flags.append("same-sign shift")
                if any(abs(r) > args.large_shift for r in (up_rel, down_rel) if r == r):
                    flags.append(f"yield shift > {100 * args.large_shift:g}%")
                rows.append({
                    "year": year, "channel": channel, "process": proc, "nuisance": nuisance,
                    "nominal": f"{n_int:.6g}", "up": f"{up_int:.6g}", "down": f"{down_int:.6g}",
                    "up_rel_pct": f"{100 * up_rel:.3f}", "down_rel_pct": f"{100 * down_rel:.3f}",
                    "flags": "; ".join(flags),
                })
            safe = re.sub(r"[^A-Za-z0-9_.+-]", "_", nuisance)
            draw_nuisance(nuisance, by_proc, nominal, f"{year} {channel}", os.path.join(out_dir, safe))
            n_plots += 1

    summary = os.path.join(args.output_dir, "systematics_summary.csv")
    with open(summary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["year"])
        writer.writeheader()
        writer.writerows(rows)
    n_flagged = sum(1 for r in rows if r.get("flags"))
    print(f"Wrote {n_plots} nuisance plots (png+pdf) under {args.output_dir}")
    print(f"Summary: {summary} ({len(rows)} process/nuisance rows, {n_flagged} flagged)")
    flag_counts = Counter(f for r in rows for f in r["flags"].split("; ") if f)
    for flag, count in flag_counts.most_common():
        print(f"  WARNING {count:5d} x {flag}")
    if orphan_warnings:
        print(f"  WARNING {len(orphan_warnings):5d} x variation without nominal (unused): {orphan_warnings[:5]}")

    error_file = os.path.join(args.output_dir, "validation_errors.txt")
    with open(error_file, "w") as handle:
        handle.write("\n".join(errors) + ("\n" if errors else ""))
    if errors:
        print(f"TEMPLATE VALIDATION FAILED: {len(errors)} hard error(s), see {error_file}")
        for line in errors[:20]:
            print(f"  ERROR {line}")
        if not args.no_fail:
            raise SystemExit(1)
    else:
        print("TEMPLATE VALIDATION PASSED: no non-finite or negative template bins")


if __name__ == "__main__":
    main()
