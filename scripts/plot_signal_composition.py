#!/usr/bin/env python3
"""
Signal-composition summary plot (HIG-19-006 Fig. style): for every category (VBF-DNN score bin,
or group of bins, in each jj-eta region) draw (1) the fractional signal composition per
production mode with the expected signal yield, (2) S/(S+B) in %, (3) S/sqrt(B).

Inputs are the nominal stage-3 templates (vbf_h-peak_<year>.root): processes ending in "_hmm"
are signal, everything else except data_obs is background. Yields are summed over --years.
B is the full h-peak window yield (not a HWHM window as in HIG-19-006), so S/(S+B) is lower.

How to run (repo root, needs stage-3 templates from run_stage3_vbf.py):
    ./run_in_pixi.sh default python scripts/plot_signal_composition.py \
        --base-dir <ntuple dir> --postfix <stage3 postfix> --years <year[,year]> [--bin-groups ...]

Example:
    ./run_in_pixi.sh default python scripts/plot_signal_composition.py \
        --base-dir /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterEvents_Oct29_tightPassLepVeto_PUDNNTrain1Oct29_JERStrat4_JetIDVars_LumiSplit_HPO200 \
        --postfix Nov01_2026_Syst_DNN2024Ray200_DY105 --years 2024 \
        --lumi-text "2024 (13.6 TeV)" --regions both_fwd25 --bin-groups "1-9,10-11,12,13,14-15"
    (omit --bin-groups for one row per DNN bin)

    Inclusive jj region (stage-3 run with -r all), all Run-3 years summed:
    ./run_in_pixi.sh default python scripts/plot_signal_composition.py \
        --base-dir /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterEvents_Oct26_tightPassLepVeto_OfficialRecomendation_Systematics_LumiSplit \
        --postfix Nov10_2026_NoSyst_DNNOct23Ray050_DY105 --regions all \
        --years 2022preEE,2022postEE,2023,2023BPix,2024,2025,2026 --lumi-text "Run 3 (13.6 TeV)"

    All three jj regions in one plot, with per-region bin groups ("region:groups;..."):
    ./run_in_pixi.sh default python scripts/plot_signal_composition.py \
        --base-dir /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterEvents_Oct29_tightPassLepVeto_PUDNNTrain1Oct29_JERStrat4_JetIDVars_LumiSplit_HPO200 \
        --postfix Nov01_2026_Syst_DNN2024Ray200_DY105 --years 2024 --lumi-text "2024 (13.6 TeV)" \
        --bin-groups "both_central:1-8,9,10,11-12;one_fwd25_one_central:1-13,14-16,17-18,19-20,21-22;both_fwd25:1-9,10-11,12,13,14-15" \
        --tag CC_CF_FF_grouped
"""

import argparse
import csv
import glob
import math
import os
import sys

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

REGION_LABEL = {
    "both_central": "CC",
    "one_fwd25_one_central": "CF",
    "both_fwd25": "FF",
    "all": "incl.",
}
# process -> (legend label, hex colour); HIG-19-006 colour scheme
SIGNAL_STYLE = {
    "ggH_hmm": ("ggH", "#1f1fd6"),
    "qqH_hmm": ("VBF", "#2ca02c"),
    "WH_hmm": ("WH", "#d62728"),
    "WplusH_hmm": ("W^{+}H", "#d62728"),
    "WminusH_hmm": ("W^{-}H", "#ff7f7f"),
    "ZH_hmm": ("ZH", "#ff7f0e"),
    "ggZH_hmm": ("ggZH", "#f2c400"),
    "ttH_hmm": ("ttH", "#9467bd"),
    "tH_hmm": ("tH", "#e377c2"),
}
FALLBACK_COLOURS = ["#8c564b", "#17becf", "#7f7f7f", "#bcbd22"]


def parse_args():
    """Parse command-line options (inputs, regions, bin grouping, labels, output dir)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", required=True, help="ntuple label dir holding stage3_datacards_* dirs")
    p.add_argument("--postfix", required=True, help="stage-3 postfix before _jj_<region>")
    p.add_argument("--years", default="2024", help="comma list; yields are summed")
    p.add_argument("--regions", default="both_central,one_fwd25_one_central,both_fwd25")
    p.add_argument("--bin-groups", default="",
                   help='1-based DNN-bin groups, e.g. "1-6,7-8,9" (all regions) or per region '
                        '"both_central:1-8,9;both_fwd25:1-9,10" (unlisted regions: one row per bin)')
    p.add_argument("--tag", default="", help="output file tag (default derived from --bin-groups)")
    p.add_argument("--lumi-text", default="(13.6 TeV)")
    p.add_argument("--cms-text", default="#it{Simulation Preliminary}")
    p.add_argument("--outdir", default="", help="default: validation/signal_composition/<label>/<postfix>_<years>")
    return p.parse_args()


def find_template(base, postfix, region, year):
    """Return the single vbf_h-peak_<year>.root template path for one jj region; exit if 0 or >1 match."""
    # the inclusive jj region ("all") has no _jj_<region> suffix in the stage-3 dir name
    sfx = "" if region == "all" else f"_jj_{region}"
    pat = f"{base}/stage3_datacards_{postfix}{sfx}/stage3_templates_*/*/vbf_h-peak_{year}.root"
    hits = sorted(glob.glob(pat))
    if len(hits) != 1:
        sys.exit(f"Expected exactly one template for {pat}, found {len(hits)}: {hits}")
    return hits[0]


def read_nominal(path):
    """Return {process: [bin contents]} for nominal histograms, plus number of bins."""
    f = ROOT.TFile.Open(path)
    out, nbins = {}, None
    for key in f.GetListOfKeys():
        name = key.GetName()
        if name.endswith("Up") or name.endswith("Down") or name == "data_obs":
            continue
        h = f.Get(name)
        if not isinstance(h, ROOT.TH1):
            continue
        nb = h.GetNbinsX()
        if nbins is not None and nb != nbins:
            sys.exit(f"{path}: {name} has {nb} bins, others {nbins}")
        nbins = nb
        out[name] = [h.GetBinContent(i) for i in range(1, nb + 1)]
    f.Close()
    return out, nbins


def region_group_spec(spec, region):
    """Pick the bin-group spec for one region: the whole string if it has no "region:" parts,
    else that region's entry ("" -> one row per bin if the region is not listed)."""
    if ":" not in spec:
        return spec
    for part in spec.split(";"):
        name, _, groups = part.partition(":")
        if name.strip() == region:
            return groups.strip()
    return ""


def parse_groups(spec, nbins):
    """Turn "1-5,6-8,..." into [(lo, hi), ...] 1-based bin ranges; default one group per bin.

    Warns if the groups do not cover every bin exactly once.
    """
    if not spec:
        return [(i, i) for i in range(1, nbins + 1)]
    groups = []
    for tok in spec.split(","):
        lo, _, hi = tok.strip().partition("-")
        lo, hi = int(lo), int(hi or lo)
        if not (1 <= lo <= hi <= nbins):
            sys.exit(f"bin group {tok} outside 1..{nbins}")
        groups.append((lo, hi))
    covered = sorted(b for lo, hi in groups for b in range(lo, hi + 1))
    if covered != list(range(1, nbins + 1)):
        print(f"WARNING: bin groups do not cover each of 1..{nbins} exactly once: {covered}")
    return groups


def build_rows(args):
    """Sum nominal templates over years and build one row per (region, bin group).

    Each row holds per-signal yields, total S and total B. Also returns the ordered list
    of signal processes (HIG-19-006 order first, unknown *_hmm processes after).
    """
    years = [y.strip() for y in args.years.split(",") if y.strip()]
    rows, signals = [], []
    for region in [r.strip() for r in args.regions.split(",") if r.strip()]:
        summed, nbins = {}, None
        for year in years:
            path = find_template(args.base_dir, args.postfix, region, year)
            print(f"[{region} {year}] {path}")
            procs, nb = read_nominal(path)
            if nbins is not None and nb != nbins:
                sys.exit(f"{region}: binning differs between years")
            nbins = nb
            for proc, vals in procs.items():
                acc = summed.setdefault(proc, [0.0] * nb)
                summed[proc] = [a + v for a, v in zip(acc, vals)]
        for proc in summed:
            if proc.endswith("_hmm") and proc not in signals:
                signals.append(proc)
        for lo, hi in parse_groups(region_group_spec(args.bin_groups, region), nbins):
            sig = {p: sum(v[lo - 1:hi]) for p, v in summed.items() if p.endswith("_hmm")}
            bkg = sum(sum(v[lo - 1:hi]) for p, v in summed.items() if not p.endswith("_hmm"))
            binlab = f"{lo}" if lo == hi else f"{lo}-{hi}"
            rows.append({
                "label": f"VBF {REGION_LABEL.get(region, region)} DNN-bin {binlab}",
                "region": region, "bins": binlab, "sig": sig, "S": sum(sig.values()), "B": bkg,
            })
    # fixed order: known modes first (HIG-19-006 order), then anything new
    order = [p for p in SIGNAL_STYLE if p in signals] + [p for p in signals if p not in SIGNAL_STYLE]
    return rows, order


def style_for(proc, idx):
    """Return (legend label, ROOT colour) for a signal process; unknown ones get a fallback colour."""
    if proc in SIGNAL_STYLE:
        lab, hexc = SIGNAL_STYLE[proc]
    else:
        lab, hexc = proc.replace("_hmm", ""), FALLBACK_COLOURS[idx % len(FALLBACK_COLOURS)]
    return lab, ROOT.TColor.GetColor(hexc)


def draw(rows, order, args, outbase):
    """Draw the three panels (composition, S/(S+B), S/sqrt(B)) and save <outbase>.pdf/.png.

    Returns the per-row S/(S+B) in % and S/sqrt(B) lists.
    """
    n = len(rows)
    height = max(500, 60 + 28 * n)
    c = ROOT.TCanvas("c", "", 1500, height)
    keep = []
    # pad x-ranges: composition | S/(S+B) | S/sqrt(B)
    edges = [(0.0, 0.47), (0.47, 0.735), (0.735, 1.0)]
    pads = []
    for i, (x0, x1) in enumerate(edges):
        pad = ROOT.TPad(f"p{i}", "", x0, 0.0, x1, 1.0)
        pad.SetLeftMargin(0.36 if i == 0 else 0.05)
        pad.SetRightMargin(0.02 if i else 0.11)
        pad.SetTopMargin(40.0 / height)
        pad.SetBottomMargin(60.0 / height)
        pad.Draw()
        pads.append(pad)
    styles = {p: style_for(p, i) for i, p in enumerate(order)}
    px = 15  # font 43/63 sizes are in pixels, so every pad gets the same text size

    def frame(pad, name, xmax, xtitle, labels):
        """Draw an empty axis frame with one y slot per row (labels only on the first panel)
        and dashed lines between jj regions."""
        pad.cd()
        # tick length is a fraction of pad width here; keep it the same in pixels for all pads
        wfrac = pad.GetAbsWNDC() * (1 - pad.GetLeftMargin() - pad.GetRightMargin())
        h = ROOT.TH2F(name, "", 1, 0, xmax, n, 0, n)
        for i, r in enumerate(rows):
            h.GetYaxis().SetBinLabel(n - i, r["label"] if labels else "")
        h.GetXaxis().SetTitle(xtitle)
        for ax in (h.GetXaxis(), h.GetYaxis()):
            ax.SetLabelFont(43)
            ax.SetTitleFont(43)
            ax.SetLabelSize(px)
            ax.SetTitleSize(px + 2)
        h.GetXaxis().SetTitleOffset(1.6)
        h.GetXaxis().SetNdivisions(505)
        h.GetXaxis().SetTickLength(0.012 * height / (1500.0 * wfrac) if wfrac > 0 else 0.03)
        h.GetXaxis().CenterTitle()
        if not labels:
            h.GetYaxis().SetLabelSize(0)
        h.GetYaxis().SetTickLength(0)
        h.Draw("AXIS")
        # dashed separators between jj regions
        for i in range(1, n):
            if rows[i]["region"] != rows[i - 1]["region"]:
                ln = ROOT.TLine(0, n - i, xmax, n - i)
                ln.SetLineStyle(2)
                ln.SetLineColor(ROOT.kGray + 2)
                ln.Draw()
                keep.append(ln)
        keep.append(h)
        return h

    def box(x0, y, x1, colour):
        """Draw one filled horizontal bar segment from x0 to x1 in row slot y."""
        b = ROOT.TBox(x0, y + 0.12, x1, y + 0.88)
        b.SetFillColor(colour)
        b.SetLineWidth(0)
        b.Draw()
        keep.append(b)

    # --- panel 1: composition ---
    frame(pads[0], "f0", 100, "Signal composition (%)", True)
    for i, r in enumerate(rows):
        y, x = n - 1 - i, 0.0
        for p in order:
            frac = 100.0 * r["sig"].get(p, 0.0) / r["S"] if r["S"] > 0 else 0.0
            if frac > 0:
                box(x, y, x + frac, styles[p][1])
                x += frac
        t = ROOT.TLatex(1.5, y + 0.5, f"#bf{{{r['S']:.2f} expected events}}")
        t.SetTextAlign(12)
        t.SetTextFont(63)
        t.SetTextSize(px - 2)
        t.SetTextColor(ROOT.kWhite)
        t.Draw()
        keep.append(t)
    ytop = 1.0 - 45.0 / height
    leg = ROOT.TLegend(0.895, ytop - 26.0 * len(order) / height, 1.0, ytop)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(43)
    leg.SetTextSize(px + 1)
    for p in order:
        dummy = ROOT.TH1F(f"leg_{p}", "", 1, 0, 1)
        dummy.SetFillColor(styles[p][1])
        dummy.SetLineWidth(0)
        leg.AddEntry(dummy, styles[p][0], "f")
        keep.append(dummy)
    leg.Draw()
    lat = ROOT.TLatex()
    lat.SetNDC()
    lat.SetTextFont(43)
    lat.SetTextSize(px + 5)
    lat.SetTextAlign(11)
    lat.DrawLatex(0.36, 1.0 - 32.0 / height, "#bf{CMS} " + args.cms_text)

    # dominant mode colours the S/B bars, as in HIG-19-006
    def dom(r):
        """Colour of the row's dominant signal process (grey if no signal), used for S/B bars."""
        return styles[max(order, key=lambda p: r["sig"].get(p, 0.0))][1] if r["S"] > 0 else ROOT.kGray

    sob = [100.0 * r["S"] / (r["S"] + r["B"]) if r["S"] + r["B"] > 0 else 0.0 for r in rows]
    ssb = [r["S"] / math.sqrt(r["B"]) if r["B"] > 0 else 0.0 for r in rows]
    for k, (vals, title) in enumerate([(sob, "S/(S+B) (%)"), (ssb, "S/#sqrt{B}")], start=1):
        frame(pads[k], f"f{k}", max(vals + [1e-6]) * 1.15, title, False)
        for i, r in enumerate(rows):
            if vals[i] > 0:
                box(0.0, n - 1 - i, vals[i], dom(r))
    pads[2].cd()
    lat.SetTextAlign(31)
    lat.SetTextSize(px + 3)
    lat.DrawLatex(0.98, 1.0 - 32.0 / height, args.lumi_text)

    for ext in ("pdf", "png"):
        c.SaveAs(f"{outbase}.{ext}")
    return sob, ssb


def main():
    """Build rows, draw the plot, write the yield CSV and print a summary table."""
    args = parse_args()
    rows, order = build_rows(args)
    for r in rows:
        if r["S"] <= 0:
            print(f"WARNING: {r['label']} has no expected signal")
    label = os.path.basename(os.path.normpath(args.base_dir))
    outdir = args.outdir or os.path.join(
        "validation", "signal_composition", label, f"{args.postfix}_{args.years.replace(',', '_')}")
    os.makedirs(outdir, exist_ok=True)
    tag = args.tag or ("binned" if not args.bin_groups else "groups_" + args.bin_groups.replace(",", "_"))
    tag = tag.replace(":", "-").replace(";", "__")
    outbase = os.path.join(outdir, f"signal_composition_{tag}")
    sob, ssb = draw(rows, order, args, outbase)

    with open(outbase + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "region", "dnn_bins", "S", "B"] + order + ["S_over_SplusB_pct", "S_over_sqrtB"])
        for r, a, b in zip(rows, sob, ssb):
            w.writerow([r["label"], r["region"], r["bins"], f"{r['S']:.4f}", f"{r['B']:.3f}"]
                       + [f"{r['sig'].get(p, 0.0):.4f}" for p in order] + [f"{a:.4f}", f"{b:.4f}"])
    comb = math.sqrt(sum(x * x for x in ssb))
    print(f"\n{'category':34s} {'S':>8s} {'B':>10s} {'S/(S+B)%':>9s} {'S/sqrtB':>8s}")
    for r, a, b in zip(rows, sob, ssb):
        print(f"{r['label']:34s} {r['S']:8.3f} {r['B']:10.2f} {a:9.3f} {b:8.4f}")
    print(f"Quadrature sum of S/sqrt(B) over categories: {comb:.3f}")
    print(f"Wrote {outbase}.{{pdf,png,csv}}")


if __name__ == "__main__":
    main()
