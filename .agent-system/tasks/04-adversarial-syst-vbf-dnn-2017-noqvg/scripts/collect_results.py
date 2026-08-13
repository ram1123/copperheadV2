#!/usr/bin/env python3
"""Collect the 2017 no-QvG lambda x score_cut grid into one table.

Per run:

  pre-fit    -- the number the study is trying to raise
  stat-only  -- nuisances frozen; the ceiling pre-fit is working towards
  headroom   -- (stat_only - prefit)/prefit, the fraction of sensitivity the
                systematics cost. The loss under test is supposed to shrink this
                WITHOUT dragging stat-only down with it.
  quadZ      -- sqrt(sum_bins S^2/B), a combine-free proxy, reported twice: over
                ALL bins and over only the bins clearing the occupancy floor. The
                gap is how much of a run's apparent significance rests on bins the
                fit cannot support. The predecessor task had a point draw 75% of
                its quadrature Z^2 from a single bin holding 0.0018 background
                events, which faked a turnover in lambda -- hence the second column.
  tv_bkg /
  tv_sig     -- worst (over up/down and over Total/mu_roccor) total-variation
                distance between the nominal template and the variation template
                after renormalising to the nominal yield. This is the mechanism
                the loss acts on: if the consistency term works, these fall.

Two baselines matter and both are printed:

  step1  -- lambda = 0, one training. What the analysis has today.
  null   -- step1 followed by two further warm-started trainings with no penalty.
            Grid points must be read against THIS one, because it carries the same
            retraining cost with none of the loss.

    collect_results.py [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

LABEL = "Run2_NanoV15_forVBFChannel_July06_2026_jetUncRedo"
SAVE_PATH = Path("/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean") / LABEL
SEP = "=" * 40

#: postfixes look like Aug12_2026_2017nq_lam10cut06 / ..._step1 / ..._null
POINT_RE = re.compile(r"_lam(?P<lam>[0-9]+)cut(?P<cut>none|[0-9]+)(?:seed(?P<seed>[0-9]+))?$")


def parse_significance(path: Path, year: str):
    """(prefit, stat_only) from the LAST block for `year`, or (None, None).

    produce_significance.sh appends, so a directory can hold several blocks; the
    last one for the year wins, which is what a re-run is meant to mean.
    """
    if not path.exists():
        return None, None
    prefit = statonly = None
    for block in path.read_text().split(SEP):
        if f"Second argument: {year}" not in block:
            continue
        values = re.findall(r"^Significance:\s+([0-9.eE+-]+)", block, re.MULTILINE)
        if len(values) >= 1:
            prefit = float(values[0])
        statonly = float(values[1]) if len(values) >= 2 else None
    return prefit, statonly


def parse_syst_frozen(path: Path):
    """Significance with ONLY the two shape nuisances frozen, or None.

    Written by syst_only_significance.sh. This is the leg produce_significance.sh
    does not compute: its "stat only" freezes allConstrainedNuisances, which takes
    the autoMCStats parameters with it, so the headroom built from it is the cost
    of the systematics AND of the finite MC statistics together. Since the binning
    is re-derived per model, the MC-stat part swings for reasons unrelated to the
    loss -- lambda=0.5/cut=0.6 reached a 52.6% "headroom" on an unremarkable
    pre-fit purely that way. syst_headroom below is the part the loss can act on.
    """
    if not path.exists():
        return None
    values = re.findall(r"^Significance:\s+([0-9.eE+-]+)", path.read_text(),
                        re.MULTILINE)
    return float(values[-1]) if values else None


def quadrature(postfix: str, task_dir: Path, min_b: float, min_n_eff: float):
    """(quadZ all bins, quadZ over bins clearing the floor, dominant bad bin)."""
    path = task_dir / f"occupancy_{postfix}.json"
    if not path.exists():
        return None, None, None
    per_bin = json.loads(path.read_text())["per_bin"]
    s = np.array([r["S"] for r in per_bin])
    b = np.array([r["B"] for r in per_bin])
    n_eff = np.array([r["n_eff"] for r in per_bin])
    with np.errstate(divide="ignore", invalid="ignore"):
        z2 = np.where(b > 0, s**2 / b, 0.0)
    good = (b >= min_b) & (n_eff >= min_n_eff)
    worst = int(np.argmax(z2)) if z2.size else None
    if worst is not None and good[worst]:
        worst = None                     # only interesting if it is unsupported
    return float(np.sqrt(z2.sum())), float(np.sqrt(z2[good].sum())), worst


def shape_metrics(postfix: str, task_dir: Path):
    """Worst-case nominal-vs-variation shape distance per group, plus the detail.

    `n_bins` travels with these numbers on purpose. Total-variation distance is
    computed on binned templates, and merging bins is a contraction -- a coarser
    binning can only lower TV. Since the binning is re-derived per run, a fall in
    tv_* is only evidence about the loss once n_bins is held roughly fixed. The
    summary checks the correlation rather than assuming it away.
    """
    path = task_dir / "shape" / f"shape_{postfix}.json"
    if not path.exists():
        return {}
    report = json.loads(path.read_text())
    out = {}
    for group, g in report.get("groups", {}).items():
        if g.get("n_bins") is not None:
            out["n_bins"] = g["n_bins"]
        tv = [s["max_shape_tv"] for s in g.get("variations", {}).values()
              if "max_shape_tv" in s]
        rms = [s["max_shape_rms"] for s in g.get("variations", {}).values()
               if "max_shape_rms" in s]
        if tv:
            out[f"tv_{group}"] = max(tv)
        if rms:
            out[f"rms_{group}"] = max(rms)
        for nu, s in g.get("variations", {}).items():
            if "max_shape_tv" in s:
                out[f"tv_{group}_{nu}"] = s["max_shape_tv"]
    return out


def decode(postfix: str):
    """(lambda, cut, seed) from a grid postfix; (None, None, None) otherwise.

    The slug drops the decimal point (1.0 -> "10", 0.005 -> "0005"), which is not
    invertible on its own, so the decode is table-driven from the values the
    driver actually sweeps.
    """
    m = POINT_RE.search(postfix)
    if not m:
        return None, None, None
    lam_by_slug = {"50": 5.0, "10": 1.0, "05": 0.5, "001": 0.01, "0005": 0.005}
    cut_by_slug = {"none": None, "06": 0.6, "10": 1.0, "20": 2.0}
    lam = lam_by_slug.get(m.group("lam"))
    cut = cut_by_slug.get(m.group("cut"), "?")
    seed = m.group("seed")
    return lam, cut, (int(seed) if seed else None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", default="stage3_datacards_*2017nq*")
    ap.add_argument("--year", default="2017")
    ap.add_argument("--out", default=None)
    ap.add_argument("--task-dir", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--min-b", type=float, default=0.5)
    ap.add_argument("--min-n-eff", type=float, default=10.0)
    args = ap.parse_args(argv)

    task_dir = Path(args.task_dir)
    rows = []
    for d in sorted(SAVE_PATH.glob(args.pattern)):
        postfix = d.name.replace("stage3_datacards_", "")
        prefit, statonly = parse_significance(
            d / f"score_{LABEL}" / "significance.txt", args.year)
        if prefit is None:
            continue
        lam, cut, seed = decode(postfix)
        qa, qc, worst = quadrature(postfix, task_dir, args.min_b, args.min_n_eff)
        syst_frozen = parse_syst_frozen(
            d / f"score_{LABEL}" / "significance_systfrozen.txt")
        row = {
            "postfix": postfix,
            "lambda": lam,
            "score_cut": cut,
            "seed": seed,
            "prefit": prefit,
            "stat_only": statonly,
            "syst_frozen": syst_frozen,
            "headroom": (statonly - prefit) / prefit if statonly else None,
            "syst_headroom": ((syst_frozen - prefit) / prefit
                              if syst_frozen else None),
            "quadZ_all": qa,
            "quadZ_clean": qc,
            "dominant_unsupported_bin": worst,
        }
        row.update(shape_metrics(postfix, task_dir))
        rows.append(row)

    if not rows:
        print(f"no results yet under {SAVE_PATH}/{args.pattern}", file=sys.stderr)
        return 1

    ref = next((r for r in rows if r["postfix"].endswith("_step1")), None)
    null = next((r for r in rows if r["postfix"].endswith("_null")), None)
    base = null or ref
    for r in rows:
        r["vs_step1"] = ((r["prefit"] - ref["prefit"]) / ref["prefit"]
                         if ref and ref["prefit"] else None)
        r["vs_null"] = ((r["prefit"] - null["prefit"]) / null["prefit"]
                        if null and null["prefit"] else None)

    width = max(len(r["postfix"]) for r in rows)
    print(f"\n{args.year} VBF, no QvG inputs, Total + mu_roccor shape nuisances only\n")
    hdr = (f"{'run':<{width}} {'lam':>6} {'cut':>5} {'pre-fit':>9} {'stat-only':>10} "
           f"{'syst_hr':>8} {'tot_hr':>8} {'vs step1':>9} {'vs null':>9} "
           f"{'quadZ_ok':>9} {'bins':>5} {'tv_bkg':>8} {'tv_sig':>8}")
    print(hdr)
    print("-" * len(hdr))
    def cell(row, key, width_, fmt, pct=False):
        v = row.get(key)
        if v is None:
            return " " * width_
        return format(v * 100 if pct else v, fmt)

    for r in rows:
        lam = f"{r['lambda']:6.3g}" if r["lambda"] is not None else "     -"
        if r["lambda"] is None:
            cut = "    -"                       # step1 / null carry no penalty
        elif r["score_cut"] is None:
            cut = "  all"                       # score_cut = all True
        elif isinstance(r["score_cut"], float):
            cut = f"{r['score_cut']:5.2f}"
        else:
            cut = "    ?"
        print(" ".join([
            f"{r['postfix']:<{width}}", lam, cut,
            cell(r, "prefit", 9, "9.5f"),
            cell(r, "stat_only", 10, "10.5f"),
            cell(r, "syst_headroom", 8, "+7.2f", pct=True) + ("%" if r.get("syst_headroom") is not None else " "),
            cell(r, "headroom", 8, "+7.2f", pct=True) + ("%" if r["headroom"] is not None else " "),
            cell(r, "vs_step1", 9, "+8.2f", pct=True) + ("%" if r["vs_step1"] is not None else " "),
            cell(r, "vs_null", 9, "+8.2f", pct=True) + ("%" if r["vs_null"] is not None else " "),
            cell(r, "quadZ_clean", 9, "9.4f"),
            f"{r['n_bins']:5d}" if r.get("n_bins") else " " * 5,
            cell(r, "tv_background", 8, "8.4f"),
            cell(r, "tv_signal", 8, "8.4f"),
        ]))

    if ref:
        print(f"\nstep1 reference = {ref['postfix']} (lambda = 0, one training)")
    if null:
        print(f"null control    = {null['postfix']} (two extra trainings, no penalty)")
        if ref and ref["prefit"]:
            drift = (null["prefit"] - ref["prefit"]) / ref["prefit"] * 100
            print(f"  the retraining procedure alone moves pre-fit by {drift:+.2f}%; "
                  f"grid points are attributable to the loss only beyond that.")
    if base and base.get("headroom") is not None:
        print(f"\nheadroom at the baseline ({base['postfix']}): "
              f"{base['headroom'] * 100:.2f}%")

    flagged = [r for r in rows if r["dominant_unsupported_bin"] is not None]
    if flagged:
        print("\nWARNING: the single largest contributor to quadZ is a bin BELOW the "
              f"occupancy floor (B >= {args.min_b}, n_eff >= {args.min_n_eff}) in:")
        for r in flagged:
            drop = 100 * (1 - (r["quadZ_clean"] / r["quadZ_all"]) ** 2)
            print(f"  {r['postfix']}: bin {r['dominant_unsupported_bin']}; "
                  f"{drop:.0f}% of Z^2 rests on unsupported bins")
        print("  Compare these runs on quadZ_ok, not on pre-fit.")

    print("\nsyst_hr  = (syst_frozen - prefit)/prefit: cost of the Total/mu_roccor shape")
    print("           nuisances ALONE, autoMCStats left floating. This is what the loss targets.")
    print("tot_hr   = (stat_only - prefit)/prefit, the legacy column: freezes the shape")
    print("           nuisances AND autoMCStats, so it also carries the MC-statistical cost,")
    print("           which swings with the per-run rebinning. Do not read it as 'systematics'.")
    print("quadZ_ok = sqrt(sum S^2/B) over bins clearing the occupancy floor")
    print("tv_*     = worst nominal-vs-variation total-variation distance over "
          "{Total, mu_roccor} x {up, down},")
    print("           computed after renormalising each variation to the nominal "
          "yield. Lower = the")
    print("           systematic moves the template shape less, which is what the "
          "consistency term targets.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"year": args.year,
             "reference": ref["postfix"] if ref else None,
             "null_control": null["postfix"] if null else None,
             "runs": rows}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
