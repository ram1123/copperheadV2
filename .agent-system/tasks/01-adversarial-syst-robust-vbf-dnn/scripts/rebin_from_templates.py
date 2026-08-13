#!/usr/bin/env python3
"""Derive a DNN binning whose every bin is populated in EVERY year.

Why not `scan_bins_for_dnn.py --min-background-per-bin`: that script's own TODO
(next to `bkg_globs`) records that the background it sees disagrees with Stage-2
by 2-9x, and states the consequence outright -- "bins that comfortably clear the
guards here come out (nearly) empty in the stage2 h-peak plots, worst in
2016postVFP". Tuning its guard is tuning against numbers the fit never sees, which
is how the current binning got a 2016postVFP bin holding 0.0058 background events.

This works from the Stage-3 template ROOT files instead, i.e. exactly the yields
the fit consumes. Merging bins is just summing, so candidate binnings can be
evaluated on existing templates without re-running Stage-2/3 at all.

Occupancy metric: effective MC entries n_eff = B^2 / var(B) per year, which is
what `autoMCStats` actually cares about; a bin can carry a respectable weighted
yield off one event and still be worthless.

    python rebin_from_templates.py [--floor-neff 10] [--floor-b 0.5] [--write]
"""
from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import uproot
import yaml

REPO = Path(__file__).resolve().parents[4]
BINNING = REPO / "configs" / "MVA" / "VBF" / "dnn_binning.yaml"
OUT = Path(__file__).resolve().parent.parent

S = ("/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/"
     "Run2_NanoV15_forVBFChannel_July06_2026_jetUncRedo")
RUNS = {
    "seed1": "advBaselineSeedCfg12345",
    "seed2": "advBaselineSeed20260810",
    "seed3": "advBaselineSeed777001",
}
YEARS = ["2018", "2017", "2016postVFP", "2016preVFP"]
BKG = ["DY", "DYVBF", "EWK", "TT+ST", "VV", "VVV"]
SIG = ["qqH_hmm", "ggH_hmm"]


def load(run_tag: str, year: str):
    f = glob.glob(f"{S}/stage3_datacards_Aug10_2026_{run_tag}/"
                  f"stage3_templates_Aug10_2026_{run_tag}/score_*/vbf_h-peak_{year}.root")
    if not f:
        raise SystemExit(f"no template for {run_tag} {year}")
    r = uproot.open(f[0])
    keys = {k.split(";")[0] for k in r.keys()}

    def tot(procs):
        v = e = None
        for p in procs:
            if p in keys:
                vv = np.asarray(r[p].values(), float)
                ee = np.asarray(r[p].variances(), float)
                v = vv.copy() if v is None else v + vv
                e = ee.copy() if e is None else e + ee
        return v, e

    b, be = tot(BKG)
    s, _ = tot(SIG)
    return b, be, s


def group_sums(x, groups):
    return np.array([x[lo:hi].sum() for lo, hi in groups])


def asimov_z(s, b):
    """Same additive Asimov Z^2 the repo's scanner uses, summed over bins."""
    s = np.maximum(np.asarray(s, float), 0.0)
    b = np.maximum(np.asarray(b, float), 0.0)
    z2 = np.where(b > 1e-9, 2.0 * ((s + b) * np.log1p(s / np.maximum(b, 1e-30)) - s), 2.0 * s)
    return float(np.sqrt(z2.sum()))


def build_groups(data, nbins, floor_neff, floor_b):
    """Merge from the high-score end down until every year clears the floors.

    Only the tail is touched: once we reach bins that already pass on their own,
    the rest of the binning is left exactly as it was, so low-score resolution
    (where the statistics are ample) is preserved.
    """
    def passes(lo, hi):
        for year in YEARS:
            for run in RUNS:
                b, be, _ = data[(run, year)]
                bs, bes = b[lo:hi].sum(), be[lo:hi].sum()
                if bs < floor_b:
                    return False
                if bes <= 0 or bs * bs / bes < floor_neff:
                    return False
        return True

    groups, hi = [], nbins
    while hi > 0:
        lo = hi - 1
        while lo > 0 and not passes(lo, hi):
            lo -= 1
        if not passes(lo, hi):          # ran out of bins to absorb
            groups.append((0, hi))
            hi = 0
            break
        groups.append((lo, hi))
        hi = lo
        # once single bins pass on their own, keep the original binning below
        if all(passes(i, i + 1) for i in range(hi)):
            groups.extend([(i, i + 1) for i in range(hi)])
            hi = 0
    return sorted(groups)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor-neff", type=float, default=10.0)
    ap.add_argument("--floor-b", type=float, default=0.5)
    ap.add_argument("--write", action="store_true", help="write configs/MVA/VBF/dnn_binning.yaml")
    args = ap.parse_args()

    edges = list(yaml.safe_load(BINNING.read_text())["edges"])
    nbins = len(edges) - 1
    data = {(run, y): load(tag, y) for run, tag in RUNS.items() for y in YEARS}

    print("=== current binning, per-seed Asimov Z (quadrature over years) ===")
    def per_run_z(groups):
        out = {}
        for run in RUNS:
            z2 = 0.0
            for y in YEARS:
                b, _, s = data[(run, y)]
                z2 += asimov_z(group_sums(s, groups), group_sums(b, groups)) ** 2
            out[run] = float(np.sqrt(z2))
        return out

    cur = [(i, i + 1) for i in range(nbins)]
    zc = per_run_z(cur)
    spread = (max(zc.values()) - min(zc.values())) / np.mean(list(zc.values()))
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in zc.items()) + f"   spread={100*spread:.2f}%")

    print(f"\n=== candidate floors (n_eff per year, B per year) ===")
    best = None
    for fn, fb in [(5, 0.2), (10, 0.5), (20, 1.0), (30, 2.0)]:
        g = build_groups(data, nbins, fn, fb)
        z = per_run_z(g)
        sp = (max(z.values()) - min(z.values())) / np.mean(list(z.values()))
        print(f"  n_eff>={fn:<3} B>={fb:<4} -> {len(g)} bins   "
              + "  ".join(f"{k}={v:.4f}" for k, v in z.items())
              + f"   spread={100*sp:.2f}%")
        if fn == args.floor_neff and fb == args.floor_b:
            best = (g, z, sp)

    groups = build_groups(data, nbins, args.floor_neff, args.floor_b)
    new_edges = [edges[lo] for lo, _ in groups] + [edges[groups[-1][1]]]
    zn = per_run_z(groups)
    spn = (max(zn.values()) - min(zn.values())) / np.mean(list(zn.values()))

    print(f"\n=== chosen: n_eff>={args.floor_neff}, B>={args.floor_b} ===")
    print(f"  {nbins} bins -> {len(groups)} bins")
    print(f"  merged groups (original bin indices): {[g for g in groups if g[1]-g[0]>1]}")
    print(f"  per-seed Z: " + "  ".join(f"{k}={v:.4f}" for k, v in zn.items()))
    print(f"  seed spread: {100*spread:.2f}%  ->  {100*spn:.2f}%")
    print("\n  worst-year occupancy per NEW bin (min over seeds):")
    print(f"  {'bin':>4}{'lo':>8}{'hi':>8}{'min B':>10}{'min n_eff':>11}")
    for j, (lo, hi) in enumerate(groups):
        mb = min(data[(r, y)][0][lo:hi].sum() for r in RUNS for y in YEARS)
        mn = min((data[(r, y)][0][lo:hi].sum() ** 2) / max(data[(r, y)][1][lo:hi].sum(), 1e-30)
                 for r in RUNS for y in YEARS)
        print(f"  {j:>4}{edges[lo]:>8.3f}{edges[hi]:>8.3f}{mb:>10.4g}{mn:>11.2f}")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": "merge Stage-3 template bins until every year clears the floors",
        "floor_neff_per_year": args.floor_neff,
        "floor_background_per_year": args.floor_b,
        "old_n_bins": nbins,
        "new_n_bins": len(groups),
        "merged_groups": [list(g) for g in groups],
        "new_edges": new_edges,
        "per_seed_asimov_Z_old": zc,
        "per_seed_asimov_Z_new": zn,
        "seed_spread_old_pct": 100 * spread,
        "seed_spread_new_pct": 100 * spn,
    }
    (OUT / "rebin_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n  wrote {OUT / 'rebin_report.json'}")

    if args.write:
        BINNING.write_text(
            "# REBINNED "
            f"{datetime.now().date()} by "
            ".agent-system/tasks/01-adversarial-syst-robust-vbf-dnn/scripts/rebin_from_templates.py\n"
            "#\n"
            "# Derived by merging the previous hand-set edges until EVERY data-taking\n"
            "# year clears an occupancy floor measured on the Stage-3 templates the fit\n"
            f"# actually consumes: n_eff = B^2/var(B) >= {args.floor_neff} and B >= {args.floor_b}\n"
            "# per year, evaluated as the worst case over three baseline seeds.\n"
            "#\n"
            "# NOT derived from scan_bins_for_dnn.py's background, which its own TODO\n"
            "# documents as 2-9x off from Stage-2 -- that mismatch is what left a\n"
            "# 2016postVFP bin holding 0.0058 background events (n_eff = 0.03) and\n"
            "# inflating that year's significance by ~68%.\n"
            f"n_bins: {len(new_edges) - 1}\n"
            "edges:\n" + "".join(f"- {e}\n" for e in new_edges)
        )
        print(f"  WROTE {BINNING}")
    else:
        print("  (dry run; pass --write to update configs/MVA/VBF/dnn_binning.yaml)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
