#!/usr/bin/env python3
"""Occupancy guard on the Stage-3 templates a re-derived binning produced.

Why
---
This study re-runs ``scan_bins_for_dnn.py`` at every lambda.  That scanner picks
edges from *its own* background estimate, which is measured to disagree with what
Stage-2 actually produces by 2-9x (see the TODO next to ``bkg_globs`` in
``scan_bins_for_dnn.py``).  In the predecessor task that mismatch left a bin
holding 0.00576 background events built from a single MC event -- effective
entries ``n_eff = B^2/var(B) = 0.03`` -- which contributed a spurious
``S/sqrt(B)`` comparable to an entire year and inflated the reference by ~8%.

A per-lambda binning can reintroduce that at any point in the sweep.  This script
makes it visible on every run instead of only when someone thinks to look: it
reads the templates the fit consumes, and reports the per-bin background yield
and effective entries.

Resolution is driven by the datacard's own ``shapes`` line, so it follows the
Stage-3 layout rather than assuming it.

    check_template_occupancy.py <stage3_datacard_dir> --year 2017
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import uproot

SIGNAL = ("ggH_hmm", "qqH_hmm")
BACKGROUND = ("DY", "DYVBF", "EWK", "TT+ST", "VV", "VVV")

#: Thresholds the predecessor task settled on when it rebinned 21 -> 17 bins.
MIN_N_EFF = 10.0
MIN_B = 0.5


def template_from_datacard(card: Path) -> Path:
    """Resolve the ROOT file from the datacard's `shapes` line."""
    for line in card.read_text().splitlines():
        parts = line.split()
        if parts[:1] == ["shapes"] and len(parts) >= 4:
            return (card.parent / parts[3]).resolve()
    raise ValueError(f"no 'shapes' line in {card}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage3_dir", help="dir holding datacard_vbf_SR_<year>.txt")
    ap.add_argument("--year", default="2017")
    ap.add_argument("--postfix", default=None, help="run label recorded in the output")
    ap.add_argument("--out", default=None, help="write the report here as JSON")
    ap.add_argument("--min-n-eff", type=float, default=MIN_N_EFF)
    ap.add_argument("--min-b", type=float, default=MIN_B)
    ap.add_argument("--signal", default=",".join(SIGNAL))
    ap.add_argument("--background", default=",".join(BACKGROUND))
    args = ap.parse_args(argv)

    card = Path(args.stage3_dir) / f"datacard_vbf_SR_{args.year}.txt"
    # The card may already be stripped; the shapes line is untouched either way.
    if not card.exists():
        card_full = card.with_suffix(card.suffix + ".full")
        if not card_full.exists():
            print(f"  OCCUPANCY: no datacard at {card}", file=sys.stderr)
            return 2
        card = card_full

    path = template_from_datacard(card)
    if not path.exists():
        print(f"  OCCUPANCY: template missing at {path}", file=sys.stderr)
        return 2

    signal = [x for x in args.signal.split(",") if x]
    background = [x for x in args.background.split(",") if x]

    with uproot.open(path) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        missing = [p for p in signal + background if p not in keys]
        if missing:
            print(f"  OCCUPANCY: {path} is missing {missing}", file=sys.stderr)
            return 2
        s = np.sum([f[p].values() for p in signal], axis=0)
        b = np.sum([f[p].values() for p in background], axis=0)
        b_var = np.sum([f[p].variances() for p in background], axis=0)

    # n_eff = B^2/var(B): how many unweighted MC events a bin's yield is really
    # worth. Barlow-Beeston treats a bin with n_eff ~ 1 as essentially unmeasured.
    with np.errstate(divide="ignore", invalid="ignore"):
        n_eff = np.where(b_var > 0, b**2 / b_var, 0.0)

    bad = [i for i in range(len(b))
           if b[i] < args.min_b or n_eff[i] < args.min_n_eff]

    report = {
        "postfix": args.postfix,
        "year": args.year,
        "template": str(path),
        "n_bins": int(len(b)),
        "thresholds": {"min_n_eff": args.min_n_eff, "min_b": args.min_b},
        "total_background": float(b.sum()),
        "total_signal": float(s.sum()),
        "per_bin": [
            {"bin": i, "S": float(s[i]), "B": float(b[i]), "n_eff": float(n_eff[i])}
            for i in range(len(b))
        ],
        "failing_bins": bad,
        "passed": not bad,
    }

    print(f"  OCCUPANCY {args.postfix or path.name}: {len(b)} bins, "
          f"B={b.sum():.1f}, S={s.sum():.3f}")
    for i in range(len(b)):
        flag = "  <-- BELOW FLOOR" if i in bad else ""
        print(f"    bin {i:2d}  S={s[i]:9.4f}  B={b[i]:10.4f}  n_eff={n_eff[i]:9.2f}{flag}")
    if bad:
        print(f"  OCCUPANCY WARNING: bins {bad} are below the floor "
              f"(B >= {args.min_b}, n_eff >= {args.min_n_eff}). This is the artefact "
              f"the predecessor task rebinned away; treat this point's significance "
              f"as unreliable until checked.")
    else:
        print("  OCCUPANCY: all bins clear the floor.")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"  OCCUPANCY: report -> {args.out}")

    # Deliberately 0 even when bins fail: this is a guard that must not abort a
    # chain, only make the problem impossible to miss.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
