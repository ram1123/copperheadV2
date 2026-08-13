#!/usr/bin/env python3
"""Compare Stage-3 templates across runs: per-bin yields, purity, and migration.

Written for the adversarial-systematics study, where the significance changed far
more than any training metric did. `val_auc_weighted` moved by 0.0001 while the
top-bin S/B fell 24%, so training metrics could not see the damage at all -- only
the binned templates could. This is the tool that made the damage visible, and it
is study-agnostic: any change that moves events between score bins shows up here.

What to look at:

  * S/B per bin      -- purity. The high-score bins are where the significance is
                        made; losing purity there is what costs sensitivity, and it
                        can happen while the total yields stay fixed.
  * migration         -- total background is conserved by construction, so a run
                        that "gains" background in the tail took it from the bulk.
                        A one-way flow into the top bins is the signature of a
                        confidence-inflation pathology rather than better ordering.
  * quadrature S/sqrt(B) -- a cheap stand-in for the stat-only significance that
                        needs no `combine` call, useful for triage before running
                        the full chain.

Example
-------
    pixi run -e default python new_features/adversarial_syst_vbf_dnn_study/scripts/\\
        compare_template_migration.py \\
        --run null=Aug10_2026_advLambdaOFFKeepDeg \\
        --run lam0.005=Aug11_2026_advVarAB_lam0005keepDeg \\
        --run lam0.082=Aug11_2026_advVarAB_lam0082keepDeg \\
        --year 2018

Layout assumed (as produced by `scripts/produce_combine_cards.sh`):

    <stage3-base>/stage3_datacards_<postfix>/
                  stage3_templates_<postfix>/
                  score_<label>/vbf_h-peak_<year>.root
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import uproot

DEFAULT_LABEL = "Run2_NanoV15_forVBFChannel_July06_2026_jetUncRedo"
DEFAULT_BASE = "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean"
SIGNAL = ("ggH_hmm", "qqH_hmm")
BACKGROUND = ("DY", "DYVBF", "EWK", "TT+ST", "VV", "VVV")


def template_path(base: str, label: str, postfix: str, year: str) -> Path:
    return (Path(base) / label / f"stage3_datacards_{postfix}"
            / f"stage3_templates_{postfix}" / f"score_{label}"
            / f"vbf_h-peak_{year}.root")


def load(path: Path, signal, background):
    """Return (S, B) summed over the nominal (non-variation) histograms."""
    with uproot.open(path) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        missing = [p for p in list(signal) + list(background) if p not in keys]
        if missing:
            raise KeyError(f"{path}: missing processes {missing}")
        s = sum(f[p].values() for p in signal)
        b = sum(f[p].values() for p in background)
    return np.asarray(s, float), np.asarray(b, float)


def quadrature(s: np.ndarray, b: np.ndarray) -> float:
    """sqrt(sum_bins S^2/B); a combine-free proxy for the stat-only significance."""
    ok = b > 0
    return float(np.sqrt(np.sum(s[ok] ** 2 / b[ok])))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=POSTFIX",
                    help="repeatable; the FIRST --run is the reference for deltas")
    ap.add_argument("--year", default="2018")
    ap.add_argument("--label", default=DEFAULT_LABEL)
    ap.add_argument("--stage3-base", default=DEFAULT_BASE)
    ap.add_argument("--tail-from", type=int, default=14,
                    help="first bin counted as 'tail' in the migration summary")
    ap.add_argument("--signal", default=",".join(SIGNAL))
    ap.add_argument("--background", default=",".join(BACKGROUND))
    args = ap.parse_args(argv)

    signal = tuple(x for x in args.signal.split(",") if x)
    background = tuple(x for x in args.background.split(",") if x)

    runs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for spec in args.run:
        if "=" not in spec:
            ap.error(f"--run needs NAME=POSTFIX, got {spec!r}")
        name, postfix = spec.split("=", 1)
        path = template_path(args.stage3_base, args.label, postfix, args.year)
        if not path.exists():
            print(f"  SKIP {name}: no template at {path}", file=sys.stderr)
            continue
        runs[name] = load(path, signal, background)
    if not runs:
        print("no runs could be loaded", file=sys.stderr)
        return 2

    names = list(runs)
    ref = names[0]
    nbins = len(runs[ref][0])

    print(f"\n=== {args.year} SR, {nbins} bins "
          f"(reference = {ref}) ===\n")
    head = f"{'bin':>4} |" + "".join(f"{f'B {n}':>13}" for n in names)
    head += " |" + "".join(f"{f'S/B {n}':>13}" for n in names)
    print(head)
    print("-" * len(head))
    for i in range(nbins):
        row = f"{i:>4} |"
        for n in names:
            row += f"{runs[n][1][i]:13.2f}"
        row += " |"
        for n in names:
            s, b = runs[n][0][i], runs[n][1][i]
            row += f"{(s / b if b > 0 else float('nan')):13.3f}"
        print(row)

    print(f"\n{'run':<16}{'total B':>12}{'bulk B':>12}{'tail B':>12}"
          f"{'tail S':>10}{'migrated':>11}{'quad S/sqrtB':>14}")
    t = args.tail_from
    ref_tail = runs[ref][1][t:].sum()
    for n in names:
        s, b = runs[n]
        mig = b[t:].sum() - ref_tail
        print(f"{n:<16}{b.sum():12.1f}{b[:t].sum():12.2f}{b[t:].sum():12.2f}"
              f"{s[t:].sum():10.3f}{mig:+11.2f}{quadrature(s, b):14.4f}")

    print(f"\n'migrated' = tail background relative to {ref}. Total background is "
          f"conserved,\nso a positive value means events moved out of the bulk "
          f"(bins 0-{t - 1}) into\nthe tail (bins {t}-{nbins - 1}) -- not that any "
          f"background was created.\n")
    for n in names:
        neg = int((runs[n][1] < 0).sum())
        if neg:
            print(f"  WARNING {n}: {neg} bin(s) with negative background -- "
                  f"combine handles these badly and they inflate systematics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
