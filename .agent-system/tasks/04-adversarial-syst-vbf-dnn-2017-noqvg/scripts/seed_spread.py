#!/usr/bin/env python3
"""Turn the lambda=0.5/cut=all seed replicas into a noise band, and re-read the
grid against it.

The grid has one measurement per (lambda, score_cut) cell, so on its own it
cannot distinguish a response to the loss from run-to-run scatter. The replicas
change ONLY the training seed at a single setting, so their spread is that
scatter: same loss, same data, same hyperparameters, same warm start.

Reports, for each metric:

  sigma  -- sample standard deviation over the replicas (ddof=1)
  band   -- the +-2 sigma interval around the replica mean

and then flags which grid cells differ from the null control by more than
2 sigma. Cells inside the band are not evidence of anything; saying so is the
point of this script.

    seed_spread.py results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = [
    ("prefit", "pre-fit significance", "{:.5f}"),
    ("syst_headroom", "systematic headroom", "{:.4f}"),
    ("quadZ_clean", "quadZ_ok", "{:.4f}"),
    ("tv_background", "tv(background)", "{:.5f}"),
    ("tv_signal", "tv(signal)", "{:.5f}"),
    ("n_bins", "n_bins", "{:.1f}"),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--n-sigma", type=float, default=2.0)
    args = ap.parse_args(argv)

    runs = json.loads(Path(args.results).read_text())["runs"]

    # The replica set: same lambda, cut = all True. The original grid point (no
    # seed suffix) is the first sample -- it was run with the default seed and is
    # otherwise identical, so excluding it would throw away a measurement.
    replicas = [r for r in runs
                if r.get("lambda") == args.lam and r.get("score_cut") is None]
    if len(replicas) < 2:
        print(f"need >=2 replicas at lambda={args.lam}, cut=all; have {len(replicas)}")
        return 1

    null = next((r for r in runs if r["postfix"].endswith("_null")), None)

    print(f"\nSeed replicas at lambda={args.lam}, score_cut=all True "
          f"(n={len(replicas)}; only --seed differs)\n")
    for r in replicas:
        tag = f"seed={r['seed']}" if r.get("seed") else "seed=default"
        print(f"  {tag:<16} {r['postfix']}")

    print(f"\n{'metric':<22} {'mean':>12} {'sigma':>12} "
          f"{'+-' + str(args.n_sigma) + ' sigma band':>28}")
    print("-" * 78)
    sigmas = {}
    for key, label, fmt in METRICS:
        vals = np.array([r[key] for r in replicas
                         if r.get(key) is not None], dtype=float)
        if vals.size < 2:
            continue
        mu, sd = float(vals.mean()), float(vals.std(ddof=1))
        sigmas[key] = sd
        lo, hi = mu - args.n_sigma * sd, mu + args.n_sigma * sd
        print(f"{label:<22} {fmt.format(mu):>12} {fmt.format(sd):>12} "
              f"{'[' + fmt.format(lo) + ', ' + fmt.format(hi) + ']':>28}")

    if null is None:
        return 0

    print(f"\nGrid cells vs the null control, judged against {args.n_sigma} sigma "
          f"from the replicas:\n")
    print(f"{'run':<34} {'lam':>6} {'cut':>5} "
          f"{'d(prefit)':>11} {'n_sig':>7}   {'d(syst_hr)':>11} {'n_sig':>7}")
    print("-" * 92)
    grid = [r for r in runs if r.get("lambda") is not None]
    for r in sorted(grid, key=lambda x: (-x["lambda"],
                                         -1 if x["score_cut"] is None
                                         else x["score_cut"])):
        cut = "all" if r["score_cut"] is None else f"{r['score_cut']:.2f}"
        bits = []
        for key in ("prefit", "syst_headroom"):
            v, nv = r.get(key), null.get(key)
            if v is None or nv is None or key not in sigmas or sigmas[key] == 0:
                bits.append((None, None))
                continue
            d = v - nv
            bits.append((d, d / sigmas[key]))
        seed = f" seed={r['seed']}" if r.get("seed") else ""
        cells = []
        for d, n in bits:
            if d is None:
                cells.append(f"{'':>11} {'':>7}")
            else:
                mark = "*" if abs(n) >= args.n_sigma else " "
                cells.append(f"{d:>+11.5f} {n:>+6.1f}{mark}")
        print(f"{r['postfix'][-33:]:<34} {r['lambda']:>6.3g} {cut:>5} "
              f"{cells[0]}   {cells[1]}{seed}")

    print(f"\n* = differs from the null control by at least {args.n_sigma} sigma of "
          f"the seed-only spread.")
    print("  Unmarked cells are consistent with changing nothing but the seed, and "
          "must not be\n  ranked against each other.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
