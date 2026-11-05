#!/usr/bin/env python3
"""
Background-MC sparsity audit of the VBF stage-3 SR bins: per year, empty-B and thin bins (B < --bmin or MC rel.
error > --relmax) with their share of the counting Z^2, and the counting Z after merging bins from the top until
none is thin. Counting Z has no systematics or MC-stat uncertainty: a sparsity check, not a combine result.

Usage:
  python scripts/audit_vbf_datacard_bins.py <save_path>/stage3_datacards_<postfix>/score_<label> [--bmin 1 --relmax 0.3]
"""
import argparse
import re
from pathlib import Path

import numpy as np
import uproot


def z2(S, B):
    safe = np.where(B > 0, B, 1.0)
    return np.where(B > 0, 2 * ((S + B) * np.log1p(S / safe) - S), 0.0)


def read_sr(card: Path):
    """Signal, background and background MC variance per SR bin of one datacard."""
    lines = [line.split() for line in open(card)]
    procs = [line for line in lines if line and line[0] == "process"]
    names, idx = procs[0][1:], [int(x) for x in procs[1][1:]]
    shapes = next(line for line in lines if line and line[0] == "shapes")
    f = uproot.open(card.parent / shapes[3])
    sig = [n for n, i in zip(names, idx) if i <= 0 and n in f]
    bkg = [n for n, i in zip(names, idx) if i > 0 and n in f]
    S = sum(f[n].to_numpy()[0] for n in sig)
    B = sum(f[n].to_numpy()[0] for n in bkg)
    V = sum(f[n].variances() for n in bkg)
    return S, B, V


def merge_from_top(S, B, V, bmin, relmax):
    # TODO: bmin/relmax defaults are ad hoc, as in train_dnn.binned_ams (see its TODO)
    mS, mB = [], []
    s = b = v = 0.0
    for i in range(len(S) - 1, -1, -1):  # top score bin first
        s += S[i]; b += B[i]; v += V[i]
        if b >= bmin and np.sqrt(v) <= relmax * b:
            mS.append(s); mB.append(b)
            s = b = v = 0.0
    if s or b:
        if mS:
            mS[-1] += s; mB[-1] += b
        else:
            mS.append(s); mB.append(b)
    return np.array(mS), np.array(mB)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("card_dir", help="stage3_datacards_<postfix>/score_<label> (holds datacard_vbf_SR_<year>.txt)")
    p.add_argument("--bmin", type=float, default=1.0, help="minimum background per bin (default 1)")
    p.add_argument("--relmax", type=float, default=0.3, help="maximum background MC relative error (default 0.3)")
    p.add_argument("--empty", type=float, default=1e-3, help="B below this counts as empty (default 0.001)")
    args = p.parse_args(argv)

    cards = sorted(Path(args.card_dir).glob("datacard_vbf_SR_*.txt"))
    if not cards:
        raise SystemExit(f"no datacard_vbf_SR_<year>.txt in {args.card_dir}")
    tot, tot_merged = 0.0, 0.0
    for card in cards:
        year = re.sub(r"^datacard_vbf_SR_|\.txt$", "", card.name)
        S, B, V = read_sr(card)
        zz = z2(S, B)
        share = zz / zz.sum()
        empty = np.flatnonzero(B <= args.empty)
        thin = np.flatnonzero((B < args.bmin) | (np.sqrt(V) > args.relmax * np.maximum(B, 1e-9)))
        top = int(np.argmax(share))
        mS, mB = merge_from_top(S, B, V, args.bmin, args.relmax)
        zm = z2(mS, mB).sum()
        tot += zz.sum(); tot_merged += zm
        print(f"{year:12s} {len(S):2d} bins  Z {np.sqrt(zz.sum()):.3f} -> merged ({len(mS):2d} bins) {np.sqrt(zm):.3f}  "
              f"empty-B bins {empty.tolist()} ({share[empty].sum():.0%} of Z^2)  thin bins {len(thin)} "
              f"({share[thin].sum():.0%})  largest bin {top}: S {S[top]:.3f} B {B[top]:.3f}+-{np.sqrt(V[top]):.3f} "
              f"({share[top]:.0%})")
    print(f"{'all (quad)':12s}          Z {np.sqrt(tot):.3f} -> merged {np.sqrt(tot_merged):.3f}"
          f"   (merge: B >= {args.bmin:g}, MC rel. err. <= {args.relmax:g})")


if __name__ == "__main__":
    main()
