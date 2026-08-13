#!/usr/bin/env python3
"""Figures for the final summary: the lambda x score_cut grid, and whether the
nominal-vs-variation template agreement actually improved.

Three panels, from results.json (written by collect_results.py):

  1. heatmaps of pre-fit significance and of systematic headroom over the grid,
     each annotated with its value and drawn on a diverging scale centred on the
     NULL CONTROL -- not on the step-1 reference. A grid point is only evidence
     about the loss to the extent it beats the same retraining with no penalty.
  2. tv_background and tv_signal against lambda, one line per score_cut, with the
     step-1 and null baselines drawn as horizontal rules. This is the direct
     answer to "did nominal-vs-variation consistency improve": the curves must
     sit BELOW the null line for the mechanism to be working.
  3. headroom against tv_background -- the consistency check between the two. If
     shrinking the template shift is what buys sensitivity, these correlate; if
     they do not, the loss is moving significance by some other route and the
     mechanism story is wrong.

    plot_grid_summary.py results.json --out reports/grid_summary.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAMBDAS = [5.0, 1.0, 0.5, 0.01, 0.005]
CUTS = [None, 0.6, 1.0, 2.0]
CUT_LABEL = {None: "all True", 0.6: "atanh>0.6", 1.0: "atanh>1.0", 2.0: "atanh>2.0"}


def grid_of(runs, key):
    """[len(CUTS), len(LAMBDAS)] array of `key`, NaN where a point is missing."""
    out = np.full((len(CUTS), len(LAMBDAS)), np.nan)
    for r in runs:
        if r.get("lambda") is None:
            continue
        try:
            i, j = CUTS.index(r["score_cut"]), LAMBDAS.index(r["lambda"])
        except ValueError:
            continue
        v = r.get(key)
        if v is not None:
            out[i, j] = v
    return out


def heatmap(ax, data, title, centre, fmt="{:.4f}", invert=False):
    """Diverging map centred on `centre`. `invert` flips which end reads as good,
    for quantities where lower is better -- by reversing the colormap, not by
    negating the data, so the colorbar keeps the real values."""
    finite = data[np.isfinite(data)]
    if finite.size and centre is not None:
        span = max(abs(finite - centre).max(), 1e-12)
        vmin, vmax = centre - span, centre + span
    else:
        vmin = vmax = None
    cmap = "RdBu" if invert else "RdBu_r"
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(LAMBDAS)), [f"{l:g}" for l in LAMBDAS])
    ax.set_yticks(range(len(CUTS)), [CUT_LABEL[c] for c in CUTS])
    ax.set_xlabel("lambda")
    ax.set_title(title, fontsize=10)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                        fontsize=8)
            else:
                ax.text(j, i, "-", ha="center", va="center", fontsize=8,
                        color="0.5")
    return im


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="results.json from collect_results.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    data = json.loads(Path(args.results).read_text())
    runs = data["runs"]
    step1 = next((r for r in runs if r["postfix"].endswith("_step1")), None)
    null = next((r for r in runs if r["postfix"].endswith("_null")), None)
    base = null or step1

    fig = plt.figure(figsize=(15.5, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.42)

    # --- 1. the grid itself -------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    pre = grid_of(runs, "prefit")
    im = heatmap(ax, pre, "pre-fit significance", base["prefit"] if base else None,
                 "{:.4f}")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 1])
    head = grid_of(runs, "headroom") * 100
    # invert: LOW headroom is the good direction, so blue should mean "better"
    im = heatmap(ax, head, "systematic headroom [%]  (lower is better)",
                 base["headroom"] * 100 if base and base.get("headroom") else None,
                 "{:.2f}", invert=True)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 2])
    tvb = grid_of(runs, "tv_background")
    im = heatmap(ax, tvb, "tv(background)  (lower = variations move shape less)",
                 base.get("tv_background") if base else None, "{:.4f}", invert=True)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # --- 2. did the template agreement improve? -----------------------------
    for col, key, name in ((0, "tv_background", "background"),
                           (1, "tv_signal", "signal")):
        ax = fig.add_subplot(gs[1, col])
        g = grid_of(runs, key)
        x = np.arange(len(LAMBDAS))
        for i, c in enumerate(CUTS):
            ax.plot(x, g[i], marker="o", ms=4, label=CUT_LABEL[c])
        for r, style, lab in ((step1, ":", "step1 (lambda=0)"),
                              (null, "--", "null control")):
            if r and r.get(key) is not None:
                ax.axhline(r[key], ls=style, color="k", lw=1.2, label=lab)
        ax.set_xticks(x, [f"{l:g}" for l in LAMBDAS])
        ax.set_xlabel("lambda")
        ax.set_ylabel(f"worst shape TV distance, {name}")
        ax.set_title(f"nominal vs Total/mu_roccor up-down: {name}", fontsize=10)
        ax.legend(fontsize=7, frameon=False, ncol=2)
        ax.grid(alpha=0.25)

    # --- 3. does the mechanism explain the sensitivity? ---------------------
    ax = fig.add_subplot(gs[1, 2])
    for i, c in enumerate(CUTS):
        pts = [(r.get("tv_background"), r.get("headroom"))
               for r in runs if r.get("score_cut") == c and r.get("lambda") is not None]
        pts = [(a, b * 100) for a, b in pts if a is not None and b is not None]
        if pts:
            ax.scatter(*zip(*pts), s=26, label=CUT_LABEL[c])
    for r, marker, lab in ((step1, "*", "step1"), (null, "P", "null")):
        if r and r.get("tv_background") is not None and r.get("headroom") is not None:
            ax.scatter([r["tv_background"]], [r["headroom"] * 100], marker=marker,
                       s=130, color="k", label=lab)
    ax.set_xlabel("tv(background)")
    ax.set_ylabel("systematic headroom [%]")
    ax.set_title("mechanism check: does less template shift buy sensitivity?",
                 fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(alpha=0.25)

    sub = []
    if step1:
        sub.append(f"step1 pre-fit {step1['prefit']:.4f}")
    if null:
        sub.append(f"null {null['prefit']:.4f}")
    fig.suptitle("2017 VBF, no QvG inputs, three-step loss over lambda x score_cut"
                 + (f"   ({', '.join(sub)})" if sub else ""), fontsize=12)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
