#!/usr/bin/env python3
"""Nominal vs up/down template agreement for the nuisances the loss is shown.

Why
---
The loss under test penalises disagreement between the nominal prediction and the
prediction on systematically varied inputs. If it works, that has to show up
*downstream*: the Stage-3 templates the fit consumes should move less between
nominal and the Total / mu_roccor up/down variations. Significance alone cannot
say whether a change came from the intended mechanism -- the templates can.

What is measured, per (process, nuisance)
-----------------------------------------
Two quantities that must be kept apart, because the DNN can only act on one:

  dnorm     -- sum(var)/sum(nom) - 1. A pure yield shift. Migration into and out
               of the VBF category as jets cross the selection is mostly *not*
               something the score can undo.
  shape_*   -- computed AFTER renormalising the variation to the nominal yield,
               so it isolates how the events redistribute across score bins.
               That redistribution is exactly what the consistency term targets.

  shape_tv   = 0.5 * sum_i |v_i - n_i| over the normalised templates. Total
               variation distance; bounded in [0,1] and cheap to compare.
  shape_rms  = sqrt( sum_i n_i * ((v_i - n_i)/n_i)^2 ), the yield-weighted RMS
               relative shape shift. Dominated by bins that carry yield rather
               than by empty tails.
  shape_chi2 = sum_i (v_i - n_i)^2 / var_nom_i on the *unnormalised* templates.
               This is the fit-facing number: it is roughly how much the nuisance
               can pull given the MC statistics available to resist it.

`up` and `down` are reported separately and then summarised by their max, since a
one-sided variation is just as harmful as a symmetric one.

CAVEAT (stated in every report): the binning is re-derived per run, so per-bin
comparisons ACROSS runs are not like-for-like. shape_tv and shape_rms are
normalised and roughly binning-robust; shape_chi2 is not, and is reported for
within-run ranking only.

    variation_shape_consistency.py <stage3_datacard_dir> --year 2017 \
        --out shape_<postfix>.json --plot shape_<postfix>.png
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

#: The nuisances this study's training is actually shown, and the only ones the
#: datacard stripper leaves behind. `mu_roccor` carries a year suffix in the
#: template names while `Total` does not, so both spellings are searched.
NUISANCES = ("Total", "mu_roccor")


def template_from_datacard(card: Path) -> Path:
    """Resolve the ROOT file from the datacard's `shapes` line."""
    for line in card.read_text().splitlines():
        parts = line.split()
        if parts[:1] == ["shapes"] and len(parts) >= 4:
            return (card.parent / parts[3]).resolve()
    raise ValueError(f"no 'shapes' line in {card}")


def resolve_card(stage3_dir: Path, year: str) -> Path:
    card = stage3_dir / f"datacard_vbf_SR_{year}.txt"
    if card.exists():
        return card
    full = card.with_suffix(card.suffix + ".full")
    if full.exists():
        return full
    raise FileNotFoundError(f"no datacard at {card} (or .full)")


def variation_keys(keys: set, process: str, nuisance: str):
    """(up_key, down_key) for `process` under `nuisance`, or (None, None).

    Matches on prefix so `mu_roccor` finds `mu_roccor2017` without the caller
    having to know the year-suffix convention.
    """
    up = down = None
    for k in keys:
        if not k.startswith(f"{process}_{nuisance}"):
            continue
        if k.endswith("Up"):
            up = k
        elif k.endswith("Down"):
            down = k
    return up, down


def compare(nom, nom_var, var):
    """Shape/normalisation metrics for one variation against nominal."""
    n_sum, v_sum = float(nom.sum()), float(var.sum())
    if n_sum <= 0:
        return None
    dnorm = v_sum / n_sum - 1.0 if v_sum > 0 else None

    # Shape only: put the variation on the nominal's yield first.
    n_hat = nom / n_sum
    v_hat = var / v_sum if v_sum > 0 else np.zeros_like(var)
    d = v_hat - n_hat
    shape_tv = 0.5 * float(np.abs(d).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(n_hat > 0, d / n_hat, 0.0)
    shape_rms = float(np.sqrt(np.sum(n_hat * rel**2)))

    # Fit-facing: unnormalised shift against the nominal MC uncertainty.
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2_terms = np.where(nom_var > 0, (var - nom) ** 2 / nom_var, 0.0)
    return {
        "dnorm": dnorm,
        "shape_tv": shape_tv,
        "shape_rms": shape_rms,
        "shape_chi2": float(chi2_terms.sum()),
        "max_abs_rel_bin": float(np.max(np.abs(rel))) if rel.size else None,
        "per_bin_rel": [float(x) for x in rel],
    }


def stack(f, keys, names, suffix=""):
    """Summed values/variances over `names` (each optionally + `suffix`)."""
    vals, varis, used = None, None, []
    for name in names:
        key = f"{name}{suffix}"
        if key not in keys:
            continue
        h = f[key]
        v, e = h.values(), h.variances()
        vals = v if vals is None else vals + v
        varis = e if varis is None else varis + e
        used.append(key)
    return vals, varis, used


def make_plot(report, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [g for g in ("background", "signal") if g in report["groups"]]
    nuis = report["nuisances"]
    if not groups or not nuis:
        return

    ncol = len(groups) * len(nuis)
    fig, axes = plt.subplots(
        2, ncol, figsize=(4.2 * ncol, 6.0), sharex="col",
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.06, "wspace": 0.28},
        squeeze=False,
    )

    col = 0
    for group in groups:
        g = report["groups"][group]
        nom = np.asarray(g["nominal"])
        x = np.arange(len(nom))
        for nu in nuis:
            entry = g["variations"].get(nu)
            ax, rax = axes[0][col], axes[1][col]
            if entry is None:
                ax.text(0.5, 0.5, f"{nu}: absent", ha="center", transform=ax.transAxes)
                col += 1
                continue

            ax.step(x, nom, where="mid", color="k", lw=1.6, label="nominal")
            for side, color in (("up", "tab:red"), ("down", "tab:blue")):
                if entry.get(side) is None:
                    continue
                v = np.asarray(entry[side]["values"])
                ax.step(x, v, where="mid", color=color, lw=1.1, ls="--", label=side)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(nom > 0, v / nom, np.nan)
                rax.step(x, ratio, where="mid", color=color, lw=1.1)

            ax.set_yscale("log")
            ax.set_title(f"{group} - {nu}", fontsize=10)
            ax.legend(fontsize=8, frameon=False)
            rax.axhline(1.0, color="k", lw=0.8)
            rax.set_ylabel("var / nom", fontsize=8)
            rax.set_xlabel("DNN score bin", fontsize=8)
            rax.tick_params(labelsize=8)
            ax.tick_params(labelsize=8)
            col += 1

    axes[0][0].set_ylabel("events", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage3_dir")
    ap.add_argument("--year", default="2017")
    ap.add_argument("--postfix", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--signal", default=",".join(SIGNAL))
    ap.add_argument("--background", default=",".join(BACKGROUND))
    ap.add_argument("--nuisances", default=",".join(NUISANCES))
    args = ap.parse_args(argv)

    try:
        card = resolve_card(Path(args.stage3_dir), args.year)
        path = template_from_datacard(card)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  SHAPE: {exc}", file=sys.stderr)
        return 2
    if not path.exists():
        print(f"  SHAPE: template missing at {path}", file=sys.stderr)
        return 2

    signal = [x for x in args.signal.split(",") if x]
    background = [x for x in args.background.split(",") if x]
    nuisances = [x for x in args.nuisances.split(",") if x]

    report = {
        "postfix": args.postfix,
        "year": args.year,
        "template": str(path),
        "nuisances": nuisances,
        "groups": {},
        "caveat": (
            "The binning is re-derived per run. shape_tv/shape_rms are computed on "
            "yield-normalised templates and are roughly binning-robust; shape_chi2 "
            "and per-bin numbers are for within-run ranking only."
        ),
    }

    with uproot.open(path) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        for group, members in (("background", background), ("signal", signal)):
            nom, nom_var, used = stack(f, keys, members)
            if nom is None:
                print(f"  SHAPE: no {group} templates in {path}", file=sys.stderr)
                continue
            entry = {
                "processes": used,
                "n_bins": int(len(nom)),
                "total": float(nom.sum()),
                "nominal": [float(v) for v in nom],
                "variations": {},
            }
            for nu in nuisances:
                sides = {}
                for side, tail in (("up", "Up"), ("down", "Down")):
                    vals, _, vused = None, None, []
                    for name in members:
                        up_k, down_k = variation_keys(keys, name, nu)
                        k = up_k if side == "up" else down_k
                        if k is None:
                            # A process without this nuisance keeps its nominal
                            # shape in the fit, so that is what must be summed in.
                            k = name if name in keys else None
                            if k is None:
                                continue
                        v = f[k].values()
                        vals = v if vals is None else vals + v
                        vused.append(k)
                    if vals is None:
                        continue
                    m = compare(nom, nom_var, vals)
                    if m is None:
                        continue
                    m["values"] = [float(v) for v in vals]
                    m["keys"] = vused
                    sides[side] = m
                if not sides:
                    continue
                sides["max_shape_tv"] = max(
                    s["shape_tv"] for s in sides.values() if isinstance(s, dict))
                sides["max_shape_rms"] = max(
                    s["shape_rms"] for s in sides.values() if isinstance(s, dict))
                sides["max_abs_dnorm"] = max(
                    abs(s["dnorm"]) for s in sides.values()
                    if isinstance(s, dict) and s.get("dnorm") is not None)
                entry["variations"][nu] = sides
            report["groups"][group] = entry

    tag = args.postfix or path.name
    print(f"  SHAPE {tag}: nominal vs variation templates")
    for group, g in report["groups"].items():
        for nu, sides in g["variations"].items():
            bits = []
            for side in ("up", "down"):
                s = sides.get(side)
                if not s:
                    continue
                bits.append(f"{side}: dnorm={s['dnorm']:+.4f} tv={s['shape_tv']:.4f} "
                            f"rms={s['shape_rms']:.4f} chi2={s['shape_chi2']:.2f}")
            print(f"    {group:<11} {nu:<10} " + " | ".join(bits))

    if args.plot:
        try:
            make_plot(report, Path(args.plot), f"{tag} - {args.year}")
            print(f"  SHAPE: plot -> {args.plot}")
        except Exception as exc:                       # plotting must never abort a chain
            print(f"  SHAPE: plot failed ({exc})", file=sys.stderr)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"  SHAPE: report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
