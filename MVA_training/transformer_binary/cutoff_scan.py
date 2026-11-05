#!/usr/bin/env python3
"""
Transformer cutoff scan (mva-categorizer-10 "constrained best"), Run-2 years summed; settings and sample groups in
configs/MVA/transformer_categorizer.yaml (cutoff_scan). Within the b-tag veto: VBF = score < cutoff, ggH otherwise.
Per cutoff, relative to the cut-based categories (applyRegionCatCuts):
  * significance (h-peak MC): Z = S/sqrt(B) per category; Z_combine = hypot(R_ggH*ref_ggH, R_VBF*ref_VBF)
  * flatness (h-sidebands, VBF): data/MC-ratio chi2/ndof per check variable <= reference + k_sigma*sqrt(2/ndof)
  * yield floor (h-sidebands): VBF-category check-MC yield >= the cut-based one
Selected: the highest Z_combine passing all three. --weight mva10 removes the Z-pT weight; analysis = wgt_nominal.

Usage (repo root, PYTHONPATH=$PWD):
  python -m MVA_training.transformer_binary.cutoff_scan --scored <stage1_output of the scored copy> --out <dir> \
      [--compare <cutoff>]
"""

import argparse
import glob
import json
import os

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from MVA_training.transformer_binary import config
from modules import selection

# configs/MVA/transformer_categorizer.yaml: cutoff_scan (samples: samples.yaml groups, per year)
CFG = config.section("cutoff_scan")
CHECK_VARIABLES = {name: tuple(binning) for name, binning in CFG["check_variables"].items()}
REPORT_ONLY = {name: tuple(binning) for name, binning in CFG["report_only"].items()}
COMBINE_REFERENCE = CFG["combine_reference"]
K_SIGMA = float(CFG["k_sigma"])
CUTOFFS = np.linspace(0.0, 1.0, int(CFG["n_grid"]) + 1)  # VBF = score < CUTOFFS[i] = score bins 0..i-1
SCORE = selection.transformer_score_field()
SELECTION_COLUMNS = ("nBtagLoose", "nBtagMedium", "jj_mass", "jj_dEta", "jet1_pt", "njets")


def sample_roles(year):
    """{process: (signal class or None, in the significance background, in the data/MC checks)} for `year`."""
    roles = {}
    for cls, procs in config.signal_samples(year).items():
        roles.update({p: (cls, False, False) for p in procs})
    check = {p for procs in config.samples("background", CFG["check_groups"], year).values() for p in procs}
    for procs in config.samples("background", CFG["background_groups"], year).values():
        roles.update({p: (None, True, p in check) for p in procs})
    return roles


def load(scored, year, sample, role, weight_mode="mva10"):
    """Columns needed here, with the reference categories and regions as boolean columns."""
    files = sorted(glob.glob(os.path.join(scored, year, "compacted", sample, "**", "*.parquet"), recursive=True))
    if not files:
        return None
    names = pq.read_schema(files[0]).names
    wanted = [SCORE, "dimuon_mass", "wgt_nominal", "separate_wgt_zpt", *CHECK_VARIABLES]
    for base in SELECTION_COLUMNS:
        wanted.append(next(c for c in (f"{base}_nominal", base) if c in names))
    table = pq.read_table(files, columns=[c for c in wanted if c in names])
    events = ak.from_arrow(table)
    events = ak.with_field(events, np.arange(len(events)), "row_id")
    frame = pd.DataFrame({c: ak.to_numpy(ak.fill_none(events[c], np.nan)) for c in [SCORE, "dimuon_mass", *CHECK_VARIABLES]})
    if sample.startswith("data_"):
        frame["weight"] = 1.0
    else:
        weight = ak.fill_none(events["wgt_nominal"], 0.0)
        if weight_mode == "mva10" and "separate_wgt_zpt" in events.fields:
            weight = weight / ak.fill_none(events["separate_wgt_zpt"], 1.0)
        frame["weight"] = ak.to_numpy(weight).astype(np.float64)
    for region in ("h-peak", "h-sidebands"):
        for category in ("vbf", "ggh", "bJetVeto"):
            kept = selection.applyRegionCatCuts(events, category=category, region_name=region, process=sample,
                                                variation="nominal", year=year)
            mask = np.zeros(len(events), dtype=bool)
            mask[ak.to_numpy(kept["row_id"])] = True
            frame[f"{region}|{category}"] = mask
    frame["year"], frame["sample"] = year, sample
    frame["signal_class"], frame["is_bkg"], frame["is_check_mc"] = role
    frame["is_data"] = sample.startswith("data_")
    return frame


def cumulative(score, weight, mask):
    """[sum of weight over mask & score < CUTOFFS[i]] for every i, and the same for weight**2."""
    hist, _ = np.histogram(score[mask], bins=CUTOFFS, weights=weight[mask])
    hist2, _ = np.histogram(score[mask], bins=CUTOFFS, weights=weight[mask] ** 2)
    return np.concatenate([[0.0], np.cumsum(hist)]), np.concatenate([[0.0], np.cumsum(hist2)])


def cumulative_2d(score, values, weight, mask, binning):
    """(n_cutoffs, n_var_bins) cumulative histograms of `values` for score < cutoff."""
    lo, hi, n = binning
    edges = np.linspace(lo, hi, n + 1)
    h, _, _ = np.histogram2d(score[mask], values[mask], bins=[CUTOFFS, edges], weights=weight[mask])
    h2, _, _ = np.histogram2d(score[mask], values[mask], bins=[CUTOFFS, edges], weights=weight[mask] ** 2)
    zero = np.zeros((1, n))
    return np.vstack([zero, np.cumsum(h, axis=0)]), np.vstack([zero, np.cumsum(h2, axis=0)])


def flatness(data_counts, mc_counts, mc_variance):
    """mva-03's chi2/ndof of the data/MC ratio around its weighted-mean flat line."""
    valid = (mc_counts > 0) & (data_counts > 0)
    ndof = int(valid.sum()) - 1
    if ndof < 1:
        return float("nan"), ndof
    ratio = data_counts[valid] / mc_counts[valid]
    ratio_err = ratio * np.sqrt(1.0 / data_counts[valid] + mc_variance[valid] / mc_counts[valid] ** 2)
    w = 1.0 / ratio_err**2
    mean = np.sum(ratio * w) / np.sum(w)
    return float(np.sum((ratio - mean) ** 2 * w) / ndof), ndof


def histogram(values, weight, binning):
    lo, hi, n = binning
    edges = np.linspace(lo, hi, n + 1)
    return np.histogram(values, bins=edges, weights=weight)[0], np.histogram(values, bins=edges, weights=weight**2)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scored", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--years", default=",".join(map(str, CFG["years"])), type=lambda s: s.split(","))
    parser.add_argument("--compare", type=float, default=None, help="also report this cutoff (e.g. Xudong's)")
    parser.add_argument("--weight", choices=("mva10", "analysis"), default="mva10")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    frames = []
    for year in args.years:
        roles = sample_roles(year)
        samples = sorted(os.listdir(os.path.join(args.scored, year, "compacted")))
        for sample in samples:
            if sample in roles or sample.startswith("data_"):
                frame = load(args.scored, year, sample, roles.get(sample, (None, False, False)), args.weight)
                if frame is not None:
                    frames.append(frame)
                    print(f"{year}/{sample}: {len(frame)} rows", flush=True)
    ev = pd.concat(frames, ignore_index=True)
    score, weight = ev[SCORE].to_numpy(), ev["weight"].to_numpy()
    is_ggh_sig, is_vbf_sig = (ev["signal_class"] == "ggH").to_numpy(), (ev["signal_class"] == "VBF").to_numpy()
    is_bkg, is_check_mc, is_data = (ev[c].to_numpy(dtype=bool) for c in ("is_bkg", "is_check_mc", "is_data"))
    peak, side = ev["h-peak|bJetVeto"].to_numpy(), ev["h-sidebands|bJetVeto"].to_numpy()

    # ---- significance (h-peak MC) ----
    scan = pd.DataFrame({"cutoff": CUTOFFS})
    totals = {}
    for name, mask in (("S_ggH", is_ggh_sig), ("S_VBF", is_vbf_sig), ("B", is_bkg)):
        in_vbf, _ = cumulative(score, weight, peak & mask)
        total = float(weight[peak & mask].sum())
        totals[name] = total
        scan[f"{name}_vbfcat"], scan[f"{name}_gghcat"] = in_vbf, total - in_vbf
    _, b2 = cumulative(score, weight, peak & is_bkg)
    scan["n_eff_bkg_vbfcat"] = np.where(b2 > 0, scan["B_vbfcat"] ** 2 / np.where(b2 > 0, b2, 1), 0.0)

    def ref_yield(mask, region, category):
        return float(weight[mask & ev[f"{region}|{category}"].to_numpy()].sum())

    ref = {
        "S_ggH_gghcat": ref_yield(is_ggh_sig, "h-peak", "ggh"), "B_gghcat": ref_yield(is_bkg, "h-peak", "ggh"),
        "S_VBF_vbfcat": ref_yield(is_vbf_sig, "h-peak", "vbf"), "B_vbfcat": ref_yield(is_bkg, "h-peak", "vbf"),
        "S_ggH_vbfcat": ref_yield(is_ggh_sig, "h-peak", "vbf"),
    }
    ref["Z_ggH"] = ref["S_ggH_gghcat"] / np.sqrt(ref["B_gghcat"])
    ref["Z_VBF"] = ref["S_VBF_vbfcat"] / np.sqrt(ref["B_vbfcat"])
    with np.errstate(divide="ignore", invalid="ignore"):
        scan["Z_ggH"] = np.where(scan["B_gghcat"] > 0, scan["S_ggH_gghcat"] / np.sqrt(scan["B_gghcat"]), np.nan)
        scan["Z_VBF"] = np.where(scan["B_vbfcat"] > 0, scan["S_VBF_vbfcat"] / np.sqrt(scan["B_vbfcat"]), np.nan)
    scan["valid"] = (scan["B_gghcat"] > 0) & (scan["B_vbfcat"] > 0) & np.isfinite(scan["Z_ggH"]) & np.isfinite(scan["Z_VBF"])
    scan["R_ggH"], scan["R_VBF"] = scan["Z_ggH"] / ref["Z_ggH"], scan["Z_VBF"] / ref["Z_VBF"]
    scan["Z_combine"] = np.sqrt((scan["R_ggH"] * COMBINE_REFERENCE["ggH"]) ** 2 + (scan["R_VBF"] * COMBINE_REFERENCE["VBF"]) ** 2)
    scan["vbf_signal_efficiency"] = scan["S_VBF_vbfcat"] / totals["S_VBF"]
    scan["ggh_signal_into_vbf"] = scan["S_ggH_vbfcat"] / totals["S_ggH"]
    ref_z_combine = float(np.hypot(COMBINE_REFERENCE["ggH"], COMBINE_REFERENCE["VBF"]))

    # ---- data/MC flatness and yield floor (h-sidebands) ----
    ref_side_vbf = side & ev["h-sidebands|vbf"].to_numpy()
    ref_flat = {}
    for var, binning in {**CHECK_VARIABLES, **REPORT_ONLY}.items():
        values = ev[var].to_numpy()
        d, _ = histogram(values[ref_side_vbf & is_data], np.ones((ref_side_vbf & is_data).sum()), binning)
        m, m2 = histogram(values[ref_side_vbf & is_check_mc], weight[ref_side_vbf & is_check_mc], binning)
        ref_flat[var] = flatness(d, m, m2)
        data_cum, _ = cumulative_2d(score, values, np.ones(len(ev)), side & is_data, binning)
        mc_cum, mc_var = cumulative_2d(score, values, weight, side & is_check_mc, binning)
        chi = [flatness(data_cum[i], mc_cum[i], mc_var[i]) for i in range(len(CUTOFFS))]
        scan[f"chi2_per_ndof_{var}"] = [c for c, _ in chi]
        scan[f"ndof_{var}"] = [n for _, n in chi]
        if var in CHECK_VARIABLES:
            tolerance = K_SIGMA * np.sqrt(2.0 / np.maximum(scan[f"ndof_{var}"], 1))
            scan[f"flatness_ok_{var}"] = (scan[f"ndof_{var}"] > 0) & np.isfinite(ref_flat[var][0]) & (
                scan[f"chi2_per_ndof_{var}"] <= ref_flat[var][0] + tolerance)
    scan["flatness_ok_all_vars"] = scan[[f"flatness_ok_{v}" for v in CHECK_VARIABLES]].all(axis=1)
    side_mc, _ = cumulative(score, weight, side & is_check_mc)
    side_data, _ = cumulative(score, np.ones(len(ev)), side & is_data)
    scan["bkg_mc_yield_sidebands_vbfcat"], scan["data_sidebands_vbfcat"] = side_mc, side_data
    ref_floor = float(weight[ref_side_vbf & is_check_mc].sum())
    scan["bkg_yield_ok"] = scan["bkg_mc_yield_sidebands_vbfcat"] >= ref_floor

    # ---- selection ----
    scan["eligible"] = scan["valid"] & scan["flatness_ok_all_vars"] & scan["bkg_yield_ok"]
    scan.to_csv(os.path.join(args.out, "cutoff_scan.csv"), index=False)
    columns = ["cutoff", "Z_combine", "R_ggH", "R_VBF", "vbf_signal_efficiency", "ggh_signal_into_vbf", "S_VBF_vbfcat",
               "S_ggH_vbfcat", "B_vbfcat", "bkg_mc_yield_sidebands_vbfcat", "data_sidebands_vbfcat",
               *[f"chi2_per_ndof_{v}" for v in {**CHECK_VARIABLES, **REPORT_ONLY}], "flatness_ok_all_vars", "bkg_yield_ok"]
    unconstrained = scan[scan["valid"]].sort_values("Z_combine", ascending=False).iloc[0]
    eligible = scan[scan["eligible"]]
    summary = {
        "reference": {**ref, "Z_combine": ref_z_combine, "sidebands_vbf_mc_yield_floor": ref_floor,
                      "flatness": {v: {"chi2_per_ndof": c, "ndof": n} for v, (c, n) in ref_flat.items()}},
        "weight": args.weight, "n_cutoffs": len(scan), "n_valid": int(scan["valid"].sum()),
        "n_passing_flatness": int((scan["valid"] & scan["flatness_ok_all_vars"]).sum()),
        "n_passing_yield_floor": int(scan["bkg_yield_ok"].sum()), "n_eligible": int(len(eligible)),
        "eligible_band": [float(eligible["cutoff"].min()), float(eligible["cutoff"].max())] if len(eligible) else None,
        "unconstrained_best": unconstrained[columns].to_dict(),
        "constrained_best": eligible.sort_values(["Z_combine", "cutoff"], ascending=[False, False]).iloc[0][columns].to_dict()
        if len(eligible) else None,
    }
    if args.compare is not None:
        i = int(np.searchsorted(CUTOFFS, args.compare))  # nearest grid point at or above
        summary["compared_cutoff"] = {"requested": args.compare, **scan.iloc[i][columns + ["eligible"]].to_dict()}
    with open(os.path.join(args.out, "cutoff_selection.json"), "w") as stream:
        json.dump(summary, stream, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
