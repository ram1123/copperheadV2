#!/usr/bin/env python3
"""
Cutflow of one flag-skim muon working point (official electron veto, exactly 2 muons) on a v2 flag-skim
stage-1 ntuple (do_flag_skim, docs/flag_skim.md), scanned over the trigger-match dR, down to the cutbased
VBF category. Same cuts/weights as get_yields.py --flag_skim_selection (modules.selection: FLAG_SKIM_*,
filterRegion, applyRegionCatCuts vbf), split into steps; samples from the --sample_config groups.
Regions: z-peak, h-sidebands, h-peak, signal. The analysis is blinded: in h-peak/signal, data is not
counted from the mass-window step on; the last row is always the VBF category in h-peak, MC only.

How to run (worktree root, default pixi env):
  ./run_in_pixi.sh default python scripts/flag_skim_cutflow.py --input <label>/stage1_output -y <year> \
      [--wp looseID_IP] [--drs 0p4 0p3 0p2 0p15 0p1] [--region h-sidebands] [--out <dir>]
Example:
  ./run_in_pixi.sh default python scripts/flag_skim_cutflow.py \
      --input /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FlagSkimV2_Oct30_OfficialRecomendation_Systematics_LumiSplit/stage1_output \
      -y 2024 --wp looseID_IP --out validation/flag_skim/2024_v2/cutflow_looseIDIP
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from modules.sample_config import get_bkg_sig_dicts
from modules.selection import _round_for_cut

WP_SF = {"looseID": "LooseID", "mediumID": "MediumID", "tightID": "TightID"}
# regions that contain 115-135 GeV: data is never counted from the mass-window step on (analysis is blinded)
BLINDED_REGIONS = ("h-peak", "signal")
REGIONS = {
    "z-peak": lambda m: (m >= 70.0) & (m < 110.0),
    "h-sidebands": lambda m: ((m >= 110.0) & (m < 115.0)) | ((m >= 135.0) & (m < 150.0)),
    "h-peak": lambda m: (m >= 115.0) & (m < 135.0),
    "signal": lambda m: (m >= 110.0) & (m < 150.0),
}


def steps_for(df, wp, dr, region):
    """Ordered (name, cumulative mask) on the rows of one parquet file."""
    p = f"pair_{wp}"
    m = df["dimuon_mass"].to_numpy()
    nbl, nbm, nj = (df[c].fillna(-1).to_numpy() for c in ("nBtagLoose_nominal", "nBtagMedium_nominal", "njets_nominal"))
    btag = (nbl >= 2) | ((nbm >= 1) & (nj >= 2))
    jet1 = df["jet1_pt_nominal"].to_numpy() > 35
    mjj = df["jj_mass_nominal"].to_numpy() > 400
    deta = _round_for_cut(df["jj_dEta_nominal"].to_numpy()) > 2.5
    prim = df[f"{p}_isPrimary"].to_numpy(bool)
    muon = [
        (f"{wp} pair (>=2 muons)", prim),
        ("exactly 2 muons", df[f"nMuons_{wp}"].to_numpy() == 2),
        ("opposite sign", df[f"{p}_OS"].to_numpy(bool)),
        ("electron veto (official)", df["pass_officialEleVeto"].to_numpy(bool)),
        (f"trigger match dR<{dr.replace('p', '.')}", df[f"{p}_trigMatch_dR{dr}"].to_numpy(bool)),
    ]
    cat = [("b-jet veto", ~btag), ("jet1 pT>35", jet1), ("mjj>400", mjj), ("dEta_jj>2.5", deta)]
    out, cum = [("stage-1 skim (all events)", df["pair_row"].to_numpy() == 0)], None
    for name, mask in muon + [(f"mass {region}", REGIONS[region](m))] + cat:
        cum = mask if cum is None else cum & mask
        out.append((name, cum))
    # same chain with the h-peak window instead (MC only: data blinded)
    cum = out[len(muon)][1] & REGIONS["h-peak"](m)
    for _, mask in cat:
        cum = cum & mask
    out.append(("VBF category, h-peak (MC only)", cum))
    return out


def process_file(args):
    path, wp, drs, region, is_data = args
    names = set(pq.read_schema(path).names)
    p = f"pair_{wp}"
    cols = ["pair_row", "dimuon_mass", "nBtagLoose_nominal", "nBtagMedium_nominal", "njets_nominal",
            "jet1_pt_nominal", "jj_mass_nominal", "jj_dEta_nominal", f"{p}_isPrimary", f"nMuons_{wp}", f"{p}_OS",
            "pass_officialEleVeto", "wgt_nominal", "separate_wgt_zpt", "muIDIso_sf_MediumID",
            f"muIDIso_sf_{WP_SF[wp.split('_')[0]]}"] + [f"{p}_trigMatch_dR{d}" for d in drs]
    df = pq.read_table(path, columns=[c for c in dict.fromkeys(cols) if c in names]).to_pandas()
    if is_data:
        w = np.ones(len(df))
        w_wp = w
    else:
        w = df["wgt_nominal"].fillna(0.0).to_numpy()
        if "separate_wgt_zpt" in df:  # as get_yields.py: no Z-pT weight
            w = w / df["separate_wgt_zpt"].fillna(1.0).to_numpy()
        nom = df["muIDIso_sf_MediumID"].to_numpy()
        sf = df[f"muIDIso_sf_{WP_SF[wp.split('_')[0]]}"].to_numpy()
        w_wp = w * np.where(nom != 0, sf / np.where(nom != 0, nom, 1.0), 1.0)  # as applyFlagSkimSelection
    rows = []
    for dr in drs:
        for i, (step, mask) in enumerate(steps_for(df, wp, dr, region)):
            if is_data and ("MC only" in step or (region in BLINDED_REGIONS and i > 5)):  # blinded
                continue
            ww = w if i == 0 else w_wp
            rows.append((dr, i, step, int(mask.sum()), float(ww[mask].sum())))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="<label>/stage1_output")
    ap.add_argument("-y", "--year", required=True)
    ap.add_argument("--wp", default="looseID_IP", help="flag WP tag, e.g. looseID_IP, mediumID_noIP, tightID_IP")
    ap.add_argument("--drs", nargs="+", default=["0p4", "0p3", "0p2", "0p15", "0p1"])
    ap.add_argument("--region", default="h-sidebands", choices=sorted(REGIONS))
    ap.add_argument("--sample_config", default=os.environ.get("SAMPLE_CONFIG") or "configs/samples/samples.yaml")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = f"{args.input}/{args.year}/compacted"
    bkg, sig, _ = get_bkg_sig_dicts(yaml_path=args.sample_config, year=args.year)
    groups = {"data": sorted(os.path.basename(d) for d in glob.glob(f"{base}/data*"))}
    groups.update({g: s for g, s in sig.items()})
    groups.update({g: s for g, s in bkg.items() if g != "DYVBF"})  # get_yields skips DYVBF without the filter study

    jobs = []
    for g, samples in groups.items():
        for s in samples:
            for f in sorted(glob.glob(f"{base}/{s}/**/*.parquet", recursive=True)):
                jobs.append((g, s, (f, args.wp, args.drs, args.region, g == "data")))
    print(f"{len(jobs)} files")
    recs = []
    with ProcessPoolExecutor(args.workers) as ex:
        for (g, s, _), rows in zip(jobs, ex.map(process_file, [j[2] for j in jobs], chunksize=4)):
            recs += [(g, s) + r for r in rows]
    t = pd.DataFrame(recs, columns=["group", "sample", "dr", "step_idx", "step", "n", "w"])
    t = t.groupby(["group", "dr", "step_idx", "step"], as_index=False)[["n", "w"]].sum()

    os.makedirs(args.out, exist_ok=True)
    t.to_csv(f"{args.out}/cutflow_long.csv", index=False)
    wide = t.pivot_table(index=["dr", "step_idx", "step"], columns="group", values="w", aggfunc="sum").fillna(0.0)
    sig_g = [g for g in sig if g in wide]
    bkg_g = [g for g in bkg if g in wide and g != "DYVBF"]
    wide["signal"] = wide[sig_g].sum(axis=1)
    wide["background"] = wide[bkg_g].sum(axis=1)
    wide["data/bkg"] = wide["data"] / wide["background"].replace(0, np.nan)
    wide["S/sqrt(S+B)"] = wide["signal"] / np.sqrt((wide["signal"] + wide["background"]).replace(0, np.nan))
    wide = wide.reset_index()
    blinded = wide["step"].str.contains("MC only")
    if args.region in BLINDED_REGIONS:
        blinded |= wide["step_idx"] > 5
    wide.loc[blinded, ["data", "data/bkg"]] = np.nan
    wide.to_csv(f"{args.out}/cutflow_wide.csv", index=False)
    pd.set_option("display.width", 250)
    for dr, sub in wide.groupby("dr", sort=False):
        print(f"\n=== {args.wp}, official e-veto, 2 muons, trigger dR<{dr.replace('p', '.')} ({args.year}) ===")
        print(sub.drop(columns=["dr", "step_idx"]).to_string(index=False, float_format=lambda x: f"{x:.5g}"))
    print(f"\nWrote {args.out}/cutflow_long.csv, cutflow_wide.csv")


if __name__ == "__main__":
    main()
