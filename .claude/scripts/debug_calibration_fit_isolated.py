#!/usr/bin/env python3
"""
Isolated single-category debug harness for the EBE mass resolution calibration fit
(src/lib/ebeMassResCalibration/basic_class_for_calibration.py::generateBWxDCB_RooCMSShape_plot).

Runs exactly one category's Z-peak fit + plot, outside the full getCalibrationFactor.py
pipeline (no argparse, no Dask, no per-category subprocess orchestration) -- useful for
quickly reproducing/bisecting fit or plotting issues without paying the cost of a full
run_analysis_pipeline.sh invocation. This is how the 2026-09 investigations tracked down:
  - the per-category RooFit object-name collision (heap corruption after the first fit)
  - the PDF-vs-PNG canvas.SaveAs() crash (see .claude/reports/investigations/
    2026-09-14_ebe-calib-fit-audit-and-run.md for the full writeup)

Usage (run inside the `default` pixi environment):
    cd /cvmfs/cms-af.opensciencegrid.org/paf/pixi/copperheadV2
    CONDA_OVERRIDE_CUDA=12.4 pixi run -e default python3 -u \
        /path/to/repo/.claude/scripts/debug_calibration_fit_isolated.py \
        --input-path /work/projects/hmm/<user>/hmm_ntuples/copperheadV1clean/<label> \
        --year 2025 --sample MC --category 30-45_BB

If a cached skim (skim_zpeak/<Sample>_<year>/zpeak_skim.parquet) already exists under
validation/ebeMassResCalibration/<label from --input-path>/, it's reused; otherwise this
builds it fresh from the compacted stage1 output (slower, first run only). If a cached
mass_<category>.npy exists for a category, that is used instead of extracting fresh from
the skim -- pass --force-fresh-mass to bypass that cache (e.g. to check whether an issue
is specific to the np.load() code path vs. a freshly computed array).
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CALIB_DIR = REPO_ROOT / "src" / "lib" / "ebeMassResCalibration"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CALIB_DIR))
# All output/cache paths below are relative to the repo root (matching
# getCalibrationFactor.py's own convention), not wherever this script happens to be
# invoked from (e.g. the CVMFS pixi project dir, which is also read-only).
os.chdir(REPO_ROOT)

import numpy as np
import pandas as pd
import dask.dataframe as dd

from basic_class_for_calibration import get_calib_categories, generateBWxDCB_RooCMSShape_plot, CONFIG


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-path", required=True, help="label's root save dir, e.g. .../hmm_ntuples/copperheadV1clean/<label>")
    ap.add_argument("--year", required=True)
    ap.add_argument("--sample", choices=["MC", "Data"], required=True)
    ap.add_argument("--category", required=True, help="e.g. 30-45_BB")
    ap.add_argument("--extra-string", default="V1")
    ap.add_argument("--out-dir", default=None, help="default: same output_dir the real pipeline would use")
    ap.add_argument("--force-fresh-mass", action="store_true", help="bypass mass_<cat>.npy cache even if present")
    args = ap.parse_args()

    label = Path(args.input_path).name
    dir_tag = label
    skim_dir = f"validation/ebeMassResCalibration/{dir_tag}/skim_zpeak/{args.sample}_{args.year}"
    out_dir = args.out_dir or f"validation/ebeMassResCalibration/{dir_tag}/binned/{args.year}/{args.sample}_{args.year}_{args.extra_string}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(skim_dir, exist_ok=True)

    npy_path = f"{skim_dir}/mass_{args.category}.npy"
    if os.path.exists(npy_path) and not args.force_fresh_mass:
        print(f"Loading cached mass array: {npy_path}", flush=True)
        mass = np.load(npy_path)
    else:
        skim_path = f"{skim_dir}/zpeak_skim.parquet"
        if os.path.exists(skim_path):
            print(f"Reading cached skim: {skim_path}", flush=True)
            ddf = dd.read_parquet(skim_path)
        else:
            load_path = str(Path(args.input_path) / "stage1_output" / args.year / "compacted")
            if args.sample == "MC":
                dy_candidates = ["dyTo2L_M-50_incl", "dyTo2Mu_M-50_aMCatNLO"]
                dy_sample = next((c for c in dy_candidates if os.path.isdir(f"{load_path}/{c}")), dy_candidates[0])
                input_glob = f"{load_path}/{dy_sample}/*/*.parquet"
            else:
                input_glob = f"{load_path}/data_*/*/*.parquet"
            print(f"Building skim from {input_glob} (no cache found)...", flush=True)
            ddf = dd.read_parquet(input_glob)[CONFIG["fields_with_errors"]]
            ddf = ddf[(ddf["dimuon_mass"] > CONFIG["zcr_filter_range"][0]) & (ddf["dimuon_mass"] < CONFIG["zcr_filter_range"][1])]
            ddf.to_parquet(skim_path, write_index=False, overwrite=True)
            ddf = dd.read_parquet(skim_path)

        cats = get_calib_categories(ddf)
        if args.category not in cats:
            sys.exit(f"Unknown category '{args.category}'. Available: {sorted(cats)}")
        print(f"Extracting mass array for {args.category} (fresh compute)...", flush=True)
        mass = ddf.loc[cats[args.category], "dimuon_mass"].compute().to_numpy()

    print(f"n events: {len(mass)}", flush=True)
    if mass.size == 0:
        sys.exit(f"Category {args.category} has 0 events -- nothing to fit.")

    df_fit = pd.DataFrame(columns=["cat_name", "fit_val", "fit_err"])
    print("Calling generateBWxDCB_RooCMSShape_plot...", flush=True)
    df_fit = generateBWxDCB_RooCMSShape_plot(
        mass, args.category, nbins=CONFIG["nbins"], df_fit=df_fit, output_dir=out_dir,
        logfile="CalibrationLog.txt", ifbinned=True, inputFilePath=args.input_path,
        fit_cfg_path=str(CALIB_DIR / "fit_config.yml"),
    )
    print("DONE:", flush=True)
    print(df_fit, flush=True)
    print(f"Plot: {out_dir}/calibration_fitCat{args.category}.png", flush=True)


if __name__ == "__main__":
    main()
