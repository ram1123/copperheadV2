#!/usr/bin/env python3

import os
import sys
import time
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

import argparse

import dask
dask.config.set(scheduler="threads")  # or "single-threaded", "processes"

from dask import delayed
import dask.dataframe as dd
import awkward as ak

from datetime import datetime

import correctionlib

import logging
from modules.utils import logger
from modules.trials import get_stage1_path
from modules.dask_utils import get_dask_client
from modules.dask_utils import close_dask_client
from cli.common_argparser import build_common_parser


from basic_class_for_calibration import (
    get_calib_categories,
    generateBWxDCB_RooCMSShape_plot,
    closure_test_from_df,
    plot_closure_comparison_calibrated_uncalibrated,
    save_calibration_json,
    filter_region,
    closure_test_resolution_binning,
    CONFIG,
    plot_histogram,
)

from basic_class_for_calibration import timed

CURRENT_DIR = Path(__file__).resolve().parent

def _setup_path():
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))

def backup_file(filepath):
    if os.path.exists(filepath):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{filepath}.{timestamp}.bak"
        os.system(f"cp {filepath} {backup_path}")
        logger.info(f"Backed up {filepath} to {backup_path}")

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def save_dataframe_to_csv(df, path, description="DataFrame"):
    df.to_csv(path, index=False)
    logger.info(f"Saved {description} to {path}")


def step1_mass_fitting_zcr(ddf, output_dir="", skim_dir="", fix_fitting_one_cat=None, ifbinned=False, inputFilePath=""):
    logger.info("=== Step 1: Mass fitting in ZCR ===")
    tstart = time.time()

    data_categories = get_calib_categories(ddf)

    logger.debug("Data categories for fitting:")
    for cat_name in data_categories.keys():
        logger.debug(f" - {cat_name}")

    df_fit = pd.DataFrame(columns=["cat_name", "fit_val", "fit_err"])
    for cat_name, mask in data_categories.items():
        if fix_fitting_one_cat and cat_name != fix_fitting_one_cat:
            # logger.debug(f"Skipping category {cat_name}, as no re-fitting required.")
            continue

        if os.path.exists(f"{skim_dir}/mass_{cat_name}.npy"):
            logger.info(f"Loading cached mass for category {cat_name} from {skim_dir}/mass_{cat_name}.npy")
            mass = np.load(f"{skim_dir}/mass_{cat_name}.npy")
        else:
            logger.info(f"Extracting mass for category {cat_name}...")
            with timed(f"extract mass for category {cat_name}"):
                mass = ddf.loc[mask, "dimuon_mass"].compute().to_numpy()
            # save to numpy array
            with timed(f"save mass numpy for category {cat_name}"):
                if fix_fitting_one_cat is not None:
                    logger.debug("Save to numpy when refitting only one category")
                    np.save(f"{skim_dir}/mass_{cat_name}.npy", mass)

        if mass.size == 0:
            logger.debug(f"Category {cat_name} has no events, skipping.")
            continue
        df_fit = generateBWxDCB_RooCMSShape_plot(
            mass,
            cat_name,
            nbins=CONFIG["nbins"],
            df_fit=df_fit,
            output_dir=output_dir,
            logfile="CalibrationLog.txt",
            ifbinned=ifbinned,
            inputFilePath=inputFilePath,
            fit_cfg_path=f"{CURRENT_DIR}/fit_config.yml",
        )

    logger.info("Step 1 completed in {:.2f} s".format(time.time() - tstart))
    return df_fit


def _build_fixcat_subprocess_argv(args, year, cat_name):
    """Re-invoke this same script for exactly one category, in its own OS process."""
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--NanoAODv", str(args.NanoAODv),
        "--years", str(year),
        "--input_path", args.input_path,
        "--extraString", args.extraString,
        "--steps", "step1",
        "--fixCat", cat_name,
        "--log-level", logging.getLevelName(args.log_level),
        "--no-dask-client",
    ]
    if args.ifbinned:
        argv.append("--ifbinned")
    if args.isMC:
        argv.append("--isMC")
    return argv


def run_step1_categories_isolated(args, year, ddf, output_dir, skim_dir):
    """
    Run each calibration category's Z-peak fit in its own OS subprocess (reusing the
    already-tested --fixCat code path), instead of looping over all categories in one
    long-lived process.

    Why: the production fit (generateBWxDCB_RooCMSShape_plot, using the custom
    RooCMSShape ROOT plugin) has shown intermittent glibc heap corruption
    ("free(): invalid next size") that manifests only *after* a category's fit,
    plots and JSON are fully written to disk -- not deterministically tied to any
    one line of Python/RooFit code (an isolated single-category repro of the exact
    same code succeeded cleanly, while the same category crashed when run through
    this script's normal per-year loop). Recompiling the bundled RooCMSShape_cc.so
    against this pixi environment's ROOT (6.32.02) surfaced a cling ABI-compatibility
    warning (__GLIBCXX__ 20240521 at compile time vs 20250605 at runtime), consistent
    with an intermittent ABI-mismatch-driven memory corruption rather than a pure
    logic bug. Isolating each category in its own process means a crash at the very
    end of category N's work cannot corrupt or kill category N+1 -- each subprocess
    starts with a clean heap.
    """
    logger.info("=== Step 1 (per-category subprocess isolation) ===")
    tstart = time.time()

    categories = list(get_calib_categories(ddf).keys())
    logger.info(f"{len(categories)} categories to fit for year {year}: {categories}")

    fit_results_path = f"{output_dir}/fit_results.csv"
    fit_params_path = f"{output_dir}/fit_params.json"
    if not os.path.exists(fit_results_path):
        pd.DataFrame(columns=["cat_name", "fit_val", "fit_err"]).to_csv(fit_results_path, index=False)

    max_attempts = 2
    subprocess_timeout_s = 150

    def _has_row(cat_name):
        df_check = pd.read_csv(fit_results_path) if os.path.exists(fit_results_path) else pd.DataFrame(columns=["cat_name"])
        return cat_name in set(df_check.get("cat_name", []))

    def _append_row(cat_name, fit_val, fit_err):
        df_check = pd.read_csv(fit_results_path) if os.path.exists(fit_results_path) else pd.DataFrame(columns=["cat_name", "fit_val", "fit_err"])
        df_check = df_check[df_check["cat_name"] != cat_name]
        df_check = pd.concat([df_check, pd.DataFrame([{"cat_name": cat_name, "fit_val": fit_val, "fit_err": fit_err}])], ignore_index=True)
        df_check.to_csv(fit_results_path, index=False)

    def _try_recover_from_json(cat_name):
        """
        save_fit_params_to_json runs BEFORE the plotting/canvas-cleanup code that is
        where the crash/hang has been observed to occur, so fit_params.json can hold a
        valid result for a category whose subprocess never returned cleanly to append
        its own fit_results.csv row. Recover it directly rather than re-fitting.
        """
        if not os.path.exists(fit_params_path):
            return False
        with open(fit_params_path) as f:
            all_fits = json.load(f)
        entry = all_fits.get(cat_name)
        if not entry or entry.get("sigma") is None:
            return False
        status = entry.get("status")
        if status != 0:
            logger.warning(
                f"[{cat_name}] recovering fit from {fit_params_path} despite non-zero "
                f"fit status={status} (MIGRAD/HESSE did not fully converge) -- treat "
                "this category's calibration factor as lower-confidence; re-check with "
                "--fixCat after tuning fit_config.yml bounds for this category."
            )
        _append_row(cat_name, entry["sigma"], entry["sigma_err"])
        logger.info(f"[{cat_name}] recovered fit_val={entry['sigma']:.4g} +/- {entry['sigma_err']:.4g} from fit_params.json")
        return True

    failures = []
    for cat_name in categories:
        if _has_row(cat_name):
            logger.info(f"[{cat_name}] already has a fit_results.csv row, skipping.")
            continue
        if _try_recover_from_json(cat_name):
            logger.info(f"[{cat_name}] recovered from a prior run's fit_params.json, skipping refit.")
            continue

        succeeded = False
        for attempt in range(1, max_attempts + 1):
            argv = _build_fixcat_subprocess_argv(args, year, cat_name)
            logger.info(f"[{cat_name}] launching isolated subprocess (attempt {attempt}/{max_attempts})")
            logger.debug(f"[{cat_name}] argv: {argv}")
            try:
                result = subprocess.run(argv, timeout=subprocess_timeout_s)
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                # subprocess.run() already killed and reaped the child on timeout.
                # Some crashes in this fit (glibc heap corruption in the custom
                # RooCMSShape ROOT plugin) print "*** Break *** abort" but then hang
                # instead of actually terminating -- a plain non-zero exit code alone
                # is not enough to detect that case, hence the timeout.
                logger.warning(
                    f"[{cat_name}] subprocess did not exit within {subprocess_timeout_s}s "
                    "and was killed (likely hung after a post-fit crash, not still fitting)"
                )
                returncode = None

            if returncode not in (0, None):
                logger.warning(
                    f"[{cat_name}] subprocess exited with code {returncode} "
                    "(often a post-fit ROOT/PyROOT cleanup crash after output was already "
                    "written -- checking fit_results.csv / fit_params.json before treating "
                    "this as a real failure)"
                )

            if _has_row(cat_name) or _try_recover_from_json(cat_name):
                succeeded = True
                break
            logger.warning(f"[{cat_name}] attempt {attempt}/{max_attempts} produced no recoverable output.")

        if not succeeded:
            logger.error(
                f"[{cat_name}] no row and nothing recoverable from {fit_params_path} after "
                f"{max_attempts} isolated attempts -- this category's fit genuinely failed."
            )
            failures.append(cat_name)

    if failures:
        logger.error(f"{len(failures)}/{len(categories)} categories produced no output: {failures}")
    else:
        logger.info(f"All {len(categories)} categories produced output.")

    logger.info("Step 1 (isolated) completed in {:.2f} s".format(time.time() - tstart))
    return pd.read_csv(fit_results_path)


def median_bootstrap_err(x, n_boot=300, seed=12345):
    """Bootstrap uncertainty on the median."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    meds = np.median(x[idx], axis=1)
    return np.std(meds, ddof=1)


def step2_mass_resolution(df, output_dir="tmp", pdfFile_ExtraText="", n_boot=300):
    """
    Compute ONLY predicted (NonCal) event-by-event mass resolution and summarize per calib category:
      - median
      - bootstrap error on median
      - N events
    """
    logger.info("=== Step 2: Predicted resolution (NonCal only) ===")
    tstart = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # Compute predicted resolution in dask
    df2 = df.assign(
        dpt1 = (df["mu1_ptErr"] / df["mu1_pt"]) * (df["dimuon_mass"] / 2),
        dpt2 = (df["mu2_ptErr"] / df["mu2_pt"]) * (df["dimuon_mass"] / 2),
        dimuon_ebe_mass_res_NonCal = lambda x: np.sqrt(x["dpt1"]**2 + x["dpt2"]**2),
    )

    logger.info("Computing dask dataframe...")
    with timed("compute dask df for step2"):
        df = df2.compute()

    logger.info("Dask dataframe computed.")
    # Category masks (your function works with pandas too)
    calib_cats = get_calib_categories(df)

    rows = []
    for cat_name, mask in calib_cats.items():
        cat_df = df[mask]
        if cat_df.empty:
            logger.warning(f"Category {cat_name} empty, skipping.")
            continue

        vals = cat_df["dimuon_ebe_mass_res_NonCal"].to_numpy()
        med = float(np.median(vals))
        med_err = float(
            median_bootstrap_err(vals, n_boot=n_boot, seed=hash(cat_name) % (2**32))
        )

        rows.append(
            {
                "cat_name": cat_name,
                "n_events": int(len(vals)),
                "median_val_NonCal": med,
                "median_err_NonCal": med_err,
            }
        )

        # optional diagnostic hist
        plot_histogram(
            vals,
            CONFIG["nbins"],
            (0.5, 3.0),
            "Dimuon mass resolution (GeV)",
            "Events",
            f"{cat_name}: median={med:.4f} +/- {med_err:.4f}",
            f"{output_dir}/mass_resolution_{cat_name}_NonCal_{pdfFile_ExtraText}.pdf",
            median=med,
        )

    out = pd.DataFrame(rows)
    logger.info("Step 2 completed in {:.2f} s".format(time.time() - tstart))
    return out


def step3_compute_calibration(df_fit, df_res):
    df_merged = pd.merge(df_fit, df_res, on="cat_name", how="inner")
    df_merged["calibration_factor"] = df_merged["fit_val"] / df_merged["median_val_NonCal"]
    return df_merged


def main():

    parser = build_common_parser()
    parser.add_argument("--isMC", action="store_true", help="Run on MC samples (default: False)")
    # binned or unbinned fitting
    parser.add_argument("--ifbinned", action="store_true", help="Use binned fitting (default: unbinned)")
    parser.add_argument(
        "--closure_test",
        action="store_true",
        help="Run closure test instead of computing calibration (default: False)",
    )
    parser.add_argument("--fixCat", type=str, default=None, help="Fit only one category")
    parser.add_argument("--backup", action="store_true", help="Enable backup before overwrite")
    parser.add_argument("--extraString", type=str, default="", help="Additional string to add to the output directory name")
    parser.add_argument(
        "--no-dask-client",
        action="store_true",
        help="Run without creating a Dask client (use default scheduler)",
    )
    # which steps to run
    parser.add_argument(
        "--steps",
        choices=["step1", "step2", "step3", "all"],
        default="all",
        help="Steps to run (default: all)",
    )
    args = parser.parse_args()

    years = args.years
    isMC = args.isMC
    ifbinned = args.ifbinned
    fix_fitting_one_cat = args.fixCat
    isMCString = "MC" if isMC else "Data"

    print(f"binned fitting: {ifbinned}")
    print(f"extra string: {args.extraString}")

    if args.input_path:
        # args.input_path (wired through -l/--label by common_workflow.sh's
        # build_calib_cmd, and by the Snakemake MassCalibration* rules) is the label's
        # root save dir -- same convention used by save_SF_rootFiles.py/compact_parquet_data.py
        # (f"{input_path}/stage1_output/{year}/..."). Previously this was ignored in favor
        # of configs/trials.yml's "current" trial, so -l/-c silently had no effect on which
        # stage-1 output got calibrated unless $HMM_TRIAL happened to be set to match.
        stage1_dir = str(Path(args.input_path) / "stage1_output")
        logger.info(f"Using stage1 output from --input_path: {stage1_dir}")
    else:
        stage1_dir = get_stage1_path()  # default = "current"
        logger.info(f"--input_path not given; falling back to configs/trials.yml: {stage1_dir}")
    LOAD_PATH = str(Path(stage1_dir) / "{year}" / "compacted")
    logger.info(f"Using LOAD_PATH: {LOAD_PATH}")

    dir_tag = LOAD_PATH.split("/")[-4] # Fetch the label from the path
    logger.info(f"output dir_tag: {dir_tag}")

    client = None  # IMPORTANT

    # This script's own module-level dask.config.set(scheduler="threads") already
    # forces .compute() calls onto local threads in THIS process regardless of any
    # distributed client that gets created below -- so a real distributed.Client
    # (local or Gateway-backed) provides no compute benefit here. Worse: creating one
    # is actively harmful, because a live distributed Client (unlike the plain
    # "threads" config scheduler) DOES take over .compute() routing and enforces its
    # own per-worker memory_limit -- step2_mass_resolution's single-shot
    # ddf.compute() has been observed needing >10 GiB for one task/partition, which
    # hard-fails with distributed.MemoryError against CONFIG["memory_limit"]="8 GiB"
    # per worker. So --use_gateway/the plain local-client path both now degrade to no
    # client at all (matching the one combination proven to run this pipeline
    # end-to-end), rather than connecting to Gateway or spinning up a local cluster.
    if args.no_dask_client:
        logger.warning("Running WITHOUT a Dask client (default scheduler)")
    elif args.use_gateway:
        from dask_gateway import Gateway

        gateway = Gateway(
            "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
            proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
        )
        if not gateway.list_clusters():
            logger.warning(
                "No running Dask Gateway cluster found for this user (and connecting "
                "one would not help -- see comment above); running WITHOUT a Dask client."
            )
        else:
            logger.warning(
                "A Dask Gateway cluster exists but will NOT be used for this run -- "
                "see comment above for why a live distributed client is unsafe for "
                "this script; running WITHOUT a Dask client instead."
            )
    else:
        logger.warning(
            "Running WITHOUT a Dask client (default scheduler) -- see comment above "
            "for why a local distributed client is unsafe for this script."
        )

    for year in years:
        logger.info(f"Processing year: {year}")
        # output directory format: validation/ebeMassResCalibration/<binned/unbinned>/<isMCString>_<year>_<extraString>
        output_dir = f"validation/ebeMassResCalibration/{dir_tag}/{'binned' if ifbinned else 'unbinned'}/{year}/{isMCString}_{year}_{args.extraString}"
        skim_dir  = f"validation/ebeMassResCalibration/{dir_tag}/skim_zpeak/{isMCString}_{year}"
        print(f"Output directory: {output_dir}")
        # sys.exit(0)

        create_directory(f"{output_dir}")
        create_directory(f"{skim_dir}")
        CalibrationJSONFile = f"res_calib_BS_correction_{year}_{isMCString}_nanoAODv{args.NanoAODv}.json"

        if isMC:
            # INPUT_DATASET = f"{LOAD_PATH.format(year=year)}/dy*MiNNLO/*/*.parquet"
            # Sample naming differs by NanoAOD campaign/label: nanoAODv12 (Run2/2022-2023)
            # labels use "dyTo2L_M-50_incl"; nanoAODv15 2024-conditions (2024/2025/2026)
            # labels use "dyTo2Mu_M-50_aMCatNLO". Pick whichever exists for this label/year.
            dy_candidates = ["dyTo2L_M-50_incl", "dyTo2Mu_M-50_aMCatNLO"]
            dy_sample = next(
                (c for c in dy_candidates if os.path.isdir(f"{LOAD_PATH.format(year=year)}/{c}")),
                dy_candidates[0],
            )
            INPUT_DATASET = f"{LOAD_PATH.format(year=year)}/{dy_sample}/*/*.parquet"
        else:
            INPUT_DATASET = f"{LOAD_PATH.format(year=year)}/data_*/*/*.parquet"

        logger.info(f"Input dataset: {INPUT_DATASET}")

        logger.info(f"Steps to run: {args.steps}")

        if args.backup:
            backup_file(f"{output_dir}/fit_results.csv")
            backup_file(f"{output_dir}/resolution_results.csv")
            backup_file(f"{output_dir}/calibration_factors.csv")
            backup_file(f"{output_dir}/calibration_factors.tex")
            backup_file(f"{output_dir}/calibration_factors_rounded.tex")
            backup_file(f"{output_dir}/calibration_factors_precision.tex")
            backup_file(f"{output_dir}/{CalibrationJSONFile}")
            backup_file(f"{output_dir}/calibration_results_calibrated.csv")
            backup_file(f"{output_dir}/fit_params.json")

        skim_path = f"{skim_dir}/zpeak_skim.parquet"
        if os.path.exists(skim_path):
            logger.info(f"Using cached skim: {skim_path}")
            with timed("read cached skim"):
                ddf = dd.read_parquet(skim_path)
        else:
            logger.info("Building Z-peak skim...")
            with timed("compute df_computed for step1"):
                ddf = dd.read_parquet(INPUT_DATASET)[CONFIG["fields_with_errors"]]
                ddf = ddf[(ddf["dimuon_mass"] > CONFIG["zcr_filter_range"][0]) & (ddf["dimuon_mass"] < CONFIG["zcr_filter_range"][1])]
                ddf.to_parquet(skim_path, write_index=False, overwrite=True)
                ddf = dd.read_parquet(skim_path)

        if not args.closure_test:
            if args.steps == "step1" or args.steps == "all":
                if fix_fitting_one_cat:
                    # A single category was explicitly requested (either directly by the
                    # user, or because we ARE one of run_step1_categories_isolated's
                    # per-category subprocesses) -- run it in-process as before.
                    df_fit = step1_mass_fitting_zcr(
                        ddf,
                        output_dir,
                        skim_dir=skim_dir,
                        fix_fitting_one_cat=fix_fitting_one_cat,
                        ifbinned=ifbinned,
                        inputFilePath=LOAD_PATH.format(year=year),
                    )
                else:
                    # Full run over every category: isolate each one in its own
                    # subprocess (see run_step1_categories_isolated's docstring for why).
                    df_fit = run_step1_categories_isolated(
                        args, year, ddf, output_dir, skim_dir,
                    )
                if fix_fitting_one_cat:
                    # df_fit currently contains ONLY the newly refit category from step1
                    df_fit_new = df_fit.copy()

                    # load old results (may not exist yet, e.g. this is the very first
                    # category ever fit for this output_dir -- start from empty rather
                    # than crashing)
                    if os.path.exists(f"{output_dir}/fit_results.csv"):
                        df_fit_old = pd.read_csv(f"{output_dir}/fit_results.csv")
                    else:
                        df_fit_old = pd.DataFrame(columns=["cat_name", "fit_val", "fit_err"])
                    df_fit_old["orig_idx"] = df_fit_old.index

                    # keep only the new row for the category under consideration
                    df_fit_new = df_fit_new[df_fit_new["cat_name"] == fix_fitting_one_cat]
                    if df_fit_new.empty:
                        # Critical: refit did not produce any row for the requested category.
                        # Log detailed context to aid debugging before raising.
                        available_cats = (
                            df_fit["cat_name"].unique().tolist()
                            if isinstance(df_fit, pd.DataFrame) and "cat_name" in df_fit.columns
                            else None
                        )
                        logger.critical(
                            "Refit produced no row for category '%s'. "
                            "Refit dataframe shape: %s. Available categories in refit results: %s",
                            fix_fitting_one_cat,
                            getattr(df_fit, "shape", None),
                            available_cats,
                        )
                        raise RuntimeError(f"Refit produced no row for category {fix_fitting_one_cat}")

                    # append + overwrite the category with the new values
                    df_fit_merged = pd.concat([df_fit_old, df_fit_new], ignore_index=True)

                    # drop duplicates keeping the LAST (the new one)
                    df_fit_merged = df_fit_merged.drop_duplicates(subset=["cat_name"], keep="last")

                    # restore original order: use old orig_idx; new row goes to the end if orig_idx missing
                    df_fit_merged = (
                        df_fit_merged.sort_values(by="orig_idx", na_position="last")
                                    .drop(columns="orig_idx")
                                    .reset_index(drop=True)
                    )
                    df_fit = df_fit_merged.copy()
                save_dataframe_to_csv(df_fit, f"{output_dir}/fit_results.csv", "fit results")

            if args.steps == "step2" or args.steps == "all":
                df_res = step2_mass_resolution(
                    ddf,
                    output_dir=output_dir,
                    pdfFile_ExtraText="",
                    n_boot=300,
                )
                save_dataframe_to_csv(df_res, f"{output_dir}/median_for_cats.csv", "median results")

            if fix_fitting_one_cat or args.steps == "step3" or args.steps == "all":
                # read df_fit and df_res from previous steps
                df_fit = pd.read_csv(f"{output_dir}/fit_results.csv")
                df_res = pd.read_csv(f"{output_dir}/median_for_cats.csv")

                df_merged = step3_compute_calibration(df_fit, df_res)
                save_dataframe_to_csv(df_merged, f"{output_dir}/calibration_factors.csv", "calibration factors")

            if fix_fitting_one_cat or args.steps == "all":
                df_merged = pd.read_csv(f"{output_dir}/calibration_factors.csv")
                # Save LaTeX tables
                for fmt, rounding in [(f"calibration_factors.tex", None),
                                    (f"calibration_factors_rounded.tex", 4),
                                    (f"calibration_factors_precision.tex", 3)]:
                    df_tmp = df_merged[
                        [
                            "cat_name",
                            "fit_val",
                            "fit_err",
                            "median_val_NonCal",
                            "calibration_factor",
                        ]
                    ].copy()
                    if rounding is not None:
                        for col in ["fit_val", "fit_err", "median_val_NonCal", "calibration_factor"]:
                            df_tmp[col] = df_tmp[col].map(lambda x: f"{x:.{rounding}f}")
                    df_tmp.to_latex(f"{output_dir}/{fmt}", index=False)

                save_calibration_json(df_merged, f"{output_dir}/{CalibrationJSONFile}")

        else:
            closure_csv = f"{output_dir}/closure_results_resolutionBinning.csv"

            # if calibration_results_calibrated.csv exists, skip calibration
            if os.path.exists(closure_csv):
                logger.info(f"{closure_csv} exists, skipping calibration step.")
                df_closure = pd.read_csv(closure_csv)
            else:
                with timed("Total closure test time:"):
                    df_closure = closure_test_resolution_binning(
                        ddf,
                        output_dir=f"{output_dir}/closure_test",
                        CalibrationFactorJSONFile=f"{output_dir}/{CalibrationJSONFile}",
                        ifbinned=ifbinned,
                        pdfFile_ExtraText="",
                        fix_bin=fix_fitting_one_cat,
                    )
                # if fix_fitting_one_cat is not None, then update the existing closure_csv
                if fix_fitting_one_cat is not None and os.path.exists(closure_csv):
                    df_existing = pd.read_csv(closure_csv)
                    # remove the row corresponding to fix_fitting_one_cat
                    df_existing = df_existing[
                        df_existing["cat_name"] != int(fix_fitting_one_cat)
                    ]
                    # append the new row
                    df_closure = pd.concat([df_existing, df_closure], ignore_index=True)
                df_closure.to_csv(closure_csv, index=False)
            print("plot closure comparison calibrated vs uncalibrated...")
            plot_closure_comparison_calibrated_uncalibrated(
                df_closure, output_dir
            )
    if client is not None:
        close_dask_client()

if __name__ == "__main__":
    main()
