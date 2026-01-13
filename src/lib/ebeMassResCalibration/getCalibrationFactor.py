#!/usr/bin/env python3

import os
import sys
import time
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
from contextlib import contextmanager

import correctionlib

import logging
from modules.utils import logger
from modules.trials import get_stage1_path
from modules.daskHelper import get_dask_client
from modules.daskHelper import close_dask_client
from modules.daskHelper import get_dask_gateway_client

from basic_class_for_calibration import (
    get_calib_categories,
    generateBWxDCB_plot,
    closure_test_from_df,
    plot_closure_comparison_calibrated_uncalibrated,
    save_calibration_json,
    filter_region,
    closure_test_resolution_binning,
    CONFIG,
    plot_histogram,
)


@contextmanager
def timed(msg):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    logger.info(f"[TIMER] {msg}: {dt:.3f} s")


def _setup_path():
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

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

    if fix_fitting_one_cat:
        data_categories = data_categories[fix_fitting_one_cat]

    print("Data categories for fitting:")
    for cat_name in data_categories.keys():
        print(f" - {cat_name}")

    df_fit = pd.DataFrame(columns=["cat_name", "fit_val", "fit_err"])
    for cat_name, mask in data_categories.items():
        if fix_fitting_one_cat and cat_name != fix_fitting_one_cat:
            logger.debug(f"Skipping category {cat_name}, as no re-fitting required.")
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
                    np.save(f"{skim_dir}/mass_{cat_name}.npy", mass)

        if mass.size == 0:
            logger.debug(f"Category {cat_name} has no events, skipping.")
            continue
        df_fit = generateBWxDCB_plot(
            mass,
            cat_name,
            nbins=CONFIG["nbins"],
            df_fit=df_fit,
            output_dir=output_dir,
            logfile="CalibrationLog.txt",
            ifbinned=ifbinned,
            inputFilePath=inputFilePath,
        )

    logger.info("Step 1 completed in {:.2f} s".format(time.time() - tstart))
    return df_fit


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
        dimuon_ebe_mass_res_NonCalc = lambda x: np.sqrt(x["dpt1"]**2 + x["dpt2"]**2),
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

        vals = cat_df["dimuon_ebe_mass_res_NonCalc"].to_numpy()
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
    parser = argparse.ArgumentParser(description="Mass resolution calibration workflow")
    parser.add_argument("--isMC", action="store_true", help="Run on MC samples (default: False)")
    # binned or unbinned fitting
    parser.add_argument("--ifbinned", action="store_true", help="Use binned fitting (default: unbinned)")
    parser.add_argument(
        "--closure_test",
        action="store_true",
        help="Run validation instead of computing calibration (default: False)",
    )
    parser.add_argument("--fixCat", type=str, default=None, help="Fit only one category")
    parser.add_argument("--years", nargs="+", default=["2018", "2017", "2016postVFP", "2016preVFP"], help="List of years to process")
    parser.add_argument("--backup", action="store_true", help="Enable backup before overwrite")
    parser.add_argument("--extraString", type=str, default="", help="Additional string to add to the output directory name")
    parser.add_argument(
        "--use_gateway",
        dest="use_gateway",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, uses dask gateway client instead of local",
    )
    parser.add_argument(
        "--nanoAODv",
        dest="nanoAODv",
        default="12"
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

    stage1_dir = get_stage1_path()  # default = "current"
    LOAD_PATH = str(Path(stage1_dir) / "{year}" / "compacted")
    logger.info(f"Using LOAD_PATH: {LOAD_PATH}")

    dir_tag = LOAD_PATH.split("/")[-4] # Fetch the label from the path
    logger.info(f"output dir_tag: {dir_tag}")

    # Initialize Dask client
    if args.use_gateway:
        logger.info("Using Dask Gateway client")
        # client = get_dask_gateway_client()
        from dask_gateway import Gateway

        gateway = Gateway(
            "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
            proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
        )
        cluster_info = gateway.list_clusters()[0]  # get the first cluster by default. There only should be one anyways
        client = gateway.connect(cluster_info.name).get_client()
        logger.info("Gateway Client created")

        client.run(_setup_path)

        # sanity check
        client.run(lambda: __import__("basic_class_for_calibration"))
    else:
        logger.info("Using local Dask client")
        client = get_dask_client(
            n_workers=CONFIG["n_workers"],
            threads_per_worker=CONFIG["threads_per_worker"],
            memory_limit=CONFIG["memory_limit"],
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
        CalibrationJSONFile = f"res_calib_BS_correction_{year}_{isMCString}_nanoAODv{args.nanoAODv}.json"

        if isMC:
            INPUT_DATASET = f"{LOAD_PATH.format(year=year)}/dy*MiNNLO/*/*.parquet"
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
                df_fit = step1_mass_fitting_zcr(
                    ddf,
                    output_dir,
                    skim_dir=skim_dir,
                    fix_fitting_one_cat=fix_fitting_one_cat,
                    ifbinned=ifbinned,
                    inputFilePath=LOAD_PATH.format(year=year),
                )
                if fix_fitting_one_cat:
                    # df_fit currently contains ONLY the newly refit category from step1
                    df_fit_new = df_fit.copy()

                    # load old results
                    df_fit_old = pd.read_csv(f"{output_dir}/fit_results.csv")
                    df_fit_old["orig_idx"] = df_fit_old.index

                    # keep only the new row for the category under consideration
                    df_fit_new = df_fit_new[df_fit_new["cat_name"] == fix_fitting_one_cat]
                    if df_fit_new.empty:
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
                df_closure = closure_test_resolution_binning(
                    ddf,
                    output_dir=f"{output_dir}/closure_test",
                    CalibrationFactorJSONFile=f"{output_dir}/{CalibrationJSONFile}",
                    ifbinned=ifbinned,
                    pdfFile_ExtraText="",
                )
                df_closure.to_csv(closure_csv, index=False)
            print("plot closure comparison calibrated vs uncalibrated...")
            plot_closure_comparison_calibrated_uncalibrated(
                df_closure, output_dir
            )
    close_dask_client()

if __name__ == "__main__":
    main()
