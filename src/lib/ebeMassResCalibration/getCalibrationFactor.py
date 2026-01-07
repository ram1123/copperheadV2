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
)

# Configuration constants
CONFIG = {
    "n_workers": 16,
    "threads_per_worker": 1,
    "memory_limit": "8 GiB",
    "zcr_filter_range": (75, 105),
    "nbins": 120,
    "fields_of_interest": ["mu1_pt", "mu1_eta", "mu2_eta", "dimuon_mass"],
    "fields_with_errors": ["mu1_pt", "mu1_ptErr", "mu2_pt", "mu2_ptErr", "mu1_eta", "mu2_eta", "dimuon_mass"],
}

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

def plot_histogram(data, bins, range, xlabel, ylabel, title, output_path, median=None):
    plt.figure()
    plt.hist(data, bins=bins, range=range, color='C0', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if median is not None:
        plt.axvline(median, color='red', linestyle='dashed', linewidth=2, label=f"Median: {median:.4f}")
        plt.legend()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved plot to {output_path}")

def step1_mass_fitting_zcr(data_events, output_dir="", fix_fitting_one_cat=None, ifbinned=False, inputFilePath=""):
    logger.info("=== Step 1: Mass fitting in ZCR ===")
    tstart = time.time()

    data_categories = get_calib_categories(data_events)

    print("Data categories for fitting:")
    for cat_name in data_categories.keys():
        print(f" - {cat_name}")

    df_fit = pd.DataFrame(columns=["cat_name", "fit_val", "fit_err"])
    for cat_name, mask in data_categories.items():
        if fix_fitting_one_cat and cat_name != fix_fitting_one_cat:
            logger.debug(f"Skipping category {cat_name}, as no re-fitting required.")
            continue
        mass = ak.to_numpy(data_events["dimuon_mass"][mask])
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

def step2_mass_resolution(df, out_string="", output_dir="tmp", CalibrationFactorJSONFile=None, pdfFile_ExtraText="", UseFullSampleForCalibration=False, ifbinned=False, inputFilePath=""):
    logger.info("=== Step 2: Mass resolution calculation ===")
    tstart = time.time()

    create_directory(f"{output_dir}")

    if CalibrationFactorJSONFile:
        # For validation choose randomly 50% of the data
        # Create a pseudo-random mask using entry index
        logger.debug(f"Entries before truncate: {len(df)}")
        if UseFullSampleForCalibration:
            # When we use full sample for calibration, then we just need to get
            # the randomly 50% of the data for the closure test
            df = df.map_partitions(lambda part: part[np.random.rand(len(part)) < 0.5])

        logger.debug(f"Entries after truncate: {len(df)}")

        correction_set = correctionlib.CorrectionSet.from_file(CalibrationFactorJSONFile)
        correction = correction_set["BS_ebe_mass_res_calibration"]
        df = df.map_partitions(lambda part: part.assign(
            calibration=correction.evaluate(
                part["mu1_pt"].to_numpy(),
                np.abs(part["mu1_eta"]).to_numpy(),
                np.abs(part["mu2_eta"]).to_numpy()
            )
        ))
    else:
        df = df.assign(calibration=1.0)

    df = df.assign(
        muon_E = df["dimuon_mass"] / 2,
        dpt1 = (df["mu1_ptErr"] / df["mu1_pt"]) * (df["dimuon_mass"] / 2),
        dpt2 = (df["mu2_ptErr"] / df["mu2_pt"]) * (df["dimuon_mass"] / 2),
        dimuon_ebe_mass_res_NonCalc = lambda x: np.sqrt(x["dpt1"]**2 + x["dpt2"]**2),
    )
    if CalibrationFactorJSONFile:
        df = df.assign(dimuon_ebe_mass_res_calc=lambda x: x["dimuon_ebe_mass_res_NonCalc"] * x["calibration"])

    result = df.compute()
    calib_cats = get_calib_categories(result)

    df_fit = pd.DataFrame(columns=["cat_name", "fit_val", "fit_err"])
    res_results = []
    res_results_NonCal = []
    for cat_name, mask in calib_cats.items():
        cat_data = result[mask]
        if cat_data.empty:
            logger.warning(f"Category {cat_name} has no events, skipping.")
            continue

        med_noncal = cat_data["dimuon_ebe_mass_res_NonCalc"].median()
        res_results_NonCal.append({"cat_name": cat_name, "median_val_NonCal": med_noncal})
        plot_histogram(cat_data["dimuon_ebe_mass_res_NonCalc"], CONFIG["nbins"], (0.5, 3.0),
                       "Dimuon mass resolution (GeV)", "Events",
                       f"Category {cat_name}\nMedian NonCal = {med_noncal:.4f} GeV",
                       f"{output_dir}/mass_resolution_{cat_name}_NonCalibrated_{pdfFile_ExtraText}.pdf",
                       median=med_noncal)

        if CalibrationFactorJSONFile:
            med_cal = cat_data["dimuon_ebe_mass_res_calc"].median()
            res_results.append({"cat_name": cat_name, "median_val": med_cal})
            plot_histogram(cat_data["dimuon_ebe_mass_res_calc"], CONFIG["nbins"], (0.5, 3.0),
                           "Dimuon mass resolution (GeV)", "Events",
                           f"Category {cat_name}\nMedian Cal = {med_cal:.4f} GeV",
                           f"{output_dir}/mass_resolution_{cat_name}_Calibrated_{pdfFile_ExtraText}.pdf",
                           median=med_cal)

            # fit it
            mass = ak.to_numpy(result["dimuon_mass"][mask])
            df_fit = generateBWxDCB_plot(mass, cat_name, nbins=CONFIG["nbins"], df_fit=df_fit, out_string=out_string, logfile=f"CalibrationLog_{pdfFile_ExtraText}.txt", pdfFile_ExtraText=pdfFile_ExtraText, ifbinned=False, inputFilePath="", output_dir=output_dir)
            logger.debug("------"*20)
            logger.debug(df_fit)
            logger.debug("------"*20)

    df_res = pd.merge(df_fit, pd.DataFrame(res_results), on="cat_name", how="inner") if CalibrationFactorJSONFile else pd.DataFrame()
    df_res_noncal = pd.DataFrame(res_results_NonCal)
    return pd.merge(df_res, df_res_noncal, on="cat_name", how="inner") if CalibrationFactorJSONFile else df_res_noncal

def step3_compute_calibration(df_fit, df_res):
    df_merged = pd.merge(df_fit, df_res, on="cat_name", how="inner")
    df_merged["calibration_factor"] = df_merged["fit_val"] / df_merged["median_val_NonCal"]
    return df_merged


def main():
    parser = argparse.ArgumentParser(description="Mass resolution calibration workflow")
    parser.add_argument("--isMC", action="store_true", help="Run on MC samples (default: False)")
    # binned or unbinned fitting
    parser.add_argument("--ifbinned", action="store_true", help="Use binned fitting (default: unbinned)")
    parser.add_argument("--validate", action="store_true", help="Run validation instead of computing calibration (default: False)")
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
    args = parser.parse_args()

    years = args.years
    isMC = args.isMC
    ifbinned = args.ifbinned
    ComputeCalibrationFactors = not args.validate
    fix_fitting_one_cat = args.fixCat
    isMCString = "MC" if isMC else "Data"
    UseFullSampleForCalibration = True

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
        # output_dir = f"validation/ebeMassResCalibration/{dir_tag}/{'binned' if ifbinned else 'unbinned'}/{year}/{isMCString}_{year}_{args.extraString}"
        output_dir = f"validation/ebeMassResCalibration/{dir_tag}/{'binned' if ifbinned else 'unbinned'}/{isMCString}_{year}_{args.extraString}"
        print(f"Output directory: {output_dir}")
        # sys.exit(0)

        if not UseFullSampleForCalibration:
            output_dir += "_PartialSampleTrain75Val25"
        create_directory(f"{output_dir}")
        CalibrationJSONFile = f"res_calib_BS_correction_{year}_{isMCString}_nanoAODv12.json"

        if isMC:
            INPUT_DATASET = f"{LOAD_PATH.format(year=year)}/dy*MiNNLO/*/*.parquet"
        else:
            INPUT_DATASET = f"{LOAD_PATH.format(year=year)}/data_*/*/*.parquet"

        logger.info(f"Input dataset: {INPUT_DATASET}")
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

        ddf = dd.read_parquet(INPUT_DATASET)[CONFIG["fields_with_errors"]]
        ddf = ddf[(ddf["dimuon_mass"] > CONFIG["zcr_filter_range"][0]) & (ddf["dimuon_mass"] < CONFIG["zcr_filter_range"][1])]
        if UseFullSampleForCalibration:
            # Use all events
            df_computed = ddf[CONFIG["fields_of_interest"]].compute()
            data_events = ak.Array(df_computed.to_dict(orient="list"))
            ######### Use all events: END
        else:
            # Use only 75% of the events for calibration and 25% for validation
            ddf_full = ddf.reset_index(drop=True)

            # Get total size and define split index
            total_len = len(ddf_full)
            split_idx = int(total_len * 0.75)

            df_computed = ddf_full.compute()
            df_train = df_computed.iloc[:split_idx]
            df_valid = df_computed.iloc[split_idx:]
            data_events = ak.Array(df_train[CONFIG["fields_of_interest"]].to_dict(orient="list"))

        if ComputeCalibrationFactors:
            df_fit = step1_mass_fitting_zcr(
                data_events,
                output_dir,
                fix_fitting_one_cat=fix_fitting_one_cat,
                ifbinned=ifbinned,
                inputFilePath=LOAD_PATH.format(year=year),
            )
            if fix_fitting_one_cat:
                df_fit = pd.read_csv(f"{output_dir}/fit_results.csv")
                df_fit["orig_idx"] = df_fit.index  # Save original index
                # append the fixed category
                df_fit = pd.concat([df_fit, df_fit[df_fit["cat_name"] == fix_fitting_one_cat]])
                # drop duplicates based on "cat_name", keeping the last occurrence
                df_fit = df_fit.drop_duplicates(subset=["cat_name"], keep="last")
                # Sort by original index to maintain order
                df_fit = df_fit.sort_values(by="orig_idx").drop(columns="orig_idx").reset_index(drop=True)

                save_dataframe_to_csv(df_fit, f"{output_dir}/fit_results.csv", "fit results")

                df_res = pd.read_csv(f"{output_dir}/resolution_results.csv")
            else:
                save_dataframe_to_csv(df_fit, f"{output_dir}/fit_results.csv", "fit results")

                if not UseFullSampleForCalibration: ddf = dd.from_pandas(df_train)
                df_res = step2_mass_resolution(
                    ddf,
                    output_dir=output_dir,
                    UseFullSampleForCalibration=UseFullSampleForCalibration,
                    ifbinned=ifbinned,
                    inputFilePath=LOAD_PATH.format(year=year),
                )
                save_dataframe_to_csv(df_res, f"{output_dir}/resolution_results.csv", "resolution results")

            df_merged = step3_compute_calibration(df_fit, df_res)
            save_dataframe_to_csv(df_merged, f"{output_dir}/calibration_factors.csv")

            # Save LaTeX tables
            for fmt, rounding in [(f"calibration_factors.tex", None),
                                (f"calibration_factors_rounded.tex", 4),
                                (f"calibration_factors_precision.tex", 3)]:
                df_tmp = df_merged[["cat_name", "fit_val", "fit_err", "median_val_NonCal", "calibration_factor"]]
                if rounding is not None:
                    for col in ["fit_val", "fit_err", "median_val_NonCal", "calibration_factor"]:
                        df_tmp[col] = df_tmp[col].map(lambda x: f"{x:.{rounding}f}")
                df_tmp.to_latex(f"{output_dir}/{fmt}", index=False)

            save_calibration_json(df_merged, f"{output_dir}/{CalibrationJSONFile}")

        else:
            if not UseFullSampleForCalibration: ddf = dd.from_pandas(df_valid)
            # if calibration_results_calibrated.csv exists, skip calibration
            if os.path.exists(f"{output_dir}/calibration_results_calibrated.csv"):
                logger.info(f"{output_dir}/calibration_results_calibrated.csv exists, skipping calibration step.")
                df_res_calibrated = pd.read_csv(f"{output_dir}/calibration_results_calibrated.csv")
            else:
                df_res_calibrated = step2_mass_resolution(
                    ddf,
                    output_dir=output_dir,
                    CalibrationFactorJSONFile=f"{output_dir}/{CalibrationJSONFile}",
                    UseFullSampleForCalibration=UseFullSampleForCalibration,
                    ifbinned=ifbinned,
                    inputFilePath=LOAD_PATH.format(year=year),
                )
                df_res_calibrated.to_csv(f"{output_dir}/calibration_results_calibrated.csv", index=False)
            print("plot closure comparison calibrated vs uncalibrated...")
            plot_closure_comparison_calibrated_uncalibrated(
                df_res_calibrated, output_dir
            )
    close_dask_client()

if __name__ == "__main__":
    main()
