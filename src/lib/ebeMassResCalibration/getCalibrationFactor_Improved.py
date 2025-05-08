#!/usr/bin/env python3

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse

import dask
dask.config.set(scheduler="threads")  # or "single-threaded", "processes"

from dask import delayed
import dask.dataframe as dd
import awkward as ak
from distributed import Client

import correctionlib
import logging
from modules.utils import logger
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

def step1_mass_fitting_zcr(data_events, out_string="", fix_fitting_one_cat=None):
    logger.info("=== Step 1: Mass fitting in ZCR ===")
    tstart = time.time()

    data_categories = get_calib_categories(data_events)

    df_fit = pd.DataFrame(columns=["cat_name", "fit_val", "fit_err"])
    for cat_name, mask in data_categories.items():
        if fix_fitting_one_cat and cat_name != fix_fitting_one_cat:
            logger.debug(f"Skipping category {cat_name}, as no re-fitting required.")
            continue
        mass = ak.to_numpy(data_events["dimuon_mass"][mask])
        if mass.size == 0:
            logger.debug(f"Category {cat_name} has no events, skipping.")
            continue
        df_fit = generateBWxDCB_plot(mass, cat_name, nbins=CONFIG["nbins"], df_fit=df_fit, out_string=out_string, logfile="CalibrationLog.txt")

    logger.info("Step 1 completed in {:.2f} s".format(time.time() - tstart))
    return df_fit

def step2_mass_resolution(df, out_string="", CalibrationFactorJSONFile=None, pdfFile_ExtraText="", UseFullSampleForCalibration=False):
    logger.info("=== Step 2: Mass resolution calculation ===")
    tstart = time.time()

    create_directory(f"plots/{out_string}")

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
                       f"plots/{out_string}/mass_resolution_{cat_name}_NonCalibrated_{pdfFile_ExtraText}.pdf",
                       median=med_noncal)

        if CalibrationFactorJSONFile:
            med_cal = cat_data["dimuon_ebe_mass_res_calc"].median()
            res_results.append({"cat_name": cat_name, "median_val": med_cal})
            plot_histogram(cat_data["dimuon_ebe_mass_res_calc"], CONFIG["nbins"], (0.5, 3.0),
                           "Dimuon mass resolution (GeV)", "Events",
                           f"Category {cat_name}\nMedian Cal = {med_cal:.4f} GeV",
                           f"plots/{out_string}/mass_resolution_{cat_name}_Calibrated_{pdfFile_ExtraText}.pdf",
                           median=med_cal)

            # fit it
            mass = ak.to_numpy(result["dimuon_mass"][mask])
            df_fit = generateBWxDCB_plot(mass, cat_name, nbins=CONFIG["nbins"], df_fit=df_fit, out_string=out_string, logfile=f"CalibrationLog_{pdfFile_ExtraText}.txt", pdfFile_ExtraText=pdfFile_ExtraText)
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
    parser.add_argument("--validate", action="store_true", help="Run validation instead of computing calibration (default: False)")
    parser.add_argument("--fixCat", type=str, default=None, help="Fit only one category")
    parser.add_argument("--years", nargs="+", default=["2018", "2017", "2016postVFP", "2016preVFP"], help="List of years to process")

    args = parser.parse_args()

    years = args.years
    isMC = args.isMC
    ComputeCalibrationFactors = not args.validate
    fix_fitting_one_cat = args.fixCat
    isMCString = "MC" if isMC else "Data"
    UseFullSampleForCalibration = True

    for year in years:
        logger.info(f"Processing year: {year}")
        if UseFullSampleForCalibration:
            out_string = f"{year}_{isMCString}_CalibrateWithFullSample"
        else:
            out_string = f"{year}_{isMCString}_Train75_Val25"
        create_directory(f"plots/{out_string}")
        CalibrationJSONFile = f"res_calib_BS_correction_{year}_nanoAODv12.json"

        if isMC:
            INPUT_DATASET = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/April19_NanoV12/stage1_output/{year}/f1_0/dy*MiNNLO/*/*.parquet"
        else:
            INPUT_DATASET = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/April19_NanoV12/stage1_output/{year}/f1_0/data_*/*/*.parquet"

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
            df_fit = step1_mass_fitting_zcr(data_events, out_string, fix_fitting_one_cat=fix_fitting_one_cat)
            if fix_fitting_one_cat:
                # save last csv file, as backup
                os.system(f"cp plots/{out_string}/resolution_results.csv plots/{out_string}/resolution_results_backup.csv")
                # update the f"plots/{out_string}/fit_results{out_string}.csv" with the new cat_name available in df_fit
                # then save it to the same file
                df_fit = pd.read_csv(f"plots/{out_string}/fit_results{out_string}.csv")
                df_fit = pd.concat([df_fit, df_fit[df_fit["cat_name"] == fix_fitting_one_cat]])
                df_fit = df_fit.drop_duplicates(subset=["cat_name"], keep="last")
                df_fit = df_fit.reset_index(drop=True)
                save_dataframe_to_csv(df_fit, f"plots/{out_string}/fit_results_{fix_fitting_one_cat}.csv", "fit results")

                df_res = pd.read_csv(f"plots/{out_string}/resolution_results.csv")
            else:
                save_dataframe_to_csv(df_fit, f"plots/{out_string}/fit_results.csv", "fit results")

                if not UseFullSampleForCalibration: ddf = dd.from_pandas(df_train)
                df_res = step2_mass_resolution(ddf, out_string, UseFullSampleForCalibration=UseFullSampleForCalibration)
                save_dataframe_to_csv(df_res, f"plots/{out_string}/resolution_results.csv", "resolution results")

            df_merged = step3_compute_calibration(df_fit, df_res)
            save_dataframe_to_csv(df_merged, f"plots/{out_string}/calibration_factors{fix_fitting_one_cat}.csv")

            # Save LaTeX tables
            for fmt, rounding in [(f"calibration_factors{fix_fitting_one_cat}.tex", None),
                                  (f"calibration_factors_rounded{fix_fitting_one_cat}.tex", 4),
                                  (f"calibration_factors_precision{fix_fitting_one_cat}.tex", 3)]:
                df_tmp = df_merged[["cat_name", "fit_val", "fit_err", "median_val_NonCal", "calibration_factor"]]
                if rounding:
                    df_tmp = df_tmp.round(rounding)
                df_tmp.to_latex(f"plots/{out_string}/{fmt}", index=False)

            save_calibration_json(df_merged, f"plots/{out_string}/{CalibrationJSONFile}")

        else:
            if not UseFullSampleForCalibration: ddf = dd.from_pandas(df_valid)
            df_res_calibrated = step2_mass_resolution(ddf, out_string,
                                                      CalibrationFactorJSONFile=f"plots/{out_string}/{CalibrationJSONFile}",
                                                      UseFullSampleForCalibration=UseFullSampleForCalibration)
            df_res_calibrated.to_csv(f"plots/{out_string}/calibration_results_calibrated.csv", index=False)
            plot_closure_comparison_calibrated_uncalibrated(df_res_calibrated, out_string)

if __name__ == "__main__":
    main()
