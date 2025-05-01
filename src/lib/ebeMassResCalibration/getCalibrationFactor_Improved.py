#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dask_awkward as dak
import dask.dataframe as dd
import awkward as ak
from distributed import Client

from basic_class_for_calibration import get_calib_categories
from basic_class_for_calibration import generateBWxDCB_plot
from basic_class_for_calibration import generateVoigtian_plot
from basic_class_for_calibration import generateBWxDCB_plot_bkgErfxExp
from basic_class_for_calibration import closure_test_from_df
from basic_class_for_calibration import closure_test_from_df_BothBeforeAndAfter
from basic_class_for_calibration import closure_test_from_df_BothBeforeAndAfter_OnSameCanvas
from basic_class_for_calibration import save_calibration_json
from basic_class_for_calibration import filter_region

import logging
from modules.utils import logger

##############################
# Step 1: Mass Fitting in ZCR
##############################
def step1_mass_fitting_zcr(parquet_path, out_string = ""):
    logger.info("=== Step 1: Mass fitting in ZCR ===")
    tstart = time.time()

    # Start a Dask distributed client.
    client = Client(n_workers=16, threads_per_worker=1, processes=True, memory_limit='8 GiB')

    # Load the data with dask_awkward.
    data_events = dak.from_parquet(parquet_path)

    # Apply a ZCR filter: here we assume that the field "z_peak" is defined.
    # region_filter = ak.fill_none(data_events["z_peak"], value=False)
    # data_events = data_events[region_filter]
    data_events = filter_region(data_events, "z_peak")


    # Only select the fields needed for calibration (muon1 and muon2 eta, pt, and dimuon mass)
    fields_of_interest = ["mu1_pt", "mu1_eta", "mu2_eta", "dimuon_mass"]
    data_events = data_events[fields_of_interest]

    # Bring the data into memory as an Awkward array.
    data_events = ak.zip({field: data_events[field] for field in data_events.fields}).compute()

    # Build calibration category masks.
    data_categories = get_calib_categories(data_events)
    logger.debug(f"Total number of categories (Step 1): {len(data_categories)}")

    nbins = 100
    # fit_results = []
    counter = 0

    # define a data frame to store the results
    df_fit = pd.DataFrame(columns=["cat_name", "fit_val", "fit_err"])

    # Loop over each calibration category and perform mass fitting.
    for cat_name, mask in data_categories.items():
        cat_dimuon_mass = ak.to_numpy(data_events["dimuon_mass"][mask])
        if cat_dimuon_mass.size == 0:
            logger.info(f"Category {cat_name} has no events, skipping.")
            continue
        # For example, use BWxDCB fit for the first 12 categories, Voigtian fit for the rest.
        df_fit = generateBWxDCB_plot(cat_dimuon_mass, cat_name, nbins=nbins, df_fit=df_fit, out_string=out_string, logfile=f"CalibrationLog.txt")
        # if counter < 12:
        #     # Your function should return a dict like {"cat_name": cat_name, "fit_val": <value>}
        #     df_fit = generateBWxDCB_plot(cat_dimuon_mass, cat_name, nbins=nbins, df_fit=df_fit, out_string=out_string, logfile=f"CalibrationLog.txt")
        # else:
        #     # df_fit = generateBWxDCB_plot_bkgErfxExp(cat_dimuon_mass, cat_name, nbins=nbins, df_fit=df_fit, out_string=out_string, logfile=f"CalibrationLog.txt")
        #     df_fit = generateVoigtian_plot(cat_dimuon_mass, cat_name, nbins=nbins, df_fit=df_fit, out_string=out_string, logfile=f"CalibrationLog.txt") # >= 12
        # fit_results.append(fit_info)
        logger.debug("------"*20)
        logger.debug(df_fit)
        logger.debug("------"*20)
        # time.sleep(2) # sleep for 2 second for debug
        counter += 1
        # if counter > 3:
            # logger.warning("Exiting loop after 4 iterations for debugging.")
            # break

    client.close()
    logger.debug("Step 1 completed in {:.2f} seconds.".format(time.time() - tstart))
    logger.debug("Sample fit results:")
    logger.debug(df_fit)

    # Convert list of dictionaries into a DataFrame.
    # df_fit = pd.DataFrame(fit_results)
    return df_fit

##############################
# Step 2: Mass Resolution using dask.dataframe
##############################
def step2_mass_resolution(parquet_path, out_string = ""):
    logger.info("=== Step 2: Mass resolution calculation ===")
    tstart = time.time()

    # create directory named out_string if it does not exist
    os.makedirs(f"plots/{out_string}", exist_ok=True)

    # Load the same common input parquet files.
    df = dd.read_parquet(parquet_path)

    # Only select the fields needed for calibration (muon1 and muon2 eta, pt, and dimuon mass)
    fields_of_interest = ["mu1_pt", "mu1_ptErr", "mu2_pt", "mu2_ptErr", "mu1_eta", "mu2_eta", "dimuon_mass"]
    df = df[fields_of_interest]


    # apply a ZCR filter: 80-100 GeV, without using the filter_region function
    df = df[(df['dimuon_mass'] > 76) & (df['dimuon_mass'] < 106)]

    # Compute per-event muon energy (assumed as half the dimuon mass) and error contributions.
    df = df.assign(
        muon_E = df['dimuon_mass'] / 2,
        dpt1 = (df['mu1_ptErr'] / df['mu1_pt']) * (df['dimuon_mass'] / 2),
        dpt2 = (df['mu2_ptErr'] / df['mu2_pt']) * (df['dimuon_mass'] / 2)
    )
    # Compute the absolute mass resolution.
    df = df.assign(
        dimuon_ebe_mass_res_calc = np.sqrt(df['dpt1']**2 + df['dpt2']**2)
    )
    # Here we keep the absolute resolution; if desired, you can compute a relative resolution.
    # For this merged example we use absolute resolution.

    # Bring the result into a Pandas DataFrame.
    result = df.compute()
    logger.debug("Sample mass resolution data:")
    logger.debug(result[['dimuon_ebe_mass_res_calc']].head())

    # Build calibration categories on the same result.
    calib_cats = get_calib_categories(result)

    res_results = []
    for cat_name, mask in calib_cats.items():
        cat_data = result[mask]
        if cat_data.empty:
            logger.warning(f"Category {cat_name} has no events, skipping.")
            continue
        median_val = cat_data['dimuon_ebe_mass_res_calc'].median()
        res_results.append({"cat_name": cat_name, "median_val": median_val})
        # Optionally, plot a histogram (uncomment if desired):
        plt.figure()
        plt.hist(cat_data['dimuon_ebe_mass_res_calc'], bins=100, range=(0, 5.0), color='C0', alpha=0.7)
        plt.xlabel('Dimuon mass resolution (GeV)')
        plt.ylabel('Events')
        plt.title(f"Category {cat_name}\nMedian = {median_val:.4f} GeV")
        plt.axvline(median_val, color='red', linestyle='dashed', linewidth=2,
                    label=f"Median: {median_val:.4f} GeV")
        plt.legend()
        plt.savefig(f'plots/{out_string}/mass_resolution_{cat_name}.pdf')
        plt.close()
        logger.info(f"Saved histogram for category {cat_name} (median = {median_val:.4f} GeV)")

    logger.info("Step 2 completed in {:.2f} seconds.".format(time.time() - tstart))
    df_res = pd.DataFrame(res_results)
    return df_res

##############################
# Step 3: Compute Calibration Factor
##############################
def step3_compute_calibration(df_fit, df_res):
    logger.info("=== Step 3: Compute calibration factors ===")
    # Merge the two DataFrames on "cat_name"
    df_merged = pd.merge(df_fit, df_res, on="cat_name", how="inner")
    logger.debug("Merged DataFrame:")
    logger.debug(df_merged)
    logger.debug("---"*20)
    # For example, calibration_factor = fit_val / median_val
    if "fit_val" in df_merged.columns and "median_val" in df_merged.columns:
        df_merged["calibration_factor"] = df_merged["fit_val"] / df_merged["median_val"]
    else:
        logger.warning("Warning: 'fit_val' and/or 'median_val' columns are missing in the merged DataFrame.")
        df_merged["calibration_factor"] = np.nan

    logger.debug("Computed calibration factors:")
    logger.debug(df_merged)
    logger.debug("---"*20)
    return df_merged

##############################
# Step 4: Save to CSV
##############################
def step4_save_csv(df_merged, out_csv="calibration_factors.csv"):
    logger.info("=== Step 4: Save results to CSV ===")
    df_merged.to_csv(out_csv, index=False)
    logger.debug(f"Saved merged calibration factors to {out_csv}")

##############################
# Main Entry Point
##############################
def main():
    total_time_start = time.time()

    years = ["2018", "2017", "2016postVFP", "2016preVFP"]
    for year in years:
        logger.info(f"Processing year: {year}")
        # out_String = "2018C_HIG_19_006_SignalOnlyDSCB_BkgLaundau_FloatmZ"
        # out_String = "2018C_LastMeeting_ChangeRevLandauToLandau"
        # out_String = f"{year}_SigOnlyDSCB_bkgRevLandau"
        # out_String = "2018_SigOnlyDSCB_bkgRooCMSShape"
        out_String = f"{year}_SigOnlyDSCB_bkgRooCMSShape_CrossCheck_DY"
        # INPUT_DATASET = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/April19_NanoV12/stage1_output/{year}/f1_0/data_*/*/*.parquet"
        # INPUT_DATASET = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/April19_NanoV12/stage1_output/{year}/f1_0/data_F/*/*.parquet"
        # INPUT_DATASET = "/depot/cms/users/shar1172/hmm/copperheadV1clean/April19_NanoV12/stage1_output/2018/f1_0/data_C/*/*.parquet"
        INPUT_DATASET = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/April19_NanoV12/stage1_output/{year}/f1_0/dy_*MiNNLO/*/*.parquet"

        # create directory named out_string if it does not exist
        os.makedirs(f"plots/{out_String}", exist_ok=True)

        # Step 1: Mass Fitting in ZCR
        df_fit = step1_mass_fitting_zcr(INPUT_DATASET, out_String)
        logger.debug(df_fit)
        # sys.exit(0)
        # write to a csv file
        df_fit.to_csv(f"plots/{out_String}/fit_results{out_String}.csv", index=False)

        # Step 2: Mass Resolution Calculation
        df_res = step2_mass_resolution(INPUT_DATASET, out_String)
        df_res.to_csv(f"plots/{out_String}/resolution_results{out_String}.csv", index=False)

        # debug: logger.debug the two DataFrames
        logger.debug("="*40)
        logger.debug("Sample fit results:")
        logger.debug(df_fit)
        logger.debug("Sample resolution results:")
        logger.debug(df_res)
        logger.debug("="*40)

        # Step 3: Compute the calibration factor (ratio)
        df_merged = step3_compute_calibration(df_fit, df_res)

        # Step 4: Save the final merged DataFrame to a CSV file.
        step4_save_csv(df_merged, f"plots/{out_String}/calibration_factors.csv")

        # Save the df_merged DataFrame as a table format for latex: Only field "cat_name", "fit_val", "median_val", "calibration_factor"
        df_merged.to_latex(f"plots/{out_String}/calibration_factors_WithFitError.tex", index=False)
        df_merged_tex = df_merged[["cat_name", "fit_val", "median_val", "calibration_factor"]]
        df_merged_tex.to_latex(f"plots/{out_String}/calibration_factors.tex", index=False)
        # rounding
        df_merged_tex = df_merged_tex.round(4)
        df_merged_tex.to_latex(f"plots/{out_String}/calibration_factors_rounded.tex", index=False)
        # precision
        df_merged_tex.to_latex(f"plots/{out_String}/calibration_factors_precision.tex", index=False, float_format="%.3f")
        df_merged_tex.to_latex(f"plots/{out_String}/calibration_factors_precision.tex", index=False, float_format="%.3f", longtable=True)


        # Step 5: Save the calibration factors to a JSON file.
        save_calibration_json(df_merged, f"plots/{out_String}/calibration_factors.json")

        #

        # Step 5: Closure test
        # closure_test_from_df(df_merged, out_String) # This function will give me the closure test with GeoFit. As of now, the input files does not have latest BSC applied and stage1 run with GeoFit.
        closure_test_from_df_BothBeforeAndAfter(df_merged, out_String) # This function will give me the closure test with GeoFit. As of now, the input files does not have latest BSC applied and stage1 run with GeoFit.
        closure_test_from_df_BothBeforeAndAfter_OnSameCanvas(df_merged, out_String) # This function will give me the closure test with GeoFit. As of now, the input files does not have latest BSC applied and stage1 run with GeoFit.
        logger.info("All steps completed!")
        logger.info(f"Total time elapsed: {time.time() - total_time_start:.2f} s")

if __name__ == "__main__":
    main()
