import awkward as ak
import dask_awkward as dak
import dask
import argparse
import sys
import os
import numpy as np
import json
from collections import OrderedDict
from modules.utils import filterRegion
import glob
import pickle

import logging
from modules.utils import logger
from modules import selection

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# Add it to sys.path
sys.path.insert(0, parent_dir)
# Now you can import your module
from src.lib.histogram.plotting import plotDataMC_compare


def plotStage2DNN_score(hist_dict_bySampleGroup, var, plot_settings, full_save_path, region_name, category, do_logscale=True, binning=None, lumi="", status="Private"):
    """
    hist_dict_bySampleGroup : dictionary with sample group (data, DY, VV) as keys and list of relecant hep histograms as values
    """
    # logger.info(f"hist_dict_bySampleGroup: {hist_dict_bySampleGroup}")

    data_dict = {}
    bkg_MC_dict = {}
    sig_MC_dict = {}
    plot_var = getPlotVar(var)
    if plot_var not in plot_settings.keys():
        logger.info(f"variable {var} not configured in plot settings!")
        return
    for group_name, sample_hist_l  in hist_dict_bySampleGroup.items():
        logger.info(f"{group_name} hist_list types: {[type(h) for h in sample_hist_l]}")
        logger.info(f"{group_name} hist_list len: {len(sample_hist_l)}")
        if len(sample_hist_l) == 0:
            logger.info(f"No histograms found for {group_name}, skipping!")
            continue

        logger.debug(f"Combining histograms for {group_name}...")
        logger.debug(f"Sample histograms keys: {[h.axes.name for h in sample_hist_l]}")

        for i, h in enumerate(sample_hist_l):
            logger.debug(f"Histogram {i} axes: {h.axes.name}")
            for axis in h.axes:
                logger.debug(f"  Axis: {axis.name}, type: {type(axis)}, labels: {getattr(axis, 'categories', 'None')}, edges: {getattr(axis, 'edges', 'None')}")

        # logger.info("sample_hist_l compute:")
        # logger.info(dask.compute(sample_hist_l))
        # logger.info("=" * 50 )

        sample_hist = sum(sample_hist_l)
        to_project_setting = {
            "region" : region_name,
            "channel" : category,
            "variation" : "nominal",
            # "sample_group": group_name,
        }
        logger.debug(f"to_project_setting: {to_project_setting}")
        logger.debug(f"sample_hist: {sample_hist}")

        #  Print/check the type of sample_hist and its keys
        logger.info(f"Type of sample_hist: {type(sample_hist)}")
        logger.info(f"Keys in sample_hist: {sample_hist.axes.name}")

        to_project_setting_val = to_project_setting.copy()
        logger.debug(f"to_project_setting_val: {to_project_setting_val}")
        to_project_setting_val["val_sumw2"] = "value"
        logger.debug(f"to_project_setting_val: {to_project_setting_val}")
        hist_val = sample_hist[to_project_setting_val].view()
        # ------------------------------------------------------
        to_project_setting_w2 = to_project_setting.copy()
        to_project_setting_w2["val_sumw2"] = "sumw2"
        hist_w2 = sample_hist[to_project_setting_w2].view()
        logger.info(f"to_project_setting: {to_project_setting}")
        logger.info(f"hist_val {group_name}: {hist_val}")
        logger.info(f"hist_w2 {group_name}: {hist_w2}")
        if np.sum(hist_val)==0:
            logger.info(f"Empty hist from {group_name}. Skipping!")
            continue
        hist_dict = {
            "hist_arr" : hist_val,
            "hist_w2_arr": hist_w2
        }

        if "data" in group_name:
            if region_name != "h-peak":
                data_dict = hist_dict
            else: # keep data blinded
                data_dict = {key: np.zeros_like(value) for key, value in hist_dict.items()}
        elif "ggH" in group_name or "VBF" in group_name: # signal
            sig_MC_dict[group_name] = hist_dict
        else: # bkg MC
            bkg_MC_dict[group_name] = hist_dict
    # order bkg_MC_dict in a specific way for plotting, smallest yielding process first:
    bkg_MC_order = ["VVV", "VV", "Ewk", "Top", "DY","DYJ01", "DYJ2"]
    bkg_MC_dict = {process: bkg_MC_dict[process] for process in bkg_MC_order if process in bkg_MC_dict}
    logger.info(f"data_dict : {data_dict}")
    logger.info(f"bkg_MC_dict : {bkg_MC_dict}")
    logger.info(f"sig_MC_dict : {sig_MC_dict}")

    # -------------------------------------------------------
    # All data are prepped, now plot Data/MC histogram
    # -------------------------------------------------------
    # full_save_path = args.save_path+f"/{args.year}/mplhep/Reg_{region_name}/Cat_{args.category}/{args.label}"
    # logger.info(f"full_save_path: {full_save_path}")

    if not os.path.exists(full_save_path):
        os.makedirs(full_save_path)
    # tag = "Run2_nanoAODv12_AK8jets"
    dnn_tag = "HPScan_03Sep_17bins_08Oct"  # FIXME
    full_save_fname = f"{full_save_path}/{var}_{region_name}_{dnn_tag}.pdf"
    logger.info(f"full_save_fname: {full_save_fname}")
    # raise ValueError

    if binning is None:
        binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])

    plotDataMC_compare(
        binning,
        data_dict,
        bkg_MC_dict,
        full_save_fname.replace(".pdf", "_log.pdf"),
        sig_MC_dict=sig_MC_dict,
        title = "",
        x_title = plot_settings[plot_var].get("xlabel"),
        y_title = plot_settings[plot_var].get("ylabel"),
        lumi = lumi,
        status = status,
        log_scale = do_logscale,
    )
    plotDataMC_compare(
        binning,
        data_dict,
        bkg_MC_dict,
        full_save_fname,
        sig_MC_dict=sig_MC_dict,
        title = "",
        x_title = plot_settings[plot_var].get("xlabel"),
        y_title = plot_settings[plot_var].get("ylabel"),
        lumi = lumi,
        status = status,
        log_scale = False,
    )


def getPickledHist_byFname(pickled_filelist, load_path):
    return_dict = {}
    for fname in pickled_filelist:
        with open(fname, "rb") as f:
            hist = pickle.load(f)
        key_name = fname.replace(f"{load_path}", "").replace("_hist.pkl", "")
        # logger.info(f"key_name: {key_name}")
        return_dict[key_name] = hist

    return return_dict

def arrangeHist_bySampleGroup(pickled_hist_dict):
    sample_group_dict = {  # keys are sample group names and values are string indicators
        "data": ["data"],
        "ggH": ["ggh_"],
        "VBF": ["vbf_"],
        "DY": ["dy_M-100To200_MiNNLO", "dy_M-50_MiNNLO", "dy_VBF_filter"],
        # "DYJ01": ["DYJ01"],
        # "DYJ2": ["DYJ2"],
        "Top": ["ttjets", "top", "st"],
        "Ewk": ["ewk"],
        "VV": ["ww_", "wz_"],
        # "VV": ["ww_", "wz_", "zz"],
        "VVV": ["www_", "wwz", "wzz", "zzz"],
    }

    hist_bySampleGroup = {sample_group: [] for sample_group in sample_group_dict.keys()}
    for hist_name, hist_instance in pickled_hist_dict.items():
        # loop over hist_name and add them to the appropriate sample group
        for sample_group, name_indicators in sample_group_dict.items():
            for name_indicator in name_indicators:
                if name_indicator in hist_name:
                    hist_bySampleGroup[sample_group].append(hist_instance)
                    continue

    for sample_group, hist_l in hist_bySampleGroup.items():
        logger.info(f"{sample_group}, len:{len(hist_l)}")
        # check hist_l number of bins
        for i, h in enumerate(hist_l):
            logger.warning(f"  {i} : {h.axes.name}, bins: {[getattr(axis, 'edges', 'None') for axis in h.axes]}")
    return hist_bySampleGroup

def getPlotVar(var: str):
    """
    Helper function that removes the variations in variable name if they exist
    """
    if "_nominal" in var:
        plot_var = var.replace("_nominal", "")
    else:
        plot_var = var
    return plot_var

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-label",
        "--label",
        dest="label",
        default="Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar",
        action="store",
        help="label",
    )
    parser.add_argument(
    "-cat",
    "--category",
    dest="category",
    default="vbf",
    action="store",
    help="string value production category we're working on",
    )
    parser.add_argument(
        "-save",
        "--save_path",
        dest="save_path",
        default="validation/from_stage2/",
        action="store",
        help="string value production category we're working on",
    )
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="2018",
    action="store",
    help="label",
    )
    parser.add_argument(
    "--mva_name",
    dest="mva_name",
    default="Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar",
    action="store",
    help="label",
    )
    parser.add_argument(
    "-reg",
    "--region",
    dest="region",
    default="signal",
    action="store",
    help="region value to plot, available regions are: h_peak, h_sidebands, z_peak and signal (h_peak OR h_sidebands)",
    )
    args = parser.parse_args()
    year = args.year
    if year == "run2":
        year_param = "*"
    elif year == "2016":
        year_param = "2016*"
    else:
        year_param = year
    # load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean/{args.label}/stage2_histograms/score_{args.mva_name}/{year_param}/"
    # load_path = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_17July/stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_17July/2018_h-peak_vbf_2018_UpdatedQGL_17July_Test/2018/" # FIXME
    # load_path = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_17July/stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_17July_Test/2018/"
    # load_path = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_17July/stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_17July_July31_Rebinned/2018/"
    # load_path = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/{args.label}/stage2_histograms/score_{args.mva_name}/2018_h-peak_vbf_2018_UpdatedQGL_17July_Test/{year}/" # FIXME
    # load_path = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_July31_Rebinned/2018/"
    # load_path = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_July31_Rebinned_NoSyst/2018/"

    # load_path = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_July31_Rebinnedv2_NoSyst/*/"
    # load_path = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_July31_Rebinnedv2_NoSyst/2018/"
    # load_path = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_July31_Rebinnedv2_NoSyst/2017/"
    # load_path = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_July31_Rebinnedv2_NoSyst/2016postVFP/"
    # load_path = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_July31_Rebinnedv2_NoSyst/2016preVFP/"

    # load_path = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_FixPUJetIDWgt_NoSyst/2018/"

    # load_path = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_FixPUJetIDWgt_Rebinned_NoSyst/2018/"
    # load_path = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_FixPUJetIDWgt_Rebinned_NoSyst/2017/"
    # load_path = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_FixPUJetIDWgt_Rebinned_NoSyst/2016postVFP/"
    # load_path = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_FixPUJetIDWgt_Rebinned_NoSyst/2016preVFP/"
    # load_path = f"/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_HPScan_03Sep_21bins/2018/"

    # Path with FatJet variables
    # load_path = f"/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/{args.label}/stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_HPScan_03Sep_17bins_NoSyst/{year_param}/"
    load_path = f"/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/{args.label}/stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_HPScan_03Sep_17bins/{year_param}/"

    logger.info(f"Looking for pickled histograms in: {load_path}")

    pickled_filelist = glob.glob(f"{load_path}/*.pkl")
    logger.info(f"load_path : {load_path}")
    # logger.info(f"pickled_hists : {pickled_filelist}")

    pickled_hist_dict = getPickledHist_byFname(pickled_filelist, load_path)
    logger.info(f"pickled_hist_dict.keys() : {pickled_hist_dict.keys()}")
    hist_dict_bySampleGroup = arrangeHist_bySampleGroup(pickled_hist_dict)
    logger.info(f"hist_dict_bySampleGroup.keys() : {hist_dict_bySampleGroup.keys()}")

    lumi_dict = {
        "2018" : 59.83,
        "2017" : 41.48,
        "2016postVFP": 19.50,
        "2016preVFP": 16.81,
        "2016": 36.3,
        "run2": 137,
    }
    lumi_val = lumi_dict[year]

    possible_samples = ["data", "ggh", "vbf", "dy", "ewk", "tt", "st", "ww", "wz", "zz","VVV"]

    plot_setting_fname = "src/lib/histogram/plot_settings_vbfCat_MVA_input.json"
    with open(plot_setting_fname, "r") as file:
        plot_settings = json.load(file)
    # logger.info(f"plot_settings: {plot_settings}")
    binning = selection.binning
    var = "DNN_score"
    region_name = args.region
    category = args.category
    label = args.label
    full_save_path = f"{args.save_path}/{args.year}/Reg_{region_name}/Cat_{category}/{label}/"
    plotStage2DNN_score(
        hist_dict_bySampleGroup,
        var,
        plot_settings,
        full_save_path,
        region_name,
        category,
        do_logscale=True,
        binning=binning,
        lumi=lumi_val,
        status="Private",
    )
