import awkward as ak
import dask_awkward as dak
import dask
import argparse
import sys
import os
import numpy as np
import json
import yaml
from collections import OrderedDict
from modules.selection import filterRegion
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
        elif group_name in {"ggH", "VBF"}: # signal
            sig_MC_dict[group_name] = hist_dict
        else: # bkg MC
            bkg_MC_dict[group_name] = hist_dict
    # order bkg_MC_dict in a specific way for plotting, smallest yielding process first:
    bkg_MC_order = ["VVV", "VV", "Ewk", "Top", "DYVBF", "DY","DYJ01", "DYJ2"]
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
    dnn_tag = plot_var
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
        plot_ratio_range = "default", # options: "default" or "auto" or list with format [0.8, 1.2]
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
        plot_ratio_range = "default", # options: "default" or "auto" or list with format [0.8, 1.2]
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

def load_group_process_indicators(sample_config_path):
    """
    Build {plot_group_name: [process_name, ...]} indicators from a
    samples.yaml-style config (see configs/samples/samples.yaml), instead of
    a hardcoded list that silently drops any process added/renamed later
    (e.g. MiNNLO DY: dyTo2Mu_M-50_MiNNLO, dyTo2Mu_M-100to200_MiNNLO).

    Indicators are the union of a group's default `processes` and every
    year's `processes_per_year` override, across ALL years in the file --
    not just one requested year -- since a stage2 histogram directory can
    mix multiple years (e.g. --year run2 globs 2016preVFP/2016postVFP/2017/
    2018 together) and a process name means the same sample in any year.
    """
    with open(sample_config_path, "r") as f:
        cfg = yaml.safe_load(f)

    def _union_group_processes(section):
        groups = (cfg.get(section) or {}).get("groups", {}) or {}
        out = {}
        for group_name, gcfg in groups.items():
            procs = set(gcfg.get("processes") or [])
            for year_procs in (gcfg.get("processes_per_year") or {}).values():
                procs.update(year_procs or [])
            out[group_name] = sorted(procs)
        return out

    bkg = _union_group_processes("background")
    sig = _union_group_processes("signal")

    return {
        # "data" isn't an MC sample in samples.yaml -- stage2 combines all
        # data-era files into one "data" histogram, so this stays literal.
        "data": ["data"],
        "ggH": sig.get("GGH", []),
        "VBF": sig.get("VBF", []),
        "DYVBF": bkg.get("DYVBF", []),
        "DY": bkg.get("DY", []),
        "Top": bkg.get("TT", []) + bkg.get("ST", []),
        "Ewk": bkg.get("EWK", []),
        "VV": bkg.get("VV", []),
        "VVV": bkg.get("VVV", []),
    }


def arrangeHist_bySampleGroup(pickled_hist_dict, sample_group_dict):
    """
    sample_group_dict: {plot_group_name: [process_name, ...]}, as built by
    load_group_process_indicators().
    """
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
    "--load",
    dest="load_path",
    default="Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar",
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
    "--vbf_filter_study",
    dest="do_vbf_filter_study",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Enable DY vs DY-VBF-filter study mode for grouping and output naming.",
    )
    parser.add_argument(
        "--sample-config",
        dest="sample_config",
        default="configs/samples/samples.yaml",
        help="Path to the sample configuration YAML file (same one run_stage2_vbf.py uses to resolve MC process groups).",
    )
    parser.add_argument(
    "-reg",
    "--region",
    dest="region",
    default="signal",
    action="store",
    help="region value to plot, available regions are: h_peak, h_sidebands, z_peak and signal (h_peak OR h_sidebands)",
    )
    parser.add_argument(
        "--log-level",
        default=logging.INFO,
        type=lambda x: getattr(logging, x),
        help="Configure the logging level.",
    )    
    args = parser.parse_args()

    logger.setLevel(args.log_level)
    
    year = args.year
    if year == "run2":
        year_param = "*"
    elif year == "2016":
        year_param = "2016*"
    else:
        year_param = year

    load_path = (
        f"{args.load_path}"
        f"/{year_param}"
    )

    logger.info(f"Looking for pickled histograms in: {load_path}")

    pickled_filelist = glob.glob(f"{load_path}/*.pkl")
    logger.info(f"load_path : {load_path}")
    # logger.info(f"pickled_hists : {pickled_filelist}")

    pickled_hist_dict = getPickledHist_byFname(pickled_filelist, load_path)
    logger.info(f"pickled_hist_dict.keys() : {pickled_hist_dict.keys()}")
    sample_group_dict = load_group_process_indicators(args.sample_config)
    logger.info(f"sample_group_dict (from {args.sample_config}) : {sample_group_dict}")
    hist_dict_bySampleGroup = arrangeHist_bySampleGroup(pickled_hist_dict, sample_group_dict)
    logger.info(f"hist_dict_bySampleGroup.keys() : {hist_dict_bySampleGroup.keys()}")

    # read lumi value from configs/parameters/lumi.yaml
    infile_lumi = os.path.join("configs", "parameters", "lumi.yaml")
    with open(infile_lumi, "r") as f:
        lumi_config = yaml.safe_load(f)
    lumi_dict = lumi_config.get("integrated_lumis", {})
    lumi = lumi_dict.get(year, 0.0)
    # convert from pb to fb
    lumi = round(lumi / 1000.0, 1)
    if lumi == 0.0:
        logger.error(f"lumi for year {year} is not defined!")
        raise ValueError(f"lumi for year {year} is not defined!")

    lumi_val = lumi

    plot_setting_fname = "src/lib/histogram/plot_settings_vbfCat_MVA_input.json"
    with open(plot_setting_fname, "r") as file:
        plot_settings = json.load(file)
    # logger.info(f"plot_settings: {plot_settings}")
    binning = selection.binning
    var = "DNN_score"
    region_name = args.region
    category = args.category
    output_tag = args.mva_name
    if args.do_vbf_filter_study and "_vbf_filter_study" not in output_tag:
        output_tag = f"{output_tag}_vbf_filter_study"
    full_save_path = f"{args.save_path}/{args.year}/Reg_{region_name}/Cat_{category}/{output_tag}_NoVHveto/"
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
