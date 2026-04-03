import argparse
import copy
import glob
import json
import logging
import os
import time
import sys

import awkward as ak
import dask
import dask_awkward as dak
import hist.dask as hda
import numpy as np
import tqdm

from cli.common_argparser import build_common_parser
from modules.dask_utils import close_dask_client, get_dask_client
from modules import selection
from modules.utils import logger
from src.lib.histogram.plotting import plotDataMC_compare
from modules.classify_year import is_run2, is_run3
from configs.variables.variable_lists import get_all_vars
from scripts.compact_parquet_data import ensure_compacted

# This order is for the stack plotting in the control plots
# bkg_MC_order = ["OTHER", "VV", "EWK",  "TOP", "DY", "DYVBF","DY_MINNLO", "DY_AMCATNLO", "DY_combined", "DYJ01", "DYJ2"]
bkg_MC_order = ["VV", "EWK",  "TOP", "DY"]


group_dict = {
    "DATA": {
        "2016preVFP": ["data_B", "data_C", "data_D", "data_E", "data_F"],
        "2016postVFP": ["data_F", "data_G", "data_H"],
        "2016": ["data_B", "data_C", "data_D", "data_E", "data_F", "data_G", "data_H"],
        "2017": ["data_B", "data_C", "data_D", "data_E", "data_F"],
        "2018": ["data_A", "data_B", "data_C", "data_D"],
        "run2": ["data_A", "data_B", "data_C", "data_D", "data_E", "data_F", "data_G", "data_H"],

        "2022preEE": ["data_C", "data_D"],
        "2022postEE": ["data_E", "data_F", "data_G"],
        "2023": ["data_C"],
        "2023BPix": ["data_D"],
        "2024": ["data_C", "data_D", "data_E", "data_F", "data_G", "data_H", "data_I"],
        "run3": ["data_C", "data_D", "data_E", "data_F", "data_G", "data_H", "data_I"],
    },
    "DY": {
        "2022preEE": ["dyTo2L_M-50_incl"],
        "2022postEE": ["dyTo2L_M-50_incl"],
        "2023": ["dyTo2L_M-50_incl"],
        "2023BPix": ["dyTo2L_M-50_incl"],
        "2024": ["dyTo2Mu_M-50_aMCatNLO"],

        # "2022preEE": ["dyTo2Mu_MLL_10To50", "dyTo2Mu_MLL_50To120", "dyTo2Mu_MLL_120To200"],
        # "2022postEE": ["dyTo2Mu_MLL_10To50", "dyTo2Mu_MLL_50To120", "dyTo2Mu_MLL_120To200"],        
        # "2023": ["dyTo2Mu_MLL_10To50", "dyTo2Mu_MLL_50To120", "dyTo2Mu_MLL_120To200"],
        # "2023BPix": ["dyTo2Mu_MLL_10To50", "dyTo2Mu_MLL_50To120", "dyTo2Mu_MLL_120To200"],
        # "2024": ["dyTo2Mu_MLL_10To50", "dyTo2Mu_MLL_50To120", "dyTo2Mu_MLL_120To200"],
    },
    "EWK": {
        "2022preEE": ["ewk_mmjj_mll_105_160"],
        "2022postEE": ["ewk_mmjj_mll_105_160"],
        "2023": ["ewk_mmjj_mll_105_160"],
        "2023BPix": ["ewk_mmjj_mll_105_160"],
        "2024": ["ewk_mmjj_mll_105_160"],
    },
    "TOP": [
        # "tt_inclusive",
        "ttjets_dl",
        "ttjets_sl",
        # "ttjets_fh",
        # "st_tw_top",
        # "st_tw_antitop",
        # "st_t_top",
        # "st_t_antitop",
    ],
    "VV": [
        "ww_2l2nu",
        "wz_3lnu",
        "wz_2l2q",
        "wz_1l1nu2q",
        "zz_2l2q",
        "zz_2l2u",
        "zz_2l2nu",
        "zz_4l",
    ],
    # "OTHER": ["www", "wwz", "wzz", "zzz"],
    "ggH": ["ggh_powhegPS"],
    "VBF": {
        "2022preEE": ["vbf_powheg_dipole"],
        "2022postEE": ["vbf_powheg_dipole"],
        "2023": ["vbf_powheg"],
        "2023BPix": ["vbf_powheg"],
        "2024": ["vbf_powheg"],
    },
}

def parseGroupProcesses(group_dict, year: str):
    """
    helper function that simplifies group_dict to be
    specific to one year.
    """
    year_specific_group_dict = {}
    for group_name, processes in group_dict.items():
        logger.debug(f"Group '{group_name}' processes (original): {processes}")
        if type(processes) is dict:
            processes = processes[year]
        year_specific_group_dict[group_name] = processes
    logger.debug(f"Group dict specific to year {year}: {year_specific_group_dict}")
    return year_specific_group_dict

def find_group_name(process_name, group_dict_param):
    # Avoid redefining group_dict from outer scope
    for group_name, processes in group_dict_param.items():
        if process_name in processes:
            return group_name
    return "other"


def fillHist(sample_hist, var, to_fill_setting, values, weights):
    values_filter = values!=-999.0
    values = values[values_filter]
    weights = weights[values_filter]
    to_fill_setting[var] = values
    to_fill_value = to_fill_setting.copy()
    to_fill_value["val_sumw2"] = "value"
    sample_hist.fill(**to_fill_value, weight=weights)

    to_fill_sumw2 = to_fill_setting.copy()
    to_fill_sumw2["val_sumw2"] = "sumw2"
    sample_hist.fill(**to_fill_sumw2, weight=weights * weights)
    return sample_hist


def getPlotVar(var_param: str):
    """
    Helper function that removes the variations in variable name if they exist
    """
    if "_nominal" in var_param:
        plot_var = var_param.replace("_nominal", "")
    else:
        plot_var = var_param
    return plot_var


if __name__ == "__main__":
    parser = build_common_parser()
    parser.add_argument(
        "-bkgorder",
        "--background_order",
        dest="background_samples",
        # default=["OTHER", "EWK", "VV", "TOP", "DY", "DYVBF", "DYJ01", "DYJ2"],
        default=["EWK", "VV", "TOP", "DY"],
        nargs="*",
        type=str,
        action="store",
        help="list of bkg samples represented by shorthands: DY, TT, ST, DB (diboson), EWK",
    )
    parser.add_argument(
    "-var",
    "--variables",
    dest="variables",
    # default=["dimuon", "mu"],
    # default=["dijet", "jet"],
    # default=["dimuon", "dijet", "jet", "mu"],
    default=["dimuon", "dijet", "jet"],
    nargs="*",
    type=str,
    action="store",
    help="list of variables to plot (ie: jet, mu, dimuon)",
    )
    parser.add_argument(
    "-min",
    "--minimum_set",
    dest="minimum_set",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="If true, plots minimum set of variables for validation",
    )
    parser.add_argument(
    "-load",
    "--load_path",
    dest="load_path",
    default="/depot/cms/users/yun79/results/stage1/test_full/f0_1",
    action="store",
    help="load path",
    )
    parser.add_argument(
    "-lumi",
    "--lumi",
    dest="lumi",
    default="",
    action="store",
    help="string value of integrated luminosity to label",
    )
    parser.add_argument(
    "--status",
    dest="status",
    default="Preliminary",
    action="store",
    help="Status of results ie Private, Preliminary, In Progress",
    )
    parser.add_argument(
    "--no_ratio",
    dest="no_ratio",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="doesn't plot Data/MC ratio",
    )
    parser.add_argument(
    "--linear_scale",
    dest="linear_scale",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="If true, provide plots in linear scale",
    )
    parser.add_argument(
    "-reg",
    "--region",
    dest="regions",
    default=[ "h-sidebands", "z-peak", "signal", "h-peak" ],
    # default=["signal", "h-peak" ],
    nargs="*",
    type=str,
    action="store",
    help="region value to plot, available regions are: h_peak, h_sidebands, z_peak and signal (h_peak OR h_sidebands)",
    )
    parser.add_argument(
    "-cat",
    "--category",
    dest="category",
    default="nocat",
    action="store",
    help="define production mode category. optionsare ggh, vbf and nocat (no category cut)",
    )
    parser.add_argument(
    "--vbf_filter_study",
    dest="do_vbf_filter_study",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="If true, apply vbf filter cut for dy samples",
    )
    parser.add_argument(
    "--remove_zpt_weights",
    dest="remove_zpt_weights",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="If true, remove z-pt weights from the events",
    )
    parser.add_argument(
        "--use_dnn_zpt_weights",
        dest="use_dnn_zpt_weights",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, use DNN-based z-pt weights for the events",
    )
    parser.add_argument(
        "--njets",
        dest="njets",
        choices=["inclusive", "0", "1", "2"],
        default="inclusive",
        help="jet multiplicity selection: 'inclusive' or exactly '0', '1', or '2'",
    )
    parser.add_argument(
        "--jj-eta-region",
        dest="jj_eta_region",
        default="all",
        choices=[
            "all",
            "jj_both_central",
            "jj_non_central",
            "jj_one_fwd25_one_central",
            "jj_one_he_one_central",
            "jj_one_fwd30_one_central",
            "jj_both_fwd25",
            "jj_both_he",
            "jj_both_fwd30",
            "jj_one_he_one_fwd30",
        ],
        help=(
            "Select dijet eta topology using jet1_eta/jet2_eta. "
            "'central' = |eta|<2.5, 'he' = 2.5<|eta|<3.0, "
            "'fwd25' = |eta|>2.5, 'fwd30' = |eta|>3.0. Default: all"
        ),
    )    
    # add dnn score to the plotting variable list
    parser.add_argument(
     "--dnn-score",
     dest="dnn_score",
     default=False,
     action=argparse.BooleanOptionalAction,
     help="If true, include DNN score in the plots",
    )
    parser.add_argument(
     "--addVars",
     dest="addVars",
     default=False,
     action=argparse.BooleanOptionalAction,
     help="If true, add additional variables to the plots",
    )
    parser.add_argument(
        "--use-compacted",
        dest="use_compacted",
        default="",
        type=str,
       help="Path to the compacted parquet files"
    )

    # ---------------------------------------------------------
    # gather arguments
    args = parser.parse_args()
    logger.setLevel(args.log_level)
    logger.info(f"args: {args}")
    logger.info(f"region: {args.regions}")

    group_dict = parseGroupProcesses(group_dict, args.year)

    if is_run3(args.year):
        CM_energy = 13.6  # TeV
    elif is_run2(args.year):
        CM_energy = 13.0  # TeV
    else:
        raise ValueError(f"Unsupported year: {args.year}")

    if args.lumi == "":
        # read lumi value from configs/parameters/lumi.yaml
        infile_lumi = os.path.join("configs", "parameters", "lumi.yaml")
        import yaml
        with open(infile_lumi, "r") as f:
            lumi_config = yaml.safe_load(f)
        lumi_dict = lumi_config.get("integrated_lumis", {})
        args.lumi = lumi_dict.get(args.year, 0.0)
        # convert from pb to fb
        args.lumi = round(args.lumi / 1000.0, 1)
        if args.lumi == 0.0:
            logger.error(f"lumi for year {args.year} is not defined!")
            raise ValueError(f"lumi for year {args.year} is not defined!")

    # FIXME: Later try to get the lumi from the copperhead_processor get_sample_info function
    # from modules.get_sample_info import get_sample_info
    # sample_info = get_sample_info("./configs/datasets/dataset_nanoAODv12_run3.yaml", dataset, year) # FIXME: hardcoded filename
    # integrated_lumi = sample_info["total_lumi_pb"]
    logger.warning(f"lumi: {args.lumi}")

    # if cat is vbf and njet is < 2 then skip the program
    # if args.category.lower() == "vbf" and (args.njets == "0" or args.njets == "1"):
    #     logger.error("VBF category requires at least 2 jets! Exiting the program.")
    #     raise ValueError("VBF category requires at least 2 jets!")

    if (args.do_vbf_filter_study):
        #  Remove the "z-peak" region from the args.regions if it exists
        if "z-peak" in args.regions:
            logger.info("Removing z-peak region from the regions!")
            args.regions.remove("z-peak")
        else:
            logger.warning("z-peak region is not in the regions, nothing to remove!")
    # else:
    #     # Remove the "DYVBF" group from the group_dict if it exists
    #     if "DYVBF" in group_dict:
    #         logger.info("Removing DYVBF from the group_dict!")
    #         del group_dict["DYVBF"]
    #     else:
    #         logger.warning("DYVBF is not in the group_dict, nothing to remove!")

    # If the args.regions is empty, exit the program
    if len(args.regions) == 0:
        logger.error("No regions specified! Exiting the program.")
        raise ValueError("No regions specified!")

    available_processes = []

    logger.info("group_dict: {group_dict}".format(group_dict=group_dict))
    # take data
    data_samples = args.data_samples
    if len(data_samples) > 0:
        for data_letter in data_samples:
            available_processes.append(f"data_{data_letter.upper()}")

    # take bkg
    background_samples = args.background_samples
    if len(background_samples) > 0:
        for bkg_sample in background_samples:
            bkg_sample_upper = bkg_sample.upper()
            if bkg_sample_upper in group_dict:
                available_processes.extend(group_dict[bkg_sample_upper])
            else:
                logger.warning(f"unknown background {bkg_sample} was given!")

    # take sig
    sig_samples = args.sig_samples
    if len(sig_samples) > 0:
        for sig_sample in sig_samples:
            sig_sample_upper = sig_sample
            if sig_sample_upper in group_dict:
                available_processes.extend(group_dict[sig_sample_upper])
            else:
                logger.warning(f"unknown signal {sig_sample} was given!")

    logger.info(f"available_processes: {available_processes}")
    # gather variables to plot:
    kinematic_vars = ['pt', 'eta', 'phi']
    if args.minimum_set: kinematic_vars = ['pt', 'eta']
    variables2plot = get_all_vars(args.minimum_set)  # get the full list of variables from the config file

    variables2plot_orig = copy.deepcopy(variables2plot)
    if "jj_mass_nominal" in variables2plot:
        variables2plot += ["jj_mass_nominal_range2"] # add another range to plot
    if "dimuon_mass" in variables2plot:
        variables2plot += ["dimuon_mass_zpeak"] # add another range to plot
    logger.info(f"variables2plot: {variables2plot}")
    # sys.exit()
    # obtain plot settings from config file

    if args.category == "ggh":
        plot_setting_fname = "./src/lib/histogram/plot_settings_gghCat_BDT_input.json"
    else: # in no cat case, just use vbfCat plot settings
        plot_setting_fname = "./src/lib/histogram/plot_settings_vbfCat_MVA_input.json"

    logger.debug(f"plot_setting_fname: {plot_setting_fname}")

    with open(plot_setting_fname, "r") as file:
        plot_settings = json.load(file)
    status = args.status.replace("_", " ")

    # define client for parallelization
    client = get_dask_client(args.use_gateway, cluster_index=args.cluster_index)
    logger.info(f"client: {client}")

    # record time
    time_step = time.time()

    # check if the compacted path exists
    if args.use_compacted != "":
        # path name should contain the string "f1_0" which is the default load path, otherwise throw error
        if "f1_0" not in args.load_path:
            raise ValueError("The load path should contain the string 'f1_0' to use the compacted path! Exiting the program.")

        # run compact script for each process
        for process in available_processes:
            compacted_path_DNN = os.path.join(args.load_path, process, "0")
            ensure_compacted(args.year, process, args.load_path, compacted_path_DNN)

        args.load_path = (args.load_path).replace("f1_0", args.use_compacted)

    logger.info(f"Using parquet files from {args.load_path}")
    # load saved parquet files. This increases memory use, but increases runtime significantly
    loaded_events = {} # intialize dictionary containing all the arrays
    for process in tqdm.tqdm(available_processes):
        full_load_path = (args.load_path+f"/{process}/*/*.parquet").replace("//", "/")
        logger.info(f"length of files: {len(glob.glob(full_load_path))}")
        logger.info(f"full_load_path: {full_load_path}")
        try:
            # FIXME: add the filter and selection while loading the parquet file
            events = dak.from_parquet(full_load_path)
            # target_chunksize = 250_000
            # events = events.repartition(rows_per_partition=target_chunksize)
        except Exception:
            logger.warning("full_load_path: %s Not available. Skipping", full_load_path)
            continue
        logger.debug(f"events.fields: {events.fields}")

        # ------------------------------------------------------
        # select only needed variables to load to save run time
        # ------------------------------------------------------

        fields2load = variables2plot_orig + [
            "wgt_nominal",
            "nBtagLoose_nominal",
            "nBtagMedium_nominal",
            "njets_nominal",
            "dimuon_mass",
            "zeppenfeld_nominal",
            "jj_mass_nominal",
            "jet1_pt_nominal",
            "jj_dEta_nominal",
            "dimuon_pt",
            "jet2_pt_nominal",
            "jj_pt_nominal",
            "zeppenfeld_nominal",
        ]

        is_data = "data" in process.lower()
        if not is_data: # MC sample
            # fields2load += ["gjj_mass", "gjj_dR", "gjet1_pt", "gjet2_pt"]
            fields2load += ["gjj_mass"]
            if "separate_wgt_zpt_wgt" in events.fields and args.remove_zpt_weights:
                logger.debug("Append separate_wgt_zpt_wgt to fields2load!")
                fields2load.append("separate_wgt_zpt_wgt")

            if (
                "zpt_wgt_reco_dnn" in events.fields
                and "separate_wgt_zpt_wgt" in events.fields
                and args.use_dnn_zpt_weights
            ):
                logger.debug("Append separate_wgt_zpt_wgt and zpt_wgt_reco_dnn to fields2load!")
                fields2load.append("separate_wgt_zpt_wgt")
                fields2load.append("zpt_wgt_reco_dnn")
        # filter out redundant fields by using the set object
        fields2load = list(set(fields2load))
        logger.debug(f"fields2load: {fields2load}")

        # check if all fields to load are in the events
        # fields_in_events = events.fields
        # for field in fields2load:
        #     if field not in fields_in_events:
        #         logger.warning(f"field {field} not in events, removing from fields2load!")

        # # TOREMOVE
        # if "separate_wgt_qgl_wgt" in events.fields:
        #     logger.info("removing separate_wgt_qgl_wgt!")
        #     events["wgt_nominal"] = events["wgt_nominal"] / events["separate_wgt_qgl_wgt"] # remove zpt wgt
        if "separate_wgt_zpt_wgt" in events.fields and args.remove_zpt_weights:
            logger.warning("removing separate_wgt_zpt_wgt!")
            events["wgt_nominal"] = events["wgt_nominal"] / events["separate_wgt_zpt_wgt"] # remove zpt wgt

        if (
            "separate_wgt_zpt_wgt" in events.fields
            and "zpt_wgt_reco_dnn" in events.fields
            and args.use_dnn_zpt_weights
            ):
            logger.warning("removing separate_wgt_zpt_wgt and applying zpt_wgt_reco_dnn!")
            events["wgt_nominal"] = events["wgt_nominal"] / events["separate_wgt_zpt_wgt"] # remove zpt wgt
            events["wgt_nominal"] = events["wgt_nominal"] * events["zpt_wgt_reco_dnn"] # apply the weights obtained from the DNN
        # if "dy" in process.lower():
        #     # scale the weights for DY samples by 3.0
        #     logger.warning("Scaling DY weights by 3.0 after removing zpt weights!")
        #     events["wgt_nominal"] = events["wgt_nominal"] * (1997.0/2124.08)

        loaded_events[process] = events
    logger.info("finished loading parquet files!")
    # mplhep style starts here --------------------------------------
    logger.info("Using mplhep style for plotting!")
    import matplotlib
    import matplotlib.pyplot as plt
    import mplhep as hep
    # hep.style.use("CMS")
    # Load CMS style including color-scheme (it's an editable dict)
    plt.style.use(hep.style.CMS)
    # this mplhep implementation assumes non-empty data; otherwise, it will crash
    # Dictionary for histograms and binnings

    # initialize histograms
    # FIXME: Is it mandatory to use all regions and channels name below? Or I can just replace it with args.regions and args.category?
    regions = ["z-peak", "signal", "h-peak", "h-sidebands"] # full list of possible regions to loop over
    channels = ["nocat", "vbf", "ggh"] # full list of possible channels to loop over
    variations = ["nominal"]
    sample_groups = list(group_dict.keys()) + ["other"]
    logger.info(f"sample_groups: {sample_groups}")
    sample_hist = (
            hda.Hist.new.StrCat(regions, name="region")
            .StrCat(channels, name="channel")
            .StrCat(["value", "sumw2"], name="val_sumw2")
            .StrCat(sample_groups, name="sample_group")
    )
    # add axis for systematic variation
    sample_hist_dictByVar = {}
    sample_hist = sample_hist.StrCat(variations, name="variation")

    # Initialize histograms for each variable to be plotted.
    logger.info("{style}Initializing histograms for each variable to be plotted.{style}".format(
        style="\n" + "="*50 + "\n",))
    for var in tqdm.tqdm(variables2plot):
        # for process in available_processes:
        if "_nominal" in var:
            plot_var = var.replace("_nominal", "")
        else:
            plot_var = var
        if plot_var not in plot_settings.keys():
            logger.warning(f"variable {var} not configured in plot settings!")
            continue
        if var == "dnn_vbf_score_atanh":
            # custom non-uniform bin edges from validation plot
            # binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
            binning = np.array(plot_settings[plot_var]["binning_nonuniform"])
        elif var == "dnn_vbf_score":
            # binning = np.array(plot_settings[plot_var]["binning_nonuniform"])
            binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
            logger.warning(f"Using non-uniform binning for {var} variable!")
            logger.warning(f"binning: {binning}")
        else:
            binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
        # if region_name == "z-peak" and plot_var == "dimuon_mass": # When z-peak region is selected, use different binning for mass
        # binning = np.linspace(*plot_settings[var]["binning_zpeak_linspace"])
        logger.debug(f"var: {var}")
        sample_hist_dictByVar[var] = sample_hist.Var(binning, name=var).Double()

    # fill the histograms
    logger.info("{style}Filling histograms for each variable.{style}".format(
        style="\n" + "="*50 + "\n",))
    sample_hist_dictByVar2compute = {}
    for var in tqdm.tqdm(variables2plot):
        sample_hist_empty = sample_hist_dictByVar[var]
        sample_hist_l = []
        var_step = time.time()
        # for process in available_processes:
        if "_nominal" in var:
            plot_var = var.replace("_nominal", "")
        else:
            plot_var = var
        if plot_var not in plot_settings.keys():
            logger.warning(f"variable {var} not configured in plot settings!")
            continue
        # -----------------------------------------------
        # intialize variables for filling histograms
        if var == "dnn_vbf_score_atanh":
            # binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
            binning = np.array(plot_settings[plot_var]["binning_nonuniform"])
        elif var == "dnn_vbf_score":
            # binning = np.array(plot_settings[plot_var]["binning_nonuniform"])
            binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
            logger.warning(f"Using non-uniform binning for {var} variable!")
            logger.warning(f"binning: {binning}")
        else:
            binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
        # if region_name == "z-peak" and plot_var == "dimuon_mass": # When z-peak region is selected, use different binning for mass
        # binning = np.linspace(*plot_settings[var]["binning_zpeak_linspace"])
        if args.linear_scale:
            do_logscale = False
        else:
            do_logscale = True
        # also check if logscale config is mentioned in plot_settings, if yes, that config takes priority
        # if "logscale" in plot_settings[plot_var].keys():
        #     do_logscale = plot_settings[plot_var]["logscale"]
        logger.debug(f"do_logscale: {do_logscale} ")

        for process in available_processes:
            sample_hist = copy.deepcopy(sample_hist_empty)
            logger.debug(f"process: {process}")
            # logger.debug(f"sample_hist: {sample_hist}")
            logger.debug(f"regions: {args.regions}")
            for region_name in args.regions:
                # for each process make new hist
                try:
                    events = loaded_events[process]
                except:
                    logger.debug(f"skipping {process}")
                    continue
                is_data = "data" in process.lower()
                logger.debug(f"is_data: {is_data}")

                # -----------------------------------------------
                # obtain the category selection
                # ------------------------------------------------
                # take the mass region and category cuts
                # ------------------------------------------------
                events = dak.map_partitions(
                    selection.applyRegionCatCuts,
                    events,
                    args.category,
                    region_name,
                    process,
                    "nominal",
                    args.do_vbf_filter_study,
                    jj_eta_region=args.jj_eta_region,
                    njets_selection=str(args.njets),
                )

                #  FOR DEBUG PURPOSES
                # if process == "dy_M-100To200_aMCatNLO":
                #     wgt_nominal = events.wgt_nominal
                #     logger.info(f"wgt_nominal = {wgt_nominal[0]}")
                #     wgt_sum = ak.sum(wgt_nominal).compute()
                #     logger.info(f"wgt_sum = {wgt_sum}")
                #     raise ValueError("Terminate the program.")

                # extract weights
                if is_data:
                    weights = (ak.fill_none(events["wgt_nominal"], value=0.0))
                    fraction_weight = 1/events.fraction
                else: # MC
                    weights = ak.fill_none(events["wgt_nominal"], value=0.0)

                    # To stich the DY aMC@NLO and MiNNLO samples, we need to divide the weight of MiNNLO sample by Luminosity (59830.0)
                    # if "dy_M-100To200_MiNNLO" in process or "dy_M-50_MiNNLO" in process :
                    # weights = weights / 59830.0 # FIXME: this is hardcoded value, should be replaced with lumi value from config file

                    # weights = weights/events.wgt_nominal_muID/ events.wgt_nominal_muIso / events.wgt_nominal_muTrig #  quick test
                    # temporary over write
                    # logger.info(f"events.fields: {events.fields}")
                    if "separate_wgt_zpt_wgt" in events.fields and args.remove_zpt_weights:
                        logger.debug("removing Zpt rewgt!")
                        weights = weights/events["separate_wgt_zpt_wgt"]

                    if (
                        "separate_wgt_zpt_wgt" in events.fields
                        and "zpt_wgt_reco_dnn" in events.fields
                        and args.use_dnn_zpt_weights
                    ):
                        logger.debug("removing separate_wgt_zpt_wgt and applying zpt_wgt_reco_dnn!")
                        weights = weights / events["separate_wgt_zpt_wgt"]
                        weights = weights * events["zpt_wgt_reco_dnn"]  # apply the weights obtained from the DNN

                    # for some reason, some nan weights are still passes ak.fill_none() bc they're "nan", not None, this used to be not a problem
                    # could be an issue of copying bunching of parquet files from one directory to another, but not exactly sure
                    # weights = np.nan_to_num(weights, nan=0.0)
                    fraction_weight = ak.ones_like(events["wgt_nominal"])  # MC is already normalized by lumisonity, so no need for scaling by fraction

                # handle arctanh transform of dnn_vbf_score
                # if var == "dnn_vbf_score_atanh":
                #     raw = ak.fill_none(events["dnn_vbf_score"], value=-999.0)
                #     values = np.arctanh((raw))  # arctanh transform
                # values = np.arctanh((raw+1)/2.0)  # arctanh transform
                # overwrite variable names with two bin ranges
                if ("_range2" in var):
                    var_reduced = var.replace("_range2","")
                    values = ak.fill_none(events[var_reduced], value=-999.0)
                elif ("_zpeak" in var):
                    var_reduced = var.replace("_zpeak","")
                    values = ak.fill_none(events[var_reduced], value=-999.0)
                else:
                    values = ak.fill_none(events[var], value=-999.0)

                #### TODO: Add overflow bins to the last bin

                # MC samples are already normalized by their xsec*lumi, but data is not
                if process in group_dict["DATA"]: # FIXME: Why weights with data?
                    logger.debug(f"{process} is in data processes")
                    weights = weights*fraction_weight
                group_name = find_group_name(process, group_dict)
                to_fill_setting = {
                "region" : region_name,
                "channel" : args.category,
                "variation" : "nominal",
                "sample_group": group_name,
                }
                sample_hist = fillHist(sample_hist, var, to_fill_setting, values, weights)

            sample_hist_l.append(sample_hist)

        sample_hist_dictByVar2compute[var] = sample_hist_l

    # logger.debug(f"sample_hist_dictByVar2compute: {sample_hist_dictByVar2compute}")

    # done with looping over process and variables we now compute
    logger.info("{style}Computing histograms.{style}".format(
        style="\n" + "="*50 + "\n",))
    logger.debug(f"sample_groups: {sample_groups}")
    logger.debug(f"variables2plot: {variables2plot}")
    sample_hist_dictByVarComputed = dask.compute(sample_hist_dictByVar2compute)[0]
    for region_name in args.regions:
        for var in tqdm.tqdm(variables2plot):
            data_dict = {}
            bkg_MC_dict = {}
            sig_MC_dict = {}
            for group_name in sample_groups:
                sample_hist_l = sample_hist_dictByVarComputed[var]
                sample_hist = sum(sample_hist_l)
                to_project_setting = {
                    "region" : region_name,
                    "channel" : args.category,
                    "variation" : "nominal",
                    "sample_group": group_name,
                }

                to_project_setting_val = to_project_setting.copy()
                logger.debug(f"to_project_setting_val: {to_project_setting_val}")
                logger.debug(f"sample_hist: {sample_hist}")
                logger.debug(f"sample_hist_l: {sample_hist_l}")

                to_project_setting_val["val_sumw2"] = "value"
                hist_val = sample_hist[to_project_setting_val].project(var).values()
                # ------------------------------------------------------
                to_project_setting_w2 = to_project_setting.copy()
                to_project_setting_w2["val_sumw2"] = "sumw2"
                hist_w2 = sample_hist[to_project_setting_w2].project(var).values()
                if np.sum(hist_val)==0: # skip processes that doesn't have anything
                    logger.debug(f"hist_val is empty for {group_name} in {var}, skipping!")
                    continue
                hist_dict = {
                    "hist_arr" : hist_val,
                    "hist_w2_arr": hist_w2
                }

                logger.debug(f"group_name: {group_name}\t hist_dict: {hist_dict}")
                if "DATA" in group_name: # data
                    if region_name != "h-peak":
                        data_dict = hist_dict
                    else: # keep data blinded
                        data_dict = {key: np.zeros_like(value) for key, value in hist_dict.items()}
                elif "ggH" == group_name or "VBF" == group_name: # signal
                    sig_MC_dict[group_name] = hist_dict
                else: # bkg MC
                    bkg_MC_dict[group_name] = hist_dict
            # order bkg_MC_dict in a specific way for plotting, smallest yielding process first:
            logger.debug(f"bkg_MC_order: {bkg_MC_order}")
            logger.debug(f"bkg_MC_dict: {bkg_MC_dict}")
            bkg_MC_dict = {process: bkg_MC_dict[process] for process in bkg_MC_order if process in bkg_MC_dict}
            logger.debug(f"data_dict: {data_dict}")
            logger.debug(f"bkg_MC_dict: {bkg_MC_dict}")
            if len(data_dict) ==0:
                logger.warning(f"empty histograms for {var} skipping!")
                continue

            # if sampels DY_MINNLO (D1) or DY_AMCATNLO(D2) are in the bkg_MC_dict, then merge them using formula
            # content_combined = (Content_D1/(Sigma_D1)^2 + Content_D2/(Sigma_D2)^2) / (1/(Sigma_D1)^2 + 1/(Sigma_D2)^2)
            if "DY_MINNLO" in bkg_MC_dict and "DY_AMCATNLO" in bkg_MC_dict:
                logger.info("Merging DY MINNLO and AMCATNLO samples!")
                hist_D1 = bkg_MC_dict["DY_MINNLO"]["hist_arr"]
                hist_D2 = bkg_MC_dict["DY_AMCATNLO"]["hist_arr"]
                hist_D1_w2 = bkg_MC_dict["DY_MINNLO"]["hist_w2_arr"]  # This is variance per bin
                hist_D2_w2 = bkg_MC_dict["DY_AMCATNLO"]["hist_w2_arr"]

                # Avoid division by zero: set variance to inf (weight 0) where either is zero
                valid = (hist_D1_w2 > 0) & (hist_D2_w2 > 0)

                combined_content = np.zeros_like(hist_D1)
                combined_w2 = np.zeros_like(hist_D1)

                # Weighted average and variance where both have entries
                combined_content[valid] = (
                    hist_D1[valid] / hist_D1_w2[valid] + hist_D2[valid] / hist_D2_w2[valid]
                ) / (1.0 / hist_D1_w2[valid] + 1.0 / hist_D2_w2[valid])
                combined_w2[valid] = 1.0 / (1.0 / hist_D1_w2[valid] + 1.0 / hist_D2_w2[valid])

                # Use single sample where only one has entries
                only1 = (hist_D1_w2 > 0) & (hist_D2_w2 == 0)
                only2 = (hist_D2_w2 > 0) & (hist_D1_w2 == 0)
                combined_content[only1] = hist_D1[only1]
                combined_w2[only1] = hist_D1_w2[only1]
                combined_content[only2] = hist_D2[only2]
                combined_w2[only2] = hist_D2_w2[only2]

                bkg_MC_dict["DY_combined"] = {
                    "hist_arr": combined_content,
                    "hist_w2_arr": combined_w2
                }
                bkg_MC_dict["DY"] = bkg_MC_dict.pop("DY_combined")

                # remove old samples
                del bkg_MC_dict["DY_MINNLO"]
                del bkg_MC_dict["DY_AMCATNLO"]

            logger.debug(f"bkg_MC_dict: {bkg_MC_dict}")
            # ---------------------------------------------------
            # All data are prepped, now plot Data/MC histogram
            # -------------------------------------------------------
            # if args.remove_zpt_weights, then update the args.label
            zpt_postfix = "default_zpt_weights"
            if args.remove_zpt_weights:
                logger.debug("Removing zpt weights from the events!")
                zpt_postfix = "no_zpt_weights"
            if args.use_dnn_zpt_weights:
                logger.warning("Using DNN-based zpt weights for the events!")
                zpt_postfix = "dnn_zpt_weights"
            if args.jj_eta_region != "all":
                zpt_postfix += f"_{args.jj_eta_region}"

            if args.year == "*":
                full_save_path = args.save_path+f"/AllYear/mplhep/Reg_{region_name}/Cat_{args.category}/njet_{args.njets}/{zpt_postfix}"
            else:
                full_save_path = args.save_path+f"/{args.year}/mplhep/Reg_{region_name}/Cat_{args.category}/njet_{args.njets}/{zpt_postfix}"
            logger.debug(f"full_save_path: {full_save_path}")

            if not os.path.exists(full_save_path):
                os.makedirs(full_save_path)
            full_save_fname = f"{full_save_path}/{var}.pdf"

            plot_var = getPlotVar(var)
            if plot_var not in plot_settings.keys():
                logger.warning(f"variable {var} not configured in plot settings!")
                continue
            binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
            if region_name == "z-peak" and plot_var == "dimuon_mass": # When z-peak region is selected, use different binning for mass
                binning = np.linspace(*plot_settings[var]["binning_zpeak_linspace"])
            # elif var == "dnn_vbf_score":
            #     # binning = np.array(plot_settings[var]["binning_nonuniform"])
            #     binning = np.linspace(*plot_settings[var]["binning_zpeak_linspace"])
            elif var == "dnn_vbf_score_atanh":
                binning = np.array(plot_settings[var]["binning_nonuniform"])

            plotDataMC_compare(
                binning,
                data_dict,
                bkg_MC_dict,
                full_save_fname.replace(".pdf", "_log.pdf"),
                sig_MC_dict=sig_MC_dict,
                title = "",
                x_title = plot_settings[plot_var].get("xlabel"),
                y_title = plot_settings[plot_var].get("ylabel"),
                lumi = args.lumi,
                status = status,
                log_scale = do_logscale,
                CenterOfMass = CM_energy,
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
                lumi = args.lumi,
                status = status,
                log_scale = False,
                CenterOfMass=CM_energy,
                plot_ratio_range = "default", # options: "default" or "auto" or list with format [0.8, 1.2]
            )

    close_dask_client()
    logger.info("Plots are saved to %s", full_save_path)
    time_elapsed = round(time.time() - time_step, 3)
    logger.info(f"Finished in {time_elapsed} s.")
