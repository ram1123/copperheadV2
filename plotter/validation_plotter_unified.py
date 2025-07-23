import sys
import awkward as ak
import dask_awkward as dak
import numpy as np
import json
import argparse
import os
from src.lib.histogram.ROOT_utils import setTDRStyle, CMS_lumi
from src.lib.histogram.plotting import plotDataMC_compare
from distributed import Client
import time
import tqdm
import glob
import copy
import hist.dask as hda
import dask
import logging
from modules.utils import logger

from scripts.compact_parquet_data import ensure_compacted

# This order is for the stack plotting in the control plots
# bkg_MC_order = ["AddTop", "OTHER", "EWK", "VVContinuum", "VV", "TOP", "DY", "DYVBF"]
# bkg_MC_order = ["AddTop", "OTHER", "EWK", "VVContinuum", "VV", "TOP", "DY"]
# bkg_MC_order = ["OTHER", "EWK", "VV", "TOP", "DY", "DYVBF"]
bkg_MC_order = ["OTHER", "EWK", "VV", "TOP", "DY", "DYVBF","DY_MINNLO", "DY_AMCATNLO", "DY_combined"]
# bkg_MC_order = ["OTHER", "EWK", "VV", "TOP", "DY"]

DY_aMCatNLO = ["dy_M-100To200_aMCatNLO", "dy_M-50_aMCatNLO"]
# DY_aMCatNLO = ["dy_M-100To200_aMCatNLO"]

DY_MiNNLO = ["dy_M-100To200_MiNNLO", "dy_M-50_MiNNLO"]

DY_HTBinned = [
    "dy_M-4to50_HT-70to100", "dy_M-4to50_HT-100to200", "dy_M-4to50_HT-200to400", "dy_M-4to50_HT-400to600", "dy_M-4to50_HT-600toInf",
    "dy_M-50_HT-70to100", "dy_M-50_HT-100to200", "dy_M-50_HT-200to400", "dy_M-50_HT-400to600", "dy_M-50_HT-600to800", "dy_M-50_HT-800to1200", "dy_M-50_HT-1200to2500", "dy_M-50_HT-2500toInf"
]

DYVBF = ["dy_VBF_filter"]


group_dict = {
    "DATA": ["data_A", "data_B", "data_C", "data_D", "data_E",  "data_F", "data_G", "data_H"],

    # "DY": DY_aMCatNLO,
    "DY": DY_MiNNLO,
    # "DY_MINNLO": DY_MiNNLO ,
    # "DY_AMCATNLO":   DY_aMCatNLO,
    "DYVBF": ["dy_VBF_filter"],

    "TOP": ["ttjets_dl", "ttjets_sl", "st_tw_top", "st_tw_antitop", "st_t_top", "st_t_antitop"],
    # "AddTop": ["st_s_lep", "TTTJ", "TTTT","TTTW", "TTWjets_LNu", "TTWJets_QQ", "TTWW", "TTZ_LLnunu", "tZq_ll"],

    "EWK": ["ewk_lljj_mll50_mjj120"],

    "VV": ["ww_2l2nu", "wz_3lnu", "wz_2l2q", "wz_1l1nu2q", "zz"],
    # "VVContinuum": ["GluGluContin_ZZ2e2mu", "GluGluContin_ZZ2mu2nu", "GluGluContin_ZZ2mu2tau", "GluGluContin_ZZ4mu", "GluGluContin_ZZ4tau"],

    "OTHER": ["www", "wwz", "wzz", "zzz"],
    "GGH": ["ggh_powhegPS"],
    "VBF": ["vbf_powheg_dipole"]
}

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


def applyRegionCatCuts(events, category: str, region_name: str, njets: str, process: str, do_vbf_filter_study: bool):
    # do mass region cut
    mass = events.dimuon_mass
    z_peak = ((mass > 70) & (mass < 110))
    h_sidebands =  ((mass > 110) & (mass < 115.03)) | ((mass > 135.03) & (mass < 150))
    h_peak = ((mass > 115.03) & (mass < 135.03))
    if region_name == "signal":
        region = h_sidebands | h_peak
    elif region_name == "h-peak":
        region = h_peak
    elif region_name == "h-sidebands":
        region = h_sidebands
    elif region_name == "z-peak":
        region = z_peak
    else:
        print(f"ERROR: acceptable region names are: z-peak, h-sidebands, h-peak, signal. Got {region_name} instead!")
        raise ValueError

    prod_cat_cut =  ak.ones_like(region, dtype="bool")

    # FIXME: add cut to veto pileup jets: pT < 50 GeV in 2.5 < | eta(j) | 4.0
    # pileup_jet_veto = ((events.jet1_pt_nominal < 50) & (abs(events.jet1_eta_nominal) > 2.5) & (abs(events.jet1_eta_nominal) < 4.0)) | (events.jet1_pt_nominal < 70)
    # pileup_jet_veto = ak.fill_none(pileup_jet_veto, value=False)
    # prod_cat_cut = prod_cat_cut & ~pileup_jet_veto

    # Remove jets beyong 2.5 rapidity
    # high_eta_jet_veto = (abs(events.jet1_eta_nominal) > 2.5) | (abs(events.jet2_eta_nominal) > 2.5)
    # high_eta_jet_veto = ak.fill_none(high_eta_jet_veto, value=False)
    # prod_cat_cut = prod_cat_cut & ~high_eta_jet_veto

    # do category cut
    if category.lower() == "nocat":
        # print("nocat mode!")
        pass
    else: # VBF or ggH
        btagLoose_filter = ak.fill_none((events.nBtagLoose_nominal >= 2), value=False)
        btagMedium_filter = ak.fill_none((events.nBtagMedium_nominal >= 1), value=False) & ak.fill_none((events.njets_nominal >= 2), value=False)
        btag_cut = btagLoose_filter | btagMedium_filter
        # vbf_cut = ak.fill_none(events.vbf_cut, value=False) # in the future none values will be replaced with False
        vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35)
        vbf_cut = ak.fill_none(vbf_cut, value=False)
        if category.lower() == "vbf":
            # print("vbf mode!")
            prod_cat_cut =  prod_cat_cut & vbf_cut
            prod_cat_cut = prod_cat_cut & ~btag_cut # btag cut is for VH and ttH categories
            if do_vbf_filter_study and process.startswith("dy_"):
                """
                Apply VBF filter, generator level di-jet invariant mass cut of 350 GeV
                This is for the stiching the inclusive and VBF DY samples.
                For inclusive DY samples, we apply the cut of < 350 GeV
                For VBF DY samples, we apply the cut of >= 350 GeV
                NOTE: For the inclusive DY category, we apply the cut (gjj_mass > 350 GeV)
                and then invert it to select inclusive DY events.
                This approach is necessary due to the behavior of NaN values when applying
                comparison operators in awkward arrays:

                - If we use (gjj_mass < 350 GeV), any event where gjj_mass is NaN will result in False,
                so those events will be excluded from the selection.
                Example:
                    - Event 1: gjj_mass = 250 GeV  --> (250 < 350) = True  (selected)
                    - Event 2: gjj_mass = NaN      --> (NaN < 350) = False (not selected)

                - If we use (gjj_mass > 350 GeV), events with NaN will also result in False,
                but if we then invert the selection (~), those NaN events will be included:
                Example:
                    - Event 1: gjj_mass = 250 GeV  --> (250 > 350) = False --> ~False = True  (selected)
                    - Event 2: gjj_mass = NaN      --> (NaN > 350) = False --> ~False = True (selected)

                Therefore, by using (gjj_mass > 350 GeV) and inverting the mask, we ensure that both
                events with gjj_mass < 350 GeV and events with NaN values are included in the inclusive DY selection.
                This guarantees that no events are lost due to NaN values in gjj_mass.
                """
                if ("dy_VBF_filter" in process):
                    logger.warning(f"Apply VBF filter gen cut > 350 for VBF DY!: process = {process}")
                    vbf_filter = ak.fill_none((events.gjj_mass > 350), value=False)
                    # logger.debug(f"{process}: events before filter = {ak.num(events, axis=0).compute()}")
                    # logger.debug(f"{process}: events after filter = {ak.sum(vbf_filter).compute()}")

                    prod_cat_cut =  (prod_cat_cut
                                & vbf_filter
                    )
                elif (process == "dy_M-100To200_MiNNLO" or
                        process == "dy_M-50_MiNNLO" or
                        process == "dy_M-100To200_aMCatNLO" or
                        process == "dy_M-50_aMCatNLO"):
                    logger.warning(f"Apply inverted VBF filter gen cut > 350 for inc. DY!: process = {process}")

                    vbf_filter = ak.fill_none((events.gjj_mass > 350), value=False)
                    # logger.debug(f"{process}: events before filter = {ak.num(events, axis=0).compute()}")
                    # logger.debug(f"{process}: events after filter = {ak.sum(vbf_filter).compute()}")

                    prod_cat_cut =  (
                        prod_cat_cut
                        & ~vbf_filter
                    )
                else:
                    logger.warning(f"no extra processing for {process}")
                    pass

        elif category.lower() == "ggh":
            # print("ggH mode!")
            prod_cat_cut =  prod_cat_cut & ~vbf_cut
            prod_cat_cut = prod_cat_cut & ~btag_cut # btag cut is for VH and ttH categories
        else:
            print("Error: invalid category option!")
            raise ValueError

    # add njets cut
    if njets != "inclusive":
        if njets == "0":
            njets_cut = (events.njets_nominal == 0)
        elif njets == "1":
            njets_cut = (events.njets_nominal == 1)
        elif njets == "2":
            njets_cut = (events.njets_nominal >= 2)
        else:
            logger.error(f"ERROR: njets value {njets} is not supported! Use 0, 1 or 2.")
            raise ValueError
        njets_cut = ak.fill_none(njets_cut, value=False)
        prod_cat_cut = prod_cat_cut & njets_cut

    category_selection = (
        prod_cat_cut &
        region
    )

    events = events[category_selection]
    return events


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="2018",
    action="store",
    help="string value of year we are calculating",
    )
    parser.add_argument(
    "-data",
    "--data",
    dest="data_samples",
    default=["A", "B", "C", "D", "E", "F", "G", "H"],
    # default=["A"],
    nargs="*",
    type=str,
    action="store",
    help="list of data samples represented by alphabetical letters A-H",
    )
    parser.add_argument(
    "-bkg",
    "--background",
    dest="bkg_samples",
    # default=["DY", "TOP", "EWK", "VV", "OTHER"],
    # default = ["AddTop", "OTHER", "EWK", "VVContinuum", "VV", "TOP", "DY"],
    # default = ["OTHER", "EWK", "VV", "TOP", "DY", "DYVBF"],
    # default = ["OTHER", "EWK", "VV", "DY", "DYVBF"],
    # default = ["OTHER", "EWK", "VV",  "TOP", "DY", "DY_MiNNLO", "DY_aMCatNLO"],
    default = ["OTHER", "EWK", "VV", "TOP", "DY", "DYVBF"],
    # default = ["OTHER", "EWK", "VV", "TOP", "DY" ],
    # default = ["AddTop", "OTHER", "EWK", "VVContinuum", "VV", "TOP", "DY", "DYVBF"],
    nargs="*",
    type=str,
    action="store",
    help="list of bkg samples represented by shorthands: DY, TT, ST, DB (diboson), EWK",
    )
    parser.add_argument(
    "-sig",
    "--signal",
    dest="sig_samples",
    default=["VBF","GGH"],
    # default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of sig samples represented by shorthands: ggH, VBF",
    )
    parser.add_argument(
    "-var",
    "--variables",
    dest="variables",
    # default=["dimuon", "mu"],
    # default=["dijet", "jet"],
    default=["dimuon", "dijet", "jet", "mu"],
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
    "-label",
    "--label",
    dest="label",
    default="",
    action="store",
    help="label",
    )
    parser.add_argument(
    "-save",
    "--save_path",
    dest="save_path",
    default="./validation/figs/",
    action="store",
    help="save path",
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
    default="",
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
    "--use_gateway",
    dest="use_gateway",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="If true, uses dask gateway client instead of local",
    )
    # parser.add_argument(
    # "--vbf",
    # dest="vbf_cat_mode",
    # default=False,
    # action=argparse.BooleanOptionalAction,
    # help="If true, apply vbf cut for vbf category, else, ggH category cut",
    # )
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
        "--njets",
        dest="njets",
        choices=["inclusive", "0", "1", "2"],
        default="inclusive",
        help="jet multiplicity selection: 'inclusive' or exactly '0', '1', or '2'",
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
     "--log-level",
     default=logging.INFO,
     type=lambda x: getattr(logging, x),
     help="Configure the logging level."
     )

    #---------------------------------------------------------
    # gather arguments
    args = parser.parse_args()
    logger.setLevel(args.log_level)
    logger.info(f"args: {args}")
    logger.info(f"region: {args.regions}")

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

    # if args.remove_zpt_weights, then update the args.label
    if args.remove_zpt_weights:
        if args.label == "":
            args.label = "no_zpt_weights"
        else:
            args.label += "_no_zpt_weights"


    available_processes = []

    logger.info("group_dict: {group_dict}".format(group_dict=group_dict))
    # take data
    data_samples = args.data_samples
    if len(data_samples) > 0:
        for data_letter in data_samples:
            available_processes.append(f"data_{data_letter.upper()}")

    # take bkg
    bkg_samples = args.bkg_samples
    if len(bkg_samples) > 0:
        for bkg_sample in bkg_samples:
            bkg_sample_upper = bkg_sample.upper()
            if bkg_sample_upper in group_dict:
                available_processes.extend(group_dict[bkg_sample_upper])
            else:
                logger.warning(f"unknown background {bkg_sample} was given!")

    # take sig
    sig_samples = args.sig_samples
    if len(sig_samples) > 0:
        for sig_sample in sig_samples:
            sig_sample_upper = sig_sample.upper()
            if sig_sample_upper in group_dict:
                available_processes.extend(group_dict[sig_sample_upper])
            else:
                logger.warning(f"unknown signal {sig_sample} was given!")

    logger.info(f"available_processes: {available_processes}")
    # gather variables to plot:
    kinematic_vars = ['pt', 'eta', 'phi']
    if args.minimum_set: kinematic_vars = ['pt']
    variables2plot = []
    if args.dnn_score:
        variables2plot.append("dnn_vbf_score")
        variables2plot.append("atanh_dnn_vbf_score")
    if len(args.variables) == 0:
        logger.error("no variables to plot!")
        raise ValueError
    if args.minimum_set: args.variables = ["dimuon", "mu"] # if minimum set is requested, only plot dimuon, and mu variables
    for particle in args.variables:
        if "dimuon" in particle:
            variables2plot.append(f"{particle}_mass")
            variables2plot.append(f"{particle}_pt")
            variables2plot.append(f"{particle}_eta")
            if args.minimum_set: # if minimum set is requested, only plot pt and mass
                continue
            variables2plot.append(f"{particle}_phi")
            variables2plot.append(f"{particle}_cos_theta_cs")
            variables2plot.append(f"{particle}_phi_cs")
            variables2plot.append(f"{particle}_cos_theta_eta")
            variables2plot.append(f"{particle}_phi_eta")
            variables2plot.append(f"mmj_min_dPhi_nominal")
            variables2plot.append(f"mmj_min_dEta_nominal")
            variables2plot.append(f"ll_zstar_log_nominal")
            variables2plot.append(f"dimuon_ebe_mass_res")
            variables2plot.append(f"dimuon_ebe_mass_res_rel")
            variables2plot.append(f"{particle}_rapidity")
            variables2plot.append("MET_pt")
            variables2plot.append("MET_phi")
            variables2plot.append("MET_sumEt")
            variables2plot.append("acoplanarity")
            variables2plot.append("PV_npvs")
            variables2plot.append("PV_npvsGood")
        elif "dijet" in particle:
            variables2plot.append(f"jj_dEta_nominal")
            variables2plot.append(f"jj_mass_nominal")
            variables2plot.append(f"jj_pt_nominal")
            variables2plot.append(f"jj_dPhi_nominal")
            variables2plot.append(f"zeppenfeld_nominal")
            variables2plot.append(f"rpt_nominal")
            variables2plot.append(f"pt_centrality_nominal")
            variables2plot.append(f"nsoftjets2_nominal")
            variables2plot.append(f"htsoft2_nominal")
            variables2plot.append(f"nsoftjets5_nominal")
            variables2plot.append(f"htsoft5_nominal")

            # --------------------------------------------------
            # variables2plot.append(f"gjj_mass")

        elif ("mu" in particle) :
            for kinematic in kinematic_vars:
                # plot both leading and subleading muons/jets
                variables2plot.append(f"{particle}1_{kinematic}")
                variables2plot.append(f"{particle}2_{kinematic}")
            if not args.minimum_set: # if minimum set is requested, only plot pt and mass
                variables2plot.append(f"{particle}1_pt_over_mass")
                variables2plot.append(f"{particle}2_pt_over_mass")
        elif ("jet" in particle):
            variables2plot.append(f"njets_nominal")
            for kinematic in kinematic_vars:
                # plot both leading and subleading muons/jets
                variables2plot.append(f"{particle}1_{kinematic}_nominal")
                variables2plot.append(f"{particle}2_{kinematic}_nominal")
            variables2plot.append(f"jet1_qgl_nominal")
            variables2plot.append(f"jet2_qgl_nominal")

        else:
            logger.warning(f"Unsupported variable: {particle} is given!")


    variables2plot_orig = copy.deepcopy(variables2plot)
    if "jj_mass_nominal" in variables2plot:
        variables2plot += ["jj_mass_nominal_range2"] # add another range to plot
    logger.info(f"variables2plot: {variables2plot}")
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
    if args.use_gateway:
        from dask_gateway import Gateway
        gateway = Gateway(
            "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
            proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
        )
        cluster_info = gateway.list_clusters()[0]  # get the first cluster by default. There only should be one anyways
        client = gateway.connect(cluster_info.name).get_client()
        logger.info("Gateway Client created")
    else:
        client = Client(n_workers=64, threads_per_worker=1, processes=True, memory_limit='10 GiB')
        logger.info("Local scale Client created")
    # record time
    time_step = time.time()

    # check if the compacted path exists
    COMPACTED_PATH = (args.load_path).replace("f1_0", "compacted")

    for process in available_processes:
        compacted_path_DNN = os.path.join(COMPACTED_PATH, process, "0")
        ensure_compacted(args.year, process, args.load_path, compacted_path_DNN)
    args.load_path = COMPACTED_PATH


    # load saved parquet files. This increases memory use, but increases runtime significantly
    loaded_events = {} # intialize dictionary containing all the arrays
    for process in tqdm.tqdm(available_processes):
        full_load_path = args.load_path+f"/{process}/*/*.parquet"
        if len(glob.glob(full_load_path)) ==0: # check if there's files in the load path
            full_load_path = args.load_path+f"/{process}/*.parquet" # try coppperheadV1 path, if this also is empty, then skip
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

        # filter out redundant fields by using the set object
        fields2load = list(set(fields2load))
        logger.info(f"fields2load: {fields2load}")

        # # TOREMOVE
        # if "separate_wgt_qgl_wgt" in events.fields:
        #     logger.info("removing separate_wgt_qgl_wgt!")
        #     events["wgt_nominal"] = events["wgt_nominal"] / events["separate_wgt_qgl_wgt"] # remove zpt wgt
        if "separate_wgt_zpt_wgt" in events.fields and args.remove_zpt_weights:
            logger.warning("removing separate_wgt_zpt_wgt!")
            events["wgt_nominal"] = events["wgt_nominal"] / events["separate_wgt_zpt_wgt"] # remove zpt wgt

        loaded_events[process] = events
    logger.info("finished loading parquet files!")
    # mplhep style starts here --------------------------------------
    logger.info("Using mplhep style for plotting!")
    import mplhep as hep
    import matplotlib.pyplot as plt
    import matplotlib
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
        if var == "atanh_dnn_vbf_score":
            # custom non-uniform bin edges from validation plot
            binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
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
        #-----------------------------------------------
        # intialize variables for filling histograms
        if var == "atanh_dnn_vbf_score":
            binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
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

                #-----------------------------------------------
                # obtain the category selection
                # ------------------------------------------------
                # take the mass region and category cuts
                # ------------------------------------------------
                events = dak.map_partitions(applyRegionCatCuts,events, args.category, region_name, args.njets, process, args.do_vbf_filter_study)

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
                        logger.info("removing Zpt rewgt!")
                        weights = weights/events["separate_wgt_zpt_wgt"]

                    # for some reason, some nan weights are still passes ak.fill_none() bc they're "nan", not None, this used to be not a problem
                    # could be an issue of copying bunching of parquet files from one directory to another, but not exactly sure
                    # weights = np.nan_to_num(weights, nan=0.0)
                    fraction_weight = ak.ones_like(events["wgt_nominal"])  # MC is already normalized by lumisonity, so no need for scaling by fraction

                # handle arctanh transform of dnn_vbf_score
                if var == "atanh_dnn_vbf_score":
                    raw = ak.fill_none(events["dnn_vbf_score"], value=-999.0)
                    values = np.arctanh((raw))  # arctanh transform
                    # values = np.arctanh((raw+1)/2.0)  # arctanh transform
                # overwrite variable names with two bin ranges
                elif ("_range2" in var):
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
                #------------------------------------------------------
                to_project_setting_w2 = to_project_setting.copy()
                to_project_setting_w2["val_sumw2"] = "sumw2"
                hist_w2 = sample_hist[to_project_setting_w2].project(var).values()
                if np.sum(hist_val)==0: # skip processes that doesn't have anything
                    logger.warning(f"hist_val is empty for {group_name} in {var}, skipping!")
                    continue
                hist_dict = {
                    "hist_arr" : hist_val,
                    "hist_w2_arr": hist_w2
                }

                logger.debug(f"group_name: {group_name}\t hist_dict: {hist_dict}")
                if "DATA" in group_name: # data
                    data_dict = hist_dict
                elif "GGH" == group_name or "VBF" == group_name: # signal
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
            full_save_path = args.save_path+f"/{args.year}/mplhep/Reg_{region_name}/Cat_{args.category}/njet_{args.njets}/{args.label}"
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
            # elif var == "atanh_dnn_vbf_score":
            #     binning = np.linspace(*plot_settings[var]["binning_linspace"])
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
            )
    time_elapsed = round(time.time() - time_step, 3)
    logger.info(f"Finished in {time_elapsed} s.")
