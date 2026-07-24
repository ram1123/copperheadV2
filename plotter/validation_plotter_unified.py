import argparse
import copy
import glob
import itertools
import json
import logging
import os
import time
import sys
from pathlib import Path

import awkward as ak
import hist
import numpy as np
import tqdm
from coffea import processor
from coffea.nanoevents import BaseSchema

import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep

from cli.common_argparser import build_common_parser
from modules.dask_utils import close_dask_client, get_dask_client
from modules import selection
from modules.utils import logger
from src.lib.histogram.plotting import plotDataMC_compare
from modules.classify_year import is_run2, is_run3
from configs.variables.variable_lists import get_all_vars
from scripts.compact_parquet_data import ensure_compacted

# Load CMS style including color-scheme once, regardless of which entry point
# (standalone CLI or the bulk driver in run_plotter.py) is used.
plt.style.use(hep.style.CMS)

DATASET_SEPARATOR = "::"

# Full static category lists for the Hist axes -- these never change; which of
# their values actually get filled/plotted is controlled separately (by the
# `regions` list passed to the processor, and by which single `category` a
# caller asks for).
FULL_REGIONS = ["z-peak", "signal", "h-peak", "h-sidebands"]
FULL_CHANNELS = ["nocat", "vbf", "ggh", "bJetVeto"]
VARIATIONS = ["nominal"]

ZPT_POSTFIX_BY_OPTION = {
    "default": "default_zpt_weights",
    "no_zpt": "no_zpt_weights",
    "dnn_zpt": "dnn_zpt_weights",
}

# This order is for the stack plotting in the control plots
# bkg_MC_order = ["OTHER", "VV", "EWK",  "TOP", "DY", "DYVBF","DY_MINNLO", "DY_AMCATNLO", "DY_combined", "DYJ01", "DYJ2"]
bkg_MC_order = ["VV", "EWK",  "TOP", "DY", "DYVBF"]


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
        "2016preVFP": [
            # "dyTo2Mu_M-100to200_MiNNLO",
            # "dy_M-100To200_MiNNLO",# run2 nanoV12
            # "dy_M-50_MiNNLO", # run2 nanoV12
            # "dy_M-50_aMCatNLO", # run2 nanoV12
            # "dyTo2L_M-50_aMCatNLO", # run2 nanoV15
            "dyTo2Mu_M-50_MiNNLO", # run2 nanoV15
            "dyTo2Mu_M-100to200_MiNNLO", # run2 nanoV15
        ],
        "2016postVFP": [
            # "dyTo2Mu_M-100to200_MiNNLO",
            # "dy_M-100To200_MiNNLO",# run2 nanoV12
            # "dy_M-50_MiNNLO", # run2 nanoV12
            # "dy_M-50_aMCatNLO", # run2 nanoV12
            # "dyTo2L_M-50_aMCatNLO", # run2 nanoV15
            "dyTo2Mu_M-50_MiNNLO", # run2 nanoV15
            "dyTo2Mu_M-100to200_MiNNLO", # run2 nanoV15
        ],
        "2017": [
            # "dyTo2Mu_M-100to200_MiNNLO",
            # "dy_M-100To200_MiNNLO",# run2 nanoV12
            # "dy_M-50_MiNNLO", # run2 nanoV12
            # "dy_M-50_aMCatNLO", # run2 nanoV12
            # "dyTo2L_M-50_aMCatNLO", # run2 nanoV15
            "dyTo2Mu_M-50_MiNNLO", # run2 nanoV15
            "dyTo2Mu_M-100to200_MiNNLO", # run2 nanoV15
        ],
        "2018": [
            # "dyTo2Mu_M-100to200_MiNNLO",
            # "dy_M-100To200_MiNNLO",# run2 nanoV12
            # "dy_M-50_MiNNLO", # run2 nanoV12
            # "dy_M-50_aMCatNLO", # run2 nanoV12
            # "dyTo2L_M-50_aMCatNLO", # run2 nanoV15
            "dyTo2Mu_M-50_MiNNLO", # run2 nanoV15
            "dyTo2Mu_M-100to200_MiNNLO", # run2 nanoV15
        ],
        "2022preEE": ["dyTo2L_M-50_incl"],
        "2022postEE": ["dyTo2L_M-50_incl"],
        "2023": ["dyTo2L_M-50_incl"],
        "2023BPix": ["dyTo2L_M-50_incl"],
        "2024": ["dyTo2Mu_M-50_aMCatNLO"],

        # "2022preEE": ["dyTo2L_M-50_incl", "dy_VBF_filter"],
        # "2022postEE": ["dyTo2L_M-50_incl", "dy_VBF_filter"],
        # "2023": ["dyTo2L_M-50_incl", "dy_VBF_filter"],
        # "2023BPix": ["dyTo2L_M-50_incl", "dy_VBF_filter"],
        # "2024": ["dyTo2Mu_M-50_aMCatNLO", "dy_VBF_filter"],


        # "2022preEE": ["dyTo2Mu_MLL_10To50", "dyTo2Mu_MLL_50To120", "dyTo2Mu_MLL_120To200"],
        # "2022postEE": ["dyTo2Mu_MLL_10To50", "dyTo2Mu_MLL_50To120", "dyTo2Mu_MLL_120To200"],
        # "2023": ["dyTo2Mu_MLL_10To50", "dyTo2Mu_MLL_50To120", "dyTo2Mu_MLL_120To200"],
        # "2023BPix": ["dyTo2Mu_MLL_10To50", "dyTo2Mu_MLL_50To120", "dyTo2Mu_MLL_120To200"],
        # "2024": ["dyTo2Mu_MLL_10To50", "dyTo2Mu_MLL_50To120", "dyTo2Mu_MLL_120To200"],

        # "2022preEE": ["dyTo2Mu_MLL_10To50", "dyTo2Mu_MLL_50To120", "dyTo2Mu_MLL_120To200"],
        # "2022postEE": ["dyTo2Mu_MLL_10To50", "dyTo2Mu_MLL_50To120", "dyTo2Mu_MLL_120To200"],
        # "2023": ["dyTo2Mu_MLL_10To50", "dyTo2Mu_MLL_50To120", "dyTo2Mu_MLL_120To200"],
        # "2023BPix": ["dyTo2Mu_MLL_10To50", "dyTo2Mu_MLL_50To120", "dyTo2Mu_MLL_120To200"],
        # "2024": ["dyTo2Mu_MLL_10To50", "dyTo2Mu_MLL_50To120", "dyTo2Mu_MLL_120To200"],
    },
    "DYVBF": {
        "2016preVFP": ["dy_VBF_filter"],
        "2016postVFP": ["dy_VBF_filter"],
        "2017": ["dy_VBF_filter"],
        "2018": ["dy_VBF_filter"],
        "2022preEE": ["dy_VBF_filter"],
        "2022postEE": ["dy_VBF_filter"],
        "2023": ["dy_VBF_filter"],
        "2023BPix": ["dy_VBF_filter"],
        "2024": ["dy_VBF_filter"],
    },
    "EWK": {
        "2016preVFP": ["ewk_zlljj"],
        "2016postVFP": ["ewk_zlljj"],
        "2017": ["ewk_zlljj"],
        "2018": ["ewk_zlljj"],
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
        "2016preVFP": ["vbf_powheg_dipole"],
        "2016postVFP": ["vbf_powheg_dipole"],
        "2017": ["vbf_powheg_dipole"],
        "2018": ["vbf_powheg_dipole"],
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
            if year not in processes:
                raise KeyError(
                    f"Year '{year}' is not configured for process group '{group_name}'."
                )
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


class ValidationHistProcessor(processor.ProcessorABC):
    """
    Fill per-(category, njets, zpt_option, variable) validation histograms for one
    process (dataset), eagerly per chunk. category, njets, and zpt_option are all
    swept together in one pass over the data: category needs its own Hist template
    per binning group (ggh vs. nocat/vbf use different bin counts for some
    variables), while njets and zpt_option are plain StrCat axes since they don't
    affect binning.
    """

    def __init__(
        self,
        hist_templates_by_category,
        categories,
        njets_options,
        zpt_options,
        regions,
        do_vbf_filter_study,
        jj_eta_region,
        group_dict_by_year,
    ):
        # hist_templates_by_category[category] is already pre-filtered to
        # variables configured in that category's plot_settings (built by
        # build_hist_templates()), so it's the source of truth for which
        # variables get filled per category.
        self.hist_templates_by_category = hist_templates_by_category
        self.categories = categories
        self.njets_options = njets_options
        self.zpt_options = zpt_options
        self.regions = regions
        self.do_vbf_filter_study = do_vbf_filter_study
        self.jj_eta_region = jj_eta_region
        self.group_dict_by_year = group_dict_by_year

    def process(self, events):
        dataset_key = events.metadata["sample"]
        year = events.metadata["year"]
        is_data = "data" in dataset_key.lower()

        # Precompute the available zpt-weight-variant fields once per chunk.
        # A zpt_option missing from this map (e.g. "no_zpt" when the sample
        # lacks "separate_wgt_zpt") falls back to "wgt_nominal" below,
        # matching the silent per-sample fallback of the original per-combo
        # script.
        zpt_field_map = {"default": "wgt_nominal"}
        if not is_data:
            if "separate_wgt_zpt" in events.fields:
                events["__wgt_no_zpt"] = events["wgt_nominal"] / events["separate_wgt_zpt"]
                zpt_field_map["no_zpt"] = "__wgt_no_zpt"
                if "zpt_wgt_gen" in events.fields:
                    events["__wgt_dnn_zpt"] = events["__wgt_no_zpt"] * events["zpt_wgt_gen"]
                    zpt_field_map["dnn_zpt"] = "__wgt_dnn_zpt"

        group_name = find_group_name(dataset_key, self.group_dict_by_year[year])

        hist_by_category_var = {
            category: {var: copy.deepcopy(t) for var, t in templates.items()}
            for category, templates in self.hist_templates_by_category.items()
        }

        for category in self.categories:
            templates = self.hist_templates_by_category[category]
            for njets in self.njets_options:
                if category == "vbf" and njets != "inclusive":
                    continue

                for region_name in self.regions:
                    region_events = selection.applyRegionCatCuts(
                        events,
                        category,
                        region_name,
                        dataset_key,
                        "nominal",
                        self.do_vbf_filter_study,
                        jj_eta_region=self.jj_eta_region,
                        njets_selection=njets,
                        year=year,
                    )

                    # weights depend only on (region, zpt_option), not on var,
                    # so resolve them once per region and reuse across vars.
                    weights_by_zpt = {}
                    if is_data:
                        base_weights = ak.to_numpy(ak.fill_none(region_events["wgt_nominal"], value=0.0))
                        base_weights = base_weights * (1.0 / ak.to_numpy(region_events.fraction))
                        for zpt_option in self.zpt_options:
                            weights_by_zpt[zpt_option] = base_weights
                    else:
                        for zpt_option in self.zpt_options:
                            wgt_field = zpt_field_map.get(zpt_option, "wgt_nominal")
                            weights_by_zpt[zpt_option] = ak.to_numpy(
                                ak.fill_none(region_events[wgt_field], value=0.0)
                            )

                    for var in templates:
                        if "_range2" in var:
                            value_var = var.replace("_range2", "")
                        elif "_zpeak" in var:
                            value_var = var.replace("_zpeak", "")
                        else:
                            value_var = var

                        if value_var not in region_events.fields:
                            logger.warning(
                                f"Variable '{value_var}' not found for process '{dataset_key}' "
                                f"in region '{region_name}'. Skipping histogram fill for '{var}'."
                            )
                            continue

                        values = ak.to_numpy(ak.fill_none(region_events[value_var], value=-999.0))

                        for zpt_option in self.zpt_options:
                            to_fill_setting = {
                                "region": region_name,
                                "channel": category,
                                "variation": "nominal",
                                "sample_group": group_name,
                                "njets": njets,
                                "zpt_option": zpt_option,
                            }
                            hist_by_category_var[category][var] = fillHist(
                                hist_by_category_var[category][var],
                                var,
                                to_fill_setting,
                                values,
                                weights_by_zpt[zpt_option],
                            )

        return {
            f"{year}{DATASET_SEPARATOR}{dataset_key}": {
                "chunks": processor.value_accumulator(int, 1),
                "hist_by_category_var": hist_by_category_var,
            }
        }

    def postprocess(self, accumulator):
        return accumulator


def load_plot_settings(category: str) -> dict:
    """Resolve and load the plot_settings JSON for a given category."""
    if category == "ggh":
        plot_setting_fname = "./src/lib/histogram/plot_settings_gghCat_BDT_input.json"
    else:  # in no cat case, just use vbfCat plot settings
        plot_setting_fname = "./src/lib/histogram/plot_settings_vbfCat_MVA_input.json"
    logger.debug(f"plot_setting_fname: {plot_setting_fname}")
    with open(plot_setting_fname, "r") as file:
        return json.load(file)


def build_hist_templates(variables2plot, plot_settings, sample_groups, njets_options, zpt_options):
    """Build the empty per-variable hist.Hist templates for one category's binning group."""
    sample_hist = (
        hist.Hist.new.StrCat(FULL_REGIONS, name="region")
        .StrCat(FULL_CHANNELS, name="channel")
        .StrCat(["value", "sumw2"], name="val_sumw2")
        .StrCat(sample_groups, name="sample_group")
        .StrCat(VARIATIONS, name="variation")
        .StrCat(njets_options, name="njets")
        .StrCat(zpt_options, name="zpt_option")
    )

    hist_templates = {}
    for var in tqdm.tqdm(variables2plot):
        plot_var = getPlotVar(var)
        if plot_var not in plot_settings.keys():
            logger.warning(f"variable {var} not configured in plot settings!")
            continue
        if var == "dnn_vbf_score_atanh":
            # custom non-uniform bin edges from validation plot
            binning = np.array(plot_settings[plot_var]["binning_nonuniform"])
        elif var == "dnn_vbf_score":
            binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
            logger.warning(f"Using non-uniform binning for {var} variable!")
            logger.warning(f"binning: {binning}")
        else:
            binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
        logger.debug(f"var: {var}")
        hist_templates[var] = sample_hist.Var(binning, name=var).Double()
    return hist_templates


def resolve_year_context(
    year,
    background_samples,
    sig_samples,
    data_samples,
    do_vbf_filter_study,
    lumi_override="",
):
    """
    Resolve everything that depends on `year` but not on category/njets/zpt_option:
    the year-specific group_dict, lumi, center-of-mass energy, and the list of
    available (data + bkg + sig) process names.
    """
    group_dict_year = parseGroupProcesses(group_dict, year)

    if is_run3(year):
        CM_energy = 13.6  # TeV
    elif is_run2(year):
        CM_energy = 13.0  # TeV
    else:
        raise ValueError(f"Unsupported year: {year}")

    if lumi_override == "":
        infile_lumi = os.path.join("configs", "parameters", "lumi.yaml")
        import yaml
        with open(infile_lumi, "r") as f:
            lumi_config = yaml.safe_load(f)
        lumi_dict = lumi_config.get("integrated_lumis", {})
        lumi = lumi_dict.get(year, 0.0)
        # convert from pb to fb
        lumi = round(lumi / 1000.0, 1)
        if lumi == 0.0:
            logger.error(f"lumi for year {year} is not defined!")
            raise ValueError(f"lumi for year {year} is not defined!")
    else:
        lumi = lumi_override
    logger.warning(f"lumi: {lumi}")

    if not do_vbf_filter_study and "DYVBF" in group_dict_year:
        logger.info("Removing DYVBF from the group_dict because --vbf_filter_study was not passed.")
        del group_dict_year["DYVBF"]

    available_processes = []
    for data_letter in data_samples:
        available_processes.append(f"data_{data_letter.upper()}")

    for bkg_sample in background_samples:
        bkg_sample_upper = bkg_sample.upper()
        if bkg_sample_upper == "DYVBF" and not do_vbf_filter_study:
            logger.info("Skipping DYVBF because --vbf_filter_study was not passed.")
            continue
        if bkg_sample_upper in group_dict_year:
            available_processes.extend(group_dict_year[bkg_sample_upper])
            if (
                do_vbf_filter_study
                and bkg_sample_upper == "DY"
                and "DYVBF" in group_dict_year
            ):
                logger.info("Adding DYVBF samples because --vbf_filter_study was passed.")
                available_processes.extend(group_dict_year["DYVBF"])
        else:
            logger.warning(f"unknown background {bkg_sample} was given!")

    for sig_sample in sig_samples:
        if sig_sample in group_dict_year:
            available_processes.extend(group_dict_year[sig_sample])
        else:
            logger.warning(f"unknown signal {sig_sample} was given!")

    return {
        "lumi": lumi,
        "CM_energy": CM_energy,
        "group_dict": group_dict_year,
        "available_processes": available_processes,
    }


def build_fileset_for_year(year, load_path, available_processes, use_compacted):
    """
    Build the coffea fileset entries (one per process) for one year, bootstrapping
    the compacted parquet path via ensure_compacted() when requested. Returns a
    dict keyed by plain process name; the caller prefixes keys with the year for
    cross-year fileset merging.
    """
    load_path = str(load_path)
    if use_compacted != "":
        # path name should contain the string "f1_0" which is the default load path, otherwise throw error
        if "f1_0" not in load_path:
            raise ValueError(
                "The load path should contain the string 'f1_0' to use the compacted path! "
                "Exiting the program."
            )
        compacted_base_path = load_path.replace("f1_0", use_compacted)
        for process in available_processes:
            compacted_path_DNN = os.path.join(compacted_base_path, process, "0")
            ensure_compacted(year, process, load_path, compacted_path_DNN)
        load_path = compacted_base_path

    logger.info(f"Using parquet files from {load_path}")
    fileset = {}
    for process in tqdm.tqdm(available_processes):
        full_load_path = (load_path + f"/{process}/*/*.parquet").replace("//", "/")
        files = glob.glob(full_load_path)
        logger.info(f"length of files: {len(files)}")
        logger.info(f"full_load_path: {full_load_path}")
        if len(files) == 0:
            logger.warning("full_load_path: %s Not available. Skipping", full_load_path)
            continue
        fileset[process] = {
            "files": files,
            "treename": "Events",
            "metadata": {"year": year, "sample": process},
        }
    return fileset


def run_validation_runner(fileset, processor_instance, client, chunksize=50_000, treereduction=2):
    """Run one coffea Runner pass over the given fileset, dask-parallelized via `client`.

    treereduction is lowered from coffea's default of 20: each reduce task gathers
    up to `treereduction` still-unreduced per-chunk histogram payloads onto one
    worker and accumulates them all at once (coffea.processor.executor._reduce),
    so a smaller fan-in bounds peak per-worker memory during the final reduction
    rounds at the cost of more (lighter) reduction rounds overall.
    """
    if not fileset:
        logger.warning("No samples left to process; fileset is empty.")
        return {}
    runner = processor.Runner(
        executor=processor.DaskExecutor(client=client, treereduction=treereduction),
        schema=BaseSchema,
        format="parquet",
        chunksize=chunksize,
    )
    return runner(fileset, processor_instance=processor_instance)


def generate_combo_plots(
    year,
    category,
    njets,
    zpt_option,
    region_name,
    var,
    sample_hist_lookup,
    sample_groups,
    plot_settings,
    save_path,
    lumi,
    status,
    CM_energy,
    do_logscale,
    do_vbf_filter_study,
    jj_eta_region,
):
    """
    Project the accumulated histograms for one (year, category, njets, zpt_option,
    region, var) combo into Data/bkg-MC/sig-MC arrays and save the PDF+txt pair.
    Returns the save directory on success, or None if there was nothing to plot.
    """
    data_dict = {}
    bkg_MC_dict = {}
    sig_MC_dict = {}

    sample_hist = sample_hist_lookup[year][category][njets][var]
    if sample_hist is None:
        logger.debug(f"no histograms found for {year} {category} {njets} {var}, skipping!")
        return None

    for group_name in sample_groups:
        to_project_setting = {
            "region": region_name,
            "channel": category,
            "variation": "nominal",
            "sample_group": group_name,
            "njets": njets,
            "zpt_option": zpt_option,
        }

        to_project_setting_val = to_project_setting.copy()
        to_project_setting_val["val_sumw2"] = "value"
        hist_val = sample_hist[to_project_setting_val].project(var).values()
        to_project_setting_w2 = to_project_setting.copy()
        to_project_setting_w2["val_sumw2"] = "sumw2"
        hist_w2 = sample_hist[to_project_setting_w2].project(var).values()
        if np.sum(hist_val) == 0:  # skip processes that doesn't have anything
            logger.debug(f"hist_val is empty for {group_name} in {var}, skipping!")
            continue
        hist_dict = {
            "hist_arr": hist_val,
            "hist_w2_arr": hist_w2,
        }

        if "DATA" in group_name:  # data
            if region_name != "h-peak":
                data_dict = hist_dict
            else:  # keep data blinded
                data_dict = {key: np.zeros_like(value) for key, value in hist_dict.items()}
        elif group_name in ("ggH", "VBF"):  # signal
            sig_MC_dict[group_name] = hist_dict
        else:  # bkg MC
            bkg_MC_dict[group_name] = hist_dict

    # order bkg_MC_dict in a specific way for plotting, smallest yielding process first:
    bkg_MC_dict = {process: bkg_MC_dict[process] for process in bkg_MC_order if process in bkg_MC_dict}
    if len(data_dict) == 0:
        logger.warning(f"empty histograms for {var} skipping!")
        return None

    # if samples DY_MINNLO (D1) or DY_AMCATNLO(D2) are in the bkg_MC_dict, then merge them using formula
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
            "hist_w2_arr": combined_w2,
        }
        bkg_MC_dict["DY"] = bkg_MC_dict.pop("DY_combined")

        # remove old samples
        del bkg_MC_dict["DY_MINNLO"]
        del bkg_MC_dict["DY_AMCATNLO"]

    # ---------------------------------------------------
    # All data are prepped, now plot Data/MC histogram
    # -------------------------------------------------------
    zpt_postfix = ZPT_POSTFIX_BY_OPTION[zpt_option]
    if do_vbf_filter_study:
        zpt_postfix += "_vbf_filter_study"
    if jj_eta_region != "all":
        zpt_postfix += f"_{jj_eta_region}"

    if year == "*":
        full_save_path = f"{save_path}/AllYear/mplhep/Reg_{region_name}/Cat_{category}/njet_{njets}/{zpt_postfix}"
    else:
        full_save_path = f"{save_path}/{year}/mplhep/Reg_{region_name}/Cat_{category}/njet_{njets}/{zpt_postfix}"

    if not os.path.exists(full_save_path):
        os.makedirs(full_save_path)
    full_save_fname = f"{full_save_path}/{var}.pdf"

    plot_var = getPlotVar(var)
    if plot_var not in plot_settings.keys():
        logger.warning(f"variable {var} not configured in plot settings!")
        return None
    binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
    if region_name == "z-peak" and plot_var == "dimuon_mass":  # When z-peak region is selected, use different binning for mass
        binning = np.linspace(*plot_settings[var]["binning_zpeak_linspace"])
    elif var == "dnn_vbf_score_atanh":
        binning = np.array(plot_settings[var]["binning_nonuniform"])

    plotDataMC_compare(
        binning,
        data_dict,
        bkg_MC_dict,
        full_save_fname.replace(".pdf", "_log.pdf"),
        sig_MC_dict=sig_MC_dict,
        title="",
        x_title=plot_settings[plot_var].get("xlabel"),
        y_title=plot_settings[plot_var].get("ylabel"),
        lumi=lumi,
        status=status,
        log_scale=do_logscale,
        CenterOfMass=CM_energy,
        plot_ratio_range="default",  # options: "default" or "auto" or list with format [0.8, 1.2]
    )
    plotDataMC_compare(
        binning,
        data_dict,
        bkg_MC_dict,
        full_save_fname,
        sig_MC_dict=sig_MC_dict,
        title="",
        x_title=plot_settings[plot_var].get("xlabel"),
        y_title=plot_settings[plot_var].get("ylabel"),
        lumi=lumi,
        status=status,
        log_scale=False,
        CenterOfMass=CM_energy,
        plot_ratio_range="default",  # options: "default" or "auto" or list with format [0.8, 1.2]
    )
    return full_save_path


def _derive_zpt_options(remove_zpt_weights_options, add_dnn_zpt_weights_options):
    """Map (remove_zpt_weights, use_dnn_zpt_weights) boolean-list config into the
    "default"/"no_zpt"/"dnn_zpt" enum ValidationHistProcessor expects."""
    zpt_options = []
    if False in remove_zpt_weights_options and False in add_dnn_zpt_weights_options:
        zpt_options.append("default")
    if True in remove_zpt_weights_options:
        zpt_options.append("no_zpt")
    if True in add_dnn_zpt_weights_options:
        zpt_options.append("dnn_zpt")
    if not zpt_options:
        raise ValueError("No valid zpt_option combination derived from the given config.")
    return zpt_options


def _run_validation_scope(
    jj_eta_region,
    do_vbf_filter_study,
    region_list,
    client,
    years,
    categories,
    njets_options,
    remove_zpt_weights_options,
    add_dnn_zpt_weights_options,
    min_set_of_vars,
    load_path_template,
    save_root,
    background_samples,
    sig_samples,
    data_samples,
    status,
    linear_scale,
    use_compacted,
    dry_run,
):
    """
    Run one consolidated Dask pass (year x category x njets x zpt_option, all
    computed together) for a single fixed (jj_eta_region, vbf_filter_study,
    region_list) scope. `client` is None during a dry run.
    """
    # if vbf_filter_study: remove z-peak from region (same rule as before)
    fill_regions = [r for r in region_list if not (do_vbf_filter_study and r == "z-peak")]
    if len(fill_regions) == 0:
        logger.warning(
            f"No regions left to fill for jj_eta_region={jj_eta_region} "
            f"vbf_filter_study={do_vbf_filter_study} after z-peak removal; skipping this scope."
        )
        return

    zpt_options = _derive_zpt_options(remove_zpt_weights_options, add_dnn_zpt_weights_options)

    save_path = save_root / f"VBFfilter_{do_vbf_filter_study}"
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 80)
    logger.info(
        f"Combo matrix: years={years} categories={categories} njets={njets_options} "
        f"zpt_options={zpt_options} regions={fill_regions} jj_eta_region={jj_eta_region} "
        f"vbf_filter_study={do_vbf_filter_study}"
    )

    variables2plot = get_all_vars(min_set_of_vars)
    if "jj_mass_nominal" in variables2plot:
        variables2plot += ["jj_mass_nominal_range2"]
    if "dimuon_mass" in variables2plot:
        variables2plot += ["dimuon_mass_zpeak"]
    logger.info(f"variables2plot: {variables2plot}")

    sample_groups = list(group_dict.keys()) + ["other"]
    if not do_vbf_filter_study and "DYVBF" in sample_groups:
        sample_groups.remove("DYVBF")
    logger.info(f"sample_groups: {sample_groups}")

    plot_settings_by_category = {cat: load_plot_settings(cat) for cat in categories}
    hist_templates_by_category = {
        cat: build_hist_templates(
            variables2plot, plot_settings_by_category[cat], sample_groups, njets_options, zpt_options
        )
        for cat in categories
    }

    if dry_run:
        logger.info(
            f"[dry-run] would build fileset for years={years} and run one Dask pass over "
            f"{len(categories)} categories x {len(njets_options)} njets x {len(zpt_options)} "
            "zpt_options; no data read."
        )
        return

    group_dict_by_year = {}
    lumi_by_year = {}
    CM_energy_by_year = {}
    fileset = {}
    for year in years:
        load_path = Path(str(load_path_template).format(year=year))
        year_ctx = resolve_year_context(
            year, background_samples, sig_samples, data_samples, do_vbf_filter_study, lumi_override=""
        )
        group_dict_by_year[year] = year_ctx["group_dict"]
        lumi_by_year[year] = year_ctx["lumi"]
        CM_energy_by_year[year] = year_ctx["CM_energy"]
        logger.info(f"{year} available_processes: {year_ctx['available_processes']}")

        year_fileset = build_fileset_for_year(
            year, load_path, year_ctx["available_processes"], use_compacted
        )
        for process, entry in year_fileset.items():
            fileset[f"{year}{DATASET_SEPARATOR}{process}"] = entry
    logger.info(f"finished building fileset! ({len(fileset)} datasets across {len(years)} year(s))")

    logger.info("{style}Filling histograms.{style}".format(
        style="\n" + "=" * 50 + "\n",))
    # Reduce eagerly as each dataset's result comes back (running total per
    # (year, category, njets, var), not a growing list summed at plot time) --
    # otherwise every process's histogram from every (category, njets) sub-pass
    # stays live in memory simultaneously until the plotting loop finally sums
    # them all at the end, which is its own (separate, driver-side) memory
    # blowup on top of the per-Runner-call one below.
    #
    # Keyed by njets too (not just year/category/var): each (category, njets)
    # sub-pass below builds its own scoped_templates with a njets axis
    # containing only that one njets value, so histograms from different njets
    # sub-passes have different axis shapes and cannot be `+=`-merged together
    # (boost-histogram raises "axes not mergable"). Only ever accumulate across
    # results that came from the same sub-pass (same njets key).
    sample_hist_lookup = {
        year: {
            cat: {njets: {var: None for var in hist_templates_by_category[cat]} for njets in njets_options}
            for cat in categories
        }
        for year in years
    }
    # Run one Runner pass per (category, njets) pair -- each pass still consolidates
    # every year and zpt_option together, but category and njets are the two axes
    # that multiply the size of every histogram returned per chunk (category needs
    # its own Hist templates since binning differs; njets is an axis on top of that).
    # Folding all of them into a single pass made the per-chunk/per-merge payload
    # ~(len(categories) x len(njets_options)) times larger than a single combo's
    # baseline, which was overflowing worker memory during the tree-reduction
    # coffea's Runner does to accumulate results across chunks. Splitting by
    # (category, njets) bounds that back down (at the cost of re-reading the
    # fileset once per (category, njets) pair instead of once total) while still
    # keeping the year- and zpt_option-consolidation wins, which don't inflate
    # per-chunk memory the way category/njets do.
    for category in categories:
        plot_settings = plot_settings_by_category[category]
        for njets in njets_options:
            if category == "vbf" and njets != "inclusive":
                continue
            logger.info(
                f"Filling histograms for category={category} njets={njets} "
                f"(years={years}, zpt_options={zpt_options})..."
            )
            scoped_templates = build_hist_templates(
                variables2plot, plot_settings, sample_groups, [njets], zpt_options
            )
            processor_instance = ValidationHistProcessor(
                hist_templates_by_category={category: scoped_templates},
                categories=[category],
                njets_options=[njets],
                zpt_options=zpt_options,
                regions=fill_regions,
                do_vbf_filter_study=do_vbf_filter_study,
                jj_eta_region=jj_eta_region,
                group_dict_by_year=group_dict_by_year,
            )
            results = run_validation_runner(fileset, processor_instance, client)
            for dataset_key, output in results.items():
                result_year, _process = dataset_key.split(DATASET_SEPARATOR, maxsplit=1)
                for var, h in output["hist_by_category_var"][category].items():
                    slot = sample_hist_lookup[result_year][category][njets]
                    if slot[var] is None:
                        slot[var] = h
                    else:
                        slot[var] += h

    logger.info("\n" + "=" * 80)
    logger.info("Generating plots for all combos...")
    job_idx = 0
    for year in years:
        for category in categories:
            plot_settings = plot_settings_by_category[category]
            for njets in njets_options:
                # skip meaningless combos
                if category == "vbf" and njets != "inclusive":
                    logger.debug(f"Skipping vbf with njets={njets} (not meaningful)")
                    continue
                for zpt_option in zpt_options:
                    job_idx += 1
                    logger.info(f"[{job_idx:04d}] {year} {category} njets={njets} zpt_option={zpt_option}")
                    for region_name in fill_regions:
                        for var in hist_templates_by_category[category]:
                            generate_combo_plots(
                                year,
                                category,
                                njets,
                                zpt_option,
                                region_name,
                                var,
                                sample_hist_lookup,
                                sample_groups,
                                plot_settings,
                                str(save_path),
                                lumi_by_year[year],
                                status,
                                CM_energy_by_year[year],
                                not linear_scale,
                                do_vbf_filter_study,
                                jj_eta_region,
                            )
    logger.info(f"Generated plots for {job_idx} (year, category, njets, zpt_option) combos.")


def run_bulk_validation(
    years,
    categories,
    njets_options,
    jj_eta_regions,
    vbf_filter_study_options,
    remove_zpt_weights_options,
    add_dnn_zpt_weights_options,
    region_options,
    min_set_of_vars,
    load_path_template,
    save_root,
    background_samples,
    sig_samples,
    data_samples,
    status="Preliminary",
    linear_scale=False,
    use_gateway=True,
    cluster_index=0,
    use_compacted="compacted",
    dry_run=False,
):
    """
    Run the full validation-plot sweep: every (jj_eta_region, vbf_filter_study,
    region_list) scope, sharing one Dask client across all of them. Within each
    scope, year x category x njets x zpt_option are all computed together in as
    few coffea Runner passes as possible (see _run_validation_scope and
    ValidationHistProcessor above).

    This is the single entry point run_plotter.py calls -- all the orchestration
    complexity (Dask client lifecycle, fileset construction, the Runner passes,
    plot generation) lives here so run_plotter.py can stay a thin config wrapper.
    """
    scopes = list(itertools.product(jj_eta_regions, vbf_filter_study_options, region_options))
    logger.info(f"Running {len(scopes)} (jj_eta_region, vbf_filter_study, region_list) scope(s).")

    client = None
    if not dry_run:
        client = get_dask_client(use_gateway, cluster_index=cluster_index)
        logger.info(f"client: {client}")

    for jj_eta_region, do_vbf_filter_study, region_list in scopes:
        _run_validation_scope(
            jj_eta_region,
            do_vbf_filter_study,
            region_list,
            client,
            years,
            categories,
            njets_options,
            remove_zpt_weights_options,
            add_dnn_zpt_weights_options,
            min_set_of_vars,
            load_path_template,
            save_root,
            background_samples,
            sig_samples,
            data_samples,
            status,
            linear_scale,
            use_compacted,
            dry_run,
        )

    if not dry_run:
        close_dask_client()


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

    if args.remove_zpt_weights and args.use_dnn_zpt_weights:
        raise ValueError(
            "Use either --remove_zpt_weights or --use_dnn_zpt_weights, not both."
        )

    if (args.do_vbf_filter_study):
        #  Remove the "z-peak" region from the args.regions if it exists
        if "z-peak" in args.regions:
            logger.info("Removing z-peak region from the regions!")
            args.regions.remove("z-peak")
        else:
            logger.warning("z-peak region is not in the regions, nothing to remove!")

    # If the args.regions is empty, exit the program
    if len(args.regions) == 0:
        logger.error("No regions specified! Exiting the program.")
        raise ValueError("No regions specified!")

    status = args.status.replace("_", " ")
    do_logscale = not args.linear_scale

    years = [args.year]
    categories = [args.category]
    njets_options = [str(args.njets)]
    if args.remove_zpt_weights:
        zpt_option = "no_zpt"
    elif args.use_dnn_zpt_weights:
        zpt_option = "dnn_zpt"
    else:
        zpt_option = "default"
    zpt_options = [zpt_option]

    # gather variables to plot:
    kinematic_vars = ['pt', 'eta', 'phi']
    if args.minimum_set: kinematic_vars = ['pt', 'eta']
    variables2plot = get_all_vars(args.minimum_set)  # get the full list of variables from the config file

    if "jj_mass_nominal" in variables2plot:
        variables2plot += ["jj_mass_nominal_range2"] # add another range to plot
    if "dimuon_mass" in variables2plot:
        variables2plot += ["dimuon_mass_zpeak"] # add another range to plot
    logger.info(f"variables2plot: {variables2plot}")

    # sample_groups: fixed group-name list, independent of year (only the DYVBF
    # removal rule -- itself independent of year -- can change it).
    sample_groups = list(group_dict.keys()) + ["other"]
    if not args.do_vbf_filter_study and "DYVBF" in sample_groups:
        sample_groups.remove("DYVBF")
    logger.info(f"sample_groups: {sample_groups}")

    plot_settings_by_category = {cat: load_plot_settings(cat) for cat in categories}

    logger.info("{style}Initializing histograms for each variable to be plotted.{style}".format(
        style="\n" + "="*50 + "\n",))
    hist_templates_by_category = {
        cat: build_hist_templates(
            variables2plot, plot_settings_by_category[cat], sample_groups, njets_options, zpt_options
        )
        for cat in categories
    }

    # define client for parallelization
    client = get_dask_client(args.use_gateway, cluster_index=args.cluster_index)
    logger.info(f"client: {client}")

    # record time
    time_step = time.time()

    group_dict_by_year = {}
    lumi_by_year = {}
    CM_energy_by_year = {}
    fileset = {}
    for year in years:
        year_ctx = resolve_year_context(
            year,
            args.background_samples,
            args.sig_samples,
            args.data_samples,
            args.do_vbf_filter_study,
            lumi_override=args.lumi,
        )
        group_dict_by_year[year] = year_ctx["group_dict"]
        lumi_by_year[year] = year_ctx["lumi"]
        CM_energy_by_year[year] = year_ctx["CM_energy"]
        logger.info(f"available_processes: {year_ctx['available_processes']}")

        year_fileset = build_fileset_for_year(
            year, args.load_path, year_ctx["available_processes"], args.use_compacted
        )
        for process, entry in year_fileset.items():
            fileset[f"{year}{DATASET_SEPARATOR}{process}"] = entry
    logger.info("finished building fileset!")

    # fill the histograms: one coffea Runner pass, chunked and dask-parallelized,
    # filling every (category, njets, zpt_option, variable) combo per chunk (eager
    # hist.Hist, mirrors run_stage2_vbf.py's CoffeaStage2VBFProcessor pattern).
    logger.info("{style}Filling histograms for each variable.{style}".format(
        style="\n" + "="*50 + "\n",))
    processor_instance = ValidationHistProcessor(
        hist_templates_by_category=hist_templates_by_category,
        categories=categories,
        njets_options=njets_options,
        zpt_options=zpt_options,
        regions=args.regions,
        do_vbf_filter_study=args.do_vbf_filter_study,
        jj_eta_region=args.jj_eta_region,
        group_dict_by_year=group_dict_by_year,
    )
    results = run_validation_runner(fileset, processor_instance, client)

    # Reduce eagerly as each dataset's result comes back (running total per
    # (year, category, njets, var), not a growing list summed at plot time) so
    # memory stays bounded by the number of distinct keys, not the number of
    # processes. Keyed by njets to match generate_combo_plots()'s shared lookup
    # shape with run_bulk_validation() -- here there's only ever one njets value
    # (categories/njets_options are both size-1 for this single-combo CLI path),
    # so it's just a trivial single-key level, not a merge-compatibility concern.
    sample_hist_lookup = {
        year: {
            cat: {njets: {var: None for var in hist_templates_by_category[cat]} for njets in njets_options}
            for cat in categories
        }
        for year in years
    }
    for dataset_key, output in results.items():
        result_year, _process = dataset_key.split(DATASET_SEPARATOR, maxsplit=1)
        for cat, var_hists in output["hist_by_category_var"].items():
            for var, h in var_hists.items():
                for njets in njets_options:
                    slot = sample_hist_lookup[result_year][cat][njets]
                    if slot[var] is None:
                        slot[var] = h
                    else:
                        slot[var] += h

    logger.info("{style}Generating plots.{style}".format(
        style="\n" + "="*50 + "\n",))
    last_save_path = args.save_path
    for year in years:
        for category in categories:
            plot_settings = plot_settings_by_category[category]
            for njets in njets_options:
                if category == "vbf" and njets != "inclusive":
                    continue
                for zpt_option in zpt_options:
                    for region_name in args.regions:
                        for var in tqdm.tqdm(hist_templates_by_category[category]):
                            result_path = generate_combo_plots(
                                year,
                                category,
                                njets,
                                zpt_option,
                                region_name,
                                var,
                                sample_hist_lookup,
                                sample_groups,
                                plot_settings,
                                args.save_path,
                                lumi_by_year[year],
                                status,
                                CM_energy_by_year[year],
                                do_logscale,
                                args.do_vbf_filter_study,
                                args.jj_eta_region,
                            )
                            if result_path:
                                last_save_path = result_path

    close_dask_client()
    logger.info("Plots are saved to %s", last_save_path)
    time_elapsed = round(time.time() - time_step, 3)
    logger.info(f"Finished in {time_elapsed} s.")
