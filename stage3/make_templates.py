import numpy as np
import pandas as pd

# from python.workflow import parallelize
# from python.variable import Variable
# from python.io import load_stage2_output_hists, save_template, mkdir, load_stage2_output_df2hists

# import warnings

# warnings.simplefilter(action="ignore", category=FutureWarning)
# from uproot3_methods.classes.TH1 import from_numpy
import glob
import pickle
import itertools
import ROOT
import os

import logging
from modules.utils import logger
logger.setLevel(logging.INFO)

class Variable(object):
    def __init__(self, name_, caption_, nbins_, xmin_, xmax_):
        self.name = name_
        self.caption = caption_
        self.nbins = nbins_
        self.xmin = xmin_
        self.xmax = xmax_

decorrelation_scheme = {
    "LHERen": ["DY", "EWK", "ggH", "TT+ST"],
    "LHEFac": ["DY", "EWK", "ggH", "TT+ST"],
    "pdf_2rms": ["DY", "VBF", "ggH"], # ["DY", "qqH_hmm", "ggH_hmm"],
    # "pdf_2rms": ["DY", "qqH_hmm", "ggH_hmm"],
}
shape_only = [
    "wgt_LHERen_up",
    "wgt_LHERen_down",
    "wgt_LHEFac_up",
    "wgt_LHEFac_down",
    "wgt_qgl_up",
    "wgt_qgl_down",
    "wgt_pdf_2rms_up",
    "wgt_pdf_2rms_down",
]

def load_stage2_output_hists(argset, parameters, dataset):
    year = argset["year"]
    var_name = argset["var_name"]
    global_path = parameters.get("global_path", None)


    if (global_path is None):
        logger.error("global_path is not set in parameters!")
        raise ValueError("global_path is not set in parameters!")
        # return

    path = f"{global_path}/stage2_histograms/{var_name}/{year}/"
    paths = glob.glob(f"{path}/{dataset}*.pkl")

    logger.debug(f"dataset: {dataset}")
    logger.debug(f"var_name: {var_name}")
    logger.debug(f"path: {path}")
    logger.debug(f"paths: {paths}")
    hist_df = pd.DataFrame()
    for path in paths:
        with open(path, "rb") as handle:
            hist = pickle.load(handle)
            new_row = {
                "year": year,
                "var_name": var_name,
                "dataset": dataset,
                "hist": hist,
            }
            hist_df = pd.concat([hist_df, pd.DataFrame([new_row])])
            hist_df.reset_index(drop=True, inplace=True)
            logger.debug(f"Loaded histogram for {dataset} in {year} with variable {var_name}: {hist}")
            # logger.debug(f"Loaded histogram for {dataset} in {year} with variable {var_name}")
            logger.debug(f"hist_df shape: {hist_df.shape}")
    if hist_df.shape[0] == 0:
        logger.debug(f"No histograms found for {dataset} in {year} with variable {var_name}")
        return pd.DataFrame()

    return hist_df

def getTH1D_from_numpy(group_hist, bin_edges, group_sumw2, centers, name):
    logger.debug("=================== getTH1D_from_numpy ============")
    logger.debug("Creating ROOT.TH1D histogram from numpy arrays")
    logger.debug(f"Group histogram: {group_hist}")
    logger.debug(f"Bin edges: {bin_edges}")
    logger.debug(f"Group sumw2: {group_sumw2}")
    logger.debug(f"Centers: {centers}")
    logger.debug(f"Histogram name: {name}")
    # Create a ROOT.TH1D histogram
    hist_values = group_hist
    hist_w2 = group_sumw2
    hist_name = name
    n_bins = len(hist_values)
    hist = ROOT.TH1D(hist_name, "", n_bins, bin_edges[0], bin_edges[-1])

    # Fill the ROOT.TH1D histogram with the bin contents
    logger.debug(f"Filling histogram {hist_name} with {n_bins} bins")
    for i, value in enumerate(hist_values):
        hist.SetBinContent(i + 1, value)  # Bin index starts at 1 in ROOT
        logger.debug(f"Set bin {i + 1} content to {value}")

    # Enable Sumw2 and overwrite it
    hist.Sumw2()  # Ensure Sumw2 is enabled
    sumw2_array = hist.GetSumw2()

    # Overwrite the Sumw2 array values
    # Overwrite Sumw2 with weights_squared
    for i, w2 in enumerate(hist_w2, start=1):  # Start at 1 for ROOT bin indexing
        # logger.info(f"w2 {w2}, for idx {i}")
        sumw2_array[i] = w2

    # # Debugging: Verify the new Sumw2 values
    # logger.info("Updated Sumw2 values:")
    # for i in range(0, n_bins + 1):
    #     logger.info(f"Bin {i}, Sumw2: {sumw2_array[i]}")
    # logger.info("Updated hist values:")
    # for i in range(0, n_bins + 1):
    #     logger.info(f"Bin {i}, content: {hist.GetBinContent(i)}")
    return hist

def to_templates(client, parameters, hist_df=None):
    # datasets = list(parameters["datasets"]) # original
    # datasets = list(parameters["datasets"]) + ["ewk_lljj_mll105_160_py_dipole", "vbf_powheg_herwig"] # manually add partonShower
    datasets = list(parameters["datasets"])
    if hist_df is None:
        logger.info("Loading histograms from stage2 output")
        logger.debug(f"datasets: {datasets}")
        logger.debug(f"parameters: {parameters}")

        argset_load = {
            "year": parameters["years"][0],
            "var_name": parameters["templates_vars"][0],
            "dataset": datasets,
        }

        hist_rows = []
        for dataset in datasets:
            hist_row = load_stage2_output_hists(argset_load, parameters, dataset)
            logger.debug(f"hist_row: {hist_row}")
            hist_rows.append(hist_row)

        hist_df = pd.concat(hist_rows).reset_index(drop=True)
        logger.debug(f"hist_df: {hist_df}")
        if hist_df.shape[0] == 0:
            logger.info("No templates to create!")
            return []

    # argset = {
    #     "year": parameters["years"],
    #     "region": parameters["regions"],
    #     "channel": parameters["channels"],
    #     "var_name": [
    #         v for v in hist_df.var_name.unique() if v in parameters["templates_vars"]
    #     ],
    #     "hist_df": [hist_df],
    # }
    # yield_dfs = parallelize(make_templates, argset, client, parameters, seq=True)

    argset = {
        "year": parameters["years"],
        "region": parameters["regions"],
        "channel": parameters["channels"],
        "var_name": [
            v for v in hist_df.var_name.unique() if v in parameters["templates_vars"]
        ],
        "hist_df": [hist_df],
    }
    logger.info(f"argset: {argset}")
    # generate all combinations within argset and loop through them

    combinations = list(itertools.product(*argset.values()))

    # Convert to list of dictionaries
    combination_dicts = [
        dict(zip(argset.keys(), combination)) for combination in combinations
    ]
    logger.info(f"Generated {len(combination_dicts)} combinations from argset")
    logger.info(f"Combination dicts: {combination_dicts}")

    # loop through combinations
    yield_dfs = []
    for combo in combination_dicts:
        logger.debug(f"argset combo: {combo}")
        yield_df = make_templates(combo, parameters)
        logger.info(f"yield_df: {yield_df}")
        if yield_df is None or yield_df.empty:
            logger.error(f"argset combo: {combo}")
            logger.error(f"parameters: {parameters}")
            raise ValueError("No templates to create for this combination!")
        yield_dfs.append(yield_df)


    yield_df = pd.concat(yield_dfs).reset_index(drop=True)
    return yield_df


def make_templates(args, parameters={}):
    logger.info("============= make_templates ============")
    logger.debug(f"args: {args}")
    logger.info(f"parameters: {parameters}")
    year = args["year"]
    logger.debug(f"make_template year: {year}")
    logger.debug(f'args["hist_df"].year: {args["hist_df"].year}')

    region = args["region"]
    channel = args["channel"]
    var_name = args["var_name"]
    hist_df = args["hist_df"].loc[
        (args["hist_df"].var_name == var_name) & (args["hist_df"].year == year)
    ]
    logger.debug(f"hist_df: {hist_df}")
    if "2016" in year:
        year_savepath = year
        year = "2016"
    else:
        year_savepath = year

    # if var_name in parameters["variables_lookup"].keys():
    #     var = parameters["variables_lookup"][var_name]
    # else:
    #     var = Variable(var_name, var_name, 50, 0, 5)
    var = Variable(var_name, var_name, 50, 0, 5)

    if hist_df.shape[0] == 0:
        logger.error(f"No templates found for {var_name} in {year} for {region} and {channel}. Skipping!")
        return

    yield_rows = []
    templates = []

    groups = list(set(parameters["grouping"].values()))
    logger.info(f"groups: {groups}")
    logger.info(f"hist_df.dataset.unique(): {hist_df.dataset.unique()}")

    for group in groups:
        datasets = []
        for d in hist_df.dataset.unique():
            if d not in parameters["grouping"].keys():
                continue
            if parameters["grouping"][d] != group:
                continue
            datasets.append(d)

        if len(datasets) == 0:
            continue

        logger.info(f"datasets: {datasets}")

        # make a list of systematics;
        # avoid situation where different datasets have incompatible systematics
        wgt_variations = []
        for dataset in datasets:
            n_partitions = len(hist_df.loc[hist_df.dataset == dataset, "hist"].values)
            for i in range(n_partitions):
                new_wgt_vars = list(
                    hist_df.loc[hist_df.dataset == dataset, "hist"]
                    .values[i]
                    .axes["variation"]
                )
                # logger.info(f"new_wgt_vars: {new_wgt_vars}")
                if len(wgt_variations) == 0:
                    wgt_variations = new_wgt_vars
                else:
                    wgt_variations = list(set(wgt_variations) & set(new_wgt_vars))

        # logger.info(f"wgt_variations: {wgt_variations}")
        # manually add parton shower variations start -------------------------------
        add_VBF_PartonShower = False
        add_EWK_PartonShower = False
        for wgt_variation in wgt_variations:
            if "qqH_hmm" ==group:
                add_VBF_PartonShower = True
                break
            elif "EWK" ==group:
                add_EWK_PartonShower = True
                break
        if add_VBF_PartonShower:
            wgt_variations += ["qqH_hmm_SignalPartonShowerUp", "qqH_hmm_SignalPartonShowerDown"]
        if add_EWK_PartonShower:
            wgt_variations += ["EWK_EWKPartonShowerUp", "EWK_EWKPartonShowerDown"]

        # logger.info(f"add_VBF_PartonShower: {add_VBF_PartonShower}")
        # logger.info(f"add_EWK_PartonShower: {add_EWK_PartonShower}")
        # manually add parton shower variations end -------------------------------
        logger.info(f"wgt_variations: {wgt_variations}")

        for variation in wgt_variations:
            logger.info(f"variation: {variation}")
            logger.info(f"channel: {channel}")

            group_hist = []
            group_sumw2 = []

            slicer_nominal = {
                "region": region,
                "channel": channel,
                "variation": "nominal",
                "val_sumw2": "value",
            }
            slicer_value = {
                "region": region,
                "channel": channel,
                "variation": variation,
                "val_sumw2": "value",
            }


            #Parton Shower case start -----------------------------
            if ("PartonShower" in variation):
                slicer_sumw2 = { # slicer_sumw2 needs to be overwritten
                    "region": region,
                    "channel": channel,
                    "variation": "nominal",
                    "val_sumw2": "sumw2",
                }
                if ("qqH_hmm" in variation):
                    baseline_dataset = "vbf_powheg_dipole"
                    variation_dataset = "vbf_powheg_herwig"
                elif ("EWK" in variation):
                    # ewk_lljj_mll50_mjj120_hist.pkl
                    baseline_dataset = "ewk_lljj_mll105_160_ptj0"
                    variation_dataset = "ewk_lljj_mll105_160_py_dipole"
                else:
                    logger.info("no parton shower exists for this sample!")
                    raise ValueError
                # vals_baseline = hist_df.loc[hist_df.dataset == baseline_dataset, "hist"].values
                logger.info(f'hist_df.loc[hist_df.dataset == baseline_dataset, "hist"]: {hist_df.loc[hist_df.dataset == baseline_dataset, "hist"]}')
                # hist_baseline = hist_df.loc[hist_df.dataset == baseline_dataset, "hist"].values.sum()

                baseline_hists = hist_df.loc[hist_df.dataset == baseline_dataset, "hist"].tolist()
                if not baseline_hists:
                    raise ValueError(f"No histograms found for baseline dataset {baseline_dataset}")

                # now sum them via Python, which will invoke your hist + hist properly
                hist_baseline = sum(baseline_hists)

                the_hist_nominal_baseline = hist_baseline[slicer_nominal].project(var.name).values()
                the_sumw2_baseline = hist_baseline[slicer_sumw2].project(var.name).values()


                # logger.info(f"{group} the_hist_nominal_baseline: {the_hist_nominal_baseline}")
                # logger.info(f"{group} the_sumw2_baseline: {the_sumw2_baseline}")

                # vals_variation = hist_df.loc[hist_df.dataset == variation_dataset, "hist"].values
                # hist_variation = hist_df.loc[hist_df.dataset == variation_dataset, "hist"].values.sum() # no need to sum() different histograms yet

                variation_hists = hist_df.loc[hist_df.dataset == variation_dataset, "hist"].tolist()
                if not variation_hists:
                    logger.info(f"No template found for {group} variation_dataset: {variation_dataset}. Skipping!")
                    continue

                hist_variation = sum(variation_hists)

                if not variation_hists:
                    continue


                logger.info(f"{group} hist_variation: {hist_variation}")
                logger.info(f"{group} hist_df: {hist_df}")
                logger.info(f"{group} variation_dataset: {variation_dataset}")

                the_hist_nominal_variation = hist_variation[slicer_nominal].project(var.name).values()
                the_sumw2_variation = hist_variation[slicer_sumw2].project(var.name).values()

                logger.info(f"{group} the_hist_nominal_variation: {the_hist_nominal_variation}")
                logger.info(f"{group} the_sumw2_variation: {the_sumw2_variation}")
                logger.info(f"{group} the_hist_nominal_variation: {type(the_hist_nominal_variation)}")
                logger.info(f"{group} the_sumw2_variation: {type(the_sumw2_variation)}")


                edges = hist_baseline[slicer_nominal].project(var.name).axes[0].edges
                edges = np.array(edges)
                logger.info(f"edges: {edges}")
                centers = (edges[:-1] + edges[1:]) / 2.0
                name = variation

                if "Up" in variation:
                    group_hist = the_hist_nominal_baseline - (the_hist_nominal_baseline -  the_hist_nominal_variation)
                elif "Down" in variation:
                    group_hist = the_hist_nominal_baseline + (the_hist_nominal_baseline -  the_hist_nominal_variation)
                else:
                    logger.info("unknown variation in parton shower")
                    raise ValueError

                # logger.info(f"group_hist: {group_hist}")
                # group_sumw2 = the_sumw2_variation*0
                group_sumw2 = 2*the_sumw2_baseline + the_sumw2_variation


                # logger.info(f"variation name: {name}")
                th1 = from_numpy([group_hist, edges])
                th1._fName = name
                th1._fSumw2 = np.array(np.append([0], group_sumw2)) # -> np.array([0, group_sumw2])
                th1._fTsumw2 = np.array(group_sumw2).sum()
                th1._fTsumwx2 = np.array(group_sumw2 * centers).sum() #-> this is w2*x distibution
                templates.append(th1)

                # variation_fixed = variation.replace("VBF_", "").replace("EWK_", "")
                variation_fixed = variation.replace("qqH_hmm_", "").replace("EWK_", "")

                # logger.info(f"variation_fixed: {variation_fixed}")
                yield_row = {
                        "var_name": var_name,
                        "group": group,
                        "region": region,
                        "channel": channel,
                        "year": year,
                        "variation": variation_fixed,
                        "yield": group_hist.sum(),
                }
                # logger.info(f"yield_rows: {yield_rows}")
                yield_rows.append(yield_row)
                continue # done parton shower, skip the rest of the loop
            # Parton Shower case end -----------------------------
            # do the normal for loop if not PartonShower

            slicer_sumw2 = {
                "region": region,
                "channel": channel,
                "variation": variation,
                "val_sumw2": "sumw2",
            }
            logger.debug(f"datasets: {datasets}")
            logger.debug(f"slicer_value: {slicer_value}")
            logger.debug(f"slicer_nominal: {slicer_nominal}")
            logger.debug(f"hist_df: {hist_df}")
            for dataset in datasets:
                try:
                    # hist = hist_df.loc[hist_df.dataset == dataset, "hist"].values.sum()

                    # my attempt start -----------------------------------------------------------
                    vals = hist_df.loc[hist_df.dataset == dataset, "hist"].values
                    logger.debug(f"vals for dataset {dataset}: {vals}")
                    if len(vals) == 0:
                        logger.debug(f"No template found for {dataset} in {year}. Skipping!")
                        continue
                    #---------------------------------------------------------
                    # available_axes = ['region', 'channel', 'val_sumw2', 'score_vbf', 'variation'] # debugging
                    # for axes in available_axes:
                    #     logger.info(f"testing axes: {axes}")
                    #     projection = vals[slicer_value].project(var.name)#.values().sum()
                    #     logger.info(f"testing projection: {projection}")
                    # logger.info(f"make_templates vals: {vals}")
                    # sliced_val = vals[slicer_value]
                    # logger.info(f"testing sliced_val: {sliced_val}")
                    # projection = vals[slicer_value].project(var.name).sum()
                    # logger.info(f"testing projection: {projection}")
                    #---------------------------------------------------------
                    # logger.info(f"make_templates len vals: {len(vals)}")
                    # logger.info(f"make_templates type(vals[0]): {type(vals[0])}")
                    # for histogram in list(vals)[:4]:
                    val_l = list(vals)
                    logger.info(f"make_templates len(val_l): {len(val_l)}")
                    logger.debug(f"make_templates val_l: {val_l}")
                    # bad_idxs = [4, 6, 7, 8, 10, 13, 15, 16, 17, 25, 28, 34, 41, 42, 51, 53, 55, 58, 60, 73, 78, 80, 81, 82, 83, 91, 92, 99, 101, 102, 104, 121]
                    bad_idxs = []
                    hist_sum = val_l[0]
                    logger.debug(f"hist_sum: {hist_sum}")

                    for hist_idx in range(1, len(val_l)):
                        logger.info(f"make_templates hist_idx: {hist_idx}")
                        histogram = val_l[hist_idx]
                        # axes_l = [axis.label for axis in histogram.axes]
                        # logger.info(f"{hist_idx} axes_l: {axes_l}")
                        if hist_idx in bad_idxs:
                            logger.debug(f"Skipping bad histogram index {hist_idx} for dataset {dataset}")
                            continue
                        try:
                            hist_sum = hist_sum+histogram
                        except Exception as e:
                            # logger.info(f"Exception {e}")
                            bad_idxs.append(hist_idx)
                            # logger.info(f"bad idx {hist_idx} with error {e}")
                        # logger.info(f"make_templates histogram: {histogram}")
                        # logger.info(f"make_templates histogram.axes: {histogram.axes}")
                        # np_val = histogram.values()
                        # logger.info(f"make_templates histogram.values(): {np_val}")
                        # logger.info(f"make_templates histogram.values().shape: {np_val.shape}")

                    # logger.info(f"make_templates type(vals): {type(vals)}")
                    # logger.info(f"make_templates axes: {vals.axes}")
                    # hist = np.sum(vals.values().flatten())
                    if len(bad_idxs) > 0:
                        logger.info(f"{dataset} bad_idxs: {bad_idxs}")
                    # hist = vals.sum()
                    hist = hist_sum
                    # vals = list(vals)
                    # hist = np.array([val.values() for val in vals]).sum(axis=0)
                    # logger.info(f"make_templates his.shapet: {hist.shape}")
                    # raise ValueError
                    # my attempt end -----------------------------------------------------------

                except Exception as e:
                    logger.info(f"Could not merge histograms for {dataset} due to error {e}")
                    continue

                try:
                    logger.debug(f"hist_sum: {hist_sum}")
                    logger.debug(f"hist: {hist}")
                    logger.debug(f"slicer_value: {slicer_value}")
                    logger.debug(f"var.name: {var.name}")
                    # logger.debug(f"hist_sum.view(): {hist_sum.view()}")
                    # logger.debug(f"hist.view(): {hist.view()}")
                    # slicer_value["val_sumw2"] = "sumw2"
                    the_hist = hist[slicer_value].project(var.name).values()
                except Exception as e:
                    logger.info(f"Could not project histograms for {dataset} due to error {e}")
                    continue
                the_hist_nominal = hist[slicer_nominal].project(var.name).values()
                the_sumw2 = hist[slicer_sumw2].project(var.name).values()

                if variation in shape_only:
                    if the_hist.sum() != 0:
                        scale = the_hist_nominal.sum() / the_hist.sum()
                    else:
                        scale = 1.0
                    the_hist = the_hist * scale
                    the_sumw2 = the_sumw2 * scale

                # temporary overwrite for TT+ST group ----------------
                # if group=="TT+ST":
                #     scale=1.081915477
                #     the_hist = the_hist * scale
                #     the_sumw2 = the_sumw2 * scale
                # if group=="Data":
                #     # logger.info("data is present!")
                #     scale=3975.000/3939
                #     the_hist = the_hist * scale
                #     the_sumw2 = the_sumw2 * scale
                # # elif group=="DYJ01":
                # #     scale=1389.6971467649/2525.096684
                # #     the_hist = the_hist * scale
                # #     the_sumw2 = the_sumw2 * scale
                # # elif group=="DYJ2":
                # #     scale=2265.5777773395/1104.819084
                # #     the_hist = the_hist * scale
                # #     the_sumw2 = the_sumw2 * scale
                # elif group=="ggH":
                #     scale=9.240/9.599384
                #     the_hist = the_hist * scale
                #     the_sumw2 = the_sumw2 * scale
                # elif group=="VBF":
                #     scale=11.784/11.936208
                #     the_hist = the_hist * scale
                #     the_sumw2 = the_sumw2 * scale
                # elif group=="EWK":
                #     scale=125.749/126.96623
                #     the_hist = the_hist * scale
                #     the_sumw2 = the_sumw2 * scale
                # elif group=="VV":
                #     scale=78.102/80.901781
                #     the_hist = the_hist * scale
                #     the_sumw2 = the_sumw2 * scale



                # -------------------------------------

                logger.debug(f"the_hist: {the_hist}")

                if (the_hist.sum() < 0) or (the_sumw2.sum() < 0):
                    logger.debug(f"Negative histogram found for {group} in {year} for {region} and {channel}. Skipping!")
                    continue

                if len(group_hist) == 0:
                    group_hist = the_hist
                    group_sumw2 = the_sumw2
                else:
                    group_hist += the_hist
                    group_sumw2 += the_sumw2

                edges = hist[slicer_value].project(var.name).axes[0].edges
                edges = np.array(edges)
                centers = (edges[:-1] + edges[1:]) / 2.0
            logger.debug(f"group_hist: {group_hist}")

            if len(group_hist) == 0:
                logger.debug(f"No histograms found for group {group} in {year} for {region} and {channel}. Skipping!")
                continue
            if sum(group_hist) == 0:
                logger.debug(f"Sum of histogram for group {group} is zero in {year} for {region} and {channel}. Skipping!")
                continue

            if group == "Data":
                name = "data_obs"
            else:
                name = group

            logger.info(f"group: {group}")
            logger.info(f"variation: {variation}")
            if variation == "nominal":
                # variation_core = variation.replace("wgt_", "")
                # variation_core = variation_core.replace("_up", "")
                # variation_core = variation_core.replace("_down", "")
                # logger.info(f"variation_core: {variation_core}")

                # else:
                variation_fixed = variation
            else:
                variation_core = variation.replace("wgt_", "")
                variation_core = variation_core.replace("_up", "")
                variation_core = variation_core.replace("_down", "")
                logger.info(f"variation_core: {variation_core}")
                suffix = ""
                if variation_core in decorrelation_scheme.keys():
                    group_LHE = group
                    logger.info(f"group_LHE: {group_LHE}")
                    if group_LHE == "DYJ2" or group_LHE == "DYJ01" :
                        group_LHE = "DY"
                    elif "qqH" in group_LHE :
                        group_LHE = "VBF"
                    elif "ggH" in group_LHE :
                        group_LHE = "ggH"
                    logger.info(f"group_LHE after: {group_LHE}")
                    if group_LHE in decorrelation_scheme[variation_core]:
                        if variation_core == "pdf_2rms" :
                            suffix = "_"+group_LHE+str(year)
                            logger.info(f"pdf_2rms suffix: {suffix}")
                        else:
                            suffix = "_"+group_LHE
                    else:
                        continue
                elif variation_core in ["muID", "muIso", "muTrig"]:
                    suffix = str(year)
                elif variation_core in ["pu", "l1prefiring"]:
                    suffix = "_wgt"+str(year)
                elif variation_core in ["qgl"]:
                    suffix = "_wgt"



                # TODO: decorrelate LHE, QGL, PDF uncertainties
                variation_fixed = variation.replace("wgt_", "")
                variation_fixed = variation_fixed.replace("_up", f"{suffix}Up")
                variation_fixed = variation_fixed.replace("_down", f"{suffix}Down")
                group_name = group
                name = f"{group_name}_{variation_fixed}"
                logger.info(f"name: {name}")

            # logger.info(f"variation name: {name}")
            # logger.info(f"var_name: {var_name}")
            # logger.info(f"variation_fixed: {variation_fixed}")
            # th1 = from_numpy([group_hist, edges])
            # th1._fName = name
            # th1._fSumw2 = np.array(np.append([0], group_sumw2)) # -> np.array([0, group_sumw2])
            # th1._fTsumw2 = np.array(group_sumw2).sum()
            # th1._fTsumwx2 = np.array(group_sumw2 * centers).sum() #-> this is w*(x**2) distibution
            logger.debug(f"Creating TH1D histogram for group {group} with variation {variation_fixed}")
            th1 = getTH1D_from_numpy(group_hist, edges, group_sumw2, centers, name)
            templates.append(th1)

            yield_rows.append(
                {
                    "var_name": var_name,
                    "group": group,
                    "region": region,
                    "channel": channel,
                    "year": year,
                    "variation": variation_fixed,
                    "yield": group_hist.sum(),
                }
            )

    if parameters["save_templates"]:
        out_dir = parameters["global_path"]
        # mkdir(out_dir)
        # out_dir += "/" + parameters["label"]
        # mkdir(out_dir)
        out_dir += "/" + "stage3_templates"
        # mkdir(out_dir)
        out_dir += "/" + var.name
        # mkdir(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        # out_fn = f"{out_dir}/{channel}_{region}_{year}.root"
        out_fn = f"{out_dir}/{channel}_{region}_{year_savepath}.root"

        logger.info(f"out_fn: {out_fn}")
        # save_template(templates, out_fn, parameters)

        # Save all histograms to the same ROOT file
        output_file = ROOT.TFile(out_fn, "RECREATE")
        for hist in templates:
            hist.Write()  # Write each histogram to the file
        # Close the file
        output_file.Close()

    yield_df = pd.DataFrame(yield_rows)
    return yield_df
