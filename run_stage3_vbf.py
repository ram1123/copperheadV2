import argparse
import time

from cli.common_argparser import build_common_parser
from modules.utils import logger
from stage3.dy_matched_jets import (
    has_matched_jet_histograms,
    split_dy_grouping,
    stage2_histogram_directory,
)
from stage3.make_datacards import build_datacards
from stage3.make_templates import to_templates
from modules.sample_config import get_all_dicts
parser = build_common_parser()
parser.add_argument(
    "-nv",
    "--no_variations",
    dest="no_variations",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="If true, runs with all variations, otherwise only nominal",
)
args = parser.parse_args()

years = args.years if args.years else [args.year]

# Use the exact year string (e.g. "2016preVFP", "2016postVFP") to resolve
# sample groups, instead of collapsing to a bare "2016" that matches no
# `processes_per_year` key in samples.yaml and silently falls back to
# group defaults that may not exist on disk (e.g. DY's "dyTo2L_M-50_incl",
# EWK's "ewk_mmjj_mll_105_160" for 2016preVFP/2016postVFP).
year = years[0]

stage2_model_suffix = args.save_postfix if args.save_postfix else ""
if args.do_vbf_filter_study:
    stage2_model_suffix = (
        f"{stage2_model_suffix}_vbf_filter_study"
        if stage2_model_suffix
        else "vbf_filter_study"
    )

# global parameters
parameters = {
    # < general settings >
    "log_level": args.log_level,
    "years": years,
    "global_path": args.input_path,
    "global_path_postfix": stage2_model_suffix,
    "outpath_postfix": stage2_model_suffix,
    "label": args.label,
    "channels": ["vbf"],
    "regions": ["h-peak", "h-sidebands"],
    "no_variations": args.no_variations,
    "syst_variations": ["nominal"],
    # "syst_variations": ['nominal', 'Absolute', 'Absolute2018', 'BBEC1', 'BBEC12018', 'EC2', 'EC22018', 'HF', 'HF2018', 'RelativeBal', 'RelativeSample2018', 'FlavorQCD', 'jer1', 'jer2', 'jer3', 'jer4', 'jer5', 'jer6', ],
    # "syst_variations": ['nominal', 'Absolute', f'Absolute_{year}', 'BBEC1', f'BBEC1_{year}', 'EC2', f'EC2_{year}', 'HF', f'HF_{year}', 'RelativeBal', f'RelativeSample_{year}', 'FlavorQCD', 'jer1', 'jer2', 'jer3', 'jer4', 'jer5', 'jer6', ],
    # < plotting settings >
    "plot_vars": [],  # "dimuon_mass"],
    # "variables_lookup": variables_lookup,
    "dnn_models": {
        "vbf": [args.label],
    },
    "bdt_models": {},
    #
    # < templates and datacards >
    "save_templates": True,
    "templates_vars": [],  # "dimuon_mass"],
}

_, _, parameters["grouping"] = get_all_dicts(
    yaml_path=args.sample_config,
    year=year,
)

stage2_histogram_path = stage2_histogram_directory(
    args.input_path,
    f"score_{args.label}",
    stage2_model_suffix,
    args.no_variations,
    year,
)
divide_dy_into_matched_jets = has_matched_jet_histograms(stage2_histogram_path)
parameters["divide_dy_into_matched_jets"] = divide_dy_into_matched_jets
if divide_dy_into_matched_jets:
    parameters["grouping"] = split_dy_grouping(parameters["grouping"])
logger.info(
    "divide_dy_into_matched_jets=%s (stage2 directory: %s)",
    divide_dy_into_matched_jets,
    stage2_histogram_path,
)

parameters["plot_groups"] = {
    "stack": (
        ["DY_matched01J", "DY_matched2J", "EWK", "TT+ST", "VV", "VVV"]
        if divide_dy_into_matched_jets
        else ["DY", "EWK", "TT+ST", "VV", "VVV"]
    ),
    "step": ["VBF", "ggH"],
    "errorbar": ["Data"],
}


if __name__ == "__main__":
    start_time = time.time()

    # add MVA scores to the list of variables to plot
    dnn_models = list(parameters["dnn_models"].values())
    bdt_models = list(parameters["bdt_models"].values())
    for models in dnn_models + bdt_models:
        for model in models:
            parameters["plot_vars"] += ["score_" + model]
            parameters["templates_vars"] += ["score_" + model]

    parameters["datasets"] = parameters["grouping"].keys()
    logger.info(f"parameters: {parameters}")

    # save templates to ROOT files
    yield_df = to_templates(parameters)
    logger.info(f'run stage3 yield_df: {yield_df}')
    if yield_df is None or yield_df.empty:
        logger.error("Yield DataFrame is empty. Cannot build datacards.")
        raise ValueError("Yield DataFrame is empty. Cannot build datacards.")

    # For sanity check save the yield_df to a CSV file
    yield_df.to_csv(f"yield_df_{parameters['label']}_{parameters['outpath_postfix']}.csv", index=False)

    datacard_str = parameters["dnn_models"]["vbf"][0]
    logger.info(f"datacard_str: {datacard_str}")

    # make datacards
    build_datacards(f"score_{datacard_str}", yield_df, parameters)
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the elapsed time
    logger.info(f"Execution time: {execution_time:.4f} seconds")
