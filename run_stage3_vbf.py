"""
Stage-3 for the VBF channel: turns stage-2 DNN-score histograms into Combine shape templates
(stage3_templates_*/) and SR/SB datacards (stage3_datacards_*/) for one year.
`--years 2025_2026` is a merged stage-3 year: it sums the 2025 and 2026 stage-2 histograms
into one set of templates with one set of year-uncorrelated nuisances (PC guideline).

How to run (repo root, `default` pixi env; stage-2 must have run for the year, or for both
2025 and 2026 when using 2025_2026). Usually driven by `run_analysis_pipeline.sh -m 3`:
    ./run_in_pixi.sh default python run_stage3_vbf.py --years <year> -input <save_path> \
        -l <label> --save_postfix <postfix> [--no_variations] [--jj_eta_region <region>] \
        [--vbf_filter_study [--dy_scheme separate]] [--signal_xsec_rescale]

`--signal_xsec_rescale` (pipeline: SIGNAL_XSEC_RESCALE=1) is the interim fix that rescales the
signal for stage-1 made with BR(H->mumu) = 2.6e-4. Cards go to <postfix>_SigXS.

Example:
    WITH_VARIATIONS=1 bash run_analysis_pipeline.sh -m 3 -y 2025_2026 -v 15 -o Sep29_2026 \
        -c configs/datasets/dataset_nanoAODv15_run3.yaml \
        -l Run3_nanoAODv15_FilterEvents_Sep22_tightPassLepVeto_OfficialRecomendation_Systematics
"""
import argparse
import os
import time

from cli.common_argparser import build_common_parser
from modules.utils import logger
from stage3.edit_datacard4DY_matchedJets import (
    has_matched_jet_histograms,
    split_dy_grouping,
    stage2_histogram_directory,
)
from stage3.make_datacards import build_datacards
from stage3.make_templates import to_templates
from modules.classify_year import component_years, is_run2
from modules.sample_config import get_all_dicts
from omegaconf import OmegaConf
parser = build_common_parser()
parser.add_argument(
    "-nv",
    "--no_variations",
    dest="no_variations",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="If true, runs with all variations, otherwise only nominal",
)
parser.add_argument(
    "--jj_eta_region",
    dest="jj_eta_region",
    default="all",
    action="store",
    help=(
        "Must match the --jj_eta_region a stage-2 run for this label/save_postfix was built "
        "with (default 'all'): only used to locate the matching stage2_histograms/stage3_ "
        "datacards output directory."
    ),
)
parser.add_argument(
    "--dy_scheme",
    dest="dy_scheme",
    default="matched",
    choices=["matched", "separate"],
    help=(
        "With matched-jet DY histograms: 'matched' (default) merges all DY samples into "
        "DY_matched01J/DY_matched2J; 'separate' keeps one process per sample group (DY, "
        "DYVBF with --vbf_filter_study), each with its own rateParam, and writes to "
        "<save_postfix>_DYsep."
    ),
)
parser.add_argument(
    "--signal_xsec_rescale",
    dest="signal_xsec_rescale",
    default=False,
    action="store_true",
    help=(
        "Interim fix: rescale ggh_powhegPS/vbf_powheg templates from the BR(H->mumu) = 2.6e-4 "
        "used in Run-3 stage-1 to the LHCHXSWG sigma x BR (stage3/make_templates.py "
        "SIGNAL_XSEC_BR_RESCALE). Only for stage-1 made before the dataset-YAML fix; writes to "
        "<save_postfix>_SigXS."
    ),
)
args = parser.parse_args()

years = args.years if args.years else [args.year]

year = years[0]

stage2_model_suffix = args.save_postfix if args.save_postfix else ""
# 'separate' reads the same stage-2 histograms but must not overwrite the 'matched' cards
stage3_output_suffix = (
    f"{stage2_model_suffix}_DYsep" if args.dy_scheme == "separate" else stage2_model_suffix
)
if args.signal_xsec_rescale:
    if any(is_run2(comp) for y in years for comp in component_years(y)):
        raise ValueError("--signal_xsec_rescale only applies to Run-3 years (BR 2.6e-4 issue).")
    stage3_output_suffix = f"{stage3_output_suffix}_SigXS" if stage3_output_suffix else "SigXS"
for _tag, _on in (("vbf_filter_study", args.do_vbf_filter_study),
                  (args.jj_eta_region, args.jj_eta_region != "all")):
    if _on:
        stage2_model_suffix = f"{stage2_model_suffix}_{_tag}" if stage2_model_suffix else _tag
        stage3_output_suffix = f"{stage3_output_suffix}_{_tag}" if stage3_output_suffix else _tag

# global parameters
parameters = {
    # < general settings >
    "log_level": args.log_level,
    "years": years,
    "global_path": args.input_path,
    "global_path_postfix": stage2_model_suffix,
    "outpath_postfix": stage3_output_suffix,
    "label": args.label,
    "channels": ["vbf"],
    "regions": ["h-peak", "h-sidebands"],
    "no_variations": args.no_variations,
    "signal_xsec_rescale": args.signal_xsec_rescale,
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

# A merged year (e.g. 2025_2026) sums its component years' stage-2 histograms, so it
# takes the union of their sample groupings; a process must map to one group in all.
parameters["grouping"] = {}
for comp_year in component_years(year):
    _, _, comp_grouping = get_all_dicts(yaml_path=args.sample_config, year=comp_year)
    for dataset, group in comp_grouping.items():
        if parameters["grouping"].setdefault(dataset, group) != group:
            raise ValueError(
                f"{dataset} is grouped as {parameters['grouping'][dataset]} and {group} "
                f"in the component years of {year}; stage-3 can't sum it into one template."
            )

# Whether alpha_s is emitted as its own `alpha_s_unc` nuisance next to `pdf_unc`, or
# combined with it into `pdf_alpha_s_unc`. Per era, from stage3/VBF/switches.yaml
# (stage3-only switches, kept out of configs/parameters/); stage3/make_templates.py
# raises if an era is missing.
stage3_switches = OmegaConf.load(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "stage3", "VBF", "switches.yaml"
    )
)["switches"]
parameters["split_pdf_alpha_s"] = {
    y: bool(stage3_switches["split_pdf_alpha_s"][y]) for y in years
}

stage2_histogram_paths = [
    stage2_histogram_directory(
        args.input_path,
        f"score_{args.label}",
        stage2_model_suffix,
        args.no_variations,
        comp_year,
    )
    for comp_year in component_years(year)
]
dy_split_per_path = [has_matched_jet_histograms(p) for p in stage2_histogram_paths]
if len(set(dy_split_per_path)) != 1:
    raise ValueError(
        f"The component years of {year} disagree on the DY matched-jet split "
        f"({dict(zip(map(str, stage2_histogram_paths), dy_split_per_path))}); rerun stage-2 "
        f"with the same divide_dy_into_matched_jets setting for all of them."
    )
divide_dy_into_matched_jets = dy_split_per_path[0]
stage2_histogram_path = ", ".join(map(str, stage2_histogram_paths))
if args.dy_scheme == "separate" and not divide_dy_into_matched_jets:
    raise ValueError("--dy_scheme separate needs stage-2 histograms split into matched jets.")
dy_separate = args.dy_scheme == "separate"
# rateParams follow the processes: DY_matched01J/2J for 'matched', DY/DYVBF for 'separate'
parameters["divide_dy_into_matched_jets"] = divide_dy_into_matched_jets and not dy_separate
parameters["dy_separate_rate_params"] = dy_separate
if divide_dy_into_matched_jets:
    parameters["grouping"] = split_dy_grouping(
        parameters["grouping"], keep_sample_groups=dy_separate
    )
logger.info(
    "divide_dy_into_matched_jets=%s (stage2 directory: %s)",
    divide_dy_into_matched_jets,
    stage2_histogram_path,
)

parameters["plot_groups"] = {
    "stack": (
        ["DY_matched01J", "DY_matched2J", "EWK", "TT+ST", "VV", "VVV"]
        if parameters["divide_dy_into_matched_jets"]
        else ["DY", "DYVBF", "EWK", "TT+ST", "VV", "VVV"]
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
