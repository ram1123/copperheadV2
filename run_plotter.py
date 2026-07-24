import itertools
import logging
import sys
from pathlib import Path

from modules.dask_utils import close_dask_client, get_dask_client
from modules.trials import get_stage1_path
from modules.utils import logger
from configs.variables.variable_lists import get_all_vars
from plotter.validation_plotter_unified import (
    DATASET_SEPARATOR,
    ValidationHistProcessor,
    build_fileset_for_year,
    build_hist_templates,
    generate_combo_plots,
    group_dict as base_group_dict,
    load_plot_settings,
    resolve_year_context,
    run_validation_runner,
)

logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# CLI flags (no argparse)
# -----------------------------------------------------------------------------
ARGS = set(sys.argv[1:])
DRY_RUN = "--dry-run" in ARGS
FORCE = "--force" in ARGS
DEBUG = "--debug" in ARGS

if DEBUG:
    logger.setLevel(logging.DEBUG)

# -----------------------------------------------------------------------------
# User config
# -----------------------------------------------------------------------------
# stage1_dir = Path(get_stage1_path())  # default = "current"
# stage1_dir = Path("/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_June25PR_test/stage1_output")  # default = "current"
# stage1_dir = Path("/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_June28Compact_test/stage1_output")  # default = "current"
# stage1_dir = Path("/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_forVBFChannel_June26_2026_jetUnc/stage1_output")  # default = "current"
stage1_dir = Path("/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_forVBFChannel_July06_2026_jetUncRedo/stage1_output")  # default = "current"

LOAD_PATH = stage1_dir / "{year}" / "f1_0"
logger.info(f"Using LOAD_PATH: {LOAD_PATH}")

# Get the main directory based on run2 and nanoAOD version from the load path, e.g. Run3_nanoAODv12
# Example: Run3_nanoAODv12_02Feb_FilterJetsHorn30GeV -> Run3_nanoAODv12
stage1_name = stage1_dir.parent.name
parts = stage1_name.split("_")
logger.info(f"Parsed stage1_name: {stage1_name}, parts: {parts}")
outputDir = "_".join(parts[:2]) if len(parts) >= 2 else stage1_name
logger.info(f"outputDir: {outputDir}")

SAVE_TAG = "incDY"
# SAVE_TAG = "incDY_DNNWgts"
SAVE_ROOT = Path("./validation/figs") / outputDir / f"{stage1_name}_{SAVE_TAG}"
logger.info(f"Using SAVE_ROOT: {SAVE_ROOT}")

# # Prevent overwriting
# if SAVE_ROOT.exists() and not FORCE:
#     raise RuntimeError(f"SAVE_ROOT exists: {SAVE_ROOT} (use --force to proceed)")

# years = ["2022preEE", "2022postEE", "2023", "2023BPix", "2024"]
# years = ["2022preEE"]
# years = ["2018"]
# years = ["2016preVFP"]
# years = ["2016postVFP", "2016preVFP"]
years = ["2018", "2017", "2016postVFP", "2016preVFP"]
# years = ["2017"]
# years = ["2023"]
# years = ["2023BPix"]
# years = ["2024"]

# categories = ["nocat", "vbf", "ggh"]
# categories = ["nocat", "vbf", "ggh", "bJetVeto"]
# categories = ["nocat", "ggh"]
# categories = ["ggh"]
# categories = ["vbf"]
categories = ["nocat"]

# year x category x njets x zpt_option are computed together in ONE Dask pass per
# (jj_eta_region, vbf_filter_study, region_list) "scope" -- see
# ValidationHistProcessor in validation_plotter_unified.py. jj_eta_region,
# vbf_filter_study, and region_options can each list multiple values; run_all_combos()
# loops over their product, running one Dask pass (reusing the same client) per scope,
# since each of these three changes which files/processes get read or which regions
# get filled -- unlike year/category/njets/zpt_option, they can't be folded into a
# single shared Hist axis.
JJ_ETA_REGIONS = [
    "all",
#     "jj_both_central",
#     "jj_non_central",
#     "jj_one_fwd25_one_central",
#     "jj_one_he_one_central",
#     "jj_one_fwd30_one_central",
#     "jj_both_fwd25",
#     "jj_both_he",
#     "jj_both_fwd30",
#     "jj_one_he_one_fwd30",
]

# Boolean flags
# vbf_filter_study_options = [False, True]  # True/False list
vbf_filter_study_options = [False]  # True/False list
remove_zpt_weights_options = [False, True]  # True/False list
add_dnn_zpt_weights_options = [False]  # True/False list
min_set_of_vars = False  # minimal set of vars

region_options = [
    ["h-sidebands", "z-peak"],
    # ["z-peak"],
    # ["h-sidebands"],
]

njets_options = ["inclusive"]
# njets_options = ["inclusive", "0", "1", "2"]
# njets_options = ["inclusive", "0", "1", "2"]
# njets_options = [ "2"]

# background/signal/data shorthand lists match validation_plotter_unified.py's
# own CLI defaults -- run_plotter.py never overrode these via subprocess flags.
BACKGROUND_SAMPLES = ["EWK", "VV", "TOP", "DY"]
SIG_SAMPLES = ["VBF", "ggH"]
DATA_SAMPLES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

STATUS = "Preliminary"
LINEAR_SCALE = False
USE_GATEWAY = True
CLUSTER_INDEX = 0
USE_COMPACTED = "compacted"  # "", "compacted", "compacted_WithDNNScore"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _derive_zpt_options(remove_zpt_weights_options, add_dnn_zpt_weights_options):
    zpt_options = []
    if False in remove_zpt_weights_options and False in add_dnn_zpt_weights_options:
        zpt_options.append("default")
    if True in remove_zpt_weights_options:
        zpt_options.append("no_zpt")
    if True in add_dnn_zpt_weights_options:
        zpt_options.append("dnn_zpt")
    if not zpt_options:
        raise ValueError("No valid zpt_option combination derived from run_plotter.py config.")
    return zpt_options


def _run_scope(jj_eta_region, do_vbf_filter_study, region_list, client):
    """
    Run one consolidated Dask pass (year x category x njets x zpt_option, all
    computed together) for a single fixed (jj_eta_region, vbf_filter_study,
    region_list) scope. `client` is None during a --dry-run.
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

    save_path = SAVE_ROOT / f"VBFfilter_{do_vbf_filter_study}"
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

    sample_groups = list(base_group_dict.keys()) + ["other"]
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

    if DRY_RUN:
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
        load_path = Path(str(LOAD_PATH).format(year=year))
        year_ctx = resolve_year_context(
            year, BACKGROUND_SAMPLES, SIG_SAMPLES, DATA_SAMPLES, do_vbf_filter_study, lumi_override=""
        )
        group_dict_by_year[year] = year_ctx["group_dict"]
        lumi_by_year[year] = year_ctx["lumi"]
        CM_energy_by_year[year] = year_ctx["CM_energy"]
        logger.info(f"{year} available_processes: {year_ctx['available_processes']}")

        year_fileset = build_fileset_for_year(
            year, load_path, year_ctx["available_processes"], USE_COMPACTED
        )
        for process, entry in year_fileset.items():
            fileset[f"{year}{DATASET_SEPARATOR}{process}"] = entry
    logger.info(f"finished building fileset! ({len(fileset)} datasets across {len(years)} year(s))")

    logger.info("{style}Filling histograms.{style}".format(
        style="\n" + "=" * 50 + "\n",))
    # Reduce eagerly as each dataset's result comes back (running total per
    # (year, category, var), not a growing list summed at plot time) -- otherwise
    # every process's histogram from every (category, njets) sub-pass stays live
    # in memory simultaneously until the plotting loop finally sums them all at
    # the end, which is its own (separate, driver-side) memory blowup on top of
    # the per-Runner-call one below.
    sample_hist_lookup = {
        year: {cat: {var: None for var in hist_templates_by_category[cat]} for cat in categories}
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
                    if sample_hist_lookup[result_year][category][var] is None:
                        sample_hist_lookup[result_year][category][var] = h
                    else:
                        sample_hist_lookup[result_year][category][var] += h

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
                                STATUS,
                                CM_energy_by_year[year],
                                not LINEAR_SCALE,
                                do_vbf_filter_study,
                                jj_eta_region,
                            )
    logger.info(f"Generated plots for {job_idx} (year, category, njets, zpt_option) combos.")


def run_all_combos():
    scopes = list(itertools.product(JJ_ETA_REGIONS, vbf_filter_study_options, region_options))
    logger.info(f"Running {len(scopes)} (jj_eta_region, vbf_filter_study, region_list) scope(s).")

    client = None
    if not DRY_RUN:
        client = get_dask_client(USE_GATEWAY, cluster_index=CLUSTER_INDEX)
        logger.info(f"client: {client}")

    for jj_eta_region, do_vbf_filter_study, region_list in scopes:
        _run_scope(jj_eta_region, do_vbf_filter_study, region_list, client)

    if not DRY_RUN:
        close_dask_client()


if __name__ == "__main__":
    run_all_combos()
