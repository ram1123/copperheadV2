import logging
import sys
from pathlib import Path

from modules.trials import get_stage1_path
from modules.utils import logger
from plotter.validation_plotter_unified import run_bulk_validation

logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# CLI flags (no argparse)
# -----------------------------------------------------------------------------
ARGS = set(sys.argv[1:])
DRY_RUN = "--dry-run" in ARGS
# Bypass the `_status` done markers under SAVE_ROOT/_status/ and recompute every
# (category, njets) sub-pass from scratch, e.g. after adding new samples to an
# already-"done" combo (the done markers are file-existence-based and can't
# detect that on their own -- mirrors run_stage1.py's --rerun).
FORCE = "--force" in ARGS
DEBUG = "--debug" in ARGS

if DEBUG:
    logger.setLevel(logging.DEBUG)

# -----------------------------------------------------------------------------
# User config
# -----------------------------------------------------------------------------
stage1_dir = Path(get_stage1_path())  # default = "current"

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

# years = ["2022preEE", "2022postEE", "2023", "2023BPix", "2024"] # Run3 years
years = ["2018", "2017", "2016postVFP", "2016preVFP"] # Run2 years

categories = ["nocat", "vbf", "ggh", "bJetVeto"]


# year x category x njets x zpt_option are computed together in ONE Dask pass per
# (jj_eta_region, vbf_filter_study, region_list) "scope" -- see run_bulk_validation()
# / ValidationHistProcessor in validation_plotter_unified.py. jj_eta_region,
# vbf_filter_study, and region_options can each list multiple values; run_bulk_validation()
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
vbf_filter_study_options = [False, True]  # True/False list
remove_zpt_weights_options = [False, True]  # True/False list
add_dnn_zpt_weights_options = [False]  # True/False list
min_set_of_vars = False  # minimal set of vars

region_options = [
    ["h-sidebands", "z-peak"],
]

njets_options = ["inclusive", "0", "1", "2"]

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


if __name__ == "__main__":
    run_bulk_validation(
        years=years,
        categories=categories,
        njets_options=njets_options,
        jj_eta_regions=JJ_ETA_REGIONS,
        vbf_filter_study_options=vbf_filter_study_options,
        remove_zpt_weights_options=remove_zpt_weights_options,
        add_dnn_zpt_weights_options=add_dnn_zpt_weights_options,
        region_options=region_options,
        min_set_of_vars=min_set_of_vars,
        load_path_template=LOAD_PATH,
        save_root=SAVE_ROOT,
        background_samples=BACKGROUND_SAMPLES,
        sig_samples=SIG_SAMPLES,
        data_samples=DATA_SAMPLES,
        status=STATUS,
        linear_scale=LINEAR_SCALE,
        use_gateway=USE_GATEWAY,
        cluster_index=CLUSTER_INDEX,
        use_compacted=USE_COMPACTED,
        dry_run=DRY_RUN,
        force_rerun=FORCE,
    )
