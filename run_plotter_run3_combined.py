import logging
import sys
from pathlib import Path

from modules.utils import logger
from plotter.validation_plotter_unified import run_bulk_validation

logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# CLI flags (no argparse) -- mirrors run_plotter.py
# -----------------------------------------------------------------------------
ARGS = set(sys.argv[1:])
DRY_RUN = "--dry-run" in ARGS
FORCE = "--force" in ARGS
DEBUG = "--debug" in ARGS

if DEBUG:
    logger.setLevel(logging.DEBUG)

# -----------------------------------------------------------------------------
# User config -- Run-3 combined (all 7 years summed into one plot per combo)
# -----------------------------------------------------------------------------
STAGE1_NAME = "Run3_nanoAODv12_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation_Systematics"
BASE_DIR = Path("/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean")
LOAD_PATH = BASE_DIR / STAGE1_NAME / "stage1_output" / "{year}" / "f1_0"
logger.info(f"Using LOAD_PATH: {LOAD_PATH}")

SAVE_TAG = "incDY"
SAVE_ROOT = Path("./validation/figs") / "Run3_combined" / f"{STAGE1_NAME}_{SAVE_TAG}"
logger.info(f"Using SAVE_ROOT: {SAVE_ROOT}")

years = ["2022preEE", "2022postEE", "2023", "2023BPix", "2024", "2025", "2026"]  # Run3 years, all combined into one plot

categories = ["nocat", "vbf", "ggh", "bJetVeto"]

# See run_plotter.py for the full explanation of how these axes combine into
# Dask passes; unchanged here except for combine_years below.
JJ_ETA_REGIONS = [
    "all",
]

vbf_filter_study_options = [False, True]  # True/False list
remove_zpt_weights_options = [False, True]  # True/False list
add_dnn_zpt_weights_options = [False]  # True/False list
min_set_of_vars = False

region_options = [
    ["h-sidebands", "z-peak"],
]

njets_options = ["inclusive", "0", "1", "2"]

BACKGROUND_SAMPLES = ["EWK", "VV", "TOP", "DY"]
SIG_SAMPLES = ["VBF", "ggH"]
DATA_SAMPLES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

STATUS = "Preliminary"
LINEAR_SCALE = False
USE_GATEWAY = True
CLUSTER_INDEX = 0
USE_COMPACTED = "compacted"  # "", "compacted", "compacted_WithDNNScore"
FORCE_COMPACT = False


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
        force_compact=FORCE_COMPACT,
        combine_years=True,
    )
