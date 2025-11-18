import subprocess
import itertools
import logging
from pathlib import Path
import sys

from modules.utils import logger
from modules.trials import get_stage1_path

logger.setLevel(logging.INFO)

DRY_RUN = len(sys.argv) > 1 and sys.argv[1] == "--dry-run"

base_script = ["python", "plotter/validation_plotter_unified.py"]

stage1_dir = get_stage1_path()  # default = "current"
LOAD_PATH = str(Path(stage1_dir) / "{year}" / "f1_0")
logger.info(f"Using LOAD_PATH: {LOAD_PATH}")

SAVE_PATH = f"./validation/figs/Run3_nanoAODv12_23October/{LOAD_PATH.split('/')[-4]}/"
logger.info(f"Using SAVE_PATH: {SAVE_PATH}")

# years = ["2018", "2017", "2016postVFP", "2016preVFP", "2016", "*"]
# years = ["2022preEE"]
# years = ["2022postEE"]
# years = ["2022preEE", "2022postEE", "2023", "2023BPix"]
# years = ["2022postEE", "2023", "2023BPix"]
# years = ["2023", "2023BPix"]
years = ["2023"]
# years = ["*"]

# categories = ["vbf", "ggh", "nocat"]
# categories = ["nocat", "ggh"]
# categories = ["vbf", "ggh"]
# categories = ["vbf"]
categories = ["nocat"]

# Boolean flags
vbf_filter_study_options = [False]  # True to apply VBF filter study, False to skip it
remove_zpt_weights_options = [False]  # True to remove zpt weights, False to keep them
debug_options = False
min_set_of_vars = False  # If True, only use a minimal set of variables  to plot

region_options = [
    # ["h-sidebands", "z-peak", "signal", "h-peak"]
    # ["h-sidebands", "signal", "h-peak"]
    ["h-sidebands", "z-peak"]
    # ["z-peak"]
    # ["h-sidebands"]
]
# njets_options = ["inclusive", "0", "1", "2"]  # inclusive = No cut on nJets
# njets_options = ["0", "1", "2"]  # inclusive = No cut on nJets
# njets_options = [ "0", "1"]  # inclusive = No cut on nJets
njets_options = ["inclusive"]  # inclusive = No cut on nJets
# njets_options = ["2"]  # inclusive = No cut on nJets

def build_command(year, save_path, load_path, cat, vbf_filter_study, remove_zpt_weights, region, njets):
    cmd = (
        base_script +
        ["-y", year,
         "--save_path", save_path,
         "--load", load_path,
         "-cat", cat,
         "--use-compacted", "compacted",  # options: "", "compacted", "compacted_WithDNNScore"
         "--use_gateway",
        #  "--dnn-score"
         ]
    )

    if debug_options:
        cmd += ["--log-level",  "DEBUG"]

    if min_set_of_vars:
        cmd += ["--minimum_set"]

    if vbf_filter_study:
        cmd += ["--vbf_filter_study"]

    if region:
        cmd += ["--region"] + region
    if njets is not None:
        cmd += ["--njets", str(njets)]

    if remove_zpt_weights:
        cmd.append("--remove_zpt_weights")

    return cmd

def run_all_combos():
    i = 1
    for year in years:
        # save_path = f"{SAVE_PATH}"
        if year == "2016":
            load_path = LOAD_PATH.format(year=str(year)+"*")
        else:
            load_path = LOAD_PATH.format(year=year)
        combo_iter = itertools.product(
            categories,
            vbf_filter_study_options,
            remove_zpt_weights_options,
            region_options,
            njets_options
        )
        for cat, vbf_flag, zpt_flag, region, njets in combo_iter:
            save_path = str(Path(f"{SAVE_PATH}") / f"VBFfilter_{vbf_flag}")
            # if cat == "ggh" and vbf_flag:
            #     logger.debug(f"Skipping ggh with vbf_filter_study: {i}")
            #     continue  # skip --vbf_filter_study for ggh, not meaningful

            if cat == "vbf" and (not (njets == "inclusive" )):
                logger.debug(f"Skipping vbf with njets: {njets}")
                continue  # skip njets for vbf, not meaningful

            # skip if vbf_filter_study_options is True then remove "z-peak" from region
            if vbf_flag and "z-peak" in region:
                region = [r for r in region if r != "z-peak"]
                logger.debug(f"Removing 'z-peak' from region for vbf_filter_study: {region}")

            cmd = build_command(year, save_path, load_path, cat, vbf_flag, zpt_flag, region, njets)
            logger.info(f"[{year}][{cat}][{i}] Running: {' '.join(cmd)}")
            if not DRY_RUN:
                subprocess.run(cmd, check=True)
            i += 1

if __name__ == "__main__":
    run_all_combos()
