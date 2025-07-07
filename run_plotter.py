import subprocess
import itertools
import logging
from modules.utils import logger
from pathlib import Path
import sys

logger.setLevel(logging.INFO)

DRY_RUN = len(sys.argv) > 1 and sys.argv[1] == "--dry-run"

base_script = ["python", "plotter/validation_plotter_unified.py"]
# base_script = ["python", "validation_plotter_unified.py"]

# SAVE_PATH = "./validation/figs/Run2_nanoAODv12_08June/CrossCheck_MiNNLO/"
# SAVE_PATH = "./validation/figs/Run2_nanoAODv12_08June/CrossCheck_aMCatNLO/"
# SAVE_PATH = "./validation/figs/Run2_nanoAODv12_08June/CrossCheck_VBFFilterAndAMCATNLO_MjjCutForAllCats/PUjetVeto50GeV_2p5_4p0_Jet70GeV/"
# SAVE_PATH = "./validation/figs/Run2_nanoAODv12_08June/CrossCheck_VBFFilterAndAMCATNLO_7July/SwitchedCutToHyeonCode_AddedbackToFun_InvertgJJcut/"
SAVE_PATH = "./validation/figs/Run2_nanoAODv12_08June/CrossCheck_VBFFilterAndAMCATNLO_7July/UpdatedVBFFilterMjjCut/"
# SAVE_PATH = "./validation/figs/Run2_nanoAODv12_08June/CrossCheck_VBFFilterOnly_diMuonLess100GeV/"
# SAVE_PATH = "./validation/figs/Run2_nanoAODv12_08June/CrossCheck_VBFFilterPlusaMCatNLO/"
# SAVE_PATH = "./validation/figs/Run2_nanoAODv12_08June/CrossCheck_VBFFilterPlusaMCatNLO_NoMjjCut/"

# LOAD_PATH = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/stage1_output/{year}/f1_0/"
LOAD_PATH = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/stage1_output/{year}/compacted/"
# years = ["2016preVFP", "2016postVFP", "2017", "2018"]
years = ["2018"]

# categories = ["vbf", "ggh", "nocat"]
# categories = ["nocat", "ggh"]
# categories = ["vbf", "ggh"]
categories = ["vbf"]
# categories = ["nocat"]

# Boolean flags
vbf_filter_study_options = [True]
remove_zpt_weights_options = [False]
debug_options = False
min_set_of_vars = False  # If True, only use a minimal set of variables  to plot

region_options = [
    # ["h-sidebands", "z-peak", "signal", "h-peak"]
    ["h-sidebands", "signal", "h-peak"]
    # ["h-sidebands", "z-peak"]
    # ["h-sidebands"]
]
# njets_options = ["inclusive", "0", "1", "2"]  # inclusive = No cut on nJets
# njets_options = ["inclusive", "0", "1"]  # inclusive = No cut on nJets
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
         "--use_gateway"
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
    i = 0
    for year in years:
        # save_path = f"{SAVE_PATH}"
        load_path = LOAD_PATH.format(year=year)
        combo_iter = itertools.product(
            categories,
            vbf_filter_study_options,
            remove_zpt_weights_options,
            region_options,
            njets_options
        )
        for cat, vbf_flag, zpt_flag, region, njets in combo_iter:
            i += 1
            save_path = str(Path(f"{SAVE_PATH}") / f"VBFfilter_{vbf_flag}")
            # if cat == "ggh" and vbf_flag:
            #     logger.debug(f"Skipping ggh with vbf_filter_study: {i}")
            #     continue  # skip --vbf_filter_study for ggh, not meaningful

            if cat == "vbf" and (not (njets == "inclusive" or njets == "2")):
                logger.info(f"Skipping vbf with njets: {njets}")
                continue  # skip njets for vbf, not meaningful

            cmd = build_command(year, save_path, load_path, cat, vbf_flag, zpt_flag, region, njets)
            logger.info(f"[{year}][{cat}][{i}] Running: {' '.join(cmd)}")
            if not DRY_RUN:
                subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_all_combos()
