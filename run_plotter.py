import subprocess
import itertools
import logging
from modules.utils import logger

logger.setLevel(logging.INFO)


base_script = ["python", "plotter/validation_plotter_unified.py"]
SAVE_PATH = "./validation/figs/Run2_nanoAODv12_08June/RamCode_VBFFilterSeparate/"

LOAD_PATH = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/stage1_output/{year}/f1_0/"

# years = ["2016preVFP", "2016postVFP", "2017", "2018"]
years = ["2018"]

# categories = ["ggh", "vbf", "nocat"]
# categories = ["vbf", "ggh"]
categories = ["vbf"]

# Boolean flags
vbf_filter_study_options = [True]
remove_zpt_weights_options = [False]

region_options = [
    ["h-sidebands", "z-peak", "signal", "h-peak"]
]
njets_options = [None, 0, 1, 2]  # None = not set

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

    if vbf_filter_study:
        cmd.append("--vbf_filter_study")

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
        save_path = f"{SAVE_PATH}"
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
            if cat == "ggh" and vbf_flag:
                logger.debug(f"Skipping ggh with vbf_filter_study: {i}")
                continue  # skip --vbf_filter_study for ggh, not meaningful

            if cat == "vbf" and njets is not None:
                logger.debug(f"Skipping vbf with njets: {i}")
                continue  # skip njets for vbf, not meaningful

            cmd = build_command(year, save_path, load_path, cat, vbf_flag, zpt_flag, region, njets)
            logger.info(f"[{year}][{cat}][{i}] Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_all_combos()
