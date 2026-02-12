import itertools
import logging
import subprocess
import sys
from pathlib import Path
from shlex import join as shjoin

from modules.trials import get_stage1_path
from modules.utils import logger

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
BASE_SCRIPT = ["python", "plotter/validation_plotter_unified.py"]

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

SAVE_TAG = "incDY_11Feb"
SAVE_ROOT = Path("./validation/figs") / outputDir / f"{stage1_name}_{SAVE_TAG}"
logger.info(f"Using SAVE_ROOT: {SAVE_ROOT}")

# Prevent overwriting
if SAVE_ROOT.exists() and not FORCE:
    raise RuntimeError(f"SAVE_ROOT exists: {SAVE_ROOT} (use --force to proceed)")

# years = ["2022preEE", "2022postEE", "2023", "2023BPix", "2024"]
years = ["2022preEE"]
# years = ["2022postEE"]
# years = ["2023"]
# years = ["2023BPix"]
# years = ["2024"]

# categories = ["nocat", "vbf", "ggh"]
# categories = ["ggh"]
# categories = ["vbf"]
categories = ["nocat"]

# Boolean flags
vbf_filter_study_options = [False]  # True/False list
remove_zpt_weights_options = [False]  # True/False list
min_set_of_vars = False  # minimal set of vars

region_options = [
    ["h-sidebands", "z-peak"],
]

njets_options = ["inclusive"]
# njets_options = ["inclusive", "0", "1", "2"]
# njets_options = ["0", "1", "2"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def build_command(
    year: str,
    save_path: Path,
    load_path: Path,
    cat: str,
    vbf_filter_study: bool,
    remove_zpt_weights: bool,
    region: list[str],
    njets: str,
) -> list[str]:
    cmd = (
        BASE_SCRIPT
        + ["-y", year]
        + ["--save_path", str(save_path)]
        + ["--load", str(load_path)]
        + ["-cat", cat]
        + ["--use-compacted", "compacted"]  # "", "compacted", "compacted_WithDNNScore"
        + ["--use_gateway", "--cluster_index", "0"]
        + ["--njets", str(njets)]
    )

    if DEBUG:
        cmd += ["--log-level", "DEBUG"]

    if min_set_of_vars:
        cmd += ["--minimum_set"]

    if vbf_filter_study:
        cmd += ["--vbf_filter_study"]

    if region:
        cmd += ["--region", *region]

    if remove_zpt_weights:
        cmd += ["--remove_zpt_weights"]

    return cmd


def run_all_combos():
    job_idx = 0

    for year in years:
        load_path = Path(str(LOAD_PATH).format(year=year))

        combo_iter = itertools.product(
            categories,
            vbf_filter_study_options,
            remove_zpt_weights_options,
            region_options,
            njets_options,
        )

        for cat, vbf_flag, zpt_flag, region_list, njets in combo_iter:
            # skip meaningless combos
            if cat == "vbf" and njets != "inclusive":
                logger.debug(f"Skipping vbf with njets={njets} (not meaningful)")
                continue

            # if vbf_filter_study: remove z-peak from region
            region = [r for r in region_list if not (vbf_flag and r == "z-peak")]

            # structured save dirs (much easier to browse)
            save_path = (
                SAVE_ROOT
                / f"VBFfilter_{vbf_flag}"
                # / year
                # / cat
                # / f"njets_{njets}"
                # / f"zptRemoved_{zpt_flag}"
            )
            save_path.mkdir(parents=True, exist_ok=True)

            cmd = build_command(
                year=year,
                save_path=save_path,
                load_path=load_path,
                cat=cat,
                vbf_filter_study=vbf_flag,
                remove_zpt_weights=zpt_flag,
                region=region,
                njets=njets,
            )

            job_idx += 1
            logger.info("\n" + "=" * 80)
            logger.info(
                f"[{job_idx:04d}] {year} {cat} njets={njets} vbf={vbf_flag} zptRm={zpt_flag}"
            )
            logger.info(shjoin(cmd))

            if not DRY_RUN:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run_all_combos()
