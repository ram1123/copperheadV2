#!/bin/bash
# Regenerate missing/corrupt diagnostic plots for the EBE mass resolution calibration
# (src/lib/ebeMassResCalibration/). A category is "missing" when it has a row in that
# output dir's calibration_factors.csv but no calibration_fitCat<cat>.png next to it --
# e.g. because the category's mass array was cached to mass_<cat>.npy by an earlier
# --fixCat subprocess (see run_step1_categories_isolated in getCalibrationFactor.py) but
# the plot itself never got written (crashed before the PNG save, or ran before the
# PDF->PNG fix and only has an old corrupt .pdf).
#
# Scans every validation/ebeMassResCalibration/<label>/binned/<year>/<MC|Data>_<year>_V1/
# directory that already has a calibration_factors.csv, finds categories with no PNG,
# and re-invokes getCalibrationFactor.py --fixCat <cat> --steps step1 for each -- this
# hits the cached mass_<cat>.npy (fast, no re-read of the input skim) and re-fits+re-plots
# just that one category, in its own subprocess (crash-isolated, like the main pipeline).
#
# Usage:
#   bash .claude/scripts/regen_missing_calibration_plots.sh
#   bash .claude/scripts/regen_missing_calibration_plots.sh --dry-run   # just list what's missing
#
# Must be run inside the `default` pixi environment (ROOT/RooFit + coffea stack), e.g.:
#   cd /cvmfs/cms-af.opensciencegrid.org/paf/pixi/copperheadV2
#   CONDA_OVERRIDE_CUDA=12.4 pixi run -e default bash /path/to/repo/.claude/scripts/regen_missing_calibration_plots.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

BASE="validation/ebeMassResCalibration"
LOG_FILE="${REGEN_LOG_FILE:-/tmp/regen_missing_plots.log}"
PROGRESS_FILE="${REGEN_PROGRESS_FILE:-/tmp/regen_missing_plots.progress}"
: > "$PROGRESS_FILE"

# label directory name -> (NanoAODv, input_path). Both nanoAODv12 (2022/2023) and
# nanoAODv15 (2024/2025/2026) label families share the same {base_dir}/<label> layout
# (see configs/trials.yml / workflow/Snakefile's label_for_year), so this is derived
# from the label string itself rather than hardcoded per year.
nano_for_label() {
    case "$1" in
        *nanoAODv12*) echo 12 ;;
        *nanoAODv15*) echo 15 ;;
        *) echo "regen: cannot infer NanoAODv from label '$1'" >&2; return 1 ;;
    esac
}

regen_one() {
    local nano="$1" input_path="$2" year="$3" sample="$4" cat="$5"
    local ismc_flag=""
    [[ "$sample" == "MC" ]] && ismc_flag="--isMC"
    timeout 120 python3 -u src/lib/ebeMassResCalibration/getCalibrationFactor.py \
        --NanoAODv "$nano" --years "$year" \
        --input_path "$input_path" \
        --extraString V1 --steps step1 --fixCat "$cat" \
        --log-level INFO --no-dask-client --ifbinned $ismc_flag \
        >> "$LOG_FILE" 2>&1
}

total_missing=0
while IFS= read -r csv; do
    d="$(dirname "$csv")"
    # .../<label>/binned/<year>/<Sample>_<year>_V1
    year_dir="$(dirname "$(dirname "$d")")"
    year="$(basename "$year_dir")"
    label="$(basename "$(dirname "$year_dir")")"
    sample_dirname="$(basename "$d")"          # e.g. MC_2025_V1
    sample="${sample_dirname%%_*}"             # MC or Data
    input_path="${BASE}/${label}"
    nano="$(nano_for_label "$label")" || continue

    cats=$(tail -n +2 "$csv" | cut -d, -f1)
    for c in $cats; do
        [[ -f "$d/calibration_fitCat${c}.png" ]] && continue
        total_missing=$((total_missing + 1))
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "MISSING  $label  $year  $sample  $c"
            continue
        fi
        echo "=== [$label $year $sample] regenerating $c ===" >> "$LOG_FILE"
        regen_one "$nano" "$input_path" "$year" "$sample" "$c"
        if [[ -f "$d/calibration_fitCat${c}.png" ]]; then
            echo "OK $label $year $sample $c" >> "$PROGRESS_FILE"
        else
            echo "STILL_MISSING $label $year $sample $c" >> "$PROGRESS_FILE"
        fi
    done
done < <(find "$BASE" -name calibration_factors.csv)

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Total missing: $total_missing"
else
    echo "ALL_REGEN_DONE (checked $total_missing)" >> "$PROGRESS_FILE"
    echo "Done. See $LOG_FILE and $PROGRESS_FILE"
fi
