#!/bin/bash
#
# splice_sample_dir.sh
#
# Preserve an existing sample's stage-1 output by renaming it aside, then
# symlink in a replacement from elsewhere in its place -- e.g. splicing a
# freshly-reprocessed DY sample into an existing label's tree so every
# downstream step (compact, stage2/3, z-pT derivation, ...) that expects ONE
# label directory keeps working unmodified, while the old sample stays on
# disk for comparison rather than being lost.
#
# Handles BOTH levels a sample can exist at -- f1_0/<sample> (raw stage-1
# output) and compacted/<sample> -- in one command, since a sample normally
# needs both spliced together for downstream steps to see a consistent
# picture. Nothing outside these two per-level path pairs is ever touched.
#
# Usage:
#   bash scripts/splice_sample_dir.sh <old_base> <new_base> <sample> [--yes]
#
#   <old_base>  .../stage1_output/<year> directory of the label to preserve
#               and splice into, e.g.
#               /work/projects/hmm/<user>/hmm_ntuples/copperheadV1clean/<LabelA>/stage1_output/<year>
#   <new_base>  .../stage1_output/<year> directory of the label to splice
#               FROM, e.g.
#               /work/projects/hmm/<user>/hmm_ntuples/copperheadV1clean/<LabelB_DYRerun>/stage1_output/<year>
#   <sample>    Sample directory name under f1_0/ and compacted/, e.g.
#               dyTo2Mu_M-50_aMCatNLO
#
# What it does per level (f1_0, then compacted), only after --yes:
#   1. mv  <old_base>/<level>/<sample>  <old_base>/<level>/<sample>_preRerun_<timestamp>   (never deletes)
#   2. ln -s  <new_base>/<level>/<sample> (absolute)   <old_base>/<level>/<sample>
#
# A level is skipped (with a message, not an error) if <new_base>/<level>/<sample>
# doesn't exist -- e.g. compacted/ commonly doesn't exist yet for a
# just-reprocessed label until `-m compact` has been run for it. f1_0 is
# expected to always exist on both sides; that one DOES error if missing.
#
# Example:
#   OLD_BASE="/work/projects/hmm/$USER/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation/stage1_output/2025"
#   NEW_BASE="/work/projects/hmm/$USER/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation_DYReRun/stage1_output/2025"
#   bash scripts/splice_sample_dir.sh "$OLD_BASE" "$NEW_BASE" dyTo2Mu_M-50_aMCatNLO --yes
#
# Dry-run by default (prints the plan for every level, changes nothing) --
# pass --yes to actually apply it.

set -euo pipefail

usage() {
    echo "Usage: $0 <old_base> <new_base> <sample> [--yes]" >&2
    echo "  <old_base>/<new_base> are .../stage1_output/<year> directories." >&2
    echo "  Dry-run by default; pass --yes to actually rename + symlink." >&2
    exit 1
}

[[ $# -lt 3 ]] && usage

OLD_BASE="$1"
NEW_BASE="$2"
SAMPLE="$3"
DO_APPLY=0
if [[ "${4:-}" == "--yes" ]]; then
    DO_APPLY=1
elif [[ $# -ge 4 ]]; then
    echo "Unrecognized fourth argument: ${4}" >&2
    usage
fi

LEVELS=("f1_0" "compacted")

# --- splice one <level>/<sample> path pair; returns via echo'd plan lines ---
splice_one() {
    local level="$1"
    local old_path="${OLD_BASE%/}/${level}/${SAMPLE}"
    local new_path="${NEW_BASE%/}/${level}/${SAMPLE}"

    if [[ ! -d "$new_path" ]]; then
        if [[ "$level" == "f1_0" ]]; then
            echo "ERROR: new path does not exist or is not a directory: $new_path" >&2
            exit 1
        fi
        echo "[skip] ${level}: new path does not exist yet, nothing to splice: $new_path"
        return 0
    fi

    if [[ ! -d "$old_path" ]]; then
        echo "ERROR: old path does not exist or is not a directory: $old_path" >&2
        exit 1
    fi
    if [[ -L "$old_path" ]]; then
        echo "ERROR: old path is already a symlink -- refusing to touch it" \
             "(looks like this level was already spliced): $old_path" >&2
        exit 1
    fi

    # Absolute so the symlink stays valid regardless of which directory it's
    # later read from (a relative target resolves relative to the symlink's
    # own location, not the caller's cwd).
    local new_path_abs
    new_path_abs="$(cd "$new_path" && pwd)"

    local timestamp backup_path
    timestamp="$(date +%Y%m%d_%H%M%S)"
    backup_path="${old_path%/}_preRerun_${timestamp}"

    if [[ -e "$backup_path" ]]; then
        echo "ERROR: backup destination already exists, refusing to overwrite: $backup_path" >&2
        exit 1
    fi

    echo "[${level}] Plan:"
    echo "    mv  '${old_path}'  '${backup_path}'"
    echo "    ln -s  '${new_path_abs}'  '${old_path}'"

    if [[ "$DO_APPLY" -ne 1 ]]; then
        return 0
    fi

    mv "$old_path" "$backup_path"
    ln -s "$new_path_abs" "$old_path"

    echo "[${level}] Done. Old data preserved at: ${backup_path}"
    echo "[${level}] ${old_path} -> $(readlink -f "$old_path")"
}

for level in "${LEVELS[@]}"; do
    splice_one "$level"
    echo
done

if [[ "$DO_APPLY" -ne 1 ]]; then
    echo "Dry run only -- nothing changed. Re-run with --yes to apply."
fi
