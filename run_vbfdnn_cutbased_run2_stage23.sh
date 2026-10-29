#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Run the existing Run-2 cut-based VBF-DNN through Stage 2 and Stage 3.

Usage:
  ./run_vbfdnn_cutbased_run2_stage23.sh [all|stage2|stage3]

Default:
  all                         Run Stage 2, then Stage 3 once per year.

Useful overrides:
  YEARS_CSV=2018              Comma-separated years (default: all four Run-2 eras).
  NO_VARIATIONS=1             Nominal-only smoke test; use a distinct POSTFIX.
  POSTFIX=<tag>               Output tag.
  USE_GATEWAY=0               Use local Dask instead of the existing Gateway cluster.
  N_WORKERS=1                 Local worker count when USE_GATEWAY=0.
  CLUSTER_INDEX=0             Existing Dask Gateway cluster index.
  SAMPLE_CONFIG=<path>        Stage-2/3 sample grouping YAML. The default is the
                              production-specific Run-2 NanoV15 mapping.
  OUT_BASE=<path>             Parent directory for this user's output.

Examples:
  ./run_vbfdnn_cutbased_run2_stage23.sh all

  YEARS_CSV=2018 NO_VARIATIONS=1 POSTFIX=CutBasedVBFDNN_2018_smoke \
    ./run_vbfdnn_cutbased_run2_stage23.sh all

Notes:
  - Scale the existing Dask Gateway cluster above zero workers before a Gateway run.
  - This script intentionally does not enable either Transformer VBF-channel option.
  - The default sample config maps the external production's st_tW_*,
    st_tchannel_*, zz_2l2q, zz_4l and ewk_zlljj directory names.
  - Re-running Stage 2 with the same postfix is incremental: complete existing
    sample histograms are skipped and newly configured samples are added.
EOF
}

MODE="${1:-all}"
case "${MODE}" in
    -h|--help)
        usage
        exit 0
        ;;
    all|stage2|stage3) ;;
    *)
        echo "ERROR: mode must be all, stage2, or stage3; got '${MODE}'." >&2
        usage >&2
        exit 2
        ;;
esac
if [[ "$#" -gt 1 ]]; then
    echo "ERROR: unexpected arguments: ${*:2}" >&2
    usage >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${REPO_ROOT}"

STAGE1_SRC="${STAGE1_SRC:-/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_BtagZptFix_sept29_2026}"
OUT_BASE="${OUT_BASE:-/work/projects/hmm/hu1027/hmm_ntuples/copperheadV1clean}"
LABEL="${LABEL:-Run2_NanoV15_BtagZptFix_sept29_2026_CutBasedVBFDNN}"
OUT_ROOT="${OUT_ROOT:-${OUT_BASE}/${LABEL}}"

MODEL_DIR="${MODEL_DIR:-/work/users/yun79/sideHustle2/copperheadV2/dnn/trained_models/Run2_NanoV15_forVBFChannel_July06_2026_jetUncRedo/2018-2017-2016postVFP-2016preVFP_h-peak_vbf}"
MODEL_TAG="${MODEL_TAG:-trained_best_optuna_100Trials_w_oneHotEncodeYear_w_MiNNLO_DY}"

POSTFIX="${POSTFIX:-CutBasedVBFDNN_Oct29_2026}"
DEFAULT_SAMPLE_CONFIG="${REPO_ROOT}/configs/samples/samples_run2_nanov15_btagzptfix_sept29_2026.yaml"
SAMPLE_CONFIG="${SAMPLE_CONFIG:-${DEFAULT_SAMPLE_CONFIG}}"
YEARS_CSV="${YEARS_CSV:-2016preVFP,2016postVFP,2017,2018}"
IFS=',' read -r -a YEARS <<< "${YEARS_CSV}"

NO_VARIATIONS="${NO_VARIATIONS:-0}"
USE_GATEWAY="${USE_GATEWAY:-1}"
CLUSTER_INDEX="${CLUSTER_INDEX:-0}"
N_WORKERS="${N_WORKERS:-1}"
DY_MATCHED_JETS="${DY_MATCHED_JETS:-reco_012}"
DY_SCHEME="${DY_SCHEME:-matched}"
VBF_FILTER_STUDY="${VBF_FILTER_STUDY:-1}"
ALLOW_DEFAULT_DNN_BINNING="${ALLOW_DEFAULT_DNN_BINNING:-1}"
JJ_ETA_REGION="${JJ_ETA_REGION:-all}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_file() {
    [[ -s "$1" ]] || die "missing or empty file: $1"
}

require_dir() {
    [[ -d "$1" ]] || die "missing directory: $1"
}

case "${NO_VARIATIONS}" in 0|1) ;; *) die "NO_VARIATIONS must be 0 or 1" ;; esac
case "${USE_GATEWAY}" in 0|1) ;; *) die "USE_GATEWAY must be 0 or 1" ;; esac
case "${VBF_FILTER_STUDY}" in 0|1) ;; *) die "VBF_FILTER_STUDY must be 0 or 1" ;; esac
case "${ALLOW_DEFAULT_DNN_BINNING}" in 0|1) ;; *) die "ALLOW_DEFAULT_DNN_BINNING must be 0 or 1" ;; esac
[[ "${#YEARS[@]}" -gt 0 ]] || die "YEARS_CSV did not contain a year"

require_dir "${STAGE1_SRC}/stage1_output"
require_dir "${MODEL_DIR}/${MODEL_TAG}"
require_file "${MODEL_DIR}/training_features.pkl"
require_file "${MODEL_DIR}/preprocess_manifest.json"
require_file "${MODEL_DIR}/${MODEL_TAG}/manifest.json"
require_file "${SAMPLE_CONFIG}"

for fold in 0 1 2 3; do
    require_file "${MODEL_DIR}/scalers_${fold}.npz"
    require_file "${MODEL_DIR}/${MODEL_TAG}/fold${fold}/best_torchscript.pt"
done

for year in "${YEARS[@]}"; do
    require_dir "${STAGE1_SRC}/stage1_output/${year}/compacted"
done

mkdir -p "${OUT_ROOT}"
STAGE1_LINK="${OUT_ROOT}/stage1_output"
if [[ -e "${STAGE1_LINK}" || -L "${STAGE1_LINK}" ]]; then
    [[ "$(realpath "${STAGE1_LINK}")" == "$(realpath "${STAGE1_SRC}/stage1_output")" ]] \
        || die "${STAGE1_LINK} already exists but does not point to ${STAGE1_SRC}/stage1_output"
else
    ln -s "${STAGE1_SRC}/stage1_output" "${STAGE1_LINK}"
fi

echo "Run-2 cut-based VBF-DNN Stage 2/3 configuration"
echo "  mode:             ${MODE}"
echo "  years:            ${YEARS[*]}"
echo "  Stage-1 input:    $(realpath "${STAGE1_LINK}")"
echo "  output root:      ${OUT_ROOT}"
echo "  analysis label:   ${LABEL}"
echo "  model directory:  ${MODEL_DIR}"
echo "  model tag:        ${MODEL_TAG}"
echo "  postfix:          ${POSTFIX}"
echo "  sample config:    ${SAMPLE_CONFIG}"
echo "  variations:       $([[ "${NO_VARIATIONS}" == 1 ]] && echo nominal-only || echo full)"
echo "  Dask Gateway:     ${USE_GATEWAY}"
echo
if [[ "$(realpath "${SAMPLE_CONFIG}")" == "$(realpath "${DEFAULT_SAMPLE_CONFIG}")" ]]; then
    echo "The default sample mapping matches this external Stage-1 production."
    echo "2017 st_schannel_had/st_schannel_lep remain intentionally excluded pending validation."
else
    echo "Using caller-supplied SAMPLE_CONFIG; verify that it matches the Stage-1 directory names."
fi
echo

variation_args=()
if [[ "${NO_VARIATIONS}" == 1 ]]; then
    variation_args+=(--no_variations)
fi

vbf_filter_args=()
if [[ "${VBF_FILTER_STUDY}" == 1 ]]; then
    vbf_filter_args+=(--vbf_filter_study)
fi

jj_region_args=()
if [[ "${JJ_ETA_REGION}" != all ]]; then
    jj_region_args+=(--jj_eta_region "${JJ_ETA_REGION}")
fi

binning_args=()
if [[ "${ALLOW_DEFAULT_DNN_BINNING}" == 1 ]]; then
    binning_args+=(--allow_default_dnn_binning)
fi

dask_args=()
if [[ "${USE_GATEWAY}" == 1 ]]; then
    dask_args+=(--use_gateway --cluster_index "${CLUSTER_INDEX}")
else
    dask_args+=(--n_workers "${N_WORKERS}")
fi

run_stage2() {
    local -a cmd=(
        "${REPO_ROOT}/run_in_pixi.sh" default python run_stage2_vbf.py
        --years "${YEARS[@]}"
        -input "${OUT_ROOT}"
        -l "${LABEL}"
        --model_path "${MODEL_DIR}"
        --model_tag "${MODEL_TAG}"
        -data A B C D E F G H
        -bkg DY TT ST VV EWK VVV
        -sig ggH VBF
        --sample-config "${SAMPLE_CONFIG}"
        --save_postfix "${POSTFIX}"
        --dy_matched_jets "${DY_MATCHED_JETS}"
        "${vbf_filter_args[@]}"
        "${jj_region_args[@]}"
        "${variation_args[@]}"
        "${binning_args[@]}"
        "${dask_args[@]}"
    )
    echo "Running Stage 2:"
    printf ' %q' "${cmd[@]}"
    echo
    "${cmd[@]}"
}

run_stage3() {
    local year
    for year in "${YEARS[@]}"; do
        local -a cmd=(
            "${REPO_ROOT}/run_in_pixi.sh" default python run_stage3_vbf.py
            --years "${year}"
            -input "${OUT_ROOT}"
            -l "${LABEL}"
            --sample-config "${SAMPLE_CONFIG}"
            --save_postfix "${POSTFIX}"
            --dy_scheme "${DY_SCHEME}"
            "${vbf_filter_args[@]}"
            "${jj_region_args[@]}"
            "${variation_args[@]}"
        )
        echo "Running Stage 3 for ${year}:"
        printf ' %q' "${cmd[@]}"
        echo
        "${cmd[@]}"
    done
}

case "${MODE}" in
    stage2) run_stage2 ;;
    stage3) run_stage3 ;;
    all)
        run_stage2
        run_stage3
        ;;
esac

echo "Completed mode '${MODE}'."
