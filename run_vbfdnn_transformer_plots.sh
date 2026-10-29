#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./run_vbfdnn_transformer_plots.sh <RUN2_MODEL> <RUN3_MODEL> [all|dnn|templates|impacts|lhscan]

Generate the nominal-only validation/statistics plots for one completed Run-2
and one completed Run-3 Transformer Stage-2/3 production.  The default step is
"all".  Existing non-empty impacts and likelihood-scan products are reused.

Examples:
  ./run_vbfdnn_transformer_plots.sh \
    Run2_VBFDNN_Transformer_lt0p022 \
    Run3_VBFDNN_Transformer_lt0p019 all

  DRY_RUN=1 ./run_vbfdnn_transformer_plots.sh \
    Run2_VBFDNN_Transformer_lt0p022 \
    Run3_VBFDNN_Transformer_lt0p019 all

Optional environment variables:
  SAVE_ROOT                 Production root (default: Purdue Hmumu project root)
  DNN_PLOT_ROOT             Stage-2 plot root (default: validation/stage2_dnn_score)
  TEMPLATE_PLOT_ROOT        Template plot root (default: validation/stage3_templates)
  RUN2_ANALYSIS_LABEL       Run-2 output label (default: RUN2_MODEL)
  RUN3_ANALYSIS_LABEL       Run-3 output label (default: RUN3_MODEL)
  RUN2_FINAL_POSTFIX        Run-2 Stage-2/3 postfix (default: TransformerStage23)
  RUN3_FINAL_POSTFIX        Run-3 Stage-2/3 postfix (default: TransformerStage23)
  COMBINED_ANALYSIS_LABEL   Combined output label
  COMBINED_FINAL_POSTFIX    Combined stats postfix (default: Run2Run3)
  JJ_ETA_REGION             Shared jj region (default: all)
  FORCE_STATS=1             Re-run impacts/lhscan even when complete outputs exist
  DRY_RUN=1                 Validate inputs and print commands without executing them

This runner intentionally skips Stage-2 and Stage-3 Up/Down systematic plots:
the Transformer production it targets is nominal-only.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

log() {
    echo "[$(date '+%F %T')] $*"
}

require_dir() {
    [[ -d "$1" ]] || die "Required directory is missing: $1"
}

require_file() {
    [[ -s "$1" ]] || die "Required non-empty file is missing: $1"
}

validate_token() {
    local name="$1" value="$2"
    [[ "${value}" =~ ^[A-Za-z0-9._+-]+$ ]] || \
        die "${name} contains unsupported characters: ${value}"
}

run_cmd() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" != "1" ]]; then
        "$@"
    fi
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }
[[ "$#" -ge 2 && "$#" -le 3 ]] || { usage >&2; exit 2; }

RUN2_MODEL="$1"
RUN3_MODEL="$2"
STEP="${3:-all}"
case "${STEP}" in
    all|dnn|templates|impacts|lhscan) ;;
    *) die "Unknown step '${STEP}'" ;;
esac

validate_token RUN2_MODEL "${RUN2_MODEL}"
validate_token RUN3_MODEL "${RUN3_MODEL}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

AF_USERNAME="${NB_USER:-$(whoami)}"
DEFAULT_SAVE_ROOT="/work/projects/hmm/${AF_USERNAME}/hmm_ntuples/copperheadV1clean"
SAVE_ROOT="$(realpath -m "${SAVE_ROOT:-${DEFAULT_SAVE_ROOT}}")"
DNN_PLOT_ROOT="$(realpath -m "${DNN_PLOT_ROOT:-${REPO_ROOT}/validation/stage2_dnn_score}")"
TEMPLATE_PLOT_ROOT="$(realpath -m "${TEMPLATE_PLOT_ROOT:-${REPO_ROOT}/validation/stage3_templates}")"
RUN_IN_PIXI="${REPO_ROOT}/run_in_pixi.sh"
STATS_PIPELINE="${REPO_ROOT}/run_stats_pipeline_VBF.sh"

RUN2_LABEL="${RUN2_ANALYSIS_LABEL:-${RUN2_MODEL}}"
RUN3_LABEL="${RUN3_ANALYSIS_LABEL:-${RUN3_MODEL}}"
RUN2_POSTFIX="${RUN2_FINAL_POSTFIX:-TransformerStage23}"
RUN3_POSTFIX="${RUN3_FINAL_POSTFIX:-TransformerStage23}"
COMBINED_POSTFIX="${COMBINED_FINAL_POSTFIX:-Run2Run3}"
COMBINED_LABEL="${COMBINED_ANALYSIS_LABEL:-Run2Run3_${RUN2_MODEL}__${RUN3_MODEL}}"
JJ_ETA_REGION="${JJ_ETA_REGION:-all}"
FORCE_STATS="${FORCE_STATS:-0}"
DRY_RUN="${DRY_RUN:-0}"

RUN2_DATASET_CONFIG="$(realpath -m "${RUN2_DATASET_CONFIG:-${REPO_ROOT}/configs/datasets/dataset_nanoAODv15_run2.yaml}")"
RUN3_DATASET_CONFIG="$(realpath -m "${RUN3_DATASET_CONFIG:-${REPO_ROOT}/configs/datasets/dataset_nanoAODv15_run3.yaml}")"

for token_name in RUN2_LABEL RUN3_LABEL RUN2_POSTFIX RUN3_POSTFIX COMBINED_LABEL COMBINED_POSTFIX JJ_ETA_REGION; do
    validate_token "${token_name}" "${!token_name}"
done
[[ "${FORCE_STATS}" == "0" || "${FORCE_STATS}" == "1" ]] || die "FORCE_STATS must be 0 or 1"
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1"
require_file "${RUN_IN_PIXI}"
require_file "${STATS_PIPELINE}"
require_file "${RUN2_DATASET_CONFIG}"
require_file "${RUN3_DATASET_CONFIG}"

jj_suffix() {
    [[ "${JJ_ETA_REGION}" == "all" ]] || printf '_%s' "${JJ_ETA_REGION}"
}

stage2_suffix() {
    printf '_vbf_filter_study%s_NoSyst' "$(jj_suffix)"
}

card_suffix() {
    local postfix="$1"
    printf '%s_vbf_filter_study%s' "${postfix}" "$(jj_suffix)"
}

stage2_dir() {
    local label="$1" postfix="$2"
    printf '%s/%s/stage2_histograms/score_%s_%s%s' \
        "${SAVE_ROOT}" "${label}" "${label}" "${postfix}" "$(stage2_suffix)"
}

card_dir() {
    local label="$1" postfix="$2"
    printf '%s/%s/stage3_datacards_%s/score_%s' \
        "${SAVE_ROOT}" "${label}" "$(card_suffix "${postfix}")" "${label}"
}

template_dir() {
    local label="$1" postfix="$2" suffix
    suffix="$(card_suffix "${postfix}")"
    printf '%s/%s/stage3_datacards_%s/stage3_templates_%s/score_%s' \
        "${SAVE_ROOT}" "${label}" "${suffix}" "${suffix}" "${label}"
}

RUN2_CARD_DIR="$(card_dir "${RUN2_LABEL}" "${RUN2_POSTFIX}")"
RUN3_CARD_DIR="$(card_dir "${RUN3_LABEL}" "${RUN3_POSTFIX}")"
COMBINED_CARD_DIR="$(card_dir "${COMBINED_LABEL}" "${COMBINED_POSTFIX}")"
RUN2_CARD_PATH="${RUN2_CARD_DIR}/HMuMu_13TeV_Run2.txt"
RUN3_CARD_PATH="${RUN3_CARD_DIR}/HMuMu_13TeV_Run3.txt"

plot_dnn_set() {
    local label="$1" postfix="$2"
    shift 2
    local hist_dir mva_name year region
    hist_dir="$(stage2_dir "${label}" "${postfix}")"
    mva_name="${label}_${postfix}$(stage2_suffix)"
    require_dir "${hist_dir}"
    for year in "$@"; do
        for region in h-sidebands h-peak; do
            run_cmd "${RUN_IN_PIXI}" default python plotter/plot_DNN_score.py \
                --load "${hist_dir}" \
                --label "${label}" \
                --category vbf \
                --year "${year}" \
                --region "${region}" \
                --mva_name "${mva_name}" \
                --save_path "${DNN_PLOT_ROOT}" \
                --sample-config "${REPO_ROOT}/configs/samples/samples.yaml" \
                --vbf_filter_study \
                --log-level ERROR
        done
    done
}

plot_nominal_templates() {
    local label="$1" postfix="$2" input output
    input="$(template_dir "${label}" "${postfix}")"
    output="${TEMPLATE_PLOT_ROOT}/${label}/$(card_suffix "${postfix}")/nominal_templates_negative_bins.pdf"
    require_dir "${input}"
    run_cmd "${RUN_IN_PIXI}" combine python scripts/plot_template_negative_bins.py \
        --input-dir "${input}" \
        --output "${output}" \
        --tag "${label} nominal-only"
}

run_single_stats() {
    local mode="$1" year="$2" label="$3" postfix="$4" dataset="$5"
    run_cmd env \
        WITH_VARIATIONS=0 \
        JJ_ETA_REGION="${JJ_ETA_REGION}" \
        SIGNAL_XSEC_RESCALE=0 \
        "${RUN_IN_PIXI}" combine bash "${STATS_PIPELINE}" \
        -m "${mode}" -y "${year}" -v 15 -c "${dataset}" \
        -l "${label}" -o "${postfix}" -S "${SAVE_ROOT}" -V
}

run_combined_stats() {
    local mode="$1"
    run_cmd env \
        WITH_VARIATIONS=0 \
        JJ_ETA_REGION="${JJ_ETA_REGION}" \
        SIGNAL_XSEC_RESCALE=0 \
        VBF_RUN2_CARD_PATH="${RUN2_CARD_PATH}" \
        VBF_RUN3_CARD_PATH="${RUN3_CARD_PATH}" \
        VBF_RUN2_MODEL_LABEL="${RUN2_MODEL}" \
        VBF_RUN3_MODEL_LABEL="${RUN3_MODEL}" \
        "${RUN_IN_PIXI}" combine bash "${STATS_PIPELINE}" \
        -m "${mode}" -y Run2Run3 -v 15 -c "${RUN3_DATASET_CONFIG}" \
        -l "${COMBINED_LABEL}" -o "${COMBINED_POSTFIX}" -S "${SAVE_ROOT}" -V
}

stats_outputs_complete() {
    local kind="$1" year="$2" postfix="$3" dir="$4" path suffix
    case "${kind}" in
        impacts)
            for suffix in r1 r0; do
                [[ -s "${dir}/impacts_${year}_${postfix}_${suffix}.json" ]] || return 1
                [[ -s "${dir}/impacts_${year}_${postfix}_${suffix}.pdf" ]] || return 1
            done
            ;;
        lhscan)
            for suffix in '' .freeze_systs .freeze_systs_mcstat .statonly; do
                path="${dir}/higgsCombine.lhscan${year}_${postfix}.with_syst${suffix}.MultiDimFit.mH125.root"
                [[ -s "${path}" ]] || return 1
            done
            [[ -s "${dir}/lh_scan_${year}_${postfix}.pdf" ]] || return 1
            [[ -s "${dir}/lh_scan_${year}_${postfix}.png" ]] || return 1
            ;;
        *) return 1 ;;
    esac
}

ensure_stats_product() {
    local kind="$1" year="$2" postfix="$3" dir="$4" context="$5"
    local mode
    case "${kind}" in
        impacts) mode=7 ;;
        lhscan) mode=8 ;;
        *) die "Unknown stats product '${kind}'" ;;
    esac

    require_file "${dir}/HMuMu_13TeV_${year}.txt"
    require_file "${dir}/HMuMu_13TeV_${year}.root"
    if [[ "${FORCE_STATS}" == "0" ]] && stats_outputs_complete "${kind}" "${year}" "${postfix}" "${dir}"; then
        log "Reusing ${kind}: ${dir}"
        return
    fi

    case "${context}" in
        run2) run_single_stats "${mode}" Run2 "${RUN2_LABEL}" "${RUN2_POSTFIX}" "${RUN2_DATASET_CONFIG}" ;;
        run3) run_single_stats "${mode}" Run3 "${RUN3_LABEL}" "${RUN3_POSTFIX}" "${RUN3_DATASET_CONFIG}" ;;
        combined) run_combined_stats "${mode}" ;;
        *) die "Unknown stats context '${context}'" ;;
    esac

    if [[ "${DRY_RUN}" != "1" ]]; then
        stats_outputs_complete "${kind}" "${year}" "${postfix}" "${dir}" || \
            die "${kind} command finished without the complete expected output set in ${dir}"
    fi
}

run_dnn_plots() {
    plot_dnn_set "${RUN2_LABEL}" "${RUN2_POSTFIX}" \
        2016preVFP 2016postVFP 2017 2018 Run2
    plot_dnn_set "${RUN3_LABEL}" "${RUN3_POSTFIX}" \
        2022preEE 2022postEE 2023 2023BPix 2024 2025 2026 Run3
}

run_template_plots() {
    plot_nominal_templates "${RUN2_LABEL}" "${RUN2_POSTFIX}"
    plot_nominal_templates "${RUN3_LABEL}" "${RUN3_POSTFIX}"
}

run_impacts_plots() {
    ensure_stats_product impacts Run2 "${RUN2_POSTFIX}" "${RUN2_CARD_DIR}" run2
    ensure_stats_product impacts Run3 "${RUN3_POSTFIX}" "${RUN3_CARD_DIR}" run3
    ensure_stats_product impacts Run2Run3 "${COMBINED_POSTFIX}" "${COMBINED_CARD_DIR}" combined
}

run_lhscan_plots() {
    ensure_stats_product lhscan Run2 "${RUN2_POSTFIX}" "${RUN2_CARD_DIR}" run2
    ensure_stats_product lhscan Run3 "${RUN3_POSTFIX}" "${RUN3_CARD_DIR}" run3
    ensure_stats_product lhscan Run2Run3 "${COMBINED_POSTFIX}" "${COMBINED_CARD_DIR}" combined
}

case "${STEP}" in
    dnn) run_dnn_plots ;;
    templates) run_template_plots ;;
    impacts) run_impacts_plots ;;
    lhscan) run_lhscan_plots ;;
    all)
        run_dnn_plots
        run_template_plots
        run_impacts_plots
        run_lhscan_plots
        ;;
esac

cat <<EOF

Nominal-only plot products:
  Run-2 DNN plots:       ${DNN_PLOT_ROOT}/${RUN2_LABEL}/${RUN2_POSTFIX}$(stage2_suffix)/{2016preVFP,2016postVFP,2017,2018,Run2}/
  Run-3 DNN plots:       ${DNN_PLOT_ROOT}/${RUN3_LABEL}/${RUN3_POSTFIX}$(stage2_suffix)/{2022preEE,2022postEE,2023,2023BPix,2024,2025,2026,Run3}/
  Run-2 template PDF:    ${TEMPLATE_PLOT_ROOT}/${RUN2_LABEL}/$(card_suffix "${RUN2_POSTFIX}")/nominal_templates_negative_bins.pdf
  Run-3 template PDF:    ${TEMPLATE_PLOT_ROOT}/${RUN3_LABEL}/$(card_suffix "${RUN3_POSTFIX}")/nominal_templates_negative_bins.pdf
  Run-2 impacts/lhscan:  ${RUN2_CARD_DIR}/
  Run-3 impacts/lhscan:  ${RUN3_CARD_DIR}/
  Combined impacts/scan: ${COMBINED_CARD_DIR}/
EOF

log "Completed step '${STEP}'"
