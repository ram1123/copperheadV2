#!/usr/bin/env bash
set -euo pipefail

# One-click Transformer-selected VBF-DNN matrix:
#   Run 2 score < 0.200, Run 2 score < 0.022,
#   Run 3 score < 0.200, Run 3 score < 0.019.
# Preprocessing is performed sequentially because Purdue AF permits one Dask
# Gateway cluster per user. Final training uses one shared 16-worker Ray
# cluster: four models x four folds x one GPU per fold.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

GLOBAL_PYTHON="/work/pixi/global/.pixi/envs/default/bin/python"
RAY_SUBMITTER="MVA_training/VBF_run3/submit_ray_dnn.py"
SCORE_COLUMN="transf_ggh_vs_vbf_binary_score"

RUN2_INPUT="/depot/cms/users/hu1027/Hmumu_Categorization_Transformer_output/scored_output/run2_stage23_vbfdnn_nominal_sep08_ray/stage1_output/"
RUN3_INPUT="/depot/cms/users/hu1027/Hmumu_Categorization_Transformer_output/binary_output/run3_oct23_binary_all_eras/stage23_scored/"
RUN2_CONFIG="configs/dnn_run2_vbf.yaml"
RUN3_CONFIG="configs/dnn_run3_vbf.yaml"
RUN2_DATASET="configs/datasets/dataset_nanoAODv12_run2.yaml"
RUN3_DATASET="configs/datasets/dataset_nanoAODv15_run3.yaml"
RUN2_YEARS="2016preVFP,2016postVFP,2017,2018"
RUN3_YEARS="2022preEE,2022postEE,2023,2023BPix,2024,2025,2026"
RUN2_YEAR_SLUG="2016preVFP-2016postVFP-2017-2018"
RUN3_YEAR_SLUG="2022preEE-2022postEE-2023-2023BPix-2024-2025-2026"

RUN2_T020_LABEL="${RUN2_T020_LABEL:-Run2_VBFDNN_Transformer_lt0p200}"
RUN2_TOPT_LABEL="${RUN2_TOPT_LABEL:-Run2_VBFDNN_Transformer_lt0p022}"
RUN3_T020_LABEL="${RUN3_T020_LABEL:-Run3_VBFDNN_Transformer_lt0p200}"
RUN3_TOPT_LABEL="${RUN3_TOPT_LABEL:-Run3_VBFDNN_Transformer_lt0p019}"
TRAIN_LABEL="${TRAIN_LABEL:-trained_ray16}"

CPUS_PER_FOLD="${CPUS_PER_FOLD:-4}"
RAM_PER_WORKER_GB="${RAM_PER_WORKER_GB:-16}"
MIN_GPU_MEMORY_GB="${MIN_GPU_MEMORY_GB:-5}"
RAY_MAX_RETRIES="${RAY_MAX_RETRIES:-4}"
POLL_SECONDS="${POLL_SECONDS:-20}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"

die() {
    echo "[ERROR] $*" >&2
    exit 2
}

[[ -x "${GLOBAL_PYTHON}" ]] || die "Global Python is missing: ${GLOBAL_PYTHON}"
[[ -x "./run_in_pixi.sh" ]] || die "./run_in_pixi.sh is missing or not executable."
[[ -f "${RAY_SUBMITTER}" ]] || die "Ray submitter is missing: ${RAY_SUBMITTER}"
for required in \
    "${RUN2_INPUT}" "${RUN3_INPUT}" \
    "${RUN2_CONFIG}" "${RUN3_CONFIG}" \
    "${RUN2_DATASET}" "${RUN3_DATASET}"; do
    [[ -e "${required}" ]] || die "Required input is missing: ${required}"
done

# Purdue AF rejects Ray GPU-cluster submissions from a notebook session that
# already owns a GPU.
if command -v nvidia-smi >/dev/null 2>&1; then
    die "Run this script from a CPU-only Purdue AF session (no notebook GPU)."
fi

preprocess_one() {
    local run_name="$1"
    local label="$2"
    local threshold="$3"
    local input_root="$4"
    local config="$5"
    local dataset="$6"
    local nanoaod="$7"
    local years="$8"

    echo
    echo "[preprocess] ${run_name}: ${SCORE_COLUMN} < ${threshold}"
    DNN_CONFIG="${config}" \
    DNN_INPUT_BASE="${input_root}" \
    DNN_CATEGORY_SELECTION="transformer" \
    TRANSFORMER_SCORE_COLUMN="${SCORE_COLUMN}" \
    TRANSFORMER_SCORE_THRESHOLD="${threshold}" \
    ./run_in_pixi.sh default bash run_analysis_pipeline.sh \
        -c "${dataset}" \
        -v "${nanoaod}" \
        -l "${label}" \
        -y "${years}" \
        -m dnn_pre \
        -k
}

if [[ "${SKIP_PREPROCESS}" == "1" ]]; then
    echo "[preprocess] SKIP_PREPROCESS=1; validating existing outputs at submission."
else
    preprocess_one "Run 2 t=0.200" "${RUN2_T020_LABEL}" "0.2" "${RUN2_INPUT}" "${RUN2_CONFIG}" "${RUN2_DATASET}" "12" "${RUN2_YEARS}"
    preprocess_one "Run 2 t=0.022" "${RUN2_TOPT_LABEL}" "0.022" "${RUN2_INPUT}" "${RUN2_CONFIG}" "${RUN2_DATASET}" "12" "${RUN2_YEARS}"
    preprocess_one "Run 3 t=0.200" "${RUN3_T020_LABEL}" "0.2" "${RUN3_INPUT}" "${RUN3_CONFIG}" "${RUN3_DATASET}" "15" "${RUN3_YEARS}"
    preprocess_one "Run 3 t=0.019" "${RUN3_TOPT_LABEL}" "0.019" "${RUN3_INPUT}" "${RUN3_CONFIG}" "${RUN3_DATASET}" "15" "${RUN3_YEARS}"
fi

run2_t020_data="${REPO_ROOT}/dnn/trained_models/${RUN2_T020_LABEL}/${RUN2_YEAR_SLUG}_h-peak_vbf_all"
run2_topt_data="${REPO_ROOT}/dnn/trained_models/${RUN2_TOPT_LABEL}/${RUN2_YEAR_SLUG}_h-peak_vbf_all"
run3_t020_data="${REPO_ROOT}/dnn/trained_models/${RUN3_T020_LABEL}/${RUN3_YEAR_SLUG}_h-peak_vbf_all"
run3_topt_data="${REPO_ROOT}/dnn/trained_models/${RUN3_TOPT_LABEL}/${RUN3_YEAR_SLUG}_h-peak_vbf_all"

declare -a model_names=(
    "Run 2 score<0.200"
    "Run 2 score<0.022"
    "Run 3 score<0.200"
    "Run 3 score<0.019"
)
declare -a model_configs=(
    "${REPO_ROOT}/${RUN2_CONFIG}"
    "${REPO_ROOT}/${RUN2_CONFIG}"
    "${REPO_ROOT}/${RUN3_CONFIG}"
    "${REPO_ROOT}/${RUN3_CONFIG}"
)
declare -a model_data_dirs=(
    "${run2_t020_data}"
    "${run2_topt_data}"
    "${run3_t020_data}"
    "${run3_topt_data}"
)

training_complete() {
    local out_dir="$1"
    local fold
    [[ -f "${out_dir}/manifest.json" ]] || return 1
    for fold in 0 1 2 3; do
        [[ -f "${out_dir}/fold${fold}/best_torchscript.pt" ]] || return 1
    done
}

declare -a submission_files=()
for index in 0 1 2 3; do
    data_dir="${model_data_dirs[${index}]}"
    out_dir="${data_dir}/${TRAIN_LABEL}"
    submission_file="${out_dir}/ray_submission.json"
    if [[ -f "${submission_file}" ]]; then
        echo "[cleanup] Removing previous Ray submission record: ${submission_file}"
        rm -f -- "${submission_file}"
    fi
    if training_complete "${out_dir}"; then
        echo "[skip] ${model_names[${index}]} is already complete: ${out_dir}"
        continue
    fi

    submit_cmd=(
        "${GLOBAL_PYTHON}" "${RAY_SUBMITTER}" submit
        --config "${model_configs[${index}]}"
        --data-dir "${data_dir}"
        --out-dir "${out_dir}"
        --cluster-workers 16
        --cpus-per-worker "${CPUS_PER_FOLD}"
        --ram-per-worker-gb "${RAM_PER_WORKER_GB}"
        --min-gpu-memory-gb "${MIN_GPU_MEMORY_GB}"
        --max-retries "${RAY_MAX_RETRIES}"
    )

    echo
    echo "[submit] ${model_names[${index}]}"
    echo "         data: ${data_dir}"
    echo "         out:  ${out_dir}"
    "${submit_cmd[@]}"
    submission_files+=("${out_dir}/ray_submission.json")
done

if [[ "${#submission_files[@]}" == "0" ]]; then
    echo "All four model trainings were already complete."
    exit 0
fi

echo
echo "Submitted ${#submission_files[@]} model job(s) to one exact-size 16-GPU Ray cluster."
echo "Each job schedules four independent folds with one GPU per fold."
echo "Waiting for every submitted job..."

declare -a wait_pids=()
for submission in "${submission_files[@]}"; do
    "${GLOBAL_PYTHON}" "${RAY_SUBMITTER}" wait \
        --submission "${submission}" \
        --poll-seconds "${POLL_SECONDS}" &
    wait_pids+=("$!")
done

wait_status=0
for pid in "${wait_pids[@]}"; do
    if ! wait "${pid}"; then
        wait_status=1
    fi
done

if [[ "${wait_status}" != "0" ]]; then
    die "At least one Ray model job failed; inspect its ray_submission.json with the status --logs command."
fi

echo
echo "All requested four-fold VBF-DNN trainings completed successfully."
for index in 0 1 2 3; do
    echo "  ${model_names[${index}]} -> ${model_data_dirs[${index}]}/${TRAIN_LABEL}"
done
