#!/usr/bin/env bash
set -euo pipefail

# Concurrently train the two already-preprocessed Run-2 Transformer-selected
# VBF-DNNs:
#   transf_ggh_vs_vbf_binary_score < 0.200
#   transf_ggh_vs_vbf_binary_score < 0.022
# Each model has four folds and each fold reserves one GPU, for eight GPUs total.
# This script intentionally performs neither preprocessing nor HPO.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

GLOBAL_PYTHON="/work/pixi/global/.pixi/envs/default/bin/python"
RAY_SUBMITTER="MVA_training/VBF_run3/submit_ray_dnn.py"
RUN2_CONFIG="${REPO_ROOT}/configs/dnn_run2_vbf.yaml"
RUN2_YEAR_SLUG="2016preVFP-2016postVFP-2017-2018"
RUN2_T020_LABEL="${RUN2_T020_LABEL:-Run2_VBFDNN_Transformer_lt0p200}"
RUN2_TOPT_LABEL="${RUN2_TOPT_LABEL:-Run2_VBFDNN_Transformer_lt0p022}"
# Keep this separate from the stale pre-fix trained_ray8 submission record.
TRAIN_LABEL="${TRAIN_LABEL:-trained_ray8_concurrent}"

CPUS_PER_FOLD="${CPUS_PER_FOLD:-4}"
RAM_PER_WORKER_GB="${RAM_PER_WORKER_GB:-16}"
MIN_GPU_MEMORY_GB="${MIN_GPU_MEMORY_GB:-5}"
RAY_MAX_RETRIES="${RAY_MAX_RETRIES:-4}"
POLL_SECONDS="${POLL_SECONDS:-20}"

die() {
    echo "[ERROR] $*" >&2
    exit 2
}

[[ -x "${GLOBAL_PYTHON}" ]] || die "Global Python is missing: ${GLOBAL_PYTHON}"
[[ -f "${RAY_SUBMITTER}" ]] || die "Ray submitter is missing: ${RAY_SUBMITTER}"
[[ -f "${RUN2_CONFIG}" ]] || die "Run-2 config is missing: ${RUN2_CONFIG}"

# Purdue AF refuses Ray GPU-cluster submissions from a notebook session that
# already owns a GPU.
if command -v nvidia-smi >/dev/null 2>&1; then
    die "Run this script from a CPU-only Purdue AF session (no notebook GPU)."
fi

run2_t020_data="${REPO_ROOT}/dnn/trained_models/${RUN2_T020_LABEL}/${RUN2_YEAR_SLUG}_h-peak_vbf_all"
run2_topt_data="${REPO_ROOT}/dnn/trained_models/${RUN2_TOPT_LABEL}/${RUN2_YEAR_SLUG}_h-peak_vbf_all"

declare -a model_names=(
    "Run 2 score<0.200"
    "Run 2 score<0.022"
)
declare -a model_data_dirs=(
    "${run2_t020_data}"
    "${run2_topt_data}"
)

validate_preprocessing() {
    local data_dir="$1"
    local split fold path
    [[ -f "${data_dir}/preprocess_manifest.json" ]] || \
        die "Missing preprocess manifest: ${data_dir}/preprocess_manifest.json"
    for split in train validation evaluation; do
        for fold in 0 1 2 3; do
            path="${data_dir}/data_df_${split}_${fold}.parquet"
            [[ -f "${path}" ]] || die "Missing preprocessed fold file: ${path}"
        done
    done
    for fold in 0 1 2 3; do
        path="${data_dir}/scalers_${fold}.npz"
        [[ -f "${path}" ]] || die "Missing scaler: ${path}"
    done
}

training_complete() {
    local out_dir="$1"
    local fold
    [[ -f "${out_dir}/manifest.json" ]] || return 1
    for fold in 0 1 2 3; do
        [[ -f "${out_dir}/fold${fold}/best_torchscript.pt" ]] || return 1
    done
}

for data_dir in "${model_data_dirs[@]}"; do
    validate_preprocessing "${data_dir}"
done

declare -a submission_files=()
for index in 0 1; do
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
        --config "${RUN2_CONFIG}"
        --data-dir "${data_dir}"
        --out-dir "${out_dir}"
        --cluster-workers 8
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
    echo "Both Run-2 model trainings are already complete."
    exit 0
fi

echo
echo "Submitted ${#submission_files[@]} model job(s) to one exact-size 8-GPU Ray cluster."
echo "Each model schedules four independent folds with one GPU per fold."
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
echo "Both Run-2 four-fold VBF-DNN trainings completed successfully."
for index in 0 1; do
    echo "  ${model_names[${index}]} -> ${model_data_dirs[${index}]}/${TRAIN_LABEL}"
done
