#!/usr/bin/env bash
set -euo pipefail

# Train the already-preprocessed Run-2 Transformer-selected VBF-DNN for
# transf_ggh_vs_vbf_binary_score < 0.200.
# One Ray cluster has exactly four one-GPU workers: one GPU per fold.
# This script intentionally does not run preprocessing or HPO.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

GLOBAL_PYTHON="/work/pixi/global/.pixi/envs/default/bin/python"
RAY_SUBMITTER="MVA_training/VBF_run3/submit_ray_dnn.py"
RUN2_CONFIG="${REPO_ROOT}/configs/dnn_run2_vbf.yaml"
RUN2_YEAR_SLUG="2016preVFP-2016postVFP-2017-2018"
RUN2_T020_LABEL="${RUN2_T020_LABEL:-Run2_VBFDNN_Transformer_lt0p200}"
TRAIN_LABEL="${TRAIN_LABEL:-trained_ray4}"

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

data_dir="${REPO_ROOT}/dnn/trained_models/${RUN2_T020_LABEL}/${RUN2_YEAR_SLUG}_h-peak_vbf_all"
out_dir="${data_dir}/${TRAIN_LABEL}"

validate_preprocessing() {
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
    local fold
    [[ -f "${out_dir}/manifest.json" ]] || return 1
    for fold in 0 1 2 3; do
        [[ -f "${out_dir}/fold${fold}/best_torchscript.pt" ]] || return 1
    done
}

validate_preprocessing
submission_file="${out_dir}/ray_submission.json"
if [[ -f "${submission_file}" ]]; then
    echo "[cleanup] Removing previous Ray submission record: ${submission_file}"
    rm -f -- "${submission_file}"
fi
if training_complete; then
    echo "Run-2 score<0.200 training is already complete: ${out_dir}"
    exit 0
fi

submit_cmd=(
    "${GLOBAL_PYTHON}" "${RAY_SUBMITTER}" submit
    --config "${RUN2_CONFIG}"
    --data-dir "${data_dir}"
    --out-dir "${out_dir}"
    --cluster-workers 4
    --cpus-per-worker "${CPUS_PER_FOLD}"
    --ram-per-worker-gb "${RAM_PER_WORKER_GB}"
    --min-gpu-memory-gb "${MIN_GPU_MEMORY_GB}"
    --max-retries "${RAY_MAX_RETRIES}"
    --wait
    --poll-seconds "${POLL_SECONDS}"
)

echo "[submit] Run 2 score<0.200"
echo "         data: ${data_dir}"
echo "         out:  ${out_dir}"
echo "         Ray:  4 folds x 1 GPU = 4 GPUs"
"${submit_cmd[@]}"

echo "Run-2 score<0.200 four-fold VBF-DNN training completed: ${out_dir}"
