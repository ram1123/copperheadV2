#!/bin/bash
set -euo pipefail

# Recommended usage:
# - Serious HPO: full 40 GB A100, ~8 CPU cores, ~32 GB RAM
# - Quick debug: A100 MIG 1g.5gb, ~4 CPU cores, ~16 GB RAM
#
# Override any of the variables below from the shell, for example:
#   TAG=my_tag HPO_TRIALS=12 HPO_FOLDS=0,1 bash run_dnn.sh

# /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn25GeV_Apr09_tightPassLepVeto_NoJER_v2/stage1_output/2022preEE/f1_0/
LABEL="Run3_nanoAODv12_FilterJetsHorn25GeV_pySR_Apr09_tightPassLepVeto_NoJER"
# LABEL="Run3_nanoAODv12_FilterJetsHorn25GeV_Apr09_tightPassLepVeto_NoJER_v2"
CONFIG="${CONFIG:-configs/dnn_run3_vbf.yaml}"
BASE_PATH="${BASE_PATH:-/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/${LABEL}/stage1_output/}"
TAG="${TAG:-${LABEL}}"
YEARS="${YEARS:-2022preEE,2022postEE,2023,2023BPix,2024}"
# YEARS="${YEARS:-2022postEE}"
REGION="${REGION:-h-peak}"
CATEGORY="${CATEGORY:-vbf}"

USE_DASK_GATEWAY="${USE_DASK_GATEWAY:-1}"
CLUSTER_INDEX="${CLUSTER_INDEX:-0}"

# HPO defaults:
# - Use multiple folds by default for robustness.
# - Start with a moderate trial count; increase after checking stability.
# HPO_FOLDS="${HPO_FOLDS:-0,1}"
HPO_FOLDS="${HPO_FOLDS:-0,1,2,3}"
HPO_TRIALS="${HPO_TRIALS:-50}"
# HPO_LABEL="${HPO_LABEL:-v2_multifold_100Trials_bothVBFggH}"
HPO_LABEL="${HPO_LABEL:-v2_multifold_050Trials_MLLBinnedDY}"

TRAIN_LABEL="${TRAIN_LABEL:-trained_best_optuna_${HPO_LABEL}}"

OUT_BASE="dnn/trained_models/${TAG}/${YEARS//,/-}_${REGION}_${CATEGORY}"
HPO_OUT_DIR="${OUT_BASE}/hpo_optuna/${HPO_LABEL}"
TRAIN_OUT_DIR="${OUT_BASE}/${TRAIN_LABEL}"
OPTUNA_BEST_JSON="${OPTUNA_BEST_JSON:-${HPO_OUT_DIR}/optuna_best.json}"

PREPROCESS_CMD=(
  python MVA_training/VBF_run3/preprocess_dnn.py
  --config "${CONFIG}"
  --base-path "${BASE_PATH}"
  --tag "${TAG}"
  --years "${YEARS}"
)

if [[ "${USE_DASK_GATEWAY}" == "1" ]]; then
  PREPROCESS_CMD+=(--use-dask-gateway --cluster-index "${CLUSTER_INDEX}")
fi

echo "[run_dnn] Output base: ${OUT_BASE}"
echo "[run_dnn] HPO output:  ${HPO_OUT_DIR}"
echo "[run_dnn] Train output:${TRAIN_OUT_DIR}"
echo "[run_dnn] HPO folds:   ${HPO_FOLDS}"
echo "[run_dnn] HPO trials:  ${HPO_TRIALS}"
echo "[run_dnn] Optuna best: ${OPTUNA_BEST_JSON}"

# Run the pre-process script
"${PREPROCESS_CMD[@]}"

time python MVA_training/VBF_run3/hpo_optuna.py \
  --config "${CONFIG}" \
  --data-dir "${OUT_BASE}/" \
  --out-dir "${HPO_OUT_DIR}" \
  --n-trials "${HPO_TRIALS}" \
  --folds "${HPO_FOLDS}"

time python MVA_training/VBF_run3/train_dnn.py \
  --config "${CONFIG}" \
  --data-dir "${OUT_BASE}/" \
  --out-dir "${TRAIN_OUT_DIR}" \
  --optuna-best-json "${OPTUNA_BEST_JSON}"
