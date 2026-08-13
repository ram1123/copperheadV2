#!/usr/bin/env bash
# Seed replicas at lambda = 0.5, score_cut = all True.
#
#   run_seed_replicas.sh [seed ...]      (default: 20260812 777 31337)
#
# Why
# ---
# The 16-point grid has no repeat measurements, so nothing in it can be told
# apart from run-to-run scatter. The only noise estimate available is the pair of
# lambda = 0 runs (step1 vs null), which differ by 0.85% in pre-fit and 0.43
# points in systematic headroom -- yet lambda = 0.01, a hundredth of the largest
# weight, moves pre-fit by 6-7% and spans 7.3 points of syst_hr between two cuts.
# Either the loss is wildly non-monotone at negligible weight, or the spread is
# noise. Repeat measurements settle it; ranking the grid without them does not.
#
# What is varied
# --------------
# ONLY --seed, on steps 2 and 3. Everything else -- config, data, optuna
# hyperparameters, warm-start source, lambda, cut -- is identical to the existing
# lambda=0.5/cut=all point, which serves as the first sample of the set.
#
# Step 1 is deliberately NOT re-seeded: every grid point warm-starts from the same
# step-1 model, so the variance that separates grid points from each other is the
# variance of steps 2-3 plus the per-model rebinning and the chain. That is what
# this measures. (Re-seeding step 1 as well would measure a larger variance than
# the one the grid comparisons actually suffer from.)
#
# Runs after the grid and the lambda=5 follow-up, because chain_2017.sh locks the
# global DNN binning file and concurrent chains would corrupt each other's edges.
set -uo pipefail

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(20260812 777 31337)

REPO=/work/users/yun79/sideHustle2/copperheadV2
LABEL=Run2_NanoV15_forVBFChannel_July06_2026_jetUncRedo
SAVE_PATH=/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/$LABEL
YEAR=2017
DATE=Aug12_2026

TASKDIR="$REPO/.agent-system/tasks/04-adversarial-syst-vbf-dnn-2017-noqvg"
LOGDIR="$TASKDIR/logs"
STATUS="$LOGDIR/sweep.status"
CHAIN="$TASKDIR/scripts/chain_2017.sh"
CONFIG="$REPO/configs/dnn_run2_vbf_noQvG.yaml"

export PREP_TAG="Y2017_noQvG_sweep_${DATE}"
DATA_DIR="$REPO/dnn/trained_models/$PREP_TAG/${YEAR}_h-peak_vbf"
MODEL_BASE="$REPO/dnn/trained_models/$LABEL/${YEAR}_h-peak_vbf_noQvG"
OPTUNA="$REPO/dnn/trained_models/$LABEL/2018-2017-2016postVFP-2016preVFP_h-peak_vbf/hpo_optuna/100Trials_w_VBF_filterFoldAll/optuna_best.json"
STEP1_TAG="trained_2017nq_step1_${DATE}"

LAM=0.5

cd "$REPO"
say() { echo "$@" | tee -a "$STATUS"; }

say "=== SEED REPLICAS: waiting for the grid and the lambda=5 follow-up $(date -u +%FT%TZ)"
while pgrep -u "$USER" -f "run_noqvg_grid\.sh|run_lam5_extra\.sh" >/dev/null 2>&1; do
    sleep 120
done
while [ -d "$LOGDIR/binning.lock" ]; do sleep 60; done
say "=== SEED REPLICAS: starting, seeds ${SEEDS[*]} $(date -u +%FT%TZ)"

train() {
    local tag="$1"; shift
    local what="$1"; shift
    if [ -f "$MODEL_BASE/$tag/manifest.json" ]; then
        say "=== SKIP $what: $tag already complete"; return 0
    fi
    if [ -e "$MODEL_BASE/$tag" ]; then
        say "=== SKIP $what: $tag exists but is incomplete; tags are never reused."; return 1
    fi
    say "=== START $what tag=$tag $(date -u +%FT%TZ)"
    local t0 rc; t0=$(date +%s)
    pixi run -e default python MVA_training/VBF_run3/train_dnn.py \
        --config "$CONFIG" --data-dir "$DATA_DIR" \
        --out-dir "$MODEL_BASE/$tag" --optuna-best-json "$OPTUNA" \
        "$@" > "$LOGDIR/train_$tag.log" 2>&1
    rc=$?
    say "=== END   $what exit=$rc wall=$(( $(date +%s) - t0 ))s $(date -u +%FT%TZ)"
    return $rc
}

for SEED in "${SEEDS[@]}"; do
    S2_TAG="trained_2017nq_lam05_cutnone_seed${SEED}_s2_${DATE}"
    S3_TAG="trained_2017nq_lam05_cutnone_seed${SEED}_s3_${DATE}"
    POSTFIX="${DATE}_2017nq_lam05cutnoneseed${SEED}"

    say "=== SEED REPLICA seed=$SEED (lambda=$LAM, cut=all) $(date -u +%FT%TZ)"

    train "$S2_TAG" "step2 seed=$SEED" \
        --seed "$SEED" \
        --init-from "$MODEL_BASE/$STEP1_TAG" \
        --use_adversarial --adversarial-lambda "$LAM" \
        --adversarial-label-only --adversarial-mask-label-term \
        --adversarial-consistency-smoothing none --adversarial-detach-nominal \
        || { say "=== SKIP seed=$SEED: step 2 failed."; continue; }

    train "$S3_TAG" "step3 seed=$SEED" \
        --seed "$SEED" \
        --init-from "$MODEL_BASE/$S2_TAG" \
        --use_adversarial --adversarial-lambda "$LAM" \
        --adversarial-mask-label-term \
        --adversarial-consistency-smoothing none --adversarial-detach-nominal \
        || { say "=== SKIP seed=$SEED: step 3 failed."; continue; }

    if [ -f "$SAVE_PATH/stage3_datacards_${POSTFIX}/score_${LABEL}/significance.txt" ]; then
        say "=== SKIP chain seed=$SEED: significance already extracted"
    else
        bash "$CHAIN" "$S3_TAG" "$POSTFIX"
    fi

    S3DIR="$SAVE_PATH/stage3_datacards_${POSTFIX}/score_${LABEL}"
    if [ -f "$S3DIR/HMuMu_13TeV_${YEAR}.txt" ] && [ ! -f "$S3DIR/significance_systfrozen.txt" ]; then
        pixi run -e combine bash "$TASKDIR/scripts/syst_only_significance.sh" \
            "$S3DIR" "$YEAR" 2>&1 | tee -a "$STATUS"
    fi

    HIST="$SAVE_PATH/stage2_histograms/score_${LABEL}_${POSTFIX}"
    if [ -d "$HIST" ]; then
        pixi run -e default python plotter/plot_stage2_variation_pdfs.py \
            --hist-path "$HIST" \
            --outdir "$TASKDIR/plots/stage2_variation_validation/${POSTFIX}" \
            --years "$YEAR" 2>&1 | tail -2 | tee -a "$STATUS"
    fi
done

say "=== SEED REPLICAS RESULTS $(date -u +%FT%TZ)"
pixi run -e default python "$TASKDIR/scripts/collect_results.py" \
    --out "$TASKDIR/results.json" 2>&1 | tee -a "$STATUS"
pixi run -e default python "$TASKDIR/scripts/seed_spread.py" \
    "$TASKDIR/results.json" 2>&1 | tee -a "$STATUS"
say "=== SEED REPLICAS DONE $(date -u +%FT%TZ)"
