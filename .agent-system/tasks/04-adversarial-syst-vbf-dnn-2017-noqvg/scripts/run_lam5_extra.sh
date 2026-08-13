#!/usr/bin/env bash
# Follow-up point requested by the analyst: lambda = 5.0 at score_cut = all True.
#
#   run_lam5_extra.sh
#
# Waits for the main grid driver to finish, then runs the single extra point
# through the same three-step schedule and the same chain, and finally the
# syst-frozen combine leg and the Stage-2 variation PDFs.
#
# It WAITS rather than running concurrently because chain_2017.sh takes a lock on
# the global DNN binning file; two chains at once would score one model against
# the other's edges, and the lock turns that into a hard failure that would cost
# a grid point.
#
# lambda = 5.0 is 5x the largest value in the grid. At lambda = 1.0 the penalty
# already outweighs the nominal term ~12:1 and moves only ~1.3% over training,
# so this point tests whether the pre-fit cost keeps scaling with lambda or
# saturates once the penalty is far past the entropy floor it cannot cross.
set -uo pipefail

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

LAM=5.0
S2_TAG="trained_2017nq_lam50_cutnone_s2_${DATE}"
S3_TAG="trained_2017nq_lam50_cutnone_s3_${DATE}"
POSTFIX="${DATE}_2017nq_lam50cutnone"

cd "$REPO"
say() { echo "$@" | tee -a "$STATUS"; }

# ---- wait for the grid to finish -------------------------------------------
say "=== LAM5 EXTRA: waiting for the main grid driver $(date -u +%FT%TZ)"
while ! grep -q "=== 2017 noQvG GRID DONE" "$LOGDIR/driver.log" 2>/dev/null; do
    sleep 120
done
# and for any chain still holding the binning lock
while [ -d "$LOGDIR/binning.lock" ]; do sleep 60; done
say "=== LAM5 EXTRA: grid finished, starting lambda=$LAM cut=all $(date -u +%FT%TZ)"

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

# STEP 2: variations against the truth label only. score_cut = all True, so no cut.
train "$S2_TAG" "step2 lambda=$LAM cut=all" \
    --init-from "$MODEL_BASE/$STEP1_TAG" \
    --use_adversarial --adversarial-lambda "$LAM" \
    --adversarial-label-only --adversarial-mask-label-term \
    --adversarial-consistency-smoothing none --adversarial-detach-nominal \
    || { say "=== ABORT LAM5: step 2 failed."; exit 1; }

# STEP 3: add the consistency term against the nominal score.
train "$S3_TAG" "step3 lambda=$LAM cut=all" \
    --init-from "$MODEL_BASE/$S2_TAG" \
    --use_adversarial --adversarial-lambda "$LAM" \
    --adversarial-mask-label-term \
    --adversarial-consistency-smoothing none --adversarial-detach-nominal \
    || { say "=== ABORT LAM5: step 3 failed."; exit 1; }

if [ -f "$SAVE_PATH/stage3_datacards_${POSTFIX}/score_${LABEL}/significance.txt" ]; then
    say "=== SKIP chain lambda=$LAM: significance already extracted"
else
    bash "$CHAIN" "$S3_TAG" "$POSTFIX"
fi

# ---- the syst-frozen combine leg (autoMCStats left floating) ----------------
S3DIR="$SAVE_PATH/stage3_datacards_${POSTFIX}/score_${LABEL}"
if [ -f "$S3DIR/HMuMu_13TeV_${YEAR}.txt" ] && [ ! -f "$S3DIR/significance_systfrozen.txt" ]; then
    say "=== LAM5 EXTRA: syst-frozen significance $(date -u +%FT%TZ)"
    pixi run -e combine bash "$TASKDIR/scripts/syst_only_significance.sh" \
        "$S3DIR" "$YEAR" 2>&1 | tee -a "$STATUS"
fi

# ---- Stage-2 variation PDFs for inspection ---------------------------------
HIST="$SAVE_PATH/stage2_histograms/score_${LABEL}_${POSTFIX}"
if [ -d "$HIST" ]; then
    say "=== LAM5 EXTRA: stage-2 variation PDFs $(date -u +%FT%TZ)"
    pixi run -e default python plotter/plot_stage2_variation_pdfs.py \
        --hist-path "$HIST" \
        --outdir "$TASKDIR/plots/stage2_variation_validation/${POSTFIX}" \
        --years "$YEAR" 2>&1 | tail -2 | tee -a "$STATUS"
else
    say "=== LAM5 EXTRA WARN: no stage-2 histograms at $HIST; skipping the PDFs."
fi

say "=== LAM5 EXTRA RESULTS $(date -u +%FT%TZ)"
pixi run -e default python "$TASKDIR/scripts/collect_results.py" \
    --out "$TASKDIR/results.json" 2>&1 | tee -a "$STATUS"
say "=== LAM5 EXTRA DONE $(date -u +%FT%TZ)"
