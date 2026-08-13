#!/usr/bin/env bash
# The 2017 no-QvG three-step sweep over lambda x score_cut, end to end.
#
#   run_noqvg_grid.sh [--skip-null] [lambda ...]
#
# Every phase is resumable: anything already on disk is skipped, so the driver can
# be killed and restarted without losing work.
#
#   0  preprocess 2017 fold parquets WITHOUT the two QvG features
#      (configs/dnn_run2_vbf_noQvG.yaml), variation set 'sweep' = JEC Total +
#      mu_roccor, up/down = 4 variations.
#   1  STEP 1, the warm-up: original_loss(pred_nominal, label) to early stopping.
#      This is BOTH the lambda = 0 reference and the warm-start source for every
#      grid point.
#   2  chain step 1 -> the reference pre-fit and stat-only significance.
#   3  the null control: two further warm-started trainings with NO penalty term,
#      chained. Without it, "the grid degrades" cannot be told apart from "training
#      a converged model twice more at full LR degrades". The predecessor task
#      established this is not a hypothetical -- see its final-summary.md.
#   4  the grid. For each (lambda, score_cut):
#        STEP 2  warm start from step 1, penalty = sum_i loss(pred_i, label)[cut]
#        STEP 3  warm start from step 2, penalty =
#                sum_i [ 2*loss(pred_i, pred_nom)[cut] + loss(pred_i, label)[cut] ]
#        then chain the STEP 3 model only (analyst decision; step 2 is recorded
#        through its training metrics).
#   5  collect.
#
# Ordering is lambda descending and cut ascending, so the most aggressive
# configurations -- largest weight, largest selected population -- land first and
# a collapse is visible in the first hour rather than the last.
set -uo pipefail

SKIP_NULL=0
if [ "${1:-}" = "--skip-null" ]; then SKIP_NULL=1; shift; fi

REPO=/work/users/yun79/sideHustle2/copperheadV2
LABEL=Run2_NanoV15_forVBFChannel_July06_2026_jetUncRedo
STAGE1="/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/$LABEL/stage1_output"
YEAR=2017
DATE=Aug12_2026

TASKDIR="$REPO/.agent-system/tasks/04-adversarial-syst-vbf-dnn-2017-noqvg"
LOGDIR="$TASKDIR/logs"; mkdir -p "$LOGDIR"
STATUS="$LOGDIR/sweep.status"
DASK="$TASKDIR/scripts/dask_cluster.py"
CHAIN="$TASKDIR/scripts/chain_2017.sh"

CONFIG="$REPO/configs/dnn_run2_vbf_noQvG.yaml"
export PREP_TAG="Y2017_noQvG_sweep_${DATE}"
DATA_DIR="$REPO/dnn/trained_models/$PREP_TAG/${YEAR}_h-peak_vbf"
MODEL_BASE="$REPO/dnn/trained_models/$LABEL/${YEAR}_h-peak_vbf_noQvG"
OPTUNA="$REPO/dnn/trained_models/$LABEL/2018-2017-2016postVFP-2016preVFP_h-peak_vbf/hpo_optuna/100Trials_w_VBF_filterFoldAll/optuna_best.json"

STEP1_TAG="trained_2017nq_step1_${DATE}"
STEP1_POSTFIX="${DATE}_2017nq_step1"

# lambda descending; score_cut ascending, "none" first (it selects every event and
# is therefore the strongest version of the penalty).
LAMBDAS=("$@")
[ ${#LAMBDAS[@]} -eq 0 ] && LAMBDAS=(1.0 0.5 0.01 0.005)
CUTS=(none 0.6 1.0 2.0)

cd "$REPO"
say() { echo "$@" | tee -a "$STATUS"; }

[ -f "$OPTUNA" ]  || { say "=== REFUSING: optuna_best.json missing at $OPTUNA"; exit 2; }
[ -f "$CONFIG" ]  || { say "=== REFUSING: $CONFIG missing"; exit 2; }

say "=== 2017 noQvG GRID START $(date -u +%FT%TZ)"
say "=== LAMBDAS ${LAMBDAS[*]}   CUTS ${CUTS[*]}"

# `train` <tag> <log-suffix> <extra args...>
train() {
    local tag="$1"; shift
    local what="$1"; shift
    if [ -f "$MODEL_BASE/$tag/manifest.json" ]; then
        say "=== SKIP $what: $tag already complete"
        return 0
    fi
    if [ -e "$MODEL_BASE/$tag" ]; then
        say "=== SKIP $what: $tag exists but is incomplete; tags are never reused."
        return 1
    fi
    say "=== START $what tag=$tag $(date -u +%FT%TZ)"
    local t0 rc
    t0=$(date +%s)
    pixi run -e default python MVA_training/VBF_run3/train_dnn.py \
        --config "$CONFIG" \
        --data-dir "$DATA_DIR" \
        --out-dir "$MODEL_BASE/$tag" \
        --optuna-best-json "$OPTUNA" \
        "$@" \
        > "$LOGDIR/train_$tag.log" 2>&1
    rc=$?
    say "=== END   $what exit=$rc wall=$(( $(date +%s) - t0 ))s $(date -u +%FT%TZ)"
    return $rc
}

# ---- phase 0: preprocessing ----------------------------------------------------
if [ -f "$DATA_DIR/preprocess_manifest.json" ]; then
    say "=== SKIP preprocess: $DATA_DIR already built"
else
    say "=== START preprocess tag=$PREP_TAG (no QvG features) $(date -u +%FT%TZ)"
    pixi run -e default python "$DASK" create --cores 2 --memory 25 --workers 40 \
        >> "$LOGDIR/dask.log" 2>&1
    t0=$(date +%s)
    pixi run -e default python MVA_training/VBF_run3/preprocess_dnn.py \
        --config "$CONFIG" \
        --base-path "$STAGE1" \
        --tag "$PREP_TAG" \
        --years "$YEAR" \
        --no-plots \
        --include-systematic-variations \
        --variation-set sweep \
        --use-dask-gateway --cluster-index 0 \
        > "$LOGDIR/preprocess_$PREP_TAG.log" 2>&1
    rc=$?
    say "=== END   preprocess exit=$rc wall=$(( $(date +%s) - t0 ))s $(date -u +%FT%TZ)"
    pixi run -e default python "$DASK" stop >> "$LOGDIR/dask.log" 2>&1
    [ $rc -eq 0 ] || { say "=== ABORT: preprocessing failed."; exit 1; }
fi

# The feature list every downstream stage inherits. Assert once, loudly.
pixi run -e default python - "$DATA_DIR/training_features.pkl" <<'PY' | tee -a "$STATUS"
import pickle, sys
feats = pickle.load(open(sys.argv[1], "rb"))
bad = [f for f in feats if "QvG" in f]
print(f"=== FEATURES {len(feats)} training features; QvG columns present: {bad}")
sys.exit(1 if bad else 0)
PY
[ "${PIPESTATUS[0]}" -eq 0 ] || { say "=== ABORT: preprocessing kept a QvG feature."; exit 1; }

# ---- phase 1: STEP 1, the warm-up / lambda = 0 reference ------------------------
train "$STEP1_TAG" "step1 warm-up (lambda=0)" \
    || { say "=== ABORT: step-1 training failed; nothing downstream can run."; exit 1; }

# ---- phase 2: the reference chain ----------------------------------------------
if [ -f "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/$LABEL/stage3_datacards_${STEP1_POSTFIX}/score_${LABEL}/significance.txt" ]; then
    say "=== SKIP chain step1: significance already extracted"
else
    bash "$CHAIN" "$STEP1_TAG" "$STEP1_POSTFIX"
fi

# ---- phase 3: the three-phase null control --------------------------------------
# Same warm-start structure as a grid point (step1 -> A -> B, fresh optimizer each
# time), penalty absent. Anything this run does to the significance is the cost of
# the retraining procedure, not of the loss.
if [ "$SKIP_NULL" -eq 0 ]; then
    NULL2="trained_2017nq_null2_${DATE}"
    NULL3="trained_2017nq_null3_${DATE}"
    NULL_POSTFIX="${DATE}_2017nq_null"
    if train "$NULL2" "null control phase 2 (warm start, no penalty)" \
            --init-from "$MODEL_BASE/$STEP1_TAG" \
        && train "$NULL3" "null control phase 3 (warm start, no penalty)" \
            --init-from "$MODEL_BASE/$NULL2"; then
        if [ -f "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/$LABEL/stage3_datacards_${NULL_POSTFIX}/score_${LABEL}/significance.txt" ]; then
            say "=== SKIP chain null: significance already extracted"
        else
            bash "$CHAIN" "$NULL3" "$NULL_POSTFIX"
        fi
    else
        say "=== WARN: null control failed; the grid still runs but its points lose their attribution baseline."
    fi
fi

# ---- phase 4: the grid ----------------------------------------------------------
for LAM in "${LAMBDAS[@]}"; do
  for CUT in "${CUTS[@]}"; do
    LSLUG=$(echo "$LAM" | tr -d '.')
    CSLUG=$(echo "$CUT" | tr -d '.')
    S2_TAG="trained_2017nq_lam${LSLUG}_cut${CSLUG}_s2_${DATE}"
    S3_TAG="trained_2017nq_lam${LSLUG}_cut${CSLUG}_s3_${DATE}"
    POSTFIX="${DATE}_2017nq_lam${LSLUG}cut${CSLUG}"

    # score_cut = "all True" is expressed as no cut at all: the mask is a no-op
    # and skipping it avoids an arbitrary sentinel threshold in the manifest.
    CUT_ARGS=()
    [ "$CUT" != "none" ] && CUT_ARGS=(--adversarial-high-score-cut "$CUT")

    say "=== GRID POINT lambda=$LAM cut=$CUT $(date -u +%FT%TZ)"

    # STEP 2: variations against the truth label only.
    train "$S2_TAG" "step2 lambda=$LAM cut=$CUT" \
        --init-from "$MODEL_BASE/$STEP1_TAG" \
        --use_adversarial --adversarial-lambda "$LAM" \
        --adversarial-label-only \
        --adversarial-mask-label-term \
        --adversarial-consistency-smoothing none \
        --adversarial-detach-nominal \
        "${CUT_ARGS[@]}" \
        || { say "=== SKIP POINT lambda=$LAM cut=$CUT: step 2 failed."; continue; }

    # STEP 3: add the consistency term against the nominal score.
    train "$S3_TAG" "step3 lambda=$LAM cut=$CUT" \
        --init-from "$MODEL_BASE/$S2_TAG" \
        --use_adversarial --adversarial-lambda "$LAM" \
        --adversarial-mask-label-term \
        --adversarial-consistency-smoothing none \
        --adversarial-detach-nominal \
        "${CUT_ARGS[@]}" \
        || { say "=== SKIP POINT lambda=$LAM cut=$CUT: step 3 failed."; continue; }

    if [ -f "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/$LABEL/stage3_datacards_${POSTFIX}/score_${LABEL}/significance.txt" ]; then
        say "=== SKIP chain lambda=$LAM cut=$CUT: significance already extracted"
    else
        bash "$CHAIN" "$S3_TAG" "$POSTFIX"
    fi

    pixi run -e default python "$TASKDIR/scripts/collect_results.py" \
        --out "$TASKDIR/results.json" >> "$LOGDIR/collect.log" 2>&1
  done
done

# ---- phase 5: results ------------------------------------------------------------
say "=== 2017 noQvG GRID RESULTS $(date -u +%FT%TZ)"
pixi run -e default python "$TASKDIR/scripts/collect_results.py" \
    --out "$TASKDIR/results.json" 2>&1 | tee -a "$STATUS"
say "=== 2017 noQvG GRID DONE $(date -u +%FT%TZ)"
