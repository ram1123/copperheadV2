#!/usr/bin/env bash
# Re-evaluate an already-trained model on SOMEONE ELSE'S bin edges.
#
#   chain_fixed_binning.sh <training_tag> <save_postfix> <binning_yaml>
#
# Why
# ---
# Every run in this study re-derived its own binning, and the seed replicas
# measured what that costs: sigma(pre-fit) = 0.072 (7.7%) at a FIXED setting,
# against a 0.190 spread across the entire lambda x score_cut grid. The grid is
# therefore unrankable -- the binning moves the answer more than the loss does.
#
# This script removes that variable. It installs a given binning file and runs
# Stage-2/3 on it, so two models can be compared on identical edges.
#
# It skips `compact` and `scan_bins_for_dnn.py` entirely, and that is correct
# rather than a shortcut: run_stage2_vbf.py reads the UNTAGGED compacted Stage-1
# parquets (`stage1_output/<year>/f1_0` -> get_compacted_path) and evaluates the
# DNN itself from --model_path/--model_tag. The DNN-score compaction exists only
# to feed the bin scanner, which is precisely the step being bypassed. So the
# whole re-evaluation is a few minutes instead of ~25.
#
# modules/selection.py reads configs/MVA/VBF/dnn_binning.yaml at IMPORT time, so
# installing the file before launching Stage-2 is sufficient -- and is also why
# the binning lock is taken here as well.
set -uo pipefail

TAG="${1:?usage: chain_fixed_binning.sh <training_tag> <save_postfix> <binning_yaml>}"
POSTFIX="${2:?usage: chain_fixed_binning.sh <training_tag> <save_postfix> <binning_yaml>}"
BINNING_SRC="${3:?usage: chain_fixed_binning.sh <training_tag> <save_postfix> <binning_yaml>}"

REPO=/work/users/yun79/sideHustle2/copperheadV2
LABEL=Run2_NanoV15_forVBFChannel_July06_2026_jetUncRedo
SAVE_PATH=/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/$LABEL
MODEL_PATH="$REPO/dnn/trained_models/$LABEL/2017_h-peak_vbf_noQvG"
DATASET=configs/datasets/dataset_nanoAODv15_run2.yaml
YEAR=2017
TASKDIR="$REPO/.agent-system/tasks/04-adversarial-syst-vbf-dnn-2017-noqvg"
LOGDIR="$TASKDIR/logs"; mkdir -p "$LOGDIR"
SHAPEDIR="$TASKDIR/shape"; mkdir -p "$SHAPEDIR"
STATUS="$LOGDIR/chain.status"
BINNING_YAML="$REPO/configs/MVA/VBF/dnn_binning.yaml"
DASK="$TASKDIR/scripts/dask_cluster.py"
LOCK="$LOGDIR/binning.lock"
STAGE3_DIR="$SAVE_PATH/stage3_datacards_${POSTFIX}/score_${LABEL}"

cd "$REPO"
say() { echo "$@" | tee -a "$STATUS"; }
dask_up()   { pixi run -e default python "$DASK" create --cores 2 --memory 25 \
                  --workers "${DASK_WORKERS:-40}" --min-workers 10 >> "$LOGDIR/dask.log" 2>&1; }
dask_down() { pixi run -e default python "$DASK" stop >> "$LOGDIR/dask.log" 2>&1; }

[ -d "$MODEL_PATH/$TAG" ]  || { say "=== ABORT $TAG: model dir missing: $MODEL_PATH/$TAG"; exit 2; }
[ -f "$BINNING_SRC" ]      || { say "=== ABORT $TAG: binning file missing: $BINNING_SRC"; exit 2; }

# Preprocessing artifacts beside the model tags, as Stage-2 expects.
PREP_DATA_DIR="$REPO/dnn/trained_models/${PREP_TAG:-Y2017_noQvG_sweep_Aug12_2026}/${YEAR}_h-peak_vbf"
for f in training_features.pkl training_features.json \
         scalers_0.npz scalers_1.npz scalers_2.npz scalers_3.npz; do
    [ -f "$PREP_DATA_DIR/$f" ] && cp -f "$PREP_DATA_DIR/$f" "$MODEL_PATH/$f"
done

if ! mkdir "$LOCK" 2>/dev/null; then
    say "=== ABORT $TAG: another chain holds $LOCK; the binning file is global."
    exit 4
fi
# Always put the previous binning back, whatever happens.
BACKUP="$LOGDIR/dnn_binning_before_${POSTFIX}.yaml"
cp "$BINNING_YAML" "$BACKUP"
restore() { cp -f "$BACKUP" "$BINNING_YAML"; rmdir "$LOCK" 2>/dev/null; }
trap restore EXIT

cp -f "$BINNING_SRC" "$BINNING_YAML"
say "=== FIXED BINNING $TAG <- $(basename "$BINNING_SRC") " \
    "n_bins=$(grep -m1 '^n_bins:' "$BINNING_YAML" | awk '{print $2}')"

step() {   # step <mode> <label>
    local mode="$1" name="$2" t0 rc
    say "=== START $TAG $name $(date -u +%FT%TZ)"
    t0=$(date +%s)
    pixi run -e default bash stage1_loop_Improved.sh \
        -c "$DATASET" -v 15 -l "$LABEL" -y "$YEAR" -m "$mode" -k \
        -o "$POSTFIX" -P "$MODEL_PATH" -T "$TAG" \
        >> "$LOGDIR/chain_${POSTFIX}_m${mode}.log" 2>&1
    rc=$?
    say "=== END   $TAG $name exit=$rc wall=$(( $(date +%s) - t0 ))s $(date -u +%FT%TZ)"
    return $rc
}

dask_up || { say "=== ABORT $TAG: no dask cluster for Stage-2."; exit 3; }
step 2  "stage2"  || { say "=== ABORT $TAG: stage2 failed."; dask_down; exit 1; }
step 2p "stage2p" || say "=== WARN  $TAG: -m 2p failed; it does not feed the significance."
step 3  "stage3"  || { say "=== ABORT $TAG: stage3 failed."; dask_down; exit 1; }
dask_down

[ -d "$STAGE3_DIR" ] || { say "=== ABORT $TAG: no stage3 dir at $STAGE3_DIR"; exit 1; }

pixi run -e default python "$TASKDIR/scripts/check_template_occupancy.py" \
    "$STAGE3_DIR" --postfix "$POSTFIX" --year "$YEAR" \
    --out "$TASKDIR/occupancy_${POSTFIX}.json" 2>&1 | tee -a "$STATUS"

pixi run -e default python "$TASKDIR/scripts/variation_shape_consistency.py" \
    "$STAGE3_DIR" --postfix "$POSTFIX" --year "$YEAR" \
    --out "$SHAPEDIR/shape_${POSTFIX}.json" \
    --plot "$SHAPEDIR/shape_${POSTFIX}.png" 2>&1 | tee -a "$STATUS"

pixi run -e default python scripts/strip_datacard_variations.py \
    "$STAGE3_DIR" --years "$YEAR" --keep Total,mu_roccor 2>&1 | tee -a "$STATUS"

pixi run -e combine sh scripts/produce_combine_cards.sh "$STAGE3_DIR" "$YEAR" \
    >> "$LOGDIR/chain_${POSTFIX}_cards.log" 2>&1 \
    || say "=== WARN  $TAG: produce_combine_cards.sh failed"
pixi run -e combine bash scripts/produce_significance.sh "$STAGE3_DIR" "$YEAR" "$POSTFIX" \
    >> "$LOGDIR/chain_${POSTFIX}_signif.log" 2>&1 \
    || say "=== WARN  $TAG: produce_significance.sh failed"
pixi run -e combine bash "$TASKDIR/scripts/syst_only_significance.sh" \
    "$STAGE3_DIR" "$YEAR" 2>&1 | tee -a "$STATUS"

HIST="$SAVE_PATH/stage2_histograms/score_${LABEL}_${POSTFIX}"
if [ -d "$HIST" ]; then
    pixi run -e default python plotter/plot_stage2_variation_pdfs.py \
        --hist-path "$HIST" \
        --outdir "$TASKDIR/plots/stage2_variation_validation/${POSTFIX}" \
        --years "$YEAR" 2>&1 | tail -2 | tee -a "$STATUS"
fi

grep -E "^Significance:" "$STAGE3_DIR/significance.txt" 2>/dev/null | tail -2 | tee -a "$STATUS"
say "=== DONE  $TAG (fixed binning) $(date -u +%FT%TZ)"
