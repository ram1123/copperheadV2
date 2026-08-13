#!/usr/bin/env bash
# One trained 2017 model -> its own DNN binning -> Stage-2 -> Stage-3 -> stripped
# datacards -> 2017 pre-fit and stat-only significance -> occupancy and template
# shape-consistency reports.
#
#   chain_2017.sh <training_tag> <save_postfix>
#
# Carried over from 03-adversarial-syst-vbf-dnn-2017-fast, with three changes:
#
#   1. The model base directory is 2017_h-peak_vbf_noQvG, kept separate from the
#      QvG-carrying task. Stage-2 resolves training_features.pkl and scalers_*.npz
#      from the model base, so two feature sets CANNOT share one base directory --
#      the second run would silently score with the first run's feature list.
#   2. The mirror of the preprocessing artifacts is forced and then verified to be
#      QvG-free. `cp -n` would leave a stale file in place, which is exactly the
#      failure mode change 1 exists to prevent.
#   3. variation_shape_consistency.py runs on the Stage-3 templates, so every run
#      records how far the Total / mu_roccor up/down templates sit from nominal.
#
# WARNING: the binning yaml is a single global file. Two chains must never run at
# the same time, or one will score against the other's edges. The lock below makes
# that a hard failure instead of a silent one.
set -uo pipefail

TAG="${1:?usage: chain_2017.sh <training_tag> <save_postfix>}"
POSTFIX="${2:?usage: chain_2017.sh <training_tag> <save_postfix>}"

REPO=/work/users/yun79/sideHustle2/copperheadV2
LABEL=Run2_NanoV15_forVBFChannel_July06_2026_jetUncRedo
SAVE_PATH=/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/$LABEL
MODEL_PATH="$REPO/dnn/trained_models/$LABEL/2017_h-peak_vbf_noQvG"
DATASET=configs/datasets/dataset_nanoAODv15_run2.yaml
YEAR=2017
TASKDIR="$REPO/.agent-system/tasks/04-adversarial-syst-vbf-dnn-2017-noqvg"
LOGDIR="$TASKDIR/logs"; mkdir -p "$LOGDIR"
BINDIR="$TASKDIR/binning"; mkdir -p "$BINDIR"
SHAPEDIR="$TASKDIR/shape"; mkdir -p "$SHAPEDIR"
STATUS="$LOGDIR/chain.status"
BINNING_YAML="$REPO/configs/MVA/VBF/dnn_binning.yaml"
DASK="$TASKDIR/scripts/dask_cluster.py"
LOCK="$LOGDIR/binning.lock"

STAGE3_DIR="$SAVE_PATH/stage3_datacards_${POSTFIX}/score_${LABEL}"
COMPACT_TAG="${POSTFIX}_FixDimuonMass"

cd "$REPO"
say() { echo "$@" | tee -a "$STATUS"; }
dask_up()   { pixi run -e default python "$DASK" create --cores 2 --memory 25 \
                  --workers "${DASK_WORKERS:-40}" --min-workers 10 >> "$LOGDIR/dask.log" 2>&1; }
dask_down() { pixi run -e default python "$DASK" stop >> "$LOGDIR/dask.log" 2>&1; }

if [ ! -d "$MODEL_PATH/$TAG" ]; then
    say "=== ABORT $TAG: trained model directory missing: $MODEL_PATH/$TAG"
    exit 2
fi

# compact_parquet_data.resolve_vbf_training_layout and run_stage2_vbf both expect
# training_features.pkl / scalers_*.npz BESIDE the trained-model tags, not in the
# preprocessing tag's own directory. Mirror them every time (-f, not -n) so a tag
# from a different feature set can never be scored with a stale list.
PREP_DATA_DIR="$REPO/dnn/trained_models/${PREP_TAG:-Y2017_noQvG_sweep_Aug12_2026}/${YEAR}_h-peak_vbf"
for f in training_features.pkl training_features.json \
         scalers_0.npz scalers_1.npz scalers_2.npz scalers_3.npz; do
    [ -f "$PREP_DATA_DIR/$f" ] && cp -f "$PREP_DATA_DIR/$f" "$MODEL_PATH/$f"
done
if [ ! -f "$MODEL_PATH/training_features.pkl" ]; then
    say "=== ABORT $TAG: no training_features.pkl under $MODEL_PATH; compact cannot resolve the layout."
    exit 2
fi
# The whole point of this task is that no QvG column reaches inference. Assert it
# on the file Stage-2 will actually read, not on the config it came from.
if ! pixi run -e default python - "$MODEL_PATH/training_features.pkl" <<'PY'
import pickle, sys
feats = pickle.load(open(sys.argv[1], "rb"))
bad = [f for f in feats if "QvG" in f]
print(f"  FEATURES: {len(feats)} training features, QvG columns: {bad}")
sys.exit(1 if bad else 0)
PY
then
    say "=== ABORT $TAG: $MODEL_PATH/training_features.pkl still carries a QvG feature."
    exit 2
fi

# Serialise on the global binning file.
if ! mkdir "$LOCK" 2>/dev/null; then
    say "=== ABORT $TAG: another chain holds $LOCK. The DNN binning is a single"
    say "    global file (configs/MVA/VBF/dnn_binning.yaml); chains cannot overlap."
    exit 4
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

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

# ---- 1. compact: write this model's DNN score onto the Stage-1 parquets --------
dask_up || { say "=== ABORT $TAG: no dask cluster for the compact step."; exit 3; }
step compact "compact" || { say "=== ABORT $TAG: compact failed."; dask_down; exit 1; }

# ---- 2. re-derive the binning (local client; the gateway cluster must be down) --
dask_down
say "=== START $TAG scan_bins $(date -u +%FT%TZ)"
t0=$(date +%s)
pixi run -e default python MVA_training/VBF_run3/scan_bins_for_dnn.py \
    --stage1-path "$SAVE_PATH" \
    --compacted-tag "$COMPACT_TAG" \
    --year "$YEAR" \
    --min-background-per-bin 0.05 \
    >> "$LOGDIR/chain_${POSTFIX}_scanbins.log" 2>&1
rc=$?
say "=== END   $TAG scan_bins exit=$rc wall=$(( $(date +%s) - t0 ))s $(date -u +%FT%TZ)"
[ $rc -eq 0 ] || { say "=== ABORT $TAG: bin scan failed; refusing to run Stage-2 on stale edges."; exit 1; }

cp "$BINNING_YAML" "$BINDIR/dnn_binning_${POSTFIX}.yaml"
say "=== BINNING $TAG n_bins=$(grep -m1 '^n_bins:' "$BINNING_YAML" | awk '{print $2}') archived to binning/dnn_binning_${POSTFIX}.yaml"

# ---- 3. Stage-2 / Stage-3 ------------------------------------------------------
dask_up || { say "=== ABORT $TAG: no dask cluster for Stage-2."; exit 3; }
step 2  "stage2"  || { say "=== ABORT $TAG: stage2 failed."; dask_down; exit 1; }
step 2p "stage2p" || say "=== WARN  $TAG: -m 2p (validation plots) failed; it does not feed the significance."
step 3  "stage3"  || { say "=== ABORT $TAG: stage3 failed."; dask_down; exit 1; }
dask_down

if [ ! -d "$STAGE3_DIR" ]; then
    say "=== ABORT $TAG: expected stage3 dir not found: $STAGE3_DIR"
    exit 1
fi

# ---- 4. occupancy guard: does any bin repeat the near-empty-bin artefact? -------
pixi run -e default python "$TASKDIR/scripts/check_template_occupancy.py" \
    "$STAGE3_DIR" --postfix "$POSTFIX" --year "$YEAR" \
    --out "$TASKDIR/occupancy_${POSTFIX}.json" 2>&1 | tee -a "$STATUS"

# ---- 5. nominal vs Total/mu_roccor up-down template agreement -------------------
# Run BEFORE the stripper: it reads the .full card if the stripped one is gone,
# but running first keeps the report independent of the stripper's behaviour.
pixi run -e default python "$TASKDIR/scripts/variation_shape_consistency.py" \
    "$STAGE3_DIR" --postfix "$POSTFIX" --year "$YEAR" \
    --out "$SHAPEDIR/shape_${POSTFIX}.json" \
    --plot "$SHAPEDIR/shape_${POSTFIX}.png" 2>&1 | tee -a "$STATUS"

# ---- 6. keep only the shape nuisances the training was shown -------------------
say "=== START $TAG strip_datacards $(date -u +%FT%TZ)"
pixi run -e default python scripts/strip_datacard_variations.py \
    "$STAGE3_DIR" --years "$YEAR" --keep Total,mu_roccor 2>&1 | tee -a "$STATUS"

# ---- 7. combine ----------------------------------------------------------------
say "=== START $TAG combine $(date -u +%FT%TZ)"
t0=$(date +%s)
pixi run -e combine sh scripts/produce_combine_cards.sh "$STAGE3_DIR" "$YEAR" \
    >> "$LOGDIR/chain_${POSTFIX}_cards.log" 2>&1 \
    || say "=== WARN  $TAG: produce_combine_cards.sh failed for $YEAR"
pixi run -e combine bash scripts/produce_significance.sh "$STAGE3_DIR" "$YEAR" "$POSTFIX" \
    >> "$LOGDIR/chain_${POSTFIX}_signif.log" 2>&1 \
    || say "=== WARN  $TAG: produce_significance.sh failed for $YEAR"
say "=== END   $TAG combine wall=$(( $(date +%s) - t0 ))s $(date -u +%FT%TZ)"

say "=== PATHS $TAG stage3=$STAGE3_DIR"
grep -A1 "Prefit Significance:\|Significance (Stat Only):" "$STAGE3_DIR/significance.txt" 2>/dev/null \
    | tail -8 | tee -a "$STATUS"
say "=== DONE  $TAG $(date -u +%FT%TZ)"
