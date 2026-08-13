#!/usr/bin/env bash
# Isolate the SHAPE-systematic cost from the MC-statistical one.
#
#   syst_only_significance.sh <stage3_score_dir> <year>
#
# Why this exists
# ---------------
# produce_significance.sh computes its "stat only" leg with
# `--freezeParameters allConstrainedNuisances`, which freezes the autoMCStats
# (Barlow-Beeston) parameters ALONG WITH the shape nuisances. So the headroom
# built from it,
#
#     headroom = (stat_only - prefit) / prefit
#
# is the combined cost of the systematics AND of the finite MC statistics -- and
# because this study re-derives the binning per model, the MC-stat part varies
# run to run for reasons that have nothing to do with the loss. Measured on
# lambda=0.5/cut=0.6: stat-only 1.277 against 0.80-0.94 for every other run, with
# an unremarkable pre-fit, i.e. the headroom column was reading MC statistics.
#
# This script adds the leg that was missing: freeze ONLY the two shape nuisances
# the training is shown, leaving autoMCStats floating. Then
#
#     syst_headroom = (syst_frozen - prefit) / prefit
#
# is the quantity the consistency term is actually supposed to shrink.
#
# Appends a "Significance (Syst Frozen)" block to significance.txt, so the
# existing parser keeps working and collect_results.py can pick it up by name.
set -uo pipefail

DIR="${1:?usage: syst_only_significance.sh <stage3_score_dir> <year>}"
YEAR="${2:?usage: syst_only_significance.sh <stage3_score_dir> <year>}"
CARD="$DIR/HMuMu_13TeV_${YEAR}.txt"

[ -f "$CARD" ] || { echo "no combine card at $CARD" >&2; exit 2; }

# The nuisance names as they appear in the datacard after stripping.
FREEZE="Total,mu_roccor${YEAR}"

{
    echo "========================================"
    echo ""
    echo "date: $(date)"
    echo "First argument: $DIR"
    echo "Second argument: $YEAR"
    echo "Third argument: systfrozen_${YEAR}"
    echo ""
    echo "Significance (Syst Frozen):"
} >> "$DIR/significance_systfrozen.txt"

combineTool.py -d "$CARD" -M Significance -m 125 --expectSignal=1 \
    -n "_${YEAR}_systfrozen_" -t -1 --rMin -2 --rMax 5 \
    --freezeParameters "$FREEZE" \
    > "$DIR/prefitsignificance_SystFrozen.log" 2>&1

grep "Significance:" "$DIR/prefitsignificance_SystFrozen.log" \
    >> "$DIR/significance_systfrozen.txt"
tail -2 "$DIR/significance_systfrozen.txt"
