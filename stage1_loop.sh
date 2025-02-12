#!/bin/bash
# Stop execution on any error
set -e

# If there is `from src.<something> import <something>
# export PYTHONPATH=$PYTHONPATH:/depot/cms/users/shar1172/copperheadV2/

# Default settings
datasetYAML="configs/datasets/dataset.yaml"
year="2018"
NanoAODv=9
data_l="A B C D"
bkg_l="DY Top VV"
sig_l=""
chunksize=300000
label="Feb11_WithPurdueZptWgt"
save_path="/depot/cms/users/$USER/hmm/copperheadV1clean/$label/"

# Get mode from command-line argument
mode=$1  # Options: "prestage" or "stage1"

echo "Selected mode: $mode"
echo "Save Path: $save_path"

if [[ "$mode" == "0" ]]; then
    echo "Running pre-stage..."
    python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l  --NanoAODv $NanoAODv --log-level DEBUG
    # python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l  --NanoAODv $NanoAODv --use_gateway --log-level DEBUG
    # python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l  --NanoAODv $NanoAODv --use_gateway --log-level DEBUG -frac 0.1
elif [[ "$mode" == "1" ]]; then
    echo "Running stage1..."
    python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv --use_gateway --log-level DEBUG
else
    echo "Error: Invalid mode. Please use '0' for prestage or '1' for stage1"
    exit 1
fi
