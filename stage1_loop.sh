#!/bin/bash
# Stop execution on any error
set -e

# If there is `from src.<something> import <something>
# export PYTHONPATH=$PYTHONPATH:/depot/cms/users/shar1172/copperheadV2/

# Default settings
datasetYAML="configs/datasets/dataset.yaml"
NanoAODv=9

year="2018"
# data_l="A B C D"
data_l=""

# year="2017"
# data_l="B C D E F"

# year="2016"
# data_l="F G H"

# year="2016APV"
# data_l="B C D E F"


bkg_l="DY Top VV"
# bkg_l="Top"
sig_l=""

chunksize=300000
label="Feb11_WithPurdueZptWgt_DY_WithoutLHECut_16Feb"
save_path="/depot/cms/users/$USER/hmm/copperheadV1clean/$label/"

# Get mode from command-line argument
mode=$1  # Options: "prestage" or "stage1"

echo "Selected mode: $mode"
echo "Save Path: $save_path"

if [[ "$mode" == "0" || "$mode" == "all" ]]; then
    echo "Running pre-stage..."
    if [[ "$2" == "test" ]]; then
        echo "python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l  --NanoAODv $NanoAODv --use_gateway --log-level DEBUG -frac 0.1"
        python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l  --NanoAODv $NanoAODv --use_gateway --log-level DEBUG -frac 0.1
    else
        echo "python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l  --NanoAODv $NanoAODv --use_gateway --log-level INFO"
        python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l  --NanoAODv $NanoAODv --use_gateway --log-level INFO
    fi  
    # python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l  --NanoAODv $NanoAODv --log-level DEBUG
    # python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l  --NanoAODv $NanoAODv --use_gateway --log-level DEBUG --xcache
elif [[ "$mode" == "1" || "$mode" == "all" ]]; then
    echo "Running stage1..."
    # if 2nd arg is available then run with test mode
    if [[ "$2" == "test" ]]; then
        echo "python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv --use_gateway --log-level DEBUG --test_mode"
        python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv --use_gateway --log-level DEBUG --test_mode
    else
        echo "python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv --use_gateway --log-level INFO"
        python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv --use_gateway --log-level INFO
    fi
elif [[ "$mode" == "2" ]]; then
    echo "Running validation step..."
    python validation/zpt_rewgt/validation.py
else
    echo "Error: Invalid mode. Please use '0' for prestage or '1' for stage1 or 2 for making validation plots."
    exit 1
fi
