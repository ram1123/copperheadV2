#!/bin/bash
# Stop execution on any error
set -e

echo "Starting program on " `date`


# Default settings
datasetYAML="configs/datasets/dataset.yaml"
NanoAODv=9

# years=("2018" "2017" "2016postVFP" "2016preVFP")
years=("2016postVFP" "2016preVFP") 

declare -A data_l_dict # Associative array because of non-integer key.
data_l_dict["2016preVFP"]="B C D E F"
data_l_dict["2016postVFP"]="F G H"
data_l_dict["2017"]="B C D E F"
data_l_dict["2018"]="A B C D"

bkg_l="DY Top VV"
# bkg_l=""
sig_l=""

chunksize=300000
label="WithPurdueZptWgt_DYWithoutLHECut_16Feb_AllYear"
save_path="/depot/cms/users/$USER/hmm/copperheadV1clean/$label/"

mode=$1  # Options: "prestage" or "stage1"

echo "Selected mode: $mode"
echo "Save Path: $save_path"

for year in "${years[@]}"; do
    data_l="${data_l_dict[$year]}"
    echo "Data: $data_l"

    if [[ "$mode" == "0" ]]; then
        echo "Running pre-stage for year $year..."
        if [[ "$2" == "test" ]]; then
            echo "python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l  --NanoAODv $NanoAODv --use_gateway --log-level DEBUG -frac 0.1"
            python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l  --NanoAODv $NanoAODv --use_gateway --log-level DEBUG -frac 0.1
        else
            echo "python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l  --NanoAODv $NanoAODv --use_gateway --log-level INFO"
            python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l  --NanoAODv $NanoAODv --use_gateway --log-level INFO
        fi
    elif [[ "$mode" == "1" ]]; then
        echo "Running stage1 for year $year..."
        if [[ "$2" == "test" ]]; then
            echo "python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv --use_gateway --log-level DEBUG --test_mode"
            python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv --use_gateway --log-level DEBUG --test_mode
        else
            echo "python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv --use_gateway --log-level INFO"
            python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv --use_gateway --log-level INFO
        fi
    elif [[ "$mode" == "all" ]]; then
        python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l  --NanoAODv $NanoAODv --use_gateway --log-level INFO
        python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv --use_gateway --log-level INFO
    elif [[ "$mode" == "2" ]]; then
        echo "Running validation step..." # FIXME: Hardcoded year and other value in the validation.py.
        python validation/zpt_rewgt/validation.py
    else
        echo "Error: Invalid mode. Please use '0' for prestage or '1' for stage1 or 2 for making validation plots."
        exit 1
    fi
done
echo "Ending program on " `date`
