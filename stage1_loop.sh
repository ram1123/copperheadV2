#!/bin/bash
# Stop execution on any error
set -e

echo "Starting program on " `date`


# Default settings
datasetYAML="configs/datasets/dataset.yaml"
NanoAODv=9

# years=("2018" "2017" "2016postVFP" "2016preVFP")
# years=("2016postVFP" "2016preVFP") 
years=("2018") 

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
# load_path = f"/depot/cms/users/{username}/hmm/copperheadV1clean/{label}/stage1_output/{year}/f1_0/"

mode=$1  # Options: "prestage" or "stage1"

echo "Selected mode: $mode"
echo "Save Path: $save_path"

for year in "${years[@]}"; do
    data_l="${data_l_dict[$year]}"
    echo "Data: $data_l"

    command0="python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l  --NanoAODv $NanoAODv --use_gateway "
    command1="python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv --use_gateway "
    command2="python validation/zpt_rewgt/validation.py -y $year --label $label --in $save_path --data $data_l --background $bkg_l --signal $sig_l "

    if [[ "$2" == "test" ]]; then
        command0+="--log-level DEBUG -frac 0.1"
        command1+="--log-level DEBUG --test_mod"
        command2+="--log-level DEBUG --debug"
    else
        command0+=" --log-level INFO"
        command1+=" --log-level INFO"
        command2+=" --log-level INFO"    
    fi
    
    if [[ "$mode" == "0" ]]; then
        echo "Running pre-stage for year $year..."
        echo "Executing: $command0"  # Print the command for debugging
        eval $command0        
    elif [[ "$mode" == "1" ]]; then
        echo "Running stage1 for year $year..."
        echo "Executing: $command1"  # Print the command for debugging
        eval $command1  
    elif [[ "$mode" == "all" ]]; then
        echo "Running pre-stage for year $year..."
        echo "Executing: $command0"  # Print the command for debugging
        eval $command0 
        
        echo "Running stage1 for year $year..."
        echo "Executing: $command1"  # Print the command for debugging
        eval $command1          
    elif [[ "$mode" == "2" ]]; then
        echo "Running validation step..." 
        echo "Executing: $command2"  # Print the command for debugging
        eval $command2
    else
        echo "Error: Invalid mode. Please use '0' for prestage or '1' for stage1 or 2 for making validation plots."
        exit 1
    fi
done
echo "Ending program on " `date`
