#!/bin/bash
# Stop execution on any error
set -e

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h            Show this help message"
    echo "  -c <file>    Dataset YAML file (default: configs/datasets/dataset.yaml)"
    echo "  -m <mode>    Mode: 0 (prestage), 1 (stage1), all (both), or val (validation)"
    echo "  -v <version> NanoAOD version (default: 9)"
    echo "  -y <year>    Year (default: 2018)"
    echo "  -l <label>   Label (default: Default_nanoAODv9)"
    echo "  -s           Skip bad files (default: 0)"
    echo "  -d           Enable debug mode (default: 0)"
    exit 1
}

# Set default values
datasetYAML="configs/datasets/dataset.yaml"
NanoAODv="9"
year="2018"
label="Default_nanoAODv9"
debug="0"
mode="0"
skipBadFiles="0"

options=":hc:m:v:y:l:d"
while getopts $options option; do
    case "$option" in
        h) usage ;;
        c) datasetYAML=$OPTARG ;;
        m) mode=$OPTARG ;;
        v) NanoAODv=$OPTARG ;;
        y) year=$OPTARG ;;
        l) label=$OPTARG ;;
        s) skipBadFiles="1" ;;
        d) debug="1" ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
    esac
done


declare -A data_l_dict # Associative array because of non-integer key.
data_l_dict["2016preVFP"]="B C D E F"
data_l_dict["2016postVFP"]="F G H"
data_l_dict["2017"]="B C D E F"
data_l_dict["2018"]="A B C D"

bkg_l="DY Top VV"
sig_l=""

years=("2018" "2017" "2016postVFP" "2016preVFP")

# If debug is on, then run only for one era in each year.
if [[ "$debug" == "1" ]]; then
    echo "Debug mode is on. Running only for 2018."
    years=("2018")
    # Also update the associated data list.
    data_l_dict["2018"]="A"
    bkg_l="DY"
fi

chunksize=300000
save_path="/depot/cms/users/$USER/hmm/copperheadV1clean/$label/"

# log file with timestamp
log_file="log_$(date +%Y%m%d_%H%M%S).txt"
echo "Starting program on " `date`
echo "Starting program on " `date` > $log_file
echo "Chunk size: $chunksize" >> $log_file
echo "Save Path: $save_path" >> $log_file
echo "Selected mode: $mode" >> $log_file

for year in "${years[@]}"; do
    data_l="${data_l_dict[$year]}"
    echo "Data: $data_l"
    echo "year: $year" >> $log_file
    echo "Data: $data_l" >> $log_file
    echo "Background: $bkg_l" >> $log_file
    echo "Signal: $sig_l" >> $log_file

    command0="python run_prestage.py --chunksize $chunksize -y $year --yaml $datasetYAML --data $data_l --background $bkg_l --signal $sig_l  --NanoAODv $NanoAODv --use_gateway "
    command1="python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv --use_gateway "
    command2="python validation/zpt_rewgt/validation.py -y $year --label $label --in $save_path --data $data_l --background $bkg_l --signal $sig_l  --use_gateway "
    command3="python src/lib/ebeMassResCalibration/ebeMassResPlotter.py --path $save_path"

    if [[ "$debug" == "1" ]]; then
        command0+="--log-level DEBUG " # -frac 0.1
        command1+="--log-level DEBUG --test_mod"
        command2+="--log-level DEBUG --debug"
    else
        command0+=" --log-level INFO"
        command1+=" --log-level INFO"
        command2+=" --log-level INFO"
    fi

    if [[ "$skipBadFiles" == "1" ]]; then
        command0+=" --skipBadFiles"
    fi

    if [[ "$mode" == "0" ]]; then
        echo "Running pre-stage for year $year..."
        echo "Executing: $command0"  # Print the command for debugging
        echo "command0: $command0" >> $log_file
        eval $command0
    elif [[ "$mode" == "1" ]]; then
        echo "Running stage1 for year $year..."
        echo "Executing: $command1"  # Print the command for debugging
        echo "command1: $command1" >> $log_file
        eval $command1
    elif [[ "$mode" == "all" ]]; then
        echo "Running pre-stage for year $year..."
        echo "Executing: $command0"  # Print the command for debugging
        echo "command0: $command0" >> $log_file
        eval $command0

        echo "Running stage1 for year $year..."
        echo "Executing: $command1"  # Print the command for debugging
        echo "command1: $command1" >> $log_file
        eval $command1
    elif [[ "$mode" == "val" ]]; then
        echo "Running validation step..."
        echo "Executing: $command2"  # Print the command for debugging
        echo "command2: $command2" >> $log_file
        eval $command2
    # Run the mass calibration fitting step
    elif [[ "$mode" == "fit" ]]; then
        echo "Running mass calibration"
        echo "Executing: $command3"  # Print the command for debugging
        echo "command: $command3" >> $log_file
        eval $command3
    else
        echo "Error: Invalid mode. Please use '0' for prestage or '1' for stage1 or 2 for making validation plots."
        exit 1
    fi
done
echo "Ending program on " `date`

echo "Ending program on " `date` >> $log_file
