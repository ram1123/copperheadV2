#!/bin/bash
# Stop execution on any error
set -e

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h            Show this help message"
    echo "  -c <file>    Dataset YAML file (default: configs/datasets/dataset.yaml)"
    echo "  -m <mode>    Mode: 0 (prestage), 1 (stage1), all (both), zpt_val (validation), calib (mass calibration) (default: 0)"
    echo "  -v <version> NanoAOD version (default: 9)"
    echo "  -y <year>    Year (default: (\"2018\" \"2017\" \"2016postVFP\" \"2016preVFP\"))"
    echo "  -l <label>   Label (default: Default_nanoAODv9)"
    echo "  -s           Skip bad files (default: 0)"
    echo "  -d           Enable debug mode (default: 0)"
    echo "  -f           Run only 10% of the sampeles for debugging (default: 0)"
    exit 1
}

# Set default values
datasetYAML="configs/datasets/dataset_nanoAODv12.yaml"
NanoAODv="12"
years=("2018" "2017" "2016postVFP" "2016preVFP")
label="Default_nanoAODv9"
debug="0"
mode="all"
skipBadFiles="0"

options=":hc:m:v:y:l:df"
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
        f) frac="1" ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
    esac
done

# function to print and execute the command
function run_command() {
    echo "Executing: $1"
    echo "Executing: $1" >> $log_file
    eval $1
}

# if year is not set then take the default value of years
if [[ -z "$year" ]]; then
    years=("2018" "2017" "2016postVFP" "2016preVFP")
    echo "Reading default year: 2018, 2017 and 2016"
else
    years=($year)
    echo "Reading year: $years"
fi

declare -A data_l_dict # Associative array because of non-integer key.
data_l_dict["2016preVFP"]="B C D E F"
data_l_dict["2016postVFP"]="F G H"
data_l_dict["2017"]="B C D E F"
data_l_dict["2018"]="A B C D"
data_l_dict["2022preEE"]="C D"
data_l_dict["2022postEE"]="E F G"

# data_l_dict["2018"]=""
# data_l_dict["2017"]=""
# data_l_dict["2016postVFP"]=""
# data_l_dict["2016preVFP"]=""
bkg_l="DY Top VV EWK VVV"
# bkg_l="Top VV EWK VVV"
# bkg_l=""
# sig_l="Higgs"
sig_l=""

# If debug is on, then run only for one era in each year.
if [[ "$debug" == "1" ]]; then
    echo "Debug mode is on. Running only for 2018."
    # years=("2016postVFP" "2016preVFP")
    years=("2018")
    # Also update the associated data list.
    data_l_dict["2018"]="C"
    data_l_dict["2017"]="B C D E F"
    data_l_dict["2016preVFP"]="B C D E F"
    data_l_dict["2016postVFP"]="F G H"
    # data_l_dict["2017"]="B C D E F"
    # data_l_dict["2022preEE"]="C D"
    # data_l_dict["2022preEE"]=""
    bkg_l=""
    sig_l=""
fi

chunksize=300000
save_path="/depot/cms/users/$USER/hmm/copperheadV1clean/$label/"


# Check if any log_*.txt file exists then move it to log_old folder
if [ -f log_*.txt ]; then
    if [ ! -d log_old ]; then
        mkdir log_old
    fi
    mv log_*.txt log_old/
fi

# log file with timestamp
log_file="log_$(date +%Y%m%d_%H%M%S).txt"
echo "Starting program on " `date`
echo "Starting program on " `date` > $log_file
echo "Chunk size: $chunksize" >> $log_file
echo "Save Path: $save_path" >> $log_file
echo "Selected mode: $mode" >> $log_file
echo "Selected NanoAOD version: $NanoAODv" >> $log_file
echo "Selected years: ${years[@]}" >> $log_file
echo "Selected years: ${years[@]}"

for year in "${years[@]}"; do
    data_l="${data_l_dict[$year]}"
    echo "Data: $data_l"
    echo "year: $year" >> $log_file
    echo "Data: $data_l" >> $log_file
    echo "Background: $bkg_l" >> $log_file
    echo "Signal: $sig_l" >> $log_file

    # command0="python run_prestage.py --chunksize $chunksize -y $year --yaml $datasetYAML --data $data_l --background $bkg_l --signal $sig_l  --NanoAODv $NanoAODv --xcache "
    command0="python run_prestage.py --chunksize $chunksize -y $year --yaml $datasetYAML --data $data_l --background $bkg_l --signal $sig_l  --NanoAODv $NanoAODv  --use_gateway  "
    command1="python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv --use_gateway  --max_file_len 2500 "
    command2="python validation/zpt_rewgt/validation.py -y $year --label $label --in $save_path --data $data_l --background $bkg_l --signal $sig_l  --use_gateway "

    command3="python src/lib/ebeMassResCalibration/ebeMassResPlotter.py --path $save_path"
    command4="python src/lib/ebeMassResCalibration/calibration_factor.py --path $save_path"

    if [[ "$debug" == "1" ]]; then
        command0+="--log-level DEBUG " #
        command1+="--log-level DEBUG " #
        command2+="--log-level DEBUG --debug"
    else
        command0+=" --log-level INFO"
        command1+=" --log-level INFO"
        command2+=" --log-level INFO"
    fi

    if [[ "$frac" == "1" ]]; then
        command0+=" -frac 0.1"
        command1+=" --test_mode"
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
    elif [[ "$mode" == "zpt_fit" ]]; then
        echo "Running fitting step..."

        # Loop over the number of bins
        # for nbins in 200; do
        # for nbins in 25 50 100 200 500; do
        nbins="CustomBins"
        command2a="python data/zpt_rewgt/fitting/do_fitting.py --run_label $label --year $year --input_path $save_path --debug --outAppend _March17"
        command2b="python data/zpt_rewgt/fitting/do_f_test.py --run_label $label --year $year --input_path $save_path --debug --nbins $nbins --outAppend _March17"
        command2c="python data/zpt_rewgt/fitting/get_goodnessOfFit.py --run_label $label --year $year --input_path $save_path --debug --nbins $nbins --outAppend _March17"

        # command2b_New="python data/zpt_rewgt/fitting/get_ZpT_Weights.py --run_label $label --year $year --input_path $save_path --debug --nbins $nbins --outAppend _bin$nbins"

        echo "Executing: $command2a"
        echo "command2a: $command2a" >> $log_file
        # eval $command2a

        echo "Executing: $command2b"
        echo "command2b: $command2b" >> $log_file
        # eval $command2b

        echo "Executing: $command2b_New"
        echo "command2b_New: $command2b_New" >> $log_file
        # eval $command2b_New

        echo "Executing: $command2c"
        echo "command2c: $command2c" >> $log_file
        eval $command2c
        # done

    elif [[ "$mode" == "zpt_val" ]]; then
        echo "Running validation step..."
        echo "Executing: $command2"  # Print the command for debugging
        echo "command2: $command2" >> $log_file
        eval $command2
    # Run the mass calibration fitting step
    elif [[ "$mode" == "calib" ]]; then
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
