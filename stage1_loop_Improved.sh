#!/bin/bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  -h            Show this help message
  -c <file>     Dataset YAML file (default: configs/datasets/dataset_nanoAODv12.yaml)
  -m <mode>     Mode: 0 (prestage), 1 (stage1), 2 (stage2), 3 (stage3), all,
                zpt_fit|zpt_fit0|zpt_fit1|zpt_fit2|zpt_fit12, zpt_val, calib,
                compact, dnn|dnn_pre|dnn_train|dnn_var_rank (default: all)
  -v <version>  NanoAOD version (default: 12)
  -y <year>     Year (default: 2018 2017 2016postVFP 2016preVFP)
  -l <label>    Label (default: Default_nanoAODv9)
  -n <njet>     nJet value (optional, default: 0)
  -b <bins>     Number of bins (optional, default: 100)
  -o <outAppend>  String to append to output files (default: today's date)
  -r <region>   DNN training region (default: h-peak)
  -t <category> DNN training category (default: vbf)
  -p <postfix>  Postfix string to append to output directory for stage2 and 3 (default: "")
  -s            Skip bad files (default: 0)
  -d            Enable debug mode (0/1/2; default: 0)
  -f            Run only 10% of samples for debugging (default: 0)
  -k            Enable Dask/Gateway (default: 0)
EOF
    exit 1
}

# ---------- Default values ----------
datasetYAML="configs/datasets/dataset_nanoAODv12.yaml"
NanoAODv="12"
declare -a years=("2018PR" "2018" "2017" "2016postVFP" "2016preVFP" "2016" "run2")
label="Default_nanoAODv9"
debug="0"
mode="all"
skipBadFiles="0"
frac="0"
njet="0"
nbin="100"
PWD="$(pwd)"
outAppend="$(date +%b%d_%Y)"   # Default: today's date, e.g. Jun24_2025
region="h-peak" # h-peak, h-sideband, signal
category="vbf"
postfix=""
dask="0"

# ----------- Parse options -----------
while getopts ":hc:m:v:y:l:n:b:d:o:r:t:p:sfk" option; do
    case "$option" in
        h) usage ;;
        c) datasetYAML="$OPTARG" ;;
        m) mode="$OPTARG" ;;
        v) NanoAODv="$OPTARG" ;;
        y) IFS=', ' read -r -a years <<< "$OPTARG" ;;
        l) label="$OPTARG" ;;
        n) njet="$OPTARG" ;;
        b) nbin="$OPTARG" ;;
        d) debug="$OPTARG" ;;
        o) outAppend="$OPTARG" ;;
        r) region="$OPTARG" ;;
        t) category="$OPTARG" ;;
        p) postfix="$OPTARG" ;;
        s) skipBadFiles="1" ;;
        f) frac="1" ;;
        k) dask="1" ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
    esac
done

# ----------- Check environment and load modules -----------
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "No conda environment detected. Activate the appropriate env and retry."
    exit 1
fi

# if DNN training is enabled, check if the conda environment is `pfn_env` else it should be `yun_coffea_latest`
# if [[ "$mode" == "dnn" || "$mode" == "dnn_pre" || "$mode" == "dnn_train" || "$mode" == "dnn_var_rank" ]]; then
if [[ "$mode" == "dnn" || "$mode" == "dnn_train" || "$mode" == "dnn_var_rank" ]]; then
    if [[ "$CONDA_PREFIX" != *"pfn_env"* ]]; then
        echo "Please run this script in the pfn_env conda environment for DNN training"
        exit 1
    fi
elif [[ "$mode" == "zpt_fit" || "$mode" == "zpt_fit0" || "$mode" == "zpt_fit1" || "$mode" == "zpt_fit2" || "$mode" == "zpt_fit12" ]]; then
    if [[ "$CONDA_PREFIX" != *"coffea_latest"* ]]; then
        echo "Please run this script in the coffea_latest conda environment for ZpT fitting"
        exit 1
    fi
else
    if [[ "$CONDA_PREFIX" != *"yun_coffea_latest"* ]]; then
        echo "Please run this script in the yun_coffea_latest conda environment"
        exit 1
    fi
fi

# ----------- Utility functions -----------
log_dir="log_old"
[ -d "$log_dir" ] || mkdir -p "$log_dir"

# Move any previous log_*.txt to log_old/
shopt -s nullglob
for f in log_*.txt; do
    mv "$f" "$log_dir/"
done
shopt -u nullglob

log_file="log_$(date +%Y%m%d_%H%M%S).txt"
exec 3>>"$log_file"  # FD 3 for logging

log() { echo "$@" | tee -a "$log_file"; }

# save_path="/depot/cms/users/$USER/hmm/copperheadV1clean/$label/"
save_path="/depot/cms/hmm/$USER/hmm_ntuples/copperheadV1clean/$label/"
# save_path="/store/user/rasharma/hmm/copperheadV1clean/$label/"

trap 'log "Program FAILED on $(date)"; exec 3>&- ' ERR

declare -A data_l_dict=(
    [2018PR]="A"
    [2016preVFP]="B C D E F"
    [2016postVFP]="F G H"
    [2016]="B C D E F G H"
    [2017]="B C D E F"
    [2018]="A B C D"
    [2022preEE]="C D"
    [2022postEE]="E F G"
    [run2]="A B C D E F G H"
)

# bkg_l="DY TT ST VV EWK VVV"
bkg_l="DY Top VV EWK VVV"
# bkg_l=""

# sig_l="VBF"
sig_l="Higgs"
# sig_l=""

if [[ "$debug" -ge 1 ]]; then
    log "Debug mode ON "
    # years=("2016preVFP")
    data_l_dict["2016preVFP"]=""
    data_l_dict["2016postVFP"]=""
    data_l_dict["2017"]=""
    data_l_dict["2018"]=""

    bkg_l="DY Top VV EWK VVV"
    # bkg_l="DY"
    # bkg_l="Top"

    sig_l="Higgs"
    # sig_l=""
fi

chunksize=300000
max_file_len=2500 # 2500 for data, 5 for MC

echo "Running with the following parameters:"
echo "  Dataset YAML: $datasetYAML"
echo "  NanoAOD version: $NanoAODv"
echo "  Years: ${years[@]}"
echo "  Label: $label"
echo "  Save path: $save_path"
echo "  Debug mode: $debug"
echo "  Mode: $mode"
echo "  Skip bad files: $skipBadFiles"
echo "  Fraction: $frac"
echo "  nJet: $njet"
echo "  Number of bins: $nbin"
echo "  Output append: $outAppend"
echo "  Region: $region"
echo "  Category: $category"

# ----------- Main loop -----------
for year in "${years[@]}"; do
    data_l="${data_l_dict[$year]}"
    log "Processing year: $year"
    log "  Data: $data_l"
    log "  Background: $bkg_l"
    log "  Signal: $sig_l"
    log "  NanoAODv: $NanoAODv"
    log "  Save path: $save_path"

    # ---- Command templates ----
    # command0="python run_prestage.py --chunksize $chunksize -y $year --yaml $datasetYAML --data $data_l --background $bkg_l --signal $sig_l  --NanoAODv $NanoAODv "
    command0="python run_prestage.py --chunksize $chunksize -y $year --yaml $datasetYAML --data $data_l --background $bkg_l --signal $sig_l  --NanoAODv $NanoAODv --xcache  "

    # INFO: If running with JES variation use the max file length = 350, else 2500
    # command1="python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv --max_file_len $max_file_len --isCutflow --rerun"
    command1="python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv  --max_file_len $max_file_len --rerun  --skipSamples "
    # command1="python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv  --max_file_len $max_file_len "

    ### DNN training parameters
    training_fold=3
    model_path="${PWD}/dnn/trained_models"
    # model_label="${label}"
    model_label="Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt" # THis name was hardcoded for older runs.

    # NOTE: This DNN is trained with all year but name contains hardcoded string "2018"
    # model_label_forCompact="2018_${region}_${category}_2018_UpdatedQGL_17July_Test" # August training
    model_label_forCompact="run2_${region}_${category}_ScanHyperParamV1" # Latest training; 03 Sep 2025
    # model_label_forCompact="run2_h-peak_vbf_BestHPButSmallHidden_128_64_32_maxAUC" # 10 Sep 2025: Same as training on 03 Sep 2025, except with old hidden layers
    # model_label_forCompact="run2_h-peak_vbf_BestHPOld_NewSoftJetVarV0" # 12 Sep 2025 training: Trained with same architecture as 03 Sep 2025, Just added new soft jet variables

    # compact_tag="03September"
    compact_tag="19September"

    # command_compact="python scripts/compact_parquet_data.py -y $year -l $save_path -m $model_path/$model_label/$model_label_forCompact --add_dnn_score  --fix_dimuon_mass --tag $compact_tag  "
    command_compact="python scripts/compact_parquet_data.py -y $year -l $save_path  "

    # rename "Top" to "TT ST" in the $bkg_l for stage2
    # FIXME: This is a temporary fix, will try to sync the naming convention in the stage2 python script.
    bkg_l_stage2="$bkg_l"
    if [[ "$bkg_l_stage2" == *"Top"* ]]; then
        bkg_l_stage2="${bkg_l_stage2/Top/TT ST}"
    fi
    # use option "--no_variations" with stage2 if you want to run with only nominal weights
    command2="python run_stage2_vbf.py --model_path $model_path/$model_label/$model_label_forCompact --model_label $model_label   --base_path $save_path -y $year -data $data_l -bkg $bkg_l_stage2 -sig $sig_l --save_postfix $postfix  "
    # command2="python run_stage2_vbf.py --model_path $model_path/$model_label/$model_label_forCompact --model_label $model_label   --base_path $save_path -y $year -data $data_l -bkg $bkg_l_stage2 -sig $sig_l --save_postfix $postfix --no_variations "

    # command3="python run_stage3_vbf.py --base_path $save_path -y $year  --save_postfix $postfix --out_postfix ${postfix}_aMCatNLO "
    command3="python run_stage3_vbf.py --base_path $save_path -y $year  --save_postfix $postfix --out_postfix ${postfix}_MiNNLO "
    # command3="python run_stage3_vbf.py --base_path $save_path -y $year  --save_postfix $postfix --out_postfix ${postfix}_DY012 "
    # command3="python run_stage3_vbf.py --base_path $save_path -y $year  --save_postfix $postfix --out_postfix ${postfix}_MiNNLOSplitMjj "
    # command3="python run_stage3_vbf.py --base_path $save_path -y $year  --save_postfix $postfix --out_postfix ${postfix}_MiNNLO_NoDYVBF "
    # command3="python run_stage3_vbf.py --base_path $save_path -y $year  --save_postfix $postfix --out_postfix ${postfix}_aMCatNLO_NoDYVBF "

    command4="python validation/zpt_rewgt/validation.py -y $year --label $label --in $save_path --data $data_l --background $bkg_l --signal $sig_l   "
    command5="python src/lib/ebeMassResCalibration/ebeMassResPlotter.py --path $save_path"
    command6="python src/lib/ebeMassResCalibration/calibration_factor.py --path $save_path"

    # Logging/debug options
    if [[ "$debug" -ge 2 ]]; then
        command0+=" --log-level DEBUG "
        command1+=" --log-level DEBUG "
        # command3+=" --log-level DEBUG "
        command4+=" --log-level DEBUG --debug "
    else
        command0+=" --log-level INFO "
        command1+=" --log-level INFO "
        command4+=" --log-level INFO "
    fi

    if [[ "$frac" == "1" ]]; then
        command0+=" -frac 0.1"
        command1+=" --test_mode"
    fi
    [[ "$skipBadFiles" == "1" ]] && command0+=" --skipBadFiles"

    if [[ "$dask" == "1" ]]; then
        command0+=" --use_gateway "
        command1+=" --use_gateway "
        command2+=" --use_gateway "
        command4+=" --use_gateway "
        command5+=" --use_gateway "
        command6+=" --use_gateway "
        command_compact+=" --use_gateway "
    fi

    # ---- Mode switch ----
    case "$mode" in
        0)
            log "Running pre-stage for year $year..."
            log "Command: $command0"
            eval "$command0"
            ;;
        1)
            log "Running stage1 for year $year..."
            log "Command: $command1"
            eval "$command1"
            ;;
        2)
            log "Running stage2 for year $year..."
            log "Command: $command2"
            eval "$command2"
            ;;
        2p)
            log "Running the validation of stage2 (i.e. data/mc plot for dnn score) for year $year..."
            region2p="h-sidebands"
            command2p1="python plotter/plot_DNN_score.py -label $label -cat $category -y ${year} --region ${region2p}"
            log "Command: $command2p1"
            eval "$command2p1"

            region2p="h-peak"
            command2p2="python plotter/plot_DNN_score.py -label $label -cat $category -y ${year} --region ${region2p}"
            log "Command: $command2p2"
            eval "$command2p2"
            ;;
        3)
            log "Running stage3 for year $year..."
            log "Command: $command3"
            eval "$command3"
            ;;
        all)
            log "Running pre-stage for year $year..."
            log "Command: $command0"
            eval "$command0"
            log "Running stage1 for year $year..."
            log "Command: $command1"
            eval "$command1"
            ;;
        zpt_fit|zpt_fit0|zpt_fit1|zpt_fit2|zpt_fit12)
            log "Running ZpT fitting step(s)..."
            dy_sample="aMCatNLO" # FIXME: Hardcoded DY sample name: aMCatNLO or MiNNLO
            cmd0="python data/zpt_rewgt/fitting/save_SF_rootFiles.py -l $label -y $year -dy_sample $dy_sample "
            cmd1="python data/zpt_rewgt/fitting/do_f_test.py               -l $label -y $year --dy_sample $dy_sample --nbins $nbin --njet $njet --outAppend $outAppend --debug"
            cmd2="python data/zpt_rewgt/fitting/get_polyFit.py             -l $label -y $year --dy_sample $dy_sample --nbins $nbin --njet $njet --outAppend $outAppend"
            [[ "$mode" =~ ^(zpt_fit0|zpt_fit)$ ]] && { log "Command0: $cmd0"; eval "$cmd0"; }
            [[ "$mode" =~ ^(zpt_fit1|zpt_fit|zpt_fit12)$ ]] && { log "Command1: $cmd1"; eval "$cmd1"; }
            [[ "$mode" =~ ^(zpt_fit2|zpt_fit|zpt_fit12)$ ]] && { log "Command2: $cmd2"; eval "$cmd2"; }
            ;;
        zpt_val)
            log "Running ZpT validation..."
            log "Command: $command4"
            eval "$command4"
            ;;
        calib)
            log "Running mass calibration..."
            log "Command: $command5"
            eval "$command5"
            ;;
        compact)
            log "Compacting parquet data for year $year..."
            log "Command: $command_compact"
            eval "$command_compact"
            ;;
        dnn|dnn_pre|dnn_train|dnn_var_rank)
            log "Running DNN step(s) for year $year..."
            cmd_preproc="python MVA_training/VBF/dnn_preprocessor.py --label $label --region $region --category $category --year $year --log-level INFO "
            # Alternative cmd_train configurations (uncomment and adjust as needed):
            # -- Bayesian Optimization:
            # cmd_train="python MVA_training/VBF/dnn_train.py --label $label --region $region --category $category --year $year --bo --bo-trials 75 --bo-epochs 100 --bo-fold 0 --n-epochs 100 --batch-size 15536 --log-level INFO "
            # cmd_train="python MVA_training/VBF/dnn_train.py --label $label --region $region --category $category --year $year --bo --bo-trials 21 --bo-epochs 100 --bo-fold 0 --n-epochs 100 --batch-size 15536 --log-level INFO "
            # -- Quick test:
            # cmd_train="python MVA_training/VBF/dnn_train.py --label $label --region $region --category $category --year $year --bo --bo-trials 3 --bo-epochs 5 --bo-fold 0 --n-epochs 5 --batch-size 15536 --log-level INFO "
            # cmd_train="python MVA_training/VBF/dnn_train.py --label $label --region $region --category $category --year $year --n-epochs 5 --batch-size 15536 --log-level INFO "
            # Active configuration:
            cmd_train="python MVA_training/VBF/dnn_train.py --label $label --region $region --category $category --year $year --n-epochs 100 --log-level INFO "
            cmd_var_rank="python MVA_training/VBF/variable_ranking.py "

            if [[ "$mode" == "dnn_pre" || "$mode" == "dnn" ]]; then
                if [[ "$dask" == "1" ]]; then
                    cmd_preproc+=" --use_gateway "
                fi
                log "Running DNN preprocessor..."
                log "Command: $cmd_preproc"
                eval "$cmd_preproc"
            fi

            if [[ "$mode" == "dnn_train" || "$mode" == "dnn" ]]; then
                if [[ "$debug" == "1" ]]; then
                    cmd_train+=" --debug "
                fi
                log "Running DNN training..."
                log "Command: $cmd_train"
                eval "$cmd_train"
            fi

            if [[ "$mode" == "dnn_var_rank" ]]; then
                log "Running variable ranking..."
                log "Command: $cmd_var_rank"
                eval "$cmd_var_rank"
            fi
            ;;
        *)
            echo "Error: Invalid mode. See -h for the full list of supported modes."
            usage
            ;;
    esac
done

log "Program ended on $(date)"
exec 3>&-
