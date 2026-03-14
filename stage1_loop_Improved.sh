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
  -o <save_postfix>  String to append to output files (default: today's date)
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
declare -a years=("2018PR" "2018" "2017" "2016postVFP" "2016preVFP" "2016" "run2" "run3")
label="Default_nanoAODv9"
debug="0"
mode="all"
skipBadFiles="0"
frac="0"
njet="0"
nbin="100"
PWD="$(pwd)"
save_postfix="$(date +%b%d_%Y)"   # Default: today's date, e.g. Jun24_2025
region="h-peak" # h-peak, h-sideband, signal
category="vbf"
postfix=""
dask="0"
cluster_index="0"
isMC="0"
isSync="0"

# ----------- Default save paths -----------
save_path_depot="/depot/cms/hmm/$USER/hmm_ntuples/copperheadV1clean/$label/"
save_path_work="/work/projects/hmm/$USER/hmm_ntuples/copperheadV1clean/$label/"
save_path_local="/depot/cms/users/$USER/hmm/copperheadV1clean/$label/"
save_path_eos="/store/user/rasharma/hmm/copperheadV1clean/$label/"

save_path="$save_path_depot"   # default

# ----------- Parse options -----------
while getopts ":hc:m:v:y:l:n:b:d:o:r:t:p:i:M:ksfS:z" option; do
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
        o) save_postfix="$OPTARG" ;;
        r) region="$OPTARG" ;;
        t) category="$OPTARG" ;;
        p) postfix="$OPTARG" ;;
        i) cluster_index="$OPTARG" ;;
        M) isMC="$OPTARG" ;;
        k) dask="1" ;;
        s) skipBadFiles="1" ;;
        f) frac="1" ;;
        S) save_path="$OPTARG" ;;
        z) isSync="1" ;;        
        \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
    esac
done

# ----------- Check environment and load modules -----------
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "No conda environment detected. Activate the appropriate env and retry."
    exit 1
fi

# if DNN training is enabled, check if the conda environment is `pfn_env` else it should be `coffea_latest`
# if [[ "$mode" == "dnn" || "$mode" == "dnn_pre" || "$mode" == "dnn_train" || "$mode" == "dnn_var_rank" ]]; then
# if [[ "$mode" == "dnn" || "$mode" == "dnn_train" || "$mode" == "dnn_var_rank" ]]; then
#     if [[ "$CONDA_PREFIX" != *"pfn_env"* ]]; then
#         echo "Please run this script in the pfn_env conda environment for DNN training"
#         exit 1
#     fi
# else
#     if [[ "$CONDA_PREFIX" != *"coffea_latest"* ]]; then
#         echo "Please run this script in the coffea_latest conda environment"
#         exit 1
#     fi
# fi

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

trap 'log "Program FAILED on $(date)"; exec 3>&- ' ERR
log "Program started on $(date)"

declare -A data_l_dict=(
    [2018PR]="A"
    [2016preVFP]="B C D E F"
    [2016postVFP]="F G H"
    [2016]="B C D E F G H"
    [2017]="B C D E F"
    [2018]="A B C D"
    [2022preEE]="C D"
    [2022postEE]="E F G"
    [2023]="C"
    [2023BPix]="D"
    [2024]="C D E F G H I"
    [run2]="A B C D E F G H"
    [run3]="C D E F G H I"
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
    # data_l_dict["2022preEE"]=""
    data_l_dict["2022postEE"]=""
    # data_l_dict["2023"]=""
    # data_l_dict["2023BPix"]=""
    # data_l_dict["2024"]=""

    bkg_l=""
    # bkg_l="DY Top VV EWK VVV"
    # bkg_l=""

    sig_l="Higgs"
    # sig_l=""
fi

chunksize=600000
max_file_len=900 # 2500 for data, 5 for MC

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
echo "  Output append: $save_postfix"
echo "  Region: $region"
echo "  Category: $category"
echo "  isMC: $isMC"


# ----------- Main loop -----------
for year in "${years[@]}"; do
    data_l="${data_l_dict[$year]}"
    log "Processing year: $year"
    log "  Data: $data_l"
    log "  Background: $bkg_l"
    log "  Signal: $sig_l"
    log "  NanoAODv: $NanoAODv"
    log "  Save path: $save_path"

    # ########## PRE-STAGE command ##########
    command0="python run_prestage.py --chunksize $chunksize -y $year --yaml $datasetYAML --data $data_l --background $bkg_l --signal $sig_l  --NanoAODv $NanoAODv  "

    # ########## STAGE-1 command ##########
    # INFO: If running with JES variation use the max file length = 350, else 2500
    # command1="python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv --max_file_len $max_file_len --yaml $datasetYAML  --isCutflow "
    # command1="python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv --max_file_len $max_file_len --yaml $datasetYAML  --isCutflow --rerun "
    command1="python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv  --max_file_len $max_file_len --yaml $datasetYAML  --skipSamples "
    # command1="python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv  --max_file_len $max_file_len --yaml $datasetYAML  "

    if [[ "${isSync}" == "1" ]]; then
        command0+=" --sync "
        command1+=" --sync --isCutflow "
    fi

    ### DNN training parameters
    training_fold=4
    model_label="${label}"
    model_trained_path="./dnn/trained_models/Run3_nanoAODv12_02Feb_FilterJetsHorn30GeV/2022preEE-2022postEE-2023-2023BPix-2024_h-peak_vbf"
    training_tag="trained_best_optuna_03trail_v3"

    # ########## Compact command ##########
    # command_compact="python scripts/compact_parquet_data.py -y $year --input_path $save_path -m $model_trained_path/$training_tag --add_dnn_score  --fix_dimuon_mass --tag $save_postfix  "
    command_compact="python scripts/compact_parquet_data.py -y $year --input_path $save_path  "

    # rename "Top" to "TT ST" in the $bkg_l for stage2
    # FIXME: This is a temporary fix, will try to sync the naming convention in the stage2 python script.
    bkg_l_stage2="$bkg_l"
    if [[ "$bkg_l_stage2" == *"Top"* ]]; then
        bkg_l_stage2="${bkg_l_stage2/Top/TT ST}"
    fi

    # ########## STAGE-2 command ##########
    # use option "--no_variations" with stage2 if you want to run with only nominal weights
    command2="python run_stage2_vbf.py -y $year -input $save_path -l $label --model_tag $training_tag --model_path $model_trained_path -data $data_l -bkg $bkg_l_stage2 -sig $sig_l --save_postfix ${save_postfix} --no_variations "
    # command2="python run_stage2_vbf.py -y $year -input $save_path -l $label --model_tag $training_tag --model_path $model_trained_path -data $data_l -bkg $bkg_l_stage2 -sig $sig_l --save_postfix ${save_postfix}  "

    # ########## STAGE-3 command ##########
    command3="python run_stage3_vbf.py --years $year -input $save_path -l $label  --save_postfix ${save_postfix} --no_variations "
    # command3="python run_stage3_vbf.py --years $year -input $save_path -l $label  --save_postfix ${save_postfix} "

    # ########## Z-PT VALIDATION command ##########
    command4="python validation/zpt_rewgt/validation.py -y $year --label $label --in $save_path --data $data_l --background $bkg_l --signal $sig_l   "

    # ########## Calibration commands ##########
    if [[ "$isMC" == "1" ]]; then
        mcArg="--isMC"
    else
        mcArg=" "
    fi

    echo "mcArc : $mcArg"

    command5="python src/lib/ebeMassResCalibration/getCalibrationFactor.py --NanoAODv $NanoAODv --years $year --extraString $postfix --ifbinned --steps all $mcArg  "
    command6="python src/lib/ebeMassResCalibration/getCalibrationFactor.py --NanoAODv $NanoAODv --years $year --extraString $postfix --ifbinned --closure_test $mcArg "

    # category="30-45_OB"
    # command5="python src/lib/ebeMassResCalibration/getCalibrationFactor.py --NanoAODv $NanoAODv --years $year --extraString $postfix --ifbinned $mcArg --steps step1   --fixCat ${category} "

    # category=10
    # command5="python src/lib/ebeMassResCalibration/getCalibrationFactor.py --NanoAODv $NanoAODv --years $year --extraString $postfix --ifbinned $mcArg --closure_test  --fixCat ${category} --no-dask-client "
    # ########## Calibration commands ##########

    # Logging/debug options
    if [[ "$debug" -ge 2 ]]; then
        command0+=" --log-level DEBUG "
        command1+=" --log-level DEBUG "
        command2+=" --log-level DEBUG "
        command3+=" --log-level DEBUG "
        command4+=" --log-level DEBUG --debug "
        command5+=" --log-level DEBUG "
        command_compact+=" --log-level DEBUG "
    else
        command0+=" --log-level INFO "
        command1+=" --log-level INFO "
        command2+=" --log-level INFO "
        command3+=" --log-level INFO "
        command4+=" --log-level INFO "
        command5+=" --log-level INFO "
        command_compact+=" --log-level INFO "
    fi

    if [[ "$frac" == "1" ]]; then
        command0+=" -frac 0.1"
        command1+=" --test_mode"
    fi
    [[ "$skipBadFiles" == "1" ]] && command0+=" --skipBadFiles"

    if [[ "$dask" -ge 1 ]]; then
        command0+=" --use_gateway "
        command1+=" --use_gateway "
        command2+=" --use_gateway "
        command4+=" --use_gateway "
        command5+=" --use_gateway "
        command6+=" --use_gateway "
        command_compact+=" --use_gateway "
    fi
    if [[ "$cluster_index" != "0" ]]; then
        command0+=" --cluster_index $cluster_index "
        command1+=" --cluster_index $cluster_index "
        command2+=" --cluster_index $cluster_index "
        command3+=" --cluster_index $cluster_index "
        command4+=" --cluster_index $cluster_index "
        command5+=" --cluster_index $cluster_index "
        command6+=" --cluster_index $cluster_index "
        command_compact+=" --cluster_index $cluster_index "
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
        4)
            log "Running stage4 (datacard preparation) for year $year..."
            # ########## STAGE-4 command ##########
            SRC_DIR="$save_path/stage3_datacards_${save_postfix}"
            DEST_DIR="/depot/cms/private/users/shar1172/CombineSetup/CMSSW_14_1_0_pre4/src/HiggsAnalysis/CombinedLimit/HMuMu_StatisticalAnalysis/run3_prelims/"
            echo "Copying datacards from $SRC_DIR to $DEST_DIR/stage3_datacards_${save_postfix}/"
            mkdir -p "${DEST_DIR}"
            rsync -av --delete "${SRC_DIR}" "${DEST_DIR}/${label}/"
            # -a : preserve permissions, timestamps, symbolic links,
            # -v : verbose output
            # --delete : delete files in the destination that are not in the source
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
            dy_sample="INCamcatnloFXFX" # FIXME: Hardcoded DY sample name: aMCatNLO or MiNNLO or amcatnloFXFX or powheg or INCamcatnloFXFX
            cmd0="python src/copperhead/zpt_rewgt/derive/save_SF_rootFiles.py -l $label -y $year --input_path $save_path -dy_sample $dy_sample --use_gateway --cluster_index $cluster_index"
            cmd1="python src/copperhead/zpt_rewgt/derive/do_f_test.py               -l $label -y $year --dy_sample $dy_sample --nbins $nbin --njet $njet --save_postfix $save_postfix --debug"
            cmd2="python src/copperhead/zpt_rewgt/derive/get_polyFit.py             -l $label -y $year --dy_sample $dy_sample  --njet $njet --save_postfix $save_postfix"
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
        calib_closure)
            log "Running mass calibration..."
            log "Command: $command6"
            eval "$command6"
            ;;
        compact)
            log "Compacting parquet data for year $year..."
            log "Command: $command_compact"
            eval "$command_compact"
            ;;
        dnn|dnn_pre|dnn_train|dnn_var_rank)
            log "Running DNN step(s) for year $year..."
            # cmd_preproc="python MVA_training/VBF/run3_model/dnn_preprocessor.py --label $label --region $region --category $category --year $year --log-level DEBUG "
            cmd_preproc="python MVA_training/VBF_new/preprocess_dnn.py --label $label --region $region --category $category --year $year --log-level DEBUG "
            # Alternative cmd_train configurations (uncomment and adjust as needed):

            # -- Bayesian Optimization:
            # cmd_train="python MVA_training/VBF/run3_model/dnn_train.py --label $label --region $region --category $category --year $year --bo --bo-trials 55 --bo-epochs 100 --bo-fold 0 --n-epochs 100 --batch-size 15536 --log-level INFO "
            # cmd_train="python MVA_training/VBF/run3_model/dnn_train.py --label $label --region $region --category $category --year $year --bo --bo-trials 51 --bo-epochs 51 --bo-fold 0 --n-epochs 51 --batch-size 15536 --log-level INFO "

            # -- Quick test:
            # cmd_train="python MVA_training/VBF/run3_model/dnn_train.py --label $label --region $region --category $category --year $year --bo --bo-trials 3 --bo-epochs 3 --bo-fold 0 --n-epochs 3 --batch-size 15536 --log-level INFO "
            # cmd_train="python MVA_training/VBF/run3_model/dnn_train.py --label $label --region $region --category $category --year $year --n-epochs 3 --batch-size 15536 --log-level DEBUG "

            # Active configuration:
            cmd_train="python MVA_training/VBF/run3_model/dnn_train.py --label $label --region $region --category $category --year $year --n-epochs 51 --log-level INFO "
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
