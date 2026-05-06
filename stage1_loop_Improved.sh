#!/bin/bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  -h            Show this help message
  -c <file>     Dataset YAML file (default: configs/datasets/dataset_nanoAODv12.yaml)
  -m <mode>     Mode: 0 (prestage), 1 (stage1), 2 (stage2), 3 (stage3), all,
                4 (copy datacards), 5|combine_vbf (build VBF combined cards/workspaces),
                6|combine_vbf_significance, 7|combine_vbf_impacts,
                8|combine_vbf_lhscan, 9|combine_vbf_all,
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
declare -a years=("2018PR" "2018" "2017" "2016postVFP" "2016preVFP" "2016" "2022preEE" "2022postEE" "2023" "2023BPix" "2024" "2025" "2026" "Run2" "Run3" "Run2Run3")
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
save_path_depot="/depot/cms/hmm/$USER/hmm_ntuples/copperheadV1clean"
save_path_work="/work/projects/hmm/$USER/hmm_ntuples/copperheadV1clean"
save_path_local="/depot/cms/users/$USER/hmm/copperheadV1clean"
save_path_eos="/store/user/rasharma/hmm/copperheadV1clean"

save_path="$save_path_work"   # default

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
save_path=${save_path}/${label}

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

combine_vbf_cards() {
    local card_dir="$1"
    local out_txt="$2"
    shift
    shift
    (
        cd "${card_dir}" || exit 1
        combineCards.py "$@" > "${out_txt}"
    )
}

vbf_card_dir() {
    echo "$save_path/stage3_datacards_${save_postfix}/score_${label}"
}

vbf_card_stem() {
    local year="$1"
    case "$year" in
        run2) echo "HMuMu_13TeV_Run2" ;;
        run3) echo "HMuMu_13TeV_Run3" ;;
        run2run3|run2+run3) echo "HMuMu_13TeV_Run2Run3" ;;
        *) echo "HMuMu_13TeV_${year}" ;;
    esac
}

ensure_vbf_card() {
    local year="$1"
    year="${year//$'\r'/}"
    year="${year#"${year%%[![:space:]]*}"}"
    year="${year%"${year##*[![:space:]]}"}"
    local card_dir
    card_dir="$(vbf_card_dir)"
    local stem
    stem="$(vbf_card_stem "$year")"
    local card_path="${card_dir}/${stem}.txt"

    log "ensure_vbf_card: requested year='${year}' card='${card_path}'"

    if [[ -s "${card_path}" ]]; then
        log "ensure_vbf_card: reusing existing non-empty card ${card_path}"
        return 0
    fi

    rm -f "${card_path}"

    case "$year" in
        2016preVFP|2016postVFP|2017|2018|2022preEE|2022postEE|2023|2023BPix|2024|2025|2026)
            local sr="datacard_vbf_SR_${year}.txt"
            local sb="datacard_vbf_SB_${year}.txt"
            log "ensure_vbf_card: matched single-year SR/SB case for ${year}"
            if [[ ! -f "${card_dir}/${sr}" || ! -f "${card_dir}/${sb}" ]]; then
                log "Missing SR/SB VBF datacards for ${year}:"
                log "  ${card_dir}/${sr}"
                log "  ${card_dir}/${sb}"
                return 1
            fi
            log "Building ${stem}.txt"
            combine_vbf_cards "${card_dir}" "${stem}.txt" "SR_${year}=${sr}" "SB_${year}=${sb}"
            ;;
        2016)
            log "ensure_vbf_card: matched combined 2016 case"
            ensure_vbf_card "2016preVFP" || return 1
            ensure_vbf_card "2016postVFP" || return 1
            log "Building ${stem}.txt"
            combine_vbf_cards "${card_dir}" "${stem}.txt" \
                "preVFP=HMuMu_13TeV_2016preVFP.txt" \
                "postVFP=HMuMu_13TeV_2016postVFP.txt"
            ;;
        Run2|run2)
            log "ensure_vbf_card: matched Run2 case"
            ensure_vbf_card "2016" || return 1
            ensure_vbf_card "2017" || return 1
            ensure_vbf_card "2018" || return 1
            log "Building ${stem}.txt"
            combine_vbf_cards "${card_dir}" "${stem}.txt" \
                "y2016=HMuMu_13TeV_2016.txt" \
                "y2017=HMuMu_13TeV_2017.txt" \
                "y2018=HMuMu_13TeV_2018.txt"
            ;;
        Run3|run3)
            log "ensure_vbf_card: matched Run3 case"
            ensure_vbf_card "2022preEE" || return 1
            ensure_vbf_card "2022postEE" || return 1
            ensure_vbf_card "2023" || return 1
            ensure_vbf_card "2023BPix" || return 1
            ensure_vbf_card "2024" || return 1
            # ensure_vbf_card "2025" || return 1
            # ensure_vbf_card "2026" || return 1
            log "Building ${stem}.txt"
            combine_vbf_cards "${card_dir}" "${stem}.txt" \
                "y2022preEE=HMuMu_13TeV_2022preEE.txt" \
                "y2022postEE=HMuMu_13TeV_2022postEE.txt" \
                "y2023=HMuMu_13TeV_2023.txt" \
                "y2023BPix=HMuMu_13TeV_2023BPix.txt" \
                "y2024=HMuMu_13TeV_2024.txt" 
                # "y2025=HMuMu_13TeV_2025.txt" \
                # "y2026=HMuMu_13TeV_2026.txt"
            ;;
        Run2Run3|run2run3|Run2+Run3|run2+run3)
            log "ensure_vbf_card: matched Run2Run3 case"
            ensure_vbf_card "Run2" || return 1
            ensure_vbf_card "Run3" || return 1
            log "Building ${stem}.txt"
            combine_vbf_cards "${card_dir}" "${stem}.txt" \
                "Run2=HMuMu_13TeV_Run2.txt" \
                "Run3=HMuMu_13TeV_Run3.txt"
            ;;
        *)
            log "Unsupported VBF combine year: ${year}"
            return 1
            ;;
    esac

    if [[ ! -s "${card_path}" ]]; then
        log "Failed to build non-empty VBF combined card: ${card_path}"
        return 1
    fi
}

ensure_vbf_workspace() {
    local year="$1"
    local card_dir
    card_dir="$(vbf_card_dir)"
    local stem
    stem="$(vbf_card_stem "$year")"
    ensure_vbf_card "$year" || return 1
    (
        cd "${card_dir}"
        if [[ ! -f "${stem}.root" ]]; then
            text2workspace.py "${stem}.txt" -m 125
        fi
    )
}

extract_significance_value() {
    local log_path="$1"
    sed -n 's/.*Significance:[[:space:]]*\([-+0-9.eE][0-9.eE+-]*\).*/\1/p' "${log_path}" | head -n 1
}

collect_vbf_significance_summary() {
    local card_dir
    card_dir="$(vbf_card_dir)"
    local summary_csv="${card_dir}/vbf_significance_summary_${save_postfix}.csv"
    local tmp_rows
    tmp_rows="$(mktemp "${card_dir}/.vbf_significance_rows_XXXXXX.csv")"
    : > "${tmp_rows}"
    # local ordered_years=(
    #     2016preVFP 2016postVFP 2016 2017 2018 Run2
    #     2022preEE 2022postEE 2023 2023BPix 2024 2025 2026 Run3 Run2Run3
    # )
    local ordered_years=(
        2022preEE 2022postEE 2023 2023BPix 2024 Run3
    )
    local y stem sig_log stat_log sig_val stat_val
    for y in "${ordered_years[@]}"; do
        stem="$(vbf_card_stem "$y")"
        sig_log="${card_dir}/${stem}_prefitsignificance.log"
        stat_log="${card_dir}/${stem}_prefitsignificance_StatOnly.log"
        echo "sig_log: ${sig_log}"
        echo "stat_log: ${stat_log}"
        if [[ -f "${sig_log}" || -f "${stat_log}" ]]; then
            sig_val="NA"
            stat_val="NA"
            if [[ -f "${sig_log}" ]]; then
                sig_val="$(extract_significance_value "${sig_log}" 2>/dev/null || echo "NA")"
            fi
            if [[ -f "${stat_log}" ]]; then
                stat_val="$(extract_significance_value "${stat_log}" 2>/dev/null || echo "NA")"
            fi
            printf "%s,%s,%s,%s\n" "${y}" "${stem}.txt" "${sig_val}" "${stat_val}" >> "${tmp_rows}"
        fi
    done

    {
        echo "year,card,significance,significance_statonly"
        cat "${tmp_rows}"
    } > "${summary_csv}"
    rm -f "${tmp_rows}"

    log "Collected VBF significance summary from logs:"
    log "  ${summary_csv}"
}

run_vbf_significance() {
    local year="$1"
    local card_dir
    card_dir="$(vbf_card_dir)"
    local stem
    stem="$(vbf_card_stem "$year")"
    ensure_vbf_card "$year" || return 1
    (
        cd "${card_dir}"
        combineTool.py -d "${stem}.txt" -M Significance -m 125 --expectSignal=1 -n "_${year}_${save_postfix}_" -t -1 --rMin -2 --rMax 5 > "${stem}_prefitsignificance.log"
        combineTool.py -d "${stem}.txt" -M Significance -m 125 --expectSignal=1 -n "_${year}_${save_postfix}_" -t -1 --rMin -2 --rMax 5 --freezeParameters allConstrainedNuisances > "${stem}_prefitsignificance_StatOnly.log"
    )
}

run_vbf_impacts() {
    local year="$1"
    local card_dir
    card_dir="$(vbf_card_dir)"
    local stem
    stem="$(vbf_card_stem "$year")"
    ensure_vbf_card "$year" || return 1
    ensure_vbf_workspace "$year" || return 1
    (
        cd "${card_dir}"
        combineTool.py -M Impacts -d "${stem}.root" -m 125 --freezeParameters MH -n ".impacts_${year}_${save_postfix}" --setParameterRanges r=-5.0,5.0 --doInitialFit --robustFit 1 -t -1 --expectSignal 1
        combineTool.py -M Impacts -d "${stem}.root" -m 125 --freezeParameters MH -n ".impacts_${year}_${save_postfix}" --setParameterRanges r=-5.0,5.0 --doFits --robustFit 1 -t -1 --expectSignal 1 --parallel 60
        combineTool.py -M Impacts -d "${stem}.root" -m 125 --freezeParameters MH -n ".impacts_${year}_${save_postfix}" --setParameterRanges r=-5.0,5.0 -o "impacts_${year}_${save_postfix}.json" -t -1 --expectSignal 1 --parallel 60
        plotImpacts.py -i "impacts_${year}_${save_postfix}.json" -o "impacts_${year}_${save_postfix}"
    )
}

run_vbf_lhscan() {
    local year="$1"
    local card_dir
    card_dir="$(vbf_card_dir)"
    local stem
    stem="$(vbf_card_stem "$year")"
    ensure_vbf_card "$year" || return 1
    ensure_vbf_workspace "$year" || return 1
    (
        cd "${card_dir}"
        log "Producing likelihood scan for ${year} in ${card_dir}, output name suffix ${save_postfix}"
        log "Datacard: ${card_dir}/${stem}.txt"
        combine -M MultiDimFit "${stem}.root" -m 125 --freezeParameters MH -n ".lhscan${year}_${save_postfix}.with_syst" --algo grid --points 100 --setParameterRanges r=-5.0,5.0 -t -1 --expectSignal 1
        combine -M MultiDimFit "${stem}.root" -m 125 --freezeParameters MH,allConstrainedNuisances -n ".lhscan${year}_${save_postfix}.with_syst.statonly" --algo grid --points 100 --setParameterRanges r=-5.0,5.0 -t -1 --expectSignal 1
        plot1DScan.py "higgsCombine.lhscan${year}_${save_postfix}.with_syst.MultiDimFit.mH125.root" \
            --main-label "With systematics" \
            --main-color 1 \
            --others "higgsCombine.lhscan${year}_${save_postfix}.with_syst.statonly.MultiDimFit.mH125.root:Stat-only:2" \
            -o "lh_scan_${year}_${save_postfix}"
    )
}

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
    # model_dir="2022preEE-2022postEE-2023-2023BPix-2024_h-peak_vbf"
    model_dir="2022postEE_h-peak_vbf"
    training_tag="trained_best_optuna_v1_multifold_050Trials"

    model_trained_path="./dnn/trained_models/${label}/${model_dir}"

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
    sig_l_stage2="ggH VBF"
    variation=false
    if ${variation}; then
        command2="python run_stage2_vbf.py -y $year -input $save_path -l $label --model_tag $training_tag --model_path $model_trained_path -data $data_l -bkg $bkg_l_stage2 -sig $sig_l_stage2 --save_postfix ${save_postfix}  "
        command3="python run_stage3_vbf.py --years $year -input $save_path -l $label  --save_postfix ${save_postfix} "
        variation_tag=""
    else
        command2="python run_stage2_vbf.py -y $year -input $save_path -l $label --model_tag $training_tag --model_path $model_trained_path -data $data_l -bkg $bkg_l_stage2 -sig $sig_l_stage2 --save_postfix ${save_postfix} --no_variations "
        command3="python run_stage3_vbf.py --years $year -input $save_path -l $label  --save_postfix ${save_postfix} --no_variations "
        variation_tag="_NoSyst"
    fi

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
            mva_name_suffix="_${save_postfix}${variation_tag}"
            load_path="${save_path}/stage2_histograms/score_${label}${mva_name_suffix}"
            command2p1="python plotter/plot_DNN_score.py --load $load_path -label $label -cat $category -y ${year} --region ${region2p} --mva_name ${label}${mva_name_suffix} --log-level DEBUG"
            log "Command: $command2p1"
            eval "$command2p1"

            region2p="h-peak"
            command2p2="python plotter/plot_DNN_score.py --load $load_path -label $label -cat $category -y ${year} --region ${region2p} --mva_name ${label}${mva_name_suffix} --log-level DEBUG"
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
        5|combine_vbf)
            log "Building combined VBF card/workspace for year $year..."
            ensure_vbf_card "$year"
            ensure_vbf_workspace "$year"
            ;;
        6|combine_vbf_significance)
            log "Running VBF significance for year $year..."
            run_vbf_significance "$year"
            ;;
        7|combine_vbf_impacts)
            log "Running VBF impacts for year $year..."
            run_vbf_impacts "$year"
            ;;
        8|combine_vbf_lhscan)
            log "Running VBF likelihood scan for year $year..."
            run_vbf_lhscan "$year"
            ;;
        9|combine_vbf_all)
            log "Running full VBF Combine chain for year $year..."
            ensure_vbf_card "$year"
            ensure_vbf_workspace "$year"
            run_vbf_significance "$year"
            collect_vbf_significance_summary
            run_vbf_impacts "$year"
            run_vbf_lhscan "$year"
            ;;
        10|combine_vbf_summary)
            log "Collecting VBF significance summary from logs..."
            collect_vbf_significance_summary
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
            cmd0="python src/copperhead/zpt_rewgt/derive/save_SF_rootFiles.py -l $label -y $year --input_path $save_path -dy_sample $dy_sample"
            if [[ "$dask" -ge 1 ]]; then
                cmd0+=" --use_gateway"
            fi
            if [[ "$cluster_index" != "0" ]]; then
                cmd0+=" --cluster_index $cluster_index"
            fi
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
            cmd_preproc="python MVA_training/VBF_run3/preprocess_dnn.py --label $label --region $region --category $category --year $year --log-level DEBUG "
            # Alternative cmd_train configurations (uncomment and adjust as needed):

            # -- Bayesian Optimization:
            # cmd_train="python MVA_training/VBF_run3/train_dnn.py --label $label --region $region --category $category --year $year --bo --bo-trials 55 --bo-epochs 100 --bo-fold 0 --n-epochs 100 --batch-size 15536 --log-level INFO "
            # cmd_train="python MVA_training/VBF_run3/train_dnn.py --label $label --region $region --category $category --year $year --bo --bo-trials 51 --bo-epochs 51 --bo-fold 0 --n-epochs 51 --batch-size 15536 --log-level INFO "

            # -- Quick test:
            # cmd_train="python MVA_training/VBF_run3/train_dnn.py --label $label --region $region --category $category --year $year --bo --bo-trials 3 --bo-epochs 3 --bo-fold 0 --n-epochs 3 --batch-size 15536 --log-level INFO "
            # cmd_train="python MVA_training/VBF_run3/train_dnn.py --label $label --region $region --category $category --year $year --n-epochs 3 --batch-size 15536 --log-level DEBUG "

            # Active configuration:
            cmd_train="python MVA_training/VBF_run3/train_dnn.py --label $label --region $region --category $category --year $year --n-epochs 51 --log-level INFO "
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
