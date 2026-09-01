#!/bin/bash

set -euo pipefail

common_defaults() {
    dataset_yaml="configs/datasets/dataset_nanoAODv12.yaml"
    nanoaod_version="12"
    declare -ga years=("2018PR" "2018" "2017" "2016postVFP" "2016preVFP" "2016" "2022preEE" "2022postEE" "2023" "2023BPix" "2024" "2025" "2026" "Run2" "Run3" "Run2Run3")
    label="Default_nanoAODv9"
    mode="all"
    debug_level="0"
    skip_bad_files="0"
    debug_fraction="0"
    njet="0"
    nbin="100"
    save_postfix="$(date +%b%d_%Y)"
    region="h-peak"
    category="vbf"
    postfix=""
    dask_gateway="0"
    cluster_index="0"
    is_mc="0"
    is_sync="0"
    compact_add_dnn_score="${COMPACT_ADD_DNN_SCORE:-0}"
    with_variations="${WITH_VARIATIONS:-0}"
    do_vbf_filter_study="${DO_VBF_FILTER_STUDY:-0}"
    chunksize="600000"
    max_file_len="900"
    save_root="/work/projects/hmm/$USER/hmm_ntuples/copperheadV1clean"
}

parse_common_args() {
    while getopts ":hc:m:v:y:l:n:b:d:o:r:t:p:i:M:S:ksfzDV" opt; do
        case "${opt}" in
            h) usage ;;
            c) dataset_yaml="${OPTARG}" ;;
            m) mode="${OPTARG}" ;;
            v) nanoaod_version="${OPTARG}" ;;
            y) IFS=' ' read -r -a years <<< "$(printf '%s' "${OPTARG}" | tr ',' ' ')" ;;
            l) label="${OPTARG}" ;;
            n) njet="${OPTARG}" ;;
            b) nbin="${OPTARG}" ;;
            d) debug_level="${OPTARG}" ;;
            o) save_postfix="${OPTARG}" ;;
            r) region="${OPTARG}" ;;
            t) category="${OPTARG}" ;;
            p) postfix="${OPTARG}" ;;
            i) cluster_index="${OPTARG}" ;;
            M) is_mc="${OPTARG}" ;;
            S) save_root="${OPTARG}" ;;
            k) dask_gateway="1" ;;
            s) skip_bad_files="1" ;;
            f) debug_fraction="1" ;;
            z) is_sync="1" ;;
            D) compact_add_dnn_score="1" ;;
            V) do_vbf_filter_study="1" ;;
            *) usage ;;
        esac
    done
    shift $((OPTIND - 1)) || true
    if [[ "$#" -gt 0 ]]; then
        echo "Unexpected positional arguments: $*" >&2
        usage
    fi
    save_path="${save_root}/${label}"
    dnn_years_csv="$(join_by "," "${years[@]}")"
    dnn_years_slug="${dnn_years_csv//,/-}"
    dnn_config="${DNN_CONFIG:-configs/dnn_run3_vbf.yaml}"
    dnn_hpo_folds="${HPO_FOLDS:-0,1,2,3}"
    dnn_hpo_trials="${HPO_TRIALS:-50}"
    dnn_hpo_label="${HPO_LABEL:-v1_multifold_050Trials}"
    dnn_train_label="${TRAIN_LABEL:-trained_best_optuna_${dnn_hpo_label}}"
    # MODEL_YEARS lets the DNN model directory reference a different (e.g. combined-year
    # trained) model than the years actually processed via -y; defaults to -y's years.
    dnn_model_years_csv="${MODEL_YEARS:-${dnn_years_csv}}"
    dnn_model_years_slug="${dnn_model_years_csv//,/-}"
    dnn_base_dir="dnn/trained_models/${label}/${dnn_model_years_slug}_${region}_${category}"
    dnn_hpo_dir="${dnn_base_dir}/hpo_optuna/${dnn_hpo_label}"
    dnn_best_json="${OPTUNA_BEST_JSON:-${dnn_hpo_dir}/optuna_best.json}"
    dnn_model_path="./${dnn_base_dir}"

    # PU-DNN (jet-level HS-vs-PU classifier, MVA_training/pileup_dnn/train_pu_dnn.py)
    # is unrelated to the VBF category DNN above: it trains on stage1's own
    # compacted output and is consumed back inside stage1 (do_use_pu_dnn_score),
    # not on stage2 output. Sample-name globs are the training script's own
    # --use-glob patterns, resolved against each year's compacted/ dir below.
    pu_dnn_dy_glob="${PU_DNN_DY_GLOB:-dyTo2Mu_M-50_aMCatNLO}"
    pu_dnn_ttbar_glob="${PU_DNN_TTBAR_GLOB:-ttjets_*}"
    pu_dnn_ewk_glob="${PU_DNN_EWK_GLOB:-ewk_*}"
    pu_dnn_regions="${PU_DNN_REGIONS:-HEpos HEneg HFpos HFneg}"
    pu_dnn_out_tag="${PU_DNN_OUT_TAG:-}"
}

setup_logging() {
    log_dir="log_old"
    mkdir -p "${log_dir}"
    shopt -s nullglob
    for f in log_*.txt; do
        mv "${f}" "${log_dir}/"
    done
    shopt -u nullglob

    log_file="log_$(date +%Y%m%d_%H%M%S).txt"
    exec > >(tee -a "${log_file}") 2>&1
    exec 3>>"${log_file}"
}

log() {
    echo "$@"
}

die() {
    log "ERROR: $*"
    exit 1
}

join_by() {
    local delimiter="$1"
    shift
    local first="${1:-}"
    shift || true
    printf '%s' "${first}"
    local item
    for item in "$@"; do
        printf '%s%s' "${delimiter}" "${item}"
    done
}

print_cmd() {
    local rendered=""
    local arg
    for arg in "$@"; do
        if [[ "${arg}" == *[[:space:]]* ]]; then
            rendered+="\"${arg}\" "
        else
            rendered+="${arg} "
        fi
    done
    log "Command: ${rendered% }"
}

run_cmd() {
    print_cmd "$@"
    "$@"
}

run_cmd_timed() {
    print_cmd "$@"
    time "$@"
}

load_year_maps() {
    declare -gA year_data_map=(
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
        [2025]="B C D E F G"
        [2026]="A B D"
        [run2]="A B C D E F G H"
        [run3]="C D E F G H I"
    )
    bkg_groups="DY Top VV EWK VVV"
    sig_groups="Higgs"

    if [[ "${debug_level}" -ge 1 ]]; then
        log "Debug mode ON"
        # year_data_map["2016preVFP"]=""
        # year_data_map["2016postVFP"]=""
        # year_data_map["2017"]=""
        # year_data_map["2018"]=""
        # year_data_map["2022postEE"]=""
        bkg_groups=""
        sig_groups=""
    fi
}

data_streams_for_year() {
    local year="$1"
    printf '%s' "${year_data_map[${year}]:-}"
}

debug_flag() {
    if [[ "${debug_level}" -ge 2 ]]; then
        printf 'DEBUG'
    else
        printf 'INFO'
    fi
}

append_gateway_args() {
    if [[ "${dask_gateway}" == "1" ]]; then
        printf '%s\n' "--use_gateway"
    fi
    if [[ "${cluster_index}" != "0" ]]; then
        printf '%s\n' "--cluster_index" "${cluster_index}"
    fi
}

append_prestage_args() {
    if [[ "${debug_fraction}" == "1" ]]; then
        printf '%s\n' "-frac" "0.1"
    fi
    if [[ "${skip_bad_files}" == "1" ]]; then
        printf '%s\n' "--skipBadFiles"
    fi
    if [[ "${is_sync}" == "1" ]]; then
        printf '%s\n' "--sync"
    fi
}

append_stage1_args() {
    if [[ "${debug_fraction}" == "1" ]]; then
        printf '%s\n' "--test_mode"
    fi
    if [[ "${is_sync}" == "1" ]]; then
        printf '%s\n' "--sync" "--isCutflow"
    fi
}

stage2_bkg_groups() {
    local out="${bkg_groups}"
    if [[ "${out}" == *"Top"* ]]; then
        out="${out/Top/TT ST}"
    fi
    printf '%s' "${out}"
}

variation_suffix() {
    if [[ "${with_variations}" == "1" ]]; then
        printf ''
    else
        printf '_NoSyst'
    fi
}

build_prestage_cmd() {
    local year="$1"
    local data_streams="$2"
    local -a data_args=()
    local -a bkg_args=()
    local -a sig_args=()
    local token
    for token in ${data_streams}; do
        data_args+=("${token}")
    done
    for token in ${bkg_groups}; do
        bkg_args+=("${token}")
    done
    for token in ${sig_groups}; do
        sig_args+=("${token}")
    done
    local cmd=(
        python run_prestage.py
        --chunksize "${chunksize}"
        -y "${year}"
        --yaml "${dataset_yaml}"
        --data "${data_args[@]}"
        --background "${bkg_args[@]}"
        --signal "${sig_args[@]}"
        --NanoAODv "${nanoaod_version}"
        --log-level "$(debug_flag)"
    )
    while IFS= read -r arg; do
        [[ -n "${arg}" ]] && cmd+=("${arg}")
    done < <(append_prestage_args)
    while IFS= read -r arg; do
        [[ -n "${arg}" ]] && cmd+=("${arg}")
    done < <(append_gateway_args)
    printf '%s\0' "${cmd[@]}"
}

build_stage1_cmd() {
    local year="$1"
    local cmd=(
        python -W ignore run_stage1.py
        -y "${year}"
        --save_path "${save_path}"
        --NanoAODv "${nanoaod_version}"
        --max_file_len "${max_file_len}"
        --yaml "${dataset_yaml}"
        --skipSamples
        --log-level "$(debug_flag)"
    )
    while IFS= read -r arg; do
        [[ -n "${arg}" ]] && cmd+=("${arg}")
    done < <(append_stage1_args)
    while IFS= read -r arg; do
        [[ -n "${arg}" ]] && cmd+=("${arg}")
    done < <(append_gateway_args)
    printf '%s\0' "${cmd[@]}"
}

build_compact_cmd() {
    local year="$1"
    local cmd=(
        python scripts/compact_parquet_data.py
        -y "${year}"
        --input_path "${save_path}"
        --log-level "$(debug_flag)"
    )
    if [[ "${compact_add_dnn_score}" == "1" ]]; then
        cmd+=(
            -m "${dnn_model_path}"
            --model_tag "${dnn_train_label}"
            --add_dnn_score
            --fix_dimuon_mass
            --save_postfix "${save_postfix}"
        )
    fi
    while IFS= read -r arg; do
        [[ -n "${arg}" ]] && cmd+=("${arg}")
    done < <(append_gateway_args)
    printf '%s\0' "${cmd[@]}"
}

build_pu_dnn_train_cmd() {
    local year="$1"
    local compacted_dir="${save_path}/stage1_output/${year}/compacted"
    local out_tag="${pu_dnn_out_tag:-run${year}_dy_top_ewk_$(date +%b%d)}"
    local -a region_args=()
    local token
    for token in ${pu_dnn_regions}; do
        region_args+=("${token}")
    done
    local cmd=(
        python MVA_training/pileup_dnn/train_pu_dnn.py
        -i
        "${compacted_dir}/${pu_dnn_dy_glob}/*/*.parquet"
        "${compacted_dir}/${pu_dnn_ttbar_glob}/*/*.parquet"
        "${compacted_dir}/${pu_dnn_ewk_glob}/*/*.parquet"
        --use-glob
        -o "validation/pu_dnn/${out_tag}"
        --regions "${region_args[@]}"
    )
    # Raw passthrough for the training script's many hyperparameter/plotting
    # flags (epochs, lr, pt-min/max, ...) so this wrapper doesn't need to
    # hand-mirror every one of them; word-split is intentional here.
    if [[ -n "${PU_DNN_EXTRA_ARGS:-}" ]]; then
        local -a extra_args=(${PU_DNN_EXTRA_ARGS})
        cmd+=("${extra_args[@]}")
    fi
    printf '%s\0' "${cmd[@]}"
}

build_stage2_cmd() {
    local year="$1"
    local -a data_args=()
    local -a bkg_args=()
    local -a sig_args=()
    local token
    for token in $(data_streams_for_year "${year}"); do
        data_args+=("${token}")
    done
    for token in $(stage2_bkg_groups); do
        bkg_args+=("${token}")
    done
    for token in ggH VBF; do
        sig_args+=("${token}")
    done
    local cmd=(
        python run_stage2_vbf.py
        -y "${year}"
        -input "${save_path}"
        -l "${label}"
        --model_tag "${dnn_train_label}"
        --model_path "${dnn_model_path}"
        -data "${data_args[@]}"
        -bkg "${bkg_args[@]}"
        -sig "${sig_args[@]}"
        --save_postfix "${save_postfix}"
        --log-level "$(debug_flag)"
    )
    if [[ "${with_variations}" != "1" ]]; then
        cmd+=(--no_variations)
    fi
    if [[ "${do_vbf_filter_study}" == "1" ]]; then
        cmd+=(--vbf_filter_study)
    fi
    while IFS= read -r arg; do
        [[ -n "${arg}" ]] && cmd+=("${arg}")
    done < <(append_gateway_args)
    printf '%s\0' "${cmd[@]}"
}

build_stage2_plot_cmd() {
    local year="$1"
    local region_name="$2"
    local stage2_suffix="$(variation_suffix)"
    if [[ "${do_vbf_filter_study}" == "1" ]]; then
        stage2_suffix="_vbf_filter_study${stage2_suffix}"
    fi
    local load_path="${save_path}/stage2_histograms/score_${label}_${save_postfix}${stage2_suffix}"
    local mva_name="${label}_${save_postfix}${stage2_suffix}"
    local cmd=(
        python plotter/plot_DNN_score.py
        --load "${load_path}"
        -label "${label}"
        -cat "${category}"
        -y "${year}"
        --region "${region_name}"
        --mva_name "${mva_name}"
        --log-level DEBUG
    )
    if [[ "${do_vbf_filter_study}" == "1" ]]; then
        cmd+=(--vbf_filter_study)
    fi
    printf '%s\0' "${cmd[@]}"
}

build_stage3_cmd() {
    local year="$1"
    local cmd=(
        python run_stage3_vbf.py
        --years "${year}"
        -input "${save_path}"
        -l "${label}"
        --save_postfix "${save_postfix}"
        --log-level "$(debug_flag)"
    )
    if [[ "${with_variations}" != "1" ]]; then
        cmd+=(--no_variations)
    fi
    if [[ "${do_vbf_filter_study}" == "1" ]]; then
        cmd+=(--vbf_filter_study)
    fi    
    if [[ "${cluster_index}" != "0" ]]; then
        cmd+=(--cluster_index "${cluster_index}")
    fi
    printf '%s\0' "${cmd[@]}"
}

build_calib_cmd() {
    local year="$1"
    local closure_mode="$2"
    local cmd=(
        python src/lib/ebeMassResCalibration/getCalibrationFactor.py
        --NanoAODv "${nanoaod_version}"
        --years "${year}"
        -l "${label}"
        --input_path "${save_path}"
        --extraString "${postfix}"
        --ifbinned
        --log-level "$(debug_flag)"
    )
    if [[ "${closure_mode}" == "full" ]]; then
        cmd+=(--steps all)
    else
        cmd+=(--closure_test)
    fi
    if [[ "${is_mc}" == "1" ]]; then
        cmd+=(--isMC)
    fi
    while IFS= read -r arg; do
        [[ -n "${arg}" ]] && cmd+=("${arg}")
    done < <(append_gateway_args)
    printf '%s\0' "${cmd[@]}"
}

run_mode_from_nul() {
    local -a cmd=()
    while IFS= read -r -d '' token; do
        cmd+=("${token}")
    done
    run_cmd "${cmd[@]}"
}

run_mode_from_nul_timed() {
    local -a cmd=()
    while IFS= read -r -d '' token; do
        cmd+=("${token}")
    done
    run_cmd_timed "${cmd[@]}"
}

run_zpt_fit() {
    local year="$1"
    local dy_sample="IncDY_aMCatNLO_PySR07MayV2"
    # local dy_sample="IncDY_aMCatNLO_PySR07MayV2_ShapeNormOnly"
    local -a cmd0=(python src/copperhead/zpt_rewgt/derive/save_SF_rootFiles.py -l "${label}" -y "${year}" --input_path "${save_path}" -dy_sample "${dy_sample}")
    local -a cmd1=(python src/copperhead/zpt_rewgt/derive/do_f_test.py -l "${label}" -y "${year}" --dy_sample "${dy_sample}" --nbins "${nbin}" --njet "${njet}" --save_postfix "${save_postfix}" --debug)
    local -a cmd2=(python src/copperhead/zpt_rewgt/derive/get_polyFit.py -l "${label}" -y "${year}" --dy_sample "${dy_sample}" --njet "${njet}" --save_postfix "${save_postfix}")

    if [[ "${dask_gateway}" == "1" ]]; then
        cmd0+=(--use_gateway)
    fi
    if [[ "${cluster_index}" != "0" ]]; then
        cmd0+=(--cluster_index "${cluster_index}")
    fi

    case "${mode}" in
        zpt_fit0|zpt_fit) run_cmd "${cmd0[@]}" ;;
    esac
    case "${mode}" in
        zpt_fit1|zpt_fit|zpt_fit12) run_cmd "${cmd1[@]}" ;;
    esac
    case "${mode}" in
        zpt_fit2|zpt_fit|zpt_fit12) run_cmd "${cmd2[@]}" ;;
    esac
}

run_dnn_workflow_once() {
    local dnn_mode="$1"
    local run_preprocess="0"
    local run_hpo="0"
    local run_train="0"

    case "${dnn_mode}" in
        dnn) run_preprocess="1"; run_hpo="1"; run_train="1" ;;
        dnn_pre) run_preprocess="1" ;;
        dnn_hpo) run_hpo="1"; run_train="1" ;;
        dnn_train) run_train="1" ;;
        dnn_var_rank) die "Variable ranking is only available in the legacy Run-2 DNN flow." ;;
    esac

    local -a pre_cmd=(
        python MVA_training/VBF_run3/preprocess_dnn.py
        --config "${dnn_config}"
        --base-path "${save_path}/stage1_output/"
        --tag "${label}"
        --years "${dnn_years_csv}"
    )
    if [[ "${dask_gateway}" == "1" ]]; then
        pre_cmd+=(--use-dask-gateway)
        if [[ "${cluster_index}" != "0" ]]; then
            pre_cmd+=(--cluster-index "${cluster_index}")
        fi
    fi

    local -a hpo_cmd=(
        python MVA_training/VBF_run3/hpo_optuna.py
        --config "${dnn_config}"
        --data-dir "${dnn_base_dir}/"
        --out-dir "${dnn_hpo_dir}"
        --n-trials "${dnn_hpo_trials}"
        --folds "${dnn_hpo_folds}"
    )
    if [[ -n "${HPO_TIMEOUT_MIN:-}" ]]; then
        hpo_cmd+=(--timeout-min "${HPO_TIMEOUT_MIN}")
    fi

    local -a train_cmd=(
        python MVA_training/VBF_run3/train_dnn.py
        --config "${dnn_config}"
        --data-dir "${dnn_base_dir}/"
        --out-dir "${dnn_base_dir}/${dnn_train_label}"
    )

    log "Running DNN workflow for years=${dnn_years_csv}"
    log "  DNN config: ${dnn_config}"
    log "  DNN base dir: ${dnn_base_dir}"
    log "  DNN HPO dir: ${dnn_hpo_dir}"
    log "  DNN best json: ${dnn_best_json}"

    [[ "${run_preprocess}" == "1" ]] && run_cmd "${pre_cmd[@]}"
    [[ "${run_hpo}" == "1" ]] && run_cmd_timed "${hpo_cmd[@]}"
    if [[ "${run_train}" == "1" ]]; then
        if [[ -f "${dnn_best_json}" ]]; then
            train_cmd+=(--optuna-best-json "${dnn_best_json}")
        else
            log "WARNING: ${dnn_best_json} not found; training will use config hyperparameters."
        fi
        run_cmd_timed "${train_cmd[@]}"
    fi
}

stage3_output_postfix() {
    local postfix="${save_postfix}"
    if [[ "${do_vbf_filter_study}" == "1" ]]; then
        postfix="${postfix}_vbf_filter_study"
    fi
    printf '%s' "${postfix}"
}

vbf_card_dir() {
    printf '%s' "${save_path}/stage3_datacards_$(stage3_output_postfix)/score_${label}"
}

vbf_card_stem() {
    local year="$1"
    case "${year}" in
        run2|Run2) printf 'HMuMu_13TeV_Run2' ;;
        run3|Run3) printf 'HMuMu_13TeV_Run3' ;;
        run2run3|Run2Run3|run2+run3|Run2+Run3) printf 'HMuMu_13TeV_Run2Run3' ;;
        *) printf 'HMuMu_13TeV_%s' "${year}" ;;
    esac
}

combine_vbf_cards() {
    local card_dir="$1"
    local out_txt="$2"
    shift 2
    (
        cd "${card_dir}" || exit 1
        combineCards.py "$@" > "${out_txt}"
    )
}

ensure_vbf_card() {
    local year="$1"
    local card_dir
    card_dir="$(vbf_card_dir)"
    local stem
    stem="$(vbf_card_stem "${year}")"
    local card_path="${card_dir}/${stem}.txt"

    if [[ -s "${card_path}" ]]; then
        return 0
    fi
    rm -f "${card_path}"

    case "${year}" in
        2016preVFP|2016postVFP|2017|2018|2022preEE|2022postEE|2023|2023BPix|2024|2025|2026)
            local sr="datacard_vbf_SR_${year}.txt"
            local sb="datacard_vbf_SB_${year}.txt"
            [[ -f "${card_dir}/${sr}" && -f "${card_dir}/${sb}" ]] || die "Missing VBF SR/SB datacards for ${year}"
            combine_vbf_cards "${card_dir}" "${stem}.txt" "SR_${year}=${sr}" "SB_${year}=${sb}"
            ;;
        2016)
            ensure_vbf_card 2016preVFP
            ensure_vbf_card 2016postVFP
            combine_vbf_cards "${card_dir}" "${stem}.txt" \
                "preVFP=HMuMu_13TeV_2016preVFP.txt" \
                "postVFP=HMuMu_13TeV_2016postVFP.txt"
            ;;
        Run2|run2)
            ensure_vbf_card 2016
            ensure_vbf_card 2017
            ensure_vbf_card 2018
            combine_vbf_cards "${card_dir}" "${stem}.txt" \
                "y2016=HMuMu_13TeV_2016.txt" \
                "y2017=HMuMu_13TeV_2017.txt" \
                "y2018=HMuMu_13TeV_2018.txt"
            ;;
        Run3|run3)
            ensure_vbf_card 2022preEE
            ensure_vbf_card 2022postEE
            ensure_vbf_card 2023
            ensure_vbf_card 2023BPix
            ensure_vbf_card 2024
            combine_vbf_cards "${card_dir}" "${stem}.txt" \
                "y2022preEE=HMuMu_13TeV_2022preEE.txt" \
                "y2022postEE=HMuMu_13TeV_2022postEE.txt" \
                "y2023=HMuMu_13TeV_2023.txt" \
                "y2023BPix=HMuMu_13TeV_2023BPix.txt" \
                "y2024=HMuMu_13TeV_2024.txt"
            ;;
        Run2Run3|run2run3|Run2+Run3|run2+run3)
            ensure_vbf_card Run2
            ensure_vbf_card Run3
            combine_vbf_cards "${card_dir}" "${stem}.txt" \
                "Run2=HMuMu_13TeV_Run2.txt" \
                "Run3=HMuMu_13TeV_Run3.txt"
            ;;
        *)
            die "Unsupported VBF combine year: ${year}"
            ;;
    esac

    [[ -s "${card_path}" ]] || die "Failed to build non-empty VBF card ${card_path}"
}

ensure_vbf_workspace() {
    local year="$1"
    local card_dir
    card_dir="$(vbf_card_dir)"
    local stem
    stem="$(vbf_card_stem "${year}")"
    ensure_vbf_card "${year}"
    (
        cd "${card_dir}"
        [[ -f "${stem}.root" ]] || text2workspace.py "${stem}.txt" -m 125
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
    local ordered_years=(2022preEE 2022postEE 2023 2023BPix 2024 Run3)
    local year stem sig_log stat_log sig_val stat_val
    for year in "${ordered_years[@]}"; do
        stem="$(vbf_card_stem "${year}")"
        sig_log="${card_dir}/${stem}_prefitsignificance.log"
        stat_log="${card_dir}/${stem}_prefitsignificance_StatOnly.log"
        if [[ -f "${sig_log}" || -f "${stat_log}" ]]; then
            sig_val="NA"
            stat_val="NA"
            [[ -f "${sig_log}" ]] && sig_val="$(extract_significance_value "${sig_log}" 2>/dev/null || printf 'NA')"
            [[ -f "${stat_log}" ]] && stat_val="$(extract_significance_value "${stat_log}" 2>/dev/null || printf 'NA')"
            printf '%s,%s,%s,%s\n' "${year}" "${stem}.txt" "${sig_val}" "${stat_val}" >> "${tmp_rows}"
        fi
    done
    {
        echo "year,card,significance,significance_statonly"
        cat "${tmp_rows}"
    } > "${summary_csv}"
    rm -f "${tmp_rows}"
    log "Collected VBF significance summary: ${summary_csv}"
}

extract_expected_limit() {
    local log_path="$1"
    sed -n 's/.*Expected[[:space:]]*50\.0%:[[:space:]]*r[[:space:]]*<[[:space:]]*\([-+0-9.eE][0-9.eE+-]*\).*/\1/p' "${log_path}" | head -n 1
}

collect_vbf_limit_summary() {
    local card_dir
    card_dir="$(vbf_card_dir)"
    local summary_csv="${card_dir}/vbf_expected_limit_summary_${save_postfix}.csv"
    local tmp_rows
    tmp_rows="$(mktemp "${card_dir}/.vbf_limit_rows_XXXXXX.csv")"
    : > "${tmp_rows}"
    local ordered_years=(2022preEE 2022postEE 2023 2023BPix 2024 Run3)
    local year stem lim_log stat_log lim_val stat_val
    for year in "${ordered_years[@]}"; do
        stem="$(vbf_card_stem "${year}")"
        lim_log="${card_dir}/${stem}_expectedlimit.log"
        stat_log="${card_dir}/${stem}_expectedlimit_StatOnly.log"
        if [[ -f "${lim_log}" || -f "${stat_log}" ]]; then
            lim_val="NA"
            stat_val="NA"
            [[ -f "${lim_log}" ]] && lim_val="$(extract_expected_limit "${lim_log}" 2>/dev/null || printf 'NA')"
            [[ -f "${stat_log}" ]] && stat_val="$(extract_expected_limit "${stat_log}" 2>/dev/null || printf 'NA')"
            printf '%s,%s,%s,%s\n' "${year}" "${stem}.txt" "${lim_val}" "${stat_val}" >> "${tmp_rows}"
        fi
    done
    {
        echo "year,card,expected_limit_median,expected_limit_median_statonly"
        cat "${tmp_rows}"
    } > "${summary_csv}"
    rm -f "${tmp_rows}"
    log "Collected VBF expected-limit summary: ${summary_csv}"
}

run_vbf_limit() {
    local year="$1"
    local card_dir
    card_dir="$(vbf_card_dir)"
    local stem
    stem="$(vbf_card_stem "${year}")"
    ensure_vbf_card "${year}"
    (
        cd "${card_dir}"
        # Blinded analysis: --run blind computes the expected limit from the Asimov
        # background-only dataset instead of unblinding real data.
        combineTool.py -d "${stem}.txt" -M AsymptoticLimits -m 125 --run blind -n "_${year}_${save_postfix}_" --rMin -2 --rMax 5 > "${stem}_expectedlimit.log"
        combineTool.py -d "${stem}.txt" -M AsymptoticLimits -m 125 --run blind -n "_${year}_${save_postfix}_" --rMin -2 --rMax 5 --freezeParameters allConstrainedNuisances > "${stem}_expectedlimit_StatOnly.log"
    )
}

run_vbf_significance() {
    local year="$1"
    local card_dir
    card_dir="$(vbf_card_dir)"
    local stem
    stem="$(vbf_card_stem "${year}")"
    ensure_vbf_card "${year}"
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
    stem="$(vbf_card_stem "${year}")"
    ensure_vbf_workspace "${year}"
    # Blinded analysis: no observed impacts. Run both Asimov scenarios instead —
    # r=1 (SM signal injected) and r=0 (background-only) — so a nuisance that only
    # ranks high under one hypothesis is visible.
    local r_inject tag
    for r_inject in 1 0; do
        tag="r${r_inject}"
        (
            cd "${card_dir}"
            combineTool.py -M Impacts -d "${stem}.root" -m 125 --freezeParameters MH -n ".impacts_${year}_${save_postfix}_${tag}" --setParameterRanges r=-5.0,5.0 --doInitialFit --robustFit 1 -t -1 --expectSignal "${r_inject}"
            combineTool.py -M Impacts -d "${stem}.root" -m 125 --freezeParameters MH -n ".impacts_${year}_${save_postfix}_${tag}" --setParameterRanges r=-5.0,5.0 --doFits --robustFit 1 -t -1 --expectSignal "${r_inject}" --parallel 60
            combineTool.py -M Impacts -d "${stem}.root" -m 125 --freezeParameters MH -n ".impacts_${year}_${save_postfix}_${tag}" --setParameterRanges r=-5.0,5.0 -o "impacts_${year}_${save_postfix}_${tag}.json" -t -1 --expectSignal "${r_inject}" --parallel 60
            plotImpacts.py -i "impacts_${year}_${save_postfix}_${tag}.json" -o "impacts_${year}_${save_postfix}_${tag}"
        )
    done
}

run_vbf_lhscan() {
    local year="$1"
    local card_dir
    card_dir="$(vbf_card_dir)"
    local stem
    stem="$(vbf_card_stem "${year}")"
    ensure_vbf_workspace "${year}"
    (
        cd "${card_dir}"
        combine -M MultiDimFit "${stem}.root" -m 125 --freezeParameters MH -n ".lhscan${year}_${save_postfix}.with_syst" --algo grid --points 100 --setParameterRanges r=-5.0,5.0 -t -1 --expectSignal 1
        combine -M MultiDimFit "${stem}.root" -m 125 --freezeParameters MH,allConstrainedNuisances -n ".lhscan${year}_${save_postfix}.with_syst.statonly" --algo grid --points 100 --setParameterRanges r=-5.0,5.0 -t -1 --expectSignal 1
        plot1DScan.py "higgsCombine.lhscan${year}_${save_postfix}.with_syst.MultiDimFit.mH125.root" \
            --main-label "With systematics" \
            --main-color 1 \
            --others "higgsCombine.lhscan${year}_${save_postfix}.with_syst.statonly.MultiDimFit.mH125.root:Stat-only:2" \
            -o "lh_scan_${year}_${save_postfix}"
    )
}

require_workflow_root() {
    [[ -f "run_stage2_vbf.py" && -f "run_stage3_vbf.py" ]] || die "Run this script from the copperheadV2 checkout."
}

print_run_configuration() {
    echo "Running with the following parameters:"
    echo "  Dataset YAML: ${dataset_yaml}"
    echo "  NanoAOD version: ${nanoaod_version}"
    echo "  Years: ${years[*]}"
    echo "  Label: ${label}"
    echo "  Save path: ${save_path}"
    echo "  Debug mode: ${debug_level}"
    echo "  Mode: ${mode}"
    echo "  Skip bad files: ${skip_bad_files}"
    echo "  Fraction: ${debug_fraction}"
    echo "  nJet: ${njet}"
    echo "  Number of bins: ${nbin}"
    echo "  Output append: ${save_postfix}"
    echo "  Region: ${region}"
    echo "  Category: ${category}"
    echo "  VBF filter study: ${do_vbf_filter_study}"
    echo "  isMC: ${is_mc}"
}
