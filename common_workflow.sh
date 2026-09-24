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
    is_cutflow="0"
    force_compact="0"
    switches_yaml_file=""
    compact_add_dnn_score="${COMPACT_ADD_DNN_SCORE:-0}"
    with_variations="${WITH_VARIATIONS:-0}"
    do_vbf_filter_study="${DO_VBF_FILTER_STUDY:-0}"
    chunksize="600000"
    max_file_len="900"
    save_root="/work/projects/hmm/$USER/hmm_ntuples/copperheadV1clean"
}

parse_common_args() {
    while getopts ":hc:m:v:y:l:n:b:d:o:r:t:p:i:M:S:w:ksfzZFDV" opt; do
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
            w) switches_yaml_file="${OPTARG}" ;;
            k) dask_gateway="1" ;;
            s) skip_bad_files="1" ;;
            f) debug_fraction="1" ;;
            z) is_sync="1" ;;
            Z) is_cutflow="1" ;;
            F) force_compact="1" ;;
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
    # Effective jet-eta topology the VBF DNN's dijet pair is restricted to
    # ("all" = none, or one of modules/selection.py's PAIR_JJ_ETA_REGIONS, e.g.
    # jj_both_central, jj_non_central, jj_both_he, ...). Source of truth is
    # analysis.jj_eta_region in the DNN config YAML (${dnn_config}); the
    # JJ_ETA_REGION env var, if set, overrides it. Resolved here so this wrapper
    # and preprocess_dnn.py agree on the value that gets baked into dnn_base_dir
    # below and passed via --jj-eta-region.
    if [[ -n "${JJ_ETA_REGION:-}" ]]; then
        dnn_jj_eta_region="${JJ_ETA_REGION}"
    else
        dnn_jj_eta_region="$(python3 -c '
import sys, yaml
try:
    cfg = yaml.safe_load(open(sys.argv[1])) or {}
    print((cfg.get("analysis") or {}).get("jj_eta_region") or "all")
except Exception:
    print("all")
' "${dnn_config}" 2>/dev/null || echo all)"
        dnn_jj_eta_region="${dnn_jj_eta_region:-all}"
    fi
    dnn_hpo_folds="${HPO_FOLDS:-0,1,2,3}"
    dnn_hpo_trials="${HPO_TRIALS:-50}"
    dnn_hpo_label="${HPO_LABEL:-v1_multifold_050Trials}"
    dnn_train_label="${TRAIN_LABEL:-trained_best_optuna_${dnn_hpo_label}}"
    # MODEL_YEARS lets the DNN model directory reference a different (e.g. combined-year
    # trained) model than the years actually processed via -y; defaults to -y's years.
    dnn_model_years_csv="${MODEL_YEARS:-${dnn_years_csv}}"
    dnn_model_years_slug="${dnn_model_years_csv//,/-}"
    dnn_base_dir="dnn/trained_models/${label}/${dnn_model_years_slug}_${region}_${category}_${dnn_jj_eta_region}"
    dnn_hpo_dir="${dnn_base_dir}/hpo_optuna/${dnn_hpo_label}"
    dnn_best_json="${OPTUNA_BEST_JSON:-${dnn_hpo_dir}/optuna_best.json}"
    dnn_model_path="./${dnn_base_dir}"

    # jj_eta_region for the actual VBF stage-2 category selection / stage-3 datacards --
    # independent of dnn_jj_eta_region above (that one is what phase-space subset the DNN was
    # trained on; this one is what phase-space subset stage-2/3 restrict the VBF category to
    # when filling histograms/building datacards -- the two are conceptually separate knobs
    # and don't have to match, though a typical workflow sets them the same).  Same JJ_ETA_REGION
    # env var as above is reused as the override for convenience, but there is no YAML
    # source-of-truth fallback here since stage2/3 aren't driven by a persistent config file
    # the way DNN training is -- unset means "all".
    jj_eta_region="${JJ_ETA_REGION:-all}"

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
    # `exec > >(...)`'s process substitution is a detached background job --
    # bash does not wait for it automatically.
    TEE_PID="$!"
    exec 3>>"${log_file}"
}

finish_logging() {
    # Let setup_logging's tee flush and exit cleanly before this script's
    # process — and whatever is capturing its stdout — tears down. Safe to
    # call even if setup_logging was never invoked (TEE_PID unset).
    if [[ -n "${TEE_PID:-}" ]]; then
        exec 1>&- 2>&-  # close our end so tee's stdin sees EOF and it can exit
        # Bounded: don't hang forever if something unexpected still holds the
        # pipe open (e.g. a leaked fd in a backgrounded child) -- 10s is far
        # more than tee needs to flush a plain text log file.
        ( sleep 10; kill "${TEE_PID}" 2>/dev/null || true ) &
        local watchdog_pid=$!
        wait "${TEE_PID}" 2>/dev/null || true
        kill "${watchdog_pid}" 2>/dev/null || true
        wait "${watchdog_pid}" 2>/dev/null || true
    fi
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
        # --sync makes run_stage1.py read prestage_output/processor_samples_<year>_NanoAODv<v>_sync.json
        # instead of the plain (non-suffixed) file -- its only effect (run_stage1.py:359-361). Without
        # it, -z silently falls back to the plain file, which is NOT guaranteed to be the small sync
        # sample set: both filenames are keyed only by year+NanoAODv, so a real/full prestage run for
        # the same year+version overwrites the plain file with production-scale samples. Confirmed live:
        # for 2022preEE v12 the plain file's data_C had 108 files / 158M events vs. the _sync.json's 3
        # small samples -- update_sync_references.sh appeared to hang because stage-1 was actually
        # processing 158M real events instead of ~1k sync events.
        printf '%s\n' "--sync"
    fi
    if [[ "${is_cutflow}" == "1" ]]; then
        # -z/--sync (sample-list selection) and -Z/--isCutflow (per-chunk cutflow shard
        # output) are independent: a caller may want either alone, or both together (e.g.
        # the CI sync-regression test wants both -- small sample set AND cutflow JSON).
        printf '%s\n' "--isCutflow"
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
    if [[ -n "${switches_yaml_file}" ]]; then
        cmd+=(--switches-yaml "${switches_yaml_file}")
    fi
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
    if [[ "${force_compact}" == "1" ]]; then
        cmd+=(--rerun)
    fi
    while IFS= read -r arg; do
        [[ -n "${arg}" ]] && cmd+=("${arg}")
    done < <(append_gateway_args)
    printf '%s\0' "${cmd[@]}"
}

run_cutflow_merge() {
    # Merges the per-chunk cutflow_*.npz shards stage-1 writes (-Z/--isCutflow)
    # into one whole-dataset cutflow per sample, via scripts/merge_cutflow_npz_file.py.
    # Every sample directory lives under stage1_output/<year>/f1_0/<sample>/ --
    # same layout build_stage1_cmd's --save_path writes to and every other
    # f1_0-based reader in this repo (fetch_hists_for_zpt_weights.py,
    # categorizer.py, ...) already assumes.
    local year="$1"
    local base_dir="${save_path}/stage1_output/${year}/f1_0"
    if [[ ! -d "${base_dir}" ]]; then
        log "No stage1 output at ${base_dir}; skipping cutflow merge for year ${year}."
        return
    fi
    local sample_dir sample_name out_json found
    for sample_dir in "${base_dir}"/*/; do
        [[ -d "${sample_dir}" ]] || continue
        sample_name="$(basename "${sample_dir}")"
        found="$(find "${sample_dir}" -name 'cutflow_*.npz' -print -quit)"
        if [[ -z "${found}" ]]; then
            log "No cutflow_*.npz under ${sample_dir}; skipping ${sample_name} (${year})."
            continue
        fi
        out_json="${sample_dir%/}/cutflow_merged_${sample_name}.json"
        run_cmd python scripts/merge_cutflow_npz_file.py "${sample_dir}" -o "${out_json}"
    done
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
    if [[ "${jj_eta_region}" != "all" ]]; then
        cmd+=(--jj_eta_region "${jj_eta_region}")
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
    if [[ "${jj_eta_region}" != "all" ]]; then
        stage2_suffix="_${jj_eta_region}${stage2_suffix}"
    fi
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
    if [[ "${jj_eta_region}" != "all" ]]; then
        cmd+=(--jj_eta_region "${jj_eta_region}")
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
    local -a cmd2=(python src/copperhead/zpt_rewgt/derive/get_polyFit.py -l "${label}" -y "${year}" --dy_sample "${dy_sample}" --njet "${njet}" --save_postfix "${save_postfix}" --input_path "${save_path}")

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
        --jj-eta-region "${dnn_jj_eta_region}"
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
    log "  DNN jj_eta_region: ${dnn_jj_eta_region}"
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
    if [[ "${jj_eta_region}" != "all" ]]; then
        postfix="${postfix}_${jj_eta_region}"
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
            ensure_vbf_card 2025
            ensure_vbf_card 2026
            combine_vbf_cards "${card_dir}" "${stem}.txt" \
                "y2022preEE=HMuMu_13TeV_2022preEE.txt" \
                "y2022postEE=HMuMu_13TeV_2022postEE.txt" \
                "y2023=HMuMu_13TeV_2023.txt" \
                "y2023BPix=HMuMu_13TeV_2023BPix.txt" \
                "y2024=HMuMu_13TeV_2024.txt" \
                "y2025=HMuMu_13TeV_2025.txt" \
                "y2026=HMuMu_13TeV_2026.txt"
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

JJ_CENTRAL_NONCENTRAL_REGIONS=(jj_both_central jj_non_central)

# --- VBF jj-eta-region (central + non-central) combination --------------------------------
# Combines the already-built jj_both_central and jj_non_central per-year cards into one card,
# treating the two phase spaces as separate channels 
#
# "jj_combined" is a bash-only pseudo-value for ${jj_eta_region}, recognized only by these
# functions (via vbf_card_dir()/stage3_output_postfix()) -- it is never a valid
# run_stage2_vbf.py/run_stage3_vbf.py --jj_eta_region choice, since stage2/3 always need one
# real phase space to select events by. Works for any year token ensure_vbf_card accepts,
# including pseudo-years like Run3 -- combines that region's own Run3-combined card.
ensure_vbf_jjcombined_card() {
    local year="$1"
    local saved_jj_eta_region="${jj_eta_region}"
    jj_eta_region="jj_combined"

    local combined_dir
    combined_dir="$(vbf_card_dir)"
    mkdir -p "${combined_dir}"
    local stem
    stem="$(vbf_card_stem "${year}")"
    local card_path="${combined_dir}/${stem}.txt"

    if [[ -s "${card_path}" ]]; then
        jj_eta_region="${saved_jj_eta_region}"
        return 0
    fi
    rm -f "${card_path}"

    local region region_dir region_card rel_path
    local -a combine_args=()
    for region in "${JJ_CENTRAL_NONCENTRAL_REGIONS[@]}"; do
        jj_eta_region="${region}"
        ensure_vbf_card "${year}"
        region_dir="$(vbf_card_dir)"
        region_card="${region_dir}/${stem}.txt"
        [[ -f "${region_card}" ]] || die "Missing ${region} VBF card for ${year}: ${region_card}"
        rel_path="$(realpath --relative-to="${combined_dir}" "${region_card}")"
        combine_args+=("${region}=${rel_path}")
    done

    jj_eta_region="jj_combined"
    combine_vbf_cards "${combined_dir}" "${stem}.txt" "${combine_args[@]}"
    [[ -s "${card_path}" ]] || die "Failed to build non-empty combined jj-region VBF card ${card_path}"

    jj_eta_region="${saved_jj_eta_region}"
}

# Runs card+workspace+significance+limit under jj_eta_region="jj_combined" for one year, then
# restores the caller's jj_eta_region. Reuses ensure_vbf_workspace/run_vbf_significance/
# run_vbf_limit unchanged: since ensure_vbf_jjcombined_card already wrote the combined card,
# their own internal ensure_vbf_card call just finds it already there and returns immediately.
run_vbf_jjcombined_significance_and_limit() {
    local year="$1"
    local saved_jj_eta_region="${jj_eta_region}"
    jj_eta_region="jj_combined"

    ensure_vbf_jjcombined_card "${year}"
    ensure_vbf_workspace "${year}"
    run_vbf_significance "${year}"
    run_vbf_limit "${year}"

    jj_eta_region="${saved_jj_eta_region}"
}

run_vbf_jjcombined_impacts() {
    local year="$1"
    local saved_jj_eta_region="${jj_eta_region}"
    jj_eta_region="jj_combined"

    ensure_vbf_jjcombined_card "${year}"
    run_vbf_impacts "${year}"

    jj_eta_region="${saved_jj_eta_region}"
}

collect_vbf_jjcombined_summaries() {
    local saved_jj_eta_region="${jj_eta_region}"
    jj_eta_region="jj_combined"

    collect_vbf_significance_summary
    collect_vbf_limit_summary

    jj_eta_region="${saved_jj_eta_region}"
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
    local ordered_years=(2022preEE 2022postEE 2023 2023BPix 2024 2025 2026 Run3)
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
    local ordered_years=(2022preEE 2022postEE 2023 2023BPix 2024 2025 2026 Run3)
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

# Number of entries in a ROOT TTree, or 0 if the file/tree is missing/unreadable.
# Used to detect a Combine fit that silently produced an empty output file (which
# combineTool.py's own collection step doesn't catch until much later, crashing with
# an unhelpful TypeError) rather than to inspect fit content.
combine_root_tree_entries() {
    local root_path="$1"
    local tree_name="$2"
    python3 -c "
import ROOT
ROOT.gErrorIgnoreLevel = ROOT.kFatal
try:
    f = ROOT.TFile.Open('${root_path}')
except OSError:
    f = None
if not f or f.IsZombie():
    print(0)
else:
    t = f.Get('${tree_name}')
    print(t.GetEntries() if t else 0)
" 2>/dev/null
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
            # --cminDefaultMinimizerStrategy: neither strategy converges uniformly across
            # all year/region/r_inject combos. Strategy 1's Hesse step fails to converge
            # (Edm stuck above tolerance) when the Asimov dataset sits almost exactly at
            # the nominal templates (e.g. 2023BPix r=1) -- strategy 0 fixes that. But
            # strategy 0 in turn fails outright for some combos (e.g. 2022preEE
            # jj_both_central r=0: initial-fit ROOT output has 0 entries), while strategy 2
            # fails on others that strategy 0 handles fine (e.g. 2022preEE jj_non_central
            # r=1). So: try strategy 0 first (cheaper, usually sufficient), and only if its
            # initial-fit output has no entries in the "limit" tree, discard it and retry
            # with strategy 2 -- then use whichever strategy actually converged for the
            # rest of this (year, r_inject)'s --doFits/-o/plotImpacts steps, so they stay
            # consistent with the initial fit they're profiling around.
            local initial_root="higgsCombine_initialFit_.impacts_${year}_${save_postfix}_${tag}.MultiDimFit.mH125.root"
            local strategy=0
            rm -f "${initial_root}"
            combineTool.py -M Impacts -d "${stem}.root" -m 125 --freezeParameters MH -n ".impacts_${year}_${save_postfix}_${tag}" --setParameterRanges r=-5.0,5.0 --doInitialFit --robustFit 1 --cminDefaultMinimizerStrategy "${strategy}" -t -1 --expectSignal "${r_inject}"
            local entries
            entries="$(combine_root_tree_entries "${initial_root}" limit)"
            if [[ "${entries:-0}" -lt 1 ]]; then
                echo "run_vbf_impacts: strategy 0 initial fit for year=${year} tag=${tag} produced no entries, retrying with strategy 2"
                rm -f "${initial_root}"
                strategy=2
                combineTool.py -M Impacts -d "${stem}.root" -m 125 --freezeParameters MH -n ".impacts_${year}_${save_postfix}_${tag}" --setParameterRanges r=-5.0,5.0 --doInitialFit --robustFit 1 --cminDefaultMinimizerStrategy "${strategy}" -t -1 --expectSignal "${r_inject}"
            fi
            combineTool.py -M Impacts -d "${stem}.root" -m 125 --freezeParameters MH -n ".impacts_${year}_${save_postfix}_${tag}" --setParameterRanges r=-5.0,5.0 --doFits --robustFit 1 --cminDefaultMinimizerStrategy "${strategy}" -t -1 --expectSignal "${r_inject}" --parallel 60
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
        # Named (lnN/shape/param) systematics only - excludes autoMCStats bin-by-bin
        # stat parameters and the DY rateParams, which stay floating in the
        # "MCStat+DYNorm" scan below.
        local named_systs
        named_systs="$(awk '$2=="lnN" || $2=="shape" || $2=="param" {print $1}' "${stem}.txt" | sort -u | paste -sd, -)"
        combine -M MultiDimFit "${stem}.root" -m 125 --freezeParameters MH -n ".lhscan${year}_${save_postfix}.with_syst" --algo grid --points 100 --setParameterRanges r=-5.0,5.0 -t -1 --expectSignal 1
        combine -M MultiDimFit "${stem}.root" -m 125 --freezeParameters "MH,${named_systs}" -n ".lhscan${year}_${save_postfix}.with_syst.mcstat_dynorm" --algo grid --points 100 --setParameterRanges r=-5.0,5.0 -t -1 --expectSignal 1
        combine -M MultiDimFit "${stem}.root" -m 125 --freezeParameters MH,allConstrainedNuisances -n ".lhscan${year}_${save_postfix}.with_syst.statonly" --algo grid --points 100 --setParameterRanges r=-5.0,5.0 -t -1 --expectSignal 1
        plot1DScan.py "higgsCombine.lhscan${year}_${save_postfix}.with_syst.MultiDimFit.mH125.root" \
            --main-label "with-syst" \
            --main-color 1 \
            --others "higgsCombine.lhscan${year}_${save_postfix}.with_syst.mcstat_dynorm.MultiDimFit.mH125.root:Stat+DYNorm:2" \
               "higgsCombine.lhscan${year}_${save_postfix}.with_syst.statonly.MultiDimFit.mH125.root:Stat-only:4" \
            --breakdown "Syst,DYNorm,Stat" \
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
    echo "  jj_eta_region (stage2/3): ${jj_eta_region}"
    echo "  isMC: ${is_mc}"
}
