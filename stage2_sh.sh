#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "$(date)"

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
# label="Run3_nanoAODv12_FilterJetsHorn25GeV_pySR_Apr03_tightPassLepVeto_NoJER_JetIDFix"
# label="Run3_nanoAODv12_FilterJetsHorn25GeV_HE30GeV_Apr03_tightPassLepVeto_NoJER_JetIDFix"
# label="Run3_nanoAODv12_FilterJetsHorn25GeV_Apr03_tightPassLepVeto_NoJER_JetIDFix"
label="Run3_nanoAODv12_FilterJetsHorn25GeV_pySR_Apr09_tightPassLepVeto_NoJER"
# base_path="/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean"
base_path="/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean"

stage2_load_path="${base_path}/${label}/stage1_output"

category="ggh"

do_hyperparam_search="1" # true (enable hyperparameter search for BDT)
train_bdt="0"

step="${1:-}"
year="${2:-all}"
n_trials="${3:-050}"
date_tag="${4:-26Apr}"

# model_name="Run3_08March_HPeachFold_train_with_bestHP"
# model_name="Run3_09March_Check"
# model_name="Run3_10March_005Trials"
# model_name="Run3_10March_020Trials"
# model_name="Run3_10March_020Trials_EBEinTraining" # Run3_10March_020Trials_EBEinTraining
# model_name="Run3_10March_020Trials_RemovedAddHyperPars"
# model_name="Run3_10March_100Trials"
# model_name="Run3_10March_020Trials_oneHotEncoding"
# model_name="Run3_10March_020Trials_oneHotEncoding_Scan"
# model_name="Run3_09Mar_HPScan_100Trials"
# model_name="Run3_09Mar_HPScan_500Trials"
# model_name="Run3_07Apr_005Trials_pySR"
# model_name="Run3_07Apr_005Trials_HE30GeV"
# model_name="Run3_07Apr_005Trials"
# model_name="Run3_07Apr_100Trials_pySR"
model_name="Run3_${n_trials}Trials_pySR_${date_tag}"

# bdt_Run3_100Trials_pySR_13Apr_2022postEE

# Run3_3Trials_pySR_13Apr

if [[ -z "${step}" ]]; then
    echo "Usage: $0 <step> <year|all>"
    echo "Example: $0 0 all"
    echo "Example: $0 4 2023"
    exit 1
fi

# all_years=(2022preEE 2022postEE 2023 2023BPix 2024 all)
# all_years=(2022preEE 2022postEE 2023 2023BPix 2024)
all_years=(all)

if [[ "${year}" == "all" ]]; then
    years=("${all_years[@]}")
    model_trainYear="all"
    label_tag="AllYear_${date_tag}"
else
    years=("${year}")
    model_trainYear="${year}"
    label_tag="${year}_${date_tag}"
fi


stage2_label="${model_name}_${category}_${label_tag}"
stage2_save_path="${base_path}/${label}/${stage2_label}/stage2_output"

mva_base_path="${SCRIPT_DIR}"
trainer_script="MVA_training/ggH_BDT/my_trainer_withWeight_gpu.py"
stage3_variants_script="run_stage3_core_pdf_variants.sh"
run_stage3_core_pdf_variants="0"
stage3_baseline_variant_tag="all_core_pdfs"
bias_run_mode="${BIAS_RUN_MODE:-local}"
bias_local_jobs="${BIAS_LOCAL_JOBS:-32}"
bias_fitdiag_parallel="${BIAS_FITDIAG_PARALLEL:-32}"
bias_array_tasks="${BIAS_ARRAY_TASKS:-500}"
bias_expect_signal="${BIAS_EXPECT_SIGNAL:-1}"
bias_max_chi2ndf="${BIAS_MAX_CHI2NDF:-2.0}"


# -----------------------------------------------------
# Helpers
# -----------------------------------------------------
print_box() {
    local msg="$1"
    local len=${#msg}
    local border
    border=$(printf '%*s' "$((len + 4))" '' | tr ' ' '=')

    echo "$border"
    echo "| $msg |"
    echo "$border"
}

print_config() {
    print_box "Run configuration"
    echo "step             : ${step}"
    echo "year input       : ${year}"
    echo "years expanded   : ${years[*]}"
    echo "label            : ${label}"
    echo "category         : ${category}"
    echo "model_name       : ${model_name}"
    echo "model_trainYear  : ${model_trainYear}"
    echo "Search HP        : ${do_hyperparam_search}"
    echo "Number of trials : ${n_trials}"
    echo "bias_run_mode    : ${bias_run_mode}"
    echo "bias_local_jobs  : ${bias_local_jobs}"
    echo "bias_fitdiag_par : ${bias_fitdiag_parallel}"
    echo "bias_array_tasks : ${bias_array_tasks}"
    echo "bias_expect_sig  : ${bias_expect_signal}"
    echo "bias_max_chi2ndf : ${bias_max_chi2ndf}"
    echo "stage2_label     : ${stage2_label}"
    echo "base_path        : ${base_path}"
    echo "stage2_load_path : ${stage2_load_path}"
    echo "stage2_save_path : ${stage2_save_path}"
    echo "mva_base_path    : ${mva_base_path}"
    echo "script_dir       : ${SCRIPT_DIR}"
}

init_run_log() {
    local log_dir="output/bdt_${model_name}_${model_trainYear}/run_logs"
    mkdir -p "${log_dir}"
    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    RUN_LOG="${log_dir}/run_${timestamp}_step${step}_year${year}.log"

    exec > >(tee -a "${RUN_LOG}") 2>&1

    print_box "Run log created"
    echo "RUN_LOG: ${RUN_LOG}"
    echo ""
}

resolve_stage3_output_label() {
    local stage3_label="$1"
    if [[ "${run_stage3_core_pdf_variants}" == "1" ]]; then
        echo "${stage3_label}_${stage3_baseline_variant_tag}"
    else
        echo "${stage3_label}"
    fi
}

# -----------------------------------------------------
# Start logging + print config
# -----------------------------------------------------
init_run_log
print_config

if [[ ! -f "run_stage2.py" || ! -f "run_stage3.py" ]]; then
    echo "Required workflow entrypoints are missing. Run this script from the copperheadV2 checkout."
    exit 1
fi

if [[ "${run_stage3_core_pdf_variants}" == "1" && ! -f "${stage3_variants_script}" ]]; then
    echo "Missing stage3 variants wrapper: ${stage3_variants_script}"
    exit 1
fi

if [[ "${step}" == "0" && ! -f "${trainer_script}" ]]; then
    echo "Missing ggH trainer script: ${trainer_script}"
    echo "Step 0 cannot run in this checkout until that file is restored or the script path is updated."
    exit 1
fi


# -----------------------------------------------------
# Step 0: Train BDT
# Only run once, not looped over years here.
# -----------------------------------------------------
if [[ ( "${step}" == "0" || "${step}" == "all" ) && "${train_bdt}" == "1" ]]; then
    # mass_decorrelation_strat="default" # no mass decorrelation
    # mass_decorrelation_strat="peking" # peking's mass flattening
    mass_decorrelation_strat="targetZpeakMass" # target distribution Zpeak mass
    # mass_decorrelation_strat="targetHpeakMass" # target distribution Hpeak mass
    # mass_decorrelation_strat="targetHsidebandMass" # target distribution lower H sidebands and uppper ZCR mass window

    trainer_args=(
        python "${trainer_script}"
        --name "${model_name}"
        --year "${year}"
        -load "${stage2_load_path}"
        --massDeCorrStrat "${mass_decorrelation_strat}"
        --n_trials "${n_trials}"
    )

    if [[ "${do_hyperparam_search}" == "1" ]]; then
        print_box "Step 0: Scan the hyperparameters for the BDT"
        "${trainer_args[@]}" \
            -param_search "${do_hyperparam_search}"
    fi

    print_box "Step 0: Training the BDT for ggH"
    do_hyperparam_search="0" # true (enable hyperparameter search for BDT)
    "${trainer_args[@]}" \
        -param_search "${do_hyperparam_search}"
fi

# -----------------------------------------------------
# Step 1: Apply stage2 for BDT edge calculation
# -----------------------------------------------------
if [[ "${step}" == "1" || "${step}" == "all" ]]; then
    local_years="all"
    save_path="output/bdt_${model_name}_${model_trainYear}/ggH/score_edge_generation/${local_years}"
    echo "save_path: ${save_path}"

    for year_i in "${years[@]}"; do
        print_box "Step 1: Apply selection and add BDT score branch (${year_i})"
        sample_l="ggh vbf data"

        if [[ "${year_i}" != "all" ]]; then
            python run_stage2.py \
                -load ${stage2_load_path} \
                -save ${stage2_save_path} \
                --samples ${sample_l} \
                -cat ${category} \
                --fraction 1.0 \
                --year ${year_i} \
                --model_name ${model_name} \
                --model_trainYear ${model_trainYear} \
                --mva_base_path ${mva_base_path} \
                --edge_cfg_path "${save_path}/${stage2_label}" \
                # --do_jecUnc # FIXME: make do_jecUnc optional
        fi
    done
fi

# -----------------------------------------------------
# Step 2: Obtain target yield
# This one can take multiple years at once.
# -----------------------------------------------------
if [[ "${step}" == "2" || "${step}" == "all" ]]; then
    # This step for only year == all
    print_box "Step 2: Obtain the target yield"
    local_years="all"

    save_path="output/bdt_${model_name}_${model_trainYear}/ggH/score_edge_generation/${local_years}"
    mkdir -p "${save_path}"
    echo "save_path: ${save_path}"

    # FIXME: remove "all" from years list if exists
    unwanted_year="all"
    echo "old years list: ${years[*]}"

    echo "New years list: ${local_years[*]}"
    python stage2/ggH/score_edge_generation/determine_score_edge.py \
        -load ${stage2_save_path} \
        --years ${local_years[@]} \
        --label ${stage2_label} \
        -save ${save_path}

    python stage2/ggH/score_edge_generation/validation.py \
        --label "${stage2_label}" \
        -save "${save_path}"
fi

# -----------------------------------------------------
# Step 3: Obtain BDT edges
# -----------------------------------------------------
if [[ "${step}" == "3" || "${step}" == "all" ]]; then
    local_years="all"
    save_path="output/bdt_${model_name}_${model_trainYear}/ggH/score_edge_generation/${local_years}"
    echo "save_path: ${save_path}"

    for year_i in "${years[@]}"; do
        if [[ "${year_i}" != "all" ]]; then
            print_box "Step 3: Obtain the BDT score edges (${year_i})"
            python stage2/ggH/calculate_score_edges.py \
                -load ${stage2_save_path} \
                --year ${year_i} \
                -save "${save_path}/${stage2_label}" \
                --edge_cfg_path "${save_path}/${stage2_label}"
        fi
    done
fi

# -----------------------------------------------------
# Step 4: Run stage2 again to save BDT score and sub-category
# -----------------------------------------------------
if [[ "${step}" == "4" || "${step}" == "all" ]]; then
    local_years="all"
    save_path="output/bdt_${model_name}_${model_trainYear}/ggH/score_edge_generation/${local_years}"
    echo "save_path: ${save_path}"

    for year_i in "${years[@]}"; do
        print_box "Step 4: Run stage2 again to save BDT score + sub-category (${year_i})"
        sample_l="data ggh vbf dy tt ww wz"

        if [[ "${year_i}" != "all" ]]; then
            python run_stage2.py \
                -load ${stage2_load_path} \
                -save ${stage2_save_path} \
                --samples ${sample_l} \
                -cat ${category} \
                --fraction 1.0 \
                --year ${year_i} \
                --model_name ${model_name} \
                --model_trainYear ${model_trainYear} \
                --mva_base_path ${mva_base_path} \
                --edge_cfg_path "${save_path}/${stage2_label}"
        fi
    done
fi

# -----------------------------------------------------
# Step 5: Validation plots
# -----------------------------------------------------
if [[ "${step}" == "5" || "${step}" == "all" ]]; then
    for year_i in "${years[@]}"; do
        print_box "Step 5(a): Obtain the BDT score plot (${year_i})"
        sample_l="data ggh vbf dy tt ww wz"
        save_path="output/bdt_${model_name}_${model_trainYear}/ggH/categorization/plots/${year_i}"
        region="h-sidebands"
        mkdir -p "${save_path}"

        python validation/ggH/categorization/validation_plot.py \
            -label ${label} \
            -cat ${stage2_label} \
            --samples ${sample_l} \
            -y ${year_i} \
            --region ${region} \
            --base_path ${base_path} \
            -save ${save_path}
    done
fi

# -----------------------------------------------------
# Step 6: More validation plots / tables
# -----------------------------------------------------
if [[ "${step}" == "6" || "${step}" == "all" ]]; then
    edge_save_path="output/bdt_${model_name}_${model_trainYear}/ggH/score_edge_generation/all"
    echo "edge_save_path: ${edge_save_path}"

    for year_i in "${years[@]}"; do
        plot_save_path="output/bdt_${model_name}_${model_trainYear}/ggH/categorization/plots/${year_i}"
        mkdir -p "${plot_save_path}"

        print_box "Step 6: Prepare extra validation inputs (${year_i})"
        sample_l="ggh vbf dy tt ww wz"

        if [[ "${year_i}" != "all" ]]; then
            # Stage-2 doesn't accept the year == all option.
            python run_stage2.py \
                -load ${stage2_load_path} \
                -save ${stage2_save_path} \
                --samples ${sample_l} \
                -cat ${category} \
                --fraction 1.0 \
                --year ${year_i} \
                --model_name ${model_name} \
                --model_trainYear ${model_trainYear} \
                --mva_base_path ${mva_base_path} \
                --edge_cfg_path "${edge_save_path}/${stage2_label}" \
                --do_6p7
        fi

        region="signal"

        print_box "Step 6.1: Fig 6.7 (${year_i})"
        python validation/ggH/categorization/plot_6_7.py \
            -label ${label} \
            -cat ${stage2_label} \
            -y ${year_i} \
            --region ${region} \
            --base_path ${base_path} \
            --model_name ${model_name} \
            --bdt_year ${model_trainYear} \
            -save ${plot_save_path} \
            --mva_base_path ${mva_base_path}

        print_box "Step 6.2: Fig 6.8 (${year_i})"
        python validation/ggH/categorization/plot_6_8.py \
            -label ${label} \
            -cat ${stage2_label} \
            -y ${year_i} \
            --region ${region} \
            --base_path ${base_path} \
            -save ${plot_save_path}

        print_box "Step 6.3: Fig 6.13 (${year_i})"
        python validation/ggH/categorization/plot_6_13.py \
            -label ${label} \
            -cat ${stage2_label} \
            -y ${year_i} \
            --region ${region} \
            --base_path ${base_path} \
            -save ${plot_save_path}

        print_box "Step 6.4: Fig 6.19 (${year_i})"
        python validation/ggH/categorization/plot_6_19.py \
            -label ${label} \
            -cat ${stage2_label} \
            -y ${year_i} \
            --region ${region} \
            --base_path ${base_path} \
            -save ${plot_save_path}

        print_box "Step 6.5: Table 6.2 and 6.12 (${year_i})"
        python validation/ggH/categorization/getTable_6_2And6_12.py \
            -label ${label} \
            -cat ${stage2_label} \
            -y ${year_i} \
            --region ${region} \
            --base_path ${base_path} \
            -save ${plot_save_path}
    done
fi

# -----------------------------------------------------
# Step 7: Workspace
# -----------------------------------------------------
if [[ "${step}" == "7" || "${step}" == "dcall" || "${step}" == "all" || "${step}" == "bias" ]]; then
    for year_i in "${years[@]}"; do
        print_box "Step 7: Get workspace year: (${year_i})"
        stage3_label="${label}_X_${model_name}_${label_tag}"
        save_path="output/bdt_${model_name}_${model_trainYear}"
        mkdir -p "${save_path}"

        echo "stage2_save_path: ${stage2_save_path}"

        if [[ "${run_stage3_core_pdf_variants}" == "1" ]]; then
            bash "${stage3_variants_script}" \
                -load ${stage2_save_path} \
                -cat ${category} \
                --year ${year_i} \
                --label ${stage3_label} \
                -save ${save_path}
        else
            python run_stage3.py \
                -load ${stage2_save_path} \
                -cat ${category} \
                --year ${year_i} \
                --label ${stage3_label} \
                -save ${save_path}
        fi
    done
fi

# -----------------------------------------------------
# Step 8: Datacard
# -----------------------------------------------------
if [[ "${step}" == "8" || "${step}" == "dcall" || "${step}" == "all" ]]; then
    for year_i in "${years[@]}"; do
        print_box "Step 8: Obtain datacard year: (${year_i})"
        stage3_label="${label}_X_${model_name}_${label_tag}"
        stage3_output_label="$(resolve_stage3_output_label "${stage3_label}")"
        save_path="output/bdt_${model_name}_${model_trainYear}"
        mkdir -p "${save_path}"

        echo "stage2_save_path: ${stage2_save_path}"

        python stage2/ggH_datacard/generate_datacard.py \
            -load ${stage2_save_path} \
            -cat ${category} \
            --year ${year_i} \
            --label ${stage3_output_label} \
            -save ${save_path}

        echo "Copy the scripts to: ${save_path}/stage3/${year_i}/${stage3_output_label}/datacards/"
        cp scripts/get_significance.sh "${save_path}/stage3/${year_i}/${stage3_output_label}/datacards/"
        cp scripts/get_impactPlots.sh "${save_path}/stage3/${year_i}/${stage3_output_label}/datacards/"

        echo "Copy the background datacards to the same path"
        cp stage3/bkg_datacards_template/*bkg*.txt "${save_path}/stage3/${year_i}/${stage3_output_label}/datacards/"

        echo "Go to path given below and run"
        echo "cd ${save_path}/stage3/${year_i}/${stage3_output_label}/datacards/"
        echo "bash get_significance.sh"
        echo "bash get_impactPlots.sh"
    done
fi

# -----------------------------------------------------
# Step 9: Run Significance
# -----------------------------------------------------
if [[ "${step}" == "9" || "${step}" == "dcall" || "${step}" == "all" ]]; then
    for year_i in "${years[@]}"; do
        print_box "Step 9: Run significance for year: (${year_i})"
        stage3_label="${label}_X_${model_name}_${label_tag}"
        stage3_output_label="$(resolve_stage3_output_label "${stage3_label}")"
        save_path="output/bdt_${model_name}_${model_trainYear}"
        mkdir -p "${save_path}"

        echo "stage2_save_path: ${stage2_save_path}"
        echo "Go to path given below and run"

        cd "${save_path}/stage3/${year_i}/${stage3_output_label}/datacards/"
        bash get_significance.sh
        cd - >/dev/null
    done
fi

# -----------------------------------------------------
# Step 11: Bias test submission
# Requires the combined significance datacard from step 9.
# -----------------------------------------------------
if [[ "${step}" == "11" || "${step}" == "bias" || "${step}" == "all" ]]; then
    bias_template_dir="validation/ggH/bias_test/FuncCandidateVsCorePdfBias"

    for year_i in "${years[@]}"; do
        print_box "Step 11: Prepare and submit bias test for year: (${year_i})"
        stage3_label="${label}_X_${model_name}_${label_tag}"
        stage3_output_label="$(resolve_stage3_output_label "${stage3_label}")"
        save_path="output/bdt_${model_name}_${model_trainYear}"
        datacard_dir="${save_path}/stage3/${year_i}/${stage3_output_label}/datacards"
        bias_output_dir="${SCRIPT_DIR}/${save_path}/stage3/${year_i}/${stage3_output_label}/bias_test"
        bias_job_dir="${bias_output_dir}/FuncCandidateVsCorePdfBias"
        core_workspace_dir="${SCRIPT_DIR}/${datacard_dir}/my_workspace"
        fitfunc_workspace_dir="${bias_output_dir}/workspaces"
        combined_datacard="${SCRIPT_DIR}/${datacard_dir}/datacard_comb_sig_all_ggh.txt"

        mkdir -p "${save_path}"

        if [[ ! -f "${combined_datacard}" ]]; then
            echo "Missing combined datacard: ${combined_datacard}"
            echo "Run step 9 first so the significance workflow creates datacard_comb_sig_all_ggh.txt."
            exit 1
        fi

        if [[ ! -d "${core_workspace_dir}" ]]; then
            echo "Missing stage3 workspace directory: ${core_workspace_dir}"
            exit 1
        fi

        python validation/ggH/bias_test/run_bias_test.py \
            -load ${stage2_save_path} \
            -cat ${category} \
            --year ${year_i} \
            --label ${stage3_output_label} \
            -save ${save_path} \
            --max-chi2ndf ${bias_max_chi2ndf}

        if [[ ! -d "${fitfunc_workspace_dir}" ]]; then
            echo "Missing bias-test workspace directory: ${fitfunc_workspace_dir}"
            exit 1
        fi

        mkdir -p "${bias_job_dir}"
        mkdir -p "${bias_job_dir}/corePdf_workspace" "${bias_job_dir}/funcCandidate_workspace" "${bias_job_dir}/slurm_log"

        cp -f "${core_workspace_dir}/"*.root "${bias_job_dir}/corePdf_workspace/"
        cp -f "${fitfunc_workspace_dir}/"*.root "${bias_job_dir}/funcCandidate_workspace/"
        cp -f "${combined_datacard}" "${bias_job_dir}/datacard_comb_sig_all_ggh_corePdf.txt"

        sed 's#my_workspace/#funcCandidate_workspace/#g' "${bias_job_dir}/datacard_comb_sig_all_ggh_corePdf.txt" > "${bias_job_dir}/datacard_comb_sig_all_ggh_fitFuncCand.txt"
        sed -i.bak 's#my_workspace/#corePdf_workspace/#g' "${bias_job_dir}/datacard_comb_sig_all_ggh_corePdf.txt"
        rm -f "${bias_job_dir}/datacard_comb_sig_all_ggh_corePdf.txt.bak"
        if [[ -f "${bias_output_dir}/selected_truth_function_indices.txt" ]]; then
            cp -f "${bias_output_dir}/selected_truth_function_indices.txt" "${bias_job_dir}/"
        fi
        printf '%s\n' "${bias_expect_signal}" > "${bias_job_dir}/bias_truth_r.txt"

        BIAS_FITDIAG_PARALLEL="${bias_fitdiag_parallel}" \
        BIAS_ARRAY_TASKS="${bias_array_tasks}" \
        BIAS_EXPECT_SIGNAL="${bias_expect_signal}" \
        sh "${bias_template_dir}/slurm_wrapper.sh" "${bias_job_dir}" "${bias_run_mode}" "${bias_local_jobs}"

        echo "Bias jobs submitted from: ${bias_job_dir}"
        echo "After the Slurm jobs finish, run:"
        echo "bash ${SCRIPT_DIR}/stage2_sh.sh bias_collect ${year_i} ${n_trials} ${date_tag}"
    done
fi

# -----------------------------------------------------
# Step 12: Bias test aggregation
# Run this after the Slurm jobs from step 11 finish.
# -----------------------------------------------------
if [[ "${step}" == "12" || "${step}" == "bias_collect" ]]; then
    for year_i in "${years[@]}"; do
        print_box "Step 12: Aggregate bias test results for year: (${year_i})"
        stage3_label="${label}_X_${model_name}_${label_tag}"
        stage3_output_label="$(resolve_stage3_output_label "${stage3_label}")"
        save_path="output/bdt_${model_name}_${model_trainYear}"
        bias_job_dir="${SCRIPT_DIR}/${save_path}/stage3/${year_i}/${stage3_output_label}/bias_test/FuncCandidateVsCorePdfBias"

        if [[ ! -d "${bias_job_dir}" ]]; then
            echo "Missing bias job directory: ${bias_job_dir}"
            exit 1
        fi

        (
            cd "${bias_job_dir}"
            python "${SCRIPT_DIR}/validation/ggH/bias_test/FuncCandidateVsCorePdfBias/aggregate_bias_pull_fitDiagnostics.py"
        )
    done
fi

# -----------------------------------------------------
# Step 10: Run Impact
# -----------------------------------------------------
if [[ "${step}" == "10" || "${step}" == "im" || "${step}" == "all" ]]; then
    for year_i in "${years[@]}"; do
        print_box "Step 10: Run Impact for year: (${year_i})"
        stage3_label="${label}_X_${model_name}_${label_tag}"
        stage3_output_label="$(resolve_stage3_output_label "${stage3_label}")"
        save_path="output/bdt_${model_name}_${model_trainYear}"
        mkdir -p "${save_path}"

        echo "stage2_save_path: ${stage2_save_path}"
        echo "Go to path given below and run"

        cd "${save_path}/stage3/${year_i}/${stage3_output_label}/datacards/"
        bash get_impactPlots.sh
        cd - >/dev/null
    done
fi

print_box "Done"
echo "Finished at: $(date)"
echo "Run log: ${RUN_LOG}"
