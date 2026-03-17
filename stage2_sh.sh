#!/bin/bash
set -euo pipefail

echo "$(date)"

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
label="Run3_nanoAODv12_FilterJetsHorn30GeV_Feb23_tightPassLepVeto_NoJER_AddVars_v2"
base_path="/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean"
stage2_load_path="${base_path}/${label}/stage1_output"

category="ggh"

# model_name="Run3_08March_HPeachFold_train_with_bestHP"
# model_name="Run3_09March_Check"
# model_name="Run3_10March_005Trials"
model_name="Run3_10March_020Trials"
# model_name="Run3_10March_020Trials_EBEinTraining" # Run3_10March_020Trials_EBEinTraining
# model_name="Run3_10March_020Trials_RemovedAddHyperPars"
# model_name="Run3_10March_100Trials"
# model_name="Run3_10March_020Trials_oneHotEncoding"
# model_name="Run3_10March_020Trials_oneHotEncoding_Scan"
# model_name="Run3_09Mar_HPScan_100Trials"
# model_name="Run3_09Mar_HPScan_500Trials"

step="${1:-}"
year="${2:-all}"

if [[ -z "${step}" ]]; then
    echo "Usage: $0 <step> <year|all>"
    echo "Example: $0 0 all"
    echo "Example: $0 4 2023"
    exit 1
fi

model_trainYear="all"
label_tag="AllYear_17March"

stage2_label="${model_name}_${category}_${label_tag}"
stage2_save_path="${base_path}/${label}/${stage2_label}/stage2_output"

bdt_edge_config_path="configs/MVA/ggH/BDT_edges.yaml"
mva_base_path="${PWD}"

# all_years=(2022preEE 2022postEE 2023 2023BPix 2024 all)
all_years=(2022preEE 2022postEE 2023 2023BPix 2024)
# all_years=(all)

if [[ "${year}" == "all" ]]; then
    years=("${all_years[@]}")
else
    years=("${year}")
fi

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
    echo "stage2_label     : ${stage2_label}"
    echo "base_path        : ${base_path}"
    echo "stage2_load_path : ${stage2_load_path}"
    echo "stage2_save_path : ${stage2_save_path}"
    echo "mva_base_path    : ${mva_base_path}"
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

# -----------------------------------------------------
# Start logging + print config
# -----------------------------------------------------
init_run_log
print_config

# -----------------------------------------------------
# Step 0: Train BDT
# Only run once, not looped over years here.
# -----------------------------------------------------
if [[ "${step}" == "0" ]]; then
    print_box "Step 0: Training the BDT for ggH"
    do_hyperparam_search="0" # true (enable hyperparameter search)
    n_trials="51" # It is for the bayseian optimization
    # mass_decorrelation_strat="default" # no mass decorrelation
    # mass_decorrelation_strat="peking" # peking's mass flattening
    mass_decorrelation_strat="targetZpeakMass" # target distribution Zpeak mass
    # mass_decorrelation_strat="targetHpeakMass" # target distribution Hpeak mass
    # mass_decorrelation_strat="targetHsidebandMass" # target distribution lower H sidebands and uppper ZCR mass window

    python MVA_training/ggH_BDT/my_trainer_withWeight_gpu.py \
        --name ${model_name} \
        --year ${year} \
        -load ${stage2_load_path} \
        -param_search ${do_hyperparam_search} \
        --n_trials ${n_trials} \
        --massDeCorrStrat ${mass_decorrelation_strat}
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
# FIXME: Keep these two files at the output directory too
#   1. /work/users/shar1172/copperheadV2_Feb2026/configs/MVA/ggH/BDT_edges.yaml
#   2. /work/users/shar1172/copperheadV2_Feb2026/stage2/ggH/target_yields.yaml
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
    local_years="all"
    save_path="output/bdt_${model_name}_${model_trainYear}/ggH/score_edge_generation/${local_years}"
    echo "save_path: ${save_path}"

    for year_i in "${years[@]}"; do
        save_path="output/bdt_${model_name}_${model_trainYear}/ggH/categorization/plots/${year_i}"
        mkdir -p "${save_path}"

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
                --edge_cfg_path "${save_path}/${stage2_label}" \
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
            -save ${save_path} \
            --mva_base_path ${mva_base_path}

        print_box "Step 6.2: Fig 6.8 (${year_i})"
        python validation/ggH/categorization/plot_6_8.py \
            -label ${label} \
            -cat ${stage2_label} \
            -y ${year_i} \
            --region ${region} \
            --base_path ${base_path} \
            -save ${save_path}

        print_box "Step 6.3: Fig 6.13 (${year_i})"
        python validation/ggH/categorization/plot_6_13.py \
            -label ${label} \
            -cat ${stage2_label} \
            -y ${year_i} \
            --region ${region} \
            --base_path ${base_path} \
            -save ${save_path}

        print_box "Step 6.4: Fig 6.19 (${year_i})"
        python validation/ggH/categorization/plot_6_19.py \
            -label ${label} \
            -cat ${stage2_label} \
            -y ${year_i} \
            --region ${region} \
            --base_path ${base_path} \
            -save ${save_path}

        print_box "Step 6.5: Table 6.2 and 6.12 (${year_i})"
        python validation/ggH/categorization/getTable_6_2And6_12.py \
            -label ${label} \
            -cat ${stage2_label} \
            -y ${year_i} \
            --region ${region} \
            --base_path ${base_path} \
            -save ${save_path}
    done
fi


# -----------------------------------------------------
# Step 7: Workspace
# -----------------------------------------------------
if [[ "${step}" == "7" || "${step}" == "dcall" || "${step}" == "all" ]]; then
    for year_i in "${years[@]}"; do
        print_box "Step 7: Get workspace year: (${year_i})"
        stage3_label="${label}_X_${model_name}_${label_tag}"
        save_path="output/bdt_${model_name}_${model_trainYear}"
        mkdir -p "${save_path}"

        echo "stage2_save_path: ${stage2_save_path}"

        python run_stage3.py \
            -load ${stage2_save_path} \
            -cat ${category} \
            --year ${year_i} \
            --label ${stage3_label} \
            -save ${save_path}
    done
fi

# -----------------------------------------------------
# Step 8: Datacard
# -----------------------------------------------------
if [[ "${step}" == "8" || "${step}" == "dcall" || "${step}" == "all" ]]; then
    for year_i in "${years[@]}"; do
        print_box "Step 8: Obtain datacard year: (${year_i})"
        stage3_label="${label}_X_${model_name}_${label_tag}"
        save_path="output/bdt_${model_name}_${model_trainYear}"
        mkdir -p "${save_path}"

        echo "stage2_save_path: ${stage2_save_path}"

        python stage2/ggH_datacard/generate_datacard.py \
            -load ${stage2_save_path} \
            -cat ${category} \
            --year ${year_i} \
            --label ${stage3_label} \
            -save ${save_path}

        echo "Copy the script: scripts/get_significance.sh to path: ${save_path}/stage3/${year_i}/${stage3_label}/datacards/"
        cp scripts/get_significance.sh ${save_path}/stage3/${year_i}/${stage3_label}/datacards/
        cp scripts/get_impactPlots.sh ${save_path}/stage3/${year_i}/${stage3_label}/datacards/

        echo "Copy the background datacards to the same path"
        cp stage3/bkg_datacards_template/*bkg*.txt ${save_path}/stage3/${year_i}/${stage3_label}/datacards/
        
        echo "Go to path given below and run"
        echo "cd ${save_path}/stage3/${year_i}/${stage3_label}/datacards/"
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
        save_path="output/bdt_${model_name}_${model_trainYear}"
        mkdir -p "${save_path}"

        echo "stage2_save_path: ${stage2_save_path}"
        echo "Go to path given below and run"
        cd ${save_path}/stage3/${year_i}/${stage3_label}/datacards/"
        bash get_significance.sh
        cd -
    done
    
fi

# -----------------------------------------------------
# Step 9: Run Significance
# -----------------------------------------------------
if [[ "${step}" == "10" || "${step}" == "im" || "${step}" == "all" ]]; then
    for year_i in "${years[@]}"; do
        print_box "Step 10: Run Impact for year: (${year_i})"
        stage3_label="${label}_X_${model_name}_${label_tag}"
        save_path="output/bdt_${model_name}_${model_trainYear}"
        mkdir -p "${save_path}"

        echo "stage2_save_path: ${stage2_save_path}"
        echo "Go to path given below and run"
        cd ${save_path}/stage3/${year_i}/${stage3_label}/datacards/"
        bash get_impactPlots.sh
    done
    
fi

print_box "Done"
echo "Finished at: $(date)"
echo "Run log: ${RUN_LOG}"
