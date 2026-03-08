#!/bin/bash
set -e
# run stage2 twice. First to generate BDT scores (we assume that an appropriate BDT is already trained, then generate score bin edges once more, then finally run stage2 again to save both bdt scores and ggH sub-category index

label="Run3_nanoAODv12_FilterJetsHorn30GeV_Feb23_tightPassLepVeto_NoJER_AddVars_v2"
base_path="/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean"
stage2_load_path="/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/$label/stage1_output"

category="ggh"


model_name="Run3_04March_test"

stage2_label="${model_name}_${category}_perYr"
stage2_save_path="/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/${label}/${stage2_label}/stage2_output"

bdt_edge_config_path="configs/MVA/ggH/BDT_edges.yaml"

year="2022postEE"
# year="2022preEE"
# year="2023"
# year="2023BPix"

mva_base_path="/work/users/yun79/Run2_MVA_trainer"

step=$1

print_box() {
    local msg="$1"
    local len=${#msg}
    local border=$(printf '%*s' "$((len + 4))" '' | tr ' ' '=')

    echo "$border"
    echo "| $msg |"
    echo "$border"
}

# -----------------------------------------------------
# Train BDT
# -----------------------------------------------------
if [[ "$step" == "0" ]]; then
    print_box "Training the BDT for ggH"
    do_hyperparam_search="0" # false
    # mass_decorrelation_strat="default" # no mass decorrelation
    # mass_decorrelation_strat="peking" # peking's mass flattening
    # mass_decorrelation_strat="targetZpeakMass" # target distribution Zpeak mass
    mass_decorrelation_strat="targetHpeakMass" # target distribution Hpeak mass
    # mass_decorrelation_strat="targetHsidebandMass" # target distribution lower H sidebands and uppper ZCR mass window

    python MVA_training/ggH_BDT/my_trainer_withWeight_gpu.py --name ${model_name} --year ${year} -load  ${stage2_load_path} -param_search ${do_hyperparam_search} --massDeCorrStrat ${mass_decorrelation_strat}
fi



# -----------------------------------------------------
# for BDT edge calculation
# -----------------------------------------------------
if [[ "$step" == "1"|| ""$step"" == "all" ]]; then
    print_box "Step: 1: Apply selection and add the BDT score branch to the ouput parquet file."
    sample_l="ggh vbf data"
    python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model_name --mva_base_path $mva_base_path
fi



# -----------------------------------------------------
# Obtain the target yield
# -----------------------------------------------------
# NOTE: when calculating over multiple years, you need to list them all in --years argument or use "all" option
if [[ "$step" == "2"|| ""$step"" == "all" ]]; then
    print_box "Step: 2: Obtain the target yield"
    save_path="stage2/ggH/score_edge_generation/"
    python stage2/ggH/score_edge_generation/determine_score_edge.py -load $stage2_save_path --years ${year} --label ${stage2_label} -save ${save_path}
    python stage2/ggH/score_edge_generation/validation.py --label ${stage2_label} -save ${save_path}
    # NOTE: Above step will print the **target yield** on the terminal.
    # Then manually write the target yield to file
    # [../stage2/ggH/target_yields.yaml](../stage2/ggH/target_yields.yaml),
    # in the ouput path.
    #  04 MARCH 2026: UPDATED THIS. NOW IT WRITE AUTOMATICALLY.
fi



# -----------------------------------------------------
# Obtain the BDT edges
# -----------------------------------------------------
if [[ "$step" == "3" || ""$step"" == "all" ]]; then
    print_box "Step: 3: Obtain the BDT score edges for categorization"
    sample_l="ggh vbf"
    python stage2/ggH/calculate_score_edges.py -load $stage2_save_path --year $year --edge_cfg_path ${bdt_edge_config_path}
fi

if [[ "$step" == "4" || ""$step"" == "all" ]]; then
    print_box "Step: 4: Run stage-2 again to save the BDT score and sub-category index in the output parquet file."
    # sample_l="data ggh vbf dy ewk tt ww wz zz other"
    sample_l="data ggh vbf dy  tt ww wz"
    python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model_name --mva_base_path $mva_base_path
fi

if [[ "$step" == "5" || ""$step"" == "all" ]]; then
    print_box "Step: 5: Obtain the BDT score plot"
    sample_l="data ggh vbf dy tt ww wz"
    save_path="validation/ggH/categorization/plots"
    region="h-sidebands"
    python validation/ggH/categorization/validation_plot.py -label $label -cat $stage2_label --samples $sample_l -y $year --region ${region} --base_path ${base_path} -save ${save_path}

    print_box "Step: 5.5: Obtain few more validation plots"
    sample_l="ggh vbf dy tt ww wz"
    # Step-a: Get the files
    python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model_name --do_6p7 --mva_base_path $mva_base_path

    # Step-b: Get the plot
    region="signal"
    bdt_year=${year}
    python validation/ggH/categorization/plot_6_7.py -label $label -cat $stage2_label -y ${year} --region ${region} --base_path ${base_path} --model_name ${model_name} --bdt_year ${bdt_year} -save ${save_path} --mva_base_path $mva_base_path
    python validation/ggH/categorization/plot_6_13.py -label $label -cat $stage2_label -y ${year} --region ${region} --base_path ${base_path} -save ${save_path}
    python validation/ggH/categorization/plot_6_19.py -label $label -cat $stage2_label -y ${year} --region ${region} --base_path ${base_path} -save ${save_path}
    python validation/ggH/categorization/plot_6_8.py -label $label -cat $stage2_label -y ${year} --region ${region} --base_path ${base_path} -save ${save_path}

    # ERROR: 
    python validation/ggH/categorization/getTable_6_2And6_12.py -label $label -cat $stage2_label -y ${year} --region ${region} --base_path ${base_path} -save ${save_path}
fi

if [[ "$step" == "6" || ""$step"" == "all" ]]; then
    print_box "Step: 6: Get the workspace"
    stage3_label="${label}_X_${model_name}_perYr"
    echo "stage2_save_path: ${stage2_save_path}"
    python run_stage3.py -load $stage2_save_path -cat $category --year $year --label $stage3_label
fi


if [[ "$step" == "7" || ""$step"" == "all" ]]; then
    print_box "Step: 7: Obtain the datacard"
    stage3_label="${label}_X_${model_name}_perYr"
    echo "stage2_save_path: ${stage2_save_path}"
    python stage2/ggH_datacard/generate_datacard.py -load $stage2_save_path -cat $category --year $year --label $stage3_label
fi
