#!/bin/bash
set -e
# run stage2 twice. First to generate BDT scores (we assume that an appropriate BDT is already trained, then generate score bin edges once more, then finally run stage2 again to save both bdt scores and ggH sub-category index

label="Run3_nanoAODv15_24Jan2025"
stage2_load_path="/depot/cms/hmm/yun79/hmm_ntuples/copperheadV1clean/$label/stage1_output"




category="ggh"


model="Run3PrelimResultsJan25_2026_NoAnnhilateWgts"

# stage2_save_path="/depot/cms/hmm/yun79/hmm_ntuples/copperheadV1clean/$label/${model}_${category}/stage2_output" 
stage2_save_path="/depot/cms/hmm/yun79/hmm_ntuples/copperheadV1clean/$label/${model}_${category}_memoryRefactor/stage2_output" 


bdt_edge_config_path="/work/users/yun79/sideHustle2/copperheadV2/configs/MVA/ggH/BDT_edges.yaml"


# # -----------------------------------------------------
# for BDT edge calculation
# year="2024"
# # sample_l="ggh vbf dy ewk tt st ww wz zz other" 
# sample_l="ggh vbf data" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model

# year="2023BPix"
# # sample_l="ggh vbf dy ewk tt st ww wz zz other" 
# sample_l="ggh vbf data" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model

# year="2023"
# # sample_l="ggh vbf dy ewk tt st ww wz zz other" 
# sample_l="ggh vbf data" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model

# year="2022postEE"
# # sample_l="ggh vbf dy ewk tt st ww wz zz other" 
# sample_l="ggh vbf data" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model

# year="2022preEE"
# # sample_l="ggh vbf dy ewk tt st ww wz zz other" 
# sample_l="ggh vbf data" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
# #------------------------------------------------------

# year="2024"
# sample_l="ggh vbf" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
# python stage2/ggH/calculate_score_edges.py -load $stage2_save_path --year $year --edge_cfg_path ${bdt_edge_config_path}
# sample_l="data ggh vbf dy ewk tt ww wz zz other" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model

# year="2023BPix"
# sample_l="ggh vbf" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
# python stage2/ggH/calculate_score_edges.py -load $stage2_save_path --year $year --edge_cfg_path ${bdt_edge_config_path}
# sample_l="data ggh vbf dy ewk tt ww wz zz other" 
# # sample_l="data ggh vbf" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model


# year="2023"
# sample_l="ggh vbf" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
# python stage2/ggH/calculate_score_edges.py -load $stage2_save_path --year $year --edge_cfg_path ${bdt_edge_config_path}
# sample_l="data ggh vbf dy ewk tt ww wz zz other" 
# # sample_l="data ggh vbf" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model

# year="2022postEE"
# sample_l="ggh vbf" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
# python stage2/ggH/calculate_score_edges.py -load $stage2_save_path --year $year --edge_cfg_path ${bdt_edge_config_path}
# sample_l="data ggh vbf dy ewk tt ww wz zz other" 
# # sample_l="data ggh vbf" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model

# year="2022preEE"
# sample_l="ggh vbf" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
# python stage2/ggH/calculate_score_edges.py -load $stage2_save_path --year $year --edge_cfg_path ${bdt_edge_config_path}
# sample_l="data ggh vbf dy ewk tt ww wz zz other" 
# # sample_l="data ggh vbf" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model

# ------------------
# stage2 specifically for fig 6.7
# ------------------
# year="2024"
# sample_l="ggh vbf dy ewk tt"  
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model --do_6p7 # --do_jecUnc
# year="2023BPix"
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model --do_6p7 # --do_jecUnc
# year="2023"
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model --do_6p7 # --do_jecUnc
# year="2022postEE"
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model --do_6p7 # --do_jecUnc
# year="2022preEE"
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model --do_6p7 # --do_jecUnc



# ------------------
# Begin stage3
# ------------------


# stage3_label="${label}_X_${model}_w_allYearBDT_w_newBDTTargetYields_recreateOct24Run2_2025"
# stage3_label="${label}_X_${model}_w_allYearBDT_w_newBDTTargetYields_recreate1_87SigOct28_2025"
# stage3_label="${label}_X_${model}_w_allYearBDT_w_newBDTTargetYields_recreate1_87SigOct28_2025_BDTJune23Recreated_Run2"
# stage3_label="${label}_X_${model}_recreate1_87SigOct29_2025_BDT_Run3"
# stage3_label="${label}_X_${model}_recreate1_87SigOct29_2025_BDT_Run3_repeat"
# stage3_label="${label}_X_${model}_recreate1_87SigOct29_2025_BDT_Run3_recalculateBDTEdges"
# stage3_label="${label}_X_${model}_recreate1_87SigOct29_2025_BDT_Run3_recalculateTargYieldNBDTEdges"
# stage3_label="${label}_X_${model}_recreate1_87SigOct31_2025_BDT_AddRpt_recalculateTargYieldNBDTEdges"
# stage3_label="${label}_X_${model}_recreate1_87_test"
# stage3_label="${label}_X_${model}_recreate2_00SigNov11_2015"
# stage3_label="${label}_X_${model}"
# stage3_label="${label}_X_${model}_diffTargetYield"
# stage3_label="${label}_X_${model}_repeat"
stage3_label="${label}_X_${model}_memoryRefactor"

echo "stage2 path: ${stage2_save_path}"
# year="all"
year="2024"
python run_stage3.py -load $stage2_save_path -cat $category --year $year --label $stage3_label

# year="2018"
# python run_stage3.py -load $stage2_save_path -cat $category --year $year --label $stage3_label
# year="2017"
# python run_stage3.py -load $stage2_save_path -cat $category --year $year --label $stage3_label
# year="2016postVFP"
# python run_stage3.py -load $stage2_save_path -cat $category --year $year --label $stage3_label
# year="2016preVFP"
# python run_stage3.py -load $stage2_save_path -cat $category --year $year --label $stage3_label