#!/bin/bash
set -e
# run stage2 twice. First to generate BDT scores (we assume that an appropriate BDT is already trained, then generate score bin edges once more, then finally run stage2 again to save both bdt scores and ggH sub-category index

sample_l="data ggh vbf" 

# label="V2_Jan17_JecDefault_valerieZpt"
label="V2_Jan29_JecOn_TrigMatchFixed_2016UlJetIdFix"
stage2_load_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/stage1_output"

category="ggh"
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/$category/stage2_output" # I like to specify the category in the save path
stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/20Mar2025_$category/stage2_output" # I like to specify the category in the save path


model="V2_UL_Jan18_2025"
# model="V2_UL_Jan19_2025_addTTST_noVBF"
# model="V2_UL_Jan19_2025_addTTST"
# model="V2_UL_Jan19_2025_addTtStEwkVv"
# model="V2_UL_Jan19_2025_addEwkVv"
# model="V2_UL_Jan30_2025_DyGghVbf"
# model="V2_UL_Feb03_2025_DyTtStVvEwkGghVbf"

year="2018"
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
python stage2/ggH/calculate_score_edges.py -load $stage2_save_path --year $year 
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model

year="2017"
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
python stage2/ggH/calculate_score_edges.py -load $stage2_save_path --year $year 
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model


year="2016postVFP"
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
python stage2/ggH/calculate_score_edges.py -load $stage2_save_path --year $year 
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model

year="2016preVFP"
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
python stage2/ggH/calculate_score_edges.py -load $stage2_save_path --year $year 
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model

# ------------------

# # stage3_label="${label}_X_${model}"
# # stage3_label="${label}_X_${model}_matchUCSD_values"
# # stage3_label="test"
# stage3_label="simple_fit_test_16Mar2025"
# # stage3_label="${label}_X_${model}_ucsdFitFuncs_newBinEdges"
# # stage3_label="${label}_X_${model}_Feb05_coreFuncFixed_newBinEdges"
# # stage3_label="${label}_X_${model}_Feb09_newBinEdges_bySig"
# # stage3_label="${label}_X_${model}_Feb11_newBinEdges_bySig_useRooDoubleCBFast_hPeakSigFit"
# # stage3_label="${label}_X_${model}_Feb15_newBinEdges"
# # stage3_label="${label}_X_${model}_Feb16_testBinEdges"
# # stage3_label="${label}_X_${model}_Feb16_testBinEdges2"
# # stage3_label="${label}_X_${model}_Feb19_fullSigFitRange"


# year="all"
# python run_stage3.py -load $stage2_save_path -cat $category --year $year --label $stage3_label



