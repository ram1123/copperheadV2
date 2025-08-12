#!/bin/bash
set -e
# run stage2 twice. First to generate BDT scores (we assume that an appropriate BDT is already trained, then generate score bin edges once more, then finally run stage2 again to save both bdt scores and ggH sub-category index

# sample_l="data ggh vbf" 
sample_l="data ggh vbf dy ewk tt st ww wz zz" 

# label="V2_Jan17_JecDefault_valerieZpt"
# label="V2_Jan29_JecOn_TrigMatchFixed_2016UlJetIdFix"
# label="UpdatedDY_100_200_CrossSection_24Feb_jetpuidOff"
# label="UpdatedDY_100_200_CrossSection_24Feb_jetpuidOff_newZptWgt25Mar2025"
# label="DYMiNNLO_jetpuidOff_newZptWgt25Mar2025"
# label="DYMiNNLO_30Mar2025"
# label="DYamcNLO_11Apr2025"
label="DYMiNNLO_11Apr2025"

stage2_load_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/stage1_output"

category="ggh"
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/$category/stage2_output" # I like to specify the category in the save path
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/20Mar2025_$category/stage2_output" # I like to specify the category in the save path



# model="V2_UL_Jan19_2025_addTTST_noVBF"
# model="V2_UL_Jan19_2025_addTTST"
# model="V2_UL_Jan19_2025_addTtStEwkVv"
# model="V2_UL_Jan19_2025_addEwkVv"
# model="V2_UL_Jan30_2025_DyGghVbf"
# model="V2_UL_Feb03_2025_DyTtStVvEwkGghVbf"
# model="V2_UL_Mar20_2025_DyTtStVvEwkGghVbf"


# model="V2_UL_Jan18_2025" # this is what I have been using for a long time

# model="V2_UL_Mar24_2025_DyTtStVvEwkGghVbf_scale_pos_weight"
# model="V2_UL_Mar25_2025_DyGghVbf_scale_pos_weight_newZpt"
# model="V2_UL_Mar26_2025_DyGghVbf_scale_pos_weight_dyMiNNLO"
# model="V2_UL_Mar26_2025_DyTtStVvEwkGghVbf_scale_pos_weight_dyMiNNLO"
# model="V2_UL_Mar30_2025_DyMiNNLOGghVbf"
# model="V2_UL_Mar30_2025_DyMiNNLOGghVbf_removeJetVar"
# model="V2_UL_Mar30_2025_DyMiNNLOGghVbf_removeAllJetVar"
# model="V2_UL_Mar30_2025_DyMiNNLOGghVbf_justDimuPt"
# model="V2_UL_Mar30_2025_DyMiNNLOGghVbf_onlyMuVar"
# model="V2_UL_Mar30_2025_DyMiNNLOGghVbf_onlyMuVar_ZeppenJjMass_DeltaVars"
# model="V2_UL_Mar30_2025_DyMiNNLOGghVbf_onlyMuVar_ZeppenJjMass_DeltaVars"
# model="V2_UL_Apr09_2025_DyMinnloTtStVvEwkGghVbf_hyperParamOnScaleWgt0_75"
# model="V2_UL_Apr11_2025_DyTtStVvEwkGghVbf"
# model="V2_UL_Apr11_2025_DyMinnloTtStVvEwkGghVbf"
model="V2_UL_Apr11_2025_DyMinnloTtStVvEwkGghVbf_bTagMediumFix"

stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}/stage2_output" 

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
# Begin stage3
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
stage3_label="${label}_X_${model}"

year="all"
python run_stage3.py -load $stage2_save_path -cat $category --year $year --label $stage3_label



