#!/bin/bash
set -e
# we assume that the workflow stage1, stage2 have been already run


label="Run3_nanoAODv12_02Feb_FilterJetsHorn30GeV"

category="ggh"

model_name="Run3PrelimResultsFeb09_2026_jecjer"

stage2_save_path="/depot/cms/hmm/yun79/hmm_ntuples/copperheadV1clean/$label/${model_name}_${category}_perYr/stage2_output" 


# # ------------------
# # Begin stage3
# # ------------------

# stage3_label="${label}_X_${model_name}_all7FitFuncs"
stage3_label="${label}_X_${model_name}_allFitFuncCandidates"

year="all"
python run_bias_test.py -load $stage2_save_path -cat $category --year $year --label $stage3_label


