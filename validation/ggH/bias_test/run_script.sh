#!/bin/bash
set -e
# we assume that the workflow stage1, stage2 have been already run



category="ggh"

label="Run3_nanoAODv15_28Feb_JetsHorn30GeV_NoJer_tightPassLepVeto"
model_name="Mar06_2026_zPeakShapeMatch_negWgtPairAnnhilate_tuned"


# stage2_save_path="/depot/cms/hmm/yun79/hmm_ntuples/copperheadV1clean/$label/${model_name}_${category}_allYr_redo/stage2_output"
label="Run3_nanoAODv12_FilterJetsHorn30GeV_Feb23_tightPassLepVeto_NoJER_AddVars_v2"
base_path="/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean"
stage2_save_path="${base_path}/${label}/Run3_10March_020Trials_ggh_AllYear/stage2_output"


# # ------------------
# # Begin stage3
# # ------------------

# stage3_label="${label}_X_${model_name}_all7FitFuncs"
stage3_label="${label}_X_${model_name}_allFitFuncCandidates"

year="all"
python run_bias_test.py -load $stage2_save_path -cat $category --year $year --label $stage3_label


