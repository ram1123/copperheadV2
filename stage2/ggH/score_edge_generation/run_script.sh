#!/bin/bash
set -e

# label="V2_Jan29_JecOn_TrigMatchFixed_2016UlJetIdFix"
# label="DYMiNNLO_30Mar2025"
# label="DYamcNLO_11Apr2025"
label="DYMiNNLO_11Apr2025"

sample_l="data ggh vbf" 

category="ggh"
# model="V2_UL_Apr09_2025_DyMinnloTtStVvEwkGghVbf_hyperParamOnScaleWgt0_75"
# model="V2_UL_Apr11_2025_DyTtStVvEwkGghVbf"
model="V2_UL_Apr11_2025_DyMinnloTtStVvEwkGghVbf"


# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/$category/stage2_output" 
stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}/stage2_output"  # I like to specify the category in the save path


years="2016preVFP 2016postVFP 2017 2018"
# years="2017"
python determine_score_edge.py -load $stage2_save_path --years ${years}