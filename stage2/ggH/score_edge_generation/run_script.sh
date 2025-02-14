#!/bin/bash
set -e

label="V2_Jan29_JecOn_TrigMatchFixed_2016UlJetIdFix"
sample_l="data ggh vbf" 
stage2_load_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/stage1_output"

category="ggh"
stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/$category/stage2_output" # I like to specify the category in the save path

model="V2_UL_Jan18_2025"

year="2018"
python determine_score_edge.py -load $stage2_save_path --year $year 