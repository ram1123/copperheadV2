#!/bin/bash
set -e

label="V2_Jan29_JecOn_TrigMatchFixed_2016UlJetIdFix"
sample_l="data ggh vbf" 

category="ggh"
stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/$category/stage2_output" # I like to specify the category in the save path


python determine_score_edge.py -load $stage2_save_path 