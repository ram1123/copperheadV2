#!/bin/bash
set -e
# run stage2 twice. First to generate BDT scores (we assume that an appropriate BDT is already trained, then generate score bin edges once more, then finally run stage2 again to save both bdt scores and ggH sub-category index

# sample_l="data dy ewk tt st ww wz zz" 
sample_l="data ggh vbf dy ewk tt st ww wz zz" 

label="V2_Jan29_JecOn_TrigMatchFixed_2016UlJetIdFix"
stage2_load_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/stage1_output"

category="20Mar2025_ggh"


python validation_plot.py -label $label -cat $category --samples $sample_l