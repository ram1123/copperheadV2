#!/bin/bash

year="2018"
# year="2017"
# year="2016postVFP"
# year="2016preVFP"


sample_l="data ggh vbf" 
stage2_load_path="/depot/cms/users/yun79/hmm/copperheadV1clean/V2_Jan16_JecDefault_plotEveryonesZptWgt/"
stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/V2_Jan16_JecDefault_plotEveryonesZptWgt/ggh/stage2_output" 
category="ggh"
model="V2_UL_Jan18_2025"
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
