#!/bin/bash

# run stage2 twice. First to generate BDT scores (we assume that an appropriate BDT is already trained, then generate score bin edges once more, then finally run stage2 again to save both bdt scores and ggH sub-category index

# year="2017"
# year="2016postVFP"
# year="2016preVFP"


sample_l="data ggh vbf" 
stage2_load_path="/depot/cms/users/yun79/hmm/copperheadV1clean/V2_Jan17_JecDefault_valerieZpt/stage1_output"

category="ggh"
stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/V2_Jan17_JecDefault_valerieZpt/$category/stage2_output" # I like to specify the category in the save path

model="V2_UL_Jan18_2025"


# year="2018"
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
# python stage2/ggH/calculate_score_edges.py -load $stage2_save_path --year $year 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model

# year="2017"
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
# python stage2/ggH/calculate_score_edges.py -load $stage2_save_path --year $year 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model


year="2016postVFP"
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
python stage2/ggH/calculate_score_edges.py -load $stage2_save_path --year $year 
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model

# year="2016preVFP"
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
# python stage2/ggH/calculate_score_edges.py -load $stage2_save_path --year $year 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
