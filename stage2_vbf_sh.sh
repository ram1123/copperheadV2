#!/bin/bash


model_label="Dec31_stage2_test"
stage1_lable="V2_Dec22_HEMVetoOnZptOn_RerecoBtagSF_XS_Rereco_BtagWPsFixed"
year="2018"
python run_stage2_vbf.py --model_label $model_label --run_label $stage1_lable -y $year --use_gateway 
# python run_stage2_vbf.py --model_label $model_label --run_label $stage1_lable -y $year