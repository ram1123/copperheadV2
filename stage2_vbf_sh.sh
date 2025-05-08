#!/bin/bash


model_label="Dec31_stage2_test"
# stage1_lable="V2_Dec22_HEMVetoOnZptOn_RerecoBtagSF_XS_Rereco_BtagWPsFixed"
# stage1_lable="test_test"
stage1_lable="V2_Jan09_test_test"

year="2018"
data_samples=""
bkg_samples="DY"
sig_samples=""
# data_samples="A B C D"
# bkg_samples="DY TT"
# sig_samples="VBF GGH"
python run_stage2_vbf.py --model_label $model_label --run_label $stage1_lable -y $year -data $data_samples -bkg $bkg_samples -sig $sig_samples --use_gateway 
# python run_stage2_vbf.py --model_label $model_label --run_label $stage1_lable -y $year -data $data_samples -bkg $bkg_samples -sig $sig_samples