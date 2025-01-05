#!/bin/bash


model_label="Dec31_stage2_test"
stage1_lable="V2_Dec22_HEMVetoOnZptOn_RerecoBtagSF_XS_Rereco_BtagWPsFixed"
year="2018"
# data_samples=""
# bkg_samples=""
data_samples="A B C D"
bkg_samples="DY TT"
sig_samples="VBF GGH"
# python run_stage2_vbf.py --model_label $model_label --run_label $stage1_lable -y $year -data $data_samples -bkg $bkg_samples -sig $sig_samples --use_gateway 
python run_stage2_vbf.py --model_label $model_label --run_label $stage1_lable -y $year -data $data_samples -bkg $bkg_samples -sig $sig_samples