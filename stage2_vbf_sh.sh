#!/bin/bash


# model_label="Dec31_stage2_test"
model_label="test"


# stage1_label="V2_Dec22_HEMVetoOnZptOn_RerecoBtagSF_XS_Rereco_BtagWPsFixed"
# stage1_label="V2_Dec22_HEMVetoOnZptOn_ULBtagSF_XS_Rereco_BtagWPsFixed"
# stage1_label="test_test"
# stage1_label="V2_Jan09_test_test"
stage1_label="April19_NanoV12"

# stage1_path="/depot/cms/users/yun79/hmm/copperheadV1clean/${stage1_label}"
stage1_path="/depot/cms/users/shar1172/hmm/copperheadV1clean/${stage1_label}"
# model_path="/work/users/yun79/valerie/fork/copperheadV2/MVA_training/VBF/dnn/trained_models"
model_path="/depot/cms/private/users/shar1172/copperheadV2_MergeFW/MVA_training/VBF/dnn/trained_models"


year="2018"
data_samples=""
bkg_samples="TT"
sig_samples=""
# data_samples="A B C D"
# bkg_samples="DY TT"
# sig_samples="VBF GGH"
python run_stage2_vbf.py --model_path $model_path --model_label $model_label --run_label $stage1_path -y $year -data $data_samples -bkg $bkg_samples -sig $sig_samples --use_gateway 
# python run_stage2_vbf.py --model_path $model_path --model_label $model_label --run_label $stage1_path -y $year -data $data_samples -bkg $bkg_samples -sig $sig_samples