#!/bin/bash
# Stop execution on any error
set -e

# year="2018"
# year="2017"
# year="2016postVFP"
# year="2016preVFP"

# data_l="A B C D E F G H"
# data_l="C"
bkg_l="DY Top VV EWK Higgs VVV"
# bkg_l="Top"
# bkg_l="TT ST VV OTHER"
sig_l="Higgs"
data_l=""
# bkg_l=""
# sig_l=""
chunksize=300000
# chunksize=50000
# save_path="/depot/cms/users/shar1172/hmm/copperheadV1clean/March25_NanoAODv9_WithUpdatedZptWgt/" # FIXME: ReRun with 2017B
# save_path="/depot/cms/users/shar1172/hmm/copperheadV1clean/TestCutFlow/"
# save_path="/depot/cms/users/shar1172/hmm/copperheadV1clean/April09_NanoV09/"

# year="2016postVFP"
# year="2016preVFP"
# year="2018"
# NanoAODv=9
# datasetYAML="configs/datasets/dataset_nanoAODv9.yaml"
# python run_prestage.py --chunksize $chunksize -y $year --yaml $datasetYAML --data $data_l --background $bkg_l --signal $sig_l --NanoAODv $NanoAODv --log-level DEBUG --xcache
# python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv  --use_gateway --log-level DEBUG
# python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv

year="2018"
NanoAODv=12
save_path="/depot/cms/users/shar1172/hmm/copperheadV1clean/April09_NanoV12/"
datasetYAML="configs/datasets/dataset_nanoAODv12.yaml"
python run_prestage.py --chunksize $chunksize -y $year --yaml $datasetYAML --data $data_l --background $bkg_l --signal $sig_l --NanoAODv $NanoAODv --log-level INFO
# python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv  --use_gateway --log-level DEBUG

# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/V2_Jan29_JecOn_TrigMatchFixed_2016UlJetIdFix/"
# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/test_test/"
# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/stage1_performance_test/"
# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/UpdatedDY_100_200_CrossSection_24Feb/"
# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/VBFFilter_DY_validation_2025Mar14/zpt_rewgt_params.yaml"


# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/UpdatedDY_100_200_CrossSection_24Feb_jetpuidOff/"
# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/UpdatedDY_100_200_CrossSection_24Feb_jetpuidOff_newZptWgt25Mar2025/"
# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/DYMiNNLO_jetpuidOff_newZptWgt25Mar2025/"


# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/NanoV12_01April2025/"

# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/HemVetoStudy_04Apr2025/"

# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/DYMiNNLO_30Mar2025/"

# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/DYMiNNLO_11Apr2025/"
# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/DYamcNLO_11Apr2025/"
# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/DYMiNNLO_HemVetoOff_17Apr2025/"


# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/DYMiNNLO_HemVetoOff_18Apr2025_singleMuTrigMatch/"

# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/cutflow_18Apr2025_singleMuTrigMatch/"
# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/test/"
# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/cutflow_27Apr2025_matchRereco/"
# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/jetHornStudy_29Apr2025_JecOnJerOff/"
# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/jetHornStudy_29Apr2025_JecOnJerOn/"
# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/jetHornStudy_29Apr2025_JecOnJerOn_tightJetPuId/"
# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/jetHornStudy_29Apr2025_JecOnJerStrat1/"
# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/jetHornStudy_29Apr2025_JecOnJerStrat2/"
# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/jetHornStudy_29Apr2025_JecOnJerStrat2_jetHornPtCut50/"
# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/jetHornStudy_29Apr2025_JecOnJerStrat2_jetHornPtCut30_jetHornTightPuId/"
# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/jetHornStudy_29Apr2025_JecOnJerStrat2_jetHornPtCut30/"
# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/jetHornStudy_29Apr2025_JecOnJerOff_jetHornPtCut30/"
# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/jetHornStudy_29Apr2025_JecOnJerStrat1_jetHornPtCut30/"
save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/jetHornStudy_29Apr2025_JecOnJerStrat1n2_jetHornPtCut30/"


# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/test_test/"


year="2018"
# year="2018_RERECO"
# year="2018_V12"

# python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l #--use_gateway # --skipBadFiles
# python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l -frac 0.2
# python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l -frac 1.0


# bkg_l="DY ST"
# year="2016postVFP"
# python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l
# NanoAODv=9
# python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv  --use_gateway
# python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv  --use_gateway
# python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv  --use_gateway
# python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv  --use_gateway

