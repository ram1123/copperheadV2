#!/bin/bash
# Stop execution on any error
# set -e

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

# year="2017"
# python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l
# NanoAODv=9
# python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv  --use_gateway

# bkg_l="DY ST"
# year="2016postVFP"
# python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l
# NanoAODv=9
# python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv  --use_gateway
# python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv  --use_gateway
# python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv  --use_gateway
# python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv  --use_gateway

# year="2016preVFP"
# python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l
# NanoAODv=9
# python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv --use_gateway
