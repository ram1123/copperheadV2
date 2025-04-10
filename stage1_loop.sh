#!/bin/bash
# Stop execution on any error
# set -e

# year="2018"
# year="2017"
# year="2016postVFP"
# year="2016preVFP"

# data_l="A B C D E F G H"
# data_l="C"
# bkg_l="DY TT ST VV OTHER"
bkg_l="DY"
# bkg_l="TT ST VV OTHER"
# sig_l="ggH VBF"
data_l=""
# bkg_l=""
sig_l=""
chunksize=300000
# chunksize=50000
# save_path="/depot/cms/users/shar1172/hmm/copperheadV1clean/March25_NanoAODv9_WithUpdatedZptWgt/" # FIXME: ReRun with 2017B
# save_path="/depot/cms/users/shar1172/hmm/copperheadV1clean/TestCutFlow/"
save_path="/depot/cms/users/shar1172/hmm/copperheadV1clean/April09_NanoV12/" # FIXME: ReRun with 2017B

# year="2016postVFP"
# year="2016preVFP"
year="2018"
# NanoAODv=9
# python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l --xcache
# python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l -frac 0.10

NanoAODv=12
datasetYAML="configs/datasets/dataset_nanoAODv12.yaml"
# python run_prestage.py --chunksize $chunksize -y $year --yaml $datasetYAML --data $data_l --background $bkg_l --signal $sig_l --NanoAODv $NanoAODv --log-level DEBUG
python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv  --use_gateway --log-level DEBUG
# python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv

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
