#!/bin/bash
# Stop execution on any error
set -e

# If there is `from src.<something> import <something>
# export PYTHONPATH=$PYTHONPATH:/depot/cms/users/shar1172/copperheadV2/

year="2018"
# year="2017"
# year="2016postVFP"
# year="2016preVFP"

NanoAODv=9

# data_l="A B C D E F G H"
data_l="A B C D"
bkg_l="dy_M-50"
sig_l=""
chunksize=300000
save_path="/depot/cms/users/$USER/hmm/copperhead_outputs_10Feb"

python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l  --NanoAODv $NanoAODv --use_gateway --log-level DEBUG

python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv --use_gateway --log-level DEBUG 
# python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv  --log-level DEBUG &> run_stage1_localCluster.log
