#!/bin/bash
# Stop execution on any error
# set -e

# year="2018"
# year="2017"
# year="2016postVFP"
# year="2016preVFP"

# data_l="A B C D E F G H"
data_l=""
bkg_l="DY"
sig_l="ggH VBF"
chunksize=300000
# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/V2_Jan16_JecDefault_plotEveryonesZptWgt/"
# save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/V2_Jan17_JecDefault_plotEveryonesZptWgt/"
save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/V2_Jan17_JecDefault_valerieZpt/"

# year="2018"
# python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l
# NanoAODv=9
# python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv  --use_gateway

# year="2017"
# python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l
# NanoAODv=9
# python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv  --use_gateway

# bkg_l="DY ST"
year="2016postVFP"
# python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l
NanoAODv=9
python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv  --use_gateway
python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv  --use_gateway
python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv  --use_gateway
python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv  --use_gateway

# year="2016preVFP"
# python run_prestage.py --chunksize $chunksize -y $year --data $data_l --background $bkg_l --signal $sig_l
# NanoAODv=9
# python -W ignore run_stage1.py -y $year --save_path $save_path --NanoAODv $NanoAODv --use_gateway
