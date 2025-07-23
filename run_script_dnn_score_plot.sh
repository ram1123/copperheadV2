#!/bin/bash
set -e


# label="fullRun_Jun23_2025_1n2Revised"
# label="Run2_nanoAODv12_08June"
label="Run2_nanoAODv12_UpdatedQGL_17July"

category="vbf"
# mvaName="fullRun_Jun23_2025_1n2Revised_2018_VBFOnly"
# mvaName="Run2_nanoAODv12_08June_2018"
mvaName="Run2_nanoAODv12_UpdatedQGL_17July"


year="2018"
lumi="59.83"

# year="2017"
# lumi="41.48"

# year="2016postVFP"
# lumi="19.50"

# year="2016preVFP"
# lumi="16.81"

# year="all"
# lumi="137"

# plot stage1 output
region="h-sidebands"
python plotter/plot_DNN_score.py -label $label -cat $category -y ${year} --region ${region} --mva_name ${mvaName} --lumi ${lumi}
region="h-peak"
python plotter/plot_DNN_score.py -label $label -cat $category -y ${year} --region ${region} --mva_name ${mvaName} --lumi ${lumi}
