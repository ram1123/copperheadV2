#!/bin/bash
# Stop execution on any error
set -e

# This script is a wrapper for plotting skimmed variables after stage1 step is complete.

data_l="A B C D E F G H"
bkg_l="DY TT ST VV EWK OTHER"
sig_l="ggH VBF"

status="Private_Work"

# this is label you used to save your stage1 outputs
label="jetHornStudy_29Apr2025_JecOnJerStrat2"
username="yun79"

# Lumi list for ref -----------------------------------------
#
#     "2018" : 59.97,
#     "2017" : 41.5,
#     "2016postVFP": 19.5,
#     "2016preVFP": 16.8,
#     "2022preEE" : None,
# -----------------------------------------------------------



year="2018"
lumi="59.97"

# year="2017"
# lumi="41.5"


# thsi is the full path where stage1 is saved
load_path="/depot/cms/users/${username}/hmm/copperheadV1clean/${label}/stage1_output/${year}/f1_0/"


# vars2plot="jet dijet dimuon mu"
vars2plot="jet"

region="z-peak"

python validation_plotter_unified.py -y $year --load_path $load_path -var $vars2plot --data $data_l --background $bkg_l --signal $sig_l --lumi $lumi --status $status -cat nocat -reg $region --label $label 

region="signal"
python validation_plotter_unified.py -y $year --load_path $load_path -var $vars2plot --data $data_l --background $bkg_l --signal $sig_l --lumi $lumi --status $status -cat ggh -reg $region --label $label 
