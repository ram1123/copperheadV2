#!/bin/bash
# set -e


# first rederive the SF root files for fitting
year="2016preVFP"
# year="all"
# label="UpdatedDY_100_200_CrossSection_24Feb_jetpuidOff"
label="DYMiNNLO_30Mar2025"

# python save_SF_rootFiles.py -l ${label} -y ${year} # --use_gateway

# do F test
# python do_f_test.py

# year="2018"
python get_polyFit.py -l ${label} -y ${year} #--save_config