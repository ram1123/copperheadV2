#!/bin/bash
# set -e


# first rederive the SF root files for fitting
year="2018"
label="UpdatedDY_100_200_CrossSection_24Feb_jetpuidOff"

python save_SF_rootFiles.py -l ${label} -y ${year}