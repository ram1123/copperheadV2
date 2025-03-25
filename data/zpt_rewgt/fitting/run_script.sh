#!/bin/bash
# set -e


# first rederive the SF root files for fitting
year="2018"
# year="all"
label="UpdatedDY_100_200_CrossSection_24Feb_jetpuidOff"

# python save_SF_rootFiles.py -l ${label} -y ${year}

# do F test
# python do_f_test.py

year="2018"
python get_polyFit.py -l ${label} -y ${year}