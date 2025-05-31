#!/bin/bash
# set -e


# first rederive the SF root files for fitting
# year="2018"
# year="all"
# label="UpdatedDY_100_200_CrossSection_24Feb_jetpuidOff"
# label="DYMiNNLO_jetpuidOff_newZptWgt25Mar2025"
# label="March25_NanoAODv9_"

# ==================================
# # year="2017"
# ==================================
# year="2017"
# label="DYMiNNLO_30Mar2025"

# year="2016postVFP"
# label="March25_NanoAODv9_WithUpdatedZptWgt"

year="2018"
label="April09_NanoV12"

#  below path for the file `save_SF_rootFiles.py` to  fetch the root files
# base_path = f"/depot/cms/users/yun79/hmm/copperheadV1clean/{label}/stage1_output/{year}/f1_0/"  # 2017 Hyeon stage-1; Minlo samples
# /depot/cms/users/shar1172/hmm/copperheadV1clean/March25_NanoAODv9_WithUpdatedZptWgt//stage1_output/2016postVFP/f1_0/dy_M-100To200_MiNNLO/0

# Get SF root files
# python save_SF_rootFiles.py -l ${label} -y ${year}

# do F test
nbin="100"
njet=2
# python do_f_test_RooFit.py --run_label ${label} --year ${year} --nbins ${nbin} --njet ${njet}
# python do_f_test.py --run_label ${label} --year ${year} --nbins ${nbin} --njet ${njet} --outAppend "final" --debug
python get_polyFit.py -l ${label} -y ${year} --nbins ${nbin} --njet ${njet} --outAppend "final"
