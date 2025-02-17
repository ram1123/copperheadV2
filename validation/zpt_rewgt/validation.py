import os
import sys
import logging
from modules.utils import logger

data_l = ['A', 'B', 'C', 'D']
# bkg_l = ['DY']
bkg_l = ['DY','TT','ST','VV']
# sig_l = ['ggH', 'VBF']
sig_l = []

vars2plot = ['dimuon']

lumi_dict = {
    "2018" : 59.83,
    "2017" : 41.48,
    "2016postVFP": 19.5,
    "2016preVFP": 16.81,
}
status = "Private_Work"
region = "z-peak"

"""
First we plot Zpt off stage1 plots to match AN-19-124 fig 4.3
"""
# label = "V2_Jan09_ForZptReWgt"
# label = "V2_Jan17_JecDefault_plotEveryonesZptWgt"
# label = "V2_Jan22_JecDefault_plotEveryonesZptWgt"
label = "Feb11_WithPurdueZptWgt"

year = "2018"
# year = "2017"
# year = "2016postVFP"
# year = "2016preVFP"
lumi = lumi_dict[year]


username = os.environ.get("USER") or os.environ.get("USERNAME")
if username:
    print(f"Username: {username}")
else:
    print("Username not found in environment variables.")
    sys.exit()


# load_path = f"/depot/cms/users/yun79/hmm/copperheadV1clean/{label}/stage1_output/{year}/f1_0/"
load_path = f"/depot/cms/users/{username}/hmm/copperheadV1clean/{label}/stage1_output/{year}/f1_0/"
# load_path = "/depot/cms/users/shar1172/hmm/copperhead_outputs_11Feb/stage1_output/2018/f1_0/"

if not os.path.exists(load_path):
    print(f"Path: {load_path} does not exits")
    sys.exit()
else:
    print(f"Path of ntuples: {load_path}")


plot_setting = "./validation/zpt_rewgt/plot_settings_Zpt_reWgt.json"

keep_zpt_on = True # dummy value



vars2plot = ' '.join(vars2plot)
data_l = ' '.join(data_l)
bkg_l = ' '.join(bkg_l)
sig_l = ' '.join(sig_l)

njet = -1

# for njet in [-1]:
for njet in [-1, 0, 1, 2]:
    # for cat in ["nocat"]:
    for cat in ["vbf", "ggh", "nocat"]:
        if cat == "vbf" and njet < 2: continue
        zpt_name = "yes_zpt"
        os.system(f"python validation/zpt_rewgt/zpt_validation_plotter.py -y {year} --load_path {load_path}  -var {vars2plot} --data {data_l} --background {bkg_l} --signal {sig_l} --lumi {lumi} --status {status} -cat {cat} -reg {region} --label {label} --plot_setting {plot_setting} --zpt_on {keep_zpt_on} --jet_multiplicity {njet} --zpt_wgt_name {zpt_name}")
    
        zpt_name = "no_zpt"
        os.system(f"python validation/zpt_rewgt/zpt_validation_plotter.py -y {year} --load_path {load_path}  -var {vars2plot} --data {data_l} --background {bkg_l} --signal {sig_l} --lumi {lumi} --status {status} -cat {cat} -reg {region} --label {label} --plot_setting {plot_setting} --zpt_on {keep_zpt_on} --jet_multiplicity {njet} --zpt_wgt_name {zpt_name}")
