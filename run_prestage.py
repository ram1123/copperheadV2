import awkward as ak
from coffea.dataset_tools import rucio_utils
from coffea.dataset_tools.preprocess import preprocess
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
import json
import os
import argparse
import dask
dask.config.set({'logging.distributed': 'error'})
from distributed import LocalCluster, Client
import time
import copy
import tqdm
import uproot
import random
# random.seed(9002301)
import re
import glob
# import warnings
# warnings.filterwarnings("error", module="coffea.*")



datasets = {
    "2023" : {
        "data_E" : "/Muon/Run2023E-PromptReco-v1/NANOAOD",
        "dy_M-50": "/DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8/Run3Summer23NanoAODv12-PilotMuonHits_130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
        "ttjets_dl": "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
        "ww_2l2nu": "/WWto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4/NANOAODSIM",
        "wz_3lnu": "/WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
        "wz_2l2q": "/WZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
        "wz_1l1nu2q": "/WZtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
        "zz": "/ZZ_TuneCP5_13p6TeV_pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
    },
    "2022postEE": {
        "data_E": "/Muon/Run2022E-16Dec2023-v1/NANOAOD",
        "data_F": "/Muon/Run2022F-16Dec2023-v1/NANOAOD",
        "data_G": "/Muon/Run2022G-19Dec2023-v2/NANOAOD",
        "dy_M-50": "/DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v5-v2/NANOAODSIM",
        "ttjets_dl": "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
        "ww_2l2nu": "/WWto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
        "wz_3lnu": "/WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
        "wz_2l2q": "/WZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
        "wz_1l1nu2q": "/WZtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
        "zz": "/ZZ_TuneCP5_13p6TeV_pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
        "ggh_powheg":"/GluGluHto2Mu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM",
        "vbf_powheg" : "/VBFHto2Mu_M-125_TuneCP5_withDipoleRecoil_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM",
    },
    "2022preEE": {
        "data_C": "/Muon/Run2022C-16Dec2023-v1/NANOAOD",
        "data_D": "/Muon/Run2022D-16Dec2023-v1/NANOAOD",
        "dy_M-50": "/DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
        "ttjets_dl": "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
        "ww_2l2nu": "/WWto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
        "wz_3lnu": "/WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
        "wz_2l2q": "/WZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
        "wz_1l1nu2q": "/WZtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
        "zz": "/ZZ_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
        "ggh_powheg":"/GluGluHto2Mu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM",
        "vbf_powheg" : "/VBFHto2Mu_M-125_TuneCP5_withDipoleRecoil_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM",
    },
    "2016preVFP": {
        "data_B": "/SingleMuon/Run2016B-ver2_HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD",
        "data_C": "/SingleMuon/Run2016C-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD",
        "data_D": "/SingleMuon/Run2016D-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD",
        "data_E": "/SingleMuon/Run2016E-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD",
        "data_F": "/SingleMuon/Run2016F-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD",
        "dy_M-50": "/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "dy_M-100To200": "/DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "ttjets_dl": "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "ttjets_sl": "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        # "ttw": "",
        # "ttz": "",
        "st_tw_top": "/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "st_tw_antitop": "/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM", 
        "ww_2l2nu": "/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "wz_3lnu": "/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "wz_2l2q": "/WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM",
        "wz_1l1nu2q": "/WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "zz": "/ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        # "www": "",
        # "wwz": "",
        # "wzz": "",
        # "zzz": "",
        "ewk_lljj_mll50_mjj120": "/EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "ggh_powheg":"/GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        # "ggh_amc": "",
        # "ggh_amcPS": "",
        # "ggh_powhegPS": "",
        "vbf_powheg": "/VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        # "vbf_powhegPS": "",
        # "vbf_powheg_herwig": "",
        # "vbf_powheg_dipole": "",
        # "wmh": "",
        # "wph": "",
        # "tth": "",
        # "zh": "",
    },
    "2016postVFP": {
        "data_F": "/SingleMuon/Run2016F-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD",
        "data_G": "/SingleMuon/Run2016G-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD",
        "data_H": "/SingleMuon/Run2016H-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD",
        "dy_M-50": "/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "dy_M-100To200": "/DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
        "ttjets_dl": "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "ttjets_sl": "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        # "ttw": "",
        # "ttz": "",
        "st_tw_top":"/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "st_tw_antitop":"/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM", 
        "ww_2l2nu": "/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "wz_3lnu": "/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "wz_2l2q": "/WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
        "wz_1l1nu2q": "/WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "zz": "/ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        # "www": "",
        # "wwz": "",
        # "wzz": "",
        # "zzz": "",
        "ewk_lljj_mll50_mjj120": "/EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "ggh_powheg":"/GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        # "ggh_amc": "",
        # "ggh_amcPS": "",
        # "ggh_powhegPS": "",
        "vbf_powheg": "/VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        # "vbf_powhegPS": "",
        # "vbf_powheg_herwig": "",
        # "vbf_powheg_dipole": "",
        # "wmh": "",
        # "wph": "",
        # "tth": "",
        # "zh": "",
    },
    "2017": {
        "data_B": "/SingleMuon/Run2017B-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD",
        "data_C": "/SingleMuon/Run2017C-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD",
        "data_D": "/SingleMuon/Run2017D-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD",
        "data_E": "/SingleMuon/Run2017E-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD",
        "data_F": "/SingleMuon/Run2017F-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD",
        "dy_M-50": "/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "dy_M-100To200": "/DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "ttjets_dl": "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "ttjets_sl": "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        # "ttw": "",
        # "ttz": "",
        "st_tw_top":"/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "st_tw_antitop":"/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM", 
        "ww_2l2nu": "/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "wz_3lnu": "/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "wz_2l2q": "/WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "wz_1l1nu2q": "/WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "zz": "/ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        # "www": "",
        # "wwz": "",
        # "wzz": "",
        # "zzz": "",
        "ewk_lljj_mll50_mjj120": "/EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "ggh_powheg":"/GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        # "ggh_amc": "",
        # "ggh_amcPS": "",
        # "ggh_powhegPS": "",
        "vbf_powheg": "/VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        # "vbf_powhegPS": "",
        # "vbf_powheg_herwig": "",
        # "vbf_powheg_dipole": "",
    },
    "2018": {
        "data_A": "/SingleMuon/Run2018A-UL2018_MiniAODv2_NanoAODv9-v2/NANOAOD",
        "data_B": "/SingleMuon/Run2018B-UL2018_MiniAODv2_NanoAODv9-v2/NANOAOD",
        "data_C": "/SingleMuon/Run2018C-UL2018_MiniAODv2_NanoAODv9-v2/NANOAOD",
        "data_D": "/SingleMuon/Run2018D-UL2018_MiniAODv2_NanoAODv9-v1/NANOAOD",
        "dy_M-50": "/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X*/NANOAODSIM",
        "dy_M-100To200": "/DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "dy_VBF_filter": "dummy",
        "ttjets_dl": "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "ttjets_sl": "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        # # "ttw": "",
        # # "ttz": "",
        "st_tw_top":"/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "st_tw_antitop":"/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM", 
        "ww_2l2nu": "/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
        "wz_3lnu": "/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
        "wz_2l2q": "/WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "wz_1l1nu2q": "/WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "zz": "/ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        # # "www": "",
        # # "wwz": "",
        # # "wzz": "",
        # # "zzz": "",
        "ewk_lljj_mll50_mjj120": "/EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "ggh_powheg":"/GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        # # "ggh_amc": "",
        # # "ggh_amcPS": "",
        # # "ggh_powhegPS": "",
        "vbf_powheg": "/VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        # # "vbf_powhegPS": "",
        # # "vbf_powheg_herwig": "",
        # # "vbf_powheg_dipole": "",
    },
}

def get_Xcache_filelist(fnames: list):
    new_fnames = []
    for fname in fnames:
        root_file = re.findall(r"/store.*", fname)[0]
        x_cache_fname = "root://cms-xcache.rcac.purdue.edu/" + root_file
        new_fnames.append(x_cache_fname)
    return new_fnames
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="2018",
    action="store",
    help="year value. The options are: 2016preVFP, 2016postVFP, 2017, 2018",
    )
    parser.add_argument(
    "-ch",
    "--chunksize",
    dest="chunksize",
    default="10000",
    action="store",
    help="chunksize",
    )
    parser.add_argument(
    "-frac",
    "--change_fraction",
    dest="fraction",
    default=None,
    action="store",
    help="change fraction of steps of the data",
    )
    # parser.add_argument(
    # "-in_str",
    # "--input_string",
    # dest="input_string",
    # default=None,
    # action="store",
    # help="string representation of samples to process, in the format of Year_{year}/DataRun_{A,B,C,D)}/Bkg_{DY,tt, etc}/Sig_{ggH, VBF}",
    # )
    parser.add_argument(
    "-data",
    "--data",
    dest="data_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of data samples represented by alphabetical letters A-H",
    )
    parser.add_argument(
    "-bkg",
    "--background",
    dest="bkg_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of bkg samples represented by shorthands: DY, TT, ST, DB (diboson), EWK",
    )
    parser.add_argument(
    "-sig",
    "--signal",
    dest="sig_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of sig samples represented by shorthands: ggH, VBF",
    )
    parser.add_argument(
    "--use_gateway",
    dest="use_gateway",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, uses dask gateway client instead of local",
    )
    parser.add_argument(
    "--xcache",
    dest="xcache",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, uses xcache root file paths",
    )
    parser.add_argument(
    "--skipBadFiles",
    dest="skipBadFiles",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, uses skips bad files when calling preprocessing",
    )
    args = parser.parse_args()
    time_step = time.time()
    # print(f"args.bkg_samples: {args.bkg_samples}")
    os.environ['XRD_REQUESTTIMEOUT']="2400" # some root files via XRootD may timeout with default value
    if args.fraction is None: # do the normal prestage setup
        allowlist_sites=["T2_US_Purdue"] # take data only from purdue for now
        total_events = 0
        # get dask client
        # turning off seperate client test start --------------------------------------------------------
        if args.use_gateway:
            from dask_gateway import Gateway
            gateway = Gateway(
                "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
                proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
            )
            cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
            client = gateway.connect(cluster_info.name).get_client()
            print("Gateway Client created")
        else: # use local cluster
            # cluster = LocalCluster(processes=True)
            # cluster.adapt(minimum=8, maximum=31) #min: 8 max: 32
            # client = Client(cluster)
            client = Client(n_workers=15,  threads_per_worker=1, processes=True, memory_limit='30 GiB')
            print("Local scale Client created")
        # turning off seperate client test end --------------------------------------------------------
        big_sample_info = {}
        year = args.year
        dataset = datasets[year]
        new_sample_list = []
       
        # take data
        data_l =  [sample_name for sample_name in dataset.keys() if "data" in sample_name]
        data_samples = args.data_samples
        # print(f"data_samples: {data_samples}")
        # print(f"data_l: {data_l}")
        if len(data_samples) >0:
            for data_letter in data_samples:
                for sample_name in data_l:
                    if data_letter in sample_name:
                        new_sample_list.append(sample_name)
        # take bkg
        bkg_samples = args.bkg_samples
        print(f"bkg_samples: {bkg_samples}")
        if len(bkg_samples) >0:
            for bkg_sample in bkg_samples:
                if bkg_sample.upper() == "DY": # enforce upper case to prevent confusion
                    # new_sample_list.append("dy_M-50")
                    # new_sample_list.append("dy_M-100To200")
                    new_sample_list.append("dy_VBF_filter")
                elif bkg_sample.upper() == "TT": # enforce upper case to prevent confusion
                    new_sample_list.append("ttjets_dl")
                    new_sample_list.append("ttjets_sl")
                elif bkg_sample.upper() == "ST": # enforce upper case to prevent confusion
                    new_sample_list.append("st_tw_top")
                    new_sample_list.append("st_tw_antitop")
                elif bkg_sample.upper() == "VV": # enforce upper case to prevent confusion
                    new_sample_list.append("ww_2l2nu")
                    new_sample_list.append("wz_3lnu")
                    new_sample_list.append("wz_2l2q")
                    new_sample_list.append("wz_1l1nu2q")
                    new_sample_list.append("zz")
                elif bkg_sample.upper() == "EWK": # enforce upper case to prevent confusion
                    new_sample_list.append("ewk_lljj_mll50_mjj120")
                else:
                    print(f"unknown background {bkg_sample} was given!")
            
        # take sig
        sig_samples = args.sig_samples
        if len(sig_samples) >0:
            for sig_sample in sig_samples:
                if sig_sample.upper() == "GGH": # enforce upper case to prevent confusion
                    new_sample_list.append("ggh_powheg")
                elif sig_sample.upper() == "VBF": # enforce upper case to prevent confusion
                    new_sample_list.append("vbf_powheg")
                else:
                    print(f"unknown signal {sig_sample} was given!")
        
        dataset = dict([(sample_name, dataset[sample_name]) for sample_name in new_sample_list])
        print(f"new dataset: {dataset.keys()}")

        for sample_name in tqdm.tqdm(dataset.keys()):
            if sample_name == "dy_VBF_filter":
                """
                temporary condition bc IDK a way to integrate prod/phys03 CMSDAS datasets into rucio utils
                """
                # test start -----------------------------------------------------------
                # load_path = "/eos/purdue/store/user/vscheure/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/UL18_Nano/240514_124107/"
                # # fnames = glob.glob(f"{load_path}/*/*.root")
                # # fnames = glob.glob(f"{load_path}/0002/*.root")
                # fnames = glob.glob(f"{load_path}/0000/*.root")
                # test end -----------------------------------------------------------
                load_path = "/depot/cms/users/yun79/private_samples/MC/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/UL18_Nano"
                fnames = glob.glob(f"{load_path}/*.root")
            else:
                das_query = dataset[sample_name]
                print(f"das_query: {das_query}")
                
                rucio_client = rucio_utils.get_rucio_client()
                outlist, outtree = rucio_utils.query_dataset(
                    das_query,
                    client=rucio_client,
                    tree=True,
                    scope="cms",
                )
                outfiles,outsites,sites_counts =rucio_utils.get_dataset_files_replicas(
                    outlist[0],
                    allowlist_sites=allowlist_sites,
                    mode="full",
                    client=rucio_client,
                    # partial_allowed=True
                )
                fnames = [file[0] for file in outfiles if file != []]
                # fnames = [fname.replace("root://eos.cms.rcac.purdue.edu//", "/eos/purdue") for fname in fnames] # replace xrootd prefix bc it's causing file not found error
                
                # random.shuffle(fnames)
                if args.xcache:
                    fnames = get_Xcache_filelist(fnames)
                # print(f"fnames: {fnames}")
            print(f"sample_name: {sample_name}")
            print(f"len(fnames): {len(fnames)}")

            
            """
            run through each file and collect total number of 
            """
            preprocess_metadata = {
                "sumGenWgts" : None,
                "nGenEvts" : None,
                "data_entries" : None,
            }
            if "data" in sample_name: # data sample
                file_input = {fname : {"object_path": "Events"} for fname in fnames}
                events = NanoEventsFactory.from_root(
                        file_input,
                        metadata={},
                        schemaclass=NanoAODSchema,
                        uproot_options={"timeout":2400},
                ).events()
                preprocess_metadata["data_entries"] = int(ak.num(events.Muon.pt, axis=0).compute()) # convert into 32bit precision as 64 bit precision isn't json serializable
                total_events += preprocess_metadata["data_entries"] 
            else: # if MC
                file_input = {fname : {"object_path": "Runs"} for fname in fnames}
                # print(f"file_input: {file_input}")
                # print(f"file_input: {file_input}")
                # print(len(file_input.keys()))
                runs = NanoEventsFactory.from_root(
                        file_input,
                        metadata={},
                        schemaclass=BaseSchema,
                        uproot_options={"timeout":2400},
                ).events()
                # runs = uproot.dask(file_input, handler=uproot.XRootDSource)
                
                preprocess_metadata["sumGenWgts"] = float(ak.sum(runs.genEventSumw).compute()) # convert into 32bit precision as 64 bit precision isn't json serializable
                preprocess_metadata["nGenEvts"] = int(ak.sum(runs.genEventCount).compute()) # convert into 32bit precision as 64 bit precision isn't json serializable
                total_events += preprocess_metadata["nGenEvts"] 

    
            val = "Events"
            file_dict = {}
            for file in fnames:
                file_dict[file] = val
            final_output = {
                sample_name :{"files" :file_dict}
            }
            
            step_size = int(args.chunksize)
            files_available, files_total = preprocess(
                final_output,
                step_size=step_size,
                align_clusters=False,
                skip_bad_files=args.skipBadFiles,
            )
            pre_stage_data = files_available
            # add in metadata
            pre_stage_data[sample_name]['metadata'] = preprocess_metadata
            # add in faction -> for later use
            pre_stage_data[sample_name]['metadata']['fraction'] = 1.0
            pre_stage_data[sample_name]['metadata']['original_fraction'] = 1.0
            # if preprocess_metadata["data_entries"] is not None: # Data
            if "data" in sample_name: # data sample
                pre_stage_data[sample_name]['metadata']["is_mc"] = False
            else: # MC
                pre_stage_data[sample_name]['metadata']["is_mc"] = True
            pre_stage_data[sample_name]['metadata']["dataset"] = sample_name
            big_sample_info.update(pre_stage_data)
        
        #save the sample info
        directory = "./config"
        filename = directory+"/processor_samples.json"
        dupli_fname = directory+"/fraction_processor_samples.json" # duplicated fname in case you want to skip fractioning
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filename, "w") as file:
                json.dump(big_sample_info, file)
        with open(dupli_fname, "w") as file:
                json.dump(big_sample_info, file)
    
        elapsed = round(time.time() - time_step, 3)
        print(f"Finished everything in {elapsed} s.")
        print(f"Total Events in files {total_events}.")
        
    else: # take the pre existing samples.json and prune off files we don't need
        fraction = float(args.fraction)
        sample_path = "./config/processor_samples.json"
        with open(sample_path) as file:
            samples = json.loads(file.read())
        new_samples = copy.deepcopy(samples) # copy old sample, overwrite it later
        if fraction < 1.0: # else, just save the original samples and new samples
            for sample_name, sample in tqdm.tqdm(samples.items()):
                is_data = "data" in sample_name
                tot_N_evnts = sample['metadata']["data_entries"] if is_data else sample['metadata']["nGenEvts"]
                new_N_evnts = int(tot_N_evnts*fraction)
                old_N_evnts = new_samples[sample_name]['metadata']["data_entries"] if is_data else new_samples[sample_name]['metadata']["nGenEvts"]
                if is_data:
                    print("data!")
                    new_samples[sample_name]['metadata']["data_entries"] = new_N_evnts
                else:
                    new_samples[sample_name]['metadata']["nGenEvts"] = new_N_evnts
                    new_samples[sample_name]['metadata']["sumGenWgts"] *= new_N_evnts/old_N_evnts # just directly multiply by fraction for this since this is already float and this is much faster
                # new_samples[sample_name]['metadata']["fraction"] = fraction
                # state new fraction
                new_samples[sample_name]['metadata']['fraction'] = new_N_evnts/old_N_evnts
                print(f"new_samples[sample_name]['metadata']['fraction']: {new_samples[sample_name]['metadata']['fraction']}")
                # new_samples[sample_name]['metadata']["original_fraction"] = fraction
                
                # loop through the files to correct the steps
                event_counter = 0 # keeps track of events of multiple root files
                stop_flag = False
                new_files = {}
                for file, file_dict in sample["files"].items():
                    if stop_flag:
                        del new_samples[sample_name]["files"][file] # delete the exess files
                        continue
                    new_steps = []
                    # loop through step sizes to correct fractions
                    for step_iteration in file_dict["steps"]:
                        new_step_lim = new_N_evnts-event_counter
                        if step_iteration[1] < new_step_lim:
                            new_steps.append(step_iteration)
                        else:  # change the upper limit
                            new_steps.append([
                                step_iteration[0],
                                new_step_lim
                            ])
                            stop_flag = True
                            break
                    new_samples[sample_name]["files"][file]["steps"] = new_steps # overwrite new steps
                    # add the end step val to the event_counter
                    if not stop_flag: # update variables and move to next file
                        end_idx = len(file_dict["steps"])-1
                        event_counter += file_dict["steps"][end_idx][1]

        #save the sample info
        directory = "./config"
        filename = directory+"/fraction_processor_samples.json"
        with open(filename, "w") as file:
                json.dump(new_samples, file)
    
        elapsed = round(time.time() - time_step, 3)
        print(f"Finished everything in {elapsed} s.")


