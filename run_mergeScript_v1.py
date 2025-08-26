#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count

# --- paths ---
REPO_ROOT = Path("/depot/cms/private/users/shar1172/copperheadV2_main")
SCRIPT = REPO_ROOT / "scripts" / "mergeNanoAODRootFiles.py"

# Be gentle on I/O; bump if storage can handle it (e.g. 24 or 32 on your 64-core node)
N_PROCS = min(11, cpu_count())

# (input_path, output_subdir, output_filename)
JOBS_UL2018 = [
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD/UL2018/GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8",
        "GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8",
        "GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/hyeonseo/Run2UL/UL2018/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/nanoV12_lxplus",
        "DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_M-105To160_VBFFilter_lxplus.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/hyeonseo/Run2UL/UL2018/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/nanoV12_hammer",
        "DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_M-105To160_VBFFilter_hammer.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/DYJetsToMuMu_M-50_massWgtFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
        "DYJetsToMuMu_M-50_massWgtFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
        "DYJetsToMuMu_M50_MiNNLO.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/DYJetsToMuMu_M-100to200_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
        "DYJetsToMuMu_M-100to200_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
        "DYJetsToMuMu_M100to200_MiNNLO.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
        "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
        "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
        "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
        "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_GT36/UL2018/SingleMuon_Run2018A",
        "SingleMuon_Run2018A",
        "SingleMuon_Run2018A.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_GT36/UL2018/SingleMuon_Run2018B",
        "SingleMuon_Run2018B",
        "SingleMuon_Run2018B.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_GT36/UL2018/SingleMuon_Run2018C",
        "SingleMuon_Run2018C",
        "SingleMuon_Run2018C.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_GT36/UL2018/SingleMuon_Run2018D",
        "SingleMuon_Run2018D",
        "SingleMuon_Run2018D.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/Run2_CustomNanoAODv12/UL2018/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL18NanoAODv9",
        "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL18NanoAODv9",
        "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL18NanoAODv9.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/Run2_CustomNanoAODv12/UL2018/ST_s-channel_4f_hadronicDecays_TuneCP5_13TeV-amcatnlo-pythia8",
        "ST_s-channel_4f_hadronicDecays_TuneCP5_13TeV-amcatnlo-pythia8",
        "ST_s-channel_4f_hadronicDecays_TuneCP5_13TeV-amcatnlo-pythia8.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD/UL2018/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9",
        "WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9",
        "WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/ZZ_TuneCP5_13TeV-pythia8",
        "ZZ_TuneCP5_13TeV-pythia8",
        "ZZ_TuneCP5_13TeV-pythia8.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8",
        "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8",
        "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8",
        "WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8",
        "WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/WZZ_TuneCP5_13TeV-amcatnlo-pythia8",
        "WZZ_TuneCP5_13TeV-amcatnlo-pythia8",
        "WZZ_TuneCP5_13TeV-amcatnlo-pythia8.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8",
        "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8",
        "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole",
        "EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole",
        "EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole.root",
        "UL2018",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8",
        "VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8",
        "VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8.root",
        "UL2018",
    ),
]


JOBS_UL2017 = [
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/SingleMuon_Run2017B",
    #     "SingleMuon_Run2017B",
    #     "SingleMuon_Run2017B.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/SingleMuon_Run2017C",
    #     "SingleMuon_Run2017C",
    #     "SingleMuon_Run2017C.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/SingleMuon_Run2017D",
    #     "SingleMuon_Run2017D",
    #     "SingleMuon_Run2017D.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/SingleMuon_Run2017E",
    #     "SingleMuon_Run2017E",
    #     "SingleMuon_Run2017E.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/SingleMuon_Run2017F",
    #     "SingleMuon_Run2017F",
    #     "SingleMuon_Run2017F.root",
    #     "UL2017",
    # ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL17NanoAODv9",
        "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL17NanoAODv9",
        "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL17NanoAODv9.root",
        "UL2017",
    ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    #     "DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    #     "DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/hyeonseo/Run2UL/UL2017/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/nanoV12_hammer",
    #     "DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8",
    #     "DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/DYJetsToMuMu_M-50_massWgtFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
    #     "DYJetsToMuMu_M-50_massWgtFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
    #     "DYJetsToMuMu_M-50_massWgtFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/DYJetsToMuMu_M-100to200_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
    #     "DYJetsToMuMu_M-100to200_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
    #     "DYJetsToMuMu_M-100to200_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
    #     "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
    #     "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
    #     "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
    #     "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
    #     "ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
    #     "ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
    #     "ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
    #     "ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
    #     "ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
    #     "ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
    #     "ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
    #     "ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
    #     "WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
    #     "WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    #     "WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    #     "WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    #     "WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    #     "WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    #     "WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    #     "WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/ZZ_TuneCP5_13TeV-pythia8",
    #     "ZZ_TuneCP5_13TeV-pythia8",
    #     "ZZ_TuneCP5_13TeV-pythia8.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8",
    #     "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8",
    #     "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL17NanoAODv9",
    #     "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL17NanoAODv9",
    #     "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL17NanoAODv9.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8",
    #     "WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8",
    #     "WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/WZZ_TuneCP5_13TeV-amcatnlo-pythia8",
    #     "WZZ_TuneCP5_13TeV-amcatnlo-pythia8",
    #     "WZZ_TuneCP5_13TeV-amcatnlo-pythia8.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8",
    #     "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8",
    #     "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole",
    #     "EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole",
    #     "EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8",
    #     "GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8",
    #     "GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/GluGluHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8",
    #     "GluGluHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8",
    #     "GluGluHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8.root",
    #     "UL2017",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8",
    #     "VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8",
    #     "VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8.root",
    #     "UL2017",
    # ),
]

JOBS_UL2016postVFP = [
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODv9",
        "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODv9",
        "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODv9.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/hyeonseo/Run2UL/UL2016postVFP/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/nanoV12_hammer/",
        "DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/DYJetsToMuMu_M-50_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
        "DYJetsToMuMu_M-50_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
        "DYJetsToMuMu_M-50_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/DYJetsToMuMu_M-100to200_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
        "DYJetsToMuMu_M-100to200_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
        "DYJetsToMuMu_M-100to200_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
        "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
        "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
        "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
        "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
        "WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
        "WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/ZZ_TuneCP5_13TeV-pythia8",
        "ZZ_TuneCP5_13TeV-pythia8",
        "ZZ_TuneCP5_13TeV-pythia8.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8",
        "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8",
        "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL16NanoAODv9",
        "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL16NanoAODv9",
        "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL16NanoAODv9.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8",
        "WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8",
        "WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/WZZ_TuneCP5_13TeV-amcatnlo-pythia8",
        "WZZ_TuneCP5_13TeV-amcatnlo-pythia8",
        "WZZ_TuneCP5_13TeV-amcatnlo-pythia8.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8",
        "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8",
        "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole",
        "EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole",
        "EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8",
        "GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8",
        "GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/GluGluHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8",
        "GluGluHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8",
        "GluGluHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8.root",
        "UL2016postVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8",
        "VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8",
        "VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8.root",
        "UL2016postVFP",
    ),
]

JOBS_UL2016preVFP = [
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/SingleMuon_Run2016B",
    #     "SingleMuon_Run2016B",
    #     "SingleMuon_Run2016B.root",
    #     "UL2016preVFP",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/SingleMuon_Run2016C",
    #     "SingleMuon_Run2016C",
    #     "SingleMuon_Run2016C.root",
    #     "UL2016preVFP",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/SingleMuon_Run2016D",
    #     "SingleMuon_Run2016D",
    #     "SingleMuon_Run2016D.root",
    #     "UL2016preVFP",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/SingleMuon_Run2016E",
    #     "SingleMuon_Run2016E",
    #     "SingleMuon_Run2016E.root",
    #     "UL2016preVFP",
    # ),
    # (
    #     "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/SingleMuon_Run2016F",
    #     "SingleMuon_Run2016F",
    #     "SingleMuon_Run2016F.root",
    #     "UL2016preVFP",
    # ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODAPVv9",
        "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODAPVv9",
        "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODAPVv9.root",
        "UL2016preVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8.root",
        "UL2016preVFP",
    ),
    ( # REDO
        "/eos/purdue/store/user/hyeonseo/Run2UL/UL2016preVFP/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/nanoV12_hammer",
        "DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8.root",
        "UL2016preVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/DYJetsToMuMu_M-50_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
        "DYJetsToMuMu_M-50_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
        "DYJetsToMuMu_M-50_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos.root",
        "UL2016preVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/DYJetsToMuMu_M-100to200_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
        "DYJetsToMuMu_M-100to200_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
        "DYJetsToMuMu_M-100to200_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos.root",
        "UL2016preVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
        "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
        "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8.root",
        "UL2016preVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
        "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
        "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8.root",
        "UL2016preVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8.root",
        "UL2016preVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8.root",
        "UL2016preVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8.root",
        "UL2016preVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8.root",
        "UL2016preVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
        "WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
        "WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8.root",
        "UL2016preVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8.root",
        "UL2016preVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8.root",
        "UL2016preVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8.root",
        "UL2016preVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/ZZ_TuneCP5_13TeV-pythia8",
        "ZZ_TuneCP5_13TeV-pythia8",
        "ZZ_TuneCP5_13TeV-pythia8.root",
        "UL2016preVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8",
        "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8",
        "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8.root",
        "UL2016preVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL16NanoAODAPVv9",
        "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL16NanoAODAPVv9",
        "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL16NanoAODAPVv9.root",
        "UL2016preVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8",
        "WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8",
        "WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8.root",
        "UL2016preVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/WZZ_TuneCP5_13TeV-amcatnlo-pythia8",
        "WZZ_TuneCP5_13TeV-amcatnlo-pythia8",
        "WZZ_TuneCP5_13TeV-amcatnlo-pythia8.root",
        "UL2016preVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8",
        "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8",
        "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8.root",
        "UL2016preVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole",
        "EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole",
        "EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole.root",
        "UL2016preVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8",
        "GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8",
        "GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8.root",
        "UL2016preVFP",
    ),
    (
        "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8",
        "VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8",
        "VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8.root",
        "UL2016preVFP",
    ),
]

# year = "UL2018"
year = "UL2017"
# year = "UL2016preVFP"
# year = "UL2016postVFP"
BASE_OUT = f"/store/user/rasharma/Run2_CustomNanoAODv12/hadded/{year}"

JOBS = []
if year == "UL2018":
    JOBS = JOBS_UL2018
elif year == "UL2017":
    JOBS = JOBS_UL2017
elif year == "UL2016preVFP":
    JOBS = JOBS_UL2016preVFP
elif year == "UL2016postVFP":
    JOBS = JOBS_UL2016postVFP

def run_job(job):
    inpath, outsub, fname, year = job
    outdir = Path(BASE_OUT) / outsub
    # outdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()

    # Prepend repo to PYTHONPATH so "from modules..." works
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH','')}"
    cmd = [
        "python", "scripts/mergeNanoAODRootFiles.py",
        "-i", inpath,
        "-o", str(outdir),
        "-f", fname,
        "-y", year
    ]
    print(f"→ {outsub}")
    # Inherit stdout/stderr so you see progress/errors directly
    res = subprocess.run(cmd)
    status = "OK" if res.returncode == 0 else f"FAIL ({res.returncode})"
    print(f"← {outsub}: {status}")
    return res.returncode


if __name__ == "__main__":
    print(f"Total jobs: {len(JOBS)} | processes: {N_PROCS}")
    with Pool(processes=N_PROCS) as pool:
        rcodes = pool.map(run_job, JOBS)
    nfail = sum(1 for r in rcodes if r != 0)
    if nfail:
        print(f"\nDone with failures: {nfail}/{len(JOBS)}")
        raise SystemExit(1)
    print("\nAll jobs finished successfully.")
