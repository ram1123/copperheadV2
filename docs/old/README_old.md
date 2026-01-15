## 🐍 Copperhead V2, - Columnar Parallel Pythonic framEwork for Run3 H&rarr;µµ Decay search

setup:
```bash
git clone https://github.com/green-cabbage/copperheadV2.git
cd copperheadV2
conda env create -f conda_envs/env.yml
```
If accessing datasets via `xRootD` will be needed:
```bash
source setup_proxy.sh
```

This would start the GRID certificate password prompt, and then once given the password, the conda env would be activated. Once voms proxy and conda env is activated, we can start on the tutorial.ipynb for ggH category

For VBF-category, our temporary implementation is to activate coffea_latest env and them execute stage1_sh.sh -> stage2_vbf_sh.sh -> stage3_vbf_sh.sh


## Planned High-level arrangement of code in CopperheadV2

| Task | Directory Location of Relevant Code |
| ------------- | ------------- |
|  Event by event dimuon mass calibration   | ./lib/ebeMassResCalibration/ |
|  Zpt weight calculation   | ./lib/ZptWgtCalculation/ |
|  ggH production channel MVA training | ./lib/MVA_training/ggH/ |
|  VBF production channel MVA training | ./lib/MVA_training/VBF/ |
|  General corrections (stage1) | ./lib/corrections/ |
|  Roofit fitting (stage3) | ./lib/fit_models/ |
|  Parameters (metadata) | ./parameters/*.yaml|



## Table of correction weight file locations:

| Correction | Year (Data/MC) | Local Location | Central Source |
| ------------- | ------------- | ------------- |------------- |
|   Rochester Correction  | 2024 |  | |
|   | 2023 |   | |
|   | 2022 |   | |
|   | 2018 | data/roch_corr/RoccoR2018UL.txt  | |
|   | 2017 | data/roch_corr/RoccoR2017UL.txt  | |
|   | 2016postVFP | data/roch_corr/RoccoR2016bUL.txt  | |
|   | 2016preVFP | data/roch_corr/RoccoR2016aUL.txt  | |
| Zpt SF | 2024 |  | |
|   | 2023 |   | |
|   | 2022 |   | |
|   | 2018, 2017, 2016 | data/zpt_weights.histo.json | Locally Calculated |
| NNLOPS (ggH) | 2024 |  | |
|   | 2023 |   | |
|   | 2022 |   | |
|   | 2018, 2017, 2016 | data/NNLOPS_reweight.root | |
|   Lumi Mask  | 2024 |  | |
|   | 2023 |   | |
|   | 2022 | data/Cert_Collisions2022_355100_362760_Golden.json | https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis#2022_Analysis_Summary_Table |
|   | 2018 | data/lumimasks/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt  | |
|   | 2017 | data/lumimasks/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt  | |
|   | 2016 | data/lumimasks/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt  | |
|   Event by event Dimuon mass resolution  | 2024 |  | Locally Calculated |
|   | 2023 |   | Locally Calculated  |
|   | 2022 |   | Locally Calculated  |
|   | 2018 | /data/res_calib/res_calib_{Data or MC}_2018.root  | Locally Calculated  |
|   | 2017 | /data/res_calib/res_calib_{Data or MC}_2017.root   | Locally Calculated  |
|   | 2016 | /data/res_calib/res_calib_{Data or MC}_2016.root   | Locally Calculated  |
|   Jet PU ID SF  | 2024 |  | |
|   | 2023 |   | |
|   | 2022 |   | |
|   | 2018, 2017, 2016 | data/PUID_106XTraining_ULRun2_EffSFandUncties_v1.root | |
|   PU reweight   | 2024 Data |  | |
|   | 2024 MC |  | |
|   | 2023 Data |   | |
|   | 2023 MC |   | |
|   | 2022 Data |   | |
|   | 2022 MC |   | https://github.com/cms-sw/cmssw/blob/CMSSW_12_6_X/SimGeneral/MixingModule/python/Run3_2022_LHC_Simulation_10h_2h_cfi.py |
|   | 2018 Data | data/pileup/puData2018_UL_withVar.root | |
|   | 2018 MC | data/pileup/mcPileup2018.root  | https://github.com/cms-sw/cmssw/blob/CMSSW_12_6_X/SimGeneral/MixingModule/python/mix_2018_25ns_UltraLegacy_PoissonOOTPU_cfi.py |
|   | 2017 Data | data/pileup/puData2017_UL_withVar.root| |
|   | 2017 MC | data/pileup/mcPileup2017.root  | https://github.com/cms-sw/cmssw/blob/CMSSW_12_6_X/SimGeneral/MixingModule/python/mix_2017_25ns_UltraLegacy_PoissonOOTPU_cfi.py |
|   | 2016 Data | data/pileup/puData2016_UL_withVar.root | |
|   | 2016 MC | data/pileup/pileup_profile_Summer16.root  | https://github.com/cms-sw/cmssw/blob/CMSSW_12_6_X/SimGeneral/MixingModule/python/mix_2016_25ns_UltraLegacy_PoissonOOTPU_cfi.py |
| Muon ID SF  | 2024 |  | |
|   | 2023 |   | https://twiki.cern.ch/twiki/bin/view/CMS/MuonRun3_2023 |
|   | 2022postEE | data/muon_sf/year2022postEE/ScaleFactors_Muon_Z_ID_ISO_2022_EE_schemaV2.json  | https://twiki.cern.ch/twiki/bin/view/CMS/MuonRun32022  |
|   |   |  | https://gitlab.cern.ch/cms-muonPOG/muonefficiencies/-/tree/master/Run3/2022_EE/2022_Z |
|   | 2022preEE | data/muon_sf/year2022preEE/ScaleFactors_Muon_Z_ID_ISO_2022_schemaV2.json  | https://twiki.cern.ch/twiki/bin/view/CMS/MuonRun32022  |
|   |   |  | https://gitlab.cern.ch/cms-muonPOG/muonefficiencies/-/tree/master/Run3/2022/2022_Z | |   | 2018  | data/muon_sf/year2018/MuonSF_Run2018_UL_ID.root | |
|   | 2017 | data/muon_sf/year2017/MuonSF_Run2017_UL_ID.root  | |
|   | 2016postVFP | data/muon_sf/year2016/MuonSF_Run2016_UL_ID.root  | |
|   | 2016preVFP | data/muon_sf/year2016/MuonSF_Run2016_UL_HIPM_ID.root  | |
| Muon ISO SF  | 2024 |  | |
|   | 2023 |   | https://twiki.cern.ch/twiki/bin/view/CMS/MuonRun3_2023 |
|   | 2022postEE | data/muon_sf/year2022postEE/ScaleFactors_Muon_Z_ID_ISO_2022_EE_schemaV2.json  | https://twiki.cern.ch/twiki/bin/view/CMS/MuonRun32022  |
|   |   |  | https://gitlab.cern.ch/cms-muonPOG/muonefficiencies/-/tree/master/Run3/2022_EE/2022_Z |
|   | 2022preEE | data/muon_sf/year2022preEE/ScaleFactors_Muon_Z_ID_ISO_2022_schemaV2.json  | https://twiki.cern.ch/twiki/bin/view/CMS/MuonRun32022  |
|   |   |  | https://gitlab.cern.ch/cms-muonPOG/muonefficiencies/-/tree/master/Run3/2022/2022_Z |
|   | 2018  | data/muon_sf/year2018/MuonSF_Run2018_UL_ISO.root | |
|   | 2017 | data/muon_sf/year2017/MuonSF_Run2017_UL_ISO.root  | |
|   | 2016postVFP | data/muon_sf/year2016/MuonSF_Run2016_UL_ISO.root  | |
|   | 2016preVFP | data/muon_sf/year2016/MuonSF_Run2016_UL_HIPM_ISO.root  | |
| Muon Trig SF  | 2024 |  | |
|   | 2023 |   | https://twiki.cern.ch/twiki/bin/view/CMS/MuonRun3_2023 |
|   | 2022postEE |   | https://twiki.cern.ch/twiki/bin/view/CMS/MuonRun32022 |
|   |   |  | https://gitlab.cern.ch/cms-muonPOG/muonefficiencies/-/tree/master/Run3/2022_EE/2022_Z/HLT/json |
|   | 2022preEE |   | https://twiki.cern.ch/twiki/bin/view/CMS/MuonRun32022 |
|   |   |  | https://gitlab.cern.ch/cms-muonPOG/muonefficiencies/-/tree/master/Run3/2022/2022_Z/HLT/json |
|   | 2018  | data/muon_sf/mu2018/EfficienciesStudies_2018_trigger_EfficienciesAndSF_2018Data_BeforeMuonHLTUpdate.root | |
|   | 2017 | data/muon_sf/mu2017/EfficienciesAndSF_RunBtoF_Nov17Nov2017.root | |
|   | 2016postVFP | data/muon_sf/mu2016/EfficienciesStudies_2016_trigger_EfficienciesAndSF_RunGtoH.root  | |
|   | 2016preVFP | data/muon_sf/mu2016/EfficienciesStudies_2016_trigger_EfficienciesAndSF_RunBtoF.root | |
| B Tag SF  | 2024 |  | |
|   | 2023 |   | |
|   | 2022 |   | |
|   | 2018  | data/btag/DeepCSV_106XUL18SF.csv | |
|   | 2017 | data/btag/DeepCSV_106XUL17SF.csv  | |
|   | 2016 | data/btag/DeepCSV_2016LegacySF_V1.csv | |
| JER  | 2024 |  | |
|   | 2023 |   | https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/JME|
|   | 2022 |   | https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/JME|
|   | 2018  | data/jec/Fall17_V3_MC_PtResolution_AK4PFchs.jr.txt | |
|   | 2017 | data/jec/Autumn18_V7_MC_PtResolution_AK4PFchs.jr.txt | |
|   | 2016 | data/jec/Summer16_25nsV1_MC_PtResolution_AK4PFchs.jr.txt | |
| Jet Resolution SF  | 2024 |  | |
|   | 2023 |   | https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/JME|
|   | 2022 |   | https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/JME|
|   | 2018  | data/jec/Autumn18_V7_MC_SF_AK4PFchs.jersf.txt | |
|   | 2017 | data/jec/Fall17_V3_MC_SF_AK4PFchs.jersf.txt | |
|   | 2016 | data/jec/Summer16_25nsV1_MC_SF_AK4PFchs.jersf.txt | |
| JEC  | 2024 Data |  | |
|   | 2024 MC |  | |
|   | 2023 Data |   | https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/JME|
|   | 2023 MC |   | https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/JME|
|   | 2022 Data |   | https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/JME|
|   | 2022 MC |   | https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/JME|
|   | 2018 Data | data/jec/Autumn18_Run{A, B, C and D}_V19_DATA_Uncertainty_AK4PFchs.junc.txt | |
|   | 2018 MC | data/jec/Autumn18_V19_MC_{L1FastJet, L2Relative, L3Absolute, L2L3Residual and Uncertainty}_AK4PFchs.jec.txt | |
|   | 2017 Data | data/jec/Fall17_17Nov2017{B, C , DE, and F}_V32_DATA_Uncertainty_AK4PFchs.junc.txt | |
|   | 2017 MC | data/jec/Fall17_17Nov2017_V32_MC_{L1FastJet, L2Relative, L3Absolute L2L3Residual and Uncertainty}_AK4PFchs.jec.txt | |
|   | 2016 Data | data/jec/Summer16_07Aug2017{BCD, EF and GH}_V11_DATA_Uncertainty_AK4PFchs.junc.txt | |
|   | 2016 MC | data/jec/Summer16_07Aug2017_V11_MC_{L1FastJet, L3Absolute, L2L3Residual and Uncertainty}_AK4PFchs.jec.txt | |
| Jet Map Veto  | 2024 |  | |
|   | 2023 |   | https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/JME|
|   | 2022 |   | https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/JME|

## Table of samples for Run2 and Run3:

| Sample | Year | Path |
| ------------- | ------------- | ------------- |
| Data  | 2024 |  |
| Data  | 2023 | /Muon/Run2023E-PromptReco-v1/NANOAOD |
| Data  | 2022postEE | /Muon/Run2022{E,F}-16Dec2023-v1/NANOAOD |
|   |  | /Muon/Run2022G-19Dec2023-v2/NANOAOD |
| Data  | 2022preEE | /Muon/Run2022{C,D}-16Dec2023-v1/NANOAOD |
| Data  | 2018 | /SingleMuon/Run2018{A,B,C}-UL2018_MiniAODv2_NanoAODv9-v2/NANOAOD |
|  |  | /SingleMuon/Run2018D-UL2018_MiniAODv2_NanoAODv9-v1/NANOAOD |
| Data  | 2017 | /SingleMuon/Run2017{B,C,D,E,F}-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD  |
| Data  | 2016postVFP | /SingleMuon/Run2016{F,G,H}-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD |
| Data  | 2016preVFP | /SingleMuon/Run2016{B-ver2_,C-,D-,E-,F-}HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD  |
| ggH signal  | 2024 |  |
| ggH signal  | 2023 |  |
| ggH signal  | 2022postEE | /GluGluHto2Mu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM |
| ggH signal  | 2022preEE | /GluGluHto2Mu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM  |
| ggH signal  | 2018 | /GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM |
| ggH signal  | 2017 | /GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM |
| ggH signal  | 2016postVFP | /GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM |
| ggH signal  | 2016preVFP | /GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM|
| VBF signal  | 2024 |  |
| VBF signal  | 2023 |  |
| VBF signal  | 2022postEE | /VBFHto2Mu_M-125_TuneCP5_withDipoleRecoil_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM |
| VBF signal  | 2022preEE | /VBFHto2Mu_M-125_TuneCP5_withDipoleRecoil_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM |
| VBF signal  | 2018 | /VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM |
| VBF signal  | 2017 | /VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM |
| VBF signal  | 2016postVFP | /VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM |
| VBF signal  | 2016preVFP | /VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM |
| DY + 2J | 2022postEE | /DYto2Mu-2Jets_MLL-105To160_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM |
| DY + 2J | 2022preEE | /DYto2Mu-2Jets_MLL-105To160_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM |
| DYm100To200  | 2024 |  |
| DYm100To200  | 2023 |  |
| DYm100To200  | 2022postEE | /DYto2Mu_MLL-120to200_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM (amcatnlo not available) |
| DYm100To200  | 2022preEE | /DYto2Mu_MLL-120to200_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM (amcatnlo not available) |
| DYm100To200  | 2018 | /DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM |
| DYm100To200  | 2017 | /DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM |
| DYm100To200  | 2016postVFP | /DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM |
| DYm100To200  | 2016preVFP | /DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM |
| DYm50  | 2024 |  |
| DYm50  | 2023 | /DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8/Run3Summer23NanoAODv12-PilotMuonHits_130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM |
| DYm50  | 2022postEE | /DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v5-v2/NANOAODSIM |
|   |  |  OR /DYto2Mu_MLL-50to120_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM |
| DYm50  | 2022preEE | /DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM |
|   |  | OR /DYto2Mu_MLL-50to120_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM  |
| DYm50  | 2018 | /DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X*/NANOAODSIM |
| DYm50  | 2017 | /DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM |
| DYm50  | 2016postVFP | /DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM |
| DYm50  | 2016preVFP | /DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM |
| Top anti-Top pair | 2024 |  |
| Top anti-Top pair | 2023 | /TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM |
| Top anti-Top pair | 2022postEE | /TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM |
| Top anti-Top pair | 2022preEE | /TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM |
| Top anti-Top pair | 2018 | /TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM |
| |  | /TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM |
| Top anti-Top pair | 2017 | /TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM |
| |  | /TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM |
| Top anti-Top pair | 2016postVFP | /TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM |
| |  | /TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM |
| Top anti-Top pair | 2016preVFP | /TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM |
| |  | /TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM |
| Single Top  | 2024 |  |
| Single Top  | 2023 |  |
| Single Top  | 2022postEE |  |
| Single Top  | 2022preEE |  |
| Single Top  | 2018 | /ST_tW_{top,antitop}_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM |
| Single Top  | 2017 | /ST_tW_{top,antitop}_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM |
| Single Top  | 2016postVFP | /ST_tW_{top,antitop}_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM |
| Single Top  | 2016preVFP | /ST_tW_{top,antitop}_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM |
| Diboson  | 2024 |  |
| Diboson  | 2023 | /WWto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4/NANOAODSIM |
|   |  | /WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM" |
|   |  | /WZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM |
|   |  | /WZtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM |
|   |  |  /ZZ_TuneCP5_13p6TeV_pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM |
| Diboson  | 2022postEE | /WWto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM |
|   |  | /WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM |
|   |  | /WZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM |
|   |  | /WZtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM |
|   |  | /ZZ_TuneCP5_13p6TeV_pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM |
| Diboson  | 2022preEE | /WWto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM |
|   |  | /WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM |
|   |  | /WZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM |
|   |  | /WZtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM |
|   |  | /ZZ_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM |
| Diboson  | 2018 | "/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM |
|   |  | /WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM |
|   |  | /WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM |
|   |  | /WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM |
|   |  |  /ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM |
| Diboson  | 2017 | /WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM  |
|   |  | /WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM |
|   |  | /WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM |
|   |  | /WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM |
|   |  | /ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM |
| Diboson  | 2016postVFP | /WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM |
|   |  | /WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM |
|   |  | /WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM |
|   |  | /WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM |
|   |  | /ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM |
| Diboson  | 2016preVFP |  /WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM |
|   |  |  /WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM |
|   |  |  /WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM |
|   |  |  /WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM |
|   |  |  /ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM |
| Electro Weak  | 2024 |  |
| Electro Weak  | 2023 |  |
| Electro Weak  | 2022postEE |  |
| Electro Weak  | 2022preEE |  |
| Electro Weak  | 2018 | /EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM |
| Electro Weak  | 2017 | /EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM |
| Electro Weak  | 2016postVFP | /EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM |
| Electro Weak  | 2016preVFP | /EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM |
