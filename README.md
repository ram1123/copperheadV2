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

This would start the GRID certificate password prompt, and then once given the password, the conda env would be activated. Once voms proxy and conda env is activated, we can start on the tutorial.ipynb

## Table of correction weight file locations:

| Correction | Year (Data/MC) | Local Location | Central Link |
| ------------- | ------------- | ------------- |------------- |
|   Rochester Correction  | 2024 |  | |
|   | 2023 |   | |
|   | 2022 |   | |
|   | 2018 | data/roch_corr/RoccoR2018.txt  | |
|   | 2017 | data/roch_corr/RoccoR2017.txt  | |
|   | 2016 | data/roch_corr/RoccoR2016.txt  | |
| Zpt SF | 2024 |  | |
|   | 2023 |   | |
|   | 2022 |   | |
|   | 2018, 2017, 2016 | data/zpt_weights.histo.json | |
| NNLOPS (ggH) | 2024 |  | |
|   | 2023 |   | |
|   | 2022 |   | |
|   | 2018, 2017, 2016 | data/NNLOPS_reweight.root | |
|   Lumi Mask  | 2024 |  | |
|   | 2023 |   | |
|   | 2022 |   | |
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
|   | 2018, 2017, 2016 | data/JetPUID_106XTraining_ULRun2_EffSFandUncties_v1.root | |
|   PU reweight   | 2024 Data |  | |
|   | 2024 MC |  | |
|   | 2023 Data |   | |
|   | 2023 MC |   | |
|   | 2022 Data |   | |
|   | 2022 MC |   | |
|   | 2018 Data | data/pileup/puData2018_UL_withVar.root | |
|   | 2018 MC | data/pileup/mcPileup2018.root  | |
|   | 2017 Data | data/pileup/puData2017_UL_withVar.root| |
|   | 2017 MC | data/pileup/mcPileup2017.root  | |
|   | 2016 Data | data/pileup/puData2016_UL_withVar.root | |
|   | 2016 MC | data/pileup/pileup_profile_Summer16.root  | |
| Muon ID SF | 2018  | data/muon_sf/year2018/MuonSF_Run2018_UL_ID.root | |
|   | 2017 | data/muon_sf/year2017/MuonSF_Run2017_UL_ID.root  | |
|   | 2016postVFP | data/muon_sf/year2016/MuonSF_Run2016_UL_ID.root  | |
|   | 2016preVFP | data/muon_sf/year2016/MuonSF_Run2016_UL_HIPM_ID.root  | |
| Muon ISO SF | 2018  | data/muon_sf/year2018/MuonSF_Run2018_UL_ISO.root | |
|   | 2017 | data/muon_sf/year2017/MuonSF_Run2017_UL_ISO.root  | |
|   | 2016postVFP | data/muon_sf/year2016/MuonSF_Run2016_UL_ISO.root  | |
|   | 2016preVFP | data/muon_sf/year2016/MuonSF_Run2016_UL_HIPM_ISO.root  | |
| Muon Trig SF | 2018  | data/muon_sf/mu2018/EfficienciesStudies_2018_trigger_EfficienciesAndSF_2018Data_BeforeMuonHLTUpdate.root | |
|   | 2017 | data/muon_sf/mu2017/EfficienciesAndSF_RunBtoF_Nov17Nov2017.root | |
|   | 2016postVFP | data/muon_sf/mu2016/EfficienciesStudies_2016_trigger_EfficienciesAndSF_RunGtoH.root  | |
|   | 2016preVFP | data/muon_sf/mu2016/EfficienciesStudies_2016_trigger_EfficienciesAndSF_RunBtoF.root | |
| B Tag SF | 2018  | data/btag/DeepCSV_106XUL18SF.csv | |
|   | 2017 | data/btag/DeepCSV_106XUL17SF.csv  | |
|   | 2016 | data/btag/DeepCSV_2016LegacySF_V1.csv | |
| JER | 2018  | data/jec/Fall17_V3_MC_PtResolution_AK4PFchs.jr.txt | |
|   | 2017 | data/jec/Autumn18_V7_MC_PtResolution_AK4PFchs.jr.txt | |
|   | 2016 | data/jec/Summer16_25nsV1_MC_PtResolution_AK4PFchs.jr.txt | |
| JER SF | 2018  | data/jec/Autumn18_V7_MC_SF_AK4PFchs.jersf.txt | |
|   | 2017 | data/jec/Fall17_V3_MC_SF_AK4PFchs.jersf.txt | |
|   | 2016 | data/jec/Summer16_25nsV1_MC_SF_AK4PFchs.jersf.txt | |
| JEC | 2018 Data | data/jec/Autumn18_Run{A, B, C and D}_V19_DATA_Uncertainty_AK4PFchs.junc.txt | |
|   | 2017 Data | data/jec/Fall17_17Nov2017{B, C , DE, and F}_V32_DATA_Uncertainty_AK4PFchs.junc.txt | |
|   | 2016 Data | data/jec/Summer16_07Aug2017{BCD, EF and GH}_V11_DATA_Uncertainty_AK4PFchs.junc.txt | |
|  | 2018 MC | data/jec/Autumn18_V19_MC_{L1FastJet, L2Relative, L3Absolute, L2L3Residual and Uncertainty}_AK4PFchs.jec.txt | |
|   | 2017 MC | data/jec/Fall17_17Nov2017_V32_MC_{L1FastJet, L2Relative, L3Absolute L2L3Residual and Uncertainty}_AK4PFchs.jec.txt | |
|   | 2016 MC | data/jec/Summer16_07Aug2017_V11_MC_{L1FastJet, L3Absolute, L2L3Residual and Uncertainty}_AK4PFchs.jec.txt | |

