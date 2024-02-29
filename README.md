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

| Correction | Year (Data/MC) | Location |
| ------------- | ------------- | ------------- |
| Rochester | 2018 | data/roch_corr/RoccoR2018.txt  |
|   | 2017 | data/roch_corr/RoccoR2017.txt  |
|   | 2016 | data/roch_corr/RoccoR2016.txt  |
| Zpt | 2018, 2017, 2016 | data/zpt_weights.histo.json |
| Lumi Mask  | 2018 | data/lumimasks/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt  |
|   | 2017 | data/lumimasks/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt  |
|   | 2016 | data/lumimasks/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt  |
| Event by event Dimuon mass resolution | 2018 | /data/res_calib/res_calib_{Data or MC}_2018.root  |
|   | 2017 | /data/res_calib/res_calib_{Data or MC}_2017.root    |
|   | 2016 | /data/res_calib/res_calib_{Data or MC}_2016.root   |
| Jet PU ID | 2018, 2017, 2016 | data/JetPUID_106XTraining_ULRun2_EffSFandUncties_v1.root |
| PU ID | 2018 Data | data/pileup/puData2018_UL_withVar.root |
|   | 2018 MC | data/pileup/mcPileup2018.root  |
|   | 2017 Data | data/pileup/puData2017_UL_withVar.root|
|   | 2017 MC | data/pileup/mcPileup2017.root  |
|   | 2016 Data | data/pileup/puData2016_UL_withVar.root |
|   | 2016 MC | data/pileup/pileup_profile_Summer16.root  |

