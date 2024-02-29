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

## Table of correction and its weight file locations:

| Correction | Year | Location |
| ------------- | ------------- | ------------- |
| Rochester | 2018 | data/roch_corr/RoccoR2018.txt  |
|   | 2018 | data/roch_corr/RoccoR2018.txt  |
