# 🐍 Copperhead V2, - Columnar Parallel Pythonic framEwork for Run3 H&rarr;µµ Decay search

## setup

```bash
git clone https://github.com/green-cabbage/copperheadV2.git
cd copperheadV2
source setup_env.sh
# Run first two column of DaskGatewaySLURM.ipynb to start the DASK.
```

## Step - 1: Skim, $Z_{p_T}$ correction


## Step - 2: Get Z-pT reweight

1. Get weight, data/DY, in the jet multiplicity bins
   * Code located in `data/Zpt_rewgt/fitting/do_fitting.py`
   * It extracts p_T(mumu) from data and DY in the Z-peak region. Also, the ratio of data/dy in nJet bins. Then save them as .root file
2. Fit the ratio data/dy: `do_f_test.py`
   * From here, get the polynomial that fits our data best as per f-test.
3. Run `get_goodnessofFit.py`: Fits and saves the polynomial info in the YAML file.
4. How to save the weight into the skimmed file:
   - Run: `work/users/shar1172/HMuMu/copperheadV2/src/copperhead_processor.py`
   - The function that saves weight in above script is `getZptWgts()`