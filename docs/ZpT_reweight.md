---
title: Z pT Reweighting
---

# Introduction

In DY production, the transverse momentum (pT) of the Z boson is not perfectly modelled in Monte Carlo (MC) simulations. This mismodelling can lead to discrepancies between data and MC in analyses that rely on accurate Z boson kinematics. To address this issue, we apply a reweighting procedure to correct the Z-pT distribution in DY MC to better match the observed data.

In our framework, we have implemented a systematic approach to derive the Z-pT reweighting factors. This involves three steps, they are:
1. Extract the Z-pT distributions from both data and DY MC in the Z-peak region.
2. Fit the ratio of data to MC Z-pT distributions using polynomial functions in three different pT ranges as the behavior of the ratio varies across these ranges. The rough boundaries are:
   - Low pT: 0 - 10 GeV
   - Medium pT: 10 - 100 GeV
   - High pT: 100 - 200 GeV
3. Once we get the three fit functions, we then sum them up to get the final reweighting function, its order and coefficients are stored in a YAML file. Finally, we apply these reweighting factors to the DY MC events based on their Z boson pT.

# DY pt mismodelling correction: Technical details

Several steps are needed to get the Z-pT weights.

**Step-1**: Obtain the histograms of Z-pT in data and MC, in `.root` format.
```bash
bash stage1_loop_Improved.sh  -l label_for_ntuple -y 2018 -m "zpt_fit0" -n 0
```

Arguments:
- `-l`: label for the ntuple.
- `-y`: year of the data.
- `-m`: mode to run, here it is `zpt_fit0` for step-1, which gets the histograms of Z-pT in data and MC.
- `-n`: jet bin to consider, `0` for zero jet bin, `1` for one jet bin and `2` for greater than equal to two jet bin.


**Step-2**: Fit the ratio histograms to determine the best polynomial order for each pT range.

```bash
bash stage1_loop_Improved.sh -l label_for_ntuple -y 2018 -m "zpt_fit1" -n 0
```

Arguments:
- `-l`: label for the ntuple.
- `-y`: year of the data.
- `-m`: mode to run, here it is `zpt_fit1` for step-2, which fits the ratio histograms to determine the best polynomial order for each pT range.
- `-n`: jet bin to consider, `0` for zero jet bin, `1` for one jet bin and `2` for greater than equal to two jet bin.


This step uses the f-test to determine the best polynomial order for the fit.

**NOTE** : Before running this step, make sure to update bins and ranges in [bin_definitions.py](../src/copperhead/zpt_rewgt/derive/bin_definitions.py). If the final fit in next step is not good enough, you may need to adjust the binning and ranges here.

**Step-3**: Now we calculate the Z-pT weights using the fit functions obtained in the previous step.

```bash
bash stage1_loop_Improved.sh -v 12 -l label_for_ntuple -y 2018  -m "zpt_fit2" -n 0
```

Arguments:
- `-v`: version of the output files.
- `-l`: label for the ntuple.
- `-y`: year of the data.
- `-m`: mode to run, here it is `zpt_fit2` for step-3, which obtains the final function for Z-pT reweighting.
- `-n`: jet bin to consider, `0` for zero jet bin, `1` for one jet bin and `2` for greater than equal to two jet bin.

Some important notes:
1. If the fit quality in step-3 is not good enough, you may need to go back to step-2 and readjust the boundaries of the three functions, or the bin widths in [bin_definitions.py](../src/copperhead/zpt_rewgt/derive/bin_definitions.py).
1. If the boundaries and bin widths looks fine, but not the fit quality, you may need to manually set the polynomial orders (to one higher order, generally it works) in the the `zpt_fit_config.yaml` file generated in step-2, and then rerun step-3.


**Step-4**: Finally, commit the YAML files having the Z-pT reweighting function parameters to the repository for future use in analysis. This YAML file saves the polynomial orders and coefficients for each pT range, for each year and jet bin. Furthermore, it also contains the number of bins and ranges used in the derivation of the Z-pT weights.
