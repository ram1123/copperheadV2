---
title: Event by Event Mass Resolution Calibration
---

# Introduction

# Input Information
<!--
Z mass: 91.1880 GeV from PDG -->
<!-- Natural Z width 2.4955 GeV from PDG -->

- Z boson mass: 91.1880 GeV (from PDG)
- Natural Z width: 2.4955 GeV (from PDG)
- H boson mass: 125.200 GeV (from PDG)
- H boson width: 3.7 MeV (from PDG)


## Event by Event (EBE) mass resolution calibration


- Run the script [`getCalibrationFactor.py`](../src/lib/ebeMassResCalibration/getCalibrationFactor.py) to get the calibration factor for the EBE mass resolution.
- The script will generate a JSON file with the calibration factors for different categories.
- Copy the generated JSON file to the path `data/res_calib/` and update the path and name of this JSON file in the config file: `configs/parameters/correction_filelist.yaml`.

### How to run

**NOTE:** Need to update the path of the input parquet files in the code [src/lib/ebeMassResCalibration/getCalibrationFactor.py](../src/lib/ebeMassResCalibration/getCalibrationFactor.py) before running.

<!-- ```bash
python src/lib/ebeMassResCalibration/getCalibrationFactor.py --years "2018"
python src/lib/ebeMassResCalibration/getCalibrationFactor.py --years "2018" --fixCat "30-45_EE" --backup
# python src/lib/ebeMassResCalibration/getCalibrationFactor.py --years "2018" --isMC
# time(python src/lib/ebeMassResCalibration/getCalibrationFactor.py --isMC  --years "2016preVFP")
# time(python src/lib/ebeMassResCalibration/getCalibrationFactor.py --isMC --validate --years "2016preVFP")
# time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April19_NanoV12_UpdatedMassCalib -y 2018 -m all)
``` -->

```bash
bash stage1_loop_Improved.sh -v 12 -c configs/datasets/dataset_nanoAODv12.yaml -m "calib"
```

- To adjust the fitting one can change the parameters in the script `src/lib/ebeMassResCalibration/getCalibrationFactor.py`
- Once we get the json file, copy it to the path `data/res_calib/` and update the path and name of this json file in the config file: `configs/parameters/correction_filelist.yaml`
- Then re-run stage-1 to get the updated mass calibration. **REMEMBER TO SWITCH ON THE BSC OPTION**.

#### Closure test

For the closure test, to avoid the bias introduced by the original binning scheme, we use an alternative binning scheme. The bin edges are defined as

```python
CLOSURE_BINS = [
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.0),
    (1.0, 1.1),
    (1.1, 1.2),
    (1.3, 1.4),
    (1.4, 1.5),
    (1.5, 1.7),
    (1.7, 2.0),
    (2.0, 2.5),
    (2.5, 3.5),
]
```

- To run the closure test use the following command:

```bash
python src/lib/ebeMassResCalibration/getCalibrationFactor.py --nanoAODv 12 --years "2018" --ifbinned  --closure_test
```



## RooFit Information

- If using GPU, then setup environment using:

   ```bash
   source /cvmfs/sft.cern.ch/lcg/views/LCG_106b_cuda/x86_64-el8-gcc11-opt/setup.sh
   ```

- To get the chi2/NDF use this method:

   ```python
   new_nfree_params = fit_result.floatParsFinal().getSize()
   chi2_ndf = frame.chiSquare("model_bsOn", "hist_bsOn", new_nfree_params)
   ```

- To get the fit result use this method:


## Things to note

1. DCB convoluted with BW function is not same as BW convoluted with DCB function.
2. Fit Z-peak with BW convoluted with DCB function.
3. Fit H-peak with DCB funtion
