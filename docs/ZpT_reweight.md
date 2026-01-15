---
title: Z pT Reweighting
---

# Introduction

## DY pt mismodelling correction

Seveal steps are needed to get the Z-pT weights.

**Step-1**: Obtain the histograms of Z-pT in data and MC.
```bash
bash stage1_loop_Improved.sh -v 12 -c configs/datasets/dataset_nanoAODv12.yaml -l label_for_ntuple -y 2018 -m "zpt_fit0" -n 0
```

**Step-2**: After several checks we decided that to fit the ratio of Z-pT distribution in data and MC using the three functions in low, medium and high pT range. So, in this step we fit the ratio histograms and find the polynomial degrees that fits best in each range.

```bash
bash stage1_loop_Improved.sh -v 12 -c configs/datasets/dataset_nanoAODv12.yaml -l label_for_ntuple -y 2018 -m "zpt_fit1" -n 0
```

Where the option:
- `-n`: specifies which jet bin to consider for fitting. `0` for zero jet bin, `1` for one jet bin and `2` for greater than equal to two jet bin.


**Step-3**: Now we calculate the Z-pT weights using the fit functions obtained in the previous step.

```bash
bash stage1_loop_Improved.sh -v 12 -c configs/datasets/dataset_nanoAODv12.yaml -l label_for_ntuple -y 2018  -m "zpt_fit2" -n 0
```


# OLD Notes

## Introduction

There are three steps to run:
1. `step-1`: get the root file with the histogram that contains the ratio of data and MC (DY) in the z-peak region.
2. `step-2`: Use the f-test to determine the best polynomial order for the fit.
   1. **NOTE** : Before running this step, make sure to update bins and ranges in [bin_definitions.py](../src/copperhead/zpt_rewgt/derive/bin_definitions.py)
3. `step-3`: Use the best polynomial order to fit the data and get the reweighting factors.


## Technical Details

As usual we can run all three steps from our centeralized script `stage1_loop_Improved.sh`:

```bash
bash stage1_loop_Improved.sh -l <Label> -y <Year> -m <StepToRun>
```

Where
- `<Label>` is the label for the run, using which is finds the stage1 output files.
- `<Year>` is the year of the data, e.g. `2017`, `2018`, `2016preVFP`, `2016postVFP`.
- `<StepToRun>` is the step to run, e.g. `zpt_fit`, `zpt_fit0`, `zpt_fit1`, `zpt_fit2`, `zpt_fit12`.
   - `zpt_fit` runs all three steps.
    - `zpt_fit0` runs the first step, which is to get the root file with the histogram that contains the ratio of data and MC (DY) in the z-peak region.
    - `zpt_fit1` runs the second step, which is to use the f-test to determine the best polynomial order for the fit. This step needs two additional arguments:
        - `--nbins`: number of bins in the z-peak region, e.g. `20`.
        - `--njet`: number of jets in the event, e.g. `0`, `1`, `2`.
        - `--outAppend`: a string to append to the output file name, e.g. `-v12`.
    - `zpt_fit2` runs the third step, which is to use the best polynomial order to fit the data and get the reweighting factors. This step needs the same additional arguments as `zpt_fit1`.

## Technical Details to improve fits


## Example commands

***Used by me to run the script***

- Generally, at first I get the root file with the histogram that contains the ratio of data and MC (DY) in the z-peak region, using `zpt_fit0`:

    ```bash
    time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_UpdatedQGL_17July -y "2018 2017 2016preVFP 2016postVFP" -m zpt_fit0)
    ```

- Then I run the second and third step for one year at a time, using `zpt_fit1` and `zpt_fit2` (or `zpt_fit12`):

    ```bash
    time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_UpdatedQGL_17July -y 2018 -m zpt_fit1 -n 0)
    time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_UpdatedQGL_17July -y 2018 -m zpt_fit2 -n 0)
    ```


## Details of step-1

- Code location: [save_SF_rootFiles.py](../src/copperhead/zpt_rewgt/derive/save_SF_rootFiles.py)


## Details of step-2
- Code location: [do_f_test.py](../src/copperhead/zpt_rewgt/derive/do_f_test.py)
## Details of step-3
- Code location: [get_polyFit.py](../src/copperhead/zpt_rewgt/derive/get_polyFit.py)


**At the end don't forgot to commit the two YAML files. First file contains the details of the fit and the second file contains the reweighting factors.**

# References/Important links
