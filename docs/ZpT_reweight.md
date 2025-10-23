# Introduction

There are three steps to run:
1. `step-1`: get the root file with the histogram that contains the ratio of data and MC (DY) in the z-peak region.
2. `step-2`: Use the f-test to determine the best polynomial order for the fit.
   1. **NOTE** : Before running this step, make sure to update bins and ranges in [bin_definitions.py](data/zpt_rewgt/fitting/bin_definitions.py)
3. `step-3`: Use the best polynomial order to fit the data and get the reweighting factors.

# Technical Details

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

- Code location: [save_SF_rootFiles.py](data/zpt_rewgt/fitting/save_SF_rootFiles.py)


## Details of step-2
- Code location: [do_f_test.py](data/zpt_rewgt/fitting/do_f_test.py)

## Details of step-3
- Code location: [get_polyFit.py](data/zpt_rewgt/fitting/get_polyFit.py)


**At the end don't forgot to commit the two YAML files. First file contains the details of the fit and the second file contains the reweighting factors.**

# References/Important links
