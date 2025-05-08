# Introduction

# Technical Details

## Obtain reweighting factors
Before running the script `run_script.sh`, please update `label`, `year` and which step to run.

```bash
cd data/zpt_rewgt/fitting
bash run_script.sh
```

There are three steps to run:
1. `step1`: get the root file with the histogram that contains the ratio of data and MC (DY) in the z-peak region.
2. `step2`: Use the f-test to determine the best polynomial order for the fit.
3. `step3`: Use the best polynomial order to fit the data and get the reweighting factors.

# References/Important links
