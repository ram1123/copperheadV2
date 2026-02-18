# Introduction

# Technical Details

## Obtain DNN data/MC plots
Before running the script `run_script.sh`, please update `label`, `year` and `mva_name`. Furthermore, we assume that stage2 is run for VBF category, which saves the stage2 histograms in: `/depot/cms/users/yun79/hmm/copperheadV1clean/{label}/stage2_histograms/score_{mva_name}/{year_param}/`
if region is specified as `h-peak`, the data histogram is automatically blinded.

```bash
cd ./validation/ggH/categorization
bash run_script.sh
```

## Running bias test

```bash
cd ./validation/ggH/bias_test
bash run_script.sh
```
## Getting NLL scan curves
copy output root files from `./validation/ggH/bias_test/run_script.sh` to `HMuMuCombine/NLL_scan` (github link:"https://github.com/green-cabbage/HMuMuCombine/tree/master") and paste them to my_workspace and then follow the instructions in `HMuMuCombine/NLL_scan/README.md`


## Getting core function biass test values
copy output root files from `./validation/ggH/bias_test/run_script.sh` to `HMuMuCombine/coreFuncScan` (github link:"https://github.com/green-cabbage/HMuMuCombine/tree/master") and paste them to my_workspace and then follow the instructions in `HMuMuCombine/coreFuncScan/README.md`



# References/Important links
