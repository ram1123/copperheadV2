#Documentation

# Doing scans
We first assume that bias fit test from copperheadV2 is done. Look at the directory's doc for instructions.

## Step0: copy the workspace and datacard
Copy the relevant datacard to this directory. Moreover, copy pre-fit workspace outputs from running `copperheadV2/validation/ggH/bias_test/run_script.sh` to `funcCandidate_workspace`, and copy stage3 pre-fit workspace outputs from from running `copperheadV2/stage2_sh.sh` to `corePdf_workspace`. The datacards we will be using is save here, but you can update the sysmtematics with the latest one (only the workspace root file paths have to be changed to appropriate paths)


## Step1: get bias test for core-PDF for each of the individual fit function candidate fit to h-sidebands data

This would take lots of computing resources, so we use slurm. The script to use is:

```bash
sh slurm_wrapper.sh

```
then extract the results via:
```bash
python aggregate_bias_pull_fitDiagnostics.py
```


