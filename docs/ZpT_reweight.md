---
title: Z pT Reweighting
---

# Overview

The Z-pT reweighting flow derives a DY correction in the Z-peak region, stores the fitted function parameters in YAML, and then applies the corresponding event weight in stage-1. In the current code, the derivation uses compacted stage-1 parquet outputs when they exist and falls back to `f1_0` otherwise.

The main scripts are:

1. [save_SF_rootFiles.py](../src/copperhead/zpt_rewgt/derive/save_SF_rootFiles.py)
2. [do_f_test.py](../src/copperhead/zpt_rewgt/derive/do_f_test.py)
3. [get_polyFit.py](../src/copperhead/zpt_rewgt/derive/get_polyFit.py)
4. [validation_plotter_unified.py](../plotter/validation_plotter_unified.py)

# Inputs

Before running the Z-pT derivation, you need:

1. A completed stage-1 run for the target year.
2. A compacted stage-1 output if possible.
3. The correct DY sample choice for the year:
   - Run-2: usually `MiNNLO` or `aMCatNLO`
   - Run-3: usually `INCamcatnloFXFX`
   - 2024 special case in the code uses `dyTo2Mu_M-50_aMCatNLO`

# Step 0: Save Data And DY Histograms

This step reads stage-1 parquet files, removes any already-applied Z-pT weight from DY, makes `Data`, `DY`, and `Data / DY` histograms in each jet bin, and saves them into ROOT workspaces.

```bash
bash stage1_loop_Improved.sh \
  -c configs/datasets/dataset_nanoAODv12_run3.yaml \
  -v 12 \
  -l Run3_nanoAODv12_myLabel \
  -y "2022postEE" \
  -m zpt_fit0 \
  -n 0 \
  -k -i 0
```

Output location:

`validation/zpt_rewgt/<label>/<dy_sample>/<year>/`

Important details:

1. The script looks in `stage1_output/<year>/compacted` first, then `stage1_output/<year>/f1_0`.
2. It always divides out `separate_wgt_zpt` before deriving new weights, so the fit starts from the unweighted DY shape.
3. It loops over `njet = 0, 1, >=2` internally.

# Step 1: Run The F-test

This step rebins the `Data / DY` histogram using [bin_definitions.py](../src/copperhead/zpt_rewgt/derive/bin_definitions.py), scans polynomial orders, and writes the preferred orders for the low- and mid-pT regions into `zpt_fit_config.yaml`.

```bash
bash stage1_loop_Improved.sh \
  -c configs/datasets/dataset_nanoAODv12_run3.yaml \
  -v 12 \
  -l Run3_nanoAODv12_myLabel \
  -y "2022postEE" \
  -m zpt_fit1 \
  -n 0
```

Output location:

`validation/zpt_rewgt/<label>/<dy_sample>/<year>/fTest_<save_postfix>/`

Files written here include:

1. `zpt_fit_config.yaml`
2. `fTest_results_<year>_njet<n>_...txt`
3. diagnostic fit PDFs

Important details:

1. The low-pT (`f0`) and mid-pT (`f1`) fits are stored separately.
2. If the fit choice looks unstable, adjust the ranges or custom bins in [bin_definitions.py](../src/copperhead/zpt_rewgt/derive/bin_definitions.py) and rerun.

# Step 2: Build The Final Piecewise Function

This step reads the `zpt_fit_config.yaml` from step 1, performs the final piecewise fit, produces goodness-of-fit plots, and saves the final function coefficients into the year- and jet-dependent YAML used later by the processor.

```bash
bash stage1_loop_Improved.sh \
  -c configs/datasets/dataset_nanoAODv12_run3.yaml \
  -v 12 \
  -l Run3_nanoAODv12_myLabel \
  -y "2022postEE" \
  -m zpt_fit2 \
  -n 0
```

Main outputs:

1. `validation/zpt_rewgt/<label>/<dy_sample>/<year>/gof_<save_postfix>/`
2. `validation/zpt_rewgt/<label>/<dy_sample>/zpt_rewgt_params_<dy_sample>.yaml`

The final YAML stores:

1. fit orders
2. fit ranges
3. bin edges
4. function coefficients and errors
5. high-pT linear tail parameters

# Step 3: Copy Or Register The Final YAML

Once the final YAML looks good, copy or merge it into the repository location used by the analysis code for your production setup. That YAML is what [copperhead_processor.py](../src/copperhead_processor.py) reads when stage-1 computes `separate_wgt_zpt`.

# Step 4: Validate The Weight

The actual validation workflow is done with [validation_plotter_unified.py](../plotter/validation_plotter_unified.py), not the older dedicated Z-pT validation scripts.

The validation is simply:

1. make the normal plots with the stored Z-pT weight applied
2. make the same plots again with `--remove_zpt_weights`
3. compare the two outputs
4. repeat that for jet multiplicities:
   - `inclusive`
   - `0`
   - `1`
   - `2` meaning `>=2`

Example commands:

```bash
python plotter/validation_plotter_unified.py \
  -y 2022postEE \
  --load /work/projects/hmm/$USER/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_myLabel/stage1_output/2022postEE/f1_0 \
  --use-compacted compacted \
  --save_path validation/figs/Run3_nanoAODv12/Run3_nanoAODv12_myLabel/VBFfilter_False \
  -cat nocat \
  --region z-peak \
  --njets inclusive
```

```bash
python plotter/validation_plotter_unified.py \
  -y 2022postEE \
  --load /work/projects/hmm/$USER/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_myLabel/stage1_output/2022postEE/f1_0 \
  --use-compacted compacted \
  --save_path validation/figs/Run3_nanoAODv12/Run3_nanoAODv12_myLabel/VBFfilter_False \
  -cat nocat \
  --region z-peak \
  --njets inclusive \
  --remove_zpt_weights
```

Then rerun the same pair for:

1. `--njets 0`
2. `--njets 1`
3. `--njets 2`

In this plotter, `--njets 2` means `>= 2` jets.

Validation output path:

`<save_path>/<year>/mplhep/Reg_<region>/Cat_<category>/njet_<inclusive|0|1|2>/<default_zpt_weights|no_zpt_weights>/`

# Snakemake Mapping

If you use [workflow/Snakefile](../workflow/Snakefile), the relevant rules are:

1. `zpt0`
2. `zpt1`
3. `zpt2`
4. `plots`

Those rules call `stage1_loop_Improved.sh` with:

1. `-m zpt_fit0`
2. `-m zpt_fit1`
3. `-m zpt_fit2`
4. plotting with and without `--remove_zpt_weights`

The workflow now passes gateway flags through those rules correctly.

# Common Checks

If the validation looks wrong, check these first:

1. The DY sample choice passed with `--dy_sample` matches the year and production campaign.
2. `zpt_fit_config.yaml` exists under `fTest_<save_postfix>/`.
3. The final `zpt_rewgt_params_<dy_sample>.yaml` contains the target year and all three jet bins.
4. The validation is reading the intended compacted or `f1_0` stage-1 directory.
5. The `--remove_zpt_weights` plots are really dividing out `separate_wgt_zpt`.
