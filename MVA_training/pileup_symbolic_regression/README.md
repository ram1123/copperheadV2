# PySR pileup regression

This directory contains the symbolic-regression workflow used to build simple jet-level pileup-rejection formulas with [PySR](https://ai.damtp.cam.ac.uk/pysr/v1.5.9/).

The training is meant for forward jets in the low-`pT` turn-on region:
- `25 <= pT < 50`
- `HE`: `2.5 <= |eta| < 3.0`
- `HF`: `|eta| >= 3.0`

Signed regions are trained separately:
- `HEpos`
- `HEneg`
- `HFpos`
- `HFneg`

## Main entrypoint

Run:

```bash
python MVA_training/pileup_symbolic_regression/run_pysr.py \
  -i "/path/to/part*.parquet" \
  -o validation/pySR/example_run \
  --features-yaml MVA_training/pileup_symbolic_regression/configs/features.yaml \
  --mode train
```

Important options:
- `--mode train`: train one symbolic model per region
- `--mode validate`: apply saved equations and thresholds
- `--mode rescan`: rescan working points without retraining
- `--pt-min`: lower `pT` bound for training, default `25`
- `--pt-turnoff`: upper `pT` bound for training, default `50`
- `--hs-eff`: target hard-scatter efficiency used to choose the threshold

## Inputs

The parquet input is expected to contain nominal jet branches for up to four jets, including:
- `jet*_eta_nominal`
- `jet*_pt_nominal`
- `jet*_hasMatchedGenJet_nominal`

Internally the code flattens the jet prefixes into a single jet table, derives extra features, and labels jets with:
- `y_hs = True` for hard-scatter jets
- `y_hs = False` for pileup jets

## Outputs

Training writes one summary JSON per region plus a global summary:
- `summary_HEpos.json`
- `summary_HEneg.json`
- `summary_HFpos.json`
- `summary_HFneg.json`
- `summary_all.json`

Each summary stores:
- the best symbolic equation
- the feature list used
- the threshold
- the threshold direction
- the achieved pileup rejection
- the valid `pT` range

These summaries are what the downstream PySR jet filtering uses.

## Helper modules

- [run_pysr.py](/Users/ramkrishnasharma/Documents/New%20project%203/copperheadV2/MVA_training/pileup_symbolic_regression/run_pysr.py): CLI entrypoint
- [pysrpu/pipeline.py](/Users/ramkrishnasharma/Documents/New%20project%203/copperheadV2/MVA_training/pileup_symbolic_regression/pysrpu/pipeline.py): training, validation, and rescan flow
- [pysrpu/regions.py](/Users/ramkrishnasharma/Documents/New%20project%203/copperheadV2/MVA_training/pileup_symbolic_regression/pysrpu/regions.py): eta-region definitions
- [configs/features.yaml](/Users/ramkrishnasharma/Documents/New%20project%203/copperheadV2/MVA_training/pileup_symbolic_regression/configs/features.yaml): region-dependent feature lists

## Plotters

- [plot_real_fake_jets_stack.py](/Users/ramkrishnasharma/Documents/New%20project%203/copperheadV2/MVA_training/pileup_symbolic_regression/plot_real_fake_jets_stack.py): stacked real/fake jet plots
- [corr_region_real_fake.py](/Users/ramkrishnasharma/Documents/New%20project%203/copperheadV2/MVA_training/pileup_symbolic_regression/corr_region_real_fake.py): 2D region correlation plots

Run the stacked real/fake jet plots with:

```bash
python MVA_training/pileup_symbolic_regression/plot_real_fake_jets_stack.py \
  -i "/path/to/part*.parquet" \
  -o validation/compare_real_fake/example \
  --apply-cleaning
```

Useful options:
- `--region HEpos`, `HEneg`, `HFpos`, `HFneg`, `HE`, `HF`, or `inclusive`
- `--normalize`
- `--apply-cleaning`

Run the 2D correlation plots with:

```bash
python MVA_training/pileup_symbolic_regression/corr_region_real_fake.py \
  -i "/path/to/part*.parquet" \
  -o validation/corr_example \
  --prefix jet1_ \
  --apply-cleaning
```

Typical prefixes are:
- `--prefix jet1_`
- `--prefix jet2_`
