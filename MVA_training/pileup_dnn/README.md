# PU/HS Jet DNN

This workflow trains flat-input DNN classifiers for pileup-jet rejection using
stage-1 parquet output.

By default it trains two separate models:

- `HE`: `2.5 <= |eta| < 3.0`
- `HF`: `|eta| >= 3.0`

Only jets in the turn-on region are used for training:

- `25 <= pT < 50 GeV`

The label is read from `jet*_hasMatchedGenJet_nominal`:

- `1`: hard-scatter jet
- `0`: pileup jet

The DNN input features are restricted to:

- `logpt`
- `chEmEF`
- `chHEF`
- `neEmEF`
- `neHEF`
- `muEF`
- `chMultiplicity`
- `neMultiplicity`
- `nConstituents`
- `nElectrons`
- `nMuons`
- `muonSubtrFactor`
- `muonSubtrDeltaEta`
- `muonSubtrDeltaPhi`
- `mass`
- `area`
- `rawFactor`

The raw `pt` and `eta` branches are still read for pT selection, eta-region
selection, and validation plots, but they are not model inputs except through
the derived `logpt`.

## Example

```bash
python MVA_training/pileup_dnn/train_pu_dnn.py \
  -i "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterJetsHorn25GeV_Apr14_tightPassLepVeto_NoJER_pySRTraining/stage1_output/2024/f1_0/dyTo2Mu_M-50_aMCatNLO/0/part*.parquet" \
  --use-glob \
  -o validation/pu_dnn/run2024 \
  --regions HE HF \
  --pt-min 25 \
  --pt-max 50 \
  --epochs 50 \
  --batch-size 4096
```

Use multiple `-i` entries to train on DY, TOP, and EWK together:

```bash
python MVA_training/pileup_dnn/train_pu_dnn.py \
  -i \
    "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterJets_May08_tightPassLepVeto_NoJER/stage1_output/2024/compacted/dyTo2Mu_M-50_aMCatNLO/*/*.parquet" \
    "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterJets_May08_tightPassLepVeto_NoJER/stage1_output/2024/compacted/ttjets_*/*/*.parquet" \
    "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterJets_May08_tightPassLepVeto_NoJER/stage1_output/2024/compacted/ewk_*/*/*.parquet" \
  --use-glob \
  -o validation/pu_dnn/run2024_dy_top_ewk_balanced \
  --regions HE HF

python MVA_training/pileup_dnn/train_pu_dnn.py \
  -i \
    "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJets_June02_tightPassLepVeto_NoJER/stage1_output/2022postEE/compacted/dy_M-50_aMCatNLO/*/*.parquet" \
    "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJets_June02_tightPassLepVeto_NoJER/stage1_output/2022postEE/compacted/ttjets_*/*/*.parquet" \
    "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJets_June02_tightPassLepVeto_NoJER/stage1_output/2022postEE/compacted/ewk_*/*/*.parquet" \
  --use-glob \
  -o validation/pu_dnn/run2022postEE_dy_top_ewk_02June \
  --regions HEpos HEneg HFpos HFneg
```

The expanded input file list is saved to `inputs.json` in the output directory.

The DNN uses class-balanced training weights by default, which is the DNN
analogue of the HS/PU balancing used in the PySR training. It also balances the
default sample groups `DY`, `TOP`, and `EWK` so each group contributes equal
total training weight within each HS/PU class. This keeps a large DY sample from
dominating TOP/EWK.

Add `--use-weights` to start from MC event weights when `wgt_nominal` is present.
The sample and class balancing are then applied on top of those weights.

Useful switches:

- `--no-class-balance`: disable HS/PU class balancing
- `--no-sample-balance`: disable DY/TOP/EWK sample-group balancing
- `--sample-balance-groups DY TOP EWK`: choose which inferred groups to balance

To train signed regions, use:

```bash
--regions HEpos HEneg HFpos HFneg
```

## Outputs

Each region gets its own directory, for example `validation/pu_dnn/run2024/HE`.

Important model files:

- `checkpoint.pt`: PyTorch checkpoint with model metadata
- `model_best.pt`: model state dict
- `model_torchscript.pt`: deployable TorchScript model
- `features.json`: final feature list
- `scaler.json`: median/mean/std used before inference

PySR-comparable files:

- `summary_HE.json`, `summary_HF.json`
- `summary_all.json`
- `inputs.json`
- `sample_composition.json`
- `rescan_HE.json`, `rescan_HF.json`
- `wp_vs_pt_HE.csv`, `wp_vs_pt_HF.csv`
- `eff_and_rej_vs_pt_*`
- `score_real_fake_*`
- `stack_score_*`, `stack_pt_*`, `stack_eta_*`

Extra DNN validation files:

- `roc_*`
- `precision_recall_*`
- `confusion_matrix_*`
- `training_history_*`
- `feature_importance_*`
- `feature_shape_*`

If `puIdDisc` is available, the script also writes matching baseline plots and a
baseline summary block so the DNN can be compared against the existing PU ID.
