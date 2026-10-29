# Transformer-selected VBF-DNN versus the cut-based VBF-DNN

## Purpose and terminology

This document explains the difference between the newly trained
Transformer-selected VBF-DNNs and the existing cut-based VBF-DNN path at the
current branch HEAD. Here, “HEAD cut-based model” means the repository's
default/legacy cut-based workflow; it does **not** mean a neural-network output
head.

The most important point is:

> The new model is not a new head attached to the binary Transformer. It is a
> separately trained VBF-versus-background MLP. The upstream binary Transformer
> is used only to define which events enter the VBF channel and the downstream
> MLP training sample.

The two classifiers therefore have different jobs and opposite score
interpretations:

- `transf_ggh_vs_vbf_binary_score` is the upstream ggH-versus-VBF category
  score. It is ggH-oriented, so **lower** values are more VBF-like and the VBF
  selection is `score < threshold`.
- The downstream VBF-DNN is trained with VBF signal as label `1` and background
  as label `0`. Its sigmoid output is therefore VBF-signal-like, so **higher**
  values are more signal-like. Stage 2 stores
  `dnn_vbf_score_atanh = atanh(sigmoid(logit))` for template binning.

## The two analysis paths

```text
Cut-based baseline
------------------
Stage-1 events
  -> H-peak + b-tag veto + jet1_pt/jj_mass/jj_dEta cuts
  -> train the VBF-DNN on this cut-defined phase space
  -> apply the same cut-defined channel in Stage 2
  -> bin the downstream VBF-DNN score

Transformer-selected path
-------------------------
Stage-1 events scored by the upstream binary Transformer
  -> H-peak + b-tag veto + binary score < threshold
  -> train a new VBF-DNN on this Transformer-defined phase space
  -> read the selection and preprocessing contract from the training manifest
  -> reproduce that channel in Stage 2
  -> bin the newly trained downstream VBF-DNN score
```

The upstream Transformer score is a **gate**, not an input feature of the
downstream VBF-DNN. It does not appear in `training_features.pkl`.

## Developer-facing comparison

| Aspect | Cut-based VBF-DNN | New Transformer-selected VBF-DNN |
|---|---|---|
| Channel definition | `jet1_pt > 35 GeV`, `jj_mass > 400 GeV`, `jj_dEta > 2.5`, plus the b-tag veto | `transf_ggh_vs_vbf_binary_score < threshold`, plus the same b-tag veto; no `jet1_pt`, `jj_mass`, or `jj_dEta` channel cut |
| Training mass region | `115 <= dimuon_mass < 135 GeV` | Same |
| Jet multiplicity implied by the channel | Effectively at least two reconstructed jets, because the dijet cuts must be defined and pass | Inclusive; 0-, 1-, and 2+-jet events may enter if they pass the Transformer score and b-tag veto |
| Source data | Ordinary compacted Stage-1 Parquet | Parquet already carrying the out-of-fold binary Transformer score |
| Role of the Transformer | None | Defines the channel only; it is not jointly trained with the VBF-DNN |
| Downstream task | Binary VBF signal (`1`) versus DY/TOP/EWK background (`0`) | Same task, but learned in a different selected phase space |
| Model family | Fully connected MLP returning one logit | Same MLP implementation, but retrained from scratch with different configured architecture/hyperparameters |
| Cross-validation | Four deterministic event folds | Same four-fold scheme |
| Missing jet/dijet values | Historical cut-path cleaner; missing-jet states are mostly irrelevant after the dijet cuts | Explicit raw-space sentinels are required because 0/1-jet events are part of the phase space |
| Scaling | Per-fold weighted mean/std computed from that fold's training rows | Same algorithm, but new scalers are learned from the Transformer-selected rows, including sentinel values |
| Stage-2 input contract | Model/scaler paths supplied by the caller; selection comes from the cut-based code path | `preprocess_manifest.json` is authoritative for input root, score column, threshold/direction, years, features, folds, and sentinel policy |
| Systematics in the current workflow | The cut path supports the existing variation-aware Stage-2 flow | Current binary-Transformer-scored inputs and manifests are nominal-only; Stage 2 rejects a variation-enabled run |
| DNN binning | Existing cut-model binning | Must use a model-specific scan and `configs/MVA/VBF/dnn_binning.yaml` entry; cut-model bins are not reusable |

## What changed in the trained network

The code still instantiates the `MLP` class in
[`train_dnn.py`](../MVA_training/VBF_run3/train_dnn.py): a sequence of linear
layers, optional batch normalization, activation, dropout, and one output
logit. However, “same model family” does not mean “same trained model.”

For the concrete Run-2 cut-based model wired by
[`run_vbfdnn_cutbased_run2_stage23.sh`](../run_vbfdnn_cutbased_run2_stage23.sh),
the stored 100-trial Optuna result uses:

- hidden layers `[768, 381]`;
- SiLU activation;
- dropout `[0.104336..., 0.104336...]`;
- no batch normalization;
- batch size `1024` and label smoothing `0.001885...`.

The new Run-2 Transformer-selected models use the fixed values in
[`configs/dnn_run2_vbf.yaml`](../configs/dnn_run2_vbf.yaml):

- hidden layers `[384, 192, 96]`;
- ReLU activation;
- dropout `[0.480653..., 0.004194..., 0.053222...]`;
- batch normalization;
- batch size `8192` and label smoothing `0.040490...`.

The new Run-3 models use the same `[384, 192, 96]` MLP definition from
[`configs/dnn_run3_vbf.yaml`](../configs/dnn_run3_vbf.yaml), but have their own
30-feature contract and Run-3 training controls. The new final trainings used
the YAML values directly (`optuna.best_json` is `null` in their training
manifests); they are not the old cut-model Optuna checkpoint continued or
fine-tuned.

Consequently, a comparison between the paths changes at least two things at
once:

1. the phase-space definition used to build the training and analysis channel;
2. the actual downstream MLP architecture/hyperparameters and fitted weights.

It should not be described as a controlled comparison of selection alone unless
the downstream architecture and training settings are first held fixed.

## Selection contracts

### Cut-based baseline

The default path calls `applyRegionCatCuts(..., category="vbf")` in
[`modules/selection.py`](../modules/selection.py). Ignoring optional HE/HF
mitigation and DY sample-overlap filtering, the reconstructed VBF category is

```text
115 <= dimuon_mass < 135 GeV
and not (nBtagLoose >= 2 or (nBtagMedium >= 1 and njets >= 2))
and jet1_pt > 35 GeV
and jj_mass > 400 GeV
and rounded(jj_dEta, 4 decimals) > 2.5
```

The `category_selection: auto` value currently present in the Run-3 YAML does
not enable Transformer selection. The loader deliberately normalizes `auto` to
`cut` to preserve historical behavior. Transformer selection is an explicit
opt-in.

### Transformer-selected path

The new preprocessing path calls `apply_transformer_training_selection()` in
[`preprocess_dnn.py`](../MVA_training/VBF_run3/preprocess_dnn.py). Its contract
is

```text
115 <= dimuon_mass < 135 GeV
and not (nBtagLoose >= 2 or (nBtagMedium >= 1 and njets >= 2))
and isfinite(transf_ggh_vs_vbf_binary_score)
and transf_ggh_vs_vbf_binary_score < threshold
```

Equality to the threshold is rejected. No extra jet multiplicity, leading-jet
`pT`, dijet-mass, or dijet-separation requirement is added.

The concrete trained working points currently represented in the repository
are:

| Run | Model label | Binary Transformer VBF gate | Downstream inputs |
|---|---|---:|---:|
| Run 2 | `Run2_VBFDNN_Transformer_lt0p200` | score `< 0.200` | 36 |
| Run 2 | `Run2_VBFDNN_Transformer_lt0p022` | score `< 0.022` | 36 |
| Run 3 | `Run3_VBFDNN_Transformer_lt0p200` | score `< 0.200` | 30 |
| Run 3 | `Run3_VBFDNN_Transformer_lt0p019` | score `< 0.019` | 30 |

The threshold is part of the immutable preprocessing manifest. A directory
created for one threshold must not be reused for another threshold.

## Missing jets and feature preprocessing

This is the largest data-contract difference. The cut-based channel requires a
passing dijet system, while the Transformer-selected channel can contain events
with zero or one reconstructed jet. Those events do not have physical values
for every jet and dijet feature.

The new path therefore applies feature-specific sentinels **before** computing
the per-fold scaler:

| Missing or invalid feature type | Raw sentinel |
|---|---:|
| Jet/dijet `pT`, mass, log, or other non-negative quantity | `0` |
| Jet eta or phi | `-5` |
| UParT QvG | `-1` |
| `jj_dEta` | `-1` |
| `rpt` | `-1` |
| `ll_zstar_log` | `-20` |
| `mmj_min_dEta` | `-1` |
| `pt_centrality` | `-200` |

`pt_centrality` is clipped in raw space to `[-200, 200]`. Standardized inputs
are not clipped. Sentinel rows participate in the normal training-fold weighted
mean/std calculation, so the new scaler files are model artifacts and cannot be
replaced by scalers from the cut-based model.

Stage 2 must run the same cleaner and use the exact per-fold scaler feature
order. The manifest-driven implementation validates this rather than silently
accepting a mismatched scaler or feature list.

## Cross-validation and inference

Both paths use four deterministic folds with `shuffle: false`; the base fold is
`event % 4`. For training instance `i`, folds `[i, i+1]` are training, fold
`i+2` is validation, and fold `i+3` is evaluation, all modulo four. Stage 2 uses
the model for which a given event was in the held-out evaluation fold, so the
downstream score is out-of-fold.

The new Ray backend changes scheduling, not the statistical model: four
independent fold models are trained concurrently, with one GPU per fold. It is
not data-parallel training of one network.

## Stage 2/3 behavior and current limitations

The cut-based path reads the ordinary Stage-1 production and reapplies the VBF
kinematic cuts for each relevant variation. The Transformer-selected path is
enabled with `--binary-transformer-manifest`; it reads the scored input root and
the complete selection/preprocessing contract from that manifest.

Important boundaries for developers:

- Do not use `--use_transformer_vbf_channel` for these models. That is a
  separate legacy 2017 three-class Transformer path with a different score and
  direction. The two options are intentionally mutually exclusive.
- Do not parse the threshold from `MODEL_LABEL`. The manifest is the source of
  truth.
- Do not add the old dijet cuts after applying the binary score. Doing so would
  create a third phase space that matches neither training contract.
- Do not replace the new sentinel cleaner with `fillEventNans()` or the legacy
  cleaner. Training and inference preprocessing must be identical.
- Do not reuse cut-based scalers, TorchScript checkpoints, or DNN bin edges.
- Keep Run-2 and Run-3 artifacts separate. Their 36- and 30-input contracts are
  incompatible.
- The current new path is nominal-only. Its output can support nominal/blinded
  development comparisons, but it is not yet equivalent to the cut-based
  full-systematics production.

For H-sideband templates, the new Stage-2 path defaults to pinning the DNN's
`dimuon_mass` input to `125 GeV`, consistent with the existing Stage-2 behavior.
`--binary-transformer-sideband-mass raw` is an explicit alternative and changes
the inference contract.

## How to interpret comparisons

Training AUC, a bin-scanner's `total_asimov_Z`, Stage-2 yields, and final
likelihood sensitivity answer different questions. In particular, the bin scan
values in `dnn_binning.yaml` are diagnostics used to choose score bins; they are
not a final expected significance or limit.

An apples-to-apples physics comparison should use:

1. the same data-taking eras and sample definitions;
2. explicit channel definitions and model manifests;
3. independent, model-specific DNN binning;
4. the same SR/SB and blinding policy;
5. the same systematic-nuisance scope; and
6. final Stage-3 datacards and the same Combine procedure.

Until the Transformer-selected path has a validated systematic treatment, any
comparison with the full-systematics cut-based result must be labeled
**nominal-only**, not as a production replacement.

## Source-of-truth files

- Training selection, folds, scalers, and manifest writing:
  [`MVA_training/VBF_run3/preprocess_dnn.py`](../MVA_training/VBF_run3/preprocess_dnn.py)
- Missing-jet sentinel implementation:
  [`MVA_training/VBF_run3/utils/pre_scale_cleaning.py`](../MVA_training/VBF_run3/utils/pre_scale_cleaning.py)
- MLP, loss, and Ray four-fold training:
  [`MVA_training/VBF_run3/train_dnn.py`](../MVA_training/VBF_run3/train_dnn.py)
- Run-specific feature/model contracts:
  [`configs/dnn_run2_vbf.yaml`](../configs/dnn_run2_vbf.yaml) and
  [`configs/dnn_run3_vbf.yaml`](../configs/dnn_run3_vbf.yaml)
- Cut-based category definition:
  [`modules/selection.py`](../modules/selection.py)
- Manifest-driven Stage-2 inference:
  [`run_stage2_vbf.py`](../run_stage2_vbf.py)
- New Stage-2/3 runner:
  [`run_vbfdnn_transformer_stage23.sh`](../run_vbfdnn_transformer_stage23.sh)
- Concrete Run-2 cut-based comparison runner:
  [`run_vbfdnn_cutbased_run2_stage23.sh`](../run_vbfdnn_cutbased_run2_stage23.sh)

