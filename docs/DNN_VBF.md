title: DNN for VBF
---

## Transformer-selected VBF-DNN training (Run 2 and Run 3)

This is an opt-in preprocessing and training path.  If the new options below
are absent, the existing cut-based categorisation and sequential local training
are unchanged.

Ready-to-run wrappers for the fixed `score < 0.2` working point are available
at the repository root:

```bash
./run_vbfdnn_transformer_run2_lt0p2.sh
./run_vbfdnn_transformer_run3_lt0p2.sh
```

Run them from a CPU-only Purdue AF notebook session. Each wrapper performs
preprocessing, 50-trial/four-fold HPO, and final four-fold Ray training with one
GPU per fold. Set `RUN_LABEL=...` before the command only if a different output
label is needed.

Both supplied scored datasets use the ggH-oriented binary column
`transf_ggh_vs_vbf_binary_score`; therefore the Transformer VBF category is
defined as

```text
transf_ggh_vs_vbf_binary_score < THRESHOLD
```

The score threshold is deliberately a run-time argument, not a hard-coded
number. In this mode the only other event selections are
`115 <= dimuon_mass < 135 GeV` and the b-tag veto
`not (nBtagLoose >= 2 or (nBtagMedium >= 1 and njets >= 2))`. No `jet1_pt`,
`jj_mass`, or `jj_dEta` VBF cut is applied.

Use a threshold-specific label. The preprocessing manifest freezes the score
column, threshold, comparison direction, mass window, and b-tag veto; reusing a
complete label with a different threshold is rejected.

### Run 2 preprocessing

Choose `THRESHOLD` before running (`0.20` below is only an example):

```bash
THRESHOLD=0.20  # example; replace with the chosen working point
LABEL=run2_transformer_t${THRESHOLD/./p}

DNN_CONFIG=configs/dnn_run2_vbf.yaml \
DNN_INPUT_BASE=/depot/cms/users/hu1027/Hmumu_Categorization_Transformer_output/scored_output/run2_stage23_vbfdnn_nominal_sep08_ray/stage1_output/ \
DNN_CATEGORY_SELECTION=transformer \
TRANSFORMER_SCORE_THRESHOLD="${THRESHOLD}" \
./run_in_pixi.sh default bash run_analysis_pipeline.sh \
  -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 \
  -l "${LABEL}" -y "2016preVFP 2016postVFP 2017 2018" -m dnn_pre
```

Run 2 uses the 36-input manifest in `configs/dnn_run2_vbf.yaml` and reads the
scored layout `<base>/<year>/compacted/<sample>/<shard>/*.parquet`.

### Run 3 preprocessing

```bash
THRESHOLD=0.20  # example; replace with the chosen working point
LABEL=run3_transformer_t${THRESHOLD/./p}

DNN_CONFIG=configs/dnn_run3_vbf.yaml \
DNN_INPUT_BASE=/depot/cms/users/hu1027/Hmumu_Categorization_Transformer_output/binary_output/run3_oct23_binary_all_eras/stage23_scored/ \
DNN_CATEGORY_SELECTION=transformer \
TRANSFORMER_SCORE_THRESHOLD="${THRESHOLD}" \
./run_in_pixi.sh default bash run_analysis_pipeline.sh \
  -c configs/datasets/dataset_nanoAODv15_run3.yaml -v 15 \
  -l "${LABEL}" \
  -y "2022preEE 2022postEE 2023 2023BPix 2024 2025 2026" -m dnn_pre
```

Run 3 uses the separate 30-input manifest in `configs/dnn_run3_vbf.yaml`. Its
scored layout is `<base>/<year>/<sample>/<shard>/*.parquet`; the preprocessor
detects this layout without changing the config used by the old compacted path.

### Four-fold training on four Ray GPUs

After preprocessing, submit the final four folds with one GPU per fold. Reuse
the same `LABEL`, years, and Run-specific `DNN_CONFIG` as above:

```bash
DNN_CONFIG=configs/dnn_run2_vbf.yaml \
DNN_TRAIN_BACKEND=ray \
TRAIN_LABEL=trained_transformer \
./run_in_pixi.sh default bash run_analysis_pipeline.sh \
  -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 \
  -l "${LABEL}" -y "2016preVFP 2016postVFP 2017 2018" -m dnn_train
```

Run 3 uses its own config and year set:

```bash
DNN_CONFIG=configs/dnn_run3_vbf.yaml \
DNN_TRAIN_BACKEND=ray \
TRAIN_LABEL=trained_transformer \
./run_in_pixi.sh default bash run_analysis_pipeline.sh \
  -c configs/datasets/dataset_nanoAODv15_run3.yaml -v 15 \
  -l "${LABEL}" \
  -y "2022preEE 2022postEE 2023 2023BPix 2024 2025 2026" -m dnn_train
```

`DNN_TRAIN_BACKEND=ray` submits a Purdue AF Ray job with exactly four one-GPU
workers. The workflow waits for the Ray terminal state, so `dnn_train` returns
success only after all four folds and the final manifest are complete. If the
configured `optuna_best.json` exists it is applied, otherwise the YAML
hyperparameters are used, matching the old `dnn_train` behavior.

The equivalent low-level submit command is:

```bash
/work/pixi/global/.pixi/envs/default/bin/python \
  MVA_training/VBF_run3/submit_ray_dnn.py submit \
  --config <run-specific-config.yaml> \
  --data-dir <preprocessing-output-directory> \
  --out-dir <preprocessing-output-directory>/trained_transformer
```

The low-level command is asynchronous by default; add `--wait` to give it the
same completion semantics as `run_analysis_pipeline.sh`.

Check the submitted job and its logs with:

```bash
/work/pixi/global/.pixi/envs/default/bin/python \
  MVA_training/VBF_run3/submit_ray_dnn.py status \
  --submission <training-output-directory>/ray_submission.json --logs
```

Each Ray task writes only its own `fold0/` through `fold3/` directory. The
driver writes `manifest.json` only after all four tasks succeed. HPO remains the
existing, separate `dnn_hpo` phase; the new four-GPU backend applies to final
four-fold training.

### Four models concurrently on 16 Ray GPUs

The repository-root wrapper below preprocesses and trains these four independent
models in one invocation:

| Model | Transformer VBF selection |
|---|---|
| Run 2 fixed working point | `score < 0.200` |
| Run 2 scanned optimum | `score < 0.022` |
| Run 3 fixed working point | `score < 0.200` |
| Run 3 scanned optimum | `score < 0.019` |

Run it from a CPU-only Purdue AF session:

```bash
./run_vbfdnn_transformer_ray16_matrix.sh
```

Preprocessing runs sequentially because one user can own only one Dask Gateway
cluster. The script then submits all four training jobs without waiting between
submissions. They share one Ray cluster requested with exactly 16 one-GPU
workers; each model schedules four independent fold tasks, so the requested
steady-state layout is `4 models x 4 folds x 1 GPU = 16 GPUs`. The script waits
for all four Ray jobs and returns nonzero if any job fails.

This command performs final training with the hyperparameters already stored in
the corresponding Run 2 or Run 3 YAML. It does not run a new Optuna scan. The
threshold-specific preprocessing and scaler artifacts remain in separate output
directories, and the final models are written under `trained_ray16/` in each
directory.

Useful optional controls are:

```bash
# Reuse already completed preprocessing outputs.
SKIP_PREPROCESS=1 ./run_vbfdnn_transformer_ray16_matrix.sh

# Per-fold CPU/RAM requests (defaults shown).
CPUS_PER_FOLD=4 RAM_PER_WORKER_GB=16 \
  ./run_vbfdnn_transformer_ray16_matrix.sh
```

The Ray wrapper scripts remove only their target output directory's previous
`ray_submission.json` before checking/submitting training. Model files,
checkpoints, metrics, and preprocessing artifacts are not removed. Deleting a
submission record does not stop a still-running Ray job, so do not launch the
same wrapper twice concurrently.

An exact 16-worker request waits until all required GPUs are simultaneously
available. The default minimum GPU memory is 5 GB so both AF A100 slices and
T4 workers are eligible. Each fold still reserves exactly one GPU.

To concurrently train only the two already-preprocessed Run-2 models, use:

```bash
./run_vbfdnn_transformer_run2_ray8.sh
```

This submits `score < 0.200` and `score < 0.022` without waiting between the
two submissions. Both jobs share one exact-size eight-worker Ray cluster, so
the steady-state layout is `2 models x 4 folds x 1 GPU = 8 GPUs`. The script
performs no preprocessing and no HPO, waits for both jobs, and writes outputs
under `trained_ray8_concurrent/`. This distinct output name avoids the stale
`trained_ray8/` record created before the Ray module-entrypoint fix.

To train only the already-preprocessed Run-2 `score < 0.200` model, request four
GPUs with:

```bash
./run_vbfdnn_transformer_run2_t0p2_ray4.sh
```

This script performs no preprocessing and no HPO. It validates the existing
Run-2 `score < 0.200` fold/scaler artifacts, submits its four folds to one
exact-size four-worker Ray cluster, waits for completion, and writes the result
under `trained_ray4/`.

The other three models have matching independent four-GPU wrappers:

```bash
# Run 2, scanned optimum
./run_vbfdnn_transformer_run2_t0p022_ray4.sh

# Run 3, fixed working point
./run_vbfdnn_transformer_run3_t0p2_ray4.sh

# Run 3, scanned optimum
./run_vbfdnn_transformer_run3_t0p019_ray4.sh
```

Each script validates its existing four-fold preprocessing artifacts, removes
only its old `trained_ray4/ray_submission.json`, requests exactly four one-GPU
workers, waits for completion, and writes its model under `trained_ray4/`.

### Missing-jet sentinel contract

Transformer selection admits 0/1-jet events. Before scaling, the new mode uses
these values:

| Condition | Fill value |
|---|---:|
| `njets < 1`, any `jet1_*` | feature-type sentinel below |
| `njets < 2`, any `jet2_*` | feature-type sentinel below |
| `njets < 2`, `jj_mass*` and other non-negative mass/log inputs | `0` |
| `njets < 2`, `jj_dEta_nominal` | `-1` |
| `njets < 2`, `rpt_nominal` | `-1` |
| `njets < 2`, `ll_zstar_log_nominal` | `-20` |
| `njets < 1`, `mmj_min_dEta_nominal` | `-1` |
| `njets < 2`, `pt_centrality_nominal` | `-200` |
| UParT QvG | `-1` |
| Jet eta or phi | `-5` |
| Other undefined jet/dijet pt, mass, log, etc. | `0` |

Any remaining `NaN` or `Inf` in a training feature is replaced by that
feature's corresponding sentinel, and preprocessing fails if a non-finite
training value remains before scaler construction. `pt_centrality_nominal` is
the sole raw observable clipped, to `[-200, 200]`, matching the historical
cleaner. The sentinels then participate in the existing train-fold weighted
mean/std calculation exactly like every other row. There is no valid-only
scaler and no clipping of standardized values.



## Input Features for DNN in VBF Analysis

The following features are used as input to the Deep Neural Network (DNN) for the Vector Boson Fusion (VBF) analysis:

- **Dimuon kinematics and angles**
  - `dimuon_mass`
  - `dimuon_pt`, `dimuon_pt_log`
  - `dimuon_rapidity`
  - `dimuon_cos_theta_cs`, `dimuon_phi_cs`

- **Event-by-event mass resolution**
  - `dimuon_ebe_mass_res`
  - `dimuon_ebe_mass_res_rel`

- **Jet kinematics**
  - Leading jet: `jet1_pt`, `jet1_eta`, `jet1_phi`
  - Subleading jet: `jet2_pt`, `jet2_eta`, `jet2_phi`

- **Jet flavor/shape information**
  - `jet1_qgl`, `jet2_qgl`

- **Dijet system**
  - `jj_mass`, `jj_mass_log`
  - `jj_dEta`

- **Additional event activity**
  - `htsoft2`
  - `nsoftjets5`

- **VBF topology variables**
  - `rpt`
  - `ll_zstar_log`
  - `mmj_min_dEta`
  - `pt_centrality`

- **Data-taking period**
  - `year`


## DNN Training

### Preprocessing Steps

#### Preprocessing validations

1. Feature distributions before and after preprocessing
1. Correlation matrices before and after preprocessing
1. Plot all input features after preprocessing, from the output parquet files
    1. Add mean and stddev values to the plots

For this, there are two scripts available in the `plotter/` directory:

1. [plot_vbfdnn_input_features_compare.py](../plotter/plot_vbfdnn_input_features_compare.py): to compare feature distributions between signal and background samples before preprocessing.
1. [`dnn_preprocessing_validation.py`](../plotter/dnn_preprocessing_validation.py): to validate the preprocessing steps by plotting the feature distributions from the preprocessed parquet files. For these plots, the mean should be around 0 and the standard deviation around 1. **Note:** the variable `year`  and `nsoftjets5` are not standardized.

### DNN Training

#### DNN preprocessing

NOTE: this step uses dask gateway for parallization. Example:
```bash
python MVA_training/VBF_run3/preprocess_dnn.py --config configs/dnn_run2_vbf.yaml \
--base-path /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026/stage1_output \
--tag Run2_NanoV12_forVBFChannel_Apr29_2026 \
--year 2017 \
--use-dask-gateway --cluster-index 0
```

#### DNN Hyperparameter Optimization
Once preprocessing is done, do hyperparameter optimization (NOTE: using GPU session will greatly accelerate this).
example:
```bash
VA_training/VBF_run3/hpo_optuna.py --config configs/dnn_run2_vbf.yaml \
--data-dir dnn/trained_models/Run2_NanoV12_forVBFChannel_Apr29_2026/2017_h-peak_vbf/ \
--out-dir dnn/trained_models/Run2_NanoV12_forVBFChannel_Apr29_2026/2017_h-peak_vbf/hpo_optuna/v2_108Trials \
--n-trials 5 --folds 0
```
#### DNN Training Command

Example:
```bash
time python MVA_training/VBF_run3/train_dnn.py --config configs/dnn_run2_vbf.yaml \
--data-dir dnn/trained_models/Run2_NanoV12_forVBFChannel_Apr29_2026/2017_h-peak_vbf/  \
--out-dir dnn/trained_models/Run2_NanoV12_forVBFChannel_Apr29_2026/2017_h-peak_vbf/trained_best_optuna \
--optuna-best-json dnn/trained_models/Run2_NanoV12_forVBFChannel_Apr29_2026/2017_h-peak_vbf/hpo_optuna/v2_108Trials/optuna_best.json
```


### VBF stage2 and stage3

#### Selecting the DNN model + jj_eta_region phase space

Current entry point is `run_analysis_pipeline.sh` (see `CLAUDE.md`), driving
`run_stage2_vbf.py`/`run_stage3_vbf.py` via `common_workflow.sh` -- the Run 2-era
`stage1_loop_Improved.sh` commands further below are historical examples. Two independent env
vars control which trained model gets loaded and which dijet |eta| phase space stage-2/3
restrict the VBF category selection to:

- `MODEL_YEARS` -- comma-separated years used to build the DNN model directory name
  (`dnn/trained_models/<label>/<MODEL_YEARS>_<region>_<category>_<JJ_ETA_REGION>`), independent
  of the years `-y` actually processes (e.g. run one year's stage2/3 while loading a model
  trained on a combined set of years). Defaults to `-y`'s years.
- `JJ_ETA_REGION` -- restricts both the DNN model subdirectory *and* the stage-2 VBF category
  event selection to one of `modules/selection.py`'s `PAIR_JJ_ETA_REGIONS`
  (`jj_both_central`, `jj_non_central`, `jj_one_fwd25_one_central`, `jj_one_he_one_central`,
  `jj_one_fwd30_one_central`, `jj_both_fwd25`, `jj_both_he`, `jj_both_fwd30`,
  `jj_one_he_one_fwd30`) or `all` (no restriction). When unset, the DNN-model-directory side
  falls back to `analysis.jj_eta_region` in the DNN config YAML (`configs/dnn_run3_vbf.yaml`
  unless `$DNN_CONFIG` points elsewhere); the stage2/3 category-selection side falls back to
  `all` directly (no YAML). These are two independent knobs sharing one env var for
  convenience -- a typical workflow sets them to match (apply the model only within the phase
  space it was trained on), but they don't have to.

**Before running, check which phase-space variant is actually trained**:
`ls dnn/trained_models/<label>/` and look inside the `_<jj_eta_region>`-suffixed directory you
intend to use for a `trained_best_optuna_*/` subdirectory. A directory with only preprocessing
artifacts (`data_df_*.parquet`, `hpo_optuna/`, no `trained_best_optuna_*/`) has been
preprocessed but not yet trained -- stage-2 will fail looking for a model file that doesn't
exist there yet.

Example (stage2 + stage2_plot + stage3, mode `23`), restricting to the both-jets-central
phase space with a model trained on the same:
```bash
time JJ_ETA_REGION=jj_both_central bash run_analysis_pipeline.sh \
  -c configs/datasets/dataset_nanoAODv15_run3.yaml -v 15 \
  -l <label> -y "<years>" -m 23 -k
```
Leave `JJ_ETA_REGION` unset to keep the previous, unrestricted-selection behavior.

#### Historical (Run 2, `stage1_loop_Improved.sh`) examples

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2017" -m 2 -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2017" -m 2p -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2017" -m 3 -k)
```


### DNN bin edge optimization

Change the `command_compact` in `stage1_loop_Improved.sh`, ie
```bash
command_compact="python scripts/compact_parquet_data.py -y $year --input_path $save_path -m $model_trained_path --model_tag $training_tag --add_dnn_score --fix_dimuon_mass --save_postfix $save_postfix "
```

then run compacted command again (no need to delete things), ie

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2018 2017 2016postVFP 2016preVFP" -m compacted -k)

```

We re-bin the DNN score edgges. Configure the stage1 label, years and MC samples and then run the python

```bash
python MVA_training/VBF_run3/scan_bins_for_dnn.py
```


## Combine step

Datacards, workspaces, significance, limits and impacts are produced with `run_stats_pipeline_VBF.sh`;
see [Statistics](Statistics.md).
