# VBF DNN full-Run3 (2022preEE-2026) training run — monitoring

**Date:** 2026-09-09
**Status:** Done — completed successfully, no errors, at 20:42 UTC

## Command being monitored

```
bash run_analysis_pipeline.sh -c configs/datasets/dataset_nanoAODv15_run3.yaml \
  -v 15 -l Run3_nanoAODv15_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation \
  -y "2022preEE,2022postEE,2023,2023BPix,2024,2025,2026" -m dnn -k
```
User-launched (not via this session's Bash tool), so no live stdout was available —
progress verified by inspecting output artifacts on disk + Dask cluster state instead.

## Data availability confirmed

All 7 requested years have compacted stage1 output under this label
(`stage1_output/<year>/compacted/`), including `vbf_powheg`/`vbf_powheg_dipole` signal and
`ggh_powhegPS`/`ggh_amcatnlo`, matching `configs/dnn_run3_vbf.yaml`'s
`samples.signal.processes_per_year`. This resolves the earlier session's blocker (2022-2023
stage1 output for the old nanoAODv12 label had been cleaned up) — stage1 has since been
(re)run for 2022preEE/2022postEE/2023/2023BPix under this current nanoAODv15 tag, and 2025/2026
data now exist too (today is 2026-09-09).

## Two uncommitted local changes discovered (not made by me)

1. **`common_workflow.sh`**: `dnn_hpo_trials` default changed from `HPO_TRIALS:-50` to
   `HPO_TRIALS:-11`. This run does an 11-trial Optuna HPO, not 50 — materially faster
   (previous full-5-year 50-trial run took ~12.2h for HPO alone; this one is on track for
   roughly ~2h for HPO at ~11 min/trial average, see timing below).
2. **`MVA_training/VBF_run3/preprocess_dnn.py`**: `events_to_dataframe`'s call to
   `applyRegionCatCuts` now passes `jj_eta_region="jj_both_central"`. The VBF DNN is now
   trained **only on events where both leading jets have |eta| <= 2.5** (central detector
   region for both jets of the dijet pair), not the full inclusive VBF-category dijet
   topology used in the previous (Aug23) run. This is likely what the user meant by
   "central region" DNN training in the earlier part of this conversation — implemented as
   a `jj_eta_region` restriction inside the existing `category="vbf"` path, not a new
   category. Worth flagging explicitly to the user since it changes what the resulting
   model actually represents (only training on "both-jets-central" VBF topology) vs the
   earlier Aug23 model (no `jj_eta_region` restriction).

Neither change is committed (`git status`/`git diff` on both files). Not modified by me.

## Progress observed (via output-directory artifacts + `get_dask_cluster_info`)

- Preprocessing: complete (`preprocess_manifest.json`, `scalers_{0..3}.npz`,
  `data_df_{train,validation,evaluation}_{0..3}.parquet`, sanity plots all present) —
  finished ~17:29 UTC.
- Dask Gateway cluster `cms.da633090339442ae9a11eb6ca6a951c9` (k8s, 2 cores/10 GiB per
  worker) used for preprocessing only, now at 0 workers (expected — HPO/train stages are
  local-torch only, no Dask).
- HPO (`hpo_optuna/v1_multifold_050Trials/optuna_trials/`): trials 0-4 complete, trial 5
  in progress (fold1) as of 18:29:42 UTC. Per-trial wall time so far: trial0 7.3 min,
  trial1 7.8 min, trial2 14.6 min, trial3 15.4 min, trial4 9.4 min (avg ~10.9 min/trial).
  At this rate, remaining ~5.5 trials -> HPO likely finishes ~19:30-19:45 UTC, final
  `train_dnn.py` 4-fold run (~10-15 min based on one HPO trial's cost) after that.
  **Update 19:02 UTC**: trial5 finished (12.7 min), trial6 in progress (fold3, folds
  0-2 done). 7/11 trials underway, no errors, no stalls. Revised estimate: HPO done
  ~20:00-20:05 UTC, final training done ~20:15-20:20 UTC.
  **Update 19:28 UTC**: trials 0-7 complete, trial8 in progress (fold0, ~8 min in).
  9/11 trials underway. No errors anywhere in the output tree. `optuna_best.json` not
  written yet (only appears once the study finishes). On pace with the 20:00-20:20 UTC
  estimate above.
  **Update 20:07 UTC**: trial9 (10th of 11) just finished. Only trial10 (11th, last)
  remains. No errors. On pace to finish HPO within ~15 min, final `train_dnn.py` run
  after that.
- No error/traceback artifacts found anywhere under the run's output tree; no stalled
  timestamps (most recent trial subdirectory is actively being written to).

## HPO: all 11 trials attempted, 10 completed + 1 pruned (not a bug)

Trials 0-9 (10 trials) each ran all 4 folds to completion; final (weighted val AUC) values:
0.878, 0.756, 0.898, 0.789, 0.836, 0.896, 0.897, 0.893, 0.862, **0.899** (trial 9 = best).
Trial 10 (the 11th/last) only has a `fold0/` dir — cut short by
`optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3)` in `hpo_optuna.py`:
the pruner is inactive for the first 10 trials (needs them to build a median baseline), so
trial 10 was the *only* trial eligible for pruning, and its early value (0.728) was below
median -> folds 1-3 skipped intentionally. Net effect: with `HPO_TRIALS=11` and
`n_startup_trials=10`, the pruner does essentially nothing useful (only ever 1 trial could
be cut) — worth lowering `n_startup_trials` or raising `HPO_TRIALS` if pruning is meant to
actually save time.

`optuna_best.json`: best trial = 9, weighted val AUC **0.8991**, params: `activation=relu,
batch_norm=False, n_layers=2, first_width=1024, shrink_factor=0.567, drop_mode=per_layer,
dropout_l0=0.162, dropout_l1=0.218, lr=3.54e-4, weight_decay=4.86e-4,
label_smoothing=0.0153, grad_clip_norm=1.68, batch_size=2048` -> applied as
`hidden=[1024, 581]` in the final training.

## Final training (`trained_best_optuna_v1_multifold_050Trials/`) — all 4 folds, no errors

| fold | train AUC_w | val AUC_w | eval AUC_w | best_epoch | KS sig | KS bkg |
|---|---|---|---|---|---|---|
| 0 | 0.8992 | 0.8933 | 0.9010 | 45 | 0.0045 | 0.0410 |
| 1 | 0.8973 | 0.8993 | 0.8925 | 38 | 0.0055 | 0.0305 |
| 2 | 0.8984 | 0.8935 | 0.9038 | 44 | 0.0030 | 0.0226 |
| 3 | 0.8983 | 0.9046 | 0.8938 | 43 | 0.0062 | 0.0270 |

Assessment:
- Train/val/eval AUC agree within ~0.01 in every fold -> no gross overfitting on AUC, and
  folds are internally consistent (tight ~0.893-0.905 spread).
- Signal KS (train vs val) tiny (0.003-0.006), same as the Aug23 run. **Background KS is
  notably higher this time (0.023-0.041) vs the Aug23 run's 0.006-0.012** — a real,
  ~3-4x larger train/val background-shape difference. Not alarming on its own (still
  <0.05) but a genuine change worth watching if this model is compared against the
  earlier one.
- Same train_loss >> val_loss inversion as the Aug23 run (e.g. fold0: train 1.02 vs val
  0.33) — same pre-existing loss-normalization-scale artifact already flagged in that
  review (per-batch weight normalization + label smoothing), not new to this run.
- **Overall AUC dropped from ~0.92-0.93 (Aug23 run) to ~0.89-0.90 (this run).** Two
  confounded changes happened at once, so this can't be cleanly attributed to one cause:
  (1) `jj_eta_region="jj_both_central"` discards forward-jet topology, and large
  pseudorapidity separation between the two jets is one of the classic, most powerful
  VBF-vs-background discriminators — restricting to both-jets-central events plausibly
  and physically should reduce separation power; (2) `HPO_TRIALS` 50->11 means far less
  hyperparameter search, so the "best" found here (0.899) is likely a worse optimum than
  50 trials would find, independent of the region restriction. An ablation (same
  `jj_eta_region`, 50 trials; or same 11 trials, no region restriction) would be needed to
  split the two effects if that matters for the analysis.

## Follow-up (not done here)

- Have not confirmed with the user whether the `jj_both_central` restriction and
  `HPO_TRIALS=11` are intentional keepers vs. things to revert/retune before this is
  treated as "the" VBF DNN.

## HPO resume to 51 trials (2026-09-09, started 21:09:40 UTC)

User asked to extend the study to 51 total trials (40 more on top of the 11 already
done). Found that `-m dnn_hpo` is **not a valid top-level mode** — `run_analysis_pipeline.sh`'s
mode dispatcher only accepts `dnn|dnn_pre|dnn_train|dnn_var_rank` even though
`run_dnn_workflow_once()`'s internal case statement has a `dnn_hpo` branch (dead/unreachable
from the CLI); confirmed by the user hitting `ERROR: Invalid analysis mode 'dnn_hpo'.`
directly. `-m dnn` would have worked but re-runs preprocessing (new Dask cluster, ~10-15 min
of redundant deterministic work) first.

Instead, invoked `hpo_optuna.py`/`train_dnn.py` directly with the same `--data-dir`/`--out-dir`
paths the wrapper would have used, `--n-trials 40`. Confirmed via log
(`[I ...] Using an existing study with name 'vbf_dnn_hpo' instead of creating a new one.`)
that it correctly resumed the existing `optuna_study.db` (trials 0-10 untouched, new trials
start at trial_00011) rather than restarting. Running in background, log at
`.../2022preEE-2022postEE-2023-2023BPix-2024-2025-2026_h-peak_vbf/hpo_resume_51trials.log`.

Estimated ~10.5h for the 40 new trials (this dataset/config averaged ~16 min/trial for the
first 10, slower than the 5-year run's ~11 min/trial) + ~15-30 min final retrain. The final
retrain step will overwrite the current `trained_best_optuna_v1_multifold_050Trials/fold{0-3}/*`
artifacts reviewed above with the new best-of-51 result.

## User-stated follow-up TODOs (2026-09-09, after this run completes)

1. Make the `jj_eta_region` used by the VBF DNN preprocessing configurable (currently
   hardcoded to `"jj_both_central"` in `preprocess_dnn.py`'s `events_to_dataframe` ->
   `applyRegionCatCuts` call, see above) — user wants to be able to choose which jet-eta
   region to train on (e.g. `jj_both_central`, `jj_non_central`, `all`, ... per
   `modules/selection.py`'s `PAIR_JJ_ETA_REGIONS`), presumably via a CLI/config option
   threaded through `preprocess_dnn.py` (and `common_workflow.sh`/`run_analysis_pipeline.sh`
   if driven that way).
2. The DNN output directory naming (`make_output_dir` in `preprocess_dnn.py`, currently
   `<tag>/<years>_<region>_<category>`; mirrored in `common_workflow.sh`'s `dnn_base_dir`)
   should encode the `jj_eta_region` choice too, so different jet-eta-region trainings
   don't collide/overwrite each other's output.
3. Separately: resolve the current branch's (`Week_June10`) merge conflicts against
   `main` (unrelated to the DNN work above).

Not started — explicitly deferred by the user until after the current training run
finishes.
