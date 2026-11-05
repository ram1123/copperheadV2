---
title: Transformer VBF channel
---

# VBF channel defined by the binary transformer

The cut-based VBF channel requires two jets with `jj_mass > 400`, `jj_dEta > 2.5`, `jet1_pt > 35` and `jet2_pt > 25`.
In the transformer channel, the b-tag-vetoed events are split by a binary ggH-vs-VBF transformer score instead:

| channel | selection (after the b-tag veto) |
| --- | --- |
| VBF | `score < cutoff` |
| ggH | `score >= cutoff` |

The score is the out-of-fold ggH probability, computed from nominal kinematics only, so JES/JER variations never
move an event between channels.

`configs/MVA/transformer_categorizer.yaml` holds everything about the categoriser:

* `score_field`, `fold_field`: the stage-1 columns the scoring writes (`transf_ggh_vs_vbf_binary_score`, `transf_fold_id`);
* `cutoff`: one value per run period (Run 2: `0.088`);
* `samples`: which `configs/samples/samples.yaml` signal groups are the two training classes;
* `training`, `hpo`: the transformer's training defaults and HPO search space;
* `cutoff_scan`: the cutoff scan's background groups, check variables and settings.

Sample names always come from `samples.yaml`, resolved per year. The VBF DNN is then retrained on the events of this channel, with the binned AMS as its
HPO objective.

Compared with the cut-based workflow, only these steps change:

* **stage 1 → scored copy**: the transformer score is added to a copy of the stage-1 parquet;
* **DNN preprocessing**: `-m dnn_pre -T`;
* **stage 2**: `-m 2 -T -L compacted`.

Stage-2 plots (`2p`), stage 3 and combine are run exactly as before.

All commands run from the repo root. Python steps use the AF global environment
(`/work/pixi/global/.pixi/envs/default/bin/python`, written `python` below); GPU steps go to the AF Ray cluster
(`source ~/.ray_auth.env` first if your session needs it).

## 1. Transformer models

The Run-2 result was produced with the Sep08 production checkpoints of
[b-hive-h-mu-mu](https://gitlab.cern.ch/xudongh/b-hive-h-mu-mu), four folds (`fold{0..3}`). `MVA_training/transformer_binary`
reproduces that training (`train.py`, documented in its docstring) if you want to retrain:

```bash
# optional HPO (Ray Tune; the first trial is the Sep08 setting)
python -m MVA_training.transformer_binary.hpo submit --stage1 <stage1_output> \
    --years 2016postVFP,2016preVFP,2017,2018 --cache-dir <dir> --out <hpo dir> --num-samples 80
# train the four folds (selection + refit), one GPU each; plan format in ray_train.py's docstring
python -m MVA_training.transformer_binary.ray_train submit --plan plan.json
```

## 2. Score stage 1 into a copy

```bash
python -m MVA_training.transformer_binary.ray_score submit --stage1 <stage1_output> \
    --years 2016postVFP,2016preVFP,2017,2018 \
    --models '<model dir>/refit/fold{fold}_seed123456/model.pt' \
    --out-root <scored root>/stage1_output --out-subdir compacted --mass-window 110,150
```

Every event, data and MC, is scored by the fold model that never trained on it. `--mass-window 110,150` keeps the
h-peak + h-sidebands range only (the copy is ~30x smaller). Make the copy look like a run directory for the
pipeline:

```bash
mkdir -p <save_root>/<label> && ln -s <scored root>/stage1_output <save_root>/<label>/stage1_output
```

## 3. Cutoff

```bash
PYTHONPATH=$PWD python -m MVA_training.transformer_binary.cutoff_scan \
    --scored <save_root>/<label>/stage1_output --out <scan dir> --weight analysis
```

The scan reports the significance, the h-sideband data/MC flatness and the background-yield floor per cutoff
(details in the script's docstring). Write the chosen value into `configs/MVA/transformer_categorizer.yaml`. The
Run-2 value `0.088` passes the yield floor; no Run-2 cutoff passes the flatness gate, so that gate was not applied.

## 4. VBF DNN

**Preprocess** the transformer-channel events into four folds:

```bash
DNN_CONFIG=configs/dnn_run2_vbf.yaml bash run_analysis_pipeline.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 \
    -S <save_root> -l <label> -y "2018 2017 2016postVFP 2016preVFP" -m dnn_pre -T
# -> dnn/trained_models/<label>/2018-2017-2016postVFP-2016preVFP_h-peak_vbf_all   (D below)
```

0/1-jet events, which the transformer channel admits, get the training missing-jet sentinels. The rule is
recorded in `D/preprocess_manifest.json` (`missing_jet_rule`), and stage 2 applies the same rule automatically.

**HPO and training** on Ray GPUs:

```bash
python MVA_training/VBF_run3/ray_dnn.py submit hpo --config configs/dnn_run2_vbf.yaml --data-dir D \
    --out-dir D/hpo_optuna/<hpo label> --n-trials 96 --workers 8 --seed 12345 --label-smoothing-range 0.0,0.02
python MVA_training/VBF_run3/ray_dnn.py submit train --config configs/dnn_run2_vbf.yaml --data-dir D \
    --out-dir D/trained_best_optuna_<hpo label> --optuna-best-json D/hpo_optuna/<hpo label>/optuna_best.json
```

The trial objective and the early-stopping monitor both default to `early_stopping.monitor` of the config,
`val_ams_absWgt`. Rerunning the HPO command with the same `--out-dir` and `--seed` adds trials to the same study.
The reference model used 146 trials (96 + 50).

**Compare** against another model on the held-out folds:

```bash
python MVA_training/VBF_run3/evaluate_heldout.py --config configs/dnn_run2_vbf.yaml --data-dir D --out heldout.json \
    --model NEW=D:trained_best_optuna_<hpo label> --model OLD=<other model dir>:<its training tag>
```

### Binned AMS metric

`train_dnn.binned_ams` computes the ggH channel's binned AMS
(`stage2/ggH/score_edge_generation/determine_score_edge.py::calculate_AMS`), `sqrt(sum 2((s+b)ln(1+s/b)-s))`, on
`metrics.ams.n_bins` (20) equal-signal-weight score bins. The bins are merged from the top score down until each has
`B >= metrics.ams.bmin` (0.25 on one validation fold, about `B >= 1` on the full sample) and background MC relative
error `<= metrics.ams.relmax` (0.3).

| monitor | weights in s and b |
| --- | --- |
| `val_ams_absWgt` (default) | absolute weights |
| `val_ams` | signed weights, as `calculate_AMS` is used in the ggH channel |

The merging is always decided on absolute weights. The weighted AUC (`val_auc_weighted`) does not see how pure the
top score bins are, which is where the VBF significance comes from.

!!! warning "TODO"
    The merging rule (`bmin`, `relmax`) is ad hoc: it comes from a bin-audit script, not from a derivation, and differs
    from the bin guards of `scan_bins_for_dnn.py` that the final binning uses (raw S+B >= 15, S >= 0.05, B >= 0.05, no MC
    error cut). It should be derived or aligned with those guards.

!!! note "Run-to-run noise"
    Retraining a trial with the same settings and seed does not reproduce its HPO value exactly on GPU: the best
    trial's mean validation AMS was 0.476 in the HPO and 0.466 when retrained. That difference is as large as the
    spread among the top eight trials. Compare models on the held-out folds, not on the HPO value.

## 5. Stage 2 → combine

Use a label for the model (`MODEL_LABEL`), and keep `-T -L compacted` on every stage-2 call:

```bash
P="bash run_analysis_pipeline.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -S <save_root> -l <label> -y \"2018 2017 2016postVFP 2016preVFP\""
export MODEL_LABEL=<label> TRAIN_LABEL=trained_best_optuna_<hpo label> STAGE2_YEARS_TOGETHER=1

# (a) score dump without variations, default binning
DUMP_SCORES=1 ALLOW_DEFAULT_DNN_BINNING=1 eval $P -m 2 -T -L compacted -o <postfix>Dump
# (b) scan the DNN bin edges on the dump and write the configs/MVA/VBF/dnn_binning.yaml entry of MODEL_LABEL
python MVA_training/VBF_run3/scan_bins_for_dnn.py \
    --stage2-scores <save_root>/<label>/stage2_histograms/score_<label>_<postfix>Dump_NoSyst \
    --output-dir <scan dir> --enforce-min-per-year --min-background-per-bin 0.01 --frac-tol 0.005 --write-config
# (c) stage 2 with variations, then the unchanged downstream steps
WITH_VARIATIONS=1 eval $P -m 2 -T -L compacted -o <postfix>
WITH_VARIATIONS=1 eval $P -m 2p -o <postfix>
WITH_VARIATIONS=1 eval $P -m 3 -o <postfix>
./run_in_pixi.sh combine bash run_stats_pipeline_VBF.sh -m 9 -y "2018 2017 2016postVFP 2016preVFP Run2" \
    -S <save_root> -l <label> -o <postfix>
```

`--enforce-min-per-year` makes every year's bins meet the background floor. Without it, edges scanned on the years
summed can leave per-year top bins with no background MC. Check the datacards for that:

```bash
python scripts/audit_vbf_datacard_bins.py <save_root>/<label>/stage3_datacards_<postfix>/score_<label>
```

It lists per year the bins with no background, the bins with sparse background MC, and their share of the
counting Z². It also gives the counting Z after merging sparse bins from the top.

## Reference values

Run 2, transformer channel, expected VBF significance from `run_stats_pipeline_VBF.sh -m 9` (2026-11-04).
These values predate the fix of the Run-2 single-top and ZZ names in `configs/samples/samples.yaml`, so they leave out
those backgrounds; the baseline row does too.

| DNN | binning | prefit | stat-only |
| --- | --- | --- | --- |
| AMS-tuned (146 trials) | per-year floor (step 5) | 1.406 | 1.581 |
| AMS-tuned (146 trials) | edges scanned on 2017 only, used for all years | 1.674 | 2.159 |
| cut-based channel, July06 DNN | cut-based reference | 1.455 | 1.654 |

With the 2017-only edges, the 2016 years have top bins with zero background MC that carry 50–75% of those
years' Z² (audit above).
