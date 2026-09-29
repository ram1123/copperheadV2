---
title: Workflow Management
---

# Workflow management

This page documents the Snakemake-based analysis workflow in [workflow/Snakefile](../workflow/Snakefile).

The workflow is an orchestration layer. It does not reimplement the analysis itself. Most heavy steps are delegated to:

- [run_analysis_pipeline.sh](../run_analysis_pipeline.sh)
- [validation_plotter_unified.py](../plotter/validation_plotter_unified.py)

## References

- [CMS CAT hackathon](https://indico.cern.ch/event/1623559/timetable/?view=standard)
- [Snakemake CMS Tutorial](https://alefisico.github.io/snakemake-cms-tutorial/index.html)

## Files

- Main workflow: [workflow/Snakefile](../workflow/Snakefile)
- Main config: [workflow/config.yaml](../workflow/config.yaml)
- Summary helper: [workflow/summary_table.py](../workflow/summary_table.py)

## Environment

Run Snakemake from the repository root inside the default analysis environment. Interactively:

```bash
./enter_pixi.sh default
```

From a non-interactive context (`nohup`, a background job, an agent), `enter_pixi.sh` hangs waiting on
stdin. Use `run_in_pixi.sh` instead, which runs one command and exits:

```bash
./run_in_pixi.sh default snakemake -s workflow/Snakefile <target> ...
```

## Rules

Rules run by the default target `all`:

1. `stage1`
2. `stage1Compact`
3. `plots`
4. `zpt0`
5. `zpt1`
6. `zpt2`
7. `summary`

Opt-in rules (not part of `all`; request them explicitly):

- `pu_dnn_train`
- `dnn_pre` -> `dnn_hpo` -> `dnn_train` (VBF DNN)
- `stage2` -> `stage2_plot`, `stage3`
- `MassCalibrationMC`, `MassCalibrationMCClosure`, `MassCalibrationData`, `MassCalibrationDataClosure`
  (their lines in `all` are commented out)

The aggregate convenience targets are:

- `all`
- `all_stage1` (`stage1` + `stage1Compact` + `plots`)
- `DY_ReRun` (`stage1` + `stage1Compact` only)
- `plot_all` (`plots` only)
- `vbf_dnn` (VBF DNN chain)
- `stage23_all` (`stage2_plot` + `stage3` for every year in `years`)

## What each rule does

- `stage1`
  Runs stage-1 production through [run_analysis_pipeline.sh](../run_analysis_pipeline.sh) with `-m 1 -Z`
  (`-Z` also writes the cutflow), using `switches_yaml` if set.

- `stage1Compact`
  Runs the compaction step through [run_analysis_pipeline.sh](../run_analysis_pipeline.sh) with `-m compact`, then validates the compacted outputs with [compact_sanity_check.py](../scripts/compact_sanity_check.py). See also [Compact Sanity Check](CompactSanityCheck.md).
  If `cleanup_f1_0_after_compact` is true, it then **deletes** the raw `f1_0` stage-1 output for that year (see [Configuration](#configuration)).

- `plots`
  Runs [validation_plotter_unified.py](../plotter/validation_plotter_unified.py) on the compacted stage-1
  outputs, once per combination of the `plot_*` config fields.

- `zpt0`, `zpt1`, `zpt2`
  Run the Z pT fitting/training flow through [run_analysis_pipeline.sh](../run_analysis_pipeline.sh) using:
  - `-m zpt_fit0`
  - `-m zpt_fit1`
  - `-m zpt_fit2`

- `MassCalibrationMC`, `MassCalibrationMCClosure`
  Run the MC mass calibration steps through [run_analysis_pipeline.sh](../run_analysis_pipeline.sh) using:
  - `-m calib`
  - `-m calib_closure`

- `MassCalibrationData`, `MassCalibrationDataClosure`
  Run the data mass calibration steps through [run_analysis_pipeline.sh](../run_analysis_pipeline.sh) using:
  - `-m calib`
  - `-m calib_closure`

- `pu_dnn_train`
  Retrains the jet-level pileup DNN for one year with `-m pu_dnn_train`. Depends on `stage1Compact`.

- `dnn_pre`, `dnn_hpo`, `dnn_train`
  VBF DNN preprocess / Optuna HPO / final training (`-m dnn_pre|dnn_hpo|dnn_train`). One job for all
  `dnn_years` together (default: `years`), so these rules have no `{year}` wildcard. `dnn_pre` depends on
  `stage1Compact` for every year in `dnn_years`.

- `stage2`, `stage2_plot`, `stage3`
  Per-year stage-2 histograms (`-m 2`), stage-2 plots (`-m 2p`) and stage-3 datacards (`-m 3`). `stage2`
  depends on `stage1Compact` for its year and on `dnn_train`, because it evaluates the VBF DNN.

- `summary`
  Prints the SUCCESS/FAILED counts from `logs/<run_tag>/failed_jobs.log` and fails if any job failed.

## Configuration

The workflow behavior is controlled through [workflow/config.yaml](../workflow/config.yaml). Update it
before launching large runs. Every field can also be set per invocation without editing the file, e.g.
`--config use_existing_stage1=True`.

General:

- `hmm_base` -- filesystem base for the ntuple area.
- `years` -- years to run over.
- `run_tag` -- names the output, `logs/<run_tag>/` and `state/<run_tag>/` directories.
- `use_gateway`, `cluster_index` -- run on the Dask Gateway cluster with this index.

Stage-1:

- `switches_yaml` -- standalone switches file used by `stage1` instead of
  `configs/switches/switches_official.yaml`.
- `cleanup_f1_0_after_compact` -- after `stage1Compact` passes its sanity check, `rm -rf` the raw `f1_0`
  output for that year. **Irreversible**; only the compacted output is kept.

Skipping upstream steps:

- `use_existing_stage1` -- skip the stage-1 / `stage1Compact` dependencies of `plots`, `zpt0`, the
  `MassCalibration*` rules, `dnn_pre`, `pu_dnn_train` and `summary`, and drop stage-1 from `all` /
  `all_stage1`. Use when stage-1 output for `run_tag` already exists.
- `stage23_standalone` -- drop all inputs of `stage2` (`stage1Compact` and `dnn_train`), so stage-2/3 run on their own.

`use_existing_stage1` and `stage23_standalone` only remove Snakemake's dependency checks: they do not
confirm that the stage-1 output or DNN model actually exist. If they are missing, the job fails when it runs.

VBF DNN:

- `dnn_config`, `dnn_years`, `dnn_jj_eta_region` -- VBF DNN training config, the years it is trained on,
  and an optional `JJ_ETA_REGION` override.
- `dnn_model_label` -- load the stage-2 VBF DNN from `dnn/trained_models/<dnn_model_label>/` instead of
  this `run_tag`'s label (passed to the pipeline as `MODEL_LABEL`). Training output is unaffected.
- `dnn_model_years` -- years part of the model directory name to load (passed as `MODEL_YEARS`), e.g. all
  Run-3 years to use the combined-year model. Empty means each stage-2 job looks for a model named after its own year.

Stage-2 / stage-3:

- `with_variations` -- run `stage2`, `stage2_plot` and `stage3` with systematic variations (passed as
  `WITH_VARIATIONS=1`). Default `false` = nominal only, and the stage-2 output directory gets a `_NoSyst` suffix.
- `save_postfix` -- output-directory postfix shared by `stage2`, `stage2_plot` and `stage3` (the `-o` flag
  of `run_analysis_pipeline.sh`). Empty = the date the Snakemake command started, e.g. `Sep29_2026`.
- `jj_eta_region` -- dijet |eta| phase space for `stage2`, `stage2_plot` and `stage3` (passed as
  `JJ_ETA_REGION`): `all` (default) or one of `PAIR_JJ_ETA_REGIONS` in
  [modules/selection.py](../modules/selection.py), e.g. `jj_both_central`, `jj_non_central`. It also selects
  the DNN model directory `<model years>_h-peak_vbf_<jj_eta_region>`.

Plotting (`plots` rule):

- `plot_vbf_filter_study` -- pass `--vbf_filter_study` to the plotter.
- `plot_categories` -- subset of `nocat` / `vbf` / `ggh`.
- `plot_njets_plots` -- per non-vbf category, which of `inclusive` / `0` / `1` / `2` jet bins to plot.
- `plot_jj_eta_regions`, `plot_single_jet_eta_regions` -- dijet / single-jet eta-region splits.
- `plot_zpt_modes` -- `withzpt` and/or `nozpt`.
- `plot_var_sets` -- `minset` and/or `fullset`.
- `plot_extra_variables` -- extra variable groups from `configs/variables/variable_lists.py`.
- `plot_output_suffix` -- suffix on the plot output directory, so a rerun with different settings does
  not overwrite earlier plots.

Z pT:

- `zpt.*`

## Common commands

### Run the full workflow

```bash
snakemake -s workflow/Snakefile \
  -j 1 \
  --resources gateway=1 \
  --rerun-incomplete \
  --restart-times 3 \
  --latency-wait 60
```

### Run with an explicit config file

```bash
snakemake -s workflow/Snakefile \
  --configfile workflow/config.yaml \
  -j 1 \
  --resources gateway=1 \
  --rerun-incomplete \
  --restart-times 3 \
  --latency-wait 60
```

### Run only plotting targets

```bash
snakemake -s workflow/Snakefile plot_all \
  -j 1 \
  --resources gateway=1 \
  --rerun-incomplete \
  --restart-times 3 \
  --latency-wait 60 \
  --config use_existing_stage1=True
```

Without `use_existing_stage1=True`, `plots` first schedules `stage1`/`stage1Compact` for any year whose
`.done` marker is missing.

### Run only the DNN

```bash
# VBF DNN (dnn_pre -> dnn_hpo -> dnn_train)
snakemake -s workflow/Snakefile vbf_dnn \
  -j 1 --rerun-incomplete --restart-times 3 --latency-wait 60 \
  --config use_existing_stage1=True

# Pileup DNN, one year
snakemake -s workflow/Snakefile state/<run_tag>/pu_dnn_train_2024.done \
  -j 1 --rerun-incomplete --restart-times 3 --latency-wait 60 \
  --config use_existing_stage1=True
```

If `dnn_train.done` already exists, `vbf_dnn` has nothing to do. Add `--forcerun dnn_pre` to rerun the
whole chain, or `--forcerun dnn_train` to retrain only.

### Run only stage-2 and stage-3

```bash
snakemake -s workflow/Snakefile stage23_all \
  -j 4 \
  --resources gateway=1 \
  --rerun-incomplete \
  --restart-times 3 \
  --latency-wait 60 \
  --config stage23_standalone=True
```

This runs `stage2` -> `stage2_plot` and `stage3` for every year in `years`. For a single year, request
`state/<run_tag>/stage3_<year>.done` instead of `stage23_all`. Years already marked done are skipped;
add `--forcerun stage2` to redo them.

To include systematic variations, add `with_variations=True` to `--config` (or set it in
[workflow/config.yaml](../workflow/config.yaml)). It applies to all three rules at once. Keep it that way:
`stage2_plot` finds the stage-2 output by its `_NoSyst` suffix, so the rules must agree. The `.done`
markers do not record which setting was used, so a year that already ran nominal-only needs
`--forcerun stage2` to rerun with variations.

To restrict the VBF selection to one dijet-eta region, add `jj_eta_region=<region>` (e.g.
`jj_both_central`). All three rules get the same region. It is also added to their state and log names
(`state/<run_tag>/stage3_<year>_<region>.done`), so each region runs and is tracked separately from the
`all` run and from each other. For the two-region stats combination (`run_stats_pipeline_VBF.sh -m 12`),
run this once with `jj_both_central` and once with `jj_non_central`:

```bash
snakemake -s workflow/Snakefile stage23_all \
  -j 4 --resources gateway=1 --rerun-incomplete --restart-times 3 --latency-wait 60 \
  --config stage23_standalone=True jj_eta_region=jj_both_central
```

`stage2_plot` and `stage3` also find the stage-2 output by its `save_postfix`. By default this is fixed
once, at the date the Snakemake command starts, so a single invocation stays consistent even past
midnight. If you resume on a later day, or want to run only `stage2_plot`/`stage3` on an existing stage-2
output, set `save_postfix` to the postfix that stage-2 output was written with (e.g.
`--config save_postfix=Sep28_2026`).

### Stage-2 with a DNN model from a different run tag

By default stage-2 loads the model from

```
dnn/trained_models/<label>/<model years>_h-peak_vbf_<jj_eta_region>/trained_best_optuna_<HPO_LABEL>/fold*/best_torchscript.pt
```

where `<label>` is `Run2_nanoAODv<nano>_<run_tag>` or `Run3_nanoAODv<nano>_<run_tag>` depending on the
year, and `<model years>` is the stage-2 job's own year. To reuse a model trained under another tag, e.g.
the combined-year Aug30 model:

```yaml
# workflow/config.yaml
dnn_model_label: "Run3_nanoAODv12_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation_Systematics"
dnn_model_years: ["2022preEE", "2022postEE", "2023", "2023BPix", "2024", "2025", "2026"]
```

Set `dnn_model_years` whenever the model directory only has a combined-year model: stage-2 runs one year
at a time, so without it each job looks for `<year>_h-peak_vbf_<jj_eta_region>` and fails if that is
missing. `<jj_eta_region>` comes from the `jj_eta_region` config field when it is not `all`, otherwise
from `analysis.jj_eta_region` in `dnn_config`. The selection region and the model's training region are
therefore the same whenever `jj_eta_region` is set, and the model directory for that region must exist.
Outside Snakemake, the same override is the `MODEL_LABEL` / `MODEL_YEARS` env vars of
[run_analysis_pipeline.sh](../run_analysis_pipeline.sh).

### Force rerun a rule

```bash
snakemake -s workflow/Snakefile \
  -j 1 \
  --resources gateway=1 \
  --rerun-incomplete \
  --restart-times 3 \
  --latency-wait 60 \
  --forcerun stage1Compact
```

### Rerun plots without reopening the full DAG

```bash
snakemake -s workflow/Snakefile plot_all \
  -j 3 \
  --resources gateway=1 \
  --rerun-incomplete \
  --restart-times 3 \
  --latency-wait 60 \
  --allowed-rules plots \
  -R plots \
  --config use_existing_stage1=True
```

## Inspect workflow state

### Summary table

```bash
snakemake -s workflow/Snakefile --summary
```

```bash
snakemake -s workflow/Snakefile --summary | python workflow/summary_table.py
```

### Visualize the DAG

```bash
snakemake -s workflow/Snakefile --dag | dot -Tpng > dag.png
snakemake -s workflow/Snakefile --dag | dot -Tpdf > dag.pdf
```

### Visualize the rule graph

```bash
snakemake -s workflow/Snakefile --rulegraph | dot -Tpng > rulegraph.png
snakemake -s workflow/Snakefile --rulegraph | dot -Tpdf > rulegraph.pdf
```

## Notes

- The Snakemake workflow now uses [run_analysis_pipeline.sh](../run_analysis_pipeline.sh), not the legacy `stage1_loop_Improved.sh`, for analysis production.
- `use_existing_stage1` / `stage23_standalone` in [workflow/config.yaml](../workflow/config.yaml) let you run plotting, the DNNs, or stage-2/3 on already produced upstream outputs without regenerating them (see [Configuration](#configuration)).
- Plotting, Z pT, and mass calibration should ideally consume explicit input paths from the workflow configuration or command construction rather than relying on unrelated defaults in external YAML files.

## Known cleanup direction

One workflow-design issue to keep improving is input-path ownership:

- plotting
- Z pT
- mass calibration

These steps should continue to take their effective input locations from Snakemake-controlled paths and configuration, so the workflow remains reproducible and easy to reroute.
