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

Use the default analysis environment before running Snakemake:

```bash
./enter_pixi.sh default
```

## Current main rules

The current workflow drives these top-level analysis steps:

1. `stage1`
2. `stage1Compact`
3. `plots`
4. `zpt0`
5. `zpt1`
6. `zpt2`
7. `MassCalibrationMC`
8. `MassCalibrationMCClosure`
9. `MassCalibrationData`
10. `MassCalibrationDataClosure`
11. `summary`

The aggregate convenience targets are:

- `all`
- `all_stage1`
- `plot_all`

## What each rule does

- `stage1`
  Runs the nominal stage-1 production through [run_analysis_pipeline.sh](../run_analysis_pipeline.sh) with `-m 1`.

- `stage1Compact`
  Runs the compaction step through [run_analysis_pipeline.sh](../run_analysis_pipeline.sh) with `-m compact`, then validates the compacted outputs.

- `plots`
  Runs [validation_plotter_unified.py](../plotter/validation_plotter_unified.py) on the stage-1 outputs.

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

## Configuration

The workflow behavior is controlled through [workflow/config.yaml](../workflow/config.yaml).

Important fields include:

- `hmm_base`
- `years`
- `run_tag`
- `use_gateway`
- `cluster_index`
- `zpt.*`

Update [workflow/config.yaml](../workflow/config.yaml) before launching large runs.

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
  --latency-wait 60
```

### Force rerun a rule

```bash
snakemake -s workflow/Snakefile \
  -j 1 \
  --resources gateway=2 \
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
  -R plots
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
- `use_existing_stage1` in [workflow/config.yaml](../workflow/config.yaml) is useful when you want to reuse already produced stage-1 outputs instead of regenerating them.
- Plotting, Z pT, and mass calibration should ideally consume explicit input paths from the workflow configuration or command construction rather than relying on unrelated defaults in external YAML files.

## Known cleanup direction

One workflow-design issue to keep improving is input-path ownership:

- plotting
- Z pT
- mass calibration

These steps should continue to take their effective input locations from Snakemake-controlled paths and configuration, so the workflow remains reproducible and easy to reroute.
