# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Copperhead V2 is a columnar, Dask/coffea-based analysis framework for the CMS Run 3 H→µµ search. It turns
NanoAOD ROOT files into skimmed parquet ntuples (stage-1), trains DNN for VBF channel, BDT for ggH channel, then builds histograms/categories from those ntuples
(stage-2), and produces Combine datacards/statistical results (stage-3 + VBF stats pipeline). It also hosts
several ML training subprojects (pileup jet ID DNN, VBF DNN, ggH BDT, Z-pT reweighting, symbolic regression)
used to derive corrections/discriminants consumed by stage-1/2.

## Environment setup

This repo uses **pixi** with multiple named environments composed from features (see `pixi.toml`). Always
enter one before running anything — do not assume a bare `python3` on PATH has the right packages.

```bash
./enter_pixi.sh default   # day-to-day analysis: coffea, ROOT, ML, symbolic regression, workflow tools
./enter_pixi.sh combine   # statistical inference / Higgs Combine (different ROOT pin than default)
./enter_pixi.sh ci        # minimal stack used by CI sync checks
```

`enter_pixi.sh` cd's into a shared CVMFS pixi project (`/cvmfs/cms-af.opensciencegrid.org/paf/pixi/copperheadV2`),
runs `pixi run -e <env> bash -lc '...'`, then returns to your working directory — it does not use a local
`pixi install`. Each environment also has a `-legacy` variant pinned to an older `coffea`/`dask-awkward`
release line (`default-legacy`, `ci-legacy`, `combine-legacy`) for migration/back-compat work.

Known issue: bare `pixi shell`/`default`(-legacy) can fail with a `__cuda` virtual-package mismatch on
machines without a real GPU, since those environments include the `cuda` feature. Fix:
`export CONDA_OVERRIDE_CUDA=12.4` (must be >= the `cuda` entry on the `linux-64-cuda` platform variant in
`pixi.toml`'s `[workspace] platforms`) before invoking pixi (see `docs/known_issues.md`). This is scoped
per-feature via that named platform variant, not workspace-wide — `ci`/`ci-legacy`/`combine`/`combine-legacy`
don't include the `cuda` feature and resolve on GPU-less machines with no override needed.

Scripts that need a specific env's dependencies (e.g. `torch`, `ROOT`) will fail with `ModuleNotFoundError`
if run with the wrong interpreter — check which pixi environment provides what in `pixi.toml`
(`[feature.*.dependencies]` blocks) rather than assuming.

## Common commands

Lint:
```bash
pixi run lint          # ruff check .
```

Stage-1 sync/regression check (what CI runs, see below) is the closest thing to a test suite — there is no
pytest suite. To validate a stage-1 change locally, rerun the sync sample and diff against `test/reference/`:
```bash
bash scripts/update_sync_references.sh          # refresh all reference years after an intentional change
bash scripts/update_sync_references.sh 2017      # single year
```

Snakemake tasks (`pixi.toml [tasks]`, thin wrappers around `workflow/Snakefile`):
```bash
pixi run stage1       # snakemake ... all_stage1
pixi run stage1full   # snakemake ... (full DAG)
pixi run plot         # snakemake ... plot_all
pixi run unlock       # snakemake --unlock
pixi run stage2       # bash stage2_sh.sh all all
pixi run check        # sanity-import coffea/awkward/uproot
```

Preferred direct entry point is `run_analysis_pipeline.sh` (wraps `run_prestage.py` / `run_stage1.py` /
`run_stage2*.py` / `run_stage3*.py`), driven by shared helpers in `common_workflow.sh`:
```bash
bash run_analysis_pipeline.sh -m 0 -y 2022preEE -v 12 -c configs/datasets/dataset_nanoAODv12_run3.yaml   # prestage
bash run_analysis_pipeline.sh -m 1 -y 2022preEE -v 12 -c configs/datasets/dataset_nanoAODv12_run3.yaml   # stage1
```
Modes (`-m`): `0|prestage`, `1|stage1`, `1a|compact`, `2|stage2`, `2p|stage2_plot`, `3|stage3`, `all`,
`zpt_fit*`, `calib`/`calib_closure`, `dnn|dnn_pre|dnn_train|dnn_var_rank`. The old monolithic
`stage1_loop_Improved.sh` still exists for transition/reference but new work should use
`run_analysis_pipeline.sh`.

VBF statistical pipeline (needs the `combine` pixi env, produced after stage-3 datacards exist):
```bash
./enter_pixi.sh combine
bash run_stats_pipeline_VBF.sh -m 10 -y Run3 -l label_for_ntuple   # full stage2/stage3/stats chain
```
Modes: `4` build VBF card/workspace (`-y` accepts a single year or a pseudo-year like
`Run3`/`Run2`/`Run2Run3` to combine already-built per-year cards into one), `5` significance,
`6` impacts, `7` likelihood scan, `8` build card/workspace + significance + collect summary,
`9` collect significance summaries only, `10` full stage2/stage3/stats chain (misnomer:
despite being named `vbf_limit` this runs significance, not `AsymptoticLimits`), `11` expected
95% CL limit via `AsymptoticLimits --run blind` + collect summary. `-y` accepts a
comma-separated list; each mode's steps run once per year in that list, so pass a single
pseudo-year (e.g. `Run3`) rather than the list of individual years to get one combined result
instead of per-year ones.

The analysis is blinded, so `6` (impacts) and `11` (limit) never fit real data: impacts run
twice per year — Asimov `r=1` (signal injected) and `r=0` (background-only) — instead of an
observed scenario, and the limit is the expected (Asimov, `--run blind`) limit.

Control/validation plots — edit input/output paths and dataset lists in `run_plotter.py` and
`plotter/validation_plotter_unified.py` first, then:
```bash
python run_plotter.py
```

Dask client for interactive/notebook work: open `DaskGatewaySLURM.ipynb` and run cells up to "Create Dask
Client"; remember to run the teardown cells when done to free gateway resources.

## Architecture

**Pipeline stages** (see `docs/Introduction.md`, `docs/workflow_management.md`):
1. **Prestage** (`run_prestage.py`) — resolves dataset YAML → per-sample JSON of input ROOT files + metadata
   (event counts, etc.), via DAS or explicit `/store` paths.
2. **Stage-1** (`run_stage1.py` → `src/stage1/runner_adapter.py` → `src/copperhead_processor.py`'s
   `EventProcessor`, a coffea `ProcessorABC`) — applies object/event selections, corrections (JEC/JER,
   Rochester, muon SF, FSR recovery, b-tag, PU, Z-pT), computes weights, writes skimmed parquet + cutflow
   JSON. Corrections live under `src/corrections/` (one module per correction: `jet.py`, `rochester.py`,
   `muon_sf.py`, `fsr_recovery.py`, `evaluator.py` for PU/b-tag/PDF/QGL weights, `zpt_dnn.py`, etc.).
3. **Compact** — post-processes/merges stage-1 parquet output into a leaner layout consumed by everything
   downstream (validated by `scripts/compact_sanity_check.py`).
4. **Stage-2** (`run_stage2.py`, `run_stage2_vbf.py`) — builds histograms/categories from compacted ntuples;
   category logic in `configs/categories/` and `src/lib/categorizer.py`.
5. **Stage-3** (`run_stage3.py`, `run_stage3_vbf.py`, plus `_ucsd`/`_validation` variants) — produces
   Combine datacards/templates (`stage3/make_datacards.py`, `stage3/make_templates.py`,
   `stage3/bkg_datacards_template/`) and ggH datacard generation (`stage2/ggH_datacard/`).
6. **VBF stats pipeline** (`run_stats_pipeline_VBF.sh`) — Combine significance/impacts/likelihood-scan chain
   on top of stage-3 datacards; requires the `combine` pixi env (different ROOT pin than `default`).

**Orchestration layers**, from outer to inner:
- `workflow/Snakefile` + `workflow/config.yaml` — Snakemake DAG over years/categories; delegates actual work
  to `run_analysis_pipeline.sh` and `plotter/validation_plotter_unified.py` rather than reimplementing logic.
  Key config fields: `hmm_base`, `years`, `run_tag`, `use_gateway`, `use_existing_stage1`, `zpt.*`.
- `run_analysis_pipeline.sh` + `common_workflow.sh` — the real driver; `common_workflow.sh` has one
  `build_<mode>_cmd` function per pipeline mode plus `run_zpt_fit`/`run_dnn_workflow_once`/
  `run_vbf_significance`/`run_vbf_impacts`/`run_vbf_lhscan` helpers.
- Individual `run_*.py` scripts — argparse entry points invoked by the shell layer; can also be run directly
  for debugging a single step.

**Config layout** (`configs/`): `datasets/*.yaml` (per-nanoAOD-version, per-run dataset lists incl. `sync_*`
used by CI), `parameters/*.yaml` (JEC, muon, electron, trigger, cross sections, luminosity,
`switches.yaml` for year-keyed feature flags, `correction_filelist.yaml`/`SF_filelist.yaml` pointing at
correction payloads), `categories/` (category cut definitions), `variables/variable_lists.py`, `samples/`,
`MVA/` (BDT subcategory calculation configs).

**Shared utilities** (`modules/`): generic, stage-agnostic helpers — `dask_utils.py` (client setup/teardown),
`xrootd_utils.py` (redirector fallback/normalization for AAA errors), `correctionlib_file_cache.py`,
`awkward_utils.py`, `vector_operations.py` (Lorentz-vector/CS-frame kinematics), `selection.py`,
`job_status.py` (stage-1 run summaries), `sample_config.py`, `git_utils.py` (embeds git commit/state into
output dirs — used by training/validation scripts for provenance), `root_2dColorProfile.py` (shared PyROOT
gradient palette for 2D plots — call `set_gradient_style()` once before drawing).

**MVA_training/** — standalone-ish training scripts producing artifacts consumed by stage-1/2 (e.g. the
pileup-jet DNN, VBF category DNN, Z-pT reweighting fits, ggH BDT — the latter is a git submodule pointing at
`ram1123/Run2_MVA_trainer`). These scripts each have their own CLI and typically support a `--replot-only`
style flag to regenerate plots from saved predictions without rerunning training/inference — check argparse
in the specific script before assuming a full rerun is needed. Validation outputs land under `validation/`,
organized by run label / region subdirectories.

**Testing model**: there is no pytest suite. Correctness is checked by the stage-1 **sync/regression test**
(`.github/workflows/sync-stage1.yml`, `ci`/`ci-legacy` pixi env): it reruns stage-1 on small reference
samples for years 2017 and 2022preEE, dumps event kinematics + cutflow JSON via
`scripts/sync_parquet_dimuon.py`, and diffs against snapshots in `test/reference/`. Any stage-1-affecting
change (touching `run_stage1.py`, `src/**`, `data/**`, `modules/**`, `configs/**`) should be validated this
way, and the reference snapshots regenerated via `scripts/update_sync_references.sh` if the change is
intentional. `pylint.yml` runs pylint on PR-changed files only via `workflow_dispatch` (not automatic).

## Working conventions specific to this repo

- ROOT plotting code (PyROOT, not matplotlib) follows the pattern in
  `MVA_training/pileup_symbolic_regression/corr_region_real_fake.py`: `ROOT.gROOT.SetBatch(True)`,
  `modules.root_2dColorProfile.set_gradient_style()` for the shared palette, `gStyle.SetOptStat(0)`,
  explicit `SetMinimum`/`SetMaximum` for comparable Z-axis ranges across multi-panel canvases.
- Scripts that embed provenance (e.g. `MVA_training/pileup_dnn/train_pu_dnn.py`) write `git_state.json` via
  `modules/git_utils.py` into their output directory — keep this pattern when adding new training scripts.
- `hf*` jet variables (`hfsigmaEtaEta`, `hfsigmaPhiPhi`, `hfcentralEtaStripSize`,
  `hfadjacentEtaStripsSize`, `hfEmEF`, `hfHEF`) are only physically meaningful for HF (forward, |eta| ≥ 3)
  jets; NanoAOD fills them with a `-1` sentinel for non-HF jets. Always guard on region/`-1` before
  histogramming these.
- Year strings are the canonical partition key throughout (`2016preVFP`, `2016postVFP`, `2017`, `2018`,
  `2022preEE`, `2022postEE`, `2023`, `2023BPix`, `2024`, `2025`); `modules/classify_year.py` maps these to
  `is_run2`/`is_run3`, and `configs/parameters/switches.yaml` keys feature flags by the same strings.

## Project scope

This repository contains a CMS physics analysis.

Before changing analysis logic, determine:

- physics process and final state;
- data-taking era;
- data or simulation;
- NanoAOD campaign and version;
- object working points;
- correction and scale-factor versions.

Never assume that Run 2, Run 3, and Phase-2 recommendations are
interchangeable.

## Working rules

- Inspect the existing implementation before suggesting changes.
- Make the smallest change that satisfies the request.
- Preserve existing user changes.
- Do not change physics selections without explaining the physics impact.
- Do not invent CMS recommendations, correction names, or working points.
- Run relevant validation after modifying code.
- Report commands executed and their results.
- Do not commit, push, merge, or delete files unless explicitly requested.

## CMS recommendations

For work involving physics objects, invoke the `cms-object-guidelines` skill.

Distinguish clearly among:

1. official CMS or POG recommendations;
2. analysis-specific requirements;
3. implementation choices;
4. suggestions that require validation.

If the stored reference does not cover the relevant era, campaign, or working
point, report that the recommendation requires authoritative verification.

## Agent coordination

For complex tasks involving multiple independent investigations, invoke the
`coordinate` skill.

The main agent must:

- read `.claude/reports/registry.md`;
- open only relevant historical reports;
- select appropriate specialists;
- provide each specialist with sufficient task-specific context;
- synthesize and verify their results;
- update the registry after substantial work.

Sub-agents must not read or update the registry unless the main agent
explicitly instructs them to do so.

## Validation

Before reporting that work is complete:

- inspect the resulting diff;
- run the smallest relevant test;
- run broader validation when practical;
- distinguish successful tests from tests that could not be run;
- identify assumptions that were not verified.

## Repository commands

Add this project's actual commands here, for example:

- Environment setup: `source ...`
- Build: `...`
- Unit tests: `...`
- Analysis test: `...`
- Formatting: `...`