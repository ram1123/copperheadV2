# Add `jj_eta_region` (all / jj_both_central / jj_non_central / ...) to VBF stage-2 and stage-3

**Date:** 2026-09-12
**Requested via:** `/coordinate` — "run stage-2/3 for VBF in jj phase spaces all, jj_both_central,
or jj_non_central; currently not included in stage-2/3, please add"
**Status:** Implemented, verified at the bash/argparse/logic level (see Verification — no live
Dask/Combine run; VOMS proxy blocked it, see Not done).

## Context found before implementing

The underlying selection mask logic already existed and was already generic/reusable:
`modules/selection.py::applyRegionCatCuts` has taken a `jj_eta_region: str = "all"` kwarg since
the 2026-09-04 njets-first-region-split work (`PAIR_JJ_ETA_REGIONS` incl. `jj_both_central`/
`jj_non_central`, validated against real data as an exact partition per the 2026-09-03
investigation report), and it already does its own validation/`ValueError`s — no new selection
logic was needed, only new callers. The 2026-09-10 DNN-preprocessing work
(`MVA_training/VBF_run3/preprocess_dnn.py`) made `jj_eta_region` configurable for **DNN
training** input, a separate, independent knob from what this task needed (VBF **category
selection** at stage-2/3) — mirrored its conventions (env var name, CLI flag pattern,
suffix-encoded output paths) but did not touch or reuse its code, since training-sample
phase-space and analysis-category phase-space don't have to be the same value.

## What was implemented

1. **`run_stage2_vbf.py`** (the only place doing real event selection here):
   - `jet1_eta`/`jet2_eta` added to `SHIFTED_SELECTION_VARIABLES` and `columns_for_selection()`'s
     `base_names` — needed because `applyRegionCatCuts` falls back to computing the region mask
     from these when no precomputed mask field exists in `events` (confirmed via the compacted
     parquet schema: no precomputed `jj_both_central`-style field exists, so this fallback path
     is what actually runs). Same `resolve_variation_field` fallback-to-nominal treatment as the
     existing jet-derived columns (`jet1_pt`, `jj_mass`, ...).
   - New `--jj_eta_region` CLI arg, `choices=["all"] + selection.PAIR_JJ_ETA_REGIONS` (deliberately
     **excludes** `SINGLE_JET_ETA_REGIONS`: this script's VBF category always needs >=2 real jets
     via `vbf_cut`, so a njets==1 region would silently select zero events rather than erroring —
     narrowed the choices to prevent that footgun rather than leaving it to blow up quietly).
   - `CoffeaStage2VBFProcessor.__init__` gained `jj_eta_region="all"`, stored and passed into the
     existing `applyRegionCatCuts(...)` call.
   - `histDirName` gets `_<jj_eta_region>` appended when not `"all"` — same position/pattern as
     the existing `do_vbf_filter_study` -> `_vbf_filter_study` suffix (before `_NoSyst`).
2. **`run_stage3_vbf.py`** (reads already-built stage-2 histograms; does no event selection):
   - New `--jj_eta_region` CLI arg (plain string, default `"all"`) — purely for locating the
     matching stage-2/3 output directory, mirroring the existing `--vbf_filter_study` handling
     verbatim: appended to `stage2_model_suffix` (-> `global_path_postfix`/`outpath_postfix`) in
     the same position/pattern.
3. **`common_workflow.sh`**:
   - New `jj_eta_region="${JJ_ETA_REGION:-all}"` resolved in `parse_common_args()` — reuses the
     `JJ_ETA_REGION` env var name already established for the DNN side, but resolved
     **independently** (no YAML source-of-truth; unset simply means `"all"`), since these are two
     separate knobs that don't have to agree.
   - `stage3_output_postfix()` appends `_<jj_eta_region>` (after `_vbf_filter_study`), same pattern
     as the existing `do_vbf_filter_study` branch — this one function feeds `vbf_card_dir()`, so
     the whole VBF stats pipeline (significance/impacts/lhscan/limit) automatically targets the
     right directory with no separate change needed there.
   - `build_stage2_cmd`, `build_stage3_cmd`: pass `--jj_eta_region "${jj_eta_region}"` when not
     `"all"` (mirroring the existing `--vbf_filter_study` passthrough).
   - `build_stage2_plot_cmd`: `stage2_suffix` gets `_<jj_eta_region>` inserted between the
     `_vbf_filter_study` and `_NoSyst` pieces, in the same relative position `run_stage2_vbf.py`'s
     own `histDirName` uses it, so the two independently-computed path strings stay byte-identical
     (verified, see below).
   - `print_run_configuration()`: added a `jj_eta_region (stage2/3): ...` line for visibility.
4. **Docs**: `run_analysis_pipeline.sh`'s `JJ_ETA_REGION` doc block (added by the 2026-09-10 DNN
   work) rewritten to describe both independent uses (DNN-training vs stage2/3-category-selection)
   under the one shared env var name. `run_stats_pipeline_VBF.sh`'s `usage()` gained a short
   `JJ_ETA_REGION` note (mismatch vs. what stage2/3 was actually built with looks exactly like a
   wrong `-o`: `Missing VBF SR/SB datacards for <year>` — the same class of gotcha hit and fixed
   for the date-postfix issue in the 2026-09-10 stats-pipeline-mode9-run report).

## Design decision: default `"all"` never changes existing output paths

Unlike the DNN-preprocessing convention (which always appends `_<jj_eta_region>`, even for
`"all"`), every path here only gets the `_<jj_eta_region>` suffix when the value is **not**
`"all"`. Deliberate: stage-2/3 already has a large, actively-used footprint of un-suffixed output
directories (every existing label in `commands.md`, the just-completed `Sep09_2026` stage-3
datacards this session's earlier task ran significance against) — silently renaming the default
path would have orphaned all of that. Verified this holds: with `JJ_ETA_REGION` unset, every
generated command and path below is byte-identical to before this change (no `_all` anywhere).

## Verification

- `python3 -c "import ast; ast.parse(...)"` on `run_stage2_vbf.py`, `run_stage3_vbf.py`: pass.
- `bash -n` on `common_workflow.sh`, `run_analysis_pipeline.sh`, `run_stats_pipeline_VBF.sh`: pass.
- `--help` (direct pixi `default`-env interpreter, no proxy needed) on both `run_stage2_vbf.py`
  and `run_stage3_vbf.py`: `--jj_eta_region` registers with the expected choices/help text.
- Direct functional test: sourced `common_workflow.sh`, ran `common_defaults` +
  `parse_common_args -y 2024 -l TestLabel -o TestPostfix`, then `build_stage2_cmd`/
  `build_stage3_cmd`/`build_stage2_plot_cmd`/`stage3_output_postfix`, twice — once with
  `JJ_ETA_REGION` unset, once with `JJ_ETA_REGION=jj_both_central`:
  - Unset: no `--jj_eta_region` flag anywhere, no `_jj_..`/`_all` suffix anywhere — identical to
    pre-change behavior.
  - Set to `jj_both_central`: `run_stage2_vbf.py` gets `--jj_eta_region jj_both_central`, whose
    internal `histDirName` computation (traced by hand) produces
    `score_TestLabel_TestPostfix_jj_both_central_NoSyst`; `build_stage2_plot_cmd`'s independently
    computed `load_path` lands on the exact same string; `run_stage3_vbf.py`'s
    `stage2_model_suffix`/`global_path_postfix` computation (also traced by hand) produces
    `TestPostfix_jj_both_central`, which `stage3/make_templates.py::load_stage2_output_hists`
    turns into the same `score_TestLabel_TestPostfix_jj_both_central_NoSyst` path stage-2 wrote to.
    `stage3_output_postfix()` -> `TestPostfix_jj_both_central` for `vbf_card_dir()` too. All four
    independently-built path strings agree.

## Not done / left for the user

- **No live Dask/Combine smoke test against real data.** `./enter_pixi.sh` requires a valid VOMS
  proxy; the one on disk had expired mid-session and regenerating it needs an interactive grid
  passphrase this session cannot supply. Everything above was verified at the argparse/bash-logic
  level, which covers the actual code paths this change touches, but a real
  `run_stage2_vbf.py --jj_eta_region jj_both_central` run against a small sample (confirming the
  `jet1_eta`/`jet2_eta` columns resolve correctly inside a real Dask task and the event count
  drops sensibly vs `--jj_eta_region all`) has not been done. Recommend doing this before relying
  on the feature for a real result.
- `stage2_plot`'s underlying `plotter/plot_DNN_score.py` was not inspected/changed — only the
  wrapper's `load_path`/`mva_name` construction was updated to match stage-2's new output
  location; if that script does its own independent path assumptions, it wasn't checked.
- CLAUDE.md's VBF pipeline section was not updated — the per-script `usage()`/`--help` text is
  the authoritative reference already, consistent with how the 2026-09-10 DNN `jj_eta_region` work
  also didn't touch CLAUDE.md.

## Files changed

`run_stage2_vbf.py`, `run_stage3_vbf.py`, `common_workflow.sh`, `run_analysis_pipeline.sh`,
`run_stats_pipeline_VBF.sh`.
