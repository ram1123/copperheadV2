# Make VBF DNN `jj_eta_region` configurable + encode it in the output dir name

**Date:** 2026-09-10
**Requested via:** `/coordinate` (user-supplied task description, two items)
**Status:** Done, verified

## What was done

1. **`MVA_training/VBF_run3/preprocess_dnn.py`**
   - `PreprocessConfig` (frozen dataclass) gained a `jj_eta_region: str` field.
   - `load_config()` reads `cfg["analysis"].get("jj_eta_region", "all")` (fallback only for
     configs predating this key — the shipped `dnn_run3_vbf.yaml` sets it explicitly).
   - `events_to_dataframe()`'s `applyRegionCatCuts(...)` call now passes
     `jj_eta_region=cfg.jj_eta_region` instead of the previously-hardcoded
     `"jj_both_central"` literal.
   - `make_output_dir()` gained a `jj_eta_region: str` param, appended to the output path:
     `f"{years_slug}_{region}_{category}_{jj_eta_region}"`.
   - New `--jj-eta-region` CLI override (mirrors the existing `--category`/`--region`
     override pattern via `object.__setattr__` in `main()`).
   - Manifest dict and two log lines (`[preprocess] ...`, `[main] ...`) now record
     `jj_eta_region` alongside `category`/`region`, for provenance.
2. **`common_workflow.sh`**: `dnn_jj_eta_region` is resolved as **`JJ_ETA_REGION` env var if
   set, else `analysis.jj_eta_region` parsed from the DNN config YAML (`${dnn_config}`) via
   a small inline `python3 -c` snippet, else `"all"`** (graceful fallback if the YAML is
   missing/unparseable); `dnn_base_dir` includes `_${dnn_jj_eta_region}`;
   `run_dnn_workflow_once()`'s `pre_cmd` passes `--jj-eta-region "${dnn_jj_eta_region}"`
   explicitly (so the exact value bash resolved is what `preprocess_dnn.py` gets — no
   re-read/drift); added a `log` line for visibility.
   **NOTE — this replaced a first, broken version** (`dnn_jj_eta_region="${JJ_ETA_REGION:-all}"`):
   because the wrapper *always* passes `--jj-eta-region`, and `preprocess_dnn.py`'s CLI
   override unconditionally wins over the YAML, that first version made editing
   `analysis.jj_eta_region` in the YAML a no-op through `run_analysis_pipeline.sh` (it always
   forced `all`). The user hit exactly this (set YAML to `jj_non_central`, banner still said
   `jj_eta_region: all`). Fixed so the YAML is the source of truth and the env var is only an
   override. Same fix applied to `run_dnn.sh`.
3. **`run_analysis_pipeline.sh`**: usage()/env-var docs updated to document `JJ_ETA_REGION`
   (mirroring the existing `MODEL_YEARS` doc block).
4. **`configs/dnn_run3_vbf.yaml`**: added explicit `analysis.jj_eta_region: "all"` with a
   comment listing valid values (`"all"` + `modules/selection.py`'s `PAIR_JJ_ETA_REGIONS`:
   `jj_both_central`, `jj_non_central`, `jj_one_fwd25_one_central`,
   `jj_one_he_one_central`, `jj_one_fwd30_one_central`, `jj_both_fwd25`, `jj_both_he`,
   `jj_both_fwd30`, `jj_one_he_one_fwd30`).
5. **`run_dnn.sh`** (a separate, `common_workflow.sh`-independent driver script for the same
   preprocess/HPO/train pipeline, not in the user's original file list — found by the
   code-reviewer, fixed here): added its own `JJ_ETA_REGION` var (same default/values),
   passes `--jj-eta-region` to `preprocess_dnn.py`, and appends it to `OUT_BASE` so its
   `hpo_optuna.py`/`train_dnn.py` `--data-dir` still points at the directory
   `preprocess_dnn.py` actually writes.
6. **`MVA_training/VBF_run3/train_dnn.py`**: one-line docstring example path updated to
   include `_<JJ_ETA_REGION>` (comment-only, no behavior change).
7. **`MVA_training/VBF_run3/hpo_optuna.py`**: checked directly (user asked explicitly
   whether this file had been reviewed) — confirmed it needs no functional change: it
   takes `--data-dir`/`--out-dir` as opaque required CLI strings and only ever builds
   paths relative to them; its own `load_config` import (line 33) is `train_dnn.py`'s
   `TrainConfig.load_config`, a separate function from `preprocess_dnn.py`'s
   `PreprocessConfig.load_config` that never reads `analysis.category`/`region`/
   `jj_eta_region` at all, so it was never coupled to this change. Fixed the same class of
   stale-docstring-example nit as item 6 (module docstring's example path lacked the new
   suffix).

No changes made to `MVA_training/VBF_run2_legacy/**`, which has a similar-looking hardcoded
`{year}_{region}_{category}{DIR_TAG}` output-path pattern in its own `dnn_preprocessor.py` —
deliberately out of scope (separate, frozen legacy Run-2 flow per `CLAUDE.md`'s mode table).

## Source-of-truth / default resolution (after the follow-up fix)

The effective `jj_eta_region` for a run is resolved as:
1. `JJ_ETA_REGION` env var, if set and non-empty (a one-off override); else
2. `analysis.jj_eta_region` in the DNN config YAML (`configs/dnn_run3_vbf.yaml` by default,
   or `$DNN_CONFIG`) — **this is the persistent source of truth, edit it here**; else
3. `"all"` (fallback for a YAML that predates this key / can't be parsed).

Both `common_workflow.sh` and `run_dnn.sh` resolve this the same way and use the result for
both the `--jj-eta-region` passthrough and the `_<jj_eta_region>` output-dir suffix.
`preprocess_dnn.py` run directly (no `--jj-eta-region`) uses YAML step 2 / fallback step 3.

## Default value: "all" (not "jj_both_central") — a mid-task correction, not mine

I initially implemented the default as `"jj_both_central"` (preserving the exact behavior of
the just-completed/still-resuming full-Run3 training run, per "preserve existing user
changes"), and verified it end-to-end that way. Partway through, `common_workflow.sh` and
`configs/dnn_run3_vbf.yaml` changed on disk (concurrent edit, not made by me — flagged by the
harness) to set the default to `"all"` (no dijet-eta restriction) instead. Per instruction I
did not revert this — it's a legitimate, deliberate default choice for the repo owner to make
(arguably the more defensible one: `"all"` is the unbiased default for a newly-generalized
option, with `jj_both_central` — an ad hoc, undocumented restriction before this change —
now just one explicit, fully-available choice among many). I updated my own
comments/docstrings (which had said "default matches current both-jets-central behavior") to
match this actual default, and re-verified the whole chain end-to-end against it (see
Verification below). **Practical consequence**: a `dnn`/`dnn_pre`/`dnn_train` run through
either driver script with `JJ_ETA_REGION` unset now applies **no** jj-eta restriction by
default — different from the just-completed full-Run3 run reviewed earlier this session
(which had `jj_both_central` hardcoded). Set `JJ_ETA_REGION=jj_both_central` (or
`--jj-eta-region jj_both_central` directly) to reproduce that run's selection.

## Bug found and fixed via code-reviewer sub-agent

`run_dnn.sh` (missed in the initial scope — not one of the two files the user named) computed
its own `OUT_BASE` without the new `jj_eta_region` suffix, while `preprocess_dnn.py`'s
`make_output_dir()` now unconditionally appends it. Left unfixed, `run_dnn.sh`'s preprocessing
step would write to `.../<years>_<region>_<category>_<jj_eta_region>/` while its
`hpo_optuna.py`/`train_dnn.py` steps would read/write `.../<years>_<region>_<category>/`
(no suffix) — a `FileNotFoundError` on a fresh run, or silent training against stale data from
a directory predating this change on a reused path. Fixed to mirror `common_workflow.sh`'s
convention exactly (see item 5 above).

## Verification

- `bash -n` on `common_workflow.sh`, `run_analysis_pipeline.sh`, `run_dnn.sh`: all pass.
- `python -m py_compile` on `preprocess_dnn.py`, `train_dnn.py`: both pass.
- `preprocess_dnn.py --help` (inside the `default` pixi env): `--jj-eta-region` registers
  correctly with the expected help text.
- Direct Python check (`load_config` + `make_output_dir`, inside the pixi env): YAML default
  resolves to `cfg.jj_eta_region == "all"`; `object.__setattr__` override to
  `"jj_non_central"` works; `make_output_dir` produces distinct, correctly-suffixed paths for
  two different `jj_eta_region` values.
- `common_workflow.sh` (sourced, `parse_common_args` called directly): `dnn_base_dir` resolves
  to `..._all` with `JJ_ETA_REGION` unset, and `..._jj_both_central` with
  `JJ_ETA_REGION=jj_both_central` — consistent with the Python side.
- `run_dnn.sh`'s `OUT_BASE` pattern manually traced: `..._all` with the var unset, matching
  `common_workflow.sh`'s convention.
- `applyRegionCatCuts` (`modules/selection.py`) already validates `jj_eta_region` against
  `"all"` + `PAIR_JJ_ETA_REGIONS`/`SINGLE_JET_ETA_REGIONS` and raises a clear `ValueError`
  for anything else — no duplicate validation added on the bash/config side by design.
- Not run: an actual end-to-end `preprocess_dnn.py` invocation against real stage1 data (would
  require a Dask Gateway cluster and real compacted parquet; the unit-level checks above cover
  the actual code paths this change touches without that cost).

## Observed during this work (not part of the task, noted for awareness)

`git status` showed several touched files as `MM` (staged **and** further modified) partway
through — evidence that another process (most likely the user, in their own terminal) ran
`git add` on these files mid-edit, capturing a snapshot that includes most but not all of my
changes into the index. I did not stage or commit anything myself, per `CLAUDE.md`'s "do not
commit ... unless explicitly requested" rule. Worth re-`git add`-ing before committing, so the
commit captures the final state reviewed here rather than the earlier partial snapshot.
