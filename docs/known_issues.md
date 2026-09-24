---
title: Known issues
---

# Known issues

## Pixi CUDA issue

When I try to run `pixi shell` command, it gives the following error:

```bash
[shar1172@purdue-af-182 copperheadV2_Feb2026_depot]$ pixi shell
Error:   × Cannot install environment 'default'
  ╰─▶ Virtual package '__cuda' does not match any of the available virtual packages on your machine: [__glibc=2.28=0, __unix=0=0, __archspec=1=zen2, __linux=4.18.0=0]
  help:  You can mock the virtual package by overriding the environment variable, e.g.: '`CONDA_OVERRIDE_CUDA=12.0`'
```

This only happens for environments that include the `cuda` feature (currently `default` and
`default-legacy` — the day-to-day analysis environments with ROOT/ML/symbolic-regression). Fix:
set `CONDA_OVERRIDE_CUDA` before running `pixi shell`, to a value >= the `cuda` entry on the
`linux-64-cuda` platform in `pixi.toml`'s `[workspace] platforms` list (currently `12.4`; check
that file if this stops working again after a version bump there).

```bash
export CONDA_OVERRIDE_CUDA=12.4
pixi shell
```

Environments that don't include the `cuda` feature (`ci`, `ci-legacy`, `combine`,
`combine-legacy`) resolve on GPU-less machines without any override — the CUDA requirement is
scoped per-feature via a named platform variant (`linux-64-cuda`) rather than applied
workspace-wide, so plain `pixi shell -e ci` (etc.) just works.

## Orphaned/pre-existing broken scripts (fixed 2026-09-12)

A repo-wide syntax sweep (`ast.parse` over every tracked `.py` file, done as part of a branch-merge
review) found 5 files that failed to parse. All 5 were confirmed broken identically on `main`,
`Week_June10`, and their common ancestor — i.e. pre-existing, unrelated to any particular merge or
PR, just never noticed because none of them is imported by anything that runs in CI or the normal
pipeline. Fixed to at least parse/import cleanly; noted below where a file is still not truly
functional beyond that.

- **`Run3_Resolution_Scripts/config/ntuples_path.py`** — every line was indented as if still inside
  a function body, with nothing to hold that indentation (a stray, never-imported copy of variables
  that are defined inline inside `Run3_Resolution_Scripts/fit_mass_resolution/fit_ZorHpeak.py`).
  Fixed: dedented to valid module-level assignments. Still not imported by anything; kept only as a
  historical reference snippet of old input paths.
- **`configs/MVA/MVA_subCat_calculation/sigEffScore.py`** — `scoreEdgesBySigEff(...)` was a bare
  function signature with no body at all, and has no callers anywhere in the repo. Fixed: gave it a
  `raise NotImplementedError` body with a docstring explaining it was never implemented, rather than
  inventing a sub-category edge-finding algorithm.
- **`plotter/stitchingValidation.py`** — two spots (an `axis_title = {...}...` line and a 4-line
  block building/saving the output canvas) had lost their indentation and fallen out of the
  enclosing `for plot_var in [...]:` loop body. Fixed: re-indented both to match the surrounding
  loop (8 spaces) — the obvious original intent, since one plot should be saved per `(Year,
  plot_var)` iteration.
- **`src/lib/categorizer.py`** — `CategorizerMVA.runMVA(self):` had no body. This whole file is an
  abandoned early-development stub (verified: `CategorizerBase`/`CategorizerCutSelect`/
  `CategorizerMVA` are never imported anywhere); the category logic actually used in the analysis
  lives in `modules/selection.py` (`applyRegionCatCuts`, `PAIR_JJ_ETA_REGIONS`/
  `SINGLE_JET_ETA_REGIONS`) and `configs/categories/`. Fixed only the parse error (added a
  `raise NotImplementedError` body to `runMVA`, added the missing `import argparse`) and added a
  module docstring documenting its dead status. **Still not functional as real code**: every
  `categorize(...)` method is missing `self`, `for key, val in cat_dict:` should be
  `cat_dict.items()`, and the `__main__` block has a pre-existing `yeawr` typo — left as-is rather
  than guessing at a real implementation for genuinely unfinished/abandoned logic.
- **`validation/DY_VBFFilter_production/VBF_filter_comparison.py`** — `quickPlotByHardProcess(...)`
  had a truncated `from_hardProcess = ` assignment (no right-hand side) and referenced an `exclude`
  variable that isn't a parameter of this function (its sibling `quickPlotByNMuon(...)`, which this
  was clearly copy-pasted from, does have `exclude=False`). Fixed: removed the incomplete/unused
  `from_hardProcess` line (never read afterward) and added the missing `exclude=False` parameter,
  copied verbatim from the sibling function rather than inventing GenPart "hard process" flag
  selection logic.

## `applyRegionCatCuts` imported from the wrong module (fixed 2026-09-12)

`plotter/get2Dplots.py` and `plotter/readParquetUsingRDataFrame.py` both did
`from plotter.validation_plotter_unified import applyRegionCatCuts` — that function has never been
defined in `validation_plotter_unified.py` (confirmed absent on `main`, `Week_June10`, and their
common ancestor alike; that module only ever calls `selection.applyRegionCatCuts` internally, never
re-exports it), so both scripts failed with `ImportError` the moment they were run, unrelated to any
particular merge. Fixed: import `applyRegionCatCuts` from `modules.selection` instead (the real
implementation), and updated both call sites' keyword arguments to match its actual signature — add
the required `variation` argument (`"nominal"`, matching every other caller in the repo) and rename
the `njets` keyword to `njets_selection` (the two scripts' `category`/`region_name`/`process`/
`do_vbf_filter_study` keywords already matched the real signature, so only these two needed to
change).

## `test/reference/switches_official.yaml` drifts out of sync with new required switch keys (fixed 2026-09-12)

`test/reference/switches_official.yaml` is a separately-maintained, intentionally-frozen snapshot of
`configs/switches/switches_official.yaml` that `.github/workflows/sync-stage1.yml` (and
`scripts/update_sync_references.sh --use-reference-switches`) copies over the real config before
running the sync sample, so CI's stage-1 output stays reproducible regardless of what production
switches currently say. Problem: nothing keeps this snapshot's *set of keys* in sync as
`configs/switches/switches_official.yaml` gains new switches over time. `src/copperhead_processor.py` reads
some switches via a plain `self.config["switches"]["some_key"]` (no default) — if such a key is
missing from the frozen snapshot, every sample crashes with a `KeyError` the moment
`--use-reference-switches` is used, since that mode replaces the whole file rather than merging keys.

Hit in practice with `do_reject_high_rawFactor_jets` (added to production switches_official.yaml after this
snapshot was last regenerated). Fixed by adding it to `test/reference/switches_official.yaml` with `false` for
every year — matching production's own universal-off default, so this is a zero-behavior-change fix,
not a new physics choice. A second key, `do_met_xy_correction`, was also found missing from the
snapshot but is read via `.config["switches"].get("do_met_xy_correction", False)`, so its absence
silently defaults to `False` (matching production everywhere anyway) — harmless, left as-is.

**If you add a new switch key that's read via a plain `["..."]` lookup (no `.get(..., default)`),
add it to `test/reference/switches_official.yaml` too** (with whatever value reproduces old/pre-change
behavior), or `--use-reference-switches` will start failing with a `KeyError` for every sample. To
check for drift directly:

```python
import yaml
prod = yaml.safe_load(open("configs/switches/switches_official.yaml"))["switches"]
ref = yaml.safe_load(open("test/reference/switches_official.yaml"))["switches"]
print("missing from test/reference/switches_official.yaml:", sorted(set(prod) - set(ref)))
```
