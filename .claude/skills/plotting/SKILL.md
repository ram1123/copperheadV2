---
name: plotting
description: Control/validation-plot conventions and entry points — run_plotter.py, validation_plotter_unified.py, PyROOT styling for MVA_training plots, the HF-sentinel guard rule. Analysis-specific, not a CMS POG topic.
---

# Plotting Conventions

Use this skill for work involving:

- `run_plotter.py` or `plotter/validation_plotter_unified.py`;
- PyROOT (not matplotlib) plotting scripts under `MVA_training/` or `plotter/`;
- the `hf*` jet-variable sentinel rule;
- picking which of two similarly-named plotting scripts is the maintained one.

Nothing here is a CMS recommendation — it's this repo's own plotting conventions, kept
consistent so plots from different scripts/authors are comparable.

## Reference selection

- `references/root-conventions.md` — PyROOT style rules (batch mode, gradient
  palette, `SetOptStat(0)`, explicit `SetMinimum`/`SetMaximum`), and the `hf*`
  sentinel-guard rule.
- `references/validation-plots.md` — how `run_plotter.py` /
  `validation_plotter_unified.py` work (dataset/process groups, categories, regions,
  the Z-pT weight comparison options), and the one known stale/duplicate script.

## Review checklist

1. New PyROOT plotting code follows `root-conventions.md` (batch mode set before any
   drawing; shared gradient palette for 2D plots; explicit axis ranges so
   multi-panel canvases are comparable).
2. Any `hf*` jet variable is guarded against the `-1` non-HF sentinel before being
   histogrammed.
3. Before writing a new plotting script, check whether an existing one under
   `plotter/` or `MVA_training/*/` already does the same thing — this repo has at
   least one confirmed stale duplicate (see `validation-plots.md`).
4. `run_plotter.py`'s input/output paths and `validation_plotter_unified.py`'s
   dataset/process-group lists are edited per-use, not read from a shared config —
   confirm they were actually updated for the request at hand before trusting a plot.
5. If a plotted variable depends on `do_add_jet_ID_vars` (or another switch gating
   which columns exist in stage‑1 parquet), confirm that switch was on for the run
   being plotted.

## Reporting categories

- implementation defect (e.g. missing sentinel guard, wrong script used);
- optional improvement (style/consistency);
- verification required (e.g. whether a variable was actually saved for this run).

Before treating a finding as new, check `.claude/reports/registry.md` — plotting
scripts have already been reviewed for stale duplicates and switch-gated columns.
