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

## MuonScaRe resolution smearing can give astronomically large or negative muon pT in Run 3 MC (open, found 2026-09-29)

**Symptom.** Some Run 3 MC muons leave stage-1 with pT of 10^17–10^61 GeV, sometimes negative,
mostly at |eta| ≈ 1.8–2.2. Found with `scripts/sync_parquet_dimuon.py` in a dir-vs-dir compare
of two stage-1 runs: 6 DY events were flagged only on `dimuon_mass`. The muon pT values were
identical in both runs; the dimuon mass computed from them is float64 cancellation noise
(e.g. -2.2e47, 0, 1e11 GeV). So those differences are noise and point to bad inputs, not a code
change between the two runs.

**Cause.** `pt_resol` in `src/corrections/MuonScaRe.py` smears as `pt * (1 + k*std*rndm)`,
where `rndm = CrystallBall.invcdf(u)` for a uniform `u` from the `HashPRNG`. The tail branch
goes as `(NC/u)**(1/(n-1))`, so when the payload's `cb_params` tail parameter `n` is close to 1
the exponent is huge and ordinary `u` values explode (n = 1.004 gives an exponent of about 250).
`filter_boundaries` only resets pT when the *input* pT is outside [26, 200] GeV or the output is
NaN. It never checks that the output is finite and sensible, so huge or negative values pass through.

**Scale per payload** (`data/roch_corr/`: repo `invcdf` over |eta| 0–2.4 x nTrackerLayers 6–18,
312 bins; "bad" = |rndm| > 100 or non-finite for u down to 1e-12):

| Payload | Bad bins | Worst |
|---|---|---|
| 2022_Summer22 | 54 | n = 1.008 |
| 2022_Summer22EE | 52 | n = 1.004 at \|eta\| ≈ 2.05 |
| 2023_Summer23 | 50 | n = 1.012 at \|eta\| ≈ 2.05–2.15 |
| 2023_Summer23BPix | 52 | n = 1.003 |
| 2024_Summer24 | 42 | \|rndm\| up to ~1e19 |
| 2025_muon_scalesmearing_VXBS | 26 | \|rndm\| ≤ ~1.4e3 (mild) |

Some bins also have `n = 0`, `alpha = 0` (apparently unfitted). Those give NaN, which
`filter_boundaries` already resets to the input pT.

**Expected impact (not measured).** Affected events get a meaningless dimuon mass, so the
110–150 GeV window should drop them in stage-2: a small MC efficiency loss rather than signal-region
contamination. The 6 events in the sync diff are a lower bound, since only events whose mass noise
happened to differ between runs showed up. To size it, count muons with pT > 1e4 or pT < 0 per
sample and year in the stage-1 MC parquet.

**Open questions before fixing.** Not yet checked against the official MUO MuonScaRe reference
code: does it behave the same way, and does MUO recommend a guard? One candidate fix is to keep the
unsmeared pT when the smeared value is non-finite, negative, or far from the input. That would be an
implementation choice, not a CMS recommendation, and it changes stage-1 output (sync references would
need regenerating).

## VBF stage-3/stats problems found while running the jj-region scan (fixed 2026-10-04/05)

Found while running `scripts/run_jj_region_scan.sh -o Oct04_2026_Syst` (label
`Run3_nanoAODv15_FilterEvents_Sep22_tightPassLepVeto_OfficialRecomendation_Systematics_LumiSplit`).
The template-level ones were caught by `scripts/plot_template_systematics.py`, now a mandatory
gate after stage-3 (Snakemake `stage3_validate`, scan-script step `validate`). Open follow-ups
are in `To-Do.md`.

**1. One-sided shape nuisance breaks `text2workspace`.** `RuntimeError: Failed to find
DY_matched01J_LHEFac_DY_matched01JUp`. In low-statistics cards (e.g. `jj_both_fwd25` 2023 SR,
`DY_matched01J` nominal 1.6 events from a few large negative-weight events) one side of a variation
had a negative yield and `make_templates` dropped it, but `make_datacards` declared the nuisance
if *either* side existed, and looked at all regions of the group at once. Fix:
`stage3/make_datacards.py` declares a shape nuisance for a process only if both Up and Down exist
in that card's own region/channel, and logs a warning otherwise.

**2. Stats pipeline reused stale cards/workspaces.** `ensure_vbf_card`/`ensure_vbf_workspace`
(`common_workflow.sh`) returned early if `HMuMu_13TeV_<year>.txt`/`.root` existed, so after a
stage-3 rerun the fit silently used the old card. Fix: rebuild whenever an input card is newer
(`vbf_card_is_current`); same for the jj-combined card.

**3. Likelihood scan without a best-fit point.** `plot1DScan.py` `assert bestfit is not None`:
the scan's initial fit did not converge (no `quantileExpected == -1` row). Fix: `run_vbf_lhscan`
uses `--robustFit 1` with minimizer strategy 0, falling back to 2, like the impacts, and fails
loudly if neither gives a best fit.

**4. Negative template bins and signal-only bins break the background-only fit.** Negative-weight
NLO MC (mostly `DY_matched2J`) left negative bins (59-73 per region), and some high-score bins
had signal but zero background. A background-only (r=0) Asimov fit then has a kink at its own
minimum (r<0 predicts negative yields) and never converges, so r=0 impacts failed for
`jj_both_fwd25`. Fix in `stage3/make_templates.py`: clip negative bins of every non-data template
(nominal and variations) to 0, keeping sumw2; then give any bin with total background <= 0 a
1e-5 floor on the largest background, in its nominal and all its variations (no shape effect),
with the datacard yield updated. Before/after plots:
`validation/stage3_templates/<label>/Oct04_2026_Syst_<region>/templates_nominal_{unclipped,clipped}.pdf`.

**5. JES `Total` double-counted.** Stage-1 writes the `Total` JES variation next to the split
sources, and stage-2 histogrammed every discovered shape variation, so every datacard had `Total`
*and* the split sources (JES counted twice; `Total` was the largest nuisance, up to +-39%). Fix:
`run_stage2_vbf.py` skips `Total_up/down` (training's `stage2_shape_variations` still returns it,
the DNN "sweep" mode needs it).

**6. Fake identical JES/JER shifts from eta float noise.** 321 (process, nuisance) pairs had
Up == Down != nominal, with the *same* shift for unrelated sources (2023 `ttjets_dl`,
`jj_both_central` SB: `HF`, `HF_2023`, `jer5`, `jer6` all -0.649%). Cause: NanoAOD stores jet eta
coarsely, so many jets sit at exactly |eta| = 2.5; stage-1 recomputes the varied eta from the
shifted four-vector (2.5 -> 2.50000004), so every JES/JER variation moved those jets across the
`|eta| <= 2.5` region cut. Fix: `modules/selection.py` rounds |eta| and `jj_dEta` to 1e-4 before
the eta-boundary cuts (`ETA_CUT_DECIMALS`). Needs a stage-2 rerun; validated: `HF`/`jer5`/`jer6`
now select exactly the nominal events in `jj_both_central`.
Follow-up (fixed 2026-10-06): that rounding used `np.round`, which is not a ufunc and raises
`NotImplementedError` on dask-awkward arrays, so every `preprocess_dnn.py` run (it passes lazy
arrays to `applyRegionCatCuts`) crashed after 15-20 s. Now `_round_for_cut` =
`np.rint(x * 1e4) / 1e4`, numpy's own `around` arithmetic, bit-identical to `np.round(x, 4)` on
numpy, awkward and dask-awkward (2M values plus the 2.5/3.0 boundaries checked). Keep every
numpy call in `applyRegionCatCuts` a ufunc.

## Run 2 DNN binning silently used for the Run 3 per-region VBF models (fixed 2026-10-05)

**Symptom.** `configs/MVA/VBF/dnn_binning.yaml` held a single binning, scanned 2026-08-08 on Run 2
NanoV15 ntuples with the Run 2 DNN, and stage-2 used it for every model and jj region. The Run 3
per-region DNNs put their scores in different ranges (signal peaks near 2.2 in `jj_both_central`,
1.6-1.7 in `jj_one_fwd25_one_central`, 2.3 in `jj_both_fwd25`), so many of the 24 Run 2 bins were
empty or had a negative background (negative-weight MC) -- the bins stage-3 later had to clip and
floor -- and `jj_both_fwd25` had a single-event "golden" bin (2.57-2.87: S = 0.079, B = 0.0012 +-
5600%, fake Asimov Z = 0.72).

**Fix.** The config is keyed `models.<DNN model label>.<jj region>` (jj region `all` included), with
the old binning kept as `default`. `modules.selection.resolve_dnn_binning` makes stage-2 stop when
the entry is missing, unless `--allow_default_dnn_binning` is given; stage-2 writes the entry it used
next to the histograms, which `plotter/plot_DNN_score.py` reads. Entries come from
`MVA_training/VBF_run3/scan_bins_for_dnn.py --stage2-scores ... --write-config`, run on per-event
scores dumped by stage-2 (`--dump_scores`), i.e. exactly the fit's selection, weights and model.
First entries (plain Asimov scan, h-peak, all Run 3 years, model
`Run3_nanoAODv12_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation_Systematics`):
`jj_both_central` 16 bins (Asimov Z 0.655 -> 0.656), `jj_one_fwd25_one_central` 14 bins
(1.649 -> 1.728), `jj_both_fwd25` 18 bins (1.506 -> 1.507, without the fake bin). The scan does
not yet account for background MC statistics (To-Do.md item 5). The old file's per-bin Run 2 scan
report is kept in `validation/dnn_binning_scan/default_run2/scan_report_from_config.txt`.
