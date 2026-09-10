---
title: Review of plotter/plot_real_fake_jets_stack.py (real vs fake jet stack plots)
date: 2026-09-03
type: investigation
status: reference
---

## Question

Does `plotter/plot_real_fake_jets_stack.py` produce real (HS) vs fake (PU) jet
contribution plots, and how should it be run?

## Answer

Yes. For each variable in its `JET_ID_VARIABLES` list (jet1_/jet2_ kinematics,
energy fractions, multiplicities, flavour, HF-noise shape vars), it books a
`THStack` of Fake (PU, gen-unmatched) + Real (HS, gen-matched) on top of a
"Total" black-point overlay, using a per-jet gen-match column
(`hasMatchedGenJet_nominal` by default) and a `pt`/`|eta|` preselection.
Physics definition is standard and correct: real = `genJetIdx != -1`
(hard-scatter-matched), fake = `genJetIdx == -1` (no gen jet match, PU
candidate) — see `src/copperhead_processor.py:3100-3101`.

## Key findings

1. **Duplicate/stale script.** `plotter/plot_real_fake_jets_stack.py` is an
   earlier, less complete snapshot of
   `MVA_training/pileup_symbolic_regression/plot_real_fake_jets_stack.py`
   (confirmed via diff — same core logic and column-naming conventions). The
   `MVA_training` copy is the one actually documented and referenced (with
   real example invocations) in
   `MVA_training/pileup_symbolic_regression/README.md`; the `plotter/` copy is
   not referenced anywhere else in the repo (checked via repo-wide grep). The
   `MVA_training` version adds: `--region` (inclusive/central/HE/HF/HEpos/
   HEneg/HFpos/HFneg), `--apply-cleaning` (HE/HF nConstituents+area cut),
   a Fake/Total ratio subpad, Freedman-Diaconis auto-binning, and glob-based
   multi-file input with column-pruned reads. **Recommendation: use the
   `MVA_training/pileup_symbolic_regression/` copy**, not the `plotter/` one,
   unless there's a specific reason to keep them separate.

2. **Most variables require `do_add_jet_ID_vars: true`.** All the
   energy-fraction/multiplicity/flavour/HF-noise columns the script plots
   (`chEmEF`, `chHEF`, `neEmEF`, `neHEF`, `muEF`, `*Multiplicity`,
   `nConstituents`, `nElectrons`, `nMuons`, `hadronFlavour`, `partonFlavour`,
   `hf*`) are only written to stage-1 parquet when
   `switches.do_add_jet_ID_vars` is `true` for that year
   (`src/copperhead_processor.py:3137-3170`). **This switch is `false` for
   every year in the current `configs/parameters/switches.yaml`** (including
   the new `"2026"` entry). Verified empirically against the CI sync-test
   2017 DY MC parquet (`test/output/label_output/stage1_output/2017/f1_0/
   dy_M-50_aMCatNLO/0/*.parquet`): only 4 of 38 `JET_ID_VARIABLES` are present
   (jet1/jet2 `pt`/`eta`); the other 34 get `[WARN] Missing column ...
   -> skipping`. To get the full variable set, stage-1 must be rerun for the
   target year with `do_add_jet_ID_vars` flipped to `true`, or an older
   campaign output where it was already enabled must be used (e.g. the
   `*_FilterJetsHorn25GeV_Apr03_tightPassLepVeto_NoJER_JetIDFix` campaign
   referenced in the `MVA_training` README's example commands).
   `jet1_nSVs_nominal`/`jet2_nSVs_nominal` specifically are never written at
   all — `nSVs` is commented out of the `do_add_jet_ID_vars` field list
   (`src/copperhead_processor.py:3149`) — so those two entries in
   `JET_ID_VARIABLES` will always be skipped regardless of the switch.

3. **MC-only; crashes (not skips) on data input.** `hasMatchedGenJet_{variation}`
   is only written `if is_mc` (`src/copperhead_processor.py:3098`). The
   script's per-variable loop checks `var not in df.columns` and skips
   gracefully, but the genmatch-column check inside `get_real_fake_masks`
   raises an uncaught `KeyError` — so pointing either script at a data-only
   parquet crashes the whole run on the first variable that *is* present,
   rather than skipping cleanly.

4. **Minor latent bug, not currently triggered.** Default `--genmatch-mode
   genJetIdx` doesn't implement ">=0 real / <0 fake" as its own docstring
   describes; it does exact `gen==1`/`gen==0` comparison. This only works
   today because the default genmatch column (`hasMatchedGenJet_nominal`) is
   boolean-like (`True`/`False`/`None`, confirmed via direct read: dtype
   `object`, values `[None True False]` — `None` rows fall into neither
   bucket, which is correct since padded/missing jets have no real pt/eta
   either and are excluded by the preselection anyway). If ever pointed at an
   actual multi-valued `genJetIdx` column with this mode, jets matched to gen
   index ≥ 2 would silently fall into neither real nor fake, breaking
   stack == total. Low priority since no current caller does this.

## How to run (recommended: MVA_training copy)

Needs ROOT → use the `default` (or `combine`) pixi env, not `ci`.

```bash
./enter_pixi.sh default
python MVA_training/pileup_symbolic_regression/plot_real_fake_jets_stack.py \
  -i "/path/to/stage1_output/<year>/compacted/<sample>/0/part*.parquet" \
  -o validation/compare_real_fake/<label> \
  --apply-cleaning
```

Useful options: `--region {inclusive,central,HE,HF,HEpos,HEneg,HFpos,HFneg}`,
`--normalize`, `--apply-cleaning`. Input must be an **MC** sample (needs
gen-match truth) whose stage-1 run had `do_add_jet_ID_vars: true` for that
year to get more than the pt/eta plots.

## Unresolved / follow-up

- Whether to delete/redirect `plotter/plot_real_fake_jets_stack.py` in favor
  of the `MVA_training` copy, or keep both — user's call, not made here.
- No stage-1 campaign currently has `do_add_jet_ID_vars: true` for Run3
  years in the live `switches.yaml`; only historical campaign output (or a
  fresh rerun with the switch flipped) has the full variable set.
