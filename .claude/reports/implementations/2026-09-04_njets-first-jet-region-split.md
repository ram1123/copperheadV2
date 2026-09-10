# Jet-multiplicity-first, then jet-region split for Cat_ggh/Cat_nocat

**Date:** 2026-09-04
**Type:** Implementation
**Scope:** `modules/selection.py`, `plotter/validation_plotter_unified.py`, `workflow/Snakefile`.
**Follow-up to:** `.claude/reports/investigations/2026-09-03_jj-eta-region-data-mc-ratio.md` (the
"inconsistent plot" investigation that first surfaced this design gap) and the same-day
coordinate-skill diagnostic (chat only, no file) that proposed this fix.

## Requested physics logic (user-specified)

- `Cat_vbf`: jet-region splitting can use inclusive/both-jets/one-jet conditions.
- `Cat_ggh`/`Cat_nocat`: split by njets first (`0j`/`1j`/`>=2j`); `0j` gets no region split; `1j`
  gets single-jet region split only (no two-jet conditions); `>=2j` gets the existing
  inclusive/both/one-jet conditions.

## What changed

**(a) `modules/selection.py`** -- added `SINGLE_JET_ETA_REGIONS` (`single_central`,
`single_fwd25`, `single_he`, `single_fwd30`), parallel to the existing `PAIR_JJ_ETA_REGIONS` (the
former inline `masks` dict, now also exposed as a module constant). Each single-jet mask bakes
`njets==1` directly into its own definition (`is_single_jet & j1_c`, etc.) -- a deliberate design
choice over leaving them purely eta-based and relying on the caller to also pass
`njets_selection="1"`: this way a `single_*` region is correct regardless of what
`njets_selection` is passed alongside it (matches the njets==1 subset, or is empty), rather than
silently wrong if the caller forgets to pair it correctly.

**(b) `modules/selection.py`** -- three new `ValueError`s in `applyRegionCatCuts`, replacing what
was previously a silent empty-selection for each: `njets_selection="0"` + any region other than
`"all"`; `njets_selection="1"` + a `PAIR_JJ_ETA_REGIONS` name; `njets_selection="2"` + a
`SINGLE_JET_ETA_REGIONS` name.

**(c) `workflow/Snakefile`** -- `PLOT_TARGETS` construction replaced with a per-category helper
(`_jet_region_targets_for`): `vbf` keeps the existing pair-region matrix, now explicitly labeled
`njets_plot="2"` (was `"inclusive"`, mislabeled -- vbf's own category cut already requires
njets>=2 via `jj_mass`/`jj_dEta`, so `njets_plot` should say so); `ggh`/`nocat` get the plain
inclusive/`all` overview target (unchanged), plus new `0j`/`all`-only, `1j`/single-region, and
`2j`/pair-region targets. `wildcard_constraints` and `PLOT_NJETS` updated to admit the new
`jj_eta_region`/`njets_plot` values.

**Bonus fix, same block**: `j1_c`/`j2_c` used `a1 < 2.5` and `j1_f25`/`j2_f25` used `a1 > 2.5` --
both strict, leaving `|eta| == 2.5` exactly (and similarly `== 3.0` for `he`/`fwd30`) uncovered by
either bucket. I'd called this "measure-zero, negligible" in the prior investigation; that
assumption was wrong -- confirmed on real data (see Verification). Fixed to half-open
(`<=2.5`/`>2.5`, `>2.5 & <=3.0`/`>3.0`) so every event lands in exactly one bucket. This also
silently improves the *existing* `jj_both_central`/etc. masks by the same tiny amount (not just
the new single-jet ones), since both reuse the same `j1_c`/`j1_he`/`j1_f30` definitions -- a real
but tiny (~0.01% of events) change to already-generated plots if regenerated.

**Not changed**: naming convention and rename-vs-add-alongside were both resolved per the user's
"proceed with your suggestion" -- single-jet names as above, and the existing `njet_inclusive`
pair-region outputs are left in place, not renamed/moved (new targets added alongside).

## Verification performed

Real-data checks against `ttjets_dl` 2026 (`Run3_nanoAODv15_FilterEvents_Aug30_..._OfficialRecomendation`
compacted output, 26,149,069 raw events), via `modules.selection.applyRegionCatCuts` directly:

- `njets==0` + `njets==1` + `njets>=2` == `n_all` exactly (2,330,905 == 2,330,905) -- unchanged
  from the prior investigation, re-confirmed.
- `jj_both_central` + `jj_non_central` (njets_selection="2") == `njets>=2` exactly (1,957,629) --
  pair-mask partition still exact after the boundary fix.
- `single_central` + `single_fwd25` (njets_selection="1") == `n1_all` exactly (331,834) --
  **only exact after the boundary fix**; before it, off by 12 (331,822), all 12 events sitting at
  exactly `|jet1_eta| == 2.5`.
- `single_he` + `single_fwd30` == `single_fwd25` exactly (23,692) -- same fix, was off by 5 before
  (events at exactly `|eta| == 3.0`).
- All four intended `ValueError`s confirmed to actually raise (`jj_both_central`+`njets=0`,
  `single_central`+`njets=0`, `jj_both_central`+`njets=1`, `single_central`+`njets=2`).
- `single_central` sanity: for `njets_selection="1"`, all returned events have `njets_nominal==1`
  and `max(|jet1_eta_nominal|) == 2.5` (i.e. the boundary point is now correctly included).
- `snakemake -s workflow/Snakefile plot_all -n --nolock` dry-run: DAG builds cleanly, 46 total
  jobs (45 `plots` + `plot_all`), all newly-required jobs are exactly the new `0j`/`1j` targets
  plus a few previously-uncomputed `2j` ones -- the pre-existing `2j`/pair-region `.done` markers
  are correctly recognized as already satisfied, not regenerated. No execution performed.

## Not yet done

- No actual plot images regenerated/inspected (only the DAG dry-run was verified) -- the user's
  original acceptance criterion ("re-generate ggh/nocat control plots with the new split, check
  the enriched .txt stats") is still open.
- Whether the `jj_both_he`/`jj_both_fwd30` Data/MC ratios from the prior investigation
  (0.94/1.25/1.73/1.58) shift measurably from the boundary fix -- expected negligible (~0.01% of
  events) but not explicitly re-measured.
- The `njets==1` bucket's own central/fwd25/he/fwd30 Data/MC trend has not been examined at all --
  new territory this change makes visible for the first time.
