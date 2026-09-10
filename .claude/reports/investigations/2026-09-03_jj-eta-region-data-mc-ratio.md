# jj_eta_region control plots: apparent "inconsistency" between panels

**Date:** 2026-09-03
**Type:** Investigation (read-only, no code changed)
**Scope:** `modules/selection.py::applyRegionCatCuts`'s `jj_eta_region` masking, as used by
`plotter/validation_plotter_unified.py`.
**Trigger:** User observed 3 side-by-side `dimuon_mass` control plots (region h-sidebands, cat
nocat, njets inclusive, `no_zpt_weights_vbf_filter_study`) for `jj_eta_region` = `all` /
`jj_both_central` / `jj_non_central`, that "does not look consistent", and asked to cross-check
for a bug in `applyRegionCatCuts` before any edit.

## Data used

`Run3_nanoAODv15_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation` (the loose-jet-cut
scenario, no HE/HF pT mitigation applied at stage-1), year 2026. `.txt` summaries next to each PDF
(enriched this session with per-sample yields, integrated Data/MC ratio, chi2/ndof) at
`validation/figs/Run3_nanoAODv15/Run3_nanoAODv15_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation/2026/mplhep/Reg_h-sidebands/Cat_nocat/njet_inclusive/no_zpt_weights_vbf_filter_study{,_jj_both_central,_jj_non_central,_jj_both_he,_jj_both_fwd30}/dimuon_mass.txt`.

## Verification performed

1. **Direct code review** of the `jj_eta_region` mask block (`modules/selection.py:175-225`):
   `jj_both_central = j1_c & j2_c`, `jj_non_central = ~(j1_c & j2_c)` -- exact logical complements,
   confirmed via truth table (all 4 True/False/None combinations of j1_c/j2_c).
2. **Direct empirical check against real production data** (not just code review): loaded the full
   `ttjets_dl` 2026 compacted parquet (26,149,069 raw events) and called `applyRegionCatCuts`
   myself with each `jj_eta_region`/`njets_selection` combination:
   - `njets==0` (41,442) + `njets==1` (331,834) + `njets>=2` (1,957,629) = 2,330,905 = `n_all`
     exactly.
   - `jj_both_central` (1,572,975) + `jj_non_central` (384,654) = 1,957,629 = `njets>=2` **exactly**.
   - Conclusion: the partition is exact -- no double-counting, no gap, no boundary bug in practice
     (despite `j1_c = a1<2.5` / `j1_f25 = a1>2.5` technically leaving the single point `a1==2.5`
     uncovered by either -- measure-zero, not observed to matter here).
3. **applyRegionCatCuts's caller** (`plotter/validation_plotter_unified.py:705-716`,
   `dak.map_partitions(selection.applyRegionCatCuts, events, args.category, region_name, process,
   "nominal", args.do_vbf_filter_study, jj_eta_region=args.jj_eta_region,
   njets_selection=str(args.njets), year=year)`) -- positional/keyword argument order checked
   against the function signature, matches correctly. Each `jj_eta_region` panel is a fully
   separate `python plotter/validation_plotter_unified.py` invocation (per
   `workflow/Snakefile`'s `rule plots` / `PLOT_TARGETS` product), so no shared-state/caching
   contamination between panels is possible either.

**Conclusion: no bug found in `applyRegionCatCuts` or its caller.**

## What's actually driving the "inconsistent" look

Pulled the enriched `.txt` summaries directly (not re-derived, already computed by the plotting
script):

| Selection | Data/MC (integrated) | chi2/ndof |
|---|---|---|
| all (inclusive) | 1.089 | 32.0 |
| `jj_both_central` (both \|eta\|<2.5) | 0.939 | 2.9 |
| `jj_non_central` (>=1 jet \|eta\|>2.5) | 1.251 | 17.9 |
| `jj_both_he` (both jets 2.5<\|eta\|<3.0) | 1.725 | 6.9 |
| `jj_both_fwd30` (both jets \|eta\|>3.0) | 1.579 | 4.9 |

A monotonic trend: the more forward the jets in the selection, the larger the data excess over MC.
This matches the known Run3 CMS JME issue already on record in this project's own memory
(`jme-horn-region-official-recommendation.md`, `jer-strat-4-official-jme-mitigation.md`): poor
data/MC modeling for forward jets in the HE region (2.5<|eta|<3.0), which is exactly why the
official HE pT>=50 GeV mitigation cut exists. This is the **loose-cut** stage-1 output (no
mitigation applied), so this is the expected symptom appearing precisely where the physics predicts
it, not a selection/plotting bug.

## Not yet done (reported findings first per user's request)

- Re-run the same 3(+2) panels through the post-hoc `--he-pt-cut`/`--hf-pt-cut` mitigation
  (`scripts/run_horn_ptcut_plots.sh`, `modules.selection.apply_jet_horn_ptcut`, added this session)
  and check whether `jj_non_central`/`jj_both_he` ratios move back toward 1.0 -- would confirm the
  HE-region-mismodeling explanation directly rather than by pattern-matching to prior findings.
- Not checked: whether the effect is data-era-specific, DY-vs-TOP-specific, or present identically
  in 2024/2025 (only 2026 checked here).

## Verification

- Command: interactive `python3` snippet loading `ttjets_dl/0/*.parquet` via `ak.from_parquet` and
  calling `selection.applyRegionCatCuts` directly (pixi `default` env). Output reproduced above.
- Result: partition exactness confirmed for one process/year; Data/MC ratios read directly from
  already-generated `.txt` files (not recomputed).
