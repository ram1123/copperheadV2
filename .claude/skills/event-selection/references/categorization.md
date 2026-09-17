# Event Selection & Categorization — Vetoes, VBF, ggH

Responsible: **analysis-specific — no CMS POG governs category-cut values.** The
numeric thresholds trace to the H→µµ analysis note (AN‑19‑124, lines ~827/830, per the
code's own comments), not a POG recommendation.

## Stored sources

| # | Source | Location | Snapshot / verified |
|---|--------|----------|---------------------|
| C1 | Class‑based category wrapper, used by `run_stage2.py` (ggH BDT path) | `configs/categories/categories.yml` (declares which cut classes AND together per category); `configs/categories/category_cuts.py` (`Custom*` classes, read via `inspect.getmembers` — comment cites AN‑19‑124 L827/830) | 2026‑09‑17 |
| C2 | Function‑based categorizer, used by `run_stage2_vbf.py` (VBF DNN path) and most plotting/validation/MVA‑preprocessing scripts | `modules/selection.py`: `filterRegion`, `applyRegionCatCuts`, `applyRegionCatCutsByScore`, `apply_jet_horn_ptcut`, `PAIR_JJ_ETA_REGIONS`, `SINGLE_JET_ETA_REGIONS` | 2026‑09‑17 |
| C3 | Category list used by the plotting drivers | `run_plotter.py` (`categories = ["nocat","vbf","ggh","bJetVeto"]`), `plotter/validation_plotter_unified.py` (`FULL_CHANNELS`, same 4) | 2026‑09‑17 |
| C4 | Dead code — do not use as a reference | `src/lib/categorizer.py` — its own module docstring says it is "abandoned/incomplete scratch," never imported anywhere in the repo | 2026‑09‑17 |
| R1 | Registry reports on this exact logic | `.claude/reports/registry.md`: 2026‑09‑03 jj‑eta‑region review, 2026‑09‑04 njets‑first split, 2026‑09‑09 crosscheck, 2026‑09‑10 jj_eta_region‑configurable | see entries |

Classification tags: **[Analysis‑specific]**, **[Implementation]**, **[Verify]**.

---

## 1. Where this sits

Upstream (not in this file): extra‑electron veto (`cms-object-guidelines/electrons.md`
— event rejected if any electron passes the MVA‑iso WP90 preselection); the dimuon
requirement (`cms-object-guidelines/muons.md` §2 — exactly 2 opposite‑charge,
trigger‑matched muons); good jets and their b‑tag working points
(`cms-object-guidelines/jets.md`, `b-tagging.md`).

This file: b‑veto → optional VH veto → mass window → VBF / ggH / nocat / bJetVeto
category → (optionally) njets/jj_eta_region sub‑partition.

Downstream: category‑scoped MVA scoring (`mva` skill), then stats templates (`stats`
skill).

---

## 2. ⚠ Two parallel category implementations — check both

| Path | Category logic lives in | Consumers |
|------|--------------------------|-----------|
| ggH BDT stage‑2 | `configs/categories/category_cuts.py` classes + `categories.yml` (`baseline: [InvBtagCut]`, `vbf: [VbfCut, Jet1PtCut]`, `ggh: [InvVbfCut]`) | `run_stage2.py` |
| VBF DNN stage‑2 + everything else | `modules/selection.py`'s `applyRegionCatCuts` (hardcoded literals) | `run_stage2_vbf.py`, `plotter/validation_plotter_unified.py`, `plotter/plot_2d.py`, `plotter/get2Dplots.py`, `plotter/plot_vbfdnn_input_features_compare.py`, `MVA_training/VBF_run3/{preprocess_dnn.py,scan_bins_for_dnn.py}`, `scripts/{get_yields.py,findbins.py,count_dimuon_mass_window.py,jetPUId_yieldTable.py,sync_parquet_dimuon.py}` |

Both encode the **same** physics numbers (b‑veto: `nBtagLoose≥2` OR
(`nBtagMedium≥1` AND `njets≥2`); VBF: `jj_mass>400 & jj_dEta>2.5 & jet1_pt>35`) but as
two independently maintained code paths. There is no shared import between them — a
fix or change in one does **not** propagate to the other. **When touching either,
check the other for the same drift.**

`category_cuts.py` also defines `CustomInvJet1PtCut` (`~jet1_ptCut`), which
`categories.yml` never references — appears unused; flag before relying on it.

---

## 3. Veto stack

- **Extra‑electron veto** — upstream, in `electrons.md`. Listed here only for
  completeness of the full chain.
- **Dimuon requirement** — upstream, in `muons.md` §2.
- **b‑veto** (`CustomInvBtagCut.filterCategory` / the `~btag_cut` term in
  `applyRegionCatCuts`): reject the event if
  `(nBtagLoose_nominal ≥ 2) OR (nBtagMedium_nominal ≥ 1 AND njets_nominal ≥ 2)`.
  Applied to **`vbf`, `ggh`, and `bJetVeto`** alike; **not** applied to `nocat`
  (baseline/no additional cut). Working‑point definitions: `b-tagging.md`.
- **VH veto** (`do_VH_veto`, optional — every caller found in the repo passes
  `False`): `nfatJets_drmuon == 0` AND `MET_pt < 150 GeV`. Cross‑ref `fat-jets.md`,
  `met.md`. Not part of the default VBF/ggH split.
- **DY VBF‑filter stitching cut** (`do_vbf_filter_study`, off by default): for DY
  samples only, `gjj_mass > 300 GeV` (Run 3) / `> 350 GeV` (Run 2), keeping or
  rejecting events depending on whether the sample name contains `dy_vbf_filter`.
  This is a **sample‑stitching** cut, not a physics veto — only relevant when
  validating that stitching.

---

## 4. Category definitions (`category` argument: `nocat` / `vbf` / `ggh` / `bJetVeto`)

| Category | Definition |
|----------|------------|
| `nocat` | mass window only — no category cut at all |
| `vbf` | `jj_mass > 400 GeV & jj_dEta > 2.5 & jet1_pt > 35 GeV` (AN‑19‑124 L827) **AND** b‑veto |
| `ggh` | **NOT** the VBF cut above **AND** b‑veto (AN‑19‑124 L830) — everything that fails VBF topology but clears the b‑veto; **not** restricted to 0/1‑jet events |
| `bJetVeto` | b‑veto only, no VBF/ggH split — its own 4th category (`run_plotter.py`/`validation_plotter_unified.py` both carry 4 categories, not 3) |

Optional HE/HF forward‑jet pT mitigation (`vbf_he_ptcut`/`vbf_hf_ptcut` in
`applyRegionCatCuts`) folds directly into `vbf_cut` itself, so it changes **both**
`vbf` (uses `vbf_cut`) and `ggh` (uses `~vbf_cut`) consistently — an event whose
jet1/jet2 pair no longer qualifies as VBF‑quality falls to `ggh` rather than being
dropped from both. Boundaries: HE = `2.5 < |η| ≤ 3.0`, HF = `|η| > 3.0` — the same
region split as `jetHorn_region` in `src/copperhead_processor.py`'s `jet_loop` and the
official‑mitigation table in `cms-object-guidelines/jets.md` §5. Check the thresholds
there before trusting a run that has this enabled.

**Score‑based alternative** — `applyRegionCatCutsByScore` assigns `vbf`/`ggh` by
`transf_vbf_score > transf_ggh_score` instead of the cut‑based `vbf_cut`. Confirm
which categorizer a given script actually calls before assuming the cut‑based
definition applies.

---

## 5. Mass window (`filterRegion`, `region_name` argument)

| Region | `dimuon_mass` range (GeV) |
|--------|---------------------------|
| `z-peak` | 70 ≤ m < 110 |
| `h-peak` | 115 ≤ m < 135 |
| `h-sidebands` | 110 ≤ m < 115, or 135 ≤ m < 150 |
| `signal` | `h-peak` ∪ `h-sidebands` |
| `full` | `z-peak` ∪ `h-sidebands` ∪ `h-peak` |

Combined last: `category_selection = prod_cat_cut & region` (`applyRegionCatCuts`,
end of function) — category cut, njets/jj_eta_region cut, and mass window are all
ANDed together into the final event mask.

---

## 6. `njets_selection` and `jj_eta_region` partitions

Validation‑plot and MVA‑scoping granularity — **not** part of the category
definition itself, but used heavily in control plots and to scope which events a VBF
DNN was trained on.

- `njets_selection`: `inclusive` / `0` / `1` / `2` (meaning ≥2).
- `PAIR_JJ_ETA_REGIONS` (need `njets ≥ 2`, jet1 **and** jet2): `jj_both_central`,
  `jj_non_central`, `jj_one_fwd25_one_central`, `jj_one_he_one_central`,
  `jj_one_fwd30_one_central`, `jj_both_fwd25`, `jj_both_he`, `jj_both_fwd30`,
  `jj_one_he_one_fwd30`.
- `SINGLE_JET_ETA_REGIONS` (self‑gate `njets == 1`, jet1 only): `single_central`,
  `single_fwd25`, `single_he`, `single_fwd30`.
- Mixing a pair region with `njets_selection="1"`, or a single‑jet region with
  `njets_selection="2"`, **raises `ValueError`** (guarded in `applyRegionCatCuts`) —
  it would otherwise silently return zero events.
- Boundaries are deliberately **half‑open**: central = `|η| ≤ 2.5`, HE =
  `2.5 < |η| ≤ 3.0`, fwd25 = `|η| > 2.5`, fwd30/HF = `|η| > 3.0`. This was fixed after
  a real data check found events sitting exactly at `|η| = 2.5` falling into neither
  bucket under an earlier strict‑inequality split (12/23692 single‑jet events in a
  2026 `ttjets_dl` check — see R1, 2026‑09‑04).
- This is the same `jj_eta_region` axis the VBF DNN's training scope depends on
  (`mva/vbf-dnn.md`'s open item) — a model trained on `jj_both_central` only should
  not be scored, or trusted, outside that region.

---

## 7. Review checklist

1. Identify which category implementation the code under review actually calls
   (`category_cuts.py`+`categories.yml`, `applyRegionCatCuts`, or
   `applyRegionCatCutsByScore`) — never assume.
2. b‑veto literals (`nBtagLoose≥2`, `nBtagMedium≥1 & njets≥2`) match between whichever
   implementation is in play.
3. VBF‑cut literals (`jj_mass>400`, `jj_dEta>2.5`, `jet1_pt>35`) match AN‑19‑124
   L827/830 in both implementations.
4. If HE/HF mitigation is active, its thresholds match `jets.md` §5 for the year.
5. `do_VH_veto` / `do_vbf_filter_study` explicitly set, not silently defaulted.
6. `region_name` matches the intended signal/control‑region definition.
7. `njets_selection` + `jj_eta_region` combination is valid and uses the half‑open
   boundary convention.
8. If interpreting a VBF DNN score, its training `jj_eta_region` scope is known.

---

## 8. Cross‑check / known findings (registry)

| Finding | Detail |
|---------|--------|
| 2026‑09‑03 jj‑eta‑region review | `jj_both_central`/`jj_non_central` verified as an **exact partition** against 26M real `ttjets_dl` 2026 events — no bug found there |
| 2026‑09‑03 / 2026‑09‑09 | A genuine, monotonic Data/MC excess with jet forwardness across these same regions — tied to the known HE‑region JME mismodeling (`jets.md` §4.3), not a category‑logic bug |
| 2026‑09‑04 | Half‑open boundary fix (§6); `Cat_ggh`/`Cat_nocat` control plots split by njets (0j/1j/≥2j) first |
| 2026‑09‑09 / 2026‑09‑10 | VBF DNN training's `jj_eta_region` was hardcoded, then made configurable — a model‑vs‑scoring‑region mismatch risk if not tracked (`mva/vbf-dnn.md`) |
| — | `src/lib/categorizer.py` is dead code by its own docstring — do not use it as a reference |

---

## 9. Evidence summary

| Item | Nature | Source | Established? |
|------|--------|--------|--------------|
| b‑veto definition | analysis‑specific | C1, C2 | yes — identical literals in both implementations, as of this review |
| VBF cut definition | analysis‑specific (AN‑19‑124 L827) | C1, C2 | yes |
| ggH cut definition | analysis‑specific (AN‑19‑124 L830) | C1, C2 | yes |
| 4‑category set (`nocat`/`vbf`/`ggh`/`bJetVeto`) | analysis‑specific | C3 | yes |
| VH veto | analysis‑specific, optional | C2 | yes; off by default everywhere checked |
| Two‑implementation drift risk | implementation | C1, C2 | flagged — no drift found *at this review*, re‑check after any category‑logic change |
| njets/jj_eta_region boundary convention | implementation | C2, R1 | yes — half‑open, previously fixed |
| Score‑based categorizer availability | implementation | C2 | yes — confirm which one a script uses |

## Last verified

- Local source review: 2026‑09‑17
