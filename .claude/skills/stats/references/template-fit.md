# Template (Shape) Datacards — `make_templates.py` / `make_datacards.py`

Responsible: **analysis-specific** — Combine datacard mechanics are standard HEP
statistics practice; the rate/lumi nuisance *values* here are this analysis's own.

## Stored sources

| # | Source | Location | Snapshot / verified |
|---|--------|----------|---------------------|
| C1 | Template generation | `stage3/make_templates.py` (`Variable` class, ROOT template histograms per process/systematic) | 2026‑09‑17 |
| C2 | Datacard writer (main / VBF path) | `stage3/make_datacards.py` — `rename_regions`, `signal_groups`, `rate_syst_lookup`, `lumi_syst`, `nuisance_titles`, the `shapes * <bin> <file> $PROCESS $PROCESS_$SYSTEMATIC` line | 2026‑09‑17 |
| C3 | ggH datacard generator (separate entry point) | `stage2/ggH_datacard/generate_datacard.py`, `run_script.sh` | 2026‑09‑17 |
| C4 | Existing ggH background cards | `stage3/bkg_datacards_template/datacard_cat{0,1,2,3,4}_ggh_bkg.txt` (+ `_test` variants) — 5 categories, matching the BDT score-edge subcategorization (`mva/ggh-bdt.md`) | 2026‑09‑17 |
| C5 | Datacard editing utility | `stage3/edit_datacard4DY_matchedJets.py` | 2026‑09‑17 |

Classification tags: **[Analysis-specific]**, **[Implementation]**, **[Verify]**.

---

## 1. Pipeline

`make_templates.py` (C1) reads stage‑2 histograms and writes per‑process,
per‑systematic **ROOT template histograms**. `make_datacards.py` (C2) then writes the
Combine `.txt` datacard(s) referencing those templates:
`shapes * {bin} {templates_file} $PROCESS $PROCESS_$SYSTEMATIC`. `signal_groups =
["qqH_hmm", "ggH_hmm"]`; mass regions are renamed for Combine as `h-peak→SR`,
`h-sidebands→SB`, `z-peak→Z` (`rename_regions`).

`stage2/ggH_datacard/generate_datacard.py` (C3) is a **separate** ggH-specific
datacard entry point, distinct from `make_datacards.py` — both import
`src/lib/fit_functions.py` (see `parametric-fit.md`), so the ggH path mixes shape
templates with an analytic background component. Confirm which entry point produced
any specific ggH datacard under review.

---

## 2. Nuisances actually written into the datacard

- **Shape nuisances**: every systematic variation column found in the stage‑2
  output (`all_nuisances`, collected while looping the MC dataframe) gets a
  `nuisance   shape` row automatically — this is the default and, currently, the
  **only** per-process systematic type actually active.
- **`lumi_syst` (`lnN`)**: a per-year flat luminosity uncertainty dict (C2) — the
  *only* rate-type nuisance currently written. **Values must match
  `cms-object-guidelines/lumi.md`** (e.g. the 2025 entry is 5%, uncorrelated,
  sourced from the official LUM recommendation there — added after a real
  `KeyError` when 2025 had no entry). If a year has no entry, `make_datacards.py`
  explicitly raises/warns rather than silently omitting it ("no official CMS LUM
  luminosity uncertainty exists yet for this year").
- **`rate_syst_lookup` (`lnN`, e.g. `XsecAndNorm{DYJ2,EWK,TT+ST,VV,ggH}`)**: defined
  per year (C2) but **the code that would inject these into the datacard is
  currently commented out** ("FIXME: Temporarily commenting out the rate
  systematics since we are not using them in the datacard for now"). **[Verify]**
  before assuming any cross-section/acceptance rate uncertainty is actually in a
  current datacard — as of this review, none is, except `lumi_syst`.

### Known data-quality issue

Every Run 3 year's `rate_syst_lookup` entry (`2022preEE`…`2024`) carries a
`# FIXME: update the values` comment and is a **direct copy of the 2018 numbers**
(`DYJ2 1.12320`, `EWK 1.05779`, `TT+ST 1.18582`, `VV 1.05615`, `ggH 1.38313`,
identical across all of them). Harmless *while* the injection stays commented out —
but a landmine if someone re-enables rate systematics without re-deriving these
first.

---

## 3. Review checklist

1. Which entry point produced this datacard — `make_datacards.py` or
   `stage2/ggH_datacard/generate_datacard.py`?
2. `lumi_syst[year]` matches `lumi.md`'s current recommendation for that year.
3. Rate systematics (`rate_syst_lookup`) are confirmed off (as of this review) — if a
   change turns them back on, the Run‑3 values must be re-derived first, not reused
   from 2018.
4. Shape nuisance list (`all_nuisances`) matches the systematic variations actually
   intended for this run (see `systematics.md` for how those get switched on).
5. Category count/naming (e.g. ggH's 5 `catN_ggh` cards) matches the current
   subcategorization scheme (`mva/ggh-bdt.md`'s `BDT_edges.yaml`).

---

## 4. Evidence summary

| Item | Nature | Source | Established? |
|------|--------|--------|--------------|
| Template → datacard pipeline | implementation | C1, C2 | yes |
| Two datacard entry points (main vs ggH-specific) | implementation | C2, C3 | yes |
| Shape nuisances = default, always on | implementation | C2 | yes |
| `lumi_syst` lnN, cross-checked vs `lumi.md` | analysis-specific | C2 | yes — keep in sync manually |
| `rate_syst_lookup` lnN | analysis-specific | C2 | defined but **not injected** (commented out); Run‑3 values stale (2018 copy) |
| ggH 5-category background cards | implementation | C4 | yes |

## Last verified

- Local source review: 2026‑09‑17
