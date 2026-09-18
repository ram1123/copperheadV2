# Parametric (RooFit) Background/Signal Shapes — `src/lib/fit_functions.py`

Responsible: **analysis-specific** — the specific RooFit function families used
(FEWZ×Bernstein, BWZ‑Redux) are standard choices in CMS H→µµ-style analyses, but the
fitted coefficients and category assignment are this analysis's own.

## Stored sources

| # | Source | Location | Snapshot / verified |
|---|--------|----------|---------------------|
| C1 | Analytic PDF library | `src/lib/fit_functions.py` — `getFEWZ_roospline`, `getFEWZ_vals`, `MakeFEWZxBernDof3` (+ `_ucsd` variant), `MakeFEWZxBern`, `MakeBWZ_Redux`, all built on `ROOT`/RooFit (`RooRealVar`, `RooProdPdf`, …) | 2026‑09‑17 |
| C2 | Consumers | `run_stage3.py`, `run_stage3_validation.py`, `run_stage3_ucsd.py`, `stage2/ggH_datacard/generate_datacard.py`, `validation/ggH/categorization/{plot_6_19.py,getTable_6_2And6_12.py}`, `validation/ggH/bias_test/run_bias_test.py` | 2026‑09‑17 |
| C3 | Related RooFit derivation with known fit-stability lessons | `corrections/ebe-mass-calibration.md` (EBE calibration is a different RooFit fit — the Z-peak resolution measurement — but hit the same class of MIGRAD/HESSE stability issues); `.claude/reports/investigations/2026-09-15_roofit-category-fit-instability-debugging-guide.md` | see report |

Classification tags: **[Analysis-specific]**, **[Implementation]**, **[Verify]**.

---

## 1. What this is

An **analytic, unbinned dimuon-mass background model** (RooFit `RooAbsPdf`s), as an
alternative/complement to the binned MC-template background in `template-fit.md`.
Function families provided (C1):

- **FEWZ × Bernstein** (`MakeFEWZxBernDof3`, `MakeFEWZxBern`) — a fixed-shape FEWZ
  NNLO Drell-Yan spline (`getFEWZ_roospline`, loaded from a saved
  `ucsd_workspace/fewz.root` RooSpline1D, `"fewz_1j_spl_order1_cat_ggh"`) multiplied
  by a floating Bernstein polynomial of configurable degree of freedom (`dof`).
- **BWZ-Redux** (`MakeBWZ_Redux`) — a Breit-Wigner × exponential-family background
  shape, with hardcoded starting values/bounds for its `a`/`b`/`c` coefficients
  (visible directly in the function body, with several commented-out alternate
  starting points from earlier tuning attempts).

Both return a `RooProdPdf`/pdf object plus a dict of the fit's `RooRealVar`
coefficients, for use in a `RooAddPdf`/`RooSimultaneous`-style category fit.

---

## 2. Where it's used (C2)

Imported by **both** `run_stage3.py` (the main stage‑3 entry point) and
`stage2/ggH_datacard/generate_datacard.py` (the separate ggH-specific datacard
generator noted in `template-fit.md`) — so the parametric background model is not
confined to one channel's code path. **[Verify]** the precise split of which
category/channel combinations use the parametric path vs. a pure binned template,
rather than assuming a strict "ggH = parametric, VBF = template" mapping — this file
only establishes which scripts import `fit_functions.py`, not which final datacards
end up using it.

Also used by `validation/ggH/{categorization,bias_test}/` scripts — i.e. this
machinery is exercised directly in bias-test and categorization validation work, not
only in the production datacard path.

---

## 3. Fit-stability lessons (from C3 — read the full report before fighting a
### category that won't converge)

The same RooFit category-fit failure modes documented for the EBE calibration apply
here, since both are RooFit fits to a dimuon-mass-like spectrum:

- a "converged" fit can still be wrong — sanity-check against neighboring
  categories/bins;
- a structured (non-random) pull-plot residual means a real unmodeled feature —
  trim the fit range, don't force signal/background parameters to absorb it;
- forcing a physically-constant parameter (e.g. a resonance pole mass/width) to a
  fixed external value can push a real shift onto another parameter and make the
  fit worse, not better;
- float RooFit background parameters one at a time where possible — floating two
  correlated ones together can hang MIGRAD;
- zero HESSE errors alongside an otherwise good fit usually means a parameter is
  pinned at a hard bound (a flat likelihood direction) — document it, don't ignore
  it.

---

## 4. Review checklist

1. Which datacard/category actually uses this parametric path vs. the binned
   template path (`template-fit.md`) — confirmed for the specific result under
   review, not assumed from the channel name alone.
2. `getFEWZ_roospline`'s external `ucsd_workspace/fewz.root` dependency is present
   and current wherever this code runs.
3. Any newly floated/refit coefficient followed the fit-stability lessons in §3
   before being trusted.
4. Hardcoded starting values/bounds in `MakeBWZ_Redux` (and the commented-out
   alternates) are the ones actually intended for the fit being reviewed — check the
   live code, not just this summary.

---

## 5. Evidence summary

| Item | Nature | Source | Established? |
|------|--------|--------|--------------|
| FEWZ×Bernstein / BWZ-Redux function library | analysis-specific | C1 | yes |
| Used by both main stage-3 and the ggH-specific datacard generator | implementation | C2 | yes |
| Exact channel/category → parametric-vs-template mapping | implementation | — | **[Verify]** |
| RooFit fit-stability lessons | analysis-specific | C3 | yes — condensed from a dedicated debugging report |

## Last verified

- Local source review: 2026‑09‑17
