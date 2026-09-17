# Event-by-Event (EBE) Dimuon Mass-Resolution Calibration

Responsible: **analysis-specific** — EBE resolution calibration against the Z peak is
standard practice in CMS H→µµ-style analyses, but the calibration factors are this
analysis's own fit; there is no POG number to check against.

## Stored sources

| # | Source | Location | Snapshot / verified |
|---|--------|----------|---------------------|
| C1 | Procedure doc (physics formula, PDG constants, ZCR→signal-region logic) | `docs/eventbyeventMassResolution.md` | 2026‑09‑17 |
| C2 | Application | `src/copperhead_processor.py`: `get_mass_resolution()` (~L2682), gated by `doing_BS_correction = switches["do_beamConstraint"]` | 2026‑09‑17 |
| C3 | Calibration-factor payload | `configs/parameters/correction_filelist.yaml` (`BS_res_calib_path`, per year, `MC`/`Data` — same table as `cms-object-guidelines/muons.md` §4.1) | 2026‑09‑17 |
| C4 | Derivation code | `src/lib/ebeMassResCalibration/{basic_class_for_calibration.py,getCalibrationFactor.py,fit_config.yml,PDFs/}` (`PDFs/` includes a `RooCMSShape` PDF) | 2026‑09‑17 |
| C5 | Recent debugging history | `.claude/reports/investigations/2026-09-14_ebe-calib-fit-audit-and-run.md` (execution attempt, currently failing for 2025 nanoAODv15); `2026-09-15_roofit-category-fit-instability-debugging-guide.md` (fit-stability lessons) | see reports |
| C6 | Validation | `src/lib/ebeMassResCalibration/closure_test.ipynb`; `README.md` "Per-event mass calibration" section | 2026‑09‑17 |

Classification tags: **[Analysis‑specific]**, **[Implementation]**, **[Verify]**.

---

## 1. What it corrects

Not a momentum correction — it calibrates the analysis's **estimate of the per-event
dimuon mass resolution**, so that the predicted resolution matches what's actually
observed in data. This estimate feeds category definitions, the VBF DNN
(`dimuon_ebe_mass_res(_rel)` inputs — see the `mva` skill), and the stats fit
(`stats` skill, parametric path).

### Predicted (uncalibrated) resolution (C1, C2)

$$
\sigma_{m_{\mu\mu}}^{\text{pred}} = \frac{m_{\mu\mu}}{2}
\sqrt{\left(\frac{\Delta p_T(\mu_1)}{p_T(\mu_1)}\right)^2 +
\left(\frac{\Delta p_T(\mu_2)}{p_T(\mu_2)}\right)^2}
$$

implemented directly as `sigma = ((mu1.ptErr/mu1.pt * E)**2 + (mu2.ptErr/mu2.pt *
E)**2)**0.5` with `E = dimuon.mass / 2` (`get_mass_resolution`).

PDG constants used to initialize the fit models (C1): Z mass 91.1880 GeV, Z natural
width 2.4955 GeV, H mass 125.200 GeV, H width 3.7 MeV.

---

## 2. Calibration factor — coupled to the beam-spot-constrained muon momentum

`get_mass_resolution(..., doing_BS_correction=...)`: the calibration factor is **only
applied when `doing_BS_correction` is true**, and `doing_BS_correction` is set
straight from `switches["do_beamConstraint"]` (C2). This is *not* a free-standing
switch of its own — it rides on the same beam-spot-constraint switch documented in
`cms-object-guidelines/muons.md` §4.1.

- Payload: correctionlib set **`BS_ebe_mass_res_calibration`**, evaluated on
  `(mu1.pt, |mu1.eta|, |mu2.eta|)`, from the JSON at `BS_res_calib_path["MC"]` (MC) or
  `["Data"]` (Data) for the year (C3).
- `calibration = 1.0` (no-op) whenever `doing_BS_correction` is false.
- Final calibrated resolution: `dimuon_ebe_mass_res = uncalibrated * calibration`;
  `dimuon_ebe_mass_res_rel = dimuon_ebe_mass_res / dimuon.mass`.

**Operational reminder** (README.md): after regenerating a calibration JSON, copy it
to `data/res_calib/`, update the path in `correction_filelist.yaml`, then re-run
stage-1 — and **remember to switch on the beam-spot-constraint option**, or the new
JSON is silently never read.

---

## 3. Derivation (C1, C4)

- Derived in the **Z control region (ZCR)** (Z→µµ resonance), then applied
  consistently to the H signal region and H sidebands (C1) — i.e. the calibration
  itself is a Z-peak measurement, reused unchanged for the mass-shifted analysis
  region.
- RooFit-based fit of the Z-peak mass shape in bins of muon kinematics
  (`basic_class_for_calibration.py`, `getCalibrationFactor.py`, `fit_config.yml`),
  using a `RooCMSShape`-family PDF (C4) for signal/background, producing the JSON
  consumed in §2.
- Validate with `closure_test.ipynb` before trusting a new derivation (C6).

### Known fit-stability gotchas (from C5's roofit debugging guide — read that report
### for the full recipe before re-deriving a category that fights the fitter)

- A "converged" MIGRAD/HESSE result can still be a bad fit — cross-check the
  extracted resolution against neighboring bins (same η at other pT, other η at the
  same pT).
- A **structured** (non-random) residual pattern in the pull plot means a real,
  unmodeled feature in the data — usually fixed by trimming the fit range, not by
  constraining signal-tail parameters to force a fit through it.
- Fixing physically-constant parameters (Breit-Wigner pole mass/width) to the PDG
  values isn't always safest — if a category's real peak is genuinely shifted, that
  forces the shift onto `mean` and can overcompensate.
- Floating multiple `RooCMSShape` background parameters together is risky — floating
  `exp_beta` and `exp_gamma` together made MIGRAD hang; float one at a time.
- Zero HESSE errors with an otherwise good fit (flat pulls, χ²/ndf ≈ 1–2) usually
  means one parameter sits at a hard bound (a genuine flat likelihood direction) —
  document it as a known limitation rather than silently accepting a fabricated error.

### Current known status

As of the 2026‑09‑14 audit (C5), the derivation pipeline **fails to complete** for
2025 (NanoAODv15, MC) — "crashes on first category, unresolved." **Do not assume this
pipeline currently produces a usable 2025 calibration** without checking the registry
for a more recent resolution.

---

## 4. Review checklist

1. `do_beamConstraint` state confirmed for the era being reviewed — the calibration
   factor is a silent no-op (1.0) if it's off.
2. `BS_res_calib_path` for the year points at the intended JSON, and it's the same
   payload muons.md §4.1 documents (they must stay in sync — same config key).
3. If re-deriving: check `.claude/reports/registry.md` and the two linked reports
   before assuming the fit will converge cleanly — the known gotchas above apply.
4. Downstream consumers (`dimuon_ebe_mass_res(_rel)` in the VBF DNN, category/fit
   templates) are using the calibrated, not the raw uncalibrated, quantity if that's
   the analysis's intent.
5. For the current known-broken era (2025 nanoAODv15 as of 2026‑09‑14), check whether
   a fix has since landed before relying on its calibration JSON.

---

## 5. Evidence summary

| Item | Nature | Source | Established? |
|------|--------|--------|--------------|
| Predicted-resolution formula | analysis‑specific | C1, C2 | yes |
| Calibration gated by `do_beamConstraint` | implementation | C2 | yes |
| Calibration payload path/keying | implementation | C3 | yes — cross-checked against `muons.md` |
| RooFit derivation method | analysis‑specific | C4 | yes |
| Fit-stability gotchas | analysis‑specific | C5 | yes — condensed from a dedicated debugging report |
| 2025 nanoAODv15 pipeline status | implementation | C5 | **known broken as of 2026‑09‑14 — re-check registry** |

## Last verified

- Local source review: 2026‑09‑17
