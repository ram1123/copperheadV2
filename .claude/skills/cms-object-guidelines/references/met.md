# Missing Transverse Momentum — Stored CMS Recommendations

Responsible POG: **JME** (JetMET — MET group).

MET object used throughout this analysis: **PuppiMET** (`PuppiMET_pt`, `PuppiMET_phi`,
`PuppiMET_sumEt`). MET enters MET‑based event cleaning and a few category / DNN input
variables; the analysis is not MET‑driven.

## Stored sources

| # | Source | Location | Snapshot / verified |
|---|--------|----------|---------------------|
| S1 | Pointer to the official MET filter list | `docs/Run3_all_basic_Information.md` (MET / Noise Filters) → `https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2#Run_3_2022_and_2023_data_and_MC` | local review 2026‑08‑31 |
| C1 | Implementation | `src/copperhead_processor.py` — PuppiMET blocks; `compute_jet_veto_jetfilter` (~L662); PuppiMET–jet horn recipe (~L489) | 2026‑08‑31 |
| — | Type‑I propagation | via the JEC/JER sequence in `jets.md` §2 | 2026‑08‑31 |

Not covered by stored sources → **Authoritative CMS verification required**: the exact
`Flag_*` filter list per era (data vs MC), MET‑φ (XY) correction applicability to
PuppiMET, and any PuppiMET scale/resolution uncertainty prescription.

Classification tags: **[JME official]**, **[Analysis‑specific]**, **[Implementation]**,
**[Verify]**.

---

## 1. Required context

Run 2 vs Run 3; exact era; data vs MC; the MET flavour (**PuppiMET** here, not
`MET`/`CaloMET`); whether the analysis needs MET‑based uncertainties.

---

## 2. Type‑I / propagation

- The AK4 JEC and JER changes applied in stage‑1 are propagated to `PuppiMET`
  (`jets.md` §2). **[Implementation]**
- **No** explicit MET‑XY (φ) or hadronic‑recoil correction is present in the configs.
  **[Verify]** whether the target era's JME recommendation requires a MET‑φ correction on
  PuppiMET (typically small/absent for PUPPI — confirm, do not assume).

---

## 3. MET noise / event filters  **[Verify]**

- The official list to apply is the one linked from S1.
- The exact `Flag_*` branches applied were **not located** in
  `src/copperhead_processor.py` during this review. Confirm which of the recommended
  filters are applied for each era and on data vs MC — typically:
  `goodVertices`, `globalSuperTightHalo2016Filter`,
  `EcalDeadCellTriggerPrimitiveFilter`, `BadPFMuonFilter`, `BadPFMuonDzFilter`,
  `hfNoisyHitsFilter`, `eeBadScFilter` (data only), `ecalBadCalibFilter`.

---

## 4. Analysis‑specific MET treatments (C1)  **[Analysis‑specific / Implementation]**

- **PuppiMET–jet "horn" recipe** (~L489): reject the event if `PuppiMET_pt > 100 GeV`
  **and** there is ≥ 1 AK4 jet with `Δφ(PuppiMET, jet) > 2.9` (plus an EM‑fraction
  condition on the jet). Targets fake MET from the 2.5 < \|η\| < 3.0 jet mismodelling —
  related to the JER known issue in `jets.md` §4.3.
- **Jet‑veto‑map jet filter** (`compute_jet_veto_jetfilter`, ~L662): when the
  jet‑veto‑map *jet* filter removes jets, `PuppiMET_pt` and `PuppiMET_sumEt` are
  recomputed; `PuppiMET_phi` is deliberately **not** reset (avoids a spurious peak at
  φ = 0).

---

## 5. Review checklist

1. MET flavour = PuppiMET everywhere it is used.
2. JEC/JER changes propagated to MET (Type‑I).
3. MET‑φ correction decision recorded (currently none — §2).
4. Recommended MET filter flags identified for the era and confirmed applied on data
   and MC (§3).
5. Analysis‑specific MET cleaning (§4) intended and documented.
6. Systematics: if MET‑based uncertainties are needed, the PuppiMET
   scale/resolution prescription is resolved.

---

## 6. Cross‑check vs this repo's config (as of 2026‑08‑31)

| Observation | Detail |
|-------------|--------|
| MET filters | recommended `Flag_*` list not found applied in `copperhead_processor.py` — verify |
| MET‑φ correction | none present — verify whether required for PuppiMET in the target era |
| MET reset on jet‑veto filter | `PuppiMET_pt`/`sumEt` recomputed, `PuppiMET_phi` intentionally kept |

---

## 7. Evidence summary

| Item | POG | Eras | Source | Established? |
|------|-----|------|--------|--------------|
| PuppiMET as the MET object | analysis | all | C1 | yes — analysis choice |
| Type‑I propagation of JEC/JER | JME | all | `jets.md` §2, C1 | yes |
| MET‑φ (XY) correction | JME | all | — | not applied; **[Verify]** if required |
| MET noise / event filter list | JME | all | S1 | pointer only; **application not confirmed** |
| PuppiMET scale/resolution uncertainty | JME | all | — | **Authoritative CMS verification required** |
| Analysis MET–jet horn recipe; jet‑veto MET reset | analysis | Run 3 | C1 | yes — implementation confirmed |

## Last verified

- Local source review: 2026‑08‑31
- Current POG recommendation: pending
