# Pileup — Stored CMS Recommendations

Responsible POG: **LUM** (pileup reweighting, minimum‑bias cross‑section). The
forward‑jet pileup‑ID DNN (§B) is an analysis discriminant with no POG.

This file covers **two distinct things**:

- **A. Pileup reweighting** — an event weight correcting the MC pileup profile to data.
- **B. Pileup‑jet‑ID DNN** — a per‑jet classifier to reject forward pileup jets.

## Stored sources

| # | Source | Location | Snapshot / verified |
|---|--------|----------|---------------------|
| C1 | PU weight payload paths | `configs/parameters/SF_filelist.yaml` (`pu_file_mc`, `pu_file_data`) | 2026‑08‑31 |
| C2 | Switches | `configs/parameters/switches.yaml` (`do_pu_wgt`, `do_use_pu_dnn_score`) | 2026‑08‑31 |
| C3 | PU weight implementation | `src/corrections/evaluator.py` (`pu_lookups`, `pu_reweight`, `pu_evaluator`), `src/corrections/weight.py` | 2026‑08‑31 |
| C4 | PU‑jet DNN | `src/corrections/pu_dnn.py`; `MVA_training/pileup_dnn/{train_pu_dnn.py,README.md}`; `pu_dnn_model_dir` in `SF_filelist.yaml` | 2026‑08‑31 |

Not covered → **Authoritative CMS verification required**: the minimum‑bias
cross‑section and its variation for the target era, and the LUM `puWeights.json.gz`
version.

Classification tags: **[LUM official payload]**, **[Analysis‑specific]**,
**[Implementation]**, **[Verify]**, **[Placeholder]**.

---

## A. Pileup reweighting

### A.1 Switch and payloads (C1, C2)

`do_pu_wgt` = **true for every year**. `src/corrections/evaluator.py` builds the weight
from the ratio of the data and MC `Pileup_nTrueInt` profiles.

| Era group | MC profile (`pu_file_mc`) | Data profile (`pu_file_data`) |
|-----------|---------------------------|-------------------------------|
| Run 2 (2016–2018) | `data/pileup/mix_20XX_25ns_UltraLegacy_PoissonOOTPU_cfi.yaml` | `data/pileup/puData20XX_UL_withVar.root` |
| 2022 / 2023 | LUM `puWeights.json.gz` (CAT `metadata/LUM/*`, 2024‑01‑31) | `dummy_only_pu_file_mc_is_needed` |
| 2024 | `puWeights_BCDEFGHI.json.gz` (CAT, 2025‑12‑02) | dummy |
| 2025 | `puWeights_2025pp_Golden_Summer24_25ns_69200ub.json.gz` (CAT, 2026‑06‑05) — **69200 µb ⇒ 69.2 mb** minimum‑bias xsec, golden | dummy |
| 2026 | **reuses the 2025 payload** — no `LUM/*2026*` on cvmfs | **[Placeholder]** |

### A.2 Variations (C3)

`pu_lookups` maps `nom` / `up` / `down` → the `pileup` / `pileup_plus` / `pileup_minus`
branches (Run 2 ROOT hists) or the JSON `nominal` / `up` / `down` (Run 3). The
up/down come from shifting the minimum‑bias cross‑section (nominally 69.2 mb ± ~4.6%).
Carried as a weight systematic.

### A.3 Verify

- The minimum‑bias cross‑section (69.2 mb) and its ± variation for the target era.
- 2026 reuses the 2025 `puWeights` — placeholder until a real 2026 LUM payload exists.
- `pu_file_data` for 2022/2023/2024/2025 is `dummy` (Run 3 uses the JSON directly);
  confirm no code path still tries to read it.

---

## B. Pileup‑jet‑ID DNN  **[Analysis‑specific]**

### B.1 Status

`do_use_pu_dnn_score` = **false for every year** — not part of the default pipeline.
Mutually exclusive with `do_use_pySR_score` (a symbolic‑regression alternative).

### B.2 Purpose

Reject **pileup jets in the forward "horn" turn‑on region** — `25 ≤ pT < 50 GeV`, in the
four signed regions `HEpos` / `HEneg` (`2.5 ≤ |η| ≤ 3.0`) and `HFpos` / `HFneg`
(`|η| > 3.0`). This is the analysis's ML implementation of the JME forward‑jet
mitigation family (see `jets.md` §4.3 / §5).

### B.3 Training (`MVA_training/pileup_dnn/train_pu_dnn.py`, C4)

- Flat‑input MLP (PyTorch), one model per region.
- Label from `jet*_hasMatchedGenJet_nominal`: `1` = hard‑scatter jet, `0` = pileup jet.
- 17 PF‑ID input features: `logpt`, `chEmEF`, `chHEF`, `neEmEF`, `neHEF`, `muEF`,
  `chMultiplicity`, `neMultiplicity`, `nConstituents`, `nElectrons`, `nMuons`,
  `muonSubtrFactor`, `muonSubtrDeltaEta`, `muonSubtrDeltaPhi`, `mass`, `area`,
  `rawFactor`. Raw `pt` / `eta` are used only for selection and plots (except via
  `logpt`).
- Class‑balanced training weights + DY / TOP / EWK sample‑group balancing by default;
  `--use-weights` starts from MC `wgt_nominal`.
- Outputs per region: `model_torchscript.pt`, `scaler.json` (median/mean/std),
  `features.json`, `summary_<region>.json`, plus ROC / WP‑vs‑pT / efficiency artefacts.

### B.4 Application (`src/corrections/pu_dnn.py`, C4)

- Loads `model_torchscript.pt` + `scaler.json` per region from `pu_dnn_model_dir`
  (`SF_filelist.yaml`). **2022/2023 models exclude `puIdDisc`; 2024/2025 models include
  it** (feature lists differ — do not mix).
- 2025 / 2026 `pu_dnn_model_dir` entries are the 2024‑trained model (**[Placeholder]**).
- No efficiency / mistag scale factor exists for this DNN → **[Verify]** before it is
  used in any result.

---

## C. Legacy Run 2 PU jet ID

For Run 2 CHS jets, `jet.yaml` sets `jet_puid: loose` and `jetpuid_sf_file` /
`jmar_sf_file` provide the JME PU‑jet‑ID scale factors (`eval_jetpuid_sf`,
`get_jetpuid_weights*` in `evaluator.py`). See `jets.md` §8. Run 3 PUPPI jets do not use
a legacy PU jet ID.

---

## D. Review checklist

1. `do_pu_wgt` on; MC/data profile payloads match the era; 2026 placeholder understood.
2. PU up/down variations wired into the weight systematics.
3. Minimum‑bias xsec (69.2 mb) and its variation confirmed for the era.
4. If `do_use_pu_dnn_score` is on: region model dir matches the era's feature set
   (`puIdDisc` in/out); no SF applied — impact validated; not mixed with `pySR`.
5. Run 2: `jet_puid` WP + PU‑jet‑ID SF applied (`jets.md` §8).

---

## E. Evidence summary

| Item | POG | Eras | Source | Established? |
|------|-----|------|--------|--------------|
| PU reweighting on (`do_pu_wgt`) | LUM | all | C2 | yes |
| Run 2 MC/data PU profiles | LUM | 2016–2018 | C1, C3 | yes |
| Run 3 `puWeights.json.gz` | LUM | 2022–2025 | C1 | yes; **2026 = placeholder** |
| Minimum‑bias xsec (69.2 mb) + variation | LUM | all | C1 (filename) | value inferred from payload name; **[Verify]** |
| PU‑jet‑ID DNN (regions, features, training) | analysis | Run 3 | C4 | yes (implementation); **off by default, no SF** |
| Legacy Run 2 PU jet ID + SF | JME | 2016–2018 | `jets.md` §8 | see `jets.md` |

## Last verified

- Local source review: 2026‑08‑31
- Current POG recommendation: pending (minimum‑bias xsec + 2026 payload)
