# Pileup-Jet-ID DNN — Forward-Jet Fake Rejection

Responsible: **analysis-specific — no CMS POG recommendation applies.** This is the
analysis's own ML implementation of the JME forward-jet mitigation family (see
`cms-object-guidelines/jets.md` §4.3 / §5); it is not a POG product and has no
official efficiency/mistag scale factor.

## Stored sources

| # | Source | Location | Snapshot / verified |
|---|--------|----------|---------------------|
| C1 | Switch | `configs/parameters/switches.yaml` (`do_use_pu_dnn_score`) | 2026‑08‑31 |
| C2 | Training | `MVA_training/pileup_dnn/{train_pu_dnn.py,README.md}` | 2026‑08‑31 |
| C3 | Application | `src/corrections/pu_dnn.py`; `pu_dnn_model_dir` in `configs/parameters/SF_filelist.yaml` | 2026‑08‑31 |
| R1 | Registry — 2025/2026 placeholder status confirmed generically | `.claude/reports/registry.md`, 2026‑09‑09 crosscheck (`investigations/2026-09-09_2025-2026-full-crosscheck.md`: "all 2025/2026 MUO/LUM/b-tag/PU-DNN/Z-pT placeholders") | 2026‑09‑09 |

Classification tags: **[Analysis‑specific]**, **[Implementation]**, **[Verify]**,
**[Placeholder]**.

---

## 1. Status

`do_use_pu_dnn_score` = **false for every year** (C1) — not part of the default
pipeline. Mutually exclusive with `do_use_pySR_score` (a symbolic-regression
alternative to the same problem).

---

## 2. Purpose

Reject **pileup jets in the forward "horn" turn-on region** — `25 ≤ pT < 50 GeV`, in
four signed regions: `HEpos`/`HEneg` (`2.5 ≤ |η| ≤ 3.0`) and `HFpos`/`HFneg`
(`|η| > 3.0`). This is the analysis's ML implementation of the JME forward-jet
mitigation family documented in `cms-object-guidelines/jets.md` §4.3/§5 — the same
physical problem those sections describe official/analysis pT-cut mitigations for.

---

## 3. Training (`MVA_training/pileup_dnn/train_pu_dnn.py`, C2)

- Flat-input MLP (PyTorch), one model **per region** (4 signed regions).
- Label from `jet*_hasMatchedGenJet_nominal`: `1` = hard-scatter jet, `0` = pileup
  jet.
- 17 PF-ID input features: `logpt`, `chEmEF`, `chHEF`, `neEmEF`, `neHEF`, `muEF`,
  `chMultiplicity`, `neMultiplicity`, `nConstituents`, `nElectrons`, `nMuons`,
  `muonSubtrFactor`, `muonSubtrDeltaEta`, `muonSubtrDeltaPhi`, `mass`, `area`,
  `rawFactor`. Raw `pt`/`eta` are used only for selection and validation plots
  (except indirectly via `logpt`).
- Class-balanced training weights, plus DY/TOP/EWK sample-group balancing by
  default; `--use-weights` starts from MC `wgt_nominal` instead.
- Per-region outputs: `model_torchscript.pt`, `scaler.json` (median/mean/std),
  `features.json`, `summary_<region>.json`, plus ROC / working-point-vs-pT /
  efficiency artefacts.

---

## 4. Application (`src/corrections/pu_dnn.py`, C3)

- Loads `model_torchscript.pt` + `scaler.json` per region from `pu_dnn_model_dir`
  (`SF_filelist.yaml`).
- **2022/2023 models exclude `puIdDisc`; 2024/2025 models include it** — the feature
  lists genuinely differ between the two groups; do not mix a model trained without
  `puIdDisc` against inputs that include (or omit) it inconsistently.
- 2025/2026 `pu_dnn_model_dir` entries reuse the **2024-trained** model
  (**[Placeholder]** — confirmed still true as of the 2026‑09‑09 registry crosscheck,
  R1).
- **No efficiency or mistag scale factor exists for this DNN.** **[Verify]** the
  data/MC efficiency impact before using it in any result — there is nothing to
  cross-check it against besides direct validation.

---

## 5. Review checklist

1. `do_use_pu_dnn_score` explicitly on/off as intended for the run being reviewed —
   it's off by default.
2. Not combined with `do_use_pySR_score` (mutually exclusive alternatives).
3. Model/feature-set match for the era: `puIdDisc` in (2024/2025-style) or out
   (2022/2023-style), consistently between training and application.
4. 2025/2026 placeholder status (2024-trained model reuse) understood before
   trusting scores for those years.
5. No SF is being silently assumed — any efficiency claim must come from direct
   validation, not an official number.

---

## 6. Evidence summary

| Item | Nature | Source | Established? |
|------|--------|--------|--------------|
| Off by default, all years | implementation | C1 | yes |
| 4-region model structure, 17-feature input list | analysis-specific | C2 | yes |
| `puIdDisc` in/out feature-set split (2022/23 vs 2024/25) | implementation | C3 | yes |
| 2025/2026 = 2024-trained-model placeholder | implementation | C3, R1 | yes — reconfirmed 2026‑09‑09 |
| Efficiency / mistag SF | — | — | **none exists — [Verify] via direct validation** |

## Last verified

- Local source review: 2026‑09‑17
