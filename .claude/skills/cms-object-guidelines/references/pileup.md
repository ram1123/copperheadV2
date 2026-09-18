# Pileup Reweighting — Stored CMS Recommendations

Responsible POG: **LUM** (pileup reweighting, minimum‑bias cross‑section).

> The forward-jet pileup-ID DNN previously documented here has moved to
> `corrections/references/pileup-jet-dnn.md` (it's an analysis-specific ML
> discriminant, not a POG-recommended pileup correction — see that skill).

## Stored sources

| # | Source | Location | Snapshot / verified |
|---|--------|----------|---------------------|
| C1 | PU weight payload paths | `configs/parameters/SF_filelist.yaml` (`pu_file_mc`, `pu_file_data`) | 2026‑08‑31 |
| C2 | Switch | `configs/parameters/switches.yaml` (`do_pu_wgt`) | 2026‑08‑31 |
| C3 | PU weight implementation | `src/corrections/evaluator.py` (`pu_lookups`, `pu_reweight`, `pu_evaluator`), `src/corrections/weight.py` | 2026‑08‑31 |
| R1 | Registry — 2025/2026 placeholder status confirmed generically | `.claude/reports/registry.md`, 2026‑09‑09 crosscheck (`investigations/2026-09-09_2025-2026-full-crosscheck.md`: "all 2025/2026 MUO/LUM/b-tag/PU-DNN/Z-pT placeholders") | 2026‑09‑09 |

Not covered → **Authoritative CMS verification required**: the minimum‑bias
cross‑section and its variation for the target era, and the LUM `puWeights.json.gz`
version.

Classification tags: **[LUM official payload]**, **[Implementation]**, **[Verify]**,
**[Placeholder]**.

---

## 1. Switch and payloads (C1, C2)

`do_pu_wgt` = **true for every year**. `src/corrections/evaluator.py` builds the weight
from the ratio of the data and MC `Pileup_nTrueInt` profiles.

| Era group | MC profile (`pu_file_mc`) | Data profile (`pu_file_data`) |
|-----------|---------------------------|-------------------------------|
| Run 2 (2016–2018) | `data/pileup/mix_20XX_25ns_UltraLegacy_PoissonOOTPU_cfi.yaml` | `data/pileup/puData20XX_UL_withVar.root` |
| 2022 / 2023 | LUM `puWeights.json.gz` (CAT `metadata/LUM/*`, 2024‑01‑31) | `dummy_only_pu_file_mc_is_needed` |
| 2024 | `puWeights_BCDEFGHI.json.gz` (CAT, 2025‑12‑02) | dummy |
| 2025 | `puWeights_2025pp_Golden_Summer24_25ns_69200ub.json.gz` (CAT, 2026‑06‑05) — **69200 µb ⇒ 69.2 mb** minimum‑bias xsec, golden | dummy |
| 2026 | **reuses the 2025 payload** — no `LUM/*2026*` on cvmfs (**[Placeholder]**, reconfirmed 2026‑09‑09, R1) | **[Placeholder]** |

---

## 2. Variations (C3)

`pu_lookups` maps `nom` / `up` / `down` → the `pileup` / `pileup_plus` / `pileup_minus`
branches (Run 2 ROOT hists) or the JSON `nominal` / `up` / `down` (Run 3). The
up/down come from shifting the minimum‑bias cross‑section (nominally 69.2 mb ± ~4.6%).
Carried as a weight systematic — see `stats/references/systematics.md`.

---

## 3. Legacy Run 2 PU jet ID

For Run 2 CHS jets, `jet.yaml` sets `jet_puid: loose` and `jetpuid_sf_file` /
`jmar_sf_file` provide the JME PU‑jet‑ID scale factors (`eval_jetpuid_sf`,
`get_jetpuid_weights*` in `evaluator.py`). See `jets.md` §8. Run 3 PUPPI jets do not use
a legacy PU jet ID. (Not to be confused with the forward pileup-jet-**DNN** — that's
an unrelated, analysis-specific discriminant, now in `corrections/references/pileup-jet-dnn.md`.)

---

## 4. Review checklist

1. `do_pu_wgt` on; MC/data profile payloads match the era; 2026 placeholder understood.
2. PU up/down variations wired into the weight systematics.
3. Minimum‑bias xsec (69.2 mb) and its variation confirmed for the era.
4. Run 2: `jet_puid` WP + PU‑jet‑ID SF applied (`jets.md` §8).
5. If the request is about the forward pileup-jet-**DNN**, this is the wrong file —
   see the `corrections` skill instead.

---

## 5. Evidence summary

| Item | POG | Eras | Source | Established? |
|------|-----|------|--------|--------------|
| PU reweighting on (`do_pu_wgt`) | LUM | all | C2 | yes |
| Run 2 MC/data PU profiles | LUM | 2016–2018 | C1, C3 | yes |
| Run 3 `puWeights.json.gz` | LUM | 2022–2025 | C1 | yes; **2026 = placeholder** |
| Minimum‑bias xsec (69.2 mb) + variation | LUM | all | C1 (filename) | value inferred from payload name; **[Verify]** |
| Legacy Run 2 PU jet ID + SF | JME | 2016–2018 | `jets.md` §8 | see `jets.md` |

## Last verified

- Local source review: 2026‑09‑17
- Current POG recommendation: pending (minimum‑bias xsec + 2026 payload)
