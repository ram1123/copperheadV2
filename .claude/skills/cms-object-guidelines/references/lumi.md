# Luminosity — Stored CMS Recommendations

Responsible POG: **LUM** (Luminosity POG) for the integrated‑luminosity values and
their uncertainty; **PdmV** for the certified‑data golden JSON (lumimask).

## Stored sources

| # | Source | Location | Snapshot / verified |
|---|--------|----------|---------------------|
| C1 | Per‑year integrated luminosity (pb⁻¹) + lumimask paths | `configs/parameters/lumi.yaml` | 2026‑08‑31 |
| C2 | Per‑era luminosity + per‑run breakdown + PdmV links | `docs/Official_recommendation.md`, `docs/Run3_all_basic_Information.md` | 2026‑08‑31 |
| C3 | Implementation — lumimask on data | `src/copperhead_processor.py` L798–805 (`coffea.lumi_tools.LumiMask`) | 2026‑08‑31 |
| C4 | Implementation — MC luminosity weight | `src/copperhead_processor.py` L1585–1601 (`weights.add("lumi", …)`) | 2026‑08‑31 |
| C5 | Helper to recompute from golden JSON | `scripts/compute_lumi_from_golden_json.sh` (untracked) | 2026‑08‑31 |

`lumi.yaml` header cites: `LumiRecommendationsRun2` TWiki and the `topeft` `lumi.json`.

Not covered → **Authoritative CMS verification required**: the final brilcalc/LUM
integrated‑luminosity numbers for Run 3, and the per‑year LUM **luminosity uncertainty**
and its cross‑year correlation model (a stage‑3 datacard nuisance — see §5).

Classification tags: **[LUM/PdmV official]**, **[Analysis‑specific]**,
**[Implementation]**, **[Verify]**.

---

## 1. Required context

Exact era; **data vs MC** (lumimask applies to data; the integrated‑luminosity value
scales MC); provenance of the value (brilcalc vs conference slide vs "Preliminary
Offline Results"); whether the LUM uncertainty is needed.

---

## 2. Lumimask / golden JSON (C1, C3)  **[PdmV official payload]**

Applied to **data only**, via `LumiMask(events.run, events.luminosityBlock)`
(`copperhead_processor.py` L800–805).

| Era | Golden JSON |
|-----|-------------|
| 2016preVFP / 2016postVFP | `Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt` |
| 2017 | `Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt` |
| 2018 | `Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt` |
| 2022preEE / 2022postEE | `Cert_Collisions2022_355100_362760_Golden.json` |
| 2023 / 2023BPix | `Cert_Collisions2023_366442_370790_Golden.json` |
| 2024 | `Cert_Collisions2024_378981_386951_Golden.json` |
| 2025 | `Cert_Collisions2025_391658_398903_Golden.json` |
| 2026 | `Cert_Collisions2026_401624_403937_golden.json` |

RERECO eras use separate ReReco JSONs under `data/lumimasks/RERECO/`.

---

## 3. Integrated luminosity for MC normalisation (C1, C4)

`lumi.yaml` `integrated_lumis` (pb⁻¹):

| Era | pb⁻¹ | Era | pb⁻¹ |
|-----|------|-----|------|
| 2016preVFP | 19500.0 | 2022preEE | 7990.0 |
| 2016postVFP | 16810.0 | 2022postEE | 26680.70 |
| 2017 | 41480.0 | 2023 | 17960.0 |
| 2018 | 59830.0 | 2023BPix | 9680.0 |
| 2016_RERECO | 35920.0 | 2024 | 109820.0 |
| 2017_RERECO | 41529.0 | 2025 | 110840.0 |
| 2018_RERECO | 59740.0 | 2026 | 25950.0 (eras A+B+D only; era C low‑PU excluded) |

MC weight: `weights.add("lumi", weight = gen_weight_ones * integrated_lumi)` where
`integrated_lumi = sample_info["total_lumi_pb"]` — i.e. the value actually used comes
from the **prestage `sample_info`**, not directly from `lumi.yaml`. Verify prestage was
run with the same numbers as `lumi.yaml`.

---

## 4. Cross‑check vs this repo (as of 2026‑08‑31)  **[Verify]**

`lumi.yaml` and `docs/Official_recommendation.md` / `docs/Run3_all_basic_Information.md`
**disagree** for several eras — these feed every MC normalisation, so reconcile before a
result:

| Era | `lumi.yaml` (pb⁻¹) | docs (pb⁻¹) |
|-----|--------------------|-------------|
| 2022preEE | 7990.0 | 7980.4 |
| 2022postEE | 26680.70 | 26671.70 |
| 2023 | 17960.0 | 17794.0 / 18063 (two docs) |
| 2023BPix | 9680.0 | 9451.0 / 9693 |
| 2024 | 109820.0 | 108960.0 |

Run 3 values are annotated as coming from Google‑Slides / "Preliminary Offline Results"
→ not yet the final LUM/brilcalc numbers. Run 2 values are ≈ the LUM UltraLegacy
recommendation — verify against `LumiRecommendationsRun2`.

---

## 5. Luminosity uncertainty

The per‑year LUM integrated‑luminosity uncertainty (order 1–2.5%, partially correlated
across years) is **not represented** in `lumi.yaml` or the processor. It must be added
as a nuisance parameter in the stage‑3 datacards. → **Authoritative CMS verification
required** for the exact per‑year values and correlation scheme.

---

## 6. Review checklist

1. Era identified; data gets the golden JSON, MC gets the integrated‑luminosity scale.
2. Lumimask path matches the era (and RERECO vs UL).
3. `sample_info["total_lumi_pb"]` used by prestage matches `lumi.yaml` for the era.
4. Run 3 luminosity value provenance understood; the `lumi.yaml` vs docs discrepancy
   (§4) resolved for the production being run.
5. 2026 uses A+B+D only (era C excluded).
6. LUM luminosity uncertainty added as a datacard nuisance in stage‑3 (§5).

---

## 7. Evidence summary

| Item | POG | Eras | Source | Established? |
|------|-----|------|--------|--------------|
| Golden JSON / lumimask paths | PdmV | all | C1, C2 | yes |
| Lumimask applied to data only | — | all | C3 | yes (implementation) |
| Integrated‑luminosity values | LUM | all | C1 | stored; Run 3 preliminary, **conflicts with docs (§4)** |
| MC lumi weight uses prestage `total_lumi_pb` | — | all | C4 | yes (implementation) — verify it matches `lumi.yaml` |
| Per‑year luminosity uncertainty + correlations | LUM | all | — | **Authoritative CMS verification required** |

## Last verified

- Local source review: 2026‑08‑31
- Current POG recommendation: pending (final Run 3 brilcalc numbers + LUM uncertainty)
