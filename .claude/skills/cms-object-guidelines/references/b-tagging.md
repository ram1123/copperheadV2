# b tagging — Stored CMS Recommendations

Responsible POG: **BTV** (B‑Tagging & Vertexing POG).

Role in this analysis: b‑tagged AK4 jets are used for a **b‑jet veto / top control**
in the VBF and ggH channels; not a signal object.

## Stored sources

| # | Source | Location | Snapshot / verified |
|---|--------|----------|---------------------|
| S1 | This analysis's b‑tag statement ("loose and medium DeepCSV") | `docs/Official_recommendation.md` (Jet Selection & Corrections table) | local review 2026‑08‑31 |
| C1 | Per‑year b‑jet acceptance + tagger working points | `configs/parameters/jet.yaml` (`btag_jet_*`, `btag_*_wp_*`) | 2026‑08‑31 |
| C2 | Scale‑factor payloads | `configs/parameters/SF_filelist.yaml` (`btag_sf_json`, `btag_sf_csv`) | 2026‑08‑31 |
| C3 | Implementation | `src/corrections/evaluator.py` (b‑tag weight evaluation) | 2026‑08‑31 |
| C4 | Working‑point provenance | `btv-wiki` URLs embedded inline in `jet.yaml` | 2026‑08‑31 |

Not covered by stored sources → **Authoritative CMS verification required**: the
recommended tagger per era, the numeric L/M/T working‑point values, the
`btagging.json.gz` version, and the SF application method + uncertainty decomposition.

Classification tags: **[BTV official]**, **[Analysis‑specific]**, **[Implementation]**,
**[Verify]**, **[Placeholder]** (unset / copied value in the config).

---

## 1. Required context

Run 2 vs Run 3; exact era; the **tagger** the analysis actually reads (DeepCSV / DeepJet
/ ParticleNet / UParT); the **working point** (loose/medium/tight); whether SFs are
applied as fixed‑WP or shape.

---

## 2. b‑jet pre‑selection (C1)  **[Analysis‑specific]**

| Requirement | Value | Classification |
|-------------|-------|----------------|
| AK4 jet pre‑selection | see `jets.md` | JME + analysis |
| pT | > **25 GeV** | `btag_jet_pt_cut` |
| \|η\| | < **2.4** (2016) / < **2.5** (2017 → Run 3) | tracker acceptance (`btag_jet_eta_cut`) |
| jet ID | **tight** | JME jet ID (`btag_jet_id`) |

---

## 3. Taggers and working points (C1, C4)

Values are **as stored in `configs/parameters/jet.yaml`**; discriminant WP = loose /
medium.

| Tagger | Run 2 UL (L / M) | Run 3 (L / M) | Provenance in config |
|--------|-----------------|---------------|----------------------|
| DeepCSV | 2016preVFP 0.2027 / 0.6001 … 2018 0.1208 / 0.4168 | not set (0.000) | btv‑wiki UL pages; S1 names "loose and medium DeepCSV" |
| DeepJet (`deepJet`) | not set | 2022preEE 0.0583 / 0.3086; 2022postEE 0.0614 / 0.3196; 2023 0.0479 / 0.2431; 2023BPix 0.048 / 0.2435 | btv‑wiki `Run3Summer22` |
| ParticleNet | not set | 2022preEE 0.047 / 0.245; 2022postEE 0.0499 / 0.2605; 2023 0.0358 / 0.1917; 2023BPix 0.0359 / 0.1919 | btv‑wiki `Run3Summer22` |
| UParT (`UParTAK4`) | 2016preVFP 0.0387 / 0.1847 … 2018 0.0308 / 0.1610 | 2024 0.0246 / 0.1272 | btv‑wiki `Run3Summer24` (2024) |

### 3.1 Placeholders / gaps — must not be used blind  **[Placeholder]**

- 2024 / 2025 / 2026 DeepJet and ParticleNet WPs are `0.000` (unset).
- 2025 / 2026 UParT WPs are copied from 2024 (`# PLACEHOLDER … pending BTV POG 2025/2026 WPs`).
- 2023 / 2023BPix DeepJet and ParticleNet WPs cite the `Run3Summer22` btv‑wiki page —
  confirm a dedicated 2023 page is not required.
- `btag_sf_json` for 2026 reuses the 2025 payload (C2).

---

## 4. Scale factors (C2)  **[BTV official payload]**

- `btag_sf_json` → **`btagging.json.gz`**: `jsonpog-integration` `POG/BTV/*_UL` (Run 2),
  CAT `metadata/BTV/*` (Run 3; 2026 = 2025 placeholder).
- `btag_sf_csv` → legacy DeepCSV CSV, Run 2 only; **all Run 3 rows marked
  `# FIXME: Update SF`** (still pointing at the 2018 UL CSV).

The SF application method — fixed‑WP (`comb` / `mujets`) vs shape / `iterativeFit`;
per‑flavour b / c / light handling; correlation model across years — is **not
established** from stored material. **[Verify]**

---

## 5. Review checklist

1. Era identified; tagger the analysis reads is one with real (non‑zero, non‑placeholder)
   WPs for that era.
2. b‑jet acceptance (pT 25, \|η\| 2.4/2.5, tight ID) applied per C1.
3. WP value provenance checked against a btv‑wiki page **for that exact campaign**
   (not `Run3Summer22` reused for 2023+).
4. `btagging.json.gz` payload matches the tagger and era; 2026 placeholder understood.
5. SF method (fixed‑WP vs shape) and flavour/uncertainty decomposition match the BTV
   recommendation for that tagger.
6. Run 3 `btag_sf_csv` `FIXME` rows are not silently used.

---

## 6. Cross‑check vs this repo's config (as of 2026‑08‑31)

| Observation | Detail |
|-------------|--------|
| Run 3 DeepJet/ParticleNet WPs for 2024–2026 | all `0.000` — unusable until set |
| 2025 / 2026 UParT WPs | copied from 2024 (placeholder) |
| 2023 / 2023BPix DeepJet & ParticleNet | provenance link is the 2022 `Run3Summer22` page |
| `btag_sf_csv` Run 3 | 2018 UL CSV reused, `# FIXME: Update SF` |
| `btag_sf_json` 2026 | reuses 2025 `btagging.json.gz` |

---

## 7. Evidence summary

| Item | POG | Eras | Source | Established? |
|------|-----|------|--------|--------------|
| b‑jet acceptance (pT/η/ID) | analysis | all | C1 | yes — analysis choice |
| DeepCSV L/M WPs | BTV | Run 2 UL | C1, C4 | values stored; verify vs current btv‑wiki |
| DeepJet / ParticleNet L/M WPs | BTV | 2022 / 2023 | C1, C4 | values stored (2023 uses a 2022 page); 2024+ unset |
| UParT L/M WPs | BTV | Run 2 + 2024 | C1, C4 | values stored; 2025/2026 placeholders |
| `btagging.json.gz` payloads | BTV | all | C2 | paths yes; 2026 placeholder |
| SF application method + uncertainty split | BTV | all | — | **Authoritative CMS verification required** |
| Recommended tagger per era | BTV | all | — | **Authoritative CMS verification required** |

## Last verified

- Local source review: 2026‑08‑31
- Current POG recommendation: pending
