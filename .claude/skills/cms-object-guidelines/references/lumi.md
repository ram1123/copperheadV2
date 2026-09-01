# Luminosity — Stored CMS Recommendations

Responsible POG: **LUM** (Luminosity POG) for the integrated‑luminosity values, their
uncertainty, and the cross‑year correlation / datacard‑nuisance scheme; **PdmV** (via
`PdmVRun3Analysis`) for the per‑era data tables, era→run‑range definitions, golden‑JSON
certification files, and the era→MC‑campaign mapping; **DQM‑DC** for the certified JSONs.

## Stored sources

| # | Source | Location | Snapshot / verified |
|---|--------|----------|---------------------|
| S1 | CMS LUM `LumiRecommendationsRun3` — "Luminosity for pp 13.6 TeV data" (per‑year brilcalc commands, normtags, totals, uncertainties, 2022–2024 combination + covariance + datacard nuisances) | `twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun3` — transcribed into §2–§3 | via user, 2026‑09‑01 |
| S2 | Public results: **LUM‑22‑001** (2022, published); **CMS‑DP‑2024‑068** (2023); **CERN‑CMS‑DP‑2026‑003** = `cds.cern.ch/record/2952191` (2024, preliminary) | referenced from S1 | 2026‑09‑01 |
| S3 | 2022–2024 combination talk | `indico.cern.ch/event/1617597/` | 2026‑09‑01 |
| S4 | CMS **PdmV `PdmVRun3Analysis`** — per‑era Delivered/Recorded/Golden‑JSON tables, era→run ranges, DQM‑DC golden‑JSON paths, era→MC‑campaign mapping, POG entry points | `twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis` — transcribed into §4–§6, §9 | via user, 2026‑09‑01 |
| S5 | CMS **`BrilcalcQuickStart`** — brilcalc setup on LXPLUS + normtag explanation | `twiki.cern.ch/twiki/bin/view/CMS/BrilcalcQuickStart` — transcribed into §2.1 | via user, 2026‑09‑01 |
| C1 | Per‑year integrated luminosity (pb⁻¹) + lumimask paths | `configs/parameters/lumi.yaml` | 2026‑08‑31 |
| C2 | Per‑era luminosity + per‑run breakdown | `docs/Official_recommendation.md`, `docs/Run3_all_basic_Information.md` | 2026‑08‑31 |
| C3 | Implementation — lumimask on data | `src/copperhead_processor.py` L798–805 (`coffea.lumi_tools.LumiMask`) | 2026‑08‑31 |
| C4 | Implementation — MC luminosity weight | `src/copperhead_processor.py` L1585–1601 (`weights.add("lumi", …)`) | 2026‑08‑31 |
| C5 | Helper to recompute from golden JSON | `scripts/compute_lumi_from_golden_json.sh` (untracked) | 2026‑08‑31 |

Not covered → **Authoritative CMS verification required**: **2025** is preliminary
(5 % uncertainty); **2026** has *no* uncertainty estimate yet; re‑pull all Run 3 values
periodically (LUM re‑calibrates earlier data retrospectively; PdmV tables are
"Preliminary Offline Results"). Run 2 (13 TeV) uncertainty / correlation scheme is not
transcribed here — use `LumiRecommendationsRun2`.

Classification tags: **[LUM official]**, **[PdmV official]**, **[DQM‑DC payload]**,
**[Implementation]**, **[Verify]**.

---

## 1. Required context

Exact era; **data vs MC** (lumimask on data; the integrated‑luminosity value scales MC);
which **normtag** and brilcalc invocation the value came from (§2); whether the result
combines multiple years (→ use the §3 nuisance scheme, not a flat uncertainty);
era→sub‑year mapping (preEE/postEE, preBPix/postBPix — §4).

---

## 2. Official LUM recommendations — pp 13.6 TeV (Run 3)  **[LUM official]**

"Total" numbers are obtained **without a certified JSON** (full run range); analyses
apply their own golden JSON. Normtags live under
`/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/`. `-b "STABLE BEAMS"` throughout.

| Era | normtag | Total delivered (recorded) | Uncertainty | Status / reference |
|-----|---------|-----------------------------|-------------|--------------------|
| **2022** | `normtag_PHYSICS.json` | 41.47 fb⁻¹ (38.01) | **1.4 %** | LUM‑22‑001, **finalized/published** |
| **2023** | `normtag_PHYSICS.json` | 32.7357 fb⁻¹ (30.1028) | **1.3 %** | CMS‑DP‑2024‑068 |
| **2024** | `normtag_PHYSICS.json` | 124.04 fb⁻¹ (114.44) | **1.6 %** (preliminary) | CERN‑CMS‑DP‑2026‑003, updated Dec 2025 |
| **2025** | `normtag_BRIL.json` | 125.07 fb⁻¹ delivered (`--begin 10491 --end 11245 --amodetag PROTPHYS`; `PROTPHYS` excludes the pO/OO/NeNe fills) | **5 %**, **uncorrelated** with previous years | preliminary, updated Feb 2026 (Z‑counting vs partial vdM tension) |
| **2026** | *none* — use `--datatag online` (does **not** use `normtag_BRIL.json`) | prompt/online only | **none available yet** | online lumi; new fills lag up to a few days; re‑calibrations applied retrospectively — redo periodically |

Command form (2022–2024): `brilcalc lumi --normtag <normtag_PHYSICS.json> -u /fb -i [json]`.
2025: `brilcalc lumi -u /fb -b "STABLE BEAMS" --normtag <normtag_BRIL.json> -i [json]`.
2026: `brilcalc lumi -b "STABLE BEAMS" --datatag online -u /fb -i [json]`.

### 2.1 Obtaining the values — running brilcalc (S5)

**Setup on LXPLUS** — a container image gives an environment‑independent brilcalc for
all users:

```bash
source /cvmfs/cms-bril.cern.ch/cms-lumi-pog/brilws-docker/brilws-env
which brilcalc          # brilcalc is enabled via an alias
brilcalc --version      # always confirm you have the latest
```

(If the container misbehaves, fall back to the pip‑install method on the TWiki; report
issues to the LUM POG conveners.)

**Typical command:**

```bash
brilcalc lumi --normtag /cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json \
  -u /fb -i [your json]
```

- `-i [your json]` — the run/lumi‑section range to integrate: the PdmV **certification
  JSON** (§6) for a finalized number, or `processedLumis.json` from a CRAB job for the
  luminosity actually analysed.
- `-o output.csv` — dump to CSV (also switches to a standard CSV format; preferred over
  a shell `>` redirect).

**Normtag — mandatory.** The normtag file pins the luminosity calibrations and detectors
for each period; the LUM POG updates it as calibrations improve, so always use the
latest.

| Normtag | Use |
|---------|-----|
| `…/Normtags/normtag_PHYSICS.json` | final approved numbers — **all pp runs**; the *only* normtag valid for a physics analysis |
| `…/Normtags/normtag_BRIL.json` | best *preliminary* calibrations, when no approved number exists yet (e.g. 2025) |

**Never run `brilcalc` without `--normtag`** (unless you know exactly why): you get the
online luminosity, which later calibration work often shifts significantly — not
accurate, precise, or stable. (2026 is the exception in §2: only `--datatag online` is
available so far.)

---

## 3. Cross‑year combination, correlation and datacard nuisances (2022–2024)  **[LUM official]**

Combination of the **Golden‑JSON** integrated luminosities (S1, S3), fb⁻¹:

```
34.75 ± 0.48 (1.4%)   [2022]
28.40 ± 0.36 (1.3%)   [2023]
110.11 ± 1.77 (1.6%)  [2024]
--------------------------------
173.26 ± 2.07 (1.2%)  [2022+2023+2024]
```

Covariance matrix (fb⁻²; parentheses are %², i.e. `100² · cov_ij / (L_i·L_j)`):

| | 2022 | 2023 | 2024 |
|--|------|------|------|
| **2022** | 0.2297 (1.9023) | 0.0231 (0.2336) | 0.1049 (0.2742) |
| **2023** | 0.0231 (0.2336) | 0.1323 (1.6403) | 0.2802 (0.8959) |
| **2024** | 0.1049 (0.2742) | 0.2802 (0.8959) | 3.1270 (2.5791) |

Total uncertainty for dataset combinations: **2022&2023 → 1.01 %**;
**2023&2024 → 1.41 %**; **2022&2023&2024 → 1.20 %**.

### Combine datacard nuisances — new reduction method (S3), *not* the Run 2 scheme

**2022 + 2023 + 2024**

| Nuisance | 2022 | 2023 | 2024 |
|----------|------|------|------|
| `lumi_1 lnN` | 1.0138 | 1.0017 | 1.0020 |
| `lumi_2 lnN` | – | 1.0127 | 1.0068 |
| `lumi_3 lnN` | – | – | 1.0144 |

**2022 + 2023** — `lumi_1`: 1.0138 / 1.0017 ; `lumi_2`: – / 1.0127.
**2023 + 2024** — `lumi_1`: 1.0115 / – ; `lumi_2`: 1.0056 / 1.0161.

Single Run 3 year → flat §2 value. **2025** enters as its own independent nuisance
(uncorrelated). **2026** has no uncertainty — flag any 2026 result.

---

## 4. PdmV per‑era data tables (S4)  **[PdmV official]**

Golden‑JSON columns are what an analysis actually integrates. "Preliminary Offline
Results" for 2024–2026.

### 4.1 Era → sub‑year → MC campaign

| Sub‑year key (repo) | Eras | MC campaign | GT (NanoAODv12 unless noted) |
|--------------------|------|-------------|------------------------------|
| `2022preEE` | 2022 C, D | `Run3Summer22` (pre EE+ leak) | `130X_mcRun3_2022_realistic_v5` |
| `2022postEE` | 2022 E, F, G | `Run3Summer22EE` (post EE+ leak) | `130X_mcRun3_2022_realistic_postEE_v6` |
| `2023` (preBPix) | 2023 C | `Run3Summer23` | `130X_mcRun3_2023_realistic_v14` |
| `2023BPix` (postBPix) | 2023 D | `Run3Summer23BPix` | `130X_mcRun3_2023_realistic_postBPix_v2` |
| `2024` | 2024 C–I | `RunIII2024Summer24` (NanoAODv15) | `150X_mcRun3_2024_realistic_v2` |
| `2025` | 2025 C–G | Summer24 MC (see `jets.md` / `pileup.md`) | — |
| `2026` | 2026 A, B, D (C = low‑PU, excluded) | Summer24 MC | — |

### 4.2 Golden‑JSON luminosity per era (fb⁻¹, with `normtag_PHYSICS.json` for 2022/2023)

| Year | Per‑era Golden JSON | Analysis total (eras used) |
|------|---------------------|-----------------------------|
| **2022** | B 0.10 · **C 5.02 · D 2.97** · **E 5.81 · F 17.78 · G 3.09** | preEE (C+D) = **7.99**, postEE (E+F+G) = **26.68**, C–G = **34.67** (B excluded) |
| **2023** | B 0.64 · **C 17.96** · **D 9.68** | preBPix (C) = **17.96**, postBPix (D) = **9.68**, C+D = **27.64** (B excluded); PdmV B+C+D total = 28.28 |
| **2024** | B 0.13 · C 7.26 · D 7.98 · E 11.42 · F 28.04 · G 38.07 · H 5.49 · I 11.56 | C–I = **109.82** (B excluded); PdmV total B–I = 109.95 |
| **2025** | B 0.25 · C 21.56 · D 25.82 · E 14.05 · F 26.69 · G 22.25 | C–G = **110.38**; PdmV total B–G = 110.63 |
| **2026** | **A 0.64 · B 15.28** · C 2.11 (low‑PU) · **D 10.03** | A+B+D = **25.95** (C excluded); PdmV total A–D = 28.06 |

Delivered / recorded totals (for reference): 2022 41.47/38.01 · 2023 32.74/30.10 ·
2024 124.04/114.44 · 2025 125.07/114.25 · 2026 33.10/30.36 fb⁻¹.

---

## 5. Era → run ranges (S4, PromptReco Tier0 eras)

Physics pp eras only (HI / pO / OO / NeNe / commissioning omitted).

| Year | Era ranges (first–last run) |
|------|------------------------------|
| 2022 | B 355065–355793 · C 355794–357486 · D 357487–359021 · E 359022–360331 · F 360332–362180 · G 362350–362760 |
| 2023 | B 366365–367079 · C 367080–369802 · D 369803–372415 · E 372417–373075 · F 373076–373861 |
| 2024 | B 378971–379411 · C 379412–380252 · D 380253–380947 · E 380948–381943 · F 381944–383779 · G 383780–385813 · H 385814–386408 · I 386409–387121 |
| 2025 | B 391531–392158 · C 392159–393609 · D 394286–395967 · E 395968–396597 · F 396598–397853 · G 397854–398903 |
| 2026 | A 401600–401836 · B 401837–402513 · C 402514–403422 (low‑PU) · D 403423–404067 |

---

## 6. Lumimask / golden JSON (C1, C3; DQM‑DC paths from S4)  **[PdmV / DQM‑DC payload]**

Applied to **data only**, via `LumiMask(events.run, events.luminosityBlock)`
(`copperhead_processor.py` L800–805). DQM‑DC base URL:
`https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions<YY>/`.

| Era | Golden JSON (repo `lumi.yaml`) | DQM‑DC cert range |
|-----|-------------------------------|-------------------|
| 2016preVFP / 2016postVFP | `Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt` | — |
| 2017 | `Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt` | — |
| 2018 | `Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt` | — |
| 2022preEE / 2022postEE | `Cert_Collisions2022_355100_362760_Golden.json` | `355100_362760` |
| 2023 / 2023BPix | `Cert_Collisions2023_366442_370790_Golden.json` | `366442_370790` |
| 2024 | `Cert_Collisions2024_378981_386951_Golden.json` | `378981_386951` |
| 2025 | `Cert_Collisions2025_391658_398903_Golden.json` | `391658_398903` (also `..._Muon.json`, `2025_lowPU.json`) |
| 2026 | `Cert_Collisions2026_401624_403937_golden.json` | `401624_403937` |

RERECO eras use separate ReReco JSONs under `data/lumimasks/RERECO/`. A `..._Muon.json`
(looser muon‑only certification) exists per Run 3 year — this analysis uses the **Golden**
JSON.

---

## 7. Integrated luminosity for MC normalisation (C1, C4)

`lumi.yaml` `integrated_lumis` (pb⁻¹):

| Era | pb⁻¹ | Era | pb⁻¹ |
|-----|------|-----|------|
| 2016preVFP | 19500.0 | 2022preEE | 7990.0 |
| 2016postVFP | 16810.0 | 2022postEE | 26680.70 |
| 2017 | 41480.0 | 2023 | 17960.0 |
| 2018 | 59830.0 | 2023BPix | 9680.0 |
| 2016_RERECO | 35920.0 | 2024 | 109820.0 |
| 2017_RERECO | 41529.0 | 2025 | 110840.0 |
| 2018_RERECO | 59740.0 | 2026 | 25950.0 (eras A+B+D; era C low‑PU excluded) |

MC weight: `weights.add("lumi", weight = gen_weight_ones * integrated_lumi)` with
`integrated_lumi = sample_info["total_lumi_pb"]` — the value used comes from the
**prestage `sample_info`**, not directly from `lumi.yaml`. Verify prestage matches.

---

## 8. Cross‑check vs this repo (as of 2026‑09‑01)

### 8.1 vs the PdmV Golden‑JSON per‑era tables (§4.2)

| Sub‑year | `lumi.yaml` (fb⁻¹) | PdmV Golden JSON (fb⁻¹) | Match |
|----------|--------------------|-------------------------|-------|
| 2022preEE | 7.990 | C+D = 7.99 | ✓ exact |
| 2022postEE | 26.681 | E+F+G = 26.68 | ✓ exact |
| 2023 (preBPix) | 17.960 | C = 17.96 | ✓ exact |
| 2023BPix | 9.680 | D = 9.68 | ✓ exact |
| 2024 | 109.820 | C–I = 109.82 | ✓ exact |
| 2025 | 110.840 | C–G = 110.38 (B–G = 110.63) | ≈ (0.2–0.5 fb⁻¹ high — re‑derive) |
| 2026 | 25.950 | A+B+D = 25.95 | ✓ exact |

So `lumi.yaml` already matches the PdmV **with‑normtag Golden JSON**, with era B
(2022/2023/2024) and era C (2026 low‑PU) correctly excluded. Only **2025 (110.84 vs
110.38/110.63)** needs re‑deriving from the 2025 golden JSON with `normtag_BRIL.json`.

### 8.2 vs the LUM combination inputs (§3)

The LUM 2022–2024 combination uses slightly different per‑year totals (2022 34.75,
2023 28.40, 2024 110.11) — different golden‑JSON snapshot and, for 2023, apparently
including era B. Use the §3 numbers for the **uncertainty and datacard nuisances**; the
per‑era **luminosity values** for normalisation come from §4.2 / `lumi.yaml`.

### 8.3 vs the repo's own docs

`lumi.yaml` still disagrees with `docs/Official_recommendation.md` /
`docs/Run3_all_basic_Information.md` for some eras (older slide values). The PdmV S4
numbers in §4.2 supersede those — update the docs to match `lumi.yaml` + §4.2.

---

## 9. POG entry points for Run 3 (S4, PdmV "Notes from POGs")

| POG | Entry point | See |
|-----|-------------|-----|
| MUO | `MuonPOG#User_Recommendations`; `MuonRun32022`, `MuonRun3_2023` | `muons.md` |
| EGM | `EgammaPOG`; `EgammaIDRecipesRun3` | `electrons.md` (ID table below) |
| TAU | `TauIDRecommendationForRun3`; `TauTrigger` | not used in H→µµ |
| LUM | `LumiRecommendationsRun3#Pileup_reweighting` | `pileup.md` §A |
| BTV | `btv-wiki.docs.cern.ch/ScaleFactors/` | `b-tagging.md` |
| JME | `JetMET#Quick_links_to_current_recommend` | `jets.md` |

EGM Run 3 offline IDs (available from CMSSW_126X / NanoV11):

| Object | ID | Working points |
|--------|----|----------------|
| Electron | `mvaEleID-RunIIIWinter22-iso` | wp80, wp90 |
| Electron | `mvaEleID-RunIIIWinter22-noIso` | wp80, wp90 |
| Electron | `cutBasedElectronID-RunIIIWinter22-V1` | veto, loose, medium, tight |
| Photon | `mvaPhoID-RunIIIWinter22-v1` | wp80, wp90 |
| Photon | `cutBasedPhotonID-RunIIIWinter22-122X-V1` | loose, medium, tight |

This analysis's `mvaIso_WP90` electron veto (see `electrons.md`) corresponds to
`mvaEleID-RunIIIWinter22-iso` **wp90**.

---

## 10. Run 2 (13 TeV)

Repo values: 2016preVFP 19.5, 2016postVFP 16.81, 2017 41.48, 2018 59.83 fb⁻¹ (2016 total
36.31; Run 2 ≈ 138 fb⁻¹) — consistent with the LUM UltraLegacy recommendation. The Run 2
uncertainty and its `lumi_13TeV_*` correlation scheme are **not transcribed here** — use
`LumiRecommendationsRun2`.

---

## 11. Review checklist

1. Era identified; data gets the golden JSON (§6), MC gets the integrated‑luminosity
   scale (§7); era→sub‑year mapping (§4.1) correct (preEE/postEE, preBPix/postBPix).
2. Lumimask path matches the era (UL vs RERECO); Golden not Muon JSON.
3. `sample_info["total_lumi_pb"]` from prestage matches §4.2 / `lumi.yaml` for the era.
4. 2025 value re‑derived (§8.1); `lumi.yaml` vs docs reconciled (§8.3).
5. 2022/2023/2024 exclude era B; 2026 excludes era C (low‑PU).
6. Datacard luminosity nuisance: single Run 3 year → flat §2 value; 2022+2023(+2024)
   combination → §3 `lumi_1/2/3 lnN` (new reduction method); 2025 → independent
   nuisance; any 2026 result → flag the missing uncertainty.

---

## 12. Evidence summary

| Item | POG | Eras | Source | Established? |
|------|-----|------|--------|--------------|
| Golden JSON / lumimask paths + DQM‑DC ranges | PdmV / DQM‑DC | all | C1, S4 | yes |
| Lumimask applied to data only | — | all | C3 | yes (implementation) |
| Per‑era Delivered/Recorded/Golden‑JSON tables | PdmV | Run 3 | S4 | yes (§4.2); 2024–2026 "Preliminary Offline" |
| Era → run ranges | PdmV | Run 3 | S4 | yes (§5) |
| Era → sub‑year → MC campaign / GT | PdmV | Run 3 | S4 | yes (§4.1) |
| Per‑year uncertainty | LUM | 2022 / 2023 / 2024 | S1, S2 | **yes**: 1.4 % / 1.3 % / 1.6 % |
| Per‑year uncertainty | LUM | 2025 | S1 | **yes**: 5 %, uncorrelated |
| Per‑year uncertainty | LUM | 2026 | S1 | **none available** — flag |
| Cross‑year covariance + combined totals + `lnN` nuisances | LUM | 2022–2024 | S1, S3 | **yes** (§3) |
| `lumi.yaml` per‑era vs PdmV Golden JSON | LUM/PdmV | Run 3 | §8.1 | **match**, except 2025 (re‑derive) |
| MC lumi weight uses prestage `total_lumi_pb` | — | all | C4 | yes — verify it matches |
| Run 2 uncertainty / correlation scheme | LUM | 2016–2018 | — | **use `LumiRecommendationsRun2`** |
| EGM Run 3 electron/photon ID names + WPs | EGM | Run 3 | S4 | yes (§9) |

## Last verified

- Official LUM Run 3 recommendation (S1–S3) + PdmV `PdmVRun3Analysis` (S4) + `BrilcalcQuickStart` (S5): via user, 2026‑09‑01
- Local source review: 2026‑08‑31
- Re‑pull Run 3 values periodically — LUM re‑calibrates earlier data retrospectively;
  2025 preliminary, 2026 has no uncertainty yet
