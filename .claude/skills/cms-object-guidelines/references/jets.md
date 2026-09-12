# AK4 / AK8 Jets — Stored CMS Recommendations

Responsible POG: **JERC** (JetMET / Jet Energy Resolution & Corrections subgroup of the
JME POG).

## Stored sources

| # | Source | Location | Snapshot / verified |
|---|--------|----------|---------------------|
| S1 | Jet Energy Scale Recommendations — CMS JME JERC | `docs/temp/JME_recommendation/Jet Energy Scale - CMS JME JERC.pdf` — `https://cms-jme-jerc.docs.cern.ch/recommendations/jes/` | page snapshot 2026‑08‑29 |
| S2 | Jet Energy Resolution Recommendations — CMS JME JERC | `docs/temp/JME_recommendation/Jet Energy Resolution - CMS JME JERC.pdf` — `https://cms-jme-jerc.docs.cern.ch/recommendations/jer/` | page snapshot 2026‑08‑29 |
| S3 | Jet Veto Map Recommendations — CMS JME JERC | `docs/temp/JME_recommendation/Jet Veto Maps - CMS JME JERC.pdf` — `https://cms-jme-jerc.docs.cern.ch/recommendations/jet-veto-maps/` | page snapshot 2026‑08‑29 |
| S4 | JME "Mitigation techniques" slide (A. Benecke, slide 14) — HE/HF forward‑jet pT‑cut mitigation | shared image; **no persistent copy in repo** | via user, 2026‑08‑30 |

Not covered by the stored snapshots (mark **Authoritative CMS verification required**
if the analysis needs them): full Jet ID working‑point definitions, PU jet ID working
points, PUPPI‑weight details, per‑source JEC uncertainty split, JER hybrid‑method
numeric matching thresholds, b tagging (see `references/b-tagging.md`), Type‑I MET
propagation (see `references/met.md`).

Classification tags used below: **[JERC official]**, **[Analysis‑specific]** (approved
per analysis, not a general prescription), **[Implementation]** (this repo's choice),
**[Verify]** (not in stored sources).

---

## 1. Required context before applying anything

Pin down, in this order:

1. **Run 2 / Run 3** — recommendations, tags and even jet content differ; never carry one
   into the other.
2. **Exact era** — 2016preVFP, 2016postVFP, 2017, 2018, 2022preEE, 2022postEE, 2023
   (PreBPix), 2023BPix (PostBPix), 2024, 2025, 2026. (2026 era C is a **dedicated
   low‑PU run** with its own JER inputs and veto‑map note — S1, S2, S3.)
3. **Data or simulation** — L2L3Residual and JER smearing are handled differently
   (§3, §4).
4. **NanoAOD campaign/version** — this selects the JEC/JER **global tag** and the jet
   collection:
   - NanoAODv9 (Run 2): AK4 **CHS** + AK8 PUPPI; JES tags `Summer19UL*_V5/V7` (S1).
   - NanoAODv15 (Run 2 re‑nano): AK4 **PUPPI** + AK8 PUPPI; JES tags
     `Summer20UL*NanoV15_V1` (S1). `Jet_jetId` branch is **not stored** in NanoAODv15 —
     jet ID must be recomputed by hand (S3, §6).
   - NanoAODv12 / v15 (Run 3): AK4 PUPPI + AK8 PUPPI.
5. **Jet radius** — AK4 vs AK8. If AK8 JECs are unavailable, apply the AK4 JECs from the
   **same data/simulation campaign** to AK8 jets (S1). **[JERC official]**
6. **Object working point** — intended jet ID / pileup WP and pT–|η| acceptance
   (analysis‑specific; not fixed by JERC).

If any of these materially changes the answer and is unknown, ask or mark the
conclusion unverified.

---

## 2. Correction sequence (order is mandatory)

```
raw jet pT
  → JEC:  L1FastJet → L2Relative → L3Absolute → L2L3Residual   (L2L3Residual: DATA only)
  → JER smearing                                                (MC only, after full JEC)
  → propagate the pT change to Type-I MET
```

- JER smearing is applied **after** the JECs. For JER uncertainties: apply nominal JEC
  first, then the JER up/down variations (S2). **[JERC official]**
- A **run‑based configuration of L2L3Residual** is implemented in all Run 2
  (NanoAODv15) and Run 3 JSON files — the residual depends on the run number of the
  event, not just the era (S1). **[JERC official]**
- **Data files are not used for JER smearing** (S2). JER smearing is MC‑only.

This repo's implementation lives in `src/corrections/jet.py` (JEC/JER) and
`src/copperhead_processor.py` `jet_loop`; level lists and tags are in
`configs/parameters/jec.yaml`.

---

## 3. Jet Energy Scale (JES / JEC)

### 3.1 Correction levels

| Sample | Levels |
|--------|--------|
| MC | L1FastJet, L2Relative, L3Absolute |
| Data | L1FastJet, L2Relative, L3Absolute, **L2L3Residual** (run‑based) |

Matches `jec_levels_mc` / `jec_levels_data` in `configs/parameters/jec.yaml`.
For 2024 data the payload splits runs into "nibs" (RunCnib1, RunFnib1/2/3, …); for
2023 data the correct sub‑era is chosen from the event's run number (S1).

### 3.2 Known issue — anomalously high‑pT forward jets  **[JERC official]**

In extremely rare Run 3 cases the L2Relative JEC showed asymptotic behaviour in a very
narrow bin, giving corrected jet pT above ~10 TeV. It mainly affects the **forward (HF)
region**. Affected‑event fractions are ~1e‑7 to a few×1e‑6, only in some production
samples.

- Mitigation: **reject jets with `Jet_rawFactor > 0.9`** (S1).
- Applies **only to affected production samples** — check the linked JERC GitLab issue
  for the specific samples before enabling.
- This repo: switch `do_reject_high_rawFactor_jets` in
  `configs/parameters/switches.yaml` (repo commit `0aa1c45`). **[Implementation]**

### 3.3 Recommended global tags (S1, 2026‑08‑29 snapshot)

**Run 3 — MC tag / DATA tag share the base name (`_MC` vs `_DATA`):**

| Era | JES tag (base) | Notes |
|-----|----------------|-------|
| 2022 PreEE | `Summer22_22Sep2023_V4` | MC = Summer22 |
| 2022 PostEE | `Summer22EE_22Sep2023_V4` | MC = Summer22EE |
| 2023 PreBPix | `Summer23Prompt23_V4` | RunCv123 / RunCv4 |
| 2023 PostBPix | `Summer23BPixPrompt23_V4` | RunD |
| 2024 | `Summer24Prompt24_V5` | RunC–I, "nibs" split; RunCnib1/Dnib1/Enib1 = ReReco |
| 2025 | `Summer24Prompt25_V3` | MC = Summer24; RunC–G Prompt |
| 2026 | `Summer24Prompt26_V1` | MC = Summer24; RunB/C/D; **era C = low‑PU** |

**Run 2 — NanoAODv15 (AK4 + AK8 PUPPI):**

| Era | JES tag |
|-----|---------|
| 2016 preVFP (APV) | `Summer20UL16APVNanoV15_V1` |
| 2016 postVFP | `Summer20UL16NanoV15_V1` |
| 2017 | `Summer20UL17NanoV15_V1` |
| 2018 | `Summer20UL18NanoV15_V1` |

**Run 2 — NanoAODv9 (AK4 CHS + AK8 PUPPI):**

| Era | JES tag |
|-----|---------|
| 2016 preVFP (APV) | `Summer19UL16APV_V7` |
| 2016 postVFP | `Summer19UL16_V7` |
| 2017 | `Summer19UL17_V5` |
| 2018 | `Summer19UL18_V5` |

Payloads: `correctionlib` JSON from `jsonpog-integration` (POG/JME) or the JERC
`cms-griddata` CAT metadata; `.txt`/`.tar`/DB also published. Visualise with the JERC
viewer.

### 3.4 JES uncertainties

- This repo uses the **regrouped / reduced** source scheme, 11 sources per year
  (`jec_unc_to_consider` in `configs/parameters/jec.yaml`): `Absolute`,
  `Absolute_<year>`, `BBEC1`, `BBEC1_<year>`, `EC2`, `EC2_<year>`, `HF`, `HF_<year>`,
  `RelativeBal`, `RelativeSample_<year>`, `FlavorQCD`. **[Implementation]** — this is the
  JME "regrouped" reduced set; the full ~25‑source list also exists.
- Precise source list / correlation model across years: **[Verify]** against the JME JEC
  uncertainty documentation for the target campaign.

---

## 4. Jet Energy Resolution (JER)

### 4.1 Method  **[JERC official]**

- **MC only.** Apply after the full JEC (§2).
- Hybrid smearing: **scaling method** when the reco jet is matched to a gen jet;
  **stochastic method** otherwise. Numeric matching thresholds (ΔR and
  |pT−pT^gen| vs. the resolution) are **[Verify]** — not quoted in the stored snapshot;
  use the JME JetResolution documentation.
- JER uncertainty: nominal JEC → JER up/down (S2). In this repo `do_jer_unc` is `false`
  by default and turned on only in the `syst` switches profile. **[Implementation]**

### 4.2 Recommended JER SF tags (S2, 2026‑08‑29 snapshot)

**Run 3 (MC):**

| Era | JER tag |
|-----|---------|
| 2022 PreEE | `Summer22_22Sep2023_JRV2_MC` |
| 2022 PostEE | `Summer22EE_22Sep2023_JRV2_MC` |
| 2023 PreBPix | `Summer23Prompt23_RunCv1234_JRV3_MC` |
| 2023 PostBPix | `Summer23BPixPrompt23_RunD_JRV3_MC` |
| 2024 | `Summer24Prompt24_JRV2_MC` |
| 2025 | `Summer24Prompt25_JRV2_MC` (smear Summer24 MC vs 2025 data) |
| 2026 RunB & RunD | `Summer24Prompt26_RunBD_JRV1_MC` (smear Summer24 MC vs 2026 data) |
| 2026 RunC (low‑PU) | `Summer24Prompt26_RunC_JRV1_MC` (separate — dedicated low‑PU run) |

**Run 2 (MC), NanoAODv15 and NanoAODv9 — same tags listed in both sections of S2:**

| Era | JER tag |
|-----|---------|
| 2016 preVFP (APV) | `Summer20UL16APV_JRV5_MC` |
| 2016 postVFP | `Summer20UL16_JRV5_MC` |
| 2017 | `Summer19UL17_JRV4_MC` |
| 2018 | `Summer19UL18_JRV3_MC` |

### 4.3 Known issue — 2.5 < |η| < 3.0, Run 3  **[Analysis‑specific]**

Data/simulation agreement of the jet **η** distribution in `2.5 < |η| < 3.0` is very
poor. Ad hoc mitigation, applied case‑by‑case; **not a general JME prescription** (S2):

- **(a)** Raise the jet pT threshold in that region (e.g. `pT > 50 GeV`) — used by
  H→Zγ.
- **(b)** Omit the **stochastic** component: correct the jet only when a gen jet is
  matched (scaling method); no stochastic smearing for unmatched jets — used by H→WW,
  **H→µµ**, VBF SUSY. **Adopting (b) requires explicit JME/JERC approval.**

This repo is an **H→µµ** analysis and implements **option (b)** as `jer_strat == 4` /
`applyStrat4()` in `src/corrections/jet.py`; base `switches.yaml` `jer_strat` = `4` for
Run 3 years (2016–2018 left at `-1`). See memory `jer-strat-4-official-jme-mitigation`.
**[Implementation]**

---

## 5. Jet horn region — HE / HF forward‑jet pT‑cut mitigation

Source **S4** (JME slide, no persistent repo copy). Regions: **HE** = `2.5 < |η| ≤ 3.0`,
**HF** = `3 < |η| < 5`, plus a `2.0 < |η| < 2.5` note for 2024. Treat as POG guidance
pending an authoritative persistent copy → effectively **[Verify]**.

| Year | HF (3<\|η\|<5) | HE (2.5<\|η\|≤3.0) | 2.0<\|η\|<2.5 |
|------|---------------|-------------------|---------------|
| 2022 | require pT > 50 GeV | require pT > 50 GeV + JER for gen‑matched jets only | n/a |
| 2023 | require pT > 50 GeV | require pT > 50 GeV + JER for gen‑matched jets only | n/a |
| 2024 | fixed — no mitigation | require pT > 50 GeV + JER for gen‑matched jets only | special L2L3Residual treatment for MC‑truth‑corrected pT < 30 GeV |
| 2025 | "fixed but worsening (radiation damage)" — **non‑definitive** | "much improved — enough for analysers?" — **non‑definitive** | "should not be a problem but please check" — **non‑definitive** |

- The "JER for gen‑matched jets only" piece of the HE mitigation **is** option (b) /
  `jer_strat=4` (§4.3).
- This repo's switches (`configs/parameters/switches.yaml`): `do_jet_horn_ptcut`
  (**HE only** — `jetHorn_region = (|η|>2.5) & (|η|≤3.0)` in `jet_loop`, never touches
  HF), and `add_pt_cut_for_HE_HF_jets` (2022/2023, broader `|η|>2.5` with no upper
  bound). As of this writing `do_jet_horn_ptcut` = `50` for 2024/2025/2026 and `false`
  for 2022/2023; `add_pt_cut_for_HE_HF_jets` = `false` for 2022/2023 despite its comment
  recommending `50` (open item). See memory `jme-horn-region-official-recommendation`.
- When a request says "the official JME recommendation", it includes these pT cuts, not
  only veto‑map handling. 2025 hedges are a **user decision point**, not something to
  resolve unilaterally (the user has chosen to treat 2025/2026 like 2024).

---

## 6. Jet Veto Maps (S3)

### 6.1 General  **[JERC official]**

- Apply the **same map to data and the corresponding MC** so both use the same
  phase space.
- **Validation is required**: produce jet (η, φ) maps before and after applying the
  veto, check the impact on other distributions, and document it in the analysis note.

### 6.2 Run 3 — **mandatory**

- **Reject the whole event** if **any** jet passing the minimal selection lies in a
  vetoed region.
- Minimal selection:
  - jet `pT > 15 GeV`
  - `tightLepVeto` jet ID (NanoAODv15: `Jet_jetId` absent → apply manually via the
    legacy Run 3 Jet ID TWiki)
  - `(chargedEmEF + neutralEmEF) < 0.9` (follows the legacy Run 2 Type‑I MET TWiki)
- 2022: the page's prose also mentions separate maps for **RunCD** vs **RunEFG**
  sub‑periods; the recommended‑maps table lists a single tag per PreEE/PostEE. Verify
  against the payload which granularity is exposed.

Recommended maps, map key `jetvetomap`:

| Era | Veto‑map tag | Note |
|-----|--------------|------|
| 2022 PreEE | `Summer22_23Sep2023_V1` | |
| 2022 PostEE | `Summer22EE_23Sep2023_V1` | flagged "affected by EE+ leak" |
| 2023 PreBPix | `Summer23Prompt23_V1` | |
| 2023 PostBPix | `Summer23BPixPrompt23_V1` | |
| 2024 | `Summer24Prompt24_V1` | |
| 2025 | `Summer24Prompt25_V1` | |
| 2026 | `Summer24Prompt26_V1` | cloned from the 2025 map |

### 6.3 Run 2

- Veto the **jet** whose axis lies in a vetoed region — **not** the whole event.
- Use the **strictest data‑driven map** (longest name). `mc` / `hw` maps are for expert
  studies; the **UL16 `h2hot_mc` map is the exception** and must be applied **in
  addition** to the strict data‑driven map.
- Loose selection: jet `pT > 15 GeV`; **tight** jet ID; PU jet ID for CHS jets with
  `pT < 50 GeV`; `(chEmEF + neEmEF) < 0.9`; jet not overlapping a PF muon
  (`ΔR < 0.2`).
- Restricted application (e.g. leading jet only, or basic map, or omit) is allowed if
  the strict veto costs too much acceptance — **validate and document**.

| Era | Tag | Map name(s) |
|-----|-----|-------------|
| 2016 (APV & non‑APV, same map) | `Summer19UL16_V0` | `h2hot_ul16_plus_hbm2_hbp12_qie11` **and** `h2hot_mc` |
| 2017 | `Summer19UL17_V2` | `h2hot_ul17_plus_hep17_plus_hbpw89` |
| 2018 | `Summer19UL18_V1` | `h2hot_ul18_plus_hem1516_plus_hbp2m1` (may drop `hem1516` for HEM‑reprocessed 2018CD + standard 2018AB; `hbp2m1` must stay) |

Derivation: CMS AN‑2022/066 + JERC UL veto‑regions presentation.

This repo: `do_jet_veto_maps_filterEvents` (drop events) and
`do_jet_veto_maps_filterJets` (drop jets) in `configs/parameters/switches.yaml`, plus
the JVM object selection in `src/copperhead_processor.py` (repo commits `89cd723`,
`1c3011c`). **[Implementation]**

---

## 7. Jet ID

- Full working‑point definitions (loose/tight/tightLepVeto per era, PF‑fraction cuts):
  **[Verify]** — not in the stored snapshots; use the JME Jet ID TWiki for the target
  campaign. Never invent the fraction cuts.
- **NanoAODv15**: `Jet_jetId` is **not stored**; recompute jet ID by hand following the
  legacy Run 3 Jet ID TWiki (S3).
- Veto‑map minimal selection uses `tightLepVeto` (Run 3) / `tight` (Run 2) — see §6.

---

## 8. Pileup mitigation / PU jet ID

- Run 2 **CHS** jets: PU jet ID is relevant for `pT < 50 GeV` (per the Run 2 veto‑map
  loose selection, §6.3). Working points and SFs: **[Verify]** (JME PU jet ID TWiki).
- Run 3 **PUPPI** jets: pileup is handled by the PUPPI weights; there is no separate
  legacy PU jet ID WP. This repo additionally develops a **pileup‑jet‑ID DNN**
  (`MVA_training/pileup_dnn/`, applied via `src/corrections/pu_dnn.py`) — an
  **[Implementation]** discriminant, not a JME product. The switch comment notes model
  variants differ by whether `puIdDisc` is an input (2024/2025 include it).

---

## 9. Review checklist for a jet implementation

1. Era, data/MC, NanoAOD version identified; jet collection (CHS vs PUPPI, AK4 vs AK8)
   consistent with the JES/JER tag chosen.
2. JEC level list correct (L2L3Residual data‑only; run‑based).
3. JES global tag matches §3.3 for that era **and** NanoAOD version.
4. `Jet_rawFactor > 0.9` rejection considered for affected Run 3 production samples.
5. JER smearing MC‑only, applied after JEC; JER SF tag matches §4.2; 2026 RunC uses the
   low‑PU tag.
6. `2.5 < |η| < 3.0` Run 3 mitigation applied consistently (option (b)/`jer_strat=4` for
   this H→µµ analysis) **and** JME/JERC approval on record.
7. HE/HF forward pT cuts (§5) match the intended scenario for the year; 2025/2026 choice
   is explicit.
8. Jet veto map: Run 3 = event rejection with the §6.2 minimal selection; Run 2 = jet
   veto with the §6.3 loose selection; correct per‑era tag; same map on data and MC;
   before/after validation done.
9. JES uncertainty sources = intended (regrouped 11‑source set here); JER up/down only
   in systematic runs.
10. MET Type‑I re‑propagated after JEC/JER changes (see `references/met.md`).

---

## 10. Cross‑check vs this repo's current `configs/parameters/jec.yaml` (as of 2026‑08‑31)

Tags in the repo that **differ from the S1/S2 snapshot** — verify which is intended
before a production run:

| Quantity | Repo `jec.yaml` | S1/S2 snapshot (2026‑08‑29) |
|----------|-----------------|------------------------------|
| JES 2022 preEE/postEE | `Summer22_22Sep2023_V3` / `Summer22EE_22Sep2023_V3` | **V4** |
| JES 2023 / 2023BPix | `Summer23Prompt23_V2` / `Summer23BPixPrompt23_V3` | **`Summer23Prompt23_V4` / `Summer23BPixPrompt23_V4`** |
| JER 2022 preEE/postEE | `Summer22_22Sep2023_JRV1_MC` / `Summer22EE_22Sep2023_JRV1_MC` | **JRV2** |
| JER 2023 / 2023BPix | `Summer23Prompt23_RunCv1234_JRV1_MC` / `Summer23BPixPrompt23_RunD_JRV1_MC` | **`…_RunCv1234_JRV3_MC` / `…_RunD_JRV3_MC`** |
| JER 2017 / 2018 (NanoV15 path) | `Summer19UL17_JRV2_MC` / `Summer19UL18_JRV2_MC` | **`Summer19UL17_JRV4_MC` / `Summer19UL18_JRV3_MC`** |
| JER 2016 preVFP/postVFP | `Summer20UL16APV_JRV3_MC` / `Summer20UL16_JRV3_MC` | **JRV5** |

Tags that **match** the snapshot: JES 2024 `Summer24Prompt24_V5`, JES 2025
`Summer24Prompt25_V3`, JES 2026 `Summer24Prompt26_V1`; JER 2024 `Summer24Prompt24_JRV2`,
JER 2025 `Summer24Prompt25_JRV2`, JER 2026 `Summer24Prompt26_RunBD_JRV1`; Run 2
NanoAODv15 JES `Summer20UL*NanoV15_V1`.

Note: the repo's Run 2 NanoAODv15 JEC JSON snapshots are dated 2026‑04‑13 (older than
the S1/S2 snapshot); the Run 3 CAT metadata snapshots are 2025‑09‑23 → 2025‑10‑07 for
2022/2023 and 2026‑07‑16 for 2024/2025. The 2026 JER `RunC` low‑PU split (S2) is **not**
separately selectable in this config (single tag per year bucket — documented limitation
in `jec.yaml`).

---

## 11. Evidence summary

| Item | POG | Eras | Source | Version / tag | Verified |
|------|-----|------|--------|---------------|----------|
| JEC levels, sequence, run‑based L2L3Residual, AK8←AK4 fallback | JERC | Run 2 (v15) + Run 3 | S1 | page 2026‑08‑29 | 2026‑08‑31 |
| `Jet_rawFactor > 0.9` rejection | JERC | Run 3, affected samples only | S1 | — | 2026‑08‑31 |
| JES global tags | JERC | all eras | S1 | see §3.3 | 2026‑08‑31 |
| JER method (MC‑only, hybrid, after JEC) | JERC | Run 2 + Run 3 | S2 | page 2026‑08‑29 | 2026‑08‑31 |
| JER SF tags | JERC | all eras | S2 | see §4.2 | 2026‑08‑31 |
| `2.5<|η|<3.0` Run 3 mitigation (a)/(b) | JERC (analysis‑specific, needs approval) | Run 3 | S2 | — | 2026‑08‑31 |
| HE/HF forward pT‑cut table | JME (slide) | 2022–2026 | S4 | slide 14, no repo copy | 2026‑08‑30 |
| Jet veto maps — Run 3 mandatory, event rejection, minimal selection, tags | JERC | Run 3 | S3 | see §6.2 | 2026‑08‑31 |
| Jet veto maps — Run 2 jet veto, loose selection, strict‑map rule, tags | JERC | Run 2 | S3 | see §6.3 | 2026‑08‑31 |
| Jet ID WP definitions | JME | all | — | **not in stored sources** | — |
| PU jet ID WPs / SFs | JME | Run 2 CHS | — | **not in stored sources** | — |
| JEC uncertainty source list | JME | all | — | **not in stored sources** (repo uses 11‑source regrouped set) | — |

If the analysis needs a number marked "not in stored sources", state:
`Authoritative CMS verification required.`

## Last verified

- Local source review (JERC docs snapshots S1–S3 + repo config): 2026‑08‑31
- HE/HF slide (S4): via user, 2026‑08‑30
- Current POG recommendation: JERC pages snapshotted 2026‑08‑29; re‑check against
  the live `cms-jme-jerc.docs.cern.ch` pages before a production run
