# Electrons — Stored CMS Recommendations

Responsible POG: **EGM** (E/Gamma POG).

Role in this analysis: electrons are a **veto object only** — there are no electrons in
the H→µµ final state. An event is rejected if any electron passes the pre‑selection
below (top / diboson suppression).

## Stored sources

| # | Source | Location | Snapshot / verified |
|---|--------|----------|---------------------|
| S1 | This analysis's electron‑veto definition | `docs/Official_recommendation.md` (Electron Selection table) | local review 2026‑08‑31 |
| S2 | EGM Run 3 offline ID names + WPs (`EgammaIDRecipesRun3`) | CMS PdmV `PdmVRun3Analysis`, "Notes from POGs / From E/Gamma" — see `lumi.md` §9 | via user, 2026‑09‑01 |
| C1 | Per‑year working‑point keys | `configs/parameters/electron.yaml` | 2026‑08‑31 |
| C2 | Implementation | `src/copperhead_processor.py` (electron‑veto block) | 2026‑08‑31 |

Per S2, the Run 3 offline MVA electron ID is `mvaEleID-RunIIIWinter22-iso` (wp80,
wp90), available from CMSSW_126X / NanoV11 — the NanoAOD `Electron_mvaIso_WP90` branch
is its **wp90** point. Still **[Verify]** for the target era: that Winter22 is the
current EGM training (not superseded for 2023/2024/2025), the Run 3 gap definition, the
correctionlib payload/version, and reconstruction + ID scale factors (needed only if
electrons become a selection object).

Classification tags: **[EGM official]**, **[Analysis‑specific]**, **[Implementation]**,
**[Verify]**.

---

## 1. Required context

Run 2 vs Run 3; exact era; NanoAOD campaign (v9 → `mvaFall17V2Iso_WP90`; v12/v15 →
`mvaIso_WP90`); whether electrons stay a veto object or are promoted to a selection
object (changes everything below).

---

## 2. Electron pre‑selection (veto definition)  **[Analysis‑specific]**

An event is **rejected** if any electron passes all of:

| Requirement | Value | Classification |
|-------------|-------|----------------|
| pT | > **20 GeV** | analysis‑specific (`electron_pt_cut`) |
| \|η\| | < **2.5** | detector acceptance (`electron_eta_cut`) |
| barrel–endcap gap | exclude **1.44 < \|η\| < 1.57** | standard EGM exclusion (S1) |
| ID | MVA ID **with isolation**, **WP90** | EGM POG working point |

Working‑point key by NanoAOD campaign (C1):

| NanoAOD | Key | EGM ID (S2) |
|---------|-----|-------------|
| v9 (Run 2 UL) | `mvaFall17V2Iso_WP90` | Fall17V2 MVA, iso, 90 % WP |
| v12 / v15 | `mvaIso_WP90` | `mvaEleID-RunIIIWinter22-iso` **wp90** |

No IP or extra isolation cut is layered on top — the MVA‑with‑iso WP is the whole
definition.

---

## 3. Energy corrections  **[Verify]**

Electron energy scale and smearing (EGM) are **not applied** — electrons only enter a
pass/fail veto. If electrons are ever promoted to a kinematic object:

- apply the EGM scale‑and‑smearing payload for the era (Run 2 EGM UL; Run 3 `electronSS`
  JSON) to data and simulation separately;
- re‑evaluate the veto pT threshold against the corrected pT.

---

## 4. Scale factors  **[Verify]**

None applied (veto‑only). If promoted to a selection object, era‑specific EGM
reconstruction and ID(+iso) scale factors from the EGM `electron.json` payload are
required, with the WP matching `mvaIso_WP90` / `mvaFall17V2Iso_WP90`, plus stat + syst
variations.

---

## 5. Review checklist

1. Era / NanoAOD version identified; correct WP key (`mvaFall17V2Iso_WP90` vs
   `mvaIso_WP90`).
2. pT (20), \|η\| (2.5), gap exclusion (1.44–1.57) applied per C1/C2.
3. Electrons used only as a veto — no energy correction / SF expected; if that changed,
   §3 and §4 now apply.
4. If promoted: EGM scale‑smearing + reco/ID SFs added and validated.

---

## 6. Evidence summary

| Item | POG | Eras | Source | Established? |
|------|-----|------|--------|--------------|
| Veto pre‑selection (pT/η/gap/WP90) | analysis | all | S1, C1 | yes — analysis choice |
| WP key per NanoAOD campaign | EGM | all | C1 | key yes; "is this the current EGM rec" **[Verify]** |
| Energy scale/smearing | EGM | all | — | not applied (veto‑only); **required if promoted** |
| Reco + ID scale factors | EGM | all | — | not applied (veto‑only); **required if promoted** |
| Run 3 MVA ID recommendation / gap definition / correctionlib version | EGM | Run 3 | — | **Authoritative CMS verification required** |

## Last verified

- Local source review: 2026‑08‑31
- Current POG recommendation: pending
