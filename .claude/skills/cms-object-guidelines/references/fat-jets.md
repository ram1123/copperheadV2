# Fat Jets (AK8) — Stored CMS Recommendations

Responsible POG: **JME** (JetMET) for AK8 JES/JER; **JMAR** for substructure tagging.

Role in this analysis: AK8 `FatJet` variables are stored **only** when the
`do_getFatJet_vars` switch is on; used for a boosted / VH‑style cross‑check, not the
main VBF or ggH categories. No primary result depends on fat jets.

## Stored sources

| # | Source | Location | Snapshot / verified |
|---|--------|----------|---------------------|
| S1 | JME JES page — AK8‑from‑AK4 fallback rule | `docs/temp/JME_recommendation/Jet Energy Scale - CMS JME JERC.pdf` (see `jets.md` S1) | 2026‑08‑29 |
| S2 | JME JER page | `docs/temp/JME_recommendation/Jet Energy Resolution - CMS JME JERC.pdf` (see `jets.md` S2) | 2026‑08‑29 |
| C1 | Implementation | `src/copperhead_processor.py` FatJet block (~L1438) | 2026‑08‑31 |
| C2 | Switch | `configs/parameters/switches.yaml` (`do_getFatJet_vars`) | 2026‑08‑31 |
| C3 | Substructure / mass SF payloads | `configs/parameters/SF_filelist.yaml` (`jmar_sf_file`) | 2026‑08‑31 |

Not covered by stored sources → **Authoritative CMS verification required**: AK8 JEC
availability per campaign, PUPPI softdrop‑mass corrections, JMS/JMR, the ParticleNet
`WvsQCD` working point, and JMAR tagging efficiency scale factors.

Classification tags: **[JME official]**, **[Analysis‑specific]**, **[Implementation]**,
**[Verify]**.

---

## 1. Required context

Run 2 vs Run 3; exact era; whether the fat‑jet path (`do_getFatJet_vars`) is actually
used in the result being reviewed; which substructure discriminant is read.

---

## 2. Pre‑selection as implemented (C1)  **[Analysis‑specific]**

`src/copperhead_processor.py` ~L1438 (the intended recipe is stated in the code comment):

| Requirement | Value | Classification |
|-------------|-------|----------------|
| pT | > **150 GeV** | analysis‑specific |
| \|η\| | < **2.4** | analysis‑specific |
| jet ID | **tight** (`FatJet_jetId >= 2`, or the `tight_id` helper if `jetId` absent) | JME jet ID |
| muon overlap | ΔR(fatjet, selected muon) > **0.8** | analysis‑specific cleaning |
| substructure (comment) | `FatJet_particleNetWithMass_WvsQCD > 0.75` | analysis‑specific tag cut |

Only the **leading** fat jet passing the above is kept; `msoftdrop` and related fields
are stored.

---

## 3. Energy corrections

- **[JME official]** AK8 PUPPI JECs come from the JME JES page (S1). **If AK8 JECs are
  unavailable, apply the AK4 JECs from the same data/simulation campaign to AK8 jets.**
- **[Verify]** PUPPI softdrop‑mass corrections and JMS / JMR (jet mass scale /
  resolution) are **not applied** — `jmar_sf_file` (C3) entries are mostly `dummy`.
  Required if the fat‑jet path enters a result.

---

## 4. Substructure taggers and scale factors  **[Verify]**

- `FatJet_particleNetWithMass_WvsQCD` is read directly, with **no** efficiency SF.
- W / Z / top tagging efficiency scale factors (JMAR) are **not** applied.
- Before any fat‑jet‑based selection enters a result, verify against the JMAR
  recommendation for the exact tagger, working point and campaign: the `WvsQCD` WP
  value, the efficiency SF, and its uncertainties.

---

## 5. Review checklist

1. Is `do_getFatJet_vars` actually on for the result under review? If not, this file
   does not apply.
2. Pre‑selection (pT 150, \|η\| 2.4, tight ID, ΔR(µ) 0.8, WvsQCD 0.75) applied per C1.
3. AK8 JEC source identified; AK4‑fallback assumption confirmed for the campaign.
4. Softdrop mass corrections / JMS / JMR decision recorded (currently none — §3).
5. `WvsQCD` WP and any JMAR SF verified against JMAR for the campaign (§4).

---

## 6. Cross‑check vs this repo's config (as of 2026‑08‑31)

| Observation | Detail |
|-------------|--------|
| `jmar_sf_file` | mostly `dummy` — no softdrop‑mass / JMS / JMR / tagging SF applied |
| substructure cut | `WvsQCD > 0.75` is in the code comment; confirm it against the code path and JMAR |
| fat‑jet path scope | exploratory only; not in the VBF/ggH primary categories |

---

## 7. Evidence summary

| Item | POG | Eras | Source | Established? |
|------|-----|------|--------|--------------|
| Pre‑selection (pT/η/ID/ΔR/WvsQCD) | analysis | all | C1 | yes — analysis choice |
| AK8 JEC / AK4 fallback | JME | Run 2 (v15) + Run 3 | S1 | rule yes; per‑campaign AK8 availability **[Verify]** |
| Softdrop mass corrections, JMS/JMR | JME | all | C3 | not applied — **[Verify]** |
| ParticleNet `WvsQCD` working point | JMAR | all | — | **Authoritative CMS verification required** |
| W/Z/top tagging efficiency SF | JMAR | all | — | **Authoritative CMS verification required** |

## Last verified

- Local source review: 2026‑08‑31
- Current POG recommendation: pending
