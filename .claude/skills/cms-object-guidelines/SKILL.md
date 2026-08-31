---
name: cms-object-guidelines
description: Apply stored official CMS POG recommendations for object selection, corrections, scale factors, and uncertainties.
---

# CMS Object Guidelines

Use this skill for work involving:

- muons;
- electrons;
- AK4 or AK8 jets;
- b tagging;
- missing transverse momentum;
- trigger-object matching;
- overlap removal;
- object corrections and scale factors;
- object-related systematic uncertainties;
- integrated luminosity and lumimask;
- pileup reweighting and the pileup-jet-ID DNN;
- the ggH BDT and VBF DNN analysis discriminants.

Photons and taus are not part of the H→μμ analysis and have no reference file.

## Required context

Before applying a recommendation, identify:

1. Run 2, Run 3, or Phase-2;
2. exact era;
3. data or simulation;
4. NanoAOD campaign and version;
5. CMSSW release, if relevant;
6. intended object working point;
7. responsible CMS POG.

If any information materially affects the recommendation and is unknown, ask
for it or mark the conclusion as unverified.

## Reference selection

Read only the relevant reference files:

- `references/muons.md`
- `references/electrons.md`
- `references/jets.md` — AK4 jets: JEC/JER, jet ID, jet veto maps, PU jet ID
- `references/fat-jets.md` — AK8 jets and substructure
- `references/b-tagging.md`
- `references/met.md` — includes MET noise / event filters
- `references/lumi.md` — integrated luminosity, golden JSON, lumi uncertainty
- `references/pileup.md` — pileup reweighting + the forward pileup-jet-ID DNN
- `references/ggh-bdt.md` — ggH-channel BDT discriminant (analysis-specific)
- `references/vbf-dnn.md` — VBF-channel DNN discriminant (analysis-specific)

Do not load every reference automatically. Overlap removal and trigger-object
matching are covered inside each object file (e.g. muon trigger matching in
`muons.md`, muon–jet cleaning in `jets.md`). Photons and taus are not part of
the H→μμ analysis; no reference file exists for them — treat any such request as
`Authoritative CMS verification required`. `ggh-bdt.md` and `vbf-dnn.md` document
analysis-specific ML discriminants — there is no CMS-POG recommendation for them;
use those files to check internal consistency, not compliance.

## Review checklist

When inspecting an object implementation, verify:

- kinematic acceptance;
- identification working point;
- isolation definition and working point;
- impact-parameter requirements;
- data-quality filters;
- correction sequence;
- scale and resolution corrections;
- data/MC scale factors;
- trigger scale factors;
- object cleaning and overlap-removal order;
- systematic uncertainty variations;
- compatibility between selection and scale factors;
- applicability to the requested era and campaign.

## Evidence rules

For every claimed official requirement, provide:

- responsible POG;
- applicable era and campaign;
- source URL or official repository;
- stored source version or Git tag, when available;
- last verification date.

Never invent or extrapolate an official recommendation.

If the stored documentation is incomplete, say:

`Authoritative CMS verification required.`

## Reporting categories

Classify findings as:

- official recommendation violation;
- analysis-specific inconsistency;
- implementation defect;
- optional improvement;
- authoritative verification required.