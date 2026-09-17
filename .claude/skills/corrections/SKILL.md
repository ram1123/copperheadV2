---
name: corrections
description: Analysis-specific corrections that don't belong to one object — Z-pT reweighting (DY mismodeling), event-by-event dimuon mass-resolution calibration, and the forward pileup-jet-ID DNN. Not fixed CMS POG numbers.
---

# Corrections (Z-pT reweighting, EBE mass calibration, pileup-jet-ID DNN)

Use this skill for work involving:

- the Drell-Yan dimuon-pT mismodeling correction (Z-pT reweighting);
- the event-by-event (EBE) dimuon mass-resolution calibration;
- the forward pileup-jet-ID DNN (`do_use_pu_dnn_score`, `src/corrections/pu_dnn.py`);
- `do_zpt`, `data/zpt_rewgt/*.yaml`, `BS_res_calib_path`, or
  `src/lib/ebeMassResCalibration/`.

Z-pT and EBE calibration are **dimuon-level** corrections; the pileup-jet-ID DNN is a
per-jet discriminant. None of the three belong to a single CMS-POG object topic,
which is why they live outside `cms-object-guidelines`. Luminosity weighting and
pileup **reweighting** (the event-weight correction, as opposed to the pileup-jet-ID
DNN) are also event-level corrections but stay documented in `cms-object-guidelines`
(`lumi.md`, `pileup.md`) — by design, since object selection needs them before the
event is even built; consult those files for those topics, this skill does not
duplicate them.

All three techniques here are standard practice in CMS H→µµ-style analyses, but none
has a POG-published number to check against — the fitted parameters, calibration
factors, and trained models are this analysis's own derivation. Treat this skill's
job as verifying the implementation is internally consistent and correctly wired, not
as checking compliance with an external recommendation.

## Reference selection

- `references/zpt-reweighting.md` — DY dimuon-pT correction: derivation pipeline,
  the "function" vs "DNN" method split, per-njet piecewise polynomial, uncertainty
  envelope, where it's switched on/off.
- `references/ebe-mass-calibration.md` — dimuon mass-resolution calibration: the
  predicted-resolution formula, the RooFit Z-peak derivation, the correctionlib
  application (coupled to the beam-spot-constrained muon momentum in `muons.md`),
  and known fit-stability gotchas.
- `references/pileup-jet-dnn.md` — forward pileup-jet rejection DNN: 4-region model
  structure, feature list, the `puIdDisc` in/out training split, and the current
  2025/2026 placeholder status.

## Review checklist

1. Confirm which method is actually active (Z-pT: `whichMethod` is a hardcoded
   variable in `copperhead_processor.py`, **not** a config switch — check it directly
   rather than trusting `switches.yaml`).
2. Confirm the correction is applied to the right sample only (Z-pT: DY MC only) and
   at the right pipeline point (both: dimuon built, before category/MVA scoring).
3. Confirm the payload/config file selected for the year + generator/NanoAOD
   combination is the one intended — both corrections have several parallel
   `data/*` files for different production tags; picking the wrong one is a real,
   easy mistake here.
4. Confirm systematic variations (if the run needs them) are actually switched on
   (`save_zpt_variations`) and understand the uncertainty derivation method in use.
5. For EBE calibration specifically: confirm `do_beamConstraint` is on if the JSON
   calibration factor is meant to apply — it is gated by that switch, not a free-
   standing correction.
6. For the pileup-jet-ID DNN specifically: confirm it's actually switched on
   (`do_use_pu_dnn_score`, off by default), not combined with `do_use_pySR_score`,
   and that no efficiency/mistag scale factor is being assumed — none exists.

## Reporting categories

- analysis-specific inconsistency;
- implementation defect (e.g. wrong config file for the year/generator);
- optional improvement;
- verification required (e.g. an open/known-broken derivation for a given era — check
  `.claude/reports/registry.md` before assuming a fix already landed).
