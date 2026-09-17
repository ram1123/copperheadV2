# Systematics — Cross-Cutting Index

Responsible: **analysis-specific mechanism**. This file does **not** duplicate any
uncertainty's numeric value — each value is documented once, with its producer. This
file explains how variations get turned **on**, how they **flow** into a datacard, and
points to where each value actually lives.

## Stored sources

| # | Source | Location | Snapshot / verified |
|---|--------|----------|---------------------|
| C1 | Systematics activation profile | `configs/parameters/switches_profiles/syst.yaml` | 2026‑09‑17 |
| C2 | Profile application mechanism | `scripts/apply_switches_profile.py`, `scripts/run_scenario.sh` (see project `CLAUDE.md` "Switches profiles") | 2026‑09‑17 |
| C3 | Stage‑1/2 variation gate | `WITH_VARIATIONS` env var (stage2/stage3's `--no_variations` flag + output-path suffix) vs. `do_jec_unc`/`do_jer_unc` (the actual stage‑1 `pt_variations` loop gates) | 2026‑09‑17, `syst.yaml` comments |
| C4 | Shape‑vs‑lnN wiring into the datacard | `template-fit.md` (this skill) | — |

Classification tags: **[Analysis-specific]**, **[Implementation]**.

---

## 1. How a systematics run is turned on

`configs/parameters/switches_profiles/syst.yaml` (C1) is a **named override set**
applied onto the shared `switches.yaml` (see `CLAUDE.md`'s "Switches profiles" note —
`switches.yaml` is one file read by every year/rule in the Snakemake DAG, so it can
only be safely changed **once, before the DAG starts**, never per-rule mid-run).

Current `syst` profile content: `do_jec_unc: true`, `do_jer_unc: true`,
`save_all_weight_variations: true`, `save_muon_roccor_variations: true`,
`save_zpt_variations: true`, `do_zpt: true`, `do_add_jet_ID_vars: false`,
`save_four_jets_kinematics: false`.

**Two separate gates, easy to conflate:**

- `do_jec_unc` / `do_jer_unc` (a `switches.yaml` key) — the **actual** gate for
  stage‑1's own `pt_variations` loop. Without these, stage‑1 only ever produces the
  `"nominal"` variation, regardless of anything else.
- `WITH_VARIATIONS=1` (an **environment variable**, not a `switches.yaml` key) —
  controls stage2/3's `--no_variations` flag and output-path suffix. It does
  **nothing** at stage‑1 by itself; it must be set explicitly on the
  `run_analysis_pipeline.sh` invocation:
  `WITH_VARIATIONS=1 bash run_analysis_pipeline.sh -m 1 ...`.

Note on `do_jer_unc` specifically: it's flagged "Not validated yet for Run-3"
directly in `switches.yaml`, but as of `jer_strat=4` (the official JME ad hoc
mitigation — `cms-object-guidelines/jets.md` §4.3), Run 3's **nominal** jets already
get JER smearing, and the JER up/down evaluation runs **unconditionally** as part of
that same smearing pass — `do_jer_unc` only controls whether those *already
computed* variations get **exposed** through stage‑1's `pt_variations` loop. It
stays `false` in the base `switches.yaml` and is turned on only by the `syst`
profile.

`do_zpt: true` in this profile assumes the Z‑pT correction (`corrections/
zpt-reweighting.md`) has already been validated and is part of the intended
production config for the years being run — edit the profile (or
`--set do_zpt=false`) if a systematics run is wanted *without* Z‑pT.

---

## 2. Where each systematic's value actually lives

Don't duplicate numbers here — go to the producer:

| Systematic | Value lives in |
|------------|-----------------|
| JES sources / regrouped scheme | `cms-object-guidelines/jets.md` §3.4 |
| JER smearing / up-down | `cms-object-guidelines/jets.md` §4 |
| Muon Rochester/scale-smearing variations | `cms-object-guidelines/muons.md` §4.2 |
| Muon ID/iso/trigger SF variations | `cms-object-guidelines/muons.md` §5 |
| b-tag SF uncertainty | `cms-object-guidelines/b-tagging.md` §4 (method **[Verify]**) |
| Pileup reweighting up/down | `cms-object-guidelines/pileup.md` §2 |
| Luminosity (`lumi_syst`, per-year `lnN`) | `cms-object-guidelines/lumi.md` §2–§3 |
| Z-pT reweighting envelope | `corrections/zpt-reweighting.md` §3 |
| EBE mass-resolution calibration | `corrections/ebe-mass-calibration.md` (calibration factor itself, not a variation) |
| VBF DNN systematics handling | `mva/vbf-dnn.md` §3 (`use_nominal_dnn_features_for_systs`, default true) |
| ggH BDT systematics handling | `mva/ggh-bdt.md` §3 (re-evaluated per variation) |

---

## 3. How variations become datacard nuisances

See `template-fit.md` §2 for the full picture; summary:

- Every systematic variation column present in the stage‑2 output becomes a
  `shape` nuisance automatically — this is the default and, as of this review, the
  **only** systematic type actually active in a produced datacard.
- `lumi_syst` is the one `lnN` nuisance actually written (per-year flat luminosity
  uncertainty — must match `lumi.md`).
- `rate_syst_lookup` (cross-section/acceptance `lnN` rate uncertainties) is defined
  but its injection into the datacard is currently **commented out** — and its
  Run‑3 values are stale 2018 copies besides. Don't assume any rate systematic
  beyond luminosity is live in a current datacard.

---

## 4. Review checklist

1. For a "with systematics" request: `do_jec_unc`/`do_jer_unc` on (not just
   `WITH_VARIATIONS`), and `WITH_VARIATIONS=1` actually passed to the CLI invocation
   (not just implied by the profile).
2. The `syst` profile was applied **once**, before the DAG started, not mid-run.
3. Whether `do_zpt` should be on or off for this systematics run — an explicit
   choice, not the profile's default silently accepted.
4. Every systematic value quoted in a review traces to its producer's file (§2
   table), not restated from memory.
5. Rate systematics (`rate_syst_lookup`) are not assumed active in the datacard —
   check `template-fit.md`'s current status first.

## Last verified

- Local source review: 2026‑09‑17
