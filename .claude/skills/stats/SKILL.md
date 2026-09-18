---
name: stats
description: Combine datacard generation (template and parametric fits), the VBF statistical pipeline, and how systematic variations flow into the fit. Analysis-specific statistics methodology, largely not a CMS POG topic.
---

# Stats (Datacards, RooFit, VBF Pipeline, Systematics)

Use this skill for work involving:

- `stage3/make_templates.py`, `stage3/make_datacards.py`,
  `stage2/ggH_datacard/generate_datacard.py`;
- RooFit analytic signal/background shapes (`src/lib/fit_functions.py`);
- `run_stats_pipeline_VBF.sh` (Combine significance/impacts/likelihood-scan chain);
- which systematic variations exist, how they're switched on, and how they end up
  as datacard nuisances.

Requires the `combine` pixi environment for anything that actually runs Combine
(`./enter_pixi.sh combine` — different ROOT pin than `default`).

Category values (rate uncertainties, blinding convention) are this analysis's own
choices; general Combine/RooFit mechanics and the LHC-style blinding convention are
common HEP statistics practice, not a CMS POG recommendation with a fixed number to
check against.

## Reference selection

- `references/template-fit.md` — binned/shape datacard path: `make_templates.py` →
  `make_datacards.py`, the `rate_syst_lookup`/`lumi_syst` nuisance dicts, the
  ggH‑category `bkg_datacards_template/` cards, and the currently‑disabled rate
  systematics.
- `references/parametric-fit.md` — RooFit analytic background/signal shapes
  (`fit_functions.py`: FEWZ×Bernstein, BWZ‑Redux, …) and their consumers.
- `references/vbf-stats-pipeline.md` — `run_stats_pipeline_VBF.sh` modes,
  year/pseudo‑year combination, blinding rules.
- `references/systematics.md` — the cross‑cutting index: the `syst` switches
  profile, `WITH_VARIATIONS`, shape‑vs‑lnN treatment, and where each uncertainty's
  actual **values** are documented (they live with the producer, not duplicated
  here).

## Review checklist

1. Identify which datacard path produced the card under review (template vs
   parametric, ggH vs VBF) — they are genuinely different code paths, not two views
   of the same thing.
2. Any rate/lnN nuisance value used is current for the year, not a stale
   Run‑2‑era placeholder still carrying a `# FIXME: update the values` comment.
3. `lumi_syst` for the year matches the current LUM recommendation
   (`cms-object-guidelines/lumi.md` §2/§3) — they must be kept in sync manually.
4. Systematic variations actually reached the datacard as intended — check
   `systematics.md` for the shape‑vs‑lnN wiring before assuming a switch alone is
   sufficient.
5. Any result presented as "observed" is checked against the blinding rules in
   `vbf-stats-pipeline.md` — this analysis is blinded; real data is never fit.

## Reporting categories

- analysis-specific inconsistency (stale rate value, lumi/nuisance mismatch);
- implementation defect;
- optional improvement;
- verification required (e.g. a Run‑3 rate uncertainty never re-derived).

Before treating a finding as new, check `.claude/reports/registry.md` — the VBF
stats-pipeline run history and the `lumi_syst` KeyError fix are already logged there.
