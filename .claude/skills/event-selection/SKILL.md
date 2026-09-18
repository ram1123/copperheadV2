---
name: event-selection
description: Event-level vetoes and VBF/ggH/nocat/bJetVeto categorization applied after object selection — b-veto, VH veto, mass window, category cuts, njets/jj_eta_region partitions. Analysis-specific, not a CMS POG topic.
---

# Event Selection & Categorization

Use this skill for work involving:

- the b-tag veto, the optional VH veto, or the DY VBF-filter stitching cut;
- the VBF / ggH / nocat / bJetVeto category definitions;
- the dimuon mass window (z-peak / h-peak / h-sidebands / signal / full);
- `njets_selection` and `jj_eta_region` partitions used in control plots and to scope
  the VBF DNN's training region;
- anything touching `modules/selection.py` or `configs/categories/`.

## Where this sits in the pipeline

Object selection (`cms-object-guidelines`: good muons, extra-electron veto, good jets,
b-tag working points) → **this skill**: b-veto + optional VH veto → mass window →
VBF/ggH/nocat/bJetVeto category → MVA scoring (`mva`) narrows/orders events within a
category → stats (`stats`) builds templates per category.

Vetoes, VBF cuts, and ggH cuts are kept in **one reference file**, not three, because
they are one interdependent decision (the b-veto gates both categories; VBF-fail and
ggH-njets-binning are the same partition problem that has already produced two real
bugs — see the file's own cross-check section).

## Reference selection

- `references/categorization.md` — the whole chain: vetoes, category definitions,
  mass window, njets/jj_eta_region partitions, and the score-based alternative
  categorizer.

## Review checklist

1. Identify which of the **two parallel category implementations** the code under
   review actually uses (see the reference file §2) — they must not be assumed
   equivalent without checking both.
2. Confirm the b-veto and VBF-cut numeric literals match between whichever
   implementation is in play and the analysis note (AN-19-124).
3. Confirm optional switches (`do_VH_veto`, `do_vbf_filter_study`, HE/HF mitigation
   pT cuts) are explicitly set, not silently defaulted.
4. Confirm `njets_selection` + `jj_eta_region` combinations are valid (a pair-topology
   region needs njets≥2; a single-jet region self-gates njets==1) and that boundary
   conventions are half-open, not strict.
5. If a VBF DNN score is being interpreted, check which `jj_eta_region` the model was
   trained on (see the `mva` skill) before trusting it outside that region.

## Reporting categories

- analysis-specific inconsistency (the two implementations disagree, or a switch is
  silently defaulted against intent);
- implementation defect;
- optional improvement;
- verification required (e.g. an unconfirmed AN-19-124 cross-check).

Never invent category-cut numbers — every threshold here must trace to
`configs/categories/category_cuts.py`, `modules/selection.py`, or the analysis note
comment that already cites it.

Before treating a finding as new, check `.claude/reports/registry.md` — this exact
category logic has already been investigated (jj_eta_region partitions, njets
splits, VBF-DNN training scope) several times.
