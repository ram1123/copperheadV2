---
name: mva
description: Analysis-specific ML discriminants used to categorize events (ggH BDT, VBF DNN) — architecture, inputs, training, and stage-2 application. Not CMS POG recommendations.
---

# MVA Discriminants (ggH BDT, VBF DNN)

Use this skill for work involving:

- the ggH-channel **BDT** or the VBF-channel **DNN**;
- their input-feature lists, training samples, or k-fold/year structure;
- how a score turns into subcategory bin edges;
- consistency between a training config and its stage-2 application.

> Naming: **ggH → BDT**, **VBF → DNN**. If a request has these swapped, flag it before
> proceeding.

Neither discriminant is a CMS POG product — there is no official recommendation to
check compliance against. These references exist to check **internal consistency**:
does stage-2 apply the model the way it was trained (same features, same order, same
fold split, same score transform)?

## Where this sits in the pipeline

Object selection (`cms-object-guidelines`) → event selection / categorization
(`event-selection`) narrows events into VBF / ggH / nocat → **this skill** scores each
category's events → the score (or its bin edges) becomes a stage-3 template axis
(`stats`).

## Reference selection

- `references/ggh-bdt.md` — ggH BDT: submodule trainer, input features, 2016 year
  merge, `BDT_edges.yaml` subcategorization.
- `references/vbf-dnn.md` — VBF DNN: PyTorch/TorchScript trainer, input features,
  k-fold inference, `sigmoid → arctanh(clip)` score transform, bin-edge scan.

Read only the one relevant to the request.

## Review checklist

1. Model/training provenance identified (submodule commit for the BDT; `--model_tag`
   for the DNN) — never assume "the current one" without checking.
2. Stage-2 input-feature list and order match the training config exactly.
3. Fold/year assignment at inference matches training (BDT: 2016 halves merged; DNN:
   event-number-based fold split).
4. Systematic variations are evaluated the way the model actually expects them
   (BDT re-evaluates per variation; DNN defaults to nominal-input approximation —
   confirm that's intended for the systematics in play).
5. Bin edges / subcategory definitions are a real computed artifact, not a placeholder
   or stale copy.
6. Score transform (if any) applied identically wherever the score is consumed
   (categorization, plotting, datacards).

## Reporting categories

- analysis-specific inconsistency (training vs. application mismatch);
- implementation defect;
- optional improvement;
- verification required (e.g. framework/version not pinned).

Never invent training details (architecture, hyperparameters, sample composition) that
aren't in the reference file or the actual training script — check the script before
asserting.

Before treating a finding as new, check `.claude/reports/registry.md` — the VBF DNN's
`jj_eta_region` training-scope question is already tracked there (see
`vbf-dnn.md`'s open item).
