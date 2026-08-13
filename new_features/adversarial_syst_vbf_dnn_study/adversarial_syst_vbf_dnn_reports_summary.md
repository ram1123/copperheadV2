# In-depth summary: adversarial-systematics VBF DNN loss-variant study

Prepared 2026-08-11 from the reports and supporting artifacts under
`.agent-system/tasks/02-adversarial-syst-vbf-dnn-loss-variants/`.

## Bottom line

The study does **not** determine whether the proposed adversarial-loss Variants A
and B improve VBF DNN significance. The only two completed variant runs used a
defective configuration in which gradients flowed through the nominal prediction
inside the consistency loss. That created a confidence-inflation or
“score-collapse” channel, especially severe under the high-score cut.

The corrected stop-gradient experiment was started but stopped after one of four
folds, before Stage 2, template production, or significance evaluation.
Consequently:

- The measured degradations should not be interpreted as evidence against
  Variants A or B.
- The predecessor study’s negative conclusion is also called into question
  because it used the same undetached formulation.
- The most defensible output is the diagnosis of the loss defect, not a
  physics-performance conclusion.

This is stated explicitly in the task's `final-summary.md`.

## What the study intended to test

The task followed up on `01-adversarial-syst-robust-vbf-dnn`, whose original
adversarial objective degraded Run-2 pre-fit significance monotonically with
increasing lambda.

The proposed variants were:

- **Variant A — consistency only:** retain only
  `lambda × sum BCE(pred_variation, pred_nominal)`, dropping the label-pulling
  term and the factor of two.
- **Variant B — tail restriction:** apply that consistency penalty only where
  `arctanh(p_nominal) > 2`, equivalent to `p_nominal > 0.964`.

The rationale was sensible: the discarded label term fights discrimination,
while most significance comes from the high-score tail, so applying decorrelation
pressure over the full score distribution may waste classifier capacity.

The study inherited rather than recomputed:

- The 17-bin score binning.
- Three rebinned baseline seeds.
- Run-2 reference significance: **1.37093**.
- Seed-noise scale: **0.00881**.
- Required +2% floor: **1.39835**.
- The keep/drop-degenerate augmented datasets.

The intended acceptance standard was demanding: at least three lambda values, a
complete train → Stage 2 → Stage 3 → combine chain, and ultimately a full
26-variation result showing pre-fit significance above 1.3984 and beyond the
seed-noise band.

## Lambda calibration

The lambda values were calibrated from measured penalty magnitudes on 20,000 real
events, rather than copied from the predecessor:

| Configuration | Raw penalty | Events retained | Lambda at nominal-loss parity |
|---|---:|---:|---:|
| Original/default loss | 3.5475 | 100% | 0.0796 |
| Variant A only | 1.1947 | 100% | 0.236 |
| A+B, cut 1.5 | 0.6194 | 16.9% | 0.456 |
| A+B, cut 2.0, no smoothing | 0.3385 | 9.4% | 0.834 |
| A+B, cut 3.0 | 0.1315 | 1.6% | 2.147 |

For keep-degenerate inputs, parity was measured as 0.816 rather than 0.834, only
a 2% difference. The initial keep grid was therefore **0.082, 0.245, and 0.82**,
corresponding approximately to 0.1×, 0.3×, and 1× parity.

After the catastrophic 0.082 result, the high point was dropped and **0.005** was
added to probe the very weak-coupling region. The calibration and decision history
are in `assumptions.md`.

## Completed significance results

Only two A+B points completed the four-fold training and four-variation
significance chain. Both used:

- Keep-degenerate inputs.
- `cut = 2.0`.
- Consistency smoothing disabled.
- `detach_nominal = false`.
- Four variations: `Total_up/down` and `mu_roccor_up/down`.

| Run | Pre-fit | Stat-only | Systematic headroom | Versus reference |
|---|---:|---:|---:|---:|
| Inherited gate reference | 1.37093 | 1.52348 | 11.13% | — |
| Null/switch-off | 1.37746 | 1.52449 | 10.67% | +0.48% |
| A+B, lambda=0.005 | 1.34935 | 1.50283 | 11.37% | -1.57% |
| A+B, lambda=0.082 | 0.82771 | 1.34432 | 62.41% | -39.62% |
| A+B, lambda=0.082, stop-gradient | — | — | — | Not completed |

The lambda=0.005 point is 0.0281 below the null, about **3.2 sigma_noise** relative
to that null. Relative to the formal reference, the goal checker reports
-2.4 sigma_noise.

A nuance in the written conclusion: stat-only significance did decrease at both
lambda values, which is one part of the desired “pre-fit up/stat-only down”
signature. The failure is that pre-fit significance also decreased, and systematic
headroom worsened. Thus neither point satisfies the full conjunctive signature;
“every component moved the wrong way” is somewhat too broad.

Per-year behavior reinforces the failure:

- At lambda=0.005, 2018, 2017, and 2016postVFP pre-fit values declined relative
  to their references; 2016preVFP improved, but not enough to rescue Run 2.
- At lambda=0.082, every year degraded sharply.
- The most extreme headroom was **93.5% in 2018** at lambda=0.082.

The raw values are in `significance_summary.json`.

## Why the loss collapsed

The central defect is mathematically credible and is also consistent with the
observed templates.

The loss was effectively:

```text
BCE(logits_variation, p_nominal)
```

but `p_nominal` remained attached to the computation graph. Therefore the
optimizer could change both:

- The varied prediction, which was supposed to match the nominal target.
- The nominal “target” itself.

For a probability target p, BCE at agreement is not zero; it equals the binary
entropy H(p). That entropy is minimized as p approaches 0 or 1. Consequently, the
loss can reduce itself by making predictions more extreme instead of making the
varied and nominal predictions genuinely stable.

For high-score events, the gradient through the target pushes the nominal score
upward. Variant B selects only `p > 0.964`, removing the corresponding
low-score/downward population that might otherwise partially balance the effect.

The hard selection mask is detached within a forward pass, but it is recomputed
from the evolving model each batch. This can create positive feedback: events
pushed over the threshold are subsequently included in the penalized tail.

Disabling consistency-term label smoothing made this worse. Smoothing had imposed
a finite target cap near 0.98; removing it exposed the unbounded
confidence-inflation direction. Accordingly, the earlier decision arguing that
smoothing should be disabled is explicitly retracted in `assumptions.md` D-8.

## Template evidence for the diagnosis

The lambda=0.082 run did not create more total background; it moved background
into the significance-bearing tail:

| 2018 SR bins | Null background | lambda=0.005 | lambda=0.082 |
|---|---:|---:|---:|
| Bulk, bins 0–13 | 3337.05 | 3334.94 | 3311.31 |
| Tail, bins 14–16 | 23.91 | 26.02 | 49.65 |
| Top bin only | 10.73 | 12.14 | 36.12 |

At lambda=0.082:

- Top-bin background increased by about **3.4×**.
- Top-bin signal increased by only **1.7×**.
- Top-bin S/B fell from **0.265 to 0.136**.
- Tail purity fell in every high-score bin.
- Quadrature S/sqrt(B) fell by 12.9%.
- The resulting concentration in a DYVBF-heavy bin made the JEC templates
  especially damaging, yielding 62.4% systematic headroom.

This is strong evidence that the loss was sharpening and migrating scores rather
than decorrelating them.

## The unfinished corrected experiment

The proposed correction was `--adversarial-detach-nominal`, which turns
`p_nominal` into a fixed target while retaining gradient flow through the varied
branch and shared model weights.

A directly comparable run at lambda=0.082 was launched. It was stopped after fold
0 completed and fold 1 began:

- Fold 0 ended at epoch 24.
- Runtime before shutdown was approximately 57 minutes.
- No final manifest was written.
- No Stage-2 scoring or Stage-3 templates were produced.
- No significance result exists.

Its partial fold-0 diagnostics were encouraging but not decisive:

| Fold-0 run | Best weighted validation AUC | Final nominal loss | Penalty change |
|---|---:|---:|---:|
| Null | 0.947003 | 0.28553 | -12.6% |
| lambda=0.082, undetached | 0.946595 | 0.29481 | -32.2% |
| lambda=0.082, stop-gradient | 0.946972 | 0.28550 | -8.2% |

The stop-gradient run tracked the null much more closely. However, the reports
correctly warn that AUC is nearly blind to high-tail damage: the predecessor saw
tiny AUC movement alongside a roughly 24% top-bin S/B loss. One fold therefore
cannot answer the physics question.

The partial tag `trained_advVarAB_lam0082_keepDegSG_Aug11_2026` must not be reused.
The reports estimate a fresh four-fold training plus significance chain at roughly
**2 hours 35 minutes**.

## Keep/drop-degenerate input issue

The task initially planned drop-degenerate inputs, but this was superseded by an
analyst decision to prioritize consistency with Stage-2 inference.

- **Keep-degenerate:** reproduces Stage-2’s current behavior, where the varied QvG
  inputs become constant -1.
- **Drop-degenerate:** removes those columns during training, but then trains
  against a perturbation Stage 2 does not actually apply.

The report concludes that neither is physically correct. A JEC shift changes
which jet is leading in about **20.4% of events**, so Stage 1 should populate each
varied QvG branch using the QvG of the jet that actually leads under that
variation. This remains the open upstream issue `REV-001`.

The raw logs also show an earlier, separate failure: the first drop-degenerate
attempts at lambda=0.083, 0.25, and 0.83 aborted because batches with no selected
high-score events returned a tensor without a gradient function. The current
source contains an attached-zero fix, but those attempts produced no scientific
results.

## Proposed replacement if stop-gradient BCE still fails

`proposal-shift-penalty.md` proposes replacing BCE with a Huber penalty on
displacement along the actual binned axis:

```text
s = arctanh(p)
penalty = weighted mean of gate(s_nom) × Huber(s_var - s_nom)
```

Key elements are:

- Detached nominal score.
- Smooth, detached gate centered at `s=2.0`.
- Gate width around 0.25.
- Huber transition around 0.25.
- Normalization by selected event weight and number of variations.
- Optional lambda warm-up.

This would make the penalty exactly zero at agreement, prevent it from lowering
its loss through confidence inflation, align the metric with Stage 3’s score axis,
and reduce domination by extreme QvG outliers.

The proposal’s own limitation is important: it still penalizes motion within a
wide histogram bin even when that motion does not change the statistical model.
The suggested long-term endpoint is therefore a differentiable, soft-binned
template-difference loss accumulated with an EMA. None of this has been
implemented or measured.

## Acceptance and artifact status

Against the formal criteria in `task.json`:

- **AC1 not met:** only two complete lambda points, both A+B and keep-degenerate;
  no three-point drop sweep and no Variant-A-only attribution.
- **AC2 met:** lambda calibration was measured and recorded.
- **AC3–AC4 substantially met for completed points:** the reports include
  pre-fit, stat-only, headroom, gate comparisons, and explicit signature
  evaluation.
- **AC5 not triggered:** no point showed the required signature, so no
  full-26-variation final run was warranted.
- **AC6 met:** setting and smoothing decisions are documented, including the
  later retraction.
- **AC7 partially/formally met:** a final summary exists, but the requested study
  itself is incomplete.
- **AC8 not met literally:** keep-degenerate Phase 1 failed, but the required
  drop-degenerate fallback did not complete before the analyst shutdown.

There is also a bookkeeping inconsistency:

- `final-summary.md` says the task was closed.
- `current-state.json` still says `implementing`.
- `task.json` still says `created`.
- `iterations/001/` contains no generator, run, or reviewer reports.
- The claimed passing tests are not accompanied by captured test output in this
  task directory.
- The entire `.agent-system/` directory is currently ignored by `.gitignore`, so
  these artifacts are not tracked despite the canonical policy saying task JSON
  artifacts should be version-controlled.

Scientifically, the correct next action is exactly what the reports recommend:
rerun the lambda=0.082 stop-gradient configuration under a fresh tag, complete all
four folds and the significance chain, and make every further decision conditional
on that result.

## Source artifacts reviewed

- `.agent-system/tasks/02-adversarial-syst-vbf-dnn-loss-variants/task.json`
- `.agent-system/tasks/02-adversarial-syst-vbf-dnn-loss-variants/current-state.json`
- `.agent-system/tasks/02-adversarial-syst-vbf-dnn-loss-variants/assumptions.md`
- `.agent-system/tasks/02-adversarial-syst-vbf-dnn-loss-variants/final-summary.md`
- `.agent-system/tasks/02-adversarial-syst-vbf-dnn-loss-variants/proposal-shift-penalty.md`
- `.agent-system/tasks/02-adversarial-syst-vbf-dnn-loss-variants/significance_summary.json`
- All files under the task's `logs/` and `scripts/` directories.
