# Assumptions and decisions — `02-adversarial-syst-vbf-dnn-loss-variants`

Follow-up to `01-adversarial-syst-robust-vbf-dnn` (closed as a negative result).
Everything marked **[measured]** was checked against real data before any compute
was committed; everything marked **[assumption]** was not.

---

## D-1. Label smoothing is DISABLED in the consistency term — `--adversarial-consistency-smoothing none`

**This is the one substantive setting chosen by the agent rather than specified by
the analyst, and it is a deviation from "use `original_loss` literally".** The user
approved proceeding on the argument rather than spending ~5 h measuring it, and
asked for it to be recorded. This is that record.

### What the function is

`bce_with_logits_loss` (`train_dnn.py`) shrinks its target toward 0.5 before
computing BCE:

```python
y = targets
if label_smoothing > 0.0:
    y = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
```

with `s = 0.04049062762704674`, an **Optuna-tuned hyperparameter** from
`configs/dnn_run2_vbf.yaml` — not introduced by either task.

### Why it is right for the nominal loss and wrong here

On hard labels it is a standard regulariser: BCE on `y in {0,1}` is minimised only
as the logit diverges, so smoothing moves the targets to 0.0202 / 0.9798, caps
confidence at a finite logit (+/-3.88) and improves calibration. Symmetric,
class-agnostic, appropriate.

The consistency term's target is not a label. It is `p_nominal`, the model's own
prediction, already a probability. Smoothing it asks the varied prediction to match
a *deliberately degraded copy* of the nominal one, so **perfect decorrelation is not
the minimum of the term** — the minimum sits about 2% lower, and the residual
gradient always points the same way: down.

**[measured]** on the config's own `s`:

| `p_nominal` | smoothed target | pull | `arctanh(p)` | `arctanh(target)` |
|---|---|---|---|---|
| 0.5000 | 0.5000 | 0 | 0.549 | 0.549 |
| 0.9640 | 0.9452 | -0.019 | 2.000 | 1.785 |
| 0.9900 | 0.9702 | -0.020 | 2.647 | 2.095 |
| 0.9990 | 0.9788 | -0.020 | 3.800 | **2.268** |

The pull is ~0 at mid-score and maximal at the extremes. In score space -- which is
what the histogram is binned in -- an event at `arctanh = 3.80` has its consistency
optimum at 2.27, i.e. dragged from the top bin (edges 1.869-7.354) down several
bins.

### Why this matters specifically for Variant B

Variant B selects `p > tanh(2.0) = 0.9640`, i.e. **only** the rows where the pull is
largest; the bulk, where it is harmless, is excluded by construction. Left on, the
term would consist entirely of tail events being dragged out of the bins that make
the significance -- the constraint would fight the objective rather than regularise
it.

### Honest limits of this argument

- It is a **prediction from the code, not a measurement**. The predecessor task
  made a smoothing-driven prediction once before (`assumptions.md` §1.2 there) and
  it was **wrong**: the observed scores moved *up*, not toward 0.5, so something
  else dominated in the default formula. What differs now is that Variant B
  restricts the term to exactly the region where this bias is largest, so it cannot
  be diluted the same way. That is a reason to expect the argument to bite here and
  not there -- it is not proof.
- The cheap decisive test exists and was declined for cost: run one lambda both
  ways (~5 h). If any result from this task is ever contested, that is the
  experiment to run.

### Scope of the change

`consistency_ls = 0.0` applies to the consistency term **only**. The nominal loss
keeps its Optuna-tuned smoothing untouched, so no regularisation is lost from the
part of the objective that trains the classifier.

---

## D-2. Inputs: drop-degenerate

**[measured]** in the predecessor task: the constant `-1` QvG columns do not cause
the discrimination loss (stat-only identical, 1.45704 vs 1.45685) but do suppress
the systematic benefit (headroom 12.01% -> 10.75%, from above the baseline to below
it). So the variant study uses `Run2_advSweep_dropDegenerate_Aug10_2026`.

## D-3. Lambda grid from a measured calibration, not a guess

**[measured]** on 20k real events with the baseline model (nominal training loss
0.2822 on the same batch):

| configuration | penalty | events kept | lambda at parity |
|---|---|---|---|
| default (predecessor) | 3.5475 | 100% | 0.0796 |
| A only | 1.1947 | 100% | 0.236 |
| A+B cut=1.5 | 0.6194 | 16.9% | 0.456 |
| **A+B cut=2.0, smoothing none** | **0.3385** | **9.4%** | **0.834** |
| A+B cut=3.0 | 0.1315 | 1.6% | 2.147 |

"Parity" = the lambda at which the penalty term equals the nominal loss. Both
variants shrink the penalty, so the *same* lambda applies far less pressure than
before: **~3x weaker for A, ~10x weaker for A+B**. Reusing the predecessor's grid
would have tested essentially nothing -- the predecessor's own failure region began
around 0.4x parity.

Phase 1 grid, as fractions of parity: **lambda = 0.083, 0.25, 0.83** (0.1x, 0.3x,
1.0x).

## D-4. What is reused rather than re-derived

The 17-bin binning, the three baseline seeds, the gate (reference **1.37093**,
`sigma_noise` **0.00881**) and the augmented inputs all carry over from the
predecessor task. ~8 h of compute saved.

**No null run is needed.** The drop and keep input sets differ only in augmented
columns, so the nominal training path is identical and the predecessor's `off`
result (1.37746) is the valid null for both. lambda=0 inertness is covered by unit
tests, and the production-scale bit-identity check (48/48 tensors) was done in the
predecessor.

## D-5. Success criterion (from the analyst)

Pre-fit **up** while stat-only **down**, with systematic headroom below the 11.13%
baseline. "Both up" would be regularisation, not decorrelation, and must not be
credited as such. Acceptance still requires >=2% on Run2 (> 1.3984), exceeding
`sigma_noise` = 0.00881, from a full-26-variation run.

## D-6. Stop rule

If all three Phase-1 points degrade, run variant A alone at 3 lambda to separate
"the tail restriction hurt" from "consistency alone does not help". If that also
fails, stop: three independent formulations of the same idea will have failed, and
the honest conclusion is that score-level decorrelation against these systematics
does not buy significance in this analysis.

---

## D-7. Input set: KEEP first, DROP only as a conditional fallback

**Analyst instruction, 2026-08-11: "I would put first priority on matching
training with stage2 inference."** This supersedes D-2.

Phase 1 therefore uses `Run2_advSweep_keepDegenerate_Aug10_2026` -- the same
constant `-1` QvG columns Stage-2 feeds when it builds the JEC templates.

### Why the instruction is mechanistically right

The consistency term teaches the model *"your prediction under variation V should
equal your prediction under nominal"*. That reduces the **fitted** systematic only
if `V` is the perturbation Stage-2 actually applies. On drop-degenerate inputs the
model would be trained to ignore a QvG shift that never occurs, while remaining
fully sensitive to the `-1` jump that **does** occur in every JEC template -- i.e.
decorrelating against the wrong perturbation.

### What this trades away, and why it is still right

The predecessor's A/B measured **drop** as better: systematic headroom 10.75% vs
keep's 12.01% at lambda=0.03. So this puts the principle above that one
measurement. There is a coherent reason the measurement should not carry over:
under the *original* formula the QvG artefact dominated the penalty and crowded out
the genuine JEC shifts. Variant A+B is ~10x weaker and confined to the top ~9% of
events, so the term may now absorb the QvG jump without wrecking discrimination.
That makes keep the more informative test, not merely the more consistent one.

**[measured]** the switch barely moves the lambda grid: parity 0.816 (keep) vs
0.834 (drop), a 2% difference, because Variant B's tail restriction means the QvG
columns contribute little to the penalty *magnitude* even when present. Phase 1
grid: **lambda = 0.082, 0.245, 0.82**.

### The condition

**If Phase 1 does not pass the final goal** -- Run2 pre-fit >= +2% over 1.37093
*and* the gain larger than `sigma_noise` = 0.00881 -- **the whole sweep is redone
on the drop-degenerate inputs.** Automated in `scripts/run_phases.sh`, decided by
`scripts/check_goal.py`; Phase 2 does not run if Phase 1 succeeds, so success costs
nothing extra.

A drop-degenerate result, if it is ever the one that passes, **must be reported
with the train/inference mismatch caveat attached**: the model would have been
trained against a perturbation Stage-2 does not apply, and the Stage-1 QvG branches
would need fixing before the number could be trusted.

### The real fix, for the record

Neither input set is correct. `keep` asserts "QvG becomes -1 under every JEC
variation" (false); `drop` asserts "QvG is unchanged by JEC" (false for the
**~20%** of events where a JEC shift changes which jet is leading -- measured from
`jet1_eta` changing in 20.4% of events under `Total_up`, and eta is a direction
that an energy-scale shift cannot move). The correct fix is upstream: Stage-1
should fill `jet1/jet2_btagUParTAK4QvG_<JEC>_up/down` with the QvG of whichever jet
leads under that variation. That is REV-001 in the predecessor task, still open.

---

## D-8. The consistency term had a score-collapse mode. Phase 1 is re-planned around `--adversarial-detach-nominal`

**[measured] 2026-08-11 from the first Phase-1 result.** This supersedes the Phase-1
grid in D-7 and partially retracts the reasoning in D-1.

### What was measured

lambda=0.082 (0.1x parity, A+B cut=2.0, smoothing none, keep-degenerate):

| | pre-fit | stat-only | headroom |
|---|---|---|---|
| null (lambda=0) | 1.37746 | 1.52449 | 10.67% |
| **A+B lambda=0.082** | **0.82771** | **1.34432** | **62.4%** |

-40% pre-fit at 0.1x parity, where the predecessor's *original* loss lost only 5.1%
at 0.38x parity. Out of family, so the templates were inspected rather than the
number being reported as a simple degradation.

2018 SR, null -> lambda=0.082: top-bin signal x1.7, **background x3.4**; S/B halves
(0.265 -> 0.136). Purity falls in every high-score bin (bin 14: 0.095 -> 0.057;
bin 15: 0.157 -> 0.073). Quadrature S/sqrt(B) 1.0841 -> 0.9438 (-12.9%), consistent
with the -11.8% stat-only. The pre-fit falls much further because the significance
now sits in one bloated DYVBF-dominated bin whose JEC templates move a lot -- hence
the 62% headroom.

### The mechanism

All runs in both tasks carry `detach_nominal: false`, so
`BCE(logits_var, p_nominal)` propagates gradient into **both** arms. Minimised over
the pair, its optimum is not "varied == nominal": BCE(q; p) >= H(p), the binary
entropy, which is minimised at `p -> 0` or `p -> 1`. The gradient w.r.t. `p_nominal`
is `-logit(q_var)`, so the term is an **entropy-minimisation / confidence-inflation**
pressure, not a decorrelation pressure.

Two brakes were removed at once in this variant:

1. **Label smoothing capped the target at 0.98**, giving a finite entropy floor.
   D-1 switched it off, making the runaway unbounded.
2. **The unrestricted formula also pushed low-score events down**, partially
   cancelling. Variant B's `p > 0.964` cut selects only events with `logit(q) > 0`,
   so the push became **unidirectional** -- up, for signal and background alike.
   Because the selection is on score and not on label, background in the tail is
   inflated hardest, which is exactly what the templates show.

### What this retracts and what it explains

- **D-1 is not vindicated by this.** Turning smoothing off did not "free" the term
  to decorrelate; it removed the bound on a pre-existing pathology. The collapse
  mode exists with smoothing on too, just capped.
- It **retro-explains the predecessor's unexplained observation** (recorded there as
  a wrong agent prediction): scores moved *up*, with background moving further.
  Same runaway, milder because bidirectional and smoothing-bounded. The predecessor's
  monotone degradation was therefore never a test of decorrelation -- it was this.

### The correction

`--adversarial-detach-nominal` stop-gradients `p_nominal`, making it a target rather
than a quantity under optimisation. This is the standard consistency-regularisation
formulation and is faithful to the task formula, in which `pred_nominal` appears as
the *argument* of `original_loss`. Gradient still reaches the shared weights through
the variation branch, which is the intended channel.

**Analyst decision, 2026-08-11:** replace the queued lambda=0.245 with a
stop-gradient run at lambda=0.082 (directly comparable to the measured point), and
re-plan Phase 1 around `detach_nominal=True` on the keep-degenerate inputs. The
undetached runs are retained as the documented negative control.

Parity calibration is unchanged: detaching alters gradients, not the penalty's
value, so the 0.1x/0.3x grid points (0.082 / 0.245) still mean what D-7 says.

Tags for the corrected configuration use the `keepDegSG` set tag (SG = stop-grad).
The in-flight lambda=0.005 undetached run is allowed to finish; it measures how the
collapse scales down and is a cheap confirmation of the mechanism.
