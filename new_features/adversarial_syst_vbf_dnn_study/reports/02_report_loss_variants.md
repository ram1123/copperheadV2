# Final summary — `02-adversarial-syst-vbf-dnn-loss-variants`

**Closed 2026-08-11T20:01Z by analyst instruction, with the decisive run stopped
mid-flight.** Read §5 before using anything here: the study ends with its central
question unanswered, and the strongest claim in it is a *diagnosis*, not a result.

Follow-up to `01-adversarial-syst-robust-vbf-dnn`. Tests the analyst's two loss
variants: **A** (consistency term only — drop the label term and the factor of 2)
and **B** (restrict the term to the high-score region, `arctanh(p) > 2.0`).

---

## 1. Headline

The variant loss was not tested as intended. Both completed runs degrade Run2
significance, and the cause was identified as a **defect in the shared adversarial
implementation, not a property of the analyst's variants**: the consistency term
had an unbounded score-collapse mode that the variants happened to amplify.

The corrected configuration (`--adversarial-detach-nominal`) was launched and
**stopped after 1 of 4 folds**. It produced no templates and no significance.

**The negative result of the predecessor task is therefore reopened, not
confirmed.** All four of its lambda points ran the same defective configuration.

---

## 2. Results

Gate (inherited, `../01-adversarial-syst-robust-vbf-dnn/gate.json`): reference
**1.37093**, `sigma_noise` **0.00881**, acceptance floor +2% = **1.3984**.

Run2, keep-degenerate inputs, 4 variations (`Total_up/down`, `mu_roccor_up/down`):

| run | pre-fit | stat-only | headroom | vs reference |
|---|---|---|---|---|
| null (lambda=0 / switch-off) | 1.37746 | 1.52449 | 10.67% | +0.48% |
| A+B lambda=0.005 | 1.34935 | 1.50283 | 11.37% | **-1.57%** |
| A+B lambda=0.082 | 0.82771 | 1.34432 | **62.41%** | **-39.62%** |
| A+B lambda=0.082, **stop-gradient** | — | — | — | **not measured (stopped)** |

Per-year:

| run | 2018 | 2017 | 2016postVFP | 2016preVFP |
|---|---|---|---|---|
| lambda=0.005 pre-fit | 0.93048 | 0.64486 | 0.46453 | 0.54382 |
| lambda=0.005 headroom | 12.17% | 15.52% | 12.85% | 7.13% |
| lambda=0.082 pre-fit | 0.47931 | 0.44247 | 0.38409 | 0.38870 |
| lambda=0.082 headroom | 93.5% | 47.9% | 24.9% | 38.2% |

Both points fail the analyst's criterion (D-5: pre-fit **up**, stat-only **down**,
headroom below baseline). Both move every component the wrong way. lambda=0.005 is
-0.028 = **-3.2 sigma** on `sigma_noise` — mild, but a real degradation, not noise.

Raw values: `significance_summary.json` (this directory).

---

## 3. What went wrong: the score-collapse mode

Full record in `assumptions.md` D-8. Summary:

`BCE(logits_var, p_nominal)` was computed with **gradient flowing into both arms**
(`detach_nominal: false`, the default, `train_dnn.py:1431-1435`). Minimised over the
pair, its optimum is not "varied == nominal": `BCE(q; p) >= H(p)`, the binary
entropy, which is minimised at `p -> 0` or `p -> 1`. The gradient w.r.t. `p_nominal`
is `-logit(q)`, so the term is a **confidence-inflation** pressure, not a
decorrelation pressure.

Variant B amplified it in two ways, both structural:

1. **It made the pressure unidirectional.** The `p > 0.964` cut selects only events
   with `logit(q) > 0`. Unrestricted, the down-push from low-score events largely
   cancels the up-push in the offset direction, leaving roughly symmetric
   sharpening, which preserves ranking. Restricted, nothing cancels.
2. **It made the selection a ratchet.** The mask is recomputed from the model's own
   output each batch (`train_dnn.py:732`), so an event lifted above the cut *joins*
   the selected set and is lifted further. Membership only grows. No restoring force.

D-1 (disabling label smoothing in the consistency term) removed the only bound: with
smoothing the target is capped at 0.98, giving a finite entropy floor. **D-1's
reasoning is retracted** — it did not free the term to decorrelate, it unbounded a
pre-existing pathology.

### Evidence (2018 SR templates, background per bin)

Total background is identical across runs (3361.0). Events **migrate**:

| bin | null | lambda=0.005 | lambda=0.082 |
|---|---|---|---|
| 0-13 (bulk) | 3337.05 | 3334.94 | **3311.31** |
| 14-16 (tail) | 23.91 | 26.02 | **49.65** |
| 16 (top bin) | 10.73 | 12.14 | **36.12** |

Top bin: background **x3.4**, signal x1.7, so S/B halves (0.265 -> 0.136). Purity
falls in every high-score bin. Quadrature S/sqrt(B) 1.0841 -> 0.9438 (-12.9%),
consistent with the -11.8% stat-only. Pre-fit falls much further (-39.6%) because
the significance now sits in one bloated DYVBF-dominated bin whose JEC templates
move a lot — hence 62% headroom. Migration is one-way and monotone in lambda
(+2.1 events into the tail at 0.005, +25.7 at 0.082).

Because the selection is on **score, not label**, and the network is smooth, the
tail cannot be lifted for signal alone; background just below the cut is swept over.

---

## 4. Consequence for the predecessor task

`01-adversarial-syst-robust-vbf-dnn/final-summary.md` concludes that score-level
decorrelation does not buy significance in this analysis. **That conclusion is not
supported by its evidence.** All four of its lambda points carried
`detach_nominal: false` (verified across every run manifest), so its monotone
degradation is consistent with having measured this collapse rather than
decorrelation.

It also **retro-explains an observation recorded there as an unexplained wrong
prediction**: scores moved *up*, with background moving further. Same runaway,
milder because bidirectional and smoothing-bounded.

That summary should be amended to note the confound. Its other findings stand —
they do not depend on the adversarial term:

- the **2016postVFP binning artefact** (bin 19, 0.00576 background events from a
  single DYVBF event, n_eff = 0.03, inflating the reference ~7.7%), fixed by
  rebinning 21 -> 17 bins from Stage-3 templates. Closes `ram1123#202`.
- the **QvG degeneracy A/B**: stat-only identical keep vs drop (1.45704 vs 1.45685),
  so the constant `-1` columns do not cause discrimination loss, but headroom
  12.01% -> 10.75%, so they do suppress systematic benefit.
- **REV-001** (still open): neither input set is correct. Stage-1 should fill
  `jet1/jet2_btagUParTAK4QvG_<JEC>_up/down` with the QvG of whichever jet leads
  under that variation — measured, a JEC shift changes the leading jet in ~20% of
  events.

---

## 5. What the shutdown leaves unresolved

The stop-gradient run at lambda=0.082 was killed at 20:01Z after **fold 0 of 4**
(24 epochs), 57 minutes into an estimated 2h15m. Tag
`trained_advVarAB_lam0082_keepDegSG_Aug11_2026`. No `manifest.json` was written, so
no Stage-2 scoring, no templates, no significance. It differed from the -39.6% run
by **exactly one flag** — same lambda, seed, inputs, variations.

**Four questions therefore close as unanswered:**

1. **Does the corrected loss recover the null?** Unknown. This was the study's
   central question and the reason the run existed.
2. **Is the predecessor's negative result an artefact or real?** Unresolved in the
   direction that matters. §4 establishes the confound exists; only this run could
   have shown whether removing it changes the outcome.
3. **Do the analyst's variants A and B have merit?** Untested. Every measurement in
   this study was taken through the collapse mode, so nothing here evaluates the
   variants on their own terms — including the two degradations in §2, which should
   **not** be cited as evidence against A or B.
4. **Was D-1 (smoothing off) harmful in itself?** Still only argued, never measured.
   The decisive experiment (one lambda both ways) was declined for cost and remains
   the test to run if D-1 is ever contested.

### Partial evidence from fold 0 — suggestive, not conclusive

| fold 0 | best `val_auc_w` | final nominal loss | penalty drop |
|---|---|---|---|
| null (lambda=0) | 0.947003 | 0.28553 | -12.6% |
| undetached lambda=0.082 | 0.946595 | 0.29481 | -32.2% |
| **stop-grad lambda=0.082** | **0.946972** | **0.28550** | **-8.2%** |

The stop-gradient run tracks the null on both discrimination metrics where the
undetached run had already separated from it, which is what the diagnosis predicts.

**This does not establish that the corrected loss works,** for a reason measured in
the predecessor task: `val_auc_weighted` is nearly blind to this damage — it moved
by 0.0001 while top-bin S/B fell 24%. AUC parity at fold 0 is therefore weak
evidence about templates, and one fold of four cannot be extrapolated to a
4-fold ensemble. **Only the significance chain decides, and it was not run.**

One observation that is informative in its own right: the undetached run reduces the
penalty far more (-32.2%) than the stop-gradient run (-8.2%) at identical lambda.
That is consistent with the undetached term reducing its own loss through the
`H(p)` floor — the collapse cheat — rather than by achieving real agreement. The
predecessor's "penalty falls 28%, so the mechanism works" reading is thus also
suspect: **honest decorrelation may be much harder to achieve than that number
suggested.** Single-fold, single-lambda; treat as a hypothesis.

### Cost to resolve

One training + chain, ~2h35m on the existing inputs. Nothing needs rebuilding: the
inputs, binning, gate and baselines are all in place, and the command is recorded in
`scripts/run_phase1_stopgrad.sh`. The partial output directory is left in place and
must not be reused as a tag.

---

## 6. Recommendation

**Do not treat this study as evidence against adversarial decorrelation.** It is
evidence that the implementation was measuring something else.

1. **Re-run the stop-gradient point** (~2.5 h) before any further design work. It is
   the cheapest decisive experiment available and everything else is conditional
   on it.
2. **If it recovers the null**, extend the lambda grid upward — the corrected term
   should tolerate far more weight than the collapsing one did — then the
   full-26-variation run at the winning lambda.
3. **If it does not**, the remaining suspects are the two structural defects of BCE
   as a consistency loss, and the replacement is specified in
   **`proposal-shift-penalty.md`**: penalise Huber displacement along the binned
   axis `arctanh(p)` with a smooth detached gate and normalisation, so that the
   minimum *is* agreement and the term can never request a higher score. That
   document also sketches the principled endpoint — penalising the soft-binned
   template difference directly.
4. **Amend the predecessor's `final-summary.md`** to record the confound.
5. **REV-001** (Stage-1 QvG under JEC) remains the correct upstream fix and is
   independent of all of the above.

---

## 7. Code and artifacts

Implementation is unchanged from the predecessor task except for the analyst's two
variant flags; no code was modified in response to the results above.

- `MVA_training/VBF_run3/train_dnn.py` — `adversarial_penalty` (variants A/B via
  `--adversarial-consistency-only`, `--adversarial-high-score-cut`),
  `--adversarial-detach-nominal`, `--adversarial-consistency-smoothing`
- `tests/test_adversarial_syst.py` (23 tests) and `tests/test_noop_vs_pre_change.py`
  — all passing, including lambda=0 bit-identity against the pre-change loss
- `assumptions.md` — decisions D-1..D-8, including the D-1 retraction
- `proposal-shift-penalty.md` — proposed replacement penalty (**not implemented,
  not measured**)
- `significance_summary.json` — raw per-year numbers
- `scripts/` — `run_variant_sweep.sh`, `run_phases.sh`, `run_phase1_stopgrad.sh`,
  `check_goal.py`

Standing constraints were observed throughout: no git commit or push, no destructive
filesystem operations, reference stage2/stage3 paths untouched, pixi environments
only, unique dated tags per lambda.
