# Adversarial-Systematics VBF DNN Study

Tests whether adding a systematics-adversarial term to the VBF DNN loss improves the
Run2 pre-fit expected significance, by training the network so that its prediction
under each shape systematic variation matches its nominal prediction.

**Outcome: no improvement, and as of phase 3 that result is no longer
confounded.** Phases 1 and 2 both ran an implementation defect that made the term
optimise *confidence* rather than *agreement*, so neither could be read as
evidence about decorrelation. **Phase 3 (2026-08-11)** ran the corrected
stop-gradient loss to completion on a cheap 2017 axis, with the control that
separates the adversarial term from the training procedure — and the term still
degrades significance and *raises* systematic headroom monotonically in lambda.

> **What this study now supports:** this particular consistency term, as a BCE
> between varied and nominal predictions, does not buy sensitivity here — with or
> without the collapse defect. It does **not** show that score-level
> decorrelation is impossible in this analysis; the redesigned penalty in
> `reports/02b_proposal_shift_penalty.md` has never been implemented or measured.
> Do not quote the phase-1 or phase-2 *numbers* at all — see their amendments.

Documents are numbered by the phase that produced them, chronologically, and the
number matches the task directory under `.agent-system/tasks/`:

| # | phase | task artifacts |
|---|---|---|
| `01_*` | phase 1 — robust baseline | `01-adversarial-syst-robust-vbf-dnn` |
| `02_*`, `02b_*` | phase 2 — loss variants (`02b` is its proposal) | `02-adversarial-syst-vbf-dnn-loss-variants` |
| `03_*` | phase 3 — 2017 fast, stop-gradient | `03-adversarial-syst-vbf-dnn-2017-fast` |
| `04_*` | phase 4 — 2017 no-QvG, three-step | `04-adversarial-syst-vbf-dnn-2017-noqvg` |

| file | what it is |
|---|---|
| `reports/03_report_2017_fast_stopgrad.md` | phase 3: the corrected loss, run to completion with a null control. **The only unconfounded result. Start here.** |
| `reports/04_report_2017_noqvg_threestep.md` | phase 4: QvG inputs removed, three-step loss, lambda x score_cut grid. **The grid cannot be ranked** — seed noise (7.7%) exceeds every effect; a fixed-binning re-read is suggestive only. Plot: `reports/04_grid_summary_noqvg.png`. |
| `reports/01_report_robust_baseline.md` | phase 1: the original loss, lambda sweep. **Carries an amendment** marking its conclusion unsupported. |
| `reports/02_report_loss_variants.md` | phase 2: the analyst's variants A and B, and the score-collapse diagnosis. |
| `reports/02b_proposal_shift_penalty.md` | proposed replacement penalty. **Not implemented, not measured.** Now the recommended next step. |
| `decisions/01_*`, `02_*`, `03_*` | every assumption and decision, flagged `[measured]` vs `[assumption]`, including one retraction |
| `data/` | significance numbers, the acceptance gate, and frozen evidence tables |
| `binning/` | the per-model DNN bin edges used by every phase-3 run |
| `scripts/compare_template_migration.py` | the analysis that made the phase-2 failure visible |

> **Phases 1–3 are a record, not a re-runnable workflow.** After phase 4 superseded
> them, the loss knobs only they used were removed from `train_dnn.py` and
> `preprocess_dnn.py`: `--adversarial-consistency-only` (variant A),
> `--adversarial-normalize` (never used by any phase), and
> `--variation-drop-degenerate` (superseded by dropping QvG from the config, which
> is what `configs/dnn_run2_vbf_noQvG.yaml` does). Their task scripts under
> `.agent-system/tasks/01-*` and `02-*` are preserved verbatim and will now fail on
> those flags. What phase 4 uses — the three-step schedule, variant B's
> `--adversarial-high-score-cut`, and the degenerate-column *detection* and warning —
> is untouched.

## The result

Run2, keep-degenerate inputs, 4 variations. Gate: reference **1.37093**,
`sigma_noise` **0.00881**, acceptance floor +2% = **1.3984**.

| run | pre-fit | stat-only | headroom | vs reference |
|---|---|---|---|---|
| null (lambda=0) | 1.37746 | 1.52449 | 10.67% | +0.48% |
| variants A+B, lambda=0.005 | 1.34935 | 1.50283 | 11.37% | -1.57% |
| variants A+B, lambda=0.082 | 0.82771 | 1.34432 | 62.41% | -39.62% |
| variants A+B, lambda=0.082, **stop-gradient** | — | — | — | **stopped at fold 0/4** |

### Phase 3 — 2017, stop-gradient, 4 variations, per-model binning (unconfounded)

Its own reference; **not comparable to the Run2 table above** (different year,
binning, and 2 shape nuisances instead of 13).

| run | pre-fit | stat-only | headroom | vs null |
|---|---|---|---|---|
| lambda=0, single phase | 0.83201 | 0.90639 | 8.94% | -2.64% |
| **lambda=0, warm-started (NULL CONTROL)** | **0.85458** | **0.92726** | **8.50%** | **+0.00%** |
| lambda=0.0884 | 0.75326 | 0.82979 | 10.16% | -11.86% |
| lambda=0.2651 | 0.72454 | 0.83059 | 14.64% | -15.22% |
| lambda=0.8836 (parity) | 0.80319 | 0.95331 | 18.69% | -6.01% |

Headroom — the quantity the term exists to shrink — rises monotonically:
8.50% -> 10.16% -> 14.64% -> 18.69%. The apparent pre-fit recovery at parity is
**88% attributable to one bin holding 0.0018 +/- 0.029 background events**; strip
sub-floor bins and that point becomes the *worst* of the five.

## Why the measurements are confounded

`BCE(logits_var, p_nominal)` was computed with gradient flowing into **both** arms
(`detach_nominal: false`, the default). Because `BCE(q; p) >= H(p)`, the binary
entropy, its optimum over the pair is `p -> 0` or `p -> 1` -- not `q == p`. The term
is a confidence-inflation pressure, not a decorrelation pressure. Variant B's
high-score cut made that pressure unidirectional *and* self-reinforcing, since the
selection is recomputed from the model's own output each batch.

Consequence in the templates (2018 SR, `data/template_migration_2018.txt`): total
background is conserved, but ~26 events **migrate** from the bulk into the tail. Top
bin background **x3.4** against signal x1.7, so S/B halves, 0.265 -> 0.136.

Full derivation and evidence: `reports/02_report_loss_variants.md` §3, and
`decisions/02_assumptions_loss_variants.md` D-8.

## Reproducing

The study's code lives in the main repo, not here -- this directory holds the
findings. The adversarial term is in `MVA_training/VBF_run3/train_dnn.py`
(`adversarial_penalty`), covered by `tests/test_adversarial_syst.py` and
`tests/test_noop_vs_pre_change.py` (23 tests, including lambda=0 bit-identity
against the pre-change loss):

```bash
pixi run -e default python -m pytest tests/test_adversarial_syst.py \
                                      tests/test_noop_vs_pre_change.py -q
```

Inspect templates from any two or more completed chains:

```bash
pixi run -e default python new_features/adversarial_syst_vbf_dnn_study/scripts/\
compare_template_migration.py \
    --run null=Aug10_2026_advLambdaOFFKeepDeg \
    --run lam0.082=Aug11_2026_advVarAB_lam0082keepDeg --year 2018
```

**The one experiment that would settle this study** (~2h35m; inputs, binning, gate
and baselines are all already in place) is a single training plus chain:

```bash
bash .agent-system/tasks/02-adversarial-syst-vbf-dnn-loss-variants/scripts/\
run_phase1_stopgrad.sh 0.082
```

That is lambda=0.082 with `--adversarial-detach-nominal`, differing from the -39.6%
run by exactly one flag. If it recovers the null, extend the lambda grid upward and
run the full 26-variation training at the winner; if it does not, implement
`reports/02b_proposal_shift_penalty.md`.

## Known limitations

- **Phase 3 has no measured noise band.** Its two lambda=0 runs share an
  objective yet differ 2.71% in pre-fit. The -11.9% and -15.2% points are 4-6x
  that and solid; the -6.0% at parity is ~2x and weak alone. Three seeds at
  lambda=0 (~1 h on the 2017 axis) would fix this and have not been run.
- **Phase 3's occupancy fails in every run, non-uniformly**: 57-88% of the
  quadrature `Z^2` rests on bins below the floor (`B >= 0.5`, `n_eff >= 10`).
  Absolute significances are inflated and the inflation can reorder runs.
  Compare on `quadZ_ok` or headroom, never raw pre-fit. The root cause is
  upstream and independent of the adversarial work: the bin scanner's
  occupancy floor is applied in *scan* units, which run 2-9x off Stage-2.
- **Phase 1 and 2 numbers are unusable.** Every lambda point in both was taken
  through the collapse mode; variants A and B were never tested on their own
  terms there. Phase 3 supersedes them.
- **The redesigned penalty is still unmeasured.** `02b_proposal_shift_penalty.md`
  remains a proposal, and it is what phase 3 recommends trying next.
- **Sweep runs use 4 variations, not the full 26.** The full set was reserved for a
  final run at a winning lambda, which never happened.
- **Neither input set is physically correct.** `keep` asserts QvG becomes -1 under
  every JEC variation; `drop` asserts QvG is unchanged by JEC. A JEC shift changes
  which jet leads in ~20% of events, so both are wrong. The upstream fix is REV-001
  in the phase-1 task, still open.
- Disabling label smoothing inside the consistency term (decision D-1) was argued
  from the code, never measured. Its reasoning is retracted in D-8; the decisive
  test is one lambda run both ways.

## Byproducts that stand independently of the adversarial term

- **2016postVFP binning artefact**, found while building the baseline: bin 19 held
  0.00576 background events from a single DYVBF MC event (`n_eff` = 0.03),
  contributing S/sqrt(B) = 0.79 -- comparable to the entire year -- and inflating
  the reference by ~7.7%. Fixed by rebinning 21 -> 17 bins from Stage-3 templates.
  Closes `ram1123#202`. See `data/rebin_report.json`.
- **QvG degeneracy A/B**: the constant `-1` columns do not cause discrimination loss
  (stat-only 1.45704 keep vs 1.45685 drop) but do suppress systematic benefit
  (headroom 12.01% -> 10.75%). See
  `data/degenerate_report_Run2_advSweep_keepDegenerate_Aug10_2026.json`.
- **`modules/systematics.py`**: single source of truth for shape-variation discovery
  and naming, shared by `run_stage2_vbf.py` and the DNN preprocessing.
