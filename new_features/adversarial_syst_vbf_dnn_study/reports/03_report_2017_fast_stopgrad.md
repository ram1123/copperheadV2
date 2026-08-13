# Phase 3 — 2017 fast-iteration axis, stop-gradient loss

**Run 2026-08-11. Task: `.agent-system/tasks/03-adversarial-syst-vbf-dnn-2017-fast/`.**

This phase answers the question phases 1 and 2 left open. It is the **first
non-confounded result in this study**: it runs the corrected (stop-gradient) loss
to completion *and* it includes the control that separates the adversarial term
from the training procedure.

**Answer: the stop-gradient consistency term does not help. It costs
significance and makes systematic headroom worse, monotonically in lambda.**

---

## Why a 2017 axis

The Run-2 axis cost 2 h 45 m per training, which is why phase 2 closed with its
decisive run stopped at fold 0 of 4. Confining everything to 2017, keeping only
the 4 variations the training is shown, and warm-starting each lambda from the
converged lambda = 0 model brought a full train-and-chain cycle to ~40 min.
**Five trainings and five complete chains ran in ~3 h 10 m**, against 2 h 45 m for
a single Run-2 training.

## Results

2017 VBF, `Total` + `mu_roccor` shape nuisances only, binning re-derived per model.

| run | lambda | pre-fit | stat-only | headroom | vs null | quadZ | quadZ_ok |
|---|---|---|---|---|---|---|---|
| `lam0` (single phase) | 0 | 0.83201 | 0.90639 | 8.94% | -2.64% | 0.9784 | 0.5548 |
| **`lam0WS` (null control)** | **0** | **0.85458** | **0.92726** | **8.50%** | **+0.00%** | 1.0454 | **0.6734** |
| `lam008836SG` | 0.0884 | 0.75326 | 0.82979 | 10.16% | **-11.86%** | 0.8732 | 0.5751 |
| `lam02651SG` | 0.2651 | 0.72454 | 0.83059 | 14.64% | **-15.22%** | 0.8711 | 0.5626 |
| `lam08836SG` | 0.8836 | 0.80319 | 0.95331 | 18.69% | **-6.01%** | 1.6018 | **0.5526** |

Lambda grid = 0.1x / 0.3x / 1.0x of the **measured** parity point 0.8836
(`calibration.json`). Loss: variants A + B with `--adversarial-detach-nominal`,
`--adversarial-high-score-cut 2.0`, `--adversarial-consistency-smoothing none`.

**Headroom rises monotonically:** 8.50% -> 10.16% -> 14.64% -> 18.69%. That is
the exact quantity the term exists to shrink, moving the wrong way at every
lambda over a 10x range in coupling, with no turnover.

## The control that makes this result stand

Every lambda > 0 run is trained in two phases — lambda = 0 to early stopping,
then a second training warm-started from it with a **fresh optimizer at full
learning rate**. That alone can move a converged model, so "lambda > 0 is worse"
had two candidate causes.

`run_phase2_null.sh` runs that second phase with the adversarial term absent:
identical warm start, identical fresh optimizer, identical early stopping.

**It does not degrade — it improves slightly (0.85458 vs 0.83201, +2.71%) and has
the lowest headroom of any run (8.50%).** The retraining procedure is benign, so
the degradation belongs to the adversarial term. Phases 1 and 2 had no such
control; this is the difference between a confounded and an unconfounded result.

It also means the correct baseline for lambda > 0 is `lam0WS`, which makes the
degradations *larger* (-6% to -15%), not smaller.

## The penalty barely moves — and this falsifies phase 1's reading

| run | fold-0 penalty, first -> last |
|---|---|
| lambda = 0.0884 | 0.305958 -> 0.299820 (**-2.0%**) |
| lambda = 0.8836 (parity) | 0.28724 -> 0.28838 (**+0.4%**; folds 1-3 up to +2.8%) |
| undetached lambda = 0.082 (phase 2, Run 2) | **-32.2%** |

At parity weight the penalty **rises**. With a stop-gradient target the network
essentially cannot reduce genuine variation-vs-nominal disagreement; it can only
pay for the attempt out of discrimination.

This confirms D-8 from the other side. The undetached term's 32% penalty
reduction was **not** decorrelation — it was the `BCE(q;p) >= H(p)` entropy floor
being exploited by driving `p -> 0/1`. Close that route and almost none of the
reduction survives. Phase 1's conclusion, "the penalty falls 28%, so the
mechanism works", is now **positively falsified** rather than merely suspect.

**`val_auc_weighted` is blind again:** at lambda = 0.0884 the mean fold AUC is
*higher* than at lambda = 0 (0.944895 vs 0.944608) while pre-fit falls 11.9%.
Never select on AUC for this problem.

## The one apparent turnover is a single bin

Pre-fit falls to 0.72454 at 0.3x parity then climbs to 0.80319 at parity, which
looks like a turnover. It is bin 21 of that run's 2017 h-peak template:

| process | bin-21 yield |
|---|---|
| DYVBF | **0.001819 +/- 0.029106** |
| DY, EWK, TT+ST, VV, VVV | 0.000000 |

The uncertainty is **16x the yield** — a background indistinguishable from zero,
the residue of cancelling NLO weights, `n_eff` = 0.004 — carrying 0.0592 signal
events. `S^2/B` = 1.92, i.e. **75% of that run's entire quadrature Z^2 from one
unsupported bin.**

It inflates stat-only hardest because the Barlow-Beeston parameters are
Gaussian-constrained (`prop_binSR_2017_bin*` in the combine log), so
`--freezeParameters allConstrainedNuisances` freezes them too — "stat-only" has
no MC-statistical protection at all. Strip sub-floor bins and the point falls
from `quadZ` 1.6018 to `quadZ_ok` **0.5526, the lowest of the five**. The
recovery was **88% artefact**, and the degradation is monotone across the whole
grid once corrected.

Same pathology as the 2016postVFP bin-19 artefact found in phase 1.
`collect_results.py` now reports `quadZ_ok` and names the offending bin
automatically.

## Limitations

- **No noise band for this configuration.** The two lambda = 0 runs share an
  objective yet differ 2.71% in pre-fit — a proxy for model + binning scatter.
  The -11.9% and -15.2% points are 4-6x that; **the -6.0% at parity is ~2x and is
  weak alone**. A real `sigma_noise` needs 3 seeds at lambda = 0 (~1 h), not run.
- **Occupancy fails in every run, non-uniformly**: 57-88% of `Z^2` rests on
  sub-floor bins. Absolute numbers are all inflated and the inflation can
  reorder runs. Compare on `quadZ_ok` or headroom, never raw pre-fit.
- **Per-model rebinning is intended** (analyst decision): the analysis is
  sensitive to the DNN, so each model is judged with the binning it would be
  deployed with. Verified distinct (25-27 bins, all pairs differ) and each run's
  Stage-2 axes match its own archived edges exactly.
- **Not comparable to phases 1-2**: different year, binning, and 2 shape
  nuisances instead of 13. The only valid reference is `lam0WS` in the same table.
- Run-2-tuned hyperparameters on 2017 data; eleven dead one-hot year inputs.
  Identical handicap on every point, so the lambda comparison holds; absolute
  2017 sensitivity does not.

## Recommendation

**Stop scanning lambda on this loss.** Go to the `rho(Delta s)` penalty in
`02b_proposal_shift_penalty.md` — minimised *at* agreement, measured along the
binned `arctanh(p)` axis, and structurally unable to request a higher score. On
this axis that is a ~40 min experiment.

Two cheap prerequisites: **3 seeds at lambda = 0** for a real noise band, and
**move the occupancy floor onto the Stage-3 templates** (the scanner's floor is
applied in scan units, which run 2-9x off Stage-2). The second is independent of
the adversarial work and contaminates everything this axis produces.

## Validation plots

Per-variation up/down vs nominal, all five models:

    plots/stage2_variation_validation/2017fast_{lam0,lam0WS,lam008836SG,lam02651SG,lam08836SG}/2017/

16 PDFs per run (one per sample), 26 pages each (13 shape systematics x
`h-peak`/`h-sidebands`), 403 pages per run. Note the x-axis differs per run
(different binnings), and these carry all 13 systematics — the stripper edits
datacards only, not Stage-2 output.
