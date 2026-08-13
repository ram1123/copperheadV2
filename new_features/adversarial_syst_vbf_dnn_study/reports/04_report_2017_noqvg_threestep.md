# 04-adversarial-syst-vbf-dnn-2017-noqvg — final summary

**2017 VBF DNN, QvG inputs removed, three-step systematics-aware loss, scanned over
lambda x score_cut.** Run 2026-08-12/13 on the 2017 fast axis. 25 full
Stage-2/3/combine evaluations, 40 trainings.

---

## 1. The headline

**The scan as specified cannot be ranked, because the measurement is less precise
than the effects being measured.** Repeat runs that differ only by the training
seed spread `sigma(pre-fit) = 0.072`, which is **7.7%** of the mean — while the
entire 17-point lambda x score_cut grid spans a range of 0.190. Four runs at one
identical setting cover **74% of the whole grid's range**. Not one grid cell
differs from the null control by 2 sigma.

The dominant free variable is not lambda and not score_cut: it is the **per-model
re-derived binning**. When the binning is instead held fixed, a coherent and much
smaller signal appears:

| model, all on one identical 12-bin binning | pre-fit | vs null | syst_hr |
|---|---|---|---|
| null (no penalty) | 0.74976 | — | 0.42% |
| lambda = 0.005 | 0.73770 | −1.61% | 0.71% |
| lambda = 1.0 | 0.78324 | **+4.47%** | 0.44% |
| lambda = 5.0 | 0.78184 | **+4.28%** | 0.52% |

A negligible penalty behaves like no penalty; two penalties differing 5x in weight
land within 0.19% of each other, ~4.4% above the null. That is the shape of a real
but **saturating** effect — consistent with the training diagnostics, where the
penalty falls only ~1.3% over training even at lambda = 1.0 because it sits on the
`BCE(q;p) >= H(p)` entropy floor it cannot cross.

**This is suggestive, not established.** It rests on one binning (itself
artefact-heavy), n = 1 per point, and no sigma has been measured for the
fixed-binning configuration.

---

## 2. What was asked, and what was delivered

| Request | Status |
|---|---|
| Remove QvG from DNN input, training **and** Stage-2 evaluation | Done. `configs/dnn_run2_vbf_noQvG.yaml`, 34 features. Asserted on the `training_features.pkl` Stage-2 actually loads, in preprocessing **and** again in every chain. |
| Three-step loss (warm-up / +variations-vs-label / +consistency-vs-nominal), each to early stopping | Done. `--adversarial-label-only`, `--adversarial-mask-label-term`; 9 new tests. Equation in `latex2copy.txt`. |
| Scan lambda in {0, 1.0, 0.5, 0.01, 0.005} | Done, 16 points + reference. lambda = 5.0 added on request. |
| Scan score_cut in {all True, atanh > 0.6, 1.0, 2.0} | Done. `all True` implemented as *no cut*, not a sentinel. |
| Inspect nominal vs Total/mu_roccor up/down histograms, with plots; say whether consistency improved | Done — see §5. Answer: **no, not once the binning is controlled.** |
| `plot_stage2_variation_pdfs.py` for completed parameters | Done, 25 directories. |
| Add total_background / total_signal plots + sample list txt | Done; membership read from `configs/samples/samples.yaml`, `metadata.txt` per year dir. |
| lambda = 5.0 / cut = all follow-up | Done. |
| Seed replicas at lambda = 0.5 / cut = all | Done, 3 extra seeds. |
| Fixed-binning cross-evaluations | Done, 4 (on request). |

---

## 3. The measurement-resolution problem (the central result)

### 3.1 Seed replicas

Four runs at lambda = 0.5, score_cut = all True, differing **only** in `--seed`
on steps 2 and 3:

| seed | pre-fit | syst_hr | n_bins | tv_bkg |
|---|---|---|---|---|
| default | 0.86174 | +1.22% | 22 | 0.01007 |
| 31337 | 0.88392 | +1.80% | 28 | 0.00800 |
| 20260812 | 0.99134 | +31.13% | 26 | 0.01177 |
| 777 | 1.00145 | +29.68% | 25 | 0.01080 |

`sigma(pre-fit) = 0.0720` (7.7% of mean), `sigma(syst_hr) = 16.7` points.

The distribution looks **bimodal, not Gaussian**: two seeds at pre-fit ~0.86–0.88
with syst_hr ~1.2–1.8%, two at ~0.99–1.00 with syst_hr ~30%. That is a discrete
outcome, consistent with the bin scanner either creating or not creating a
particular near-empty high-score bin which then dominates both quantities. With
n = 4 the sigma is itself poorly determined and the 2-sigma test below is crude —
but the qualitative conclusion does not depend on it.

### 3.2 Why this differs from the earlier stability measurement

The predecessor task (`01-adversarial-syst-robust-vbf-dnn`) measured seed-to-seed
noise of **0.643%** after its occupancy fix (1.687% before; `decisions.md` C-9
records 2.25% at one point, which tripped a 2% stop condition). That measurement
held the binning **fixed** and occupancy-clean.

Every run here re-derives its own binning, and the occupancy guard reports
**38–66% of quadrature Z^2 sitting on sub-floor bins in all 25 runs, including
both baselines**. The earlier figure is correct for the configuration it was
measured in and does not transfer to this one. The 12x difference *is* the
per-run rebinning.

### 3.3 Consequence

Judged against 2 sigma of the seed-only spread, **no grid cell differs from the
null control** on pre-fit or syst_hr (all |n_sigma| <= 1.4). The only points
clearing 2 sigma are two of the seed replicas themselves.

The raw grid also has no ordering in lambda. Along `cut = all True`, versus the
null: **+10.69% (lambda 5.0), −3.15% (1.0), +6.55% (0.5), −6.99% (0.01), +3.04%
(0.005)** — an 18-point spread across a 1000x range of penalty weight, with best
and worst at neither extreme.

---

## 4. The fixed-binning study

Cheap and decisive: Stage-2 reads the **untagged** compacted Stage-1 parquets
(`run_stage2_vbf.py:828-829`) and evaluates the DNN itself, so re-evaluating a
model on someone else's edges needs no `compact` and no `scan_bins` — ~4 minutes
(`scripts/chain_fixed_binning.sh`).

| model | binning | pre-fit | syst_hr | tv_bkg |
|---|---|---|---|---|
| lambda 1.0 | own (12 bins) | 0.78324 | 0.44% | 0.0047 |
| lambda 5.0 | lambda 1.0's 12 bins | 0.78184 | 0.52% | 0.0068 |
| lambda 0.005 | lambda 1.0's 12 bins | 0.73770 | 0.71% | 0.0058 |
| null | lambda 1.0's 12 bins | 0.74976 | 0.42% | 0.0053 |
| null | own (26 bins) | 0.80874 | 1.42% | 0.0102 |
| null | lambda 5.0's 23 bins | 0.82337 | 1.62% | 0.0150 |
| lambda 5.0 | own (23 bins) | 0.89517 | 0.51% | 0.0121 |

**What the headline advantages actually were:**

* **lambda = 1.0's syst_hr of 0.44% vs the null's 1.42% was the binning, not the
  loss.** The null model — never shown the penalty — gets **0.42%** on the same
  12 edges. The entire apparent 3x reduction is a property of those bin edges.
* **lambda = 5.0's +10.69% pre-fit was mostly binning too.** Granting the null the
  same 23 edges is worth +1.81% on its own; and on shared 12-bin edges lambda 5.0
  and lambda 1.0 are indistinguishable (0.78184 vs 0.78324), where on their own
  bins they looked 13.8 points apart.
* **lambda = 5.0's syst_hr of 0.51% is *not* a binning artefact** — the null gets
  1.62% on lambda 5.0's edges. So that one is a model property.

---

## 5. Did nominal-vs-variation consistency improve?

**No — not once the binning is controlled.** This was the analyst's explicit
question and the answer changed as the controls improved.

Measured on the Stage-3 templates the fit consumes, separating the pure yield
shift from the shape redistribution (`scripts/variation_shape_consistency.py`):

* **`dnorm` is untouchable.** Background `Total` up/down is **+0.2323 / −0.1795**
  in *every* run to four decimals, baselines included. Roughly a quarter of the
  Total variation is pure yield migration as jets cross the VBF selection; no
  score-space penalty can address it. This caps what the method could ever win.
* **`tv_sig` is inert.** 0.0193–0.0227 across all 25 runs. The loss never moves
  the signal templates.
* **`tv_bkg` moves, but with the binning, not the loss.** The null model rebinned
  from 26 to 12 bins goes 0.0102 -> 0.0053 *by itself*; lambda = 1.0's 0.0047 is
  barely different. On common 12-bin edges the ordering is lambda 1.0 (0.0047),
  null (0.0053), lambda 0.005 (0.0058), lambda 5.0 (0.0068) — the **strongest**
  penalty has the **worst** background agreement.
* **tv_bkg and syst_hr are uncorrelated** across the grid: Pearson +0.26,
  Spearman +0.12 (n = 11). tv_bkg vs n_bins is the stronger relation (+0.51).

Plots: `shape/shape_<postfix>.png` (per-run, 4 panels) and the full
`plots/stage2_variation_validation/<postfix>/2017/` PDF sets, including
`total_background` / `total_signal`.

---

## 6. A metric defect found and fixed

`produce_significance.sh` computes its "stat only" leg with
`--freezeParameters allConstrainedNuisances`, which freezes the **autoMCStats
Barlow-Beeston parameters along with the shape nuisances**. The headroom built
from it is therefore the cost of the systematics **and** of the finite MC
statistics together — and with the binning re-derived per model, the MC-stat part
swings for reasons unrelated to the loss.

`scripts/syst_only_significance.sh` adds the missing leg: freeze **only** `Total`
and `mu_roccor2017`, leave autoMCStats floating.

| | legacy `tot_hr` | true `syst_hr` |
|---|---|---|
| step1 | 10.73% | **1.85%** |
| null | 11.10% | **1.42%** |
| lambda 0.5 / cut 0.6 | **52.59%** | **0.85%** |
| lambda 0.5 / cut 2.0 | **59.74%** | **4.82%** |
| lambda 5.0 / cut all | **63.99%** | **0.51%** |

**The systematics this study targets cost ~1.4–1.9% of sensitivity, not the
10–11% the legacy column implies.** That is the size of the prize, and it bounds
what any version of this loss can win. The 52.6% and 59.7% "headroom" outliers
were reading MC statistics; both reproduce exactly on re-running combine, so they
are deterministic properties of those templates, not fit instabilities.

**This affects the predecessor task's headline.** Its "headroom rises 8.50% ->
18.69% monotonically in lambda" used this same contaminated metric and deserves
re-examination on the `syst_hr` basis.

---

## 7. Corrections made during this task

Recorded because each one changed a conclusion:

1. **"The spread is all noise"** — asserted before the replicas that test it had
   run. Withdrawn.
2. **"Too large to be seed noise"** — the correction after the analyst pushed
   back, leaning on the 0.643% prior. Also wrong: that prior was measured with
   fixed occupancy-clean binning and does not transfer. The replicas gave 7.7%.
3. **"tv_bkg and syst_hr broadly track each other"** — claimed on 8 points;
   collapsed to +0.26 Pearson at 11.
4. **"The tv_bkg halving is real because tv_sig didn't move"** — the stated
   control was invalid. The null model shows the same asymmetry under the same
   rebinning: coarsening compresses the steeply-falling background far more than
   the signal.
5. **"A latent bug in the year lookup"** — described as if it were in the
   analysis code. It was a defect *introduced* in the new plotter by
   reimplementing sample resolution instead of using `modules/sample_config.py`,
   which normalises the keys correctly. Stage-2 and Stage-3 were never affected.
   Fixed by deleting the duplicate and delegating to the repo's resolver.

---

## 8. Repository changes

**Modified**
* `MVA_training/VBF_run3/train_dnn.py` — `--adversarial-label-only`,
  `--adversarial-mask-label-term`, `AdversarialConfig.{label_only,mask_label_term}`
  with a mutual-exclusion guard; manifest records both. The pre-existing formula
  is bit-for-bit unchanged when the new flags are off (pinned by test).
* `plotter/plot_stage2_variation_pdfs.py` — `total_background` / `total_signal`
  summed pages; membership from `configs/samples/samples.yaml` via
  `modules.sample_config`; `metadata.txt` per year directory; ratio guides at
  0.75/1.25; shared `folded_arrays()` helper.
* `tests/test_adversarial_syst.py` — 9 new tests (58 pass).

**Added**
* `configs/dnn_run2_vbf_noQvG.yaml`
* `latex2copy.txt` — the loss as implemented, with the conventions needed to
  reproduce it.
* Task scripts: `run_noqvg_grid.sh`, `chain_2017.sh`, `chain_fixed_binning.sh`,
  `run_lam5_extra.sh`, `run_seed_replicas.sh`, `syst_only_significance.sh`,
  `variation_shape_consistency.py`, `seed_spread.py`, `collect_results.py`,
  `plot_grid_summary.py`.

**Not committed** — no `git commit`/`push` was run, per the task constraints.

---

## 9. Incidental findings worth someone's attention

1. **Five background processes in `samples.yaml` are absent from Stage-2 for
   2017**: `st_t_top`, `st_t_antitop`, `st_tw_top`, `st_tw_antitop`, `zz`. The
   background total is 14 of 19 config entries, consistently — Stage-3 agrees with
   the Stage-2 sum to 2.4e-7 — so the datacard's `TT+ST` is TT only and `VV`
   excludes ZZ. Affects no comparison here (all runs are missing the same five)
   but it is upstream of everything.
2. **`scan_bins_for_dnn.py` reports a stale `best_nbins_from_scan`.** The
   lambda = 1.0 file says `best_nbins_from_scan: 18` and "Best binning with 18
   bins" while listing 12. Post-scan merging under `min_background_per_bin`
   reduces the count without updating the reported figure.
3. **Stage-2 already restricts its shape variations to `Total` + `mu_roccor`**
   (`run_stage2_vbf.py:451-462`, an uncommitted working-tree change predating this
   task), so `strip_datacard_variations.py` is now a no-op safety net rather than
   load-bearing.
4. **The bin scanner's own reports show the artefact plainly.** The lambda = 1.0
   edges put `Z = 0.717` of a total `0.795` into one bin holding `B = 0.062`
   from 123 raw MC events.

---

## 10. Recommendation

**Do not run another lambda scan on this axis until the binning is fixed.** The
per-model re-derivation injects ~7.7% scatter, which is several times any effect
seen. Concretely, in order:

1. **Derive one occupancy-clean binning from the Stage-3 templates** and freeze
   it for all comparisons — the approach that took the predecessor's seed noise
   from 1.687% to 0.643%. `scripts/rebin_from_templates.py` from
   `01-adversarial-syst-robust-vbf-dnn` already does this.
2. **Re-measure sigma on that fixed binning** (3 seeds, ~1 h). Without it the
   +4.4% below has no error bar.
3. **Complete the fixed-binning lambda scan** — lambda = 0.01 and 0.5 are the
   missing points, ~4 min each — and see whether the step from ~0 to +4.4% is
   monotone or a two-point coincidence.
4. Only then consider the `rho(Delta s)` shift penalty from
   `proposal-shift-penalty.md`, still unimplemented.

**On the physics:** the prize is ~1.4–1.9% of sensitivity, a quarter of the Total
variation is yield migration the score cannot touch, and the consistency term sits
on an entropy floor it cannot cross. Even the optimistic reading of the
fixed-binning result does not come from improved template agreement — tv_bkg does
not order with lambda. If the +4.4% survives steps 1–3, its mechanism is still
unexplained and should be understood before the loss is adopted.

**Removing QvG is defensible on its own**, independent of the loss: it costs
0.0029 of mean val AUC (0.94461 -> 0.94176) and ~3.6% of pre-fit at lambda = 0,
while eliminating a feature whose JEC-varied branches are a `-1` sentinel for
>99.4% of events — i.e. a missing branch that Stage-2 was feeding its own DNN at
inference. That is a correctness argument, not a sensitivity one, and it should be
decided as such.

---

## 11. Artifacts

```
.agent-system/tasks/04-adversarial-syst-vbf-dnn-2017-noqvg/
  final-summary.md            this file
  assumptions.md              12 decisions taken to make the request runnable
  results.json                every run, every metric
  occupancy_<postfix>.json    per-bin S/B/n_eff, 25 runs
  shape/shape_<postfix>.{json,png}   nominal-vs-variation template metrics
  binning/dnn_binning_<postfix>.yaml edges each run used (+ _ORIGINAL)
  plots/stage2_variation_validation/<postfix>/2017/
        16 per-sample PDFs + total_background + total_signal + metadata.txt
  logs/                       driver, per-chain, per-training
  scripts/                    the rig (see §8)
```

25 evaluated runs: 16 grid + step1 + null + lambda 5.0 + 3 seed replicas + 4
fixed-binning cross-evaluations.
