# Final summary — `01-adversarial-syst-robust-vbf-dnn`

**Outcome: the objective was not achieved. The implementation is correct and
verified; the method does not work.**

Adding a systematics-consistency term to the VBF DNN loss degrades the Run2 pre-fit
expected significance monotonically in lambda. The optimum is lambda = 0, i.e. no
adversarial term. This is a physics result, not a defect: the penalty it optimises
falls exactly as designed.

Two by-products are worth more than the headline result — a real artefact inflating
the analysis reference by ~7.7% was found and fixed, and an upstream Stage-1 defect
was identified that affects every JEC template in the analysis.

---

```yaml
status: closed (negative result, human-directed)
task_dir: /work/users/yun79/sideHustle2/copperheadV2/.agent-system/tasks/01-adversarial-syst-robust-vbf-dnn
iterations_completed: 2

files_modified:
  - MVA_training/VBF_run3/train_dnn.py          # adversarial loss + flags + --seed
  - MVA_training/VBF_run3/preprocess_dnn.py     # --include-systematic-variations, --variation-set,
                                                # --variation-drop-degenerate, --files-per-chunk
  - run_stage2_vbf.py                           # re-export the shared systematics helpers
  - configs/MVA/VBF/dnn_binning.yaml            # 21 -> 17 bins (occupancy fix, see "Binning" below)
files_created:
  - modules/systematics.py                      # single source of truth for variation discovery
  - tests/test_adversarial_syst.py              # 17 tests
  - tests/test_noop_vs_pre_change.py            # bit-identity vs git HEAD
  - tests/conftest.py
  - .agent-system/tasks/01-adversarial-syst-robust-vbf-dnn/  # task artifacts, scripts, reports

variation_set_sweep: [Total_down, Total_up, mu_roccor_down, mu_roccor_up]
n_variations_sweep: 4
variation_set_full: 26 canonical per-event slots (13 sources x up/down; the 5 year-decorrelated
                    JEC sources collapse onto shared "<Source>_yearDecor_<dir>" slots)
n_variations_full: 26
augmented_data_dir_sweep: dnn/trained_models/Run2_advSweep_keepDegenerate_Aug10_2026/2018-2017-2016postVFP-2016preVFP_h-peak_vbf   # 3.4 GB, 83 cols
augmented_data_dir_sweep_drop: dnn/trained_models/Run2_advSweep_dropDegenerate_Aug10_2026/2018-2017-2016postVFP-2016preVFP_h-peak_vbf  # 3.5 GB
augmented_data_dir_full: dnn/trained_models/Run2_advFull_keepDegenerate_Aug10_2026/2018-2017-2016postVFP-2016preVFP_h-peak_vbf     # 17 GB, 413 cols -- BUILT BUT UNUSED

baseline_seed_runs:            # rebinned 17-bin binning
  - seed: 12345 (config default);  significance_Run2: 1.37746
  - seed: 20260810;                significance_Run2: 1.36091
  - seed: 777001;                  significance_Run2: 1.37443
sigma_noise: 0.00881168         # 0.643% of the rebinned reference
baseline_reproduces_reference: true
  # On the ORIGINAL binning, seed 12345 reproduced the section-2 reference EXACTLY
  # (Run2 1.52839, all years to 6 s.f.). That reference was then superseded -- see "Binning".

reference_significances_original: 2018 1.05541 / 2017 0.728025 / 2016postVFP 0.628278 /
                                  2016preVFP 0.503932 / Run2 1.52839   # 21-bin, INFLATED
statonly_ceiling_Run2_original: 1.6903
reference_significances_rebinned: Run2 1.37093 (mean of 3 seeds)       # 17-bin, the live gate
statonly_ceiling_Run2_rebinned: 1.52348

lambda_values_tried: [off, 0, 0.003, 0.03, 0.3]   (+ 0.03 on drop-degenerate inputs)
selected_lambda: none — the optimum is lambda = 0
lambda_transfer_treatment: n/a — the final phase was not run
gate_outcome: not_met
per_epoch_slowdown_sweep: 2.16x (8113 s vs 3753 s at 4 variations, batch 1024, 1.54M rows)
per_epoch_slowdown_full: not measured (final phase not run)
updown_plot_paths: not produced — no lambda was accepted, so section 6.2's
                   plot_stage2_variation_pdfs.py step had no accepted candidate to inspect
blocking_findings_resolved: REV-003, REV-004, REV-006, REV-008, REV-009, REV-011
known_issues: REV-001 (Stage-1 QvG defect, upstream), REV-002 (unweighted training loss),
              REV-007 (selection metric blind to the failure mode), REV-010 (the negative result)
```

---

## 1. The result

Gate (rebinned): reference **1.37093**, `sigma_noise` **0.00881 (0.643%)**,
+2% floor **1.3984**, +5% target **1.4395**.

| lambda | Run2 pre-fit | vs gate | stat-only | syst headroom | sigma |
|---|---|---|---|---|---|
| baseline (3 seeds) | 1.37093 | — | 1.52348 | 11.13% | — |
| off | 1.37746 | +0.48% | 1.52449 | 10.67% | +0.74 |
| 0 | 1.37746 | +0.48% | 1.52449 | 10.67% | +0.74 |
| 0.003 | 1.35899 | -0.87% | 1.51377 | 11.39% | -1.36 |
| 0.03 | 1.30085 | -5.11% | 1.45704 | 12.01% | -7.95 |
| 0.3 | 1.10780 | -19.19% | 1.24046 | 11.98% | -29.9 |
| 0.03 drop-degenerate | 1.31541 | -4.05% | 1.45685 | 10.75% | -6.30 |

Per-year values for every run are in `significance_summary.json`; regenerate the
table with `scripts/summarize_runs.py`.

**Neither the +5% target nor the human-lowered +2% floor is approached from any
direction.** Section 6.3's cap-time fallback is unavailable: it requires a genuine
interior optimum, and the best point here is the lambda -> 0 edge.

## 2. Why it fails

**The mechanism runs.** The consistency penalty at the selected epoch falls
monotonically — 3.625 (lambda=0) -> 3.529 -> 3.232 -> 2.621, a 28% reduction.
Variation predictions genuinely converge on nominal.

**Decorrelation does not buy robustness here.** The systematic headroom
`(stat_only - prefit)/prefit` *rises* from 11.13% to ~12% and stays there.

**The damage is in the tail, and the selection metric cannot see it.** At
lambda=0.03 the mean validation AUC moves by 0.0001 while S/B in the top three
score bins falls 24% (0.1786 -> 0.1352): signal and background both migrate to
higher scores, background further (+63% vs +23%). Early stopping monitors
`val_auc_weighted` per decision H-4, so checkpoints are chosen on a quantity blind
to the failure.

**Controlled A/B on the degenerate QvG columns** (input sets differing in exactly
those four columns): the artefact is **not** the cause of the discrimination loss
— stat-only is identical, 1.45704 vs 1.45685 — but it **was** suppressing the
systematic benefit, headroom 12.01% -> 10.75%, from above baseline to below it
(1.9 sigma; baseline headroom spread +/-0.67 points).

**The exchange rate is the obstacle.** Reaching +2% at unchanged stat-only needs
the headroom cut from 11.13% to ~8.5%, a 24% relative reduction. The best observed
trade bought 3.4% relative headroom for a 4.4% stat-only loss — about an order of
magnitude short.

**Untested.** The sweep decorrelates against `Total` and `mu_roccor`, 2 of the 13
shape nuisances in the datacards, while paying discrimination cost against all of
them. The full-26 phase would target every nuisance and could have a materially
better exchange rate — section 6.3's "the reduced sweep set is not a valid proxy"
case. Not run by human decision; the inputs exist and are verified.

## 3. Binning: a real analysis defect found and fixed

The section-2 reference was **inflated ~7.7%** by a single bin.

2016postVFP bin 19 held **0.00576 background events** — all of it one `DYVBF` MC
event, `n_eff` = 0.03 — against 0.06 signal, contributing S/sqrt(B) = 0.79,
comparable to the entire year's significance. The active binning was hand-edited
and had bypassed the occupancy guards in `scan_bins_for_dnn.py`. This is the known
open issue in repo HEAD `a46bae7`.

Fixed by merging old bins 16-20 (21 -> 17 bins), derived from the **Stage-3
templates the fit actually consumes** rather than from `scan_bins_for_dnn.py`,
whose own TODO documents its background as 2-9x off from Stage-2.

| | value |
|---|---|
| artefact removal | **-7.71%** |
| genuine resolution cost | **-0.27%** |
| seed-to-seed noise | **1.687% -> 0.643%** |

Old config preserved at `dnn_binning_PREVIOUS_handEdited21bins.yaml`; derivation in
`rebin_report.json` and `scripts/rebin_from_templates.py`.

**This is independent of the adversarial study and changes the analysis reference.**

## 4. Upstream defect: constant QvG JEC branches

All **48** JEC-varied `jet1/jet2_btagUParTAK4QvG_<source>_<dir>` columns are a
constant `-1` sentinel, while nominal spans the full discriminant range (3662
distinct values in one 2018 file). Stage-2 resolves the same columns at inference,
so **every JEC template in the analysis is built with both jets' QvG pinned** —
plausibly inflating the JEC shape systematic. Detection is now automatic in
`preprocess_dnn.py` (modal-share test) and reported in the manifest.

## 5. Verification (what can be relied on)

- **Switch-off is bitwise identical to the pre-change code at production scale**:
  all 8 checkpoints (4 folds x best/last) of baseline seed 1 match the reference
  model trained 2026-07-20 at commit `115c8db`, tensor for tensor.
- **lambda = 0 is bitwise identical to switch-off**: 48/48 tensors, and the full
  chain returns the same Run2 value. Meaningful, not short-circuited — the manifest
  shows 4 variations discovered and the penalty computed each epoch (4.134 ->
  3.614); it multiplies by zero. This validates running the variation forwards in
  `eval()` mode (no dropout RNG consumption, no BatchNorm running-stat pollution).
- **Variation discovery is the same object Stage-2 uses** (identity assertion),
  cross-checked per (variation, feature) on real schemas with zero mismatches.
- **Varied features use the nominal scalers**: nominal at weighted mean 6.6e-9 /
  std 1.00000, varied at means up to 1.20 and stds 0.00-1.36.
- **17 tests pass**, including two regression tests for the degeneracy detector.
- **Nothing pre-existing was overwritten** (verified by mtime on both reference
  trees); no commit, no push, no destructive operation.

## 6. Recommended next steps

1. **Take the binning fix to the analysis** (§3). It is independent of this study,
   closes `ram1123#202`, and changes the published reference.
2. **Fix the Stage-1 QvG variation branches** (§4). Affects Stage-2 inference
   generally, not just this task.
3. **If the method is revisited, in this order:**
   a. change the model-selection metric — AUC cannot see the failure (REV-007);
   b. reconsider the unweighted training loss (REV-002) — the failure is
      tail-concentrated and weights are what distinguish tail events;
   c. only then run the full-26 phase, on **drop-degenerate** inputs, at
      lambda <= 0.003 (26 variations give ~6.5x the penalty at equal lambda).
4. **Do not re-run the sweep phase as specified.** Four variations covering 2 of 13
   nuisances is not a useful proxy, and this campaign cost ~20 h of GPU and cluster
   time to establish that.

## 7. Commands

Every command, exit code and duration is in `iterations/002/run-report.json`
(17 recorded commands: 3 preprocessing passes, 3 baseline trainings, 6 baseline
chains, 5 sweep points, 1 A/B, 1 blocked final phase). Drivers are under
`scripts/`; all are re-runnable and refuse to reuse a tag.

---

## AMENDMENT 2026-08-11: this task's negative result is CONFOUNDED

Added by the follow-up task `02-adversarial-syst-vbf-dnn-loss-variants` (see its
`final-summary.md` §3-4 and `assumptions.md` D-8).

All four lambda points above ran with `detach_nominal: false` (verified across every
run manifest). In that configuration `BCE(logits_var, p_nominal)` propagates gradient
into **both** arms, and since `BCE(q; p) >= H(p)` its optimum over the pair is
`p -> 0` or `p -> 1`, not `q == p`. The term is therefore a confidence-inflation
pressure, not a decorrelation pressure.

**The conclusion "score-level decorrelation does not buy significance in this
analysis" is not supported by this evidence.** The monotone degradation reported
above is consistent with having measured that artefact instead. The corrected
configuration (`--adversarial-detach-nominal`) has not been run to a significance
number.

This also explains the observation recorded above as an unexplained wrong
prediction: scores moved *up*, with background moving further. That is the runaway,
in its milder bidirectional form.

Additionally, the reading "the penalty falls 28%, so the mechanism works" is
suspect: at identical lambda the undetached term reduces its penalty by 32.2% while
the stop-gradient term manages only 8.2% (fold 0), consistent with the undetached
reduction coming from the `H(p)` floor rather than from real agreement.

**Unaffected by this amendment** (they do not depend on the adversarial term): the
2016postVFP binning artefact and the 21 -> 17 rebinning, the QvG degeneracy A/B, the
gate and baseline seeds, and REV-001.
