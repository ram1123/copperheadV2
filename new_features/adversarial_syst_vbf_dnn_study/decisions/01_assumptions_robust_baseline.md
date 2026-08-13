# Assumptions and explicit decisions — `01-adversarial-syst-robust-vbf-dnn`

Written by the Code Generator, iteration 001. Section numbers refer to
`quick_tests/taskprompt_adversarial_syst_vbf_dnn.md`.

Everything below marked **[verified]** was checked by reading code or data in
this repository; everything marked **[assumption]** was not.

---

## 1. Decisions required by section 3.4

### 1.1 Target for `original_loss(pred_i, pred_nominal)` — **not detached** (default)

The nominal prediction enters as the soft target `sigmoid(logit_nominal)` and is
**not** detached, so both the variation branch and the nominal branch are
penalised. Exposed as `--adversarial-detach-nominal` to switch to stop-gradient.

*Rationale:* the task recommends this default. Penalising both branches lets the
network move the nominal score toward a variation-stable region rather than only
dragging variations toward a fixed nominal, which is what "decorrelation" should
mean.

### 1.2 Label smoothing inside the consistency term — **inherited** (default)

`original_loss` applies `cfg.label_smoothing` (0.0405 from the Optuna config) to
its target. The task says to use `original_loss` literally, so the default
(`--adversarial-consistency-smoothing inherit`) does exactly that.

*Flagged risk:* smoothing a target that is **already** a probability shifts the
consistency minimum from `pred_i = p_nom` to `pred_i = p_nom*(1-s) + 0.5*s`. Even
perfect decorrelation therefore leaves a residual gradient pulling variation
scores toward 0.5 — i.e. toward **flattening the score distribution**, which
section 6.3 lists explicitly as a reason to escalate to a human. If the Reviewer
sees score-distribution flattening, `--adversarial-consistency-smoothing none` is
the first thing to try. **[verified]** by reading `bce_with_logits_loss`.

### 1.3 Weighting — **plain sum, `wb` on every term**

Every term (nominal, consistency, variation-vs-label) uses the same per-event
weight tensor `wb` through the same `make_loss_weights` path. The sum over
variations is a **plain sum**, as written in the task formula.
`--adversarial-normalize` divides by `N_variations`; it is **off** by default and
is documented in the manifest as a deviation, because it rescales lambda.

**Pre-existing behaviour that makes this mostly moot — raise with the analyst.**
**[verified]** The training loop's `original_loss` does *not* actually apply event
weights. `bce_with_logits_loss(..., weights=None)` returns the **scalar** mean
BCE, so the subsequent
`sum(loss_raw * loss_w) / sum(|loss_w|)` reduces to `loss_raw` exactly (because
`make_loss_weights` takes `abs`, so `sum(w) == sum(|w|)`). The training loss is
therefore the *unweighted* mean BCE, while `evaluate()` — which calls the same
function with `weights=wb` — computes the properly weighted loss. This is why
`history["train_loss"]` and the batch-mean training loss differ by a large factor
in every run, including the un-modified baseline.

This is a real defect in the existing training code. It is **deliberately
preserved** here: fixing it would move the baseline and destroy the section-5.2
"unchanged when the switch is off" requirement. It is recorded as a finding for a
human, not silently corrected. `training_loop_loss()` in `train_dnn.py` carries
the same note.

### 1.4 Sign / asymmetry — **up and down independent**

`up` and `down` of a source are separate entries on the variation axis and each
contributes its own `2*L(pred_i, pred_nom) + L(pred_i, label)`. No pairing, no
symmetrisation. This is what the formula implies.

### 1.5 Compute cost — **measured, not estimated**

Measured on real Stage-1 inputs (2018, `vbf_powheg_dipole` + `ewk_zlljj`,
193k train rows/fold, 1 GPU), by running 3 and 13 epochs and taking the slope:

| Phase | `N_variations` | s/epoch | vs off |
|---|---|---|---|
| off | 0 | 8.6 | 1.00x |
| sweep | 4 | 10.8 | **1.26x** |
| full | 26 | 16.9 | **1.97x** |

Far below the naive `1 + N` scaling (5x / 27x) the task anticipated, because a
large share of per-epoch time is the two unchanged nominal evaluation passes
(`evaluate()` on train and val) and data loading, not the training forward pass.
Neither phase is close to infeasible. **[assumption]** that this ratio holds at
the 8x-larger real training set; the absolute time will scale, the ratio should
not change much.

### 1.6 Numerical guards

Three layers, all **[verified]** by test:

1. **Preprocessing** — an augmented column that is missing for a given
   sample/year, or non-finite for a given row, is filled from that row's
   **nominal** value (`_fill_missing_variation_columns`). An absent variation then
   satisfies `pred_i == pred_nominal` exactly and contributes zero to the
   consistency term.
2. **Preprocessing** — varied columns get the *same* pre-scaling sentinels as
   nominal, by reusing `pre_scaling_clean` on a per-variation nominal-named view
   (`_clean_variation_columns`) rather than reimplementing its rules.
3. **Dataset** — `ParquetDataset` re-checks and replaces any remaining non-finite
   varied value with the nominal value, and logs the count.

---

## 2. The four-year / year-decorrelated variation axis (section 3.3)

**Decision: 26 variations per event, on a year-agnostic canonical axis.**
This is the option section 4.1 mandates for the final phase ("full 26 variations
per event"), not the 56-wide union.

**[verified]** across the full configured sample list (4 years x 7 samples that
exist on disk):

- every year discovers exactly **26** shape variations, and all 7 samples within a
  year agree exactly (no ragged samples);
- **16** are common to all four years (Absolute, BBEC1, EC2, FlavorQCD, HF,
  RelativeBal, Total, mu_roccor — x up/down);
- **10** per year are year-decorrelated (Absolute, BBEC1, EC2, HF, RelativeSample
  x the year's own token x up/down); the union over years is **56**, the
  intersection **16**.

The implementation maps each year-decorrelated suffix onto a shared slot —
`Absolute_2018_up` and `Absolute_2016APV_up` both become
`Absolute_yearDecor_up` (`modules/systematics.canonical_variation_name`). Every
event therefore fills all 26 slots with its own year's sources, and the sum runs
over one homogeneous 26-wide axis. No 56-wide ragged tensor, no
absent-variation padding, and the effective lambda normalisation is identical for
all events.

The detection is a regex on the source name (`^<Source>_<year-token>$`), not a
hardcoded year table, so it does not need editing when Run-3 years are added.

## 3. Which features actually shift (section 3.3, second bullet)

**[verified]**, and it **corrects the task prompt's estimate.** The prompt says 17
features carry the 24 JEC variations; the true number is **15**. `nsoftjets5` and
`htsoft2` are excluded because `feature_name_for_variation` pins any feature whose
name contains `"soft"` to nominal — the soft-drop protection the task itself told
us to preserve. Per year, per event:

| | prompt | measured |
|---|---|---|
| features x 24 JEC variations | 17 | **15** |
| features x 2 `mu_roccor` variations | 6 | **6** |
| features with zero variations | 2 | **4** (`dimuon_cos_theta_cs`, `dimuon_phi_cs`, `nsoftjets5_nominal`, `htsoft2_nominal`) |
| extra columns per fold (single-year basis) | ~420 | **372** = 15x24 + 6x2 |

Confirmed end-to-end: the augmented 2018 fold parquet has exactly **372**
augmented columns for `--variation-set full` and **42** for `--variation-set
sweep` (15x2 + 6x2).

## 4. Scaling of varied features (section 3.3, point 4)

**Varied columns are standardized with the nominal feature's mean and std** — the
same per-fold `scalers_i.npz` values, never their own statistics. Re-standardizing
a variation with its own moments would absorb the shift the consistency term
exists to see.

**[verified]** on the augmented 2018 output: nominal features have weighted mean
6.6e-9 and weighted std 1.00000 exactly (by construction), while the varied
columns have weighted means up to 1.20 and stds spanning [0.00, 1.36]. If they had
been re-standardized they would all sit at (0, 1) too.

## 5. Disk and memory cost (section 3.3, point 5)

Measured on the 2018 smoke run (386k events) and scaled by the real dataset size
(3,089,832 events, factor 8.0):

| | measured (386k ev) | projected (3.09M ev) |
|---|---|---|
| nominal-only fold dir | — | 1.6 GB (existing, on disk) |
| `--variation-set sweep` (83 cols) | 536 MB | **~4.3 GB** |
| `--variation-set full` (413 cols) | 2.6 GB | **~21 GB** |

**[assumption]** Training RAM for the full set: a train fold is ~1.545M rows x 408
float32 columns ~ 2.5 GB as numpy, held twice (pandas frame + dataset copy), with
train/val/eval all resident via `_load_fold_data_cached` — order **10-12 GB RSS
per fold**. Not prohibitive on this node (503 GB), but it is the reason
`--adversarial-variation-chunk` exists for *device* memory.

Not prohibitive, so no subset/streaming strategy is proposed.

## 6. Why the variation forwards run in `eval()` mode

**[verified]** by a bit-identity test. The adversarial forward passes put the model
in `eval()` mode (and restore the previous mode in a `finally`). Two reasons, both
load-bearing for the section-4.2 "lambda=0 must reproduce switch-off" check:

- **dropout must not fire** on the variation branch, or it would consume the torch
  RNG stream and every subsequent nominal batch would differ;
- **BatchNorm must not update its running statistics** from variation batches, or
  the exported eval-mode TorchScript model that Stage-2 scores with would drift
  even at lambda = 0.

Gradients still flow — `eval()` only changes dropout/BN behaviour.
`test_lambda_zero_reproduces_switch_off_bitwise` asserts bit-identical weights.

**[assumption]/consequence:** the consistency target `p_nominal` is therefore an
*eval-mode* nominal prediction, while the first (original) loss term uses the
train-mode prediction. The two differ by dropout noise. This is a deliberate
trade: a train-mode variation branch would make the lambda=0 reproducibility check
impossible.

## 7. Model selection is still on the nominal metric

Early stopping and best-checkpoint selection monitor `val_auc_weighted`, computed
on a **nominal-only** validation set — unchanged from the baseline. The
adversarial term shapes training but does not enter model selection. Per-epoch
`train_loss_nominal` and `train_adv_penalty` are recorded in `history.json` so the
Reviewer can see whether the penalty is actually coming down.

**[assumption]** that selecting on nominal AUC is acceptable. If the sweep shows
the penalty still falling when early stopping fires, the monitor may need
revisiting — that is a config change and out of scope here.

## 8. Degenerate variation branches in Stage-1 — **raise with a human**

**[verified], and this is the most important physics finding of iteration 001.**

`jet1_btagUParTAK4QvG_<JEC>_up/down` and `jet2_btagUParTAK4QvG_<JEC>_up/down` are
a **constant `-1` sentinel** in Stage-1, for every event and every one of the 24
JEC variations, while the nominal branch spans the full discriminant range (3662
distinct values in a single 2018 `ewk_zlljj` file). Direct read of the Stage-1
parquet:

```
jet1_btagUParTAK4QvG_nominal    min=-1.0000 max= 0.9946 nunique=3662
jet1_btagUParTAK4QvG_Total_up   min=-1.0000 max=-1.0000 nunique=   1
jet1_btagUParTAK4QvG_Total_down min=-1.0000 max=-1.0000 nunique=   1
```

Consequences:

- **This is not new, and it is not confined to training.** Stage-2's
  `feature_name_for_variation` resolves the same column at inference, so every
  JEC-varied template in the *reference* datacards was already produced with both
  jets' QvG pinned to `-1`. That plausibly inflates the JEC shape systematic in
  the 1.52839 reference itself.
- **For the adversarial term it is actively harmful.** In scaled units the
  QvG "shift" is a mean |delta| of ~1.05 sigma, against 0.03-0.09 sigma for the
  genuine JEC shifts — 12 to 35 times larger. Left in, the consistency term would
  be dominated by it and the network would simply learn to ignore the QvG feature,
  losing nominal discrimination for a fake systematic.

Handling, per the task's instruction to *match* Stage-2 rather than invent:

- detection runs **always** and is loud (`find_degenerate_variation_columns`); the
  full list lands in `preprocess_manifest.json` under
  `systematic_variations.degenerate_columns` (37 of the 372 augmented columns were
  flagged on the 2018 smoke run, all of them QvG);
- the **default keeps them**, so training and Stage-2 inference agree;
- `--variation-drop-degenerate` falls them back to nominal, which makes them
  contribute exactly zero — at the cost of training/inference disagreement on
  those columns.

**This is a decision for the analyst, not for the agent.** Recommended: produce
the sweep-phase inputs *both* ways and compare, or fix the Stage-1 branch. It is
raised as a blocking question rather than resolved unilaterally.

---

## 9. Things this iteration did **not** verify

- No production training, Stage-1/2/3 or combine run has been executed
  (`approvals.production_granted` is `false` in `task.json`).
- The section-5.2 no-op check has been proven at **unit scale** against the actual
  `git HEAD` version of `train_dnn.py` (`tests/test_noop_vs_pre_change.py`), not
  yet on the real fold parquets.
- No `sigma_noise` band exists yet; no lambda has been swept; the section-2
  reference values have been re-read from the reference `significance.txt` and
  match the task table exactly.
