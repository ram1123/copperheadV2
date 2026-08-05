# Preprocessing and Training-Schedule Fixes

Task: `transformer-3class-preprocessing-fixes-2017` (continuation of `transformer-3class-study-2017`)  
Generated: `2026-08-05T16:17:29Z`  
Study: `transformer_3class_study_2017`, year `2017`

Five defects found in a sanity review of the study preprocessing against standard practice for transformer models. All five are fixed here. Every number below is recomputed from real 2017 Stage1 events or copied from a recorded run artifact.

## Summary

| Fix | What was wrong | Status |
|---|---|---|
| `FIX-1-weights` | Per-class training-weight normalization | fixed |
| `FIX-2-log1p` | log1p for non-negative global magnitudes | fixed |
| `FIX-3-amp` | AMP re-enabled: GradScaler-aware health check, plus a finite fp16 mask sentinel | fixed |
| `FIX-4-lr-schedule` | Linear warmup into cosine decay | fixed |
| `FIX-5-input-clip` | Clipping of standardized inputs | fixed |

---

## FIX-1 — Per-class training-weight normalization

The loss is sum(loss_i * w_i) / sum(w_i), so a class contributes in proportion to its summed weight. Raw absolute MC weights span ~7e4 between samples, so the signal classes contributed ~1e-5 of the gradient. The count-based inverse_sqrt_frequency factor could not help because max_events_per_class had already equalized the counts.

**Change.** Fit per-class scales on the train split that equalize each class total weight; disable class_balance.

| Class | Raw weight sum | Normalized weight sum |
|---|---|---|
| ggH | 2.98126 | 335.257 |
| VBF | 0.111114 | 335.257 |
| bkg | 1002.68 | 335.257 |

Max/min class weight-sum ratio: **9023.87 → 1**.
Equalization asserted by validate_model.py: `True`. class_balance is now `none` so the imbalance is not corrected twice.

## FIX-2 — log1p for non-negative global magnitudes

ln(clip(x, 1e-6)) mapped a physical zero to -13.8. htsoft5 is zero in a large fraction of events, so that spike dominated the standardization mean and std.

**Change.** Use log1p for htsoft2, htsoft5, MET pt, and MET sumEt; record the transform in the checkpoint.

| Feature | Zero-valued events | ln(clip) value at 0 | log1p value at 0 | ln(clip) mean/std | log1p mean/std |
|---|---|---|---|---|---|
| `global_ln_htsoft2` | 376 (4.5%) | -13.8155 | 0 | 2.132 / 3.794 | 2.937 / 1.187 |
| `global_ln_htsoft5` | 1816 (21.6%) | -13.8155 | 0 | -1.874 / 6.855 | 1.862 / 1.544 |
| `global_MET_ln_pt` | 0 (0.0%) | n/a | n/a | 3.238 / 0.8318 | 3.293 / 0.7796 |
| `global_MET_ln_e` | 0 (0.0%) | n/a | n/a | 5.754 / 0.4368 | 5.757 / 0.4353 |

- `global_ln_htsoft2`: the zero spike sat **4.4 std** from the non-zero bulk under ln(clip), now **2.59 std** under log1p.
- `global_ln_htsoft5`: the zero spike sat **2.22 std** from the non-zero bulk under ln(clip), now **1.54 std** under log1p.

## FIX-3 — AMP re-enabled

AMP was disabled with the note "float16 attention produced non-finite gradients". The observation was real but misattributed: GradScaler intentionally starts at an optimistic loss scale, overflows the first backward, skips that step, and halves the scale. The validation asserted finite gradients on that first step, so it could only ever see the designed transient.

**Change.** Judge AMP over several steps, counting only steps the scaler applied (src/amp_health.py). Separately, replace the float16 attention-mask sentinel finfo.min with a finite -1e4 to remove a genuine overflow hazard.

> **Correction to the original review.** The review originally attributed the AMP failure to the finfo.min sentinel. Tested directly on real 2017 events: with the OLD sentinel, float16 forward/backward produced finite gradients on every applied step, both at initialization and over 120 training steps. The sentinel was therefore not the cause; it is retained as hardening only.

### Root cause

GradScaler starts at an optimistic loss scale and lets the first backward overflow float16 on purpose, then halves the scale and retries. The previous validation asserted finite gradients on that first step, so it reported the scaler working as designed as a model defect and disabled AMP permanently.

- GradScaler default `init_scale`: `65536`
- Fix: probe_amp_health runs several steps and judges only the steps the scaler actually applied.

- Probe: `4` steps, `4` applied, `0` skipped by the scaler
- First step gradients finite: `True`
- Loss scale `65536` → `65536`
- **Healthy: `True`**

Whether the scaler overflows on step 0 is configuration-dependent — it varies with batch size, weighting, and the events in the first batch. In the runs recorded here it did not fire, but it was reproduced directly at batch 256 (8 of 67 gradient tensors non-finite at step 0, scale 65536 → 32768, every subsequent step finite). The point of the probe is that the outcome no longer depends on that coin flip: a skipped step is recognized as the scaler working, not as a model defect.

| Step | Loss | Loss scale | Gradients finite | Skipped by scaler |
|---|---|---|---|---|
| 0 | 1.188 | 65536 | True | False |
| 1 | 1.217 | 65536 | True | False |
| 2 | 1.2 | 65536 | True | False |
| 3 | 0.9841 | 65536 | True | False |

### Sentinel hardening (secondary)

- Old float16 sentinel: `-65504` — becomes `-inf` once an additional score below `-16` is added
- New float16 sentinel: `-10000` — stays finite: `True`
- New float32 sentinel: `-1e+09`
- Masked key still receives exactly zero attention: `True` (weight `0`)

Validation outcome: loss finite `True`, AMP healthy `True`, parameters updated `True`.

Training AMP state: requested `True`, enabled `True`, dtype `float16`, disable reason `None`.

## FIX-4 — Linear warmup into cosine decay

Plain AdamW at a constant LR; pre-norm transformers are unstable without warmup.

**Change.** LambdaLR stepped per optimizer step: linear warmup, then cosine decay to a floor.

- Schedule: `cosine`, enabled `True`
- Warmup: `8` of `164` optimizer steps (`82` per epoch x `2` epochs)
- LR floor ratio: `0.01`, final multiplier `0.0101004`
- Warmup verified on a live optimizer: base LR `0.0001` → at construction `5e-05` → after one step `0.0001` (warmup applied: `True`)

| Epoch | Optimizer steps | LR first | LR max | LR last |
|---|---|---|---|---|
| 1 | 82 | 1.25e-05 | 0.0001 | 5.54758e-05 |
| 2 | 82 | 5.44831e-05 | 5.44831e-05 | 1.01004e-06 |

Multiplier curve (step:multiplier): `0:0.125, 7:1.000, 8:1.000, 15:0.995, 30:0.952, 44:0.876, 59:0.761, 74:0.623, 89:0.475, 104:0.329, 119:0.200, 133:0.103, 148:0.035, 163:0.010`

## FIX-5 — Clipping of standardized inputs

Standardized features had unbounded tails; ParT-style pipelines clip them.

**Change.** Clip standardized object and global features to a symmetric bound persisted with the normalization statistics.

- Clip bound: `5`
- Standardized values inspected: `232911`
- Values beyond the bound before clipping: `36` (`0.0155%`)
- Max |standardized| object feature: `7.77229` → `5`
- Max |standardized| global feature: `5.31592` → `5`
- Padded tokens remain exactly zero after clipping: `True`

| Split | Max abs object | Max abs global | Bound respected |
|---|---|---|---|
| train | 5 | 5 | True |
| val | 5 | 5 | True |
| test | 5 | 4.70764 | True |

## Downstream smoke metrics

- Smoke test accuracy: `0.635669` on `4452` events
- Checkpoint-reload evaluation accuracy: `0.635669` on `4452` events

_Smoke-scale run only; not a production performance statement._

| Class | Events | Accuracy | Mean assigned probability |
|---|---|---|---|
| ggH | 1500 | 0.5907 | 0.4571 |
| VBF | 1500 | 0.778 | 0.6475 |
| bkg | 1452 | 0.5351 | 0.452 |

## Known issues left open (out of scope for this task)

- Pair features are not masked before the PairEmbed BatchNorm, so degenerate pad pairs still set its running statistics.
- InputProcess applies RMSNorm across the 5-dim feature axis, which partially undoes the per-feature standardization.
- The dimuon mass is still reconstructible from the mu1/mu2 tokens and their pairwise invariant mass.
- Normalization statistics are pooled across the heterogeneous token types.
- checkpoint_payload in scripts/train.py has unreachable plotting code after its return statement; the accuracy curve is therefore never regenerated.

## Machine-readable companions

- `reports/preprocessing_fixes_summary.yaml`
- `reports/preprocessing_fixes_summary.json`
