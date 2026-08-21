# Smoke Test Report

Generated: `2026-08-07T20:27:43Z`

- Events loaded: `2980833`
- Device: `cuda`
- CUDA memory limit request: `5.0 GiB`
- CUDA status: `CUDA used successfully under the 5-GiB allocator limit.`
- AMP requested: `True`
- AMP enabled: `True`
- AMP disable reason: `None`
- Masked attention bias (float16): `-10000.0`
- LR schedule: `cosine` warmup `8152`/`163040` steps, floor `0.01`
- Standardized clip: `5.0`
- Train class weight sums (raw): `[185.34554170869342, 14.256174578988208, 114707.14857014379]`
- Train class weight sums (normalized): `[38302.25009040348, 38302.2500968941, 38302.25009314437]`
- Train class weight sum ratio: raw `8046.138038966468` -> normalized `1.000000000169458`
- Epochs: `40`
- Requested batch size: `512`
- Physical batch size: `512`
- Gradient accumulation steps: `1`
- Effective batch size: `512`
- Trainable parameters: `1062663`
- Best checkpoint: `/work/users/yun79/sideHustle2/copperheadV2/new_features/transformer_3class_study/outputs/checkpoints/best_model.pt`
- Final checkpoint: `/work/users/yun79/sideHustle2/copperheadV2/new_features/transformer_3class_study/outputs/checkpoints/final_model.pt`
- CUDA memory summary: `/work/users/yun79/sideHustle2/copperheadV2/new_features/transformer_3class_study/outputs/metrics/cuda_memory_summary.yaml`
- Test accuracy: `0.8640514397539838`
- Max softmax sum deviation: `1.7881393432617188e-07`

## Per-Class Test Summary

- `ggH`: n=92914, acc=0.8099963407021546, mean_p_true=0.715688943862915
- `VBF`: n=192617, acc=0.8047109029836411, mean_p_true=0.7462267279624939
- `bkg`: n=161594, acc=0.9658650692476206, mean_p_true=0.9361957907676697

This is a 2-epoch smoke run and is not a production performance statement.
