# Smoke Test Report

Generated: `2026-08-05T16:58:00Z`

- Events loaded: `29685`
- Device: `cuda`
- CUDA memory limit request: `5.0 GiB`
- CUDA status: `CUDA used successfully under the 5-GiB allocator limit.`
- AMP requested: `True`
- AMP enabled: `True`
- AMP disable reason: `None`
- Masked attention bias (float16): `-10000.0`
- LR schedule: `cosine` warmup `8`/`164` steps, floor `0.01`
- Standardized clip: `5.0`
- Train class weight sums (raw): `[2.9812569388741395, 0.11111410970568159, 1002.6796182490007]`
- Train class weight sums (normalized): `[335.2573296856135, 335.25732977082953, 335.2573299478115]`
- Train class weight sum ratio: raw `9023.87303381085` -> normalized `1.0000000007820797`
- Epochs: `2`
- Requested batch size: `256`
- Physical batch size: `256`
- Gradient accumulation steps: `1`
- Effective batch size: `256`
- Trainable parameters: `1062663`
- Best checkpoint: `/work/users/yun79/sideHustle2/copperheadV2/new_features/transformer_3class_study/outputs/checkpoints/best_model.pt`
- Final checkpoint: `/work/users/yun79/sideHustle2/copperheadV2/new_features/transformer_3class_study/outputs/checkpoints/final_model.pt`
- CUDA memory summary: `/work/users/yun79/sideHustle2/copperheadV2/new_features/transformer_3class_study/outputs/metrics/cuda_memory_summary.yaml`
- Test accuracy: `0.6415094339622641`
- Max softmax sum deviation: `1.1920928955078125e-07`

## Per-Class Test Summary

- `ggH`: n=1500, acc=0.5866666666666667, mean_p_true=0.4567616879940033
- `VBF`: n=1500, acc=0.7806666666666666, mean_p_true=0.6490616798400879
- `bkg`: n=1452, acc=0.5544077134986226, mean_p_true=0.4603968560695648

This is a 2-epoch smoke run and is not a production performance statement.
