# Smoke Test Report

Generated: `2026-08-05T16:15:41Z`

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
- Trainable parameters: `1062671`
- Best checkpoint: `/work/users/yun79/sideHustle2/copperheadV2/new_features/transformer_3class_study/outputs/checkpoints/best_model.pt`
- Final checkpoint: `/work/users/yun79/sideHustle2/copperheadV2/new_features/transformer_3class_study/outputs/checkpoints/final_model.pt`
- CUDA memory summary: `/work/users/yun79/sideHustle2/copperheadV2/new_features/transformer_3class_study/outputs/metrics/cuda_memory_summary.yaml`
- Test accuracy: `0.6356693620844565`
- Max softmax sum deviation: `1.1920928955078125e-07`

## Per-Class Test Summary

- `ggH`: n=1500, acc=0.5906666666666667, mean_p_true=0.4570537805557251
- `VBF`: n=1500, acc=0.778, mean_p_true=0.6474733352661133
- `bkg`: n=1452, acc=0.5351239669421488, mean_p_true=0.4519728124141693

This is a 2-epoch smoke run and is not a production performance statement.
