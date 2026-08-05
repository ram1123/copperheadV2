# Minimal 2017 Three-Class Transformer Study

This directory is an isolated smoke-test study for a 3-class H->mumu transformer classifier using the 2017 Copperhead Stage1 parquet outputs. It validates the architecture/data/evaluation path only; it is not a production model and is not wired into the Copperhead production workflow.

Classes are encoded as:

| index | label | output column |
| --- | --- | --- |
| 0 | `ggH` | `p_ggH` |
| 1 | `VBF` | `p_VBF` |
| 2 | `bkg` | `p_bkg` |

The source architecture reference is `/work/users/hu1027/b-hive-h-mu-mu`, specifically the HMuMu ParticleTransformer code under `utils/models/particletransformer2.py` and the training config `config/HMuMu_ParT_12Apr2026.yml`. It was selected from transformer-related Git history and HMuMu entry points rather than file modification time alone; run `inspect_source_model.py` to refresh the commit, related-file, and alternate-implementation evidence.

## Quick Start

Run from the Copperhead repository root:

```bash
pixi run python new_features/transformer_3class_study/scripts/inspect_source_model.py --config new_features/transformer_3class_study/configs/study_config.yaml
pixi run python new_features/transformer_3class_study/scripts/inspect_environment.py --config new_features/transformer_3class_study/configs/study_config.yaml
pixi run python new_features/transformer_3class_study/scripts/inspect_stage1_inputs.py --config new_features/transformer_3class_study/configs/study_config.yaml --max-files-per-sample 1
pixi run python new_features/transformer_3class_study/scripts/validate_model.py --config new_features/transformer_3class_study/configs/study_config.yaml --device auto --cuda-memory-limit-gib 5 --max-files-per-sample 1 --max-events-per-class 600 --batch-size 64
pixi run python new_features/transformer_3class_study/scripts/train.py --config new_features/transformer_3class_study/configs/study_config.yaml --device auto --cuda-memory-limit-gib 5 --max-files-per-sample 1 --max-events-per-class 10000 --epochs 2 --batch-size 256 --smoke-test
pixi run python new_features/transformer_3class_study/scripts/evaluate.py --config new_features/transformer_3class_study/configs/study_config.yaml --checkpoint new_features/transformer_3class_study/outputs/checkpoints/best_model.pt --split test --device auto --cuda-memory-limit-gib 5
```

## Architecture and Features

- The transformer sees five object tokens: `mu1`, `mu2`, `jet1`, `jet2`, and `dimuon`.
- Object tokens follow the source feature convention: `physObj_ln_pt`, `physObj_ln_e`, `physObj_eta`, `physObj_sin_phi`, `physObj_cos_phi`, `physObj_id`, `physObj_pt_4v`, `physObj_eta_4v`, `physObj_phi_4v`, and `physObj_energy_4v`.
- Global features are `global_ln_htsoft2`, `global_ln_htsoft5`, `global_MET_ln_pt`, `global_MET_ln_e`, `global_MET_sin_phi`, `global_MET_cos_phi`, and `global_n_jets`.
- The source HMuMu transformer uses class order `bkg`, `ggH`, `VBF`; this study deliberately remaps to `ggH`, `VBF`, `bkg` to match the task request.
- Padding is defined by `physObj_pt_4v <= 0`, with padded rows zeroed and excluded from attention through the padding mask/pairwise bias.
- Normalization is fitted only on the training split and saved into each checkpoint. Non-finite feature values are replaced with zero after bounded log/energy transformations.
- The Stage1 branch names differ slightly from the source preprocessing. This study resolves nominal Copperhead columns such as `jet1_pt_nominal`, `htsoft2_nominal`, `PuppiMET_pt`, and `njets_nominal` in `src/features.py`.

## Samples and Weights

- `inspect_stage1_inputs.py` reads `configs/samples/samples.yaml` for 2017, resolves Stage1 directories, excludes recorded data, and drops `dy_VBF_filter` to avoid inclusive/filtered DY overlap.
- The resolved sample manifest is `configs/samples_2017_resolved.yaml`; it records the original YAML entry, resolved Stage1 path, assigned class, class index, status, and reason for each included sample.
- DY Z pT overlap is handled explicitly by using `wgt_nominal / separate_wgt_zpt` when `separate_wgt_zpt` is present.
- Training uses the absolute event weight for cross entropy so signed negative weights are not passed directly into the loss. Signed and absolute weight summaries are reported for inspected inputs.

## CUDA and Batching

- `--device auto` selects CUDA when available and CPU otherwise. `--device cuda` fails if CUDA is unavailable.
- When CUDA is used, the PyTorch allocator is capped with `torch.cuda.set_per_process_memory_fraction` before model allocation. This limits PyTorch allocator behavior, not CUDA context, driver, library, or non-PyTorch allocations.
- CUDA mixed precision is enabled when CUDA is selected and the runtime config allows it.
- Training runs a representative forward/backward memory test before the epoch loop. If the requested physical batch size does not fit, it retries the sequence `256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1` and records the chosen physical batch size plus gradient accumulation.
- CUDA memory and batch-size evidence is saved to `outputs/metrics/cuda_memory_summary.yaml`.

## Outputs

- Checkpoints: `outputs/checkpoints/best_model.pt` and `outputs/checkpoints/final_model.pt`.
- Metrics: `outputs/metrics/*.yaml` and `outputs/metrics/*.json`.
- Predictions: `outputs/predictions/*.parquet`, including `source_file`, `event_weight`, `data_split`, `p_ggH`, `p_VBF`, `p_bkg`, `predicted_class`, and `predicted_class_index`.
- Reports: `reports/source_model_report.md`, `reports/input_compatibility_report.md`, `reports/smoke_test_report.md`, and `reports/evaluation_summary.md`.
- Logs: `logs/`.

## Known Limitations

- The two-epoch smoke checkpoint can collapse to a background-only argmax classifier. That is a pipeline validation result, not a physics-performance claim.
- Only smoke-scale file/event caps are used here. Production integration would need a separate task for full Run 2 training, hyperparameter scans, systematic validation, and production inference wiring.
- This study intentionally omits Luigi, law, HTCondor, and production workflow wrappers.
