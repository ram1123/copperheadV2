# Source Model Inspection

Generated: `2026-08-03T21:43:33Z`

## Source Repository

- Path: `/work/users/hu1027/b-hive-h-mu-mu`
- Branch: `dev_may26_binary`
- Commit: `a64b79cf5ddc1ade41dbdb071347108e11f6463e`
- Dirty status: `M commands.md
 M pixi.lock
 M pixi.toml
 M run_score_multiple_models.sh
?? .gitattributes
?? b_hive_environment.yaml
?? vbf_dnn_note_model_spec.md`
- Selected model file: `/work/users/hu1027/b-hive-h-mu-mu/utils/models/particletransformer2.py`
- Selected config: `/work/users/hu1027/b-hive-h-mu-mu/config/HMuMu_ParT_12Apr2026.yml`
- Selected class: `ParticleTransformer_HMuMu` at source line `1124`

## Recency and Selection Evidence

The selected implementation is the HMuMu-specific 3-class transformer in `utils/models/particletransformer2.py`, paired with `config/HMuMu_ParT_12Apr2026.yml` and the current scoring/evaluation utilities. Git history affecting these transformer-related files was inspected before relying on filenames or timestamps.

- `ff2f6dc Improve the writting scores back`
- `07bc4e6 Fix plotting for ablation test, and add scoring feature for binary model`
- `5e72a86 Repair the scoring script`
- `7cd5f5e Repair the score writing back script`
- `5d718be "Add binary classier"`
- `62f1702 Fix weighted inference`
- `b25642c Add weighted train and modify preprocess`
- `3c9f000 Modify the pairwise features`
- `8a70efb Add Ablation Feature`
- `2e3c5d0 Add early stop`
- `51ea5a2 Fix some errors`
- `b8a1a48 Add feature: compact and add score to parquet files.`

## Related Source Files

- Training: `['utils/models/base_model.py']`
- Evaluation/inference: `['scripts/score_parquet_with_hmumu_model.py', 'scripts/plot_attention_maps.py']`
- Preprocessing: `['utils/dataset/structured_arrays.py', 'utils/coffea_processors/pf_candidate_and_vertex.py']`

## Transformer Entry Points Considered

- `CLS_TransformerEncoder`
- `CLS_TransformerEncoderLayer`
- `HF_TransformerEncoder`
- `HF_TransformerEncoderLayer`
- `PairEmbed`
- `ParticleTransformer2`
- `ParticleTransformer2_JetClass`
- `ParticleTransformer_HMuMu`
- `ParticleTransformer_HMuMu_GgHVsVBF_w_wgt`
- `ParticleTransformer_HMuMu_w_wgt`
- `RMSNorm`
- `SwiGLU`

## Alternate Implementations Not Selected

- `ParticleTransformer_HMuMu_w_wgt` line `856`: Weighted source variant uses the same HMuMu architecture but extracts train_wgt from the input for loss weighting; the study implements external absolute event weights explicitly.
- `ParticleTransformer_HMuMu_GgHVsVBF_w_wgt` line `1068`: Binary ggH-vs-VBF specialization cannot provide the requested three output probabilities including background.
- `ParticleTransformer2_JetClass` line `757`: JetClass entry point is a generic/non-HMuMu architecture and does not match the Hmumu object/global-token contract.

## Source Feature Contract

- Global features: `['global_ln_htsoft2', 'global_ln_htsoft5', 'global_MET_ln_pt', 'global_MET_ln_e', 'global_MET_sin_phi', 'global_MET_cos_phi', 'global_n_jets', 'train_wgt', 'event_wgt']`
- Object features: `['physObj_ln_pt', 'physObj_ln_e', 'physObj_eta', 'physObj_sin_phi', 'physObj_cos_phi', 'physObj_id', 'physObj_pt_4v', 'physObj_eta_4v', 'physObj_phi_4v', 'physObj_energy_4v']`
- Object tokens: `5`
- Training weight feature: `train_wgt`
- Source truths/order: `['is_bkg', 'is_ggH', 'is_VBF']`

## Source Architecture

- Defaults: `num_enc=3`, `num_head=8`, `embed_dim=128`, `dropout=0.1`, `swiglu=True`, `build_4v=True`.
- Object token layout: continuous features, `physObj_id`, and four raw kinematic slots `pt_4v`, `eta_4v`, `phi_4v`, `energy_4v`.
- Object IDs: padding=0, mu1=1, mu2=2, jet1=3, jet2=4, dimuon=5, global token=6.
- Pairwise attention bias: `PairEmbed` over `pt`, `eta`, `phi`, and `energy`, producing per-head attention bias.
- Padding mask: padded objects are identified by `pt_4v == 0.0`; pairwise attention receives a large negative mask for padded rows/columns.
- Global token: first seven global features are embedded separately and appended only for class-token attention.
- Pooling/head: learned CLS token attends over encoded object tokens plus global token, then RMSNorm and a linear class head produce logits.
- Source output order: `is_bkg`, `is_ggH`, `is_VBF`.
- Source weighted variant extracts `train_wgt` from global inputs for event-weighted cross entropy; the unweighted variant excludes it from the global token to avoid feature leakage.

## Study Feature Contract

- Study object order: `['mu1', 'mu2', 'jet1', 'jet2', 'dimuon']`
- Study object feature order: `['physObj_ln_pt', 'physObj_ln_e', 'physObj_eta', 'physObj_sin_phi', 'physObj_cos_phi', 'physObj_id', 'physObj_pt_4v', 'physObj_eta_4v', 'physObj_phi_4v', 'physObj_energy_4v']`
- Study global feature order: `['global_ln_htsoft2', 'global_ln_htsoft5', 'global_MET_ln_pt', 'global_MET_ln_e', 'global_MET_sin_phi', 'global_MET_cos_phi', 'global_n_jets']`
- Study class order: `['ggH', 'VBF', 'bkg']`

## Study Deviations

- Output order is remapped to `ggH`, `VBF`, `bkg`.
- The implementation is copied as a minimal local architecture rather than importing the co-worker repo at runtime.
- `PuppiMET_sumEt` is used as the available Stage1 proxy for the source `global_MET_ln_e` feature.
- `train_wgt` and `event_wgt` are kept as external metadata/weights rather than being fed into the seven-feature global token.
- The local smoke model keeps the source default compact HMuMu dimensions: 3 object-encoder layers, 8 heads, and 128 embedding dimensions.

## Source Files Intentionally Not Copied

- Luigi/law task wrappers and batch-submission orchestration.
- Production dataset constructors and Dask/Coffea production processors.
- Source scoring wrappers beyond the documented feature and inference conventions.
