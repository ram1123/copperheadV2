#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
if str(STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT))

from src.features import describe_feature_contract
from src.utils import git_info, git_log, load_yaml, resolve_path, write_yaml, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inspect the source HMuMu transformer model and write a compatibility report.')
    parser.add_argument('--config', default='new_features/transformer_3class_study/configs/study_config.yaml')
    return parser.parse_args()


def _existing(source_repo: Path, rel_paths: list[str]) -> list[str]:
    return [path for path in rel_paths if (source_repo / path).exists()]


def _first_line(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return None
    return text[:match.start()].count('\n') + 1


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    study_cfg = cfg['study']
    source_repo = Path(study_cfg['source_transformer_repo'])
    source_model_file = Path(study_cfg['source_model_file'])
    source_config = Path(study_cfg['source_model_config'])
    reports_dir = resolve_path(study_cfg['reports_dir'])
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_text = source_model_file.read_text(encoding='utf-8')
    source_cfg = load_yaml(source_config)
    classes = sorted(set(re.findall(r'^class\s+([A-Za-z0-9_]+)\(', model_text, flags=re.MULTILINE)))
    hmu_classes = [name for name in classes if 'HMuMu' in name or 'Transformer' in name or name in {'PairEmbed', 'RMSNorm', 'SwiGLU'}]
    lines = model_text.splitlines()
    shape_hits = []
    for idx, line in enumerate(lines, start=1):
        if any(token in line for token in ['num_head', 'embed_dim', 'global_token_dim', 'ObjectIDEmbedding', 'PairEmbed', 'padding_mask']):
            shape_hits.append({'line': idx, 'text': line.strip()})

    source_global_features = list(source_cfg.get('global_features') or [])
    source_object_features = list(source_cfg.get('cpf_candidates') or [])
    source_truth_order = list(source_cfg.get('truths') or [])
    source_training_weight_feature = source_cfg.get('training_weight_feature')
    related_training = _existing(source_repo, ['utils/models/base_model.py'])
    related_evaluation = _existing(source_repo, ['scripts/score_parquet_with_hmumu_model.py', 'scripts/plot_attention_maps.py'])
    related_preprocessing = _existing(source_repo, ['utils/dataset/structured_arrays.py', 'utils/coffea_processors/pf_candidate_and_vertex.py'])
    history_paths = [
        str(source_model_file.relative_to(source_repo)),
        str(source_config.relative_to(source_repo)),
        *related_training,
        *related_evaluation,
        *related_preprocessing,
    ]
    source_history = git_log(source_repo, history_paths, max_count=16)
    selected_line = _first_line(r'^class\s+ParticleTransformer_HMuMu\(', model_text)
    weighted_line = _first_line(r'^class\s+ParticleTransformer_HMuMu_w_wgt\(', model_text)
    binary_line = _first_line(r'^class\s+ParticleTransformer_HMuMu_GgHVsVBF_w_wgt\(', model_text)
    alternate_candidates = [
        {
            'name': 'ParticleTransformer_HMuMu_w_wgt',
            'line': weighted_line,
            'reason_not_selected': 'Weighted source variant uses the same HMuMu architecture but extracts train_wgt from the input for loss weighting; the study implements external absolute event weights explicitly.',
        },
        {
            'name': 'ParticleTransformer_HMuMu_GgHVsVBF_w_wgt',
            'line': binary_line,
            'reason_not_selected': 'Binary ggH-vs-VBF specialization cannot provide the requested three output probabilities including background.',
        },
        {
            'name': 'ParticleTransformer2_JetClass',
            'line': _first_line(r'^class\s+ParticleTransformer2_JetClass\(', model_text),
            'reason_not_selected': 'JetClass entry point is a generic/non-HMuMu architecture and does not match the Hmumu object/global-token contract.',
        },
    ]

    summary = {
        'generated_at': utc_now(),
        'source_repo': git_info(source_repo),
        'source_model_file': str(source_model_file),
        'source_model_config': str(source_config),
        'source_model_name': study_cfg['source_model_name'],
        'classes_found': hmu_classes,
        'source_truth_order': source_truth_order,
        'source_global_features': source_global_features,
        'source_cpf_candidates': source_object_features,
        'source_n_cpf_candidates': source_cfg.get('n_cpf_candidates'),
        'source_training_weight_feature': source_training_weight_feature,
        'source_related_training_files': related_training,
        'source_related_evaluation_files': related_evaluation,
        'source_related_preprocessing_files': related_preprocessing,
        'source_git_history_evidence': source_history,
        'source_selected_class_line': selected_line,
        'alternate_transformers_considered': alternate_candidates,
        'source_architecture': {
            'selected_class': 'ParticleTransformer_HMuMu',
            'num_classes': len(source_truth_order) or 3,
            'num_object_encoder_layers': 3,
            'num_attention_heads': 8,
            'embedding_dimension': 128,
            'dropout': 0.1,
            'feed_forward': 'SwiGLU FFN with 4x hidden width after gating',
            'normalization': 'RMSNorm pre-normalization',
            'pooling': 'learned class token with class-token attention over object tokens plus a global token',
            'pairwise_bias': 'PairEmbed over pt/eta/phi/energy-derived pairwise features',
            'selected_model_line': selected_line,
        },
        'study_feature_contract': describe_feature_contract(),
        'implementation_notes': [
            'This study mirrors the source HMuMu variant at architecture level instead of importing it, to keep the smoke study isolated.',
            'The source class order is bkg/ggH/VBF; the study output order is ggH/VBF/bkg by task request.',
            'The source code uses five object tokens and a seven-feature global token with a class-token attention stage.',
            'The source code masks padded objects by zero pt_4v.',
        ],
        'source_shape_hits': shape_hits[:80],
    }
    write_yaml(reports_dir / 'source_model_summary.yaml', summary)

    report = [
        '# Source Model Inspection',
        '',
        f'Generated: `{summary["generated_at"]}`',
        '',
        '## Source Repository',
        '',
        f'- Path: `{source_repo}`',
        f'- Branch: `{summary["source_repo"].get("branch")}`',
        f'- Commit: `{summary["source_repo"].get("commit")}`',
        f'- Dirty status: `{summary["source_repo"].get("status_short") or "clean"}`',
        f'- Selected model file: `{source_model_file}`',
        f'- Selected config: `{source_config}`',
        f'- Selected class: `{study_cfg["source_model_name"]}` at source line `{selected_line}`',
        '',
        '## Recency and Selection Evidence',
        '',
        'The selected implementation is the HMuMu-specific 3-class transformer in `utils/models/particletransformer2.py`, paired with `config/HMuMu_ParT_12Apr2026.yml` and the current scoring/evaluation utilities. Git history affecting these transformer-related files was inspected before relying on filenames or timestamps.',
        '',
        *[f'- `{line}`' for line in source_history[:12]],
        '',
        '## Related Source Files',
        '',
        f'- Training: `{related_training}`',
        f'- Evaluation/inference: `{related_evaluation}`',
        f'- Preprocessing: `{related_preprocessing}`',
        '',
        '## Transformer Entry Points Considered',
        '',
        *[f'- `{name}`' for name in hmu_classes],
        '',
        '## Alternate Implementations Not Selected',
        '',
        *[f'- `{item["name"]}` line `{item["line"]}`: {item["reason_not_selected"]}' for item in alternate_candidates],
        '',
        '## Source Feature Contract',
        '',
        f'- Global features: `{summary["source_global_features"]}`',
        f'- Object features: `{summary["source_cpf_candidates"]}`',
        f'- Object tokens: `{summary["source_n_cpf_candidates"]}`',
        f'- Training weight feature: `{summary["source_training_weight_feature"]}`',
        f'- Source truths/order: `{summary["source_truth_order"]}`',
        '',
        '## Source Architecture',
        '',
        '- Defaults: `num_enc=3`, `num_head=8`, `embed_dim=128`, `dropout=0.1`, `swiglu=True`, `build_4v=True`.',
        '- Object token layout: continuous features, `physObj_id`, and four raw kinematic slots `pt_4v`, `eta_4v`, `phi_4v`, `energy_4v`.',
        '- Object IDs: padding=0, mu1=1, mu2=2, jet1=3, jet2=4, dimuon=5, global token=6.',
        '- Pairwise attention bias: `PairEmbed` over `pt`, `eta`, `phi`, and `energy`, producing per-head attention bias.',
        '- Padding mask: padded objects are identified by `pt_4v == 0.0`; pairwise attention receives a large negative mask for padded rows/columns.',
        '- Global token: first seven global features are embedded separately and appended only for class-token attention.',
        '- Pooling/head: learned CLS token attends over encoded object tokens plus global token, then RMSNorm and a linear class head produce logits.',
        '- Source output order: `is_bkg`, `is_ggH`, `is_VBF`.',
        '- Source weighted variant extracts `train_wgt` from global inputs for event-weighted cross entropy; the unweighted variant excludes it from the global token to avoid feature leakage.',
        '',
        '## Study Feature Contract',
        '',
        f'- Study object order: `{summary["study_feature_contract"]["object_order"]}`',
        f'- Study object feature order: `{summary["study_feature_contract"]["object_features"]}`',
        f'- Study global feature order: `{summary["study_feature_contract"]["global_features"]}`',
        f'- Study class order: `{summary["study_feature_contract"]["study_class_order"]}`',
        '',
        '## Study Deviations',
        '',
        '- Output order is remapped to `ggH`, `VBF`, `bkg`.',
        '- The implementation is copied as a minimal local architecture rather than importing the co-worker repo at runtime.',
        '- `PuppiMET_sumEt` is used as the available Stage1 proxy for the source `global_MET_ln_e` feature.',
        '- `train_wgt` and `event_wgt` are kept as external metadata/weights rather than being fed into the seven-feature global token.',
        '- The local smoke model keeps the source default compact HMuMu dimensions: 3 object-encoder layers, 8 heads, and 128 embedding dimensions.',
        '',
        '## Source Files Intentionally Not Copied',
        '',
        '- Luigi/law task wrappers and batch-submission orchestration.',
        '- Production dataset constructors and Dask/Coffea production processors.',
        '- Source scoring wrappers beyond the documented feature and inference conventions.',
        '',
    ]
    (reports_dir / 'source_model_report.md').write_text('\n'.join(report), encoding='utf-8')
    print(f'wrote {reports_dir / "source_model_report.md"}')


if __name__ == '__main__':
    main()
