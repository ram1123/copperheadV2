#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

STUDY_ROOT = Path(__file__).resolve().parents[1]
if str(STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT))

from src.data import build_bundle, load_events, assign_splits
from src.features import INDEX_TO_CLASS, OUTPUT_COLUMNS
from src.model import build_model
from src.utils import cuda_memory_snapshot, ensure_dirs, load_yaml, resolve_path, runtime_config, select_device, set_seed, setup_logging, utc_now, write_json, write_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate a trained 3-class H->mumu transformer checkpoint.')
    parser.add_argument('--config', default='new_features/transformer_3class_study/configs/study_config.yaml')
    parser.add_argument('--checkpoint', default='new_features/transformer_3class_study/outputs/checkpoints/best_model.pt')
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='test')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default=None)
    parser.add_argument('--cuda-memory-limit-gib', type=float, default=None)
    parser.add_argument('--max-files-per-sample', type=int, default=None)
    parser.add_argument('--max-events-per-sample', type=int, default=None)
    parser.add_argument('--max-events-per-class', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def metrics_from_probs(probs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    pred = probs.argmax(axis=1)
    confusion = np.zeros((3, 3), dtype=int)
    for truth, guess in zip(labels, pred):
        confusion[int(truth), int(guess)] += 1
    per_class = {}
    for idx, name in INDEX_TO_CLASS.items():
        mask = labels == idx
        per_class[name] = {
            'n_events': int(mask.sum()),
            'accuracy': float((pred[mask] == idx).mean()) if mask.any() else None,
            'mean_assigned_probability': float(probs[mask, idx].mean()) if mask.any() else None,
        }
    return {
        'n_events': int(labels.size),
        'accuracy': float((pred == labels).mean()) if labels.size else None,
        'confusion': confusion.tolist(),
        'per_class': per_class,
        'mean_probabilities': {col: float(probs[:, idx].mean()) for idx, col in enumerate(OUTPUT_COLUMNS)} if labels.size else {},
        'max_softmax_sum_deviation': float(np.max(np.abs(probs.sum(axis=1) - 1.0))) if labels.size else None,
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_path(args.config))
    checkpoint_path = resolve_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_cfg = checkpoint.get('config') or cfg
    study_cfg = checkpoint_cfg['study']
    training_cfg = checkpoint_cfg.get('training', {})
    limits = checkpoint.get('training_limits', {})
    rt_cfg = runtime_config(checkpoint_cfg)

    device_name = args.device or rt_cfg.get('device', 'auto')
    cuda_limit = float(args.cuda_memory_limit_gib if args.cuda_memory_limit_gib is not None else rt_cfg['cuda'].get('max_memory_gib', 5.0))
    output_dir = resolve_path(study_cfg['output_dir'])
    reports_dir = resolve_path(study_cfg['reports_dir'])
    logs_dir = resolve_path(study_cfg['logs_dir'])
    ensure_dirs([output_dir / 'metrics', output_dir / 'predictions', reports_dir, logs_dir])
    logger = setup_logging(logs_dir / f'evaluate_{utc_now().replace(":", "")}.log')

    set_seed(int(training_cfg.get('seed', 12345)))
    device, device_info = select_device(device_name, cuda_limit)
    model = build_model(checkpoint.get('model_config') or checkpoint_cfg.get('model', {})).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # The pair standardization lives in FixedStandardize buffers, so load_state_dict
    # restores it with the weights. Fail loudly rather than silently scoring with the
    # identity transform if the checkpoint predates the fitted statistics.
    if not model.pair_normalization_fitted:
        raise RuntimeError(
            'Checkpoint has unfitted pair normalization buffers; it predates the '
            'valid-pair-only pair standardization and cannot be evaluated with this code.'
        )
    model.eval()

    max_files = args.max_files_per_sample if args.max_files_per_sample is not None else limits.get('max_files_per_sample')
    max_sample = args.max_events_per_sample if args.max_events_per_sample is not None else limits.get('max_events_per_sample')
    max_class = args.max_events_per_class if args.max_events_per_class is not None else limits.get('max_events_per_class')
    batch_size = int(args.batch_size if args.batch_size is not None else limits.get('physical_batch_size') or limits.get('batch_size') or training_cfg.get('batch_size', 256))

    df = load_events(
        resolve_path(study_cfg['resolved_samples_yaml']),
        Path(study_cfg['stage1_root']),
        str(study_cfg['dataset_subdir']),
        max_files_per_sample=max_files,
        max_events_per_sample=max_sample,
        max_events_per_class=max_class,
    )
    split_df = assign_splits(df, float(training_cfg.get('train_fraction', 0.70)), float(training_cfg.get('val_fraction', 0.15)), int(training_cfg.get('seed', 12345)))
    eval_df = split_df if args.split == 'all' else split_df[split_df['split'] == args.split].reset_index(drop=True)
    if len(eval_df) == 0:
        raise RuntimeError(f'No events available for split {args.split}.')
    bundle, _ = build_bundle(eval_df, normalization=checkpoint['normalization'], fit=False)

    probs_all = []
    labels_all = []
    with torch.inference_mode():
        for start in range(0, len(bundle.labels), batch_size):
            stop = min(start + batch_size, len(bundle.labels))
            objects = torch.from_numpy(bundle.objects[start:stop]).to(device)
            global_features = torch.from_numpy(bundle.global_features[start:stop]).to(device)
            mask = torch.from_numpy(bundle.padding_mask[start:stop]).to(device)
            probs = torch.softmax(model(objects, global_features, mask), dim=1)
            probs_all.append(probs.cpu().numpy())
            labels_all.append(bundle.labels[start:stop])
    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    metrics = {
        'generated_at': utc_now(),
        'checkpoint': str(checkpoint_path),
        'pair_normalization': checkpoint.get('pair_normalization'),
        'disco': checkpoint.get('disco'),
        'split': args.split,
        'device': device_info,
        'limits': {'max_files_per_sample': max_files, 'max_events_per_sample': max_sample, 'max_events_per_class': max_class, 'batch_size': batch_size},
        'metrics': metrics_from_probs(probs, labels),
        'cuda_memory_final': cuda_memory_snapshot(),
    }
    write_yaml(output_dir / 'metrics' / f'eval_{args.split}_metrics.yaml', metrics)
    write_json(output_dir / 'metrics' / f'eval_{args.split}_metrics.json', metrics)

    pred = bundle.metadata.copy().reset_index(drop=True)
    for idx, col in enumerate(OUTPUT_COLUMNS):
        pred[col] = probs[:, idx]
    pred['predicted_class_index'] = probs.argmax(axis=1)
    pred['predicted_class'] = [INDEX_TO_CLASS[int(idx)] for idx in pred['predicted_class_index']]
    pred['pred_class_index'] = pred['predicted_class_index']
    pred['pred_class'] = pred['predicted_class']
    pred.to_parquet(output_dir / 'predictions' / f'eval_{args.split}_predictions.parquet', index=False)

    report = [
        '# Evaluation Summary',
        '',
        f'Generated: `{metrics["generated_at"]}`',
        '',
        f'- Checkpoint: `{checkpoint_path}`',
        f'- Split: `{args.split}`',
        f'- Events: `{metrics["metrics"]["n_events"]}`',
        f'- Accuracy: `{metrics["metrics"]["accuracy"]}`',
        f'- Max softmax sum deviation: `{metrics["metrics"]["max_softmax_sum_deviation"]}`',
        '',
        '## Per-Class Summary',
        '',
    ]
    for name, row in metrics['metrics']['per_class'].items():
        report.append(f'- `{name}`: n={row["n_events"]}, acc={row["accuracy"]}, mean_p_true={row["mean_assigned_probability"]}')
    report.extend(['', 'This summary belongs to the minimal smoke-test checkpoint unless a larger checkpoint is passed explicitly.', ''])
    (reports_dir / 'evaluation_summary.md').write_text('\n'.join(report), encoding='utf-8')
    logger.info('evaluation metrics: %s', metrics['metrics'])
    print(json.dumps({'split': args.split, 'events': metrics['metrics']['n_events'], 'accuracy': metrics['metrics']['accuracy']}, indent=2))


if __name__ == '__main__':
    main()
