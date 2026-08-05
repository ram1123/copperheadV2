from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from .features import (
    CLASS_TO_INDEX,
    OUTPUT_COLUMNS,
    build_event_weights,
    build_global_features,
    build_object_features,
    canonicalize_frame,
    columns_to_read,
    resolve_columns,
)
from .losses import apply_class_weight_scales, class_weight_sums, fit_class_weight_scales
from .utils import load_yaml, resolve_path

ALIASES_BY_SAMPLE = {
    'st_tw_top': ['st_tW_top'],
    'st_tw_antitop': ['st_tW_antitop'],
    'st_t_top': ['st_tchannel_top'],
    'st_t_antitop': ['st_tchannel_antitop'],
    'zz': ['zz_2l2q', 'zz_4l'],
}
EXCLUDE_EXACT = {'dy_VBF_filter'}


@dataclass(frozen=True)
class ResolvedSample:
    name: str
    class_name: str
    class_index: int
    source_group: str
    yaml_name: str
    n_files: int


@dataclass
class TensorBundle:
    objects: np.ndarray
    global_features: np.ndarray
    padding_mask: np.ndarray
    labels: np.ndarray
    event_weights: np.ndarray
    train_weights: np.ndarray
    raw_train_weights: np.ndarray
    row_indices: np.ndarray
    metadata: pd.DataFrame


class HmumuTensorDataset(Dataset):
    def __init__(self, bundle: TensorBundle) -> None:
        self.objects = torch.from_numpy(bundle.objects.astype(np.float32))
        self.global_features = torch.from_numpy(bundle.global_features.astype(np.float32))
        self.padding_mask = torch.from_numpy(bundle.padding_mask.astype(bool))
        self.labels = torch.from_numpy(bundle.labels.astype(np.int64))
        self.weights = torch.from_numpy(bundle.train_weights.astype(np.float32))
        self.row_indices = torch.from_numpy(bundle.row_indices.astype(np.int64))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        return (
            self.objects[idx],
            self.global_features[idx],
            self.padding_mask[idx],
            self.labels[idx],
            self.weights[idx],
            self.row_indices[idx],
        )


def parquet_files_for_sample(stage1_root: Path, dataset_subdir: str, sample: str) -> list[Path]:
    sample_dir = stage1_root / dataset_subdir / sample
    return sorted(sample_dir.glob('*/*.parquet'))


def available_stage1_samples(stage1_root: Path, dataset_subdir: str) -> set[str]:
    root = stage1_root / dataset_subdir
    if not root.exists():
        return set()
    return {path.name for path in root.iterdir() if path.is_dir()}


def _processes_for_year(group_data: dict[str, Any], year: int) -> list[str]:
    per_year = group_data.get('processes_per_year') or {}
    value = per_year.get(str(year), per_year.get(year))
    if isinstance(value, list):
        return list(value)
    if value in ({}, None):
        return list(group_data.get('processes') or [])
    return list(group_data.get('processes') or [])


def _expand_candidate(yaml_name: str, available: set[str]) -> list[str]:
    if yaml_name in EXCLUDE_EXACT:
        return []
    if yaml_name in available:
        return [yaml_name]
    expanded = []
    for alias in ALIASES_BY_SAMPLE.get(yaml_name, []):
        if alias in available:
            expanded.append(alias)
    return expanded


def resolve_samples_from_official_yaml(samples_yaml: Path, stage1_root: Path, dataset_subdir: str, year: int) -> dict[str, Any]:
    raw = load_yaml(samples_yaml)
    available = available_stage1_samples(stage1_root, dataset_subdir)
    samples: list[ResolvedSample] = []
    excluded: list[dict[str, str]] = []
    seen: set[tuple[str, int]] = set()

    class_plan = [
        ('signal', 'GGH', 'ggH'),
        ('signal', 'VBF', 'VBF'),
    ]
    for section, group, class_name in class_plan:
        group_data = raw.get(section, {}).get('groups', {}).get(group, {})
        for yaml_name in _processes_for_year(group_data, year):
            expanded = _expand_candidate(yaml_name, available)
            if not expanded:
                excluded.append({'yaml_name': yaml_name, 'source_group': f'{section}/{group}', 'reason': 'not present in Stage1 compacted directory'})
            for sample_name in expanded:
                key = (sample_name, CLASS_TO_INDEX[class_name])
                if key in seen:
                    excluded.append({'yaml_name': yaml_name, 'source_group': f'{section}/{group}', 'reason': f'duplicate of {sample_name}'})
                    continue
                seen.add(key)
                n_files = len(parquet_files_for_sample(stage1_root, dataset_subdir, sample_name))
                samples.append(ResolvedSample(sample_name, class_name, CLASS_TO_INDEX[class_name], f'{section}/{group}', yaml_name, n_files))

    for group, group_data in (raw.get('background', {}).get('groups') or {}).items():
        for yaml_name in _processes_for_year(group_data, year):
            if yaml_name in EXCLUDE_EXACT:
                excluded.append({'yaml_name': yaml_name, 'source_group': f'background/{group}', 'reason': 'excluded overlap sample for this inclusive-DY smoke study'})
                continue
            expanded = _expand_candidate(yaml_name, available)
            if not expanded:
                excluded.append({'yaml_name': yaml_name, 'source_group': f'background/{group}', 'reason': 'not present in Stage1 compacted directory'})
            for sample_name in expanded:
                key = (sample_name, CLASS_TO_INDEX['bkg'])
                if key in seen:
                    excluded.append({'yaml_name': yaml_name, 'source_group': f'background/{group}', 'reason': f'duplicate of {sample_name}'})
                    continue
                seen.add(key)
                n_files = len(parquet_files_for_sample(stage1_root, dataset_subdir, sample_name))
                samples.append(ResolvedSample(sample_name, 'bkg', CLASS_TO_INDEX['bkg'], f'background/{group}', yaml_name, n_files))

    for data_name in sorted(name for name in available if name.startswith('data_')):
        excluded.append({'yaml_name': data_name, 'source_group': 'data', 'reason': 'data excluded from supervised MC training'})

    sample_items = []
    for sample in samples:
        item = asdict(sample)
        item.update({
            'sample_name': sample.name,
            'original_yaml_entry': sample.yaml_name,
            'resolved_stage1_path': str(stage1_root / dataset_subdir / sample.name),
            'assigned_class': sample.class_name,
            'status': 'included',
            'reason': 'resolved from 2017 samples.yaml and present in Stage1 compacted directory',
        })
        sample_items.append(item)
    return {
        'year': year,
        'stage1_root': str(stage1_root),
        'dataset_subdir': dataset_subdir,
        'class_mapping': CLASS_TO_INDEX,
        'sample_resolution_notes': [
            'Parsed from configs/samples/samples.yaml for the requested year.',
            'Data samples are excluded.',
            'dy_VBF_filter is excluded as a DY overlap guard.',
            'Historical ST and zz aliases are resolved to existing Stage1 directory names when needed.',
            'For DY samples, train_wgt is wgt_nominal divided by separate_wgt_zpt when that branch exists and is nonzero.',
        ],
        'samples': sample_items,
        'excluded': excluded,
    }


def load_resolved_samples(path: Path) -> list[ResolvedSample]:
    data = load_yaml(path)
    samples = []
    for item in data.get('samples', []):
        samples.append(ResolvedSample(
            name=str(item['name']),
            class_name=str(item['class_name']),
            class_index=int(item['class_index']),
            source_group=str(item.get('source_group', 'unknown')),
            yaml_name=str(item.get('yaml_name', item['name'])),
            n_files=int(item.get('n_files', 0)),
        ))
    return samples


def _stable_int(*parts: object) -> int:
    text = '|'.join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:16], 16)


def _sample_budget(samples: list[ResolvedSample], max_events_per_sample: int | None, max_events_per_class: int | None) -> dict[str, int | None]:
    budgets: dict[str, int | None] = {}
    by_class: dict[int, list[ResolvedSample]] = {}
    for sample in samples:
        by_class.setdefault(sample.class_index, []).append(sample)
    for class_index, class_samples in by_class.items():
        per_class = None
        if max_events_per_class is not None:
            per_class = int(math.ceil(max_events_per_class / max(len(class_samples), 1)))
        for sample in class_samples:
            values = [value for value in [max_events_per_sample, per_class] if value is not None]
            budgets[sample.name] = min(values) if values else None
    return budgets


def _read_one_file(path: Path, class_name: str, class_index: int, sample_name: str, source_group: str, row_limit: int | None) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    resolution = resolve_columns(pf.schema_arrow.names)
    if resolution.missing_required:
        raise RuntimeError(f'{path} is missing required columns: {resolution.missing_required}')
    table = pf.read(columns=columns_to_read(resolution))
    frame = canonicalize_frame(table.to_pandas(), resolution)
    if row_limit is not None:
        frame = frame.head(row_limit)
    frame = frame.reset_index(drop=True)
    frame['_source_file'] = str(path)
    frame['_source_row'] = np.arange(len(frame), dtype=np.int64)
    frame['sample'] = sample_name
    frame['source_group'] = source_group
    frame['true_class'] = class_name
    frame['true_class_index'] = class_index
    return frame


def load_events(
    resolved_samples_path: Path,
    stage1_root: Path,
    dataset_subdir: str,
    max_files_per_sample: int | None = None,
    max_events_per_sample: int | None = None,
    max_events_per_class: int | None = None,
) -> pd.DataFrame:
    samples = load_resolved_samples(resolved_samples_path)
    budgets = _sample_budget(samples, max_events_per_sample, max_events_per_class)
    frames: list[pd.DataFrame] = []
    for sample in samples:
        files = parquet_files_for_sample(stage1_root, dataset_subdir, sample.name)
        if max_files_per_sample is not None:
            files = files[:max_files_per_sample]
        budget = budgets.get(sample.name)
        loaded = 0
        for path in files:
            remaining = None if budget is None else max(budget - loaded, 0)
            if remaining == 0:
                break
            frame = _read_one_file(path, sample.class_name, sample.class_index, sample.name, sample.source_group, remaining)
            loaded += len(frame)
            if len(frame):
                frames.append(frame)
    if not frames:
        raise RuntimeError('No events were loaded from the resolved Stage1 samples.')
    out = pd.concat(frames, ignore_index=True)
    if max_events_per_class is not None:
        out['_limit_hash'] = [
            _stable_int(sample, event, source_file, source_row)
            for sample, event, source_file, source_row in zip(
                out['sample'], out['event'], out['_source_file'], out['_source_row']
            )
        ]
        out = out.sort_values('_limit_hash').groupby('true_class_index', group_keys=False).head(max_events_per_class)
        out = out.drop(columns=['_limit_hash']).reset_index(drop=True)
    return out


def assign_splits(df: pd.DataFrame, train_fraction: float, val_fraction: float, seed: int) -> pd.DataFrame:
    out = df.copy()
    hashes = [
        _stable_int(seed, class_index, sample, event, source_file, source_row)
        for class_index, sample, event, source_file, source_row in zip(
            out['true_class_index'], out['sample'], out['event'], out['_source_file'], out['_source_row']
        )
    ]
    out['_split_hash'] = hashes
    out['split'] = 'train'
    pieces: list[pd.DataFrame] = []
    for _, group in out.sort_values('_split_hash').groupby('true_class_index', sort=True):
        n = len(group)
        if n == 1:
            labels = ['train']
        elif n == 2:
            labels = ['train', 'test']
        else:
            n_train = max(1, min(n - 2, int(round(n * train_fraction))))
            n_val = max(1, min(n - n_train - 1, int(round(n * val_fraction))))
            n_test = n - n_train - n_val
            labels = ['train'] * n_train + ['val'] * n_val + ['test'] * n_test
        group = group.copy()
        group['split'] = labels
        pieces.append(group)
    out = pd.concat(pieces, ignore_index=True).drop(columns=['_split_hash'])
    return out


DEFAULT_STANDARDIZED_CLIP = 5.0


def fit_normalization(
    objects: np.ndarray,
    global_features: np.ndarray,
    padding_mask: np.ndarray,
    standardized_clip: float | None = DEFAULT_STANDARDIZED_CLIP,
) -> dict[str, Any]:
    valid = ~padding_mask
    object_values = objects[:, :, :5][valid]
    if object_values.size == 0:
        raise RuntimeError('No valid object tokens available to fit normalization.')
    obj_mean = object_values.mean(axis=0).astype(np.float32)
    obj_std = object_values.std(axis=0).astype(np.float32)
    glob_mean = global_features.mean(axis=0).astype(np.float32)
    glob_std = global_features.std(axis=0).astype(np.float32)
    obj_std = np.where(obj_std > 1.0e-6, obj_std, 1.0).astype(np.float32)
    glob_std = np.where(glob_std > 1.0e-6, glob_std, 1.0).astype(np.float32)
    return {
        'object_continuous_mean': obj_mean.tolist(),
        'object_continuous_std': obj_std.tolist(),
        'global_mean': glob_mean.tolist(),
        'global_std': glob_std.tolist(),
        # Carried with the statistics so evaluate.py, which restores `normalization`
        # straight from the checkpoint, reproduces the training-time clipping exactly.
        'standardized_clip': None if standardized_clip is None else float(standardized_clip),
    }


def apply_normalization(objects: np.ndarray, global_features: np.ndarray, padding_mask: np.ndarray, stats: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    out_obj = objects.copy().astype(np.float32)
    out_global = global_features.copy().astype(np.float32)
    obj_mean = np.asarray(stats['object_continuous_mean'], dtype=np.float32)
    obj_std = np.asarray(stats['object_continuous_std'], dtype=np.float32)
    glob_mean = np.asarray(stats['global_mean'], dtype=np.float32)
    glob_std = np.asarray(stats['global_std'], dtype=np.float32)
    out_obj[:, :, :5] = (out_obj[:, :, :5] - obj_mean) / obj_std
    out_global = (out_global - glob_mean) / glob_std
    out_obj = np.nan_to_num(out_obj, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    out_global = np.nan_to_num(out_global, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    # Clip the standardized values, then zero the padded rows. Order matters: clipping
    # after zeroing would be a no-op on pads only if the bound is positive, and zeroing
    # after clipping keeps padded tokens at exactly 0 regardless of the bound.
    clip = stats.get('standardized_clip', DEFAULT_STANDARDIZED_CLIP)
    if clip is not None:
        bound = float(clip)
        out_obj[:, :, :5] = np.clip(out_obj[:, :, :5], -bound, bound)
        out_global = np.clip(out_global, -bound, bound)
    out_obj[:, :, :5][padding_mask] = 0.0
    return out_obj.astype(np.float32), out_global.astype(np.float32)


def build_bundle(
    df: pd.DataFrame,
    normalization: dict[str, Any] | None = None,
    fit: bool = False,
    standardized_clip: float | None = DEFAULT_STANDARDIZED_CLIP,
) -> tuple[TensorBundle, dict[str, Any]]:
    objects, padding_mask = build_object_features(df)
    global_features = build_global_features(df)
    event_weights, raw_train_weights = build_event_weights(df)
    labels = df['true_class_index'].to_numpy(dtype=np.int64)
    if fit:
        normalization = fit_normalization(objects, global_features, padding_mask, standardized_clip)
        # Weight normalization travels inside `normalization` on purpose: evaluate.py
        # restores that single dict from the checkpoint, so nesting it here means the
        # evaluation path reproduces the training-time weighting with no extra plumbing.
        normalization['weight_normalization'] = fit_class_weight_scales(raw_train_weights, labels)
    if normalization is None:
        raise ValueError('normalization must be supplied unless fit=True')
    weight_stats = normalization.get('weight_normalization')
    if weight_stats is None:
        raise ValueError("normalization is missing 'weight_normalization'; refit with fit=True")
    train_weights = apply_class_weight_scales(raw_train_weights, labels, weight_stats)
    objects, global_features = apply_normalization(objects, global_features, padding_mask, normalization)
    metadata_columns = ['sample', 'source_group', 'true_class', 'true_class_index', 'event', 'run', 'luminosityBlock', '_source_file', '_source_row', 'split']
    metadata = df[[col for col in metadata_columns if col in df.columns]].copy().reset_index(drop=True)
    metadata['source_file'] = metadata['_source_file'] if '_source_file' in metadata else ''
    metadata['source_row'] = metadata['_source_row'] if '_source_row' in metadata else -1
    metadata['data_split'] = metadata['split'] if 'split' in metadata else ''
    metadata['event_weight'] = event_weights
    metadata['train_weight'] = train_weights
    metadata['raw_train_weight'] = raw_train_weights
    bundle = TensorBundle(
        objects=objects,
        global_features=global_features,
        padding_mask=padding_mask.astype(bool),
        labels=labels,
        event_weights=event_weights,
        train_weights=train_weights,
        raw_train_weights=raw_train_weights,
        row_indices=np.arange(len(df), dtype=np.int64),
        metadata=metadata,
    )
    return bundle, normalization


def build_loaders(
    df: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
    standardized_clip: float | None = DEFAULT_STANDARDIZED_CLIP,
) -> tuple[dict[str, DataLoader], dict[str, TensorBundle], dict[str, Any], pd.DataFrame]:
    split_df = assign_splits(df, train_fraction=train_fraction, val_fraction=val_fraction, seed=seed)
    bundles: dict[str, TensorBundle] = {}
    normalization: dict[str, Any] | None = None
    for split in ['train', 'val', 'test']:
        part = split_df[split_df['split'] == split].reset_index(drop=True)
        if split == 'train':
            bundle, normalization = build_bundle(part, fit=True, standardized_clip=standardized_clip)
        else:
            bundle, normalization = build_bundle(part, normalization=normalization, fit=False)
        bundles[split] = bundle
    loaders = {
        split: DataLoader(
            HmumuTensorDataset(bundle),
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, bundle in bundles.items()
    }
    assert normalization is not None
    return loaders, bundles, normalization, split_df


def class_counts(df: pd.DataFrame) -> dict[str, int]:
    return {OUTPUT_COLUMNS[int(idx)].replace('p_', ''): int(count) for idx, count in df['true_class_index'].value_counts().sort_index().items()}


def weight_summary(df: pd.DataFrame) -> dict[str, Any]:
    event_weight, train_weight = build_event_weights(df)
    finite = np.isfinite(event_weight)
    return {
        'event_count': int(len(df)),
        'negative_weight_events': int(np.sum(event_weight < 0.0)),
        'negative_weight_fraction': float(np.mean(event_weight < 0.0)) if len(event_weight) else 0.0,
        'signed_weight_sum': float(np.nansum(event_weight)),
        'absolute_weight_sum': float(np.nansum(np.abs(event_weight))),
        'train_weight_sum': float(np.nansum(train_weight)),
        'min_event_weight': float(np.nanmin(event_weight)) if len(event_weight) else None,
        'max_event_weight': float(np.nanmax(event_weight)) if len(event_weight) else None,
        'nan_weight_count': int(np.sum(np.isnan(event_weight))),
        'infinite_weight_count': int(np.sum(~finite & ~np.isnan(event_weight))),
    }


def preprocessing_diagnostics(
    bundles: dict[str, TensorBundle],
    normalization: dict[str, Any],
    num_classes: int = 3,
) -> dict[str, Any]:
    """Evidence that the weighting and clipping fixes actually took effect.

    Reported per split so the train-only fit and its reuse on val/test are both visible.
    """
    clip = normalization.get('standardized_clip')
    per_split: dict[str, Any] = {}
    for split, bundle in bundles.items():
        raw_sums = class_weight_sums(bundle.raw_train_weights, bundle.labels, num_classes)
        normalized_sums = class_weight_sums(bundle.train_weights, bundle.labels, num_classes)
        positive_raw = [value for value in raw_sums if value > 0.0]
        positive_norm = [value for value in normalized_sums if value > 0.0]
        continuous = bundle.objects[:, :, :5][~bundle.padding_mask]
        obj_abs_max = float(np.abs(continuous).max()) if continuous.size else None
        glob_abs_max = float(np.abs(bundle.global_features).max()) if bundle.global_features.size else None
        per_split[split] = {
            'n_events': int(bundle.labels.shape[0]),
            'class_event_counts': [int((bundle.labels == index).sum()) for index in range(num_classes)],
            'raw_class_weight_sums': raw_sums,
            'normalized_class_weight_sums': normalized_sums,
            'raw_class_weight_sum_ratio': (
                float(max(positive_raw) / min(positive_raw)) if len(positive_raw) > 1 else None
            ),
            'normalized_class_weight_sum_ratio': (
                float(max(positive_norm) / min(positive_norm)) if len(positive_norm) > 1 else None
            ),
            'max_abs_standardized_object_feature': obj_abs_max,
            'max_abs_standardized_global_feature': glob_abs_max,
            'clip_bound_respected': (
                None if clip is None
                else bool(
                    (obj_abs_max is None or obj_abs_max <= float(clip) + 1.0e-5)
                    and (glob_abs_max is None or glob_abs_max <= float(clip) + 1.0e-5)
                )
            ),
        }
    return {
        'standardized_clip': clip,
        'weight_normalization': normalization.get('weight_normalization'),
        'per_split': per_split,
    }


def weight_summary_rows(df: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(weight_summary(group))
        rows.append(row)
    return rows
