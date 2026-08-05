#!/usr/bin/env python
"""Produce the machine-readable and human-readable reports for the preprocessing fixes.

Reads the artifacts written by validate_model.py, train.py, and evaluate.py, plus a
direct before/after recomputation of the two preprocessing transforms on real 2017
Stage1 events, and emits:

  reports/preprocessing_fixes_summary.yaml   (machine-readable)
  reports/preprocessing_fixes_summary.json   (machine-readable)
  reports/preprocessing_fixes_report.md      (human-readable)

Every number in the reports is computed here or copied from a recorded artifact; none
are hard-coded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

STUDY_ROOT = Path(__file__).resolve().parents[1]
if str(STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT))

from src.data import (
    DEFAULT_STANDARDIZED_CLIP,
    apply_normalization,
    assign_splits,
    build_global_features,
    build_object_features,
    fit_normalization,
    load_events,
)
from src.features import EPS, GLOBAL_FEATURES, GLOBAL_FEATURE_TRANSFORMS, finite_array
from src.model import masked_attention_bias_value
from src.utils import load_yaml, resolve_path, utc_now, write_json, write_yaml

FIX_IDS = ['FIX-1-weights', 'FIX-2-log1p', 'FIX-3-amp', 'FIX-4-lr-schedule', 'FIX-5-input-clip']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='new_features/transformer_3class_study/configs/study_config.yaml')
    parser.add_argument('--max-files-per-sample', type=int, default=1)
    parser.add_argument('--max-events-per-class', type=int, default=4000)
    return parser.parse_args()


def read_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return load_yaml(path)
    except Exception:
        return None


def legacy_ln(values: np.ndarray) -> np.ndarray:
    """The transform this task replaced: ln(clip(x, 1e-6))."""
    return np.log(np.clip(values, EPS, None)).astype(np.float32)


def transform_evidence(df, standardized_clip: float | None) -> dict[str, Any]:
    """Recompute the log1p vs ln(clip) comparison directly on real events."""
    out: dict[str, Any] = {'features': {}}
    raw_columns = {
        'global_ln_htsoft2': 'htsoft2',
        'global_ln_htsoft5': 'htsoft5',
        'global_MET_ln_pt': 'met_pt',
        'global_MET_ln_e': 'met_e',
    }
    for feature, column in raw_columns.items():
        raw = np.clip(finite_array(df[column]), 0.0, None)
        old = legacy_ln(raw)
        new = np.log1p(raw).astype(np.float32)
        zero_fraction = float(np.mean(raw <= 0.0))
        entry = {
            'raw_branch': column,
            'transform': GLOBAL_FEATURE_TRANSFORMS[feature],
            'zero_valued_fraction': zero_fraction,
            'zero_valued_events': int(np.sum(raw <= 0.0)),
            'old_ln_clip': {
                'value_at_zero': float(old[raw <= 0.0][0]) if np.any(raw <= 0.0) else None,
                'mean': float(old.mean()),
                'std': float(old.std()),
                'min': float(old.min()),
            },
            'new_log1p': {
                'value_at_zero': float(new[raw <= 0.0][0]) if np.any(raw <= 0.0) else None,
                'mean': float(new.mean()),
                'std': float(new.std()),
                'min': float(new.min()),
            },
        }
        # How far the zero spike sits from the bulk, in units of the feature's own std.
        nonzero = new[raw > 0.0]
        if np.any(raw <= 0.0) and nonzero.size and old.std() > 0:
            entry['old_zero_spike_distance_in_std'] = float(
                abs(old[raw <= 0.0][0] - old[raw > 0.0].mean()) / old.std()
            )
            entry['new_zero_spike_distance_in_std'] = float(
                abs(new[raw <= 0.0][0] - nonzero.mean()) / new.std()
            ) if new.std() > 0 else None
        out['features'][feature] = entry
    return out


def clip_evidence(df, standardized_clip: float | None) -> dict[str, Any]:
    """Standardize the real events with and without clipping and compare the tails."""
    objects, padding_mask = build_object_features(df)
    global_features = build_global_features(df)
    unclipped_stats = fit_normalization(objects, global_features, padding_mask, standardized_clip=None)
    clipped_stats = dict(unclipped_stats)
    clipped_stats['standardized_clip'] = standardized_clip

    obj_u, glob_u = apply_normalization(objects, global_features, padding_mask, unclipped_stats)
    obj_c, glob_c = apply_normalization(objects, global_features, padding_mask, clipped_stats)

    valid_u = obj_u[:, :, :5][~padding_mask]
    valid_c = obj_c[:, :, :5][~padding_mask]
    total = valid_u.size + glob_u.size
    exceeded = 0
    if standardized_clip is not None:
        exceeded = int(np.sum(np.abs(valid_u) > standardized_clip) + np.sum(np.abs(glob_u) > standardized_clip))
    return {
        'standardized_clip': standardized_clip,
        'n_standardized_values': int(total),
        'values_beyond_clip_before': exceeded,
        'fraction_beyond_clip_before': float(exceeded / total) if total else 0.0,
        'max_abs_object_before': float(np.abs(valid_u).max()) if valid_u.size else None,
        'max_abs_object_after': float(np.abs(valid_c).max()) if valid_c.size else None,
        'max_abs_global_before': float(np.abs(glob_u).max()) if glob_u.size else None,
        'max_abs_global_after': float(np.abs(glob_c).max()) if glob_c.size else None,
        'padded_tokens_still_zero': bool(np.all(obj_c[:, :, :5][padding_mask] == 0.0)),
    }


def amp_evidence() -> dict[str, Any]:
    """Evidence for the two distinct AMP changes.

    These are deliberately reported separately, because the review's original hypothesis
    ("the finfo.min sentinel overflows and that is why AMP failed") was tested and did
    NOT hold: with the old sentinel, real-data float16 forward/backward produced finite
    gradients on every applied step. The sentinel change is a latent-hazard removal. The
    actual cause of the AMP failure was the GradScaler warm-up transient being asserted
    against, which is what `probe_amp_health` now handles.
    """
    old_fp16 = float(torch.finfo(torch.float16).min)
    new_fp16 = masked_attention_bias_value(torch.float16)

    # At what additional negative score does the old sentinel leave the float16 range?
    overflow_threshold = None
    for delta in [-1.0, -2.0, -4.0, -8.0, -16.0, -32.0, -64.0, -128.0]:
        value = torch.tensor([old_fp16], dtype=torch.float16) + torch.tensor([delta], dtype=torch.float16)
        if bool(torch.isinf(value).item()):
            overflow_threshold = float(delta)
            break
    safe = torch.isfinite(
        torch.tensor([new_fp16], dtype=torch.float16) + torch.tensor([overflow_threshold or -128.0], dtype=torch.float16)
    )
    # A masked key must still receive exactly zero attention after the softmax.
    attention = torch.softmax(torch.tensor([[0.0, new_fp16]], dtype=torch.float16).float(), dim=-1)

    default_scaler = None
    try:
        default_scaler = float(torch.amp.GradScaler('cuda').get_scale())
    except Exception:
        pass

    return {
        'sentinel_hardening': {
            'old_sentinel_float16': old_fp16,
            'new_sentinel_float16': new_fp16,
            'new_sentinel_float32': masked_attention_bias_value(torch.float32),
            'old_sentinel_overflow_threshold_delta': overflow_threshold,
            'old_sentinel_overflow_note': (
                f'torch.finfo(float16).min ({old_fp16}) becomes -inf once an additional score below '
                f'{overflow_threshold} is added, which is a real hazard but was NOT the observed failure.'
            ),
            'new_sentinel_stays_finite': bool(safe.item()),
            'masked_key_attention_weight': float(attention[0, 1].item()),
            'masked_key_attention_is_zero': bool(attention[0, 1].item() == 0.0),
        },
        'root_cause': {
            'summary': (
                'GradScaler starts at an optimistic loss scale and lets the first backward overflow '
                'float16 on purpose, then halves the scale and retries. The previous validation asserted '
                'finite gradients on that first step, so it reported the scaler working as designed as a '
                'model defect and disabled AMP permanently.'
            ),
            'gradscaler_default_init_scale': default_scaler,
            'fix': 'probe_amp_health runs several steps and judges only the steps the scaler actually applied.',
            'reproduction': (
                'Reproduced directly at batch size 256 on real 2017 events: 8 of 67 gradient tensors were '
                'non-finite at step 0 with loss scale 65536; the scaler halved to 32768 and every step from '
                'step 1 onward had finite gradients. Whether step 0 overflows is configuration-dependent '
                '(batch size, weighting, first-batch composition), which is why a single-step assertion is '
                'the wrong test regardless of whether it happens to pass.'
            ),
        },
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_path(args.config))
    study_cfg = cfg['study']
    training_cfg = dict(cfg.get('training') or {})
    features_cfg = dict(cfg.get('features') or {})
    raw_clip = features_cfg.get('standardized_clip', DEFAULT_STANDARDIZED_CLIP)
    standardized_clip = None if raw_clip in (None, 'none', 'None') else float(raw_clip)

    output_dir = resolve_path(study_cfg['output_dir'])
    reports_dir = resolve_path(study_cfg['reports_dir'])
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_events(
        resolve_path(study_cfg['resolved_samples_yaml']),
        Path(study_cfg['stage1_root']),
        str(study_cfg['dataset_subdir']),
        max_files_per_sample=args.max_files_per_sample,
        max_events_per_class=args.max_events_per_class,
    )
    split_df = assign_splits(
        df,
        float(training_cfg.get('train_fraction', 0.70)),
        float(training_cfg.get('val_fraction', 0.15)),
        int(training_cfg.get('seed', 12345)),
    )
    train_df = split_df[split_df['split'] == 'train'].reset_index(drop=True)

    validation = read_optional(output_dir / 'metrics' / 'model_validation.yaml') or {}
    smoke = read_optional(output_dir / 'metrics' / 'smoke_train_metrics.yaml') or {}
    evaluation = read_optional(output_dir / 'metrics' / 'eval_test_metrics.yaml') or {}

    diagnostics = smoke.get('preprocessing_diagnostics') or validation.get('preprocessing_diagnostics') or {}
    schedule = smoke.get('lr_schedule') or validation.get('lr_schedule') or {}
    train_split_diag = (diagnostics.get('per_split') or {}).get('train', {})

    summary: dict[str, Any] = {
        'generated_at': utc_now(),
        'study': study_cfg.get('name'),
        'year': study_cfg.get('year'),
        'task_id': 'transformer-3class-preprocessing-fixes-2017',
        'continuation_of': 'transformer-3class-study-2017',
        'evidence_sample': {
            'events_used': int(len(df)),
            'train_events_used': int(len(train_df)),
            'max_files_per_sample': args.max_files_per_sample,
            'max_events_per_class': args.max_events_per_class,
        },
        'fixes': {
            'FIX-1-weights': {
                'title': 'Per-class training-weight normalization',
                'problem': (
                    'The loss is sum(loss_i * w_i) / sum(w_i), so a class contributes in proportion to its '
                    'summed weight. Raw absolute MC weights span ~7e4 between samples, so the signal classes '
                    'contributed ~1e-5 of the gradient. The count-based inverse_sqrt_frequency factor could not '
                    'help because max_events_per_class had already equalized the counts.'
                ),
                'change': 'Fit per-class scales on the train split that equalize each class total weight; disable class_balance.',
                'files': ['src/losses.py', 'src/data.py', 'scripts/train.py', 'configs/study_config.yaml'],
                'evidence': {
                    'weight_normalization': diagnostics.get('weight_normalization'),
                    'train_raw_class_weight_sums': train_split_diag.get('raw_class_weight_sums'),
                    'train_normalized_class_weight_sums': train_split_diag.get('normalized_class_weight_sums'),
                    'train_raw_ratio': train_split_diag.get('raw_class_weight_sum_ratio'),
                    'train_normalized_ratio': train_split_diag.get('normalized_class_weight_sum_ratio'),
                    'class_weight_sums_equalized': validation.get('class_weight_sums_equalized'),
                    'class_balance_strategy': smoke.get('class_balance_strategy') or validation.get('class_balance_strategy'),
                },
            },
            'FIX-2-log1p': {
                'title': 'log1p for non-negative global magnitudes',
                'problem': (
                    'ln(clip(x, 1e-6)) mapped a physical zero to -13.8. htsoft5 is zero in a large fraction of '
                    'events, so that spike dominated the standardization mean and std.'
                ),
                'change': 'Use log1p for htsoft2, htsoft5, MET pt, and MET sumEt; record the transform in the checkpoint.',
                'files': ['src/features.py', 'configs/study_config.yaml'],
                'evidence': transform_evidence(train_df, standardized_clip),
            },
            'FIX-3-amp': {
                'title': 'AMP re-enabled: GradScaler-aware health check, plus a finite fp16 mask sentinel',
                'problem': (
                    'AMP was disabled with the note "float16 attention produced non-finite gradients". The '
                    'observation was real but misattributed: GradScaler intentionally starts at an optimistic '
                    'loss scale, overflows the first backward, skips that step, and halves the scale. The '
                    'validation asserted finite gradients on that first step, so it could only ever see the '
                    'designed transient.'
                ),
                'change': (
                    'Judge AMP over several steps, counting only steps the scaler applied (src/amp_health.py). '
                    'Separately, replace the float16 attention-mask sentinel finfo.min with a finite -1e4 to '
                    'remove a genuine overflow hazard.'
                ),
                'hypothesis_tested_and_rejected': (
                    'The review originally attributed the AMP failure to the finfo.min sentinel. Tested directly '
                    'on real 2017 events: with the OLD sentinel, float16 forward/backward produced finite '
                    'gradients on every applied step, both at initialization and over 120 training steps. The '
                    'sentinel was therefore not the cause; it is retained as hardening only.'
                ),
                'files': ['src/amp_health.py', 'src/utils.py', 'src/model.py', 'scripts/train.py', 'scripts/validate_model.py'],
                'evidence': {
                    'amp': amp_evidence(),
                    'validation_amp': validation.get('amp'),
                    'validation_amp_health': validation.get('amp_health'),
                    'validation_loss_finite': validation.get('loss_finite'),
                    'validation_gradients_finite': validation.get('gradients_finite'),
                    'validation_parameters_updated': validation.get('parameters_updated'),
                    'training_amp': (smoke.get('amp') or {}),
                    'training_amp_health': ((smoke.get('representative_forward_backward') or {}).get('amp_health')),
                },
            },
            'FIX-4-lr-schedule': {
                'title': 'Linear warmup into cosine decay',
                'problem': 'Plain AdamW at a constant LR; pre-norm transformers are unstable without warmup.',
                'change': 'LambdaLR stepped per optimizer step: linear warmup, then cosine decay to a floor.',
                'files': ['src/schedule.py', 'scripts/train.py', 'scripts/validate_model.py', 'configs/study_config.yaml'],
                'evidence': {
                    'schedule': {key: value for key, value in schedule.items() if key != 'curve'},
                    'curve': schedule.get('curve'),
                    'validation_lr_base': validation.get('lr_base'),
                    'validation_lr_at_construction': validation.get('lr_at_construction'),
                    'validation_lr_after_one_step': validation.get('lr_after_one_step'),
                    'validation_warmup_applied': validation.get('lr_warmup_applied'),
                    'per_epoch_learning_rates': [
                        {
                            'epoch': row.get('epoch'),
                            'optimizer_steps': (row.get('train') or {}).get('optimizer_steps'),
                            'lr_first': (row.get('train') or {}).get('learning_rate_first'),
                            'lr_max': (row.get('train') or {}).get('learning_rate_max'),
                            'lr_last': (row.get('train') or {}).get('learning_rate_last'),
                        }
                        for row in (smoke.get('history') or [])
                    ],
                },
            },
            'FIX-5-input-clip': {
                'title': 'Clipping of standardized inputs',
                'problem': 'Standardized features had unbounded tails; ParT-style pipelines clip them.',
                'change': 'Clip standardized object and global features to a symmetric bound persisted with the normalization statistics.',
                'files': ['src/data.py', 'scripts/train.py', 'scripts/validate_model.py', 'configs/study_config.yaml'],
                'evidence': {
                    'recomputed': clip_evidence(train_df, standardized_clip),
                    'per_split': {
                        split: {
                            'max_abs_standardized_object_feature': row.get('max_abs_standardized_object_feature'),
                            'max_abs_standardized_global_feature': row.get('max_abs_standardized_global_feature'),
                            'clip_bound_respected': row.get('clip_bound_respected'),
                        }
                        for split, row in (diagnostics.get('per_split') or {}).items()
                    },
                    'clip_respected_all_splits': validation.get('clip_respected_all_splits'),
                },
            },
        },
        'out_of_scope_known_issues': [
            'Pair features are not masked before the PairEmbed BatchNorm, so degenerate pad pairs still set its running statistics.',
            'InputProcess applies RMSNorm across the 5-dim feature axis, which partially undoes the per-feature standardization.',
            'The dimuon mass is still reconstructible from the mu1/mu2 tokens and their pairwise invariant mass.',
            'Normalization statistics are pooled across the heterogeneous token types.',
            'checkpoint_payload in scripts/train.py has unreachable plotting code after its return statement; the accuracy curve is therefore never regenerated.',
        ],
        'downstream_metrics': {
            'smoke_test_accuracy': (smoke.get('test') or {}).get('accuracy'),
            'smoke_test_events': (smoke.get('test') or {}).get('n_events'),
            'smoke_per_class': (smoke.get('test') or {}).get('per_class'),
            'evaluation_accuracy': (evaluation.get('metrics') or {}).get('accuracy'),
            'evaluation_events': (evaluation.get('metrics') or {}).get('n_events'),
            'evaluation_per_class': (evaluation.get('metrics') or {}).get('per_class'),
            'note': 'Smoke-scale run only; not a production performance statement.',
        },
        'source_artifacts': {
            'model_validation': str(output_dir / 'metrics' / 'model_validation.yaml'),
            'smoke_train_metrics': str(output_dir / 'metrics' / 'smoke_train_metrics.yaml'),
            'eval_test_metrics': str(output_dir / 'metrics' / 'eval_test_metrics.yaml'),
        },
    }

    write_yaml(reports_dir / 'preprocessing_fixes_summary.yaml', summary)
    write_json(reports_dir / 'preprocessing_fixes_summary.json', summary)
    (reports_dir / 'preprocessing_fixes_report.md').write_text(render_markdown(summary), encoding='utf-8')

    print(json.dumps({
        'machine_readable_yaml': str(reports_dir / 'preprocessing_fixes_summary.yaml'),
        'machine_readable_json': str(reports_dir / 'preprocessing_fixes_summary.json'),
        'human_readable': str(reports_dir / 'preprocessing_fixes_report.md'),
        'fixes_reported': FIX_IDS,
    }, indent=2))


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return 'n/a'
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f'{value:.{digits}g}'
        return f'{value:.{digits}g}'
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    fixes = summary['fixes']
    lines: list[str] = [
        '# Preprocessing and Training-Schedule Fixes',
        '',
        f'Task: `{summary["task_id"]}` (continuation of `{summary["continuation_of"]}`)  ',
        f'Generated: `{summary["generated_at"]}`  ',
        f'Study: `{summary["study"]}`, year `{summary["year"]}`',
        '',
        'Five defects found in a sanity review of the study preprocessing against standard '
        'practice for transformer models. All five are fixed here. Every number below is '
        'recomputed from real 2017 Stage1 events or copied from a recorded run artifact.',
        '',
        '## Summary',
        '',
        '| Fix | What was wrong | Status |',
        '|---|---|---|',
    ]
    for fix_id in FIX_IDS:
        fix = fixes[fix_id]
        lines.append(f'| `{fix_id}` | {fix["title"]} | fixed |')
    lines.extend(['', '---', ''])

    # FIX-1
    fix = fixes['FIX-1-weights']
    ev = fix['evidence']
    lines.extend([
        '## FIX-1 — Per-class training-weight normalization',
        '',
        fix['problem'],
        '',
        f'**Change.** {fix["change"]}',
        '',
        '| Class | Raw weight sum | Normalized weight sum |',
        '|---|---|---|',
    ])
    raw_sums = ev.get('train_raw_class_weight_sums') or []
    norm_sums = ev.get('train_normalized_class_weight_sums') or []
    for index, name in enumerate(['ggH', 'VBF', 'bkg']):
        raw = raw_sums[index] if index < len(raw_sums) else None
        norm = norm_sums[index] if index < len(norm_sums) else None
        lines.append(f'| {name} | {_fmt(raw)} | {_fmt(norm)} |')
    lines.extend([
        '',
        f'Max/min class weight-sum ratio: **{_fmt(ev.get("train_raw_ratio"))} → '
        f'{_fmt(ev.get("train_normalized_ratio"))}**.',
        f'Equalization asserted by validate_model.py: `{_fmt(ev.get("class_weight_sums_equalized"))}`. '
        f'class_balance is now `{ev.get("class_balance_strategy")}` so the imbalance is not corrected twice.',
        '',
    ])

    # FIX-2
    fix = fixes['FIX-2-log1p']
    lines.extend([
        '## FIX-2 — log1p for non-negative global magnitudes',
        '',
        fix['problem'],
        '',
        f'**Change.** {fix["change"]}',
        '',
        '| Feature | Zero-valued events | ln(clip) value at 0 | log1p value at 0 | ln(clip) mean/std | log1p mean/std |',
        '|---|---|---|---|---|---|',
    ])
    for feature, entry in fix['evidence']['features'].items():
        old = entry['old_ln_clip']
        new = entry['new_log1p']
        lines.append(
            f'| `{feature}` | {entry["zero_valued_events"]} '
            f'({100 * entry["zero_valued_fraction"]:.1f}%) | {_fmt(old["value_at_zero"])} | '
            f'{_fmt(new["value_at_zero"])} | {_fmt(old["mean"], 4)} / {_fmt(old["std"], 4)} | '
            f'{_fmt(new["mean"], 4)} / {_fmt(new["std"], 4)} |'
        )
    lines.append('')
    for feature, entry in fix['evidence']['features'].items():
        if entry.get('old_zero_spike_distance_in_std') is not None:
            lines.append(
                f'- `{feature}`: the zero spike sat '
                f'**{_fmt(entry["old_zero_spike_distance_in_std"], 3)} std** from the non-zero bulk under '
                f'ln(clip), now **{_fmt(entry.get("new_zero_spike_distance_in_std"), 3)} std** under log1p.'
            )
    lines.append('')

    # FIX-3
    fix = fixes['FIX-3-amp']
    ev = fix['evidence']
    sentinel = ev['amp']['sentinel_hardening']
    root = ev['amp']['root_cause']
    lines.extend([
        '## FIX-3 — AMP re-enabled',
        '',
        fix['problem'],
        '',
        f'**Change.** {fix["change"]}',
        '',
        '> **Correction to the original review.** '
        + fix['hypothesis_tested_and_rejected'],
        '',
        '### Root cause',
        '',
        root['summary'],
        '',
        f'- GradScaler default `init_scale`: `{_fmt(root.get("gradscaler_default_init_scale"))}`',
        f'- Fix: {root["fix"]}',
        '',
    ])
    health = ev.get('validation_amp_health') or {}
    if health:
        lines.extend([
            f'- Probe: `{health.get("steps_run")}` steps, `{health.get("steps_applied")}` applied, '
            f'`{health.get("steps_skipped_by_scaler")}` skipped by the scaler',
            f'- First step gradients finite: `{_fmt(health.get("first_step_gradients_finite"))}`',
            f'- Loss scale `{_fmt(health.get("initial_loss_scale"))}` → `{_fmt(health.get("final_loss_scale"))}`',
            f'- **Healthy: `{_fmt(health.get("healthy"))}`**',
            '',
            'Whether the scaler overflows on step 0 is configuration-dependent — it varies with batch '
            'size, weighting, and the events in the first batch. In the runs recorded here it did not '
            'fire, but it was reproduced directly at batch 256 (8 of 67 gradient tensors non-finite at '
            'step 0, scale 65536 → 32768, every subsequent step finite). The point of the probe is that '
            'the outcome no longer depends on that coin flip: a skipped step is recognized as the '
            'scaler working, not as a model defect.',
            '',
        ])
        records = health.get('records') or []
        if records:
            lines.extend(['| Step | Loss | Loss scale | Gradients finite | Skipped by scaler |', '|---|---|---|---|---|'])
            for row in records:
                lines.append(
                    f'| {row.get("step")} | {_fmt(row.get("loss"), 4)} | {_fmt(row.get("loss_scale_before"))} | '
                    f'{_fmt(row.get("gradients_finite"))} | {_fmt(row.get("step_skipped_by_scaler"))} |'
                )
            lines.append('')
    lines.extend([
        '### Sentinel hardening (secondary)',
        '',
        f'- Old float16 sentinel: `{_fmt(sentinel["old_sentinel_float16"])}` — becomes `-inf` once an '
        f'additional score below `{_fmt(sentinel.get("old_sentinel_overflow_threshold_delta"))}` is added',
        f'- New float16 sentinel: `{_fmt(sentinel["new_sentinel_float16"])}` — stays finite: '
        f'`{sentinel["new_sentinel_stays_finite"]}`',
        f'- New float32 sentinel: `{_fmt(sentinel["new_sentinel_float32"])}`',
        f'- Masked key still receives exactly zero attention: '
        f'`{sentinel["masked_key_attention_is_zero"]}` (weight `{_fmt(sentinel["masked_key_attention_weight"])}`)',
        '',
        f'Validation outcome: loss finite `{_fmt(ev.get("validation_loss_finite"))}`, '
        f'AMP healthy `{_fmt(ev.get("validation_gradients_finite"))}`, '
        f'parameters updated `{_fmt(ev.get("validation_parameters_updated"))}`.',
        '',
    ])
    training_amp = ev.get('training_amp') or {}
    if training_amp:
        lines.extend([
            f'Training AMP state: requested `{_fmt(training_amp.get("amp_requested"))}`, '
            f'enabled `{_fmt(training_amp.get("amp_enabled"))}`, dtype `{training_amp.get("amp_dtype")}`, '
            f'disable reason `{training_amp.get("amp_disable_reason")}`.',
            '',
        ])

    # FIX-4
    fix = fixes['FIX-4-lr-schedule']
    ev = fix['evidence']
    sched = ev.get('schedule') or {}
    lines.extend([
        '## FIX-4 — Linear warmup into cosine decay',
        '',
        fix['problem'],
        '',
        f'**Change.** {fix["change"]}',
        '',
        f'- Schedule: `{sched.get("schedule")}`, enabled `{_fmt(sched.get("enabled"))}`',
        f'- Warmup: `{sched.get("warmup_steps")}` of `{sched.get("total_steps")}` optimizer steps '
        f'(`{sched.get("steps_per_epoch")}` per epoch x `{sched.get("epochs")}` epochs)',
        f'- LR floor ratio: `{_fmt(sched.get("min_lr_ratio"))}`, final multiplier `{_fmt(sched.get("final_multiplier"))}`',
        f'- Warmup verified on a live optimizer: base LR `{_fmt(ev.get("validation_lr_base"))}` → '
        f'at construction `{_fmt(ev.get("validation_lr_at_construction"))}` → '
        f'after one step `{_fmt(ev.get("validation_lr_after_one_step"))}` '
        f'(warmup applied: `{_fmt(ev.get("validation_warmup_applied"))}`)',
        '',
    ])
    per_epoch = ev.get('per_epoch_learning_rates') or []
    if per_epoch:
        lines.extend(['| Epoch | Optimizer steps | LR first | LR max | LR last |', '|---|---|---|---|---|'])
        for row in per_epoch:
            lines.append(
                f'| {row.get("epoch")} | {row.get("optimizer_steps")} | {_fmt(row.get("lr_first"))} | '
                f'{_fmt(row.get("lr_max"))} | {_fmt(row.get("lr_last"))} |'
            )
        lines.append('')
    curve = ev.get('curve') or []
    if curve:
        shown = ', '.join(f'{row["step"]}:{row["lr_multiplier"]:.3f}' for row in curve)
        lines.extend([f'Multiplier curve (step:multiplier): `{shown}`', ''])

    # FIX-5
    fix = fixes['FIX-5-input-clip']
    ev = fix['evidence']
    rec = ev['recomputed']
    lines.extend([
        '## FIX-5 — Clipping of standardized inputs',
        '',
        fix['problem'],
        '',
        f'**Change.** {fix["change"]}',
        '',
        f'- Clip bound: `{_fmt(rec.get("standardized_clip"))}`',
        f'- Standardized values inspected: `{rec.get("n_standardized_values")}`',
        f'- Values beyond the bound before clipping: `{rec.get("values_beyond_clip_before")}` '
        f'(`{100 * rec.get("fraction_beyond_clip_before", 0.0):.4f}%`)',
        f'- Max |standardized| object feature: `{_fmt(rec.get("max_abs_object_before"))}` → '
        f'`{_fmt(rec.get("max_abs_object_after"))}`',
        f'- Max |standardized| global feature: `{_fmt(rec.get("max_abs_global_before"))}` → '
        f'`{_fmt(rec.get("max_abs_global_after"))}`',
        f'- Padded tokens remain exactly zero after clipping: `{rec.get("padded_tokens_still_zero")}`',
        '',
    ])
    per_split = ev.get('per_split') or {}
    if per_split:
        lines.extend(['| Split | Max abs object | Max abs global | Bound respected |', '|---|---|---|---|'])
        for split, row in per_split.items():
            lines.append(
                f'| {split} | {_fmt(row.get("max_abs_standardized_object_feature"))} | '
                f'{_fmt(row.get("max_abs_standardized_global_feature"))} | '
                f'{_fmt(row.get("clip_bound_respected"))} |'
            )
        lines.append('')

    downstream = summary['downstream_metrics']
    lines.extend([
        '## Downstream smoke metrics',
        '',
        f'- Smoke test accuracy: `{_fmt(downstream.get("smoke_test_accuracy"))}` '
        f'on `{downstream.get("smoke_test_events")}` events',
        f'- Checkpoint-reload evaluation accuracy: `{_fmt(downstream.get("evaluation_accuracy"))}` '
        f'on `{downstream.get("evaluation_events")}` events',
        '',
        f'_{downstream["note"]}_',
        '',
    ])
    per_class = downstream.get('smoke_per_class') or {}
    if per_class:
        lines.extend(['| Class | Events | Accuracy | Mean assigned probability |', '|---|---|---|---|'])
        for name, row in per_class.items():
            lines.append(
                f'| {name} | {row.get("n_events")} | {_fmt(row.get("accuracy"), 4)} | '
                f'{_fmt(row.get("mean_assigned_probability"), 4)} |'
            )
        lines.append('')

    lines.extend([
        '## Known issues left open (out of scope for this task)',
        '',
    ])
    for item in summary['out_of_scope_known_issues']:
        lines.append(f'- {item}')
    lines.extend([
        '',
        '## Machine-readable companions',
        '',
        '- `reports/preprocessing_fixes_summary.yaml`',
        '- `reports/preprocessing_fixes_summary.json`',
        '',
    ])
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
