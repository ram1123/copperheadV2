#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

STUDY_ROOT = Path(__file__).resolve().parents[1]
if str(STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT))

from src.amp_health import probe_amp_health
from src.data import DEFAULT_STANDARDIZED_CLIP, build_loaders, load_events, preprocessing_diagnostics
from src.losses import inverse_sqrt_class_weights, weighted_cross_entropy
from src.model import build_model, masked_attention_bias_value
from src.schedule import build_lr_scheduler, sample_schedule
from src.utils import (
    amp_info,
    cuda_cleanup,
    cuda_memory_snapshot,
    cuda_reset_peak,
    load_yaml,
    make_grad_scaler,
    resolve_path,
    runtime_config,
    select_device,
    set_seed,
    write_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a real-data forward/backward validation for the transformer smoke model.')
    parser.add_argument('--config', default='new_features/transformer_3class_study/configs/study_config.yaml')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default=None)
    parser.add_argument('--cuda-memory-limit-gib', type=float, default=None)
    parser.add_argument('--max-files-per-sample', type=int, default=1)
    parser.add_argument('--max-events-per-class', type=int, default=600)
    parser.add_argument('--batch-size', type=int, default=64)
    return parser.parse_args()


def autocast_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True)
    return nullcontext()


def _sums_are_equal(sums: list[float], rel_tol: float = 1.0e-5) -> bool:
    """True when every populated class carries the same total training weight."""
    populated = [value for value in sums if value > 0.0]
    if len(populated) < 2:
        return True
    return (max(populated) - min(populated)) <= rel_tol * max(populated)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_path(args.config))
    study_cfg = cfg['study']
    training_cfg = cfg.get('training', {})
    rt_cfg = runtime_config(cfg)
    cuda_limit = float(args.cuda_memory_limit_gib if args.cuda_memory_limit_gib is not None else rt_cfg['cuda']['max_memory_gib'])
    device_name = args.device or rt_cfg.get('device', 'auto')

    set_seed(int(training_cfg.get('seed', study_cfg.get('seed', 12345))))
    device, device_info = select_device(device_name, cuda_limit)
    amp_summary = amp_info(device, bool(rt_cfg['cuda'].get('automatic_mixed_precision', True)))
    amp_summary['amp_requested'] = bool(rt_cfg['cuda'].get('automatic_mixed_precision', True))
    amp_summary['masked_attention_bias_float16'] = masked_attention_bias_value(torch.float16)
    amp_summary['masked_attention_bias_float32'] = masked_attention_bias_value(torch.float32)
    amp_summary['full_precision_required'] = None
    # This script is the representative float16 check that train.py's AMP fallback keys
    # off. It deliberately does NOT pre-disable AMP: the finite-loss / finite-gradient
    # assertion at the end of main() is the evidence, and it must be allowed to fail.
    use_amp = bool(amp_summary['amp_enabled'])

    resolved_samples = resolve_path(study_cfg['resolved_samples_yaml'])
    if not resolved_samples.exists():
        raise FileNotFoundError(f'Resolved sample file is missing: {resolved_samples}. Run inspect_stage1_inputs.py first.')
    df = load_events(
        resolved_samples,
        Path(study_cfg['stage1_root']),
        str(study_cfg['dataset_subdir']),
        max_files_per_sample=args.max_files_per_sample,
        max_events_per_class=args.max_events_per_class,
    )
    features_cfg = dict(cfg.get('features') or {})
    raw_clip = features_cfg.get('standardized_clip', DEFAULT_STANDARDIZED_CLIP)
    standardized_clip = None if raw_clip in (None, 'none', 'None') else float(raw_clip)
    loaders, bundles, normalization, _ = build_loaders(
        df,
        batch_size=args.batch_size,
        num_workers=int(training_cfg.get('num_workers', 0)),
        train_fraction=float(training_cfg.get('train_fraction', 0.70)),
        val_fraction=float(training_cfg.get('val_fraction', 0.15)),
        seed=int(training_cfg.get('seed', 12345)),
        standardized_clip=standardized_clip,
    )
    diagnostics = preprocessing_diagnostics(bundles, normalization)

    model = build_model(cfg.get('model', {})).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get('learning_rate', 1.0e-4)),
        weight_decay=float(training_cfg.get('weight_decay', 0.01)),
    )
    class_balance = str(training_cfg.get('class_balance', 'none') or 'none').lower()
    class_weight = (
        inverse_sqrt_class_weights(torch.from_numpy(bundles['train'].labels), num_classes=3).to(device)
        if class_balance == 'inverse_sqrt_frequency'
        else None
    )

    # Exercise the schedule on this optimizer so warmup/decay is validated, not just built.
    # LambdaLR rescales param_groups[0]['lr'] at construction, so capture the base first.
    base_lr = float(optimizer.param_groups[0]['lr'])
    # Use the configured epoch count, not a single epoch, so the schedule built here has
    # the same shape as the one training will use and the warmup ramp is observable.
    steps_per_epoch = max(1, len(loaders['train']))
    total_steps = steps_per_epoch * max(1, int(training_cfg.get('epochs', 2)))
    scheduler, schedule_summary = build_lr_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=training_cfg.get('warmup_steps'),
        warmup_fraction=float(training_cfg.get('warmup_fraction', 0.05)),
        schedule=str(training_cfg.get('lr_schedule', 'cosine')),
        min_lr_ratio=float(training_cfg.get('min_lr_ratio', 0.01)),
    )
    schedule_summary['steps_per_epoch'] = steps_per_epoch
    schedule_summary['epochs'] = int(training_cfg.get('epochs', 2))
    schedule_summary['curve'] = sample_schedule(
        total_steps,
        warmup_steps=training_cfg.get('warmup_steps'),
        warmup_fraction=float(training_cfg.get('warmup_fraction', 0.05)),
        min_lr_ratio=float(training_cfg.get('min_lr_ratio', 0.01)),
    )
    lr_at_construction = float(optimizer.param_groups[0]['lr'])

    batch = next(iter(loaders['train']))
    objects, global_features, mask, labels, weights, _ = batch
    objects = objects.to(device)
    global_features = global_features.to(device)
    mask = mask.to(device)
    labels = labels.to(device)
    weights = weights.to(device)
    before = [param.detach().cpu().clone() for param in model.parameters() if param.requires_grad]

    cuda_cleanup()
    cuda_reset_peak(device)

    # GradScaler-aware health probe. A single step is not enough to judge AMP: the scaler
    # deliberately overflows the first backward and halves its scale, so asserting finite
    # gradients on step 0 rejects healthy float16 training. See src/amp_health.py.
    logits_holder: dict[str, torch.Tensor] = {}

    def forward_loss() -> torch.Tensor:
        out = model(objects, global_features, mask)
        logits_holder['logits'] = out
        return weighted_cross_entropy(out, labels, weights, class_weight=class_weight)

    amp_health = probe_amp_health(
        model,
        optimizer,
        forward_loss,
        device,
        use_amp,
        steps=int(training_cfg.get('amp_probe_steps', 4)),
        gradient_clip_norm=float(training_cfg.get('gradient_clip_norm', 5.0)),
        init_scale=training_cfg.get('amp_init_scale'),
    )
    logits = logits_holder['logits']
    loss = torch.tensor(amp_health['records'][-1]['loss'])
    probs = torch.softmax(logits.detach().float(), dim=1)
    gradients_finite = bool(amp_health['healthy'])
    if scheduler is not None:
        scheduler.step()
    lr_after_one_step = float(optimizer.param_groups[0]['lr'])
    after = [param.detach().cpu() for param in model.parameters() if param.requires_grad]
    parameters_updated = any(not torch.equal(old, new) for old, new in zip(before, after))
    memory = cuda_memory_snapshot(device)

    summary: dict[str, Any] = {
        'device': device_info,
        'amp': amp_summary,
        'events_loaded': int(len(df)),
        'batch_size': int(labels.numel()),
        'input_shapes': {
            'objects': list(objects.shape),
            'global_features': list(global_features.shape),
            'padding_mask': list(mask.shape),
            'labels': list(labels.shape),
        },
        'tensor_devices': {
            'model': next(model.parameters()).device.type,
            'objects': objects.device.type,
            'global_features': global_features.device.type,
            'padding_mask': mask.device.type,
            'labels': labels.device.type,
            'loss': loss.device.type,
            'logits': logits.device.type,
        },
        'logit_shape': list(logits.shape),
        'probability_shape': list(probs.shape),
        'max_softmax_sum_deviation': float(torch.max(torch.abs(probs.sum(dim=1) - 1.0)).detach().cpu()),
        'loss': float(loss.detach().cpu()),
        'loss_finite': bool(torch.isfinite(loss).detach().cpu()),
        'gradients_finite': gradients_finite,
        'amp_health': amp_health,
        'parameters_updated': parameters_updated,
        'class_balance_strategy': class_balance,
        'preprocessing_diagnostics': diagnostics,
        'standardized_clip': standardized_clip,
        'clip_respected_all_splits': all(
            row['clip_bound_respected'] is not False for row in diagnostics['per_split'].values()
        ),
        'class_weight_sums_equalized': _sums_are_equal(
            diagnostics['per_split']['train']['normalized_class_weight_sums']
        ),
        'lr_schedule': schedule_summary,
        'lr_base': base_lr,
        'lr_at_construction': lr_at_construction,
        'lr_after_one_step': lr_after_one_step,
        # A single warmup step is degenerate: lr_lambda(0) = (0+1)/1 = 1.0, so there is no
        # ramp to observe. Only assert the ramp when the schedule has room for one.
        'lr_warmup_steps': schedule_summary.get('warmup_steps'),
        'lr_warmup_observable': bool(scheduler is not None and int(schedule_summary.get('warmup_steps') or 0) >= 2),
        'lr_warmup_applied': (
            None
            if scheduler is None or int(schedule_summary.get('warmup_steps') or 0) < 2
            else bool(lr_at_construction < base_lr - 1.0e-12)
        ),
        'cuda_memory': memory,
        'memory_limit_satisfied': device.type != 'cuda' or memory.get('max_reserved_gib', 0.0) <= float(device_info.get('cuda_allocator_limit_gib_effective', cuda_limit)) + 1.0e-6,
    }
    failures = []
    if not summary['loss_finite']:
        failures.append('loss is not finite')
    if not summary['gradients_finite']:
        failures.append(f'AMP health probe failed: {amp_health.get("diagnosis")}')
    if not summary['parameters_updated']:
        failures.append('parameters did not update')
    if not summary['clip_respected_all_splits']:
        failures.append(f"standardized features exceed the clip bound {standardized_clip}")
    if not summary['class_weight_sums_equalized']:
        failures.append(
            f"per-class training weight sums are not equal: "
            f"{diagnostics['per_split']['train']['normalized_class_weight_sums']}"
        )
    if summary['lr_warmup_applied'] is False:
        failures.append(
            f'LR warmup did not reduce the initial LR below the base LR {base_lr} '
            f'(warmup_steps={summary["lr_warmup_steps"]})'
        )
    if failures:
        raise RuntimeError(f'Model validation failed: {"; ".join(failures)}. Summary: {summary}')
    output_dir = resolve_path(study_cfg['output_dir'])
    output_dir.joinpath('metrics').mkdir(parents=True, exist_ok=True)
    write_yaml(output_dir / 'metrics' / 'model_validation.yaml', summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
