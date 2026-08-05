#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

STUDY_ROOT = Path(__file__).resolve().parents[1]
if str(STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT))

from src.amp_health import probe_amp_health
from src.data import DEFAULT_STANDARDIZED_CLIP, build_loaders, class_counts, load_events, preprocessing_diagnostics
from src.features import INDEX_TO_CLASS, OUTPUT_COLUMNS, describe_feature_contract
from src.losses import inverse_sqrt_class_weights, total_objective, weighted_cross_entropy
from src.model import build_model, masked_attention_bias_value
from src.schedule import build_lr_scheduler, sample_schedule
from src.utils import (
    amp_info,
    cuda_cleanup,
    cuda_memory_snapshot,
    cuda_reset_peak,
    ensure_dirs,
    load_yaml,
    make_grad_scaler,
    resolve_path,
    select_device,
    set_seed,
    source_provenance,
    setup_logging,
    runtime_config,
    utc_now,
    write_json,
    write_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a minimal 3-class H->mumu transformer smoke model.')
    parser.add_argument('--config', default='new_features/transformer_3class_study/configs/study_config.yaml')
    parser.add_argument('--samples', default=None, help='Optional resolved samples YAML override.')
    parser.add_argument('--year', type=int, default=None)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default=None)
    parser.add_argument('--cuda-memory-limit-gib', type=float, default=None)
    parser.add_argument('--max-files-per-sample', type=int, default=None)
    parser.add_argument('--max-events-per-sample', type=int, default=None)
    parser.add_argument('--max-events-per-class', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--smoke-test', action='store_true')
    return parser.parse_args()


def batch_to_device(batch, device: torch.device):
    objects, global_features, mask, labels, weights, dimuon_mass, row_indices = batch
    return (
        objects.to(device, non_blocking=True),
        global_features.to(device, non_blocking=True),
        mask.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
        weights.to(device, non_blocking=True),
        dimuon_mass.to(device, non_blocking=True),
        row_indices,
    )


def autocast_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True)
    return nullcontext()


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    class_weight=None,
    gradient_clip_norm: float | None = None,
    use_amp: bool = False,
    grad_scaler=None,
    gradient_accumulation_steps: int = 1,
    scheduler=None,
    disco_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_weight = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros(3, 3, dtype=torch.long)
    learning_rates: list[float] = []
    dcorr_values: list[float] = []
    disco_penalties: list[float] = []
    if train:
        optimizer.zero_grad(set_to_none=True)
    n_batches = len(loader)
    accumulation = max(int(gradient_accumulation_steps), 1)
    for batch_idx, batch in enumerate(loader, start=1):
        objects, global_features, mask, labels, weights, dimuon_mass, _ = batch_to_device(batch, device)
        with torch.set_grad_enabled(train):
            with autocast_context(device, use_amp):
                logits = model(objects, global_features, mask)
                loss, loss_info = total_objective(
                    logits, labels, weights, class_weight=class_weight,
                    dimuon_mass=dimuon_mass, **(disco_config or {}),
                )
            if loss_info.get('disco_dcorr') is not None:
                dcorr_values.append(float(loss_info['disco_dcorr']))
                disco_penalties.append(float(loss_info['disco_penalty']))
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss encountered: {loss.detach().cpu().item()}')
        if train:
            scaled_loss = loss / accumulation
            if use_amp and grad_scaler is not None:
                grad_scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            if batch_idx % accumulation == 0 or batch_idx == n_batches:
                if use_amp and grad_scaler is not None:
                    grad_scaler.unscale_(optimizer)
                if gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                learning_rates.append(float(optimizer.param_groups[0]['lr']))
                if use_amp and grad_scaler is not None:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # Stepped per optimizer step, not per epoch, so warmup is measured in
                # updates. Under AMP a skipped (inf-gradient) step still advances the
                # schedule; that is one step of drift at most and keeps the LR curve
                # reproducible from the step count alone.
                if scheduler is not None:
                    scheduler.step()
        batch_weight = float(torch.clamp(weights, min=0.0).sum().detach().cpu())
        total_loss += float(loss.detach().cpu()) * max(batch_weight, 1.0)
        total_weight += max(batch_weight, 1.0)
        pred = torch.argmax(logits.detach(), dim=1)
        correct += int((pred == labels).sum().detach().cpu())
        total += int(labels.numel())
        for truth, guess in zip(labels.detach().cpu(), pred.detach().cpu()):
            confusion[int(truth), int(guess)] += 1
    metrics: dict[str, Any] = {
        'loss': total_loss / max(total_weight, 1.0),
        'accuracy': correct / max(total, 1),
        'n_events': total,
        'confusion': confusion.tolist(),
    }
    if dcorr_values:
        metrics['disco_dcorr_mean'] = float(np.mean(dcorr_values))
        metrics['disco_dcorr_max'] = float(np.max(dcorr_values))
        metrics['disco_dcorr_batches'] = len(dcorr_values)
        metrics['disco_penalty_mean'] = float(np.mean(disco_penalties))
    if train:
        metrics['optimizer_steps'] = len(learning_rates)
        metrics['learning_rate_first'] = learning_rates[0] if learning_rates else None
        metrics['learning_rate_last'] = learning_rates[-1] if learning_rates else None
        metrics['learning_rate_max'] = max(learning_rates) if learning_rates else None
    return metrics


def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs_all = []
    labels_all = []
    rows_all = []
    with torch.inference_mode():
        for batch in loader:
            objects, global_features, mask, labels, _, _, row_indices = batch_to_device(batch, device)
            probs = torch.softmax(model(objects, global_features, mask), dim=1)
            probs_all.append(probs.cpu().numpy())
            labels_all.append(labels.cpu().numpy())
            rows_all.append(row_indices.cpu().numpy())
    return np.concatenate(probs_all), np.concatenate(labels_all), np.concatenate(rows_all)


def prediction_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
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
        'accuracy': float((pred == labels).mean()) if labels.size else None,
        'n_events': int(labels.size),
        'confusion': confusion.tolist(),
        'per_class': per_class,
        'max_softmax_sum_deviation': float(np.max(np.abs(probs.sum(axis=1) - 1.0))) if labels.size else None,
        'mean_probabilities': {col: float(probs[:, idx].mean()) for idx, col in enumerate(OUTPUT_COLUMNS)} if labels.size else {},
    }


def save_predictions(path: Path, metadata: pd.DataFrame, probs: np.ndarray, labels: np.ndarray, rows: np.ndarray) -> None:
    pred = metadata.iloc[rows].reset_index(drop=True).copy()
    for idx, col in enumerate(OUTPUT_COLUMNS):
        pred[col] = probs[:, idx]
    pred['predicted_class_index'] = probs.argmax(axis=1)
    pred['predicted_class'] = [INDEX_TO_CLASS[int(idx)] for idx in pred['predicted_class_index']]
    pred['pred_class_index'] = pred['predicted_class_index']
    pred['pred_class'] = pred['predicted_class']
    pred['label_check'] = labels
    path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(path, index=False)


def maybe_plot(history: list[dict[str, Any]], outdir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not history:
        return
    epochs = [row['epoch'] for row in history]
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [row['train']['loss'] for row in history], marker='o', label='train')
    ax.plot(epochs, [row['val']['loss'] for row in history], marker='o', label='val')
    ax.set_xlabel('epoch')
    ax.set_ylabel('weighted CE loss')
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / 'loss_curve.png', dpi=150)
    plt.close(fig)


def install_pair_normalization(model, normalization: dict[str, Any], logger=None) -> None:
    """Install the valid-pair-only statistics into the model's FixedStandardize buffers.

    They live in the state_dict, so evaluation restores them with the weights.
    """
    stats = (normalization or {}).get('pair_normalization')
    if not stats:
        raise ValueError("normalization is missing 'pair_normalization'; refit with fit=True")
    model.set_pair_normalization(stats['pair_feature_mean'], stats['pair_feature_std'])
    if logger is not None:
        logger.info('pair normalization (valid pairs only): mean=%s std=%s degenerate_fraction=%.4f',
                    stats['pair_feature_mean'], stats['pair_feature_std'], stats['degenerate_pair_fraction'])


def resolve_class_weight(training_cfg: dict[str, Any], labels, device, logger=None):
    """Build the optional count-based class-weight vector.

    Defaults to disabled. Per-class training-weight normalization already equalizes each
    class's contribution to the loss, so stacking `inverse_sqrt_frequency` on top of it
    would balance the same imbalance twice. It stays available for explicit opt-in, with
    a warning, because the two mechanisms are not interchangeable when max_events_per_class
    is not set.
    """
    strategy = str(training_cfg.get('class_balance', 'none') or 'none').lower()
    if strategy in {'none', 'off', 'false'}:
        return None
    if strategy != 'inverse_sqrt_frequency':
        raise ValueError(f'Unsupported class_balance {strategy!r}; expected inverse_sqrt_frequency or none.')
    if logger is not None:
        logger.warning(
            'class_balance=inverse_sqrt_frequency is stacked on top of per-class training-weight '
            'normalization; both correct class imbalance, so this double-counts it.'
        )
    return inverse_sqrt_class_weights(torch.from_numpy(labels), num_classes=3).to(device)


def batch_size_attempts(requested: int) -> list[int]:
    sequence = [256, 128, 64, 32, 16, 8, 4, 2, 1]
    attempts = [value for value in sequence if value <= requested]
    if requested not in attempts:
        attempts.insert(0, requested)
    return attempts or [1]


def representative_memory_test(
    model,
    optimizer,
    bundle,
    device: torch.device,
    requested_batch_size: int,
    class_weight,
    use_amp: bool,
    gradient_clip_norm: float,
    amp_probe_steps: int = 4,
    amp_init_scale: float | None = None,
) -> dict[str, Any]:
    attempts = []
    if device.type != 'cuda':
        batch = min(requested_batch_size, len(bundle.labels))
        return {
            'requested_batch_size': requested_batch_size,
            'physical_batch_size': batch,
            'attempts': [{'batch_size': batch, 'status': 'cpu_no_cuda'}],
            'input_shapes': {
                'objects': list(bundle.objects[:batch].shape),
                'global_features': list(bundle.global_features[:batch].shape),
                'padding_mask': list(bundle.padding_mask[:batch].shape),
            },
            'logit_shape': [batch, 3],
            'probability_shape': [batch, 3],
            'loss_finite': True,
            'gradients_finite': True,
        }
    for physical in batch_size_attempts(requested_batch_size):
        actual = min(physical, len(bundle.labels))
        if actual <= 0:
            continue
        try:
            cuda_cleanup()
            cuda_reset_peak(device)
            objects = torch.from_numpy(bundle.objects[:actual]).to(device)
            global_features = torch.from_numpy(bundle.global_features[:actual]).to(device)
            mask = torch.from_numpy(bundle.padding_mask[:actual]).to(device)
            labels = torch.from_numpy(bundle.labels[:actual]).to(device)
            weights = torch.from_numpy(bundle.train_weights[:actual]).to(device)
            optimizer.zero_grad(set_to_none=True)
            # GradScaler-aware: judging AMP on a single step would flag the scaler's
            # designed first-step overflow as a model failure. See src/amp_health.py.
            holder: dict[str, torch.Tensor] = {}

            def forward_loss():
                out = model(objects, global_features, mask)
                holder['logits'] = out
                return weighted_cross_entropy(out, labels, weights, class_weight=class_weight)

            amp_health = probe_amp_health(
                model,
                optimizer,
                forward_loss,
                device,
                use_amp,
                steps=amp_probe_steps,
                gradient_clip_norm=gradient_clip_norm,
                init_scale=amp_init_scale,
            )
            logits = holder['logits']
            loss = torch.tensor(amp_health['records'][-1]['loss'])
            gradients_finite = bool(amp_health['healthy'])
            probs = torch.softmax(logits.detach().float(), dim=1)
            snapshot = cuda_memory_snapshot(device)
            optimizer.zero_grad(set_to_none=True)
            attempts.append({'batch_size': actual, 'status': 'success', 'cuda_memory': snapshot})
            return {
                'requested_batch_size': requested_batch_size,
                'physical_batch_size': actual,
                'attempts': attempts,
                'input_shapes': {
                    'objects': list(objects.shape),
                    'global_features': list(global_features.shape),
                    'padding_mask': list(mask.shape),
                    'labels': list(labels.shape),
                },
                'logit_shape': list(logits.shape),
                'probability_shape': list(probs.shape),
                'max_softmax_sum_deviation': float(torch.max(torch.abs(probs.sum(dim=1) - 1.0)).detach().cpu()),
                'loss': float(loss.detach().cpu()),
                'loss_finite': bool(torch.isfinite(loss).detach().cpu()),
                'gradients_finite': gradients_finite,
                'amp_health': amp_health,
                'cuda_memory': snapshot,
            }
        except torch.cuda.OutOfMemoryError as exc:
            attempts.append({'batch_size': actual, 'status': 'cuda_oom', 'error': str(exc), 'cuda_memory': cuda_memory_snapshot(device)})
            optimizer.zero_grad(set_to_none=True)
            cuda_cleanup()
    raise RuntimeError(f'No candidate batch size fit under the CUDA memory limit. Attempts: {attempts}')


def checkpoint_payload(
    model,
    optimizer,
    cfg: dict[str, Any],
    normalization: dict[str, Any],
    history: list[dict[str, Any]],
    training_limits: dict[str, Any],
    runtime_summary: dict[str, Any],
    epoch: int,
    validation_metric: float,
    schedule_summary: dict[str, Any] | None = None,
    preprocessing_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': cfg.get('model', {}),
        'normalization': normalization,
        'feature_configuration': describe_feature_contract(),
        'preprocessing_statistics': normalization,
        'weight_normalization': normalization.get('weight_normalization'),
        'standardized_clip': normalization.get('standardized_clip'),
        'pair_normalization': normalization.get('pair_normalization'),
        'lr_schedule': schedule_summary or {},
        'preprocessing_diagnostics': preprocessing_summary or {},
        'padding_conventions': {
            'object_padding_value': 0.0,
            'padding_mask': 'object token is padded when physObj_pt_4v <= 0',
        },
        'mask_conventions': {
            'object_self_attention': 'padded key tokens receive a finite large-negative pairwise attention bias',
            'class_attention': 'class token attends to object tokens plus a non-padded global token',
            'masked_bias_float16': masked_attention_bias_value(torch.float16),
            'masked_bias_float32': masked_attention_bias_value(torch.float32),
        },
        'config': cfg,
        'class_mapping': {'ggH': 0, 'VBF': 1, 'bkg': 2},
        'class_order': ['ggH', 'VBF', 'bkg'],
        'output_columns': OUTPUT_COLUMNS,
        'random_seed': int(cfg.get('training', {}).get('seed', cfg.get('study', {}).get('seed', 12345))),
        'source_model_provenance': source_provenance(cfg),
        'training_weight_strategy': cfg.get('training', {}).get('weight_strategy', 'absolute_event_weight'),
        'evaluation_weight_strategy': cfg.get('training', {}).get('evaluation_weight_strategy', 'signed_nominal'),
        'runtime_device': runtime_summary.get('device'),
        'cuda_memory_configuration': runtime_summary.get('cuda_memory'),
        'training_limits': training_limits,
        'history': history,
        'epoch': epoch,
        'validation_metric': validation_metric,
        'saved_at': utc_now(),
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [row['train']['accuracy'] for row in history], marker='o', label='train')
    ax.plot(epochs, [row['val']['accuracy'] for row in history], marker='o', label='val')
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / 'accuracy_curve.png', dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_path(args.config))
    study_cfg = cfg['study']
    training_cfg = dict(cfg.get('training', {}))
    rt_cfg = runtime_config(cfg)
    cuda_cfg = dict(rt_cfg['cuda'])
    if args.year is not None and args.year != int(study_cfg['year']):
        raise ValueError(f'This study config is for {study_cfg["year"]}, not {args.year}.')

    epochs = int(args.epochs if args.epochs is not None else training_cfg.get('epochs', 2))
    requested_batch_size = int(args.batch_size if args.batch_size is not None else training_cfg.get('batch_size', 256))
    max_files_per_sample = args.max_files_per_sample if args.max_files_per_sample is not None else training_cfg.get('max_files_per_sample')
    max_events_per_sample = args.max_events_per_sample
    max_events_per_class = args.max_events_per_class if args.max_events_per_class is not None else training_cfg.get('max_events_per_class')
    device_name = args.device or rt_cfg.get('device', 'auto')
    cuda_limit = float(args.cuda_memory_limit_gib if args.cuda_memory_limit_gib is not None else cuda_cfg.get('max_memory_gib', 5.0))

    output_dir = resolve_path(study_cfg['output_dir'])
    reports_dir = resolve_path(study_cfg['reports_dir'])
    logs_dir = resolve_path(study_cfg['logs_dir'])
    ensure_dirs([output_dir / 'checkpoints', output_dir / 'metrics', output_dir / 'plots', output_dir / 'predictions', reports_dir, logs_dir])
    logger = setup_logging(logs_dir / f'train_{utc_now().replace(":", "")}.log')

    set_seed(int(training_cfg.get('seed', 12345)))
    device, device_info = select_device(device_name, cuda_limit)
    amp_summary = amp_info(device, bool(cuda_cfg.get('automatic_mixed_precision', True)))
    amp_summary['amp_requested'] = bool(cuda_cfg.get('automatic_mixed_precision', True))
    amp_summary['full_precision_required'] = None
    amp_summary['masked_attention_bias_float16'] = masked_attention_bias_value(torch.float16)
    amp_summary['masked_attention_bias_float32'] = masked_attention_bias_value(torch.float32)
    amp_summary['amp_disable_reason'] = None
    # AMP is no longer switched off unconditionally. The previous non-finite gradients
    # came from a float16 attention bias of finfo(float16).min = -65504, which overflowed
    # to -inf once added to the scores. With the finite sentinel in model.py that is
    # fixed, so AMP stays on unless the representative forward/backward below actually
    # observes non-finite gradients — the fallback is now evidence-driven.
    use_amp = bool(amp_summary['amp_enabled'])
    logger.info('device info: %s', device_info)
    logger.info('amp info: %s', amp_summary)

    resolved_samples = resolve_path(args.samples or study_cfg['resolved_samples_yaml'])
    if not resolved_samples.exists():
        raise FileNotFoundError(f'Resolved sample file is missing: {resolved_samples}. Run inspect_stage1_inputs.py first.')
    df = load_events(
        resolved_samples,
        Path(study_cfg['stage1_root']),
        str(study_cfg['dataset_subdir']),
        max_files_per_sample=max_files_per_sample,
        max_events_per_sample=max_events_per_sample,
        max_events_per_class=max_events_per_class,
    )
    logger.info('loaded %s events; class counts %s', len(df), class_counts(df))
    features_cfg = dict(cfg.get('features') or {})
    raw_clip = features_cfg.get('standardized_clip', DEFAULT_STANDARDIZED_CLIP)
    standardized_clip = None if raw_clip in (None, 'none', 'None') else float(raw_clip)
    loaders, bundles, normalization, split_df = build_loaders(
        df,
        batch_size=requested_batch_size,
        num_workers=int(training_cfg.get('num_workers', 0)),
        train_fraction=float(training_cfg.get('train_fraction', 0.70)),
        val_fraction=float(training_cfg.get('val_fraction', 0.15)),
        seed=int(training_cfg.get('seed', 12345)),
        standardized_clip=standardized_clip,
    )
    split_counts = split_df.groupby(['split', 'true_class']).size().unstack(fill_value=0).to_dict()
    logger.info('split counts: %s', split_counts)
    logger.info('weight normalization: %s', normalization['weight_normalization'])
    logger.info('standardized clip: %s', standardized_clip)

    model = build_model(cfg.get('model', {})).to(device)
    install_pair_normalization(model, normalization, logger)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_cfg.get('learning_rate', 1.0e-4)), weight_decay=float(training_cfg.get('weight_decay', 0.01)))
    n_parameters = int(sum(param.numel() for param in model.parameters() if param.requires_grad))
    logger.info('trainable parameters: %s', n_parameters)
    class_weight = resolve_class_weight(training_cfg, bundles['train'].labels, device, logger)

    rep_summary = representative_memory_test(
        model,
        optimizer,
        bundles['train'],
        device,
        requested_batch_size,
        class_weight,
        use_amp,
        float(training_cfg.get('gradient_clip_norm', 5.0)),
        amp_probe_steps=int(training_cfg.get('amp_probe_steps', 4)),
        amp_init_scale=training_cfg.get('amp_init_scale'),
    )
    # Evidence-driven AMP fallback: only fall back to full precision if the representative
    # forward/backward actually produced non-finite gradients, and record what was seen.
    if use_amp and not rep_summary.get('gradients_finite', True):
        amp_summary['amp_enabled'] = False
        amp_summary['amp_dtype'] = None
        amp_summary['gradient_scaler_enabled'] = False
        amp_summary['amp_disable_reason'] = (
            'Representative float16 forward/backward produced non-finite gradients; '
            'falling back to full precision.'
        )
        amp_summary['full_precision_required'] = amp_summary['amp_disable_reason']
        use_amp = False
        logger.warning('%s', amp_summary['amp_disable_reason'])
        cuda_cleanup()
        model = build_model(cfg.get('model', {})).to(device)
        install_pair_normalization(model, normalization)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_cfg.get('learning_rate', 1.0e-4)), weight_decay=float(training_cfg.get('weight_decay', 0.01)))
        rep_summary = representative_memory_test(
            model,
            optimizer,
            bundles['train'],
            device,
            requested_batch_size,
            class_weight,
            use_amp,
            float(training_cfg.get('gradient_clip_norm', 5.0)),
            amp_probe_steps=int(training_cfg.get('amp_probe_steps', 4)),
            amp_init_scale=training_cfg.get('amp_init_scale'),
        )
    amp_summary['amp_validated_gradients_finite'] = bool(rep_summary.get('gradients_finite', True))
    logger.info('amp after representative validation: %s', amp_summary)

    physical_batch_size = int(rep_summary['physical_batch_size'])
    gradient_accumulation_steps = max(1, int(math.ceil(requested_batch_size / max(physical_batch_size, 1))))
    if physical_batch_size != requested_batch_size:
        logger.info('adapted physical batch size from %s to %s', requested_batch_size, physical_batch_size)
        loaders, bundles, normalization, split_df = build_loaders(
            df,
            batch_size=physical_batch_size,
            num_workers=int(training_cfg.get('num_workers', 0)),
            train_fraction=float(training_cfg.get('train_fraction', 0.70)),
            val_fraction=float(training_cfg.get('val_fraction', 0.15)),
            seed=int(training_cfg.get('seed', 12345)),
            standardized_clip=standardized_clip,
        )
        class_weight = resolve_class_weight(training_cfg, bundles['train'].labels, device)
    cuda_cleanup()
    model = build_model(cfg.get('model', {})).to(device)
    install_pair_normalization(model, normalization, logger)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_cfg.get('learning_rate', 1.0e-4)), weight_decay=float(training_cfg.get('weight_decay', 0.01)))
    grad_scaler = make_grad_scaler(use_amp)

    steps_per_epoch = max(1, int(math.ceil(len(loaders['train']) / max(gradient_accumulation_steps, 1))))
    total_optimizer_steps = steps_per_epoch * epochs
    scheduler, schedule_summary = build_lr_scheduler(
        optimizer,
        total_steps=total_optimizer_steps,
        warmup_steps=training_cfg.get('warmup_steps'),
        warmup_fraction=float(training_cfg.get('warmup_fraction', 0.05)),
        schedule=str(training_cfg.get('lr_schedule', 'cosine')),
        min_lr_ratio=float(training_cfg.get('min_lr_ratio', 0.01)),
    )
    schedule_summary['steps_per_epoch'] = steps_per_epoch
    schedule_summary['epochs'] = epochs
    schedule_summary['curve'] = sample_schedule(
        total_optimizer_steps,
        warmup_steps=training_cfg.get('warmup_steps'),
        warmup_fraction=float(training_cfg.get('warmup_fraction', 0.05)),
        min_lr_ratio=float(training_cfg.get('min_lr_ratio', 0.01)),
    )
    logger.info('lr schedule: %s', {key: value for key, value in schedule_summary.items() if key != 'curve'})

    mass_window = training_cfg.get('disco_mass_window')
    disco_config = {
        'disco_lambda': float(training_cfg.get('disco_lambda', 0.0)),
        'disco_score_mode': str(training_cfg.get('disco_score', 'signal_sum')),
        'disco_target_class': int(training_cfg.get('disco_target_class', 2)),
        'disco_mass_window': tuple(mass_window) if mass_window else None,
        'disco_monitor': bool(training_cfg.get('disco_monitor', True)),
    }
    logger.info('disco config: %s', disco_config)

    diagnostics = preprocessing_diagnostics(bundles, normalization)
    logger.info('preprocessing diagnostics: %s', diagnostics['per_split']['train'])

    history = []
    best_val = float('inf')
    best_path = output_dir / 'checkpoints' / 'best_model.pt'
    final_path = output_dir / 'checkpoints' / 'final_model.pt'
    training_limits = {
        'max_files_per_sample': max_files_per_sample,
        'max_events_per_sample': max_events_per_sample,
        'max_events_per_class': max_events_per_class,
        'requested_batch_size': requested_batch_size,
        'physical_batch_size': physical_batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'effective_batch_size': physical_batch_size * gradient_accumulation_steps,
        'epochs': epochs,
        'smoke_test': bool(args.smoke_test),
    }
    for epoch in range(1, epochs + 1):
        cuda_reset_peak(device)
        train_metrics = run_epoch(
            model,
            loaders['train'],
            device,
            optimizer=optimizer,
            class_weight=class_weight,
            gradient_clip_norm=float(training_cfg.get('gradient_clip_norm', 5.0)),
            use_amp=use_amp,
            grad_scaler=grad_scaler,
            gradient_accumulation_steps=gradient_accumulation_steps,
            scheduler=scheduler,
            disco_config=disco_config,
        )
        train_memory = cuda_memory_snapshot(device)
        val_metrics = run_epoch(model, loaders['val'], device, optimizer=None, class_weight=class_weight, use_amp=use_amp, disco_config=disco_config)
        val_memory = cuda_memory_snapshot(device)
        row = {'epoch': epoch, 'train': train_metrics, 'val': val_metrics, 'cuda_memory': val_memory, 'train_cuda_memory': train_memory}
        history.append(row)
        logger.info('epoch %s train=%s val=%s', epoch, train_metrics, val_metrics)
        if val_metrics['loss'] < best_val:
            best_val = float(val_metrics['loss'])
            runtime_summary = {'device': device_info, 'amp': amp_summary, 'cuda_memory': cuda_memory_snapshot(device)}
            torch.save(checkpoint_payload(model, optimizer, cfg, normalization, history, training_limits, runtime_summary, epoch, best_val, schedule_summary, diagnostics), best_path)

    runtime_summary = {'device': device_info, 'amp': amp_summary, 'cuda_memory': cuda_memory_snapshot(device)}
    torch.save(checkpoint_payload(model, optimizer, cfg, normalization, history, training_limits, runtime_summary, epochs, best_val, schedule_summary, diagnostics), final_path)

    probs, labels, rows = predict(model, loaders['test'], device)
    test_metrics = prediction_metrics(probs, labels)
    final_memory = cuda_memory_snapshot(device)
    cuda_summary = {
        'cuda_available': bool(device_info.get('cuda_available')),
        'cuda_used': device.type == 'cuda',
        'device_index': device_info.get('cuda_device_index'),
        'device_name': device_info.get('cuda_device_name'),
        'device_total_memory_gib': device_info.get('cuda_total_memory_gib'),
        'configured_allocator_limit_gib': cuda_limit,
        'configured_allocator_fraction': device_info.get('cuda_allocator_fraction'),
        'allocator_limit_scope': 'PyTorch CUDA allocator only; CUDA context, driver, and non-PyTorch allocations are not strictly capped.',
        'amp_enabled': amp_summary['amp_enabled'],
        'amp_dtype': amp_summary['amp_dtype'],
        'gradient_scaler_enabled': amp_summary['gradient_scaler_enabled'],
        'requested_batch_size': requested_batch_size,
        'physical_batch_size': physical_batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'effective_batch_size': physical_batch_size * gradient_accumulation_steps,
        'batch_size_attempts': rep_summary.get('attempts', []),
        'representative_forward_backward': rep_summary,
        'peak_allocated_gib': final_memory.get('max_allocated_gib'),
        'peak_reserved_gib': final_memory.get('max_reserved_gib'),
        'target_peak_allocated_gib': cuda_cfg.get('target_peak_allocated_gib', 4.5),
        'memory_limit_satisfied': (
            device.type != 'cuda'
            or (
                final_memory.get('max_reserved_gib', 0.0) <= float(device_info.get('cuda_allocator_limit_gib_effective', cuda_limit)) + 1.0e-6
                and final_memory.get('max_allocated_gib', 0.0) <= float(cuda_cfg.get('target_peak_allocated_gib', 4.5)) + 1.0e-6
            )
        ),
    }
    write_yaml(output_dir / 'metrics' / 'cuda_memory_summary.yaml', cuda_summary)
    metrics = {
        'generated_at': utc_now(),
        'smoke_test': bool(args.smoke_test),
        'device': device_info,
        'amp': amp_summary,
        'number_of_trainable_parameters': n_parameters,
        'limits': {
            'max_files_per_sample': max_files_per_sample,
            'max_events_per_sample': max_events_per_sample,
            'max_events_per_class': max_events_per_class,
            'epochs': epochs,
            'requested_batch_size': requested_batch_size,
            'physical_batch_size': physical_batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'effective_batch_size': physical_batch_size * gradient_accumulation_steps,
        },
        'class_counts_loaded': class_counts(df),
        'split_counts': split_df.groupby(['split', 'true_class']).size().reset_index(name='n').to_dict(orient='records'),
        'lr_schedule': schedule_summary,
        'preprocessing_diagnostics': diagnostics,
        'disco': disco_config,
        'pair_normalization': normalization.get('pair_normalization'),
        'class_balance_strategy': str(training_cfg.get('class_balance', 'none') or 'none'),
        'history': history,
        'test': test_metrics,
        'representative_forward_backward': rep_summary,
        'cuda_memory_final': final_memory,
        'cuda_memory_summary': cuda_summary,
        'checkpoints': {'best': str(best_path), 'final': str(final_path)},
    }
    write_yaml(output_dir / 'metrics' / 'smoke_train_metrics.yaml', metrics)
    write_json(output_dir / 'metrics' / 'smoke_train_metrics.json', metrics)
    save_predictions(output_dir / 'predictions' / 'smoke_test_predictions.parquet', bundles['test'].metadata, probs, labels, rows)
    maybe_plot(history, output_dir / 'plots')

    report = [
        '# Smoke Test Report',
        '',
        f'Generated: `{metrics["generated_at"]}`',
        '',
        f'- Events loaded: `{len(df)}`',
        f'- Device: `{device_info.get("selected")}`',
        f'- CUDA memory limit request: `{cuda_limit} GiB`',
        f'- CUDA status: `{"CUDA used successfully under the 5-GiB allocator limit." if cuda_summary["cuda_used"] and cuda_summary["memory_limit_satisfied"] else "CUDA was unavailable; CPU fallback was used." if not cuda_summary["cuda_used"] else "CUDA used but memory-limit validation failed."}`',
        f'- AMP requested: `{amp_summary.get("amp_requested")}`',
        f'- AMP enabled: `{amp_summary["amp_enabled"]}`',
        f'- AMP disable reason: `{amp_summary.get("amp_disable_reason")}`',
        f'- Masked attention bias (float16): `{amp_summary.get("masked_attention_bias_float16")}`',
        f'- LR schedule: `{schedule_summary.get("schedule")}` warmup `{schedule_summary.get("warmup_steps")}`/`{schedule_summary.get("total_steps")}` steps, floor `{schedule_summary.get("min_lr_ratio")}`',
        f'- Standardized clip: `{diagnostics.get("standardized_clip")}`',
        f'- Train class weight sums (raw): `{diagnostics["per_split"]["train"]["raw_class_weight_sums"]}`',
        f'- Train class weight sums (normalized): `{diagnostics["per_split"]["train"]["normalized_class_weight_sums"]}`',
        f'- Train class weight sum ratio: raw `{diagnostics["per_split"]["train"]["raw_class_weight_sum_ratio"]}` -> normalized `{diagnostics["per_split"]["train"]["normalized_class_weight_sum_ratio"]}`',
        f'- Epochs: `{epochs}`',
        f'- Requested batch size: `{requested_batch_size}`',
        f'- Physical batch size: `{physical_batch_size}`',
        f'- Gradient accumulation steps: `{gradient_accumulation_steps}`',
        f'- Effective batch size: `{physical_batch_size * gradient_accumulation_steps}`',
        f'- Trainable parameters: `{n_parameters}`',
        f'- Best checkpoint: `{best_path}`',
        f'- Final checkpoint: `{final_path}`',
        f'- CUDA memory summary: `{output_dir / "metrics" / "cuda_memory_summary.yaml"}`',
        f'- Test accuracy: `{test_metrics["accuracy"]}`',
        f'- Max softmax sum deviation: `{test_metrics["max_softmax_sum_deviation"]}`',
        '',
        '## Per-Class Test Summary',
        '',
    ]
    for name, row in test_metrics['per_class'].items():
        report.append(f'- `{name}`: n={row["n_events"]}, acc={row["accuracy"]}, mean_p_true={row["mean_assigned_probability"]}')
    report.extend(['', 'This is a 2-epoch smoke run and is not a production performance statement.', ''])
    (reports_dir / 'smoke_test_report.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'best_checkpoint': str(best_path), 'test_accuracy': test_metrics['accuracy'], 'events_loaded': len(df)}, indent=2))


if __name__ == '__main__':
    main()
