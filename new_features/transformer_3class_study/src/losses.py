from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .disco import disco_penalty


def weighted_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    event_weight: torch.Tensor | None = None,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.cross_entropy(logits, target, reduction='none', weight=class_weight)
    if event_weight is None:
        return loss.mean()
    weights = torch.clamp(event_weight.float(), min=0.0)
    denom = torch.clamp(weights.sum(), min=1.0)
    return (loss * weights).sum() / denom


def total_objective(
    logits: torch.Tensor,
    target: torch.Tensor,
    event_weight: torch.Tensor | None = None,
    class_weight: torch.Tensor | None = None,
    dimuon_mass: torch.Tensor | None = None,
    disco_lambda: float = 0.0,
    disco_score_mode: str = 'signal_sum',
    disco_target_class: int = 2,
    disco_mass_window: tuple[float, float] | None = None,
    disco_monitor: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Weighted cross-entropy plus an optional lambda * dCorr^2 decorrelation penalty.

    `disco_lambda` defaults to 0.0. At exactly 0 the penalty term is never constructed, so
    it contributes nothing to the loss value and adds no node to the autograd graph -- the
    gradients are bit-identical to running without decorrelation at all. When monitoring is
    on the dCorr is still evaluated, but under no_grad, so it is observable without
    influencing training.
    """
    classification = weighted_cross_entropy(logits, target, event_weight, class_weight)
    info: dict[str, Any] = {
        'classification_loss': float(classification.detach().cpu()),
        'disco_lambda': float(disco_lambda),
        'disco_enabled': bool(disco_lambda > 0.0),
        'disco_applied': False,
        'disco_dcorr': None,
        'disco_penalty': 0.0,
        'disco_n_selected': 0,
        'disco_skip_reason': None,
    }

    if disco_lambda > 0.0:
        dcorr, disco_info = disco_penalty(
            logits, target, dimuon_mass, event_weight,
            target_class_index=disco_target_class,
            score_mode=disco_score_mode,
            mass_window=disco_mass_window,
        )
        penalty = disco_lambda * dcorr * dcorr
        total = classification + penalty
        info.update({
            'disco_applied': bool(disco_info['applied']),
            'disco_dcorr': disco_info.get('dcorr'),
            'disco_penalty': float(penalty.detach().cpu()),
            'disco_n_selected': disco_info['n_selected'],
            'disco_skip_reason': disco_info['skip_reason'],
        })
        info['total_loss'] = float(total.detach().cpu())
        return total, info

    if disco_monitor and dimuon_mass is not None:
        # Observation only: no_grad keeps this entirely out of the autograd graph.
        with torch.no_grad():
            _, disco_info = disco_penalty(
                logits.detach(), target, dimuon_mass, event_weight,
                target_class_index=disco_target_class,
                score_mode=disco_score_mode,
                mass_window=disco_mass_window,
            )
        info.update({
            'disco_applied': False,
            'disco_dcorr': disco_info.get('dcorr'),
            'disco_n_selected': disco_info['n_selected'],
            'disco_skip_reason': disco_info['skip_reason'],
            'disco_monitored': True,
        })

    info['total_loss'] = float(classification.detach().cpu())
    return classification, info


def inverse_sqrt_class_weights(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    counts = torch.bincount(labels.cpu(), minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = 1.0 / torch.sqrt(counts)
    return weights / weights.mean()


def fit_class_weight_scales(
    train_weights: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 3,
) -> dict[str, Any]:
    """Fit per-class rescaling factors that equalize each class's total training weight.

    The loss is sum(loss_i * w_i) / sum(w_i), so a class contributes to the gradient in
    proportion to its *summed* weight, not its event count. Raw absolute MC weights span
    about 7e4 between samples (dyTo2Mu |w| median 1.09, vbf_powheg_dipole 1.58e-5), so
    without this the signal classes are numerically absent from the objective.

    Balancing on counts instead (inverse_sqrt_class_weights) does not help here:
    max_events_per_class already equalizes counts, so that factor collapses to ~1.

    Scales are fitted on the train split only and reused unchanged for val and test, the
    same discipline the standardization statistics follow.
    """
    weights = np.asarray(train_weights, dtype=np.float64)
    labels = np.asarray(labels)
    total = float(weights.sum())
    per_class_sum: list[float] = []
    per_class_count: list[int] = []
    for class_index in range(num_classes):
        selection = labels == class_index
        per_class_sum.append(float(weights[selection].sum()))
        per_class_count.append(int(selection.sum()))

    present = [value for value in per_class_sum if value > 0.0]
    if not present:
        raise RuntimeError('No positive training weight in any class; cannot fit class weight scales.')

    # Target: every populated class contributes the same total weight, and the grand
    # total is preserved so the loss stays on a comparable scale across configurations.
    target_per_class = total / max(len(present), 1)
    scales = [
        (target_per_class / class_sum) if class_sum > 0.0 else 0.0
        for class_sum in per_class_sum
    ]
    return {
        'strategy': 'per_class_equal_sum',
        'fitted_on': 'train',
        'num_classes': int(num_classes),
        'class_weight_scales': [float(value) for value in scales],
        'raw_class_weight_sums': per_class_sum,
        'class_event_counts': per_class_count,
        'target_weight_sum_per_class': float(target_per_class),
        'raw_max_over_min_class_weight_sum': (
            float(max(present) / min(present)) if min(present) > 0.0 else None
        ),
    }


def apply_class_weight_scales(
    train_weights: np.ndarray,
    labels: np.ndarray,
    stats: dict[str, Any],
) -> np.ndarray:
    """Apply fitted per-class scales, returning a new weight array."""
    scales = np.asarray(stats['class_weight_scales'], dtype=np.float64)
    labels = np.asarray(labels)
    out = np.asarray(train_weights, dtype=np.float64).copy()
    valid = (labels >= 0) & (labels < scales.shape[0])
    out[valid] = out[valid] * scales[labels[valid]]
    return out.astype(np.float32)


def class_weight_sums(train_weights: np.ndarray, labels: np.ndarray, num_classes: int = 3) -> list[float]:
    """Summed training weight per class — the quantity the loss actually responds to."""
    weights = np.asarray(train_weights, dtype=np.float64)
    labels = np.asarray(labels)
    return [float(weights[labels == index].sum()) for index in range(num_classes)]
