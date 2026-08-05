from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


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
