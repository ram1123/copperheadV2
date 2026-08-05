from __future__ import annotations

import math
from typing import Any

import torch


def resolve_warmup_steps(
    total_steps: int,
    warmup_steps: int | None = None,
    warmup_fraction: float = 0.05,
) -> int:
    """Pick the warmup length, preferring an explicit step count over the fraction.

    Always leaves at least one decay step so the schedule cannot be pure warmup, and
    always uses at least one warmup step when the run is long enough to have two.
    """
    if total_steps <= 1:
        return 0
    if warmup_steps is not None:
        resolved = int(warmup_steps)
    else:
        resolved = int(round(total_steps * float(warmup_fraction)))
        if resolved < 1:
            resolved = 1
    return max(0, min(resolved, total_steps - 1))


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int | None = None,
    warmup_fraction: float = 0.05,
    schedule: str = 'cosine',
    min_lr_ratio: float = 0.01,
) -> tuple[torch.optim.lr_scheduler.LambdaLR | None, dict[str, Any]]:
    """Linear warmup into cosine decay, stepped once per optimizer step.

    Pre-norm transformers trained with AdamW are unstable at a constant LR from step
    zero: the attention and FFN blocks see large gradients before the normalization
    statistics settle. Warmup is the standard remedy, and cosine decay to a small floor
    is the standard companion.
    """
    schedule = (schedule or 'none').lower()
    total_steps = max(int(total_steps), 1)
    resolved_warmup = resolve_warmup_steps(total_steps, warmup_steps, warmup_fraction)
    floor = max(float(min_lr_ratio), 0.0)

    summary: dict[str, Any] = {
        'schedule': schedule,
        'total_steps': total_steps,
        'warmup_steps': resolved_warmup,
        'warmup_fraction_requested': float(warmup_fraction),
        'min_lr_ratio': floor,
        'base_learning_rates': [group['lr'] for group in optimizer.param_groups],
    }

    if schedule in {'none', 'constant'}:
        summary['enabled'] = False
        return None, summary

    if schedule != 'cosine':
        raise ValueError(f"Unsupported lr_schedule {schedule!r}; expected 'cosine', 'constant', or 'none'.")

    def lr_lambda(step: int) -> float:
        # LambdaLR calls with step=0 before the first optimizer step.
        if resolved_warmup > 0 and step < resolved_warmup:
            return float(step + 1) / float(resolved_warmup)
        decay_total = max(total_steps - resolved_warmup, 1)
        progress = min(max(step - resolved_warmup, 0) / decay_total, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    summary['enabled'] = True
    summary['peak_multiplier'] = 1.0
    summary['final_multiplier'] = lr_lambda(total_steps - 1)
    return scheduler, summary


def sample_schedule(
    total_steps: int,
    warmup_steps: int | None = None,
    warmup_fraction: float = 0.05,
    min_lr_ratio: float = 0.01,
    n_samples: int = 12,
) -> list[dict[str, float]]:
    """Materialize the multiplier curve for reporting, without touching an optimizer."""
    total_steps = max(int(total_steps), 1)
    resolved_warmup = resolve_warmup_steps(total_steps, warmup_steps, warmup_fraction)
    floor = max(float(min_lr_ratio), 0.0)

    def multiplier(step: int) -> float:
        if resolved_warmup > 0 and step < resolved_warmup:
            return float(step + 1) / float(resolved_warmup)
        decay_total = max(total_steps - resolved_warmup, 1)
        progress = min(max(step - resolved_warmup, 0) / decay_total, 1.0)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    if total_steps <= n_samples:
        steps = list(range(total_steps))
    else:
        steps = sorted({int(round(index * (total_steps - 1) / (n_samples - 1))) for index in range(n_samples)})
        for boundary in (resolved_warmup - 1, resolved_warmup):
            if 0 <= boundary < total_steps:
                steps.append(boundary)
        steps = sorted(set(steps))
    return [{'step': int(step), 'lr_multiplier': float(multiplier(step))} for step in steps]
