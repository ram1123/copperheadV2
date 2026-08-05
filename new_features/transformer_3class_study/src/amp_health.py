"""GradScaler-aware assessment of whether float16 training is actually healthy.

Why this module exists
----------------------
The previous iteration of this study disabled AMP permanently with the note "float16
attention produced non-finite gradients". That observation was real but was misread.
`torch.amp.GradScaler` starts at `init_scale = 65536` on purpose: it multiplies the loss
by an optimistic factor, lets the first backward overflow float16, detects the overflow,
**skips** that optimizer step, and halves the scale until gradients fit. Non-finite
gradients on step 0 are therefore the designed warm-up behaviour of the scaler, not a
defect in the model.

The old check called `scaler.unscale_()` and asserted `isfinite(param.grad)` on that very
first step, so it could only ever see the transient and conclude AMP was broken.

This module runs several steps instead, records the loss-scale trajectory, and judges AMP
on the steps the scaler actually applied.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable

import torch

from .utils import amp_step_is_skipped, grad_scaler_scale, make_grad_scaler


def autocast_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True)
    return nullcontext()


def probe_amp_health(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[], torch.Tensor],
    device: torch.device,
    use_amp: bool,
    steps: int = 4,
    gradient_clip_norm: float | None = 5.0,
    init_scale: float | None = None,
) -> dict[str, Any]:
    """Run a few forward/backward steps and report whether AMP is healthy.

    `loss_fn` must run the forward pass and return the loss; it is called inside the
    autocast context. Returns a summary whose `healthy` flag is the value callers should
    key their AMP fallback off.
    """
    scaler = make_grad_scaler(use_amp, init_scale) if use_amp else None
    records: list[dict[str, Any]] = []

    for step in range(max(int(steps), 1)):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, use_amp):
            loss = loss_fn()
        loss_finite = bool(torch.isfinite(loss).detach().cpu())

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        grads_finite = all(
            param.grad is None or bool(torch.isfinite(param.grad).all().detach().cpu())
            for param in model.parameters()
        )
        if gradient_clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        scale_before = grad_scaler_scale(scaler)
        if use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scale_after = grad_scaler_scale(scaler)
        skipped = amp_step_is_skipped(scale_before, scale_after)

        records.append({
            'step': step,
            'loss': float(loss.detach().cpu()),
            'loss_finite': loss_finite,
            'gradients_finite': grads_finite,
            'loss_scale_before': scale_before,
            'loss_scale_after': scale_after,
            'step_skipped_by_scaler': skipped,
        })

    applied = [row for row in records if not row['step_skipped_by_scaler']]
    skipped = [row for row in records if row['step_skipped_by_scaler']]
    applied_finite = [row for row in applied if row['gradients_finite'] and row['loss_finite']]

    summary: dict[str, Any] = {
        'amp_used': bool(use_amp),
        'steps_run': len(records),
        'steps_applied': len(applied),
        'steps_skipped_by_scaler': len(skipped),
        'applied_steps_with_finite_gradients': len(applied_finite),
        'first_step_gradients_finite': records[0]['gradients_finite'] if records else None,
        'initial_loss_scale': records[0]['loss_scale_before'] if records else None,
        'final_loss_scale': records[-1]['loss_scale_after'] if records else None,
        'records': records,
        # An applied step with finite gradients is the real health criterion. Steps the
        # scaler discarded never reached the weights, so their gradients are irrelevant.
        'healthy': bool(applied_finite) and all(row['loss_finite'] for row in records),
    }
    if not summary['healthy']:
        if not applied:
            summary['diagnosis'] = (
                'GradScaler skipped every probe step; the loss scale never settled. '
                'Lower init_scale or investigate genuine overflow.'
            )
        elif not applied_finite:
            summary['diagnosis'] = 'Applied steps still carried non-finite gradients; float16 is genuinely unstable here.'
        else:
            summary['diagnosis'] = 'A loss value was non-finite during the probe.'
    else:
        summary['diagnosis'] = None
    return summary
