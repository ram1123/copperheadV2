"""Weighted distance correlation, for decorrelating the classifier score from m_mumu.

Why this exists
---------------
The transformer receives both muon four-vectors, so m_mumu is a deterministic function of
its inputs: it is recoverable from the mu1/mu2 tokens directly and from the (mu1, mu2)
pairwise invariant mass. Setting the dimuon token's mass to 0 hides nothing. If the score
is allowed to learn m_mumu it will sculpt the background mass spectrum, which is fatal for
a bump hunt at 125 GeV.

Removing the information would require an input-basis change (dimuon kinematics plus
Collins-Soper decay angles, dropping the mass). Keeping the muon tokens leaves only the
soft route implemented here: penalize the distance correlation between the score and
m_mumu on background events.

Distance correlation (Szekely, Rizzo & Bakirov 2007) is 0 if and only if the two variables
are statistically independent -- unlike Pearson correlation, it catches non-linear and
non-monotone dependence. It is differentiable and has a single hyperparameter, so unlike an
adversary there is no inner optimization to balance.

The weighted formulation follows Kasieczka & Shih, "DisCo Fever: Robust Networks Through
Distance Correlation" (arXiv:2001.05310).

Caveats worth remembering
-------------------------
This is a soft constraint, strictly weaker than not giving the network the information.
It needs a lambda scan, it only constrains the distribution it was trained on, and it can
degrade under distribution shift. Prefer the basis change if it does not cost real
discrimination.
"""
from __future__ import annotations

from typing import Any

import torch

# Distance correlation is a ratio of empirical U-statistics; with only a handful of
# samples the estimate is dominated by noise and its gradient is meaningless.
MIN_EVENTS_FOR_DISCO = 8


def _weighted_double_centered(matrix: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted double centering: A_ij = a_ij - <a_i.> - <a_.j> + <a>."""
    row_mean = (matrix * weights[None, :]).sum(dim=1, keepdim=True)
    col_mean = (matrix * weights[:, None]).sum(dim=0, keepdim=True)
    grand_mean = (matrix * weights[:, None] * weights[None, :]).sum()
    return matrix - row_mean - col_mean + grand_mean


def distance_correlation(
    x: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Weighted distance correlation between two 1-D samples.

    Returns dCorr in [0, 1]: 0 for independent variables, 1 for a deterministic linear
    relation. Differentiable in both arguments.
    """
    x = x.reshape(-1).float()
    y = y.reshape(-1).float()
    if x.shape != y.shape:
        raise ValueError(f'distance_correlation expects equal shapes, got {tuple(x.shape)} and {tuple(y.shape)}')
    n = x.shape[0]
    if n < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)

    if weights is None:
        weights = torch.ones(n, device=x.device, dtype=x.dtype)
    else:
        weights = weights.reshape(-1).float().clamp_min(0.0)
    total = weights.sum()
    if float(total) <= 0.0:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    weights = weights / total  # normalized so the centering terms are plain weighted means

    a = (x[:, None] - x[None, :]).abs()
    b = (y[:, None] - y[None, :]).abs()
    centered_a = _weighted_double_centered(a, weights)
    centered_b = _weighted_double_centered(b, weights)

    joint = weights[:, None] * weights[None, :]
    dcov2 = (centered_a * centered_b * joint).sum()
    dvar_x2 = (centered_a * centered_a * joint).sum()
    dvar_y2 = (centered_b * centered_b * joint).sum()

    denominator = torch.sqrt(torch.clamp(dvar_x2 * dvar_y2, min=0.0))
    # A constant input makes dVar zero and the ratio undefined; report no correlation.
    if float(denominator) <= eps:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    dcorr2 = torch.clamp(dcov2, min=0.0) / (denominator + eps)
    return torch.sqrt(torch.clamp(dcorr2, min=0.0))


def decorrelation_score(probabilities: torch.Tensor, mode: str = 'signal_sum') -> torch.Tensor:
    """The scalar whose correlation with m_mumu we care about.

    'signal_sum' (p_ggH + p_VBF) is the natural choice: it is the signal-vs-background
    discriminant that would define analysis categories, so it is the quantity whose
    sculpting of the background mass spectrum matters.
    """
    mode = (mode or 'signal_sum').lower()
    if mode == 'signal_sum':
        return probabilities[:, 0] + probabilities[:, 1]
    if mode == 'p_bkg':
        return probabilities[:, 2]
    if mode == 'p_ggh':
        return probabilities[:, 0]
    if mode == 'p_vbf':
        return probabilities[:, 1]
    raise ValueError(f"Unsupported disco_score {mode!r}; expected signal_sum, p_bkg, p_ggH or p_VBF.")


def disco_penalty(
    logits: torch.Tensor,
    labels: torch.Tensor,
    dimuon_mass: torch.Tensor | None,
    weights: torch.Tensor | None = None,
    target_class_index: int = 2,
    score_mode: str = 'signal_sum',
    mass_window: tuple[float, float] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Distance correlation between the classifier score and m_mumu on one class.

    Returns (dcorr, info). The caller multiplies by lambda; this function never applies it,
    so the same value can be monitored whether or not the penalty is switched on.

    Computed on background events only (`target_class_index`): sculpting of the *background*
    mass spectrum is what breaks the bump hunt. Signal is expected to peak in m_mumu, so
    decorrelating it would be actively wrong.
    """
    info: dict[str, Any] = {
        'n_selected': 0,
        'score_mode': score_mode,
        'target_class_index': int(target_class_index),
        'applied': False,
        'skip_reason': None,
    }
    zero = torch.zeros((), device=logits.device, dtype=torch.float32)
    if dimuon_mass is None:
        info['skip_reason'] = 'dimuon_mass unavailable'
        return zero, info

    selection = labels == target_class_index
    if mass_window is not None:
        low, high = mass_window
        selection = selection & (dimuon_mass >= low) & (dimuon_mass <= high)

    n_selected = int(selection.sum())
    info['n_selected'] = n_selected
    if n_selected < MIN_EVENTS_FOR_DISCO:
        info['skip_reason'] = f'only {n_selected} target-class events in batch (< {MIN_EVENTS_FOR_DISCO})'
        return zero, info

    probabilities = torch.softmax(logits.float(), dim=1)
    score = decorrelation_score(probabilities, score_mode)[selection]
    mass = dimuon_mass[selection].float()
    selected_weights = weights[selection].float() if weights is not None else None

    dcorr = distance_correlation(score, mass, selected_weights)
    info['applied'] = True
    info['dcorr'] = float(dcorr.detach().cpu())
    info['dcorr_squared'] = float((dcorr * dcorr).detach().cpu())
    return dcorr, info
