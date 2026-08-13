# Proposal: displacement-based adversarial penalty (`rho(Delta s)`)

**Status: proposal only. Not implemented, not measured.** Filed at the analyst's
request on 2026-08-11 to be carried into `final-summary.md`. Do not treat any
statement here as a result.

Motivated by D-8 (the score-collapse mode). The stop-gradient fix removes the
gradient leak into `p_nominal`, but two defects of `BCE(q; p)` survive detaching.

## Why BCE is the wrong divergence here

1. **Non-zero floor.** `BCE(q; p) >= H(p)`, the binary entropy of the target. At
   perfect agreement (`q == p`) the term equals `H(p)`, which runs from 0.693 at
   `p=0.5` to 0.056 at `p=0.99`. So the term's magnitude varies ~12x across the
   score range for reasons nobody chose, and "perfectly decorrelated" is not a
   zero of the objective.
2. **Wrong metric.** BCE measures disagreement in probability space; Stage-3 bins
   in `arctanh(score)`. Near `p -> 1`, where the significance lives, a large move
   along the histogram axis is a tiny move in `p`, so BCE under-penalises exactly
   the migrations that matter. (Top-bin edges 1.869-7.354 in arctanh units.)

## The form

    penalty = sum_v  E[ w * g(s_nom) * rho(s_v - s_nom) ]  /  E[ w * g(s_nom) ]

with `s = arctanh(p)` (the binned axis), `s_nom` **detached**, `g` a smooth
detached gate `sigmoid((s_nom - c)/tau)` replacing the hard `p > tanh(2.0)` cut,
and `rho` the Huber loss. Suggested `c = 2.0`, `tau = 0.25`, `huber_delta = 0.25`.

```python
def adversarial_penalty_shift(logits_var, p_nominal, wb, adv):
    eps = 1e-6
    s_nom = torch.atanh(p_nominal.float().clamp(eps, 1 - eps)).detach()          # [B]
    s_var = torch.atanh(torch.sigmoid(logits_var.float()).clamp(eps, 1 - eps))   # [B,V]
    gate = torch.sigmoid((s_nom - adv.gate_center) / adv.gate_width).detach()
    w = (wb * gate).unsqueeze(1)
    d = s_var - s_nom.unsqueeze(1)
    dl = adv.huber_delta
    rho = torch.where(d.abs() <= dl, 0.5 * d.pow(2), dl * (d.abs() - 0.5 * dl))
    return (w * rho).sum() / w.sum().clamp_min(eps) / logits_var.shape[1]
```

## What each piece buys

| property | `BCE(q; p)` | `rho(Delta s)` |
|---|---|---|
| value at perfect agreement | `H(p)`, varies 12x | exactly 0 |
| can the term request a *higher* score? | yes -- this is the collapse | no, by construction |
| metric matches the binned axis | no (probability space) | yes |
| gate membership trainable | yes -> ratchet (D-8) | no (detached + shift-based) |
| lambda stable across training | no | yes (normalised by gated weight) |
| outlier events | unbounded | linear tail (Huber) |

The ratchet dies twice over: the gate is detached, **and** `rho` is minimised at
`Delta s = 0` regardless of where `s` sits, so nothing pushes events into or out of
the gated region. "Agree" and "be confident" become orthogonal rather than entangled
in a single number.

Huber rather than L2 matters specifically because the degenerate QvG `-1` columns
produce a few events with enormous `Delta s`; under L2 those would dominate.

## Refinements

- **Warm up lambda** over the first few epochs. At initialisation the score
  distribution is meaningless, so the gated region is arbitrary. This also disposes
  of the empty-selection edge case that currently needs `logits_var.sum() * 0.0`.
- **Penalise spread**, `Var_v(s_v)`, instead of deviation from nominal: closer to
  what the nuisance parameter constrains, and it stops privileging nominal as a
  reference point.

## The principled endpoint, if `rho(Delta s)` is not enough

Penalise the **template** difference directly, which is the quantity the fit sees:
soft-assign events to bins via a temperature-softmax over the Stage-3 edges,
accumulate per-bin yields for nominal and varied (EMA across batches -- one batch is
far too noisy), and penalise `sum_b (N_v,b - N_nom,b)^2 / max(N_nom,b, 1)`.

This fixes a real weakness of `rho(Delta s)`: the top bin spans 1.869-7.354, so
movement *inside* it costs nothing in the fit yet is still penalised. Only a binned
objective knows that. It is the correct target; it costs EMA state and a temperature
to tune, so it is the second thing to try, not the first.

## How to test it cheaply

Behind `--adversarial-penalty-form shift`, with unit tests: `rho(0) == 0`; gate and
`s_nom` carry no gradient; the normalised term is invariant to batch composition;
bit-identity with the switch-off run at `lambda = 0`. Then one training at the
`rho`-calibrated parity point on keep-degenerate inputs, compared against the
stop-gradient BCE run as baseline.
