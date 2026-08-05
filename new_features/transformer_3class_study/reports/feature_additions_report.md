# DisCo Mass Decorrelation and Masked Pair Features

Task: `transformer-3class-disco-and-pair-mask-2017` (continuation of `transformer-3class-preprocessing-fixes-2017`)  
Generated: `2026-08-05T16:58:39Z`  
Study: `transformer_3class_study_2017`, year `2017`  
Evidence sample: `11997` events, `8398` in the train split

## Summary

| Feature | Status | Default |
|---|---|---|
| DisCo decorrelation penalty against m_mumu | added | **disabled** (`disco_lambda = 0`) |
| Masked pair features + valid-pair-only standardization | added | active |

---

## Feature 1 — DisCo decorrelation penalty

The transformer receives both muon four-vectors, so m_mumu is recoverable from the mu1/mu2 tokens and from the (mu1, mu2) pairwise invariant mass. Setting the dimuon token mass to 0 hides nothing. A score that learns m_mumu sculpts the background mass spectrum, which is fatal for a bump hunt at 125 GeV.

**Change.** total_loss = weighted_cross_entropy + lambda * dCorr^2(score, m_mumu), evaluated on background events per batch with the per-event training weights. lambda defaults to 0.0.

The penalty is **off by default**: at `disco_lambda = 0` the term is never constructed, so it adds nothing to the loss and no node to the autograd graph. `disco_monitor = True` still reports dCorr under `no_grad`, so the correlation is observable without training against it.

### The mass really is recoverable from the inputs

- dCorr(m_mumu, (mu1,mu2) pairwise mass feature) = **0.9576** on 2000 background events
- dCorr(m_mumu, dimuon token energy) = 0.202

The (mu1, mu2) pairwise invariant-mass feature reproduces m_mumu almost exactly, which is why setting the dimuon token mass to 0 does not decorrelate anything and why a penalty (or an input-basis change) is required.

### m_mumu is metadata only

It is carried in the batch for the penalty but never enters the model input tensors. Every object and global feature column was compared against it:

| Split | Mass present in features | Closest column | Closest max abs difference |
|---|---|---|---|
| train | **False** | `object[1][9]` | 627.84 |
| val | **False** | `object[1][6]` | 341.04 |
| test | **False** | `object[1][9]` | 615.23 |

### Lambda scan — the trade-off, measured

_4-epoch runs on 8398 train events, identical seed and initialization per lambda; indicative of the trade-off only._

| lambda | Test accuracy | Held-out background dCorr(score, m_mumu) |
|---|---|---|
| 0 | 0.6576 | **0.4017** |
| 1 | 0.6042 | **0.1421** |
| 5 | 0.4986 | **0.1638** |
| 20 | 0.448 | **0.08928** |

Going from lambda 0 to 20 changes held-out dCorr by **-0.3124** and accuracy by **-0.2096**.

### Caveats

- A soft constraint: strictly weaker than removing m_mumu from the inputs.
- Requires a lambda scan; the right value is analysis-dependent.
- Only constrains the distribution it was trained on, and can degrade under shift.
- Applied to background only on purpose - signal is expected to peak in m_mumu.

---

## Feature 2 — Masked pair features

The study dropped the source's trailing "* mask" factor, so a pair involving a padded token carried the real object's eta, phi and mass measured against a fictitious zero four-vector. Those values sit in the same range as genuine pairs and set the BatchNorm running statistics, which were then applied to the real pairs.

**Change.** (a) restore the source-style validity mask so degenerate pairs are exactly zero; (b) fit pair-feature mean/std on VALID PAIRS ONLY from the train split and replace the PairEmbed input BatchNorm1d with that fixed standardization, persisted in the checkpoint.

### How much of the grid is degenerate

| Token | Padded |
|---|---|
| mu1 | 0.0% |
| mu2 | 0.0% |
| jet1 | 27.0% |
| jet2 | 58.3% |
| dimuon | 0.0% |

- Pair-grid entries: `209950`, of which `149997` are valid
- **Degenerate fraction: 28.6%** (4.15 valid tokens per event of 5)
- Degenerate pairs exactly zero after masking: `True`
- Valid pairs unchanged by masking: `True`

### Where the damage was

| Pair feature | Unmasked, all pairs | Valid pairs only | Degenerate only | Offset | Scale |
|---|---|---|---|---|---|
| `delta_r` | 1.922 / 1.464 | 1.946 / 1.529 | 1.862 / 1.283 | 0.0162 | 1.04 |
| `abs_delta_eta` | 1.153 / 1.231 | 1.155 / 1.287 | 1.148 / 1.079 | 0.00159 | 1.05 |
| `cos_delta_phi` | 0.1389 / 0.794 | 0.1145 / 0.8101 | 0.2001 / 0.7485 | -0.0308 | 1.02 |
| `log1p_mass_squared` | 5.667 / 4.756 | 7.79 / 3.917 | 0.3561 / 1.165 | 0.446 | 0.824 |

"Offset" is where a genuine pair sitting at its own mean would land after standardization with the old all-pairs statistics (0 would be correct); "Scale" is the factor its spread is multiplied by (1 would be correct). The angular features were barely affected; the mass channel was not.

### Fitted statistics now in use

| Pair feature | Fitted mean | Fitted std |
|---|---|---|
| `delta_r` | 1.9457 | 1.529 |
| `abs_delta_eta` | 1.1548 | 1.2873 |
| `cos_delta_phi` | 0.11451 | 0.81014 |
| `log1p_mass_squared` | 7.79 | 3.917 |

Fitted on `train`, over `valid_pairs_only` (149997 of 209950 entries; 28.6% degenerate and excluded).

Round-trip: validation reports `pair_normalization_fitted = True`; the statistics live in the `FixedStandardize` buffers, so `load_state_dict` restores them with the weights.

## Downstream smoke metrics

- Smoke test accuracy: `0.641509` on `4452` events
- Checkpoint-reload evaluation accuracy: `0.641509` on `4452` events

_Smoke-scale run only; not a production performance statement._

| Class | Events | Accuracy | Mean assigned probability |
|---|---|---|---|
| ggH | 1500 | 0.5867 | 0.4568 |
| VBF | 1500 | 0.7807 | 0.6491 |
| bkg | 1452 | 0.5544 | 0.4604 |

## Known issues remaining

- The deeper BatchNorm1d(hidden) layers inside PairEmbed still see degenerate entries; with the input masked those collapse to a constant (the conv bias) rather than a kinematics-dependent signal. Fully masked statistics throughout PairEmbed were out of scope.
- InputProcess applies RMSNorm across the 5-dim feature axis, partially undoing the per-feature standardization.
- m_mumu is still reconstructible from the inputs; the DisCo term penalizes using it rather than removing it. A Collins-Soper input-basis change would remove it outright.
- Normalization statistics are pooled across the heterogeneous token types.
- checkpoint_payload in scripts/train.py has unreachable plotting code after its return.

## Machine-readable companions

- `reports/feature_additions_summary.yaml`
- `reports/feature_additions_summary.json`
