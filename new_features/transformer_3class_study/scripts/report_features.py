#!/usr/bin/env python
"""Machine- and human-readable reports for the DisCo penalty and the pair-feature masking.

Emits:
  reports/feature_additions_summary.yaml   (machine-readable)
  reports/feature_additions_summary.json   (machine-readable)
  reports/feature_additions_report.md      (human-readable)

Evidence is recomputed on real 2017 events, not restated. With --lambda-scan the script
also trains short models at several lambda values so the correlation/discrimination
trade-off is measured rather than asserted.
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
    assert_mass_not_in_features,
    assign_splits,
    build_bundle,
    fit_pair_normalization,
    load_events,
)
from src.disco import distance_correlation
from src.losses import total_objective
from src.model import PAIR_FEATURE_NAMES, build_model, pair_validity_mask, pairwise_hmumu_features
from src.utils import load_yaml, resolve_path, utc_now, write_json, write_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='new_features/transformer_3class_study/configs/study_config.yaml')
    parser.add_argument('--max-files-per-sample', type=int, default=1)
    parser.add_argument('--max-events-per-class', type=int, default=4000)
    parser.add_argument('--lambda-scan', default='0.0,1.0,10.0',
                        help='Comma-separated lambda values to train briefly. Empty string disables.')
    parser.add_argument('--scan-epochs', type=int, default=3)
    parser.add_argument('--scan-batch-size', type=int, default=256)
    return parser.parse_args()


def read_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return load_yaml(path)
    except Exception:
        return None


def pair_masking_evidence(bundle) -> dict[str, Any]:
    """Recompute the masked-vs-unmasked pair statistics on real events."""
    fourvec = torch.from_numpy(np.ascontiguousarray(bundle.objects[:, :, 6:10]))
    padding_mask = torch.from_numpy(bundle.padding_mask)
    valid_token = ~padding_mask
    valid_pair = (valid_token[:, :, None] & valid_token[:, None, :])

    masked = pairwise_hmumu_features(fourvec, apply_mask=True)
    unmasked = pairwise_hmumu_features(fourvec, apply_mask=False)

    features: dict[str, Any] = {}
    for index, name in enumerate(PAIR_FEATURE_NAMES):
        all_unmasked = unmasked[:, index].reshape(-1)
        all_masked = masked[:, index].reshape(-1)
        valid_only = unmasked[:, index][valid_pair]
        degenerate = unmasked[:, index][~valid_pair]
        features[name] = {
            'unmasked_all_pairs': {'mean': float(all_unmasked.mean()), 'std': float(all_unmasked.std())},
            'masked_all_pairs': {'mean': float(all_masked.mean()), 'std': float(all_masked.std())},
            'valid_pairs_only': {'mean': float(valid_only.mean()), 'std': float(valid_only.std())},
            'degenerate_pairs_unmasked': {'mean': float(degenerate.mean()), 'std': float(degenerate.std())},
            # How far the old (all-pairs) statistics displace a genuine pair once it is
            # standardized: a valid pair sitting at its own mean lands here instead of 0,
            # and its spread is scaled by this factor instead of 1.
            'valid_pair_offset_under_all_pair_stats': float(
                (valid_only.mean() - all_unmasked.mean()) / (all_unmasked.std() + 1e-12)
            ),
            'valid_pair_scale_under_all_pair_stats': float(
                valid_only.std() / (all_unmasked.std() + 1e-12)
            ),
        }

    degenerate_zero = bool(torch.all(masked.permute(0, 2, 3, 1)[~valid_pair] == 0.0))
    valid_untouched = bool(torch.allclose(
        masked.permute(0, 2, 3, 1)[valid_pair], unmasked.permute(0, 2, 3, 1)[valid_pair]
    ))
    return {
        'n_events': int(padding_mask.shape[0]),
        'pad_fraction_per_token': {
            name: float(padding_mask[:, i].float().mean())
            for i, name in enumerate(['mu1', 'mu2', 'jet1', 'jet2', 'dimuon'])
        },
        'total_pair_entries': int(valid_pair.numel()),
        'valid_pair_entries': int(valid_pair.sum()),
        'degenerate_pair_fraction': float((~valid_pair).float().mean()),
        'mean_valid_tokens_per_event': float(valid_token.sum(1).float().mean()),
        'degenerate_pairs_exactly_zero_after_mask': degenerate_zero,
        'valid_pairs_unchanged_by_mask': valid_untouched,
        'features': features,
    }


def mass_correlation_evidence(bundle, config: dict[str, Any]) -> dict[str, Any]:
    """dCorr between m_mumu and simple functions of the raw inputs, on background events.

    Establishes that the mass really is recoverable from what the network is given -- the
    motivation for the penalty existing at all.
    """
    labels = bundle.labels
    mass = torch.from_numpy(bundle.dimuon_mass.astype(np.float32))
    bkg = torch.from_numpy((labels == 2).astype(bool))
    n_bkg = int(bkg.sum())
    limit = min(n_bkg, 2000)
    index = torch.nonzero(bkg).reshape(-1)[:limit]
    mass_sel = mass[index]

    fourvec = torch.from_numpy(np.ascontiguousarray(bundle.objects[index.numpy()][:, :, 6:10]))
    pair = pairwise_hmumu_features(fourvec, apply_mask=True)
    # The (mu1, mu2) pair invariant mass IS m_mumu; slot 3 is log1p(m^2).
    mu_pair_mass = pair[:, 3, 0, 1]
    dimuon_token_energy = fourvec[:, 4, 3]
    return {
        'n_background_events': int(limit),
        'dcorr_mass_vs_mu1mu2_pair_mass_feature': float(distance_correlation(mu_pair_mass, mass_sel)),
        'dcorr_mass_vs_dimuon_token_energy': float(distance_correlation(dimuon_token_energy, mass_sel)),
        'note': (
            'The (mu1, mu2) pairwise invariant-mass feature reproduces m_mumu almost exactly, '
            'which is why setting the dimuon token mass to 0 does not decorrelate anything and '
            'why a penalty (or an input-basis change) is required.'
        ),
    }


def run_lambda_scan(
    bundles: dict[str, Any],
    normalization: dict[str, Any],
    model_cfg: dict[str, Any],
    lambdas: list[float],
    epochs: int,
    batch_size: int,
    device: torch.device,
    seed: int = 12345,
) -> list[dict[str, Any]]:
    """Train briefly at each lambda and measure held-out dCorr and accuracy.

    Deliberately small: the point is the direction and rough magnitude of the trade-off,
    not a production number.
    """
    train, test = bundles['train'], bundles['test']
    to = lambda a, dtype=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dtype)
    tr = {
        'objects': to(train.objects), 'global': to(train.global_features),
        'mask': to(train.padding_mask, torch.bool), 'labels': to(train.labels, torch.long),
        'weights': to(train.train_weights), 'mass': to(train.dimuon_mass),
    }
    te = {
        'objects': to(test.objects), 'global': to(test.global_features),
        'mask': to(test.padding_mask, torch.bool), 'labels': to(test.labels, torch.long),
        'mass': to(test.dimuon_mass),
    }
    pair_stats = normalization['pair_normalization']
    n = tr['labels'].shape[0]
    results = []

    for lam in lambdas:
        torch.manual_seed(seed)
        model = build_model(model_cfg).to(device)
        model.set_pair_normalization(pair_stats['pair_feature_mean'], pair_stats['pair_feature_std'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=0.01)
        generator = torch.Generator(device='cpu').manual_seed(seed)

        model.train()
        for _ in range(epochs):
            order = torch.randperm(n, generator=generator).to(device)
            for start in range(0, n, batch_size):
                sel = order[start:start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = model(tr['objects'][sel], tr['global'][sel], tr['mask'][sel])
                loss, _ = total_objective(
                    logits, tr['labels'][sel], tr['weights'][sel],
                    dimuon_mass=tr['mass'][sel], disco_lambda=lam, disco_monitor=False,
                )
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(te['objects'], te['global'], te['mask']), dim=1)
        pred = probs.argmax(dim=1)
        accuracy = float((pred == te['labels']).float().mean())
        bkg = te['labels'] == 2
        score = (probs[:, 0] + probs[:, 1])[bkg]
        held_out_dcorr = float(distance_correlation(score[:2000], te['mass'][bkg][:2000]))
        results.append({
            'disco_lambda': float(lam),
            'test_accuracy': accuracy,
            'held_out_background_dcorr': held_out_dcorr,
            'epochs': epochs,
            'batch_size': batch_size,
            'n_train': int(n),
            'n_test_background': int(bkg.sum()),
        })
    return results


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
    bundles: dict[str, Any] = {}
    normalization = None
    for split in ['train', 'val', 'test']:
        part = split_df[split_df['split'] == split].reset_index(drop=True)
        if split == 'train':
            bundles[split], normalization = build_bundle(part, fit=True, standardized_clip=standardized_clip)
        else:
            bundles[split], normalization = build_bundle(part, normalization=normalization, fit=False)

    pair_evidence = pair_masking_evidence(bundles['train'])
    leak = {split: assert_mass_not_in_features(bundle) for split, bundle in bundles.items()}
    correlation = mass_correlation_evidence(bundles['train'], cfg)

    scan: list[dict[str, Any]] = []
    scan_note = 'lambda scan disabled'
    if args.lambda_scan.strip():
        lambdas = [float(value) for value in args.lambda_scan.split(',') if value.strip()]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        scan = run_lambda_scan(
            bundles, normalization, cfg.get('model', {}), lambdas,
            args.scan_epochs, args.scan_batch_size, device,
            seed=int(training_cfg.get('seed', 12345)),
        )
        scan_note = (
            f'{args.scan_epochs}-epoch runs on {int(bundles["train"].labels.shape[0])} train events, '
            f'identical seed and initialization per lambda; indicative of the trade-off only.'
        )

    validation = read_optional(output_dir / 'metrics' / 'model_validation.yaml') or {}
    smoke = read_optional(output_dir / 'metrics' / 'smoke_train_metrics.yaml') or {}
    evaluation = read_optional(output_dir / 'metrics' / 'eval_test_metrics.yaml') or {}

    summary: dict[str, Any] = {
        'generated_at': utc_now(),
        'task_id': 'transformer-3class-disco-and-pair-mask-2017',
        'continuation_of': 'transformer-3class-preprocessing-fixes-2017',
        'study': study_cfg.get('name'),
        'year': study_cfg.get('year'),
        'evidence_sample': {
            'events_used': int(len(df)),
            'train_events': int(bundles['train'].labels.shape[0]),
            'max_files_per_sample': args.max_files_per_sample,
            'max_events_per_class': args.max_events_per_class,
        },
        'features': {
            'DISCO-mass-decorrelation': {
                'title': 'DisCo decorrelation penalty against m_mumu',
                'problem': (
                    'The transformer receives both muon four-vectors, so m_mumu is recoverable from the '
                    'mu1/mu2 tokens and from the (mu1, mu2) pairwise invariant mass. Setting the dimuon '
                    'token mass to 0 hides nothing. A score that learns m_mumu sculpts the background '
                    'mass spectrum, which is fatal for a bump hunt at 125 GeV.'
                ),
                'change': (
                    'total_loss = weighted_cross_entropy + lambda * dCorr^2(score, m_mumu), evaluated on '
                    'background events per batch with the per-event training weights. lambda defaults to 0.0.'
                ),
                'files': ['src/disco.py', 'src/losses.py', 'src/features.py', 'src/data.py',
                          'scripts/train.py', 'scripts/validate_model.py', 'configs/study_config.yaml'],
                'default_lambda': float(training_cfg.get('disco_lambda', 0.0)),
                'score_mode': str(training_cfg.get('disco_score', 'signal_sum')),
                'target_class_index': int(training_cfg.get('disco_target_class', 2)),
                'monitor_when_disabled': bool(training_cfg.get('disco_monitor', True)),
                'evidence': {
                    'mass_recoverable_from_inputs': correlation,
                    'mass_leak_check': leak,
                    'lambda_scan': scan,
                    'lambda_scan_note': scan_note,
                    'training_disco_config': smoke.get('disco'),
                    'training_dcorr_per_epoch': [
                        {
                            'epoch': row.get('epoch'),
                            'train_dcorr_mean': (row.get('train') or {}).get('disco_dcorr_mean'),
                            'val_dcorr_mean': (row.get('val') or {}).get('disco_dcorr_mean'),
                        }
                        for row in (smoke.get('history') or [])
                    ],
                    'validation_disco_loss_info': validation.get('disco_loss_info'),
                },
                'caveats': [
                    'A soft constraint: strictly weaker than removing m_mumu from the inputs.',
                    'Requires a lambda scan; the right value is analysis-dependent.',
                    'Only constrains the distribution it was trained on, and can degrade under shift.',
                    'Applied to background only on purpose - signal is expected to peak in m_mumu.',
                ],
            },
            'PAIR-feature-masking': {
                'title': 'Mask pair features before the PairEmbed normalization',
                'problem': (
                    'The study dropped the source\'s trailing "* mask" factor, so a pair involving a padded '
                    'token carried the real object\'s eta, phi and mass measured against a fictitious zero '
                    'four-vector. Those values sit in the same range as genuine pairs and set the BatchNorm '
                    'running statistics, which were then applied to the real pairs.'
                ),
                'change': (
                    '(a) restore the source-style validity mask so degenerate pairs are exactly zero; '
                    '(b) fit pair-feature mean/std on VALID PAIRS ONLY from the train split and replace the '
                    'PairEmbed input BatchNorm1d with that fixed standardization, persisted in the checkpoint.'
                ),
                'files': ['src/model.py', 'src/data.py', 'scripts/train.py',
                          'scripts/validate_model.py', 'scripts/evaluate.py'],
                'evidence': {
                    'recomputed': pair_evidence,
                    'fitted_pair_normalization': normalization.get('pair_normalization'),
                    'training_pair_normalization': smoke.get('pair_normalization'),
                    'evaluation_pair_normalization': evaluation.get('pair_normalization'),
                    'validation_pair_normalization_fitted': validation.get('pair_normalization_fitted'),
                },
            },
        },
        'downstream_metrics': {
            'smoke_test_accuracy': (smoke.get('test') or {}).get('accuracy'),
            'smoke_test_events': (smoke.get('test') or {}).get('n_events'),
            'smoke_per_class': (smoke.get('test') or {}).get('per_class'),
            'evaluation_accuracy': (evaluation.get('metrics') or {}).get('accuracy'),
            'evaluation_events': (evaluation.get('metrics') or {}).get('n_events'),
            'note': 'Smoke-scale run only; not a production performance statement.',
        },
        'known_issues_remaining': [
            'The deeper BatchNorm1d(hidden) layers inside PairEmbed still see degenerate entries; with the '
            'input masked those collapse to a constant (the conv bias) rather than a kinematics-dependent '
            'signal. Fully masked statistics throughout PairEmbed were out of scope.',
            'InputProcess applies RMSNorm across the 5-dim feature axis, partially undoing the per-feature '
            'standardization.',
            'm_mumu is still reconstructible from the inputs; the DisCo term penalizes using it rather than '
            'removing it. A Collins-Soper input-basis change would remove it outright.',
            'Normalization statistics are pooled across the heterogeneous token types.',
            'checkpoint_payload in scripts/train.py has unreachable plotting code after its return.',
        ],
    }

    write_yaml(reports_dir / 'feature_additions_summary.yaml', summary)
    write_json(reports_dir / 'feature_additions_summary.json', summary)
    (reports_dir / 'feature_additions_report.md').write_text(render_markdown(summary), encoding='utf-8')
    print(json.dumps({
        'machine_readable_yaml': str(reports_dir / 'feature_additions_summary.yaml'),
        'machine_readable_json': str(reports_dir / 'feature_additions_summary.json'),
        'human_readable': str(reports_dir / 'feature_additions_report.md'),
        'lambda_scan': scan,
    }, indent=2))


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return 'n/a'
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f'{value:.{digits}g}'
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    disco = summary['features']['DISCO-mass-decorrelation']
    pair = summary['features']['PAIR-feature-masking']
    pe = pair['evidence']['recomputed']

    lines: list[str] = [
        '# DisCo Mass Decorrelation and Masked Pair Features',
        '',
        f'Task: `{summary["task_id"]}` (continuation of `{summary["continuation_of"]}`)  ',
        f'Generated: `{summary["generated_at"]}`  ',
        f'Study: `{summary["study"]}`, year `{summary["year"]}`  ',
        f'Evidence sample: `{summary["evidence_sample"]["events_used"]}` events, '
        f'`{summary["evidence_sample"]["train_events"]}` in the train split',
        '',
        '## Summary',
        '',
        '| Feature | Status | Default |',
        '|---|---|---|',
        f'| DisCo decorrelation penalty against m_mumu | added | **disabled** (`disco_lambda = '
        f'{_fmt(disco["default_lambda"])}`) |',
        '| Masked pair features + valid-pair-only standardization | added | active |',
        '',
        '---',
        '',
        '## Feature 1 — DisCo decorrelation penalty',
        '',
        disco['problem'],
        '',
        f'**Change.** {disco["change"]}',
        '',
        f'The penalty is **off by default**: at `disco_lambda = {_fmt(disco["default_lambda"])}` the term is '
        'never constructed, so it adds nothing to the loss and no node to the autograd graph. '
        f'`disco_monitor = {_fmt(disco["monitor_when_disabled"])}` still reports dCorr under `no_grad`, so the '
        'correlation is observable without training against it.',
        '',
        '### The mass really is recoverable from the inputs',
        '',
    ]
    corr = disco['evidence']['mass_recoverable_from_inputs']
    lines.extend([
        f'- dCorr(m_mumu, (mu1,mu2) pairwise mass feature) = **{_fmt(corr["dcorr_mass_vs_mu1mu2_pair_mass_feature"], 4)}** '
        f'on {corr["n_background_events"]} background events',
        f'- dCorr(m_mumu, dimuon token energy) = {_fmt(corr["dcorr_mass_vs_dimuon_token_energy"], 4)}',
        '',
        corr['note'],
        '',
        '### m_mumu is metadata only',
        '',
        'It is carried in the batch for the penalty but never enters the model input tensors. Every object '
        'and global feature column was compared against it:',
        '',
        '| Split | Mass present in features | Closest column | Closest max abs difference |',
        '|---|---|---|---|',
    ])
    for split, row in disco['evidence']['mass_leak_check'].items():
        lines.append(
            f'| {split} | **{_fmt(row["mass_present_in_features"])}** | `{row["closest_column"]}` | '
            f'{_fmt(row["closest_max_abs_difference"], 5)} |'
        )
    lines.append('')

    scan = disco['evidence'].get('lambda_scan') or []
    if scan:
        lines.extend([
            '### Lambda scan — the trade-off, measured',
            '',
            f'_{disco["evidence"]["lambda_scan_note"]}_',
            '',
            '| lambda | Test accuracy | Held-out background dCorr(score, m_mumu) |',
            '|---|---|---|',
        ])
        for row in scan:
            lines.append(
                f'| {_fmt(row["disco_lambda"], 4)} | {_fmt(row["test_accuracy"], 4)} | '
                f'**{_fmt(row["held_out_background_dcorr"], 4)}** |'
            )
        baseline = next((r for r in scan if r['disco_lambda'] == 0.0), None)
        strongest = max(scan, key=lambda r: r['disco_lambda'])
        if baseline and strongest is not baseline:
            d_drop = baseline['held_out_background_dcorr'] - strongest['held_out_background_dcorr']
            a_drop = baseline['test_accuracy'] - strongest['test_accuracy']
            lines.extend([
                '',
                f'Going from lambda 0 to {_fmt(strongest["disco_lambda"], 4)} changes held-out dCorr by '
                f'**{-d_drop:+.4f}** and accuracy by **{-a_drop:+.4f}**.',
            ])
        lines.append('')

    lines.extend(['### Caveats', ''])
    for item in disco['caveats']:
        lines.append(f'- {item}')
    lines.extend(['', '---', '', '## Feature 2 — Masked pair features', '', pair['problem'], '',
                  f'**Change.** {pair["change"]}', '', '### How much of the grid is degenerate', ''])
    lines.extend([
        '| Token | Padded |',
        '|---|---|',
    ])
    for name, value in pe['pad_fraction_per_token'].items():
        lines.append(f'| {name} | {100*value:.1f}% |')
    lines.extend([
        '',
        f'- Pair-grid entries: `{pe["total_pair_entries"]}`, of which `{pe["valid_pair_entries"]}` are valid',
        f'- **Degenerate fraction: {100*pe["degenerate_pair_fraction"]:.1f}%** '
        f'({_fmt(pe["mean_valid_tokens_per_event"], 3)} valid tokens per event of 5)',
        f'- Degenerate pairs exactly zero after masking: `{pe["degenerate_pairs_exactly_zero_after_mask"]}`',
        f'- Valid pairs unchanged by masking: `{pe["valid_pairs_unchanged_by_mask"]}`',
        '',
        '### Where the damage was',
        '',
        '| Pair feature | Unmasked, all pairs | Valid pairs only | Degenerate only | Offset | Scale |',
        '|---|---|---|---|---|---|',
    ])
    for name, row in pe['features'].items():
        lines.append(
            f'| `{name}` | {_fmt(row["unmasked_all_pairs"]["mean"], 4)} / {_fmt(row["unmasked_all_pairs"]["std"], 4)} '
            f'| {_fmt(row["valid_pairs_only"]["mean"], 4)} / {_fmt(row["valid_pairs_only"]["std"], 4)} '
            f'| {_fmt(row["degenerate_pairs_unmasked"]["mean"], 4)} / {_fmt(row["degenerate_pairs_unmasked"]["std"], 4)} '
            f'| {_fmt(row["valid_pair_offset_under_all_pair_stats"], 3)} '
            f'| {_fmt(row["valid_pair_scale_under_all_pair_stats"], 3)} |'
        )
    lines.extend([
        '',
        '"Offset" is where a genuine pair sitting at its own mean would land after standardization with the '
        'old all-pairs statistics (0 would be correct); "Scale" is the factor its spread is multiplied by '
        '(1 would be correct). The angular features were barely affected; the mass channel was not.',
        '',
        '### Fitted statistics now in use',
        '',
    ])
    fitted = pair['evidence']['fitted_pair_normalization'] or {}
    if fitted:
        lines.extend([
            '| Pair feature | Fitted mean | Fitted std |',
            '|---|---|---|',
        ])
        for name, mean, std in zip(fitted['pair_feature_names'], fitted['pair_feature_mean'], fitted['pair_feature_std']):
            lines.append(f'| `{name}` | {_fmt(mean, 5)} | {_fmt(std, 5)} |')
        lines.extend([
            '',
            f'Fitted on `{fitted["fitted_on"]}`, over `{fitted["fitted_over"]}` '
            f'({fitted["valid_pair_entries"]} of {fitted["total_pair_entries"]} entries; '
            f'{100*fitted["degenerate_pair_fraction"]:.1f}% degenerate and excluded).',
            '',
            f'Round-trip: validation reports `pair_normalization_fitted = '
            f'{_fmt(pair["evidence"].get("validation_pair_normalization_fitted"))}`; the statistics live in '
            'the `FixedStandardize` buffers, so `load_state_dict` restores them with the weights.',
            '',
        ])

    downstream = summary['downstream_metrics']
    lines.extend([
        '## Downstream smoke metrics',
        '',
        f'- Smoke test accuracy: `{_fmt(downstream.get("smoke_test_accuracy"), 6)}` on '
        f'`{downstream.get("smoke_test_events")}` events',
        f'- Checkpoint-reload evaluation accuracy: `{_fmt(downstream.get("evaluation_accuracy"), 6)}` on '
        f'`{downstream.get("evaluation_events")}` events',
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

    lines.extend(['## Known issues remaining', ''])
    for item in summary['known_issues_remaining']:
        lines.append(f'- {item}')
    lines.extend([
        '',
        '## Machine-readable companions',
        '',
        '- `reports/feature_additions_summary.yaml`',
        '- `reports/feature_additions_summary.json`',
        '',
    ])
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
