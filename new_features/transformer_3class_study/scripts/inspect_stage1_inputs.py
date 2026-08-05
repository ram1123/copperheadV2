#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

STUDY_ROOT = Path(__file__).resolve().parents[1]
if str(STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT))

from src.data import parquet_files_for_sample, resolve_samples_from_official_yaml, weight_summary
from src.features import canonicalize_frame, columns_to_read, describe_feature_contract, resolve_columns
from src.utils import format_table, load_yaml, resolve_path, write_yaml, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inspect 2017 Stage1 parquet inputs for the transformer smoke study.')
    parser.add_argument('--config', default='new_features/transformer_3class_study/configs/study_config.yaml')
    parser.add_argument('--max-files-per-sample', type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_path(args.config))
    study_cfg = cfg['study']
    year = int(study_cfg['year'])
    stage1_root = Path(study_cfg['stage1_root'])
    dataset_subdir = str(study_cfg['dataset_subdir'])
    samples_yaml = resolve_path(study_cfg['copperhead_samples_yaml'])
    resolved_path = resolve_path(study_cfg['resolved_samples_yaml'])
    reports_dir = resolve_path(study_cfg['reports_dir'])
    reports_dir.mkdir(parents=True, exist_ok=True)

    resolved = resolve_samples_from_official_yaml(samples_yaml, stage1_root, dataset_subdir, year)
    write_yaml(resolved_path, resolved)

    sample_rows = []
    issue_rows = []
    weight_rows = []
    actual_columns = {}
    for sample in resolved['samples']:
        files = parquet_files_for_sample(stage1_root, dataset_subdir, sample['name'])
        inspected = files[: args.max_files_per_sample]
        row_count = 0
        missing_union: set[str] = set()
        optional_union: set[str] = set()
        column_map = {}
        inspected_frames = []
        for path in inspected:
            pf = pq.ParquetFile(path)
            row_count += pf.metadata.num_rows
            resolution = resolve_columns(pf.schema_arrow.names)
            missing_union.update(resolution.missing_required)
            optional_union.update(resolution.missing_optional)
            column_map.update(resolution.mapping)
            if not resolution.missing_required:
                table = pf.read(columns=columns_to_read(resolution))
                inspected_frames.append(canonicalize_frame(table.to_pandas(), resolution))
        actual_columns[sample['name']] = column_map
        if inspected_frames:
            weights = weight_summary(pd.concat(inspected_frames, ignore_index=True))
        else:
            weights = {
                'event_count': 0,
                'negative_weight_events': 0,
                'negative_weight_fraction': 0.0,
                'signed_weight_sum': 0.0,
                'absolute_weight_sum': 0.0,
                'min_event_weight': None,
                'max_event_weight': None,
                'nan_weight_count': 0,
                'infinite_weight_count': 0,
            }
        weight_rows.append({
            'sample': sample['name'],
            'class': sample['class_name'],
            'events': weights['event_count'],
            'negative': weights['negative_weight_events'],
            'neg_frac': round(weights['negative_weight_fraction'], 6),
            'signed_sum': round(weights['signed_weight_sum'], 6),
            'abs_sum': round(weights['absolute_weight_sum'], 6),
            'nan': weights['nan_weight_count'],
            'inf': weights['infinite_weight_count'],
        })
        sample_rows.append({
            'sample': sample['name'],
            'class': sample['class_name'],
            'group': sample['source_group'],
            'files': len(files),
            'inspected_files': len(inspected),
            'inspected_rows': row_count,
            'missing_required': ', '.join(sorted(missing_union)) or 'none',
        })
        if missing_union:
            issue_rows.append({
                'sample': sample['name'],
                'missing_required': ', '.join(sorted(missing_union)),
                'inspected_files': len(inspected),
            })

    summary = {
        'generated_at': utc_now(),
        'stage1_root': str(stage1_root),
        'dataset_subdir': dataset_subdir,
        'samples_yaml': str(samples_yaml),
        'resolved_samples_yaml': str(resolved_path),
        'n_resolved_samples': len(resolved['samples']),
        'n_excluded_entries': len(resolved['excluded']),
        'max_files_per_sample': args.max_files_per_sample,
        'sample_rows': sample_rows,
        'weight_rows': weight_rows,
        'issues': issue_rows,
        'feature_contract': describe_feature_contract(),
        'actual_columns': actual_columns,
    }
    write_yaml(reports_dir / 'input_compatibility_summary.yaml', summary)

    report = [
        '# Stage1 Input Compatibility',
        '',
        f'Generated: `{summary["generated_at"]}`',
        '',
        f'- Stage1 root: `{stage1_root}`',
        f'- Dataset subdir: `{dataset_subdir}`',
        f'- Resolved samples: `{len(resolved["samples"])}`',
        f'- Excluded entries: `{len(resolved["excluded"])}`',
        '',
        '## Resolved Samples',
        '',
        format_table(sample_rows, ['sample', 'class', 'group', 'files', 'inspected_files', 'inspected_rows', 'missing_required']),
        '',
        '## Exclusions and Overlap Guards',
        '',
        format_table(resolved['excluded'], ['yaml_name', 'source_group', 'reason']),
        '',
        '## Final Ordered Feature List',
        '',
        f'- Object token order: `{", ".join(summary["feature_contract"]["object_order"])}`',
        f'- Object feature order: `{", ".join(summary["feature_contract"]["object_features"])}`',
        f'- Global feature order: `{", ".join(summary["feature_contract"]["global_features"])}`',
        '',
        '## Token, Padding, and Alias Rules',
        '',
        '- Five object tokens are built in the fixed order `mu1`, `mu2`, `jet1`, `jet2`, `dimuon`.',
        '- Padding is indicated by `physObj_pt_4v <= 0`; padded object feature rows are zeroed and masked from attention.',
        '- Pairwise attention uses raw `pt`, `eta`, `phi`, and `energy` slots from each object token.',
        '- Continuous object slots and all global features are standardized from the training split only during training.',
        '- Non-finite feature values are replaced with zero after bounded log/energy transformations.',
        '- `PuppiMET_sumEt` is used as the Stage1 proxy for `global_MET_ln_e`.',
        '',
        '## Inspected Weight Summary',
        '',
        format_table(weight_rows, ['sample', 'class', 'events', 'negative', 'neg_frac', 'signed_sum', 'abs_sum', 'nan', 'inf']),
        '',
        '## Compatibility Result',
        '',
    ]
    if issue_rows:
        report.extend(['Missing required columns were found:', '', format_table(issue_rows, ['sample', 'missing_required', 'inspected_files']), ''])
    else:
        report.extend([
            'All inspected files contained the required transformer columns after alias resolution.',
            '',
            'DY samples with `separate_wgt_zpt` will use `wgt_nominal / separate_wgt_zpt`; other samples use `wgt_nominal`.',
            'The available Copperhead MET energy-like branch is `PuppiMET_sumEt`, used as the global MET energy proxy.',
            '',
        ])
    (reports_dir / 'input_compatibility_report.md').write_text('\n'.join(report), encoding='utf-8')
    print(f'wrote {resolved_path}')
    print(f'wrote {reports_dir / "input_compatibility_report.md"}')


if __name__ == '__main__':
    main()
