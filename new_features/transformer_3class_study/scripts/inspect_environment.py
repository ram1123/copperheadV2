#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

STUDY_ROOT = Path(__file__).resolve().parents[1]
if str(STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT))

from src.utils import environment_summary, git_info, load_yaml, resolve_path, source_provenance, write_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Record the active Pixi/PyTorch environment for the transformer study.')
    parser.add_argument('--config', default='new_features/transformer_3class_study/configs/study_config.yaml')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_path(args.config))
    study_cfg = cfg['study']
    output_dir = resolve_path(study_cfg['output_dir'])
    output_dir.joinpath('metrics').mkdir(parents=True, exist_ok=True)
    summary = environment_summary()
    summary.update({
        'copperhead_git': git_info(resolve_path('.')),
        'source_model_provenance': source_provenance(cfg),
    })
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        summary.update({
            'cuda_device_index': idx,
            'cuda_device_name': props.name,
            'cuda_total_memory_gib': round(props.total_memory / (1024 ** 3), 3),
        })
    write_yaml(output_dir / 'metrics' / 'environment_summary.yaml', summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
