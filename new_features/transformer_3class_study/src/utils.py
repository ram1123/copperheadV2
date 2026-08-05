from __future__ import annotations

import gc
import json
import logging
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

GIB = 1024 ** 3


def study_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base or repo_root()) / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f'Expected YAML mapping in {path}')
    return data


def write_yaml(path: str | Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        yaml.safe_dump(to_builtin(dict(data)), handle, sort_keys=False, width=120)


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(data), indent=2, sort_keys=False) + '\n', encoding='utf-8')


def to_builtin(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or type(value) in {str, int, float, bool}:
        return value
    if isinstance(value, str):
        return str(value)
    return str(value)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def runtime_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    runtime = dict(cfg.get('runtime') or {})
    legacy_device = dict(cfg.get('device') or {})
    cuda_cfg = dict(runtime.get('cuda') or {})
    return {
        'device': runtime.get('device', legacy_device.get('default', 'auto')),
        'require_cuda_when_available': bool(runtime.get('require_cuda_when_available', True)),
        'cuda': {
            'max_memory_gib': float(cuda_cfg.get('max_memory_gib', legacy_device.get('cuda_memory_limit_gib', 5.0))),
            'target_peak_allocated_gib': float(cuda_cfg.get('target_peak_allocated_gib', 4.5)),
            'automatic_mixed_precision': bool(cuda_cfg.get('automatic_mixed_precision', True)),
            'select_device_by_free_memory': bool(cuda_cfg.get('select_device_by_free_memory', True)),
            'allow_silent_cpu_fallback': bool(cuda_cfg.get('allow_silent_cpu_fallback', False)),
            'empty_cache_between_phases': bool(cuda_cfg.get('empty_cache_between_phases', True)),
        },
    }


def setup_logging(log_path: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding='utf-8'))
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        handlers=handlers,
        force=True,
    )
    return logging.getLogger('transformer_3class_study')


def git_info(path: str | Path) -> dict[str, str | None]:
    path = Path(path)
    out: dict[str, str | None] = {'path': str(path), 'branch': None, 'commit': None, 'status_short': None}
    if not path.exists():
        return out
    safe = f'safe.directory={path}'
    commands = {
        'branch': ['git', '-c', safe, 'rev-parse', '--abbrev-ref', 'HEAD'],
        'commit': ['git', '-c', safe, 'rev-parse', 'HEAD'],
        'status_short': ['git', '-c', safe, 'status', '--short'],
    }
    for key, cmd in commands.items():
        try:
            result = subprocess.run(cmd, cwd=path, check=False, capture_output=True, text=True)
        except OSError:
            continue
        if result.returncode == 0:
            out[key] = result.stdout.strip()
    return out


def git_log(path: str | Path, paths: list[str], max_count: int = 12) -> list[str]:
    path = Path(path)
    if not path.exists():
        return []
    safe = f'safe.directory={path}'
    cmd = ['git', '-c', safe, 'log', '--oneline', '--date=iso', f'--max-count={max_count}', '--', *paths]
    try:
        result = subprocess.run(cmd, cwd=path, check=False, capture_output=True, text=True)
    except OSError:
        return []
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def environment_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        'python_executable': sys.executable,
        'python_version': sys.version.replace('\n', ' '),
        'platform': platform.platform(),
        'torch_version': getattr(torch, '__version__', None),
        'cuda_available': bool(torch.cuda.is_available()),
        'torch_cuda_version': getattr(torch.version, 'cuda', None),
    }
    for module_name in ['awkward', 'coffea', 'numpy', 'pandas', 'pyarrow', 'yaml']:
        try:
            module = __import__(module_name)
        except Exception as exc:
            summary[f'{module_name}_version'] = f'unavailable: {exc}'
        else:
            summary[f'{module_name}_version'] = getattr(module, '__version__', 'unknown')
    return summary


def select_device(requested: str = 'auto', cuda_memory_limit_gib: float = 5.0) -> tuple[torch.device, dict[str, Any]]:
    requested = requested.lower()
    cuda_available = torch.cuda.is_available()
    info: dict[str, Any] = {
        'requested': requested,
        'cuda_available': bool(cuda_available),
        'selected': 'cpu',
        'cuda_memory_limit_gib': cuda_memory_limit_gib,
    }
    if requested not in {'auto', 'cuda', 'cpu'}:
        raise ValueError('--device must be one of auto, cuda, cpu')
    if requested == 'cuda' and not cuda_available:
        raise RuntimeError('CUDA was explicitly requested, but torch.cuda.is_available() is false.')
    if requested == 'auto':
        selected = 'cuda' if cuda_available else 'cpu'
    else:
        selected = requested
    device = torch.device(selected)
    info['selected'] = selected
    if selected == 'cuda':
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total_gib = props.total_memory / GIB
        fraction = min(max(cuda_memory_limit_gib, 0.1) / total_gib, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, idx)
        free_gib = None
        try:
            free_bytes, _ = torch.cuda.mem_get_info(idx)
            free_gib = round(free_bytes / GIB, 3)
        except Exception:
            pass
        info.update({
            'cuda_device_index': idx,
            'cuda_device_name': props.name,
            'cuda_total_memory_gib': round(total_gib, 3),
            'cuda_free_memory_gib_before': free_gib,
            'torch_cuda_version': getattr(torch.version, 'cuda', None),
            'cuda_allocator_fraction': round(fraction, 6),
            'cuda_allocator_limit_gib_effective': round(fraction * total_gib, 3),
            'cuda_allocator_function': 'torch.cuda.set_per_process_memory_fraction',
        })
    return device, info


def cuda_reset_peak(device: torch.device | None = None) -> None:
    if not torch.cuda.is_available():
        return
    idx = torch.cuda.current_device() if device is None else device.index
    if idx is None:
        idx = torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(idx)


def cuda_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cuda_memory_snapshot(device: torch.device | None = None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {'cuda_available': False}
    idx = torch.cuda.current_device() if device is None else device.index
    if idx is None:
        idx = torch.cuda.current_device()
    return {
        'cuda_available': True,
        'device_index': idx,
        'allocated_gib': round(torch.cuda.memory_allocated(idx) / GIB, 6),
        'reserved_gib': round(torch.cuda.memory_reserved(idx) / GIB, 6),
        'max_allocated_gib': round(torch.cuda.max_memory_allocated(idx) / GIB, 6),
        'max_reserved_gib': round(torch.cuda.max_memory_reserved(idx) / GIB, 6),
    }


def amp_info(device: torch.device, enabled_by_config: bool = True) -> dict[str, Any]:
    enabled = bool(device.type == 'cuda' and enabled_by_config)
    return {
        'amp_enabled': enabled,
        'amp_dtype': 'float16' if enabled else None,
        'gradient_scaler_enabled': enabled,
    }


def make_grad_scaler(enabled: bool, init_scale: float | None = None):
    if not enabled:
        return None
    kwargs: dict[str, Any] = {'enabled': True}
    if init_scale is not None:
        kwargs['init_scale'] = float(init_scale)
    try:
        return torch.amp.GradScaler('cuda', **kwargs)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(**kwargs)


def grad_scaler_scale(scaler) -> float | None:
    if scaler is None:
        return None
    try:
        return float(scaler.get_scale())
    except Exception:
        return None


def amp_step_is_skipped(scale_before: float | None, scale_after: float | None) -> bool:
    """True when GradScaler backed the loss scale off, i.e. it discarded this step.

    A skipped step is the scaler working as designed, not a model defect: it starts at a
    deliberately optimistic scale (65536 by default), lets the first backward overflow,
    then halves until the gradients fit in float16. Any finite-gradient assertion has to
    exclude these steps or it will reject healthy AMP on the very first iteration --
    which is exactly what disabled AMP in the previous iteration of this study.
    """
    if scale_before is None or scale_after is None:
        return False
    return scale_after < scale_before


def source_provenance(cfg: Mapping[str, Any]) -> dict[str, Any]:
    study_cfg = dict(cfg.get('study') or {})
    source_repo = Path(study_cfg.get('source_transformer_repo', ''))
    source_model = Path(study_cfg.get('source_model_file', ''))
    source_config = Path(study_cfg.get('source_model_config', ''))
    return {
        'source_repository': str(source_repo),
        'source_model_file': str(source_model),
        'source_model_config': str(source_config),
        'source_model_name': study_cfg.get('source_model_name'),
        'source_git': git_info(source_repo),
        'selection_note': 'Selected HMuMu ParticleTransformer implementation from source config HMuMu_ParT_12Apr2026 and transformer-related git history.',
    }


def ensure_dirs(paths: list[str | Path]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def format_table(rows: list[Mapping[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ''
    header = '| ' + ' | '.join(columns) + ' |'
    sep = '| ' + ' | '.join(['---'] * len(columns)) + ' |'
    body = []
    for row in rows:
        body.append('| ' + ' | '.join(str(row.get(col, '')) for col in columns) + ' |')
    return '\n'.join([header, sep, *body])
