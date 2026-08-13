"""No-op evidence: with --use_adversarial off, training equals the PRE-CHANGE code.

This is the section-5.2 check at unit scale. It does not take the current
`train_dnn.py` on trust: it materialises the version at `git HEAD` (i.e. before
this task's edits), imports it as a separate module, trains both on the same
synthetic fold parquets with the same seed, and requires bit-identical weights.

Run with:
    pixi run -e default python -m pytest tests/test_noop_vs_pre_change.py -q
"""
from __future__ import annotations

import importlib.util
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from MVA_training.VBF_run3 import train_dnn as train_new


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "dnn_run2_vbf.yaml"
TARGET = "MVA_training/VBF_run3/train_dnn.py"
BASELINE_REF = "HEAD"


def _load_pre_change_module(tmp_path: Path):
    try:
        src = subprocess.run(
            ["git", "show", f"{BASELINE_REF}:{TARGET}"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"cannot read {BASELINE_REF}:{TARGET} from git: {exc}")

    path = tmp_path / "train_dnn_pre_change.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("train_dnn_pre_change", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_cfg(module):
    cfg = module.load_config(str(CONFIG))
    return replace(
        cfg,
        device="cpu",
        epochs=3,
        batch_size=64,
        n_folds=4,
        es_enable=False,
        plots_enable=False,
        save_torchscript=False,
        save_history=False,
        amp_enable=False,
        num_workers=0,
    )


def _write_folds(data_dir: Path, features, n_folds=4, n=512, seed=11):
    rng = np.random.default_rng(seed)
    physical = [f for f in features if not f.startswith("year_")]
    for split in ("train", "validation", "evaluation"):
        for i in range(n_folds):
            data = {f: rng.normal(size=n).astype(np.float32) for f in physical}
            for f in features:
                if f.startswith("year_"):
                    data[f] = np.zeros(n, dtype=np.float32)
            data["year_2018"] = np.ones(n, dtype=np.float32)
            data["label"] = (rng.random(n) > 0.5).astype(np.float32)
            data["wgt_nominal"] = rng.normal(1.0, 0.2, n).astype(np.float32)
            data["event"] = np.arange(n, dtype=np.int64)
            pd.DataFrame(data).to_parquet(
                data_dir / f"data_df_{split}_{i}.parquet", index=False
            )


def _weights(out_dir: Path, name: str):
    ckpt = torch.load(
        out_dir / "fold0" / name, map_location="cpu", weights_only=False
    )
    return ckpt["model_state"]


def test_switch_off_is_bit_identical_to_pre_change_code(tmp_path):
    old = _load_pre_change_module(tmp_path)

    cfg_new = _tiny_cfg(train_new)
    cfg_old = _tiny_cfg(old)
    assert cfg_new.training_features == cfg_old.training_features

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_folds(data_dir, cfg_new.training_features)

    out_old = tmp_path / "old"
    old.set_seed(cfg_old.seed)
    old.train_one_fold(0, cfg_old, str(data_dir), str(out_old))

    out_new = tmp_path / "new"
    train_new.set_seed(cfg_new.seed)
    train_new.train_one_fold(0, cfg_new, str(data_dir), str(out_new))

    for ckpt_name in ("best.pt", "last.pt"):
        a = _weights(out_old, ckpt_name)
        b = _weights(out_new, ckpt_name)
        assert set(a) == set(b)
        for k in a:
            assert torch.equal(a[k], b[k]), f"{ckpt_name}: tensor '{k}' differs"


def test_switch_off_produces_identical_scores(tmp_path):
    """Saved-score equivalence, not just weights."""
    old = _load_pre_change_module(tmp_path)
    cfg_new = _tiny_cfg(train_new)
    cfg_old = _tiny_cfg(old)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_folds(data_dir, cfg_new.training_features)

    out_old = tmp_path / "old"
    old.set_seed(cfg_old.seed)
    old.train_one_fold(0, cfg_old, str(data_dir), str(out_old))
    out_new = tmp_path / "new"
    train_new.set_seed(cfg_new.seed)
    train_new.train_one_fold(0, cfg_new, str(data_dir), str(out_new))

    df = pd.read_parquet(data_dir / "data_df_evaluation_0.parquet")
    x = torch.from_numpy(
        df[cfg_new.training_features].to_numpy(dtype=np.float32, copy=True)
    )

    def _score(module, out_dir, cfg):
        model = module.MLP(
            input_dim=len(cfg.training_features),
            hidden=cfg.hidden,
            activation=cfg.activation,
            dropout=cfg.dropout,
            batch_norm=cfg.batch_norm,
        )
        model.load_state_dict(_weights(out_dir, "best.pt"))
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(x))

    s_old = _score(old, out_old, cfg_old)
    s_new = _score(train_new, out_new, cfg_new)
    assert torch.equal(s_old, s_new)
