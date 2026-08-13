"""Tests for train_dnn.py --init-from (warm start).

The 2017 fast-iteration study trains every lambda > 0 starting from a converged
lambda = 0 model rather than from random init.  What has to hold:

  * the weights really are the earlier run's, per fold, not a fresh draw;
  * only the *weights* cross over -- optimizer, LR schedule and early stopping
    are rebuilt, so the second phase runs to early stopping on its own;
  * a missing checkpoint fails loudly instead of silently training from scratch,
    which would look like a successful sweep point but measure nothing;
  * omitting the flag leaves the previous behaviour untouched.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from MVA_training.VBF_run3 import train_dnn
from tests.test_adversarial_syst import (
    _assert_states_identical,
    _state_dict_from,
    _tiny_cfg,
    _write_synthetic_folds,
)

VARIATIONS = ["Total_down", "Total_up", "mu_roccor_down", "mu_roccor_up"]


def _frozen_cfg():
    """A config whose training step is a no-op, so warm-started weights survive.

    lr = 0 freezes the parameters (AdamW's decoupled weight decay also scales
    with lr), and batch_norm = False removes the running statistics, which are
    buffers that would otherwise move even at lr = 0.  What is left is a run
    whose best.pt must be exactly what it was initialised with.
    """
    return replace(_tiny_cfg(), lr=0.0, batch_norm=False, epochs=1)


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    _write_synthetic_folds(d, _tiny_cfg(), VARIATIONS)
    return d


@pytest.mark.slow
def test_warm_start_loads_the_source_weights(tmp_path, data_dir):
    cfg = _frozen_cfg()

    source = tmp_path / "lam0"
    train_dnn.set_seed(cfg.seed)
    train_dnn.train_one_fold(0, cfg, str(data_dir), str(source))

    warm = tmp_path / "warm"
    train_dnn.set_seed(cfg.seed + 1)  # a different draw, so only the load can match
    train_dnn.train_one_fold(0, cfg, str(data_dir), str(warm), init_from=str(source))

    _assert_states_identical(_state_dict_from(source), _state_dict_from(warm))


@pytest.mark.slow
def test_warm_start_is_per_fold(tmp_path, data_dir):
    """Fold i must be initialised from fold i, not from fold 0 of the source."""
    cfg = _frozen_cfg()

    source = tmp_path / "lam0"
    for fold in (0, 1):
        train_dnn.set_seed(cfg.seed + fold)
        train_dnn.train_one_fold(fold, cfg, str(data_dir), str(source))

    src0 = _state_dict_from(source)
    src1 = torch.load(source / "fold1" / "best.pt", map_location="cpu",
                      weights_only=False)["model_state"]
    assert not all(torch.equal(src0[k], src1[k]) for k in src0), (
        "the two source folds came out identical; this test cannot distinguish them"
    )

    warm = tmp_path / "warm"
    train_dnn.set_seed(cfg.seed + 99)
    train_dnn.train_one_fold(1, cfg, str(data_dir), str(warm), init_from=str(source))

    got = torch.load(warm / "fold1" / "best.pt", map_location="cpu",
                     weights_only=False)["model_state"]
    _assert_states_identical(src1, got)


@pytest.mark.slow
def test_warm_start_still_trains(tmp_path, data_dir):
    """With a real learning rate the warm-started run must move off its init."""
    cfg = _tiny_cfg()

    source = tmp_path / "lam0"
    train_dnn.set_seed(cfg.seed)
    train_dnn.train_one_fold(0, cfg, str(data_dir), str(source))

    warm = tmp_path / "warm"
    train_dnn.set_seed(cfg.seed)
    train_dnn.train_one_fold(0, cfg, str(data_dir), str(warm), init_from=str(source))

    a, b = _state_dict_from(source), _state_dict_from(warm)
    assert not all(torch.equal(a[k], b[k]) for k in a), (
        "warm-started run did not update any weights -- the second phase did nothing"
    )


@pytest.mark.slow
def test_missing_checkpoint_raises(tmp_path, data_dir):
    cfg = _frozen_cfg()
    empty = tmp_path / "no-such-run"
    with pytest.raises(FileNotFoundError, match="fold 0"):
        train_dnn.train_one_fold(0, cfg, str(data_dir), str(tmp_path / "out"),
                                 init_from=str(empty))


@pytest.mark.slow
def test_no_init_from_is_unchanged(tmp_path, data_dir):
    """Default (None) must reproduce the pre-change path bit for bit."""
    cfg = _tiny_cfg()

    a = tmp_path / "a"
    train_dnn.set_seed(cfg.seed)
    train_dnn.train_one_fold(0, cfg, str(data_dir), str(a))

    b = tmp_path / "b"
    train_dnn.set_seed(cfg.seed)
    train_dnn.train_one_fold(0, cfg, str(data_dir), str(b), init_from=None)

    _assert_states_identical(_state_dict_from(a), _state_dict_from(b))


def test_cli_exposes_init_from():
    args = train_dnn.build_argparser().parse_args(
        ["--config", "c.yaml", "--data-dir", "d", "--out-dir", "o",
         "--init-from", "/some/run"]
    )
    assert args.init_from == "/some/run"
    assert Path(args.init_from).name == "run"


def test_cli_init_from_defaults_to_none():
    args = train_dnn.build_argparser().parse_args(
        ["--config", "c.yaml", "--data-dir", "d", "--out-dir", "o"]
    )
    assert args.init_from is None


def test_cli_epochs_override():
    base = ["--config", "c.yaml", "--data-dir", "d", "--out-dir", "o"]
    assert train_dnn.build_argparser().parse_args(base).epochs is None
    assert train_dnn.build_argparser().parse_args(base + ["--epochs", "1"]).epochs == 1


@pytest.mark.slow
def test_epochs_override_limits_the_loop(tmp_path, data_dir):
    """One epoch must produce a one-entry history, whatever the config says."""
    cfg = replace(_tiny_cfg(), epochs=1, save_history=True)
    out = tmp_path / "probe"
    train_dnn.set_seed(cfg.seed)
    train_dnn.train_one_fold(0, cfg, str(data_dir), str(out))

    import json
    history = json.loads((out / "fold0" / "history.json").read_text())
    assert len(history["train_loss"]) == 1
