#!/usr/bin/env python3
"""
Train a binary DNN (MLP) using preprocessed fold-parquet files produced by preprocess_dnn.py.

Inputs (per fold i):
  - data_df_train_{i}.parquet
  - data_df_validation_{i}.parquet
  - data_df_evaluation_{i}.parquet
  - scalers_{i}.npz  (optional; used mainly for bookkeeping)

Outputs (per fold i):
  - best.pt / last.pt
  - history.json
  - metrics_summary.json
  - plots/ (ROC, PR, score hists, calibration, KS, corr, learning curves)

Notes:
- Uses BCEWithLogitsLoss (returns_logits=true).
- Supports weighted loss if cfg.data.weights.use_in_loss=true.
- Weighted metrics via sklearn sample_weight where available.
- Calibration/KS implemented manually (weighted).

python train_dnn.py \
  --config configs/dnn_run3_vbf.yaml \
  --data-dir dnn/trained_models/<TAG>/<YEAR>_<REGION>_<CAT> \
  --out-dir   dnn/trained_models/<TAG>/<YEAR>_<REGION>_<CAT>/trained \
  --log-level INFO
"""

import argparse
import copy
import json
import math
import os
import random
import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import yaml
from modules.git_utils import get_git_commit, get_git_state
from modules.systematics import VARIATION_COL_SEP
from modules.utils import logger
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset


# --------------------------------------------------------------------------------
# AUC helper (sklearn missing sample_weight in some versions)
# --------------------------------------------------------------------------------
def safe_sample_weight(w: np.ndarray, mode: str = "abs") -> np.ndarray:
    """
    Make sample_weight safe for sklearn ranking metrics.
    mode:
      - "abs": use abs(w)  (recommended for HEP)
      - "clip0": negative -> 0
      - "none": return w (unsafe if negative)
    """
    w = np.asarray(w, dtype=np.float64)
    if mode == "abs":
        w = np.abs(w)
    elif mode == "clip0":
        w = np.clip(w, 0.0, None)
    elif mode == "none":
        pass
    else:
        raise ValueError(f"Unknown weight mode: {mode}")

    # avoid all-zero weights
    if not np.isfinite(w).all():
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    if np.sum(w) <= 0:
        # fallback to uniform
        w = np.ones_like(w, dtype=np.float64)
    return w


def safe_auc(
    y_true: np.ndarray, y_score: np.ndarray, sample_weight: np.ndarray
) -> float:
    """
    Robust weighted AUC:
      - guards single-class edge case
      - ensures weights are safe
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score, sample_weight=sample_weight))


def safe_ap(
    y_true: np.ndarray, y_score: np.ndarray, sample_weight: np.ndarray
) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score, sample_weight=sample_weight))


# --------------------------------------------------------------------------------------
# Dataclasses (config view)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    seed: int
    device: str
    dtype: str

    n_folds: int

    training_features: List[str]
    weight_col: str
    label_col: str

    # model
    hidden: List[int]
    activation: str
    dropout: List[float]
    batch_norm: bool
    returns_logits: bool

    # training
    epochs: int
    batch_size: int
    num_workers: int
    prefetch_factor: Optional[int]
    pin_memory: bool

    # optim
    optimizer_name: str
    lr: float
    weight_decay: float
    betas: Tuple[float, float]
    eps: float

    # scheduler
    scheduler_name: str
    rop_mode: str
    rop_factor: float
    rop_patience: int
    rop_min_lr: float
    rop_threshold: float

    # loss
    loss_name: str
    pos_weight: Optional[float]
    label_smoothing: float
    use_weight_in_loss: bool
    normalize_in_batch: bool

    # regularization
    grad_clip_norm: float
    amp_enable: bool

    # early stopping
    es_enable: bool
    es_monitor: str
    es_mode: str
    es_patience: int
    es_min_delta: float
    es_warmup_epochs: int
    es_restore_best: bool

    # plots
    plots_enable: bool
    plots_subdir: str
    score_bins: int
    score_range: Tuple[float, float]
    score_normalize: bool
    score_logy: bool
    roc_points: int
    calib_bins: int
    corr_topk: int

    # artifacts
    save_best: bool
    save_last: bool
    save_history: bool
    save_predictions: bool
    save_torchscript: bool


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(cfg_path: str) -> TrainConfig:
    cfg = _load_yaml(cfg_path)

    seed = int(cfg["meta"]["seed"])
    device = str(cfg["meta"].get("device", "cpu"))
    dtype = str(cfg["meta"].get("dtype", "float32"))

    n_folds = int(cfg["data"]["cv"]["n_folds"])

    training_features = list(cfg["features"]["training"])
    weight_col = str(cfg["data"]["weights"]["weight_col"])
    label_col = "label"

    model = cfg["model"]
    hidden = list(model["hidden"])
    activation = str(model.get("activation", "relu"))
    dropout = list(model.get("dropout", [0.0] * len(hidden)))
    batch_norm = bool(model.get("batch_norm", False))
    returns_logits = bool(model.get("output", {}).get("returns_logits", True))

    train = cfg["training"]
    epochs = int(train["epochs"])
    batch_size = int(train["batch_size"])

    dl = train.get("dataloader", {})
    num_workers = int(dl.get("num_workers", 0))
    prefetch_factor = dl.get("prefetch_factor", None)
    pin_memory = bool(dl.get("pin_memory", False))

    opt = train["optimizer"]
    optimizer_name = str(opt.get("name", "adamw"))
    lr = float(opt["lr"])
    weight_decay = float(opt.get("weight_decay", 0.0))
    betas = tuple(opt.get("betas", [0.9, 0.999]))
    eps = float(opt.get("eps", 1e-8))

    sch = train.get("scheduler", {"name": "none"})
    scheduler_name = str(sch.get("name", "none"))
    rop = sch.get("reduce_on_plateau", {})
    rop_mode = str(rop.get("mode", "min"))
    rop_factor = float(rop.get("factor", 0.5))
    rop_patience = int(rop.get("patience", 2))
    rop_min_lr = float(rop.get("min_lr", 1e-6))
    rop_threshold = float(rop.get("threshold", 1e-4))

    loss = train.get("loss", {"name": "bce_logits"})
    loss_name = str(loss.get("name", "bce_logits"))
    bce = loss.get("bce_logits", {})
    pos_weight = bce.get("pos_weight", None)
    pos_weight = None if pos_weight in (None, "null") else float(pos_weight)
    label_smoothing = float(bce.get("label_smoothing", 0.0))

    use_weight_in_loss = bool(cfg["data"]["weights"].get("use_in_loss", True))
    normalize_in_batch = bool(cfg["data"]["weights"].get("normalize_in_batch", True))

    reg = train.get("regularization", {})
    grad_clip_norm = float(reg.get("gradient_clip_norm", 0.0))
    amp_enable = bool(reg.get("amp", {}).get("enable", False))

    es = cfg.get("early_stopping", {})
    es_enable = bool(es.get("enable", True))
    es_monitor = str(es.get("monitor", "val_loss"))
    es_mode = str(es.get("mode", "min"))
    es_patience = int(es.get("patience", 5))
    es_min_delta = float(es.get("min_delta", 1e-4))
    es_warmup_epochs = int(es.get("warmup_epochs", 0))
    es_restore_best = bool(es.get("restore_best", True))

    plots = cfg.get("plots", {})
    plots_enable = bool(plots.get("enable", True))
    plots_subdir = str(plots.get("out_subdir", "plots"))

    score_hist = plots.get("score_hist", {})
    score_bins = int(score_hist.get("bins", 30))
    score_range = tuple(score_hist.get("range", [0.0, 1.0]))
    score_normalize = bool(score_hist.get("normalize", True))
    score_logy = bool(score_hist.get("logy", False))

    roc_cfg = plots.get("roc", {})
    roc_points = int(roc_cfg.get("points", 300))

    calib_cfg = plots.get("calibration", {})
    calib_bins = int(calib_cfg.get("n_bins", 10))

    corr_cfg = plots.get("corr", {})
    corr_topk = int(corr_cfg.get("topk", 30))

    art = cfg.get("artifacts", {})
    save_best = bool(art.get("save_best_model", True))
    save_last = bool(art.get("save_last_model", True))
    save_history = bool(art.get("save_history", True))
    save_predictions = bool(art.get("save_predictions", False))
    save_torchscript = bool(art.get("save_torchscript", True))

    return TrainConfig(
        seed=seed,
        device=device,
        dtype=dtype,
        n_folds=n_folds,
        training_features=training_features,
        weight_col=weight_col,
        label_col=label_col,
        hidden=hidden,
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm,
        returns_logits=returns_logits,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=None if prefetch_factor in (None, "null") else int(prefetch_factor),
        pin_memory=pin_memory,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        betas=(float(betas[0]), float(betas[1])),
        eps=eps,
        scheduler_name=scheduler_name,
        rop_mode=rop_mode,
        rop_factor=rop_factor,
        rop_patience=rop_patience,
        rop_min_lr=rop_min_lr,
        rop_threshold=rop_threshold,
        loss_name=loss_name,
        pos_weight=pos_weight,
        label_smoothing=label_smoothing,
        use_weight_in_loss=use_weight_in_loss,
        normalize_in_batch=normalize_in_batch,
        grad_clip_norm=grad_clip_norm,
        amp_enable=amp_enable,
        es_enable=es_enable,
        es_monitor=es_monitor,
        es_mode=es_mode,
        es_patience=es_patience,
        es_min_delta=es_min_delta,
        es_warmup_epochs=es_warmup_epochs,
        es_restore_best=es_restore_best,
        plots_enable=plots_enable,
        plots_subdir=plots_subdir,
        score_bins=score_bins,
        score_range=(float(score_range[0]), float(score_range[1])),
        score_normalize=score_normalize,
        score_logy=score_logy,
        roc_points=roc_points,
        calib_bins=calib_bins,
        corr_topk=corr_topk,
        save_best=save_best,
        save_last=save_last,
        save_history=save_history,
        save_predictions=save_predictions,
        save_torchscript=save_torchscript,
    )


# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    # determinism tradeoff (slower). For training stability debugging, you can enable:
    # torch.use_deterministic_algorithms(True)


# --------------------------------------------------------------------------------------
# Dataset / Dataloader
# --------------------------------------------------------------------------------------
class ParquetDataset(Dataset):
    """Nominal fold data, optionally carrying systematic-variation inputs.

    ``variation_spec`` is ``[(variation_name, {feature: augmented_column}), ...]``.
    When it is ``None`` (the default, and always the case without
    ``--use_adversarial``) this class behaves exactly as before and yields the
    original 3-tuple, so the non-adversarial training path is untouched.

    Varied inputs are *not* materialised as a dense ``[N, V, F]`` block: a JEC
    variation only shifts the jet-side features and ``mu_roccor`` only the
    dimuon-side ones, so only the shifted columns are stored and the rest of each
    variation row is the nominal row. A feature absent from a variation's map is
    therefore identical to nominal by construction and contributes exactly zero
    to the consistency term -- the same fallback Stage-2 applies at inference.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        label_col: str,
        weight_col: str,
        dtype: str,
        variation_spec: Optional[List[Tuple[str, Dict[str, str]]]] = None,
    ) -> None:
        self.features = features
        self.label_col = label_col
        self.weight_col = weight_col

        x = df[features].to_numpy(dtype=np.float32 if dtype == "float32" else np.float64, copy=True)
        y = df[label_col].to_numpy(dtype=np.float32, copy=True)
        w = df[weight_col].to_numpy(dtype=np.float32, copy=True)

        self.x = x
        self.y = y
        self.w = w

        self.variation_names: List[str] = []
        self.n_variations = 0
        if variation_spec:
            feat_pos = {f: i for i, f in enumerate(features)}
            cols: List[str] = []
            slot_idx: List[int] = []
            feat_idx: List[int] = []
            for vi, (vname, fmap) in enumerate(variation_spec):
                self.variation_names.append(vname)
                for feat, col in sorted(fmap.items()):
                    if feat not in feat_pos:
                        raise KeyError(
                            f"Variation '{vname}' maps feature '{feat}' which is not a "
                            "training feature."
                        )
                    if col not in df.columns:
                        raise KeyError(
                            f"Missing variation column '{col}' (variation '{vname}', "
                            f"feature '{feat}') in the fold parquet. Point --data-dir at "
                            "a directory produced with --include-systematic-variations."
                        )
                    cols.append(col)
                    slot_idx.append(vi)
                    feat_idx.append(feat_pos[feat])

            self.n_variations = len(variation_spec)
            self.var_slot_idx = np.asarray(slot_idx, dtype=np.int64)
            self.var_feat_idx = np.asarray(feat_idx, dtype=np.int64)
            xvar = df[cols].to_numpy(dtype=x.dtype, copy=True)

            # NaN/Inf guard: a non-finite varied value falls back to its nominal
            # counterpart, which makes that (event, variation, feature) contribute
            # exactly zero to the consistency term rather than poisoning the loss.
            bad = ~np.isfinite(xvar)
            n_bad = int(bad.sum())
            if n_bad:
                nominal_broadcast = self.x[:, self.var_feat_idx]
                xvar[bad] = nominal_broadcast[bad]
                logger.warning(
                    "[dataset] %d non-finite varied values replaced by nominal "
                    "(%.3g%% of %d).",
                    n_bad,
                    100.0 * n_bad / max(xvar.size, 1),
                    xvar.size,
                )
            self.xvar = xvar

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        if self.n_variations == 0:
            return (
                torch.from_numpy(self.x[idx]),
                torch.from_numpy(np.array(self.y[idx])),
                torch.from_numpy(np.array(self.w[idx])),
            )
        xi = self.x[idx]
        xv = np.repeat(xi[None, :], self.n_variations, axis=0)
        xv[self.var_slot_idx, self.var_feat_idx] = self.xvar[idx]
        return (
            torch.from_numpy(xi),
            torch.from_numpy(np.array(self.y[idx])),
            torch.from_numpy(np.array(self.w[idx])),
            torch.from_numpy(xv),
        )


def make_dataloader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: Optional[int],
    pin_memory: bool,
) -> DataLoader:
    kwargs: Dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    # prefetch_factor must be None if num_workers == 0
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor if prefetch_factor is not None else 2
        kwargs["persistent_workers"] = True
    return DataLoader(ds, **kwargs)


@lru_cache(maxsize=64)
def _load_fold_data_cached(
    train_path: str,
    val_path: str,
    eval_path: str,
    training_features: Tuple[str, ...],
    label_col: str,
    weight_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cache fold parquet loading across HPO trials within one Python process."""

    def _read_one(path_str: str) -> pd.DataFrame:
        path = Path(path_str)
        df = pd.read_parquet(path)
        for c in list(training_features) + [label_col, weight_col]:
            if c not in df.columns:
                raise KeyError(f"Missing column '{c}' in {path}")

        df[list(training_features)] = df[list(training_features)].apply(
            pd.to_numeric, errors="coerce"
        )
        return df.dropna(subset=list(training_features))

    return (
        _read_one(train_path),
        _read_one(val_path),
        _read_one(eval_path),
    )


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: List[int],
        activation: str,
        dropout: List[float],
        batch_norm: bool,
    ) -> None:
        super().__init__()
        if len(dropout) != len(hidden):
            raise ValueError("dropout list must have same length as hidden")

        layers: List[nn.Module] = []
        in_dim = input_dim
        act = _get_activation(activation)

        for h, p in zip(hidden, dropout):
            layers.append(nn.Linear(in_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act.__class__())  # fresh module instance
            if p and p > 0:
                layers.append(nn.Dropout(p))
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns logits shape [B]
        return self.net(x).squeeze(1)

def export_torchscript(model: nn.Module, n_features: int, out_path: Path) -> None:
    model_copy = copy.deepcopy(model).to("cpu").eval()
    example = torch.zeros(1, n_features, dtype=torch.float32)
    ts = torch.jit.trace(model_copy, example)
    ts.save(str(out_path))

# --------------------------------------------------------------------------------------
# Loss / Metrics
# --------------------------------------------------------------------------------------
def bce_with_logits_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: Optional[torch.Tensor],
    pos_weight: Optional[torch.Tensor],
    label_smoothing: float,
    normalize_in_batch: bool,
) -> torch.Tensor:
    """
    Weighted BCEWithLogitsLoss:
      - if weights is None -> standard mean
      - else -> weighted mean (optionally normalize weights in batch)
    """
    y = targets
    if label_smoothing > 0.0:
        y = y * (1.0 - label_smoothing) + 0.5 * label_smoothing

    # elementwise BCE (no reduction)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    per = loss_fn(logits, y)

    if weights is None:
        return per.mean()

    # sync weigth
    w_eff = make_loss_weights(
        weights,
        normalize_in_batch=normalize_in_batch,
        clip_abs_max=None,
    )
    return torch.sum(per * w_eff) / (torch.sum(torch.abs(w_eff)) + 1e-12)


def make_loss_weights(
    w: torch.Tensor,
    normalize_in_batch: bool = True,
    clip_abs_max: float | None = None,
) -> torch.Tensor:
    # w shape: (B,)
    w_eff = torch.abs(w)  # IMPORTANT for training stability in HEP
    if clip_abs_max is not None:
        w_eff = torch.clamp(w_eff, max=float(clip_abs_max))

    if normalize_in_batch:
        denom = torch.sum(w_eff)
        if denom > 0:
            w_eff = w_eff * (w_eff.numel() / denom)  # mean weight = 1
        else:
            w_eff = torch.ones_like(w_eff)
    return w_eff


def training_loop_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    wb: torch.Tensor,
    pos_weight: Optional[torch.Tensor],
    label_smoothing: float,
    normalize_in_batch: bool,
) -> torch.Tensor:
    """`original_loss` as the training loop has always computed it.

    This is a verbatim extraction of the inline block that used to sit in
    `train_one_fold`'s batch loop -- same calls, same operand order -- so the
    non-adversarial path stays bit-for-bit identical while the adversarial terms
    reuse exactly the same functional form.

    NOTE (pre-existing behaviour, deliberately preserved): `bce_with_logits_loss`
    with `weights=None` already returns the *scalar* mean BCE, so multiplying by
    `loss_w` and dividing by `sum|loss_w|` cancels and this reduces to the
    unweighted mean. The per-event weights therefore do not currently reach the
    training loss (they do reach `evaluate()`). Changing that would move the
    baseline and is out of scope here; see assumptions.md.
    """
    loss_raw = bce_with_logits_loss(
        logits=logits,
        targets=targets,
        weights=None,
        pos_weight=pos_weight,
        label_smoothing=label_smoothing,
        normalize_in_batch=False,
    )
    loss_w = make_loss_weights(
        w=wb,
        normalize_in_batch=normalize_in_batch,
        clip_abs_max=None,
    )
    return torch.sum(loss_raw * loss_w) / (torch.sum(torch.abs(loss_w)) + 1e-12)


@dataclass(frozen=True)
class AdversarialConfig:
    """Settings for the systematics-consistency term. See assumptions.md."""

    lam: float
    #: stop-gradient on the nominal prediction used as the consistency target
    detach_nominal: bool
    #: "inherit" reuses cfg.label_smoothing for original_loss(pred_i, pred_nom);
    #: "none" uses 0.0 there (the soft target is already a probability)
    consistency_label_smoothing: str
    #: number of variations per forward chunk; <=0 means all in one chunk
    variation_chunk: int
    #: VARIANT B -- restrict the consistency term to events whose nominal score is
    #: high, `arctanh(p_nominal) > high_score_cut`. The significance comes from the
    #: top score bins, so decorrelating across the bulk spends discrimination where
    #: it cannot pay. None disables the cut.
    high_score_cut: Optional[float] = None
    #: STEP 2 of the three-step schedule -- keep only `original_loss(pred_i, label)`
    #: and drop the consistency term entirely, i.e. the variations are trained
    #: against the truth label alongside the nominal.
    label_only: bool = False
    #: Apply `high_score_cut` to the label term as well. The three-step schedule
    #: writes `[score_cut]` on every term inside the lambda sum, whereas the
    #: original variant-B formula masked the consistency term alone; this flag is
    #: what separates the two readings. No effect when `high_score_cut` is None.
    mask_label_term: bool = False


def adversarial_penalty(
    logits_var: torch.Tensor,
    p_nominal: torch.Tensor,
    yb: torch.Tensor,
    wb: torch.Tensor,
    pos_weight: Optional[torch.Tensor],
    cfg: TrainConfig,
    adv: AdversarialConfig,
) -> torch.Tensor:
    """`sum_i [ 2*original_loss(pred_i, pred_nominal) + original_loss(pred_i, label) ]`.

    `logits_var` is `[B, V]`; `p_nominal` is `[B]` in (0,1). `up` and `down` are
    independent entries of the V axis, so the sum treats them independently, as
    the task formula implies. The sum is plain (un-normalised).

    Two forms are selectable, corresponding to the steps of the three-step
    schedule:

    * `label_only`      -> `sum_i original_loss(pred_i, label)`            (step 2)
    * default           -> the full formula above                          (step 3)

    `high_score_cut` masks the consistency term always, and the label term as well
    when `mask_label_term` is set.
    """
    consistency_ls = (
        cfg.label_smoothing if adv.consistency_label_smoothing == "inherit" else 0.0
    )

    # VARIANT B: keep only the high-score events in the consistency term.
    # `arctanh(p) > c` is equivalent to `p > tanh(c)` for c > 0 and avoids
    # arctanh's blow-up as p -> 1, which is exactly where the selected events sit.
    # The mask is a selection, not a differentiable quantity, so it is detached.
    keep = None
    empty = False
    if adv.high_score_cut is not None:
        keep = p_nominal.detach() > math.tanh(adv.high_score_cut)
        # No event in this batch is in the high-score region. Slicing to an empty
        # tensor would give NaN, so every term the cut applies to is skipped below.
        # This is not a rare edge case: from a cold start the model predicts ~0.5
        # everywhere, so NO event clears the cut and every run hits it on its first
        # batch.
        empty = not bool(keep.any())

    # STEP 2 of the three-step schedule drops the consistency term; step 3 keeps
    # both terms with the factor of 2 on the consistency side.
    include_consistency = not adv.label_only
    consistency_coeff = 2.0
    # The consistency term is always masked when a cut is set. The label term is
    # masked only when `mask_label_term` is set, which is what separates the
    # three-step reading from the earlier variant-B formula.
    mask_label = adv.mask_label_term

    total = logits_var.new_zeros(())
    contributed = False
    for v in range(logits_var.shape[1]):
        lv = logits_var[:, v]
        if include_consistency and not empty:
            if keep is None:
                total = total + consistency_coeff * training_loop_loss(
                    logits=lv,
                    targets=p_nominal,
                    wb=wb,
                    pos_weight=pos_weight,
                    label_smoothing=consistency_ls,
                    normalize_in_batch=cfg.normalize_in_batch,
                )
            else:
                total = total + consistency_coeff * training_loop_loss(
                    logits=lv[keep],
                    targets=p_nominal[keep],
                    wb=wb[keep],
                    pos_weight=pos_weight,
                    label_smoothing=consistency_ls,
                    normalize_in_batch=cfg.normalize_in_batch,
                )
            contributed = True
        if keep is None or not mask_label:
            total = total + training_loop_loss(
                logits=lv,
                targets=yb,
                wb=wb,
                pos_weight=pos_weight,
                label_smoothing=cfg.label_smoothing,
                normalize_in_batch=cfg.normalize_in_batch,
            )
            contributed = True
        elif not empty:
            total = total + training_loop_loss(
                logits=lv[keep],
                targets=yb[keep],
                wb=wb[keep],
                pos_weight=pos_weight,
                label_smoothing=cfg.label_smoothing,
                normalize_in_batch=cfg.normalize_in_batch,
            )
            contributed = True

    if not contributed:
        # Every term was cut away. The zero must stay attached to the graph: a bare
        # `new_zeros(())` has no grad_fn and the caller's
        # `scaler.scale(term).backward()` then raises "element 0 of tensors does not
        # require grad". Multiplying by zero keeps the graph and contributes exactly
        # zero gradient.
        return logits_var.sum() * 0.0
    return total


def _variation_chunks(n_variations: int, chunk: int) -> List[Tuple[int, int]]:
    if chunk is None or chunk <= 0 or chunk >= n_variations:
        return [(0, n_variations)]
    return [(s, min(s + chunk, n_variations)) for s in range(0, n_variations, chunk)]


def load_variation_spec(data_dir: str) -> List[Tuple[str, Dict[str, str]]]:
    """Read the variation axis written by `preprocess_dnn.py --include-systematic-variations`.

    Returns `[(canonical_variation, {feature: augmented_column}), ...]` ordered by
    canonical variation name, so `up` and `down` of a source sit next to each other
    and the ordering is reproducible across runs.
    """
    manifest_path = Path(data_dir) / "preprocess_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"--use_adversarial needs {manifest_path}, written by preprocess_dnn.py."
        )
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    block = manifest.get("systematic_variations") or {}
    if not block.get("enabled"):
        raise ValueError(
            f"{manifest_path} has no systematic-variation block "
            "(systematic_variations.enabled is not true). Re-run preprocess_dnn.py "
            "with --include-systematic-variations and point --data-dir at that output."
        )

    canonical = list(block["canonical_variations"])
    col_to_feat: Dict[str, str] = dict(block["column_to_nominal_feature"])

    known = set(canonical)
    per_variation: Dict[str, Dict[str, str]] = {v: {} for v in canonical}
    for col, feat in col_to_feat.items():
        if VARIATION_COL_SEP not in col:
            raise ValueError(
                f"Augmented column '{col}' is missing the '{VARIATION_COL_SEP}' separator."
            )
        variation = col.rsplit(VARIATION_COL_SEP, 1)[1]
        if variation not in known:
            raise ValueError(
                f"Augmented column '{col}' names variation '{variation}', which is not "
                f"in the manifest's canonical_variations."
            )
        per_variation[variation][feat] = col

    empty = [v for v in canonical if not per_variation[v]]
    if empty:
        raise ValueError(f"Variations with no shifted feature columns: {empty}")

    return [(v, per_variation[v]) for v in canonical]


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def weighted_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: Optional[np.ndarray],
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weighted calibration curve (reliability diagram).
    Returns (bin_mean_pred, bin_frac_pos), excluding empty bins.
    """
    y_true = y_true.astype(np.float64)
    y_prob = y_prob.astype(np.float64)
    w = np.ones_like(y_true, dtype=np.float64) if sample_weight is None else sample_weight.astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    mean_pred = []
    frac_pos = []
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        wb = np.abs(w[m])
        wsum = wb.sum()
        if wsum <= 0:
            continue
        mean_pred.append(np.sum(wb * y_prob[m]) / wsum)
        frac_pos.append(np.sum(wb * y_true[m]) / wsum)

    return np.array(mean_pred), np.array(frac_pos)


def weighted_ks_distance(
    a: np.ndarray, wa: np.ndarray, b: np.ndarray, wb: np.ndarray
) -> float:
    """
    Weighted KS distance between two 1D distributions.
    Returns max|CDFa - CDFb| using weights (absolute weights).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    wa = np.abs(np.asarray(wa, dtype=np.float64))
    wb = np.abs(np.asarray(wb, dtype=np.float64))

    # sort values
    sa = np.argsort(a)
    sb = np.argsort(b)
    a_sorted, wa_sorted = a[sa], wa[sa]
    b_sorted, wb_sorted = b[sb], wb[sb]

    cdfa = np.cumsum(wa_sorted) / (np.sum(wa_sorted) + 1e-12)
    cdfb = np.cumsum(wb_sorted) / (np.sum(wb_sorted) + 1e-12)

    # evaluate both CDFs on union grid
    grid = np.unique(np.concatenate([a_sorted, b_sorted]))
    # interpolate stepwise: for each grid point, CDF is last value with x<=grid
    def step_cdf(x_sorted, cdf_sorted, x_grid):
        out = np.searchsorted(x_sorted, x_grid, side="right") - 1
        out = np.clip(out, -1, len(cdf_sorted) - 1)
        res = np.zeros_like(x_grid, dtype=np.float64)
        m = out >= 0
        res[m] = cdf_sorted[out[m]]
        return res

    Ca = step_cdf(a_sorted, cdfa, grid)
    Cb = step_cdf(b_sorted, cdfb, grid)
    return float(np.max(np.abs(Ca - Cb)))


# --------------------------------------------------------------------------------------
# Plotting helpers (matplotlib only)
# --------------------------------------------------------------------------------------
def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_learning_curves(outdir: str, history: Dict[str, List[float]]) -> None:
    _ensure_dir(outdir)
    # loss
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curves.pdf"))
    plt.close()

    # auc
    if "train_auc_w" in history and "val_auc_w" in history:
        plt.figure()
        plt.plot(history["train_auc_w"], label="train_auc_w")
        plt.plot(history["val_auc_w"], label="val_auc_w")
        plt.xlabel("epoch")
        plt.ylabel("AUC (weighted)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "auc_curves_weighted.pdf"))
        plt.close()


def plot_roc(outdir: str, y_true: np.ndarray, y_prob: np.ndarray, w: Optional[np.ndarray]) -> Dict[str, float]:
    _ensure_dir(outdir)
    # weighted AUC
    auc_w = float(roc_auc_score(y_true, y_prob, sample_weight=w)) if w is not None else float(roc_auc_score(y_true, y_prob))
    fpr, tpr, _ = roc_curve(y_true, y_prob, sample_weight=w)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc_w:.4f}")
    plt.plot([0, 1], [0, 1])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_weighted.pdf"))
    plt.close()
    return {"auc_w": auc_w}


def plot_pr(outdir: str, y_true: np.ndarray, y_prob: np.ndarray, w: Optional[np.ndarray]) -> Dict[str, float]:
    _ensure_dir(outdir)
    precision, recall, _ = precision_recall_curve(y_true, y_prob, sample_weight=w)
    ap = float(average_precision_score(y_true, y_prob, sample_weight=w)) if w is not None else float(average_precision_score(y_true, y_prob))

    plt.figure()
    plt.plot(recall, precision, label=f"AP={ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pr_weighted.pdf"))
    plt.close()
    return {"ap_w": ap}


def plot_score_hists(
    outdir: str,
    cfg: TrainConfig,
    train: Dict[str, np.ndarray],
    val: Dict[str, np.ndarray],
) -> None:
    _ensure_dir(outdir)
    bins = cfg.score_bins
    lo, hi = cfg.score_range

    def _hist(x, w):
        h, edges = np.histogram(x, bins=bins, range=(lo, hi), weights=w, density=cfg.score_normalize)
        return h, edges

    plt.figure()
    for label_name, d in [("train", train), ("val", val)]:
        # signal
        ms = d["y"] == 1
        mb = d["y"] == 0
        hs, edges = _hist(d["p"][ms], d["w"][ms])
        hb, _ = _hist(d["p"][mb], d["w"][mb])
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.step(centers, hs, where="mid", label=f"sig {label_name}")
        plt.step(centers, hb, where="mid", label=f"bkg {label_name}")

    plt.xlabel("DNN score (sigmoid)")
    plt.ylabel("density" if cfg.score_normalize else "events")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "score_hists_train_vs_val.pdf"))

    # also save log scale version
    plt.yscale("log")
    plt.savefig(os.path.join(outdir, "score_hists_train_vs_val_logy.pdf"))
    plt.close()


def plot_calibration(outdir: str, y_true: np.ndarray, y_prob: np.ndarray, w: Optional[np.ndarray], n_bins: int) -> None:
    _ensure_dir(outdir)
    mp, fp = weighted_calibration_curve(y_true, y_prob, w, n_bins=n_bins)

    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.plot(mp, fp, marker="o")
    plt.xlabel("mean predicted prob")
    plt.ylabel("fraction of positives")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "calibration_weighted.pdf"))
    plt.close()


def plot_ks_overtraining(
    outdir: str,
    train: Dict[str, np.ndarray],
    val: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    KS(sig train vs val), KS(bkg train vs val) using weighted CDF distance.
    """
    _ensure_dir(outdir)
    ks: Dict[str, float] = {}
    for cls, name in [(1, "sig"), (0, "bkg")]:
        mt = train["y"] == cls
        mv = val["y"] == cls
        ks_val = weighted_ks_distance(train["p"][mt], train["w"][mt], val["p"][mv], val["w"][mv])
        ks[name] = ks_val

    # quick plot: overlay hist shapes (already done in score hists)
    with open(os.path.join(outdir, "ks_overtraining.json"), "w") as f:
        json.dump(ks, f, indent=2)
    return ks


def plot_corr_topk(outdir: str, df_train: pd.DataFrame, features: List[str], topk: int) -> None:
    """
    Plot correlation heatmap for top-k most correlated feature pairs (absolute corr).
    """
    _ensure_dir(outdir)
    x = df_train[features].astype(np.float64)
    corr = x.corr().to_numpy()
    # find top-k off-diagonal pairs
    n = corr.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((abs(corr[i, j]), i, j))
    pairs.sort(reverse=True)
    sel = pairs[: max(5, min(topk, len(pairs)))]
    idx = sorted(set([i for _, i, _ in sel] + [j for _, _, j in sel]))
    feat_sel = [features[i] for i in idx]
    corr_sel = x[feat_sel].corr().to_numpy()

    plt.figure(figsize=(max(6, 0.25 * len(feat_sel)), max(5, 0.25 * len(feat_sel))))
    plt.imshow(corr_sel, vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(feat_sel)), feat_sel, rotation=90, fontsize=7)
    plt.yticks(range(len(feat_sel)), feat_sel, fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "corr_topk.pdf"))
    plt.close()


# --------------------------------------------------------------------------------------
# Early stopping
# --------------------------------------------------------------------------------------
class EarlyStopping:
    def __init__(
        self,
        mode: str,
        patience: int,
        min_delta: float,
        warmup_epochs: int,
    ) -> None:
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs

        self.best: Optional[float] = None
        self.bad = 0

    def improved(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < (self.best - self.min_delta)
        return value > (self.best + self.min_delta)

    def step(self, epoch: int, value: float) -> bool:
        if epoch < self.warmup_epochs:
            return False
        if self.improved(value):
            self.best = value
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience


# --------------------------------------------------------------------------------------
# Train / Eval loops
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    pos_weight_t: Optional[torch.Tensor],
) -> Dict[str, Any]:
    model.eval()
    losses = []
    logits_all = []
    y_all = []
    w_all = []

    for batch in loader:
        # The train loader yields a 4th element (variation inputs) when the
        # adversarial term is on; metrics are always nominal-only.
        xb, yb, wb = batch[0], batch[1], batch[2]
        xb = xb.to(device)
        yb = yb.to(device)
        wb = wb.to(device)

        logits = model(xb)

        weights = wb if cfg.use_weight_in_loss else None
        loss = bce_with_logits_loss(
            logits=logits,
            targets=yb,
            weights=weights,
            pos_weight=pos_weight_t,
            label_smoothing=cfg.label_smoothing,
            normalize_in_batch=cfg.normalize_in_batch,
        )

        losses.append(float(loss.item()))
        logits_all.append(logits.detach().cpu().numpy())
        y_all.append(yb.detach().cpu().numpy())
        w_all.append(wb.detach().cpu().numpy())

    logits_np = np.concatenate(logits_all, axis=0)
    y_np = np.concatenate(y_all, axis=0).astype(np.int64)
    w_np = np.concatenate(w_all, axis=0).astype(np.float64)

    prob_np = sigmoid_np(logits_np)

    # metrics
    out: Dict[str, Any] = {}
    out["loss"] = float(np.mean(losses))
    w_metrics = safe_sample_weight(w_np, mode="abs")  # IMPORTANT: abs weights for ROC/PR

    out["auc_w"] = safe_auc(y_np, prob_np, sample_weight=w_metrics)
    out["auc"] = float(roc_auc_score(y_np, prob_np)) if len(np.unique(y_np)) > 1 else float("nan")

    out["ap_w"] = safe_ap(y_np, prob_np, sample_weight=w_metrics)
    out["ap"] = float(average_precision_score(y_np, prob_np)) if len(np.unique(y_np)) > 1 else float("nan")

    # also store the metric-weights for plots downstream
    out["w_metrics"] = w_metrics

    out["y"] = y_np
    out["p"] = prob_np
    out["w"] = w_np
    return out


def train_one_fold(
    fold_idx: int,
    cfg: TrainConfig,
    data_dir: str,
    out_dir: str,
    trial: "optuna.trial.Trial | None" = None,
    trial_step_offset: int = 0,
    evaluate_train_each_epoch: bool = True,
    run_final_evaluation: bool = True,
    adv: Optional[AdversarialConfig] = None,
    variation_spec: Optional[List[Tuple[str, Dict[str, str]]]] = None,
    init_from: Optional[str] = None,
) -> float:
    fold_dir = Path(out_dir) / f"fold{fold_idx}"
    plots_dir = fold_dir / cfg.plots_subdir
    fold_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[fold %d] start epochs=%d batch_size=%d num_workers=%d device=%s",
        fold_idx,
        cfg.epochs,
        cfg.batch_size,
        cfg.num_workers,
        cfg.device,
    )

    # Load parquet
    train_path = Path(data_dir) / f"data_df_train_{fold_idx}.parquet"
    val_path = Path(data_dir) / f"data_df_validation_{fold_idx}.parquet"
    eval_path = Path(data_dir) / f"data_df_evaluation_{fold_idx}.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing: {val_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing: {eval_path}")

    df_train, df_val, df_eval = _load_fold_data_cached(
        str(train_path),
        str(val_path),
        str(eval_path),
        tuple(cfg.training_features),
        cfg.label_col,
        cfg.weight_col,
    )

    # datasets/loaders
    # Only the TRAIN dataset carries variation inputs: the adversarial term is a
    # training-time regulariser, and val/eval keep their original nominal-only
    # metrics so `evaluate()` and the reported AUC/loss stay comparable to the
    # non-adversarial baseline.
    ds_train = ParquetDataset(
        df_train,
        cfg.training_features,
        cfg.label_col,
        cfg.weight_col,
        cfg.dtype,
        variation_spec=variation_spec if adv is not None else None,
    )
    ds_val = ParquetDataset(df_val, cfg.training_features, cfg.label_col, cfg.weight_col, cfg.dtype)
    ds_eval = ParquetDataset(df_eval, cfg.training_features, cfg.label_col, cfg.weight_col, cfg.dtype)

    if adv is not None:
        logger.info(
            "[fold %d] adversarial ON: lambda=%.6g N_variations=%d detach_nominal=%s "
            "consistency_label_smoothing=%s variation_chunk=%d",
            fold_idx,
            adv.lam,
            ds_train.n_variations,
            adv.detach_nominal,
            adv.consistency_label_smoothing,
            adv.variation_chunk,
        )
        logger.info("[fold %d] variations: %s", fold_idx, ds_train.variation_names)

    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    pin_memory = bool(cfg.pin_memory) and (device.type == "cuda")

    loader_train = make_dataloader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        pin_memory=pin_memory,
    )
    loader_val = make_dataloader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        pin_memory=pin_memory,
    )
    loader_eval = make_dataloader(
        ds_eval,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        pin_memory=pin_memory,
    )

    # model
    model = MLP(
        input_dim=len(cfg.training_features),
        hidden=cfg.hidden,
        activation=cfg.activation,
        dropout=cfg.dropout,
        batch_norm=cfg.batch_norm,
    ).to(device)

    # Warm start: initialise the weights from another run's best checkpoint for this
    # fold. Only the weights are carried over -- the optimizer, the LR schedule and
    # the early-stopping counter are all built fresh below, so the second phase is a
    # genuine "train again until early stopping" rather than a resumed epoch loop.
    # Used to run the adversarial phase from a converged lambda=0 model instead of
    # from random init, which both speeds up the sweep and makes every lambda start
    # from the identical point.
    if init_from is not None:
        init_ckpt = Path(init_from) / f"fold{fold_idx}" / "best.pt"
        if not init_ckpt.exists():
            raise FileNotFoundError(
                f"--init-from: no checkpoint for fold {fold_idx} at {init_ckpt}. "
                "The warm-start run must have completed this fold."
            )
        state = torch.load(init_ckpt, map_location=device)["model_state"]
        model.load_state_dict(state)
        logger.info("[fold %d] warm start from %s", fold_idx, init_ckpt)

    # optimizer
    opt_name = cfg.optimizer_name.lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
            eps=cfg.eps,
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
            eps=cfg.eps,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer_name}")

    # scheduler
    scheduler = None
    if cfg.scheduler_name.lower() == "reduce_on_plateau":
        scheduler_mode = cfg.rop_mode
        if cfg.es_monitor == "val_auc_weighted":
            scheduler_mode = "max"
        elif cfg.es_monitor == "val_loss":
            scheduler_mode = "min"

        if scheduler_mode != cfg.rop_mode:
            logger.warning(
                "[fold %d] Overriding ReduceLROnPlateau mode from %s to %s to match monitor=%s",
                fold_idx,
                cfg.rop_mode,
                scheduler_mode,
                cfg.es_monitor,
            )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=cfg.rop_factor,
            patience=cfg.rop_patience,
            min_lr=cfg.rop_min_lr,
            threshold=cfg.rop_threshold,
        )

    # pos_weight for BCE (optional)
    pos_weight_t = None
    if cfg.pos_weight is not None:
        pos_weight_t = torch.tensor([cfg.pos_weight], dtype=torch.float32, device=device)

    scaler = torch.amp.GradScaler(
        "cuda", enabled=(cfg.amp_enable and device.type == "cuda")
    )

    # early stopping
    early = EarlyStopping(
        mode=cfg.es_mode,
        patience=cfg.es_patience,
        min_delta=cfg.es_min_delta,
        warmup_epochs=cfg.es_warmup_epochs,
    ) if cfg.es_enable else None

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_auc_w": [],
        "val_auc_w": [],
        "lr": [],
    }

    best_metric = None
    best_epoch = None
    best_path = fold_dir / "best.pt"
    last_path = fold_dir / "last.pt"
    best_ts_path = fold_dir / "best_torchscript.pt"
    last_ts_path = fold_dir / "last_torchscript.pt"

    # train loop
    for epoch in range(cfg.epochs):
        model.train()
        running = []
        running_nominal = []
        running_penalty = []

        for batch in loader_train:
            xb, yb, wb = batch[0], batch[1], batch[2]
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(cfg.amp_enable and device.type == "cuda")):
                logits = model(xb)
                loss = training_loop_loss(
                    logits=logits,
                    targets=yb,
                    wb=wb,
                    pos_weight=pos_weight_t,
                    label_smoothing=cfg.label_smoothing,
                    normalize_in_batch=cfg.normalize_in_batch,
                )

            if adv is None:
                scaler.scale(loss).backward()
            else:
                # Term 1: the original loss, backwarded exactly as before.
                scaler.scale(loss).backward(retain_graph=False)
                running_nominal.append(float(loss.item()))

                xvb = batch[3].to(device)  # [B, V, F]
                n_var = xvb.shape[1]
                penalty_value = 0.0

                # The adversarial forwards run with the model in eval() mode on
                # purpose:
                #   * dropout must not fire, or it would consume RNG and the
                #     lambda=0 run could no longer reproduce the switch-off run;
                #   * BatchNorm must not update its running statistics from
                #     variation batches, or the exported (eval-mode) model that
                #     Stage-2 scores with would drift even at lambda=0.
                # Gradients still flow: eval() only changes dropout/BN behaviour.
                was_training = model.training
                model.eval()
                try:
                    with torch.amp.autocast(
                        "cuda", enabled=(cfg.amp_enable and device.type == "cuda")
                    ):
                        if adv.detach_nominal:
                            with torch.no_grad():
                                p_nominal = torch.sigmoid(model(xb))
                        else:
                            p_nominal = torch.sigmoid(model(xb))

                        chunks = _variation_chunks(n_var, adv.variation_chunk)
                        for ci, (lo, hi) in enumerate(chunks):
                            sub = xvb[:, lo:hi, :]
                            b, v, f = sub.shape
                            logits_var = model(sub.reshape(b * v, f)).reshape(b, v)
                            penalty = adversarial_penalty(
                                logits_var=logits_var,
                                p_nominal=p_nominal,
                                yb=yb,
                                wb=wb,
                                pos_weight=pos_weight_t,
                                cfg=cfg,
                                adv=adv,
                            )
                            penalty_value += float(penalty.item())
                            term = adv.lam * penalty
                            # Chunks accumulate into .grad; the un-detached
                            # nominal graph is shared, so keep it until the last.
                            keep = (not adv.detach_nominal) and (ci < len(chunks) - 1)
                            scaler.scale(term).backward(retain_graph=keep)
                finally:
                    if was_training:
                        model.train()

                running_penalty.append(penalty_value)
                loss = loss.detach() + adv.lam * penalty_value

            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            running.append(float(loss.item()))

        # During HPO, skip the full training-set evaluation to reduce epoch cost.
        if evaluate_train_each_epoch:
            train_eval = evaluate(model, loader_train, device, cfg, pos_weight_t)
            train_loss = float(train_eval["loss"])
            train_auc_w = float(train_eval["auc_w"])
        else:
            train_eval = None
            train_loss = float(np.mean(running)) if running else float("nan")
            train_auc_w = float("nan")
        val_eval = evaluate(model, loader_val, device, cfg, pos_weight_t)

        val_loss = float(val_eval["loss"])
        val_auc_w = float(val_eval["auc_w"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc_w"].append(train_auc_w)
        history["val_auc_w"].append(val_auc_w)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if adv is not None:
            # Batch-mean split of the epoch's training objective, so the reviewer
            # can see whether the consistency penalty is actually coming down
            # rather than the total just tracking the nominal term.
            history.setdefault("train_loss_nominal", []).append(
                float(np.mean(running_nominal)) if running_nominal else float("nan")
            )
            history.setdefault("train_adv_penalty", []).append(
                float(np.mean(running_penalty)) if running_penalty else float("nan")
            )
            logger.info(
                "[fold %d] epoch %d  adv: nominal_loss=%.6f penalty=%.6f "
                "lambda*penalty=%.6f",
                fold_idx,
                epoch,
                history["train_loss_nominal"][-1],
                history["train_adv_penalty"][-1],
                adv.lam * history["train_adv_penalty"][-1],
            )

        if evaluate_train_each_epoch:
            logger.info(
                "[fold %d] epoch %d/%d  train_loss=%.6f val_loss=%.6f  train_auc_w=%.4f val_auc_w=%.4f lr=%.3e",
                fold_idx,
                epoch,
                cfg.epochs - 1,
                train_loss,
                val_loss,
                train_auc_w,
                val_auc_w,
                history["lr"][-1],
            )
        else:
            logger.info(
                "[fold %d] epoch %d/%d  train_loss(batch)=%.6f val_loss=%.6f  val_auc_w=%.4f lr=%.3e",
                fold_idx,
                epoch,
                cfg.epochs - 1,
                train_loss,
                val_loss,
                val_auc_w,
                history["lr"][-1],
            )

        # choose monitor value
        if cfg.es_monitor == "val_loss":
            monitor_val = val_loss
        elif cfg.es_monitor == "val_auc_weighted":
            monitor_val = val_auc_w
        else:
            monitor_val = val_loss

        # scheduler step
        if scheduler is not None:
            scheduler.step(monitor_val)

        # ---------------------------
        # Optuna report + prune (NEW)
        # ---------------------------
        if trial is not None:
            # Optuna maximizes by default if we set direction="maximize" in study
            # So we report a "score" where larger is better.
            if cfg.es_monitor == "val_loss":
                score = -float(monitor_val)  # smaller loss => larger score
            else:
                score = float(monitor_val)  # auc => larger better

            # Make step unique across folds to avoid "already reported step warning"
            step = trial_step_offset + epoch
            trial.report(score, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned at fold = {fold_idx}, epoch = {epoch}, value = {score:.6f}")

        # save best
        improved = False
        if best_metric is None:
            improved = True
        else:
            if cfg.es_mode == "min":
                improved = monitor_val < best_metric
            else:
                improved = monitor_val > best_metric

        if improved:
            best_metric = monitor_val
            best_epoch = epoch
            if cfg.save_best:
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "cfg": cfg.__dict__,
                        "best_metric": best_metric,
                        "best_epoch": best_epoch,
                    },
                    best_path,
                )

            if cfg.save_torchscript:
                # Export only when requested; HPO does not consume these files.
                export_torchscript(model, len(cfg.training_features), best_ts_path)

        # early stopping
        if early is not None:
            stop = early.step(epoch, monitor_val)
            if stop:
                logger.warning("[fold %d] Early stopping at epoch=%d", fold_idx, epoch)
                break

    # save last
    if cfg.save_last:
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": len(history["train_loss"]) - 1,
                "cfg": cfg.__dict__,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
            },
            last_path,
        )
        if cfg.save_torchscript:
            # Export only when requested; HPO does not consume these files.
            export_torchscript(model, len(cfg.training_features), last_ts_path)

    # load best for final evaluation if requested
    if cfg.es_enable and cfg.es_restore_best and cfg.save_best and best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    if not run_final_evaluation:
        return float(best_metric) if best_metric is not None else float("nan")

    # final eval (train/val/eval)
    train_final = evaluate(model, loader_train, device, cfg, pos_weight_t)
    val_final = evaluate(model, loader_val, device, cfg, pos_weight_t)
    eval_final = evaluate(model, loader_eval, device, cfg, pos_weight_t)

    # save history
    if cfg.save_history:
        with open(fold_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    # plots
    if cfg.plots_enable:
        plot_learning_curves(str(plots_dir), history)

        plot_roc(str(plots_dir), val_final["y"], val_final["p"], val_final["w_metrics"])
        plot_pr(str(plots_dir), val_final["y"], val_final["p"], val_final["w_metrics"])
        plot_score_hists(str(plots_dir), cfg, train_final, val_final)
        plot_calibration(str(plots_dir), val_final["y"], val_final["p"], val_final["w_metrics"], n_bins=cfg.calib_bins)

        ks = plot_ks_overtraining(str(plots_dir), train_final, val_final)
        plot_corr_topk(str(plots_dir), df_train, cfg.training_features, topk=cfg.corr_topk)

        # Save a quick text summary of metrics
        metrics_summary = {
            "train": {k: float(train_final[k]) for k in ["loss", "auc_w", "auc", "ap_w", "ap"]},
            "val": {k: float(val_final[k]) for k in ["loss", "auc_w", "auc", "ap_w", "ap"]},
            "eval": {k: float(eval_final[k]) for k in ["loss", "auc_w", "auc", "ap_w", "ap"]},
            "ks_overtraining": ks,
            "best_monitor_value": float(best_metric) if best_metric is not None else None,
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
        }
        with open(fold_dir / "metrics_summary.json", "w") as f:
            json.dump(metrics_summary, f, indent=2)

    # optionally save predictions
    if cfg.save_predictions:
        pred_df = pd.DataFrame(
            {
                "y_val": val_final["y"],
                "p_val": val_final["p"],
                "w_val": val_final["w_metrics"],
            }
        )
        pred_df.to_parquet(fold_dir / "val_predictions.parquet", index=False)

    # ---------------------------
    # Return fold objective (NEW)
    # ---------------------------
    return float(best_metric) if best_metric is not None else float("nan")


def train_all_folds(
    cfg: TrainConfig,
    data_dir: str,
    out_dir: str,
    folds: Optional[List[int]],
    adv: Optional[AdversarialConfig] = None,
    variation_spec: Optional[List[Tuple[str, Dict[str, str]]]] = None,
    init_from: Optional[str] = None,
) -> None:
    if folds is None:
        folds = list(range(cfg.n_folds))

    for i in folds:
        logger.info("============================================================")
        logger.info("Training fold %d/%d", i, cfg.n_folds - 1)
        train_one_fold(
            i,
            cfg,
            data_dir=data_dir,
            out_dir=out_dir,
            adv=adv,
            variation_spec=variation_spec,
            init_from=init_from,
        )


# --------------------------------------------------------------------------------------
# Oputuna best hyperparameter integration
# --------------------------------------------------------------------------------------
def _reconstruct_hidden_from_optuna_params(params: Dict[str, Any]) -> List[int]:
    """Rebuild hidden list from (n_layers, first_width, shrink_factor)."""
    n_layers = int(params["n_layers"])
    first_width = int(params["first_width"])
    shrink = float(params["shrink_factor"])

    hidden: List[int] = []
    w = float(first_width)
    for _ in range(n_layers):
        hidden.append(int(round(w)))
        w = max(16.0, w * shrink)
    return hidden


def _build_dropout_from_optuna_params(
    params: Dict[str, Any], n_layers: int
) -> List[float]:
    """Rebuild dropout list for either drop_mode=one or per_layer."""
    # drop_mode is present in your study params, but we can infer from keys too.
    if "dropout" in params:
        p = float(params["dropout"])
        return [p] * n_layers

    # per-layer: dropout_l0, dropout_l1, ...
    out = []
    for i in range(n_layers):
        out.append(float(params.get(f"dropout_l{i}", 0.0)))
    return out


def load_optuna_best_json(path: str) -> Dict[str, Any]:
    """
    Reads optuna_best.json created by hpo_optuna.py:
      {
        "best_value": ...,
        "best_params": {...},
        "n_trials": ...
      }
    Returns dict with keys: best_value, best_params, n_trials.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"optuna_best.json not found: {path}")
    with open(p, "r") as f:
        d = json.load(f)
    if "best_params" not in d:
        raise KeyError(f"Missing 'best_params' in {path}")
    return d


def apply_optuna_best(
    cfg: TrainConfig, optuna_blob: Dict[str, Any]
) -> Tuple[TrainConfig, Dict[str, Any]]:
    """
    Override TrainConfig using Optuna best params.
    Returns (new_cfg, applied_params_summary).
    """
    params = optuna_blob["best_params"]

    # hidden (prefer exact list if present in user_attrs-export, else reconstruct)
    # Your current optuna_best.json probably does NOT contain user_attrs,
    # so reconstruction is the default.
    hidden = params.get("hidden", None)
    if hidden is None:
        hidden = _reconstruct_hidden_from_optuna_params(params)
    hidden = [int(x) for x in hidden]
    dropout = _build_dropout_from_optuna_params(params, n_layers=len(hidden))

    # basic overrides (only if present)
    new_cfg = replace(
        cfg,
        hidden=hidden,
        dropout=[float(x) for x in dropout],
        activation=str(params.get("activation", cfg.activation)),
        batch_norm=bool(params.get("batch_norm", cfg.batch_norm)),
        lr=float(params.get("lr", cfg.lr)),
        weight_decay=float(params.get("weight_decay", cfg.weight_decay)),
        label_smoothing=float(params.get("label_smoothing", cfg.label_smoothing)),
        grad_clip_norm=float(params.get("grad_clip_norm", cfg.grad_clip_norm)),
        batch_size=int(params.get("batch_size", cfg.batch_size)),
    )

    applied = {
        "hidden": hidden,
        "dropout": dropout,
        "activation": new_cfg.activation,
        "batch_norm": new_cfg.batch_norm,
        "lr": new_cfg.lr,
        "weight_decay": new_cfg.weight_decay,
        "label_smoothing": new_cfg.label_smoothing,
        "grad_clip_norm": new_cfg.grad_clip_norm,
        "batch_size": new_cfg.batch_size,
        "best_value": optuna_blob.get("best_value", None),
        "n_trials": optuna_blob.get("n_trials", None),
    }
    return new_cfg, applied


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train DNN from fold-parquet files (4-fold CV).")
    p.add_argument("--config", required=True, help="YAML config (model/training/features)")
    p.add_argument("--data-dir", required=True, help="Directory with data_df_train_i.parquet etc.")
    p.add_argument("--out-dir", required=True, help="Output directory for trained_models/<tag>/<year_region_cat> (or any).")
    p.add_argument("--folds", default=None, help="Comma list of folds to train (e.g. 0,1,2,3). Default: all.")
    p.add_argument("--optuna-best-json",default=None,help="Path to optuna_best.json (will override config hyperparameters).")
    p.add_argument("--log-level", default="INFO", help="Logging level (INFO/DEBUG/...)")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override the training seed (meta.seed / the optuna trial seed). Used to "
            "measure the run-to-run noise band. Default: unchanged behaviour."
        ),
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Override training.epochs from the config. Mainly for short probe runs "
            "(e.g. one epoch to measure the adversarial penalty against the nominal "
            "loss for lambda calibration). Default: unchanged."
        ),
    )
    p.add_argument(
        "--init-from",
        default=None,
        help=(
            "Warm start: directory of a previous run whose <dir>/fold<i>/best.pt is "
            "loaded as the initial weights for fold i. The optimizer, LR schedule "
            "and early-stopping counter are still built fresh, so training runs to "
            "early stopping again. Used to train the adversarial phase from a "
            "converged lambda=0 model. Default: random initialisation, as before."
        ),
    )

    adv = p.add_argument_group("systematics-adversarial loss")
    adv.add_argument(
        "--use_adversarial",
        action="store_true",
        help=(
            "Add the systematics-consistency term to the loss. Requires --data-dir to "
            "point at fold parquets produced with "
            "preprocess_dnn.py --include-systematic-variations. When this flag is "
            "absent the training path is unchanged."
        ),
    )
    adv.add_argument(
        "--adversarial-lambda",
        type=float,
        default=0.0,
        help="Weight of the adversarial sum (default 0.0).",
    )
    adv.add_argument(
        "--adversarial-detach-nominal",
        action="store_true",
        help=(
            "Stop-gradient on the nominal prediction used as the consistency target, "
            "so only the variation branch is penalised. Default: not detached "
            "(both branches are penalised)."
        ),
    )
    adv.add_argument(
        "--adversarial-consistency-smoothing",
        choices=["inherit", "none"],
        default="inherit",
        help=(
            "Label smoothing inside original_loss(pred_i, pred_nominal). 'inherit' "
            "(default) reuses cfg.label_smoothing, i.e. original_loss literally; "
            "'none' disables it, since the target is already a soft probability and "
            "smoothing it biases variation scores toward 0.5."
        ),
    )
    adv.add_argument(
        "--adversarial-label-only",
        action="store_true",
        help=(
            "STEP 2 of the three-step schedule: use "
            "lambda * sum_i original_loss(pred_i, label) only -- the variations are "
            "trained against the truth label and the consistency term is absent."
        ),
    )
    adv.add_argument(
        "--adversarial-mask-label-term",
        action="store_true",
        help=(
            "Apply --adversarial-high-score-cut to the original_loss(pred_i, label) "
            "term as well, not just to the consistency term. The three-step schedule "
            "puts [score_cut] on every term inside the lambda sum; the earlier "
            "variant-B formula masked the consistency term alone. No effect without "
            "a cut."
        ),
    )
    adv.add_argument(
        "--adversarial-high-score-cut",
        type=float,
        default=None,
        help=(
            "VARIANT B: restrict the consistency term to events with "
            "arctanh(p_nominal) > CUT (e.g. 2.0, i.e. p_nominal > 0.9640). The "
            "significance comes from the top score bins, so decorrelating across the "
            "bulk spends discrimination where it cannot pay. Changes the effective "
            "lambda scale, so re-scan lambda when enabling it."
        ),
    )
    adv.add_argument(
        "--adversarial-variation-chunk",
        type=int,
        default=0,
        help=(
            "Variations per forward/backward chunk (gradient accumulation). "
            "<=0 (default) processes all variations in one chunk; lower it if the "
            "full-variation run runs out of device memory."
        ),
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    # --log-level was declared but never applied, so the shared logger stayed at
    # its WARNING default and the per-epoch lines (including the adversarial
    # nominal_loss/penalty report) never reached the log. Mirrors what
    # preprocess_dnn.py already does with the same flag.
    logger.setLevel(args.log_level)

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    optuna_info = None
    optuna_applied = None
    active_seed = cfg.seed

    if args.optuna_best_json is not None:
        optuna_info = load_optuna_best_json(args.optuna_best_json)
        cfg, optuna_applied = apply_optuna_best(cfg, optuna_info)
        trial_seed = optuna_info.get("trial_seed", None)
        if trial_seed is not None:
            active_seed = int(trial_seed)
            set_seed(active_seed)

        logger.info("[optuna] Loaded best params from: %s", args.optuna_best_json)
        logger.info("[optuna] Applied: hidden=%s activation=%s batch_norm=%s dropout=%s",
                    optuna_applied["hidden"], optuna_applied["activation"],
                    optuna_applied["batch_norm"], optuna_applied["dropout"])
        logger.info("[optuna] Applied: lr=%.3e wd=%.3e bs=%d label_smoothing=%.4f grad_clip=%.3f",
                    optuna_applied["lr"], optuna_applied["weight_decay"],
                    optuna_applied["batch_size"], optuna_applied["label_smoothing"],
                    optuna_applied["grad_clip_norm"])
        if trial_seed is not None:
            logger.info("[optuna] Reusing winning trial seed: %d", active_seed)

    # --seed is applied last so it wins over both the config and the optuna
    # trial seed; without it, seeding is exactly as before.
    if args.seed is not None:
        active_seed = int(args.seed)
        set_seed(active_seed)
        logger.info("[seed] Overriding training seed from --seed: %d", active_seed)

    # Applied after the optuna overrides so a probe run's epoch budget is not
    # silently replaced by the tuned value.
    if args.epochs is not None:
        cfg = replace(cfg, epochs=int(args.epochs))
        logger.info("[epochs] Overriding training epochs from --epochs: %d", cfg.epochs)

    adv = None
    variation_spec = None
    if args.use_adversarial:
        variation_spec = load_variation_spec(args.data_dir)
        adv = AdversarialConfig(
            lam=float(args.adversarial_lambda),
            detach_nominal=bool(args.adversarial_detach_nominal),
            consistency_label_smoothing=str(args.adversarial_consistency_smoothing),
            variation_chunk=int(args.adversarial_variation_chunk),
            high_score_cut=(
                float(args.adversarial_high_score_cut)
                if args.adversarial_high_score_cut is not None
                else None
            ),
            label_only=bool(args.adversarial_label_only),
            mask_label_term=bool(args.adversarial_mask_label_term),
        )
        logger.info(
            "[adversarial] lambda=%.6g N_variations=%d detach_nominal=%s "
            "consistency_smoothing=%s chunk=%d "
            "label_only=%s high_score_cut=%s mask_label_term=%s",
            adv.lam,
            len(variation_spec),
            adv.detach_nominal,
            adv.consistency_label_smoothing,
            adv.variation_chunk,
            adv.label_only,
            adv.high_score_cut,
            adv.mask_label_term,
        )
        logger.info("[adversarial] variations: %s", [v for v, _ in variation_spec])

    folds = None
    if args.folds is not None:
        folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]

    # quick env print
    logger.info("Torch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
    logger.info("Active training seed: %d", active_seed)

    train_all_folds(
        cfg,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        folds=folds,
        adv=adv,
        variation_spec=variation_spec,
        init_from=args.init_from,
    )

    # Save manifest for reproducibility
    manifest = {
        # -------------------------
        # Run identity
        # -------------------------
        "run": {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "command": " ".join(sys.argv),
        },

        # -------------------------
        # Configuration
        # -------------------------
        "config": cfg.__dict__,
        "features": {
            "training_features": cfg.training_features,
            "label_col": cfg.label_col,
            "weight_col": cfg.weight_col,
        },
        "cv": {
            "n_folds": cfg.n_folds,
            "folds_run": folds if folds is not None else "all",
        },
        "optuna": {
            "best_json": str(args.optuna_best_json) if args.optuna_best_json else None,
            "best_value": (optuna_info.get("best_value") if optuna_info else None),
            "n_trials": (optuna_info.get("n_trials") if optuna_info else None),
            "trial_seed": (optuna_info.get("trial_seed") if optuna_info else None),
            "applied_params": optuna_applied,
        },

        # -------------------------
        # Data
        # -------------------------
        "data": {
            "data_dir": str(args.data_dir),
            # "years": cfg.years,
            # "regions": cfg.region,
            # "categories": cfg.category,
            # "combined_training": len(cfg.years) > 1,
        },
        # -------------------------
        # Training behavior (critical!)
        # -------------------------
        "adversarial": {
            "enabled": bool(args.use_adversarial),
            "lambda": float(args.adversarial_lambda) if args.use_adversarial else None,
            "n_variations": len(variation_spec) if variation_spec else 0,
            "variations": [v for v, _ in variation_spec] if variation_spec else [],
            "detach_nominal": bool(args.adversarial_detach_nominal),
            "consistency_label_smoothing": str(args.adversarial_consistency_smoothing),
            "variation_chunk": int(args.adversarial_variation_chunk),
            "label_only": bool(args.adversarial_label_only),
            "mask_label_term": bool(args.adversarial_mask_label_term),
            "high_score_cut": args.adversarial_high_score_cut,
            "high_score_cut_p_threshold": (
                math.tanh(args.adversarial_high_score_cut)
                if args.adversarial_high_score_cut is not None
                else None
            ),
            "seed_override": (int(args.seed) if args.seed is not None else None),
            "active_seed": int(active_seed),
        },

        "warm_start": {
            "init_from": args.init_from,
        },

        "training_behavior": {
            "loss_name": cfg.loss_name,
            "use_weight_in_loss": cfg.use_weight_in_loss,
            "normalize_weights_in_batch": cfg.normalize_in_batch,
            "pos_weight": cfg.pos_weight,
            "label_smoothing": cfg.label_smoothing,
            "grad_clip_norm": cfg.grad_clip_norm,
            "amp_enabled": cfg.amp_enable,
            "optimizer": cfg.optimizer_name,
            "scheduler": cfg.scheduler_name,
            "early_stopping": {
                "enabled": cfg.es_enable,
                "monitor": cfg.es_monitor,
                "mode": cfg.es_mode,
                "patience": cfg.es_patience,
                "min_delta": cfg.es_min_delta,
            },
        },

        "pipeline": {
            "preprocessor_version": "v1.0",
            "git_commit": get_git_commit(),
            "git_dirty": get_git_state(args.out_dir),
        },
    }

    with open(Path(args.out_dir) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Done. Success.")


if __name__ == "__main__":
    main()
