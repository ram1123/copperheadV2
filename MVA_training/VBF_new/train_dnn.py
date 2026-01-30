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
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset

import optuna

from modules.git_utils import get_git_commit, get_git_state
from modules.utils import logger


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
    )


# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # determinism tradeoff (slower). For training stability debugging, you can enable:
    # torch.use_deterministic_algorithms(True)


# --------------------------------------------------------------------------------------
# Dataset / Dataloader
# --------------------------------------------------------------------------------------
class ParquetDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        label_col: str,
        weight_col: str,
        dtype: str,
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

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.x[idx]),
            torch.from_numpy(np.array(self.y[idx])),
            torch.from_numpy(np.array(self.w[idx])),
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

    w = weights
    if normalize_in_batch:
        # keep average weight ~1 to stabilize scale
        denom = torch.mean(torch.abs(w)) + 1e-12
        w = w / denom

    return torch.sum(per * w) / (torch.sum(torch.abs(w)) + 1e-12)


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

    for xb, yb, wb in loader:
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
) -> float:
    fold_dir = Path(out_dir) / f"fold{fold_idx}"
    plots_dir = fold_dir / cfg.plots_subdir
    fold_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

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

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_eval = pd.read_parquet(eval_path)

    # sanity: required columns
    for c in cfg.training_features + [cfg.label_col, cfg.weight_col]:
        if c not in df_train.columns:
            raise KeyError(f"Missing column '{c}' in {train_path}")

    # ensure numeric
    df_train[cfg.training_features] = df_train[cfg.training_features].apply(pd.to_numeric, errors="coerce")
    df_val[cfg.training_features] = df_val[cfg.training_features].apply(pd.to_numeric, errors="coerce")
    df_eval[cfg.training_features] = df_eval[cfg.training_features].apply(pd.to_numeric, errors="coerce")

    # drop any rows with NaN in features (shouldn't happen if preprocessing is correct)
    df_train = df_train.dropna(subset=cfg.training_features)
    df_val = df_val.dropna(subset=cfg.training_features)
    df_eval = df_eval.dropna(subset=cfg.training_features)

    # datasets/loaders
    ds_train = ParquetDataset(df_train, cfg.training_features, cfg.label_col, cfg.weight_col, cfg.dtype)
    ds_val = ParquetDataset(df_val, cfg.training_features, cfg.label_col, cfg.weight_col, cfg.dtype)
    ds_eval = ParquetDataset(df_eval, cfg.training_features, cfg.label_col, cfg.weight_col, cfg.dtype)

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.rop_mode,
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
    best_path = fold_dir / "best.pt"
    last_path = fold_dir / "last.pt"

    # train loop
    for epoch in range(cfg.epochs):
        model.train()
        running = []

        for xb, yb, wb in loader_train:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(cfg.amp_enable and device.type == "cuda")):
                logits = model(xb)
                weights = wb if cfg.use_weight_in_loss else None
                loss_raw = bce_with_logits_loss(
                    logits=logits,
                    targets=yb,
                    weights=None,
                    pos_weight=pos_weight_t,
                    label_smoothing=cfg.label_smoothing,
                    normalize_in_batch=False,
                )

                loss_w = make_loss_weights(
                    w=wb,
                    normalize_in_batch=cfg.normalize_in_batch,
                    clip_abs_max=None,
                )
                loss = torch.sum(loss_raw * loss_w) / (torch.sum(torch.abs(loss_w)) + 1e-12)


            scaler.scale(loss).backward()

            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            running.append(float(loss.item()))

        # evaluate
        train_eval = evaluate(model, loader_train, device, cfg, pos_weight_t)
        val_eval = evaluate(model, loader_val, device, cfg, pos_weight_t)

        train_loss = float(train_eval["loss"])
        val_loss = float(val_eval["loss"])
        train_auc_w = float(train_eval["auc_w"])
        val_auc_w = float(val_eval["auc_w"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc_w"].append(train_auc_w)
        history["val_auc_w"].append(val_auc_w)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        logger.info(
            "[fold %d] epoch %d/%d  train_loss=%.6f val_loss=%.6f  train_auc_w=%.4f val_auc_w=%.4f lr=%.3e",
            fold_idx, epoch, cfg.epochs - 1, train_loss, val_loss, train_auc_w, val_auc_w, history["lr"][-1]
        )

        # scheduler step
        if scheduler is not None:
            if cfg.es_monitor == "val_loss":
                scheduler.step(val_loss)
            else:
                # for AUC monitor, you might want mode="max" and step(-auc) etc.
                scheduler.step(val_loss)

        # choose monitor value
        if cfg.es_monitor == "val_loss":
            monitor_val = val_loss
        elif cfg.es_monitor == "val_auc_weighted":
            monitor_val = val_auc_w
        else:
            monitor_val = val_loss

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
            if cfg.save_best:
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "cfg": cfg.__dict__,
                        "best_metric": best_metric,
                    },
                    best_path,
                )

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
            },
            last_path,
        )

    # load best for final evaluation if requested
    if cfg.es_enable and cfg.es_restore_best and cfg.save_best and best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

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


def train_all_folds(cfg: TrainConfig, data_dir: str, out_dir: str, folds: Optional[List[int]]) -> None:
    if folds is None:
        folds = list(range(cfg.n_folds))

    for i in folds:
        logger.info("============================================================")
        logger.info("Training fold %d/%d", i, cfg.n_folds - 1)
        train_one_fold(i, cfg, data_dir=data_dir, out_dir=out_dir)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train DNN from fold-parquet files (4-fold CV).")
    p.add_argument("--config", required=True, help="YAML config (model/training/features)")
    p.add_argument("--data-dir", required=True, help="Directory with data_df_train_i.parquet etc.")
    p.add_argument("--out-dir", required=True, help="Output directory for trained_models/<tag>/<year_region_cat> (or any).")
    p.add_argument("--folds", default=None, help="Comma list of folds to train (e.g. 0,1,2,3). Default: all.")
    p.add_argument("--log-level", default="INFO", help="Logging level (INFO/DEBUG/...)")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    folds = None
    if args.folds is not None:
        folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]

    # quick env print
    logger.info("Torch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))

    train_all_folds(cfg, data_dir=args.data_dir, out_dir=args.out_dir, folds=folds)

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

        # -------------------------
        # Data
        # -------------------------
        "data": {
            "data_dir": str(args.data_dir),
        },

        # -------------------------
        # Training behavior (critical!)
        # -------------------------
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
