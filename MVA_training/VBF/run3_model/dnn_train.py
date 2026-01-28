import json
import multiprocessing as mp
import os
import pickle
import threading
from pathlib import Path
from time import time as _time
import yaml

import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
import plotly.io as pio

# Put this at the VERY TOP, before importing torch, numpy, etc.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.service.utils.report_utils import get_standard_plots
from ax.storage.json_store.save import save_experiment
from ax.utils.notebook.plotting import render
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

mp.set_start_method('spawn', force=True)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# #### END: Libraries for scan HYPERPARAMETERS


plt.style.use(hep.style.CMS)
import concurrent

# torch.multiprocessing.set_sharing_strategy('file_descriptor') # reason: https://discuss.pytorch.org/t/training-crashes-due-to-insufficient-shared-memory-shm-nn-dataparallel/26396/44
import logging

from cli.common_argparser import build_common_parser

from dnn_helper import *
from modules import selection
from modules.utils import logger

from MVA_training.VBF.dnn_plotting import (
    _roc_weighted,
    cv_consistency_plots_ROOT,
    partial_dependence_curve,
    permutation_importance_auc,
    plot_auc_and_loss,
    plot_calibration_ROOT,
    plot_loss_curves,
    plot_lr,
    plot_overtraining_KS_ROOT,
    plot_pdp_ROOT,
    plot_perm_importance_ROOT,
    plot_score_feature_corr_ROOT_bar,
    plot_score_feature_corr_ROOT_heatmap,
    plot_significance,
    plot_threshold_scan_ROOT,
    plot_weight_distribution_ROOT,
    plotConfusionMatrix,
    plotPrecisionRecall,
    safe_weighted_auc,
    yield_table_after_cut,
    _wcov, _wstd
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Using device: {DEVICE}")
logger.info(f"using workers: {NWORKERS}")

def _safe_auc(y, p, w=None):
    try:
        return safe_weighted_auc(y, p, sample_weight=w)
    except ValueError:
        return float("nan")

def safe_arctanh(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -1.0 + eps, 1.0 - eps)
    return np.arctanh(x)

def transformDnnScore(dnn_scores):
    s = safe_arctanh(dnn_scores) # protection from atanh(0) or atanh(1 or -1) whose value is +/- inf
    return s


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        pt = torch.exp(-BCE_loss)  # Probabilities of correct classification
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()


class HingeLoss(nn.Module):
    """
    source: chatgpt, but verified on https://lightning.ai/docs/torchmetrics/stable/classification/hinge_loss.html
    """

    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, outputs, targets):
        # Map targets {0, 1} -> {-1, 1}
        targets = 2 * targets - 1  # Convert 0 -> -1, 1 -> 1
        # Calculate hinge loss
        loss = torch.mean(torch.clamp(1 - outputs * targets, min=0))
        return loss


class TrainingLogger:
    # Reference: https://www.geeksforgeeks.org/deep-learning/monitoring-model-training-in-pytorch-with-callbacks-and-logging/
    def __init__(self, log_interval: int = 10):
        self.log_interval = int(log_interval)
        self.training_logs = []
        self.epoch_start_time = None

    def on_epoch_begin(self, epoch: int):
        self.epoch_start_time = _time()
        logger.debug("Epoch %d starting.", epoch + 1)

    def on_epoch_end(self, epoch: int, logs=None):
        if self.epoch_start_time is None:
            self.epoch_start_time = _time()

        elapsed_time = _time() - self.epoch_start_time
        logger.info("Epoch %d finished in %.2f seconds.", epoch + 1, elapsed_time)

        if logs is None:
            logs = {}
        logs["epoch_time"] = float(elapsed_time)

        # store a copy to avoid accidental mutation by caller
        self.training_logs.append(dict(logs))

    def on_batch_end(self, batch: int, logs=None):
        if (batch + 1) % self.log_interval != 0:
            return

        logs = logs or {}
        loss = logs.get("loss", None)
        acc = logs.get("accuracy", None)

        if loss is None and acc is None:
            logger.info("Batch %d", batch + 1)
        elif loss is None:
            logger.info("Batch %d: Accuracy = %.4f", batch + 1, acc)
        elif acc is None:
            logger.info("Batch %d: Loss = %.4f", batch + 1, loss)
        else:
            logger.info("Batch %d: Loss = %.4f, Accuracy = %.4f", batch + 1, loss, acc)

    def save_logs(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.training_logs, f)
        logger.info("Training logs saved to %s", filepath)


class EarlyStopping:
    def __init__(
        self,
        patience=10,
        delta=0.0,
        mode="min",
        nfold=0,
        fold_save_path=None,
        model=None,
        training_features=None,
        verbose=True,
        trace_jit=True,
        jit_batch=100,
    ):
        self.patience = int(patience)
        self.delta = float(delta)
        self.mode = mode
        self.nfold = int(nfold)
        self.fold_save_path = fold_save_path
        self.model = model
        self.training_features = training_features or []
        self.verbose = bool(verbose)
        self.trace_jit = bool(trace_jit)
        self.jit_batch = int(jit_batch)

        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if self.mode not in ("min", "max"):
            raise ValueError("mode should be 'min' or 'max'")

        if self.mode == "min":
            self.monitor_op = lambda curr, best: curr < best - self.delta
        else:
            self.monitor_op = lambda curr, best: curr > best + self.delta

    def on_epoch_end(
        self, epoch, current_score, *, optimizer=None, scheduler=None, scaler=None
    ):
        improved = self.best_score is None or self.monitor_op(
            current_score, self.best_score
        )

        if improved:
            if self.verbose:
                logger.info(
                    "[EarlyStopping] Fold %d, Epoch %d: best %s improved from %s to %s",
                    self.nfold,
                    epoch,
                    self.mode,
                    str(self.best_score),
                    str(current_score),
                )
            self.best_score = float(current_score)
            self.counter = 0

            if self.model is not None and self.fold_save_path is not None:
                save_model_artifacts(
                    model=self.model,
                    fold_save_path=self.fold_save_path,
                    model_name="best_model",
                    training_features=self.training_features,
                    trace_jit=self.trace_jit,
                    jit_batch=self.jit_batch,
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_score=self.best_score,
                    logger=logger,
                )
            return False

        self.counter += 1
        if self.verbose:
            logger.info(
                "[EarlyStopping] Fold %d, Epoch %d: no improvement (%d/%d)",
                self.nfold,
                epoch,
                self.counter,
                self.patience,
            )

        if self.counter >= self.patience:
            logger.warning("[EarlyStopping] Triggered.")
            self.early_stop = True
            return True

        return False


def save_model_artifacts(
    *,
    model,
    fold_save_path: str,
    model_name: str,
    training_features=None,
    trace_jit: bool = True,
    jit_batch: int = 100,
    epoch: int | None = None,
    optimizer=None,
    scheduler=None,
    scaler=None,
    best_score: float | None = None,
    logger=None,
):
    """Save model artifacts (weights + optional checkpoint + optional torchscript)."""
    os.makedirs(fold_save_path, exist_ok=True)
    training_features = training_features or []

    # preserve state
    was_training = model.training
    try:
        orig_device = next(model.parameters()).device
    except StopIteration:
        orig_device = torch.device("cpu")

    # --- 1) weights (always)
    weights_path = os.path.join(fold_save_path, f"{model_name}_weights.pt")
    model.eval()
    torch.save(model.state_dict(), weights_path)

    # --- 2) checkpoint (optional)
    if optimizer is not None:
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_score": best_score,
            "training_features": training_features,
        }
        if scheduler is not None:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None:
            ckpt["scaler_state_dict"] = scaler.state_dict()

        ckpt_path = os.path.join(fold_save_path, f"{model_name}_checkpoint.pt")
        torch.save(ckpt, ckpt_path)

    # --- 3) torchscript (optional)
    if trace_jit and training_features:
        jit_path = os.path.join(fold_save_path, f"{model_name}_torchJit_ver.pt")
        dummy_input = torch.rand(
            int(jit_batch), len(training_features), dtype=torch.float32
        )

        model.to("cpu")
        with torch.inference_mode():
            traced = torch.jit.trace(model, dummy_input)
            traced.save(jit_path)
        model.to(orig_device)

    # restore original mode
    model.train(was_training)

    if logger is not None:
        logger.info("Saved model artifacts (%s) to %s", model_name, fold_save_path)


def save_model_final(model, training_features, fold_save_path, *, logger=logger):
    save_model_artifacts(
        model=model,
        fold_save_path=fold_save_path,
        model_name="final_model",
        training_features=training_features,
        trace_jit=True,
        jit_batch=100,
        optimizer=None,  # set optimizer if you want a final checkpoint too
        logger=logger,
    )


def prepare_features(df, features, variation="nominal"):
    """
    slightly different from the once in dnn_preprocecssor replacing events with df
    """
    features_var = []
    for trf in features:
        if "soft" in trf:
            variation_current = "nominal"
        else:
            variation_current = variation

        if f"{trf}_{variation_current}" in df.columns:
            features_var.append(f"{trf}_{variation_current}")
        elif trf in df.columns:
            features_var.append(trf)
        else:
            logger.info(f"Variable {trf} not found in training dataframe!")
    return features_var


class Net(nn.Module):
    def __init__(
        self,
        n_feat,
        hidden=(128, 64, 32),
        dropout=(0.2, 0.2, 0.2),
        activation="tanh",
        use_batchnorm=True,
    ):
        super(Net, self).__init__()
        h1, h2, h3 = hidden
        d1, d2, d3 = dropout
        act_map = {"relu": F.relu, "gelu": F.gelu, "selu": F.selu, "tanh": torch.tanh}
        if activation not in act_map:
            raise ValueError(f"Unknown activation {activation}")
        self.act = act_map[activation]

        self.fc1 = nn.Linear(n_feat, h1)
        self.bn1 = nn.BatchNorm1d(h1) if use_batchnorm else nn.Identity()
        self.dropout1 = nn.Dropout(d1)

        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2) if use_batchnorm else nn.Identity()
        self.dropout2 = nn.Dropout(d2)

        self.fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3) if use_batchnorm else nn.Identity()
        self.dropout3 = nn.Dropout(d3)

        self.output = nn.Linear(h3, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.dropout3(x)

        logits = self.output(x)

        # output = F.sigmoid(logits)  # for BCEWithLogitsLoss, no need to apply sigmoid here

        return logits


# Custom Dataset class
class NumpyDataset(Dataset):
    def __init__(self, input_arr, label_arr, row_index=None):
        """
        Args:
            input_arr (numpy.ndarray): Input features array.
            label_arr (numpy.ndarray): Labels array.
            row_index (numpy.ndarray, optional): Row indices for tracking. Defaults to None.
        """
        self.input_arr = torch.tensor(input_arr, dtype=torch.float32)
        self.label_arr = torch.tensor(label_arr, dtype=torch.float32)
        if row_index is None:
            row_index = np.arange(len(input_arr))
        self.row_index = torch.tensor(np.asarray(row_index), dtype=torch.int64)

    def __len__(self):
        # Returns the total number of samples
        return len(self.input_arr)

    def __getitem__(self, idx):
        # Retrieve a sample and its corresponding label
        return self.input_arr[idx], self.label_arr[idx], self.row_index[idx]


def plotSigVsBkg(
    score_dict,
    bins,
    plt_save_path,
    transformPrediction=False,
    normalize=True,
    log_scale=False,
):
    """
    TODO: add weights
    """
    fig, ax = plt.subplots()

    # configure scale first (before setting limits)
    if log_scale:
        ax.set_yscale("log")

    for stage, output_dict in score_dict.items():
        pred_total = np.asarray(output_dict["prediction"], dtype=np.float64)
        label_total = np.asarray(output_dict["label"], dtype=np.int64)
        wgt_total = np.asarray(output_dict["weight"], dtype=np.float64)

        # --- basic sanity: same length
        if not (len(pred_total) == len(label_total) == len(wgt_total)):
            logger.warning(
                "[plotSigVsBkg] Length mismatch stage=%s: pred=%d label=%d wgt=%d. Skipping.",
                stage,
                len(pred_total),
                len(label_total),
                len(wgt_total),
            )
            continue

        # --- drop non-finite rows (critical!)
        m = np.isfinite(pred_total) & np.isfinite(wgt_total)
        pred_total = pred_total[m]
        label_total = label_total[m]
        wgt_total = wgt_total[m]

        if pred_total.size == 0:
            logger.warning(
                "[plotSigVsBkg] No finite entries stage=%s. Skipping.", stage
            )
            continue

        if transformPrediction:
            pred_total = safe_arctanh(pred_total)  # must clip internally

        # split
        sig_mask = label_total == 1
        bkg_mask = label_total == 0

        dnn_sig = pred_total[sig_mask]
        dnn_bkg = pred_total[bkg_mask]
        w_sig = wgt_total[sig_mask]
        w_bkg = wgt_total[bkg_mask]

        # handle empty classes
        if dnn_sig.size == 0 or dnn_bkg.size == 0:
            logger.warning(
                "[plotSigVsBkg] Empty sig or bkg stage=%s (sig=%d bkg=%d). Skipping.",
                stage,
                dnn_sig.size,
                dnn_bkg.size,
            )
            continue

        # histogram (density=True can produce NaN if total weight/area is zero)
        hist_sig, _ = np.histogram(dnn_sig, bins=bins, weights=w_sig, density=normalize)
        hist_bkg, _ = np.histogram(dnn_bkg, bins=bins, weights=w_bkg, density=normalize)

        # sanitize histograms
        hist_sig = np.nan_to_num(hist_sig, nan=0.0, posinf=0.0, neginf=0.0)
        hist_bkg = np.nan_to_num(hist_bkg, nan=0.0, posinf=0.0, neginf=0.0)

        max_bin_content = max(float(hist_sig.max()), float(hist_bkg.max()))

        # if everything is zero (or was NaN -> 0), don’t set a bogus ylim
        if max_bin_content <= 0.0 or not np.isfinite(max_bin_content):
            logger.warning(
                "[plotSigVsBkg] max_bin_content not finite/positive stage=%s (max=%s). Skipping ylim.",
                stage,
                str(max_bin_content),
            )
            # still plot (will just show nothing), or you can continue
            # continue

        # set limits safely
        ymin = 1e-4
        if log_scale:
            ymax = max(ymin * 10.0, max_bin_content * 5.0)
        else:
            ymax = max(ymin * 10.0, max_bin_content * 1.1)
        ax.set_ylim(ymin, ymax)

        hep.histplot(
            hist_sig, bins=bins, histtype="step", label=f"Signal - {stage}", ax=ax
        )
        hep.histplot(
            hist_bkg, bins=bins, histtype="step", label=f"Bkg - {stage}", ax=ax
        )

    x_label_addendum = "normalized" if normalize else ""
    ax.set_xlabel(
        ("arctanh Score " if transformPrediction else "DNN Score ") + x_label_addendum
    )
    ax.set_ylabel("Events")
    ax.legend()
    hep.cms.label(data=True, loc=0, label="Private Work 2018", com="13", ax=ax)

    fig.savefig(plt_save_path)
    plt.close(fig)


def customROC_curve_AN(label, pred, weight, ucsd_mode=False):
    """
    generates signal and background efficiency consistent with the AN,
    as described by Fig 4.6 of Dmitry's PhD thesis
    """
    # we assume sigmoid output with labels 0 = background, 1 = signal
    thresholds = np.linspace(start=0,stop=1, num=500)
    effBkg_total = -99*np.ones_like(thresholds) # effBkg = false positive rate
    effSig_total = -99*np.ones_like(thresholds) # effSig = true positive rate
    for ix in range(len(thresholds)):
        threshold = thresholds[ix]
        # get FP and TP
        positive_filter = (pred > threshold)
        falsePositive_filter = positive_filter & (label == 0)
        FP = np.sum(weight[falsePositive_filter])#  FP = false positive
        truePositive_filter = positive_filter & (label == 1)
        TP = np.sum(weight[truePositive_filter])#  TP = true positive


        # get TN and FN
        negative_filter = (pred <= threshold) # just picked negative to be <=
        trueNegative_filter = negative_filter & (label == 0)
        TN = np.sum(weight[trueNegative_filter])#  TN = true negative
        falseNegative_filter = negative_filter & (label == 1)
        FN = np.sum(weight[falseNegative_filter])#  FN = false negative




        if ucsd_mode:
            effBkg = FP / (TN + FP) # AN-19-124 ggH Cat definition
            effSig = TP / (FN + TP) # AN-19-124 ggH Cat definition
        else:
            effBkg = TN / (TN + FP) # Dmitry PhD thesis definition
            effSig = FN / (FN + TP) # Dmitry PhD thesis definition

        effBkg_total[ix] = effBkg
        effSig_total[ix] = effSig

        # logger.info(f"ix: {ix}")
        # logger.info(f"threshold: {threshold}")
        # logger.info(f"effBkg: {effBkg}")
        # logger.info(f"effSig: {effSig}")


        # sanity check
        assert ((np.sum(positive_filter) + np.sum(negative_filter)) == len(pred))
        total_yield = FP + TP + FN + TN
        assert(np.isclose(total_yield, np.sum(weight)))
        # logger.info(f"total_yield: {total_yield}")
        # logger.info(f"np.sum(weight): {np.sum(weight)}")


    effBkg_total[np.isnan(effBkg_total)] = 1
    effSig_total[np.isnan(effSig_total)] = 1

    return (effBkg_total, effSig_total, thresholds)


def plotROC(score_dict, plt_save_path):
    """
    """
    ucsd_mode = "ucsd" in plt_save_path
    fig, ax_main = plt.subplots()
    status = "Private Work 2018"
    CenterOfMass = "13"
    hep.cms.label(data=True, loc=0, label=status, com=CenterOfMass, ax=ax_main)
    for stage, output_dict in score_dict.items():
        pred_total = output_dict["prediction"]
        label_total = output_dict["label"]
        wgt_total = output_dict["weight"]
        eff_bkg, eff_sig, thresholds = customROC_curve_AN(label_total, pred_total, wgt_total, ucsd_mode=ucsd_mode)
        plt.plot(eff_sig, eff_bkg, label=f"{stage}")

    plt.vlines(np.linspace(0,1,11), 0, 1, linestyle="dashed", color="grey")
    # plt.hlines(np.logspace(-4,0,5), 0, 1, linestyle="dashed", color="grey")
    # plt.hlines(eff_bkg, 0, eff_sig, linestyle="dashed")
    plt.xlim([0.0, 1.0])
    if ucsd_mode:
        plt.hlines(np.logspace(-4,0,5), 0, 1, linestyle="dashed", color="grey")
        plt.yscale('log')
        plt.ylim([0.001, 1.0])
    else:
        plt.ylim([0.0, 1.0])
        plt.hlines(np.linspace(0,1,11), 0, 1, linestyle="dashed", color="grey")
    plt.xlabel('$\\epsilon_{sig}$')
    plt.ylabel('$\\epsilon_{bkg}$')

    plt.legend(loc="lower right")
    # plt.title(f'ROC curve for ggH BDT {year}')
    plt.savefig(plt_save_path)
    plt.clf()
    plt.close(fig)  # Close the figure to free memory


def dnnEvaluateLoop(model, dataloader, loss_fn, device="cpu"):
    """
    Helper function running through the evaluation.

    Fixes:
      - Correct loss averaging: accumulates sum(loss * batch_size) and divides by total N.
      - Defines n_total (was missing).
      - Stores avg_loss (not raw summed loss) in return_dict["total_loss"].
      - Keeps the AUC log explicitly unweighted (no weights available in this dataloader).
    """
    model.eval()

    total_loss_sum = 0.0  # sum of (batch_loss_mean * batch_size)
    n_total = 0           # total number of events
    batch_losses = []
    pred_l = []
    label_l = []

    with torch.no_grad():
        for _, (inputs, labels, _ridx) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device).reshape((-1, 1))

            logits = model(inputs)
            loss = loss_fn(logits, labels)  # typically mean over batch
            batch_loss = float(loss.item())
            bs = int(labels.size(0))

            total_loss_sum += batch_loss * bs
            n_total += bs
            batch_losses.append(batch_loss)

            probs = torch.sigmoid(logits)
            pred_l.append(probs.detach().cpu().numpy())
            label_l.append(labels.detach().cpu().numpy())

    pred_total = np.concatenate(pred_l, axis=0).ravel() if pred_l else np.array([], dtype=float)
    label_total = np.concatenate(label_l, axis=0).ravel() if label_l else np.array([], dtype=float)

    avg_loss = (total_loss_sum / n_total) if n_total > 0 else float("nan")

    # Unweighted AUC here (weights are not provided by this dataloader)
    auc_score = safe_weighted_auc(label_total, pred_total, sample_weight=None)
    logger.info(f"Evaluation AUC (unweighted): {auc_score:.4f}")

    return_dict = {
        "label": label_total,
        "prediction": pred_total,
        "total_loss": avg_loss,          # average per-event loss
        "batch_losses": batch_losses,    # per-batch mean losses (useful for debugging)
        "n_total": n_total,
    }

    model.train()  # turn back to train mode
    return return_dict

def axplot_to_plotly(obj):
    """
    Convert Ax plot outputs (AxPlotConfig / plotly Figure / dict) into something
    plotly.io.write_html can consume.
    Returns a plotly.graph_objects.Figure or a plotly-compatible dict.
    """
    import plotly.graph_objects as go

    # Already a plotly Figure
    if isinstance(obj, go.Figure):
        return obj

    # AxPlotConfig (newer Ax): obj.data is dict with keys {'data','layout'}
    if hasattr(obj, "data"):
        try:
            return go.Figure(obj.data)
        except Exception:
            # Fallback: sometimes obj.data is already Figure-like dict
            return obj.data

    # Already plotly-compatible dict
    if isinstance(obj, dict):
        # Wrap into Figure for maximum compatibility
        try:
            return go.Figure(obj)
        except Exception:
            return obj

    raise TypeError(f"Unsupported plot object type: {type(obj)}")


def get_standard_plots_robust(experiment):
    """
    Works across Ax versions where get_standard_plots may or may not need `model`.
    Returns a list of AxPlotConfig objects (or plotly figs depending on Ax version).
    """
    from ax.plot import diagnostic
    from ax.modelbridge.registry import Models

    data = experiment.fetch_data()

    # Try the "newer signature" first: get_standard_plots(experiment=..., model=...)
    # Build a model in a version-tolerant way (keyword or positional).
    model = None
    try:
        model = Models.BOTORCH_MODULAR(experiment=experiment, data=data)
    except TypeError:
        try:
            model = Models.BOTORCH_MODULAR(experiment, data)
        except TypeError:
            model = None

    if model is not None:
        try:
            return diagnostic.get_standard_plots(experiment=experiment, model=model)
        except TypeError:
            # Some Ax versions accept only positional args here
            try:
                return diagnostic.get_standard_plots(experiment, model)
            except TypeError:
                pass

    # Fallback: older signature get_standard_plots(experiment)
    try:
        return diagnostic.get_standard_plots(experiment=experiment)
    except TypeError:
        return diagnostic.get_standard_plots(experiment)


def ValidationPlots(
    model,
    epoch,
    fold_idx,
    fold_save_path,
    df_valid,
    training_features,
    best_significance,
    score_dict,
    pred_total,
    label_total,
    df_train,
    train_loop_dict,
    valid_loop_dict,
):
    """
    Fixed version:
      - Uses a consistent "stage" for ALL per-process plots (pred/label/weight/process arrays
        are taken from score_dict[stage], not df_valid).
      - Uses wgt_total (not df_valid.wgt_nominal) when slicing by proc_filter.
      - Uses proc_total (not df_valid.process) in BOTH linear and log-scale per-process plots.
      - Adds strict sanity checks for array length consistency.
      - Keeps your ROC + Sig/Bkg plots as-is (they use score_dict stages).
    """

    # -------------------------
    # ROC + score distributions
    # -------------------------
    plt_save_path = f"{fold_save_path}/epoch{epoch}_ROC.png"
    plotROC(score_dict, plt_save_path)

    plt_save_path = f"{fold_save_path}/epoch{epoch}_ROC_ucsd.png"
    plotROC(score_dict, plt_save_path)

    bins_uniform = np.linspace(0, 1, 30)
    plt_save_path = f"{fold_save_path}/epoch{epoch}_DNN_combined_dist_bySigBkg.png"
    plotSigVsBkg(score_dict, bins_uniform, plt_save_path, transformPrediction=False)

    bins = selection.binning
    plt_save_path = (
        f"{fold_save_path}/epoch{epoch}_DNN_combined_transformedDist_bySigBkg.png"
    )
    plotSigVsBkg(score_dict, bins, plt_save_path, transformPrediction=True)

    # NOTE: if you want log-scale for the transformed plot, you probably want transformPrediction=True.
    # Keeping your call style but fix the filename logic.
    plotSigVsBkg(
        score_dict,
        bins,
        plt_save_path.replace(".png", "_log.png"),
        transformPrediction=True,
        log_scale=True,
    )

    # -------------------------
    # Choose stage for per-process plots
    # -------------------------
    stage = (
        "valid"  # or "eval" or "train" or "valid+eval" (if you include process there)
    )

    if stage not in score_dict:
        raise KeyError(
            f"stage='{stage}' not in score_dict keys={list(score_dict.keys())}. "
            "Make sure score_dict[stage] contains prediction/label/weight/process."
        )

    needed = ("prediction", "label", "weight", "process")
    for k in needed:
        if k not in score_dict[stage]:
            raise KeyError(
                f"score_dict['{stage}'] missing key '{k}'. "
                "Add 'process' alongside prediction/label/weight when building score_dict."
            )

    pred_total = np.asarray(score_dict[stage]["prediction"])
    label_total = np.asarray(score_dict[stage]["label"])
    wgt_total = np.asarray(score_dict[stage]["weight"])
    proc_total = np.asarray(score_dict[stage]["process"])

    # -------------------------
    # Sanity checks (critical)
    # -------------------------
    n = len(pred_total)
    if not (len(label_total) == len(wgt_total) == len(proc_total) == n):
        raise ValueError(
            f"[ValidationPlots] Length mismatch for stage='{stage}': "
            f"pred={len(pred_total)}, label={len(label_total)}, "
            f"weight={len(wgt_total)}, process={len(proc_total)}. "
            "This usually happens if your dataloader is shuffled and you are not reordering by row_index."
        )

    # Ensure probabilities (because transformDnnScore expects [0,1])
    if np.nanmin(pred_total) < 0.0 or np.nanmax(pred_total) > 1.0:
        raise ValueError(
            f"[ValidationPlots] pred_total outside [0,1] for stage='{stage}' "
            f"(min={np.nanmin(pred_total):.3g}, max={np.nanmax(pred_total):.3g}). "
            "You likely passed logits instead of probabilities to score_dict. "
            "Fix: store sigmoid(logits) into score_dict."
        )

    # ------------------------------------------
    # Per-process (line) plots: dN/d(score) by process
    # ------------------------------------------
    processes = ["dy", "top", "ewk", "vbf", "ggh"]

    fig, ax = plt.subplots()

    for proc in processes:
        proc_filter = (proc_total == proc)
        if not np.any(proc_filter):
            continue

        dnn_scores = pred_total[proc_filter]
        dnn_scores = transformDnnScore(dnn_scores)
        wgt_proc = wgt_total[proc_filter]

        hist_proc, bins_proc = np.histogram(dnn_scores, bins=bins, weights=wgt_proc)
        bin_centers_proc = 0.5 * (bins_proc[:-1] + bins_proc[1:])
        ax.plot(bin_centers_proc, hist_proc, label=proc, drawstyle="steps-mid")

    ax.set_xlabel("arctanh Score")
    ax.set_ylabel("Events")  # not "Density" unless you set density=True in np.histogram
    ax.set_title(f"DNN Score Distributions by Sample ({stage})")
    ax.legend()
    fig.savefig(f"{fold_save_path}/epoch{epoch}_DNN_{stage}_dist_byProcess.png")
    plt.clf()

    # ------------------------------------------
    # Per-process stacked plot (log-scale) + significance
    # ------------------------------------------
    fig, ax_main = plt.subplots()
    ax_main.set_yscale("log")
    ax_main.set_ylim(0.01, 1e9)

    bkg_processes = ["ewk", "top", "dy"]  # smallest first
    bkg_hist_l = []
    for proc in bkg_processes:
        proc_filter = proc_total == proc
        if not np.any(proc_filter):
            bkg_hist_l.append(np.zeros(len(bins) - 1))
            continue

        dnn_scores = transformDnnScore(pred_total[proc_filter])
        wgt = wgt_total[proc_filter]
        hist_proc, _ = np.histogram(dnn_scores, bins=bins, weights=wgt)
        bkg_hist_l.append(hist_proc)

    hep.histplot(
        bkg_hist_l,
        bins=bins,
        stack=True,
        histtype="fill",
        label=bkg_processes,
        sort="label_r",
        ax=ax_main,
    )

    sig_processes = ["vbf", "ggh"]
    sig_hist_l = []
    for proc in sig_processes:
        proc_filter = proc_total == proc
        if not np.any(proc_filter):
            sig_hist_l.append(np.zeros(len(bins) - 1))
            continue

        dnn_scores = transformDnnScore(pred_total[proc_filter])
        wgt = wgt_total[proc_filter]
        hist_proc, _ = np.histogram(dnn_scores, bins=bins, weights=wgt)
        sig_hist_l.append(hist_proc)

        hep.histplot(
            hist_proc,
            bins=bins,
            histtype="step",
            label=proc,
            ax=ax_main,
        )

    ax_main.set_xlabel("arctanh Score")
    ax_main.set_ylabel("Events")

    sig_hist_total = np.sum(sig_hist_l, axis=0)
    bkg_hist_total = np.sum(bkg_hist_l, axis=0)
    significance = calculateSignificance(sig_hist_total, bkg_hist_total)
    best_significance = significance

    logger.info(
        f"new best significance for fold {fold_idx} is {best_significance} from epoch {epoch} "
        f"(stage={stage})"
    )

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax_main.text(
        0.05,
        0.95,
        f"Significance: {best_significance:.3f}",
        transform=ax_main.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    plt.title(f"DNN Score Distributions ({stage})")
    plt.legend()
    plt.savefig(f"{fold_save_path}/epoch{epoch}_DNN_{stage}_stackedDist_byProcess.png")
    plt.clf()
    plt.close(fig)

    # =========================
    # Extra validation (ROOT)
    # =========================
    plot_overtraining_KS_ROOT(
        train_loop_dict["prediction"],
        train_loop_dict["label"],
        df_train.wgt_nominal.values,
        valid_loop_dict["prediction"],
        valid_loop_dict["label"],
        df_valid.wgt_nominal.values,
        save_path=f"{fold_save_path}/epoch{epoch}_KS_Overtraining.pdf",
        nbins=30,
    )

    plot_calibration_ROOT(
        valid_loop_dict["prediction"],
        valid_loop_dict["label"],
        df_valid.wgt_nominal.values,
        save_path=f"{fold_save_path}/epoch{epoch}_Calibration.pdf",
        n_bins=15,
    )

    plot_threshold_scan_ROOT(
        valid_loop_dict["prediction"],
        valid_loop_dict["label"],
        df_valid.wgt_nominal.values,
        save_path=f"{fold_save_path}/epoch{epoch}_ThresholdScan.pdf",
    )

    plot_weight_distribution_ROOT(
        df_valid.wgt_nominal.values, f"{fold_save_path}/epoch{epoch}"
    )

    return best_significance


def dnn_train(model, data_dict, fold_idx, training_features, batch_size, nepochs, save_path,
              callback=None, lr=1e-3, optimizer_name="adam", weight_decay=0.0, loss_name="bce"):
    logger.setLevel(logging.INFO)
    if len(training_features) == 0:
        logger.error("ERROR: please define the training features the DNN will train on")
        raise ValueError

    fold_save_path = f"{save_path}/fold{fold_idx}"
    if not os.path.exists(fold_save_path):
        os.makedirs(fold_save_path)

    train_losses = []
    val_losses = []

    # divide our data into 4 folds
    # input_arr_train, label_arr_train = data_dict["train"]
    # input_arr_valid, label_arr_valid = data_dict["validation"]
    # logger.info(f"data_dict.keys(): {data_dict.keys()}")
    df_train = data_dict["train"]
    df_valid = data_dict["validation"]
    df_eval = data_dict["evaluation"]
    input_arr_train = df_train[training_features].values
    label_arr_train = df_train.label.values
    input_arr_valid = df_valid[training_features].values
    label_arr_valid = df_valid.label.values
    input_arr_eval = df_eval[training_features].values
    label_arr_eval = df_eval.label.values

    # CHOOSE LOSS
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # loss_fn = torch.nn.BCELoss()
    # if loss_name == "focal":
    #     loss_fn = FocalLoss(alpha=1, gamma=2)
    # elif loss_name == "hinge":
    #     loss_fn = HingeLoss()
    # else:
    #     loss_fn = torch.nn.BCELoss()
    #     # loss_fn = torch.nn.BCEWithLogitsLoss()

    # Iterating through the DataLoader
    #
    logger.info(f"input_arr_train shape: {input_arr_train.shape}")

    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # CHOOSE OPTIMIZER
    if optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    dataset_train = NumpyDataset(input_arr_train, label_arr_train, row_index=df_train.index.to_numpy())
    dataloader_train_ordered = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=NWORKERS, pin_memory=PIN_MEMORY) # for plotting
    dataset_valid = NumpyDataset(input_arr_valid, label_arr_valid, row_index=df_valid.index.to_numpy())
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=NWORKERS, pin_memory=PIN_MEMORY)
    dataset_eval = NumpyDataset(input_arr_eval, label_arr_eval, row_index=df_eval.index.to_numpy())
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False, num_workers=NWORKERS, pin_memory=PIN_MEMORY)
    best_significance = 0
    mode = "max" # max: maximize the auc, min: minimize the loss
    early_stopping_callback = EarlyStopping(patience=5, delta=1e-3, mode=mode, nfold=fold_idx, fold_save_path=f"{fold_save_path}", model=model, training_features=training_features, verbose=True)

    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "train_auc": [], "val_auc": [],
        "significance": [], "lr": []
    }
    for epoch in range(nepochs):
        model.train()
        callback.on_epoch_begin(epoch)
        # every epoch, reshuffle train data loader (could be unncessary)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=NWORKERS, pin_memory=PIN_MEMORY)

        epoch_loss = 0
        batch_losses = []
        for batch_idx, (inputs, labels, ridx) in enumerate(dataloader_train):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).reshape((-1,1))

            optimizer.zero_grad()

            # Make predictions for this batch
            pred = model(inputs)

            logger.debug(f"inputs: {inputs}")
            logger.debug(f"labels: {labels}")
            logger.debug(f"pred: {pred}")

            # Compute the loss and its gradients
            loss = loss_fn(pred, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # ---- Accuracy per batch ----
            with torch.no_grad():
                # Convert predictions to 0/1 using threshold 0.5
                probs = torch.sigmoid(pred)
                pred_binary = (probs >= 0.5).float()

                correct = (pred_binary == labels).sum().item()
                batch_acc = correct / labels.size(0)

            # For logging: Gather data and report
            batch_loss = loss.item()
            epoch_loss += batch_loss
            batch_losses.append(batch_loss)
            callback.on_batch_end(batch_idx, logs={'loss': batch_loss, 'accuracy': batch_acc})

        train_losses.append(epoch_loss)
        logger.debug(f"fold {fold_idx} epoch {epoch}            train total loss: {epoch_loss}")
        logger.debug(f"fold {fold_idx} epoch {epoch}    train average batch loss: {np.mean(batch_losses)}")
        valid_loop_dict = dnnEvaluateLoop(model, dataloader_valid, loss_fn, device=DEVICE)
        train_loop_dict = dnnEvaluateLoop(model, dataloader_train_ordered, loss_fn, device=DEVICE)
        eval_loop_dict = dnnEvaluateLoop(model, dataloader_eval, loss_fn, device=DEVICE)
        score_dict = {
            "train": {
                "prediction": train_loop_dict["prediction"],
                "label": train_loop_dict["label"],
                "weight": df_train.wgt_nominal.values,
                "process": df_train.process.values,   # add this now (important)
            },
            "valid": {
                "prediction": valid_loop_dict["prediction"],
                "label": valid_loop_dict["label"],
                "weight": df_valid.wgt_nominal.values,
                "process": df_valid.process.values,
            },
            "eval": {
                "prediction": eval_loop_dict["prediction"],
                "label": eval_loop_dict["label"],
                "weight": df_eval.wgt_nominal.values,
                "process": df_eval.process.values,
            },
        }

        pred_total = valid_loop_dict["prediction"]
        label_total = valid_loop_dict["label"]
        valid_loss = valid_loop_dict["total_loss"]
        batch_losses = valid_loop_dict["batch_losses"]
        auc_score = safe_weighted_auc(label_total, pred_total, sample_weight=df_valid.wgt_nominal.values)
        val_losses.append(valid_loss)
        logger.debug(f"fold {fold_idx} epoch {epoch} validation total loss: {valid_loss}")
        logger.debug(f"fold {fold_idx} epoch {epoch} validation average batch loss: {np.mean(batch_losses)}")
        logger.debug(f"fold {fold_idx} epoch {epoch} validation AUC: {auc_score}")

        # call early stopping
        best_score_for_earlyStopping = 0
        if mode == "min":
            best_score_for_earlyStopping = valid_loss
        if mode == "max":
            best_score_for_earlyStopping = auc_score
        if early_stopping_callback and  early_stopping_callback.on_epoch_end(epoch, best_score_for_earlyStopping):
            logger.warning(f"Early stopping at epoch {epoch} for fold {fold_idx}")
            # save_model_final(model, training_features, fold_save_path)
            break

        # ------------------------------------------------
        # plot the score distributions
        # ------------------------------------------------
        # validate_interval = 20
        # if ((epoch==0) or ((epoch % validate_interval) == (validate_interval-1))) or (epoch==nepochs-1):
        best_significance = ValidationPlots(model, epoch, fold_idx, fold_save_path, df_valid, training_features, best_significance, score_dict, pred_total, label_total,
                                            df_train, train_loop_dict, valid_loop_dict)

        callback.on_epoch_end(epoch, logs={'loss': epoch_loss, 'auc': auc_score, 'significance': best_significance})

        # AUCs (weighted, if you want)
        train_auc = _safe_auc(train_loop_dict["label"], train_loop_dict["prediction"], w=df_train.wgt_nominal.values)
        val_auc   = _safe_auc(valid_loop_dict["label"], valid_loop_dict["prediction"], w=df_valid.wgt_nominal.values)

        # physics significance from the SAME per-epoch stacked hists you already build
        # reuse best_significance just computed by ValidationPlots:
        sig_epoch = best_significance

        # learning rate (works even without scheduler)
        curr_lr = optimizer.param_groups[0]["lr"]

        fpr, tpr = _roc_weighted(valid_loop_dict["prediction"],
                                valid_loop_dict["label"],
                                df_valid.wgt_nominal.values, n=300)

        np.savez(f"{fold_save_path}/cv_artifacts.npz",
                auc=safe_weighted_auc(valid_loop_dict["label"], valid_loop_dict["prediction"],
                                sample_weight=df_valid.wgt_nominal.values),
                fpr=fpr, tpr=tpr,
                pred=valid_loop_dict["prediction"],
                label=valid_loop_dict["label"],
                weight=df_valid.wgt_nominal.values.astype("f"),
                fold=fold_idx)

        # log into history
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(valid_loop_dict["total_loss"])
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)
        history["significance"].append(sig_epoch)
        history["lr"].append(curr_lr)
        # END of epoch loop

    # after you build train_loop_dict, valid_loop_dict, and have df_train/df_valid
    plot_overtraining_KS_ROOT(
        train_loop_dict["prediction"], train_loop_dict["label"], df_train.wgt_nominal.values,
        valid_loop_dict["prediction"], valid_loop_dict["label"], df_valid.wgt_nominal.values,
        save_path=f"{fold_save_path}/KS_Overtraining_{fold_idx}.pdf",
        nbins=30
    )
    plot_calibration_ROOT(
        valid_loop_dict["prediction"], valid_loop_dict["label"], df_valid.wgt_nominal.values,
        save_path=f"{fold_save_path}/Calibration_{fold_idx}.pdf", n_bins=15
    )

    plot_threshold_scan_ROOT(
        valid_loop_dict["prediction"], valid_loop_dict["label"], df_valid.wgt_nominal.values,
        save_path=f"{fold_save_path}/ThresholdScan_{fold_idx}.pdf"
    )

    # Build the list once:
    abs_rhos = []

    pred = np.asarray(valid_loop_dict["prediction"], dtype=float)
    w = np.asarray(df_valid.wgt_nominal.values, dtype=float)
    for f in training_features:
        x = df_valid[f].values.astype(float)
        # compute weighted |rho|: (use your existing weighted corr helpers)
        sx, sp = _wstd(x, w), _wstd(pred, w)
        rho = 0.0 if (sx==0 or sp==0) else _wcov(x, pred, w)/(sx*sp)
        abs_rhos.append((f, abs(rho), rho))

    try:
        plot_score_feature_corr_ROOT_bar(
            abs_rhos, f"{fold_save_path}/epoch{epoch}_CorrBar.pdf", topk=26
        )
    except Exception as e:
        logger.exception("[CorrHeatmap] Skipped due to error: %s", e)

    try:
        plot_score_feature_corr_ROOT_heatmap(
            abs_rhos,
            save_path=f"{fold_save_path}/CorrHeatmap_{fold_idx}.pdf",
            topk=26,
        )
    except Exception as e:
        logger.exception("[CorrHeatmap] Skipped due to error: %s", e)

    # # cats = np.where(df_valid.nJets.values>=2, "njet>=2", "njet<2")  # adapt to your column
    # # plot_score_shapes_and_roc_by_category_ROOT(
    # #     valid_loop_dict["prediction"], valid_loop_dict["label"], df_valid.wgt_nominal.values,
    # #     cats, save_prefix=f"{fold_save_path}/PerCategory_{fold_idx}"
    # # )

    # # e.g., processes in your df_valid: ["dy","top","ewk","vbf","ggh"]
    # signal_procs = ["vbf","ggh"]
    # plot_cumulative_SSB_per_process_ROOT(
    #     valid_loop_dict["prediction"], valid_loop_dict["label"], df_valid.wgt_nominal.values,
    #     df_valid.process.values, signal_procs,
    #     save_path=f"{fold_save_path}/Cumulative_SSB_{fold_idx}.pdf"
    # )

    auc_base, perm_res = permutation_importance_auc(
        model, df_valid, training_features,
        labels=valid_loop_dict["label"],
        weights=df_valid.wgt_nominal.values,
        device=DEVICE, n_repeats=1, subsample=50000  # subsample optional/speed
    )
    plot_perm_importance_ROOT(perm_res, f"{fold_save_path}/epoch{epoch}_PermImportance.pdf", topk=26)

    # ['dimuon_mass', 'dimuon_ebe_mass_res', 'dimuon_ebe_mass_res_rel', 'jj_mass_nominal', 'jj_mass_log_nominal', 'rpt_nominal', 'll_zstar_log_nominal',
    #  'jj_dEta_nominal', 'nsoftjets5_nominal', 'mmj_min_dEta_nominal', 'dimuon_pt', 'dimuon_pt_log', 'dimuon_rapidity', 'jet1_pt_nominal', 'jet1_eta_nominal', 'jet1_phi_nominal',
    #  'jet2_pt_nominal', 'jet2_eta_nominal', 'jet2_phi_nominal', 'jet1_qgl_nominal', 'jet2_qgl_nominal', 'dimuon_cos_theta_cs', 'dimuon_phi_cs', 'htsoft2_nominal',
    #  'pt_centrality_nominal', 'year']
    for feat, label in [
        ("dimuon_pt", "p_{T}^{#mu#mu}"),
        ("dimuon_ebe_mass_res", "m_{#mu#mu}^{res}"),
        ("dimuon_ebe_mass_res_rel", "m_{#mu#mu}^{res, rel}"),
        ("jet1_eta_nominal", "#eta_{jet1}"),
        ("jet1_pt_nominal", "#p_{T}^{jet1}"),
        ("jet2_eta_nominal", "#eta_{jet2}"),
        ("jet2_pt_nominal", "#p_{T}^{jet2}"),
        ("nsoftjets5_nominal", "N_{soft jets}"),
        ("rpt_nominal", "rpt"),
        ("jj_mass_nominal", "m_{jj}"),
        ("jj_dEta_nominal", "#Delta#eta_{jj}"),
    ]:
        if feat in training_features:
            gx, gy = partial_dependence_curve(
                model, df_valid, training_features, feat, df_valid.wgt_nominal.values, DEVICE,
                grid="quantile", nbins=15, subsample=40000
            )
            plot_pdp_ROOT(gx, gy, xlabel=label, save_path=f"{fold_save_path}/epoch{epoch}_PDP_{feat}.pdf")

    # If you have the threshold scan arrays, choose best_t = thr[np.argmax(SSB)]
    # Otherwise pick a fixed cut:
    best_t = 0.70

    res = yield_table_after_cut(
        valid_loop_dict["prediction"], valid_loop_dict["label"],
        df_valid.wgt_nominal.values, df_valid.process.values,
        score_cut=best_t, save_prefix=f"{fold_save_path}/epoch{epoch}"
    )
    logger.info(f"[YieldTable] S/sqrt(B) at cut {best_t:.3f}: {res['ssb']:.3f}; table: {res['txt']} / {res['csv']}")

    # Save final model state (in case early stopping did not trigger)
    save_model_final(model, training_features, fold_save_path)

    # Validation plots
    # ------------------------------------------------
    # 1. Plot the loss curves
    plot_loss_curves(train_losses, val_losses, save_path=f"{fold_save_path}/loss_curves_{fold_idx}.pdf")
    # 2. Plot the ROC curve
    plotROC(score_dict, plt_save_path=f"{fold_save_path}/ROC_curve_{fold_idx}.pdf")
    # 3. Plot the Sig/Bkg distributions
    bins = np.linspace(0, 1, 30)
    plotSigVsBkg(score_dict, bins, plt_save_path=f"{fold_save_path}/SigBkg_dist_{fold_idx}.pdf", transformPrediction=False, normalize=True)
    plotSigVsBkg(score_dict, bins, plt_save_path=f"{fold_save_path}/SigBkg_dist_{fold_idx}_log.pdf", transformPrediction=False, normalize=True, log_scale=True)
    # 4. Plot the Sig/Bkg distributions with transformed scores
    bins = selection.binning
    plotSigVsBkg(score_dict, bins, plt_save_path=f"{fold_save_path}/SigBkg_dist_transformed_{fold_idx}.pdf", transformPrediction=True, normalize=True)
    plotSigVsBkg(score_dict, bins, plt_save_path=f"{fold_save_path}/SigBkg_dist_transformed_{fold_idx}_log.pdf", transformPrediction=True, normalize=True, log_scale=True)
    # 4. Precision vs Recall curve
    plotPrecisionRecall(score_dict, plt_save_path=f"{fold_save_path}/PrecisionRecall_curve_{fold_idx}.pdf")
    # 5. Confusion matrix
    plotConfusionMatrix(score_dict, plt_save_path=f"{fold_save_path}/ConfusionMatrix_{fold_idx}.pdf")
    # 6. Feature importance
    # plotFeatureImportance(model, training_features, plt_save_path=f"{fold_save_path}/FeatureImportance_{fold_idx}.pdf")

    callback.save_logs(f"{fold_save_path}/epoch{epoch}_training_logs.pkl")
    # calculate the scale, save it
    # save the resulting df for training
    plot_auc_and_loss(
        history, save_path=f"{fold_save_path}/AUC_and_Loss_vs_Epoch_{fold_idx}.pdf"
    )
    plot_significance(
        history, save_path=f"{fold_save_path}/Significance_vs_Epoch_{fold_idx}.pdf"
    )
    plot_lr(history, save_path=f"{fold_save_path}/LearningRate_vs_Epoch_{fold_idx}.pdf")


def calculateSignificance(sig_hist, bkg_hist):
    """
    Calculate significance using the Asimov formula.
    sig_hist: array of signal histogram counts per bin
    bkg_hist: array of background histogram counts per bin
    Returns the significance value.
    """
    s = sig_hist
    b = np.where(bkg_hist > 0.0, bkg_hist, np.nan)  # avoid div by zero
    value = 2 * ( (s + b) * np.log(1 + s / b) - s )
    value = np.nansum(value)
    return np.sqrt(value)


def _make_dl(df_train, df_valid, feats, batch_size):
    ds_tr = NumpyDataset(df_train[feats].values, df_train.label.values)
    ds_va = NumpyDataset(df_valid[feats].values, df_valid.label.values)
    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NWORKERS,
        pin_memory=PIN_MEMORY,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NWORKERS,
        pin_memory=PIN_MEMORY,
    )
    return dl_tr, dl_va


@torch.no_grad()
def _valid_auc(model, dl, device):
    model.eval()
    probs, labels = [], []
    for batch in dl:
        xb, yb = batch[0].to(device), batch[1].to(device).reshape((-1, 1))
        logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p.ravel())
        labels.append(yb.cpu().numpy().ravel())
    return roc_auc_score(np.concatenate(labels), np.concatenate(probs))


def bo_evaluate(params, *, save_path, training_features, bo_fold=0, bo_epochs=30):
    """
    Ax calls this with 'params' dict. We run a short train on one fold and return AUC.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_tr = pd.read_parquet(f"{save_path}/data_df_train_{bo_fold}.parquet")
    df_va = pd.read_parquet(f"{save_path}/data_df_validation_{bo_fold}.parquet")
    feats = prepare_features(df_tr, training_features)

    # derive architecture
    h1 = int(params["hidden0"])
    h2 = max(8, int(h1 * float(params["shrink1"])))
    h3 = max(8, int(h2 * float(params["shrink2"])))
    d = float(params["dropout"])
    bs = int(params["batch_size"])

    dl_tr, dl_va = _make_dl(df_tr, df_va, feats, bs)
    model = Net(
        n_feat=len(feats),
        hidden=(h1, h2, h3),
        dropout=(d, d, d),
        activation=params["activation"],
    ).to(device)

    # loss/opt
    loss_fn = (
        FocalLoss(alpha=1, gamma=2)
        if params["loss_name"] == "focal"
        else torch.nn.BCEWithLogitsLoss()
    )
    if params["optimizer"] == "adamw":
        opt = optim.AdamW(
            model.parameters(),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
        )
    else:
        opt = optim.Adam(
            model.parameters(),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
        )

    # short training with patience
    best, bad, patience = -1.0, 0, 5
    for _ in range(int(bo_epochs)):
        model.train()
        for batch in dl_tr:
            xb, yb = batch[0].to(device), batch[1].to(device).reshape((-1, 1))
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
        auc = _valid_auc(model, dl_va, device)
        if auc > best + 1e-4:
            best, bad = auc, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best


def save_bo_artifacts(experiment, values, best_params, outdir, objective_name="auc"):
    os.makedirs(outdir, exist_ok=True)

    # 1) Full experiment JSON (reloadable)
    save_experiment(
        experiment=experiment, filepath=os.path.join(outdir, "ax_experiment.json")
    )

    # 2) Trials table straight from Ax data
    raw = experiment.fetch_data().df.copy()
    # Expected columns include: ["arm_name","metric_name","mean","sem","trial_index", ...]
    if "metric_name" not in raw.columns:
        raise RuntimeError(
            f"Ax data missing 'metric_name'. Columns: {list(raw.columns)}"
        )
    raw = raw[raw["metric_name"] == objective_name].sort_values("trial_index")

    # Be defensive about metric value column name
    mean_col = next((c for c in ["mean", "value", "data"] if c in raw.columns), None)
    if mean_col is None:
        raise RuntimeError(
            f"No metric value column found in Ax data. Columns: {list(raw.columns)}"
        )

    raw = raw.rename(columns={mean_col: objective_name})

    # Pull parameters per trial/arm
    rows = []
    for t in experiment.trials.values():
        arms = []
        if hasattr(t, "arms"):
            arms = list(t.arms)
        elif hasattr(t, "arm"):
            arms = [t.arm]

        for arm in arms:
            rows.append(
                {"trial_index": t.index, "arm_name": arm.name, **arm.parameters}
            )

    ptab = pd.DataFrame(rows)

    # Merge metric measurements with parameters
    # (If parameters are missing for some arms, keep the metric row anyway)
    df = raw.merge(ptab, on=["trial_index", "arm_name"], how="left").reset_index(
        drop=True
    )
    df.to_csv(os.path.join(outdir, "ax_trials.csv"), index=False)

    # 3) Human-readable best summary
    means, covs = values
    best_mean = float(means.get(objective_name, float("nan")))
    with open(os.path.join(outdir, "ax_best.txt"), "w") as f:
        f.write(f"Objective: {objective_name}\n")
        f.write(f"Best mean {objective_name}: {best_mean:.6f}\n")
        f.write("Best parameters:\n")
        for k, v in best_params.items():
            f.write(f"  - {k}: {v}\n")

    # 4) Plot: optimization trace (objective vs trial index)
    # Use grouped mean by trial in case of multiple arms/observations
    try:
        y_by_trial = df.groupby("trial_index")[objective_name].mean().sort_index()

        # IMPORTANT: Ax's optimization_trace_single_method expects shape (n_trials, 1)
        # not (1, n_trials)
        y_mat = y_by_trial.to_numpy().reshape(-1, 1)

        if y_mat.shape[0] >= 1:
            trace_obj = optimization_trace_single_method(
                y=y_mat,
                title=f"Optimization Trace ({objective_name} vs. trial)",
                ylabel=objective_name.upper(),
            )
            pio.write_html(
                axplot_to_plotly(trace_obj),
                file=os.path.join(outdir, "01_optimization_trace.html"),
                include_plotlyjs="cdn",
                auto_open=False,
            )
        else:
            logger.warning("[Ax] Skipping optimization trace: no trials found.")
    except Exception as e:
        logger.warning(f"[Ax] optimization_trace_single_method failed: {e}")

    # 5) Standard Ax plots (slice, contour, diagnostics)
    try:
        plots = get_standard_plots_robust(experiment)
        for i, p in enumerate(plots):
            pio.write_html(
                axplot_to_plotly(p),
                file=os.path.join(outdir, f"02_standard_plot_{i:02d}.html"),
                include_plotlyjs="cdn",
                auto_open=False,
            )
    except Exception as e:
        logger.warning(f"[Ax] get_standard_plots failed: {e}")


class BOTrialRecorder:
    def __init__(self, out_dir, objective_name="auc"):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.csv_path = Path(out_dir) / "bo_trials_live.csv"
        self.jsonl_path = Path(out_dir) / "bo_trials_live.jsonl"
        self.objective = objective_name
        self._lock = threading.Lock()
        # write CSV header if new
        if not self.csv_path.exists():
            with open(self.csv_path, "w") as f:
                f.write(
                    "trial_index,duration_sec,status,"  # fixed fields first
                    "auc,auc_sem,"  # metrics (sem kept for consistency)
                    "params_json\n"
                )  # keep params in a JSON column
        # JSONL is schemaless; no header

    def record(self, trial_index, params, auc, duration_sec, status="ok", auc_sem=""):
        row = {
            "trial_index": int(trial_index),
            "duration_sec": float(duration_sec),
            "status": str(status),
            self.objective: float(auc),
            f"{self.objective}_sem": auc_sem,
            "params": params,  # preserve types
        }
        line_csv = (
            f'{row["trial_index"]},{row["duration_sec"]:.3f},{row["status"]},'
            f'{row[self.objective]:.7f},{row[f"{self.objective}_sem"]},'
            f'{json.dumps(params, separators=(",", ":"))}\n'
        )
        line_json = json.dumps(row, separators=(",", ":")) + "\n"
        with self._lock:
            with open(self.csv_path, "a") as f:
                f.write(line_csv)
            with open(self.jsonl_path, "a") as f:
                f.write(line_json)


def main():
    parser = build_common_parser()
    parser.add_argument(
        "-cat",
        "--category",
        dest="category",
        default="vbf",
        action="store",
        help="production mode category. Options: vbf or ggh",
    )
    parser.add_argument(
        "-r",
        "--region",
        dest="region",
        default="h-peak",
        action="store",
        help="region of the data. Options: h-peak, h-sidebands, signal",
    )
    # add dnn training arguments: epoch, batch size, etc.
    parser.add_argument(
        "--n-epochs",
        default=100,
        type=int,
        help="Number of epochs to train the DNN.",
    )
    parser.add_argument(
        "--batch-size",
        default=15536,
        type=int,
        help="Batch size for training the DNN.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use only 10% of the data for debugging.",
    )
    parser.add_argument(
        "--bo",
        action="store_true",
        help="Run GP-Bayesian optimization (Ax) before training"
    )
    parser.add_argument(
        "--bo-trials",
        type=int,
        default=60
    )
    parser.add_argument(
        "--bo-epochs",
        type=int,
        default=30
    )
    parser.add_argument(
        "--bo-fold",
        type=int,
        default=0,
        help="Which fold to use for Bayesian optimization"
    )
    parser.add_argument(
        "--n-procs",
        type=int,
        default=1,
        help="Number of parallel processes"
    )
    args = parser.parse_args()

    logger.setLevel(args.log_level)
    for handler in logger.handlers:
        handler.setLevel(args.log_level)
    save_path = f"dnn/trained_models/{args.label}/{args.year}_{args.region}_{args.category}{DIR_TAG}"
    if not os.path.exists(save_path):
        raise ValueError(f"Save path {save_path} does not exist. Please run dnn_preprocessor.py first.")

    try:
        meta = {
            "label": args.label,
            "year": args.year,
            "region": args.region,
            "category": args.category,
            "n_epochs": args.n_epochs,
            "batch_size_cli": args.batch_size,
            "bo": args.bo,
            "bo_trials": args.bo_trials,
            "bo_epochs": args.bo_epochs,
            "bo_fold": args.bo_fold,
        }
        Path(save_path).mkdir(parents=True, exist_ok=True)
        with open(Path(save_path) / "run_meta.yaml", "w") as f:
            yaml.safe_dump(meta, f)
    except Exception as _e:
        logger.warning(f"Could not write run_meta.yaml: {_e}")

    with open(f'{save_path}/training_features.pkl', 'rb') as f:
        training_features = pickle.load(f)

    # best hyperparameters from the hyperparameter optimization
    #  03 Sep 2025: Best hyperparameters from the full search (45 trials)
    best_hp = {
        "hidden": (1024, 1024, 409), #(128, 64, 32),
        "dropout": (0.0, 0.0, 0.0), #(0.2, 0.2, 0.2),
        "activation": "selu", #"tanh",
        "optimizer": "adamw", #"adam",
        "lr": 0.011339465927284355, #1e-3,
        "weight_decay": 1.9522171123020773e-06, #0.0,
        "batch_size": 50048, #args.batch_size,
        "loss_name": "bce",
    }
    # # Best hyperparameters (from 03 Sep 2025 training) except hidden layers. Here hidden layers are set to (128, 64, 32) similar as last training
    # best_hp = {
    #     "hidden": (128, 64, 32), #(128, 64, 32),
    #     "dropout": (0.0, 0.0, 0.0), #(0.2, 0.2, 0.2),
    #     "activation": "selu", #"tanh",
    #     "optimizer": "adamw", #"adam",
    #     "lr": 0.011339465927284355, #1e-3,
    #     "weight_decay": 1.9522171123020773e-06, #0.0,
    #     "batch_size": 2048, #args.batch_size,
    #     "loss_name": "bce",
    # }

    if args.bo:
        search_space = [
            {"name":"hidden0",      "type":"choice", "values":[64,128,256,512,1024],
            "value_type":"int", "is_ordered": True, "sort_values": True},

            {"name":"shrink1",      "type":"range",  "bounds":[0.4, 1.0],
            "value_type":"float"},

            {"name":"shrink2",      "type":"range",  "bounds":[0.4, 1.0],
            "value_type":"float"},

            {"name":"dropout",      "type":"range",  "bounds":[0.0, 0.5],
            "value_type":"float"},

            {"name":"activation",   "type":"choice", "values":["relu","gelu","selu","tanh"],
            "value_type":"str", "is_ordered": False, "sort_values": False},

            {"name":"optimizer",    "type":"choice", "values":["adam","adamw"],
            "value_type":"str", "is_ordered": True, "sort_values": False},

            {"name":"lr",           "type":"range",  "bounds":[1e-4,3e-2], "log_scale":True,
            "value_type":"float"},

            {"name":"weight_decay", "type":"range",  "bounds":[1e-7,3e-3], "log_scale":True,
            "value_type":"float"},

            {"name":"batch_size",   "type":"choice", "values":[512,1024,2048,4096,8192,15536,30000],
            "value_type":"int", "is_ordered": True, "sort_values": True},

            {"name":"loss_name",    "type":"choice", "values":["bce","focal"],
            "value_type":"str", "is_ordered": True, "sort_values": False},
        ]

        try:
            os.makedirs(Path(save_path) / "bo_logs", exist_ok=True)
            with open(Path(save_path) / "bo_logs" / "search_space.yaml", "w") as f:
                yaml.safe_dump({"parameters": search_space}, f)
        except Exception as _e:
            logger.warning(f"Could not write search_space.yaml: {_e}")

        bo_dir = os.path.join(save_path, "bo_logs")
        recorder = BOTrialRecorder(out_dir=bo_dir, objective_name="auc")

        _TRIAL_COUNTER = {"i": 0}  # simple in-process counter

        def _eval_logged(params):
            _TRIAL_COUNTER["i"] += 1
            trial_idx = _TRIAL_COUNTER["i"]
            t0 = _time()
            status = "ok"
            try:
                auc = bo_evaluate(
                    params,
                    save_path=save_path,
                    training_features=training_features,
                    bo_fold=int(args.bo_fold),
                    bo_epochs=int(args.bo_epochs),
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                auc = 0.0
                status = "oom"
            except Exception as e:
                logger.exception(f"[Ax] Trial {trial_idx} failed: {e}")
                auc = 0.0
                status = "error"
            dur = _time() - t0

            # persist to disk (CSV + JSONL) and to logger
            recorder.record(trial_idx, params, auc, dur, status=status)
            logger.info(
                f"[Ax] Trial {trial_idx} ({status}) AUC={auc:.6f} "
                f"params={json.dumps(params, separators=(',',':'))} "
                f"t={dur:.2f}s"
            )
            return auc

        # === Save BO logs & plots ===
        bo_dir = Path(save_path) / "bo_logs"
        best_params, values, experiment, model = optimize(
            parameters=search_space,
            evaluation_function=_eval_logged,  # <— use the logging wrapper
            total_trials=int(args.bo_trials),
            minimize=False,  # we maximize AUC
            objective_name="auc",
        )

        logger.info(f"[Ax] Best parameters: {best_params}")
        logger.info(f"[Ax] Values: {values}")

        bo_out = os.path.join(save_path, "ax_bo_artifacts")
        save_bo_artifacts(experiment, values, best_params, bo_out)

        # translate to training knobs
        h1 = int(best_params["hidden0"])
        h2 = max(8, int(h1 * float(best_params["shrink1"])))
        h3 = max(8, int(h2 * float(best_params["shrink2"])))
        best_hp.update(
            {
                "hidden": (h1, h2, h3),
                "dropout": (float(best_params["dropout"]),) * 3,
                "activation": best_params["activation"],
                "optimizer": best_params["optimizer"],
                "lr": float(best_params["lr"]),
                "weight_decay": float(best_params["weight_decay"]),
                "batch_size": int(best_params["batch_size"]),
                "loss_name": best_params["loss_name"],
            }
        )
        logger.info(f"[Ax] Best parameters: {best_params}")
        logger.info(f"[Ax] Best values: {values}")
        # logger.info(f"[Ax] Best AUC ~ {values[0]['auc']:.5f}")
        logger.info(f"[Ax] Best params: {best_hp}")

    nfolds = 4 #4

    # print best_hp
    logger.info(f"Using best hyperparameters: {best_hp}")

    # Parallelization list intitializtation
    model_l = []
    data_dict_l = []
    fold_l = []
    training_features_l = []
    save_path_l = []
    batch_size_l = []
    nepochs_l = []
    callback_l = []
    for i in range(nfolds):
        model = Net(
            n_feat=len(training_features),
            hidden=best_hp["hidden"],
            dropout=best_hp["dropout"],
            activation=best_hp["activation"],
        )

        df_train = pd.read_parquet(f"{save_path}/data_df_train_{i}.parquet") # these have been already scaled
        df_valid = pd.read_parquet(f"{save_path}/data_df_validation_{i}.parquet") # these have been already scaled
        df_eval = pd.read_parquet(f"{save_path}/data_df_evaluation_{i}.parquet") # these have been already scaled

        # use only 10% stats for debug
        if args.debug:
            df_train = df_train.sample(frac=0.1, random_state=42)
            df_valid = df_valid.sample(frac=0.1, random_state=42)
            df_eval = df_eval.sample(frac=0.1, random_state=42)

        training_features = prepare_features(df_train, training_features) # add variation to the name
        logger.info(f"fold {i} training features: {training_features}")
        logger.debug(f"df_train: {df_train}")
        data_dict = {
            "train": df_train,
            "validation": df_valid,
            "evaluation": df_eval,
        }
        nepochs = args.n_epochs
        batch_size = int(best_hp["batch_size"])
        # dnn_train(model, data_dict, i, training_features, batch_size, nepochs, save_path, TrainingLogger(log_interval=10))

        # collect the input parameters
        model_l.append(model)
        data_dict_l.append(data_dict)
        fold_l.append(i)
        training_features_l.append(training_features)
        save_path_l.append(save_path)
        batch_size_l.append(batch_size)
        nepochs_l.append(nepochs)
        callback_l.append(TrainingLogger(log_interval=10))

    lr_l = [best_hp["lr"]] * nfolds
    opt_name_l = [best_hp["optimizer"]] * nfolds
    wd_l = [best_hp["weight_decay"]] * nfolds
    loss_name_l = [best_hp["loss_name"]] * nfolds

    n_procs = args.n_procs
    logger.info(f"n_procs set to: {n_procs}")
    if n_procs is None or n_procs <= 1:
        logger.info("Running fold training serially (no ProcessPoolExecutor).")
        result_l = []
        for (
            model,
            data_dict,
            fold,
            trf,
            batch_size,
            nepochs,
            save_path_i,
            callback,
            lr,
            opt_name,
            wd,
            loss_name,
        ) in zip(
            model_l,
            data_dict_l,
            fold_l,
            training_features_l,
            batch_size_l,
            nepochs_l,
            save_path_l,
            callback_l,
            lr_l,
            opt_name_l,
            wd_l,
            loss_name_l,
        ):
            r = dnn_train(
                model,
                data_dict,
                fold,
                trf,
                batch_size,
                nepochs,
                save_path_i,
                callback,
                lr,
                opt_name,
                wd,
                loss_name,
            )
            result_l.append(r)
        logger.debug(f"result_l (serial): {result_l}")
        logger.info("done (serial)!")
    else:
        logger.info(f"Running fold training with ProcessPoolExecutor, n_procs={n_procs}")
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as executor:
            result_l = list(
                executor.map(
                    dnn_train,
                    model_l,
                    data_dict_l,
                    fold_l,
                    training_features_l,
                    batch_size_l,
                    nepochs_l,
                    save_path_l,
                    callback_l,
                    lr_l,
                    opt_name_l,
                    wd_l,
                    loss_name_l,
                )
            )
        logger.debug(f"result_l (parallel): {result_l}")
        logger.info("done (parallel)!")

    # After training all folds
    fold_dirs = [f"{save_path}/fold{i}" for i in range(nfolds)]
    cv_consistency_plots_ROOT(save_path, fold_dirs, nbins=30)

    logger.info("Success!")

if __name__ == '__main__':
    main()
