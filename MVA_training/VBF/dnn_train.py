import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import dask_awkward as dak
import numpy as np
import awkward as ak
import glob
import pandas as pd
import itertools
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import os
import argparse
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import mplhep as hep
from sklearn.metrics import confusion_matrix

from time import time


plt.style.use(hep.style.CMS)
import concurrent
# torch.multiprocessing.set_sharing_strategy('file_descriptor') # reason: https://discuss.pytorch.org/t/training-crashes-due-to-insufficient-shared-memory-shm-nn-dataparallel/26396/44

import logging
from modules.utils import logger
from modules import selection

from dnn_helper import *

from MVA_training.VBF.dnn_plotting import plot_loss_curves
from MVA_training.VBF.dnn_plotting import plotPrecisionRecall
from MVA_training.VBF.dnn_plotting import plotConfusionMatrix
from MVA_training.VBF.dnn_plotting import plot_auc_and_loss
from MVA_training.VBF.dnn_plotting import plot_significance
from MVA_training.VBF.dnn_plotting import plot_lr
from MVA_training.VBF.dnn_plotting import plot_overtraining_KS_ROOT
from MVA_training.VBF.dnn_plotting import plot_calibration_ROOT
from MVA_training.VBF.dnn_plotting import plot_threshold_scan_ROOT
from MVA_training.VBF.dnn_plotting import plot_score_feature_corr_ROOT_heatmap
from MVA_training.VBF.dnn_plotting import plot_score_feature_corr_ROOT_bar
from MVA_training.VBF.dnn_plotting import plot_score_shapes_and_roc_by_category_ROOT
from MVA_training.VBF.dnn_plotting import plot_cumulative_SSB_per_process_ROOT
from MVA_training.VBF.dnn_plotting import permutation_importance_auc
from MVA_training.VBF.dnn_plotting import plot_perm_importance_ROOT
from MVA_training.VBF.dnn_plotting import partial_dependence_curve
from MVA_training.VBF.dnn_plotting import plot_pdp_ROOT
from MVA_training.VBF.dnn_plotting import plot_weight_distribution_ROOT
from MVA_training.VBF.dnn_plotting import yield_table_after_cut
from MVA_training.VBF.dnn_plotting import _roc_weighted
from MVA_training.VBF.dnn_plotting import cv_consistency_plots_ROOT
from MVA_training.VBF.dnn_plotting import safe_weighted_auc

if not torch.cuda.is_available():
    logger.warning("CUDA is not available. Using CPU for training.")
    DEVICE = "cpu"

logger.info(f"using workers: {NWORKERS}")

def _safe_auc(y, p, w=None):
    try:
        return safe_weighted_auc(y, p, sample_weight=w)
    except ValueError:
        return float("nan")

def transformDnnScore(dnn_scores):
    s = np.clip(dnn_scores, 1e-6, 1-1e-6) # protection from atanh(0) or atanh(1 or -1) whose value is +/- inf
    return np.atanh(s)


training_logs = []
class TrainingLogger:
    # Reference: https://www.geeksforgeeks.org/deep-learning/monitoring-model-training-in-pytorch-with-callbacks-and-logging/
    def __init__(self, log_interval=10):
        self.log_interval = log_interval

    def on_epoch_begin(self, epoch):
        self.epoch_start_time = time()
        logger.debug(f"Epoch {epoch + 1} starting.")

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time() - self.epoch_start_time
        logger.info(f"Epoch {epoch + 1} finished in {elapsed_time:.2f} seconds.")
        logs['epoch_time'] = elapsed_time  # Add epoch time to logs
        training_logs.append(logs)  # Collect training logs

    def on_batch_end(self, batch, logs=None):
        if (batch + 1) % self.log_interval == 0:
            logger.info(f"Batch {batch + 1}: Loss = {logs['loss']:.4f}, Accuracy = {logs['accuracy']:.4f}")

    # Save the training_logs to a file
    def save_logs(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(training_logs, f)
        logger.info(f"Training logs saved to {filepath}")


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, mode="min", nfold=0,
                 fold_save_path=None, model=None, training_features=None, verbose=True):
        """
        Args:
            patience: How many epochs to wait after last improvement.
            delta: Minimum change to qualify as improvement.
            mode: 'min' to minimize (e.g. loss), 'max' to maximize (e.g. AUC).
            fold_save_path: Path to save the best model.
            model: PyTorch model.
            training_features: List of training features (used for JIT tracing).
            verbose: If True, log info.
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.nfold = nfold
        self.fold_save_path = fold_save_path
        self.model = model
        self.training_features = training_features
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode not in ['min', 'max']:
            raise ValueError("mode should be 'min' or 'max'")

        self.monitor_op = (lambda curr, best: curr < best - delta) if mode == "min" \
                          else (lambda curr, best: curr > best + delta)

    def on_epoch_end(self, epoch, current_score):
        if self.best_score is None or self.monitor_op(current_score, self.best_score):
            if self.verbose:
                logger.info(f"[EarlyStopping] Fold {self.nfold}, Epoch {epoch}: best {self.mode} improved from {self.best_score} to {current_score}")
            self.best_score = current_score
            self.counter = 0
            # self._save_model()
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"[EarlyStopping] Fold {self.nfold}, Epoch {epoch}: no improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                logger.warning("[EarlyStopping] Triggered.")
                self.early_stop = True
                self._save_model()
                return True
        return False

    def _save_model(self):
        if self.model is None or self.fold_save_path is None:
            return
        os.makedirs(self.fold_save_path, exist_ok=True)
        self.model.eval()
        torch.save(self.model.state_dict(), f"{self.fold_save_path}/best_model_weights.pt")
        if self.training_features:
            dummy_input = torch.rand(100, len(self.training_features))
            self.model.to("cpu")
            torch.jit.trace(self.model, dummy_input).save(f"{self.fold_save_path}/best_model_torchJit_ver.pt")
            self.model.to(DEVICE)
        self.model.train()
        logger.info(f"[EarlyStopping] Model saved to {self.fold_save_path}")

def save_model_final(model, training_features, fold_save_path):
    model.eval()
    torch.save(model.state_dict(), f"{fold_save_path}/final_model_weights.pt")
    if training_features:
        dummy_input = torch.rand(100, len(training_features))
        model.to("cpu")
        torch.jit.trace(model, dummy_input).save(f"{fold_save_path}/final_model_torchJit_ver.pt")
        model.to(DEVICE)
    model.train()
    logger.info(f"[FinalModel] Saved final model to {fold_save_path}")


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
        self.fc1 = nn.Linear(n_feat, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.dropout1 = nn.Dropout(d1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.dropout2 = nn.Dropout(d2)
        self.fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3)
        self.dropout3 = nn.Dropout(d3)
        self.output = nn.Linear(h3, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.tanh(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.tanh(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.tanh(x)
        x = self.dropout3(x)

        x = self.output(x)
        output = F.sigmoid(x)
        return output


# Custom Dataset class
class NumpyDataset(Dataset):
    def __init__(self, input_arr, label_arr):
        """
        Args:
            input_arr (numpy.ndarray): Input features array.
            label_arr (numpy.ndarray): Labels array.
        """
        self.input_arr = torch.tensor(input_arr, dtype=torch.float32)
        self.label_arr = torch.tensor(label_arr, dtype=torch.float32)

    def __len__(self):
        # Returns the total number of samples
        return len(self.input_arr)

    def __getitem__(self, idx):
        # Retrieve a sample and its corresponding label
        return self.input_arr[idx], self.label_arr[idx]


def plotSigVsBkg(score_dict, bins, plt_save_path, transformPrediction=False, normalize=True, log_scale=False):
    """
    TODO: add weights
    """
    fig, ax_main = plt.subplots()
    if log_scale:
        plt.yscale('log')
        plt.ylim((0.001, 1e3))
    else:
        plt.ylim((0.0, 5.0))
    for stage, output_dict in score_dict.items():
        pred_total = output_dict["prediction"]
        label_total = output_dict["label"]
        wgt_total = output_dict["weight"]
        if transformPrediction:
            pred_total = np.arctanh(pred_total)
            if log_scale:
                plt.ylim((0.001, 1e5))
            else:
                plt.ylim((0.0, 5.0))
        dnn_scores_signal = pred_total[label_total==1]  # Simulated DNN scores for signal
        dnn_scores_background = pred_total[label_total==0]   # Simulated DNN scores for background
        wgt_total_signal = wgt_total[label_total==1]
        wgt_total_background = wgt_total[label_total==0]
        # Histogram for signal, normalized to one
        hist_signal, bins_signal = np.histogram(dnn_scores_signal, bins=bins, weights=wgt_total_signal, density=normalize)
        # bin_centers_signal = 0.5 * (bins_signal[:-1] + bins_signal[1:])

        # Histogram for background, normalized to one
        hist_background, bins_background = np.histogram(dnn_scores_background, bins=bins, weights=wgt_total_background, density=normalize)
        # bin_centers_background = 0.5 * (bins_background[:-1] + bins_background[1:])
        hep.histplot(
            hist_signal,
            bins=bins,
            histtype='step',
            label=f"Signal - {stage}",
            ax=ax_main,
        )
        hep.histplot(
            hist_background,
            bins=bins,
            histtype='step',
            label=f"Bkg - {stage}",
            ax=ax_main,
        )

    x_label_addendum = "normalized" if normalize else ""
    if transformPrediction:
        plt.xlabel(f'arctanh Score {x_label_addendum}')
    else:
        plt.xlabel(f'DNN Score {x_label_addendum}')
    plt.ylabel('Events')
    # plt.title('normalized DNN Score Distributions Sig vs Bkg')
    plt.legend()
    status = "Private Work 2018"
    CenterOfMass = "13"
    # hep.cms.label(data=True, loc=0, label=status, com=CenterOfMass, lumi=lumi, ax=ax_main)
    hep.cms.label(data=True, loc=0, label=status, com=CenterOfMass, ax=ax_main)
    plt.savefig(plt_save_path)
    plt.clf()
    plt.close(fig)  # Close the figure to free memory


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
    helper function running through the evaluation
    """
    model.eval()
    total_loss = 0
    batch_losses = []
    pred_l = []
    label_l = []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device).reshape((-1,1))
            pred = model(inputs)
            loss = loss_fn(pred, labels)
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_losses.append(batch_loss)
            pred_l.append(pred.cpu().numpy())
            label_l.append(labels.cpu().numpy())
            # x_l.append(inputs.cpu().numpy()) # sanity check

        pred_total = np.concatenate(pred_l, axis=0).flatten()
        label_total = np.concatenate(label_l, axis=0).flatten()
        # x_total = np.concatenate(x_l, axis=0) # sanity check

    # logger.info(f"pred_l: {pred_l}")
    # logger.info(f"label_l: {label_l}")
    auc_score = safe_weighted_auc(label_total, pred_total)
    return_dict = {
        "label" : label_total,
        "prediction" : pred_total,
        "total_loss" : total_loss,
        "batch_losses" : batch_losses,
    }
    model.train() # turn back to train mode
    return return_dict

def ValidationPlots(model, epoch, fold_idx, fold_save_path, df_valid, training_features, best_significance, score_dict, pred_total, label_total,
                    df_train, train_loop_dict, valid_loop_dict):
    # if ((epoch==0) or ((epoch % validate_interval) == (validate_interval-1))) or (epoch==nepochs-1):
    # if True: # (epoch==0) or ((epoch % validate_interval) == (validate_interval-1)):
    # plot ROC curve
    plt_save_path = f"{fold_save_path}/epoch{epoch}_ROC.png"
    plotROC(score_dict, plt_save_path)
    plt_save_path = f"{fold_save_path}/epoch{epoch}_ROC_ucsd.png" # plot with sig eff and bkg eff in AN-19-124
    plotROC(score_dict, plt_save_path)

    # DNN Score plot with uniform bins
    bins = np.linspace(0, 1, 30)
    plt_save_path = f"{fold_save_path}/epoch{epoch}_DNN_combined_dist_bySigBkg.png"
    plotSigVsBkg(score_dict, bins, plt_save_path, transformPrediction=False)

    # DNN Score plot with last custom bins using which signal distribution is flat
    bins = selection.binning
    plt_save_path = f"{fold_save_path}/epoch{epoch}_DNN_combined_transformedDist_bySigBkg.png"
    plotSigVsBkg(score_dict, bins, plt_save_path, transformPrediction=True)
    # raise ValueError

    # # ------------------------------------------
    # # do the signal ratio plot
    # # ------------------------------------------

    # # Histogram for signal, normalized to one
    # wgt_signal = df_valid.wgt_nominal[label_total==1]
    # hist_signal, bins_signal = np.histogram(dnn_scores_signal, bins=bins, weights=wgt_signal)
    # bin_centers_signal = 0.5 * (bins_signal[:-1] + bins_signal[1:])

    # # Histogram for background, normalized to one
    # wgt_background = df_valid.wgt_nominal[label_total==0]
    # hist_background, bins_background = np.histogram(dnn_scores_background, bins=bins, weights=wgt_background)
    # bin_centers_background = 0.5 * (bins_background[:-1] + bins_background[1:])

    # # hist_signal, bins_signal = np.histogram(dnn_scores_signal, bins=bins)
    # # bin_centers_signal = 0.5 * (bins_signal[:-1] + bins_signal[1:])

    # # hist_background, bins_background = np.histogram(dnn_scores_background, bins=bins)
    # # bin_centers_background = 0.5 * (bins_background[:-1] + bins_background[1:])
    # sigBkg_ratio = np.zeros_like(hist_background)
    # nan_filter = hist_background !=0
    # sigBkg_ratio[nan_filter] = hist_signal[nan_filter] /hist_background[nan_filter]

    #  # Plotting
    # plt.figure(figsize=(10, 6))
    # plt.plot(bin_centers_signal, sigBkg_ratio, label='Sig/Bkg', drawstyle='steps-mid')
    # plt.xlabel('arctanh Score')
    # plt.ylabel('Sig/Bkg')
    # plt.title('Sig / Bkg DNN Score Distributions ')
    # plt.legend()
    # plt.savefig(f"{fold_save_path}/epoch{epoch}_DNN_validation_dist_sigBkgRatio.png")
    # plt.clf()

    # Create histograms and normalize them separated by process samples
    processes = ["dy", "top", "ewk", "vbf", "ggh"]

    # # sanity check that pred and labels have same row idx as df_valid
    # logger.info(f"x_total: {x_total[:10, :]}")
    # logger.info(f"df_valid: {df_valid.iloc[:10]}")

    for proc in processes:
        proc_filter = df_valid.process == proc
        # logger.info(f"proc_filter: {proc_filter}")
        dnn_scores = pred_total[proc_filter]
        dnn_scores = transformDnnScore(dnn_scores)
        wgt_proc = df_valid.wgt_nominal[proc_filter]
        hist_proc, bins_proc = np.histogram(dnn_scores, bins=bins, weights=wgt_proc)
        # logger.info(f"{proc} hist: {hist_proc}")
        bin_centers_proc = 0.5 * (bins_proc[:-1] + bins_proc[1:])
        plt.plot(bin_centers_proc, hist_proc, label=proc, drawstyle='steps-mid')
    plt.xlabel('arctanh Score')
    plt.ylabel('Density')
    plt.title('Normalized DNN Score Distributions by Sample')
    plt.legend()
    plt.savefig(f"{fold_save_path}/epoch{epoch}_DNN_validation_dist_byProcess.png")
    plt.clf()

    # Do the logscale plot
    fig, ax_main = plt.subplots()

    ax_main.set_yscale('log')
    ax_main.set_ylim(0.01, 1e9)

    # stack bkg

    bkg_processes = ["ewk", "top", "dy"] # smallest samples first
    bkg_hist_l = []
    for proc in bkg_processes:
        proc_filter = df_valid.process == proc
        dnn_scores = pred_total[proc_filter]
        dnn_scores = transformDnnScore(dnn_scores)
        wgt = df_valid.wgt_nominal[proc_filter]
        hist_proc, bins_proc = np.histogram(dnn_scores, bins=bins, weights=wgt)
        # logger.info(f"{proc} hist: {hist_proc}")
        bkg_hist_l.append(hist_proc)

    hep.histplot(
        bkg_hist_l,
        bins=bins,
        stack=True,
        histtype='fill',
        label=bkg_processes,
        sort='label_r',
        ax=ax_main,
    )

    # plot signal, no stack

    sig_processes = ["vbf", "ggh"]
    sig_hist_l = []

    for proc in sig_processes:
        proc_filter = df_valid.process == proc
        dnn_scores = pred_total[proc_filter]
        # logger.info(f"min dnn_scores: {np.min(dnn_scores)}")
        # logger.info(f"max dnn_scores: {np.max(dnn_scores)}")
        dnn_scores = transformDnnScore(dnn_scores)
        wgt = df_valid.wgt_nominal[proc_filter]
        hist_proc, bins_proc = np.histogram(dnn_scores, bins=bins, weights=wgt)
        # logger.info(f"{proc} hist: {hist_proc}")
        sig_hist_l.append(hist_proc)
        hep.histplot(
            hist_proc,
            bins=bins,
            histtype='step',
            label=proc,
            # color =  "black",
            ax=ax_main,
        )
    ax_main.set_xlabel('arctanh Score')
    ax_main.set_ylabel("Events")

    sig_hist_total = np.sum(sig_hist_l)
    bkg_hist_total = np.sum(bkg_hist_l)
    significance = calculateSignificance(sig_hist_total, bkg_hist_total)
    # if significance > best_significance:
    #     best_significance = significance
    #     # save state_dict
    #     model.eval()
    #     torch.save(model.state_dict(), f'{fold_save_path}/best_model_weights.pt')
    #     # save torch jit version for coffea torch_wrapper while you're at it
    #     dummy_input = torch.rand(100, len(training_features))
    #     # temporarily move model to cpu
    #     model.to("cpu")
    #     torch.jit.trace(model, dummy_input).save(f'{fold_save_path}/best_model_torchJit_ver.pt')
    #     model.to(DEVICE)
    #     model.train() # turn model back to train mode
    #     logger.info(f"new best significance for fold {i} is {best_significance} from {epoch} epoch")

    best_significance = significance
    # save state_dict
    # model.eval()
    # torch.save(model.state_dict(), f'{fold_save_path}/best_model_weights.pt')
    # save torch jit version for coffea torch_wrapper while you're at it
    # dummy_input = torch.rand(100, len(training_features))
    # temporarily move model to cpu
    # model.to("cpu")
    # torch.jit.trace(model, dummy_input).save(f'{fold_save_path}/best_model_torchJit_ver.pt')
    # model.to(DEVICE)
    # model.train() # turn model back to train mode
    logger.info(f"new best significance for fold {fold_idx} is {best_significance} from {epoch} epoch")

    # add significance to plot
    significance = str(significance)[:5] # round to 3 d.p.
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax_main.text(0.05, 0.95, f"Significance: {significance}", transform=ax_main.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    plt.title('DNN Score Distributions')
    plt.legend()
    plt.savefig(f"{fold_save_path}/epoch{epoch}_DNN_validation_stackedDist_byProcess.png")
    plt.clf()
    plt.close(fig)  # Close the figure to free memory

    # =========================
    # Extra validation (ROOT)
    # =========================
    # 1) Train vs Val score shapes with weighted KS p-values (sig/bkg)
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

    # 2) Calibration (reliability) curve + weighted Brier (RMSE)
    plot_calibration_ROOT(
        valid_loop_dict["prediction"],
        valid_loop_dict["label"],
        df_valid.wgt_nominal.values,
        save_path=f"{fold_save_path}/epoch{epoch}_Calibration.pdf",
        n_bins=15,
    )

    # 3) Threshold scan: TPR, FPR, Precision, and S/√B vs threshold
    plot_threshold_scan_ROOT(
        valid_loop_dict["prediction"],
        valid_loop_dict["label"],
        df_valid.wgt_nominal.values,
        save_path=f"{fold_save_path}/epoch{epoch}_ThresholdScan.pdf",
    )

    plot_weight_distribution_ROOT(df_valid.wgt_nominal.values, f"{fold_save_path}/epoch{epoch}")

    return best_significance


def dnn_train(model, data_dict, fold_idx, training_features, batch_size, nepochs, save_path, callback=None):
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

    loss_fn = torch.nn.BCELoss()

    # Iterating through the DataLoader
    #
    logger.info(f"input_arr_train shape: {input_arr_train.shape}")

    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset_train = NumpyDataset(input_arr_train, label_arr_train)
    dataloader_train_ordered = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=NWORKERS, pin_memory=PIN_MEMORY) # for plotting
    dataset_valid = NumpyDataset(input_arr_valid, label_arr_valid)
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=NWORKERS, pin_memory=PIN_MEMORY)
    dataset_eval = NumpyDataset(input_arr_eval, label_arr_eval)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=True, num_workers=NWORKERS, pin_memory=PIN_MEMORY)
    best_significance = 0
    early_stopping_callback = EarlyStopping(patience=5, delta=1e-3, mode="min", nfold=fold_idx, fold_save_path=f"{fold_save_path}", model=model, training_features=training_features, verbose=True)

    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "train_auc": [], "val_auc": [],
        "significance": [], "lr": []
    }
    for epoch in range(nepochs):
        model.train()
        callback.on_epoch_begin(epoch)
        # every epoch, reshuffle train data loader (could be unncessary)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=NWORKERS, pin_memory=PIN_MEMORY)

        epoch_loss = 0
        batch_losses = []
        for batch_idx, (inputs, labels) in enumerate(dataloader_train):
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
                pred_binary = (pred >= 0.5).float()
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
            "train" :  {
                "prediction": train_loop_dict["prediction"],
                "label": train_loop_dict["label"],
                "weight": df_train.wgt_nominal.values,
            },
            "valid+eval" : {
                "prediction": np.concatenate([valid_loop_dict["prediction"], eval_loop_dict["prediction"]], axis=0),
                "label":  np.concatenate([valid_loop_dict["label"], eval_loop_dict["label"]], axis=0),
                "weight": np.concatenate([df_valid.wgt_nominal.values, df_eval.wgt_nominal.values], axis=0),
            },
        }

        pred_total = valid_loop_dict["prediction"]
        label_total = valid_loop_dict["label"]
        valid_loss = valid_loop_dict["total_loss"]
        batch_losses = valid_loop_dict["batch_losses"]
        auc_score = safe_weighted_auc(label_total, pred_total)
        val_losses.append(valid_loss)
        logger.debug(f"fold {fold_idx} epoch {epoch} validation total loss: {valid_loss}")
        logger.debug(f"fold {fold_idx} epoch {epoch} validation average batch loss: {np.mean(batch_losses)}")
        logger.debug(f"fold {fold_idx} epoch {epoch} validation AUC: {auc_score}")

        # call early stopping
        if early_stopping_callback and  early_stopping_callback.on_epoch_end(epoch, valid_loss):
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
        train_auc = _safe_auc(train_loop_dict["label"], train_loop_dict["prediction"], w=None)
        val_auc   = _safe_auc(valid_loop_dict["label"], valid_loop_dict["prediction"], w=None)

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
    from MVA_training.VBF.dnn_plotting import _wstd, _wcov

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
    S <<B approximation of asimov significance as defined in eqn 4.1 of improvements paper
    """
    value = ( sig_hist / np.sqrt(bkg_hist) )**2
    value = np.sum(value)
    return np.sqrt(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--label",
        dest="label",
        default="test",
        action="store",
        help="Unique run label (to create output path)",
    )
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
    parser.add_argument(
        "-y",
        "--year",
        dest="year",
        default="2018",
        action="store",
        help="year of the data. Options: 2016, 2017, 2018",
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
        "--log-level",
        default=logging.INFO,
        type=lambda x: getattr(logging, x),
        help="Configure the logging level."
        )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use only 10% of the data for debugging.",
    )
    args = parser.parse_args()
    logger.setLevel(args.log_level)
    for handler in logger.handlers:
        handler.setLevel(args.log_level)
    save_path = f"dnn/trained_models/{args.label}/{args.year}_{args.region}_{args.category}{DIR_TAG}"
    if not os.path.exists(save_path):
        raise ValueError(f"Save path {save_path} does not exist. Please run dnn_preprocessor.py first.")

    with open(f'{save_path}/training_features.pkl', 'rb') as f:
        training_features = pickle.load(f)

    nfolds = 4 #4

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
        model = Net(len(training_features))

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
        batch_size = args.batch_size
        dnn_train(model, data_dict, i, training_features, batch_size, nepochs, save_path, TrainingLogger(log_interval=10))

        # collect the input parameters
        model_l.append(model)
        data_dict_l.append(data_dict)
        fold_l.append(i)
        training_features_l.append(training_features)
        save_path_l.append(save_path)
        batch_size_l.append(batch_size)
        nepochs_l.append(nepochs)
        callback_l.append(TrainingLogger(log_interval=10))

    # with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
    #     # Submit each file check to the executor
    #     result_l = list(executor.map(
    #         dnn_train,
    #         model_l,
    #         data_dict_l,
    #         fold_l,
    #         training_features_l,
    #         batch_size_l,
    #         nepochs_l,
    #         save_path_l,
    #         callback_l,
    #     ))
    #     logger.debug(f"result_l: {result_l}")
    #     logger.info("Success!")

    # After training all folds
    fold_dirs = [f"{save_path}/fold{i}" for i in range(nfolds)]
    cv_consistency_plots_ROOT(save_path, fold_dirs, nbins=30)

    logger.info("Success!")

if __name__ == '__main__':
    main()
