import dask_awkward as dak
import numpy as np
import awkward as ak
import glob
import pandas as pd
import itertools
import torch
torch.multiprocessing.set_sharing_strategy('file_system') # reason: https://discuss.pytorch.org/t/training-crashes-due-to-insufficient-shared-memory-shm-nn-dataparallel/26396/44
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import argparse
from sklearn.metrics import roc_auc_score
# Add confusion matrix imports for fallback
import matplotlib.pyplot as plt
import mplhep as hep
from time import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
plt.style.use(hep.style.CMS)
import concurrent


import torch.profiler
from torch.cuda.amp import autocast, GradScaler

torch.set_float32_matmul_precision('high')


import logging
from modules.utils import logger

from dnn_helper import *


if not torch.cuda.is_available():
    logger.warning("CUDA is not available. Using CPU for training.")
    DEVICE = "cpu"

logger.info(f"using workers: {NWORKERS}")

def transformDnnScore(dnn_scores):
    return np.atanh(dnn_scores)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
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


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, mode="min",
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
                logger.info(f"[EarlyStopping] Epoch {epoch}: best {self.mode} improved from {self.best_score} to {current_score}")
            self.best_score = current_score
            self.counter = 0
            # self._save_model()
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"[EarlyStopping] Epoch {epoch}: no improvement ({self.counter}/{self.patience})")
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

def plot_loss_curves(train_losses, val_losses, save_path="loss_curves.pdf"):
    """
    Plot training and validation loss vs. epoch.

    Args:
        train_losses (list): Training loss per epoch.
        val_losses (list): Validation loss per epoch.
        save_path (str): Path to save the plot. If None, shows the plot.
    """
    epochs = list(range(len(train_losses)))
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    hep.cms.text("Private Work", loc=0)
    plt.grid(True)
    plt.savefig(save_path)
    logger.info(f"[plot_loss_curves] Saved loss curve to {save_path}")
    plt.close()

def plotPrecisionRecall(score_dict, plt_save_path):
    """
    Plot Precision-Recall curve for the given score dictionary.
    """
    from sklearn.metrics import precision_recall_curve
    fig, ax_main = plt.subplots()
    status = "Private Work 2018"
    CenterOfMass = "13"
    # hep.cms.label(data=True, loc=0, label=status, com=CenterOfMass, ax=ax_main)

    for stage, output_dict in score_dict.items():
        pred_total = output_dict["prediction"]
        label_total = output_dict["label"]
        wgt_total = output_dict["weight"]

        precision, recall, _ = precision_recall_curve(label_total, pred_total, sample_weight=wgt_total)
        ax_main.plot(recall, precision, label=f"{stage}")

    ax_main.set_xlabel('Recall')
    ax_main.set_ylabel('Precision')
    ax_main.set_title('Precision-Recall Curve')
    ax_main.legend()
    plt.savefig(plt_save_path)
    plt.clf()
    plt.close(fig)  # Close the figure to free memory

def plotConfusionMatrix(score_dict, plt_save_path):
    """
    Plot confusion matrix for the given score dictionary.
    """
    # Try importing seaborn, fallback to sklearn if unavailable
    try:
        import seaborn as sns
    except ImportError:
        from sklearn.metrics import ConfusionMatrixDisplay
        sns = None

    # status = "Private Work 2018"
    # CenterOfMass = "13"

    for stage, output_dict in score_dict.items():
        fig, ax_main = plt.subplots()
        # hep.cms.label(data=True, loc=0, ax=ax_main)
        pred_total = output_dict["prediction"]
        label_total = output_dict["label"]
        wgt_total = output_dict["weight"]

        # Convert predictions to binary (0 or 1)
        pred_binary = (pred_total > 0.5).astype(int)

        cm = confusion_matrix(label_total, pred_binary, sample_weight=wgt_total)
        if sns:
            sns.heatmap(cm, annot=True, cmap='Blues', ax=ax_main)
        else:
            ConfusionMatrixDisplay(cm).plot(ax=ax_main, cmap='Blues', values_format='d')
        ax_main.set_title(f'Confusion Matrix - {stage}')
        ax_main.set_xlabel('Predicted')
        ax_main.set_ylabel('True')

        plt.savefig(plt_save_path.replace('.pdf', f'_{stage}.pdf'))
        plt.clf()
        plt.close(fig)  # Close the figure to free memory

def plotFeatureImportance(model, features, plt_save_path):
    """
    Plot feature importance using SHAP values.
    """
    import shap

    # Assuming model is a PyTorch model and features is a list of feature names
    # Convert model to a format compatible with SHAP
    def model_predict(input_data):
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor).numpy()
        return output

    # Create a SHAP explainer
    explainer = shap.KernelExplainer(model_predict, np.zeros((1, len(features))))
    shap_values = explainer.shap_values(np.random.rand(100, len(features)))

    # Plot the feature importance
    shap.summary_plot(shap_values, features, plot_type="bar", show=False)
    plt.savefig(plt_save_path)
    plt.clf()
    plt.close()  # Close the figure to free memory

def prepare_features(events, features, variation="nominal"):
    features_var = []
    for trf in features:
        if "soft" in trf:
            variation_current = "nominal"
        else:
            variation_current = variation

        if f"{trf}_{variation_current}" in events.fields:
            features_var.append(f"{trf}_{variation_current}")
        elif trf in events.fields:
            features_var.append(trf)
        else:
            logger.warning(f"Variable {trf} not found in training dataframe!")
    return features_var

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_feat):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_feat, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)
        self.output = nn.Linear(32, 1)

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
            # eps = 1e-6
            # pred_total = np.clip(pred_total, -1 + eps, 1 - eps)
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
    TODO: add weights
    """
    ucsd_mode = "ucsd" in plt_save_path
    fig, ax_main = plt.subplots()
    status = "Private Work 2018"
    CenterOfMass = "13"
    hep.cms.label(data=True, loc=0, label=status, com=CenterOfMass, ax=ax_main)
    plt.yscale('log')
    plt.ylim((0.001, 1e3))
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
    plt.yscale("log")
    plt.ylim([0.0001, 1.0])

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
    auc_score = roc_auc_score(label_total, pred_total)
    return_dict = {
        "label" : label_total,
        "prediction" : pred_total,
        "total_loss" : total_loss,
        "batch_losses" : batch_losses,
    }
    model.train() # turn back to train mode
    return return_dict


def dnn_train(model, data_dict, nfolds, training_features, batch_size, nepochs, save_path, callback=None):
    if len(training_features) == 0:
        logger.error("ERROR: please define the training features the DNN will train on")
        raise ValueError

    fold_save_path = f"{save_path}/fold{nfolds}"
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
    # loss_fn = FocalLoss(alpha=1, gamma=2)
    # loss_fn = HingeLoss()

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
    early_stopping_callback = EarlyStopping(patience=5, delta=1e-3, mode="min", fold_save_path=f"{fold_save_path}", model=model, training_features=training_features, verbose=True)

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

            # Compute the loss and its gradients
            loss = loss_fn(pred, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            batch_loss = loss.item()
            epoch_loss += batch_loss
            batch_losses.append(batch_loss)
            # prof.step() # for profiler
        train_losses.append(epoch_loss)
        logger.debug(f"fold {nfolds} epoch {epoch}            train total loss: {epoch_loss}")
        logger.debug(f"fold {nfolds} epoch {epoch}    train average batch loss: {np.mean(batch_losses)}")
        validate_interval = 20
        if True:
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
            # auc_score = roc_auc_score(label_total, pred_total)
            val_losses.append(valid_loss)
            logger.debug(f"fold {nfolds} epoch {epoch} validation total loss: {valid_loss}")
            logger.debug(f"fold {nfolds} epoch {epoch} validation average batch loss: {np.mean(batch_losses)}")
            # logger.debug(f"fold {nfolds} epoch {epoch} validation AUC: {auc_score}")

            # call early stopping
            if early_stopping_callback and  early_stopping_callback.on_epoch_end(epoch, valid_loss):
                logger.warning(f"Early stopping at epoch {epoch} for fold {nfolds}")
                save_model_final(model, training_features, fold_save_path)
                break

    # Save final model state (in case early stopping did not trigger)
    save_model_final(model, training_features, fold_save_path)

    # Validation plots
    # ------------------------------------------------
    # 1. Plot the loss curves
    plot_loss_curves(train_losses, val_losses, save_path=f"{fold_save_path}/loss_curves_{nfolds}.pdf")
    # 2. Plot the ROC curve
    plotROC(score_dict, plt_save_path=f"{fold_save_path}/ROC_curve_{nfolds}.pdf")
    # 3. Plot the Sig/Bkg distributions
    bins = np.linspace(0, 1, 30)
    plotSigVsBkg(score_dict, bins, plt_save_path=f"{fold_save_path}/SigBkg_dist_{nfolds}.pdf", transformPrediction=False, normalize=True)
    plotSigVsBkg(score_dict, bins, plt_save_path=f"{fold_save_path}/SigBkg_dist_{nfolds}_log.pdf", transformPrediction=False, normalize=True, log_scale=True)
    # 4. Plot the Sig/Bkg distributions with transformed scores
    bins = np.array([
                    0,
                    0.07,
                    0.432,
                    0.71,
                    0.926,
                    1.114,
                    1.28,
                    1.428,
                    1.564,
                    1.686,
                    1.798,
                    1.9,
                    2.0,
                    2.8,
                ])
    plotSigVsBkg(score_dict, bins, plt_save_path=f"{fold_save_path}/SigBkg_dist_transformed_{nfolds}.pdf", transformPrediction=True, normalize=True)
    plotSigVsBkg(score_dict, bins, plt_save_path=f"{fold_save_path}/SigBkg_dist_transformed_{nfolds}_log.pdf", transformPrediction=True, normalize=True, log_scale=True)
    # 4. Precision vs Recall curve
    plotPrecisionRecall(score_dict, plt_save_path=f"{fold_save_path}/PrecisionRecall_curve_{nfolds}.pdf")
    # 5. Confusion matrix
    plotConfusionMatrix(score_dict, plt_save_path=f"{fold_save_path}/ConfusionMatrix_{nfolds}.pdf")
    # 6. Feature importance
    # plotFeatureImportance(model, training_features, plt_save_path=f"{fold_save_path}/FeatureImportance_{nfolds}.pdf")

    # calculate the scale, save it
    # save the resulting df for training
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
            logger.warning(f"Variable {trf} not found in training dataframe!")
    return features_var


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
        default=35536,
        type=int,
        help="Batch size for training the DNN.",
    )
    parser.add_argument(
        "--log-level",
        default=logging.INFO,
        type=lambda x: getattr(logging, x),
        help="Configure the logging level."
        )
    args = parser.parse_args()
    logger.setLevel(args.log_level)
    save_path = f"dnn/trained_models/{args.label}/{args.year}_{args.region}_{args.category}{DIR_TAG}"
    if not os.path.exists(save_path):
        raise ValueError(f"Save path {save_path} does not exist. Please run dnn_preprocessor.py first.")

    with open(f'{save_path}/training_features.pkl', 'rb') as f:
        training_features = pickle.load(f)

    nfolds = 4 #4
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

        training_features = prepare_features(df_train, training_features) # add variation to the name
        logger.info(f"fold {i} training features: {training_features}")
        data_dict = {
            "train": df_train,
            "validation": df_valid,
            "evaluation": df_eval,
        }
        nepochs = args.n_epochs
        batch_size = args.batch_size
        # dnn_train(model, data_dict, i, training_features, batch_size, nepochs, save_path)

        # collect the input parameters
        model_l.append(model)
        data_dict_l.append(data_dict)
        fold_l.append(i)
        training_features_l.append(training_features)
        save_path_l.append(save_path)
        batch_size_l.append(batch_size)
        nepochs_l.append(nepochs)
        callback_l.append(TrainingLogger(log_interval=10))

    with concurrent.futures.ProcessPoolExecutor(max_workers=nfolds) as executor:
        # Submit each file check to the executor
        result_l = list(executor.map(
            dnn_train,
            model_l,
            data_dict_l,
            fold_l,
            training_features_l,
            batch_size_l,
            nepochs_l,
            save_path_l,
            callback_l,
        ))
        print(f"result_l: {result_l}")
        print("Success!")


# Script entrypoint
if __name__ == '__main__':
    main()
