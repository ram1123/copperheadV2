import dask_awkward as dak
import numpy as np
import awkward as ak
import glob
import pandas as pd
import itertools
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pickle
import os 
import argparse
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
# hep.style.use("CMS")

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

# def getParquetFiles(path):
    # return glob.glob(path)

def fillEventNans(events):
    """
    checked that this function is unnecssary for vbf category, but have it for robustness
    """
    for field in events.fields:
        if "phi" in field:
            events[field] = ak.fill_none(events[field], value=-10) # we're working on a DNN, so significant deviation may be warranted
        else: # for all other fields (this may need to be changed)
            events[field] = ak.fill_none(events[field], value=0)
    return events

# def replaceSidebandMass(events):
#     for field in events.fields:
#         if "phi" in field:
#             events[field] = ak.fill_none(events[field], value=-1)
#         else: # for all other fields (this may need to be changed)
#             events[field] = ak.fill_none(events[field], value=0)
#     return events

def applyCatAndFeatFilter(events, features: list, region="h-peak", category="vbf"):
    """
    
    """
    # apply category filter
    dimuon_mass = events.dimuon_mass
    if region =="h-peak":
        region = (dimuon_mass > 115.03) & (dimuon_mass < 135.03)
    elif region =="h-sidebands":
        region = ((dimuon_mass > 110) & (dimuon_mass < 115.03)) | ((dimuon_mass > 135.03) & (dimuon_mass < 150))
    elif region =="signal":
        region = (dimuon_mass >= 110) & (dimuon_mass <= 150.0)
    
    if category.lower() == "vbf":
        btag_cut =ak.fill_none((events.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((events.nBtagMedium_nominal >= 1), value=False)
        cat_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35) 
        cat_cut = cat_cut & (~btag_cut) # btag cut is for VH and ttH categories
    elif category.lower()== "ggh":
        btag_cut =ak.fill_none((events.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((events.nBtagMedium_nominal >= 1), value=False)
        cat_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5)
        cat_cut = cat_cut & (~btag_cut) # btag cut is for VH and ttH categories
    else: # no category cut is applied
        cat_cut = ak.ones_like(dimuon_mass, dtype="bool")
        
    cat_cut = ak.fill_none(cat_cut, value=False)
    cat_filter = (
        cat_cut & 
        region 
    )
    events = events[cat_filter] # apply the category filter
    # print(f"events dimuon_mass: {events.dimuon_mass.compute()}")
    # apply the feature filter (so the ak zip only contains features we are interested)
    print(f"features: {features}")
    events = ak.zip({field : events[field] for field in features}) 
    return events


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
            print(f"Variable {trf} not found in training dataframe!")
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


def plotSigVsBkg(score_dict, bins, plt_save_path, transformPrediction=False, normalize=True):
    """
    TODO: add weights
    """
    fig, ax_main = plt.subplots()
    plt.yscale('log')
    plt.ylim((0.001, 1e3))
    for stage, output_dict in score_dict.items():
        pred_total = output_dict["prediction"]
        label_total = output_dict["label"]
        wgt_total = output_dict["weight"]
        if transformPrediction:
            pred_total = np.arctanh(pred_total)
            plt.ylim((0.001, 1e5))
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


def customROC_curve_AN(label, pred, weight):
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

        


        # effBkg = TN / (TN + FP) # Dmitry PhD thesis definition
        # effSig = FN / (FN + TP) # Dmitry PhD thesis definition
        effBkg = FP / (TN + FP) # AN-19-124 ggH Cat definition
        effSig = TP / (FN + TP) # AN-19-124 ggH Cat definition
        effBkg_total[ix] = effBkg
        effSig_total[ix] = effSig

        # print(f"ix: {ix}") 
        # print(f"threshold: {threshold}")
        # print(f"effBkg: {effBkg}")
        # print(f"effSig: {effSig}")
        
        
        # sanity check
        assert ((np.sum(positive_filter) + np.sum(negative_filter)) == len(pred))
        total_yield = FP + TP + FN + TN
        assert(np.isclose(total_yield, np.sum(weight)))
        # print(f"total_yield: {total_yield}")
        # print(f"np.sum(weight): {np.sum(weight)}")
    

    effBkg_total[np.isnan(effBkg_total)] = 1
    effSig_total[np.isnan(effSig_total)] = 1

    return (effBkg_total, effSig_total, thresholds)


def plotROC(score_dict, plt_save_path):
    """
    TODO: add weights
    """
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
        eff_bkg, eff_sig, thresholds = customROC_curve_AN(label_total, pred_total, wgt_total)
        plt.plot(eff_sig, eff_bkg, label=f"{stage}")

    plt.vlines(np.linspace(0,1,11), 0, 1, linestyle="dashed", color="grey")
    plt.hlines(np.logspace(-4,0,5), 0, 1, linestyle="dashed", color="grey")
    # plt.hlines(eff_bkg, 0, eff_sig, linestyle="dashed")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0001, 1.0])
    plt.xlabel('$\\epsilon_{sig}$')
    plt.ylabel('$\\epsilon_{bkg}$')
    plt.yscale("log")
    plt.ylim([0.0001, 1.0])
    
    plt.legend(loc="lower right")
    # plt.title(f'ROC curve for ggH BDT {year}')
    plt.savefig(plt_save_path)
    plt.clf()



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
    
    # print(f"pred_l: {pred_l}")
    # print(f"label_l: {label_l}")
    auc_score = roc_auc_score(label_total, pred_total)
    return_dict = {
        "label" : label_total,
        "prediction" : pred_total,
        "total_loss" : total_loss,
        "batch_losses" : batch_losses,
    }
    model.train() # turn back to train mode  
    return return_dict




def dnn_train(model, data_dict, training_features=[], batch_size=65536, nepochs=101, save_path=""):
    if save_path == "save_path":
        print("ERROR: please define the save path for the results")
        raise ValueError
    if len(training_features) == 0:
        print("ERROR: please define the training features the DNN will train on")
        raise ValueError
    
    # divide our data into 4 folds
    # input_arr_train, label_arr_train = data_dict["train"]
    # input_arr_valid, label_arr_valid = data_dict["validation"]
    # print(f"data_dict.keys(): {data_dict.keys()}")
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
    device = "cuda"
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset_train = NumpyDataset(input_arr_train, label_arr_train)
    dataloader_train_ordered = DataLoader(dataset_train, batch_size=batch_size, shuffle=False) # for plotting
    dataset_valid = NumpyDataset(input_arr_valid, label_arr_valid)
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
    dataset_eval = NumpyDataset(input_arr_eval, label_arr_eval)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False)
    best_significance = 0
    for epoch in range(nepochs):
        model.train()
        # every epoch, reshuffle train data loader (could be unncessary)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        
        epoch_loss = 0
        batch_losses = []
        for batch_idx, (inputs, labels) in enumerate(dataloader_train):
            inputs = inputs.to(device)
            labels = labels.to(device).reshape((-1,1))
            
    
            
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

        print(f"fold {i} epoch {epoch} train total loss: {epoch_loss}")
        print(f"fold {i} epoch {epoch} train average batch loss: {np.mean(batch_losses)}")
        validate_interval = 5
        if (epoch==0) or ((epoch % validate_interval) == (validate_interval-1)):            
            
            # x_l = [] # sanity check
            # with torch.no_grad():
            #     # valid_loss = 0
            #     # batch_losses = []
            #     # pred_l = []
            #     # label_l = []
            #     # for batch_idx, (inputs, labels) in enumerate(dataloader_valid):
            #     #     inputs = inputs.to(device)
            #     #     labels = labels.to(device).reshape((-1,1))
            #     #     pred = model(inputs)
            #     #     loss = loss_fn(pred, labels)
            #     #     batch_loss = loss.item()
            #     #     valid_loss += batch_loss
            #     #     batch_losses.append(batch_loss)
            #     #     pred_l.append(pred.cpu().numpy())
            #     #     label_l.append(labels.cpu().numpy())
            #     #     # x_l.append(inputs.cpu().numpy()) # sanity check
    
            #     # pred_total = np.concatenate(pred_l, axis=0).flatten()
            #     # label_total = np.concatenate(label_l, axis=0).flatten()
            #     # # x_total = np.concatenate(x_l, axis=0) # sanity check
                
            #     # # print(f"pred_l: {pred_l}")
            #     # # print(f"label_l: {label_l}")
            valid_loop_dict = dnnEvaluateLoop(model, dataloader_valid, loss_fn, device=device)
            train_loop_dict = dnnEvaluateLoop(model, dataloader_train_ordered, loss_fn, device=device)
            eval_loop_dict = dnnEvaluateLoop(model, dataloader_eval, loss_fn, device=device)
            score_dict = {
                "train" :  {
                    "prediction": train_loop_dict["prediction"],
                    "label": train_loop_dict["label"],
                    "weight": df_train.wgt_nominal.values,
                },
                # "validation" : {
                #     "prediction": valid_loop_dict["prediction"],
                #     "label": valid_loop_dict["label"],
                # },
                # "evaluation" :  {
                #     "prediction": eval_loop_dict["prediction"],
                #     "label": eval_loop_dict["label"],
                # },
                "valid+eval" : {
                    "prediction": np.concatenate([valid_loop_dict["prediction"], eval_loop_dict["prediction"]], axis=0),
                    "label":  np.concatenate([valid_loop_dict["label"], eval_loop_dict["label"]], axis=0),
                    "weight": np.concatenate([df_valid.wgt_nominal.values, df_eval.wgt_nominal.values], axis=0),
                },
            }

            # # debugging
            # train_label = train_loop_dict["label"]
            # random_idxs = random_indices = np.random.choice(len(train_label), size=100, replace=False)
            # train_label = train_label[random_idxs]
            # df_train_label = df_train.label.values[random_idxs]
            # # print(f"train_label: {train_label}")
            # # print(f"df_train.label: {df_train.label.values[random_idxs]}")
            # print(f"labels same: {np.all(df_train_label==train_label)}")
            
            pred_total = valid_loop_dict["prediction"]
            label_total = valid_loop_dict["label"]
            valid_loss = valid_loop_dict["total_loss"]
            batch_losses = valid_loop_dict["batch_losses"]
            auc_score = roc_auc_score(label_total, pred_total)
            print(f"fold {i} epoch {epoch} validation total loss: {valid_loss}")
            print(f"fold {i} epoch {epoch} validation average batch loss: {np.mean(batch_losses)}")
            print(f"fold {i} epoch {epoch} validation AUC: {auc_score}")

            # ------------------------------------------------
            # plot the score distributions
            # ------------------------------------------------
            fold_save_path = f"{save_path}/fold{i}"
            if not os.path.exists(fold_save_path):
                os.makedirs(fold_save_path)

            # # plot Sig vs Bkg from 0 to 1
            # # dnn_scores_signal = pred_total[label_total==1]  # Simulated DNN scores for signal
            # # dnn_scores_background = pred_total[label_total==0] # Simulated DNN scores for background
            # # bins = np.linspace(0, 1, 30) 
            # # plt_save_path = f"{fold_save_path}/epoch{epoch}_DNN_validation_dist_bySigBkg.png"
            # # plotSigVsBkg(dnn_scores_signal, dnn_scores_background, bins, plt_save_path)
            
            # # transform the score
            # pred_total = np.arctanh(pred_total)
            
            # dnn_scores_signal = pred_total[label_total==1]  # Simulated DNN scores for signal
            # dnn_scores_background = pred_total[label_total==0]   # Simulated DNN scores for background
            # # print(f"fold {i} epoch {epoch} validation pred_total: {pred_total.shape}")
            # # print(f"fold {i} epoch {epoch} validation label_total: {label_total.shape}")
            # # print(f"fold {i} epoch {epoch} validation dnn_scores_signal: {dnn_scores_signal}")
            # # print(f"fold {i} epoch {epoch} validation dnn_scores_background: {dnn_scores_background}")
            
            # # Create histograms and normalize them separated by signal and background
            # # bins = np.linspace(0, 1, 30)  # Adjust bin edges as needed
            # # bins = np.linspace(0, 2.8, 30)  # Adjust bin edges as needed


            # plot ROC curve 
            plt_save_path = f"{fold_save_path}/epoch{epoch}_ROC.png"
            plotROC(score_dict, plt_save_path)

            bins = np.linspace(0, 1, 30) 
            plt_save_path = f"{fold_save_path}/epoch{epoch}_DNN_combined_dist_bySigBkg.png"
            # plotSigVsBkg(dnn_scores_signal, dnn_scores_background, bins, plt_save_path)
            plotSigVsBkg(score_dict, bins, plt_save_path, transformPrediction=False)
            
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
            plt_save_path = f"{fold_save_path}/epoch{epoch}_DNN_combined_transformedDist_bySigBkg.png"
            
            # plotSigVsBkg(dnn_scores_signal, dnn_scores_background, bins, plt_save_path)
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
            # print(f"x_total: {x_total[:10, :]}")
            # print(f"df_valid: {df_valid.iloc[:10]}")
            
            for proc in processes:
                proc_filter = df_valid.process == proc
                # print(f"proc_filter: {proc_filter}")
                dnn_scores = pred_total[proc_filter]
                wgt_proc = df_valid.wgt_nominal[proc_filter]
                hist_proc, bins_proc = np.histogram(dnn_scores, bins=bins, weights=wgt_proc)
                # print(f"{proc} hist: {hist_proc}")
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
                wgt = df_valid.wgt_nominal[proc_filter]
                hist_proc, bins_proc = np.histogram(dnn_scores, bins=bins, weights=wgt)
                # print(f"{proc} hist: {hist_proc}")
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
                wgt = df_valid.wgt_nominal[proc_filter]
                hist_proc, bins_proc = np.histogram(dnn_scores, bins=bins, weights=wgt)
                # print(f"{proc} hist: {hist_proc}")
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
            if significance > best_significance:
                best_significance = significance
                # save state_dict
                model.eval()
                torch.save(model.state_dict(), f'{fold_save_path}/best_model_weights.pt')
                # save torch jit version for coffea torch_wrapper while you're at it
                dummy_input = torch.rand(100, len(training_features))
                # temporarily move model to cpu
                model.to("cpu")
                torch.jit.trace(model, dummy_input).save(f'{fold_save_path}/best_model_torchJit_ver.pt')
                model.to(device)
                model.train() # turn model back to train mode
                print(f"new best significance for fold {i} is {best_significance} from {epoch} epoch")

            # add significance to plot
            significance = str(significance)[:5] # round to 3 d.p.
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax_main.text(0.05, 0.95, f"Significance: {significance}", transform=ax_main.transAxes, fontsize=14, verticalalignment='top', bbox=props)

            plt.title('DNN Score Distributions')
            plt.legend()
            plt.savefig(f"{fold_save_path}/epoch{epoch}_DNN_validation_stackedDist_byProcess.png")
            plt.clf()

            
            
            
            
            

    
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
            print(f"Variable {trf} not found in training dataframe!")
    return features_var


def calculateSignificance(sig_hist, bkg_hist):
    """
    S <<B approximation of asimov significance as defined in eqn 4.1 of improvements paper
    """
    value = ( sig_hist / np.sqrt(bkg_hist) )**2
    value = np.sum(value)
    return np.sqrt(value)
   
parser = argparse.ArgumentParser()
parser.add_argument(
    "-l",
    "--label",
    dest="label",
    default="test",
    action="store",
    help="Unique run label (to create output path)",
)
args = parser.parse_args()
if __name__ == "__main__":  
    save_path = f"dnn/trained_models/{args.label}"
    # training_features = [
    #     'dimuon_mass', 'dimuon_pt', 'dimuon_pt_log', 'dimuon_eta', \
    #      'dimuon_cos_theta_cs', 'dimuon_phi_cs',
    #      'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_qgl', 'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_qgl',\
    #      'jj_mass', 'jj_mass_log', 'jj_dEta', 'rpt', 'll_zstar_log', 'mmj_min_dEta', 'nsoftjets5', 'htsoft2'
    # ]
    with open(f'{save_path}/training_features.pkl', 'rb') as f:
        training_features = pickle.load(f)
    
    nfolds = 4 #4 
    model = Net(22)
    for i in range(nfolds):       
        # input_arr_train = np.load(f"{save_path}/data_input_train_{i}.npy")
        # label_arr_train = np.load(f"{save_path}/data_label_train_{i}.npy")
        # input_arr_valid = np.load(f"{save_path}/data_input_validation_{i}.npy")
        # label_arr_valid = np.load(f"{save_path}/data_label_validation_{i}.npy")
        # data_dict = {
        #     "train": (input_arr_train, label_arr_train),
        #     "validation": (input_arr_valid, label_arr_valid)
        # }
        # dnn_train(model, data_dict, save_path=save_path)
        df_train = pd.read_parquet(f"{save_path}/data_df_train_{i}") # these have been already scaled
        df_valid = pd.read_parquet(f"{save_path}/data_df_validation_{i}") # these have been already scaled
        df_eval = pd.read_parquet(f"{save_path}/data_df_evaluation_{i}") # these have been already scaled

        training_features = prepare_features(df_train, training_features) # add variation to the name
        print(f"new training_features: {training_features}")
        data_dict = {
            "train": df_train,
            "validation": df_valid,
            "evaluation": df_eval,
        }
        nepochs = 100 # 100
        batch_size = 65536
        dnn_train(model, data_dict,training_features=training_features, save_path=save_path,batch_size=batch_size,nepochs=nepochs)



