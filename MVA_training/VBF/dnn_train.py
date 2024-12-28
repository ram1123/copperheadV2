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
import os 
import argparse
from sklearn.metrics import roc_auc_score


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


def dnn_train(model, data_dict, batch_size=1024, nepochs=301):
    # nepochs = 50 # temporary overwrite
    # divide our data into 4 folds
    input_arr_train, label_arr_train = data_dict["train"]
    input_arr_valid, label_arr_valid = data_dict["validation"]
    
    loss_fn = torch.nn.BCELoss()
    # Iterating through the DataLoader
    # 
    
    device = "cuda"
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(nepochs):
        model.train()
        dataset_train = NumpyDataset(input_arr_train, label_arr_train)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dataset_valid = NumpyDataset(input_arr_valid, label_arr_valid)
        dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
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
            # print(f"running_loss: {running_loss}")
            # if i % 1000 == 999:
            #     last_loss = running_loss / 1000 # loss per batch
            #     print('  batch {} loss: {}'.format(i + 1, last_loss))
            #     tb_x = epoch_index * len(training_loader) + i + 1
            #     tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            #     running_loss = 0.
        print(f"fold {i} epoch {epoch} train total loss: {epoch_loss}")
        print(f"fold {i} epoch {epoch} train average batch loss: {np.mean(batch_losses)}")
        if (epoch % 5) == 0:
            model.eval()
            
            valid_loss = 0
            batch_losses = []
            pred_l = []
            label_l = []
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(dataloader_valid):
                    inputs = inputs.to(device)
                    labels = labels.to(device).reshape((-1,1))
                    pred = model(inputs)
                    loss = loss_fn(pred, labels)
                    batch_loss = loss.item()
                    valid_loss += batch_loss
                    batch_losses.append(batch_loss)
                    pred_l.append(pred.cpu().numpy())
                    label_l.append(labels.cpu().numpy())
    
                pred_l = np.concatenate(pred_l, axis=0).flatten()
                label_l = np.concatenate(label_l, axis=0).flatten()
                # print(f"pred_l: {pred_l}")
                # print(f"label_l: {label_l}")
                auc_score = roc_auc_score(label_l, pred_l)
            print(f"fold {i} epoch {epoch} validation total loss: {valid_loss}")
            print(f"fold {i} epoch {epoch} validation average batch loss: {np.mean(batch_losses)}")
            print(f"fold {i} epoch {epoch} validation AUC: {auc_score}")
            model.train() # turn model back to train mode
            
    
    
    # calculate the scale, save it
    # save the resulting df for training
    
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
    nfolds = 1 #4 
    model = Net(22)
    for i in range(nfolds):       
        input_arr_train = np.load(f"{save_path}/data_input_train_{i}.npy")
        label_arr_train = np.load(f"{save_path}/data_label_train_{i}.npy")
        input_arr_valid = np.load(f"{save_path}/data_input_validation_{i}.npy")
        label_arr_valid = np.load(f"{save_path}/data_label_validation_{i}.npy")
        data_dict = {
            "train": (input_arr_train, label_arr_train),
            "validation": (input_arr_valid, label_arr_valid)
        }
        
        dnn_train(model, data_dict)



