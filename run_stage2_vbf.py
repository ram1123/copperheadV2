

from hist import Hist
import dask
import awkward as ak
import hist.dask as hda
from coffea import processor
from coffea.nanoevents.methods import candidate
from coffea.dataset_tools import (
    apply_to_fileset,
    max_chunks,
    preprocess,
)
from distributed import Client
import dask_awkward as dak
import numpy as np
from coffea.nanoevents import NanoEventsFactory
from coffea.nanoevents.schemas import PFNanoAODSchema
import awkward as ak
import dask_awkward as dak
import numpy as np

#understand coffea pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_shape):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)
        self.output = nn.Linear(32, 1)

    def forward(self, features):
        x = features
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

common_path = "/depot/cms/users/yun79/hmm/copperheadV1clean/V2_Dec22_HEMVetoOnZptOn_RerecoBtagSF_XS_Rereco_BtagWPsFixed//stage1_output/2018/f1_0/data_A/0/"
data1 = dak.from_parquet(f"{common_path}/part000.parquet")
events = data1[:3]

n_feat = 3
model = Net(n_feat)
model.eval()
input = torch.rand(100, n_feat)
torch.jit.trace(model, input).save("test_model.pt")





from coffea.ml_tools.torch_wrapper import torch_wrapper

class DNNWrapper(torch_wrapper):
    def _create_model(self):
        model = torch.jit.load(self.torch_jit)
        model.eval()
        return model
    def prepare_awkward(self, arr):
        # The input is any awkward array with matching dimension
        # apply "fill_none" to arr in order to remove "?" label of the awkward array
        default_none_val = 0
        arr = ak.fill_none(arr, value=default_none_val)
        # arr = ak.drop_none(arr)
        arr = ak.to_packed(arr)


        return [
            ak.values_astype(arr, "float32"), #only modification we do is is force float32
        ], {}


# print(events.event.compute())
input_arr = ak.concatenate( # Fold 5 event-level variables into a singular array
    [
        events.dimuon_mass[:, np.newaxis],
        events.mu2_pt[:, np.newaxis],
        events.mu1_pt[:, np.newaxis],
    ],
    axis=1,
)
print(input_arr.compute())
dwrap = DNNWrapper("test_model.pt")
dnn_score = dwrap(input_arr)
print(dnn_score) # This is the lazy evaluated dask array! Use this directly for histogram filling
print(dnn_score.compute()) # Eagerly evaluated result


regions = ["h-peak", "h-sidebands"]
channels = ["vbf"]
score_hist = (
        hda.Hist.new.StrCat(regions, name="region")
        .StrCat(channels, name="channel")
        .StrCat(["value", "sumw2"], name="val_sumw2")
)
bins = np.linspace(110, 150, num=50)
score_hist = score_hist.Var(bins, name="dnn_score")

score_hist = score_hist.Double()
to_fill = {
    "region" : "h-peak",
    "channel" : "vbf",
    "val_sumw2" : "value",
    "dnn_score" : dnn_score
    
}

score_hist.fill(**to_fill)


import matplotlib.pyplot as plt

project_dict = {
    "region" : "h-peak",
    "channel" : "vbf",
    "val_sumw2" : "value",
}

fig, ax = plt.subplots()
score_hist[project_dict].project("dnn_score").plot1d(ax=ax)
# ax.set_xscale("log")
ax.legend(title="DNN score")
plt.savefig("test.png")