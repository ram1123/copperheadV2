

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
from coffea.ml_tools.torch_wrapper import torch_wrapper
import argparse
import pickle
import time

def fillEventNans(events, category="vbf"):
    """
    checked that this function is unnecssary for vbf category, but have it for robustness
    """
    if category == "vbf":
        for field in events.fields:
            if "phi" in field:
                events[field] = ak.fill_none(events[field], value=-10) # we're working on a DNN, so significant deviation may be warranted
            else: # for all other fields (this may need to be changed)
                events[field] = ak.fill_none(events[field], value=0)
    else:
        print("ERROR: unsupported category!")
        raise ValueError
    return events

def applyCatAndFeatFilter(events, region="h-peak", category="vbf"):
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
    # print(f"features: {features}")
    # events = ak.zip({field : events[field] for field in features}) 
    return events

class DNNWrapper(torch_wrapper):
    def _create_model(self):
        model = torch.jit.load(self.torch_jit)
        model.eval()
        return model
    def prepare_awkward(self, arr):
        # The input is any awkward array with matching dimension

        # Soln #1
        default_none_val = 0
        arr = ak.fill_none(arr, value=default_none_val) # apply "fill_none" to arr in order to remove "?" label of the awkward array


        # Soln #2
        # arr = ak.drop_none(arr)


        # Soln #3
        # arr = ak.to_packed(arr)

        # print(f"arr: {arr.compute()}")
        return [
            ak.values_astype(arr, "float32"), #only modification we do is is force float32
        ], {}


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

# class Net(nn.Module):
#     def __init__(self, input_shape):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(input_shape, 128)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.dropout1 = nn.Dropout(0.2)
#         self.fc2 = nn.Linear(128, 64)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.dropout2 = nn.Dropout(0.2)
#         self.fc3 = nn.Linear(64, 32)
#         self.bn3 = nn.BatchNorm1d(32)
#         self.dropout3 = nn.Dropout(0.2)
#         self.output = nn.Linear(32, 1)

#     def forward(self, features):
#         x = features
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = F.tanh(x)
#         x = self.dropout1(x)

#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = F.tanh(x)
#         x = self.dropout2(x)

#         x = self.fc3(x)
#         x = self.bn3(x)
#         x = F.tanh(x)
#         x = self.dropout3(x)

#         x = self.output(x)
#         output = F.sigmoid(x)
#         return output



# n_feat = 3
# # model = Net(n_feat)
# # model.eval()
# # input = torch.rand(100, n_feat)
# # torch.jit.trace(model, input).save("test_model.pt")


def getFoldFilter(events, fold_vals, nfolds):
    fold_filter = ak.zeros_like(events.event, dtype="bool")
    # print(f" eval_filter b4: {eval_filter.compute()}")
    for fold_value in fold_vals:
        fold_filter = fold_filter | ((events.event % nfolds) == fold_value)
    return fold_filter

parser = argparse.ArgumentParser()
parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="2018",
    action="store",
    help="string value of year we are calculating",
)
parser.add_argument(
    "-ml",
    "--model_label",
    dest="model_label",
    default="test",
    action="store",
    help="Unique run label (to create output path)",
)
parser.add_argument(
    "-rl",
    "--run_label",
    dest="run_label",
    default="test",
    action="store",
    help="Unique run label (to create output path)",
)
parser.add_argument(
    "-gate",
    "--use_gateway",
    dest="use_gateway",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, uses dask gateway client instead of local",
    )

args = parser.parse_args()
if __name__ == "__main__":  
    start_time = time.time()
    if args.use_gateway:
        from dask_gateway import Gateway
        gateway = Gateway(
            "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
            proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
        )
        cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
        client = gateway.connect(cluster_info.name).get_client()
        print("Gateway Client created")
    # # #-----------------------------------------------------------
    else:
        from distributed import LocalCluster, Client
        cluster = LocalCluster(processes=True)
        cluster.adapt(minimum=8, maximum=31) #min: 8 max: 32
        client = Client(cluster)
        print("Local scale Client created")

    base_path = f"/depot/cms/users/yun79/hmm/copperheadV1clean/{args.run_label}"

    hist_save_path = f"{base_path}/stage2_histograms/score_{args.model_label}/"
        

    # Preprocessing
    # stage1_path = "/depot/cms/users/yun79/hmm/copperheadV1clean/V2_Dec22_HEMVetoOnZptOn_RerecoBtagSF_XS_Rereco_BtagWPsFixed//stage1_output/2018/f1_0/data_C/0"
    stage1_path = f"{base_path}/stage1_output/{args.year}/f1_0/data_C/0"
    # stage1_path = "/depot/cms/users/yun79/hmm/copperheadV1clean/V2_Dec22_HEMVetoOnZptOn_RerecoBtagSF_XS_Rereco_BtagWPsFixed//stage1_output/2018/f1_0/data_*/0"
    events = dak.from_parquet(f"{stage1_path}/*.parquet")
    # events = dak.from_parquet(f"part000.parquet")
    
    # model_trained_path = f"MVA_training/VBF/dnn/trained_models/{args.model_label}"
    model_trained_path = f"/work/users/yun79/valerie/fork/copperheadV2/MVA_training/VBF/dnn/trained_models/{args.model_label}"
    
    with open(f'{model_trained_path}/training_features.pkl', 'rb') as f:
        training_features = pickle.load(f)
    print(f"training_features: {training_features}")
    variations = ["nominal"]
    for variation in variations:
        print(f"working on {variation}")
        training_features = prepare_features(events, training_features, variation=variation) # add variations where applicable
        print(f"new training_features: {training_features}")
        print(f"new training_features: {len(training_features)}")
    
        # features2load = ["event","wgt_nominal", "nBtagLoose", "jj_dEta", "jj_mass"]
        # features2load = prepare_features(events, features2load) # add variations where applicable
        # print(f"new features2load: {features2load}")
    
        # features2load = list(set(features2load + training_features))
        # print(f"final features2load: {features2load}")
        # raise ValueError
        region = "h-peak"
        category = "vbf"
        events = applyCatAndFeatFilter(events, region=region, category=category)
        events = fillEventNans(events, category=category) # for vbf category, this may be unncessary
        
        
        
        
        
        nfolds = 4 #4 
    
        # dnn_score_l = []
    
        # events = dak.from_parquet(f"part000.parquet")
        # # events = events[:3]
        
        
        
        
        # # print(events.event.compute())
        # input_arr = ak.concatenate( # Fold 5 event-level variables into a singular array
        #     [
        #         events.dimuon_mass[:, np.newaxis],
        #         events.mu2_pt[:, np.newaxis],
        #         events.mu1_pt[:, np.newaxis],
        #     ],
        #     axis=1,
        # )
        # print(input_arr.compute())
        # dwrap = DNNWrapper("test_model.pt")
        # dnn_score = dwrap(input_arr)
        # print(dnn_score) # This is the lazy evaluated dask array! Use this directly for histogram filling
        # print(dnn_score.compute()) # Eagerly evaluated result
        # print("Success!")
    
        nan_val = -999.0
        
        input_arr_dict = { feat : nan_val*ak.ones_like(events.event) for feat in training_features}
        print(f" input_arr_dict b4: {input_arr_dict}")
        for fold in range(nfolds): 
            
            eval_folds = [(fold+f)%nfolds for f in [3]]
            print(f" eval_folds: {eval_folds}")
            # 
            # eval_filter = ak.zeros_like(events.event, dtype="bool")
            # # print(f" eval_filter b4: {eval_filter.compute()}")
            # for eval_fold in eval_folds:
            #     eval_filter = eval_filter | ((events.event % nfolds) == eval_fold)
            eval_filter = getFoldFilter(events, eval_folds, nfolds)
    
    
            
            # print(f" eval_filter after: {eval_filter.compute()}")
            # print(f" events.event: {events.event.compute()}")
            # print(f" events.event% nfolds: {events.event.compute()% nfolds}")
            
            
            for feat in training_features:
                input_arr_fold = input_arr_dict[feat] 
                input_arr_fold = ak.where(eval_filter, events[feat], input_arr_fold)
                input_arr_dict[feat] = input_arr_fold
    
            # print(f" input_arr_dict after: {input_arr_dict}")
            
        # # debug:
        # for feat in training_features:
        #     input_arr_total = input_arr_dict[feat] 
        #     print(f"{feat} input_arr_total : {input_arr_total.compute()}")
        #     # check if we missed any nan_values
        #     any_nan = ak.any(input_arr_total ==nan_val)
        #     print(f"{feat} any_nan: {any_nan.compute()}")
        #     # merge the fold values
        #     # raise ValueError
        #     # for feat in input_arr_dict.keys():
        #         # input_arr_dict[feat] = ak.concatenate(input_arr_dict[feat], axis=0) # maybe compute individually for each fold?
    
        # ---------------------------------------------------
        # Now evaluate DNN score
        # ---------------------------------------------------
        input_arr = ak.concatenate(
            [input_arr_dict[feat][:, np.newaxis] for feat in training_features], # np.newaxis is added so that we can concat on axis=1
            axis=1
        )
        dnn_score = nan_val*ak.ones_like(events.event)
        for fold in range(nfolds): 
            eval_folds = [(fold+f)%nfolds for f in [3]]
            eval_filter = getFoldFilter(events, eval_folds, nfolds)
            model_load_path = f"{model_trained_path}/fold{fold}/best_model_torchJit_ver.pt"
            dnnWrap = DNNWrapper(model_load_path)
            dnn_score_fold = dnnWrap(input_arr)
            dnn_score = ak.where(eval_filter, dnn_score, dnn_score_fold)
            # print(f"{fold} fold dnn_score: {dnn_score.compute()}")
    
        # # debug:
        # any_nan = ak.any(dnn_score ==nan_val)
        # print(f"dnn_score any_nan: {any_nan.compute()}")
        # raise ValueError
    
            
        # ---------------------------------------------------
        # Now onto converting DNN score as histograms
        # ---------------------------------------------------
    
            
    
        regions = ["h-peak", "h-sidebands"]
        channels = ["vbf"]
        score_hist = (
                hda.Hist.new.StrCat(regions, name="region")
                .StrCat(channels, name="channel")
                .StrCat(["value", "sumw2"], name="val_sumw2")
        )
        bins = np.linspace(0, 1, num=50)
        score_hist = score_hist.Var(bins, name="dnn_score")
        
        score_hist = score_hist.Double()
        to_fill = {
            "region" : "h-peak",
            "channel" : "vbf",
            "val_sumw2" : "value",
            "dnn_score" : ak.flatten(dnn_score)
            
        }
        
        score_hist.fill(**to_fill)
        print("score_hist is filled!")
        score_hist = score_hist.compute()
        # ---------------------------------------------------
        # Save Hist 
        # ---------------------------------------------------
        
        
        # ---------------------------------------------------
        # Plot Hist for debugging
        # ---------------------------------------------------
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
    print("Success!")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time : {execution_time:.4f} seconds")