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


import torch
import torch.nn as nn
import torch.nn.functional as F
from coffea.ml_tools.torch_wrapper import torch_wrapper
import argparse
import pickle
import time
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import itertools

def get_variation(wgt_variation, sys_variation):
    if "nominal" in wgt_variation:
        if "nominal" in sys_variation:
            return "nominal"
        else:
            return sys_variation
    else:
        if "nominal" in sys_variation:
            return wgt_variation
        else:
            return None


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



def getFoldFilter(events, fold_vals, nfolds):
    fold_filter = ak.zeros_like(events.event, dtype="bool")
    # print(f" eval_filter b4: {eval_filter.compute()}")
    for fold_value in fold_vals:
        fold_filter = fold_filter | ((events.event % nfolds) == fold_value)
    return fold_filter


def getStage1Samples(stage1_path, data_samples=[], sig_samples=[], bkg_samples=[]):
    """
    sig samples: VBF, GGH
    bkg smaples: DY, TT, ST, VV, EWK
    """
    sample_dict = {}
    data_l = []
    return_filelist_dict = {}
    for data_letter in data_samples:
        data_l.append(f"data_{data_letter.upper()}")

    data_filelist = []
    for sample in data_l:
        data_filelist += glob.glob(f"{stage1_path}/{sample}/*/*.parquet")
        # return_filelist_dict[sample] = glob.glob(f"{stage1_path}/{sample}/*/*.parquet")
    
    return_filelist_dict["data"] = data_filelist # keep data as one sample list for speedup

    # sample_dict["data"] = data_filelist

    # ------------------------------------
    # work on sig MC
    # ------------------------------------
    sig_sample_dict = {
        "VBF" : [ 
            "vbf_powheg_dipole", # pythia dipole
            "vbf_powheg_herwig", # herwig
            "vbf_powhegPS", # pythia 8
        ],
        "GGH" : [
            "ggh_powhegPS"
        ]
    }

    sig_sample_l = []
    for sig_sample in sig_samples:
        sig_sample = sig_sample.upper()
        if sig_sample in sig_sample_dict.keys():
            sig_sample_l += sig_sample_dict[sig_sample]
    print(f"sig_sample_l: {sig_sample_l}")

    sig_filelist = []
    for sample in sig_sample_l:
        sample_filelist = glob.glob(f"{stage1_path}/{sample}/*/*.parquet")
        if len(sample_filelist) == 0: 
            print(f"No {sample} files were found!")
            continue
        return_filelist_dict[sample] = sample_filelist 

    # sample_dict["signal"] = sig_filelist

    
    # ------------------------------------
    # work on bkg MC
    # ------------------------------------
    bkg_sample_dict = {
        "DY" : [ 
            "dy_M-100To200",
            "dy_m105_160_vbf_amc", 
            "dy_M-50", 
        ],
        "TT" : [
            "ttjets_dl",
            "ttjets_sl",
        ],
        "ST" : [
            "st_tw_top",
            "st_tw_antitop",
        ],
        "EWK" : [
            "ewk_lljj_mll105_160_ptj0", # herwig
            "ewk_lljj_mll105_160_py_dipole", # pythia dipole
        ],
        "VV" : [
            "ww_2l2nu",
            "wz_3lnu",
            "wz_2l2q",
            "wz_1l1nu2q",
            "zz",
        ],
    }

    bkg_sample_l = []
    for bkg_sample in bkg_samples:
        bkg_sample = bkg_sample.upper()
        if bkg_sample in bkg_sample_dict.keys():
           bkg_sample_l += bkg_sample_dict[bkg_sample]
    print(f"bkg_sample_l: {bkg_sample_l}")

    bkg_filelist = []
    for sample in bkg_sample_l:
        sample_filelist = glob.glob(f"{stage1_path}/{sample}/*/*.parquet")
        if len(sample_filelist) == 0: 
            print(f"No {sample} files were found!")
            continue
        return_filelist_dict[sample] = sample_filelist 
    # sample_dict["background"] = bkg_filelist

    # print(f"sample_dict: {sample_dict}")
    return return_filelist_dict

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
parser.add_argument(
    "-data",
    "--data",
    dest="data_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of data samples represented by alphabetical letters A-H",
)
parser.add_argument(
    "-bkg",
    "--background",
    dest="bkg_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of bkg samples represented by shorthands: DY, TT, ST, DB (diboson), EWK",
)
parser.add_argument(
    "-sig",
    "--signal",
    dest="sig_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of sig samples represented by shorthands: ggH, VBF",
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

    
        

    bkg_samples = args.bkg_samples
    sig_samples = args.sig_samples
    data_samples = args.data_samples
    print(f"data_samples: {data_samples}")

    # stage1_path = f"{base_path}/stage1_output/{args.year}/f1_0/data_C/0"
    stage1_path = f"{base_path}/stage1_output/{args.year}/f1_0"
    # full_sample_dict = getStage1Samples(stage1_path, data_samples=data_samples, sig_samples=sig_samples, bkg_samples=bkg_samples)
    full_sample_dict = getStage1Samples(stage1_path, data_samples=data_samples, sig_samples=sig_samples, bkg_samples=bkg_samples)
    
    for sample_type, sample_l in tqdm(full_sample_dict.items(), desc="Processing Samples"):
        if len(sample_l) ==0:
            print(f"No files for {sample_type} is found! Skipping!")
            continue
            
        events_stage1 = dak.from_parquet(sample_l)

        # reparitition events if npartitions are too little to decrease memory usage (ie histograming vbf requires > 10 GB per worker otherwise) ----------------------
        # min_partition_size = 50
        # print(f"events_stage1.npartitions b4 repartition: {events_stage1.npartitions}")
        # if events_stage1.npartitions < min_partition_size :
        #     events_stage1 = events_stage1.repartition(npartitions=min_partition_size)
        #     print(f"events_stage1.npartitions after repartition: {events_stage1.npartitions}")
        # ----------------------
        
        # Preprocessing
        # stage1_path = "/depot/cms/users/yun79/hmm/copperheadV1clean/V2_Dec22_HEMVetoOnZptOn_RerecoBtagSF_XS_Rereco_BtagWPsFixed//stage1_output/2018/f1_0/data_C/0"
        
        # stage1_path = "/depot/cms/users/yun79/hmm/copperheadV1clean/V2_Dec22_HEMVetoOnZptOn_RerecoBtagSF_XS_Rereco_BtagWPsFixed//stage1_output/2018/f1_0/data_*/0"
        # events = dak.from_parquet(f"{stage1_path}/*.parquet")
        # events = dak.from_parquet(f"part000.parquet")
        
        # model_trained_path = f"MVA_training/VBF/dnn/trained_models/{args.model_label}"
        model_trained_path = f"/work/users/yun79/valerie/fork/copperheadV2/MVA_training/VBF/dnn/trained_models/{args.model_label}"
        
        with open(f'{model_trained_path}/training_features.pkl', 'rb') as f:
            training_features = pickle.load(f)
        print(f"training_features: {training_features}")
        print(f"len training_features: {len(training_features)}")
        
        # ------------------------------------------
        # Initialize sample histograme to save later
        # ------------------------------------------
        # variations = ["nominal"] # full list of possible variations to loop over
        wgt_variations = [w for w in events_stage1.fields if ("wgt_" in w)]
        wgt_variations = wgt_variations[:3] # for testing
        print(f"wgt_variations: {wgt_variations}")
        syst_variations = []
        syst_variations = ["nominal"] 
        # syst_variations = ['nominal', 'Absolute_up', 'Absolute_down', f'Absolute_{year}_up',
        variations = []
        for w in wgt_variations:
            for v in syst_variations:
                variation = get_variation(w, v)
                if variation:
                    variations.append(variation)
        print(f"variations: {variations}")

        regions = ["h-peak", "h-sidebands"] # full list of possible regions to loop over
        channels = ["vbf"] # full list of possible channels to loop over
        score_hist = (
                hda.Hist.new.StrCat(regions, name="region")
                .StrCat(channels, name="channel")
                .StrCat(["value", "sumw2"], name="val_sumw2")
        )
        # add axis for systematic variation
        score_hist = score_hist.StrCat(variations, name="variation")
        # add score category
        bins = np.linspace(0, 1, num=13) # TODO: update this
        score_name = f"score_{args.model_label}"
        score_hist = score_hist.Var(bins, name=score_name)
        
        
        score_hist = score_hist.Double()

        # loop over configurations and fill the histogram
        loop_args = {
            "region": regions,
            "wgt_variation": wgt_variations,
            "syst_variation": syst_variations,
            "channel": channels,
        }
        loop_args = [
            dict(zip(loop_args.keys(), values))
            for values in itertools.product(*loop_args.values())
        ]
        # print(f"loop_args: {loop_args}")
        
        
        # for variation in variations:
        #     print(f"working on {variation}")
        for loop_arg in loop_args:
            print(f"loop_arg: {loop_arg}")
            # raise ValueError
        
            # features2load = ["event","wgt_nominal", "nBtagLoose", "jj_dEta", "jj_mass"]
            # features2load = prepare_features(events, features2load) # add variations where applicable
            # print(f"new features2load: {features2load}")
        
            # features2load = list(set(features2load + training_features))
            # print(f"final features2load: {features2load}")
            # raise ValueError
            # region = "h-peak"
            # category = "vbf"
            region = loop_arg["region"]
            category = loop_arg["channel"]
            syst_variation = loop_arg["syst_variation"]
            wgt_variation = loop_arg["wgt_variation"]
            variation = get_variation(wgt_variation, syst_variation)
            if not variation:
                print(f"skipping variation {variation} from {wgt_variation} and {syst_variation}")
                continue
            
            events = applyCatAndFeatFilter(events_stage1, region=region, category=category)
            events = fillEventNans(events, category=category) # for vbf category, this may be unncessary

            training_features = prepare_features(events, training_features, variation=variation) # add variations where applicable
            print(f"new training_features: {training_features}")
            print(f"new training_features: {len(training_features)}")
            
            
            
            
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
            # print(f"dnn_score b4: {dnn_score.compute()}")
            for fold in range(nfolds): 
                eval_folds = [(fold+f)%nfolds for f in [3]]
                eval_filter = getFoldFilter(events, eval_folds, nfolds)
                model_load_path = f"{model_trained_path}/fold{fold}/best_model_torchJit_ver.pt"
                dnnWrap = DNNWrapper(model_load_path)
                dnn_score_fold = dnnWrap(input_arr)
                # print(f"{fold} fold dnn_score_fold b4 flatten: {dnn_score_fold.compute()}")
                dnn_score_fold = ak.flatten(dnn_score_fold, axis=1) # DNN outpout is 2 dimensional
                
                dnn_score = ak.where(eval_filter, dnn_score, dnn_score_fold)
                # print(f"{fold} fold dnn_score_fold after flatten: {dnn_score_fold.compute()}")
                # print(f"{fold} fold dnn_score: {dnn_score.compute()}")
                
            # print(f"dnn_score b4 after: {dnn_score.compute()}")
            # # debug:
            # any_nan = ak.any(dnn_score ==nan_val)
            # print(f"dnn_score any_nan: {any_nan.compute()}")
            # raise ValueError
        
                
            # ---------------------------------------------------
            # Now onto converting DNN score as histograms
            # ---------------------------------------------------
        
                
        
            
            to_fill = {
                "region" : "h-peak",
                "channel" : "vbf",
                "variation" : variation,
                score_name : dnn_score
                
            }
            # weight = events.wgt_nominal
            weight = events[wgt_variation]
                
            # print(f"weight len: {ak.num(weight, axis=0).compute()}")
            # print(f"ak.flatten(dnn_score) len: {ak.num(dnn_score, axis=0).compute()}")
            
            to_fill_value = to_fill.copy()
            to_fill_value["val_sumw2"] = "value"
            # to_fill_value["variation"] = variation
            score_hist.fill(**to_fill_value, weight=weight)
            # score_hist.fill(**to_fill_value)
    
            to_fill_sumw2 = to_fill.copy()
            to_fill_sumw2["val_sumw2"] = "sumw2"
            # to_fill_sumw2["variation"] = variation
            score_hist.fill(**to_fill_sumw2, weight=weight * weight)
            print(f"score_hist is filled for {sample_type}, {variation} variation!")


        # ---------------------------------------------------
        # done with variation loop, compute hist
        # ---------------------------------------------------
        score_hist = score_hist.compute()
        
        

        # ---------------------------------------------------
        # Save Hist 
        # ---------------------------------------------------
        hist_save_path = f"{base_path}/stage2_histograms/score_{args.model_label}/{args.year}/"

        if not os.path.exists(hist_save_path):
            os.makedirs(hist_save_path)
        
        with open(f"{hist_save_path}/{sample_type}_hist.pkl", "wb") as file:
            pickle.dump(score_hist, file)
            # print(f"{sample_type} histogram successfully!")
            print(f"{sample_type} histogram on {hist_save_path}!")

        
        # ---------------------------------------------------
        # Plot Hist for debugging
        # ---------------------------------------------------
        
        
        project_dict = {
            "region" : "h-peak",
            "channel" : "vbf",
            "val_sumw2" : "value",
            "variation" : "nominal",
        }
        
        fig, ax = plt.subplots()
        score_hist[project_dict].project(score_name).plot1d(ax=ax)
        # ax.set_xscale("log")
        ax.legend(title="DNN score")
        plt.savefig(f"{sample_type}_test.png")
    print("Success!")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time : {execution_time:.4f} seconds")