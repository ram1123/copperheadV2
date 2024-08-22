import numpy as np
import pickle
import awkward as ak
import dask_awkward as dak
from distributed import Client
from omegaconf import OmegaConf

from typing import Tuple, List, Dict
import ROOT as rt
import glob, os

from src.lib.MVA_functions import prepare_features, evaluate_bdt, evaluate_dnn
import argparse
import time
import sys, inspect
import configs.categories.category_cuts as category_cuts

def categoryWrapper(name: str, events) -> ak.Array:
    found_cut = False
    for class_name, obj in inspect.getmembers(category_cuts):
        if ("__" not in class_name) and ("Custom" in class_name):  # only look at classes I wrote
            if name == obj.name:
                # print(f"found category cut for {name}!")
                return obj.filterCategory(events) # this is a 1-D boolean awkward array with length of events
    if not found_cut:
        print(f"ERROR: given category name {name} for categoryWrapper is not supported!")
        raise ValueError

def categoryWrapperLoop(names: List[str], events) -> ak.Array:
    if len(names) == 0:
        print(f"ERROR: given names in categoryWrapperLoop is empty!")
        raise ValueError
    bool_arrs = [categoryWrapper(name, events) for name in names]
    out_arr = bool_arrs[0]
    if len(bool_arrs) > 1:
        for ix in range(1, len(bool_arrs)):
            out_arr = out_arr & bool_arrs[ix]
    return out_arr

def getCategoryCutNames(category:str) -> List[str]:
    """
    simple wrapper function that loads configs/categories/categories.yml
    and extracts the relevelt list of strings representation of cuts for
    the given category
    """
    category = category.lower() # force all lower character for simplicity
    category_config = OmegaConf.load("configs/categories/categories.yml")
    out_list = category_config["baseline"]
    out_list += category_config[category]
    # print(f"getCategoryCutNames out_list: {out_list}")
    return out_list
        
def process4gghCategory(events: ak.Record) -> ak.Record:
    """
    Takes the given stage1 output, runs MVA, and returns a new 
    ak.Record with MVA score + relevant info from stage1 output
    for ggH category

    Params
    ------------------------------------------------------------
    events: ak.Record of stage1 output
    """
    # load and obtain MVA outputs
    events["dimuon_dEta"] = np.abs(events.mu1_pt - events.mu2_pt)
    events["dimuon_pt_log"] = np.log(events.dimuon_pt)
    events["jj_mass_log"] = np.log(events.jj_mass)
    events["ll_zstar_log"] = np.log(events.ll_zstar)
    events["mu1_pt_over_mass"] = events.mu1_pt / events.dimuon_mass
    events["mu2_pt_over_mass"] = events.mu2_pt / events.dimuon_mass

    train_feat_dict = {
        "BDTperyear_2018" : [
            'dimuon_cos_theta_cs', 'dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR', 'dimuon_eta', 'dimuon_phi', 'dimuon_phi_cs', 'dimuon_pt', 
            'dimuon_pt_log', 'jet1_eta', 'jet1_phi', 'jet1_pt', 'jet1_qgl', 'jet2_eta', 'jet2_phi', 
            'jet2_pt', 'jet2_qgl', 'jj_dEta', 'jj_dPhi', 'jj_eta', 'jj_mass', 'jj_mass_log', 
            'jj_phi', 'jj_pt', 'll_zstar_log', 'mmj1_dEta', 'mmj1_dPhi', 
            'mmj_min_dEta', 'mmj_min_dPhi', 'mmjj_eta', 'mmjj_mass', 'mmjj_phi', 'mmjj_pt', 'mu1_eta', 'mu1_iso', 
            'mu1_phi', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_iso', 'mu2_phi', 'mu2_pt_over_mass', 'zeppenfeld'
        ],
        "phifixedBDT_2018" : [
                'dimuon_cos_theta_cs', 'dimuon_eta', 'dimuon_phi_cs', 'dimuon_pt', 'jet1_eta', 'jet1_pt', 'jet2_eta', 'jet2_pt', 'jj_dEta', 'jj_dPhi', 'jj_mass', 'mmj1_dEta', 'mmj1_dPhi',  'mmj_min_dEta', 'mmj_min_dPhi', 'mu1_eta', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_pt_over_mass', 'zeppenfeld' #, 'njets'
        ], # AN 19-124
        
    }
    
    model_name = "phifixedBDT_2018"
    # model_name = "BDTperyear_2018"

    training_features = train_feat_dict[model_name]
    print(f"len(training_features): {len(training_features)}")

    # load training features from the ak.Record
    for training_feature in training_features:
        if training_feature not in events.fields:
            print(f"mssing feature: {training_feature}")

    # ----------------------------------
    # do preprocessing
    # ----------------------------------
   
    # load fields to load
    fields2load = training_features + ["h_peak", "h_sidebands", "nBtagLoose", "nBtagMedium", "vbf_cut", "dimuon_mass", "wgt_nominal_total", "mmj2_dEta", "mmj2_dPhi"]
    # load data to memory using compute()
    events = ak.zip({
        field : events[field] for field in fields2load
    }).compute()

    # filter events for ggH category
    region = (events.h_peak != 0) | (events.h_sidebands != 0) # signal region cut
    category_str = "ggh"
    cut_names = getCategoryCutNames(category_str)
    gghCat_selection = (
        categoryWrapperLoop(cut_names, events)
        & region
    )
    
    events = events[gghCat_selection]
    
    # make sure to replace nans with -99.0 values   
    none_val = -99.0
    for field in events.fields:
        events[field] = ak.fill_none(events[field], value= none_val)
        inf_cond = (np.inf == events[field]) | (-np.inf == events[field]) 
        events[field] = ak.where(inf_cond, none_val, events[field])

    parameters = {
    "models_path" : "/depot/cms/hmm/vscheure/data/trained_models/"
    }
    processed_events = evaluate_bdt(events, "nominal", model_name, training_features, parameters) 

    # load BDT score edges for subcategory divison
    BDTedges_load_path = "./configs/MVA/ggH/BDT_edges.yaml"
    edges = OmegaConf.load(BDTedges_load_path)
    year = "2018"
    edges = np.array(edges[year])
    # edges = 1-edges
    print(f"subCat BDT edges: {edges}")

    # Calculate the subCategory index 
    # original start --------------------------------------------------------------
    # BDT_score = processed_events["BDT_score"]
    # print(f"BDT_score :{BDT_score}")
    # n_edges = len(edges)
    # BDT_score_repeat = ak.concatenate([BDT_score[:,np.newaxis] for i in range(n_edges)], axis=1)
    # # BDT_score_repeat
    # n_rows = len(BDT_score_repeat)
    # edges_repeat = np.repeat(edges[np.newaxis,:],n_rows,axis=0)
    # # edges_repeat.shape
    # edge_idx = ak.sum( (BDT_score_repeat >= edges_repeat), axis=1)
    # # edge_idx = ak.sum( (BDT_score_repeat <= edges_repeat), axis=1)
    # subCat_idx =  edge_idx - 1 # sub category index starts at zero
    # print(f"subCat_idx: {subCat_idx}")
    # processed_events["subCategory_idx"] = subCat_idx
    # original end -----------------------------------------------------------------

    # new start --------------------------------------------------------------
    BDT_score = processed_events["BDT_score"]
    print(f"BDT_score :{BDT_score}")
    print(f"ak.max(BDT_score) :{ak.max(BDT_score)}")
    print(f"ak.min(BDT_score) :{ak.min(BDT_score)}")
    subCat_idx = -1*ak.ones_like(BDT_score)
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        cut = (BDT_score > lo) & (BDT_score <= hi)
        # cut = (BDT_score <= lo) & (BDT_score > hi)
        subCat_idx = ak.where(cut, i, subCat_idx)
    print(f"subCat_idx: {subCat_idx}")
    # test if any remain has -1 value
    print(f"ak.sum(subCat_idx==-1): {ak.sum(subCat_idx==-1)}")
    processed_events["subCategory_idx"] = subCat_idx
    # new end -----------------------------------------------------------------

    # filter in only the variables you need to do stage3
    fields2save = [
        "dimuon_mass",
        "BDT_score",
        "subCategory_idx",
        "wgt_nominal_total",
        "h_peak",
        "h_sidebands",
    ]
    
    processed_events = ak.zip({
        field : processed_events[field] for field in fields2save
    })
    return processed_events

def process4vbfCategory(events: ak.Record, variation="nominal") -> ak.Record:
    """
    Takes the given stage1 output, runs MVA, and returns a new 
    ak.Record with MVA score + relevant info from stage1 output
    for VBF category

    Params
    ------------------------------------------------------------
    events: ak.Record of stage1 output
    """
    # load and obtain MVA outputs
    events["dimuon_dEta"] = np.abs(events.mu1_pt - events.mu2_pt)
    events["dimuon_pt_log"] = np.log(events.dimuon_pt)
    events["jj_mass_log"] = np.log(events.jj_mass)
    events["ll_zstar_log"] = np.log(events.ll_zstar)
    events["mu1_pt_over_mass"] = events.mu1_pt / events.dimuon_mass
    events["mu2_pt_over_mass"] = events.mu2_pt / events.dimuon_mass
    events["dimuon_ebe_mass_res_rel"] = events.dimuon_ebe_mass_res / events.dimuon_mass
    events["rpt"] = events.mmjj_pt / (events.dimuon_pt + events.jet1_pt + events.jet2_pt)# as of writing this code, rpt variable is calculated, but not saved during stage1
    events["year"] = ak.ones_like(events.mu1_pt)* 2018

    # original start --------------------------------------
    # training_features = [
    #     "dimuon_mass",
    #     "dimuon_pt",
    #     "dimuon_pt_log",
    #     "dimuon_eta",
    #     "dimuon_ebe_mass_res",
    #     "dimuon_ebe_mass_res_rel",
    #     "dimuon_cos_theta_cs",
    #     "dimuon_phi_cs",
    #     # "dimuon_pisa_mass_res",
    #     # "dimuon_pisa_mass_res_rel",
    #     # "dimuon_cos_theta_cs_pisa",
    #     # "dimuon_phi_cs_pisa",
    #     "jet1_pt",
    #     "jet1_eta",
    #     "jet1_phi",
    #     "jet1_qgl",
    #     "jet2_pt",
    #     "jet2_eta",
    #     "jet2_phi",
    #     "jet2_qgl",
    #     "jj_mass",
    #     "jj_mass_log",
    #     "jj_dEta",
    #     "rpt",
    #     "ll_zstar_log",
    #     "mmj_min_dEta",
    #     "nsoftjets5",
    #     "htsoft2",
    # ]
    # model_name = "PhiFixedVBF"
    # original end --------------------------------------

    # teset start ----------------------------------
    training_features = [
        "dimuon_mass",
        "dimuon_pt",
        "dimuon_pt_log",
        "dimuon_eta",
        "dimuon_ebe_mass_res",
        "dimuon_ebe_mass_res_rel",
        "dimuon_cos_theta_cs",
        "dimuon_phi_cs",
        "jet1_pt",
        "jet1_eta",
        "jet1_phi",
        "jet1_qgl",
        "jet2_pt",
        "jet2_eta",
        "jet2_phi",
        "jet2_qgl",
        "jj_mass",
        "jj_mass_log",
        "jj_dEta",
        "rpt",
        "ll_zstar_log",
        "mmj_min_dEta",
        "nsoftjets5",
        "htsoft2",
        "year",
    ]
    model_name = "pytorch_jun27"
    # test end --------------------------------------
    len(training_features)
    # load training features from the ak.Record
    for training_feature in training_features:
        if training_feature not in events.fields:
            print(f"mssing feature: {training_feature}")

    # ----------------------------------
    # do preprocessing
    # ----------------------------------
    training_features = prepare_features(events,training_features, variation=variation, add_year=False)
    fields2load = training_features + ["h_peak", "h_sidebands", "nBtagLoose", "nBtagMedium", "vbf_cut", "dimuon_mass", "wgt_nominal_total", "mmj2_dEta", "mmj2_dPhi"]
    # load data to memory using compute()
    events = ak.zip({
        field : events[field] for field in fields2load
    }).compute()

    # filter events for VBF category
    region = (events.h_peak != 0) | (events.h_sidebands != 0) # signal region cut
    category_str = "vbf"
    cut_names = getCategoryCutNames(category_str)
    vbfCat_selection = (
        categoryWrapperLoop(cut_names, events)
        & region
    )
    
    events = events[vbfCat_selection]
    
    # make sure to replace nans with -99.0 values   
    none_val = -99.0
    for field in events.fields:
        events[field] = ak.fill_none(events[field], value= none_val)
        inf_cond = (np.inf == events[field]) | (-np.inf == events[field]) 
        events[field] = ak.where(inf_cond, none_val, events[field])

    parameters = {
    # "models_path" : "/depot/cms/hmm/vscheure/data/trained_models/",
    "models_path" : "/depot/cms/hmm/copperhead/trained_models/",
    }
    processed_events = evaluate_dnn(events, "nominal", model_name, training_features, parameters) 
    # filter in only the variables you need to do stage3
    fields2save = [
        "dimuon_mass",
        "DNN_score",
        "DNN_score_sigmoid",
        "wgt_nominal_total",
        "h_peak",
        "h_sidebands",
    ]
    fields2save += training_features # this line is for debugging
    processed_events = ak.zip({
        field : processed_events[field] for field in fields2save
    })
    return processed_events


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-save",
    "--save_path",
    dest="save_path",
    default=None,
    action="store",
    help="save path to store stage1 output files",
    )
    parser.add_argument(
    "-load",
    "--load_path",
    dest="load_path",
    default=None,
    action="store",
    help="save path to store stage1 output files",
    )
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="2018",
    action="store",
    help="string value of year we are calculating",
    )
    parser.add_argument(
    "-cat",
    "--category",
    dest="category",
    default="ggH",
    action="store",
    help="string value production category we're working on",
    )
    parser.add_argument(
    "-samp",
    "--samples",
    dest="samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of samples to process for stage2. Current valid inputs are data, signal and DY",
    )
    start_time = time.time()
    client =  Client(n_workers=31,  threads_per_worker=1, processes=True, memory_limit='4 GiB') 
    args = parser.parse_args()
    # check for valid arguments
    if args.load_path == None:
        print("load path to load stage1 output is not specified!")
        raise ValueError
    if args.save_path == None:
        print("save path is not specified!")
        raise ValueError
    if len(args.samples) == 0:
        print("samples list is zero!")
        raise ValueError
    
    load_path = f"{args.load_path}/{args.year}/f1_0"
    print(f"load_path: {load_path}")
    # load_path = "/depot/cms/users/yun79/results/stage1/test_VBF-filter_JECon_07June2024/2018/f1_0"
    # full_load_path = load_path+f"/data_C/*/*.parquet"
    # full_load_path = load_path+f"/data_D/*/*.parquet"
    # full_load_path = load_path+f"/data_*/*/*.parquet"
    # full_load_path = load_path+f"/data_A/*/*.parquet"
    # full_load_path = load_path+f"/ggh_powheg/*/*.parquet"
    # full_load_path = load_path+f"/dy_M-100To200/*/*.parquet"
    category = args.category.lower()
    for sample in args.samples:
        if sample.lower() == "data":
            full_load_path = load_path+f"/data_*/*/*.parquet"
        # elif sample.lower() == "signal":
        #     if category == "ggh": # ggH
        #         full_load_path = load_path+f"/ggh_powheg/*/*.parquet"
        #     elif category == "vbf": # VBF
        #         full_load_path = load_path+f"/vbf_powheg/*/*.parquet"
        #     else:
        #         print("unsupported category")
        #         raise ValueError
        elif sample.lower() == "ggh":
            full_load_path = load_path+f"/ggh_powheg/*/*.parquet"
        elif sample.lower() == "vbf":
            full_load_path = load_path+f"/vbf_powheg/*/*.parquet"
        elif sample.lower() == "dy":
            full_load_path = load_path+f"/dy_M-100To200/*/*.parquet"
        elif sample.lower() == "ewk":
            full_load_path = load_path+f"/ewk_lljj_mll50_mjj120/*/*.parquet"
        else:
            print(f"unsupported sample!")
            raise ValueError
            
        print(f"full_load_path: {full_load_path}")
    
        
        events = dak.from_parquet(full_load_path)
        if category == "ggh":
            processed_events = process4gghCategory(events)      
        elif category == "vbf":
            processed_events = process4vbfCategory(events) 
        else: 
            print ("unsupported category given!")
            raise ValueError
        # define save path and save
        # save_path = "/work/users/yun79/stage2_output/ggH/test"
        save_path = f"{args.save_path}/{category}/{args.year}"
        print(f"save_path: {save_path}")
        # make save path if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if "data" in full_load_path.lower():
            save_filename = f"{save_path}/processed_events_data.parquet"  
        elif "ggh_powheg" in full_load_path: # signal
            save_filename = f"{save_path}/processed_events_sigMC_ggh.parquet" 
        elif "vbf_powheg" in full_load_path: # signal
            save_filename = f"{save_path}/processed_events_sigMC_vbf.parquet" 
        elif "dy_M-100To200" in full_load_path:
            save_filename = f"{save_path}/processed_events_bkgMC_dy.parquet" 
        elif "ewk" in full_load_path:
            save_filename = f"{save_path}/processed_events_bkgMC_ewk.parquet" 
        else:
            print ("unsupported sample given!")
            raise ValueError
        print(f"save_filename: {save_filename}")
    
        # delete the file if there's already same save_filename
        try:
            os.remove(save_filename)
        except:
            pass
        ak.to_parquet(processed_events, save_filename)

    end_time = time.time()
    print(f"stage2 done in {end_time-start_time} seconds")

    