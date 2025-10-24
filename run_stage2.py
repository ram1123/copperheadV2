import numpy as np
import pickle
import awkward as ak
import dask_awkward as dak
from distributed import Client
from omegaconf import OmegaConf

from typing import Tuple, List, Dict
# import ROOT as rt
import glob, os

from src.lib.MVA_functions import prepare_features, evaluate_bdt, evaluate_dnn
import argparse
import time
import sys, inspect
import configs.categories.category_cuts as category_cuts
import json


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

def renameFieldsToV2(events):
    V2_fields = [
        "jet1_pt",
        'jet1_eta', 
        'jet2_pt', 
        'mmj1_dEta', 
        'mmj1_dPhi',  
        'jj_dEta', 
        'jj_dPhi', 
        'jj_mass', 
        'zeppenfeld', 
        'mmj_min_dEta', 
        'mmj_min_dPhi', 
        'njets',
        "nBtagLoose",
        "nBtagMedium",
        "mmj2_dEta",
        "mmj2_dPhi",
        "dimuon_rapidity",
        "wgt_nominal_total",
        "zeppenfeld",
    ]
    for V2_field in V2_fields:
        if V2_field == "dimuon_rapidity":
            V1_field = "dimuon_eta"
        elif V2_field == "wgt_nominal_total":
            V1_field = "wgt_nominal"
        else:
            V1_field = V2_field+"_nominal"
        events[V2_field] = events[V1_field]

    # manuall add in region fields
    # V2_field = "h_peak"
    # events[V2_field] = events["region"] == "h-peak"
    # V2_field = "h_sidebands"
    # events[V2_field] = events["region"] == "h-sidebands"
    return events

def categoryWrapper(name: str, events) -> ak.Array:
    """
    wrapper function to take a string representation of cuts and applying the python implementation 
    saved in configs.categories.category_cuts
    """
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
    """
    Wrapper function that implements categoryWrapper in a loop
    """
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

def getDeltaPhi(phi1,phi2):
    """
    This is the Dmitry's old direct implementation od dPhi
    """
    dphi = abs(np.mod(phi1 - phi2 + np.pi, 2 * np.pi) - np.pi)
    return dphi
        
def process4gghCategory(events: ak.Record, year:str, model_name:str) -> ak.Record:
    """
    Takes the given stage1 output, runs MVA, and returns a new 
    ak.Record with MVA score + relevant info from stage1 output
    for ggH category

    Params
    ------------------------------------------------------------
    events: ak.Record of stage1 output
    """
    # load and obtain MVA outputs
    # events["dimuon_dEta"] = np.abs(events.mu1_pt - events.mu2_pt)
    # events["dimuon_pt_log"] = np.log(events.dimuon_pt)
    # events["jj_mass_log"] = np.log(events.jj_mass)

    # # recalculate BDT variables that you're not certain is up to date from stage 1
    # min_dEta_filter  = ak.fill_none((events.mmj1_dEta < events.mmj2_dEta), value=True)
    # events["mmj_min_dEta"]  = ak.where(
    #     min_dEta_filter,
    #     events.mmj1_dEta,
    #     events.mmj2_dEta,
    # )
    # min_dPhi_filter = ak.fill_none((events.mmj1_dPhi < events.mmj2_dPhi), value=True)
    # events["mmj_min_dPhi"] = ak.where(
    #     min_dPhi_filter,
    #     events.mmj1_dPhi,
    #     events.mmj2_dPhi,
    # )
    # events["jj_dPhi"] = getDeltaPhi(events.jet1_phi, events.jet2_phi)
    


    # merged 2016preVFP and 2016postVFP for BDT training
    if "2016" in year:
        year_param = "2016"
    else:
        year_param = year

        
    model_path = f"/work/users/yun79/Run2_MVA_trainer/output/bdt_{model_name}_{year_param}"
    training_feat_path = f"{model_path}/training_features.json"
    print(f"trainig_feat_path: {training_feat_path}")
    with open(training_feat_path, 'r') as file:
        training_features = json.load(file)

    # load training features from the ak.Record
    for training_feature in training_features:
        if training_feature not in events.fields:
            print(f"mssing feature: {training_feature}")

    # ----------------------------------
    # do preprocessing
    # ----------------------------------
   
    # load fields to load
    fields2load = ["nBtagLoose", "nBtagMedium", "dimuon_mass", "wgt_nominal", "mmj2_dEta", "mmj2_dPhi", "event", "jj_mass_nominal", "jj_dEta_nominal", "jet1_pt_nominal"]
    fields2load = prepare_features(events, fields2load) # add variation to the name
    fields2load = list(set(fields2load + training_features)) # remove redundant fields

    print(f"fields2load: {fields2load}")

    # load data to memory using compute()
    # original start -------------------------------
    events = ak.zip({
        field : events[field] for field in fields2load
    }).compute()
    # original end -------------------------------

    # filter events for ggH category
    # dimuon_mass = events.dimuon_mass
    # region = (dimuon_mass >= 110) & (dimuon_mass <= 150.0) # signal region
    category_str = "ggh"
    cut_names = getCategoryCutNames(category_str)
    gghCat_selection = (
        categoryWrapperLoop(cut_names, events)
        # & region
    )
    
    events = events[gghCat_selection]
    # print(f"events num: {ak.num(events, axis=0)}")
    # raise ValueError

    # make sure to replace nans with zeros,  unless it's delta phis, in which case it's -1, as specified in line 1117 of the AN
    for field in events.fields:
        if "dPhi" in field:
            none_val = -1.0
        else:
            none_val = 0.0
        events[field] = ak.fill_none(events[field], value=none_val)
    print(f"process4gghCategory year: {year}")
    if year == "2016_RERECO": # I didn't train a separate BDT for rereco eras
        year_param = "2016preVFP"
    elif "RERECO" in year: # ie 2017_RERECO
        year_param = year.replace("_RERECO", "")
    elif "2016" in year: # we merge 2016preVFP and 2016postVFP into one 2016 for BDT training
        year_param = "2016"
    else:
        year_param = year
    parameters = {
    # "models_path" : "/depot/cms/hmm/vscheure/data/trained_models/",
        # 
        # "models_path" : "/depot/cms/users/yun79/hmm/trained_MVAs/bdt_final_2018/",
        # "models_path" : "/depot/cms/users/yun79/hmm/trained_MVAs/bdt_WgtOff_includeQGL_2018/",
        # "models_path" : f"/depot/cms/users/yun79/hmm/trained_MVAs/bdt_{model_name}_{year_param}/",
        "models_path" : f"/depot/cms/users/yun79/hmm/trained_MVAs/bdt_{model_name}_{year_param}",
        # "models_path" : model_path,
        "year" : year_param,
    }
    print(f"parameters models path: {parameters['models_path']}")
    processed_events = evaluate_bdt(events, "nominal", model_name, training_features, parameters) 

    # load BDT score edges for subcategory divison
    BDTedges_load_path = "./configs/MVA/ggH/BDT_edges.yaml"
    edges = OmegaConf.load(BDTedges_load_path)
    # edges = np.array(edges[year_param])
    edges = np.array(edges[year])
    # edges = 1-edges
    print(f"subCat BDT edges: {edges}")

    BDT_score = processed_events["BDT_score"]
    subCat_idx = np.digitize(BDT_score, edges) -1 # digitize starts at one, not zero
    processed_events["subCategory_idx"] = subCat_idx

    
    # filter in only the variables you need to do stage3
    fields2save = [
        "dimuon_mass",
        "BDT_score", # eval fold
        # "BDT_score_val", # val fold
        # "BDT_score_train", # train fold
        "subCategory_idx", # eval fold
        # "subCategory_idx_val", # val fold
        "wgt_nominal",
        # "h_peak",
        # "h_sidebands",
        "event", # This is not strictly necessary
    ]
    
    processed_events = ak.zip({
        field : processed_events[field] for field in fields2save
    })
    return processed_events

# def process4vbfCategory(events: ak.Record, variation="nominal") -> ak.Record:
#     """
#     Takes the given stage1 output, runs MVA, and returns a new 
#     ak.Record with MVA score + relevant info from stage1 output
#     for VBF category

#     Params
#     ------------------------------------------------------------
#     events: ak.Record of stage1 output
#     """
#     # load and obtain MVA outputs
#     events["dimuon_dEta"] = np.abs(events.mu1_pt - events.mu2_pt)
#     events["dimuon_pt_log"] = np.log(events.dimuon_pt)
#     events["jj_mass_log"] = np.log(events.jj_mass)
#     events["mu1_pt_over_mass"] = events.mu1_pt / events.dimuon_mass
#     events["mu2_pt_over_mass"] = events.mu2_pt / events.dimuon_mass
#     events["dimuon_ebe_mass_res_rel"] = events.dimuon_ebe_mass_res / events.dimuon_mass
#     events["rpt"] = events.mmjj_pt / (events.dimuon_pt + events.jet1_pt + events.jet2_pt)# as of writing this code, rpt variable is calculated, but not saved during stage1
#     print("Warning find a way to fix the year thing")
#     raise ValueError
#     events["year"] = ak.ones_like(events.mu1_pt)* 2018

#     # original start --------------------------------------
#     # training_features = [
#     #     "dimuon_mass",
#     #     "dimuon_pt",
#     #     "dimuon_pt_log",
#     #     "dimuon_eta",
#     #     "dimuon_ebe_mass_res",
#     #     "dimuon_ebe_mass_res_rel",
#     #     "dimuon_cos_theta_cs",
#     #     "dimuon_phi_cs",
#     #     # "dimuon_pisa_mass_res",
#     #     # "dimuon_pisa_mass_res_rel",
#     #     # "dimuon_cos_theta_cs_pisa",
#     #     # "dimuon_phi_cs_pisa",
#     #     "jet1_pt",
#     #     "jet1_eta",
#     #     "jet1_phi",
#     #     "jet1_qgl",
#     #     "jet2_pt",
#     #     "jet2_eta",
#     #     "jet2_phi",
#     #     "jet2_qgl",
#     #     "jj_mass",
#     #     "jj_mass_log",
#     #     "jj_dEta",
#     #     "rpt",
#     #     "ll_zstar_log",
#     #     "mmj_min_dEta",
#     #     "nsoftjets5",
#     #     "htsoft2",
#     # ]
#     # model_name = "PhiFixedVBF"
#     # original end --------------------------------------

#     # teset start ----------------------------------
#     training_features = [
#         "dimuon_mass",
#         "dimuon_pt",
#         "dimuon_pt_log",
#         "dimuon_eta",
#         "dimuon_ebe_mass_res",
#         "dimuon_ebe_mass_res_rel",
#         "dimuon_cos_theta_cs",
#         "dimuon_phi_cs",
#         "jet1_pt",
#         "jet1_eta",
#         "jet1_phi",
#         "jet1_qgl",
#         "jet2_pt",
#         "jet2_eta",
#         "jet2_phi",
#         "jet2_qgl",
#         "jj_mass",
#         "jj_mass_log",
#         "jj_dEta",
#         "rpt",
#         "ll_zstar_log",
#         "mmj_min_dEta",
#         "nsoftjets5",
#         "htsoft2",
#         "year",
#     ]
#     model_name = "pytorch_jun27"
#     # test end --------------------------------------
#     len(training_features)
#     # load training features from the ak.Record
#     for training_feature in training_features:
#         if training_feature not in events.fields:
#             print(f"mssing feature: {training_feature}")

#     # ----------------------------------
#     # do preprocessing
#     # ----------------------------------
#     training_features = prepare_features(events,training_features, variation=variation, add_year=False)
#     # original start -----------------------------------------------------
#     fields2load = training_features + ["h_peak", "h_sidebands", "nBtagLoose", "nBtagMedium", "vbf_cut", "dimuon_mass", "wgt_nominal_total", "mmj2_dEta", "mmj2_dPhi"]
#     # original end -----------------------------------------------------

#     # temporary save everything start -----------------------------------------------
#     # fields2load = events.fields# temp overwrite to koeep everything
#     # temporary save everything end -----------------------------------------------

#     fields2load = list(set(fields2load)) # remove redundancies
#     # load data to memory using compute()
#     events = ak.zip({
#         field : events[field] for field in fields2load
#     }).compute()

#     # filter events for VBF category
#     region = (events.h_peak != 0) | (events.h_sidebands != 0) # signal region cut
#     category_str = "vbf"
#     cut_names = getCategoryCutNames(category_str)
#     vbfCat_selection = (
#         categoryWrapperLoop(cut_names, events)
#         & region
#     )
    
#     events = events[vbfCat_selection]
    
#     # make sure to replace nans with -99.0 values   
#     none_val = -99.0
#     for field in events.fields:
#         events[field] = ak.fill_none(events[field], value= none_val)

#     parameters = {
#     # "models_path" : "/depot/cms/hmm/vscheure/data/trained_models/",
#         "models_path" : "/depot/cms/hmm/copperhead/trained_models/",
        
#     }
#     processed_events = evaluate_dnn(events, "nominal", model_name, training_features, parameters) 
#     # original start -----------------------------------------------------
#     # filter in only the variables you need to do stage3
#     fields2save = [
#         "dimuon_mass",
#         "dimuon_pt",
#         "DNN_score",
#         "DNN_score_sigmoid",
#         "wgt_nominal_total",
#         "h_peak",
#         "h_sidebands",
#     ]
#     fields2save += training_features # this line is for debugging
#     # original end -----------------------------------------------------

#     # temporary save everything start -----------------------------------------------
#     # fields2save = fields2load
#     # temporary save everything end -----------------------------------------------
    
#     processed_events = ak.zip({
#         field : processed_events[field] for field in fields2save
#     })
#     return processed_events


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
    "-model",
    "--model_name",
    dest="model_name",
    default="",
    action="store",
    help="MVA model name to load",
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
    parser.add_argument(
    "-frac",
    "--fraction",
    dest="fraction",
    default=None,
    action="store",
    help="fraction value used in stage1. By default we assume it to be 1.0",
    )
    start_time = time.time()
    client =  Client(n_workers=20,  threads_per_worker=1, processes=True, memory_limit='4 GiB') 
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
    if args.fraction is None: # fraction == 1.0
        load_path = f"{args.load_path}/{args.year}/f1_0"
    else:
        frac_str = args.fraction.replace(".", "_")
        load_path = f"{args.load_path}/{args.year}/f{frac_str}"
        # load_path = f"{args.load_path}/{args.year}/"
    print(f"load_path: {load_path}")
    category = args.category.lower()

    print(f"args.samples: {args.samples}")
    for sample in args.samples:
        if sample.lower() == "data":
            full_load_path = load_path+f"/data_*/*/*.parquet" # original
            # altering to match copperheadV1's stasge1 output to work with copperheadV2
            # full_load_path = load_path+f"/data_*/*.parquet"
            # full_load_path = load_path+f"/data_B/*.parquet"
            full_load_path = glob.glob(full_load_path)
        elif sample.lower() == "ggh":
            full_load_path = load_path+f"/ggh_powhegPS/*/*.parquet"
        elif sample.lower() == "ggh_amcps":
            full_load_path = load_path+f"/ggh_amcPS/*/*.parquet"
        elif sample.lower() == "vbf":
            full_load_path = load_path+f"/vbf_powheg_dipole/*/*.parquet"
        elif sample.lower() == "dy":
            # full_load_path = load_path+f"/dy_M-100To200/*/*.parquet"
            full_load_path = load_path+f"/dy_*/*/*.parquet"
        elif sample.lower() == "ewk":
            full_load_path = load_path+f"/ewk_lljj_mll50_mjj120/*/*.parquet"
        elif sample.lower() == "tt":
            full_load_path = load_path+f"/ttjets*/*/*.parquet"
        elif sample.lower() == "st":
            full_load_path = load_path+f"/st_tw*/*/*.parquet"
        elif sample.lower() == "ww":
            full_load_path = load_path+f"/ww_*/*/*.parquet"
        elif sample.lower() == "wz":
            full_load_path = load_path+f"/wz_*/*/*.parquet"
        elif sample.lower() == "zz":
            full_load_path = load_path+f"/zz/*/*.parquet"
        else:
            print(f"unsupported sample!")
            raise ValueError
            
        # print(f"full_load_path: {full_load_path}")
        # if "data" in full_load_path:
        #     data_filelist.append(full_load_path)
        # elif ("ggh" in full_load_path) or ("vbf" in full_load_path):
        #     sig_MC_filelist.append(full_load_path)
        # else:
        #     bkg_MC_filelist.append(full_load_path)
        
        events = dak.from_parquet(full_load_path)
        
        # making so taht copperheadV1 results work start -------------------------------------------
        # fields2load = [
        #     "jet1_pt_nominal",
        #     'jet1_eta_nominal', 
        #     'jet2_pt_nominal', 
        #     'mmj1_dEta_nominal', 
        #     'mmj1_dPhi_nominal',  
        #     'jj_dEta_nominal', 
        #     'jj_dPhi_nominal', 
        #     'jj_mass_nominal', 
        #     'zeppenfeld_nominal', 
        #     'mmj_min_dEta_nominal', 
        #     'mmj_min_dPhi_nominal', 
        #     'njets_nominal',
        #     "nBtagLoose_nominal",
        #     "nBtagMedium_nominal",
        #     "mmj2_dEta_nominal",
        #     "mmj2_dPhi_nominal",
        #     "wgt_nominal",
        #     'dimuon_mass', 
        #     'dimuon_pt', 
        #     'dimuon_eta', 
        #     'dimuon_cos_theta_cs', 
        #     'dimuon_phi_cs', 
        #     'mu1_pt_over_mass', 
        #     'mu1_eta', 
        #     'mu2_pt_over_mass', 
        #     'mu2_eta', 
        #     'zeppenfeld_nominal', 
        # ]
        # events = ak.zip({
        #     field : events[field] for field in fields2load
        # }).compute()
        # events = renameFieldsToV2(events)
        # making so taht copperheadV1 results work end -------------------------------------------
        
        
        print("done loading events!")
        if category == "ggh":
            processed_events = process4gghCategory(events, args.year, args.model_name)      
        elif category == "vbf":
            processed_events = process4vbfCategory(events) 
        else: 
            print ("unsupported category given!")
            raise ValueError
        # define save path and save
        # save_path = "/work/users/yun79/stage2_output/ggH/test"
        # save_path = f"{args.save_path}/{category}/{args.year}"
        save_path = f"{args.save_path}/{args.year}"
        print(f"save_path: {save_path}")
        # make save path if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if sample.lower() == "data":
            save_filename = f"{save_path}/processed_events_data.parquet"  
        elif sample.lower() == "ggh": # signal
            save_filename = f"{save_path}/processed_events_sigMC_ggh.parquet" 
        elif sample.lower() == "ggh_amcps": # signal
            save_filename = f"{save_path}/processed_events_sigMC_ggh_amcPS.parquet" 
        elif sample.lower() == "vbf": # signal
            save_filename = f"{save_path}/processed_events_sigMC_vbf.parquet" 
        elif sample.lower() == "dy":
            save_filename = f"{save_path}/processed_events_bkgMC_dy.parquet" 
        elif sample.lower() == "ewk":
            save_filename = f"{save_path}/processed_events_bkgMC_ewk.parquet" 
        elif sample.lower() == "tt":
            save_filename = f"{save_path}/processed_events_bkgMC_tt.parquet" 
        elif sample.lower() == "st":
            save_filename = f"{save_path}/processed_events_bkgMC_st.parquet" 
        elif sample.lower() == "ww":
            save_filename = f"{save_path}/processed_events_bkgMC_ww.parquet" 
        elif sample.lower() == "wz":
            save_filename = f"{save_path}/processed_events_bkgMC_wz.parquet" 
        elif sample.lower() == "zz":
            save_filename = f"{save_path}/processed_events_bkgMC_zz.parquet" 
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

        
        # This is ineligant, but also save the bdt edges that was presumably used
        BDTedges_load_path = "./configs/MVA/ggH/BDT_edges.yaml"
        edges = OmegaConf.load(BDTedges_load_path)
        OmegaConf.save(config=edges, f=f'{save_path}/BDT_edges.yaml')
    
    end_time = time.time()
    print(f"stage2 done in {end_time-start_time} seconds")

    