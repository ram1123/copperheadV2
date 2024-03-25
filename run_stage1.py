from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from copperhead_processor import EventProcessor
# NanoAODSchema.warn_missing_crossrefs = False
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import dask
# from dask.distributed import Client
import sys
import time
import json
from distributed import LocalCluster, Client, progress
import pandas as pd
import os
import tqdm
import warnings
import dask_awkward as dak
import glob
from itertools import islice
import copy
import argparse
from dask.distributed import performance_report
from corrections.evaluator import nnlops_weights

# dask.config.set({'logging.distributed': 'error'})
import logging
logger = logging.getLogger("distributed.utils_perf")
logger.setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

test_mode = False
np.set_printoptions(threshold=sys.maxsize)
import gc
import ctypes
def trim_memory() -> int:
     libc = ctypes.CDLL("libc.so.6")
     return libc.malloc_trim(0)
    
# # test code limiting memory leak -----------------------------------
# import gc
# client.run(gc.collect)  # collect garbage on all workers
# import ctypes
# def trim_memory() -> int:
#      libc = ctypes.CDLL("libc.so.6")
#      return libc.malloc_trim(0)
# client.run(trim_memory)

# #-------------------------------------------------------------------


def dataset_loop(processor, dataset_dict, file_idx=0, test=False, save_path=None):
    if save_path is None:
        save_path = "/depot/cms/users/yun79/results/stage1/test/" # default
        # save_path = "/depot/cms/hmm/yun79/copperheadV2/results/stage1/test/"
    if not test: # full scale implementation
        # print(f"dataset_dict: {dataset_dict['files']}")
        events = NanoEventsFactory.from_root(
            dataset_dict["files"],
            schemaclass=NanoAODSchema,
            metadata= dataset_dict["metadata"],
        ).events()
    else: # do it in a small set for developing and testing
        print("local testing")
        np.random.seed(0) 
        # test_size = 50
        test_size = 1000
        entry_start= 1000
        # metadata = {"dataset": "dy_M-50", "is_mc": True, "sumGenWgts" : 60713723011.942}
        # metadata = {"dataset": "vbf_powheg", "is_mc": True, "sumGenWgts" : 7720081.838819998}
        # metadata = {"dataset": "ggh_powheg", "is_mc": True}
        # metadata = {"dataset": "data_A", "is_mc": False}
        root_file = list(dataset_dict["files"].keys())[0]
        print(f"test root_file: {root_file}")
        events = NanoEventsFactory.from_root(
            {root_file : "Events"}, # dataset_dict is just a root file path string in test case
            schemaclass=NanoAODSchema,
            metadata= dataset_dict["metadata"],
            delayed= False,
            entry_start = entry_start,
            entry_stop = entry_start+test_size,
        ).events()
    # out_collections = coffea_processor.process(events)
    out_collections = processor.process(events)
    dataset_fraction = dataset_dict["metadata"]["fraction"]

    computed = out_collections
    # print(f"computed: {dask.compute(computed)}")
     #just reading test start--------------------------------
    # placeholder_dict =  {
    #         'mu1_pt': (computed["mu_pt"][:,0]),
    #         'mu2_pt': (computed["mu_pt"][:,1]),
       
            # 'mu1_eta': (computed["mu_eta"][:,0]),
            # 'mu2_eta': (computed["mu_eta"][:,1]),
            # 'mu1_phi': (computed["mu_phi"][:,0]),
            # 'mu2_phi': (computed["mu_phi"][:,1]),
            # 'mu1_iso': (computed["mu_iso"][:,0]),
            # 'mu2_iso': (computed["mu_iso"][:,1]),
            # # 'mu1_pt_over_mass': (computed["mu_pt_over_mass"][:,0]),
            # # 'mu2_pt_over_mass': (computed["mu_pt_over_mass"][:,1]),
            # "dimuon_mass": (computed["dimuon_mass"]),
            # "dimuon_ebe_mass_res": (computed["dimuon_ebe_mass_res"]),
            # "dimuon_ebe_mass_res_rel": (computed["dimuon_ebe_mass_res_rel"]),
            # "dimuon_pt": (computed["dimuon_pt"]),
            # # "dimuon_pt_log": (np.log(computed["dimuon_pt"])), # np functions are compatible with ak if input is ak array 
            # "dimuon_eta": (computed["dimuon_eta"]),
            # "dimuon_phi": (computed["dimuon_phi"]),
            # "dimuon_dEta": (computed["dimuon_dEta"]),
            # "dimuon_dPhi": (computed["dimuon_dPhi"]),
            # "dimuon_dR": (computed["dimuon_dR"]),
            # "dimuon_cos_theta_cs": (computed["dimuon_cos_theta_cs"]), 
            # "dimuon_phi_cs": (computed["dimuon_phi_cs"]), 
            # # # jet variables -------------------------------
            # "jet1_pt" : (computed["jet1_pt"]),
            # "jet1_eta" : (computed["jet1_eta"]),
            # "jet1_rap" : (computed["jet1_rap"]),
            # "jet1_phi" : (computed["jet1_phi"]),
            # "jet1_qgl" : (computed["jet1_qgl"]),
            # "jet1_jetId" : (computed["jet1_jetId"]),
            # "jet1_puId" : (computed["jet1_puId"]),
            # "jet2_pt" : (computed["jet2_pt"]),
            # "jet2_eta" : (computed["jet2_eta"]),
            # "jet2_rap" : (computed["jet2_rap"]),
            # "jet2_phi" : (computed["jet2_phi"]),
            # "jet2_qgl" : (computed["jet2_qgl"]),
            # "jet2_jetId" : (computed["jet2_jetId"]),
            # "jet2_puId" : (computed["jet2_puId"]),
            # "jj_mass" : (computed["jj_mass"]),
            # # "jj_mass_log" : (computed["jj_mass_log"]),
            # "jj_pt" : (computed["jj_pt"]),
            # "jj_eta" : (computed["jj_eta"]),
            # "jj_phi" : (computed["jj_phi"]),
            # "jj_dEta" : (computed["jj_dEta"]),
            # "jj_dPhi":  (computed["jj_dPhi"]),
            # "mmj1_dEta" : (computed["mmj1_dEta"]),
            # "mmj1_dPhi" : (computed["mmj1_dPhi"]),
            # "mmj1_dR" : (computed["mmj1_dR"]),
            # "mmj2_dEta" : (computed["mmj2_dEta"]),
            # "mmj2_dPhi" : (computed["mmj2_dPhi"]),
            # "mmj2_dR" : (computed["mmj2_dR"]),
            # "mmj_min_dEta" : (computed["mmj_min_dEta"]),
            # "mmj_min_dPhi" : (computed["mmj_min_dPhi"]),
            # "mmjj_pt" : (computed["mmjj_pt"]),
            # "mmjj_eta" : (computed["mmjj_eta"]),
            # "mmjj_phi" : (computed["mmjj_phi"]),
            # "mmjj_mass" : (computed["mmjj_mass"]),
            # "rpt" : (computed["rpt"]),
            # "zeppenfeld" : (computed["zeppenfeld"]),
            # "njets" : (computed["njets"]),
            # # Btagging WPs ------------------------------------
            # "nBtagLoose" : (computed["nBtagLoose"]),
            # "nBtagMedium" : (computed["nBtagMedium"]),
            # # regions -------------------------------------
            # "z_peak" : (computed["z_peak"]),
            # "h_sidebands" : (computed["h_sidebands"]),
            # "h_peak" : (computed["h_peak"]),
            # # vbf ?? ------------------------------------------------
            # "vbf_cut" : (computed["vbf_cut"]),
            # #----------------------------------------
            # "fraction" : dataset_fraction*(ak.ones_like(computed["njets"])),    
        
    # }
    # if dataset_dict["metadata"]["is_mc"]:
    #     additional_dict = {
    #          # gen jet variables -------------------------------------
    #         "gjj_mass":  (computed["gjj_mass"]),
    #         'gjet1_pt': (computed["gjet_pt"][:,0]),
    #         'gjet2_pt': (computed["gjet_pt"][:,1]),
    #         'gjet1_eta': (computed["gjet_eta"][:,0]),
    #         'gjet2_eta': (computed["gjet_eta"][:,1]),
    #         'gjet1_phi': (computed["gjet_phi"][:,0]),
    #         'gjet2_phi': (computed["gjet_phi"][:,1]),
    #         'gjet1_mass': (computed["gjet_mass"][:,0]),
    #         'gjet2_mass': (computed["gjet_mass"][:,1]),
    #         "gjj_dEta": (computed["gjj_dEta"]),
    #         "gjj_dPhi": (computed["gjj_dPhi"]),
    #         "gjj_dR": (computed["gjj_dR"]),
    #         # weights -------------------------------------
    #         "weight_nominal" : (ak.ones_like(computed["nominal"])),
    #     }
    #     placeholder_dict.update(additional_dict)
    #just reading test end--------------------------------

    # ------------------------------------------
    placeholder_dict =  {
            # 'mu1_pt': (computed["mu_pt"][:,0]),
            # 'mu2_pt': (computed["mu_pt"][:,1]),
            # 'mu1_eta': (computed["mu_eta"][:,0]),
            # 'mu2_eta': (computed["mu_eta"][:,1]),
            # 'mu1_phi': (computed["mu_phi"][:,0]),
            # 'mu2_phi': (computed["mu_phi"][:,1]),
            'mu1_pt': (computed["mu1_pt"]),
            'mu2_pt': (computed["mu2_pt"]),
            'mu1_eta': (computed["mu1_eta"]),
            'mu2_eta': (computed["mu2_eta"]),
            'mu1_phi': (computed["mu1_phi"]),
            'mu2_phi': (computed["mu2_phi"]),
            'nmuons': (computed["nmuons"]),
            # 'mu1_pt_raw': (computed["mu1_pt_raw"]),
            # 'mu2_pt_raw': (computed["mu2_pt_raw"]),
            # 'mu1_pt_fsr': (computed["mu1_pt_fsr"]),
            # 'mu2_pt_fsr': (computed["mu2_pt_fsr"]),
            'jet1_pt': (computed["jet1_pt"]),
            'jet2_pt': (computed["jet2_pt"]),
            'jet1_eta': (computed["jet1_eta"]),
            'jet2_eta': (computed["jet2_eta"]),
            'jet1_phi': (computed["jet1_phi"]),
            'jet2_phi': (computed["jet2_phi"]),
            'jet1_mass': (computed["jet1_mass"]),
            'jet2_mass': (computed["jet2_mass"]),
            'njets': (computed["njets"]),
            'weights': (computed["weights"]),
            # 'fsr_mask': (computed["fsr_mask"]),
            'dimuon_mass': (computed["dimuon_mass"]),
            'dimuon_ebe_mass_res': (computed["dimuon_ebe_mass_res"]),
            'dimuon_cos_theta_cs': (computed["dimuon_cos_theta_cs"]),
            'dimuon_phi_cs': (computed["dimuon_phi_cs"]),

            'jet1_pt_raw': (computed["jet1_pt_raw"]),
            'jet1_mass_raw': (computed["jet1_mass_raw"]),
            'jet1_rho': (computed["jet1_rho"]),
            'jet1_area': (computed["jet1_area"]),
            'jet1_pt_jec': (computed["jet1_pt_jec"]),
            'jet1_mass_jec': (computed["jet1_mass_jec"]),
            'jet2_pt_raw': (computed["jet2_pt_raw"]),
            'jet2_mass_raw': (computed["jet2_mass_raw"]),
            'jet2_rho': (computed["jet2_rho"]),
            'jet2_area': (computed["jet2_area"]),
            'jet2_pt_jec': (computed["jet2_pt_jec"]),
            'jet2_mass_jec': (computed["jet2_mass_jec"]),
            # fraction -------------------------------------
            "fraction" : dataset_fraction*(ak.ones_like(computed["njets"])), 
            # Btagging WPs ------------------------------------
            "nBtagLoose" : (computed["nBtagLoose"]),
            "nBtagMedium" : (computed["nBtagMedium"]),
            # regions -------------------------------------
            "z_peak" : (computed["z_peak"]),
            "h_sidebands" : (computed["h_sidebands"]),
            "h_peak" : (computed["h_peak"]),
            # vbf ?? ------------------------------------------------
            "vbf_cut" : (computed["vbf_cut"]),
            # "pass_leading_pt" : (computed["pass_leading_pt"]),
        
         }
    is_mc = dataset_dict["metadata"]["is_mc"]
    if is_mc:
        additional_dict = {
            'jet1_pt_gen': (computed["jet1_pt_gen"]),
            'jet2_pt_gen': (computed["jet2_pt_gen"]),
    #          # gen jet variables -------------------------------------
    #         "gjj_mass":  (computed["gjj_mass"]),
    #         'gjet1_pt': (computed["gjet_pt"][:,0]),
    #         'gjet2_pt': (computed["gjet_pt"][:,1]),
    #         'gjet1_eta': (computed["gjet_eta"][:,0]),
    #         'gjet2_eta': (computed["gjet_eta"][:,1]),
    #         'gjet1_phi': (computed["gjet_phi"][:,0]),
    #         'gjet2_phi': (computed["gjet_phi"][:,1]),
    #         'gjet1_mass': (computed["gjet_mass"][:,0]),
    #         'gjet2_mass': (computed["gjet_mass"][:,1]),
    #         "gjj_dEta": (computed["gjj_dEta"]),
    #         "gjj_dPhi": (computed["gjj_dPhi"]),
    #         "gjj_dR": (computed["gjj_dR"]),
        }
        placeholder_dict.update(additional_dict)
    #------------------------------
        
    # define save path
    fraction = round(dataset_dict["metadata"]["fraction"], 3)
    fraction_str = str(fraction).replace('.', '_')
    sample_name = dataset_dict['metadata']['dataset']
    save_path = save_path + f"/f{fraction_str}/{dataset_dict['metadata']['dataset']}/{file_idx}"
    print(f"save_path: {save_path}")
    # remove previously existing files
    filelist = glob.glob(f"{save_path}/*.parquet")
    print(f"len(filelist): {len(filelist)}")
    for file in filelist:
        os.remove(file)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #------------------------
    # do nnlops correct on normal awkward
    do_nnlops = processor.config["do_nnlops"] and ("ggh" in events.metadata["dataset"])
    if do_nnlops: # we need full computed for this
        print("doing nnlops!")
        # placeholder_dict = dask.compute(placeholder_dict)[0]
        HTX_dict = {
            "HTXS_Higgs_pt" : (computed["HTXS_Higgs_pt"]),
            "HTXS_njets30" : (computed["HTXS_njets30"]),
        }
        HTX_dict = dask.compute(HTX_dict)[0] # dask compute gives a tuple of length one
        # print(f"type(HTX_dict): {type(HTX_dict)}")
        # print(f"(HTX_dict): {(HTX_dict)}")
        nnlops_wgt = nnlops_weights(
            HTX_dict["HTXS_Higgs_pt"],
            HTX_dict["HTXS_njets30"], 
            processor.config, 
            events.metadata["dataset"]
        )
        nnlops_save_path = save_path + "/nnlops"
        # remove previously existing files
        filelist = glob.glob(f"{nnlops_save_path}/*.parquet")
        print(f"nnlops filelist: {filelist}")
        for file in filelist:
            os.remove(file)
        if not os.path.exists(nnlops_save_path):
            os.makedirs(nnlops_save_path)
        print(f"nnlops_wgt: {nnlops_wgt}")
        # save nnlops wgts to apply them later
        ak.to_parquet(ak.zip({"nnlops_wgt" : nnlops_wgt}), nnlops_save_path+"/wgt.parquet")
        # ak.to_parquet(nnlops_wgt, nnlops_save_path+"/wgt.parquet")

    #----------------------------------
    zip = ak.zip(placeholder_dict, depth_limit=1)
    # zip = dask.compute(placeholder_dict)
    # N_reasonable = 100000
    # N_reasonable = 40000
    # zip = zip.repartition(rows_per_partition=N_reasonable)
    # print(f"zip: {zip.compute()}")
    
    # zip.to_parquet(save_path, compute=True)
    skim = dak.to_parquet(zip, save_path, compute=False)
    return skim
    # if test:
    #     for var_name, ak_arr in placeholder_dict.items():
    #         sample_save_path = save_path + f"/f{fraction_str}/{dataset_dict['metadata']['dataset']}/{var_name}"
    #         if not os.path.exists(sample_save_path):
    #             os.makedirs(sample_save_path)
    #         # print(f"saving to parquet on: {sample_save_path}")
    #         ak.to_parquet(ak_arr, sample_save_path+"/array1.parquet")
    # else:
    #     for var_name, ak_arr in placeholder_dict.items():
    #         sample_save_path = save_path + f"/f{fraction_str}/{dataset_dict['metadata']['dataset']}/{var_name}"
    #         if not os.path.exists(sample_save_path):
    #             os.makedirs(sample_save_path)
    #         print(f"saving to parquet on: {sample_save_path}")
    #         # if "ttjet" in sample_name:
    #         #     print(f"computed: {dask.compute(computed)}")
    #         # zip.to_parquet(save_path, compute=False)
    #         # ak_arr.to_parquet(save_path, compute=False)
    #         # ak_arr.to_parquet(save_path)
    #         #delete preexisting parquet files
    #         filelist = glob.glob(f"{sample_save_path}/*.parquet")
    #         print(f"filelist: {filelist}")
    #         for file in filelist:
    #             try:
    #                 os.remove(file)
    #             except Exception:
    #                 pass
    #         var_step = time.time()
    #         dak.to_parquet(ak_arr,sample_save_path,compute=True)
    #         var_elapsed = round(time.time() - var_step, 3)
    #         print(f"Finished saving {sample_save_path} in {var_elapsed} s.")

# def divide_chunks(l: list, n: int): 
#     # looping till length l 
#     for i in range(0, len(l), n):  
#         yield l[i:i + n] 


def divide_chunks(data: dict, SIZE: int):
   it = iter(data)
   for i in range(0, len(data), SIZE):
      yield {k:data[k] for k in islice(it, SIZE)}


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
    "-y",
    "--year",
    dest="year",
    default="2018",
    action="store",
    help="string value of year we are calculating",
    )
    parser.add_argument(
    "--use_gateway",
    dest="use_gateway",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, uses dask gateway client instead of local",
    )
    args = parser.parse_args()
    
    time_step = time.time()
    
    warnings.filterwarnings('ignore')
    """
    Coffea Dask automatically uses the Dask Client that has been defined above
    """
    
    config_path = "./config/parameters.json"
    with open(config_path) as file:
        config = json.loads(file.read())
    coffea_processor = EventProcessor(config, test_mode=test_mode)

    if not test_mode: # full scale implementation
        # # original ---------------------------------------------------------
        # if args.use_gateway:
        #     from dask_gateway import Gateway
        #     gateway = Gateway()
        #     cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
        #     client = gateway.connect(cluster_info.name).get_client()
        #     print("Gateway Client created")
        # # #-----------------------------------------------------------
        # else:
        #     cluster = LocalCluster(processes=True, memory_limit='14 GiB', threads_per_worker=1,)
        #     # cluster.adapt(minimum=8, maximum=8)
        #     cluster.scale(1)
        #     client = Client(cluster)
        #     print("Local scale Client created")
        #     # print(f"client dashboard link: {client.dashboard_link}")
        #-------------------------------------------------------------------------------------
        #-----------------------------------------------------------
        # client = Client(n_workers=8,  threads_per_worker=1, processes=True, memory_limit='8 GiB') 
        #---------------------------------------------------------
        # print("cluster scale up")
        # sample_path = "./config/processor_samples.json"
        sample_path = "./config/fraction_processor_samples.json"
        with open(sample_path) as file:
            samples = json.loads(file.read())
        # print(f"samples.keys(): {samples.keys()}")
        total_save_path = args.save_path + f"/{args.year}"
        # with performance_report(filename="dask-report.html"):
        # for dataset, sample in samples.items():
        # dask.config.set(scheduler='single-threaded')
        for dataset, sample in tqdm.tqdm(samples.items()):
            sample_step = time.time()
            #test
            # if dataset != "ttjets_dl":
            # if "data" in dataset:
            #     continue
            # print(f"dataset: {dataset}")
            # print(f'sample["files"]: {sample["files"]}')
            # divide sample to smaller chunks
            # max_file_len = 15
            max_file_len = 6
            # max_file_len = 1
            smaller_files = list(divide_chunks(sample["files"], max_file_len))
            # print(f"smaller_files: {smaller_files}")
            for idx in tqdm.tqdm(range(len(smaller_files)), leave=False):
                # with Client(n_workers=41,  threads_per_worker=1, processes=True, memory_limit='3 GiB', silence_logs=logging.ERROR) as client:
                # with Client(n_workers=41,  threads_per_worker=1, processes=True, memory_limit='3 GiB') as client:
                with Client(n_workers=1,  threads_per_worker=1, processes=True, memory_limit='15 GiB') as client:
                    with performance_report(filename="dask-report.html"):
                        smaller_sample = copy.deepcopy(sample)
                        smaller_sample["files"] = smaller_files[idx]
                        # print(f"smaller_files[{idx}]: {smaller_files[idx]}")
                        # continue
                        var_step = time.time()
                        to_compute = dataset_loop(coffea_processor, smaller_sample, file_idx=idx, test=test_mode, save_path=total_save_path)
                        # print(f"to_compute: {to_compute}")
                        dask.compute(to_compute)
                        # futures = dask.compute(to_compute) # futures: represent the ongoing or completed computations within the Dask cluster, regardless of the underlying execution environment
                        # print(f"futures: {futures}")
                        # progress(futures)
                        # results = client.gather(futures)
    
                        # do garbage collection and memory trimming-----------
                        client.run(gc.collect)
                        client.run(trim_memory)
                        #-----------------------------------------------------
                        
                        var_elapsed = round(time.time() - var_step, 3)
                        print(f"Finished file_idx {idx} in {var_elapsed} s.")
            sample_elapsed = round(time.time() - sample_step, 3)
            print(f"Finished sample {dataset} in {sample_elapsed} s.")
                
    else:
        # dataset_loop(coffea_processor, xrootd_path+fname, test=test_mode)

        sample_path = "./config/fraction_processor_samples.json"
        with open(sample_path) as file:
            samples = json.loads(file.read())
        # print(f"samples.keys(): {samples.keys()}")

        with performance_report(filename="dask-report.html"):
            # for dataset, sample in samples.items():
            for dataset, sample in tqdm.tqdm(samples.items()):
                
                # testing
                # if ("dy_M-100To200" not in dataset) and ("data_A" not in dataset):
                # if ("dy_M-100To200" not in dataset) :
                # if ("ggh_powheg" not in dataset) and ("vbf_powheg" not in dataset):
                #     continue
                # print(f"dataset: {dataset}")
                dataset_loop(coffea_processor, sample, test=test_mode)
        
    

    

    
    elapsed = round(time.time() - time_step, 3)
    print(f"Finished everything in {elapsed} s.")