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
from corrections.evaluator import nnlops_weights, qgl_weights

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
        
    out_collections = processor.process(events)
    dataset_fraction = dataset_dict["metadata"]["fraction"]
    
    # Dmitry test 4 start ----------------------------

    skim = dak.to_parquet(out_collections, save_path, compute=False)
    return skim
    # DMitry test 4 end--------------------------------

    # ------------------------------------------
    placeholder_dict =  {
            'mu1_pt': (out_collections["mu1_pt"]),
            'mu2_pt': (out_collections["mu2_pt"]),
            'mu1_eta': (out_collections["mu1_eta"]),
            'mu2_eta': (out_collections["mu2_eta"]),
            'mu1_phi': (out_collections["mu1_phi"]),
            'mu2_phi': (out_collections["mu2_phi"]),
            'jet1_pt': (out_collections["jet1_pt"]),
            'jet2_pt': (out_collections["jet2_pt"]),
            'jet1_eta': (out_collections["jet1_eta"]),
            'jet2_eta': (out_collections["jet2_eta"]),
            'jet1_phi': (out_collections["jet1_phi"]),
            'jet2_phi': (out_collections["jet2_phi"]),
            'jet1_mass': (out_collections["jet1_mass"]),
            'jet2_mass': (out_collections["jet2_mass"]),
            'njets': (out_collections["njets"]),
            'weights': (out_collections["weights"]),
            # 'dimuon_mass': (out_collections["dimuon_mass"]),
            # 'dimuon_ebe_mass_res': (out_collections["dimuon_ebe_mass_res"]),
            # 'dimuon_cos_theta_cs': (out_collections["dimuon_cos_theta_cs"]),
            # 'dimuon_phi_cs': (out_collections["dimuon_phi_cs"]),

            'jet1_pt_raw': (out_collections["jet1_pt_raw"]),
            'jet1_mass_raw': (out_collections["jet1_mass_raw"]),
            'jet1_rho': (out_collections["jet1_rho"]),
            'jet1_area': (out_collections["jet1_area"]),
            # 'jet1_pt_jec': (out_collections["jet1_pt_jec"]),
            # 'jet1_mass_jec': (out_collections["jet1_mass_jec"]),
            'jet2_pt_raw': (out_collections["jet2_pt_raw"]),
            'jet2_mass_raw': (out_collections["jet2_mass_raw"]),
            'jet2_rho': (out_collections["jet2_rho"]),
            'jet2_area': (out_collections["jet2_area"]),
            # 'jet2_pt_jec': (out_collections["jet2_pt_jec"]),
            # 'jet2_mass_jec': (out_collections["jet2_mass_jec"]),
            # fraction -------------------------------------
            "fraction" : dataset_fraction*(ak.ones_like(out_collections["njets"])), 
            # Btagging WPs ------------------------------------
            "nBtagLoose" : (out_collections["nBtagLoose"]),
            "nBtagMedium" : (out_collections["nBtagMedium"]),
            # regions -------------------------------------
            "z_peak" : (out_collections["z_peak"]),
            "h_sidebands" : (out_collections["h_sidebands"]),
            "h_peak" : (out_collections["h_peak"]),
            # vbf ?? ------------------------------------------------
            "vbf_cut" : (out_collections["vbf_cut"]),
            # "pass_leading_pt" : (out_collections["pass_leading_pt"]),
        
         }
    is_mc = dataset_dict["metadata"]["is_mc"]
    if is_mc:
        additional_dict = {
            'jet1_pt_gen': (out_collections["jet1_pt_gen"]),
            'jet2_pt_gen': (out_collections["jet2_pt_gen"]),
    #          # gen jet variables -------------------------------------
    #         "gjj_mass":  (out_collections["gjj_mass"]),
    #         'gjet1_pt': (out_collections["gjet_pt"][:,0]),
    #         'gjet2_pt': (out_collections["gjet_pt"][:,1]),
    #         'gjet1_eta': (out_collections["gjet_eta"][:,0]),
    #         'gjet2_eta': (out_collections["gjet_eta"][:,1]),
    #         'gjet1_phi': (out_collections["gjet_phi"][:,0]),
    #         'gjet2_phi': (out_collections["gjet_phi"][:,1]),
    #         'gjet1_mass': (out_collections["gjet_mass"][:,0]),
    #         'gjet2_mass': (out_collections["gjet_mass"][:,1]),
    #         "gjj_dEta": (out_collections["gjj_dEta"]),
    #         "gjj_dPhi": (out_collections["gjj_dPhi"]),
    #         "gjj_dR": (out_collections["gjj_dR"]),
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
    
    #----------------------------------
    zip = ak.zip(placeholder_dict, depth_limit=1)
    # zip = dask.compute(placeholder_dict)
    
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
        if args.use_gateway:
            from dask_gateway import Gateway
            # gateway = Gateway()
            gateway = Gateway(
                "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
                proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
            )
            cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
            client = gateway.connect(cluster_info.name).get_client()
            print("Gateway Client created")
        # # #-----------------------------------------------------------
        else:
            # client = Client(n_workers=1,  threads_per_worker=1, processes=True, memory_limit='15 GiB') 
            client = Client(n_workers=15,  threads_per_worker=1, processes=True, memory_limit='4 GiB') 
            # client = Client(n_workers=41,  threads_per_worker=1, processes=True, memory_limit='3 GiB') 
            print("Local scale Client created")
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
        with performance_report(filename="dask-report.html"):
            for dataset, sample in tqdm.tqdm(samples.items()):
                sample_step = time.time()
                max_file_len = 15
                # max_file_len = 6
                # max_file_len = 50
                # max_file_len = 9
                smaller_files = list(divide_chunks(sample["files"], max_file_len))
                # print(f"smaller_files: {smaller_files}")
                for idx in tqdm.tqdm(range(len(smaller_files)), leave=False):
                    print("restarting workers!")
                    client.restart(wait_for_workers = False)
                    smaller_sample = copy.deepcopy(sample)
                    smaller_sample["files"] = smaller_files[idx]
                    var_step = time.time()
                    to_compute = dataset_loop(coffea_processor, smaller_sample, file_idx=idx, test=test_mode, save_path=total_save_path)
                    print(f"to_compute: {to_compute}")
                    dask.compute(to_compute)

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