from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from copperhead_processor import EventProcessor
# NanoAODSchema.warn_missing_crossrefs = False
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import dask
from dask.distributed import Client
import sys
import time
import json
from distributed import LocalCluster, Client
import pandas as pd
import os
import tqdm
import warnings

test_mode = True
np.set_printoptions(threshold=sys.maxsize)



def dataset_loop(processor, dataset_dict, test=False):
    save_path = "/depot/cms/users/yun79/results/stage1/test/"
    # save_path = "/depot/cms/hmm/yun79/copperheadV2/results/stage1/test/"
    if not test: # full scale implementation
        # print(f"dataset_dict: {dataset_dict['files']}")
        events = NanoEventsFactory.from_root(
            # samples["files"],
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
    out_collections = coffea_processor.process(events)
    dataset_fraction = dataset_dict["metadata"]["fraction"]
    """
    # print(f"copperhead2 run stage1 out_collections b4 compute: {out_collections}")
    (computed, ) = dask.compute(out_collections)
    # print(f"copperhead2 run stage1 computed after compute: {computed}")
    # print(f"copperhead2 run stage1 type(out): {type(result)}")
    
    # placeholder_dict =  {
    #         'mu1_pt': ak.to_numpy(computed["mu_pt"][:,0]),
    #         'mu2_pt': ak.to_numpy(computed["mu_pt"][:,1]),
    #         'mu1_eta': ak.to_numpy(computed["mu_eta"][:,0]),
    #         'mu2_eta': ak.to_numpy(computed["mu_eta"][:,1]),
    #         'mu1_phi': ak.to_numpy(computed["mu_phi"][:,0]),
    #         'mu2_phi': ak.to_numpy(computed["mu_phi"][:,1]),
    #         'mu1_iso': ak.to_numpy(computed["mu_iso"][:,0]),
    #         'mu2_iso': ak.to_numpy(computed["mu_iso"][:,1]),
    #         'mu1_pt_over_mass': ak.to_numpy(computed["mu_pt_over_mass"][:,0]),
    #         'mu2_pt_over_mass': ak.to_numpy(computed["mu_pt_over_mass"][:,1]),
    #         "dimuon_mass": ak.to_numpy(computed["dimuon_mass"]),
    #         "dimuon_ebe_mass_res": ak.to_numpy(computed["dimuon_ebe_mass_res"]),
    #         "dimuon_ebe_mass_res_rel": ak.to_numpy(computed["dimuon_ebe_mass_res_rel"]),
    #         "dimuon_pt": ak.to_numpy(computed["dimuon_pt"]),
    #         "dimuon_pt_log": ak.to_numpy(np.log(computed["dimuon_pt"])), # np functions are compatible with ak if input is ak array 
    #         "dimuon_eta": ak.to_numpy(computed["dimuon_eta"]),
    #         "dimuon_phi": ak.to_numpy(computed["dimuon_phi"]),
    #         "dimuon_dEta": ak.to_numpy(computed["dimuon_dEta"]),
    #         "dimuon_dPhi": ak.to_numpy(computed["dimuon_dPhi"]),
    #         "dimuon_dR": ak.to_numpy(computed["dimuon_dR"]),
    #         "dimuon_cos_theta_cs": ak.to_numpy(computed["dimuon_cos_theta_cs"]), 
    #         "dimuon_phi_cs": ak.to_numpy(computed["dimuon_phi_cs"]), 
    #         # jet variables -------------------------------
    #         "jet1_pt" : ak.to_numpy(computed["jet1_pt"]),
    #         "jet1_eta" : ak.to_numpy(computed["jet1_eta"]),
    #         "jet1_rap" : ak.to_numpy(computed["jet1_rap"]),
    #         "jet1_phi" : ak.to_numpy(computed["jet1_phi"]),
    #         "jet1_qgl" : ak.to_numpy(computed["jet1_qgl"]),
    #         "jet1_jetId" : ak.to_numpy(computed["jet1_jetId"]),
    #         "jet1_puId" : ak.to_numpy(computed["jet1_puId"]),
    #         "jet2_pt" : ak.to_numpy(computed["jet2_pt"]),
    #         "jet2_eta" : ak.to_numpy(computed["jet2_eta"]),
    #         "jet2_rap" : ak.to_numpy(computed["jet2_rap"]),
    #         "jet2_phi" : ak.to_numpy(computed["jet2_phi"]),
    #         "jet2_qgl" : ak.to_numpy(computed["jet2_qgl"]),
    #         "jet2_jetId" : ak.to_numpy(computed["jet2_jetId"]),
    #         "jet2_puId" : ak.to_numpy(computed["jet2_puId"]),
    #         "jj_mass" : ak.to_numpy(computed["jj_mass"]),
    #         "jj_mass_log" : ak.to_numpy(computed["jj_mass_log"]),
    #         "jj_pt" : ak.to_numpy(computed["jj_pt"]),
    #         "jj_eta" : ak.to_numpy(computed["jj_eta"]),
    #         "jj_phi" : ak.to_numpy(computed["jj_phi"]),
    #         "jj_dEta" : ak.to_numpy(computed["jj_dEta"]),
    #         "jj_dPhi":  ak.to_numpy(computed["jj_dPhi"]),
    #         "mmj1_dEta" : ak.to_numpy(computed["mmj1_dEta"]),
    #         "mmj1_dPhi" : ak.to_numpy(computed["mmj1_dPhi"]),
    #         "mmj1_dR" : ak.to_numpy(computed["mmj1_dR"]),
    #         "mmj2_dEta" : ak.to_numpy(computed["mmj2_dEta"]),
    #         "mmj2_dPhi" : ak.to_numpy(computed["mmj2_dPhi"]),
    #         "mmj2_dR" : ak.to_numpy(computed["mmj2_dR"]),
    #         "mmj_min_dEta" : ak.to_numpy(computed["mmj_min_dEta"]),
    #         "mmj_min_dPhi" : ak.to_numpy(computed["mmj_min_dPhi"]),
    #         "mmjj_pt" : ak.to_numpy(computed["mmjj_pt"]),
    #         "mmjj_eta" : ak.to_numpy(computed["mmjj_eta"]),
    #         "mmjj_phi" : ak.to_numpy(computed["mmjj_phi"]),
    #         "mmjj_mass" : ak.to_numpy(computed["mmjj_mass"]),
    #         "rpt" : ak.to_numpy(computed["rpt"]),
    #         "zeppenfeld" : ak.to_numpy(computed["zeppenfeld"]),
    #         "njets" : ak.to_numpy(computed["njets"]),
    #         # regions -------------------------------------
    #         "z_peak" : ak.to_numpy(computed["z_peak"]),
    #         "h_sidebands" : ak.to_numpy(computed["h_sidebands"]),
    #         "h_peak" : ak.to_numpy(computed["h_peak"]),
    #         # vbf ?? ------------------------------------------------
    #         "vbf_cut" : ak.to_numpy(computed["vbf_cut"]),
    #         #----------------------------------------
    #         "fraction" : dataset_fraction*ak.to_numpy(ak.ones_like(computed["njets"])),
    # }
    # if dataset_dict["metadata"]["is_mc"]:
    #     additional_dict = {
    #          # gen jet variables -------------------------------------
    #         "gjj_mass":  ak.to_numpy(computed["gjj_mass"]),
    #         'gjet1_pt': ak.to_numpy(computed["gjet_pt"][:,0]),
    #         'gjet2_pt': ak.to_numpy(computed["gjet_pt"][:,1]),
    #         'gjet1_eta': ak.to_numpy(computed["gjet_eta"][:,0]),
    #         'gjet2_eta': ak.to_numpy(computed["gjet_eta"][:,1]),
    #         'gjet1_phi': ak.to_numpy(computed["gjet_phi"][:,0]),
    #         'gjet2_phi': ak.to_numpy(computed["gjet_phi"][:,1]),
    #         'gjet1_mass': ak.to_numpy(computed["gjet_mass"][:,0]),
    #         'gjet2_mass': ak.to_numpy(computed["gjet_mass"][:,1]),
    #         "gjj_dEta": ak.to_numpy(computed["gjj_dEta"]),
    #         "gjj_dPhi": ak.to_numpy(computed["gjj_dPhi"]),
    #         "gjj_dR": ak.to_numpy(computed["gjj_dR"]),
    #         # weights -------------------------------------
    #         "weight_nominal" : ak.to_numpy(ak.ones_like(computed["nominal"])),
    #     }
    #     placeholder_dict.update(additional_dict)
    # placeholder = pd.DataFrame(placeholder_dict)

    # print(f"copperhead2 EventProcessor after leading pt cut placeholder: \n {placeholder.to_string()}")
    #save results 
    fraction_str = str(dataset_dict["metadata"]["original_fraction"]).replace('.', '_')
    save_path = save_path + f"/f{fraction_str}"
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    dataset = events.metadata['dataset']
    placeholder.to_csv(save_path+f"/V2stage1_{dataset}.csv")
    """
    computed = out_collections
    placeholder_dict =  {
            'mu1_pt': (computed["mu_pt"][:,0]),
            'mu2_pt': (computed["mu_pt"][:,1]),
            'mu1_eta': (computed["mu_eta"][:,0]),
            'mu2_eta': (computed["mu_eta"][:,1]),
            'mu1_phi': (computed["mu_phi"][:,0]),
            'mu2_phi': (computed["mu_phi"][:,1]),
            'mu1_iso': (computed["mu_iso"][:,0]),
            'mu2_iso': (computed["mu_iso"][:,1]),
            'mu1_pt_over_mass': (computed["mu_pt_over_mass"][:,0]),
            'mu2_pt_over_mass': (computed["mu_pt_over_mass"][:,1]),
            "dimuon_mass": (computed["dimuon_mass"]),
            "dimuon_ebe_mass_res": (computed["dimuon_ebe_mass_res"]),
            "dimuon_ebe_mass_res_rel": (computed["dimuon_ebe_mass_res_rel"]),
            "dimuon_pt": (computed["dimuon_pt"]),
            "dimuon_pt_log": (np.log(computed["dimuon_pt"])), # np functions are compatible with ak if input is ak array 
            "dimuon_eta": (computed["dimuon_eta"]),
            "dimuon_phi": (computed["dimuon_phi"]),
            "dimuon_dEta": (computed["dimuon_dEta"]),
            "dimuon_dPhi": (computed["dimuon_dPhi"]),
            "dimuon_dR": (computed["dimuon_dR"]),
            "dimuon_cos_theta_cs": (computed["dimuon_cos_theta_cs"]), 
            "dimuon_phi_cs": (computed["dimuon_phi_cs"]), 
            # # jet variables -------------------------------
            "jet1_pt" : (computed["jet1_pt"]),
            "jet1_eta" : (computed["jet1_eta"]),
            "jet1_rap" : (computed["jet1_rap"]),
            "jet1_phi" : (computed["jet1_phi"]),
            "jet1_qgl" : (computed["jet1_qgl"]),
            "jet1_jetId" : (computed["jet1_jetId"]),
            "jet1_puId" : (computed["jet1_puId"]),
            "jet2_pt" : (computed["jet2_pt"]),
            "jet2_eta" : (computed["jet2_eta"]),
            "jet2_rap" : (computed["jet2_rap"]),
            "jet2_phi" : (computed["jet2_phi"]),
            "jet2_qgl" : (computed["jet2_qgl"]),
            "jet2_jetId" : (computed["jet2_jetId"]),
            "jet2_puId" : (computed["jet2_puId"]),
            "jj_mass" : (computed["jj_mass"]),
            # "jj_mass_log" : (computed["jj_mass_log"]),
            "jj_pt" : (computed["jj_pt"]),
            "jj_eta" : (computed["jj_eta"]),
            "jj_phi" : (computed["jj_phi"]),
            "jj_dEta" : (computed["jj_dEta"]),
            "jj_dPhi":  (computed["jj_dPhi"]),
            "mmj1_dEta" : (computed["mmj1_dEta"]),
            "mmj1_dPhi" : (computed["mmj1_dPhi"]),
            "mmj1_dR" : (computed["mmj1_dR"]),
            "mmj2_dEta" : (computed["mmj2_dEta"]),
            "mmj2_dPhi" : (computed["mmj2_dPhi"]),
            "mmj2_dR" : (computed["mmj2_dR"]),
            "mmj_min_dEta" : (computed["mmj_min_dEta"]),
            "mmj_min_dPhi" : (computed["mmj_min_dPhi"]),
            "mmjj_pt" : (computed["mmjj_pt"]),
            "mmjj_eta" : (computed["mmjj_eta"]),
            "mmjj_phi" : (computed["mmjj_phi"]),
            "mmjj_mass" : (computed["mmjj_mass"]),
            "rpt" : (computed["rpt"]),
            "zeppenfeld" : (computed["zeppenfeld"]),
            "njets" : (computed["njets"]),
            # regions -------------------------------------
            "z_peak" : (computed["z_peak"]),
            "h_sidebands" : (computed["h_sidebands"]),
            "h_peak" : (computed["h_peak"]),
            # vbf ?? ------------------------------------------------
            "vbf_cut" : (computed["vbf_cut"]),
            #----------------------------------------
            "fraction" : dataset_fraction*(ak.ones_like(computed["njets"])),
    }
    zip = ak.zip(placeholder_dict, depth_limit=1)
    # N_reasonable = 100000
    N_reasonable = 40000
    # zip = zip.repartition(rows_per_partition=N_reasonable)
    print(f"zip: {zip}")
    fraction_str = str(dataset_dict["metadata"]["original_fraction"]).replace('.', '_')
    save_path = save_path + f"/f{fraction_str}/{dataset_dict['metadata']['dataset']}"
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    if not test:
        print("saving to parquet")
        zip.to_parquet(save_path, compute=False)

if __name__ == "__main__":
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
        from dask_gateway import Gateway
        # gateway = Gateway()
        # cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
        # client = gateway.connect(cluster_info.name).get_client()
        # print("Gateway Client created")
        # #-----------------------------------------------------------
        cluster = LocalCluster()
        cluster.adapt(minimum=16, maximum=16)
        # cluster.scale(63) # create 16 local workers
        client = Client(cluster)
        print("Local scale Client created")
        print(f"client dashboard link: {client.dashboard_link}")
        
        print("cluster scale up")
        # sample_path = "./config/processor_samples.json"
        sample_path = "./config/fraction_processor_samples.json"
        with open(sample_path) as file:
            samples = json.loads(file.read())
        # print(f"samples.keys(): {samples.keys()}")

        # for dataset, sample in samples.items():
        for dataset, sample in tqdm.tqdm(samples.items()):
            
            # testing
            # if ("dy_M-100To200" not in dataset) and ("data_A" not in dataset):
            # if ("dy_M-100To200" not in dataset) :
            # if ("ggh_powheg" not in dataset) and ("vbf_powheg" not in dataset):
            if ("ggh_powheg"  in dataset):
                continue
            # print(f"dataset: {dataset}")
            dataset_loop(coffea_processor, sample, test=test_mode)
    else:
        # xrootd_path = "root://eos.cms.rcac.purdue.edu/"
        # # fname = "/store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/230000/1A909DE6-CA08-434B-BDBB-B648B95BEFDF.root"
        # # fname = '/store/data/Run2018A/SingleMuon/NANOAOD/UL2018_MiniAODv2_NanoAODv9-v2/2550000/9DDF008C-B740-CA4D-B7EE-8E7E660FBD9A.root'
        # # fname = "/store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/40000/ECB51118-0153-2F40-BB6D-0204F0EE98C2.root" # "dy_M-50",
        # # fname = '/store/mc/RunIISummer20UL18NanoAODv9/GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/2810000/C4DAB63C-E2A1-A541-93A8-3F46315E362C.root' # ggh_powheg
        # fname ="/store/mc/RunIISummer20UL18NanoAODv9/VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/2820000/083C985C-C112-3B46-A053-D72C1F83309D.root" # vbf
        # dataset_loop(coffea_processor, xrootd_path+fname, test=test_mode)

        sample_path = "./config/fraction_processor_samples.json"
        with open(sample_path) as file:
            samples = json.loads(file.read())
        # print(f"samples.keys(): {samples.keys()}")

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