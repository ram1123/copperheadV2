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
# np.set_printoptions(threshold=sys.maxsize)

cluster_on = False # False
# test_size = 100
test_size = 100

xrootd_path = "root://eos.cms.rcac.purdue.edu/"
# fname = "/store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/230000/1A909DE6-CA08-434B-BDBB-B648B95BEFDF.root"
# fname = '/store/data/Run2018A/SingleMuon/NANOAOD/UL2018_MiniAODv2_NanoAODv9-v2/2550000/9DDF008C-B740-CA4D-B7EE-8E7E660FBD9A.root'
fname = "/store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/40000/ECB51118-0153-2F40-BB6D-0204F0EE98C2.root"

if __name__ == "__main__":
    time_step = time.time()
    # if local_cluster:
    #     # create local cluster
    #     cluster = Client(
    #         processes=True,
    #         n_workers=3,#int(args.num_workers), # 32
    #         #dashboard_address=dash_local,
    #         threads_per_worker=1,
    #         memory_limit="6GB",
    #     )
    #     print("Local Client created")
    # else:
    if cluster_on:
        from dask_gateway import Gateway
        gateway = Gateway()
        cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
        client = gateway.connect(cluster_info.name).get_client()
        print("Gateway Client created")
        #-----------------------------------------------------------
        # cluster = LocalCluster()
        # cluster.scale(16) # create 16 local workers
        # client = Client(cluster)
        # print("Local scale Client created")
    
    """
    Coffea Dask automatically uses the Dask Client that has been defined above
    """
    metadata = {"dataset": "dy_M-50", "is_mc": True}
    # metadata = {"dataset": "data_A", "is_mc": False}
    if cluster_on:
        print("cluster scale up")
        sample_path = "./config/processor_samples.json"
        with open(sample_path) as file:
            samples = json.loads(file.read())
        print(f"samples: {samples}")
        events = NanoEventsFactory.from_root(
            samples["files"],
            schemaclass=NanoAODSchema,
            metadata= metadata,
        ).events()
    else: # do it locally for testing
        print("local testing")
        np.random.seed(0) 
        events = NanoEventsFactory.from_root(
            {xrootd_path+fname : "Events"},
            schemaclass=NanoAODSchema,
            metadata= metadata,
            delayed= False,
            entry_stop = test_size,
        ).events()

    # high_precision = True
    # if high_precision:
    #     fields_to_change = ["pt","eta", "phi"]
    config_path = "./config/parameters.json"
    
    p = EventProcessor(config_path)
    print(f"copperhead2 run stage1 type(events): {type(events)}")
    out_collections = p.process(events)
    print(f"copperhead2 run stage1 out_collections b4 compute: {out_collections}")
    (computed, ) = dask.compute(out_collections)
    print(f"copperhead2 run stage1 computed after compute: {computed}")
    # print(f"copperhead2 run stage1 type(out): {type(result)}")

    placeholder =  pd.DataFrame({
                'mu1_pt': ak.to_numpy(computed["mu_pt"][:,0]),
                'mu2_pt': ak.to_numpy(computed["mu_pt"][:,1]),
                'mu1_eta': ak.to_numpy(computed["mu_eta"][:,0]),
                'mu2_eta': ak.to_numpy(computed["mu_eta"][:,1]),
                'mu1_phi': ak.to_numpy(computed["mu_phi"][:,0]),
                'mu2_phi': ak.to_numpy(computed["mu_phi"][:,1]),
                'mu1_iso': ak.to_numpy(computed["mu_iso"][:,0]),
                'mu2_iso': ak.to_numpy(computed["mu_iso"][:,1]),
                'mu1_pt_over_mass': ak.to_numpy(computed["mu_pt_over_mass"][:,0]),
                'mu2_pt_over_mass': ak.to_numpy(computed["mu_pt_over_mass"][:,1]),
                "dimuon_mass": ak.to_numpy(computed["dimuon_mass"]),
                "dimuon_ebe_mass_res": ak.to_numpy(computed["dimuon_ebe_mass_res"]),
                "dimuon_ebe_mass_res_rel": ak.to_numpy(computed["dimuon_ebe_mass_res_rel"]),
                "dimuon_pt": ak.to_numpy(computed["dimuon_pt"]),
                "dimuon_pt_log": ak.to_numpy(np.log(computed["dimuon_pt"])), # np functions are compatible with ak if input is ak array 
                "dimuon_eta": ak.to_numpy(computed["dimuon_eta"]),
                "dimuon_phi": ak.to_numpy(computed["dimuon_phi"]),
                "dimuon_dEta": ak.to_numpy(computed["dimuon_dEta"]),
                "dimuon_dPhi": ak.to_numpy(computed["dimuon_dPhi"]),
                "dimuon_dR": ak.to_numpy(computed["dimuon_dR"]),
                "dimuon_cos_theta_cs": ak.to_numpy(computed["dimuon_cos_theta_cs"]), 
                "dimuon_phi_cs": ak.to_numpy(computed["dimuon_phi_cs"]), 
                "gjj_mass":  ak.to_numpy(computed["gjj_mass"]),
                'gjet1_pt': ak.to_numpy(computed["gjet_pt"][:,0]),
                'gjet2_pt': ak.to_numpy(computed["gjet_pt"][:,1]),
                'gjet1_eta': ak.to_numpy(computed["gjet_eta"][:,0]),
                'gjet2_eta': ak.to_numpy(computed["gjet_eta"][:,1]),
                'gjet1_phi': ak.to_numpy(computed["gjet_phi"][:,0]),
                'gjet2_phi': ak.to_numpy(computed["gjet_phi"][:,1]),
                'gjet1_mass': ak.to_numpy(computed["gjet_mass"][:,0]),
                'gjet2_mass': ak.to_numpy(computed["gjet_mass"][:,1]),
                "gjj_dEta": ak.to_numpy(computed["gjj_dEta"]),
                "gjj_dPhi": ak.to_numpy(computed["gjj_dPhi"]),
                "gjj_dR": ak.to_numpy(computed["gjj_dR"]),
                # "gjet1_mass":  ak.to_numpy(gjet1.mass),
                # "gjet1_pt":  ak.to_numpy(gjet1.pt),
                # "gjet1_eta":  ak.to_numpy(gjet1.eta),
                # "gjet1_phi":  ak.to_numpy(gjet1.phi),
                # "gjet2_mass":  ak.to_numpy(gjet2.mass),
                # "gjet2_pt":  ak.to_numpy(gjet2.pt),
                # "gjet2_eta":  ak.to_numpy(gjet2.eta),
                # "gjet2_phi":  ak.to_numpy(gjet2.phi),
                # "gjj_dEta": ak.to_numpy(gjj_dEta),
                # "gjj_dPhi": ak.to_numpy(gjj_dPhi),
                # "gjj_dR": ak.to_numpy(gjj_dR),
                
            })
    # print(f"copperhead2 EventProcessor after leading pt cut placeholder: \n {placeholder.to_string()}")
    placeholder.to_csv("./V2placeholder.csv")

    
    elapsed = round(time.time() - time_step, 3)
    print(f"Finished everything in {elapsed} s.")