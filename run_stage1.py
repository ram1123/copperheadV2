from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from copperhead_processor import EventProcessor
# NanoAODSchema.warn_missing_crossrefs = False
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import dask
from dask.distributed import Client
import sys
np.set_printoptions(threshold=sys.maxsize)

cluster_on = False
test_size = 50
# test_size = 10000

xrootd_path = "root://eos.cms.rcac.purdue.edu/"
# fname = "/store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/230000/1A909DE6-CA08-434B-BDBB-B648B95BEFDF.root"
fname = "/store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/40000/ECB51118-0153-2F40-BB6D-0204F0EE98C2.root"

if __name__ == "__main__":
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
        # connect to existing Slurm cluster
        # parameters["client"] = Client(parameters["slurm_cluster_ip"])
        from dask_gateway import Gateway
        gateway = Gateway()
        cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
        client = gateway.connect(cluster_info.name).get_client()
        print("Gateway Client created")
    
    """
    Coffea Dask automatically uses the Dask Client that has been defined above
    """
    metadata = {"dataset": "dy_M-50"}
    if cluster_on:
        events = NanoEventsFactory.from_root(
            {xrootd_path+fname : {"object_path": "Events", "steps": [[0, 50]]}},
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
    
    p = EventProcessor()
    print(f"copperhead2 run stage1 type(events): {type(events)}")
    out_collections = p.process(events)
    result = dask.compute(out_collections)
    print(f"copperhead2 run stage1 type(out): {type(result)}")
