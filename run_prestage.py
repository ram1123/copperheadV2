import awkward as ak
from coffea.dataset_tools import rucio_utils
from coffea.dataset_tools.preprocess import preprocess
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
import json
import os
import argparse
from distributed import LocalCluster, Client
import time
import copy

datasets = {
    "2016preVFP": {
        "data_B": "/SingleMuon/Run2016B-ver2_HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD",
        "data_C": "/SingleMuon/Run2016C-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD",
        "data_D": "/SingleMuon/Run2016D-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD",
        "data_E": "/SingleMuon/Run2016E-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD",
        "data_F": "/SingleMuon/Run2016F-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD",
        "dy_M-50": "/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "dy_M-100To200": "/DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "ttjets_dl": "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "ttjets_sl": "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        # "ttw": "",
        # "ttz": "",
        "st_tw_top": "/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "st_tw_antitop": "/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM", 
        "ww_2l2nu": "/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "wz_3lnu": "/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "wz_2l2q": "/WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM",
        "wz_1l1nu2q": "/WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "zz": "/ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        # "www": "",
        # "wwz": "",
        # "wzz": "",
        # "zzz": "",
        "ewk_lljj_mll50_mjj120": "/EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "ggh_powheg":"/GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        # "ggh_amc": "",
        # "ggh_amcPS": "",
        # "ggh_powhegPS": "",
        "vbf_powheg": "/VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        # "vbf_powhegPS": "",
        # "vbf_powheg_herwig": "",
        # "vbf_powheg_dipole": "",
        # "wmh": "",
        # "wph": "",
        # "tth": "",
        # "zh": "",
    },
    "2016postVFP": {
        "data_F": "/SingleMuon/Run2016F-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD",
        "data_G": "/SingleMuon/Run2016G-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD",
        "data_H": "/SingleMuon/Run2016H-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD",
        "dy_M-50": "/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "dy_M-100To200": "/DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
        "ttjets_dl": "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "ttjets_sl": "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        # "ttw": "",
        # "ttz": "",
        "st_tw_top":"/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "st_tw_antitop":"/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM", 
        "ww_2l2nu": "/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "wz_3lnu": "/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "wz_2l2q": "/WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
        "wz_1l1nu2q": "/WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "zz": "/ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        # "www": "",
        # "wwz": "",
        # "wzz": "",
        # "zzz": "",
        "ewk_lljj_mll50_mjj120": "/EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "ggh_powheg":"/GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        # "ggh_amc": "",
        # "ggh_amcPS": "",
        # "ggh_powhegPS": "",
        "vbf_powheg": "/VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        # "vbf_powhegPS": "",
        # "vbf_powheg_herwig": "",
        # "vbf_powheg_dipole": "",
        # "wmh": "",
        # "wph": "",
        # "tth": "",
        # "zh": "",
    },
    "2017": {
        "data_B": "/SingleMuon/Run2017B-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD",
        "data_C": "/SingleMuon/Run2017C-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD",
        "data_D": "/SingleMuon/Run2017D-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD",
        "data_E": "/SingleMuon/Run2017E-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD",
        "data_F": "/SingleMuon/Run2017F-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD",
        "dy_M-50": "/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "dy_M-100To200": "/DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "ttjets_dl": "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "ttjets_sl": "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        # "ttw": "",
        # "ttz": "",
        "st_tw_top":"/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "st_tw_antitop":"/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM", 
        "ww_2l2nu": "/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "wz_3lnu": "/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "wz_2l2q": "/WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "wz_1l1nu2q": "/WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "zz": "/ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        # "www": "",
        # "wwz": "",
        # "wzz": "",
        # "zzz": "",
        "ewk_lljj_mll50_mjj120": "/EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "ggh_powheg":"/GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        # "ggh_amc": "",
        # "ggh_amcPS": "",
        # "ggh_powhegPS": "",
        "vbf_powheg": "/VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        # "vbf_powhegPS": "",
        # "vbf_powheg_herwig": "",
        # "vbf_powheg_dipole": "",
    },
    "2018": {
        "data_A": "/SingleMuon/Run2018A-UL2018_MiniAODv2_NanoAODv9-v2/NANOAOD",
        "data_B": "/SingleMuon/Run2018B-UL2018_MiniAODv2_NanoAODv9-v2/NANOAOD",
        "data_C": "/SingleMuon/Run2018C-UL2018_MiniAODv2_NanoAODv9-v2/NANOAOD",
        "data_D": "/SingleMuon/Run2018D-UL2018_MiniAODv2_NanoAODv9-v1/NANOAOD",
        "dy_M-50": "/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X*/NANOAODSIM",
        "dy_M-100To200": "/DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        # "ttjets_dl": "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        # "ttjets_sl": "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        # # "ttw": "",
        # # "ttz": "",
        # "st_tw_top":"/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        # "st_tw_antitop":"/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM", 
        # "ww_2l2nu": "/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
        # "wz_3lnu": "/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
        # "wz_2l2q": "/WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        # "wz_1l1nu2q": "/WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        # "zz": "/ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        # # "www": "",
        # # "wwz": "",
        # # "wzz": "",
        # # "zzz": "",
        # "ewk_lljj_mll50_mjj120": "/EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "ggh_powheg":"/GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        # # "ggh_amc": "",
        # # "ggh_amcPS": "",
        # # "ggh_powhegPS": "",
        "vbf_powheg": "/VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        # # "vbf_powhegPS": "",
        # # "vbf_powheg_herwig": "",
        # # "vbf_powheg_dipole": "",
    },
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="2018",
    action="store",
    help="year",
    )
    parser.add_argument(
    "-c",
    "--cluster",
    dest="cluster",
    default=None,
    action="store",
    help="if not None, use distributed cluster",
    )
    parser.add_argument(
    "-ch",
    "--chunksize",
    dest="chunksize",
    default="10000",
    action="store",
    help="chunksize",
    )
    parser.add_argument(
    "-frac",
    "--change_fraction",
    dest="fraction",
    default=None,
    action="store",
    help="change fraction of steps of the data",
    )
    args = parser.parse_args()
    time_step = time.time()
    
    if args.fraction is None: # do the normal prestage setup
        allowlist_sites=["T2_US_Purdue"] # take data only from purdue for now
        total_events = 0
        # get dask client
        if args.cluster is not None:
            from dask_gateway import Gateway
            gateway = Gateway()
            cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
            client = gateway.connect(cluster_info.name).get_client()
            print("Gateway Client created")
        else: # use local cluster
            cluster = LocalCluster()
            cluster.scale(32) 
            client = Client(cluster)
            print("Local scale Client created")
        big_sample_info = {}
        dataset = datasets[args.year]
    
        for sample_name in dataset.keys():
            print(f"prestage sample_name: {sample_name}")
            # # test
            # if "dy_M-50" not in sample_name:
            #     continue
            
            das_query = dataset[sample_name]
            
            
            client = rucio_utils.get_rucio_client()
            outlist, outtree = rucio_utils.query_dataset(
                das_query,
                client=client,
                tree=True,
                scope="cms",
            )
            outfiles,outsites,sites_counts =rucio_utils.get_dataset_files_replicas(
                outlist[0],
                allowlist_sites=allowlist_sites,
                mode="full",
                client=client,
                partial_allowed=True
            )
            fnames = [file[0] for file in outfiles if file != []]
            # print(f"prestage fnames: {fnames}")
            print(f"prestage len(fnames): {len(fnames)}")
            
            """
            run through each file and collect total number of 
            """
            preprocess_metadata = {
                "sumGenWgts" : None,
                "nGenEvts" : None,
                "data_entries" : None,
            }
            if "data" in sample_name: # data sample
                file_input = {fname : {"object_path": "Events"} for fname in fnames}
                events = NanoEventsFactory.from_root(
                        file_input,
                        metadata={},
                        schemaclass=NanoAODSchema,
                ).events()
                preprocess_metadata["data_entries"] = int(ak.num(events.Muon.pt, axis=0).compute()) # convert into 32bit precision as 64 bit precision isn't json serializable
                total_events += preprocess_metadata["data_entries"] 
            else:
                file_input = {fname : {"object_path": "Runs"} for fname in fnames}
                runs = NanoEventsFactory.from_root(
                        file_input,
                        metadata={},
                        schemaclass=BaseSchema,
                ).events()
                
                preprocess_metadata["sumGenWgts"] = float(ak.sum(runs.genEventSumw).compute()) # convert into 32bit precision as 64 bit precision isn't json serializable
                preprocess_metadata["nGenEvts"] = int(ak.sum(runs.genEventCount).compute()) # convert into 32bit precision as 64 bit precision isn't json serializable
                total_events += preprocess_metadata["nGenEvts"] 
            print(f"prestage preprocess_metadata: {preprocess_metadata}")    
    
            val = "Events"
            file_dict = {}
            for file in fnames:
                file_dict[file] = val
            # final_output = {"files" :file_dict, "metadata" : "MC"}
            # final_output = {sample_name :final_output}
            final_output = {
                sample_name :{"files" :file_dict}
            }
            
            step_size = int(args.chunksize)
            files_available, files_total = preprocess(
                final_output,
                maybe_step_size=step_size,
                align_clusters=False,
                skip_bad_files=True,
            )
            
            pre_stage_data = files_available
            # add in metadata
            pre_stage_data[sample_name]['metadata'] = preprocess_metadata
            # add in faction -> for later use
            pre_stage_data[sample_name]['metadata']['fraction'] = 1.0
            # if preprocess_metadata["data_entries"] is not None: # Data
            if "data" in sample_name: # data sample
                pre_stage_data[sample_name]['metadata']["is_mc"] = False
            else: # MC
                pre_stage_data[sample_name]['metadata']["is_mc"] = True
            pre_stage_data[sample_name]['metadata']["dataset"] = sample_name
            big_sample_info.update(pre_stage_data)
        
        #save the sample info
        directory = "./config"
        filename = directory+"/processor_samples.json"
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filename, "w") as file:
                json.dump(big_sample_info, file)
    
        elapsed = round(time.time() - time_step, 3)
        print(f"Finished everything in {elapsed} s.")
        print(f"Total Events to go thru {total_events}.")
        
    else: # take the pre existing samples.json and prune off files we don't need
        fraction = float(args.fraction)
        sample_path = "./config/processor_samples.json"
        with open(sample_path) as file:
            samples = json.loads(file.read())
        new_samples = copy.deepcopy(samples) # copy old sample, overwrite it later
        # new_samples = {}
        for sample_name, sample in samples.items():
            is_data = "data" in sample_name
            tot_N_evnts = sample['metadata']["data_entries"] if is_data else sample['metadata']["nGenEvts"]
            new_N_evnts = int(tot_N_evnts*fraction)
            print(f"datset {sample_name} new_N_evnts: {new_N_evnts} ")
            # new_samples[sample_name] = {
            #     "metadata" : sample["metadata"] # copy old metadata for now, overwrite it later
            # }
            if is_data:
                new_samples[sample_name]['metadata']["data_entries"] = new_N_evnts
            else:
                new_samples[sample_name]['metadata']["nGenEvts"] = new_N_evnts
                new_samples[sample_name]['metadata']["sumGenWgts"] *= fraction # just directly multiply by fraction for this since this is already float
            new_samples[sample_name]['metadata']["fraction"] = fraction
            
            print(f"new_samples[{sample_name}]: {new_samples[sample_name].keys()}")
            # loop through the files to correct the steps
            event_counter = 0 # keeps track of events of multiple root files
            stop_flag = False
            new_files = {}
            # for file in sample["files"]:
            for file, file_dict in sample["files"].items():
                print(f"stop_flag: {stop_flag}")
                if stop_flag:
                    del new_samples[sample_name]["files"][file] # delete the exess files
                    continue
                new_steps = []
                # loop through step sizes to correct it
                for step_iteration in file_dict["steps"]:
                # for i in range(len(file["steps"])):
                #     step_iteration= file["steps"][i]
                    new_step_lim = new_N_evnts-event_counter
                    if step_iteration[1] < new_step_lim:
                        new_steps.append(step_iteration)
                    else:  # change the upper limit
                        new_steps.append([
                            step_iteration[0],
                            new_step_lim
                        ])
                        stop_flag = True
                        print(f'event_counter+new_step_lim : {event_counter+new_step_lim}')
                        break
                # print(f'new_samples[sample_name]["files"].keys(): {new_samples[sample_name]["files"].keys()}')
                new_samples[sample_name]["files"][file]["steps"] = new_steps # overwrite new steps
                # add the end step val to the event_counter
                if not stop_flag: # update variables and move to next file
                    end_idx = len(file_dict["steps"])-1
                    event_counter += file_dict["steps"][end_idx][1]
                
                    

            
        #save the sample info
        directory = "./config"
        filename = directory+"/processor_samples.json"
        with open(filename, "w") as file:
                json.dump(new_samples, file)
    
        elapsed = round(time.time() - time_step, 3)
        print(f"Finished everything in {elapsed} s.")


