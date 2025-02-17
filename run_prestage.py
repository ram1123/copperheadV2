import awkward as ak
from coffea.dataset_tools import rucio_utils
from coffea.dataset_tools.preprocess import preprocess
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
import json
import os
import argparse
import dask
dask.config.set({'logging.distributed': 'error'})
from distributed import LocalCluster, Client
import time
import copy
import tqdm
import uproot
import random
# random.seed(9002301)
import re
import glob
# import warnings
# warnings.filterwarnings("error", module="coffea.*")
from omegaconf import OmegaConf
import sys

import logging
from modules.utils import logger

def get_Xcache_filelist(fnames: list):
    new_fnames = []
    for fname in fnames:
        root_file = re.findall(r"/store.*", fname)[0]
        x_cache_fname = "root://cms-xcache.rcac.purdue.edu/" + root_file
        new_fnames.append(x_cache_fname)
    return new_fnames

def find_keys_in_yaml(yaml_data, keys_to_find):
    """
    Recursively searches for specific keys in a nested OmegaConf YAML structure.
    """
    found_values = {}

    def recursive_search(data, parent_key=""):
        if isinstance(data, dict) or OmegaConf.is_dict(data):
            for key, value in data.items():
                if key in keys_to_find:
                    found_values[key] = value
                recursive_search(value, key)
        elif isinstance(data, list) or OmegaConf.is_list(data):
            for item in data:
                recursive_search(item, parent_key)

    recursive_search(yaml_data)
    return found_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="2018",
    action="store",        
    help="year value. The options are: 2016preVFP, 2016postVFP, 2017, 2018",
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
        "--yaml",
        dest="dataset_yaml_file",
        default="configs/datasets/dataset.yaml",
        help="path of yaml file containing the dataset names"
    )
    parser.add_argument(
    "-frac",
    "--change_fraction",
    dest="fraction",
    default=None,
    action="store",
    help="change fraction of steps of the data",
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
    parser.add_argument(
    "--use_gateway",
    dest="use_gateway",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, uses dask gateway client instead of local",
    )
    parser.add_argument(
    "--xcache",
    dest="xcache",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, uses xcache root file paths",
    )
    parser.add_argument(
    "--skipBadFiles",
    dest="skipBadFiles",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, uses skips bad files when calling preprocessing",
    )
    parser.add_argument(
    "-aod_v",
    "--NanoAODv",
    type=int,
    dest="NanoAODv",
    default=9,
    choices = [9, 12],
    help="version number of NanoAOD samples we're working with. currently, only 9 and 12 are supported",
    )
    parser.add_argument( # temp flag to test the 2 percent data discrepancy in ggH cat between mine and official workspace
    "--run2_rereco",
    dest="run2_rereco",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, uses skips bad files when calling preprocessing",
    )
    parser.add_argument(
     "--log-level",
     default=logging.ERROR,
     type=lambda x: getattr(logging, x),
     help="Configure the logging level."
     )
    args = parser.parse_args()
    time_step = time.time()

    logger.setLevel(args.log_level)
    
    os.environ['XRD_REQUESTTIMEOUT']="2400" # some root files via XRootD may timeout with default value

    year = args.year
    logger.info(f"year: {year}")
    
    if args.fraction is None: # do the normal prestage setup
        total_events = 0
        # get dask client
        if args.use_gateway:
            from dask_gateway import Gateway
            gateway = Gateway(
                "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
                proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
            )
            cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
            client = gateway.connect(cluster_info.name).get_client()
            logger.debug("Gateway Client created")
        else: # use local cluster
            cluster = LocalCluster(processes=True)
            cluster.adapt(minimum=8, maximum=31) #min: 8 max: 32
            client = Client(cluster)
            # client = Client(n_workers=15,  threads_per_worker=1, processes=True, memory_limit='30 GiB')
            logger.info("Local scale Client created")

        big_sample_info = {}
        
        # load dataset sample paths from yaml files
        datasets = OmegaConf.load(args.dataset_yaml_file)
        if args.run2_rereco: # temp condition for RERECO data case
            dataset = datasets.years[f"{year}_RERECO"]
        else: # normal
            dataset = datasets.years[f"{year}"]
        new_sample_list = []

        logger.info(f'dataset: {dataset}')
        
        # take data and add to the list: `new_sample_list[]`
        logger.debug(f'data: {dataset["Data"].keys()}')
        data_l =  [sample_name for sample_name in dataset['Data'].keys() if "data" in sample_name]
        logger.debug(data_l)
        data_samples = args.data_samples
        logger.info(f"data_samples: {data_samples}")
        
        if len(data_samples) >0:
            for data_letter in data_samples:
                for sample_name in data_l:
                    if data_letter in sample_name:
                        logger.debug(sample_name)
                        new_sample_list.append(sample_name)
                        
        logger.debug(f"new sample list: {new_sample_list}")
        
        # take bkg and add to the list: `new_sample_list[]`
        bkg_l = [sample_name for sample_name in dataset.keys() if "data" not in sample_name.lower()]
        logger.info(f"background samples: {bkg_l}")
        bkg_samples = args.bkg_samples
        logger.info(f"bkg_samples: {bkg_samples}")
        if len(bkg_samples) >0:
            for bkg_letter in bkg_samples:
                for bkg_name in bkg_l:
                    if bkg_letter in bkg_name:
                        for bkgs in dataset[bkg_name].keys():
                            new_sample_list.append(bkgs)
                        
        logger.debug(f"new sample list: {new_sample_list}")

        # take sig and add to the list: `new_sample_list[]`
        sig_samples = args.sig_samples
        logger.info(f"signal samples: {sig_samples}")
        if len(sig_samples) >0:
            for sig_sample in sig_samples: # FIXME: Why custom? We should just read from YAML file. No hardcoding needed here.
                if sig_sample.upper() == "GGH": # enforce upper case to prevent confusion
                    new_sample_list.append("ggh_powhegPS")
                elif sig_sample.upper() == "VBF": # enforce upper case to prevent confusion
                    new_sample_list.append("vbf_powheg_dipole")
                else:
                    logger.debug(f"unknown signal {sig_sample} was given!")

        logger.debug(f"Sample list: {new_sample_list}")

        dataset_dict = {}
        for sample_name_temp in new_sample_list:
            try:
                # dataset_dict[sample_name_temp] = dataset[sample_name_temp]
                temp_dict = find_keys_in_yaml(dataset, sample_name_temp)
                dataset_dict.update(temp_dict)
            except Exception as e:
                logger.error(f"Sample {sample_name} gives error {e}. Skipping")
                continue
        dataset = dataset_dict # overwrite dataset bc we don't need it anymore
        logger.debug(f"dataset: {dataset}")
        logger.debug(f"dataset keys: {dataset.keys()}")
        logger.debug(f"year: {year}")
        logger.debug(f"type(year): {type(year)}")
        logger.debug(f"Is run2_rereco: {args.run2_rereco}")

        fnames = ""

        for sample_name in tqdm.tqdm(dataset.keys()):
            is_data =  ("data" in sample_name)
            logger.debug(f"Sample Name: {sample_name}")
            logger.debug(f"is data?: {is_data}")
            # FIXME: This is temporary overwrite of samples that are private (not supported by rucio yet), thus the das_query is "dummy" in dataset yml files
            if sample_name == "dy_VBF_filter":
                """
                temporary condition bc IDK a way to integrate prod/phys03 CMSDAS datasets into rucio utils
                """
                load_path = "/eos/purdue/store/user/vscheure/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/UL18_Nano/240514_124107/"
                fnames = glob.glob(f"{load_path}/*/*.root")

            elif sample_name == "dy_m105_160_vbf_amc":
                """
                load directly from local files
                """
                load_path = "/eos/purdue/store/mc/RunIIAutumn18NanoAODv6/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/NANOAODSIM"
                fnames = glob.glob(f"{load_path}/*/*/*.root")
                
            elif sample_name == "dy_VBF_filter_fromGridpack":
                """
                load directly from local files
                """
                load_path = "/eos/purdue/store/user/hyeonseo/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/Flat_NanoAODSIMv9_CMSSW_10_6_26_BigRun/240904_151935/0000/"
                fnames = glob.glob(f"{load_path}/*.root")
            elif (year == "2016postVFP") and (sample_name == "dy_M-100To200"): # temp overwrite bc external xrootD issues
                load_path = "/eos/purdue/store/mc/RunIISummer20UL16NanoAODv9/DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v2/*/"
                fnames = glob.glob(f"{load_path}/*.root")
            elif year == "2018":
                if (sample_name == "dy_m105_160_vbf_amc"): # temporary overwrite for BDT input test Nov 14 2024
                    load_path = "/eos/purdue/store/mc/RunIIAutumn18NanoAODv6/DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/"
                    fnames = glob.glob(f"{load_path}/*/*/*/*.root")
                elif (sample_name == "ewk_lljj_mll105_160_ptj0"):
                    load_path = "/eos/purdue/store/group/local/hmm/FSRnano18MC_NANOV10b/EWK_LLJJ_MLL_105-160_SM_5f_LO_TuneCH3_13TeV-madgraph-herwig7_corrected/" # taken from RERECO
                    fnames = glob.glob(f"{load_path}/*/*/*/*.root")
                elif (sample_name == "ewk_lljj_mll105_160_py_dipole"):
                    load_path = "/eos/purdue/store/group/local/hmm/FSRnano18MC_NANOV10c/EWK_LLJJ_MLL_105-160_TuneCP5_13TeV-madgraph-pythia_dipole/" # taken from RERECO
                    fnames = glob.glob(f"{load_path}/*/*/*/*.root")
            elif year == "2017_RERECO":
                if sample_name == "data_B":
                    load_path = "/eos/purdue/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017B-31Mar2018-v1/"
                    fnames = glob.glob(f"{load_path}/*/*/*.root")
                elif sample_name == "data_C":
                    load_path = "/eos/purdue/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017C-31Mar2018-v1/"
                    fnames = glob.glob(f"{load_path}/*/*/*.root")
                elif sample_name == "data_D":
                    load_path = "/eos/purdue/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017D-31Mar2018-v1/"
                    fnames = glob.glob(f"{load_path}/*/*/*.root")
                elif sample_name == "data_E":
                    load_path = "/eos/purdue/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017E-31Mar2018-v1/"
                    fnames = glob.glob(f"{load_path}/*/*/*.root")
                elif sample_name == "data_F":
                    load_path = "/eos/purdue/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017F-31Mar2018-v1/"
                    fnames = glob.glob(f"{load_path}/*/*/*.root")

                    bad_files = [ # this is obtained from quick_tests/quick_bad_fil_collector.ipynb
                        "/eos/purdue/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017F-31Mar2018-v1/191007_095748/0001/myNanoProdData2017_NANO_1634.root",
"/eos/purdue/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017F-31Mar2018-v1/191007_095748/0001/myNanoProdData2017_NANO_1635.root",
"/eos/purdue/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017F-31Mar2018-v1/191007_095748/0001/myNanoProdData2017_NANO_1636.root",
"/eos/purdue/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017F-31Mar2018-v1/191007_095748/0001/myNanoProdData2017_NANO_1637.root",
"/eos/purdue/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017F-31Mar2018-v1/191007_095748/0001/myNanoProdData2017_NANO_1638.root",
"/eos/purdue/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017F-31Mar2018-v1/191007_095748/0001/myNanoProdData2017_NANO_1639.root",
"/eos/purdue/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017F-31Mar2018-v1/191007_095748/0001/myNanoProdData2017_NANO_1640.root",
"/eos/purdue/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017F-31Mar2018-v1/191007_095748/0001/myNanoProdData2017_NANO_1641.root",
"/eos/purdue/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017F-31Mar2018-v1/191007_095748/0001/myNanoProdData2017_NANO_1642.root",
"/eos/purdue/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017F-31Mar2018-v1/191007_095748/0001/myNanoProdData2017_NANO_1643.root",
                    ]
                    fnames = set(fnames)
                    bad_files = set(bad_files)
                    fnames = list(fnames.difference(bad_files)) # remove bad files from fnames and turn it back to a list
                elif sample_name == "ggh_amcPS": # actually amcPS
                    load_path = "/eos/purdue/store/group/local/hmm/FSRmyNanoProdMc2017_NANOV8a/GluGluHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8/"
                    fnames = glob.glob(f"{load_path}/*/*/*/*.root")
                elif sample_name == "vbf_powheg":
                    load_path = "/eos/purdue/store/group/local/hmm/FSRmyNanoProdMc2017_NANOV8a/VBFHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/" # technically vbf_powhegPS
                    fnames = glob.glob(f"{load_path}/*/*/*/*.root")

            elif year == "2016_RERECO":
                logger.debug("2016_RERECO !")
                logger.debug(f"sample_name: {sample_name}")
                if sample_name == "data_B":
                    load_path = "/eos/purdue/store/group/local/hmm/FSRNANO2016DATAV8a/SingleMuon/RunIIData17_FSRNANO2016DATAV8a_un2016B-17Jul2018_ver2-v1/"
                    fnames = glob.glob(f"{load_path}/*/*/*.root")
                elif sample_name == "data_C":
                    load_path = "/eos/purdue/store/group/local/hmm/FSRNANO2016DATAV8a/SingleMuon/RunIIData17_FSRNANO2016DATAV8a_Run2016C-17Jul2018-v1/"
                    fnames = glob.glob(f"{load_path}/*/*/*.root")
                elif sample_name == "data_D":
                    load_path = "/eos/purdue/store/group/local/hmm/FSRNANO2016DATAV8a/SingleMuon/RunIIData17_FSRNANO2016DATAV8a_Run2016D-17Jul2018-v1/"
                    fnames = glob.glob(f"{load_path}/*/*/*.root")
                elif sample_name == "data_E":
                    load_path = "/eos/purdue/store/group/local/hmm/FSRNANO2016DATAV8a/SingleMuon/RunIIData17_FSRNANO2016DATAV8a_Run2016E-17Jul2018-v1/"
                    fnames = glob.glob(f"{load_path}/*/*/*.root")
                elif sample_name == "data_F":
                    load_path = "/eos/purdue/store/group/local/hmm/FSRNANO2016DATAV8a/SingleMuon/RunIIData17_FSRNANO2016DATAV8a_Run2016F-17Jul2018-v1/"
                    fnames = glob.glob(f"{load_path}/*/*/*.root")
                elif sample_name == "data_G":
                    load_path = "/eos/purdue/store/group/local/hmm/FSRNANO2016DATAV8a/SingleMuon/RunIIData17_FSRNANO2016DATAV8a_Run2016G-17Jul2018-v1/"
                    fnames = glob.glob(f"{load_path}/*/*/*.root")
                elif sample_name == "data_H":
                    load_path = "/eos/purdue//store/group/local/hmm/FSRNANO2016DATAV8a/SingleMuon/RunIIData17_FSRNANO2016DATAV8a_Run2016H-17Jul2018-v1/"
                    fnames = glob.glob(f"{load_path}/*/*/*.root")
                elif sample_name == "ggh_amcPS": # actually amcPS
                    load_path = "/eos/purdue/store/group/local/hmm/FSRNANO2016MCV8c/GluGluHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8/"
                    fnames = glob.glob(f"{load_path}/*/*/*/*.root")
                elif sample_name == "vbf_powheg":
                    load_path = "/eos/purdue/store/group/local/hmm/FSRNANO2016MCV8a_06May2020/VBF_HToMuMu_M125_13TeV_powheg_pythia8/"
                    fnames = glob.glob(f"{load_path}/*/*/*/*.root")
            
            if year == '2018' and sample_name == "ggh_powhegPS":
                load_path = '/eos/purdue/store/mc/RunIISummer20UL18NanoAODv9/GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/'
                fnames = glob.glob(f"{load_path}/*/*.root")
            
            # This is the default method
            das_query = dataset[sample_name]
            logger.debug(f"das query: {das_query}")

            if "dummy" not in das_query:
                allowlist_sites=["T2_US_Purdue", "T2_US_MIT","T2_US_FNAL"]        
                rucio_client = rucio_utils.get_rucio_client() # INFO: Why rucio?
                try:
                    # Use the rucio_client
                    outlist, outtree = rucio_utils.query_dataset(
                        das_query,
                        client=rucio_client,
                        tree=True,
                        scope="cms",
                    )
                    outfiles, outsites, sites_counts = rucio_utils.get_dataset_files_replicas(
                        outlist[0],
                        allowlist_sites=allowlist_sites,
                        mode="full",
                        client=rucio_client,
                    )
                finally:
                    # Close the session
                    await rucio_client.close()
                fnames = [file[0] for file in outfiles if file != []]
                # fnames = [fname.replace("root://eos.cms.rcac.purdue.edu/", "/eos/purdue") for fname in fnames] # replace xrootd prefix bc it's causing file not found error
                # random.shuffle(fnames)
                if args.xcache:
                    fnames = get_Xcache_filelist(fnames)
                # logger.debug(f"fnames: {fnames}")
                        
            logger.debug(f"file names: {fnames}")
            logger.debug(f"sample_name: {sample_name}")
            logger.debug(f"das_query: {das_query}")
            logger.debug(f"len(fnames): {len(fnames)}")
            logger.debug(f"file names: {fnames}")
            
            # fnames = [fname.replace("/eos/purdue", "root://eos.cms.rcac.purdue.edu/") for fname in fnames] # replace to xrootd bc sometimes eos mounts timeout when reading 

            """
            run through each file and collect sumGenWeights, total number of events, etc.
            """
            preprocess_metadata = {
                "sumGenWgts" : None,
                "nGenEvts" : None,
                "data_entries" : None,
            }
            if is_data: # data sample
                file_input = {fname : {"object_path": "Events"} for fname in fnames}
                logger.debug(f"file_input: {file_input}")
                events = NanoEventsFactory.from_root(
                        file_input,
                        metadata={},
                        schemaclass=NanoAODSchema,
                        uproot_options={"timeout":2400},
                ).events()
                logger.debug(f"file_input: {file_input}")
                logger.debug(f"events.fields: {events.fields}")
                preprocess_metadata["data_entries"] = int(ak.num(events.Muon.pt, axis=0).compute()) # convert into 32bit precision as 64 bit precision isn't json serializable
                total_events += preprocess_metadata["data_entries"] 
            else: # if MC
                file_input = {fname : {"object_path": "Runs"} for fname in fnames}
                logger.debug(f"file_input: {file_input}")
                # logger.debug(f"file_input: {file_input}")
                # logger.debug(len(file_input.keys()))
                runs = NanoEventsFactory.from_root(
                        file_input,
                        metadata={},
                        schemaclass=BaseSchema,
                        uproot_options={"timeout":2400},
                ).events()               
                if "genEventSumw" in runs.fields:
                    preprocess_metadata["sumGenWgts"] = float(ak.sum(runs.genEventSumw).compute()) # convert into 32bit precision as 64 bit precision isn't json serializable
                    preprocess_metadata["nGenEvts"] = int(ak.sum(runs.genEventCount).compute()) # convert into 32bit precision as 64 bit precision isn't json serializable
                else: # nanoAODv6 
                    preprocess_metadata["sumGenWgts"] = float(ak.sum(runs.genEventSumw_).compute()) # convert into 32bit precision as 64 bit precision isn't json serializable
                    preprocess_metadata["nGenEvts"] = int(ak.sum(runs.genEventCount_).compute()) # convert into 32bit precision as 64 bit precision isn't json serializable
                                    
                total_events += preprocess_metadata["nGenEvts"] 

    
            # QUESTION: WHY THIS IS NOT ABOVE?
            # test start -------------------------------
            if sample_name == "dy_VBF_filter":
                """
                Starting from coffea 2024.4.1, this if statement is technically as obsolite preprocess
                can now handle thousands of root files no problem, but this "manual" is at least three 
                times faster than preprocess, so keeping this if statement for now
                """
                runs = NanoEventsFactory.from_root(
                        file_input,
                        metadata={},
                        schemaclass=BaseSchema,
                        uproot_options={"timeout":2400},
                ).events()  
                genEventCount = runs.genEventCount.compute()
                # logger.debug(f"(genEventCount): {(genEventCount)}")
                # logger.debug(f"len(genEventCount): {len(genEventCount)}")
                
                assert len(fnames) == len(genEventCount)
                file_dict = {}
                for idx in range(len(fnames)):
                    step_start = 0
                    step_end = int(genEventCount[idx]) # convert into 32bit precision as 64 bit precision isn't json serializable
                    file = fnames[idx]
                    file_dict[file] = {
                        "object_path": "Events",
                        "steps" : [[step_start,step_end]],
                        "num_entries" : step_end, 
                    }
                final_output = {
                    sample_name :{"files" :file_dict}
                }
                # logger.debug(f"final_output: {final_output}")
                pre_stage_data = final_output
            else:
                """
                everything else other than VBF filter, but starting from coffea 2024.4.1 is condition could
                techincally work on VBF filter files, just slower
                """
                val = "Events"
                file_dict = {}
                for file in fnames:
                    file_dict[file] = val
                final_output = {
                    sample_name :{"files" :file_dict}
                }
                # logger.debug(f"final_output: {final_output}")
                step_size = int(args.chunksize)
                files_available, files_total = preprocess(
                    final_output,
                    step_size=step_size,
                    align_clusters=False,
                    skip_bad_files=args.skipBadFiles,
                )
                # logger.debug(f"files_available: {files_available}")
                pre_stage_data = files_available

            # test end2  --------------------------------------------------------------
            # add in metadata
            pre_stage_data[sample_name]['metadata'] = preprocess_metadata
            # add in faction -> for later use
            pre_stage_data[sample_name]['metadata']['fraction'] = 1.0
            pre_stage_data[sample_name]['metadata']['original_fraction'] = 1.0
            # if preprocess_metadata["data_entries"] is not None: # Data
            if "data" in sample_name: # data sample
                pre_stage_data[sample_name]['metadata']["is_mc"] = False
            else: # MC
                pre_stage_data[sample_name]['metadata']["is_mc"] = True
            pre_stage_data[sample_name]['metadata']["dataset"] = sample_name
            big_sample_info.update(pre_stage_data)
        
        #save the sample info
        directory = "./prestage_output"
        filename = directory+"/processor_samples.json"
        # dupli_fname = directory+"/fraction_processor_samples.json" # duplicated fname in case you want to skip fractioning
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filename, "w") as file:
                json.dump(big_sample_info, file)
    
        elapsed = round(time.time() - time_step, 3)
        logger.info(f"Finished everything in {elapsed} s.")
        logger.info(f"Total Events in files {total_events}.")
        
    else: # take the pre existing samples.json and prune off files we don't need
        fraction = float(args.fraction)
        directory = "./prestage_output"
        sample_path = directory + "/processor_samples.json"
        with open(sample_path) as file:
            samples = json.loads(file.read())
        new_samples = copy.deepcopy(samples) # copy old sample, overwrite it later
        if fraction < 1.0: # else, just save the original samples and new samples
            for sample_name, sample in tqdm.tqdm(samples.items()):
                is_data = "data" in sample_name
                tot_N_evnts = sample['metadata']["data_entries"] if is_data else sample['metadata']["nGenEvts"]
                new_N_evnts = int(tot_N_evnts*fraction)
                old_N_evnts = new_samples[sample_name]['metadata']["data_entries"] if is_data else new_samples[sample_name]['metadata']["nGenEvts"]
                if is_data:
                    logger.debug("data!")
                    new_samples[sample_name]['metadata']["data_entries"] = new_N_evnts
                else:
                    new_samples[sample_name]['metadata']["nGenEvts"] = new_N_evnts
                    new_samples[sample_name]['metadata']["sumGenWgts"] *= new_N_evnts/old_N_evnts # just directly multiply by fraction for this since this is already float and this is much faster
                # new_samples[sample_name]['metadata']["fraction"] = fraction
                # state new fraction
                new_samples[sample_name]['metadata']['fraction'] = new_N_evnts/old_N_evnts
                logger.debug(f"new_samples[sample_name]['metadata']['fraction']: {new_samples[sample_name]['metadata']['fraction']}")
                # new_samples[sample_name]['metadata']["original_fraction"] = fraction
                
                # loop through the files to correct the steps
                event_counter = 0 # keeps track of events of multiple root files
                stop_flag = False
                new_files = {}
                for file, file_dict in sample["files"].items():
                    if stop_flag:
                        del new_samples[sample_name]["files"][file] # delete the exess files
                        continue
                    new_steps = []
                    # loop through step sizes to correct fractions
                    for step_iteration in file_dict["steps"]:
                        new_step_lim = new_N_evnts-event_counter
                        if step_iteration[1] < new_step_lim:
                            new_steps.append(step_iteration)
                        else:  # change the upper limit
                            new_steps.append([
                                step_iteration[0],
                                new_step_lim
                            ])
                            stop_flag = True
                            break
                    new_samples[sample_name]["files"][file]["steps"] = new_steps # overwrite new steps
                    # add the end step val to the event_counter
                    if not stop_flag: # update variables and move to next file
                        end_idx = len(file_dict["steps"])-1
                        event_counter += file_dict["steps"][end_idx][1]

        #save the sample info
        
        filename = directory+"/fraction_processor_samples.json"
        with open(filename, "w") as file:
                json.dump(new_samples, file)
    
        elapsed = round(time.time() - time_step, 3)
        logger.info(f"Finished everything in {elapsed} s.")

