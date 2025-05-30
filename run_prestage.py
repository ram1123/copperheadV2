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
import re
import glob
# import warnings
# warnings.filterwarnings("error", module="coffea.*")
from omegaconf import OmegaConf
import numpy as np
import uuid

import sys
from collections.abc import Sequence

import logging
from modules.utils import logger
from rich import print

def getBadFile(fname):
    try:
        up_file = uproot.open(fname)
        tmp_path = f"/tmp/{uuid.uuid4().hex}.parquet"
        if "Muon_pt" in up_file["Events"].keys():
            # apply parquet tests for lzma error
            ak.to_parquet(up_file["Events"]['Muon_pt'].array(),tmp_path)
            ak.to_parquet(up_file["Events"]['Muon_eta'].array(),tmp_path)
            ak.to_parquet(up_file["Events"]['Muon_phi'].array(),tmp_path)
            ak.to_parquet(up_file["Events"]['Muon_mass'].array(),tmp_path)
            ak.to_parquet(up_file["Events"]['Jet_pt'].array(),tmp_path)
            ak.to_parquet(up_file["Events"]['Jet_eta'].array(),tmp_path)
            ak.to_parquet(up_file["Events"]['Jet_phi'].array(),tmp_path)
            ak.to_parquet(up_file["Events"]['Jet_mass'].array(),tmp_path)
            ak.to_parquet(up_file["Events"]['Electron_pt'].array(),tmp_path)
            ak.to_parquet(up_file["Events"]['Electron_eta'].array(),tmp_path)
            ak.to_parquet(up_file["Events"]['Electron_phi'].array(),tmp_path)
            ak.to_parquet(up_file["Events"]['Electron_mass'].array(),tmp_path)

            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return "" # if no problem, return empty string
        else:
            return fname # bad file
    except Exception as e:
        # return f"An error occurred with file {fname}: {e}"
        # print(f"An error occurred with file {fname}: {e}")
        return fname # bad fileclient

# def getBadFileParallelize(filelist, max_workers=60)
#     with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
#         # Submit each file check to the executor
#         results = list(executor.map(getBadFile, filelist))

#     bad_file_l = []
#     for result in results:
#         if result != "":
#             # print(result)
#             bad_file_l.append(result)

#     return bad_file_l

def getBadFileParallelizeDask(filelist):
    """
    We assume that the dask client has already been initialized
    """
    lazy_results = []
    for fname in filelist:
        lazy_result = dask.delayed(getBadFile)(fname)
        lazy_results.append(lazy_result)
    results = dask.compute(*lazy_results)

    bad_file_l = []
    for result in results:
        if result != "":
            # print(result)
            bad_file_l.append(result)
    print(f"bad_file_l: {bad_file_l}")
    return bad_file_l


def getDatasetRootFiles(single_dataset_name: str, allowlist_sites: list)-> list:
    print(f"single_dataset_name {single_dataset_name}")
    if single_dataset_name.startswith("/eos"):
        fnames = glob.glob(f"{single_dataset_name}/*.root")
        logger.debug(f"fnames: {fnames}")
        fnames = [fname.replace("/eos/purdue", "root://eos.cms.rcac.purdue.edu/") for fname in fnames] # replace to xrootd bc sometimes eos mounts timeout when reading
    elif single_dataset_name.startswith("/depot"):
        fnames = glob.glob(f"{single_dataset_name}/*.root")
        logger.debug(f"fnames: {fnames}")
    else:
        das_query = single_dataset_name
        logger.debug(f"das query: {das_query}")
        print(f"allowlist_sites: {allowlist_sites}")

        rucio_client = rucio_utils.get_rucio_client() # INFO: Why rucio?

        outlist, outtree = rucio_utils.query_dataset(
            das_query,
            client=rucio_client,
            tree=True,
            scope="cms",
        )
        outfiles,outsites,sites_counts =rucio_utils.get_dataset_files_replicas(
            outlist[0],
            allowlist_sites=allowlist_sites,
            mode="full",
            client=rucio_client,
            # partial_allowed=True
        )
        fnames = [file[0] for file in outfiles if file != []]

    return fnames

def get_Xcache_filelist(fnames: list):
    new_fnames = []
    logger.debug(f"fnames: {fnames}")
    for fname in fnames:
        root_file = re.findall(r"/store.*", fname)[0]
        x_cache_fname = "root://cms-xcache.rcac.purdue.edu/" + root_file
        new_fnames.append(x_cache_fname)
    print(f"new_fnames: {new_fnames}")
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
    # argument for prestage output and output file
    parser.add_argument(
        "--prestage_output",
        dest="prestage_output",
        default="./prestage_output",
        action="store",
        help="path to prestage output directory",
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
            client = Client(n_workers=15,  threads_per_worker=1, processes=True, memory_limit='30 GiB')
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
        logger.info(f"data_samples to read: {data_samples}")

        if len(data_samples) >0:
            for data_letter in data_samples:
                for sample_name in data_l:
                    if data_letter in sample_name:
                        logger.debug(sample_name)
                        new_sample_list.append(sample_name)

        logger.debug(f"Loaded samples: {new_sample_list}")

        # take bkg and add to the list: `new_sample_list[]`
        bkg_l = [sample_name for sample_name in dataset.keys() if "data" not in sample_name.lower()]
        logger.info(f"background samples defined in YAML file: {bkg_l}")
        bkg_samples = args.bkg_samples
        logger.info(f"background samples asked to read: {bkg_samples}")
        if len(bkg_samples) >0:
            for bkg_letter in bkg_samples:
                for bkg_name in bkg_l:
                    if bkg_letter in bkg_name:
                        for bkgs in dataset[bkg_name].keys():
                            new_sample_list.append(bkgs)

        logger.debug(f"Loaded samples: {new_sample_list}")

        # take sig and add to the list: `new_sample_list[]`
        sig_samples = args.sig_samples
        logger.info(f"signal samples to load: {sig_samples}")
        if len(sig_samples) >0:
            for sig_sample in sig_samples:
                for bkg_letter in bkg_l: # bkg_l contains both bkg and signal samples keys
                    logger.debug(f"bkg_letter: {bkg_letter}, sig_sample: {sig_sample}")
                    if bkg_letter in sig_sample:
                        for bkgs in dataset[bkg_letter].keys():
                            new_sample_list.append(bkgs)

        logger.debug(f"Loaded samples: {new_sample_list}")

        dataset_dict = {}
        for sample_name_temp in new_sample_list:
            try:
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
            logger.debug(f"dataset[sample_name]: {dataset[sample_name]}")
            logger.debug(f"is data?: {is_data}")

            dataset_name = dataset[sample_name]
            allowlist_sites=["T2_US_Purdue", "T2_US_MIT","T2_US_FNAL", "T2_CH_CERN", "T2_US_Vanderbilt", "T2_US_Florida", "T2_IT_Pisa", "T2_DE_RWTH"]

            # print(f"type(dataset_name): {type(dataset_name)}")
            is_some_list_type = isinstance(dataset_name, Sequence) and not isinstance(dataset_name, str)
            logger.debug(f"is_some_list_type: {is_some_list_type}")
            if is_some_list_type:
                fnames = []
                for single_dataset_name in dataset_name:
                    fnames += getDatasetRootFiles(single_dataset_name, allowlist_sites)
                # print(f"fnames: {fnames}")
                # raise ValueError
            else:
                single_dataset_name = dataset_name
                fnames = getDatasetRootFiles(single_dataset_name, allowlist_sites)

            # convert to xcachce paths if requested
                if args.xcache:
                    fnames = get_Xcache_filelist(fnames)

            logger.debug(f"file names: {fnames}")
            logger.debug(f"sample_name: {sample_name}")
            logger.debug(f"len(fnames): {len(fnames)}")
            logger.debug(f"file names: {fnames}")


            """
            run through each file and collect total number of
            """
            preprocess_metadata = {
                "sumGenWgts" : None,
                "nGenEvts" : None,
                "data_entries" : None,
            }
            if is_data: # data sample
                file_input = {fname : {"object_path": "Events"} for fname in fnames}
                events = NanoEventsFactory.from_root(
                        file_input,
                        metadata={},
                        schemaclass=NanoAODSchema,
                        uproot_options={"timeout":4*2400},
                ).events()
                logger.debug(f"file_input: {file_input}")
                logger.debug(f"events.fields: {events.fields}")
                preprocess_metadata["data_entries"] = int(ak.num(events.Muon.pt, axis=0).compute()) # convert into 32bit precision as 64 bit precision isn't json serializable
                total_events += preprocess_metadata["data_entries"]
            else: # if MC
                if "MiNNLO" in sample_name: # We have spurious gen weight issue. ref: https://cms-talk.web.cern.ch/t/huge-event-weights-in-dy-powhegminnlo/8718/9
                    file_input = {fname : {"object_path": "Events"} for fname in fnames}
                    events = NanoEventsFactory.from_root(
                            file_input,
                            metadata={},
                            schemaclass=BaseSchema,
                            uproot_options={"timeout":4*2400},
                    ).events()
                    gen_wgt = np.sign(events.genWeight) # extract signs only, not magntitude
                    preprocess_metadata["sumGenWgts"]= float(ak.sum(gen_wgt).compute())
                    preprocess_metadata["nGenEvts"]= int(ak.num(gen_wgt, axis=0).compute())
                else:
                    file_input = {fname : {"object_path": "Runs"} for fname in fnames}
                    logger.debug(f"file_input: {file_input}")
                    # print(f"file_input: {file_input}")
                    runs = NanoEventsFactory.from_root(
                            file_input,
                            metadata={},
                            schemaclass=BaseSchema,
                            uproot_options={"timeout":4*2400},
                    ).events()


                    # print(f"runs.fields: {runs.fields}")
                    # if sample_name == "dy_m105_160_vbf_amc": # nanoAODv6
                    if "genEventSumw" in runs.fields:
                        # sumGenwgts = ak.sum(runs.genEventSumw).compute()
                        # sumGenwgts_v2 = ak.sum(events.genWeight).compute()
                        # gen_wgt_max = ak.max(events.genWeight).compute()
                        # big_gen_wgt = events.genWeight > 3000
                        # print(f"big_gen_wgt num: {ak.sum(big_gen_wgt).compute()}")
                        # print(f"gen_wgt_max: {gen_wgt_max}")
                        # print(f"nevents: {nevents}")
                        preprocess_metadata["sumGenWgts"] = float(ak.sum(runs.genEventSumw).compute()) # convert into 32bit precision as 64 bit precision isn't json serializable
                        preprocess_metadata["nGenEvts"] = int(ak.sum(runs.genEventCount).compute()) # convert into 32bit precision as 64 bit precision isn't json serializable
                    else: # nanoAODv6
                        preprocess_metadata["sumGenWgts"] = float(ak.sum(runs.genEventSumw_).compute()) # convert into 32bit precision as 64 bit precision isn't json serializable
                        preprocess_metadata["nGenEvts"] = int(ak.sum(runs.genEventCount_).compute()) # convert into 32bit precision as 64 bit precision isn't json serializable

                total_events += preprocess_metadata["nGenEvts"]

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
                # print(f"final_output: {final_output}")
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
                # print(f"final_output: {final_output}")
                step_size = int(args.chunksize)
                files_available, files_total = preprocess(
                    final_output,
                    step_size=step_size,
                    align_clusters=False,
                    skip_bad_files=args.skipBadFiles,
                )
                # print(f"files_available: {files_available}")
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
        directory = args.prestage_output
        filename = directory+"/processor_samples_"+year+"_NanoAODv"+str(args.NanoAODv)+".json" # INFO: Hardcoded filename
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filename, "w") as file:
                json.dump(big_sample_info, file)

        elapsed = round(time.time() - time_step, 3)
        logger.info(f"Finished everything in {elapsed} s.")
        logger.info(f"Total Events in files {total_events}.")

    else: # take the pre existing samples.json and prune off files we don't need
        fraction = float(args.fraction)
        directory = args.prestage_output
        sample_path = directory+"/processor_samples_"+year+"_NanoAODv"+str(args.NanoAODv)+".json" # INFO: Hardcoded filename
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
        filename = directory+"/fraction_processor_samples_"+year+"_NanoAODv"+str(args.NanoAODv)+".json" # INFO: Hardcoded filename
        with open(filename, "w") as file:
                json.dump(new_samples, file)

        elapsed = round(time.time() - time_step, 3)
        logger.info(f"Finished everything in {elapsed} s.")

