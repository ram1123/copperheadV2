from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from src.copperhead_processor import EventProcessor
# from src.copperhead_processor_cutflow import EventProcessor
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
import os
from omegaconf import OmegaConf
from coffea.nanoevents.methods import vector
from rich import print


# dask.config.set({'logging.distributed': 'error'})
import logging
from modules.utils import logger
from modules.utils import get_git_info

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# test_mode = False
np.set_printoptions(threshold=sys.maxsize)
import gc
import ctypes
from src.lib.get_parameters import getParametersForYr

from modules.utils import logger


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

def getSavePath(start_path: str, dataset_dict: dict, file_idx: int):
    """
    Small wrapper function that returns the directory path to save the parquets
    from stage1
    """
    fraction = round(dataset_dict["metadata"]["fraction"], 3)
    fraction_str = str(fraction).replace('.', '_')
    sample_name = dataset_dict['metadata']['dataset']
    save_path = start_path + f"/f{fraction_str}/{dataset_dict['metadata']['dataset']}/{file_idx}"
    return save_path

def dataset_loop(processor, dataset_dict, file_idx=0, test=False, save_path=None, isCutflow=False):
    if save_path is None:
        username = os.environ.get("USER") or os.environ.get("USERNAME")
        save_path = f"/depot/cms/users/{username}/results/stage1/test/" # default
        os.makedirs(save_path, exist_ok=True)
    logger.debug(f"dataset: {dataset_dict}")
    logger.debug(f"file index: {file_idx}")
    logger.debug(f"test: {test}")
    logger.debug(f"Output path: {save_path}")

    events = NanoEventsFactory.from_root(
        dataset_dict["files"],
        schemaclass=NanoAODSchema,
        metadata= dataset_dict["metadata"],
        uproot_options={
            "timeout":2400,
            # "allow_read_errors_with_report": True, # this makes process skip over OSErrors
        },
    ).events()

    out_collections = processor.process(events)

    # Save the cutflow
    if hasattr(processor, "cutflow") and isCutflow:
        fraction = round(dataset_dict["metadata"]["fraction"], 3)
        fraction_str = str(fraction).replace('.', '_')
        cutflow_save_path = f"{save_path}/f{fraction_str}"
        if not os.path.exists(cutflow_save_path):
            os.makedirs(cutflow_save_path)
        cutflow_save_path = f"{save_path}/f{fraction_str}/cutflow_{dataset_dict['metadata']['dataset']}_{file_idx}.npz"

        processor.cutflow.to_npz(cutflow_save_path).compute()
        logger.info(f"Cutflow saved to {cutflow_save_path}")

    dataset_fraction = dataset_dict["metadata"]["fraction"]

    logger.info(f"out_collections keys: {out_collections.keys()}")

    skim_dict = out_collections
    skim_dict["fraction"] = dataset_fraction*(ak.ones_like(out_collections["event"]))
    #----------------------------------
    skim_zip = ak.zip(skim_dict, depth_limit=1)
    logger.debug(f"skim_zip: {skim_zip}")
    # skim_zip.persist().to_parquet(save_path)
    # raise ValueError
    return skim_zip



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
    parser.add_argument(
    "-aod_v",
    "--NanoAODv",
    dest="NanoAODv",
    type=int,
    default=9,
    choices = [9, 12],
    help="version number of NanoAOD samples we're working with. currently, only 9 and 12 are supported",
    )
    parser.add_argument(
        "-maxfile",
        "--max_file_len",
        dest="max_file_len",
        type=int,
        default = 3000,
        help = "How many maximum files to process simultaneously.",
    )
    parser.add_argument(
     "--log-level",
     default=logging.ERROR,
     type=lambda x: getattr(logging, x),
     help="Configure the logging level."
     )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="If need to run over fractional dataset samples for test run"
    )
    # add parser to turn on the cut-flow
    parser.add_argument(
        "--isCutflow",
        action="store_true",
        help="Get the cutflow",
    )
    args = parser.parse_args()

    logger.setLevel(args.log_level)

    test_mode = args.test_mode
    logger.debug(f"Test mode: {test_mode}")

    # make NanoAODv into an interger variable
    logger.info(f"args.NanoAODv: {args.NanoAODv}")
    logger.info(f"args.year: {args.year}")

    time_step = time.time()


    warnings.filterwarnings('ignore')
    """
    Coffea Dask automatically uses the Dask Client that has been defined above
    """


    config = getParametersForYr("./configs/parameters/" , args.year)
    logger.debug(f"stage1 config: {config}")
    coffea_processor = EventProcessor(config, test_mode=test_mode, isCutflow=args.isCutflow)

    if not test_mode: # full scale implementation
        if args.use_gateway:
            from dask_gateway import Gateway
            gateway = Gateway(
                "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
                proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
            )
            # gateway = Gateway()
            logger.info("Connecting to Dask Gateway")
            logger.info(f"gateway: {gateway}")
            logger.info(f"gateway list clusters: {gateway.list_clusters()}")

            cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
            client = gateway.connect(cluster_info.name).get_client()
            logger.debug(f"client: {client}")
            logger.info("Gateway Client created")
        else:
            client = Client(n_workers=64,  threads_per_worker=1, processes=True, memory_limit='10 GiB')
            logger.info("Local scale Client created")
        #-------------------------------------------------------------------------------------
        sample_path = "./prestage_output/processor_samples_"+args.year+"_NanoAODv"+str(args.NanoAODv)+".json" # INFO: Hardcoded filename        logger.debug(f"Sample path: {sample_path}")
        logger.debug(f"Sample path: {sample_path}")
        with open(sample_path) as file:
            samples = json.loads(file.read())

        logger.debug(f'samples: {samples}')
        # add in NanoAODv info into samples metadata for coffea processor
        for dataset in samples.keys():
            samples[dataset]["metadata"]["NanoAODv"] = args.NanoAODv
        start_save_path = args.save_path + f"/stage1_output/{args.year}"
        logger.info(f"start_save_path: {start_save_path}")
        # make the directory if it doesn't exist
        os.makedirs(start_save_path, exist_ok=True)

        # Get git information; for the log. Also, it will help with debugging, if needed.
        git_commit_hash, branch_name, diff = get_git_info()
        # save this information in a file in the `start_save_path` directory
        git_info_path = os.path.join(start_save_path, "git_info.txt")
        with open(git_info_path, "w") as f:
            f.write(f"Git commit hash: {git_commit_hash}\n")
            f.write(f"Branch name: {branch_name}\n")
            f.write(f"Diff:\n{diff}\n")
        logger.info(f"git_info_path: {git_info_path}")


        with performance_report(filename="dask-report.html"):
            for dataset, sample in tqdm.tqdm(samples.items()):
                # if dataset in ["ggh_amcPS", "ggh_powhegPS"]: # FIXME: temporary line to skip some datasets for which we already have stage1 output
                #     logger.warning(f"Skipping dataset: {dataset}")
                #     continue
                # if "ewk_lljj_mll50_mjj120" not in dataset: # FIXME: temporary line to skip some datasets for which we already have stage1 output
                #     logger.warning(f"Skipping dataset: {dataset}")
                #     continue
                sample_step = time.time()
                smaller_files = list(divide_chunks(sample["files"], args.max_file_len))
                logger.debug(f"max_file_len: {args.max_file_len}")
                logger.debug(f"len(smaller_files): {len(smaller_files)}")
                for idx in tqdm.tqdm(range(len(smaller_files)), leave=False):
                    # if idx < 50 or idx > 51: continue # for testing purposes
                    # if idx < 7: continue
                    logger.info(f"Processing {dataset} file index {idx}")
                    smaller_sample = copy.deepcopy(sample)
                    smaller_sample["files"] = smaller_files[idx]
                    var_step = time.time()
                    to_persist = dataset_loop(coffea_processor, smaller_sample, file_idx=idx, test=test_mode, save_path=start_save_path)
                    save_path = getSavePath(start_save_path, smaller_sample, idx)
                    logger.info(f"save_path: {save_path}")
                    if not os.path.exists(save_path):
                        logger.debug(f"Path: {save_path} is going to be created")
                        os.makedirs(save_path)
                    else:
                        # remove previously existing files and make path if doesn't exist
                        filelist = glob.glob(f"{save_path}/*.parquet")
                        logger.warning(f"Going to delete files: len(filelist): {len(filelist)}")
                        for file in filelist:
                            os.remove(file)
                    logger.debug(f"Directory created or cleaned: {save_path}")
                    to_persist.persist().to_parquet(save_path)
                    # to_persist.to_parquet(save_path)

                    var_elapsed = round(time.time() - var_step, 3)
                    logger.info(f"Finished file_idx {idx} in {var_elapsed} s.")
                sample_elapsed = round(time.time() - sample_step, 3)
                logger.info(f"Finished sample {dataset} in {sample_elapsed} s.")

    else:
        client = Client(n_workers=12,  threads_per_worker=1, processes=True, memory_limit='10 GiB')
        logger.info("Local scale Client created")

        sample_path = "./prestage_output/fraction_processor_samples_"+args.year+"_NanoAODv"+str(args.NanoAODv)+".json" # INFO: Hardcoded filename
        with open(sample_path) as file:
            samples = json.loads(file.read())
        logger.debug(f'samples: {samples}')

        for dataset in samples.keys():
            samples[dataset]["metadata"]["NanoAODv"] = args.NanoAODv

        start_save_path = args.save_path + f"/stage1_output_test/{args.year}"
        logger.info(f"start_save_path: {start_save_path}")
        os.makedirs(start_save_path, exist_ok=True)
        with performance_report(filename="dask-report.html"):
            for dataset, sample in tqdm.tqdm(samples.items()):
                logger.debug(f"dataset: {dataset}")
                to_persist = dataset_loop(coffea_processor, sample, test=test_mode, save_path=start_save_path)

                save_path = getSavePath(start_save_path, sample, 0)
                logger.info(f"save_path: {save_path}")
                if not os.path.exists(save_path):
                    logger.debug(f"Path: {save_path} is going to be created")
                    os.makedirs(save_path)
                else:
                    # remove previously existing files and make path if doesn't exist
                    filelist = glob.glob(f"{save_path}/*.parquet")
                    logger.debug(f"Going to delete files: len(filelist): {len(filelist)}")
                    for file in filelist:
                        os.remove(file)
                logger.debug("Directory created or cleaned")
                to_persist.persist().to_parquet(save_path)
    elapsed = round(time.time() - time_step, 3)
    logger.info(f"Finished everything in {elapsed} s.")
