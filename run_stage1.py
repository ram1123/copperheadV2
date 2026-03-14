import argparse
import copy
import ctypes
from datetime import datetime
import glob
import json
import os
import subprocess
import sys
import time
import warnings
from itertools import islice

import awkward as ak
import dask
import numpy as np
import tqdm
from cli.common_argparser import build_common_parser
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from dask.distributed import performance_report

from modules.dask_utils import close_dask_client, get_dask_client
from modules.job_status import JobStatus, write_stage1_summary
from modules.utils import get_git_info, logger
from modules.xrootd_utils import AAA_ERROR_FRAGMENTS, AAA_REDIRECTORS, normalize_paths
from src.copperhead_processor import EventProcessor
from src.lib.get_parameters import getParametersForYr
from configs.skip_stage1_run import samples_to_skip, samples_to_run

dask.config.set(annotations={"retries": 5})
dask.config.set({"distributed.scheduler.default-task-retries": 5})
dask.config.set({"distributed.scheduler.worker-saturation": 1.0})

warnings.filterwarnings("ignore", category=RuntimeWarning)

np.set_printoptions(threshold=sys.maxsize)


def should_process_dataset(dataset, args, samples_to_skip=None, samples_to_run=None):
    """
    Decide whether a dataset should be processed.
    Returns True if it should run, False if it should be skipped.
    """

    # If explicit run-list is provided → highest priority
    if samples_to_run:
        return dataset in samples_to_run

    # Else, apply skip list if requested
    if args.skipSamples and samples_to_skip:
        return dataset not in samples_to_skip

    # Default → run
    return True

def get_expected_events_from_files_dict(files_dict):
    """
    Count expected input events from prestage file dict.

    Supports:
      {file: {"steps": [[start, stop], ...], ...}}
    and falls back to num_entries if needed.
    """
    total = 0

    for _, finfo in files_dict.items():
        if isinstance(finfo, dict):
            if "steps" in finfo and finfo["steps"] is not None:
                for step in finfo["steps"]:
                    total += int(step[1]) - int(step[0])
            elif "num_entries" in finfo:
                total += int(finfo["num_entries"])
    return int(total)

# #-------------------------------------------------------------------

def getSavePath(start_path: str, dataset_dict: dict, file_idx: int):
    """
    Small wrapper function that returns the directory path to save the parquets
    from stage1
    """
    fraction = round(dataset_dict["metadata"]["fraction"], 3)
    fraction_str = str(fraction).replace('.', '_')
    save_path = start_path + f"/f{fraction_str}/{dataset_dict['metadata']['dataset']}/{file_idx}"
    return save_path

def dataset_loop(processor, dataset_dict, file_idx=0, test=False, save_path=None,  isCutflow=False, dataset_yaml_file="configs/datasets/dataset.yaml"):
    if save_path is None:
        username = os.environ.get("USER") or os.environ.get("USERNAME")
        save_path = f"/depot/cms/users/{username}/results/stage1/test/" # default
        os.makedirs(save_path, exist_ok=True)
    logger.debug(f"dataset: {dataset_dict}")
    logger.debug(f"file index: {file_idx}")
    logger.debug(f"test: {test}")
    logger.debug(f"Output path: {save_path}")

    # dict to hold the max_num_elements info per sample
    dict_max_num_elements = {
        "data_": 900,  # None means no limit (use uproot's default behavior)
        "dy": 250,
        "ttjets_dl": 250,
        "ttjets_sl": 250,
    }
    max_num_elements = 500 # default
    if any(key in dataset_dict["metadata"]["dataset"] for key in dict_max_num_elements.keys()):
        max_num_elements = dict_max_num_elements[[key for key in dict_max_num_elements.keys() if key in dataset_dict["metadata"]["dataset"]][0]]
        logger.debug(f"Setting max_num_elements for {dataset_dict['metadata']['dataset']} to {max_num_elements}")
    else:
        max_num_elements = 500
    logger.info(f"max_num_elements for {dataset_dict['metadata']['dataset']} set to {max_num_elements}")

    events = NanoEventsFactory.from_root(
        dataset_dict["files"],
        schemaclass=NanoAODSchema,
        metadata= dataset_dict["metadata"],
        uproot_options={
            "timeout": 900,
            "num_workers": 1, # needs to be 1 for dask, solves vector_read error
            "max_num_elements": max_num_elements,
            # "allow_read_errors_with_report": True, # this makes process skip over OSErrors
        },
    ).events()

    processed_event_count = 0
    out_collections, processed_event_count = processor.process(events, dataset_yaml_file=dataset_yaml_file)

    # Save the cutflow
    if hasattr(processor, "cutflow") and isCutflow:
        logger.info("Saving cutflow information (NPZ and JSON)")
        
        # Ensure directory exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        base_name = f"cutflow_{dataset_dict['metadata']['dataset']}_{file_idx}"
        npz_path = os.path.join(save_path, f"{base_name}.npz")
        json_path = os.path.join(save_path, f"{base_name}.json")

        # 1. Save NPZ (Efficient for reloading into Coffea/Python later)
        # The .compute() ensures Dask finishes the task before writing
        processor.cutflow.to_npz(npz_path).compute()
        logger.info(f"NPZ saved: {npz_path}")

        # 2. Save JSON
        logger.debug(f"processor.cutflow.logger.info(): {processor.cutflow.print()}")
        try:
            cf_res = processor.cutflow
            
            # Helper to safely convert numpy values to python scalars
            def clean(val):
                return val.item() if hasattr(val, "item") else val

            # Build a structured dictionary
            # _names: list of cut names
            # _nevcutflow: cumulative counts
            # _nevonecut: individual cut counts
            combined_data = {}
            for i, name in enumerate(cf_res._names):
                combined_data[name] = {
                    "cumulative": clean(cf_res._nevcutflow[i]),
                    "individual": clean(cf_res._nevonecut[i])
                }

            with open(json_path, 'w') as f:
                json.dump(combined_data, f, indent=4)
                
            logger.info(f"JSON saved to {json_path}")

        except Exception as e:
            logger.error(f"JSON save failed: {e}")


    dataset_fraction = dataset_dict["metadata"]["fraction"]

    logger.debug(f"out_collections keys: {out_collections.keys()}")

    out_collections["fraction"] = dataset_fraction * (ak.ones_like(out_collections["event"]))
    # ----------------------------------
    skim_zip = ak.zip(out_collections, depth_limit=1)
    logger.debug(f"skim_zip: {skim_zip}")
    # skim_zip.persist().to_parquet(save_path)
    to_persist = skim_zip.persist()
    to_persist = to_persist.to_parquet(save_path, compute=False)
    persisted, processed_event_count = dask.compute(to_persist, processed_event_count)
    return processed_event_count


def divide_chunks(data: dict, SIZE: int):
   it = iter(data)
   for i in range(0, len(data), SIZE):
      yield {k:data[k] for k in islice(it, SIZE)}


def _run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def eos_mkdirs(eos_path: str, retries: int = 3, sleep: float = 2.0):
    """Create directories on EOS if they do not exist.

    To create directory use gfal command:
    gfal-mkdir davs://eos.cms.rcac.purdue.edu:9000/store/user/<username>/<directoryname>

    Args:
        eos_path (str): The EOS path where directories should be created.
        retries (int, optional): Number of retries in case of failure. Defaults to 3.
        sleep (float, optional): Sleep time between retries in seconds. Defaults to 2.0.

    FIXME: This function currently does not handle /store paths correctly. As gfal-mkdir command does not work with coffea_latest environment.
    """
    if eos_path.startswith("/depot") or eos_path.startswith("/work") or eos_path.startswith("test"):
        os.makedirs(eos_path, exist_ok=True)
        return
    if not eos_path.startswith("/store") or not eos_path.startswith("davs"):
        raise RuntimeError(f"Path does not starts with /depot or /work or /store or davs. Please check path.")
    if not eos_path.startswith("davs://eos.cms.rcac.purdue.edu:9000/"):
        eos_path = f"davs://eos.cms.rcac.purdue.edu:9000/{eos_path.lstrip('/')}"
    logger.info(f"Creating EOS directory: {eos_path}")
    cmd = ["gfal-mkdir", "-p", eos_path]
    for attempt in range(retries):
        logger.info(f"command: {' '.join(cmd)}")
        result = _run(cmd)
        if result.returncode == 0:
            logger.info(f"Successfully created EOS directory: {eos_path}")
            return
        else:
            logger.warning(f"Attempt {attempt + 1} to create EOS directory failed: {result.stderr}")
            time.sleep(sleep)
    logger.error(f"Failed to create EOS directory after {retries} attempts: {eos_path}")
    raise RuntimeError(f"Failed to create EOS directory: {eos_path}")


def _parquet_dir_has_files(p: str) -> bool:
    try:
        return any(fn.endswith(".parquet") for fn in os.listdir(p))
    except Exception:
        return False


if __name__ == "__main__":
    t0 = time.perf_counter()
    parser = build_common_parser()
    parser.add_argument(
        "-maxfile",
        "--max_file_len",
        dest="max_file_len",
        type=int,
        default = 3000,
        help = "How many maximum files to process simultaneously.",
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
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="If true, deletes the existing stage1 output directory and reruns stage1",
    )
    parser.add_argument(
        "--skipSamples",
        action="store_true",
        help="If true, skips samples listed in configs/skip_stage1_run.py",
    )
    parser.add_argument(
        "--sync",
        dest="sync",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, syncs files before preprocessing",
    )
    args = parser.parse_args()

    logger.setLevel(args.log_level)

    test_mode = args.test_mode
    logger.debug(f"Test mode: {test_mode}")

    # make NanoAODv into an interger variable
    logger.info(f"args.NanoAODv: {args.NanoAODv}")
    logger.info(f"args.year: {args.year}")
    t1 = time.perf_counter()
    logger.info(f"[Timing] Time taken to parse arguments: {round(t1 - t0, 3)} seconds")

    time_step = time.time()

    warnings.filterwarnings('ignore')
    """
    Coffea Dask automatically uses the Dask Client that has been defined above
    """

    if "2018" in args.year:
        yearForConfig = "2018" # use 2018 parameters for 2018PR as well
    else:
        yearForConfig = args.year

    config = getParametersForYr("./configs/parameters/" , yearForConfig)
    logger.debug(f"stage1 config: {config}")
    coffea_processor = EventProcessor(config, test_mode=test_mode, isCutflow=args.isCutflow)

    client = get_dask_client(args.use_gateway, cluster_index=args.cluster_index)
    if not test_mode: # full scale implementation
        t2 = time.perf_counter()
        logger.info(f"[Timing] Time taken to create Dask Client: {round(t2 - t1, 3)} seconds")
        # -------------------------------------------------------------------------------------
        sample_path = "./prestage_output/processor_samples_"+args.year+"_NanoAODv"+str(args.NanoAODv)+".json" # INFO: Hardcoded filename        logger.debug(f"Sample path: {sample_path}")
        if args.sync:
            sample_path = sample_path.replace(".json", "_sync.json") # INFO: Hardcoded sample_path        
        logger.debug(f"Sample path: {sample_path}")
        with open(sample_path) as file:
            samples = json.loads(file.read())

        logger.debug(f'samples: {samples}')
        # add in NanoAODv info into samples metadata for coffea processor
        for dataset in samples.keys():
            samples[dataset]["metadata"]["NanoAODv"] = args.NanoAODv
        start_save_path = f"{args.save_path}/stage1_output/{args.year}"
        logger.info(f"start_save_path: {start_save_path}")
        # make the directory if it doesn't exist
        eos_mkdirs(start_save_path)
        # Resumability markers ------------------------------------------------
        jobstat = JobStatus(status_dir=os.path.join(start_save_path, "_status"))

        # Get git information; for the log. Also, it will help with debugging, if needed.
        git_commit_hash, branch_name, diff = get_git_info()
        # save this information in a file in the `start_save_path` directory
        # add timestamp to the filename to avoid overwriting in case of multiple runs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        git_info_path = os.path.join(start_save_path, f"git_info_{timestamp}.txt")
        with open(git_info_path, "w") as f:
            f.write(f"Git commit hash: {git_commit_hash}\n")
            f.write(f"Branch name: {branch_name}\n")
            f.write(f"Diff:\n{diff}\n")
        logger.info(f"git_info_path: {git_info_path}")

        # if True:
        with performance_report(filename="dask-report.html"):
            for dataset, sample in tqdm.tqdm(samples.items(), desc="Processing datasets"):
                logger.info("{}{}".format("\n" * 2, "=" * 51))
                logger.info(f"===         Processing dataset: {dataset}       ===")
                logger.info(f"===         NanoAODv: {args.NanoAODv}                 ===")
                logger.info(f"===         Year: {args.year}                        ===")
                logger.info("{}{}".format("=" * 51, "\n" * 2))

                if not should_process_dataset(dataset, args, samples_to_skip, samples_to_run):
                    logger.warning(f"Skipping dataset: {dataset}")
                    continue

                sample_step = time.time()
                # dict to hold file lenght info per sample
                dict_file_length = {
                    "data_": 900,
                    "dy": 250,
                    "ttjets_dl": 250,
                    "ttjets_sl": 250,
                }
                if any(key in dataset for key in dict_file_length.keys()):
                    args.max_file_len = dict_file_length[[key for key in dict_file_length.keys() if key in dataset][0]]
                    logger.info(f"Setting max_file_len for {dataset} to {args.max_file_len}")
                else:
                    args.max_file_len = 500
                logger.info(f"max_file_len for {dataset} set to {args.max_file_len}")

                # split the sample files into smaller chunks of size args.max_file_len
                # # use only 1/4 of the total files available in sample for test mode
                # if True:
                #     total_files = len(sample["files"])
                #     print(f"total_files for {dataset}: {total_files}")
                #     print(f"files: {sample['files']}")
                #     test_files = sample["files"][100]
                #     sample["files"] = test_files
                #     logger.info(f"Test mode: Using only 1/4 of total files for {dataset}. total_files: {total_files}, test_files used: {len(test_files)}")
                smaller_files = list(divide_chunks(sample["files"], args.max_file_len))
                logger.info(f"len(smaller_files): {len(smaller_files)}")
                for idx in tqdm.tqdm(range(len(smaller_files)), leave=False):
                    # Skip if already done (unless user wants a full rerun)
                    if not args.rerun and not jobstat.should_run(dataset, idx):
                        logger.info(f"[resume] skip {dataset}[{idx}] (done marker present)")
                        continue

                    smaller_sample = copy.deepcopy(sample)
                    smaller_sample["files"] = smaller_files[idx]
                    var_step = time.time()
                    save_path = getSavePath(start_save_path, smaller_sample, idx)
                    ExpectedEvents_from_prestage = get_expected_events_from_files_dict(smaller_sample["files"])
                    logger.debug(f"ExpectedEvents_from_prestage: {ExpectedEvents_from_prestage}")

                    processed_event_count = 0

                    # Try up to several times, cycling through redirectors defined in AAA_REDIRECTORS
                    jobstat.mark_running(dataset, idx,
                        meta={
                            "split count": len(smaller_files),
                            "max_attempts": len(AAA_REDIRECTORS),
                            "args.max_file_len": args.max_file_len,
                            "redirector": None,
                            "path": save_path,
                            "git_commit_hash": git_commit_hash,
                            "git_branch": branch_name,
                            "git patch path": git_info_path,
                            })
                    for attempt, host_prefix in enumerate(AAA_REDIRECTORS, start=1):
                        try:
                            logger.info(f"[resume] attempt {attempt} for {dataset}[{idx}] using {host_prefix}")
                            # build fresh file list with this redirector
                            alt_sample = copy.deepcopy(smaller_sample)
                            # logger.info(f"alt_sample['files']: {alt_sample['files']}")

                            alt_sample["files"] = normalize_paths(smaller_sample["files"], host_prefix=host_prefix)

                            logger.debug(f"alt_sample['files']: {alt_sample['files']}")

                            # clean partial output from previous tries
                            os.system(f"rm -rf '{save_path}'")
                            eos_mkdirs(save_path)

                            # rebuild the events/out collections for this attempt
                            processed_event_count = dataset_loop(coffea_processor, alt_sample, file_idx=idx, test=test_mode, save_path=save_path, isCutflow=args.isCutflow, dataset_yaml_file=args.dataset_yaml_file)

                            logger.info(f"Expected  events: {ExpectedEvents_from_prestage}")
                            logger.info(f"Processed events: {processed_event_count}")

                            # to_persist = to_persist.persist()
                            # to_persist.to_parquet(save_path, write_metadata_file=False) # INFO: Find out difference between below and this line
                            # to_persist.to_parquet(save_path)

                            if not _parquet_dir_has_files(save_path):
                                raise RuntimeError("Parquet write produced no files.")

                            if ExpectedEvents_from_prestage != processed_event_count:
                                raise ValueError(
                                    f"Number of processed events does not match expected events: "
                                    f"expected {ExpectedEvents_from_prestage}, processed {processed_event_count} "
                                    f"(dataset={dataset}, file_idx={idx}, save_path={save_path})"
                                )

                            jobstat.mark_done(
                                dataset,
                                idx,
                                meta={
                                    "split count": len(smaller_files),
                                    "attempt": attempt,
                                    "max_attempts": len(AAA_REDIRECTORS),
                                    "args.max_file_len": args.max_file_len,
                                    "redirector": host_prefix,
                                    "path": save_path,
                                    "git_commit_hash": git_commit_hash,
                                    "git_branch": branch_name,
                                    "git patch path": git_info_path,
                                    "Expected events from pre-stage": ExpectedEvents_from_prestage,
                                    "Processed events from stage-1": processed_event_count,
                                },
                            )
                            logger.info(f"[resume] success on attempt {attempt} with {host_prefix}")
                            break  # stop trying once successful

                        except Exception as e:
                            msg = str(e)
                            tls_bad = any(frag in msg for frag in AAA_ERROR_FRAGMENTS)
                            logger.warning(
                                f"[resume] attempt {attempt} failed for {dataset}[{idx}] "
                                f"({type(e).__name__}: {e})"
                            )
                            # save the list of files that were attempted in this failure for debugging,
                            # with the redirector info in the filename
                            # as well as the error message for this failure
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            error_info_path = os.path.join(
                                start_save_path,
                                "_status",
                                f"error_{dataset}_{idx}_{timestamp}.txt",
                            )
                            with open(error_info_path, "w") as f:
                                f.write(f"Error message: {msg}\n")
                                f.write(f"Attempted files with {host_prefix}:\n")
                                f.write(f"Total files attempted: {len(alt_sample['files'])}\n")
                                for i, file in enumerate(alt_sample["files"]):
                                    f.write(f"{i:4}: {file}\n")
                            logger.info(f"Saved error info to {error_info_path}")

                            if attempt < len(AAA_REDIRECTORS) and tls_bad:
                                logger.warning(f"Retrying {dataset}[{idx}] with next redirector ...")
                                continue  # next redirector in list
                            else:
                                jobstat.mark_failed(
                                    dataset,
                                    idx,
                                    e,
                                    meta={
                                        "split count": len(smaller_files),
                                        "attempt": attempt,
                                        "max_attempts": len(AAA_REDIRECTORS),
                                        "args.max_file_len": args.max_file_len,
                                        "redirector": host_prefix,
                                        "path": save_path,
                                        "git_commit_hash": git_commit_hash,
                                        "git_branch": branch_name,
                                        "git patch path": git_info_path,
                                        "Expected events from pre-stage": ExpectedEvents_from_prestage,
                                        "Processed events from stage-1": processed_event_count,                                        
                                    },
                                )
                                logger.exception(
                                    f"[resume] write failed after {attempt} attempts for {dataset}[{idx}]"
                                )

                    var_elapsed = round(time.time() - var_step, 3)
                    logger.info(f"Finished file_idx {idx} in {var_elapsed} s.")
                sample_elapsed = round(time.time() - sample_step, 3)
                logger.info(f"Finished sample {dataset} in {sample_elapsed} s.")
                t6 = time.perf_counter()
                logger.info(f"[Timing] Time taken to process sample {dataset}: {round(t6 - t2, 3)} seconds")

    else:
        # FIXME: update this for /store usage
        sample_path = "./prestage_output/fraction_processor_samples_"+args.year+"_NanoAODv"+str(args.NanoAODv)+".json" # INFO: Hardcoded filename
        with open(sample_path) as file:
            samples = json.loads(file.read())
        logger.debug(f'samples: {samples}')

        for dataset in samples.keys():
            samples[dataset]["metadata"]["NanoAODv"] = args.NanoAODv

        start_save_path = f"{args.save_path}/stage1_output_test/{args.year}"
        logger.info(f"start_save_path: {start_save_path}")
        os.makedirs(start_save_path, exist_ok=True)
        with performance_report(filename="dask-report.html"):
            for dataset, sample in tqdm.tqdm(samples.items()):
                logger.debug(f"dataset: {dataset}")
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
                dataset_loop(coffea_processor, sample, test=test_mode, save_path=save_path, dataset_yaml_file=args.dataset_yaml_file)

    elapsed = round(time.time() - time_step, 3)

    write_stage1_summary(
        status_dir=os.path.join(start_save_path, "_status"),
        out_json_path=os.path.join(start_save_path, "_status", "stage1_summary.json"),
        logger=logger,
    )

    close_dask_client()
    logger.info(f"Finished everything in {elapsed} s.")
