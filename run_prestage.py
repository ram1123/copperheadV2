import argparse
import copy
import glob
import json
import os
import re
import time
import uuid

import awkward as ak
import dask_awkward as dak

import coffea
import dask
import numpy as np
import tqdm
import uproot
from cli.common_argparser import build_common_parser, resolve_dataset_yaml_file
from coffea.dataset_tools import rucio_utils
from coffea.dataset_tools.preprocess import preprocess
from coffea.nanoevents import BaseSchema, NanoAODSchema, NanoEventsFactory
from distributed import Client
from modules.utils import logger
from modules.xrootd_utils import AAA_ERROR_FRAGMENTS, AAA_REDIRECTORS, normalize_paths
from omegaconf import OmegaConf
from modules.dask_utils import close_dask_client, get_dask_client

# import warnings
# warnings.filterwarnings("error", module="coffea.*")

dask.config.set({'logging.distributed': 'error'})

def nanoevents_from_root_with_redirectors(
    file_input, host_prefix, attempt, schemaclass, uproot_options, metadata=None
):
    if metadata is None:
        metadata = {}

    try:
        fi_norm = normalize_paths(file_input, host_prefix)
        logger.info(f"[prestage] attempt {attempt} using redirector {host_prefix}")
        events = NanoEventsFactory.from_root(
            fi_norm,
            metadata=metadata,
            schemaclass=schemaclass,
            uproot_options=uproot_options,
        ).events()
        logger.info(f"[prestage] attempt {attempt} succeeded to read metadata with {host_prefix}")
        return events

    except Exception as e:
        msg = str(e)
        tls_bad = any(frag in msg for frag in AAA_ERROR_FRAGMENTS)
        logger.warning(
            f"[prestage] attempt {attempt} failed with {host_prefix}: {type(e).__name__}: {e} (tls_bad={tls_bad})"
        )
        raise


def preprocess_with_redirectors(
    final_output, step_size: int, align_clusters: bool, skip_bad_files: bool
):
    """
    Run coffea.dataset_tools.preprocess, retrying with different AAA redirectors
    if we hit typical XRootD / LZMA issues.
    """
    for attempt, host_prefix in enumerate(AAA_REDIRECTORS, start=1):
        try:
            # Normalize file URLs inside "files" dicts
            norm_final_output = {}
            for sname, sinfo in final_output.items():
                norm_final_output[sname] = {
                    "files": normalize_paths(sinfo["files"], host_prefix)
                }

            logger.info(
                f"[prestage] preparing files: attempt {attempt} with redirector {host_prefix}"
            )
            return preprocess(
                norm_final_output,
                step_size=step_size,
                align_clusters=align_clusters,
                skip_bad_files=skip_bad_files,
            )

        except Exception as e:
            msg = str(e)
            tls_bad = any(frag in msg for frag in AAA_ERROR_FRAGMENTS)
            logger.warning(
                f"[prestage] preprocess attempt {attempt} failed "
                f"with {host_prefix}: {type(e).__name__}: {e}"
            )
            if tls_bad and attempt < len(AAA_REDIRECTORS):
                logger.warning("[prestage] retrying preprocess with next redirector...")
                continue
            # Non-AAA error or no more redirectors
            raise


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

            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return "" # if no problem, return empty string
        else:
            return fname # bad file
    except Exception:
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

def removeBadFiles(filelist):
    bad_filelist = getBadFileParallelizeDask(filelist)
    clean_filtlist = list(set(filelist) - set(bad_filelist)) # remove bad files from the filelist
    return clean_filtlist

def getDatasetRootFiles(single_dataset_name: str, allowlist_sites: list)-> list:
    # logger.info(f"dataset name: {single_dataset_name}")
    # single_dataset_name = single_dataset_name["path"]
    logger.info(f"dataset name: {single_dataset_name}")
    if single_dataset_name.startswith("/eos"):
        fnames = glob.glob(f"{single_dataset_name}/*.root")
        logger.debug(f"fnames: {fnames}")
        fnames = [fname.replace("/eos/purdue", "root://eos.cms.rcac.purdue.edu/") for fname in fnames] # replace to xrootd bc sometimes eos mounts timeout when reading
    elif single_dataset_name.startswith("/depot") or single_dataset_name.startswith("test"):
        fnames = glob.glob(f"{single_dataset_name}/*.root")
        logger.debug(f"fnames: {fnames}")
    else:
        das_query = single_dataset_name

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
        x_cache_fname = "root://xcache.cms.rcac.purdue.edu/" + root_file
        new_fnames.append(x_cache_fname)
    # logger.debug(f"new_fnames: {new_fnames}")
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
    parser = build_common_parser()
    parser.add_argument(
    "-ch",
    "--chunksize",
    dest="chunksize",
    default="10000",
    action="store",
    help="chunksize",
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
    parser.add_argument( # temp flag to test the 2 percent data discrepancy in ggH cat between mine and official workspace
    "--run2_rereco",
    dest="run2_rereco",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="If true, uses skips bad files when calling preprocessing",
    )
    parser.add_argument(
        "--prestage_output",
        dest="prestage_output",
        default="./prestage_output",
        action="store",
        help="path to prestage output directory",
    )
    parser.add_argument(
        "--sync",
        dest="sync",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, syncs files before preprocessing",
    )

    args = parser.parse_args()

    time_step = time.time()
    logger.setLevel(args.log_level)
    os.environ['XRD_REQUESTTIMEOUT']="2400" # some root files via XRootD may timeout with default value
    year = args.year
    args.dataset_yaml_file = resolve_dataset_yaml_file(
        args.dataset_yaml_file, year, args.NanoAODv
    )
    logger.info(f"Using dataset YAML: {args.dataset_yaml_file}")
    logger.info(f"year: {year}")

    if args.fraction is None: # do the normal prestage setup
        total_events = 0

        client = get_dask_client(args.use_gateway, cluster_index=args.cluster_index)

        big_sample_info = {}
        # load dataset sample paths from yaml files
        datasets = OmegaConf.load(args.dataset_yaml_file)
        logger.debug(f'dataset: {datasets}')
        logger.debug(f'datasets.years: {datasets.years}')
        logger.debug(f'datasets.years.keys(): {datasets.years.keys()}')
        if args.run2_rereco: # temp condition for RERECO data case
            year_node = datasets.years[f"{year}_RERECO"]
        else: # normal
            year_node = datasets.years[f"{year}"]
        new_sample_list = []

        # 1) DATA: allow -data C D ... (matches keys like data_C, data_D)
        if "Data" in year_node:
            data_keys = [k for k in year_node["Data"].keys() if k.lower().startswith("data_")]
            logger.debug(data_keys)
            data_samples = args.data_samples
            logger.info(f"data_samples to read: {data_samples}")
            if len(args.data_samples) > 0:
                for data_letter in args.data_samples:
                    for sample_name in data_keys:
                        if sample_name.lower().endswith(data_letter.lower()):
                            new_sample_list.append(sample_name)
            else:
                logger.warning("No -data letters specified; skipping data.")
        else:
            logger.warning("No 'Data' block found in YAML for this year.")

        # 2) BKG/SIG groups: allow -bkg DY TT EWK VV VVV and -sig ggH VBF
        group_keys = [k for k in year_node.keys() if k != "Data"]
        logger.debug(f"Signal and background MC group keys: {group_keys}")

        def _append_group_samples(requested_names):
            for wanted in requested_names:
                for g in group_keys:
                    if g == wanted:
                        new_sample_list.extend(list(year_node[g].keys()))

        # backgrounds
        if len(args.bkg_samples) > 0:
            _append_group_samples(args.bkg_samples)

        # signals (group names should also be top-level like ggH, VBF if present)
        if len(args.sig_samples) > 0:
            _append_group_samples(args.sig_samples)

        logger.debug(f"Loaded samples (names): {new_sample_list}")

        # --- flatten YAML for just the selected samples into a dict {sample_name: sample_dict} ----
        dataset = {}

        logger.debug(f"new_sample_list: {new_sample_list}")
        for name in new_sample_list:
            # search in Data first
            if "Data" in year_node and name in year_node["Data"]:
                dataset[name] = year_node["Data"][name]
                continue
            # then search in each signal/bkg MC group
            for g in group_keys:
                if name in year_node[g]:
                    dataset[name] = year_node[g][name]
                    dataset[name]["__group__"] = g
                    break

        logger.debug(f"Number of selected samples: {len(dataset)}")
        logger.debug(f"Selected sample keys: {list(dataset.keys())}")
        logger.debug(f"Selected sample: {dataset}")
        if not dataset:
            raise RuntimeError("No samples matched your selection. Check -data/-bkg/-sig arguments vs YAML.")

        logger.info(f"Final selected dataset keys: {list(dataset.keys())}")

        fnames = ""

        for sample_name in tqdm.tqdm(dataset.keys()):
            is_data =  ("data" in sample_name)
            logger.debug(f"Sample Name: {sample_name}")
            logger.debug(f"dataset[sample_name]: {dataset[sample_name]}")
            logger.debug(f"is data?: {is_data}")

            allowlist_sites = ["T2_DE_DESY", "T2_AT_Vienna", "T2_DE_RWTH", "T2_IT_Legnaro",
                                "T2_US_Caltech", "T2_UL_Florida", "T2_US_MIT", "T2_US_Purdue", "T2_US_Wisconsin", "T2_US_Nebraska", "T2_US_Vanderbilt",
                                "T2_BE_UCL", "T2_BR_SPRACE", "T2_EE_Estonia",
                                "T2_ES_CIEMAT", "T2_ES_IFCA", "T2_FR_IPHC", "T2_PL_Swierk",
                                "T2_FR_GRIF", "T2_IN_TIFR", "T2_RU_JINR", "T2_BE_IIHE", "T2_CH_CSCS",
                                ]

            logger.debug(f"allowlist_sites: {allowlist_sites}")

            # print(f"type(dataset_name): {type(dataset_name)}")
            sample_cfg = dataset[sample_name]

            logger.debug(f"sample_name: {sample_name}")
            logger.debug(f"sample_cfg: {sample_cfg}")

            logger.debug(f"type(sample_cfg): {type(sample_cfg)}")

            skip_sample = False
            # extract list of DAS paths or local files
            if OmegaConf.is_dict(sample_cfg) and "datasets" in sample_cfg:
                ds_list = OmegaConf.to_container(sample_cfg["datasets"], resolve=True)
                if "skip_sample" in sample_cfg:
                    skip_sample = sample_cfg["skip_sample"]
                    logger.debug(f"Sample {sample_name} has skip_sample = {skip_sample}")
                    if skip_sample:
                        logger.warning(f"Skipping sample {sample_name} as per skip_sample flag.")
                        continue

            elif OmegaConf.is_list(sample_cfg):
                ds_list = OmegaConf.to_container(sample_cfg, resolve=True)
            elif isinstance(sample_cfg, str):
                ds_list = [sample_cfg]
            else:
                raise ValueError(f"Unexpected sample_cfg type {type(sample_cfg)} for {sample_name}")

            # resolve files
            fnames = []
            for single_dataset_name in ds_list:
                if "None" in single_dataset_name:
                    logger.warning(f"Sample {sample_name} has 'None' dataset; skipping.")
                    continue
                fnames += getDatasetRootFiles(single_dataset_name, allowlist_sites)

            if len(fnames) == 0:
                logger.error(f"No files found for sample {sample_name}. Skipping this sample.")
                continue

            if args.skipBadFiles: # if we want to skip bad files
                logger.info("Skipping bad files")
                logger.info(f"Number of files before removing bad files: {len(fnames)}")
                fnames = removeBadFiles(fnames)
                logger.info(f"Number of files after removing bad files: {len(fnames)}")

            # convert to xcachce paths if requested
            if is_data:
                args.xcache = False # FIXME: force turn off xcache for data. This is temporary solution, until hadded for customnanod is fixed for the data files
            if args.xcache:
                fnames = get_Xcache_filelist(fnames)

            # FIXME: Below search replace is a fix for some of files that has this string (not sure why)
            # if fnames contains `/eos/vbc/experiments/cms` remove it. 
            fnames = [f.replace("/eos/vbc/experiments/cms", "") if "/eos/vbc/experiments/cms" in f else f for f in fnames]

            logger.debug(f"sample_name: {sample_name}")
            logger.debug(f"file names: {fnames}")
            logger.debug(f"len(fnames): {len(fnames)}")

            """
            run through each file and collect total number of
            """
            preprocess_metadata = {
                "sumGenWgts" : None,
                "nGenEvts" : None,
                "data_entries" : None,
            }
            if is_data:  # data sample
                base_file_input = {fname: {"object_path": "Events"} for fname in fnames}

                for attempt, host_prefix in enumerate(AAA_REDIRECTORS, start=1):
                    try:
                        file_input = normalize_paths(base_file_input, host_prefix)
                        logger.info(
                            f"[prestage] data sample {sample_name}: attempt {attempt} using {host_prefix}"
                        )

                        events = NanoEventsFactory.from_root(
                            file_input,
                            metadata={},
                            schemaclass=NanoAODSchema,
                            uproot_options={"timeout": 4 * 2400},
                        ).events()
                        logger.debug(f"file_input: {file_input}")
                        logger.debug(f"events.fields: {events.fields}")
                        # if coffea version < 2025.3.0 then use the line below
                        if coffea.__version__ == "2024.11.0":
                            preprocess_metadata["data_entries"] = int(dak.num(events, axis=0).compute())
                        elif coffea.__version__ == "2025.3.0":
                            # For coffea version 2025.3.0, count the entries directly from ROOT files
                            n_events_total = 0
                            for fname_normalized in file_input.keys():
                                with uproot.open(f"{fname_normalized}:Events") as tree:
                                    # or: tree = uproot.open(fname_normalized)["Events"]
                                    n_events_total += tree.num_entries

                            preprocess_metadata["data_entries"] = int(n_events_total)
                        else:
                            raise RuntimeError(f"Unsupported coffea version: {coffea.__version__}")
                        total_events += preprocess_metadata["data_entries"]

                        logger.info(
                            f"[prestage] data sample {sample_name}: success on attempt {attempt} "
                            f"with {host_prefix}, entries = {preprocess_metadata['data_entries']}"
                        )
                        break  # success → stop trying redirectors

                    except Exception as e:
                        msg = str(e)
                        tls_bad = any(frag in msg for frag in AAA_ERROR_FRAGMENTS)
                        logger.warning(
                            f"[prestage] data sample {sample_name}: attempt {attempt} failed "
                            f"with {host_prefix}: {type(e).__name__}: {e}"
                        )
                        if tls_bad and attempt < len(AAA_REDIRECTORS):
                            logger.warning("[prestage] retrying data sample with next redirector...")
                            continue
                        # Non-AAA error or last redirector → propagate
                        raise
            else: # if MC
                for attempt, host_prefix in enumerate(AAA_REDIRECTORS, start=1):
                    try:
                        if "MiNNLO" in sample_name: # We have spurious gen weight issue. ref: https://cms-talk.web.cern.ch/t/huge-event-weights-in-dy-powhegminnlo/8718/9
                            file_input = {fname : {"object_path": "Events"} for fname in fnames}
                            file_input = normalize_paths(file_input, host_prefix)
                            events = nanoevents_from_root_with_redirectors(
                                file_input=file_input,
                                host_prefix=host_prefix,
                                attempt=attempt,
                                schemaclass=BaseSchema,
                                metadata={},
                                uproot_options={"timeout": 4 * 2400},
                            )
                            gen_wgt = np.sign(events.genWeight) # extract signs only, not magntitude
                            preprocess_metadata["sumGenWgts"]= float(ak.sum(gen_wgt).compute())
                            preprocess_metadata["nGenEvts"]= int(ak.num(gen_wgt, axis=0).compute())
                            break  # stop trying once successful
                        else:
                            file_input = {fname: {"object_path": "Runs"} for fname in fnames}
                            file_input = normalize_paths(file_input, host_prefix)
                            logger.debug(f"file_input: {file_input}")
                            runs = nanoevents_from_root_with_redirectors(
                                file_input=file_input,
                                host_prefix=host_prefix,
                                attempt=attempt,
                                schemaclass=BaseSchema,
                                metadata={},
                                uproot_options={"timeout": 4 * 2400},
                            )

                            # print(f"runs.fields: {runs.fields}")
                            # if sample_name == "dy_m105_160_vbf_amc": # nanoAODv6
                            if "genEventSumw" in runs.fields:
                                logger.debug("genEventSumw found: nanoAODv9 or v12")
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
                                logger.debug("genEventSumw_ found: nanoAODv6")
                                preprocess_metadata["sumGenWgts"] = float(ak.sum(runs.genEventSumw_).compute()) # convert into 32bit precision as 64 bit precision isn't json serializable
                                preprocess_metadata["nGenEvts"] = int(ak.sum(runs.genEventCount_).compute()) # convert into 32bit precision as 64 bit precision isn't json serializable
                            logger.info(f"[prestage] data sample {sample_name}: success on attempt {attempt} ")
                            break  # stop trying once successful
                    except Exception as e:
                        msg = str(e)
                        tls_bad = any(frag in msg for frag in AAA_ERROR_FRAGMENTS)
                        logger.warning(
                            f"[prestage] data sample {sample_name}: attempt {attempt} failed "
                            f"with {host_prefix}: {type(e).__name__}: {e}"
                        )
                        if tls_bad and attempt < len(AAA_REDIRECTORS):
                            logger.warning("[prestage] retrying data sample with next redirector...")
                            continue
                        # Non-AAA error or last redirector → propagate
                        raise

                total_events += preprocess_metadata["nGenEvts"]

            # test start -------------------------------
            if sample_name == "dy_VBF_filter_NoUSE": # FIXME: Temporary fix. Remove this patch
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
                files_available, files_total = preprocess_with_redirectors(
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

        # save the sample info
        directory = args.prestage_output
        filename = directory+"/processor_samples_"+year+"_NanoAODv"+str(args.NanoAODv)+".json" # INFO: Hardcoded filename
        if args.sync:
            filename = filename.replace(".json", "_sync.json") # INFO: Hardcoded filename
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filename, "w") as file:
            json.dump(big_sample_info, file, indent=2, sort_keys=True)

        close_dask_client()

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

        # save the sample info
        filename = directory+"/fraction_processor_samples_"+year+"_NanoAODv"+str(args.NanoAODv)+".json" # INFO: Hardcoded filename
        with open(filename, "w") as file:
            json.dump(new_samples, file, indent=2, sort_keys=True)

        elapsed = round(time.time() - time_step, 3)
        logger.info(f"Finished everything in {elapsed} s.")
