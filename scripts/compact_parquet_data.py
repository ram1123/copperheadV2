import os
import dask_awkward as dak
import awkward as ak
import argparse
from distributed import Client
import logging
from modules.utils import logger
import sys
import pickle
import numpy as np
from functools import partial


# Importing DNNWrapper from run_stage2_vbf.py
from run_stage2_vbf import DNNWrapper, prepare_features, fillEventNans, getFoldFilter


def ensure_compacted(year, sample, load_path, compacted_path):
    logger.debug(f"Checking compacted dataset: {compacted_path}")

    # if dir compacted_dir exists, then delete the directory
    # if os.path.exists(compacted_dir): # For debugging purposes
    #     logger.debug(f"Compacted directory exists: {compacted_dir}. Deleting the directory.")
    #     os.system(f"rm -rf {compacted_dir}")

    if not os.path.exists(compacted_path):
        logger.info(f"Compacted dataset not found. Creating at {compacted_path}")

        orig_path = os.path.join(load_path, sample)
        if not os.path.exists(orig_path):
            logger.debug(f"Original data not found at {orig_path}. Skipping.")
            return

        logger.debug(f"Reading data from {orig_path}")
        inFile = dak.from_parquet(orig_path)

        if "vbf_powheg_dipole" in sample:
            logger.warning(f"Sample {sample} is a VBF sample, so, using a smaller chunk size (100k) for repartitioning.")
            target_chunksize = 100_000
        else:
            target_chunksize = 500_000
        inFile = inFile.repartition(rows_per_partition=target_chunksize)

        logger.info(f"Writing compacted data to {compacted_path}")
        inFile.to_parquet(compacted_path)
        logger.debug(f"Dataset successfully compacted.")
    else:
        logger.debug(f"Compacted dataset already exists at {compacted_path}")

def is_typetracer_array(arr):
    # Checks if any backend is typetracer in modern awkward
    try:
        return arr.layout.backend.name == "typetracer"
    except Exception:
        # fallback for older awkward/empty arrays
        return False

def add_dnn_score(events_partition,
              training_features,
              model_cache,
              nfolds, fix_dimuon_mass):
    if getattr(events_partition.layout.backend, "name", None) == "typetracer":
        return ak.with_field(
            events_partition,
            np.empty(0, dtype=np.float32),
            "dnn_vbf_score"
        )
    # Prepare features for this partition
    features_to_use = prepare_features(events_partition, training_features)
    nan_val = -999.0
    input_arr_dict = {}
    for feat in features_to_use:
        arr = nan_val * ak.ones_like(events_partition.event)
        # If the feature is "dimuon_mass", set its value to 125.0
        if feat == "dimuon_mass" and fix_dimuon_mass:
            logger.info(f"Setting 'dimuon_mass' feature to 125.0 for all events in partition.")
            arr = 125.0 * ak.ones_like(events_partition.event)
        input_arr_dict[feat] = arr

    for fold in range(nfolds):
        eval_folds = [(fold + f) % nfolds for f in [3]]
        eval_filter = getFoldFilter(events_partition, eval_folds, nfolds)
        for feat in features_to_use:
            input_arr_fold = input_arr_dict[feat]
            input_arr_fold = ak.where(eval_filter, events_partition[feat], input_arr_fold)
            input_arr_dict[feat] = input_arr_fold
    input_arr = ak.concatenate(
        [input_arr_dict[feat][:, np.newaxis] for feat in features_to_use], axis=1
    )
    dnn_vbf_score = nan_val * ak.ones_like(events_partition.event)
    for fold in range(nfolds):
        eval_folds = [(fold + f) % nfolds for f in [3]]
        eval_filter = getFoldFilter(events_partition, eval_folds, nfolds)
        dnnWrap = model_cache[fold]
        dnn_score_fold = dnnWrap(input_arr)
        dnn_score_fold = ak.flatten(dnn_score_fold, axis=1)
        dnn_vbf_score = ak.where(eval_filter, dnn_vbf_score, dnn_score_fold)
    # return events_partition.assign(dnn_vbf_score=dnn_vbf_score)
    return ak.with_field(
        events_partition,
        dnn_vbf_score,
        "dnn_vbf_score"
    )

def compact_and_add_dnn_score(year, sample, load_path, compacted_dir, model_path, add_dnn_score_flag=False, tag="", fix_dimuon_mass=False):
    compacted_path = os.path.join(compacted_dir, sample, "0") # Added zero to match the original path structure

    compacted_dir_tagged = f"{compacted_dir}_{tag}" if tag else compacted_dir
    compacted_dir_tagged = f"{compacted_dir_tagged}_FixDimuonMass" if fix_dimuon_mass else compacted_dir_tagged
    compacted_path_DNN = os.path.join(compacted_dir_tagged, sample, "0")

    logger.debug(f"Checking compacted dataset for: {compacted_path}")
    logger.debug(f"Checking compacted dataset with DNN score for: {compacted_path_DNN}")

    if not os.path.exists(compacted_path):
        logger.debug(f"Compacted dataset not found. Creating at {compacted_path}")
        ensure_compacted(year, sample, load_path, compacted_path)

    # don't run if add_dnn_score is False
    if not add_dnn_score_flag:
        logger.info("Skipping DNN score addition as add_dnn_score is False.")
        return

    # Load the compacted dataset
    logger.debug(f"Loading compacted dataset from {compacted_path}")
    events = dak.from_parquet(compacted_path)

    # Load the DNN model
    logger.debug(f"Loading DNN model from {model_path}")
    model_trained_path = model_path
    with open(f"{model_trained_path}/training_features.pkl", "rb") as f:
        training_features = pickle.load(f)
    logger.debug(f"Training features loaded: {training_features}")

    # Load and Cache models for each fold
    model_cache = {}
    nfolds = 3  # Assuming 3 folds, adjust as necessary
    for fold in range(nfolds):
        model_load_path = f"{model_trained_path}/fold{fold}/best_model_torchJit_ver.pt"
        logger.debug(f"Loading model for fold {fold} from {model_load_path}")
        model_cache[fold] = DNNWrapper(model_load_path)
        logger.debug(f"Loaded model for fold {fold} from {model_load_path}")


    meta = ak.with_field(events._meta, np.zeros(0, dtype=np.float32), "dnn_vbf_score")
    events = dak.map_partitions(
        add_dnn_score,
        events,
        training_features=training_features,
        model_cache=model_cache,
        nfolds=nfolds,
        fix_dimuon_mass=fix_dimuon_mass,
        meta=meta,
    )

    # Save the updated events with DNN score to the compacted dataset
    events.to_parquet(compacted_path_DNN)
    logger.info(f"Updated dataset with DNN score saved to {compacted_path_DNN}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compacts parquet datasets.")
    parser.add_argument("-y", "--year", required=True, help="Year of the dataset")
    parser.add_argument("-l", "--load_path", required=True, help="Path to the original dataset")
    parser.add_argument("-c", "--compacted_dir", default="", help="Path to store the compacted dataset")
    parser.add_argument("-t", "--tag", default="", help="Tag for the compacted directory")
    parser.add_argument("-m", "--model_path", required=True, help="Path to the DNN model directory")
    parser.add_argument(
        "--fix_dimuon_mass",
        action="store_true",
        help="Fix dimuon mass to 125.0"
    )
    parser.add_argument(
        "--add_dnn_score",
        action="store_true",
        help="Add DNN score to the compacted dataset"
    )
    parser.add_argument("--use_gateway", action="store_true", help="Use Dask Gateway client")
    parser.add_argument(
     "--log-level",
     default=logging.INFO,
     type=lambda x: getattr(logging, x),
     help="Configure the logging level."
    )
    args = parser.parse_args()

    logger.setLevel(args.log_level)

    if args.use_gateway:
        from dask_gateway import Gateway
        gateway = Gateway(
            "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
            proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
        )
        cluster_info = gateway.list_clusters()[0]  # get the first cluster by default. There only should be one anyways
        client = gateway.connect(cluster_info.name).get_client()
        logger.info("Gateway Client created")
    else:
        client = Client(n_workers=64, threads_per_worker=1, processes=True, memory_limit='10 GiB')
        logger.info("Local scale Client created")

    # append /stage1_output/2018/f1_0 to load path
    args.load_path = os.path.join(args.load_path, f"stage1_output/{args.year}/f1_0")

    if not args.compacted_dir:
        logger.debug("No compacted directory provided, using default.")
        args.compacted_dir = (args.load_path).replace("f1_0", "compacted")
    logger.info(f"Compacted directory set to: {args.compacted_dir}")

    samples = os.listdir(args.load_path)
    for sample in samples:
        logger.info(f"Processing sample: {sample}")
        compact_and_add_dnn_score(args.year, sample, args.load_path, args.compacted_dir, args.model_path, args.add_dnn_score, args.tag, args.fix_dimuon_mass)
