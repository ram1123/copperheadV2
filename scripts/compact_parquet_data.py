import logging
import os
import pickle
import math
import shutil
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from distributed import wait
from cli.common_argparser import build_common_parser
from modules.dask_utils import close_dask_client, get_dask_client
from modules.job_status import JobStatus
from modules.utils import logger
from run_stage2_vbf import (
    DNNWrapper,
    getFoldFilter,
    prepare_features,
)
import glob


def sigmoid_ak(x):
    return 1.0 / (1.0 + np.exp(-x))


def clip_ak(x, min_value, max_value):
    return ak.where(x < min_value, min_value, ak.where(x > max_value, max_value, x))


def resolve_scaler_path(model_trained_path, fold):
    npz_path = os.path.join(model_trained_path, f"scalers_{fold}.npz")
    npy_path = os.path.join(model_trained_path, f"scalers_{fold}.npy")
    if os.path.exists(npz_path):
        return npz_path
    if os.path.exists(npy_path):
        return npy_path
    raise FileNotFoundError(f"Missing scaler for fold {fold}: checked {npz_path} and {npy_path}")


def load_scaler_stats(model_trained_path, fold):
    scaler_path = resolve_scaler_path(model_trained_path, fold)
    if scaler_path.endswith(".npz"):
        with np.load(scaler_path, allow_pickle=True) as scaler_file:
            features = scaler_file["features"] if "features" in scaler_file.files else None
            return scaler_file["mean"], scaler_file["std"], features
    scaler_mean, scaler_std = np.load(scaler_path, allow_pickle=True)
    return scaler_mean, scaler_std, None


def resolve_model_path(model_trained_path, fold):
    candidates = [
        os.path.join(model_trained_path, f"fold{fold}", "best_torchscript.pt"),
        os.path.join(model_trained_path, f"fold{fold}", "best_model_torchJit_ver.pt"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Missing model for fold {fold}: checked {', '.join(candidates)}"
    )


def infer_nfolds(model_trained_path, scaler_root=None):
    fold = 0
    while True:
        try:
            resolve_model_path(model_trained_path, fold)
            resolve_scaler_path(scaler_root or model_trained_path, fold)
            fold += 1
        except FileNotFoundError:
            break
    if fold == 0:
        raise FileNotFoundError(
            "No fold artifacts were found under model path: "
            f"{model_trained_path} with scaler root: {scaler_root or model_trained_path}"
        )
    return fold


def resolve_vbf_training_layout(base_path, model_tag, nfolds=4):
    """
    Resolve the preprocessing/scaler directory and the trained-model directory.

    Current stage-2 conventions are:
    - training features + scalers live directly under ``base_path``
    - fold models live under ``base_path / model_tag``
    """
    base_path = Path(base_path).resolve()
    feature_dir = base_path
    training_dir = (base_path / model_tag).resolve()

    feature_ok = (feature_dir / "training_features.pkl").is_file()
    scaler_ok = any(
        (feature_dir / f"scalers_{fold}.npz").is_file()
        or (feature_dir / f"scalers_{fold}.npy").is_file()
        for fold in range(nfolds)
    )
    model_ok = any(
        (training_dir / f"fold{fold}" / "best_torchscript.pt").is_file()
        or (training_dir / f"fold{fold}" / "best_model_torchJit_ver.pt").is_file()
        for fold in range(nfolds)
    )

    if feature_ok and scaler_ok and model_ok:
        return feature_dir, training_dir

    raise FileNotFoundError(
        "Could not resolve VBF training layout under "
        f"base_path={base_path}, model_tag={model_tag}. "
        f"feature_ok={feature_ok} scaler_ok={scaler_ok} model_ok={model_ok} "
        f"(expected training_dir={training_dir})"
    )


def group_parquet_files(file_infos, target_n_final_files):
    if target_n_final_files <= 0:
        raise ValueError(
            f"target_n_final_files must be positive, got {target_n_final_files}"
        )
    if not file_infos:
        return []

    if target_n_final_files >= len(file_infos):
        return [[file_info] for file_info in file_infos]

    groups = []
    n_files = len(file_infos)
    for group_idx in range(target_n_final_files):
        start = group_idx * n_files // target_n_final_files
        stop = (group_idx + 1) * n_files // target_n_final_files
        groups.append(file_infos[start:stop])

    return groups


def write_compacted_group(idx, group_files, compacted_path):
    group_rows = sum(rows for _, rows in group_files)
    output_path = os.path.join(compacted_path, f"part{idx}.parquet")
    logger.debug(
        "Writing compacted part %s with %s input files and %s rows to %s",
        idx,
        len(group_files),
        group_rows,
        output_path,
    )
    arrays = [ak.from_parquet(path) for path, _ in group_files]
    events = arrays[0] if len(arrays) == 1 else ak.concatenate(arrays)
    ak.to_parquet(events, output_path)
    return idx, len(group_files), group_rows, output_path

def _get_num_rows(path):
    return pq.ParquetFile(path).metadata.num_rows

def _get_file_info(path):
    return path, pq.ParquetFile(path).metadata.num_rows

def ensure_compacted(year, sample, input_path, compacted_path, client=None):
    logger.info(f"year: {year}")
    logger.info(f"samples: {sample}")
    logger.info(f"input_path: {input_path}")
    logger.info(f"Checking compacted dataset: {compacted_path}")

    if not os.path.exists(compacted_path):
        logger.info("No compacted dataset exists")
        logger.debug(f"Compacted dataset not found: {compacted_path}")

        orig_path = os.path.join(input_path, sample)
        if not os.path.exists(orig_path):
            logger.info(f"Original data not found at {orig_path}. Skipping.")
            return

        logger.debug(f"Reading data from {orig_path}")
        # check if any parquet files exist (recursively)
        parquet_files = glob.glob(os.path.join(orig_path, "**", "*.parquet"), recursive=True)

        if len(parquet_files) == 0:
            logger.warning(f"No parquet files found under {orig_path}. Skipping.")
            return

        futures = client.map(_get_num_rows, parquet_files)
        total_rows = sum(client.gather(futures))

        if total_rows == 0:
            logger.warning(f"No rows found in parquet files under {orig_path}. Skipping.")
            return

        if ("vbf_powheg" in sample):
            logger.warning(f"Sample {sample} has high density (e.g. vbf signal), so, using a smaller maximum row count (10k) per compacted file.")
            # max_num_of_rows = 100_000
            max_num_of_rows = 10_000
        elif ("top" in sample.lower()) or ("ttjets" in sample.lower()):
            logger.warning(f"Sample {sample} has high memory usage (e.g. st_tchannel_antitop), so, using a smaller maximum row count per compacted file.")
            max_num_of_rows = 1_000
        else:
            # max_num_of_rows = 300_000
            max_num_of_rows = 30_000

        target_n_final_files = min(
            len(parquet_files),
            max(1, math.ceil(total_rows / max_num_of_rows)),
        )
        logger.info(
            "Compacting %s rows from %s parquet files to %s final files",
            total_rows,
            len(parquet_files),
            target_n_final_files,
        )
        parquet_files = sorted(parquet_files)
        futures = client.map(_get_file_info, parquet_files) # TODO: merge _get_num_rows and _get_file_info, since they do the same thing.
        file_infos = client.gather(futures)
        grouped_files = group_parquet_files(file_infos, target_n_final_files)
        # print(f"Grouped files: {grouped_files}")
        logger.info(
            "Writing compacted dataset as %s parquet files",
            len(grouped_files),
        )
        os.makedirs(compacted_path, exist_ok=True)
        if client is None:
            logger.warning("No Dask client provided; writing compacted dataset sequentially")
            results = [
                write_compacted_group(idx, group_files, compacted_path)
                for idx, group_files in enumerate(grouped_files)
            ]
        else:
            n_workers = len(client.scheduler_info().get("workers", {}))
            logger.info("Writing compacted dataset with Dask client (%s workers)", n_workers)
            futures = [
                client.submit(
                    write_compacted_group,
                    idx,
                    group_files,
                    compacted_path,
                    pure=False, # Since it performs I/O, do not optimize it away as a reusable pure computation.
                )
                for idx, group_files in enumerate(grouped_files)
            ]
            results = client.gather(futures)

        for idx, n_files, n_rows, output_path in sorted(results):
            logger.debug(
                "Finished compacted part %s with %s input files and %s rows at %s",
                idx,
                n_files,
                n_rows,
                output_path,
            )
        logger.info("Dataset successfully compacted.")
    else:
        logger.warning(f"Compacted dataset already exists at {compacted_path}")

def ensure_compacted_scaled(year, samples, input_path, compacted_dir, client=None, rerun=False):
    """
    Batched sibling of ensure_compacted(): compacts many samples in one pass
    by submitting the row-count/write work for every sample together and
    gathering once, instead of blocking on the cluster per sample.
    ensure_compacted() itself is left untouched since other code depends on
    its per-sample behavior.

    Progress is tracked per sample under
    {input_path parent}/_status/compacted, using modules.job_status.JobStatus
    the same way run_stage1.py does: each sample is marked "running" before
    its write jobs are submitted and "done" only once they all succeed, so a
    crash partway through leaves already-finished samples resumable instead
    of silently re-declared "done" via directory existence alone (which
    os.makedirs'd the target dir up front, before any part file existed).
    Pass rerun=True to bypass the done markers and force a full redo, mirroring
    run_stage1.py's --rerun.
    """
    logger.info(f"year: {year}")
    logger.info(f"samples: {samples}")
    logger.info(f"input_path: {input_path}")

    status_dir = Path(input_path).resolve().parent / "_status" / "compacted"
    jobstat = JobStatus(status_dir=str(status_dir))

    pending_samples = {}
    for sample in samples:
        compacted_path = os.path.join(compacted_dir, sample, "0")
        logger.info(f"Checking compacted dataset: {compacted_path}")

        if not rerun and not jobstat.should_run(sample, 0):
            logger.info(f"[resume] skip {sample} (done marker present at {status_dir})")
            continue

        if os.path.exists(compacted_path):
            if rerun:
                logger.warning(f"--rerun set; removing existing compacted dataset at {compacted_path} and recompacting.")
                shutil.rmtree(compacted_path)
            elif glob.glob(os.path.join(compacted_path, "*.parquet")):
                logger.warning(f"Compacted dataset already exists at {compacted_path}; backfilling done marker.")
                jobstat.mark_done(sample, 0, meta={"path": compacted_path, "backfilled": True})
                continue
            else:
                logger.warning(
                    f"Found an empty compacted dataset dir at {compacted_path} "
                    "(likely left behind by an interrupted previous run); removing it and recompacting."
                )
                shutil.rmtree(compacted_path)

        orig_path = os.path.join(input_path, sample)
        if not os.path.exists(orig_path):
            logger.info(f"Original data not found at {orig_path}. Skipping.")
            continue

        parquet_files = sorted(glob.glob(os.path.join(orig_path, "**", "*.parquet"), recursive=True))
        if not parquet_files:
            logger.warning(f"No parquet files found under {orig_path}. Skipping.")
            continue

        pending_samples[sample] = (compacted_path, parquet_files)

    if not pending_samples:
        logger.info("No samples require compaction.")
        return

    for sample, (compacted_path, _) in pending_samples.items():
        jobstat.mark_running(sample, 0, meta={"path": compacted_path})

    # Fetch (path, num_rows) for every file across every pending sample in a
    # single round trip instead of one per sample.
    all_files = [
        path
        for _, parquet_files in pending_samples.values()
        for path in parquet_files
    ]
    if client is None:
        file_infos = [_get_file_info(path) for path in all_files]
    else:
        futures = client.map(_get_file_info, all_files)
        file_infos = client.gather(futures)
    rows_by_path = dict(file_infos)

    write_jobs = []  # (sample, compacted_path, idx, group_files)
    for sample, (compacted_path, parquet_files) in pending_samples.items():
        file_infos_sample = [(path, rows_by_path[path]) for path in parquet_files]
        total_rows = sum(rows for _, rows in file_infos_sample)

        if total_rows == 0:
            logger.warning(f"No rows found in parquet files for sample {sample}. Skipping.")
            jobstat.mark_failed(sample, 0, RuntimeError("no rows found in source parquet files"))
            continue

        if "vbf_powheg" in sample:
            logger.warning(f"Sample {sample} has high density (e.g. vbf signal), so, using a smaller maximum row count (10k) per compacted file.")
            max_num_of_rows = 10_000
        elif ("top" in sample.lower()) or ("ttjets" in sample.lower()):
            logger.warning(f"Sample {sample} has high memory usage (e.g. st_tchannel_antitop), so, using a smaller maximum row count per compacted file.")
            max_num_of_rows = 1_000
        else:
            max_num_of_rows = 30_000

        target_n_final_files = min(
            len(parquet_files),
            max(1, math.ceil(total_rows / max_num_of_rows)),
        )
        logger.info(
            "Compacting %s rows from %s parquet files to %s final files for sample %s",
            total_rows,
            len(parquet_files),
            target_n_final_files,
            sample,
        )
        grouped_files = group_parquet_files(file_infos_sample, target_n_final_files)
        os.makedirs(compacted_path, exist_ok=True)
        for idx, group_files in enumerate(grouped_files):
            write_jobs.append((sample, compacted_path, idx, group_files))

    if not write_jobs:
        logger.info("No compaction groups to write.")
        return

    logger.info(
        "Writing %s compacted parquet files across %s samples",
        len(write_jobs),
        len(pending_samples),
    )

    results_by_sample = {}
    failed_samples = {}
    if client is None:
        logger.warning("No Dask client provided; writing compacted datasets sequentially")
        for sample, compacted_path, idx, group_files in write_jobs:
            if sample in failed_samples:
                continue
            try:
                result = write_compacted_group(idx, group_files, compacted_path)
            except Exception as exc:
                failed_samples[sample] = exc
            else:
                results_by_sample.setdefault(sample, []).append(result)
    else:
        n_workers = len(client.scheduler_info().get("workers", {}))
        logger.info("Writing compacted datasets with Dask client (%s workers)", n_workers)
        submissions = [
            (
                sample,
                client.submit(
                    write_compacted_group,
                    idx,
                    group_files,
                    compacted_path,
                    pure=False,  # performs I/O; do not optimize away as a reusable pure computation.
                ),
            )
            for sample, compacted_path, idx, group_files in write_jobs
        ]
        wait([future for _, future in submissions])
        for sample, future in submissions:
            if future.status == "error":
                failed_samples.setdefault(sample, future.exception())
            else:
                results_by_sample.setdefault(sample, []).append(future.result())

    # A sample with any failed group is left incomplete on disk; wipe it so a
    # rerun redoes the whole sample cleanly rather than mistaking the partial
    # output for done (mirrors run_stage1.py deleting partial output before retry).
    for sample, err in failed_samples.items():
        logger.error(f"Compaction failed for sample {sample}: {err}")
        compacted_path = pending_samples[sample][0]
        shutil.rmtree(compacted_path, ignore_errors=True)
        jobstat.mark_failed(sample, 0, err, meta={"path": compacted_path})

    for sample, group_results in results_by_sample.items():
        if sample in failed_samples:
            continue
        for idx, n_files, n_rows, output_path in sorted(group_results):
            logger.debug(
                "Finished compacted part %s with %s input files and %s rows at %s (sample=%s)",
                idx,
                n_files,
                n_rows,
                output_path,
                sample,
            )
        jobstat.mark_done(
            sample,
            0,
            meta={
                "path": pending_samples[sample][0],
                "n_parts": len(group_results),
                "n_input_files": sum(r[1] for r in group_results),
                "n_rows": sum(r[2] for r in group_results),
            },
        )

    logger.info(
        "Datasets successfully compacted for %s samples (%s failed).",
        len(results_by_sample) - len(failed_samples),
        len(failed_samples),
    )

def add_dnn_score(events_partition,
                model_trained_path,
                training_features,
                model_cache,
                nfolds, fix_dimuon_mass, year):
    # One-hot year features (e.g. "year_2022preEE") don't exist as event
    # fields; synthesize them from the partition's dataset year, matching
    # run_stage2_vbf.py's CoffeaStage2VBFProcessor.evaluate_scores().
    year_onehot_features = {f for f in training_features if f.startswith("year_")}
    expected_year_col = f"year_{year}"
    if year_onehot_features and expected_year_col not in year_onehot_features:
        raise ValueError(
            f"Year '{year}' has no matching one-hot training feature. "
            f"Expected one of: "
            f"{sorted(f[len('year_'):] for f in year_onehot_features)}"
        )
    # Prepare features for this partition
    features_to_use = prepare_features(
        events_partition, training_features, year_onehot_features=year_onehot_features
    )
    no_scale_features = {
        "year",
        "nsoftjets5_nominal",
    } | year_onehot_features
    nan_val = -999.0
    input_arr_dict = {}
    for feat in features_to_use:
        arr = nan_val * ak.ones_like(events_partition.event)
        # If the feature is "dimuon_mass", set its value to 125.0
        if feat == "dimuon_mass" and fix_dimuon_mass:
            logger.info("Setting 'dimuon_mass' feature to 125.0 for all events in partition.")
            arr = 125.0 * ak.ones_like(events_partition.event)
        input_arr_dict[feat] = arr

    for fold in range(nfolds):
        # eval_folds = [(fold + f) % nfolds for f in [3]]
        eval_folds = [fold]
        eval_filter = getFoldFilter(events_partition, eval_folds, nfolds)
        scaler_mean, scaler_std, scaler_features = load_scaler_stats(model_trained_path, fold)
        scaler_mean = scaler_mean.astype(np.float64)
        scaler_std = scaler_std.astype(np.float64)
        if scaler_features is None:
            scaler_map = {
                feat: (scaler_mean[ix], scaler_std[ix])
                for ix, feat in enumerate(features_to_use[: len(scaler_mean)])
            }
        else:
            scaler_features = [str(x) for x in scaler_features]
            scaler_map = {
                feat: (scaler_mean[ix], scaler_std[ix])
                for ix, feat in enumerate(scaler_features)
            }

        missing_in_scaler = [
            feat for feat in features_to_use
            if feat not in scaler_map and feat not in no_scale_features
        ]
        if missing_in_scaler:
            raise ValueError(
                f"Features {missing_in_scaler} are missing in scaler and not in NO_SCALE features"
            )

        for ix in range(len(features_to_use)):
            feat = features_to_use[ix]
            input_arr_fold = input_arr_dict[feat]

            # scale the events feature
            if feat in year_onehot_features:
                year_value = 1.0 if feat == expected_year_col else 0.0
                in_feat = year_value * ak.ones_like(events_partition.event)
            elif feat == "dimuon_mass" and fix_dimuon_mass:
                in_feat = 125.0 * ak.ones_like(events_partition.event)
            else:
                in_feat = events_partition[feat]
            if feat in scaler_map and feat not in no_scale_features:
                mu, sigma = scaler_map[feat]
                in_feat = (in_feat - mu) / sigma
            else:
                in_feat = ak.values_astype(in_feat, "float32")

            input_arr_fold = ak.where(eval_filter, in_feat, input_arr_fold)
            input_arr_dict[feat] = input_arr_fold
    input_arr = ak.concatenate(
        [input_arr_dict[feat][:, np.newaxis] for feat in features_to_use], axis=1
    )

    if logger.isEnabledFor(logging.DEBUG):
        n_preview = min(10, len(events_partition))
        event_numbers = ak.to_numpy(ak.materialize(events_partition.event))[:n_preview]
        preview_arr = ak.to_numpy(input_arr[:n_preview])
        preview_df = pd.DataFrame(preview_arr, columns=features_to_use)
        preview_df.insert(0, "event", event_numbers)
        with pd.option_context("display.max_columns", None, "display.width", 200):
            logger.debug(f"DNN input features (first {n_preview} events):\n{preview_df}")
    

    dnn_vbf_logit = nan_val * ak.ones_like(events_partition.event)
    for fold in range(nfolds):
        # eval_folds = [(fold + f) % nfolds for f in [3]]
        eval_folds = [fold]
        eval_filter = getFoldFilter(events_partition, eval_folds, nfolds)
        dnnWrap = model_cache[fold]
        dnn_score_fold = dnnWrap(input_arr)
        dnn_score_fold = ak.flatten(dnn_score_fold, axis=None)
        dnn_vbf_logit = ak.where(eval_filter, dnn_score_fold, dnn_vbf_logit)

    valid_mask = dnn_vbf_logit != nan_val
    dnn_vbf_score = sigmoid_ak(dnn_vbf_logit)
    dnn_vbf_score = ak.where(valid_mask, dnn_vbf_score, nan_val)

    score_for_atanh = clip_ak(dnn_vbf_score, 0.0, 0.999999)
    dnn_vbf_score_atanh = np.arctanh(score_for_atanh)
    dnn_vbf_score_atanh = ak.where(valid_mask, dnn_vbf_score_atanh, nan_val)

    # return the events with the dnn_vbf_score, dnn_vbf_score_atanh, and dnn_vbf_logit fields added
    events_partition = ak.with_field(events_partition, dnn_vbf_score, "dnn_vbf_score")
    events_partition = ak.with_field(events_partition, dnn_vbf_score_atanh, "dnn_vbf_score_atanh")
    events_partition = ak.with_field(events_partition, dnn_vbf_logit, "dnn_vbf_logit")
    return events_partition

def add_dnn_score_to_file(
    input_path,
    output_path,
    model_trained_path,
    training_features,
    model_cache,
    nfolds,
    fix_dimuon_mass,
    year,
):
    events = ak.from_parquet(input_path)
    events = add_dnn_score(
        events,
        model_trained_path=model_trained_path,
        training_features=training_features,
        model_cache=model_cache,
        nfolds=nfolds,
        fix_dimuon_mass=fix_dimuon_mass,
        year=year,
    )
    ak.to_parquet(events, output_path)
    return output_path

def compact_and_add_dnn_score(
    year,
    sample,
    input_path,
    compacted_dir,
    model_path,
    add_dnn_score_flag=False,
    tag="",
    fix_dimuon_mass=False,
    model_tag="",
    client=None,
):
    compacted_path = os.path.join(compacted_dir, sample, "0") # Added zero to match the original path structure

    compacted_dir_tagged = f"{compacted_dir}_{tag}" if tag else compacted_dir
    compacted_dir_tagged = f"{compacted_dir_tagged}_FixDimuonMass" if fix_dimuon_mass else compacted_dir_tagged
    compacted_path_DNN = os.path.join(compacted_dir_tagged, sample, "0")

    logger.info(f"Checking compacted dataset for: {compacted_path}")

    # compact the dataset
    ensure_compacted(year, sample, input_path, compacted_path, client=client)

    # Add the DNN score to the compacted dataset
    if not add_dnn_score_flag:
        logger.info("Skipping DNN score addition as add_dnn_score is False.")
        return

    logger.info(f"Checking compacted dataset with DNN score for: {compacted_path_DNN}")
    if not os.path.exists(compacted_path):
        logger.warning(f"Compacted dataset missing at {compacted_path}. Skipping DNN score addition.")
        return
    # List the compacted parquet files
    parquet_files = sorted(glob.glob(os.path.join(compacted_path, "*.parquet")))
    if not parquet_files:
        logger.warning(f"No parquet files found under {compacted_path}. Skipping DNN score addition.")
        return

    # Load the DNN model
    logger.debug(f"Loading DNN model from {model_path}")
    model_path_obj = Path(model_path).resolve()
    feature_dir, training_dir = resolve_vbf_training_layout(
        model_path_obj, model_tag, nfolds=4
    )
    logger.info(
        "Resolved DNN layout with base=%s model_tag=%s -> feature_dir=%s training_dir=%s",
        model_path_obj,
        model_tag,
        feature_dir,
        training_dir,
    )

    with open(feature_dir / "training_features.pkl", "rb") as f:
        training_features = pickle.load(f)
    logger.debug(f"Training features loaded: {training_features}")

    # Load and Cache models for each fold
    model_cache = {}
    nfolds = infer_nfolds(str(training_dir), scaler_root=str(feature_dir))
    for fold in range(nfolds):
        model_load_path = resolve_model_path(str(training_dir), fold)
        logger.debug(f"Loading model for fold {fold} from {model_load_path}")
        model_cache[fold] = DNNWrapper(model_load_path)
        logger.debug(f"Loaded model for fold {fold} from {model_load_path}")

    os.makedirs(compacted_path_DNN, exist_ok=True)

    if client is None:
        logger.warning("No Dask client provided; adding DNN score sequentially")
        for input_file in parquet_files:
            output_file = os.path.join(compacted_path_DNN, os.path.basename(input_file))
            add_dnn_score_to_file(
                input_file,
                output_file,
                str(feature_dir),
                training_features,
                model_cache,
                nfolds,
                fix_dimuon_mass,
                year,
            )
    else:
        futures = [
            client.submit(
                add_dnn_score_to_file,
                input_file,
                os.path.join(compacted_path_DNN, os.path.basename(input_file)),
                str(feature_dir),
                training_features,
                model_cache,
                nfolds,
                fix_dimuon_mass,
                year,
                pure=False,  # performs I/O; do not optimize away as a reusable pure computation.
            )
            for input_file in parquet_files
        ]
        client.gather(futures)

    logger.info(f"Updated dataset with DNN score saved to {compacted_path_DNN}")

def add_dnn_score_scaled(
    year,
    samples,
    input_path,
    compacted_dir,
    model_path,
    tag="",
    fix_dimuon_mass=False,
    model_tag="",
    client=None,
    rerun=False,
):
    """
    Batched sibling of the DNN-scoring half of compact_and_add_dnn_score():
    loads the model once and scores every sample's compacted files in a
    single round of submissions instead of one sample at a time.

    Progress is tracked per sample under {input_path parent}/_status/compacted/{tag_key},
    where tag_key mirrors the tag/fix_dimuon_mass suffix used to build
    compacted_dir_tagged below -- so different --save_postfix/--fix_dimuon_mass
    runs get independent resumability instead of colliding on the same status
    files, and a rerun skips samples whose DNN-scored output is already on disk.
    Pass rerun=True to bypass the done markers and force a full redo.
    """
    compacted_dir_tagged = f"{compacted_dir}_{tag}" if tag else compacted_dir
    compacted_dir_tagged = f"{compacted_dir_tagged}_FixDimuonMass" if fix_dimuon_mass else compacted_dir_tagged

    tag_key = tag if tag else "untagged"
    if fix_dimuon_mass:
        tag_key = f"{tag_key}_FixDimuonMass"
    status_dir = Path(input_path).resolve().parent / "_status" / "compacted" / tag_key
    jobstat = JobStatus(status_dir=str(status_dir))

    logger.debug(f"Loading DNN model from {model_path}")
    model_path_obj = Path(model_path).resolve()
    feature_dir, training_dir = resolve_vbf_training_layout(
        model_path_obj, model_tag, nfolds=4
    )
    logger.info(
        "Resolved DNN layout with base=%s model_tag=%s -> feature_dir=%s training_dir=%s",
        model_path_obj,
        model_tag,
        feature_dir,
        training_dir,
    )

    with open(feature_dir / "training_features.pkl", "rb") as f:
        training_features = pickle.load(f)
    logger.debug(f"Training features loaded: {training_features}")

    model_cache = {}
    nfolds = infer_nfolds(str(training_dir), scaler_root=str(feature_dir))
    for fold in range(nfolds):
        model_load_path = resolve_model_path(str(training_dir), fold)
        logger.debug(f"Loading model for fold {fold} from {model_load_path}")
        model_cache[fold] = DNNWrapper(model_load_path)
        logger.debug(f"Loaded model for fold {fold} from {model_load_path}")

    pending_samples = {}  # sample -> (compacted_path_DNN, [(input_file, output_file), ...])
    for sample in samples:
        compacted_path = os.path.join(compacted_dir, sample, "0")
        compacted_path_DNN = os.path.join(compacted_dir_tagged, sample, "0")

        if not rerun and not jobstat.should_run(sample, 0):
            logger.info(f"[resume] skip DNN score for {sample} (done marker present at {status_dir})")
            continue

        logger.info(f"Checking compacted dataset with DNN score for: {compacted_path_DNN}")
        if not os.path.exists(compacted_path):
            logger.warning(f"Compacted dataset missing at {compacted_path}. Skipping DNN score addition.")
            continue

        parquet_files = sorted(glob.glob(os.path.join(compacted_path, "*.parquet")))
        if not parquet_files:
            logger.warning(f"No parquet files found under {compacted_path}. Skipping DNN score addition.")
            continue

        if os.path.exists(compacted_path_DNN):
            if rerun:
                logger.warning(f"--rerun set; removing existing DNN-scored dataset at {compacted_path_DNN} and rescoring.")
                shutil.rmtree(compacted_path_DNN)
            else:
                existing = glob.glob(os.path.join(compacted_path_DNN, "*.parquet"))
                if len(existing) >= len(parquet_files):
                    logger.warning(f"DNN-scored dataset already exists at {compacted_path_DNN}; backfilling done marker.")
                    jobstat.mark_done(sample, 0, meta={"path": compacted_path_DNN, "backfilled": True})
                    continue
                logger.warning(
                    f"Found an incomplete DNN-scored dataset dir at {compacted_path_DNN} "
                    "(likely left behind by an interrupted previous run); removing it and rescoring."
                )
                shutil.rmtree(compacted_path_DNN)

        os.makedirs(compacted_path_DNN, exist_ok=True)
        pending_samples[sample] = (
            compacted_path_DNN,
            [
                (input_file, os.path.join(compacted_path_DNN, os.path.basename(input_file)))
                for input_file in parquet_files
            ],
        )

    if not pending_samples:
        logger.info("No samples require DNN score addition.")
        return

    for sample, (compacted_path_DNN, _) in pending_samples.items():
        jobstat.mark_running(sample, 0, meta={"path": compacted_path_DNN})

    flat_jobs = [
        (sample, input_file, output_file)
        for sample, (_, pairs) in pending_samples.items()
        for input_file, output_file in pairs
    ]
    logger.info(
        "Adding DNN score to %s parquet files across %s samples",
        len(flat_jobs),
        len(pending_samples),
    )

    succeeded_counts = {}
    failed_samples = {}
    if client is None:
        logger.warning("No Dask client provided; adding DNN score sequentially")
        for sample, input_file, output_file in flat_jobs:
            if sample in failed_samples:
                continue
            try:
                add_dnn_score_to_file(
                    input_file,
                    output_file,
                    str(feature_dir),
                    training_features,
                    model_cache,
                    nfolds,
                    fix_dimuon_mass,
                    year,
                )
            except Exception as exc:
                failed_samples[sample] = exc
            else:
                succeeded_counts[sample] = succeeded_counts.get(sample, 0) + 1
    else:
        submissions = [
            (
                sample,
                client.submit(
                    add_dnn_score_to_file,
                    input_file,
                    output_file,
                    str(feature_dir),
                    training_features,
                    model_cache,
                    nfolds,
                    fix_dimuon_mass,
                    year,
                    pure=False,  # performs I/O; do not optimize away as a reusable pure computation.
                ),
            )
            for sample, input_file, output_file in flat_jobs
        ]
        wait([future for _, future in submissions])
        for sample, future in submissions:
            if future.status == "error":
                failed_samples.setdefault(sample, future.exception())
            else:
                future.result()
                succeeded_counts[sample] = succeeded_counts.get(sample, 0) + 1

    # A sample with any failed file is left incomplete on disk; wipe it so a
    # rerun redoes the whole sample cleanly rather than mistaking the partial
    # output for done.
    for sample, err in failed_samples.items():
        logger.error(f"DNN score addition failed for sample {sample}: {err}")
        compacted_path_DNN = pending_samples[sample][0]
        shutil.rmtree(compacted_path_DNN, ignore_errors=True)
        jobstat.mark_failed(sample, 0, err, meta={"path": compacted_path_DNN})

    for sample, (compacted_path_DNN, _) in pending_samples.items():
        if sample in failed_samples:
            continue
        jobstat.mark_done(
            sample,
            0,
            meta={"path": compacted_path_DNN, "n_files": succeeded_counts.get(sample, 0)},
        )

    logger.info(
        "Updated datasets with DNN score saved under %s (%s samples, %s failed)",
        compacted_dir_tagged,
        len(pending_samples) - len(failed_samples),
        len(failed_samples),
    )

def compact_and_add_dnn_score_scaled(
    year,
    samples,
    input_path,
    compacted_dir,
    model_path,
    add_dnn_score_flag=False,
    tag="",
    fix_dimuon_mass=False,
    model_tag="",
    client=None,
    rerun=False,
):
    """
    Batched sibling of compact_and_add_dnn_score(): compacts and scores all
    given samples for a single year together, so the cluster processes work
    across samples concurrently instead of one sample at a time.
    """
    ensure_compacted_scaled(year, samples, input_path, compacted_dir, client=client, rerun=rerun)

    if not add_dnn_score_flag:
        logger.info("Skipping DNN score addition as add_dnn_score is False.")
        return

    add_dnn_score_scaled(
        year,
        samples,
        input_path,
        compacted_dir,
        model_path,
        tag=tag,
        fix_dimuon_mass=fix_dimuon_mass,
        model_tag=model_tag,
        client=client,
        rerun=rerun,
    )


if __name__ == "__main__":
    parser = build_common_parser()
    parser.add_argument("-c", "--compacted_dir", default="", help="Path to store the compacted dataset")
    parser.add_argument("-m", "--model_path", help="Path to the DNN model directory")
    parser.add_argument(
        "--model_tag",
        default="",
        help="Trained-model tag under --model_path, matching stage-2 conventions.",
    )
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
    parser.add_argument(
        "--rerun",
        action="store_true",
        help=(
            "If true, bypasses the per-sample done markers under "
            "_status/compacted and reprocesses every sample from scratch."
        ),
    )
    args = parser.parse_args()

    logger.setLevel(args.log_level)

    if args.add_dnn_score:
        if not args.model_path:
            raise ValueError("--model_path is required when --add_dnn_score is used.")
        if not args.model_tag:
            raise ValueError("--model_tag is required when --add_dnn_score is used.")

    client = get_dask_client(args.use_gateway, cluster_index=args.cluster_index)

    # append /stage1_output/2018/f1_0 to load path
    args.input_path = os.path.join(args.input_path, f"stage1_output/{args.year}/f1_0")
    logger.info(f"Input path: {args.input_path}")

    if not args.compacted_dir:
        logger.debug("No compacted directory provided, using default.")
        args.compacted_dir = (args.input_path).replace("f1_0", "compacted")
    logger.info(f"Compacted directory set to: {args.compacted_dir}")

    samples = os.listdir(args.input_path)
    # Uncomment below lines to filter specific samples for testing
    # samples = [s for s in samples if s == "vbf_powheg_dipole"]
    # samples = [s for s in samples if "MiNNLO" in s]
    # samples = [s for s in samples if "dy_VBF_filter" in s]
    # samples = [s for s in samples if "DY" in s]
    logger.info(f"Processing {len(samples)} samples for year {args.year}: {samples}")
    compact_and_add_dnn_score_scaled(
        args.year,
        samples,
        args.input_path,
        args.compacted_dir,
        args.model_path,
        args.add_dnn_score,
        args.save_postfix,
        args.fix_dimuon_mass,
        args.model_tag,
        client=client,
        rerun=args.rerun,
    )

    close_dask_client()
