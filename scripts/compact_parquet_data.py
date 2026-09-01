import os
import pickle
import math
import time
from pathlib import Path

import awkward as ak
import dask_awkward as dak
import numpy as np
import pyarrow.parquet as pq
from cli.common_argparser import build_common_parser
from modules.dask_utils import close_dask_client, get_dask_client
from modules.utils import logger
from run_stage2_vbf import (
    DNNWrapper,
    getFoldFilter,
    prepare_features,
)
from tqdm import tqdm
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
    group_rows = sum(rows for _, rows, _ in group_files)
    output_path = os.path.join(compacted_path, f"part{idx}.parquet")
    logger.debug(
        "Writing compacted part %s with %s input files and %s rows to %s",
        idx,
        len(group_files),
        group_rows,
        output_path,
    )
    arrays = [ak.from_parquet(path) for path, _, _ in group_files]
    events = arrays[0] if len(arrays) == 1 else ak.concatenate(arrays)
    ak.to_parquet(events, output_path)
    return idx, len(group_files), group_rows, output_path


# Target *uncompressed* (in-memory) size per compacted output file. Chosen instead
# of a fixed row-count budget because different samples have very different
# per-row memory footprints (jet multiplicity, jagged-array widths, which optional
# branches got saved, etc.) -- row count alone is a poor proxy for that.
DEFAULT_TARGET_MB_PER_FILE = 250.0


def _get_file_rows_and_bytes(path):
    """Return (path, num_rows, total_uncompressed_bytes) from parquet metadata,
    without reading the actual column data."""
    meta = pq.ParquetFile(path).metadata
    total_bytes = sum(meta.row_group(i).total_byte_size for i in range(meta.num_row_groups))
    return path, meta.num_rows, total_bytes

def ensure_compacted(
    year, sample, input_path, compacted_path, client=None,
    target_mb_per_file=DEFAULT_TARGET_MB_PER_FILE,
):
    """Compact `sample`'s stage-1 parquet output into fewer, larger files.

    Returns a short status string for the caller to tally: "compacted",
    "already_exists", "no_input_dir", "no_files", or "no_rows".
    """
    logger.debug(f"year: {year}, sample: {sample}, input_path: {input_path}")

    if os.path.exists(compacted_path):
        logger.info(f"[{sample}] already compacted, skipping ({compacted_path})")
        return "already_exists"

    logger.debug(f"Compacted dataset not found: {compacted_path}")

    orig_path = os.path.join(input_path, sample)
    if not os.path.exists(orig_path):
        logger.info(f"[{sample}] no stage-1 output at {orig_path}, skipping")
        return "no_input_dir"

    logger.debug(f"Reading data from {orig_path}")
    # check if any parquet files exist (recursively)
    parquet_files = glob.glob(os.path.join(orig_path, "**", "*.parquet"), recursive=True)

    if len(parquet_files) == 0:
        logger.warning(f"[{sample}] no parquet files found under {orig_path}, skipping")
        return "no_files"

    t_start = time.perf_counter()
    parquet_files = sorted(parquet_files)
    futures = client.map(_get_file_rows_and_bytes, parquet_files)
    file_infos = client.gather(futures)
    total_rows = sum(rows for _, rows, _ in file_infos)
    total_bytes = sum(nbytes for _, _, nbytes in file_infos)

    if total_rows == 0:
        logger.warning(f"[{sample}] no rows found in parquet files under {orig_path}, skipping")
        return "no_rows"

    avg_bytes_per_row = total_bytes / total_rows
    target_bytes_per_file = target_mb_per_file * 1024 * 1024
    max_num_of_rows = max(1, round(target_bytes_per_file / avg_bytes_per_row))
    size_note = f"~{avg_bytes_per_row:.0f} B/row -> {max_num_of_rows:,} rows/file for {target_mb_per_file:.0f}MB target"

    target_n_final_files = min(
        len(parquet_files),
        max(1, math.ceil(total_rows / max_num_of_rows)),
    )
    grouped_files = group_parquet_files(file_infos, target_n_final_files)

    os.makedirs(compacted_path, exist_ok=True)
    if client is None:
        n_workers_note = "sequentially, no Dask client"
        results = [
            write_compacted_group(idx, group_files, compacted_path)
            for idx, group_files in enumerate(grouped_files)
        ]
    else:
        n_workers = len(client.scheduler_info().get("workers", {}))
        n_workers_note = f"{n_workers} workers"
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

    elapsed = time.perf_counter() - t_start
    logger.info(
        "[%s] %s rows, %s files -> %s files (%s), %s, done in %.1fs",
        sample,
        f"{total_rows:,}",
        len(parquet_files),
        len(grouped_files),
        n_workers_note,
        size_note,
        elapsed,
    )
    return "compacted"

def add_dnn_score(events_partition,
                model_trained_path,
                training_features,
                model_cache,
                nfolds, fix_dimuon_mass):
    if getattr(events_partition.layout.backend, "name", None) == "typetracer":
        out = ak.with_field(events_partition, np.empty(0, np.float32), "dnn_vbf_score")
        out = ak.with_field(out, np.empty(0, np.float32), "dnn_vbf_score_atanh")
        out = ak.with_field(out, np.empty(0, np.float32), "dnn_vbf_logit")
        return out
    # Prepare features for this partition
    features_to_use = prepare_features(events_partition, training_features)
    no_scale_features = {
        "year",
        "nsoftjets5_nominal",
    }
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
            if feat == "dimuon_mass" and fix_dimuon_mass:
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
    target_mb_per_file=DEFAULT_TARGET_MB_PER_FILE,
):
    compacted_path = os.path.join(compacted_dir, sample, "0") # Added zero to match the original path structure

    compacted_dir_tagged = f"{compacted_dir}_{tag}" if tag else compacted_dir
    compacted_dir_tagged = f"{compacted_dir_tagged}_FixDimuonMass" if fix_dimuon_mass else compacted_dir_tagged
    compacted_path_DNN = os.path.join(compacted_dir_tagged, sample, "0")

    # compact the dataset
    status = ensure_compacted(
        year, sample, input_path, compacted_path, client=client,
        target_mb_per_file=target_mb_per_file,
    )

    # Add the DNN score to the compacted dataset
    if not add_dnn_score_flag:
        return status

    logger.debug(f"Checking compacted dataset with DNN score for: {compacted_path_DNN}")
    if not os.path.exists(compacted_path):
        logger.warning(f"Compacted dataset missing at {compacted_path}. Skipping DNN score addition.")
        return status
    # Load the compacted dataset
    logger.debug(f"Loading compacted dataset from {compacted_path}")
    events = dak.from_parquet(compacted_path)

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

    meta = ak.with_field(events._meta, np.zeros(0, dtype=np.float32), "dnn_vbf_score")
    meta = ak.with_field(meta, np.zeros(0, dtype=np.float32), "dnn_vbf_score_atanh")
    meta = ak.with_field(meta, np.zeros(0, dtype=np.float32), "dnn_vbf_logit")
    events = dak.map_partitions(
        add_dnn_score,
        events,
        model_trained_path=str(feature_dir),
        training_features=training_features,
        model_cache=model_cache,
        nfolds=nfolds,
        fix_dimuon_mass=fix_dimuon_mass,
        meta=meta,
    )

    # Save the updated events with DNN score to the compacted dataset
    events.to_parquet(compacted_path_DNN)
    logger.info(f"Updated dataset with DNN score saved to {compacted_path_DNN}")
    return status


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
        "--target-mb-per-file",
        type=float,
        default=DEFAULT_TARGET_MB_PER_FILE,
        help=(
            "Target uncompressed size (MB) per compacted output file. Rows-per-file "
            "is derived per sample from its actual average row size, instead of a "
            "fixed row count."
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
    # client = get_dask_client(False, n_workers=32) # FIXME: no using dask gateway for now, as it is not working on the cluster

    # append /stage1_output/2018/f1_0 to load path
    args.input_path = os.path.join(args.input_path, f"stage1_output/{args.year}/f1_0")
    logger.info(f"Input path: {args.input_path}")

    if not args.compacted_dir:
        logger.debug("No compacted directory provided, using default.")
        args.compacted_dir = (args.input_path).replace("f1_0", "compacted")
    logger.info(f"Compacted directory set to: {args.compacted_dir}")

    samples = os.listdir(args.input_path)
    t_run_start = time.perf_counter()
    status_counts = {}
    failed_samples = []
    for i, sample in enumerate(tqdm(samples, desc="Processing samples"), start=1):
        # Uncomment below lines to filter specific samples for testing
        # if sample != "vbf_powheg_dipole": continue
        # if "MiNNLO" not in sample: continue
        # if "dy_VBF_filter" not in sample:
        # continue
        # if "DY" not in sample: continue
        tqdm.write(f"--- [{i}/{len(samples)}] {sample} ---")
        try:
            status = compact_and_add_dnn_score(
                args.year,
                sample,
                args.input_path,
                args.compacted_dir,
                args.model_path,
                args.add_dnn_score,
                args.save_postfix,
                args.fix_dimuon_mass,
                args.model_tag,
                client=client,
                target_mb_per_file=args.target_mb_per_file,
            )
        except Exception:
            logger.exception(f"[{sample}] failed")
            status = "failed"
            failed_samples.append(sample)
        status_counts[status] = status_counts.get(status, 0) + 1

    elapsed = time.perf_counter() - t_run_start
    logger.info(
        "Compaction summary: %s samples in %.1fs -> %s",
        len(samples),
        elapsed,
        ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())),
    )
    if failed_samples:
        logger.warning("Failed samples: %s", ", ".join(failed_samples))

    close_dask_client()
