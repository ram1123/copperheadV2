#!/usr/bin/env python3
"""
preprocess_dnn.py

Reads stage-1 compacted parquet files, applies region/category cuts, cleaning,
defines 4-fold splits, performs weighted scaling (train-only), and writes
per-fold parquet files (train/validation/evaluation) + scaler artifacts.

Design goals:
- Config-driven (YAML) for features/samples/behavior
- Base path comes from CLI (NOT from config)
- Deterministic CV split based on event number (optionally hashed shuffle)
- Supports per-year process names (DY etc)
- Produces ONE combined dataset across requested years

Example:
python preprocess_dnn.py \
  --config configs/dnn_run3_vbf.yaml \
  --base-path /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_23Jan_JVMFilterJets/stage1_output \
  --tag kfold_shuffleTrue \
  --years 2022preEE,2022postEE,2023BPix \
  --out-root dnn/trained_models \
  --log-level INFO
"""
import argparse
import glob as _glob
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
from typing import Iterator

import awkward as ak
import dask_awkward as dak
import numpy as np
import pandas as pd
import yaml
from modules.dask_utils import close_dask_client, get_dask_client
from modules.git_utils import get_git_commit, get_git_state
from modules.selection import applyRegionCatCuts
from modules.utils import logger
from MVA_training.VBF_run3.utils.pre_scale_cleaning import pre_scaling_clean

# Optional: keep your existing diagnostic plots (safe to disable from CLI)
try:
    from MVA_training.VBF_run3.utils.scaling_helper import (
        plot_before_after_scaling,
        plot_corr_before_after,
        plot_scaled_mean_std,
        plot_scaled_outliers,
        plot_feature_hists_from_fold_dfs,
        plot_feature_hists_compare_folds,
    )

    HAVE_SCALING_PLOTS = True
except Exception:
    HAVE_SCALING_PLOTS = False


# --------------------------------------------------------------------------------------
# Dataclasses
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessConfig:
    """Typed view of the YAML config (minimal fields used by this preprocessor)."""

    seed: int
    dtype: str

    category: str
    region: str

    required_columns: List[str]
    allow_missing_columns: bool

    weight_col: str

    n_folds: int
    strategy: str
    shuffle: bool

    glob_template: str

    # Samples are stored as dicts so we can resolve per-year.
    # signal_spec: {"label": int, "processes": [...], "processes_per_year": {year: [...]}}
    signal_spec: Dict[str, Any]
    # background_groups_spec: {group: {"processes": [...], "processes_per_year": {year: [...]}}}
    background_groups_spec: Dict[str, Dict[str, Any]]
    background_label: int

    training_features: List[str]
    do_not_scale_features: List[str]

# --------------------------------------------------------------------------------------
# YAML + sample resolution helpers
# --------------------------------------------------------------------------------------
def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def procs_for_year(spec: Dict[str, Any], year: str) -> List[str]:
    """
    Resolve processes for a given year with clear precedence:
      - if processes_per_year exists and contains `year` => use that
      - else use `processes` (default)
    """
    per = spec.get("processes_per_year") or {}
    if year in per:
        return list(per[year])
    return list(spec.get("processes", []))

def load_config(cfg_path: str) -> PreprocessConfig:
    cfg = _load_yaml(cfg_path)

    seed = int(cfg["meta"]["seed"])
    dtype = str(cfg["meta"].get("dtype", "float32"))

    category = str(cfg["analysis"]["category"])
    region = str(cfg["analysis"]["region"])

    required_columns = list(cfg["data"]["parquet"].get("required_columns", []))
    allow_missing_columns = bool(
        cfg["data"]["parquet"].get("allow_missing_columns", False)
    )

    weight_col = str(cfg["data"]["weights"]["weight_col"])

    n_folds = int(cfg["data"]["cv"]["n_folds"])
    strategy = str(cfg["data"]["cv"]["strategy"])
    shuffle = bool(cfg["data"]["cv"]["shuffle"])

    glob_template = str(cfg["samples"]["glob_template"])

    signal_spec = dict(cfg["samples"]["signal"])  # includes label, processes, processes_per_year
    background_label = int(cfg["samples"]["background"]["label"])
    background_groups_spec = dict(cfg["samples"]["background"]["groups"])  # group -> spec dict

    training_features = list(cfg["features"]["training"])
    do_not_scale_features = list(cfg["features"].get("do_not_scale_features", []))

    return PreprocessConfig(
        seed=seed,
        dtype=dtype,
        category=category,
        region=region,
        required_columns=required_columns,
        allow_missing_columns=allow_missing_columns,
        weight_col=weight_col,
        n_folds=n_folds,
        strategy=strategy,
        shuffle=shuffle,
        glob_template=glob_template,
        signal_spec=signal_spec,
        background_groups_spec=background_groups_spec,
        background_label=background_label,
        training_features=training_features,
        do_not_scale_features=do_not_scale_features,
    )

def resolve_glob(base_path: str, year: str, glob_template: str, process: str) -> List[str]:
    # tokens: {BASE}, {YEAR}, {PROCESS}
    patt = (
        glob_template.replace("{BASE}", base_path)
        .replace("{YEAR}", year)
        .replace("{PROCESS}", process)
    )
    patt = Path(patt).as_posix() # ensure no // issues in the path
    # logger.info(f"reading: {patt}")
    files = sorted(_glob.glob(patt))
    return files


def events_to_dataframe(
    events: dak.Array,
    keep_cols: List[str],
    cfg: PreprocessConfig,
    process: str,
    label_int: int,
    group: str,
    year: str,
) -> pd.DataFrame:
    """
    Apply selection, convert to pandas, attach label + group + process (+ year if requested).
    """
    events = applyRegionCatCuts(
        events,
        category=cfg.category,
        region_name=cfg.region,
        process=process,
        variation="nominal",
        do_vbf_filter_study=False,
        do_VH_veto=False,
    )

    arr = events.compute()
    arr = arr[~ak.is_none(arr)]

    present_cols = [c for c in keep_cols if c in arr.fields]
    missing_cols = sorted(list(set(keep_cols) - set(present_cols)))
    if missing_cols and (not cfg.allow_missing_columns):
        raise KeyError(f"Missing columns in '{process}': {missing_cols}")

    data: Dict[str, np.ndarray] = {}
    for c in present_cols:
        col = arr[c]
        data[c] = ak.to_numpy(col)

    df = pd.DataFrame(data)

    df["label"] = float(label_int)
    df["process"] = str(process)
    df["process_group"] = str(group)

    # Ensure event exists for deterministic folds
    if "event" not in df.columns:
        raise KeyError(
            "Column 'event' is required for deterministic folds (event % n_folds). "
            "Add it to features2load."
        )

    return df


def weighted_std(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Weighted std (per feature).

    mean = sum(w_i * x_i)/sum(w_i)
    weighted variance square = (sum(w_i* (x_i - mean)^2)) / sum(w_i)

    NOTE: Uses |w| for stability with negative weights. We can play with this and check the performance.

    """
    w = np.asarray(weights, dtype=np.float64)
    w = np.abs(w)
    x = np.asarray(values, dtype=np.float64)

    wsum = np.sum(w)
    if wsum <= 0:
        # fall back to unweighted
        return np.std(x, axis=0)

    mean = np.sum(w[:, None] * x, axis=0) / wsum
    var = np.sum(w[:, None] * (x - mean) ** 2, axis=0) / wsum
    return np.sqrt(np.maximum(var, 0.0))

# --------------------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------------------
def _compute_fold_ids(
    df: pd.DataFrame, n_folds: int, seed: int, shuffle: bool
) -> np.ndarray:
    ev = df["event"].to_numpy(dtype=np.uint64, copy=False)

    if not shuffle:
        logger.debug("[_compute_fold_ids] shuffle=False: using event %% n_folds")
        return (ev.astype(np.int64) % int(n_folds)).astype(np.int64)

    logger.debug("[_compute_fold_ids] shuffle=True: using hashed shuffle with seed=%d", seed)

    # deterministic hashed shuffle based on event and seed (stable across years)
    seed_u = np.uint64(seed)
    x = ev ^ seed_u
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xFF51AFD7ED558CCD)
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xC4CEB9FE1A85EC53)
    x ^= x >> np.uint64(33)
    return (x % np.uint64(n_folds)).astype(np.int64)


def make_output_dir(
    out_root: str, tag: str, years: str, region: str, category: str
) -> str:
    years_slug = years.replace(",", "-").replace(" ", "")
    out = Path(out_root) / tag / f"{years_slug}_{region}_{category}"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def save_feature_list(save_dir: str, features: List[str]) -> None:
    with open(os.path.join(save_dir, "training_features.pkl"), "wb") as f:
        pickle.dump(features, f)
    with open(os.path.join(save_dir, "training_features.json"), "w") as f:
        json.dump(features, f, indent=2)


def save_scaler_npz(
    save_dir: str,
    fold_idx: int,
    scale_features: List[str],
    mean: np.ndarray,
    std: np.ndarray,
    final_features: List[str],
) -> str:
    out = os.path.join(save_dir, f"scalers_{fold_idx}.npz")
    np.savez(
        out,
        features=np.array(scale_features, dtype=object),
        mean=mean,
        std=std,
        final_features=np.array(final_features, dtype=object),
    )
    return out


# --------------------------------------------------------------------------------------
# Core logic
# --------------------------------------------------------------------------------------
def chunk_list(items: List[str], chunk_size: int) -> Iterator[List[str]]:
    """Yield successive chunks from a list."""
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def preprocess(
    cfg: PreprocessConfig,
    base_path: str,
    out_dir: str,
    years: str,
    make_plots: bool,
) -> None:
    years_list = [y.strip() for y in years.split(",") if y.strip()]
    if not years_list:
        raise ValueError("No years provided.")

    # columns to load
    features2load = list(cfg.training_features) + ["event", cfg.weight_col]

    logger.info("[preprocess] years: %s", years_list)
    logger.info("[preprocess] base_path: %s", base_path)
    logger.info("[preprocess] category=%s region=%s", cfg.category, cfg.region)
    logger.info("[preprocess] features2load: %s", features2load)

    # ---- build combined dataframe across years ----
    dfs: List[pd.DataFrame] = []

    for year in years_list:
        # resolve processes per year
        sig_label = int(cfg.signal_spec["label"])
        sig_procs = procs_for_year(cfg.signal_spec, year)

        bkg_groups = cfg.background_groups_spec
        all_bkg_procs: List[Tuple[str, str]] = []  # (group, proc)
        for group_name, gspec in bkg_groups.items():
            procs = procs_for_year(gspec, year)
            for p in procs:
                all_bkg_procs.append((group_name, p))

        logger.info("------------------------------------------------------------")
        logger.info("[year %s] signal procs: %s", year, sig_procs)
        logger.info(
            "[year %s] background groups: %s",
            year,
            {g: procs_for_year(s, year) for g, s in bkg_groups.items()},
        )

        # signal
        for proc in sig_procs:
            files = resolve_glob(base_path, year, cfg.glob_template, proc)
            if not files:
                logger.warning(
                    "[year %s] Missing files for signal proc=%s. Skipping.", year, proc
                )
                continue
            logger.info("[year %s] Reading %d files: %s", year, len(files), proc)
            events = dak.from_parquet(files, columns=features2load + cfg.required_columns)

            # required columns check (best effort)
            if cfg.required_columns:
                missing_req = [
                    c for c in cfg.required_columns if c not in events.fields
                ]
                if missing_req and (not cfg.allow_missing_columns):
                    raise KeyError(
                        f"[year {year}] Process '{proc}' missing required columns: {missing_req}"
                    )

            df = events_to_dataframe(
                events=events,
                keep_cols=features2load + cfg.required_columns,
                cfg=cfg,
                process=proc,
                label_int=sig_label,
                group="vbf",
                year=year,
            )
            dfs.append(df)

        # backgrounds
        for group_name, proc in all_bkg_procs:
            files = resolve_glob(base_path, year, cfg.glob_template, proc)
            if not files:
                logger.warning(
                    "[year %s] Missing files for bkg group=%s proc=%s. Skipping.",
                    year,
                    group_name,
                    proc,
                )
                continue

            chunk_size = 50
            for ichunk, files_chunk in enumerate(chunk_list(files, chunk_size=chunk_size)):
                logger.info(
                    "[year %s] proc=%s chunk %d/%d (nfiles=%d)",
                    year,
                    proc,
                    ichunk + 1,
                    (len(files) - 1) // chunk_size + 1,
                    len(files_chunk),
                )
                events = dak.from_parquet(files_chunk, columns=features2load + cfg.required_columns)

                if cfg.required_columns:
                    missing_req = [
                        c for c in cfg.required_columns if c not in events.fields
                    ]
                    if missing_req and (not cfg.allow_missing_columns):
                        raise KeyError(
                            f"[year {year}] Process '{proc}' missing required columns: {missing_req}"
                        )

                df = events_to_dataframe(
                    events=events,
                    keep_cols=features2load + cfg.required_columns,
                    cfg=cfg,
                    process=proc,
                    label_int=cfg.background_label,
                    group=str(group_name).lower(),
                    year=year,
                )
                dfs.append(df)

    if not dfs:
        raise RuntimeError(
            "No dataframes produced. Check base_path/template/process names/cuts."
        )

    df_total = pd.concat(dfs, axis=0, ignore_index=True)
    logger.info("[preprocess] Combined dataframe: N=%d", len(df_total))

    # ---- basic checks / cleaning ----
    if cfg.weight_col not in df_total.columns:
        raise KeyError(
            f"Weight column '{cfg.weight_col}' not found in merged dataframe."
        )

    # numeric coercion check (training features only)
    # print("df_total[year].head():")
    # print(df_total["year"].head())
    for f in cfg.training_features:
        if f not in df_total.columns:
            raise KeyError(
                f"Training feature '{f}' missing from merged dataframe. (Check year feature or parquet columns.)"
            )
        non_numeric = pd.to_numeric(df_total[f], errors="coerce").isna().any()
        if non_numeric:
            raise ValueError(f"Non-numeric values found in feature column '{f}'.")

    logger.info("[preprocess] Running pre_scaling_clean...")
    df_total = pre_scaling_clean(df_total)

    # Save training feature list
    save_feature_list(out_dir, cfg.training_features)

    # ---- scaling plan ----
    do_not_scale_features = set(cfg.do_not_scale_features)
    scale_features = [f for f in cfg.training_features if f not in do_not_scale_features]
    passthrough_features = [f for f in cfg.training_features if f in do_not_scale_features]

    logger.info(
        "[preprocess] Scaling %d features, passing through %d features",
        len(scale_features),
        len(passthrough_features),
    )
    logger.debug("[preprocess] scale_features: %s", scale_features)
    logger.debug("[preprocess] passthrough_features: %s", passthrough_features)
    final_features = scale_features + passthrough_features

    # ---- folds ----
    n_folds = int(cfg.n_folds)
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    fold_ids = _compute_fold_ids(
        df_total, n_folds=n_folds, seed=cfg.seed, shuffle=cfg.shuffle
    )
    df_total["_fold_id"] = fold_ids

    # ---- write fold parquet outputs ----
    for i in range(n_folds):
        m = df_total["_fold_id"] == i
        n = int(m.sum())
        sw = float(np.abs(df_total.loc[m, cfg.weight_col]).sum())
        logger.info("[fold_id check] fold=%d  N=%d  sum|w|=%.6e", i, n, sw)

        # scheme: train=[i,i+1], val=[i+2], eval=[i+3]
        train_folds = [(i + f) % n_folds for f in (0, 1)]
        val_folds = [(i + 2) % n_folds]
        eval_folds = [(i + 3) % n_folds]

        logger.info("------------------------------------------------------------")
        logger.info(
            "[fold %d] train=%s val=%s eval=%s", i, train_folds, val_folds, eval_folds
        )

        fold_id = df_total["_fold_id"]  # pandas Series

        train_mask = fold_id.isin(train_folds)
        val_mask = fold_id.isin(val_folds)
        eval_mask = fold_id.isin(eval_folds)

        df_train = df_total.loc[train_mask].copy()
        df_val = df_total.loc[val_mask].copy()
        df_eval = df_total.loc[eval_mask].copy()

        # Scaling (weighted) computed on TRAIN only (on scale_features only)
        w_train = df_train[cfg.weight_col].to_numpy(dtype=np.float64, copy=False)
        x_scale_train = df_train[scale_features].to_numpy(dtype=np.float64, copy=False)

        # weighted mean/std
        w_abs = np.abs(w_train)
        wsum = float(np.sum(w_abs))
        if wsum > 0:
            mean = np.sum(w_abs[:, None] * x_scale_train, axis=0) / wsum
        else:
            mean = np.mean(x_scale_train, axis=0)

        std = weighted_std(x_scale_train, w_train)
        std = np.where(std < 1e-6, 1.0, std)

        def _apply_scale(df_in: pd.DataFrame) -> None:
            x = df_in[scale_features].to_numpy(dtype=np.float64, copy=True)
            x = (x - mean) / std
            df_in.loc[:, scale_features] = x.astype(
                np.float32 if cfg.dtype == "float32" else np.float64
            )

        _apply_scale(df_train)
        _apply_scale(df_val)
        _apply_scale(df_eval)

        # stable column order for training
        keep_cols = final_features + [
            "label",
            cfg.weight_col,
            "event",
            "process",
            "process_group",
        ]
        # If you created a year feature, ensure it remains (only if it is in training_features).
        df_train = df_train[keep_cols]
        df_val = df_val[keep_cols]
        df_eval = df_eval[keep_cols]

        scaler_path = save_scaler_npz(
            save_dir=out_dir,
            fold_idx=i,
            scale_features=scale_features,
            mean=mean,
            std=std,
            final_features=final_features,
        )
        logger.info("[fold %d] Saved scaler: %s", i, scaler_path)

        logger.debug("[preprocess] scaling plots enabled: %s", make_plots and HAVE_SCALING_PLOTS)
        if make_plots and HAVE_SCALING_PLOTS:
            try:
                sanity_dir = os.path.join(out_dir, "sanity_feature_plots")
                plot_before_after_scaling(
                    x_scale_train, w_train, mean, std, scale_features, sanity_dir
                )
                plot_scaled_mean_std(
                    df_train[scale_features].to_numpy(),
                    w_train,
                    scale_features,
                    sanity_dir,
                )
                plot_corr_before_after(
                    x_scale_train, df_train[scale_features].to_numpy(), sanity_dir
                )
                plot_scaled_outliers(
                    df_train[scale_features].to_numpy(), scale_features, sanity_dir
                )

                plot_feature_hists_from_fold_dfs(
                    df_train=df_train,
                    df_val=df_val,
                    df_eval=df_eval,
                    features=final_features,   # scaled + passthrough in stable order
                    label_col="label",
                    weight_col=cfg.weight_col,
                    outdir=os.path.join(sanity_dir, "sanity_feature_plots"),
                    fold_idx=i,
                    bins=60,
                    logy=False,  # set True for logy PDFs
                )
                logger.info("[fold %d] Scaling plots done.", i)

            except Exception as exc:
                logger.warning(
                    "[fold %d] Scaling plots failed (continuing): %s", i, str(exc)
                )

        # Write fold parquet files
        df_train.to_parquet(
            os.path.join(out_dir, f"data_df_train_{i}.parquet"), index=False
        )
        df_val.to_parquet(
            os.path.join(out_dir, f"data_df_validation_{i}.parquet"), index=False
        )
        df_eval.to_parquet(
            os.path.join(out_dir, f"data_df_evaluation_{i}.parquet"), index=False
        )

        logger.info(
            "[fold %d] Wrote parquets: train=%d val=%d eval=%d",
            i,
            len(df_train),
            len(df_val),
            len(df_eval),
        )

    # ---- cross-fold feature distribution comparisons ----
    if make_plots and HAVE_SCALING_PLOTS:
        logger.info("------------------------------------------------------------")
        logger.info("[preprocess] Generating cross-fold feature distribution comparisons...")
        sanity_dir = os.path.join(out_dir, "sanity_feature_plots")
        # Compare folds for TRAIN background (most stable)
        plot_feature_hists_compare_folds(
            fold_parquet_dir=out_dir,
            features=final_features,
            label_col="label",
            weight_col=cfg.weight_col,
            outdir=sanity_dir,
            n_folds=cfg.n_folds,
            split="train",
            cls=0,  # 0=bkg
            bins=60,
            logy=False,
        )

        # Compare folds for VAL background
        plot_feature_hists_compare_folds(
            fold_parquet_dir=out_dir,
            features=final_features,
            label_col="label",
            weight_col=cfg.weight_col,
            outdir=sanity_dir,
            n_folds=cfg.n_folds,
            split="validation",
            cls=0,
            bins=60,
            logy=False,
        )

        # Optional: signal overlays too (can be noisy if low stats)
        plot_feature_hists_compare_folds(
            fold_parquet_dir=out_dir,
            features=final_features,
            label_col="label",
            weight_col=cfg.weight_col,
            outdir=sanity_dir,
            n_folds=cfg.n_folds,
            split="train",
            cls=1,  # 1=sig
            bins=60,
            logy=False,
        )

    # ---- manifest for reproducibility ----
    manifest = {
        "pipeline": {
            "preprocessor_version": "v1.1",
            "git_commit": get_git_commit(),
            "git_state": get_git_state(out_dir),
        },
        "base_path": str(base_path),
        "category": cfg.category,
        "region": cfg.region,
        "years": years_list,
        "n_folds": cfg.n_folds,
        "cv": {
            "type": cfg.strategy,
            "shuffle": cfg.shuffle,
            "seed": cfg.seed,
            "n_folds": cfg.n_folds,
        },
        "weight_col": cfg.weight_col,
        "training_features": cfg.training_features,
        "scale_features": scale_features,
        "do_not_scale_features": sorted(list(do_not_scale_features)),
        "samples": {
            "signal": {
                "label": int(cfg.signal_spec["label"]),
                "processes": cfg.signal_spec.get("processes", []),
                "processes_per_year": cfg.signal_spec.get("processes_per_year", {}),
            },
            "background": {
                "label": int(cfg.background_label),
                "groups": cfg.background_groups_spec,
            },
        },
        "yields": {
            "total": {
                "n_events": int(len(df_total)),
                "sumw": float(df_total[cfg.weight_col].sum()),
            },
            "signal": {
                "n_events": int((df_total["label"] == 1).sum()),
                "sumw": float(
                    df_total.loc[df_total["label"] == 1, cfg.weight_col].sum()
                ),
            },
            "background": {
                "n_events": int((df_total["label"] == 0).sum()),
                "sumw": float(
                    df_total.loc[df_total["label"] == 0, cfg.weight_col].sum()
                ),
            },
        },
    }

    out_manifest = os.path.join(out_dir, "preprocess_manifest.json")
    with open(out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("[preprocess] Saved manifest: %s", out_manifest)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DNN preprocessing: cuts -> cleaning -> folds -> scaling -> fold-parquets (multi-year combined)"
    )
    p.add_argument(
        "--config", required=True, help="YAML config file (features/samples/etc.)"
    )

    # base path comes from CLI
    p.add_argument(
        "--base-path",
        required=True,
        help="Base path containing YEAR subdirs (e.g. .../stage1_output).",
    )
    p.add_argument("--tag", required=True, help="Run tag (used under --out-root).")
    p.add_argument(
        "--years",
        required=True,
        help="Comma-separated list of years to process (e.g. '2022preEE,2022postEE,2023BPix').",
    )
    p.add_argument(
        "--out-root", default="dnn/trained_models", help="Output root directory"
    )
    p.add_argument(
        "--no-plots", action="store_true", help="Disable scaling diagnostic plots"
    )
    p.add_argument("--log-level", default="INFO", help="Logging level")

    # quick overrides (without editing YAML)
    p.add_argument(
        "--category", default=None, help="Override analysis.category from YAML"
    )
    # dask_gateway and cluster index for Dask
    p.add_argument(
        "--cluster-index", type=int, default=0, help="Dask cluster index (default: 0)"
    )
    p.add_argument("--use-dask-gateway", action="store_true", help="Use Dask Gateway")
    p.add_argument("--region", default=None, help="Override analysis.region from YAML")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    logger.setLevel(args.log_level)

    client = get_dask_client(args.use_dask_gateway, cluster_index=args.cluster_index)
    cfg = load_config(args.config)
    if args.category is not None:
        object.__setattr__(
            cfg, "category", str(args.category)
        )  # pylint: disable=protected-access
    if args.region is not None:
        object.__setattr__(
            cfg, "region", str(args.region)
        )  # pylint: disable=protected-access

    out_dir = make_output_dir(
        out_root=args.out_root,
        tag=args.tag,
        years=args.years,
        region=cfg.region,
        category=cfg.category,
    )

    logger.info("[main] Output directory: %s", out_dir)
    logger.info(
        "[main] category=%s region=%s years=%s", cfg.category, cfg.region, args.years
    )

    preprocess(
        cfg=cfg,
        base_path=args.base_path,
        out_dir=out_dir,
        years=args.years,
        make_plots=(not args.no_plots),
    )

    logger.info("[main] Done. Success.")
    close_dask_client()


if __name__ == "__main__":
    main()
