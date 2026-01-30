#!/usr/bin/env python3
"""
preprocess_dnn.py

Reads stage-1 compacted parquet files, applies region/category cuts, cleaning,
defines 4-fold splits, performs weighted scaling (train-only), and writes
per-fold parquet files (train/validation/evaluation) + scaler artifacts.

Design goals:
- Config-driven (YAML) for features/samples/behavior
- Base path comes from CLI (NOT from config)
- Deterministic CV split (event % n_folds)
- Pylint-friendly naming and structure
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import awkward as ak
import dask_awkward as dak
import numpy as np
import pandas as pd
import yaml
from modules.git_utils import get_git_commit, get_git_state
from modules.selection import applyRegionCatCuts
from modules.utils import logger

# Your existing cleaning helper (keeps current behavior)
from MVA_training.VBF_new.utils.pre_scale_cleaning import pre_scaling_clean

# Optional: keep your existing diagnostic plots (safe to disable from CLI)
try:
    from MVA_training.VBF_new.utils.scaling_helper import (
        plot_before_after_scaling,
        plot_corr_before_after,
        plot_scaled_mean_std,
        plot_scaled_outliers,
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
    year_mode: str
    year_feature_name: str

    required_columns: List[str]
    allow_missing_columns: bool

    weight_col: str

    n_folds: int

    glob_template: str

    signal_label: int
    signal_processes: List[str]

    background_label: int
    background_groups: Dict[str, List[str]]  # group -> processes

    training_features: List[str]

    missing_strategy: str  # "median" supported here
    add_missing_flags: bool

    do_not_scale_default: Tuple[str, ...] = ("nsoftjets5_nominal",)

    def do_not_scale_features(self) -> Tuple[str, ...]:
        do_not_scale = list(self.do_not_scale_default)
        if self.year_mode == "feature" and self.year_feature_name:
            do_not_scale.append(self.year_feature_name)
        return tuple(dict.fromkeys(do_not_scale))  # unique, keep order


@dataclass(frozen=True)
class SampleMap:
    """Derived mappings from config samples."""

    process_to_label: Dict[str, int]
    process_to_group: Dict[str, str]
    signal_group_name: str = "signal"
    other_group_name: str = "OTHER"


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(cfg_path: str) -> PreprocessConfig:
    cfg = _load_yaml(cfg_path)

    seed = int(cfg["meta"]["seed"])
    dtype = str(cfg["meta"].get("dtype", "float32"))

    category = str(cfg["analysis"]["category"])
    region = str(cfg["analysis"]["region"])
    year_mode = str(cfg["analysis"].get("year_mode", "feature"))
    year_feature_name = str(cfg["analysis"].get("year_feature_name", "year"))

    required_columns = list(cfg["data"]["parquet"].get("required_columns", []))
    allow_missing_columns = bool(
        cfg["data"]["parquet"].get("allow_missing_columns", False)
    )

    weight_col = str(cfg["data"]["weights"]["weight_col"])

    n_folds = int(cfg["data"]["cv"]["n_folds"])
    glob_template = str(cfg["samples"]["glob_template"])

    signal_label = int(cfg["samples"]["signal"]["label"])
    signal_processes = list(cfg["samples"]["signal"]["processes"])

    background_label = int(cfg["samples"]["background"]["label"])
    bkg_groups_raw = cfg["samples"]["background"]["groups"]
    background_groups = {
        str(g): list(v["processes"]) for g, v in bkg_groups_raw.items()
    }

    training_features = list(cfg["features"]["training"])

    missing = cfg.get("preprocessing", {}).get("missing", {})
    missing_strategy = str(missing.get("strategy", "median"))
    add_missing_flags = bool(missing.get("add_missing_flags", False))

    if add_missing_flags:
        logger.warning(
            "[config] add_missing_flags=true is not implemented in this preprocessor; ignoring."
        )

    if missing_strategy != "median":
        raise ValueError(
            f"Unsupported missing.strategy: {missing_strategy} (supported: median)"
        )

    return PreprocessConfig(
        seed=seed,
        dtype=dtype,
        category=category,
        region=region,
        year_mode=year_mode,
        year_feature_name=year_feature_name,
        required_columns=required_columns,
        allow_missing_columns=allow_missing_columns,
        weight_col=weight_col,
        n_folds=n_folds,
        glob_template=glob_template,
        signal_label=signal_label,
        signal_processes=signal_processes,
        background_label=background_label,
        background_groups=background_groups,
        training_features=training_features,
        missing_strategy=missing_strategy,
        add_missing_flags=add_missing_flags,
    )


def build_sample_map(cfg: PreprocessConfig) -> SampleMap:
    proc_to_label: Dict[str, int] = {}
    proc_to_group: Dict[str, str] = {}

    # signal
    for p in cfg.signal_processes:
        proc_to_label[p] = cfg.signal_label
        proc_to_group[p] = "vbf"  # your signal is VBF; keep group name stable

    # backgrounds (group names from config: DY/TOP/EWK)
    for group_name, procs in cfg.background_groups.items():
        group_lower = group_name.lower()
        for p in procs:
            proc_to_label[p] = cfg.background_label
            proc_to_group[p] = group_lower

    return SampleMap(process_to_label=proc_to_label, process_to_group=proc_to_group)


def resolve_glob(base_path: str, glob_template: str, process: str) -> List[str]:
    # tokens: {BASE}, {PROCESS}
    patt = glob_template.replace("{BASE}", base_path).replace("{PROCESS}", process)
    # use python glob via Path().glob? template contains **? keep simplest:
    import glob as _glob

    files = sorted(_glob.glob(patt))
    return files


def events_to_dataframe(
    events: dak.Array,
    keep_cols: List[str],
    cfg: PreprocessConfig,
    process: str,
    label_int: int,
    group: str,
) -> pd.DataFrame:
    """
    Apply selection, convert to pandas, attach label + group + process.
    """
    # apply your selection (same signature you use)
    events = applyRegionCatCuts(
        events,
        category=cfg.category,
        region_name=cfg.region,
        process=process,
        variation="nominal",
        do_vbf_filter_study=False,
        do_VH_veto=False,
    )

    # Compute to Awkward and sanitize None records
    arr = events.compute()
    arr = arr[~ak.is_none(arr)]

    # Keep only columns that exist
    present_cols = [c for c in keep_cols if c in arr.fields]
    if (not cfg.allow_missing_columns) and (set(keep_cols) - set(present_cols)):
        missing = sorted(list(set(keep_cols) - set(present_cols)))
        raise KeyError(f"Missing columns in '{process}': {missing}")

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


def make_output_dir(
    out_root: str, tag: str, year: str, region: str, category: str
) -> str:
    # Keep it stable and readable
    out = Path(out_root) / tag / f"{year}_{region}_{category}"
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
# Main preprocessor
# --------------------------------------------------------------------------------------
def preprocess(
    cfg: PreprocessConfig,
    sample_map: SampleMap,
    base_path: str,
    out_dir: str,
    year: str,
    make_plots: bool,
) -> None:
    rng = np.random.default_rng(cfg.seed)

    # columns to load from parquet
    features2load = list(cfg.training_features) + ["event", cfg.weight_col]
    logger.debug("[preprocess] Features to load: %s", features2load)

    # Read and convert all processes -> one big dataframe
    dfs: List[pd.DataFrame] = []

    all_processes = list(cfg.signal_processes)
    for _, procs in cfg.background_groups.items():
        all_processes.extend(procs)

    logger.info("[preprocess] Base path: %s", base_path)
    logger.info("[preprocess] Total processes: %d", len(all_processes))

    for process in all_processes:
        files = resolve_glob(base_path, cfg.glob_template, process)
        if not files:
            logger.warning(
                "[preprocess] No parquets for process=%s (pattern from template). Skipping.",
                process,
            )
            continue

        logger.info(
            "[preprocess] Reading %d parquet files for process=%s", len(files), process
        )

        events = dak.from_parquet(files)

        # quick required-column validation (best-effort)
        if cfg.required_columns:
            missing_req = [c for c in cfg.required_columns if c not in events.fields]
            if missing_req and (not cfg.allow_missing_columns):
                raise KeyError(
                    f"Process '{process}' missing required columns: {missing_req}"
                )

        label_int = sample_map.process_to_label.get(process, cfg.background_label)
        group = sample_map.process_to_group.get(process, sample_map.other_group_name)

        df = events_to_dataframe(
            events=events,
            keep_cols=features2load + cfg.required_columns,
            cfg=cfg,
            process=process,
            label_int=label_int,
            group=group,
        )
        dfs.append(df)

    if not dfs:
        raise RuntimeError(
            "No dataframes produced. Check base_path/template/process names/cuts."
        )

    df_total = pd.concat(dfs, axis=0, ignore_index=True)

    # Basic weight sanitization
    if cfg.weight_col not in df_total.columns:
        raise KeyError(
            f"Weight column '{cfg.weight_col}' not found in the merged dataframe."
        )

    # Sanity check if there are any features with any non-numeric values
    for f in cfg.training_features:
        non_numeric = pd.to_numeric(df_total[f], errors="coerce").isna().any()
        if non_numeric:
            raise ValueError(f"Non-numeric values found in feature column '{f}'.")

    logger.debug("[preprocess] All training feature values are numeric.")

    # Clean before scaling (your existing behavior)
    logger.info("[preprocess] Running pre_scaling_clean...")
    df_total = pre_scaling_clean(df_total)

    # Save training feature list
    save_feature_list(out_dir, cfg.training_features)

    # Determine scaling feature order
    do_not_scale = set(cfg.do_not_scale_features())
    logger.debug("[preprocess] do_not_scale features: %s", sorted(list(do_not_scale)))

    scale_features = [f for f in cfg.training_features if f not in do_not_scale]
    passthrough_features = [f for f in cfg.training_features if f in do_not_scale]
    logger.info("[preprocess] Scaling %d features, passing through %d features",
        len(scale_features), len(passthrough_features)
    )

    final_features = scale_features + passthrough_features
    logger.debug("[preprocess] Final feature order: %s", final_features)

    # Deterministic 4-fold split (matches your current code)
    n_folds = cfg.n_folds
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    logger.info("[preprocess] Using deterministic folds: event %% %d", n_folds)

    # Make fold outputs
    for i in range(n_folds):
        # Same scheme as your script:
        train_folds = [(i + f) % n_folds for f in (0, 1)]
        val_folds = [(i + 2) % n_folds]
        eval_folds = [(i + 3) % n_folds]

        logger.info("------------------------------------------------------------")
        logger.info(
            "[fold %d] train=%s val=%s eval=%s", i, train_folds, val_folds, eval_folds
        )

        fold_id = (df_total["event"] % n_folds).astype(np.int64)

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

        # Keep final feature column order stable for training
        df_train = df_train[
            final_features
            + ["label", cfg.weight_col, "event", "process", "process_group"]
        ]
        df_val = df_val[
            final_features
            + ["label", cfg.weight_col, "event", "process", "process_group"]
        ]
        df_eval = df_eval[
            final_features
            + ["label", cfg.weight_col, "event", "process", "process_group"]
        ]

        # Save scaler artifact
        scaler_path = save_scaler_npz(
            save_dir=out_dir,
            fold_idx=i,
            scale_features=scale_features,
            mean=mean,
            std=std,
            final_features=final_features,
        )
        logger.info("[fold %d] Saved scaler: %s", i, scaler_path)

        # Optional diagnostic plots
        if make_plots and HAVE_SCALING_PLOTS:
            try:
                plot_before_after_scaling(
                    x_scale_train, w_train, mean, std, scale_features, out_dir
                )
                plot_scaled_mean_std(
                    df_train[scale_features].to_numpy(),
                    w_train,
                    scale_features,
                    out_dir,
                )
                plot_corr_before_after(
                    x_scale_train, df_train[scale_features].to_numpy(), out_dir
                )
                plot_scaled_outliers(
                    df_train[scale_features].to_numpy(), scale_features, out_dir
                )
            except Exception as exc:
                logger.warning(
                    "[fold %d] Scaling plots failed (continuing): %s", i, str(exc)
                )

        # Write fold parquet files (small enough → keep separate, as you want)
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

    # Save a small manifest for reproducibility
    manifest = {
        "pipeline": {
            "preprocessor_version": "v1.0",
            "git_commit": get_git_commit(),
            "git_state": get_git_state(out_dir),
        },
        "base_path": base_path,
        "category": cfg.category,
        "region": cfg.region,
        "year": year,
        "n_folds": cfg.n_folds,
        "weight_col": cfg.weight_col,
        "training_features": cfg.training_features,
        "scale_features": scale_features,
        "do_not_scale": sorted(list(do_not_scale)),
        "signal_processes": cfg.signal_processes,
        "background_groups": cfg.background_groups,
        "seed": cfg.seed,
        "yields": {
            "total": {
                "n_events": int(len(df_total)),
                "sumw": float(df_total[cfg.weight_col].sum()),
            },
            "signal": {
                "n_events": int((df_total.label == 1).sum()),
                "sumw": float(df_total.loc[df_total.label == 1, cfg.weight_col].sum()),
            },
            "background": {
                "n_events": int((df_total.label == 0).sum()),
                "sumw": float(df_total.loc[df_total.label == 0, cfg.weight_col].sum()),
            },
        },
    }
    with open(os.path.join(out_dir, "preprocess_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        "[preprocess] Saved manifest: %s",
        os.path.join(out_dir, "preprocess_manifest.json"),
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DNN preprocessing: cuts -> cleaning -> folds -> scaling -> fold-parquets"
    )

    p.add_argument(
        "--config", required=True, help="YAML config file (features/samples/etc.)"
    )

    # base path comes from CLI (your requirement)
    p.add_argument(
        "--base-path",
        required=True,
        help="Base path containing process subdirs (e.g. .../YEAR/compacted or .../YEAR/f1_0).",
    )

    # tag comes from CLI (your requirement)
    p.add_argument(
        "--tag",
        required=True,
        help="Run tag (used to create output directory under --out-root).",
    )

    p.add_argument(
        "--year",
        default="unknown",
        help="Year string for output naming (e.g. 2022postEE, run3, etc.)",
    )
    p.add_argument(
        "--out-root", default="dnn/trained_models", help="Output root directory"
    )
    p.add_argument(
        "--no-plots", action="store_true", help="Disable scaling diagnostic plots"
    )
    p.add_argument("--log-level", default="DEBUG", help="Logging level")

    # Allow quick override without editing YAML
    p.add_argument(
        "--category", default=None, help="Override analysis.category from YAML"
    )
    p.add_argument("--region", default=None, help="Override analysis.region from YAML")

    return p


def main() -> None:
    args = build_argparser().parse_args()
    logger.setLevel(args.log_level)

    cfg = load_config(args.config)
    if args.category is not None:
        object.__setattr__(
            cfg, "category", str(args.category)
        )  # pylint: disable=protected-access
    if args.region is not None:
        object.__setattr__(
            cfg, "region", str(args.region)
        )  # pylint: disable=protected-access

    np.random.seed(cfg.seed)

    sample_map = build_sample_map(cfg)

    out_dir = make_output_dir(
        out_root=args.out_root,
        tag=args.tag,
        year=args.year,
        region=cfg.region,
        category=cfg.category,
    )

    logger.info("[main] Output directory: %s", out_dir)
    logger.info(
        "[main] category=%s region=%s year=%s", cfg.category, cfg.region, args.year
    )

    preprocess(
        cfg=cfg,
        sample_map=sample_map,
        base_path=args.base_path,
        out_dir=out_dir,
        year=args.year,
        make_plots=(not args.no_plots),
    )

    logger.info("[main] Done. Success.")


if __name__ == "__main__":
    main()
