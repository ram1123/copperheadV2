from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import awkward as ak
import dask_awkward as dak
import numpy as np
import torch

from modules.utils import logger

# ----------------------------------------------------------------------
# Caches (per worker process)
# ----------------------------------------------------------------------
_ZPT_SCALER_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[List[str]]]] = {}
_ZPT_MODEL_CACHE: Dict[Tuple[str, str], torch.jit.RecursiveScriptModule] = {}


def load_scaler_cached(scaler_path: str):
    """
    Expects scaler.npz with either:
      - mean, scale  (sklearn StandardScaler)
      - mean, std    (training code saves this)
      - x_mean, x_std
    Optionally also contains:
      - feature_order (saved by training code)
    Returns: (mean, scale, feature_order_or_None)
    """
    if scaler_path not in _ZPT_SCALER_CACHE:
        d = np.load(scaler_path)
        keys = set(d.files)

        # --- mean/scale ---
        if {"mean", "scale"}.issubset(keys):
            mean = d["mean"].astype(np.float32)
            scale = d["scale"].astype(np.float32)
        elif {"mean", "std"}.issubset(keys):
            mean = d["mean"].astype(np.float32)
            scale = d["std"].astype(np.float32)
        else:
            raise KeyError(
                f"Unrecognized scaler keys in {scaler_path}: {d.files}. "
                "Expected (mean,scale) or (mean,std)."
            )

        # avoid divide-by-zero
        scale = np.where(scale == 0, 1.0, scale).astype(np.float32)

        # --- optional feature order ---
        feat_order = None
        if "feature_order" in keys:
            # saved as np.array(feature_order) in training
            feat_order = [str(x) for x in d["feature_order"].tolist()]

        _ZPT_SCALER_CACHE[scaler_path] = (mean, scale, feat_order)

    return _ZPT_SCALER_CACHE[scaler_path]


def load_torchscript_cached(model_path: str, device: torch.device):
    key = (model_path, str(device))
    if key not in _ZPT_MODEL_CACHE:
        m = torch.jit.load(model_path, map_location=device)
        m.eval()
        _ZPT_MODEL_CACHE[key] = m
    return _ZPT_MODEL_CACHE[key]


def _to_1d_float32(x, default=0.0):
    x = ak.fill_none(x, default)
    x = ak.to_numpy(x)
    x = np.asarray(x, dtype=np.float32)
    x[~np.isfinite(x)] = np.float32(default)
    return x


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class ZptDNNConfig:
    model_path: str
    feature_names: List[str]  # must be an ordered list!
    # IMPORTANT: TS model outputs logits -> need odds weight = exp(logit)
    output_mode: str = "logit_to_odds"  # ["logit_to_odds", "logit_to_prob", "weight"]
    batch_size: int = 262144
    device: Optional[str] = None  # "cpu" or "cuda"
    # match the training clip (0.2, 5.0). To disable set to None.
    clip_weight_min: Optional[float] = 0.2
    clip_weight_max: Optional[float] = 5.0
    # numerical stability for exp/logit
    logit_clip: float = 50.0
    # if True, enforce scaler feature order == cfg.feature_names (recommended)
    check_feature_order: bool = True


# ----------------------------------------------------------------------
# Partition-local (CONCRETE awkward only) implementation
# ----------------------------------------------------------------------
def _eval_zpt_by_njet_one_partition(
    feat_rec: ak.Array,
    njets: ak.Array,
    cfg_base: ZptDNNConfig,
    model_paths_by_cat: Dict[str, str],
    scalar_paths_by_cat: Optional[Dict[str, str]] = None,
    *args,
    **kwargs,
) -> ak.Array:
    # Typetracer guard (Dask-Awkward graph optimization phase)
    if ak.backend(njets) == "typetracer" or ak.backend(feat_rec) == "typetracer":
        return ak.values_astype(ak.ones_like(njets), np.float32)

    # device
    dev = cfg_base.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev)

    # njets routing (concrete)
    nj = ak.to_numpy(ak.fill_none(njets, 0)).astype(np.int32)
    nj = np.clip(nj, 0, 2)

    masks = {
        "0j": (nj == 0),
        "1j": (nj == 1),
        "2j": (nj >= 2),
    }

    # build full X once (unscaled here; scale per-cat if we want per-cat scalers)
    cols = []
    for name in cfg_base.feature_names:
        if name not in feat_rec.fields:
            raise KeyError(
                f"Missing feature '{name}' in partition; got={feat_rec.fields}"
            )
        cols.append(_to_1d_float32(feat_rec[name], default=0.0))
    X = np.stack(cols, axis=1)  # (N, F)

    out = np.ones((X.shape[0],), dtype=np.float32)

    for cat, m in masks.items():
        if not m.any():
            continue

        model_path = model_paths_by_cat[cat]
        model = load_torchscript_cached(model_path, device)

        Xsub = X[m]

        # Apply scaler (if provided)
        if scalar_paths_by_cat is not None:
            scaler_path = scalar_paths_by_cat[cat]
            mean, scale, feat_order = load_scaler_cached(scaler_path)

            if cfg_base.check_feature_order and feat_order is not None:
                if list(feat_order) != list(cfg_base.feature_names):
                    raise ValueError(
                        f"Feature order mismatch for {cat}!\n"
                        f"  scaler.feature_order = {feat_order}\n"
                        f"  cfg.feature_names    = {cfg_base.feature_names}\n"
                        f"Scaler path: {scaler_path}"
                    )

            if mean.shape[0] != Xsub.shape[1] or scale.shape[0] != Xsub.shape[1]:
                raise ValueError(
                    f"Scaler dim mismatch for {cat}: "
                    f"X has {Xsub.shape[1]} features, mean={mean.shape}, scale={scale.shape} "
                    f"from {scaler_path}"
                )

            Xsub = (Xsub - mean) / scale

        # inference
        ysub = np.empty((Xsub.shape[0],), dtype=np.float32)
        bs = int(cfg_base.batch_size)

        for i in range(0, Xsub.shape[0], bs):
            xb = torch.from_numpy(Xsub[i : i + bs]).to(device)
            yb = model(xb)
            if yb.ndim == 2 and yb.shape[1] == 1:
                yb = yb[:, 0]
            ysub[i : i + bs] = yb.detach().to("cpu").numpy().astype(np.float32)

        # ----------------------------------------------------------
        # Convert model output -> weight consistently with training
        # Training: logit -> p=sigmoid(logit) -> w=p/(1-p) = exp(logit)
        # ----------------------------------------------------------
        if cfg_base.output_mode == "logit_to_odds":
            # safe exp(logit)
            lc = float(cfg_base.logit_clip)
            wsub = np.exp(np.clip(ysub, -lc, lc)).astype(np.float32)

        elif cfg_base.output_mode == "logit_to_prob":
            # if we  need p(Data|x) itself for diagnostics
            lc = float(cfg_base.logit_clip)
            z = np.clip(ysub, -lc, lc)
            wsub = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

        elif cfg_base.output_mode == "weight":
            # Only if the TS model already outputs odds weight directly
            wsub = ysub.astype(np.float32, copy=False)

        else:
            raise ValueError(f"Unknown output_mode={cfg_base.output_mode}")

        # sanitize + clip (match the training )
        wsub = np.nan_to_num(wsub, nan=1.0, posinf=1.0, neginf=1.0)

        if cfg_base.clip_weight_min is not None or cfg_base.clip_weight_max is not None:
            lo = (
                cfg_base.clip_weight_min
                if cfg_base.clip_weight_min is not None
                else -np.inf
            )
            hi = (
                cfg_base.clip_weight_max
                if cfg_base.clip_weight_max is not None
                else np.inf
            )
            wsub = np.clip(wsub, lo, hi)

        out[m] = wsub

    return ak.Array(out)


# ----------------------------------------------------------------------
# Dask-facing wrapper
# ----------------------------------------------------------------------
def eval_zpt_torchscript_by_njet(
    features: Dict[str, ak.Array],
    njets: ak.Array,
    cfg_base: ZptDNNConfig,
    model_paths_by_cat: Dict[str, str],  # {"0j":..., "1j":..., "2j":...}
    scalar_paths_by_cat: Optional[Dict[str, str]] = None,
):
    # record array for alignment
    feat_rec = ak.zip(
        {k: ak.fill_none(v, 0.0) for k, v in features.items()}, depth_limit=1
    )
    logger.info(f"eval_zpt_torchscript_by_njet: feat_rec.fields={feat_rec.fields}, njets.fields={njets.fields}")

    # If dask input: do per-partition inference
    if isinstance(feat_rec, dak.Array) or isinstance(njets, dak.Array):
        meta = ak.to_backend(ak.Array(np.empty((0,), dtype=np.float32)), "typetracer")
        return dak.map_partitions(
            _eval_zpt_by_njet_one_partition,
            feat_rec,
            njets,
            cfg_base,
            model_paths_by_cat,
            scalar_paths_by_cat,
            meta=meta,
        )

    return _eval_zpt_by_njet_one_partition(
        feat_rec, njets, cfg_base, model_paths_by_cat, scalar_paths_by_cat
    )
