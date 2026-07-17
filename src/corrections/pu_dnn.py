"""PU/HS jet-ID DNN scoring for the forward-jet (HE/HF) horn cleanup.

Consumes the per-region models trained by MVA_training/pileup_dnn/train_pu_dnn.py
(``model_torchscript.pt`` + ``scaler.json`` + ``summary_<region>.json`` under one
subdirectory per region: HEpos, HEneg, HFpos, HFneg) and evaluates them on the
stage-1 jet collection using the same feature list and scaling convention used
at training time (see ``Scaler.transform`` and ``flatten_jets`` in that script).
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List

import awkward as ak
import dask_awkward as dak
import numpy as np
import torch

from modules.utils import logger

REGIONS = ("HEpos", "HEneg", "HFpos", "HFneg")

# Repo root, resolved from this file's own location so a relative base_dir
# resolves correctly regardless of the caller's cwd (driver process vs. a
# Dask worker started from a different working directory).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PU_DNN_MODEL_CACHE: Dict[str, torch.jit.RecursiveScriptModule] = {}


def region_mask_eta(eta, region: str):
    abs_eta = np.abs(eta)
    if region == "HEpos":
        return (eta > 0) & (abs_eta >= 2.5) & (abs_eta <= 3.0)
    elif region == "HEneg":
        return (eta < 0) & (abs_eta >= 2.5) & (abs_eta <= 3.0)
    elif region == "HFpos":
        return (eta > 0) & (abs_eta > 3.0)
    elif region == "HFneg":
        return (eta < 0) & (abs_eta > 3.0)
    else:
        raise ValueError(f"Unknown region {region}")


@dataclass
class PUDnnRegionConfig:
    region: str
    model_path: str
    features: List[str]
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    threshold: float
    direction: str
    pt_min: float
    pt_turnoff: float


def load_pu_dnn_configs(base_dir: str) -> Dict[str, PUDnnRegionConfig]:
    """Load one PUDnnRegionConfig per region subdirectory found under base_dir."""
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(_PROJECT_ROOT, base_dir)

    configs: Dict[str, PUDnnRegionConfig] = {}
    for region in REGIONS:
        region_dir = os.path.join(base_dir, region)
        summary_path = os.path.join(region_dir, f"summary_{region}.json")
        scaler_path = os.path.join(region_dir, "scaler.json")
        model_path = os.path.join(region_dir, "model_torchscript.pt")
        if not (
            os.path.isfile(summary_path)
            and os.path.isfile(scaler_path)
            and os.path.isfile(model_path)
        ):
            logger.warning(
                f"PU DNN artifacts missing for region {region} under {base_dir}; skipping this region."
            )
            continue

        with open(summary_path) as f:
            summary = json.load(f)
        with open(scaler_path) as f:
            scaler = json.load(f)

        if list(scaler["features"]) != list(summary["features"]):
            raise ValueError(
                f"PU DNN feature-order mismatch between scaler.json and "
                f"summary_{region}.json in {region_dir}"
            )

        configs[region] = PUDnnRegionConfig(
            region=region,
            model_path=model_path,
            features=list(scaler["features"]),
            median=np.asarray(scaler["median"], dtype=np.float32),
            mean=np.asarray(scaler["mean"], dtype=np.float32),
            scale=np.asarray(scaler["scale"], dtype=np.float32),
            threshold=float(summary["threshold"]),
            direction=str(summary["direction"]),
            pt_min=float(summary["pt_min"]),
            pt_turnoff=float(summary["pt_turnoff"]),
        )
    return configs


def add_dphi_met_jet_features(jets: ak.Array, met_phi: ak.Array) -> ak.Array:
    """Attach per-jet minDPhiMetJet/maxDPhiMetJet fields to `jets`.

    Matches train_pu_dnn.py's flatten_jets(): min/max |dphi(MET, jet)| computed
    over the two leading jets in the event only, then broadcast onto every jet
    in that event (it is really an event-level MET-topology feature).
    """
    padded = ak.pad_none(jets, target=2, axis=1)
    lead_phi = padded[:, 0].phi
    sublead_phi = padded[:, 1].phi

    def _dphi(phi1, phi2):
        d = phi1 - phi2
        d = (d + np.pi) % (2.0 * np.pi) - np.pi
        return np.abs(d)

    dphi_lead = ak.fill_none(_dphi(lead_phi, met_phi), np.nan)
    dphi_sublead = ak.fill_none(_dphi(sublead_phi, met_phi), np.nan)

    min_dphi = np.fmin(dphi_lead, dphi_sublead)
    max_dphi = np.fmax(dphi_lead, dphi_sublead)

    zeros = jets.pt * 0.0
    jets = ak.with_field(jets, zeros + min_dphi[:, np.newaxis], "minDPhiMetJet")
    jets = ak.with_field(jets, zeros + max_dphi[:, np.newaxis], "maxDPhiMetJet")
    return jets


def _load_torchscript_cached(model_path: str, device: torch.device):
    key = f"{model_path}::{device}"
    if key not in _PU_DNN_MODEL_CACHE:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        _PU_DNN_MODEL_CACHE[key] = model
    return _PU_DNN_MODEL_CACHE[key]


def _scale_features(x: np.ndarray, cfg: PUDnnRegionConfig) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    bad = ~np.isfinite(x)
    if bad.any():
        rows, cols = np.where(bad)
        x[rows, cols] = cfg.median[cols]
    x = (x - cfg.mean) / cfg.scale
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _eval_pu_dnn_one_partition(
    feat_rec: ak.Array,
    configs: Dict[str, PUDnnRegionConfig],
    device_str: str,
) -> ak.Array:
    if ak.backend(feat_rec) == "typetracer":
        pass_dummy = ak.values_astype(ak.ones_like(feat_rec["pt"]), bool)
        score_dummy = ak.values_astype(ak.ones_like(feat_rec["pt"]), np.float32)
        return ak.zip({"pu_dnn_pass": pass_dummy, "pu_dnn_score": score_dummy})

    counts = ak.num(feat_rec, axis=1)
    flat = ak.flatten(feat_rec, axis=1)
    n = len(flat)

    eta = np.asarray(ak.to_numpy(flat["eta"]), dtype=np.float32)
    pt = np.asarray(ak.to_numpy(flat["pt"]), dtype=np.float32)

    combined_pass = np.ones(n, dtype=bool)
    combined_score = np.full(n, np.nan, dtype=np.float32)

    device = torch.device(device_str)
    for region, cfg in configs.items():
        mask_active = region_mask_eta(eta, region) & (pt >= cfg.pt_min) & (pt < cfg.pt_turnoff)
        if not mask_active.any():
            continue

        cols = [np.asarray(ak.to_numpy(flat[name]), dtype=np.float32) for name in cfg.features]
        X = np.stack(cols, axis=1)[mask_active]
        X = _scale_features(X, cfg)

        model = _load_torchscript_cached(cfg.model_path, device)
        with torch.no_grad():
            logits = model(torch.from_numpy(X).to(device))
            score_sub = torch.sigmoid(logits).detach().to("cpu").numpy().astype(np.float32)

        pass_sub = (
            score_sub > cfg.threshold if cfg.direction == "keep_high" else score_sub < cfg.threshold
        )

        combined_score[mask_active] = score_sub
        combined_pass[mask_active] = pass_sub

    jagged_pass = ak.unflatten(ak.Array(combined_pass), counts)
    jagged_score = ak.unflatten(ak.Array(combined_score), counts)
    return ak.zip({"pu_dnn_pass": jagged_pass, "pu_dnn_score": jagged_score})


def eval_pu_dnn(
    jets: ak.Array,
    configs: Dict[str, PUDnnRegionConfig],
    device: str = "cpu",
) -> ak.Array:
    """Evaluate the region PU DNNs on a per-event jagged jet collection.

    `jets` must already carry minDPhiMetJet/maxDPhiMetJet (see
    add_dphi_met_jet_features) plus the native NanoAOD fields referenced by
    each region's `features` list.

    Returns a jagged record array aligned with `jets`, with fields
    `pu_dnn_pass` (bool, True outside any active region/pT window) and
    `pu_dnn_score` (float32, NaN outside any active region/pT window).
    """
    needed = {"pt", "eta"}
    for cfg in configs.values():
        needed.update(cfg.features)
    feat_rec = ak.zip({name: jets[name] for name in sorted(needed)})

    if isinstance(feat_rec, dak.Array):
        empty_bool = ak.to_backend(ak.Array([[True]])[:0], "typetracer")
        empty_score = ak.to_backend(ak.Array([[np.float32(1.0)]])[:0], "typetracer")
        meta_out = ak.zip({"pu_dnn_pass": empty_bool, "pu_dnn_score": empty_score})
        return dak.map_partitions(
            _eval_pu_dnn_one_partition,
            feat_rec,
            configs,
            device,
            meta=meta_out,
        )

    return _eval_pu_dnn_one_partition(feat_rec, configs, device)
