#!/usr/bin/env python3
"""
Train small flat-input DNNs for PU/HS jet rejection from stage-1 parquet output.

The default workflow trains two independent classifiers:
  - HE: 2.5 <= |eta| < 3.0
  - HF: |eta| >= 3.0

The output mirrors the PySR pileup workflow where possible: one summary JSON per
region, a global summary, threshold rescans, score plots, and pT turn-on
efficiency/rejection plots.

Example command:
--------------------

time python MVA_training/pileup_dnn/train_pu_dnn.py \
    -i "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJets_June02_tightPassLepVeto_NoJER/stage1_output/2022postEE/compacted/dyTo2L_M-50_incl/*/*.parquet"     "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJets_June02_tightPassLepVeto_NoJER/stage1_output/2022postEE/compacted/ttjets_*/*/*.parquet"     "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJets_June02_tightPassLepVeto_NoJER/stage1_output/2022postEE/compacted/ewk_*/*/*.parquet" \
    --use-glob \
    -o validation/pu_dnn/run2022postEE_03June_DYIncl_OnlyJetRelatedVariables \
    --regions HEpos HEneg HFpos HFneg

time python MVA_training/pileup_dnn/train_pu_dnn.py \
    -i "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJets_June02_tightPassLepVeto_NoJER/stage1_output/2022postEE/compacted/dyTo2L_M-50_incl/*/*.parquet"     "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJets_June02_tightPassLepVeto_NoJER/stage1_output/2022postEE/compacted/ttjets_*/*/*.parquet"     "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJets_June02_tightPassLepVeto_NoJER/stage1_output/2022postEE/compacted/ewk_*/*/*.parquet" \
    --use-glob \
    -o validation/pu_dnn/ablation_scan_03June \
    --regions HEpos HEneg HFpos HFneg \
    --run-ablations

python MVA_training/pileup_dnn/train_pu_dnn.py \
    -i "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJets_June02_tightPassLepVeto_NoJER/stage1_output/2022postEE/compacted/dyTo2L_M-50_incl/*/*.parquet"     "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJets_June02_tightPassLepVeto_NoJER/stage1_output/2022postEE/compacted/ttjets_*/*/*.parquet"     "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJets_June02_tightPassLepVeto_NoJER/stage1_output/2022postEE/compacted/ewk_*/*/*.parquet" \
    -o validation/pu_dnn/run2022postEE_dy_top_ewk_02June_DYIncl_dPhi_NoPt \
    --regions HEpos HEneg HFpos HFneg \
    --replot-only    
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset


CORE_VARS = ("pt", "eta", "hasMatchedGenJet")
DEFAULT_REGIONS = ("HE", "HF")
REGION_CHOICES = ("HE", "HF", "HEpos", "HEneg", "HFpos", "HFneg")
DEFAULT_PT_BINS = [25, 27, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50]

# These are the only DNN input features. Branch-name aliases protect us from
# drift between NanoAOD/stage-1 versions.
FEATURE_ALIASES = {
    "chEmEF": ("chEmEF",),
    "chHEF": ("chHEF",),
    "neEmEF": ("neEmEF",),
    "neHEF": ("neHEF",),
    "muEF": ("muEF",),
    "hfEmEF": ("hfEmEF",),
    "hfHEF": ("hfHEF",),
    "nConstituents": ("nConstituents",),
    "nElectrons": ("nElectrons",),
    "nMuons": ("nMuons",),
    "mass": ("mass",),
    "area": ("area",),
    "rawFactor": ("rawFactor",),
    "muonSubtrFactor": ("muonSubtrFactor",),
    "chMultiplicity": ("chMultiplicity",),
    "neMultiplicity": ("neMultiplicity",),
    "muonSubtrDeltaEta": ("muonSubtrDeltaEta",),
    "muonSubtrDeltaPhi": ("muonSubtrDeltaPhi",),
    "hfadjacentEtaStripsSize": ("hfadjacentEtaStripsSize",),
    "hfcentralEtaStripSize": ("hfcentralEtaStripSize",),
    "hfsigmaEtaEta": ("hfsigmaEtaEta",),
    "hfsigmaPhiPhi": ("hfsigmaPhiPhi",),
}

BASELINE_ALIASES = {
    "puIdDisc": ("puIdDisc", "puId", "puId17"),
}

MODEL_FEATURES = [
    "logpt",
    "minDPhiMetJet",
    "chEmEF",
    "chHEF",
    "neEmEF",
    "neHEF",
    "muEF",
    "chMultiplicity",
    "neMultiplicity",
    "nConstituents",
    "nElectrons",
    "nMuons",
    "muonSubtrFactor",
    "muonSubtrDeltaEta",
    "muonSubtrDeltaPhi",
    "hfadjacentEtaStripsSize",
    "hfcentralEtaStripSize",
    "hfsigmaEtaEta",
    "hfsigmaPhiPhi",
    "hfEmEF",
    "hfHEF",
]


@dataclass
class Scaler:
    features: list[str]
    median: list[float]
    mean: list[float]
    scale: list[float]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.features].to_numpy(dtype=np.float32, copy=True)
        med = np.asarray(self.median, dtype=np.float32)
        mean = np.asarray(self.mean, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        bad = ~np.isfinite(x)
        if bad.any():
            x[bad] = np.take(med, np.where(bad)[1])
        x = (x - mean) / scale
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class JetDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        if w is None:
            w = np.ones(len(y), dtype=np.float32)
        self.w = torch.as_tensor(w, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.w[idx]


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: Iterable[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        width_in = n_features
        for width in hidden:
            layers.append(nn.Linear(width_in, width))
            layers.append(nn.LayerNorm(width))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width_in = width
        layers.append(nn.Linear(width_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DNN PU/HS jet classifiers from stage-1 parquet output."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        nargs="+",
        help="One or more parquet files, globs, or directories. Pass DY, TOP, and EWK samples here.",
    )
    parser.add_argument("-o", "--output", required=True, help="Output directory.")
    parser.add_argument(
        "--regions",
        nargs="+",
        default=list(DEFAULT_REGIONS),
        choices=REGION_CHOICES,
        help="Eta regions to train. Defaults to separate HE and HF models.",
    )
    parser.add_argument("--use-glob", action="store_true", help="Expand input as a glob.")
    parser.add_argument("--variation", default="nominal", help="Stage-1 jet variation suffix.")
    parser.add_argument("--max-jets", type=int, default=2, help="Number of jet prefixes to flatten.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap after parquet load.")
    parser.add_argument("--pt-min", type=float, default=25.0, help="Minimum jet pT for training.")
    parser.add_argument("--pt-max", type=float, default=50.0, help="Maximum jet pT for training.")
    parser.add_argument(
        "--pt-bins",
        nargs="+",
        type=float,
        default=DEFAULT_PT_BINS,
        help="pT bins for HS efficiency and PU rejection plots.",
    )
    parser.add_argument("--hs-eff", type=float, default=0.80, help="Target HS efficiency.")
    parser.add_argument(
        "--hs-scan",
        nargs="+",
        type=float,
        default=None,
        help="Optional HS efficiency grid for threshold rescans.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden", nargs="+", type=int, default=[64, 64, 32])
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--use-weights",
        action="store_true",
        help="Use event weights if a weight column is available.",
    )
    parser.add_argument(
        "--no-class-balance",
        action="store_true",
        help="Do not reweight HS and PU classes to equal total weight.",
    )
    parser.add_argument(
        "--no-sample-balance",
        action="store_true",
        help="Do not reweight DY/TOP/EWK sample groups to equal total weight.",
    )
    parser.add_argument(
        "--sample-balance-groups",
        nargs="+",
        default=["DY", "TOP", "EWK"],
        help="Sample groups to balance. Defaults to DY TOP EWK.",
    )
    parser.add_argument("--weight-col", default="wgt_nominal")
    parser.add_argument("--weight-clip", type=float, default=50.0)
    parser.add_argument(
        "--importance-max-rows",
        type=int,
        default=20000,
        help="Maximum test rows used for permutation feature importance.",
    )
    parser.add_argument(
        "--importance-repeats",
        type=int,
        default=20,
        help="Number of repeated shuffles per feature for permutation importance.",
    )
    parser.add_argument(
        "--run-ablations",
        action="store_true",
        help="Train automatic feature ablation variants and write summary CSVs.",
    )
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Regenerate validation plots from saved predictions/metrics in an existing output directory.",
    )
    parser.add_argument(
        "--ablation-groups",
        nargs="+",
        default=[
            "pt=logpt",
            "met=minDPhiMetJet",
            "ptMET=logpt,minDPhiMetJet",
            "neutral=neEmEF,neHEF",
            "charged=chEmEF,chHEF,nElectrons",
            "muon=muEF,nMuons,muonSubtrFactor,muonSubtrDeltaEta,muonSubtrDeltaPhi",
            "multiplicity=chMultiplicity,neMultiplicity,nConstituents",
            "hf_strips=hfadjacentEtaStripsSize,hfcentralEtaStripSize",
            "hf_shape=hfsigmaEtaEta,hfsigmaPhiPhi",
            "hf_energy=hfEmEF,hfHEF",
        ],
        help="Named feature groups for automatic group-drop ablations as NAME=feat1,feat2,...",
    )
    parser.add_argument(
        "--plot-format",
        nargs="+",
        default=["png", "pdf"],
        choices=("png", "pdf"),
        help="Plot file formats.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip validation plots.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def expand_inputs(input_paths: list[str], use_glob: bool) -> list[str]:
    paths = []
    for input_path in input_paths:
        path = Path(input_path)
        if use_glob:
            expanded = sorted(glob(input_path))
        elif path.is_dir():
            expanded = sorted(str(p) for p in path.rglob("*.parquet"))
        else:
            expanded = [input_path]
        paths.extend(p for p in expanded if Path(p).exists())
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise FileNotFoundError(f"No parquet inputs found from: {input_paths}")
    return paths


def parquet_columns(path: str) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).schema.names
    except Exception:
        return list(pd.read_parquet(path).columns)


def infer_sample_name(path: str) -> str:
    parts = Path(path).parts
    for marker in ("f1_0", "f0_1"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    for idx, part in enumerate(parts):
        if part.startswith("compacted") and idx + 1 < len(parts):
            return parts[idx + 1]
    parent = Path(path).parent
    if parent.name.isdigit() and parent.parent.name:
        return parent.parent.name
    return parent.name


def infer_sample_group(sample_name: str) -> str:
    name = sample_name.lower()
    if name.startswith("dy") or "dyto" in name or "dy_" in name:
        return "DY"
    if (
        name.startswith("tt")
        or "ttjets" in name
        or name.startswith("st")
        or "single_top" in name
        or "top" in name
    ):
        return "TOP"
    if name.startswith("ewk") or "ewk" in name:
        return "EWK"
    return "OTHER"


def jet_col(prefix: str, var: str, variation: str) -> str:
    return f"{prefix}{var}_{variation}"


def delta_phi(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def possible_cols(prefix: str, feature: str, aliases: dict[str, tuple[str, ...]], variation: str) -> list[str]:
    cols = []
    for alias in aliases[feature]:
        cols.append(jet_col(prefix, alias, variation))
        cols.append(f"{prefix}{alias}")
    return cols


def needed_columns(available: set[str], variation: str, max_jets: int, weight_col: str) -> list[str]:
    requested: set[str] = {"event", "run", "luminosityBlock", weight_col}
    requested.update({"MET_phi", "PuppiMET_phi"})
    for jidx in range(1, max_jets + 1):
        prefix = f"jet{jidx}_"
        for var in CORE_VARS:
            requested.add(jet_col(prefix, var, variation))
            requested.add(f"{prefix}{var}")
        requested.add(jet_col(prefix, "phi", variation))
        requested.add(f"{prefix}phi")
        for feature in FEATURE_ALIASES:
            requested.update(possible_cols(prefix, feature, FEATURE_ALIASES, variation))
        for feature in BASELINE_ALIASES:
            requested.update(possible_cols(prefix, feature, BASELINE_ALIASES, variation))
    return sorted(requested & available)


def load_stage1(paths: list[str], args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    rows_left = args.max_rows

    for path in paths:
        if rows_left is not None and rows_left <= 0:
            break
        local_cols = needed_columns(set(parquet_columns(path)), args.variation, args.max_jets, args.weight_col)
        if not local_cols:
            continue
        frame = pd.read_parquet(path, columns=local_cols)
        if rows_left is not None:
            frame = frame.head(rows_left)
            rows_left -= len(frame)
        sample_name = infer_sample_name(path)
        frame["__sample_name"] = sample_name
        frame["__sample_group"] = infer_sample_group(sample_name)
        frames.append(frame)

    if not frames:
        raise ValueError("No readable parquet frames with stage-1 jet columns were found.")
    return pd.concat(frames, axis=0, ignore_index=True)


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def numeric_series(df: pd.DataFrame, colname: str, dtype=np.float32) -> np.ndarray:
    return pd.to_numeric(df[colname], errors="coerce").astype(dtype).to_numpy()


def flatten_jets(df_in: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    met_phi_col = first_existing_column(df_in, ["MET_phi", "PuppiMET_phi"])
    met_phi = numeric_series(df_in, met_phi_col) if met_phi_col is not None else None
    min_dphi_met_jet = None
    if met_phi is not None:
        dphi_stack = []
        for jidx in range(1, args.max_jets + 1):
            prefix = f"jet{jidx}_"
            phi_col = first_existing_column(
                df_in,
                [jet_col(prefix, "phi", args.variation), f"{prefix}phi"],
            )
            if phi_col is None:
                continue
            jet_phi = numeric_series(df_in, phi_col)
            dphi = np.abs(delta_phi(jet_phi, met_phi)).astype(np.float32)
            dphi[~(np.isfinite(jet_phi) & np.isfinite(met_phi))] = np.nan
            dphi_stack.append(dphi)
        if dphi_stack:
            min_dphi_met_jet = np.nanmin(np.vstack(dphi_stack), axis=0).astype(np.float32)

    for jidx in range(1, args.max_jets + 1):
        prefix = f"jet{jidx}_"
        pt_col = first_existing_column(
            df_in, [jet_col(prefix, "pt", args.variation), f"{prefix}pt"]
        )
        eta_col = first_existing_column(
            df_in, [jet_col(prefix, "eta", args.variation), f"{prefix}eta"]
        )
        label_col = first_existing_column(
            df_in,
            [
                jet_col(prefix, "hasMatchedGenJet", args.variation),
                f"{prefix}hasMatchedGenJet",
            ],
        )
        if pt_col is None or eta_col is None or label_col is None:
            continue

        pt = numeric_series(df_in, pt_col)
        eta = numeric_series(df_in, eta_col)
        y_raw = numeric_series(df_in, label_col)
        exists = np.isfinite(pt) & np.isfinite(eta) & (pt > 0) & np.isfinite(y_raw)
        if not exists.any():
            continue

        frame = pd.DataFrame(
            {
                "pt": pt[exists],
                "eta": eta[exists],
                "aeta": np.abs(eta[exists]).astype(np.float32),
                "y_hs": (y_raw[exists] > 0.5).astype(np.int8),
                "jidx": np.full(np.count_nonzero(exists), jidx - 1, dtype=np.float32),
            }
        )
        if min_dphi_met_jet is not None:
            frame["minDPhiMetJet"] = min_dphi_met_jet[exists]
        for meta_col in (
            "event",
            "run",
            "luminosityBlock",
            args.weight_col,
            "__sample_name",
            "__sample_group",
        ):
            if meta_col in df_in.columns:
                frame[meta_col] = df_in.loc[exists, meta_col].to_numpy()

        for feature in FEATURE_ALIASES:
            source = first_existing_column(
                df_in, possible_cols(prefix, feature, FEATURE_ALIASES, args.variation)
            )
            if source is not None:
                frame[feature] = numeric_series(df_in, source)[exists]

        for feature in BASELINE_ALIASES:
            source = first_existing_column(
                df_in, possible_cols(prefix, feature, BASELINE_ALIASES, args.variation)
            )
            if source is not None:
                frame[feature] = numeric_series(df_in, source)[exists]

        frames.append(frame)

    if not frames:
        raise KeyError(
            "Could not find any complete jet{N}_{pt,eta,hasMatchedGenJet}_nominal branches."
        )

    jets = pd.concat(frames, axis=0, ignore_index=True)
    jets = derive_features(jets)
    jets = cleanup_numeric(jets)
    mask_pt = (jets["pt"] >= args.pt_min) & (jets["pt"] < args.pt_max)
    jets = jets[mask_pt].copy()
    if len(jets) == 0:
        raise ValueError(f"No jets survive {args.pt_min} <= pt < {args.pt_max}.")
    return jets


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["logpt"] = np.log1p(np.clip(df["pt"].to_numpy(dtype=np.float32), 0, None))
    return df


def cleanup_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if col in {"run", "luminosityBlock", "event"}:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            values = pd.to_numeric(df[col], errors="coerce")
            values = values.mask(values <= -99.0)
            df[col] = values.astype(np.float32)
    return df


def region_mask_eta(eta: np.ndarray, region: str) -> np.ndarray:
    if region == "HE":
        return (np.abs(eta) >= 2.5) & (np.abs(eta) < 3.0)
    if region == "HF":
        return np.abs(eta) >= 3.0
    if region == "HEpos":
        return (eta >= 2.5) & (eta < 3.0)
    if region == "HEneg":
        return (eta <= -2.5) & (eta > -3.0)
    if region == "HFpos":
        return eta >= 3.0
    if region == "HFneg":
        return eta <= -3.0
    raise ValueError(f"Unknown region: {region}")


def stratified_random_split(df: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train = np.zeros(len(df), dtype=bool)
    val = np.zeros(len(df), dtype=bool)
    test = np.zeros(len(df), dtype=bool)
    y = df["y_hs"].to_numpy(dtype=np.int8)
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        idx = rng.permutation(idx)
        n_train = int(0.60 * len(idx))
        n_val = int(0.20 * len(idx))
        train[idx[:n_train]] = True
        val[idx[n_train : n_train + n_val]] = True
        test[idx[n_train + n_val :]] = True
    return train, val, test


def has_both_classes(df: pd.DataFrame, mask: np.ndarray) -> bool:
    return len(np.unique(df.loc[mask, "y_hs"].to_numpy(dtype=np.int8))) == 2


def split_by_event_or_random(df: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(df)
    if "event" in df.columns:
        event = pd.to_numeric(df["event"], errors="coerce").fillna(-1).astype(np.int64)
        mod = np.mod(np.abs(event.to_numpy()), 10)
        test = mod < 2
        val = (mod >= 2) & (mod < 4)
        train = ~(test | val)
    else:
        train, val, test = stratified_random_split(df, seed)

    # Fall back if an event modulo split accidentally empties a partition or a
    # class. This keeps validation thresholds meaningful on small test samples.
    if (
        min(train.sum(), val.sum(), test.sum()) == 0
        or not has_both_classes(df, train)
        or not has_both_classes(df, val)
        or not has_both_classes(df, test)
    ):
        tmp = df.drop(columns=["event"], errors="ignore")
        return stratified_random_split(tmp, seed)
    return train, val, test


def selected_features(df: pd.DataFrame, region: str) -> list[str]:
    keep = []
    for feat in MODEL_FEATURES:
        if feat not in df.columns:
            continue
        values = pd.to_numeric(df[feat], errors="coerce").to_numpy(dtype=np.float32)
        finite = np.isfinite(values)
        if finite.sum() == 0:
            continue
        if np.nanstd(values[finite]) <= 1e-8:
            continue
        keep.append(feat)
    return keep


def parse_ablation_groups(specs: list[str]) -> list[tuple[str, list[str]]]:
    groups = []
    for spec in specs:
        name, sep, payload = spec.partition("=")
        if not sep:
            raise ValueError(
                f"Invalid ablation group '{spec}'. Expected NAME=feat1,feat2,..."
            )
        features = [feat.strip() for feat in payload.split(",") if feat.strip()]
        if not features:
            continue
        groups.append((name.strip(), features))
    return groups


def build_ablation_variants(
    baseline_features: list[str], args: argparse.Namespace
) -> list[tuple[str, list[str], str]]:
    variants: list[tuple[str, list[str], str]] = []
    seen: set[tuple[str, ...]] = set()

    for feat in baseline_features:
        feature_set = [item for item in baseline_features if item != feat]
        key = tuple(feature_set)
        if len(feature_set) == 0 or key in seen:
            continue
        seen.add(key)
        variants.append((f"drop_{feat}", feature_set, f"drop feature {feat}"))

    for group_name, group_features in parse_ablation_groups(args.ablation_groups):
        feature_set = [feat for feat in baseline_features if feat not in set(group_features)]
        key = tuple(feature_set)
        if len(feature_set) == 0 or key in seen or len(feature_set) == len(baseline_features):
            continue
        seen.add(key)
        dropped = [feat for feat in baseline_features if feat not in feature_set]
        variants.append(
            (
                f"drop_group_{group_name}",
                feature_set,
                f"drop group {group_name}: {','.join(dropped)}",
            )
        )
    return variants


def summary_to_ablation_row(summary: dict, tag: str, notes: str) -> dict:
    metrics = summary.get("metrics", {})
    return {
        "region": summary.get("region"),
        "tag": tag,
        "notes": notes,
        "n_features": len(summary.get("features", [])),
        "features": ",".join(summary.get("features", [])),
        "threshold": summary.get("threshold"),
        "direction": summary.get("direction"),
        "hs_efficiency": summary.get("hs_efficiency"),
        "pu_rejection": summary.get("pu_rejection"),
        "test_auc": metrics.get("test_auc"),
        "val_auc": metrics.get("val_auc"),
        "train_auc": metrics.get("train_auc"),
        "test_average_precision": metrics.get("test_average_precision"),
        "best_epoch": metrics.get("best_epoch"),
        "n_train": metrics.get("n_train"),
        "n_val": metrics.get("n_val"),
        "n_test": metrics.get("n_test"),
    }


def make_scaler(df_train: pd.DataFrame, features: list[str]) -> Scaler:
    values = df_train[features].to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    median = np.nanmedian(values, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    filled = np.where(np.isfinite(values), values, median)
    mean = np.mean(filled, axis=0)
    scale = np.std(filled, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return Scaler(
        features=features,
        median=median.astype(float).tolist(),
        mean=mean.astype(float).tolist(),
        scale=scale.astype(float).tolist(),
    )


def apply_sample_group_balance(weights: np.ndarray, df: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    if args.no_sample_balance or "__sample_group" not in df.columns:
        return weights

    groups = df["__sample_group"].astype(str).to_numpy()
    target_groups = [str(group) for group in args.sample_balance_groups]
    y = df["y_hs"].to_numpy(dtype=bool) if "y_hs" in df.columns else None
    class_masks = [np.ones(len(df), dtype=bool)]
    if y is not None:
        class_masks = [y, ~y]

    balanced = weights.astype(np.float32, copy=True)
    for class_mask in class_masks:
        present_groups = [
            group
            for group in target_groups
            if np.any(class_mask & (groups == group) & (balanced > 0))
        ]
        if len(present_groups) < 2:
            continue

        target_mask = class_mask & np.isin(groups, present_groups)
        total_weight = float(np.sum(balanced[target_mask]))
        if total_weight <= 0:
            continue

        target_per_group = total_weight / len(present_groups)
        for group in present_groups:
            group_mask = class_mask & (groups == group)
            group_weight = float(np.sum(balanced[group_mask]))
            if group_weight > 0:
                balanced[group_mask] *= target_per_group / group_weight

    return balanced


def event_weights(df: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    if not args.use_weights or args.weight_col not in df.columns:
        weights = np.ones(len(df), dtype=np.float32)
    else:
        weights = pd.to_numeric(df[args.weight_col], errors="coerce").to_numpy(dtype=np.float32)
        weights = np.abs(weights)
        weights = np.nan_to_num(weights, nan=1.0, posinf=args.weight_clip, neginf=1.0)
        weights = np.clip(weights, 0.0, args.weight_clip)

    weights = apply_sample_group_balance(weights, df, args)

    if not args.no_class_balance and "y_hs" in df.columns:
        y = df["y_hs"].to_numpy(dtype=bool)
        total = np.sum(weights)
        hs_sum = np.sum(weights[y])
        pu_sum = np.sum(weights[~y])
        if total > 0 and hs_sum > 0 and pu_sum > 0:
            weights[y] *= 0.5 * total / hs_sum
            weights[~y] *= 0.5 * total / pu_sum

    mean = weights[weights > 0].mean() if np.any(weights > 0) else 1.0
    return (weights / mean).astype(np.float32)


def sample_group_summary(df: pd.DataFrame, weights: np.ndarray | None = None) -> dict:
    if "__sample_group" not in df.columns:
        return {}
    tmp = df[["__sample_group", "y_hs"]].copy()
    tmp["weight"] = weights if weights is not None else np.ones(len(tmp), dtype=np.float32)
    rows = {}
    for group, group_df in tmp.groupby("__sample_group", dropna=False):
        y = group_df["y_hs"].astype(bool)
        rows[str(group)] = {
            "n_jets": int(len(group_df)),
            "n_hs": int(y.sum()),
            "n_pu": int((~y).sum()),
            "weight_sum": float(group_df["weight"].sum()),
            "weight_hs": float(group_df.loc[y, "weight"].sum()),
            "weight_pu": float(group_df.loc[~y, "weight"].sum()),
        }
    return rows


def train_one_model(
    df_region: pd.DataFrame,
    features: list[str],
    outdir: Path,
    args: argparse.Namespace,
) -> tuple[MLP, Scaler, dict, pd.DataFrame]:
    train_mask, val_mask, test_mask = split_by_event_or_random(df_region, args.seed)

    df_train = df_region[train_mask].copy()
    df_val = df_region[val_mask].copy()
    df_test = df_region[test_mask].copy()

    scaler = make_scaler(df_train, features)
    x_train = scaler.transform(df_train)
    x_val = scaler.transform(df_val)
    x_test = scaler.transform(df_test)

    y_train = df_train["y_hs"].to_numpy(dtype=np.float32)
    y_val = df_val["y_hs"].to_numpy(dtype=np.float32)
    y_test = df_test["y_hs"].to_numpy(dtype=np.float32)
    w_train = event_weights(df_train, args)
    w_val = event_weights(df_val, args)
    w_test = event_weights(df_test, args)

    device = choose_device(args.device)
    print(f"device: {device}")
    model = MLP(len(features), args.hidden, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    train_loader = DataLoader(
        JetDataset(x_train, y_train, w_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    history = []
    best_auc = -np.inf
    best_epoch = -1
    best_state = None
    bad_epochs = 0

    for epoch in range(args.epochs):
        model.train()
        batch_losses = []
        for xb, yb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss = (loss * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        train_score = predict_scores(model, x_train, device, args.batch_size)
        val_score = predict_scores(model, x_val, device, args.batch_size)
        train_auc = safe_auc(y_train, train_score, w_train)
        val_auc = safe_auc(y_val, val_score, w_val)
        train_loss = float(np.mean(batch_losses))
        val_loss = bce_numpy(y_val, val_score, w_val)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_auc": train_auc,
                "val_auc": val_auc,
            }
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_score = predict_scores(model, x_test, device, args.batch_size)
    val_score = predict_scores(model, x_val, device, args.batch_size)
    train_score = predict_scores(model, x_train, device, args.batch_size)

    pred_cols = list(dict.fromkeys(["pt", "eta", "aeta", "y_hs", "jidx"] + features))
    pred_test = df_test[pred_cols].copy()
    for meta in ("event", "run", "luminosityBlock", "__sample_name", "__sample_group"):
        if meta in df_test.columns:
            pred_test[meta] = df_test[meta].to_numpy()
    pred_test["score"] = test_score
    pred_test["weight"] = w_test
    if "puIdDisc" in df_test.columns:
        pred_test["puIdDisc"] = df_test["puIdDisc"].to_numpy(dtype=np.float32)

    metrics = {
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "n_features": int(len(features)),
        "best_epoch": int(best_epoch),
        "train_auc": safe_auc(y_train, train_score, w_train),
        "val_auc": safe_auc(y_val, val_score, w_val),
        "test_auc": safe_auc(y_test, test_score, w_test),
        "test_average_precision": safe_ap(y_test, test_score, w_test),
        "sample_group_balance": not args.no_sample_balance,
        "sample_balance_groups": args.sample_balance_groups,
        "train_sample_groups": sample_group_summary(df_train, w_train),
        "val_sample_groups": sample_group_summary(df_val, w_val),
        "test_sample_groups": sample_group_summary(df_test, w_test),
        "history": history,
    }

    torch.save(model.state_dict(), outdir / "model_best.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "features": features,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "scaler": asdict(scaler),
        },
        outdir / "checkpoint.pt",
    )
    model.eval()
    example = torch.zeros(1, len(features), dtype=torch.float32, device=device)
    traced = torch.jit.trace(model, example)
    traced.save(str(outdir / "model_torchscript.pt"))

    (outdir / "features.json").write_text(json.dumps(features, indent=2))
    (outdir / "scaler.json").write_text(json.dumps(asdict(scaler), indent=2))
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_table(pred_test, outdir / "predictions_test.parquet")
    return model, scaler, metrics, pred_test


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    return torch.device(device_arg)


def write_table(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception:
        df.to_csv(path.with_suffix(".csv"), index=False)


def predict_scores(model: MLP, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    scores = []
    loader = DataLoader(torch.as_tensor(x, dtype=torch.float32), batch_size=batch_size)
    with torch.no_grad():
        for xb in loader:
            logits = model(xb.to(device))
            scores.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(scores).astype(np.float32)


def safe_auc(y: np.ndarray, score: np.ndarray, weight: np.ndarray | None = None) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, score, sample_weight=weight))
    except ValueError:
        return float("nan")


def safe_ap(y: np.ndarray, score: np.ndarray, weight: np.ndarray | None = None) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y, score, sample_weight=weight))
    except ValueError:
        return float("nan")


def bce_numpy(y: np.ndarray, score: np.ndarray, weight: np.ndarray | None = None) -> float:
    eps = 1e-7
    score = np.clip(score, eps, 1 - eps)
    loss = -(y * np.log(score) + (1 - y) * np.log(1 - score))
    if weight is None:
        return float(np.mean(loss))
    return float(np.sum(loss * weight) / max(np.sum(weight), eps))


def weighted_fraction(mask: np.ndarray, weight: np.ndarray | None = None) -> float:
    if weight is None:
        return float(np.mean(mask)) if len(mask) else float("nan")
    denom = np.sum(weight)
    if denom <= 0:
        return float("nan")
    return float(np.sum(weight[mask] if mask.dtype == bool else mask * weight) / denom)


def weighted_quantile(values: np.ndarray, quantile: float, weight: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=np.float64)
    quantile = float(np.clip(quantile, 0.0, 1.0))
    if len(values) == 0:
        return float("nan")
    if weight is None:
        return float(np.quantile(values, quantile))
    weight = np.asarray(weight, dtype=np.float64)
    order = np.argsort(values)
    values = values[order]
    weight = weight[order]
    cdf = np.cumsum(weight)
    if cdf[-1] <= 0:
        return float(np.quantile(values, quantile))
    cdf /= cdf[-1]
    return float(np.interp(quantile, cdf, values))


def threshold_and_direction(
    score: np.ndarray,
    y: np.ndarray,
    target_hs_eff: float,
    weight: np.ndarray | None = None,
) -> tuple[float, str, float, float]:
    y_bool = y.astype(bool)
    hs_scores = score[y_bool]
    pu_scores = score[~y_bool]
    hs_w = weight[y_bool] if weight is not None else None
    pu_w = weight[~y_bool] if weight is not None else None
    if len(hs_scores) == 0 or len(pu_scores) == 0:
        return float("nan"), "keep_high", float("nan"), float("nan")

    candidates = []
    thr_high = weighted_quantile(hs_scores, 1.0 - target_hs_eff, hs_w)
    pass_high_hs = hs_scores > thr_high
    pass_high_pu = pu_scores > thr_high
    hs_eff_high = weighted_fraction(pass_high_hs, hs_w)
    pu_rej_high = 1.0 - weighted_fraction(pass_high_pu, pu_w)
    candidates.append((thr_high, "keep_high", hs_eff_high, pu_rej_high))

    thr_low = weighted_quantile(hs_scores, target_hs_eff, hs_w)
    pass_low_hs = hs_scores < thr_low
    pass_low_pu = pu_scores < thr_low
    hs_eff_low = weighted_fraction(pass_low_hs, hs_w)
    pu_rej_low = 1.0 - weighted_fraction(pass_low_pu, pu_w)
    candidates.append((thr_low, "keep_low", hs_eff_low, pu_rej_low))

    candidates = [c for c in candidates if np.isfinite(c[0]) and np.isfinite(c[3])]
    if not candidates:
        return float("nan"), "keep_high", float("nan"), float("nan")
    return max(candidates, key=lambda item: item[3])


def pass_from_threshold(score: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    if direction == "keep_high":
        return score > threshold
    if direction == "keep_low":
        return score < threshold
    raise ValueError(f"Unknown direction: {direction}")


def wp_vs_pt(
    df: pd.DataFrame,
    pass_mask: np.ndarray,
    pt_bins: list[float],
    weight: np.ndarray | None = None,
) -> pd.DataFrame:
    rows = []
    pt = df["pt"].to_numpy(dtype=np.float32)
    y = df["y_hs"].to_numpy(dtype=bool)
    for lo, hi in zip(pt_bins[:-1], pt_bins[1:]):
        in_bin = (pt >= lo) & (pt < hi)
        hs = in_bin & y
        pu = in_bin & ~y
        hs_eff = weighted_fraction(pass_mask[hs], weight[hs] if weight is not None else None)
        pu_rej = 1.0 - weighted_fraction(pass_mask[pu], weight[pu] if weight is not None else None)
        rows.append(
            {
                "pt_low": float(lo),
                "pt_high": float(hi),
                "pt_center": float(0.5 * (lo + hi)),
                "hs_eff": hs_eff,
                "pu_rejection": pu_rej,
                "n_hs": int(hs.sum()),
                "n_pu": int(pu.sum()),
            }
        )
    return pd.DataFrame(rows)


def save_plot(fig: plt.Figure, outbase: Path, formats: Iterable[str]) -> None:
    for fmt in formats:
        fig.savefig(outbase.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)


def plot_training_history(history: list[dict], outdir: Path, region: str, formats: list[str]) -> None:
    if not history:
        return
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(h["epoch"], h["train_loss"], label="train")
    axes[0].plot(h["epoch"], h["val_loss"], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(h["epoch"], h["train_auc"], label="train")
    axes[1].plot(h["epoch"], h["val_auc"], label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Weighted AUC")
    axes[1].set_ylim(0.45, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"{region} DNN training")
    save_plot(fig, outdir / f"training_history_{region}", formats)


def plot_roc_pr(
    y: np.ndarray,
    score: np.ndarray,
    weight: np.ndarray,
    outdir: Path,
    region: str,
    formats: list[str],
) -> None:
    if len(np.unique(y)) < 2:
        return
    fpr, tpr, _ = roc_curve(y, score, sample_weight=weight)
    auc = safe_auc(y, score, weight)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, label=f"DNN AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="random")
    ax.set_xlabel("PU acceptance")
    ax.set_ylabel("HS efficiency")
    ax.set_title(f"{region} ROC")
    ax.grid(alpha=0.3)
    ax.legend()
    save_plot(fig, outdir / f"roc_{region}", formats)

    precision, recall, _ = precision_recall_curve(y, score, sample_weight=weight)
    ap = safe_ap(y, score, weight)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(recall, precision, label=f"DNN AP = {ap:.4f}")
    ax.set_xlabel("HS efficiency")
    ax.set_ylabel("HS purity")
    ax.set_title(f"{region} precision-recall")
    ax.grid(alpha=0.3)
    ax.legend()
    save_plot(fig, outdir / f"precision_recall_{region}", formats)


def plot_score_real_fake(
    score: np.ndarray,
    y: np.ndarray,
    outdir: Path,
    region: str,
    formats: list[str],
    tag: str = "dnn",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    lo = float(np.nanmin(score))
    hi = float(np.nanmax(score))
    if not np.isfinite(lo) or not np.isfinite(hi):
        plt.close(fig)
        return
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    bins = np.linspace(lo, hi, 51)
    ax.hist(score[y.astype(bool)], bins=bins, histtype="step", linewidth=2, density=True, label="HS")
    ax.hist(score[~y.astype(bool)], bins=bins, histtype="step", linewidth=2, density=True, label="PU")
    ax.set_xlabel("DNN HS score" if tag == "dnn" else tag)
    ax.set_ylabel("Normalized jets")
    ax.set_title(f"{region} score: HS vs PU")
    ax.grid(alpha=0.3)
    ax.legend()
    outbase = outdir / f"score_real_fake_{region}"
    if tag != "dnn":
        outbase = outdir / f"score_real_fake_{region}_{tag}"
    save_plot(fig, outbase, formats)


def plot_confusion(
    y: np.ndarray,
    pass_mask: np.ndarray,
    outdir: Path,
    region: str,
    formats: list[str],
) -> None:
    counts = confusion_matrix(y.astype(int), pass_mask.astype(int), labels=[0, 1])
    row_sums = counts.sum(axis=1, keepdims=True)
    cm_percent = np.divide(
        counts * 100.0,
        row_sums,
        out=np.zeros_like(counts, dtype=float),
        where=row_sums > 0,
    )
    disp = ConfusionMatrixDisplay(cm_percent, display_labels=["PU", "HS"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".1f")
    ax.set_xticklabels(["PU/reject", "HS/keep"])
    ax.set_ylabel("True class")
    ax.set_xlabel("Predicted decision")
    ax.set_title(f"{region} confusion matrix [% per true class]")
    for text in disp.text_.ravel():
        text.set_text(f"{float(text.get_text()):.1f}%")
    save_plot(fig, outdir / f"confusion_matrix_{region}", formats)


def plot_eff_rej_vs_pt(
    table: pd.DataFrame,
    outdir: Path,
    region: str,
    formats: list[str],
    tag: str = "dnn",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(table["pt_center"], table["hs_eff"], marker="o", label="HS efficiency")
    ax.plot(table["pt_center"], table["pu_rejection"], marker="s", label="PU rejection")
    ax.set_xlabel("Jet pT [GeV]")
    ax.set_ylabel("Efficiency / rejection")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{region} working point vs pT")
    ax.grid(alpha=0.3)
    ax.legend()
    outbase = outdir / f"eff_and_rej_vs_pt_{region}_validation"
    if tag != "dnn":
        outbase = outdir / f"eff_and_rej_vs_pt_{region}_{tag}"
    save_plot(fig, outbase, formats)


def plot_stacked_before_after(
    values: np.ndarray,
    y: np.ndarray,
    pass_mask: np.ndarray,
    outdir: Path,
    region: str,
    name: str,
    xlabel: str,
    formats: list[str],
    bins: int = 50,
) -> None:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return
    lo, hi = np.nanpercentile(values[finite], [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return
    hist_bins = np.linspace(lo, hi, bins + 1)
    y = y.astype(bool)

    for suffix, mask, title in [
        ("before", np.ones(len(values), dtype=bool), "before DNN cut"),
        ("after", pass_mask.astype(bool), "after DNN cut"),
    ]:
        pu_vals = values[mask & ~y & finite]
        hs_vals = values[mask & y & finite]
        pu_counts, _ = np.histogram(pu_vals, bins=hist_bins)
        hs_counts, _ = np.histogram(hs_vals, bins=hist_bins)
        total_counts = pu_counts + hs_counts
        ratio = np.divide(
            pu_counts,
            total_counts,
            out=np.zeros_like(pu_counts, dtype=np.float64),
            where=total_counts > 0,
        )
        centers = 0.5 * (hist_bins[:-1] + hist_bins[1:])
        widths = np.diff(hist_bins)

        fig, (ax, rax) = plt.subplots(
            2,
            1,
            figsize=(6, 6.2),
            sharex=True,
            gridspec_kw={"height_ratios": [3.2, 1.1], "hspace": 0.05},
        )
        ax.hist(
            [pu_vals, hs_vals],
            bins=hist_bins,
            stacked=True,
            label=["PU", "HS"],
            color=["#c44e52", "#4c72b0"],
            alpha=0.75,
        )
        ax.errorbar(
            centers,
            total_counts,
            yerr=np.sqrt(total_counts),
            fmt="o",
            color="black",
            markersize=4,
            linewidth=1,
            label="Total jets",
        )
        ax.set_ylabel("Jets")
        ax.set_title(f"{region} {name} {title}")
        ax.grid(alpha=0.3)
        ax.legend()

        rax.errorbar(
            centers,
            ratio,
            xerr=0.5 * widths,
            fmt="o",
            color="black",
            markersize=3.5,
            linewidth=1,
        )
        rax.set_ylim(0.0, 1.0)
        rax.set_ylabel("Fake/Total")
        rax.set_xlabel(xlabel)
        rax.grid(alpha=0.3)
        save_plot(fig, outdir / f"stack_{name}_{region}_{suffix}", formats)


def permutation_importance(
    model: MLP,
    scaler: Scaler,
    df_test: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if len(df_test) == 0:
        return pd.DataFrame()
    sample = df_test
    if len(sample) > args.importance_max_rows:
        sample = sample.sample(args.importance_max_rows, random_state=args.seed)
    y = sample["y_hs"].to_numpy(dtype=np.float32)
    w = sample["weight"].to_numpy(dtype=np.float32) if "weight" in sample else None
    device = choose_device(args.device)
    x = scaler.transform(sample)
    base = predict_scores(model, x, device, args.batch_size)
    base_auc = safe_auc(y, base, w)
    rng = np.random.default_rng(args.seed)
    rows = []
    n_repeats = max(1, int(args.importance_repeats))
    for idx, feature in enumerate(scaler.features):
        auc_values = []
        auc_drop_values = []
        for _ in range(n_repeats):
            x_perm = x.copy()
            x_perm[:, idx] = rng.permutation(x_perm[:, idx])
            score = predict_scores(model, x_perm, device, args.batch_size)
            auc = safe_auc(y, score, w)
            auc_values.append(auc)
            if np.isfinite(base_auc) and np.isfinite(auc):
                auc_drop_values.append(float(base_auc - auc))
        auc_arr = np.asarray(auc_values, dtype=np.float64)
        auc_drop_arr = np.asarray(auc_drop_values, dtype=np.float64)
        auc_mean = float(np.nanmean(auc_arr)) if auc_arr.size else float("nan")
        auc_std = float(np.nanstd(auc_arr, ddof=0)) if auc_arr.size else float("nan")
        auc_drop_mean = float(np.nanmean(auc_drop_arr)) if auc_drop_arr.size else float("nan")
        auc_drop_std = float(np.nanstd(auc_drop_arr, ddof=0)) if auc_drop_arr.size else float("nan")
        auc_drop_stderr = (
            float(auc_drop_std / np.sqrt(auc_drop_arr.size)) if auc_drop_arr.size else float("nan")
        )
        rows.append(
            {
                "feature": feature,
                "base_auc": float(base_auc),
                "auc": auc_mean,
                "auc_std": auc_std,
                "auc_drop": auc_drop_mean,
                "auc_drop_std": auc_drop_std,
                "auc_drop_stderr": auc_drop_stderr,
                "n_repeats": int(auc_arr.size),
            }
        )
    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False)


def plot_feature_importance(
    table: pd.DataFrame,
    outdir: Path,
    region: str,
    formats: list[str],
    top_n: int = 20,
) -> None:
    if table.empty:
        return
    top = table.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.28 * len(top) + 1.5)))
    xerr = top["auc_drop_stderr"] if "auc_drop_stderr" in top else None
    ax.barh(top["feature"], top["auc_drop"], xerr=xerr, color="#55a868", ecolor="#2f4f4f", capsize=3)
    ax.set_xlabel("Permutation AUC drop")
    ax.set_title(f"{region} DNN feature importance")
    ax.grid(axis="x", alpha=0.3)
    save_plot(fig, outdir / f"feature_importance_{region}", formats)


def plot_top_feature_shapes(
    df: pd.DataFrame,
    features: list[str],
    importance: pd.DataFrame,
    outdir: Path,
    region: str,
    formats: list[str],
    n_features: int = 6,
) -> None:
    if importance.empty:
        plot_features = features[:n_features]
    else:
        plot_features = [f for f in importance["feature"].head(n_features) if f in df.columns]
    y = df["y_hs"].to_numpy(dtype=bool)
    for feature in plot_features:
        values = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=np.float32)
        finite = np.isfinite(values)
        if finite.sum() == 0:
            continue
        lo, hi = np.nanpercentile(values[finite], [0.5, 99.5])
        if lo == hi:
            continue
        bins = np.linspace(lo, hi, 51)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(values[y & finite], bins=bins, histtype="step", density=True, linewidth=2, label="HS")
        ax.hist(values[~y & finite], bins=bins, histtype="step", density=True, linewidth=2, label="PU")
        ax.set_xlabel(feature)
        ax.set_ylabel("Normalized jets")
        ax.set_title(f"{region} {feature}")
        ax.grid(alpha=0.3)
        ax.legend()
        save_plot(fig, outdir / f"feature_shape_{region}_{feature}", formats)


def baseline_puid_plots(
    pred: pd.DataFrame,
    outdir: Path,
    region: str,
    args: argparse.Namespace,
) -> dict | None:
    if "puIdDisc" not in pred.columns:
        return None
    y = pred["y_hs"].to_numpy(dtype=np.float32)
    score = pred["puIdDisc"].to_numpy(dtype=np.float32)
    finite = np.isfinite(score)
    if finite.sum() == 0 or len(np.unique(y[finite])) < 2:
        return None
    weights = pred["weight"].to_numpy(dtype=np.float32)[finite]
    score = score[finite]
    y = y[finite]
    threshold, direction, hs_eff, pu_rej = threshold_and_direction(
        score, y, args.hs_eff, weights
    )
    pass_mask = pass_from_threshold(score, threshold, direction)
    tmp = pred.loc[finite].copy()
    table = wp_vs_pt(tmp, pass_mask, args.pt_bins, weights)
    table.to_csv(outdir / f"puid_wp_vs_pt_{region}.csv", index=False)
    if not args.no_plots:
        plot_score_real_fake(score, y, outdir, region, args.plot_format, tag="puIdDisc")
        plot_eff_rej_vs_pt(table, outdir, region, args.plot_format, tag="puIdDisc")
    return {
        "baseline": "puIdDisc",
        "threshold": float(threshold),
        "direction": direction,
        "hs_efficiency": float(hs_eff),
        "pu_rejection": float(pu_rej),
        "auc": safe_auc(y, score, weights),
    }


def make_validation_outputs(
    model: MLP,
    scaler: Scaler,
    df_region: pd.DataFrame,
    pred_test: pd.DataFrame,
    metrics: dict,
    outdir: Path,
    region: str,
    args: argparse.Namespace,
) -> dict:
    y = pred_test["y_hs"].to_numpy(dtype=np.float32)
    score = pred_test["score"].to_numpy(dtype=np.float32)
    weight = pred_test["weight"].to_numpy(dtype=np.float32)

    threshold, direction, hs_eff, pu_rej = threshold_and_direction(score, y, args.hs_eff, weight)
    pass_mask = pass_from_threshold(score, threshold, direction)
    wp_table = wp_vs_pt(pred_test, pass_mask, args.pt_bins, weight)
    wp_table.to_csv(outdir / f"wp_vs_pt_{region}.csv", index=False)

    hs_grid = args.hs_scan if args.hs_scan is not None else np.linspace(0.60, 0.98, 20)
    scan_rows = []
    for target in hs_grid:
        thr, direct, eff, rej = threshold_and_direction(score, y, float(target), weight)
        scan_rows.append(
            {
                "hs_eff_target": float(target),
                "threshold": float(thr),
                "direction": direct,
                "hs_efficiency": float(eff),
                "pu_rejection": float(rej),
            }
        )
    (outdir / f"rescan_{region}.json").write_text(json.dumps(scan_rows, indent=2))

    if not args.no_plots:
        plot_training_history(metrics["history"], outdir, region, args.plot_format)
        plot_roc_pr(y, score, weight, outdir, region, args.plot_format)
        plot_score_real_fake(score, y, outdir, region, args.plot_format)
        plot_confusion(y, pass_mask, outdir, region, args.plot_format)
        plot_eff_rej_vs_pt(wp_table, outdir, region, args.plot_format)
        plot_stacked_before_after(
            score,
            y,
            pass_mask,
            outdir,
            region,
            "score",
            "DNN HS score",
            args.plot_format,
        )
        for name, xlabel in [
            ("pt", "Jet pT [GeV]"),
            ("eta", "Jet eta"),
            ("aeta", "|Jet eta|"),
        ]:
            plot_stacked_before_after(
                pred_test[name].to_numpy(dtype=np.float32),
                y,
                pass_mask,
                outdir,
                region,
                name,
                xlabel,
                args.plot_format,
            )

    importance = permutation_importance(model, scaler, pred_test, args)
    importance.to_csv(outdir / f"feature_importance_{region}.csv", index=False)
    if not args.no_plots:
        plot_feature_importance(importance, outdir, region, args.plot_format)
        plot_top_feature_shapes(df_region, scaler.features, importance, outdir, region, args.plot_format)

    baseline = baseline_puid_plots(pred_test, outdir, region, args)

    summary = {
        "region": region,
        "model_type": "dnn",
        "model_path": str(outdir / "model_torchscript.pt"),
        "checkpoint_path": str(outdir / "checkpoint.pt"),
        "features": scaler.features,
        "threshold": float(threshold),
        "direction": direction,
        "hs_efficiency": float(hs_eff),
        "pu_rejection": float(pu_rej),
        "n_region": int(len(df_region)),
        "n_test": int(len(pred_test)),
        "pt_min": float(args.pt_min),
        "pt_turnoff": float(args.pt_max),
        "valid_pt_range": {"min": float(args.pt_min), "max": float(args.pt_max)},
        "metrics": {k: v for k, v in metrics.items() if k != "history"},
        "baseline_puIdDisc": baseline,
    }
    (outdir / f"summary_{region}.json").write_text(json.dumps(summary, indent=2))
    return summary


def replot_region_from_saved_outputs(output: Path, region: str, args: argparse.Namespace) -> bool:
    outdir = output / region
    pred_path = outdir / "predictions_test.parquet"
    metrics_path = outdir / "metrics.json"
    summary_path = outdir / f"summary_{region}.json"
    if not pred_path.exists():
        print(f"[{region}] Missing {pred_path}; cannot replot.")
        return False
    if not metrics_path.exists():
        print(f"[{region}] Missing {metrics_path}; cannot replot.")
        return False

    pred_test = pd.read_parquet(pred_path)
    metrics = json.loads(metrics_path.read_text())
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    y = pred_test["y_hs"].to_numpy(dtype=np.float32)
    score = pred_test["score"].to_numpy(dtype=np.float32)
    weight = (
        pred_test["weight"].to_numpy(dtype=np.float32)
        if "weight" in pred_test.columns
        else np.ones(len(pred_test), dtype=np.float32)
    )
    threshold = float(summary.get("threshold", float("nan")))
    direction = summary.get("direction", "keep_high")
    hs_eff = summary.get("hs_efficiency")
    pu_rej = summary.get("pu_rejection")
    if not np.isfinite(threshold):
        threshold, direction, hs_eff, pu_rej = threshold_and_direction(score, y, args.hs_eff, weight)
    pass_mask = pass_from_threshold(score, threshold, direction)
    wp_table = wp_vs_pt(pred_test, pass_mask, args.pt_bins, weight)
    wp_table.to_csv(outdir / f"wp_vs_pt_{region}.csv", index=False)

    plot_training_history(metrics.get("history", []), outdir, region, args.plot_format)
    plot_roc_pr(y, score, weight, outdir, region, args.plot_format)
    plot_score_real_fake(score, y, outdir, region, args.plot_format)
    plot_confusion(y, pass_mask, outdir, region, args.plot_format)
    plot_eff_rej_vs_pt(wp_table, outdir, region, args.plot_format)
    plot_stacked_before_after(
        score,
        y,
        pass_mask,
        outdir,
        region,
        "score",
        "DNN HS score",
        args.plot_format,
    )
    for name, xlabel in [
        ("pt", "Jet pT [GeV]"),
        ("eta", "Jet eta"),
        ("aeta", "|Jet eta|"),
    ]:
        if name in pred_test.columns:
            plot_stacked_before_after(
                pred_test[name].to_numpy(dtype=np.float32),
                y,
                pass_mask,
                outdir,
                region,
                name,
                xlabel,
                args.plot_format,
            )
    if "puIdDisc" in pred_test.columns:
        baseline_puid_plots(pred_test, outdir, region, args)
    return True


def train_region_with_features(
    df_region: pd.DataFrame,
    region: str,
    outdir: Path,
    args: argparse.Namespace,
    features: list[str],
) -> dict:
    print(
        f"[{region}] training N={len(df_region)} HS={int((df_region['y_hs'] == 1).sum())} "
        f"PU={int((df_region['y_hs'] == 0).sum())} features={len(features)} out={outdir}"
    )
    model, scaler, metrics, pred_test = train_one_model(df_region, features, outdir, args)
    return make_validation_outputs(model, scaler, df_region, pred_test, metrics, outdir, region, args)


def train_region(jets: pd.DataFrame, region: str, output: Path, args: argparse.Namespace) -> dict | None:
    mask = region_mask_eta(jets["eta"].to_numpy(dtype=np.float32), region)
    df_region = jets[mask].copy()
    if len(df_region) == 0:
        print(f"[{region}] No jets found; skipping.")
        return None
    class_counts = df_region["y_hs"].value_counts().to_dict()
    if len(class_counts) < 2:
        print(f"[{region}] Only one class present ({class_counts}); skipping.")
        return None

    features = selected_features(df_region, region)
    if not features:
        print(f"[{region}] No usable non-constant features; skipping.")
        return None

    print(f"Input features: {features}")

    outdir = output / region
    outdir.mkdir(parents=True, exist_ok=True)
    summary = train_region_with_features(df_region, region, outdir, args, features)

    if args.run_ablations:
        ablation_dir = outdir / "ablations"
        ablation_dir.mkdir(parents=True, exist_ok=True)
        rows = [summary_to_ablation_row(summary, "baseline", "full selected feature set")]
        for tag, feature_set, notes in build_ablation_variants(features, args):
            variant_outdir = ablation_dir / tag
            variant_outdir.mkdir(parents=True, exist_ok=True)
            variant_args = copy.deepcopy(args)
            variant_args.no_plots = True
            print(f"[{region}] ablation {tag}: {notes}")
            variant_summary = train_region_with_features(
                df_region, region, variant_outdir, variant_args, feature_set
            )
            rows.append(summary_to_ablation_row(variant_summary, tag, notes))
        ablation_table = pd.DataFrame(rows).sort_values(
            ["test_auc", "pu_rejection"], ascending=[False, False]
        )
        ablation_table.to_csv(outdir / f"ablation_summary_{region}.csv", index=False)
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.replot_only:
        refreshed = 0
        for region in args.regions:
            print(f"[{region}] regenerating saved validation plots")
            refreshed += int(replot_region_from_saved_outputs(output, region, args))
        print(f"Done. Replotted {refreshed} region(s) in {output}")
        return

    if not args.input:
        raise ValueError("--input is required unless --replot-only is used.")

    paths = expand_inputs(args.input, args.use_glob)
    (output / "inputs.json").write_text(
        json.dumps(
            {
                "input": args.input,
                "use_glob": args.use_glob,
                "n_files": len(paths),
                "files": paths,
            },
            indent=2,
        )
    )
    print(f"Loading {len(paths)} parquet file(s)")
    df = load_stage1(paths, args)
    print(f"Loaded event table: {len(df)} rows, {len(df.columns)} columns")
    jets = flatten_jets(df, args)
    print(
        f"Flattened jet table: {len(jets)} jets in {args.pt_min} <= pt < {args.pt_max}; "
        f"HS={int(jets['y_hs'].sum())}, PU={int((jets['y_hs'] == 0).sum())}"
    )
    (output / "sample_composition.json").write_text(
        json.dumps(sample_group_summary(jets), indent=2)
    )

    summaries = []
    for region in args.regions:
        summary = train_region(jets, region, output, args)
        if summary is not None:
            summaries.append(summary)
            (output / f"summary_{region}.json").write_text(json.dumps(summary, indent=2))

    (output / "summary_all.json").write_text(json.dumps(summaries, indent=2))
    if args.run_ablations:
        ablation_frames = []
        for region in args.regions:
            path = output / region / f"ablation_summary_{region}.csv"
            if path.exists():
                ablation_frames.append(pd.read_csv(path))
        if ablation_frames:
            pd.concat(ablation_frames, axis=0, ignore_index=True).to_csv(
                output / "ablation_summary_all.csv", index=False
            )
    print(f"Done. Wrote {len(summaries)} region summaries to {output}")


if __name__ == "__main__":
    main()
