"""
Stage-1 parquet -> transformer inputs (training and scoring), exactly as b-hive's build_model_inputs at d57c342.
Nominal kinematics only (`<name>_nominal`, else `<name>`), so the score does not move with JES/JER.
"""

import glob
import os

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from MVA_training.transformer_binary import config
from MVA_training.transformer_binary.folds import event_folds, year_index
from MVA_training.transformer_binary.model import GLOBAL_FEATURES, OBJECT_FEATURES, OBJECT_IDS, OBJECTS

MUON_MASS = 0.1056583755  # GeV
LOG_FLOOR = 1.0e-12

MODEL_INPUTS = (
    *(f"{obj}_{var}" for obj in ("mu1", "mu2", "dimuon") for var in ("pt", "eta", "phi")),
    *(f"{jet}_{var}" for jet in ("jet1", "jet2") for var in ("pt", "eta", "phi", "mass")),
    "htsoft2", "htsoft5", "MET_pt", "MET_phi", "njets",
)
EVENT_ID = ("run", "luminosityBlock", "event")
SELECTION_INPUTS = ("dimuon_mass", "nBtagLoose", "nBtagMedium", "wgt_nominal")
COLUMN_ALIASES = {"MET_pt": ("PuppiMET_pt",), "MET_phi": ("PuppiMET_phi",)}


def resolve_columns(available, names):
    """Map each canonical name to the stage1 column holding its nominal value."""
    available = set(available)
    resolved = {}
    for name in names:
        candidates = [f"{base}{suffix}" for base in (name, *COLUMN_ALIASES.get(name, ())) for suffix in ("_nominal", "")]
        match = next((c for c in candidates if c in available), None)
        if match is None:
            raise KeyError(f"stage1 has none of {candidates}")
        resolved[name] = match
    return resolved


def sample_files(stage1_root, year, sample, subdir="compacted"):
    files = sorted(glob.glob(os.path.join(stage1_root, year, subdir, sample, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {os.path.join(stage1_root, year, subdir, sample)}")
    return files


def read_columns(files, names):
    """Read canonical `names` from parquet files as a dict of numpy arrays (nulls -> 0)."""
    resolved = resolve_columns(pq.read_schema(files[0]).names, names)
    table = pa.concat_tables(pq.read_table(f, columns=list(resolved.values())) for f in files)
    out = {}
    for name, column in resolved.items():
        values = pc.fill_null(table[column].combine_chunks(), 0)
        out[name] = values.to_numpy(zero_copy_only=False)
    return out


def _log(values):
    return np.log(np.maximum(values, LOG_FLOOR)).astype(np.float32)


def build_inputs(columns, year):
    """
    columns: dict with MODEL_INPUTS (numpy, any numeric dtype)
    returns  global_features (N, 7) float32, year_idx (N,) int64, objects (N, 5, 10) float32
    """
    f32 = {name: np.asarray(columns[name], dtype=np.float32) for name in MODEL_INPUTS}
    n_events = len(f32["mu1_pt"])

    tokens = []
    for obj in OBJECTS:
        pt, eta, phi = f32[f"{obj}_pt"], f32[f"{obj}_eta"], f32[f"{obj}_phi"]
        if obj in ("mu1", "mu2"):
            mass = np.full(n_events, MUON_MASS, dtype=np.float32)
        elif obj == "dimuon":
            mass = np.zeros(n_events, dtype=np.float32)
        else:
            mass = f32[f"{obj}_mass"]
        energy = np.sqrt((pt * np.cosh(eta)) ** 2 + mass**2).astype(np.float32)
        obj_id = np.where(pt == 0.0, 0, OBJECT_IDS[obj]).astype(np.float32)
        tokens.append(np.stack([_log(pt), _log(energy), eta, np.sin(phi), np.cos(phi), obj_id, pt, eta, phi, energy], axis=1))
    objects = np.stack(tokens, axis=1).astype(np.float32)

    met_pt, met_phi = f32["MET_pt"], f32["MET_phi"]
    global_features = np.stack(
        [
            _log(f32["htsoft2"]),
            _log(f32["htsoft5"]),
            _log(met_pt),
            _log(np.abs(met_pt)),  # MET token energy; MET is massless with eta = 0
            np.sin(met_phi),
            np.cos(met_phi),
            f32["njets"],
        ],
        axis=1,
    ).astype(np.float32)

    assert objects.shape[1:] == (len(OBJECTS), len(OBJECT_FEATURES))
    assert global_features.shape[1] == len(GLOBAL_FEATURES)
    if not (np.isfinite(objects).all() and np.isfinite(global_features).all()):
        raise ValueError("non-finite transformer inputs")
    return global_features, np.full(n_events, year_index(year), dtype=np.int64), objects


def training_selection(columns):
    """h-peak [115, 135) GeV and the VH/ttH b-tag veto; no njets requirement (b-hive convention)."""
    in_peak = (columns["dimuon_mass"] >= 115.0) & (columns["dimuon_mass"] < 135.0)
    btag = (columns["nBtagLoose"] >= 2) | ((columns["nBtagMedium"] >= 1) & (columns["njets"] >= 2))
    return in_peak & ~btag


def load_signal(stage1_root, years, subdir="compacted"):
    """Selected ggH and VBF signal events of `years`, with inputs, label, weight, year and fold."""
    parts = []
    for year in years:
        for class_name, samples in config.signal_samples(year).items():  # the score is the ggH probability
            label = 1.0 if class_name == "ggH" else 0.0
            for sample in samples:
                columns = read_columns(sample_files(stage1_root, year, sample, subdir), MODEL_INPUTS + EVENT_ID + SELECTION_INPUTS)
                keep = training_selection(columns)
                columns = {name: values[keep] for name, values in columns.items()}
                global_features, year_idx, objects = build_inputs(columns, year)
                parts.append(
                    dict(
                        global_features=global_features,
                        year_idx=year_idx,
                        objects=objects,
                        is_ggH=np.full(len(year_idx), label, dtype=np.float32),
                        abs_weight=np.abs(columns["wgt_nominal"]).astype(np.float64),
                        fold=event_folds(year_idx, sample, columns["run"], columns["luminosityBlock"], columns["event"]),
                        sample=np.full(len(year_idx), sample),
                        **{name: columns[name] for name in EVENT_ID},
                    )
                )
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
