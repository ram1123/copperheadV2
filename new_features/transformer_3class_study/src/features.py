from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

MUON_MASS_GEV = 0.1056583755
EPS = 1.0e-6

CLASS_TO_INDEX = {'ggH': 0, 'VBF': 1, 'bkg': 2}
INDEX_TO_CLASS = {idx: name for name, idx in CLASS_TO_INDEX.items()}
OUTPUT_COLUMNS = ['p_ggH', 'p_VBF', 'p_bkg']

OBJECT_ORDER = ['mu1', 'mu2', 'jet1', 'jet2', 'dimuon']
OBJECT_IDS = {'pad': 0, 'mu1': 1, 'mu2': 2, 'jet1': 3, 'jet2': 4, 'dimuon': 5, 'global': 6}
OBJECT_FEATURES = [
    'physObj_ln_pt',
    'physObj_ln_e',
    'physObj_eta',
    'physObj_sin_phi',
    'physObj_cos_phi',
    'physObj_id',
    'physObj_pt_4v',
    'physObj_eta_4v',
    'physObj_phi_4v',
    'physObj_energy_4v',
]
GLOBAL_FEATURES = [
    'global_ln_htsoft2',
    'global_ln_htsoft5',
    'global_MET_ln_pt',
    'global_MET_ln_e',
    'global_MET_sin_phi',
    'global_MET_cos_phi',
    'global_n_jets',
]

# Explicit record of the transform applied to each raw branch before standardization.
# Kept next to the feature order so a checkpoint can be interpreted without reading code.
GLOBAL_FEATURE_TRANSFORMS = {
    'global_ln_htsoft2': 'log1p(max(htsoft2, 0))',
    'global_ln_htsoft5': 'log1p(max(htsoft5, 0))',
    'global_MET_ln_pt': 'log1p(max(MET_pt, 0))',
    'global_MET_ln_e': 'log1p(max(PuppiMET_sumEt, 0))',
    'global_MET_sin_phi': 'sin(MET_phi)',
    'global_MET_cos_phi': 'cos(MET_phi)',
    'global_n_jets': 'identity',
}
OBJECT_FEATURE_TRANSFORMS = {
    'physObj_ln_pt': 'ln(max(pt, 1e-6)); padded rows zeroed',
    'physObj_ln_e': 'ln(max(E, 1e-6)); padded rows zeroed',
    'physObj_eta': 'identity',
    'physObj_sin_phi': 'sin(phi)',
    'physObj_cos_phi': 'cos(phi)',
    'physObj_id': 'integer token id, excluded from standardization and from the continuous input block',
    'physObj_pt_4v': 'raw pt, pairwise-attention input only',
    'physObj_eta_4v': 'raw eta, pairwise-attention input only',
    'physObj_phi_4v': 'raw phi, pairwise-attention input only',
    'physObj_energy_4v': 'raw energy, pairwise-attention input only',
}

COLUMN_ALIASES: dict[str, list[str]] = {
    'event': ['event'],
    'run': ['run'],
    'luminosityBlock': ['luminosityBlock'],
    'year': ['year'],
    'wgt_nominal': ['wgt_nominal'],
    'separate_wgt_zpt': ['separate_wgt_zpt'],
    'mu1_pt': ['mu1_pt'],
    'mu1_eta': ['mu1_eta'],
    'mu1_phi': ['mu1_phi'],
    'mu2_pt': ['mu2_pt'],
    'mu2_eta': ['mu2_eta'],
    'mu2_phi': ['mu2_phi'],
    'jet1_pt': ['jet1_pt_nominal', 'jet1_pt'],
    'jet1_eta': ['jet1_eta_nominal', 'jet1_eta'],
    'jet1_phi': ['jet1_phi_nominal', 'jet1_phi'],
    'jet1_mass': ['jet1_mass_nominal', 'jet1_mass'],
    'jet2_pt': ['jet2_pt_nominal', 'jet2_pt'],
    'jet2_eta': ['jet2_eta_nominal', 'jet2_eta'],
    'jet2_phi': ['jet2_phi_nominal', 'jet2_phi'],
    'jet2_mass': ['jet2_mass_nominal', 'jet2_mass'],
    'dimuon_pt': ['dimuon_pt'],
    'dimuon_eta': ['dimuon_eta', 'dimuon_rapidity'],
    'dimuon_phi': ['dimuon_phi'],
    'htsoft2': ['htsoft2_nominal', 'htsoft2'],
    'htsoft5': ['htsoft5_nominal', 'htsoft5'],
    'met_pt': ['PuppiMET_pt', 'MET_pt_nominal', 'MET_pt'],
    'met_phi': ['PuppiMET_phi', 'MET_phi_nominal', 'MET_phi'],
    'met_e': ['PuppiMET_sumEt', 'MET_sumEt', 'MET_e', 'MET_E'],
    'n_jets': ['njets_nominal', 'n_jets', 'n_jets_nominal'],
}

OPTIONAL_CANONICAL = {'separate_wgt_zpt', 'run', 'luminosityBlock', 'year'}


@dataclass(frozen=True)
class ColumnResolution:
    mapping: dict[str, str]
    missing_required: list[str]
    missing_optional: list[str]


def resolve_columns(available_columns: list[str] | set[str]) -> ColumnResolution:
    available = set(available_columns)
    mapping: dict[str, str] = {}
    missing_required: list[str] = []
    missing_optional: list[str] = []
    for canonical, candidates in COLUMN_ALIASES.items():
        found = next((candidate for candidate in candidates if candidate in available), None)
        if found is None:
            if canonical in OPTIONAL_CANONICAL:
                missing_optional.append(canonical)
            else:
                missing_required.append(canonical)
        else:
            mapping[canonical] = found
    return ColumnResolution(mapping=mapping, missing_required=missing_required, missing_optional=missing_optional)


def columns_to_read(resolution: ColumnResolution) -> list[str]:
    return sorted(set(resolution.mapping.values()))


def canonicalize_frame(df: pd.DataFrame, resolution: ColumnResolution) -> pd.DataFrame:
    rename = {actual: canonical for canonical, actual in resolution.mapping.items()}
    out = df.rename(columns=rename).copy()
    for optional in OPTIONAL_CANONICAL:
        if optional not in out:
            out[optional] = np.nan
    return out


def finite_array(values: pd.Series | np.ndarray, default: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return np.where(np.isfinite(arr), arr, default).astype(np.float32)


def phi_features(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phi = finite_array(phi)
    return np.sin(phi).astype(np.float32), np.cos(phi).astype(np.float32)


def energy_from_pt_eta_mass(pt: np.ndarray, eta: np.ndarray, mass: np.ndarray | float) -> np.ndarray:
    pt = np.clip(finite_array(pt), 0.0, None)
    eta = np.clip(finite_array(eta), -10.0, 10.0)
    mass_arr = np.asarray(mass, dtype=np.float32) + np.zeros_like(pt, dtype=np.float32)
    momentum = pt * np.cosh(eta)
    energy2 = momentum * momentum + np.clip(mass_arr, 0.0, None) ** 2
    return np.sqrt(np.clip(energy2, 0.0, None)).astype(np.float32)


def object_token(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, mass: np.ndarray | float, object_id: int) -> np.ndarray:
    pt = np.clip(finite_array(pt), 0.0, None)
    eta = finite_array(eta)
    phi = finite_array(phi)
    energy = energy_from_pt_eta_mass(pt, eta, mass)
    sin_phi, cos_phi = phi_features(phi)
    token = np.stack([
        np.log(np.clip(pt, EPS, None)),
        np.log(np.clip(energy, EPS, None)),
        eta,
        sin_phi,
        cos_phi,
        np.full_like(pt, object_id, dtype=np.float32),
        pt,
        eta,
        phi,
        energy,
    ], axis=1).astype(np.float32)
    pad = (~np.isfinite(pt)) | (pt <= 0.0)
    token[pad] = 0.0
    return token


def build_object_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    tokens = []
    tokens.append(object_token(df['mu1_pt'], df['mu1_eta'], df['mu1_phi'], MUON_MASS_GEV, OBJECT_IDS['mu1']))
    tokens.append(object_token(df['mu2_pt'], df['mu2_eta'], df['mu2_phi'], MUON_MASS_GEV, OBJECT_IDS['mu2']))
    tokens.append(object_token(df['jet1_pt'], df['jet1_eta'], df['jet1_phi'], finite_array(df['jet1_mass']), OBJECT_IDS['jet1']))
    tokens.append(object_token(df['jet2_pt'], df['jet2_eta'], df['jet2_phi'], finite_array(df['jet2_mass']), OBJECT_IDS['jet2']))
    # The source preprocessing hides the dimuon mass from the transformer token by using mass=0.
    tokens.append(object_token(df['dimuon_pt'], df['dimuon_eta'], df['dimuon_phi'], 0.0, OBJECT_IDS['dimuon']))
    out = np.stack(tokens, axis=1).astype(np.float32)
    padding_mask = out[:, :, 6] <= 0.0
    return out, padding_mask


def log1p_magnitude(values: np.ndarray) -> np.ndarray:
    """log1p transform for non-negative magnitudes with an attainable physical zero.

    A plain ln(clip(x, EPS)) maps a genuine zero to ln(1e-6) = -13.8, which is ~15 units
    away from the bulk of the distribution and then dominates the standardization mean
    and std. htsoft5 is zero in 17-29% of 2017 events and htsoft2 in 3-7%, so that spike
    is not a rare edge case. log1p is monotonic, maps 0 -> 0 continuously, and agrees
    with ln to better than a percent above ~100 GeV.
    """
    return np.log1p(np.clip(values, 0.0, None)).astype(np.float32)


def build_global_features(df: pd.DataFrame) -> np.ndarray:
    met_pt = np.clip(finite_array(df['met_pt']), 0.0, None)
    met_e = np.clip(finite_array(df['met_e']), 0.0, None)
    met_phi = finite_array(df['met_phi'])
    sin_phi, cos_phi = phi_features(met_phi)
    return np.stack([
        log1p_magnitude(finite_array(df['htsoft2'])),
        log1p_magnitude(finite_array(df['htsoft5'])),
        log1p_magnitude(met_pt),
        log1p_magnitude(met_e),
        sin_phi,
        cos_phi,
        finite_array(df['n_jets']),
    ], axis=1).astype(np.float32)


def build_event_weights(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    nominal = finite_array(df['wgt_nominal'], default=1.0)
    if 'separate_wgt_zpt' in df:
        zpt = finite_array(df['separate_wgt_zpt'], default=np.nan)
        use_zpt = np.isfinite(zpt) & (np.abs(zpt) > EPS)
        event_weight = np.where(use_zpt, nominal / zpt, nominal)
    else:
        event_weight = nominal
    train_weight = np.abs(event_weight).astype(np.float32)
    train_weight = np.where(np.isfinite(train_weight) & (train_weight > 0.0), train_weight, 1.0).astype(np.float32)
    return event_weight.astype(np.float32), train_weight


def describe_feature_contract() -> dict[str, object]:
    return {
        'classes': CLASS_TO_INDEX,
        'source_class_order': ['bkg', 'ggH', 'VBF'],
        'study_class_order': ['ggH', 'VBF', 'bkg'],
        'object_order': OBJECT_ORDER,
        'object_features': OBJECT_FEATURES,
        'global_features': GLOBAL_FEATURES,
        'column_aliases': COLUMN_ALIASES,
        'global_feature_transforms': GLOBAL_FEATURE_TRANSFORMS,
        'object_feature_transforms': OBJECT_FEATURE_TRANSFORMS,
        'notes': [
            'Pairwise attention uses raw pt, eta, phi, energy from object feature slots 6:10.',
            'Continuous object slots 0:5 and all global features are normalized from the training split.',
            'PuppiMET_sumEt is used as the available Copperhead proxy for the source global_MET_ln_e branch,'
            ' and global_MET_ln_e is ln(1 + PuppiMET_sumEt).',
            'Non-negative global magnitudes use log1p, not ln(clip(x, 1e-6)): htsoft2/htsoft5 have an'
            ' attainable physical zero, and the ln floor put 17-29% of events at -13.8, which then'
            ' dominated the standardization statistics.',
            'Object ln_pt and ln_e keep a plain ln because padded object rows are zeroed and every'
            ' surviving object has pt > 0 by construction.',
        ],
    }
