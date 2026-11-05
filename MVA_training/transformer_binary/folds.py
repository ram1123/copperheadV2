"""
Year encoding and event folds of the binary transformer, bit-for-bit as b-hive (modules/year_vocab.py at d57c342).
Fold hash: (year_idx, sample, run, luminosityBlock, event), so folds do not depend on file location.
"""

import hashlib

import numpy as np

YEAR_VOCABULARIES = {
    "run2": ("2016postVFP", "2016preVFP", "2017", "2018"),
    "run3": ("2022preEE", "2022postEE", "2023", "2023BPix", "2024", "2025", "2026"),
}
N_FOLDS = 4

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_FIELD_SALTS = tuple(
    np.uint64(v) for v in (0xD1B54A32D192ED03, 0xABC98388FB8FAC03, 0x8CB92BA72F3D8DD7, 0xDB4F0B9175AE2165)
)


def run_period(year):
    """'run2' or 'run3' for a stage1 year directory name."""
    for period, vocabulary in YEAR_VOCABULARIES.items():
        if year in vocabulary:
            return period
    raise ValueError(f"Unknown year {year!r}; known: {YEAR_VOCABULARIES}")


def year_index(year):
    """Position of `year` in its run period's vocabulary (the model's year-embedding row)."""
    return YEAR_VOCABULARIES[run_period(year)].index(year)


def _splitmix64(values):
    values = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = (values + np.uint64(0x9E3779B97F4A7C15)) & _MASK64
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _MASK64
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _MASK64
    return z ^ (z >> np.uint64(31))


def _sample_hash(sample, namespace):
    digest = hashlib.blake2b(f"{namespace}\x1f{sample}".encode(), digest_size=8, person=b"bhive-fold-v1").digest()
    return np.uint64(int.from_bytes(digest, byteorder="little", signed=False))


def _event_hash(year_idx, sample, run, lumi, event, namespace):
    mixed = _splitmix64(_sample_hash(sample, namespace))
    for values, salt in zip((year_idx, run, lumi, event), _FIELD_SALTS):
        values = np.asarray(values).astype(np.uint64)
        mixed = _splitmix64(mixed ^ _splitmix64(values ^ salt))
    return mixed


def event_folds(year_idx, sample, run, lumi, event):
    """Outer fold in [0, 4) per event; model `k` never trains on fold `k`. `sample` is one name."""
    return np.asarray(_event_hash(year_idx, sample, run, lumi, event, "outer-fold") % np.uint64(N_FOLDS), dtype=np.int8)


def inner_validation_mask(year_idx, sample, run, lumi, event, fraction=0.10):
    """Independent ~10% hash partition used to pick the epoch count before the refit."""
    mixed = _event_hash(year_idx, sample, run, lumi, event, "inner-validation")
    return np.asarray(mixed % np.uint64(1_000_000) < int(round(fraction * 1_000_000)), dtype=bool)
