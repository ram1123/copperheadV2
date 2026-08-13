"""Shared shape-systematics discovery and per-feature variation resolution.

This module is the single source of truth for "what is a shape variation" and
"which column does feature F take under variation V".  It was factored out of
``run_stage2_vbf.py`` so that Stage-2 inference and the DNN
preprocessing/training code agree on the answer by construction rather than by
two hand-maintained copies.

``run_stage2_vbf.py`` re-exports :func:`discover_shape_systs` and
:func:`feature_name_for_variation` from here, so its public names are unchanged.

Nothing in this module reads files or touches Dask; callers pass in the field
names they already have.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from modules.utils import logger


# --------------------------------------------------------------------------
# Discovery (moved verbatim from run_stage2_vbf.py)
# --------------------------------------------------------------------------
DEFAULT_SHAPE_SYST_PREFIXES: List[str] = [
    "dimuon_mass_",
    "dimuon_pt_",
    "dimuon_pt_log_",
    "dimuon_eta_",
    "dimuon_rapidity_",
    "dimuon_ebe_mass_res_",
    "dimuon_ebe_mass_res_rel_",
    "mu1_pt_",
    "mu2_pt_",
    "mu1_pt_over_mass_",
    "mu2_pt_over_mass_",
    "jet1_pt_",
    "jet2_pt_",
    "jj_mass_",
    "jj_dEta_",
    "njets_",
    "nBtagLoose_",
    "nBtagMedium_",
]


def discover_shape_systs(fields, prefixes=None):
    """
    Discover available shape-variation suffixes from shifted stage1 branches.
    This covers both jet/JEC-like variations and muon-momentum shape variations.
    """
    if prefixes is None:
        prefixes = list(DEFAULT_SHAPE_SYST_PREFIXES)
    suffixes = set()
    for f in fields:
        if not (f.endswith("_up") or f.endswith("_down")):
            continue
        # Use the longest (most specific) matching prefix, not the first one in
        # list order, so e.g. "mu1_pt_over_mass_" wins over "mu1_pt_" for a field
        # like "mu1_pt_over_mass_mu_roccor_up" regardless of how `prefixes` is
        # ordered.
        matching_prefixes = [p for p in prefixes if f.startswith(p)]
        if not matching_prefixes:
            continue
        best_prefix = max(matching_prefixes, key=len)
        suffixes.add(f[len(best_prefix):])
    return sorted(suffixes)


def feature_name_for_variation(
    feat,
    variation,
    fields,
    allow_nominal_fallback=False,
    nominal_only_features=None,
):
    """Resolve DNN inputs; default caller passes nominal to match sideHustle5."""
    # training_features often already carry a literal "_nominal" suffix
    # (e.g. "jet1_phi_nominal") rather than a bare base name ("jet1_phi").
    # Strip it so the shifted candidate below is built from the base name,
    # not double-suffixed into something like "jet1_phi_nominal_nominal".
    base = feat[: -len("_nominal")] if feat.endswith("_nominal") else feat

    if variation == "nominal" or variation.startswith("wgt"):
        use_var = "nominal"
    elif "soft" in feat:
        use_var = "nominal"
    else:
        use_var = variation

    candidates = [f"{base}_{use_var}", f"{base}_nominal", feat]
    for idx, c in enumerate(candidates):
        if c in fields:
            if idx > 0:
                logger.warning(
                    f"[stage2] DNN feature '{base}_{use_var}' unavailable for "
                    f"variation '{variation}'; falling back to '{c}'."
                )
            logger.debug(
                f"[stage2][field-resolve] kind=dnn variation={variation} "
                f"var={feat} resolved={c} fallback={idx > 0}"
            )
            return c
    if allow_nominal_fallback:
        logger.warning(
            f"[stage2] DNN feature '{base}_{use_var}' (and nominal/base) unavailable "
            f"for variation '{variation}'; falling back to unverified '{feat}'."
        )
        logger.debug(
            f"[stage2][field-resolve] kind=dnn variation={variation} "
            f"var={feat} resolved={feat} fallback=True"
        )
        return feat
    raise KeyError(
        f"Feature {feat} (var={variation}) not found in fields. "
        f"Tried: {candidates}. "
        "DNN inputs are resolved to nominal fields for all variations."
    )


# --------------------------------------------------------------------------
# Stage-2 exclusions that the training side must mirror
# --------------------------------------------------------------------------
def stage2_shape_variations(fields, prefixes=None) -> List[str]:
    """Exactly the shape-variation list Stage-2 puts into ``syst_variations``.

    ``run_stage2_vbf.VbfProcessor.process`` builds it as
    ``discover_shape_systs(fields)`` filtered by ``not syst.startswith("log_")``
    (run_stage2_vbf.py, the ``syst_variations`` block).  The ``log_`` filter drops
    the spurious suffixes produced when a ``*_log_<syst>_up`` branch is matched by
    the shorter ``jj_mass_``/``dimuon_pt_`` prefix.  Training must use the same
    list or it would decorrelate against variations Stage-2 never evaluates.
    """
    return [s for s in discover_shape_systs(fields, prefixes=prefixes) if not s.startswith("log_")]


# --------------------------------------------------------------------------
# Year decorrelation
# --------------------------------------------------------------------------
#: A JEC source is year-decorrelated when its suffix carries a year token, e.g.
#: ``Absolute_2018_up`` / ``RelativeSample_2016APV_down``.  Only the event's own
#: year exists in its Stage-1 file, so the training code maps each of these onto
#: a year-agnostic canonical slot (see :func:`canonical_variation_name`).
YEAR_TOKEN_RE = re.compile(r"^(?P<source>[A-Za-z][A-Za-z0-9]*)_(?P<token>(?:19|20)\d{2}[A-Za-z]*)$")

#: Placeholder substituted for the year token in a canonical variation name.
YEAR_DECOR_TOKEN = "yearDecor"


def split_variation(variation: str) -> Tuple[str, str]:
    """Split ``"Absolute_2018_up"`` -> ``("Absolute_2018", "up")``."""
    for direction in ("_up", "_down"):
        if variation.endswith(direction):
            return variation[: -len(direction)], direction[1:]
    raise ValueError(f"Variation {variation!r} does not end in '_up' or '_down'.")


def is_year_decorrelated(variation: str) -> bool:
    """True for suffixes whose source carries an explicit year token."""
    base, _ = split_variation(variation)
    return YEAR_TOKEN_RE.match(base) is not None


def canonical_variation_name(variation: str) -> str:
    """Map a per-year variation onto a year-agnostic slot name.

    ``Absolute_2018_up`` and ``Absolute_2016APV_up`` both become
    ``Absolute_yearDecor_up``; correlated suffixes such as ``Total_up`` or
    ``mu_roccor_down`` are returned unchanged.

    This is what makes a *single* homogeneous variation axis possible for a
    training set that spans four years: every event contributes its own year's
    decorrelated source in the same slot.
    """
    base, direction = split_variation(variation)
    m = YEAR_TOKEN_RE.match(base)
    if m is None:
        return variation
    return f"{m.group('source')}_{YEAR_DECOR_TOKEN}_{direction}"


def canonical_variation_list(variations: Iterable[str]) -> List[str]:
    """Sorted, de-duplicated canonical names for an iterable of variations."""
    return sorted({canonical_variation_name(v) for v in variations})


# --------------------------------------------------------------------------
# Variation-set selection (sweep vs full)
# --------------------------------------------------------------------------
#: Reduced set used for the lambda sweep: JEC ``Total`` plus ``mu_roccor``,
#: both directions -> 4 variations, ~5x forward-pass cost.
SWEEP_VARIATION_SOURCES: Tuple[str, ...] = ("Total", "mu_roccor")

VARIATION_SETS: Tuple[str, ...] = ("sweep", "full")


def select_variations(variations: Sequence[str], variation_set: str) -> List[str]:
    """Filter a discovered per-year variation list down to the requested set.

    ``"full"``  -> every discovered shape variation (26 per event for Run-2).
    ``"sweep"`` -> only :data:`SWEEP_VARIATION_SOURCES`, up and down (4).
    """
    if variation_set not in VARIATION_SETS:
        raise ValueError(
            f"Unknown variation_set {variation_set!r}; expected one of {VARIATION_SETS}."
        )
    if variation_set == "full":
        return list(variations)
    keep = []
    for v in variations:
        base, _ = split_variation(v)
        if base in SWEEP_VARIATION_SOURCES:
            keep.append(v)
    return keep


# --------------------------------------------------------------------------
# Column naming for the variation-augmented fold parquets
# --------------------------------------------------------------------------
#: Separator between a training-feature name and a canonical variation name in
#: the augmented fold parquets.  Column names are *not* parsed back apart at
#: read time -- ``preprocess_manifest.json`` carries the explicit mapping -- but
#: the separator keeps the files readable by eye.
VARIATION_COL_SEP = "__var__"


def variation_column_name(feature: str, canonical_variation: str) -> str:
    """Augmented-parquet column holding ``feature`` under ``canonical_variation``."""
    return f"{feature}{VARIATION_COL_SEP}{canonical_variation}"


def resolve_variation_columns(
    features: Sequence[str],
    variations: Sequence[str],
    fields,
    year_onehot_prefix: str = "year_",
) -> Dict[str, Dict[str, str]]:
    """Per-variation map of ``{feature: stage1 column}`` for *shifted* features only.

    Features that fall back to nominal under a variation are omitted rather than
    stored redundantly -- that is exactly the Stage-2
    :func:`feature_name_for_variation` fallback (a JEC variation leaves the six
    dimuon features and the two CS angles at nominal; ``mu_roccor`` leaves the
    jet features at nominal; soft-drop features are pinned to nominal for every
    variation).  A feature absent from the map contributes *identically* to
    nominal, so the consistency term sees exactly zero from it.

    One-hot year features carry no event field and are always omitted.
    """
    fields = set(fields)
    out: Dict[str, Dict[str, str]] = {}
    for variation in variations:
        per_feature: Dict[str, str] = {}
        for feat in features:
            if feat.startswith(year_onehot_prefix):
                continue
            base = feat[: -len("_nominal")] if feat.endswith("_nominal") else feat
            shifted = f"{base}_{variation}"
            # Mirror feature_name_for_variation's soft-drop pinning without
            # emitting its per-call warning for the (expected) fallbacks.
            if "soft" in feat:
                continue
            if shifted in fields:
                per_feature[feat] = shifted
        out[variation] = per_feature
    return out


def assert_matches_stage2(
    features: Sequence[str],
    variations: Sequence[str],
    fields,
) -> None:
    """Cross-check :func:`resolve_variation_columns` against Stage-2 resolution.

    Raises ``AssertionError`` on any disagreement.  Used by the unit tests and by
    the preprocessing so a future edit to either path cannot silently drift.
    """
    fields = set(fields)
    resolved = resolve_variation_columns(features, variations, fields)
    for variation in variations:
        for feat in features:
            if feat.startswith("year_"):
                continue
            stage2_col = feature_name_for_variation(feat, variation, fields)
            ours = resolved[variation].get(feat)
            if ours is None:
                nominal_col = feature_name_for_variation(feat, "nominal", fields)
                assert stage2_col == nominal_col, (
                    f"variation={variation} feature={feat}: helper says 'falls back to "
                    f"nominal' ({nominal_col}) but stage2 resolves {stage2_col}"
                )
            else:
                assert stage2_col == ours, (
                    f"variation={variation} feature={feat}: helper says {ours} but "
                    f"stage2 resolves {stage2_col}"
                )
