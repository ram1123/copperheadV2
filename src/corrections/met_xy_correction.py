"""MET xy-shift ("phi modulation") correction for PuppiMET, Run 2 UL.

Inhomogeneities in detector response vs. phi introduce a modulation in the
mean x/y components of the measured MET. The xy-shift correction removes
this by fitting the mean MET x/y component vs. N(vertices) in well-balanced
Z(mu mu) events, separately per data-taking era and for MC, then subtracting
the fitted value:

    METxcorr = -(a_x * npv + b_x)
    METycorr = -(a_y * npv + b_y)
    corrected_x = met_pt * cos(met_phi) + METxcorr
    corrected_y = met_pt * sin(met_phi) + METycorr

npv is capped at 100. This is applied identically to data and MC, each with
its own fitted (a, b) coefficients -- it is not a data/MC-derived scale
factor, it is a per-sample-type kinematic recipe (like a JEC), so applying it
to data uses data's own fit, not an "MC scale factor".

Source / provenance
--------------------
- new_features/MET_phi_modulation/ref_material/metxycorrections_UL2016.pdf
  (L. Thomas, "MET XY corrections in UL2016", JME General, 28 Apr 2021) --
  method description (slides 2-3) and confirmation that the effect on
  PuppiMET specifically is present but small (slide 15, 26).
- new_features/MET_phi_modulation/ref_material/XYMETCorrection_withUL17andUL18andUL16.h
  -- the reference C++ macro linked from the slides. All numeric
  coefficients and run-number era boundaries below were parsed
  programmatically (regex) from that file, not hand-transcribed, to avoid
  transcription error; see
  .agent-system/tasks/implement-met-phi-modulation-correction/iterations/002/generator-report.md
  (2016) and iterations/003/generator-report.md (2017/2018 extension) for the
  extraction script and re-verification.

Puppi-MET, full Run 2 UL (2016preVFP/APV, 2016postVFP/nonAPV, 2017, 2018;
data and MC) coefficients are included. Data era selection is purely by run
number (matching the reference macro's own behavior, which does not gate
data-era matching on a "year" argument either) -- this is safe because UL
Run 2 run-number ranges for 2016/2017/2018 are mutually disjoint.

No uncertainty (Up/Down) variation is implemented for this correction: the
reference material provides no fit uncertainty on the (a, b) coefficients,
and CMS does not assign this correction a dedicated systematic in normal
usage (it is applied as a deterministic kinematic recipe, like a JEC central
value, not as a reweighting). This is a documented decision, not an
omission -- see the generator reports referenced above.
"""
from __future__ import annotations

import awkward as ak
import numpy as np

# Reference: https://lathomas.web.cern.ch/METStuff/XYCorrections/XYMETCorrection_withUL17andUL18andUL16.h and https://indico.cern.ch/event/1033432/contributions/4339934/attachments/2235168/3788215/metxycorrections_UL2016.pdf
# Puppi MET xy-shift coefficients: era_key -> {"x": (a, b), "y": (a, b)}.
# Parsed from XYMETCorrection_withUL17andUL18andUL16.h, `if(ispuppi){...}` block.
PUPPI_XY_COEFFS_RUN2 = {
    # --- 2016 ---
    "UL2016B": {"x": (-0.00109025, -0.338093), "y": (-0.00356058, 0.128407)},
    "UL2016C": {"x": (-0.00271913, -0.342268), "y": (0.00187386, 0.104)},
    "UL2016D": {"x": (-0.00254194, -0.305264), "y": (-0.00177408, 0.164639)},
    "UL2016E": {"x": (-0.00358835, -0.225435), "y": (-0.000444268, 0.180479)},
    "UL2016F": {"x": (0.0056759, -0.454101), "y": (-0.00962707, 0.35731)},        # preVFP (APV) part of run F
    "UL2016Flate": {"x": (0.0234421, -0.371298), "y": (-0.00997438, 0.0809178)},  # postVFP (nonAPV) part of run F
    "UL2016G": {"x": (0.0182134, -0.335786), "y": (-0.0063338, 0.093349)},
    "UL2016H": {"x": (0.015702, -0.340832), "y": (-0.00544957, 0.199093)},
    "UL2016MCAPV": {"x": (-0.0060447, -0.4183), "y": (0.008331, -0.0990046)},
    "UL2016MCnonAPV": {"x": (-0.0058341, -0.395049), "y": (0.00971595, -0.101288)},
    # --- 2017 ---
    "UL2017B": {"x": (-0.00382117, -0.666228), "y": (0.0109034, 0.172188)},
    "UL2017C": {"x": (-0.00110699, -0.747643), "y": (-0.0012184, 0.303817)},
    "UL2017D": {"x": (-0.00141442, -0.721382), "y": (-0.0011873, 0.21646)},
    "UL2017E": {"x": (0.00593859, -0.851999), "y": (-0.00754254, 0.245956)},
    "UL2017F": {"x": (0.00765682, -0.945001), "y": (-0.0154974, 0.804176)},
    "UL2017MC": {"x": (-0.0102265, -0.446416), "y": (0.0198663, 0.243182)},
    # --- 2018 ---
    "UL2018A": {"x": (-0.0073377, 0.0250294), "y": (-0.000406059, 0.0417346)},
    "UL2018B": {"x": (0.00434261, 0.00892927), "y": (0.00234695, 0.20381)},
    "UL2018C": {"x": (0.00198311, 0.37026), "y": (-0.016127, 0.402029)},
    "UL2018D": {"x": (0.00220647, 0.378141), "y": (-0.0160244, 0.471053)},
    "UL2018MC": {"x": (-0.0214557, 0.969428), "y": (0.0167134, 0.199296)},
}

# Data run-range -> era boundaries, full Run 2 UL (source: same .h file,
# if/else chain for `!isMC && ... && isUL`).
# (era_key, run_low, run_high, extra_single_run_or_None).
# The single extra run values (278769/278770) are the exact run at which the
# HIP-mitigation fix (APV/nonAPV split) lands within 2016 run F. Run-number
# ranges across 2016/2017/2018 are mutually disjoint, so this single combined
# table is safe to apply regardless of the declared `year` (matching the
# reference macro's own year-agnostic data-era matching).
RUN2_DATA_RUN_RANGES = [
    ("UL2016B", 272007, 275376, None),
    ("UL2016C", 275657, 276283, None),
    ("UL2016D", 276315, 276811, None),
    ("UL2016E", 276831, 277420, None),
    ("UL2016F", 277772, 278768, 278770),
    ("UL2016Flate", 278801, 278808, 278769),
    ("UL2016G", 278820, 280385, None),
    ("UL2016H", 280919, 284044, None),
    ("UL2017B", 297020, 299329, None),
    ("UL2017C", 299337, 302029, None),
    ("UL2017D", 302030, 303434, None),
    ("UL2017E", 303435, 304826, None),
    ("UL2017F", 304911, 306462, None),
    ("UL2018A", 315252, 316995, None),
    ("UL2018B", 316998, 319312, None),
    ("UL2018C", 319313, 320393, None),
    ("UL2018D", 320394, 325273, None),
]

# year -> MC era key.
_MC_ERA_KEY_BY_YEAR = {
    "2016preVFP": "UL2016MCAPV",
    "2016postVFP": "UL2016MCnonAPV",
    "2017": "UL2017MC",
    "2018": "UL2018MC",
}

_SUPPORTED_YEARS = tuple(_MC_ERA_KEY_BY_YEAR)


def _data_era_coeffs(run: ak.Array):
    """Per-event (a_x, b_x, a_y, b_y, matched) for Run 2 UL Puppi MET
    xy-shift, data, selected by run number alone. Events whose run doesn't
    match any known UL era get zero correction (no-op), mirroring the
    reference macro's own fallback ("couldn't find data/MC era => no
    correction applied").
    """
    zeros = ak.zeros_like(run, dtype=np.float64)
    a_x, b_x, a_y, b_y = zeros, zeros, zeros, zeros
    matched = ak.zeros_like(run, dtype=np.bool_)
    for era_key, lo, hi, extra_run in RUN2_DATA_RUN_RANGES:
        in_range = (run >= lo) & (run <= hi)
        if extra_run is not None:
            in_range = in_range | (run == extra_run)
        c = PUPPI_XY_COEFFS_RUN2[era_key]
        a_x = ak.where(in_range, c["x"][0], a_x)
        b_x = ak.where(in_range, c["x"][1], b_x)
        a_y = ak.where(in_range, c["y"][0], a_y)
        b_y = ak.where(in_range, c["y"][1], b_y)
        matched = matched | in_range
    return a_x, b_x, a_y, b_y, matched


def apply_puppi_met_xy_correction(
    met_pt: ak.Array,
    met_phi: ak.Array,
    npvs: ak.Array,
    run: ak.Array,
    year: str,
    is_mc: bool,
):
    """Return (corrected_met_pt, corrected_met_phi, n_unmatched_events).

    n_unmatched_events is a lazy (dask-awkward-compatible) scalar count of
    data events whose run number matched no known Run 2 UL era and therefore
    received zero correction; it should be 0 (or negligible) for well-formed
    UL inputs and is intended for the Code Runner to log.
    """
    if year not in _SUPPORTED_YEARS:
        raise ValueError(
            f"MET xy-shift correction coefficients are only available for "
            f"{_SUPPORTED_YEARS}, got year={year!r}. See met_xy_correction.py "
            f"docstring for how to extend to other years."
        )

    npv_capped = ak.where(npvs > 100, 100, npvs)
    npv_capped = ak.values_astype(npv_capped, np.float64)

    if is_mc:
        era_key = _MC_ERA_KEY_BY_YEAR[year]
        c = PUPPI_XY_COEFFS_RUN2[era_key]
        a_x, b_x = c["x"]
        a_y, b_y = c["y"]
        n_unmatched = 0
    else:
        a_x, b_x, a_y, b_y, matched = _data_era_coeffs(run)
        n_unmatched = ak.sum(~matched)

    met_x_corr = -(a_x * npv_capped + b_x)
    met_y_corr = -(a_y * npv_capped + b_y)

    met_x = met_pt * np.cos(met_phi) + met_x_corr
    met_y = met_pt * np.sin(met_phi) + met_y_corr

    corrected_pt = np.sqrt(met_x**2 + met_y**2)
    # np.arctan2 is the vectorized equivalent of the reference macro's manual
    # quadrant-branching atan (TMath::ATan + pi/-pi adjustments); it handles
    # all quadrants and the x==0 edge cases directly.
    corrected_phi = np.arctan2(met_y, met_x)

    return corrected_pt, corrected_phi, n_unmatched
