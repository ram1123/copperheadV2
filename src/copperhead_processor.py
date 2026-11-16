import logging
import json
import os
import time
from typing import TypeVar

import awkward as ak
import coffea.processor as processor
import numpy as np
from coffea.analysis_tools import PackedSelection, Weights
from coffea.btag_tools import BTagScaleFactor
from coffea.lumi_tools import LumiMask
from omegaconf import OmegaConf

from modules.awkward_utils import ensure_event_axis
from modules.classify_year import is_run2, is_run3, run3_btag_score
from modules.correctionlib_file_cache import get_corrset
from modules.get_sample_info import get_sample_info
from modules.utils import logger

from modules.vector_operations import (
    _delta_phi,
    cs_variables,
    delta_r_V1,
    etaFrame_variables,
    getRapidity,
)

from src.corrections.evaluator import (
    PDF_ALPHA_S_MEMBER_INDICES,
    PDF_N_EIGENVECTOR_MEMBERS,
    add_pdf_variations,
    add_stxs_variations,
    btag_fixedwp_tagger,
    btag_fixedwp_weights,
    btag_weights_jsonKeepDim,
    get_jetpuid_weights_eta_dependent,
    get_pdf_lha_id_range,
    lhe_weights,
    nnlops_weights,
    ps_weights,
    pu_evaluator,
    qgl_weights_V2,
)
from src.corrections.fsr_recovery import fsr_recoveryV1
from src.corrections.jet import (
    applyHemVeto,
    applyJetUncertaintyKinematics,
    applyUpDown,
    btag_jet_selection,
    custom_jet_id,
    fatjet_tight_id_correctionlib,
    do_jec_scale,
    do_jer_smear,
    fill_softjets_HIG19006,
    get_jec_sources,
    get_jet_variation,
    get_puId,
    jet_id,
    jet_puid,
)
from src.corrections.muon_sf import add_muon_sfs_correctionlib, muon_sfs_stat_syst_correctionlib
from src.corrections.pu_dnn import add_dphi_met_jet_features, eval_pu_dnn, load_pu_dnn_configs
from src.corrections.rochester import apply_KitMuScaleRe_Run3, apply_KitMuScaleRe_Run3_bs_split, apply_roccor
from src.corrections.met_xy_correction import apply_puppi_met_xy_correction

coffea_nanoevent = TypeVar('coffea_nanoevent')
ak_array = TypeVar('ak_array')


# ---------------------------------------------------------
# Load all region JSON configs
# ---------------------------------------------------------
def load_pysr_configs(base_dir="configs/pysr_best"):
    configs = {}
    if not os.path.isdir(base_dir):
        logger.warning(f"PySR config directory not found: {base_dir}")
        return configs

    for fname in os.listdir(base_dir):
        if not fname.endswith(".json"):
            continue
        if fname.startswith("summary_all"):
            continue

        with open(os.path.join(base_dir, fname)) as f:
            cfg = json.load(f)

        region = cfg["region"]
        configs[region] = cfg

    return configs


# ---------------------------------------------------------
# Region mask (eta-based)
# ---------------------------------------------------------
def region_mask_eta(eta, region):
    abs_eta = abs(eta)

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


# ---------------------------------------------------------
# Add derived symbolic features if needed
# ---------------------------------------------------------
def ensure_symbolic_features(jets, features):
    if "logpt" in features and "logpt" not in jets.fields:
        jets = ak.with_field(
            jets,
            np.log(ak.where(jets.pt > 1e-6, jets.pt, 1e-6)),
            "logpt",
        )

    if "lognC" in features and "lognC" not in jets.fields:
        jets = ak.with_field(
            jets,
            np.log1p(jets.nConstituents),
            "lognC",
        )

    if "hf_ratio" in features and "hf_ratio" not in jets.fields:
        jets = ak.with_field(
            jets,
            jets.hfadjacentEtaStripsSize / ak.where(
                np.abs(jets.hfcentralEtaStripSize) > 1e-6,
                jets.hfcentralEtaStripSize,
                1e-6,
            ),
            "hf_ratio",
        )

    if "hf_sigma_sum" in features and "hf_sigma_sum" not in jets.fields:
        jets = ak.with_field(
            jets,
            jets.hfsigmaEtaEta + jets.hfsigmaPhiPhi,
            "hf_sigma_sum",
        )

    return jets


# ---------------------------------------------------------
# Safe equation evaluation (awkward-safe)
# ---------------------------------------------------------
def evaluate_symbolic_equation(jets, cfg):
    local_dict = {f"x{i}": jets[feat] for i, feat in enumerate(cfg["features"])}

    safe_dict = {
        "np": np,
        "abs": np.abs,
        "Abs": np.abs,
        "sqrt": np.sqrt,
        "sqrt_abs": lambda x: np.sqrt(np.abs(x)),
        "log": lambda x: np.log(ak.where(x > 1e-6, x, 1e-6)),
        "log1p": lambda x: np.log1p(ak.where(x > 0, x, 0)),
        "tanh": np.tanh,
        "log1p_abs": lambda x: np.log1p(np.abs(x)),
        "log_abs": lambda x: np.log(np.maximum(np.abs(x), 1e-6)),
    }

    return eval(cfg["_compiled_equation"], safe_dict, local_dict)


# ---------------------------------------------------------
# Main mask builder
# ---------------------------------------------------------
def build_pysr_pu_masks(jets, configs):
    pt = jets.pt
    eta = jets.eta

    # Initialize everything as True (pass by default)
    combined_pass = ak.ones_like(pt, dtype=bool)

    outputs = {}
    regions_out = []

    for region, cfg in configs.items():

        # -----------------------
        # region + pt window
        # -----------------------
        mask_region = region_mask_eta(eta, region)
        mask_pt = (pt >= cfg["pt_min"]) & (pt < cfg["pt_turnoff"])
        mask_active = mask_region & mask_pt

        # -----------------------
        # score
        # -----------------------
        score = evaluate_symbolic_equation(jets, cfg)

        if cfg["direction"] == "keep_high":
            pass_core = score > cfg["threshold"]
        else:
            pass_core = score < cfg["threshold"]

        # -----------------------
        # Only apply inside active region
        # Outside region/pt → pass automatically
        # -----------------------
        pass_region = (~mask_active) | pass_core

        outputs[f"jet_pysr_{region}_score"] = score
        outputs[f"jet_pysr_{region}_pass"] = pass_region

        combined_pass = combined_pass & pass_region
        regions_out.append(region)

    outputs["jet_pysr_pu_pass"] = combined_pass

    return outputs, regions_out


def getJetType(NanoAODv: int, year: str):
    if is_run2(year):
        if NanoAODv<15:
            jet_type = "AK4PFchs"
        else: # nanoV15
            jet_type = "AK4PFPuppi"
    else: # run3 all jets are PUPPI
        jet_type = "AK4PFPuppi"
    return jet_type


def safe_ratio(num, den, default=0.0):
    """Element-wise safe division for awkward arrays."""
    return ak.where(den != 0, num / den, default)


def _higgs_peak_and_sidebands(mass):
    """(h_peak: 115-135 GeV, h_sidebands: 110-115 or 135-150 GeV) masks."""
    h_peak = (mass >= 115.0) & (mass < 135.0)
    h_sidebands = ((mass >= 110.0) & (mass < 115.0)) | ((mass >= 135.0) & (mass < 150.0))
    return h_peak, h_sidebands


def _leading_two(muons, key="pt"):
    """(mu1, mu2) sorted by `key`, padded with None for events with < 2 muons."""
    padded = ak.pad_none(muons, 2)
    muons_sorted = padded[ak.argsort(padded[key], ascending=False)]
    return muons_sorted[:, 0], muons_sorted[:, 1]


def build_muon_kinematic_variation_block(
    processor,
    mu1,
    mu2,
    pt1,
    pt2,
    suffix: str,
    is_mc: bool,
    doing_BS_correction: bool = False,
):
    mu1_var = ak.with_field(mu1, pt1, "pt")
    mu2_var = ak.with_field(mu2, pt2, "pt")
    dimuon_var = mu1_var + mu2_var
    dimuon_mass_res_var, calibration_var = processor.get_mass_resolution(
        dimuon_var,
        mu1_var,
        mu2_var,
        is_mc,
        doing_BS_correction=doing_BS_correction,
    )
    dimuon_mass_res_var = dimuon_mass_res_var * calibration_var

    return {
        f"mu1_pt_{suffix}": mu1_var.pt,
        f"mu2_pt_{suffix}": mu2_var.pt,
        f"mu1_pt_over_mass_{suffix}": safe_ratio(mu1_var.pt, dimuon_var.mass, default=0.0),
        f"mu2_pt_over_mass_{suffix}": safe_ratio(mu2_var.pt, dimuon_var.mass, default=0.0),
        f"dimuon_mass_{suffix}": dimuon_var.mass,
        f"dimuon_pt_{suffix}": dimuon_var.pt,
        f"dimuon_pt_log_{suffix}": np.log(ak.where(dimuon_var.pt > 0, dimuon_var.pt, 1e-6)),
        f"dimuon_eta_{suffix}": dimuon_var.eta,
        f"dimuon_rapidity_{suffix}": getRapidity(dimuon_var),
        f"dimuon_ebe_mass_res_{suffix}": dimuon_mass_res_var,
        f"dimuon_ebe_mass_res_rel_{suffix}": safe_ratio(
            dimuon_mass_res_var, dimuon_var.mass, default=0.0
        ),
    }


def _load_zpt_config_section(year: str, config_path: str, NanoAODv: int):
    logger.debug(f"zpt config file: {config_path} (year={year}, nanoAODv{NanoAODv})")
    wgt_config = OmegaConf.load(config_path)
    wgt_config = wgt_config[str(year)]
    if ("nanoAODv12" in wgt_config.keys()) or ("nanoAODv15" in wgt_config.keys()):
        try:
            wgt_config = wgt_config[f"nanoAODv{NanoAODv}"]
        except Exception as exc:
            raise ValueError(
                f"Zpt config for nanoAODv{NanoAODv} is not yet available!"
            ) from exc
    return wgt_config


def _eval_zpt_poly(wgt_config, jet_multiplicity: int, nbins, prefix: str, order: int, x, sigma_shift: float = 0.0):
    """
    Evaluate the fitted polynomial (prefix "f0" or "f1") at `x`, optionally
    shifted by `sigma_shift` sigma.

    get_polyFit.py (the derivation script) only saves each coefficient's own
    marginal fit uncertainty (`{prefix}_p{i}_err`) -- it does not save the
    coefficient covariance matrix. Absent that, the +-1 sigma envelope at a
    given x is built by combining each order's error contribution in
    quadrature (the standard fallback for combining independent error
    sources), NOT by shifting every coefficient by its own full marginal
    error in the same direction at once. The latter (this function's
    predecessor, _get_zpt_coeff) implicitly assumes every coefficient's
    error is 100% positively correlated with every other's, which is not
    physical for a polynomial fit and, combined with x^order growth for a
    6th-order polynomial, produced +-1 sigma "variations" tens of times
    larger than the nominal weight (and frequently negative) -- see
    docs/... zpt variation investigation.

    Caveat this does NOT fix: true polynomial fit coefficients are usually
    strongly (anti-)correlated, so even an independent-quadrature combination
    of marginal errors overstates the true fit uncertainty at x away from the
    fit's pivot. A fully correct band needs either the coefficient covariance
    matrix or the underlying combined-fit parameters (common_shift/low_tilt/
    mid_tilt/delta_tail_slope, computed but not persisted in get_polyFit.py's
    build_final_piecewise_coefficients) to be saved and varied instead of the
    14 derived per-order coefficients -- that requires re-running the fit
    derivation, not just this evaluation code.
    """
    nominal = 0.0
    variance = 0.0
    for i in range(order + 1):
        base = wgt_config[f"njet_{jet_multiplicity}"][nbins][f"{prefix}_p{i}"]
        term = base * (x ** i)
        nominal = nominal + term
        if sigma_shift != 0.0:
            err = wgt_config[f"njet_{jet_multiplicity}"][nbins].get(f"{prefix}_p{i}_err", 0.0)
            variance = variance + (err * (x ** i)) ** 2
    if sigma_shift == 0.0:
        return nominal
    envelope = variance ** 0.5
    return nominal + sigma_shift * envelope


def _eval_zpt_old_quadrature(
    wgt_config, jet_multiplicity, nbins, dimuon_pt, f0_order, f1_order,
    poly_fit_cutoff_min, poly_fit_cutoff_max, sigma_shift,
):
    """
    Evaluate the full 3-piece Z-pT function (also used for the nominal,
    sigma_shift=0 case -- identical to the original implementation then).
    For sigma_shift != 0 this is the legacy per-coefficient independent-
    quadrature fallback (_eval_zpt_poly) for YAML files that predate the 4
    combined-fit parameters below -- horizontal_mx itself is never varied,
    matching the original implementation (only f0/f1 are perturbed; the
    horizontal segment's slope is fixed and its intercept just follows
    continuity from f1's shifted value at xmax1).
    """
    def eval_poly(prefix, order, x):
        return _eval_zpt_poly(wgt_config, jet_multiplicity, nbins, prefix, order, x, sigma_shift)

    mx = wgt_config[f"njet_{jet_multiplicity}"][nbins]["horizontal_mx"]
    return _zpt_piecewise(dimuon_pt, eval_poly, f0_order, f1_order, poly_fit_cutoff_min, poly_fit_cutoff_max, mx)


def _zpt_piecewise(dimuon_pt, eval_poly, f0_order, f1_order, poly_fit_cutoff_min, poly_fit_cutoff_max, mx):
    """3-piece Z-pT function: f0 below xmin1, f1 (continuous at xmin1) up to xmax1, then a line of slope mx."""
    f0_val = eval_poly("f0", f0_order, dimuon_pt)
    f0_xmin = eval_poly("f0", f0_order, poly_fit_cutoff_min)
    f1_val = eval_poly("f1", f1_order, dimuon_pt)
    f1_xmin = eval_poly("f1", f1_order, poly_fit_cutoff_min)
    f1_xmax = eval_poly("f1", f1_order, poly_fit_cutoff_max)

    offset = f0_xmin - f1_xmin
    y_at_xmax = f1_xmax + offset
    intercept = y_at_xmax - mx * poly_fit_cutoff_max

    value = ak.where(poly_fit_cutoff_min >= dimuon_pt, f0_val, f1_val + offset)
    value = ak.where(poly_fit_cutoff_max < dimuon_pt, mx * dimuon_pt + intercept, value)
    return value


# How a +1 sigma shift in each of the 4 parameters get_polyFit.py's combined
# refit actually fits independently (common_shift/low_tilt/mid_tilt/
# delta_tail_slope -- see make_combined_function_reduced /
# build_final_piecewise_coefficients there) propagates into the *already
# saved* final f0_p0/f0_p1/f1_p0/f1_p1/horizontal_mx coefficients. This
# mapping is exactly linear (not an approximation) and needs only those final
# coefficients plus xmin1 -- not the raw/local per-region fit coefficients,
# which aren't saved. Derivation: common_shift and low_tilt only ever enter
# f0's coefficients (low_tilt vanishes at xmin1 by construction, so it can't
# disturb f0/f1 continuity there); mid_tilt enters f1's coefficients the same
# way *and* shifts the tail slope by the same amount, since
# final_tail_slope = derivative-of-final-f1-at-xmax1 + delta_tail_slope, and
# a +delta shift to f1's linear coefficient shifts that derivative by exactly
# +delta everywhere; delta_tail_slope only ever enters the tail slope.
_ZPT_COMBINED_PARAM_DELTAS = {
    "common_shift": lambda err, xmin1: {"f0_p0": err, "f1_p0": err},
    "low_tilt": lambda err, xmin1: {"f0_p0": -err * xmin1, "f0_p1": err},
    "mid_tilt": lambda err, xmin1: {"f1_p0": -err * xmin1, "f1_p1": err, "horizontal_mx": err},
    "delta_tail_slope": lambda err, xmin1: {"horizontal_mx": err},
}


def _eval_zpt_piecewise_with_deltas(
    wgt_config, jet_multiplicity, nbins, dimuon_pt, f0_order, f1_order,
    poly_fit_cutoff_min, poly_fit_cutoff_max, coeff_deltas,
):
    """
    Same 3-piece evaluation as _eval_zpt_old_quadrature's nominal case, but
    with the given additive `coeff_deltas` (a sparse {"f0_p0": ..., ...}
    dict, see _ZPT_COMBINED_PARAM_DELTAS) applied to the named low-order
    coefficients/horizontal_mx before evaluating -- used to measure one
    combined-fit parameter's effect on the weight at a time.
    """
    def coeff(prefix, order_idx):
        base = wgt_config[f"njet_{jet_multiplicity}"][nbins][f"{prefix}_p{order_idx}"]
        return base + coeff_deltas.get(f"{prefix}_p{order_idx}", 0.0)

    def eval_poly(prefix, order, x):
        total = 0.0
        for i in range(order + 1):
            total = total + coeff(prefix, i) * (x ** i)
        return total

    mx = wgt_config[f"njet_{jet_multiplicity}"][nbins]["horizontal_mx"] + coeff_deltas.get("horizontal_mx", 0.0)
    return _zpt_piecewise(dimuon_pt, eval_poly, f0_order, f1_order, poly_fit_cutoff_min, poly_fit_cutoff_max, mx)


_ZPT_COMBINED_PARAM_ORDER = ["common_shift", "low_tilt", "mid_tilt", "delta_tail_slope"]


def _zpt_combined_param_envelope_full_cov(
    wgt_config, jet_multiplicity, nbins, dimuon_pt, f0_order, f1_order,
    poly_fit_cutoff_min, poly_fit_cutoff_max, nominal_value,
):
    """
    +-1 sigma envelope from the 4 combined-refit parameters
    (common_shift/low_tilt/mid_tilt/delta_tail_slope), propagated through
    their FULL covariance matrix: sigma(x)^2 = g(x)^T Cov g(x), where g(x)
    is the weight's gradient w.r.t. each parameter. This is the preferred
    method -- the correct one, not an independence approximation.

    g(x) is obtained exactly (not by finite differencing): the mapping from
    each combined-fit parameter to the final f0_p0/f0_p1/f1_p0/f1_p1/
    horizontal_mx coefficients is exactly linear (_ZPT_COMBINED_PARAM_DELTAS),
    so evaluating _eval_zpt_piecewise_with_deltas with a *unit* shift in one
    parameter (instead of that parameter's own error) gives exactly
    d(weight)/d(param), with no discretization error.

    Returns None if this YAML predates the saved combined_fit_covariance
    field (i.e. was derived before get_polyFit.py started saving it), so the
    caller can fall back to _zpt_combined_param_envelope_diagonal.
    """
    cfg = wgt_config[f"njet_{jet_multiplicity}"][nbins]
    if "combined_fit_covariance" not in cfg:
        return None

    cov = cfg["combined_fit_covariance"]
    cov_order = cfg.get("combined_fit_covariance_order", _ZPT_COMBINED_PARAM_ORDER)

    gradients = {}
    for name in cov_order:
        deltas = _ZPT_COMBINED_PARAM_DELTAS[name](1.0, poly_fit_cutoff_min)
        shifted = _eval_zpt_piecewise_with_deltas(
            wgt_config, jet_multiplicity, nbins, dimuon_pt, f0_order, f1_order,
            poly_fit_cutoff_min, poly_fit_cutoff_max, deltas,
        )
        gradients[name] = shifted - nominal_value

    variance = 0.0
    for i, name_i in enumerate(cov_order):
        for j, name_j in enumerate(cov_order):
            variance = variance + gradients[name_i] * gradients[name_j] * cov[i][j]

    # Exact math guarantees variance >= 0 (a quadratic form g^T Cov g with
    # Cov positive semi-definite); guard only against floating-point noise
    # right at zero-envelope points.
    variance = ak.where(variance < 0.0, 0.0, variance)
    return variance ** 0.5


def _zpt_combined_param_envelope_diagonal(
    wgt_config, jet_multiplicity, nbins, dimuon_pt, f0_order, f1_order,
    poly_fit_cutoff_min, poly_fit_cutoff_max, nominal_value,
):
    """
    Second-tier fallback (used only for a YAML that predates the saved
    combined_fit_covariance field): +-1 sigma envelope from the 4 combined
    refit parameters, treated as independent and combined in quadrature --
    ignores the (generally small but nonzero) correlations between them that
    _zpt_combined_param_envelope_full_cov propagates correctly. Still a much
    better-motivated independence assumption than treating the 14 per-order
    f0_pN/f1_pN coefficients as independent (_eval_zpt_old_quadrature's own
    fallback): these 4 were fit together as genuinely separate MINUIT
    parameters, and 3 of the 4 are defined to vanish at their own anchor
    point specifically so they don't leak into the other regions. Returns
    None if this YAML predates even these 4 saved *_err fields, so the
    caller can fall back further.
    """
    cfg = wgt_config[f"njet_{jet_multiplicity}"][nbins]
    if not all(f"{name}_err" in cfg for name in _ZPT_COMBINED_PARAM_DELTAS):
        return None

    variance = 0.0
    for name, delta_fn in _ZPT_COMBINED_PARAM_DELTAS.items():
        err = cfg[f"{name}_err"]
        if err == 0.0:
            continue
        deltas = delta_fn(err, poly_fit_cutoff_min)
        shifted = _eval_zpt_piecewise_with_deltas(
            wgt_config, jet_multiplicity, nbins, dimuon_pt, f0_order, f1_order,
            poly_fit_cutoff_min, poly_fit_cutoff_max, deltas,
        )
        variance = variance + (shifted - nominal_value) ** 2

    return variance ** 0.5


def getZptWgts_3region(
    dimuon_pt,
    njets,
    nbins,
    year: str,
    config_path: str,
    NanoAODv: int,
    sigma_shift: float = 0.0,
):
    """
    Get Z pT weights based on polynomial fits in 3 regions.
    TODO: Implement the possibility to apply the zpt weights w.r.t. number of generated jets instead of reco jets.
    """
    wgt_config = _load_zpt_config_section(year, config_path, NanoAODv)
    zpt_wgt = ak.ones_like(dimuon_pt)
    jet_multiplicies = [0,1,2]

    for jet_multiplicity in jet_multiplicies:

        # Get cut-off regions between the polynomial fits
        poly_fit_cutoff_min = wgt_config[f"njet_{jet_multiplicity}"][nbins]["polynomial_range"]["xmin1"]
        poly_fit_cutoff_max = wgt_config[f"njet_{jet_multiplicity}"][nbins]["polynomial_range"]["xmax1"]

        # Get the function order
        f0_order = int(wgt_config[f"njet_{jet_multiplicity}"][nbins]["fit_orders"]["f0_order"])
        f1_order = int(wgt_config[f"njet_{jet_multiplicity}"][nbins]["fit_orders"]["f1_order"])

        nominal_value = _eval_zpt_old_quadrature(
            wgt_config, jet_multiplicity, nbins, dimuon_pt, f0_order, f1_order,
            poly_fit_cutoff_min, poly_fit_cutoff_max, sigma_shift=0.0,
        )

        if sigma_shift == 0.0:
            zpt_wgt_by_jet = nominal_value
        else:
            # 3-tier fallback, most- to least-correct, degrading only as far
            # as the payload YAML's own saved fields require:
            #   1) full covariance of the 4 combined-fit parameters (needs
            #      combined_fit_covariance -- get_polyFit.py's newest output)
            #   2) those same 4 parameters treated as independent (needs
            #      just their *_err fields -- older get_polyFit.py output)
            #   3) the original 14 per-order f0_pN/f1_pN coefficients,
            #      treated as independent (any payload YAML at all)
            envelope = _zpt_combined_param_envelope_full_cov(
                wgt_config, jet_multiplicity, nbins, dimuon_pt, f0_order, f1_order,
                poly_fit_cutoff_min, poly_fit_cutoff_max, nominal_value,
            )
            if envelope is None:
                envelope = _zpt_combined_param_envelope_diagonal(
                    wgt_config, jet_multiplicity, nbins, dimuon_pt, f0_order, f1_order,
                    poly_fit_cutoff_min, poly_fit_cutoff_max, nominal_value,
                )
            if envelope is not None:
                zpt_wgt_by_jet = nominal_value + sigma_shift * envelope
            else:
                # This YAML predates even the 4 combined-fit parameters --
                # fall back to the coarsest per-coefficient quadrature envelope.
                zpt_wgt_by_jet = _eval_zpt_old_quadrature(
                    wgt_config, jet_multiplicity, nbins, dimuon_pt, f0_order, f1_order,
                    poly_fit_cutoff_min, poly_fit_cutoff_max, sigma_shift=sigma_shift,
                )

        if jet_multiplicity != 2:
            njet_mask = njets == jet_multiplicity
        else:
            njet_mask = njets >= 2 # njet 2 is inclusive
        zpt_wgt = ak.where(njet_mask, zpt_wgt_by_jet, zpt_wgt) # if matching jet multiplicity, apply the values

    cutOff_mask = dimuon_pt < 200 # ignore wgts from dimuon pT > 200
    zpt_wgt = ak.where(cutOff_mask, zpt_wgt, ak.ones_like(dimuon_pt))
    return zpt_wgt

def _add_block(out, block):
    """This function is to update the output dictionary with another dictionary block,
    that will be in the output parquet file.
    Arguments:
        out: dict
            The output dictionary to be updated.
        block: dict
            The dictionary block to add to the output.
    Returns:
        out: dict
            The updated output dictionary.
    """
    out.update(block)
    return out


def pick_vbf_pairs(jets):
    """
    VBF tag-pair candidates under different criteria (study: docs/vbf_jet_selection.md).
    jets is the already-selected, pt-sorted Array of jets per event.
    Returns:
      "lead", "max_mjj", "max_deta", "mjj_deta": (jet1, jet2); "lead" is padded, so jet1 exists in 1-jet events
      "has_mjj_deta": some pair has dEta > 2.5
      "pairs_by_method": {method: (jet1, jet2, i1, i2)} with i1 < i2 the jets' indices in `jets` (jet1 = higher pT),
             None where no pair qualifies (written by the save_alt_vbf_pairs switch)
    """
    # need at least 2 jets to form a pair; combinations() will give empty for <2
    pairs = ak.combinations(jets, 2, fields=["j1", "j2"])
    ipairs = ak.argcombinations(jets, 2, fields=["i1", "i2"])  # same order as `pairs`, but returns the indeces of jets

    # pair metrics
    mjj = (pairs.j1 + pairs.j2).mass
    deta = np.abs(pairs.j1.eta - pairs.j2.eta)
    deta_cut = 2.5

    def best(score, cond=None):
        """Pair with the highest score among pairs passing cond, as (jet1, jet2, i1, i2)."""
        idx = ak.argmax(score if cond is None else ak.mask(score, cond), axis=1, keepdims=True)
        p, ip = pairs[idx], ipairs[idx]
        return ak.firsts(p.j1), ak.firsts(p.j2), ak.firsts(ip.i1), ak.firsts(ip.i2)

    pairs_by_method = {
        "leadpt": best(mjj, (ipairs.i1 == 0) & (ipairs.i2 == 1)),
        "maxmjj": best(mjj),
        "maxdeta": best(deta),
        "maxmjj_opphemi": best(mjj, pairs.j1.eta * pairs.j2.eta < 0),
        "leadpt_maxmjj": best(mjj, ipairs.i1 == 0),
        "maxmjj_bothpt30": best(mjj, (pairs.j1.pt > 30) & (pairs.j2.pt > 30)),
        "maxmjj_bothpt35": best(mjj, (pairs.j1.pt > 35) & (pairs.j2.pt > 35)),
    }
    # max mjj with deta > 2.5; falls back to max mjj when no pair passes
    has_pair_deta = ak.any(deta > deta_cut, axis=1)
    pairs_by_method["maxmjj_deta25"] = tuple(
        ak.where(has_pair_deta, a, b) for a, b in zip(best(mjj, deta > deta_cut), pairs_by_method["maxmjj"])
    )
    if hasattr(jets, "btagPNetQvG"):
        qsum = pairs.j1.btagPNetQvG + pairs.j2.btagPNetQvG
        pairs_by_method["maxqvg"] = best(qsum)
        pairs_by_method["maxqvg_deta25"] = best(qsum, deta > deta_cut)

    # leading-pt jets (the analysis pair)
    padded = ak.pad_none(jets, 2)

    return {
        "lead": (padded[:, 0], padded[:, 1]),
        "max_mjj": pairs_by_method["maxmjj"][:2],
        "max_deta": pairs_by_method["maxdeta"][:2],
        "mjj_deta": pairs_by_method["maxmjj_deta25"][:2],
        "has_mjj_deta": has_pair_deta,
        "pairs_by_method": pairs_by_method,
    }


def build_extended_jet_vars(jets, pair_dict, dimuon, mu1, mu2, met_phi, variation):
    """
    Event-level min/max/count variables over ALL selected jets (not only jet1/jet2),
    for VBF DNN feature studies. Empty jet sets give None (as jet1_* for 0-jet events);
    counts give 0.
    """
    out = {}
    abs_eta = np.abs(jets.eta)
    out[f"jets_max_abs_eta_{variation}"] = ak.max(abs_eta, axis=1)
    out[f"jets_min_abs_eta_{variation}"] = ak.min(abs_eta, axis=1)
    out[f"njets_absEta_gt2p5_{variation}"] = ak.sum(abs_eta > 2.5, axis=1)

    # PNetB / UParT scores are -1 outside the tracker (|eta| > 2.5): use valid scores only
    for tagger in ["btagPNetQvG", "btagPNetB", "btagUParTAK4QvG", "btagUParTAK4B"]:
        if not hasattr(jets, tagger):
            continue
        score = jets[tagger]
        score = score[score >= 0]
        out[f"jets_max_{tagger}_{variation}"] = ak.max(score, axis=1)
        out[f"jets_min_{tagger}_{variation}"] = ak.min(score, axis=1)

    puid = get_puId(jets)
    out[f"jets_max_puId_{variation}"] = ak.max(puid, axis=1)
    out[f"jets_min_puId_{variation}"] = ak.min(puid, axis=1)

    # muon-jet angles over 2 muons x all jets
    out[f"mujet_min_dR_{variation}"] = np.minimum(
        ak.min(jets.delta_r(mu1), axis=1), ak.min(jets.delta_r(mu2), axis=1)
    )
    dphi_mu1 = _delta_phi(jets.phi, mu1.phi)
    dphi_mu2 = _delta_phi(jets.phi, mu2.phi)
    out[f"mujet_min_dPhi_{variation}"] = np.minimum(ak.min(dphi_mu1, axis=1), ak.min(dphi_mu2, axis=1))
    out[f"mujet_max_dPhi_{variation}"] = np.maximum(ak.max(dphi_mu1, axis=1), ak.max(dphi_mu2, axis=1))

    mmj_dEta = np.abs(dimuon.eta - jets.eta)
    out[f"mmjall_min_dEta_{variation}"] = ak.min(mmj_dEta, axis=1)
    out[f"mmjall_max_dEta_{variation}"] = ak.max(mmj_dEta, axis=1)

    out[f"jetmet_min_dPhi_{variation}"] = ak.min(_delta_phi(jets.phi, met_phi), axis=1)

    # extra jets (beyond the two leading-pT jets): central-jet-veto style variables
    jet1, jet2 = pair_dict["lead"]
    extra = jets[:, 2:]
    y1 = getRapidity(jet1)
    y2 = getRapidity(jet2)
    zep_extra = (getRapidity(extra) - 0.5 * (y1 + y2)) / np.abs(y1 - y2)
    out[f"extrajets_min_abs_zeppenfeld_{variation}"] = ak.min(np.abs(zep_extra), axis=1)
    between = (extra.eta > np.minimum(jet1.eta, jet2.eta)) & (extra.eta < np.maximum(jet1.eta, jet2.eta))
    out[f"nextrajets_between_{variation}"] = ak.fill_none(ak.sum(between, axis=1), 0)
    out[f"extrajets_between_max_pt_{variation}"] = ak.max(extra.pt[between], axis=1)

    j1_mjj, j2_mjj = pair_dict["max_mjj"]
    out[f"jj_mass_maxAllPairs_{variation}"] = (j1_mjj + j2_mjj).mass
    return out


# Muon pT variation suffix -> shifted-pT muon field (muon variation block and derived DNN vars)
MUON_PT_VARIATION_FIELDS = {
    "mu_roccor_up": "pt_roch_up",
    "mu_roccor_down": "pt_roch_down",
    "mu_scale_up": "pt_roch_scale_up",
    "mu_scale_down": "pt_roch_scale_down",
    "mu_resol_up": "pt_roch_resol_up",
    "mu_resol_down": "pt_roch_resol_down",
}


# VBF-DNN derived features -> variation families that move them ("mu", "jet"); empty = nominal only.
# Kept in sync by hand with EXTRA_FEATURES in modules/vbf_dnn_features.py (stage-2/DNN on-the-fly path).
VBF_DNN_DERIVED_FEATURES = {
    "mm_pt_over_mass": ("mu",),
    "mm_pt_balance": ("mu",),
    "mu_pt_ratio": ("mu",),
    "mu_pt_sum": ("mu",),
    "mu_pt_asym": ("mu",),
    "mm_alpha_lab": (),
    "mmj1_dy_abs": ("mu", "jet"),
    "mmj1_cos_theta_star": ("mu", "jet"),
    "mmj1_mass": ("mu", "jet"),
    "mu1j1_dPhi": ("jet",),
    "mu2j1_dPhi": ("jet",),
    "mu1j2_dPhi": ("jet",),
    "mu2j2_dPhi": ("jet",),
    "mu2j1_dR": ("jet",),
    "njets_fwd25": ("jet",),
    "jj_qvg_max": ("jet",),
    "jj_qvg_min": ("jet",),
    "jj_btag_max": ("jet",),
    "jj_btag_min": ("jet",),
    "jj_pt_diff": ("jet",),
    "ht_jj": ("jet",),
}
VBF_DNN_DERIVED_SENTINEL = -1.0


def _fill_sentinel(x, fill=VBF_DNN_DERIVED_SENTINEL):
    """None (missing jet) and non-finite -> fill."""
    x = ak.fill_none(x, fill)
    return ak.where(np.isfinite(x), x, fill)


def build_vbf_dnn_derived_vars(dimuon, mu1, mu2, jet1, jet2, suffix, family, year):
    """
    VBF-DNN derived features (VBF_DNN_DERIVED_FEATURES) as {name}_{suffix}.
    family: "nominal" (all), "jet" (JES/JER: jet-dependent only) or "mu" (muon pT
    variation: muon-dependent only; pass the shifted mu1/mu2/dimuon and nominal jets).
    year: picks the b-score for jj_btag_* (Run 3: ParticleNet 2022-23, UParT 2024-26; Run 2: ParticleNet).
    Missing jet / non-finite -> -1; njets_fwd25 and ht_jj -> 0.
    """
    btag_score = run3_btag_score(year) if (is_run3(year) and "RERECO" not in year) else "btagPNetB"
    names = [n for n, fams in VBF_DNN_DERIVED_FEATURES.items() if family == "nominal" or family in fams]
    # float64 like compute_extra_features (NanoAOD float32 loses precision in cos_theta_star)
    dimuon, mu1, mu2, jet1, jet2 = (ak.values_astype(x, np.float64) for x in (dimuon, mu1, mu2, jet1, jet2))
    out = {}
    s = VBF_DNN_DERIVED_SENTINEL
    pt_sum = mu1.pt + mu2.pt
    for n in names:
        if n == "mm_pt_over_mass":
            v = _fill_sentinel(dimuon.pt / dimuon.mass, s)
        elif n == "mm_pt_balance":
            v = _fill_sentinel(dimuon.pt / pt_sum, s)
        elif n == "mu_pt_ratio":
            v = _fill_sentinel(mu2.pt / mu1.pt, s)
        elif n == "mu_pt_sum":
            v = _fill_sentinel(pt_sum, s)
        elif n == "mu_pt_asym":
            v = _fill_sentinel((mu1.pt - mu2.pt) / pt_sum, s)
        elif n == "mm_alpha_lab":
            cos_a = (np.cos(_delta_phi(mu1.phi, mu2.phi)) + np.sinh(mu1.eta) * np.sinh(mu2.eta)) / (
                np.cosh(mu1.eta) * np.cosh(mu2.eta)
            )
            cos_a = ak.where(cos_a > 1.0, 1.0, ak.where(cos_a < -1.0, -1.0, cos_a))
            v = _fill_sentinel(np.arccos(cos_a), s)
        elif n == "mmj1_dy_abs":
            v = _fill_sentinel(np.abs(getRapidity(dimuon) - getRapidity(jet1)), s)
        elif n == "mmj1_mass":
            v = _fill_sentinel((dimuon + jet1).mass, s)
        elif n == "mmj1_cos_theta_star":
            # |cos| of the mumu direction in the (mumu j1) rest frame vs that system's lab flight direction
            em, pxm, pym, pzm = dimuon.energy, dimuon.px, dimuon.py, dimuon.pz
            E, PX, PY, PZ = em + jet1.energy, pxm + jet1.px, pym + jet1.py, pzm + jet1.pz
            bx, by, bz = PX / E, PY / E, PZ / E
            b2 = bx**2 + by**2 + bz**2
            gam = 1.0 / np.sqrt(np.maximum(1.0 - b2, 1e-12))
            k = (gam - 1.0) * (bx * pxm + by * pym + bz * pzm) / ak.where(b2 > 0, b2, 1.0) - gam * em
            sx, sy, sz = pxm + k * bx, pym + k * by, pzm + k * bz
            cos_t = (sx * PX + sy * PY + sz * PZ) / (
                np.sqrt(sx**2 + sy**2 + sz**2) * np.sqrt(PX**2 + PY**2 + PZ**2)
            )
            v = _fill_sentinel(np.abs(cos_t), s)
        elif n in ("mu1j1_dPhi", "mu2j1_dPhi", "mu1j2_dPhi", "mu2j2_dPhi"):
            mu = mu1 if n[2] == "1" else mu2
            jet = jet1 if n[4] == "1" else jet2
            v = _fill_sentinel(_delta_phi(mu.phi, jet.phi), s)
        elif n == "mu2j1_dR":
            v = _fill_sentinel(np.sqrt((mu2.eta - jet1.eta) ** 2 + _delta_phi(mu2.phi, jet1.phi) ** 2), s)
        elif n == "ht_jj":
            v = _fill_sentinel(ak.fill_none(jet1.pt, 0.0) + ak.fill_none(jet2.pt, 0.0), 0.0)
        elif n == "njets_fwd25":
            fwd1 = ak.fill_none(np.abs(jet1.eta) > 2.5, False)
            fwd2 = ak.fill_none(np.abs(jet2.eta) > 2.5, False)
            v = ak.values_astype(fwd1, np.float64) + ak.values_astype(fwd2, np.float64)
        elif n.startswith(("jj_qvg", "jj_btag")):
            tag = "btagPNetQvG" if "qvg" in n else btag_score
            if not (hasattr(jet1, tag) and hasattr(jet2, tag)):
                continue  # tagger absent in this NanoAOD (e.g. Run 2)
            # tagger scores are -1 where undefined: only valid scores enter max/min
            t1 = ak.where(jet1[tag] >= 0, jet1[tag], np.nan)
            t2 = ak.where(jet2[tag] >= 0, jet2[tag], np.nan)
            both = np.maximum(t1, t2) if n.endswith("max") else np.minimum(t1, t2)
            v = ak.where(np.isnan(t1), t2, ak.where(np.isnan(t2), t1, both))
            v = _fill_sentinel(v, s)  # no jet2 or both invalid -> sentinel
        elif n == "jj_pt_diff":
            v = _fill_sentinel(jet1.pt - jet2.pt, s)
        else:
            raise KeyError(f"[build_vbf_dnn_derived_vars] no stage-1 definition for {n}")
        out[f"{n}_{suffix}"] = v
    return out


def apply_ECALBadCalib_EventFilter_recipe(
    events,
    base_mask,
    *,
    is_mc: bool,
    run_min: int = 362433,
    run_max: int = 367144,
    met_pt_min: float = 100.0,
    jet_pt_min: float = 50.0,
    eta_min: float = -0.5,
    eta_max: float = -0.1,
    phi_min: float = -2.1,
    phi_max: float = -1.8,
    emef_min: float = 0.9,
    dphi_min: float = 2.9,
):
    """
    Reference: https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2#ECal_BadCalibration_Filter_Flag

    The NanoAOD does not have enough info to rerun the filter. So please apply the following recipe:
    Reject the event if PuppiMET_pt > 100 GeV and there is at least one jet (AK4) which has
        pT > 50 GeV,
        eta within -0.5 to -0.1,
        phi within -2.1 to -1.8,
        Neutral EM energy fraction or charged EM energy fraction (branch names: Jet_neEmEF, Jet_chEmEF) > 0.9
        Δɸ(PuppiMET _phi, jet) > 2.9
    Apply it only for RunNumbers in the range 362433 to 367144 which belong to later part of 2022 and early 2023.
    DO NOT apply jet ID (branch: Jet_jetId) on the jets while implementing this recipe.
    The effect of this recipe on good events is very small (<0.2%) and it is not simulated in MC. So, the recipe is not recommended for MC.

    Parameters
    ----------
    events : coffea NanoEvents
    base_mask : ak.Array[bool]
        Existing event-quality mask to be updated.
    is_mc : bool
        Whether the sample is MC.
    Returns
    -------
    ak.Array[bool]
        Updated mask with the additional rejection applied (data only).
    """
    # no-op on MC by prescription
    if is_mc:
        return base_mask

    # defensive checks
    if not (hasattr(events, "PuppiMET") and hasattr(events, "Jet") and hasattr(events, "run")):
        logger.warning("Skipping PuppiMET-jet horn recipe: missing PuppiMET/Jet/run branches.")
        return base_mask

    run = events.run
    in_run_range = (run >= run_min) & (run <= run_max)

    met_pt = events.PuppiMET.pt
    met_phi = events.PuppiMET.phi

    jets = events.Jet  # NOTE: its recommended to not apply the Jet_jetId or any other correction
    jet_pt = jets.pt
    jet_eta = jets.eta
    jet_phi = jets.phi

    # EM fractions
    try:
        jet_neEmEF = jets.neEmEF
        jet_chEmEF = jets.chEmEF
    except Exception as e:
        logger.warning("Skipping PuppiMET-jet horn recipe: Jet.neEmEF / Jet.chEmEF not found (%r).", e)
        return base_mask

    jet_region = (
        (jet_pt > jet_pt_min)
        & (jet_eta >= eta_min) & (jet_eta <= eta_max)
        & (jet_phi >= phi_min) & (jet_phi <= phi_max)
        & ((jet_neEmEF > emef_min) | (jet_chEmEF > emef_min))
    )

    dphi_met_jet = _delta_phi(met_phi, jet_phi)
    jet_region = jet_region & (dphi_met_jet > dphi_min)

    has_bad_jet = ak.any(jet_region, axis=1)
    reject = in_run_range & (met_pt > met_pt_min) & has_bad_jet

    return base_mask & (~reject)

class EventProcessor(processor.ProcessorABC):
    def __init__(self, config: dict, test_mode=False, isCutflow=False, **kwargs):
        self.config = config
        self.isCutflow = isCutflow
        self.test_mode = test_mode

        # Initialize PackedSelection
        # Reference: https://nbviewer.org/github/scikit-hep/coffea/blob/master/binder/packedselection.ipynb
        self.selection = {}
        self.cutflow = {}
        self.cutflow_names = []

        self.pysr_configs = {}
        self.pysr_all_features = set()
        if self.config["switches"].get("do_use_pySR_score", False):
            pysr_base_dir = self.config.get("pysr_config_dir", "configs/pysr_versions/run_multi_bkg_07May_v2")
            self.pysr_configs = load_pysr_configs(pysr_base_dir)
            if not self.pysr_configs:
                raise FileNotFoundError(
                    f"PySR scoring is enabled, but no config JSON files were found in {pysr_base_dir}."
                )
            for region, cfg in self.pysr_configs.items():
                cfg["_compiled_equation"] = compile(
                    cfg["equation"], f"<pysr_{region}>", "eval"
                )
                self.pysr_all_features.update(cfg["features"])

        self.pu_dnn_configs = {}
        if self.config["switches"].get("do_use_pu_dnn_score", False):
            if self.config["switches"].get("do_use_pySR_score", False):
                raise ValueError(
                    "do_use_pySR_score and do_use_pu_dnn_score are alternative "
                    "forward-jet PU cleanup models; only one can be enabled at once."
                )
            # do_use_pu_dnn_score enabled must have an explicit model dir.
            if "pu_dnn_model_dir" not in self.config:
                raise KeyError(
                    "do_use_pu_dnn_score is enabled for this year, but no "
                    "'pu_dnn_model_dir' entry was found in the config. Add one "
                    "for this year in configs/parameters/SF_filelist.yaml."
                )
            pu_dnn_base_dir = self.config["pu_dnn_model_dir"]
            self.pu_dnn_configs = load_pu_dnn_configs(pu_dnn_base_dir)
            if not self.pu_dnn_configs:
                raise FileNotFoundError(
                    f"PU DNN scoring is enabled, but no region models were found under {pu_dnn_base_dir}."
                )

    def _jet_veto_map_mask(self, jets):
        """
        Jagged jet-veto-map value per jet (0 = not vetoed), from the correctionlib file in
        config["jet_veto_maps"] / tag config["jet_veto_maps_tag"]. Jets failing the official JME
        minimal selection (https://cms-jme-jerc.docs.cern.ch/recommendations/jet-veto-maps/#application:
        pT > 15 GeV, tightLepVeto jet ID, chEmEF + neEmEF < 0.9) are forced to 0, so a soft,
        loose-ID or high-EM-fraction jet in a vetoed region never counts.
        """
        jet_veto_maps_path = self.config.get("jet_veto_maps", None)
        logger.debug(f"jet_veto_maps_path: {jet_veto_maps_path}")
        if jet_veto_maps_path is None:
            logger.error("Jet veto maps path is not specified in the config!")
            raise ValueError("Jet veto maps path is not specified in the config!")

        cset = get_corrset(jet_veto_maps_path)
        jetVetoMapTag = self.config.get("jet_veto_maps_tag", None)
        logger.debug(f"Jet veto map tag from config: {jetVetoMapTag}")
        jet_veto_map = cset[jetVetoMapTag]

        year = self.config["year"]
        min_sel = (
            (jets.pt > 15)
            & jet_id(jets, self.config, year=year, jet_id_key="jet_veto_map_jet_id")
            & ((jets.chEmEF + jets.neEmEF) < 0.9)
        )
        # jets failing min_sel are looked up at (0, 0): never outside the map range; result forced to 0 below
        # correctionlib's evaluate returns a FLAT array: flatten, evaluate, unflatten to the jet structure
        flat_inputs = {
            "type": "jetvetomap",
            "eta": ak.flatten(ak.where(min_sel, jets.eta, 0.0)),
            "phi": ak.flatten(ak.where(min_sel, jets.phi, 0.0)),
        }
        inputs = [flat_inputs[input.name] for input in jet_veto_map.inputs]
        jet_veto_mask = ak.unflatten(jet_veto_map.evaluate(*inputs), ak.num(jets, axis=1))
        return ak.where(min_sel, jet_veto_mask, 0.0)

    def compute_jet_veto_eventfilter(self, events, jets):
        """Jet veto maps, event mode ("do_jet_veto_maps_filterEvents"): True if any jet is in a vetoed region."""
        return ak.any(self._jet_veto_map_mask(jets) != 0.0, axis=1)

    def compute_jet_veto_jetfilter(self, events, jets, PuppiMET):
        """
        Jet veto maps, jet mode ("do_jet_veto_maps_filterJets"): remove jets in vetoed regions and
        set PuppiMET pt/sumEt to zero in events with such a jet.
        """
        jet_veto_mask = self._jet_veto_map_mask(jets)

        jet_veto_eventFilter = ak.any(jet_veto_mask != 0.0, axis=1)
        jets = jets[jet_veto_mask != 100.0]

        # when jet_veto_eventFilter is True, set PuppiMET pt to zero:
        met_cond = (jet_veto_eventFilter == True)

        # fetch original  PuppiMET pt, phi, sumEt
        # NOTE: Don't reset PuppiMET.phi otherwise we will see a peak at zero in PuppiMET.phi distribution
        puppi_met_pt = PuppiMET.pt
        puppi_met_sumEt = PuppiMET.sumEt

        # Obtain new PuppiMET pt, phi, sumEt - set to zero when met_cond is True
        puppi_met_pt_new = ak.where(met_cond, ak.zeros_like(puppi_met_pt), puppi_met_pt)
        puppi_met_sumEt_new = ak.where(met_cond, ak.zeros_like(puppi_met_sumEt), puppi_met_sumEt)

        # overwrite the PuppiMET variables
        PuppiMET["pt"] = puppi_met_pt_new
        PuppiMET["sumEt"] = puppi_met_sumEt_new

        return jets, PuppiMET

    def process(self, events: coffea_nanoevent, dataset_yaml_file: str):
        t0 = time.perf_counter()
        year = self.config["year"]

        # ReInitialize PackedSelection, otherwise processor would merge selection from previous run
        self.selection = PackedSelection()

        event_filter = ak.ones_like(events.event, dtype="bool") # 1D boolean array to be used to filter out bad events
        self.processed_event_count = ak.num(events, axis=0) # For METADATA of event count
        # Debugging: Check structure of event_filter
        logger.debug(f"event_filter type: {type(event_filter)}")
        logger.debug(f"event_filter length: {len(event_filter)}")
        logger.debug(f"events length: {len(events)}")

        # Ensure event_filter matches the structure of events
        if len(event_filter) != len(events):
            raise ValueError("event_filter length does not match events length!")

        self.selection.add("TotalEntries", event_filter)

        dataset = events.metadata['dataset']
        logger.debug(f"Dataset going to read: {dataset}")
        logger.debug(f"events.metadata: {events.metadata}")
        NanoAODv = events.metadata['NanoAODv']
        self.config['NanoAODv'] = NanoAODv
        is_mc = events.metadata['is_mc']
        logger.debug(f"NanoAODv: {NanoAODv}")

        t1 = time.perf_counter()
        logger.debug(f"[timing] Metadata read time: {t1 - t0:.2f} seconds")

        # ------------------------------------------------------------#
        # Step-1: Apply the lumi mask for data only
        # ------------------------------------------------------------#
        lumi_mask = ak.ones_like(event_filter, dtype="bool")
        if not is_mc:
            logger.debug(f'self.config["lumimask"]: {self.config["lumimask"]}')
            lumi_info = LumiMask(self.config["lumimask"])
            lumi_mask = lumi_info(events.run, events.luminosityBlock)
        self.selection.add("lumi_mask", lumi_mask)

        # ------------------------------------------------------------#
        # Step-2: Apply LHE cut to remove events with dilepton mass between 100 and 200 GeV for DY_M-50 sample
        # ------------------------------------------------------------#
        is_inclusive_dy_sample = dataset.startswith("dy") and "M-50" in dataset
        if is_inclusive_dy_sample and self.config["switches"]["do_remove_dy_M100to200"]:
            # INFO: For run-2, for higher statistics, we are stiching DY_M-50 and DY_M-100to200 samples together.
            #            As the DY_M-50 sample is the inclusive sample, we need to remove the events in DY_M-50 that have
            #            dilepton mass between 100 and 200 GeV, to avoid double counting with DY_M-100to200 sample.
            # FIXME: currently, `dy_M-50` is hardcoded

            logger.debug("doing dy_M-50 LHE cut!")
            LHE_particles = events.LHEPart #has unique pdgIDs of [ 1,  2,  3,  4,  5, 11, 13, 15, 21]
            bool_filter = (abs(LHE_particles.pdgId) == 11) | (abs(LHE_particles.pdgId) == 13) | (abs(LHE_particles.pdgId) == 15)
            LHE_leptons = LHE_particles[bool_filter]

            """
            TODO: maybe we can get faster by just indexing first and second, instead of argmax and argmins
            When I had a quick look, all LHE_leptons had either two or zero leptons per event, never one,
            so just indexing first and second could work
            """
            max_idxs = ak.argmax(LHE_leptons.pdgId , axis=1,keepdims=True) # get idx for normal lepton
            min_idxs = ak.argmin(LHE_leptons.pdgId , axis=1,keepdims=True) # get idx for anti lepton
            LHE_lepton_barless = LHE_leptons[max_idxs]
            LHE_lepton_bar = LHE_leptons[min_idxs]
            LHE_dilepton_mass =  (LHE_lepton_barless +LHE_lepton_bar).mass

            LHE_filter = (((LHE_dilepton_mass > 100) & (LHE_dilepton_mass < 200)))[:,0]
            LHE_filter = ak.fill_none(LHE_filter, value=False)
            LHE_filter = (LHE_filter== False) # we want True to indicate that we want to keep the event
            event_filter = event_filter & LHE_filter

            self.selection.add("LHE_cut", LHE_filter)

        t3 = time.perf_counter()
        logger.debug(f"[timing] LHE cut time: {t3 - t1:.2f} seconds")
        # ------------------------------------------------------------#

        # ------------------------------------------------------------#
        # Step-3: Apply HLT
        # ------------------------------------------------------------#
        # Apply HLT to both Data and MC.
        # NOTE: this would probably be superfluous if you already do trigger matching
        HLT_filter = ak.zeros_like(event_filter, dtype="bool")  # start with 1D of Falses
        for HLT_str in self.config["hlt"]:
            logger.debug(f"HLT_str: {HLT_str}")
            HLT_filter = HLT_filter | ak.fill_none(events.HLT[HLT_str], value=False)
        self.selection.add("HLT_filter", HLT_filter)
        event_filter = event_filter & HLT_filter

        t4 = time.perf_counter()
        logger.debug(f"[timing] HLT and lumi mask time: {t4 - t3:.2f} seconds")
        # ------------------------------------------------------------#

        # --------------------------------------------------------#
        # Step-4: Obtain the pileup weights
        # --------------------------------------------------------#
        do_pu_wgt = self.config["switches"]["do_pu_wgt"]
        if do_pu_wgt:
            # obtain PU reweighting b4 event filtering, and apply it after we finalize event_filter
            logger.debug(f"year: {year}")
            if is_run3(year):
                run_campaign = 3
            elif is_run2(year):
                run_campaign = 2
            else:
                raise ValueError(f"Year {year} is neither Run2 nor Run3!")
            logger.debug(f"run_campaign: {run_campaign}")
            if is_mc:
                logger.debug("doing PU re-wgt!")
                pu_wgts = pu_evaluator(
                            self.config,
                            events.Pileup.nTrueInt,
                            onTheSpot=False, # False
                            Run = run_campaign,
                            is_rereco = ("RERECO" in year),
                    )

        # --------------------------------------------------------#
        # INFO: Select muons that pass pT, eta, isolation cuts,
        #            muon ID and quality flags
        #           Select events with 2 good muons, no electrons,
        #           passing quality cuts and at least one good PV
        # --------------------------------------------------------#

        # --------------------------------------------------------#
        # Step-5: Apply the event quality flags (also known as MET filters)
        # --------------------------------------------------------#
        evnt_qual_flg_selection = ak.ones_like(event_filter, dtype="bool")
        logger.debug("Applying event quality (MET-filter) flags")
        for evt_qual_flg in self.config["event_flags"]:
            logger.debug(f"evt_qual_flg: {evt_qual_flg}")
            evnt_qual_flg_selection = evnt_qual_flg_selection & events.Flag[evt_qual_flg]

        evnt_qual_flg_selection = apply_ECALBadCalib_EventFilter_recipe(events, evnt_qual_flg_selection, is_mc=is_mc)
        self.selection.add("event_quality_flags", evnt_qual_flg_selection)

        # --------------------------------------------------------
        # Step-6: Fetch the BSC corrected muon pT and pT error.
        #              If BS constrained muon variables are present.
        # --------------------------------------------------------
        doing_BS_correction = self.config["switches"]["do_beamConstraint"]
        if self.config["switches"]["do_beamConstraint"] and ("bsConstrainedChi2" in events.Muon.fields): # beamConstraint overrides geofit
            logger.debug("doing beam constraint!")
            BSConstraint_mask = (
                (events.Muon.bsConstrainedChi2 <30) # NOTE: Hardcoded chi2 cut for beam constraint
            )
            BSConstraint_mask = ak.fill_none(BSConstraint_mask, False)
            events["Muon", "pt"] = ak.where(BSConstraint_mask, events.Muon.bsConstrainedPt, events.Muon.pt)
            events["Muon", "ptErr"] = ak.where(BSConstraint_mask, events.Muon.bsConstrainedPtErr, events.Muon.ptErr)
            events["Muon", "pt_is_bsc"] = BSConstraint_mask
        else:
            events["Muon", "pt_is_bsc"] = ak.zeros_like(events.Muon.pt, dtype=bool)

        # Save raw variables before computing any corrections
        # rochester corrects pt only, but fsr_recovery changes all vals below
        events["Muon", "pt_raw"] = events.Muon.pt
        events["Muon", "eta_raw"] = events.Muon.eta
        events["Muon", "phi_raw"] = events.Muon.phi
        events["Muon", "pfRelIso04_all_raw"] = events.Muon.pfRelIso04_all

        # --------------------------------------------------------
        # Step-7: Apply Rochester correction to muon pT
        # --------------------------------------------------------
        if self.config["switches"]["do_roccor"]:
            # TODO make more elegant distinction between Run2 and Run3
            if is_run2(year):
                # Fixme: corrected and non-corrected BS should be also added to Run-2
                logger.debug("doing Run2 rochester!")
                apply_roccor(events, self.config["roccor_file"], is_mc)
            elif is_run3(year):
                logger.debug("doing Run3 KIT muon Scale Resolution!")
                roccor_nonbs = self.config.get("roccor_file_nonBS", self.config["roccor_file"])
                if roccor_nonbs != self.config["roccor_file"]:
                    # VXBS ScaRe for beam-spot-constrained pT, regular ScaRe for muons that keep NanoAOD pT
                    apply_KitMuScaleRe_Run3_bs_split(
                        events, self.config["roccor_file"], roccor_nonbs, events.Muon.pt_is_bsc, is_mc)
                else:
                    apply_KitMuScaleRe_Run3(events, self.config["roccor_file"], is_mc)
            else:
                raise ValueError(f"Year {year} is neither Run2 nor Run3!")
            events["Muon", "pt"] = events.Muon.pt_roch
        else:
            events["Muon", "pt_roch"] = events.Muon.pt

        # Optional lepton/IP veto (switch do_lepton_ip_veto, default off = official selection):
        # muon IP cuts, exactly-2-loose-muons veto, and an iso+IP-qualified electron veto down to
        # electron_pt_cut_lepton_ip_veto. Cut values: configs/parameters/{muon,electron}.yaml.
        do_lepton_ip_veto = self.config["switches"].get("do_lepton_ip_veto", False)
        # do_muon_ip_cut: muon IP cuts alone (no extra lepton vetoes), to separate the two effects
        do_muon_ip_cut = do_lepton_ip_veto or self.config["switches"].get("do_muon_ip_cut", False)
        muon_selection = (
            (events.Muon.pt_raw > self.config["muon_pt_cut"]) # pt_raw is pt b4 rochester #FIXME: Why pt_raw
            & (abs(events.Muon.eta_raw) < self.config["muon_eta_cut"])
            & events.Muon[self.config["muon_id"]]
            & (events.Muon.isGlobal | events.Muon.isTracker) # Table 3.5  AN-19-124
        )
        if do_muon_ip_cut:
            muon_ip = (abs(events.Muon.dxy) < self.config["muon_dxy_cut"]) & (abs(events.Muon.dz) < self.config["muon_dz_cut"])
            muon_selection = muon_selection & muon_ip # AN-19-124 baseline IP cuts


        # muon_sel_corrected_kinematics switch (Run 3; Round-1 for 2024): selection, loose veto, trigger match,
        # jet cleaning and SFs use ScaRe+FSR muon kinematics instead of the raw (BSC) ones
        use_corr_mu = self.config["switches"]["muon_sel_corrected_kinematics"]
        if use_corr_mu and is_run2(year):  # Run-2 SFs (add_muon_sfs_correctionlib) stay on raw kinematics
            raise ValueError(f"muon_sel_corrected_kinematics is Run-3 only, but true for {year}")
        if not use_corr_mu:  # else filled after FSR, on the corrected kinematics
            self.selection.add("muon_pT_roch", ak.any(events.Muon.pt_roch >= self.config["muon_pt_cut"], axis=1))
            self.selection.add("muon_eta", ak.any(abs(events.Muon.eta_raw) <= self.config["muon_eta_cut"], axis=1))
            self.selection.add("muon_id", ak.any(events.Muon[self.config["muon_id"]], axis=1))
            self.selection.add("muon_isGlobal_or_Tracker", ak.any(events.Muon.isGlobal | events.Muon.isTracker, axis=1))
            if do_muon_ip_cut:
                self.selection.add("muon_ip", ak.any(muon_ip, axis=1))
            self.selection.add("muon_selection", ak.any(muon_selection, axis=1))

        # calculate FSR recovery, but don't apply it until trigger matching is done
        # but apply muon iso overwrite, so base muon selection could be done
        do_fsr = self.config["switches"]["do_fsr"]
        if do_fsr:
            logger.debug("doing fsr!")
            fsr_recoveryV1(events)  # adds pt_fsr/iso_fsr fields to events.Muon
            events["Muon", "pfRelIso04_all"] = events.Muon.iso_fsr

        # pT-dependent ID/iso WP (muon_id_pt_split > 0, 2024-26: Tight ID + iso < 0.15 below 30 GeV), cut on corrected pT
        mu_id_pt_split = self.config["muon_id_pt_split"]
        if mu_id_pt_split > 0 and not use_corr_mu:
            raise ValueError(f"muon_id_pt_split > 0 needs muon_sel_corrected_kinematics, false for {year}")
        if use_corr_mu:
            # pT/eta cuts on ScaRe+FSR kinematics (FSR is applied to Muon.pt only at trigger matching)
            mu_pt_sel = events.Muon.pt_fsr if do_fsr else events.Muon.pt
            mu_eta_sel = events.Muon.eta_fsr if do_fsr else events.Muon.eta
            mu_lowpt = mu_pt_sel < mu_id_pt_split
            mu_pass_id = ak.where(mu_lowpt, events.Muon[self.config["muon_id_lowpt"]], events.Muon[self.config["muon_id"]])
            mu_iso_cut = ak.where(mu_lowpt, self.config["muon_iso_cut_lowpt"], self.config["muon_iso_cut"])
            muon_selection = (
                (mu_pt_sel > self.config["muon_pt_cut"])
                & (abs(mu_eta_sel) < self.config["muon_eta_cut"])
                & mu_pass_id
                & (events.Muon.isGlobal | events.Muon.isTracker)
            )
            if do_muon_ip_cut:
                muon_selection = muon_selection & muon_ip
            muon_selection = ak.fill_none(muon_selection, False)
            self.selection.add("muon_pT_roch", ak.any(mu_pt_sel > self.config["muon_pt_cut"], axis=1))
            self.selection.add("muon_eta", ak.any(abs(mu_eta_sel) < self.config["muon_eta_cut"], axis=1))
            self.selection.add("muon_id", ak.any(mu_pass_id, axis=1))
            self.selection.add("muon_isGlobal_or_Tracker", ak.any(events.Muon.isGlobal | events.Muon.isTracker, axis=1))
            if do_muon_ip_cut:
                self.selection.add("muon_ip", ak.any(muon_ip, axis=1))
            self.selection.add("muon_selection", ak.any(muon_selection, axis=1))
        else:
            mu_pt_sel = events.Muon.pt_raw
            mu_eta_sel = events.Muon.eta_raw
            mu_iso_cut = self.config["muon_iso_cut"]

        # apply iso portion of base muon selection, now that possible FSR photons are integrated into pfRelIso04_all as specified in line 360 of AN-19-124
        mu_pass_iso = events.Muon.pfRelIso04_all < mu_iso_cut
        muon_selection = muon_selection & mu_pass_iso
        self.selection.add("muon_iso", ak.any(mu_pass_iso, axis=1))
        if do_lepton_ip_veto:
            # Extra loose-muon veto: exactly 2 loose muons (rejects >2); selected muons are a subset of loose,
            # so with nmuons>=2 this leaves exactly the 2 selected muons
            loose_muon = (  # same pT/eta definition as the selected muons (raw, or ScaRe+FSR)
                (mu_pt_sel > self.config["loose_muon_pt_cut"])
                & (abs(mu_eta_sel) < self.config["muon_eta_cut"])
                & events.Muon[self.config["loose_muon_id"]]
                & (events.Muon.pfRelIso04_all < self.config["loose_muon_iso_cut"])
                & muon_ip
            )
            loose_muon_veto = (ak.sum(loose_muon, axis=1) == 2)


        t5 = time.perf_counter()
        logger.debug(f"[timing] Muon selection time: {t5 - t4:.2f} seconds")
        # --------------------------------------------------------
        # apply tirgger match after base muon selection and Rochester correction, but b4 FSR recovery as implied in line 373 of AN-19-124
        # use_corr_mu (Round-1): match ScaRe+FSR muons instead (offline pT, eta, phi)
        if use_corr_mu and do_fsr:
            events["Muon", "pt"] = events.Muon.pt_fsr
            events["Muon", "eta"] = events.Muon.eta_fsr
            events["Muon", "phi"] = events.Muon.phi_fsr
        trig_pt_field = "pt" if use_corr_mu else "pt_roch"
        if self.config["switches"]["do_trigger_match"]:
            do_seperate_mu1_leading_pt_cut = False
            logger.debug("doing trigger match!")
            """
            Apply trigger matching. We take the two leading pT reco muons and try to have at least one of the muons
            to be matched with the trigger object that fired our HLT. If none of the muons did it, then we reject the
            event. This operation is computationally expensive, so perhaps worth considering not implementing it if
            it has neglible impact
            reference: https://cms-nanoaod-integration.web.cern.ch/autoDoc/NanoAODv9/2018UL/doc_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1.html

            TODO: The impact this operation has onto the statistics is supposedly very low, but I have to check that
            """

            mu_id = 13
            pass_id = abs(events.TrigObj.id) == mu_id
            pass_filterbit = (events.TrigObj.filterBits & 8) > 0
            trigger_cands = events.TrigObj[pass_id & pass_filterbit]

            dr_threshold = self.config["muon_trigmatch_dr"]
            logger.debug(f"dr_threshold: {dr_threshold}")

            # at least one of the two leading muons must match a trigger object and pass the leading-pT cut
            muon_matches = []
            for mu in _leading_two(events.Muon[muon_selection]):
                # None comes from the muon padding, not trigger_cands, so False is correct
                dr_match = ak.fill_none(ak.sum(mu.delta_r(trigger_cands) < dr_threshold, axis=1) > 0, value=False)
                leading_pt_match = ak.fill_none(mu[trig_pt_field] >= self.config["muon_leading_pt"], value=False)
                muon_matches.append(dr_match & leading_pt_match)
            trigger_match = muon_matches[0] | muon_matches[1]
            event_filter = event_filter & trigger_match

            self.selection.add("trigger_match", trigger_match)
        else:
            do_seperate_mu1_leading_pt_cut = True
            logger.warning("NO trigger match! Doing leading mu pass instead!")

        t6 = time.perf_counter()
        logger.debug(f"[timing] Trigger match time: {t6 - t5:.2f} seconds")
        # --------------------------------------------------------


        # apply FSR correction, since trigger match is calculated
        if do_fsr:
            events["Muon", "pt"] = events.Muon.pt_fsr
            events["Muon", "eta"] = events.Muon.eta_fsr
            events["Muon", "phi"] = events.Muon.phi_fsr
        else:
            # if no fsr, just copy 'pt' to 'pt_fsr'
            events["Muon", "pt_fsr"] = events.Muon.pt

        t6a = time.perf_counter()
        logger.debug(f"[timing] FSR correction time: {t6a - t6:.2f} seconds")
        # -----------------------------------------------------------------

        muons = events.Muon[muon_selection]
        t6c = time.perf_counter()
        logger.debug(f"[timing] Muon selection time: {t6c - t6a:.2f} seconds")

        # do the separate mu1 leading pt cut that copperheadV1 does instead of trigger matching
        if do_seperate_mu1_leading_pt_cut:
            # cut on raw pt (sorting by corrected pt gives fewer events); ScaRe+FSR pt when use_corr_mu
            lead_pt_field = "pt" if use_corr_mu else "pt_raw"
            mu1, _ = _leading_two(muons, key=lead_pt_field)
            pass_leading_pt = ak.fill_none((mu1[lead_pt_field] > self.config["muon_leading_pt"]), value=False)
            event_filter = event_filter & pass_leading_pt

            self.selection.add("leading_muon_pt", pass_leading_pt)

        t6d = time.perf_counter()
        logger.debug(f"[timing] Separate leading muon pT cut time: {t6d - t6c:.2f} seconds")
        # count muons that pass the muon selection
        nmuons = ak.num(muons, axis=1)
        t6e = time.perf_counter()
        logger.debug(f"[timing] Count muons time: {t6e - t6d:.2f} seconds")

        # Find opposite-sign muons
        mm_charge = ak.prod(muons.charge, axis=1) # techinally not a product of two leading pT muon charge, but (nmuons==2) cut ensures that there's only two muons

        t7 = time.perf_counter()
        logger.debug(f"[timing] diMuon selection time: {t7 - t6:.2f} seconds")
        # --------------------------------------------------------#
        if NanoAODv == 9:
            electron_id = self.config["electron_id_v9"]
        elif NanoAODv == 12 or NanoAODv == 15:
            # for electron_id NanoAODv should be 12 for both 12 and 15.
            electron_id = self.config["electron_id_v12"]
        else:
            logger.error(f"Unsupported NanoAODv: {NanoAODv}")
            raise ValueError(f"Unsupported NanoAODv: {NanoAODv}")
        logger.debug(f"electron_id: {electron_id}")
        # Veto events with good quality electrons; VBF and ggH categories need zero electrons
        ecal_gap = (1.44 < abs(events.Electron.eta)) & (1.57 > abs(events.Electron.eta)) # Source: line 460 of https://cms.cern.ch/iCMS/analysisadmin/cadilines?id=1973&ancode=EGM-17-001&tp=an&line=EGM-17-001
        electron_pt_cut = self.config["electron_pt_cut_lepton_ip_veto"] if do_lepton_ip_veto else self.config["electron_pt_cut"]
        electron_selection = (
            (events.Electron.pt > electron_pt_cut)
            & (abs(events.Electron.eta) < self.config["electron_eta_cut"])
            & events.Electron[electron_id]
            & ~ecal_gap # reject electrons in ecal gap region, as specified in table 3.5 of AN-19-124
        )
        if do_lepton_ip_veto:
            # EGM-style IP cuts, split barrel/endcap on supercluster eta
            ele_is_barrel = abs(events.Electron.eta + events.Electron.deltaEtaSC) < 1.479
            electron_ip = ak.where(
                ele_is_barrel,
                (abs(events.Electron.dxy) < self.config["electron_dxy_cut_barrel"]) & (abs(events.Electron.dz) < self.config["electron_dz_cut_barrel"]),
                (abs(events.Electron.dxy) < self.config["electron_dxy_cut_endcap"]) & (abs(events.Electron.dz) < self.config["electron_dz_cut_endcap"]),
            )
            electron_selection = (
                electron_selection
                & (events.Electron.pfRelIso03_all < self.config["electron_iso_cut"])
                & electron_ip
            )
        nelectrons = ak.sum(electron_selection, axis=1)
        electron_veto = (nelectrons == 0)
        self.selection.add("electron_veto", electron_veto)
        if self.config["switches"]["do_HemVeto"]:
            HemVeto_filter, is_HemRegion = applyHemVeto(events.Jet, events.run, events.event, self.config, is_mc, NanoAODv)
            if (not self.config["switches"]["do_HemVetoStudy"]): # when we are calculating HemVeto fraction for MC, we shouldn't filter out hem veto events
                logger.debug("adding HemVeto!")
                event_filter = event_filter & HemVeto_filter
        else:
            HemVeto_filter = ak.ones_like(event_filter, dtype="bool")
            is_HemRegion = ak.ones_like(event_filter, dtype="bool")

        self.selection.add("HemVeto", HemVeto_filter == True)
        event_filter = (
                event_filter
                & lumi_mask
                & (evnt_qual_flg_selection > 0)
                & (events.PV.npvsGood > 0) # number of good primary vertex cut
        )

        pv_good = (events.PV.npvsGood > 0)
        self.selection.add("PV_npvsGood", pv_good)
        if do_lepton_ip_veto:
            # >=2 only: the loose-muon veto below is the single multiplicity cut (selected muons are a subset of loose)
            event_filter = event_filter & (nmuons >= 2)
            self.selection.add("nMuons_ge2", nmuons >= 2)
            event_filter = event_filter & loose_muon_veto
            self.selection.add("nLooseMuons_eq2", loose_muon_veto)
        else:
            event_filter = event_filter & (nmuons == 2)
            self.selection.add("nmuons", nmuons==2)

        event_filter = event_filter & (mm_charge == -1)
        self.selection.add("mm_charge", mm_charge==-1)
        event_filter = event_filter & electron_veto

        t8 = time.perf_counter()
        logger.debug(f"[timing] Electron selection filtering time: {t8 - t7:.2f} seconds")

        # calculate sum of gen weight b4 skimming off bad events
        if is_mc:
            if self.test_mode: # for small files local testing
                sumWeights = ak.sum(events.genWeight, axis=0) # for testing
            else:
                sumWeights = events.metadata['sumGenWgts']
                logger.debug(f"sumWeights: {(sumWeights)}")
        if self.config["switches"].get("do_jet_veto_maps_filterEvents", False):
            logger.debug("Applying jet veto maps!")
            jets_for_veto = events.Jet
            jet_veto_eventFilter = self.compute_jet_veto_eventfilter(events, jets_for_veto)
            event_filter = event_filter & ~jet_veto_eventFilter

            keep_after_jet_veto = ~jet_veto_eventFilter
            self.selection.add("jet_veto_maps", keep_after_jet_veto)

        # Below patch is to define dimuon mass for the cutflow before we filter out bad events
        # ------------------- Cutflow dimuon mass window: START -----------------------
        mu1_for_mass, mu2_for_mass = _leading_two(muons)
        dimuon_for_mass = mu1_for_mass + mu2_for_mass
        dimuon_mass_for_cutflow = ak.fill_none(dimuon_for_mass.mass, 0.0)
        dimuon_mass_window_cut = ( (dimuon_mass_for_cutflow > 76.0) & (dimuon_mass_for_cutflow < 106.0) )
        self.selection.add("dimuon_mass_window_76_106", dimuon_mass_window_cut)

        h_peak, h_sidebands1 = _higgs_peak_and_sidebands(dimuon_mass_for_cutflow)
        h_sidebands2 =  ((dimuon_mass_for_cutflow >= 106.0) & (dimuon_mass_for_cutflow < 115.0)) | ((dimuon_mass_for_cutflow >= 135.0) & (dimuon_mass_for_cutflow < 150.0))

        self.selection.add("h_peak_115_135", h_peak)
        self.selection.add("h_sidebands_110_115_135_150", h_sidebands1)
        self.selection.add("h_sidebands_106_115_135_150", h_sidebands2)
        # ------------------- Cutflow dimuon mass window: END -----------------------

        events = events[event_filter == True]
        muons = muons[event_filter == True]
        nmuons = ak.to_packed(nmuons[event_filter == True])

        if is_mc and do_pu_wgt:
            for variation in pu_wgts.keys():
                pu_wgts[variation] = ak.to_packed(pu_wgts[variation][event_filter == True])

        t9 = time.perf_counter()
        logger.debug(f"[timing] GEN weight and PU time: {t9 - t8:.2f} seconds")

        # --------------------------------------------------------#
        # Fill dimuon and muon variables
        # --------------------------------------------------------#

        # ---------------------------------------------------------
        # TODO: find out why we don't filter out bad events right now via
        # even_selection column, since fill muon is computationally exp
        # Last time I checked there was some errors on LHE correction shape mismatch
        # ---------------------------------------------------------

        mu1, mu2 = _leading_two(muons)

        dimuon_dR = mu1.delta_r(mu2)
        dimuon_dEta = abs(mu1.eta - mu2.eta)
        dimuon_dPhi = abs(mu1.delta_phi(mu2))
        acoplanarity = 1 - dimuon_dPhi/ np.pi  # acoplanarity = 1 - delta_phi/pi
        dimuon = mu1+mu2

        uncalibrated_dimuon_ebe_mass_res, calibration = self.get_mass_resolution(dimuon, mu1, mu2, is_mc, doing_BS_correction=doing_BS_correction)
        dimuon_ebe_mass_res = uncalibrated_dimuon_ebe_mass_res * calibration
        dimuon_ebe_mass_res_rel = dimuon_ebe_mass_res/dimuon.mass
        dimuon_cos_theta_cs, dimuon_phi_cs = cs_variables(mu1,mu2)
        dimuon_cos_theta_eta, dimuon_phi_eta = etaFrame_variables(mu1,mu2)

        t10 = time.perf_counter()
        logger.debug(f"[timing] Dimuon variables time: {t10 - t9:.2f} seconds")

        # fill genjets
        if is_mc:
            # fill gen jets for VBF filter on postprocess
            gjets = events.GenJet
            gleptons = events.GenPart[
                (
                    (abs(events.GenPart.pdgId) == 13)
                    | (abs(events.GenPart.pdgId) == 11)
                    | (abs(events.GenPart.pdgId) == 15)
                )
                & events.GenPart.hasFlags('isHardProcess')
            ]
            gl_pair = ak.cartesian({"jet": gjets, "lepton": gleptons}, axis=1, nested=True)
            dr_gl = gl_pair["jet"].delta_r(gl_pair["lepton"])
            isolated = ak.all((dr_gl > 0.3), axis=-1) # this also returns true if there's no leptons near the gjet

            # same order sorting algorithm as reco jets
            gjets = ak.to_packed(gjets[isolated])
            sorted_args = ak.argsort(gjets.pt, ascending=False)
            sorted_gjets = (gjets[sorted_args])
            gjets_sorted = ak.pad_none(sorted_gjets, target=2)

            gjet1 = gjets_sorted[:,0]
            gjet2 = gjets_sorted[:,1]
            gjj = gjet1 + gjet2

            gjj_dEta = abs(gjet1.eta - gjet2.eta)
            gjj_dPhi = abs(gjet1.delta_phi(gjet2))
            gjj_dR = gjet1.delta_r(gjet2)

            # number of gen jets
            n_genjets = ak.num(gjets, axis=1)
            # number of gen jets with pT > 25 GeV and |eta| < 4.7
            n_genjets_pt25_eta47 = ak.sum((gjets.pt > 25) & (abs(gjets.eta) < 4.7), axis=1)
            # number of gen jets with pT > 30 GeV and |eta| < 4.7
            n_genjets_pt30_eta47 = ak.sum((gjets.pt > 30) & (abs(gjets.eta) < 4.7), axis=1)

        t11 = time.perf_counter()
        logger.debug(f"[timing] GenJet variables time: {t11 - t10:.2f} seconds")

        self.prepare_jets(events, NanoAODv=NanoAODv)

        # ------------------------------------------------------------#
        # Apply JEC, get JEC and JER variations
        # ------------------------------------------------------------#
        # JER: https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetResolution
        # JES: https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC

        year = self.config["year"]
        jets = events.Jet

        jets = ensure_event_axis(jets, len(events), "jet")

        PuppiMET = events.PuppiMET
        if self.config["switches"].get("do_met_xy_correction", False):
            logger.debug("Applying PuppiMET xy-shift correction!")
            xy_met_pt, xy_met_phi, _ = apply_puppi_met_xy_correction(
                PuppiMET.pt,
                PuppiMET.phi,
                events.PV.npvs,
                events.run,
                year,
                is_mc,
            )
            PuppiMET["pt_xyCorr"] = xy_met_pt
            PuppiMET["phi_xyCorr"] = xy_met_phi
        if self.config["switches"].get("do_jet_veto_maps_filterJets", False):
            logger.debug("Applying jet veto maps!")
            jets, PuppiMET = self.compute_jet_veto_jetfilter(events, jets, PuppiMET)
            jets = ensure_event_axis(jets, len(events), "jet-post-veto")
        t12 = time.perf_counter()
        logger.debug(f"[timing] prepare jets time: {t12 - t11:.2f} seconds")

        logger.debug(f"jets type before pad_none: {jets.type}")
        logger.debug(f"jets ndim: {jets.ndim}")

        jet_default = ak.pad_none(jets, target=4, axis=1)
        jet1_default = jet_default[:, 0]
        jet2_default = jet_default[:, 1]
        save_four_jets_kinematics = self.config["switches"]["save_four_jets_kinematics"]
        if save_four_jets_kinematics:
            jet3_default = jet_default[:, 2]
            jet4_default = jet_default[:, 3]

        # -----------------------------------------------------
        # pre-selection for fatjets
        # add pre-selection for fatjets before saving the information: pT > 150 GeV and |eta| < 2.4 and pass the tight jet ID, dR(j, muons) > 0.8, FatJet_particleNetWithMass_WvsQCD > 0.75
        # Save the number of fat jets that passes this conditions
        do_getFatJet_vars = self.config["switches"].get("do_getFatJet_vars", False)
        if do_getFatJet_vars:
            fatJets = events.FatJet
            fatJets = ensure_event_axis(fatJets, len(events), "fatjet")
            nfatJets = ak.num(fatJets, axis=1)
            fatjet_selection = (
                (fatJets.pt > 150)
                & (abs(fatJets.eta) < 2.4)
                & (fatJets.particleNetWithMass_WvsQCD > 0.75) # W vs QCD discriminator
            )
            if hasattr(fatJets, "jetId"):
                fatjet_selection = fatjet_selection & (fatJets.jetId >= 2)
            elif is_run3(year):  # NanoAODv15 has no FatJet_jetId: JME correctionlib AK8PUPPI Tight
                fatjet_selection = fatjet_selection & fatjet_tight_id_correctionlib(fatJets, self.config)
            else:
                logger.warning("FatJets have no jetId field!")
                tight_id, _ = custom_jet_id(fatJets, year)
                fatjet_selection = fatjet_selection & tight_id

            fatJets = fatJets[fatjet_selection]
            nfatJets_pre = ak.num(fatJets, axis=1)

            # if nfatJets_pre > 0, we apply the dR(jet, muon) > 0.8 cut and save the number of fatjets that passes this
            # here muons are mu1 and mu2, as defined above
            fatJets_dRmu1 = fatJets.delta_r(mu1)
            fatJets_dRmu2 = fatJets.delta_r(mu2)
            fatJets_dRmu1 = ak.fill_none(fatJets_dRmu1, 999) # if there's no fatjet, set dR to a large number, set it to +999 as later I am checking min of the two numbers. So, set it to large +ve number
            fatJets_dRmu2 = ak.fill_none(fatJets_dRmu2, 999)
            fatJets_dRmu = np.minimum(fatJets_dRmu1, fatJets_dRmu2)

            fatJets = fatJets[fatJets_dRmu > 0.8]
            nfatJets_drmuon = ak.num(fatJets, axis=1)

            # keep only the leading fatjet after all the selections above
            fatJets_default = ak.pad_none(fatJets, target=1)
            fatJet1_default = fatJets_default[:, 0]

        do_jec = self.config["switches"]["do_jec"]
        # JES uncertainties are MC-only; local flag, never written back to the shared config
        do_jec_unc = self.config["switches"]["do_jec_unc"] and is_mc
        logger.debug(f"do_jec_unc: {do_jec_unc}")
        do_jer_unc = self.config["switches"]["do_jer_unc"]
        logger.debug(f"do_jer_unc: {do_jer_unc}")
        jec_unc_sources = []
        jet_type = getJetType(NanoAODv, year)
        self.config["jec_parameters"]["jet_algorithm"] = jet_type # define jet type / jet algorithm into the jec.yaml
        logger.debug(f'jec params: {self.config["jec_parameters"]}')
        if do_jec:
            logger.debug("doing JEC  (+ JER for MC)!")

            # 1) JES/JER variation labels you want to carry
            if do_jec_unc:
                jec_tag = self.config["jec_parameters"]["jec_tags"]
                if (not isinstance(jec_tag, str)):
                    jec_tag = jec_tag[f"nanoAODv{NanoAODv}"]
                jerc_load_path = self.config["jec_parameters"]["jerc_load_path"]
                cset = get_corrset(jerc_load_path, NanoAODv)
                jec_unc_sources = get_jec_sources(cset, jec_tag, jet_type=jet_type)
                variation_l = ["nominal"] + jec_unc_sources
            else:
                variation_l = ["nominal"]

            logger.debug(f"variations: {variation_l}")

            # 2) Apply JES to jets (nominal + uncertainty sources)
            jets = do_jec_scale(jets, events, self.config, is_mc, dataset, uncs=variation_l)

            # store nominal snapshot names
            jets["mass_jec"] = jets.mass
            jets["pt_jec"] = jets.pt

            logger.debug(f"year: {year}, is_mc: {is_mc}, dataset: {dataset}")

            # 3) Apply JER smearing on MC (https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution#JER_Scaling_factors_and_Uncertai)
            if is_mc and (self.config["switches"]["jer_strat"] >=0):
                logger.debug("Applying JER smearing!")
                # Only build the up/down JER-smear columns when they will be used
                # (do_jer_unc drives the pt_variations loop further below). On a
                # nominal run this skips 2/3 of the JER correctionlib evaluations
                # and the apply_jer_unc per-eta-bin columns, with no change to the
                # skim output.
                jer_syst_l = ["nom", "up", "down"] if do_jer_unc else ["nom"]
                jets = do_jer_smear(jets, self.config, events.event, syst_l=jer_syst_l)
            else:
                logger.debug(f"==> Not applying JER smearing. is_mc: {is_mc}, jer_strat: {self.config['switches']['jer_strat']}")

            # 4) Sort jets *after* final pt is set
            sorted_args = ak.argsort(jets.pt, ascending=False)
            jets = (jets[sorted_args])

            # now JER has been applied, we apply unc coeefficients to the latest value
            variation_l.remove("nominal")
            if is_mc:
                jets = applyJetUncertaintyKinematics(jets, variation_l)

        else:
            jets["mass_jec"] = jets.mass
            jets["pt_jec"] = jets.pt

        t13 = time.perf_counter()
        logger.debug(f"[timing] JEC and JER time: {t13 - t12:.2f} seconds")

        # ------------------------------------------------------------#
        # Apply genweights, PU weights and L1 prefiring weights
        # ------------------------------------------------------------#
        save_all_weight_variations = self.config["switches"].get("save_all_weight_variations", False)
        do_save_partial_weights = self.config["switches"].get("do_save_partial_weights", False)
        weights = Weights(len(events), storeIndividual=do_save_partial_weights) # none for dask awkward
        # Set by the do_pdf block below, consumed where weight_dict is built.
        pdf_member_ratios = None
        pdf_alpha_s_ratios = None
        pdf_central_weight = None
        write_pdf_members = False
        ps_ratios = None  # set by the do_ps_wgt block below
        if is_mc:
            gen_weight_ones = ak.ones_like(events.genWeight)
            if "MiNNLO" in dataset: # We have spurious gen weight issue. ref: https://cms-talk.web.cern.ch/t/huge-event-weights-in-dy-powhegminnlo/8718/9
                weights.add("genWeight", weight=np.sign(events.genWeight)) # just extract the sign, not the magnitude
            else:
                weights.add("genWeight", weight=events.genWeight)
            weights.add("genWeight_normalization", weight=gen_weight_ones/sumWeights)

            logger.debug(f"year: {year}, dataset_yaml_file: {dataset_yaml_file}")
            # FIXME: Remove this if condition later when we update the yaml file for run2 too.
            sample_info = get_sample_info(dataset_yaml_file, dataset, year) # FIXME: hardcoded filename
            logger.debug(f"sample_info: {sample_info}")

            integrated_lumi = sample_info["total_lumi_pb"]
            logger.debug(f"integrated_lumi: {integrated_lumi}")

            cross_section = sample_info["cross_section_pb"]
            logger.debug(f"cross_section (before k-factor): {cross_section}")

            kfactor = sample_info["kfactor_value"]
            cross_section = cross_section * kfactor

            logger.debug(f"kfactor: {kfactor}")
            logger.debug(f"cross_section (after k-factor): {cross_section}")

            weights.add("xsec", weight=gen_weight_ones*cross_section)
            weights.add("lumi", weight=gen_weight_ones*integrated_lumi)

            if do_pu_wgt:
                logger.debug("adding PU wgts!")
                weights.add("pu", weight=pu_wgts["nom"],weightUp=pu_wgts["up"],weightDown=pu_wgts["down"])
            # L1 prefiring weights
            if self.config["switches"]["do_l1prefiring_wgts"] and ("L1PreFiringWeight" in events.fields):
                logger.debug("adding L1 prefiring wgts!")
                L1_nom = events.L1PreFiringWeight.Nom
                L1_up = events.L1PreFiringWeight.Up
                L1_down = events.L1PreFiringWeight.Dn
                weights.add("l1prefiring",
                    weight=L1_nom,
                    weightUp=L1_up,
                    weightDown=L1_down
                )
        else: # data-> just add in ak ones for consistency
            weights.add("ones", weight=ak.values_astype(ak.ones_like(events.HLT.IsoMu24), "float32"))

        t14 = time.perf_counter()
        logger.debug(f"[timing] Weights time: {t14 - t13:.2f} seconds")

        # ------------------------------------------------------------#
        # Calculate other event weights
        # ------------------------------------------------------------#
        pt_variations = ["nominal"]
        if is_mc:
            if do_jec_unc:
                pt_variations += applyUpDown(jec_unc_sources)

            if self.config["switches"]["do_jer_unc"] and self.config["switches"]["jer_strat"] >= 0:
                pt_variations += self.config["jec_parameters"]["jer_variations"]

        logger.debug(f"pt_variations: {pt_variations}")
        if is_mc:
            do_nnlops = self.config["switches"]["do_nnlops"] and ("ggh" in events.metadata["dataset"])
            if do_nnlops:
                logger.debug("doing NNLOPS!")
                nnlopsw = nnlops_weights(events.HTXS.Higgs_pt, events.HTXS.njets30, self.config, events.metadata["dataset"])
                weights.add("nnlops", weight=nnlopsw)

            # do mu SF start -------------------------------------
            logger.debug("doing musf!")
            if is_run3(year):
                # stat and syst as separate nuisances; nominal keeps the muID/muIso/muTrig names
                for name, sf in muon_sfs_stat_syst_correctionlib(mu1, mu2, self.config).items():
                    weights.add(name, weight=sf["nom"])
                    safe_nom = ak.where(sf["nom"] != 0, sf["nom"], 1.0)
                    for comp in ("stat", "syst"):
                        up, down = sf[comp]
                        weights.add(f"{name}_{comp}",
                            weight=ak.ones_like(sf["nom"]),
                            weightUp=up / safe_nom,
                            weightDown=down / safe_nom,
                        )
            elif is_run2(year):
                muon_sfs = add_muon_sfs_correctionlib(mu1, mu2, self.config)
                for name, sf in zip(("muID", "muIso", "muTrig"), muon_sfs):
                    weights.add(name, weight=sf["nom"], weightUp=sf["up"], weightDown=sf["down"])
            else:
                raise ValueError(f"Year {year} is not recognized as Run 2 or Run 3 year for muon SFs!")

            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
            do_lhe = "LHEScaleWeight" in events.fields
            if do_lhe:
                logger.debug("doing LHE!")
                lhe_ren, lhe_fac = lhe_weights(events, events.metadata["dataset"], self.config["year"])
                weights.add("LHERen",
                    weight=ak.ones_like(lhe_ren["up"]),
                    weightUp=lhe_ren["up"],
                    weightDown=lhe_ren["down"]
                )
                weights.add("LHEFac",
                    weight=ak.ones_like(lhe_fac["up"]),
                    weightUp=lhe_fac["up"],
                    weightDown=lhe_fac["down"]
                )

            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
            dataset = events.metadata["dataset"]
            do_thu = (
                self.config["switches"]["do_THU"]
                and ("vbf" in dataset)
                and ("dy" not in dataset)
                and ("stage1_1_fine_cat_pTjet30GeV" in events.HTXS.fields)
            )
            if do_thu:
                logger.debug("doing THU weights!")
                add_stxs_variations(
                    events,
                    weights,
                    self.config,
                )

            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
            # Gate by branch-title LHA IDs using pdf_supported_lha_ids in switches_official.yaml.
            # Only 103-member NNPDF3.1 symmhessian+as sets (306000/325300) are supported;
            # other sets receive unity weights because they need different prescriptions.
            # Supported sets with corrupt weights (st_tchannel_*, tt_inclusive_amcatnlo)
            # are excluded in the Run2 dataset YAML.
            pdf_lha_ids = get_pdf_lha_id_range(events)
            logger.debug(f"{dataset} ({year}): LHEPdfWeight LHA IDs {pdf_lha_ids}")
            if pdf_lha_ids is None and ("LHEPdfWeight" in events.fields):
                logger.warning(
                    f"{dataset} ({year}): LHEPdfWeight title carries no LHA ID range, "
                    f"so PDF members are set to unity"
                )
            pdf_excluded_samples = self.config["switches"].get("pdf_excluded_samples", [])
            if dataset in pdf_excluded_samples:
                logger.warning(
                    f"{dataset} ({year}): listed in pdf_excluded_samples, "
                    f"so PDF members are set to unity"
                )
            do_pdf = (
                self.config["switches"]["do_pdf"]
                and pdf_lha_ids is not None
                and pdf_lha_ids[0] in self.config["switches"]["pdf_supported_lha_ids"]
                and dataset not in pdf_excluded_samples
            )
            # The 100 eigenvector members are carried to stage3 as separate weight
            if do_pdf:
                logger.debug("doing pdf!")
                pdf_member_ratios, pdf_alpha_s_ratios, pdf_central_weight = add_pdf_variations(
                    events, self.config, dataset
                )
                write_pdf_members = True
            else:
                if self.config["switches"]["do_pdf"]: # for other mc samples with unsuppoorted NNPDF, we emit unity members
                    # Unity members, so the per-group variation intersection in stage3
                    # does not drop pdf_unc for every other dataset in the same group.
                    # A unit ratio contributes exactly zero to F^(k) - F^(0).
                    logger.debug("skip pdf! member ratios set to unity")
                    write_pdf_members = True

            # PS ISR/FSR ratios: plain columns (no wgt_ prefix), so stage-2/3 do not turn
            # them into nuisances for every process; Run-2 applied PS to signal only.
            if self.config["switches"].get("do_ps_wgt", False):
                ps_ratios = ps_weights(events, dataset, year)
        t15 = time.perf_counter()
        logger.debug(f"[timing] some GEN event weights for syst time: {t15 - t14:.2f} seconds")

        # ------------------------------------------------------------#
        # Fill Muon variables and gjet variables
        # ------------------------------------------------------------#
        # if year length is > 4, then it contains "pre" or "post" or "BPix"
        if len(year) > 4:
            """For the DNN training, we want to add year as one of the input variables.

            The expected format for `year` is a string like "2016pre", "2016post", "2017", "2018", "2022pre", or "2022post".

            If the year contains "pre", it is mapped to .0 (e.g., "2016pre" -> 2016.0).
            If the year contains "post", it is mapped to .5 (e.g., "2016post" -> 2016.5).
            If the year does not match these patterns, it is converted directly to float (e.g., "2017" -> 2017.0).
            If the format is unexpected, this may raise a ValueError.
            """
            logger.debug(f"Year format contains more than 4 characters: {year}")
            dnn_year = float(year[:4])
            if "pre" in year:
                dnn_year += 0.0
            else:
                dnn_year += 0.5
            logger.debug(f"Mapped year to dnn_year: {dnn_year}")
        else:
            dnn_year = float(year)
        logger.debug(f"dnn_year: {dnn_year}")

        # output dict for the output parquet file
        out_dict = {}

        # Event identifiers
        _add_block(out_dict, {
            "event": events.event,
            "run": events.run,
            "luminosityBlock": events.luminosityBlock,
            "fraction": ak.ones_like(events.event) * events.metadata["fraction"],
            "year": ak.ones_like(nmuons) * dnn_year,
        })

        # Leading and sub-leading muon kinematics
        _add_block(out_dict, {
            "mu1_pt": mu1.pt,
            "mu1_ptErr": mu1.ptErr,
            "mu1_eta": mu1.eta,
            "mu1_phi": mu1.phi,

            "mu2_pt": mu2.pt,
            "mu2_ptErr": mu2.ptErr,
            "mu2_eta": mu2.eta,
            "mu2_phi": mu2.phi,

            "mu1_pt_over_mass": safe_ratio(mu1.pt, dimuon.mass, default=0.0),
            "mu2_pt_over_mass": safe_ratio(mu2.pt, dimuon.mass, default=0.0),
        })

        # Dimuon kinematics
        _add_block(
            out_dict,
            {
                "dimuon_mass": dimuon.mass,
                "dimuon_pt": dimuon.pt,
                "dimuon_pt_log": np.log(dimuon.pt),
                "dimuon_eta": dimuon.eta,
                "dimuon_rapidity": getRapidity(dimuon),
                "dimuon_phi": dimuon.phi,

                "dimuon_dEta": dimuon_dEta,
                "dimuon_dPhi": dimuon_dPhi,
                "dimuon_dR": dimuon_dR,
                "acoplanarity": acoplanarity,
                "mu_max_abs_eta": np.maximum(np.abs(mu1.eta), np.abs(mu2.eta)),
            },
        )

        # Mass resolution and angular variables
        _add_block(out_dict, {
            "uncalibrated_dimuon_ebe_mass_res": uncalibrated_dimuon_ebe_mass_res,
            "dimuon_ebe_mass_res": dimuon_ebe_mass_res,
            "dimuon_ebe_mass_res_rel": dimuon_ebe_mass_res_rel,
            "dimuon_cos_theta_cs": dimuon_cos_theta_cs,
            "dimuon_phi_cs": dimuon_phi_cs,
        })

        save_muon_roccor_variations = self.config["switches"].get(
            "save_muon_roccor_variations", True
        )
        if is_mc and save_muon_roccor_variations:
            muon_variation_block = {}
            for suffix, field in MUON_PT_VARIATION_FIELDS.items():
                if hasattr(mu1, field) and hasattr(mu2, field):
                    muon_variation_block.update(
                        build_muon_kinematic_variation_block(
                            self,
                            mu1,
                            mu2,
                            mu1[field],
                            mu2[field],
                            suffix,
                            is_mc,
                            doing_BS_correction=doing_BS_correction,
                        )
                    )
            if len(muon_variation_block) > 0:
                _add_block(out_dict, muon_variation_block)

        # MET
        _add_block(out_dict, {
            "PuppiMET_pt": PuppiMET.pt,
            "PuppiMET_phi": PuppiMET.phi,
            "PuppiMET_sumEt": PuppiMET.sumEt,
        })
        if self.config["switches"].get("do_met_xy_correction", False):
          _add_block(out_dict, {
                "PuppiMET_pt_xyCorr": PuppiMET.pt_xyCorr,
                "PuppiMET_phi_xyCorr": PuppiMET.phi_xyCorr,
            })  

        # FatJet block
        if do_getFatJet_vars:
            _add_block(out_dict, {
                "nfatJets": nfatJets,
                "nfatJets_pre": nfatJets_pre,
                "nfatJets_drmuon": nfatJets_drmuon,

                "fatJet1_default_pt_nominal": fatJet1_default.pt,
                "fatJet1_default_eta_nominal": fatJet1_default.eta,
                "fatJet1_default_phi_nominal": fatJet1_default.phi,
                "fatJet1_default_mass_nominal": fatJet1_default.mass,
                "fatJet1_default_msoftdrop_nominal": fatJet1_default.msoftdrop,
                "fatJet1_default_particleNetWithMass_WvsQCD_nominal": fatJet1_default.particleNetWithMass_WvsQCD,
            })

        # Additional jet block
        if self.config["switches"]["save_default_jet_vars"]:
            # Default jet kinematics (nominal, pre-JEC/JER snapshot)
            _add_block(out_dict, {
                "jet1_default_pt_nominal": jet1_default.pt,
                "jet1_default_eta_nominal": jet1_default.eta,
                "jet1_default_phi_nominal": jet1_default.phi,
                "jet1_default_mass_nominal": jet1_default.mass,

                "jet2_default_pt_nominal": jet2_default.pt,
                "jet2_default_eta_nominal": jet2_default.eta,
                "jet2_default_phi_nominal": jet2_default.phi,
                "jet2_default_mass_nominal": jet2_default.mass,
            })
            if save_four_jets_kinematics:
                _add_block(out_dict, {
                    "jet3_default_pt_nominal": jet3_default.pt,
                    "jet3_default_eta_nominal": jet3_default.eta,
                    "jet3_default_phi_nominal": jet3_default.phi,
                    "jet3_default_mass_nominal": jet3_default.mass,

                    "jet4_default_pt_nominal": jet4_default.pt,
                    "jet4_default_eta_nominal": jet4_default.eta,
                    "jet4_default_phi_nominal": jet4_default.phi,
                    "jet4_default_mass_nominal": jet4_default.mass,
                })
        _add_block(out_dict, {
            "PV_npvs": events.PV.npvs,
            "PV_npvsGood": events.PV.npvsGood,
        })

        # --- Muon impact parameters: always saved (needed to study/tune IP cuts from ntuples) ---
        _add_block(out_dict, {
            "mu1_dxy":        mu1.dxy,
            "mu2_dxy":        mu2.dxy,
            "mu1_dxyErr":     mu1.dxyErr,
            "mu2_dxyErr":     mu2.dxyErr,
            "mu1_dxybs":      mu1.dxybs,
            "mu2_dxybs":      mu2.dxybs,
            "mu1_dz":         mu1.dz,
            "mu2_dz":         mu2.dz,
            "mu1_dzErr":      mu1.dzErr,
            "mu2_dzErr":      mu2.dzErr,
            "mu1_ip3d":       mu1.ip3d,
            "mu2_ip3d":       mu2.ip3d,
            "mu1_sip3d":      mu1.sip3d,
            "mu2_sip3d":      mu2.sip3d,
        })

        # --- Extra muon variables  ----------------------
        save_full_muon_detail = self.config["switches"].get("save_full_muon_detail", False)
        if save_full_muon_detail:
            _add_block(out_dict, {
                "mu1_charge": mu1.charge,
                "mu2_charge": mu2.charge,
                "mu1_pt_over_mu2_pt": safe_ratio(mu1.pt, mu2.pt),
                "mu1_eta_over_mu2_eta": safe_ratio(abs(mu1.eta), abs(mu2.eta)),
                "mu1_pt_roch" : mu1.pt_roch,
                "mu1_pt_fsr" : mu1.pt_fsr,
                "mu2_pt_roch" : mu2.pt_roch,
                "mu2_pt_fsr" : mu2.pt_fsr,

                # IDs / quality flags
                "mu1_highPurity":     mu1.highPurity,
                "mu2_highPurity":     mu2.highPurity,
                "mu1_inTimeMuon":     mu1.inTimeMuon,
                "mu2_inTimeMuon":     mu2.inTimeMuon,
                "mu1_isGlobal":       mu1.isGlobal,
                "mu2_isGlobal":       mu2.isGlobal,
                "mu1_isPFcand":       mu1.isPFcand,
                "mu2_isPFcand":       mu2.isPFcand,
                "mu1_isStandalone":   mu1.isStandalone,
                "mu2_isStandalone":   mu2.isStandalone,
                "mu1_isTracker":      mu1.isTracker,
                "mu2_isTracker":      mu2.isTracker,
                "mu1_looseId":        mu1.looseId,
                "mu2_looseId":        mu2.looseId,
                "mu1_mediumId":       mu1.mediumId,
                "mu2_mediumId":       mu2.mediumId,
                "mu1_mediumPromptId": mu1.mediumPromptId,
                "mu2_mediumPromptId": mu2.mediumPromptId,
                "mu1_tightCharge":    mu1.tightCharge,
                "mu2_tightCharge":    mu2.tightCharge,
                "mu1_pdgId":          mu1.pdgId,
                "mu2_pdgId":          mu2.pdgId,

                # Isolation IDs / working points
                "mu1_miniIsoId":        mu1.miniIsoId,
                "mu2_miniIsoId":        mu2.miniIsoId,
                "mu1_miniPFRelIso_all": mu1.miniPFRelIso_all,
                "mu2_miniPFRelIso_all": mu2.miniPFRelIso_all,
                "mu1_miniPFRelIso_chg": mu1.miniPFRelIso_chg,
                "mu2_miniPFRelIso_chg": mu2.miniPFRelIso_chg,
                "mu1_multiIsoId":       mu1.multiIsoId,
                "mu2_multiIsoId":       mu2.multiIsoId,
                "mu1_pfIsoId":          mu1.pfIsoId,
                "mu2_pfIsoId":          mu2.pfIsoId,
                "mu1_pfRelIso03_all":    mu1.pfRelIso03_all,
                "mu2_pfRelIso03_all":    mu2.pfRelIso03_all,
                "mu1_pfRelIso03_chg":   mu1.pfRelIso03_chg,
                "mu2_pfRelIso03_chg":   mu2.pfRelIso03_chg,
                "mu1_pfRelIso04_all":   mu1.pfRelIso04_all,
                "mu2_pfRelIso04_all":   mu2.pfRelIso04_all,
                "mu1_puppiIsoId":       mu1.puppiIsoId,
                "mu2_puppiIsoId":       mu2.puppiIsoId,
                "mu1_tkIsoId":          mu1.tkIsoId,
                "mu2_tkIsoId":          mu2.tkIsoId,
                "mu1_tkRelIso":         mu1.tkRelIso,
                "mu2_tkRelIso":         mu2.tkRelIso,

                # Track / stations info
                "mu1_nStations":       mu1.nStations,
                "mu2_nStations":       mu2.nStations,
                "mu1_nTrackerLayers":  mu1.nTrackerLayers,
                "mu2_nTrackerLayers":  mu2.nTrackerLayers,
                "mu1_segmentComp":     mu1.segmentComp,
                "mu2_segmentComp":     mu2.segmentComp,

                # Jet matching
                "mu1_jetIdx":          mu1.jetIdx,
                "mu2_jetIdx":          mu2.jetIdx,
                "mu1_jetNDauCharged":  mu1.jetNDauCharged,
                "mu2_jetNDauCharged":  mu2.jetNDauCharged,
                "mu1_jetPtRelv2":      mu1.jetPtRelv2,
                "mu2_jetPtRelv2":      mu2.jetPtRelv2,
                "mu1_jetRelIso":       mu1.jetRelIso,
                "mu2_jetRelIso":       mu2.jetRelIso,

                # SV matching
                "mu1_svIdx":           mu1.svIdx,
                "mu2_svIdx":           mu2.svIdx,

                "nmuons": nmuons,

                "dimuon_cos_theta_eta": dimuon_cos_theta_eta,
                "dimuon_phi_eta": dimuon_phi_eta,
                "dimuon_pt_over_PuppiMET_pt": safe_ratio(dimuon.pt, PuppiMET.pt, default=0.0),
                "dimuon_pt_over_jet1_pt": safe_ratio(dimuon.pt, jet1_default.pt, default=0.0),
                "dimuon_pt_over_jet2_pt": safe_ratio(dimuon.pt, jet2_default.pt, default=0.0),
                "mu1_pt_raw": mu1.pt_raw,
                "mu2_pt_raw": mu2.pt_raw,
            })

        # ------------------------------------------------------------#
        # Correlations between the two muons
        # ------------------------------------------------------------#
        # Basic kinematic correlations
        pt_sum      = mu1.pt + mu2.pt
        pt_diff     = mu1.pt - mu2.pt
        pt_absdiff  = abs(pt_diff)
        pt_prod     = mu1.pt * mu2.pt
        pt_ratio12  = safe_ratio(mu1.pt, mu2.pt, default=1.0)
        pt_ratio21  = safe_ratio(mu2.pt, mu1.pt, default=1.0)
        pt_min      = ak.where(mu1.pt < mu2.pt, mu1.pt, mu2.pt)
        pt_max      = ak.where(mu1.pt > mu2.pt, mu1.pt, mu2.pt)
        pt_asym     = safe_ratio(mu1.pt - mu2.pt, mu1.pt + mu2.pt, default=0.0)

        eta_sum     = mu1.eta + mu2.eta
        eta_diff    = mu1.eta - mu2.eta
        eta_prod    = mu1.eta * mu2.eta

        abs_eta1    = abs(mu1.eta)
        abs_eta2    = abs(mu2.eta)
        abs_eta_sum = abs_eta1 + abs_eta2
        abs_eta_diff = abs(abs_eta1 - abs_eta2)
        abs_eta_min = ak.where(abs_eta1 < abs_eta2, abs_eta1, abs_eta2)

        # Isolation correlations (using 04-cone since you already use it as base)
        iso1 = mu1.pfRelIso04_all
        iso2 = mu2.pfRelIso04_all
        iso_sum     = iso1 + iso2
        iso_diff    = iso1 - iso2
        iso_absdiff = abs(iso_diff)
        iso_prod    = iso1 * iso2
        iso_min     = ak.where(iso1 < iso2, iso1, iso2)
        iso_max     = ak.where(iso1 > iso2, iso1, iso2)
        iso_asym    = safe_ratio(iso1 - iso2, iso1 + iso2, default=0.0)

        # Impact-parameter–related correlations
        dxy1, dxy2 = mu1.dxy, mu2.dxy
        dz1,  dz2  = mu1.dz,  mu2.dz
        sip1, sip2 = mu1.sip3d, mu2.sip3d

        dxy_sum     = dxy1 + dxy2
        dxy_diff    = dxy1 - dxy2
        dxy_absdiff = abs(dxy_diff)
        dz_sum      = dz1 + dz2
        dz_diff     = dz1 - dz2
        dz_absdiff  = abs(dz_diff)

        sip_sum     = sip1 + sip2
        sip_diff    = sip1 - sip2
        sip_absdiff = abs(sip_diff)
        sip_prod    = sip1 * sip2
        sip_min     = ak.where(sip1 < sip2, sip1, sip2)
        sip_max     = ak.where(sip1 > sip2, sip1, sip2)

        # Track quality correlations
        nStations1, nStations2 = mu1.nStations, mu2.nStations
        nTrkLayers1, nTrkLayers2 = mu1.nTrackerLayers, mu2.nTrackerLayers

        nStations_min = ak.where(nStations1 < nStations2, nStations1, nStations2)
        nStations_max = ak.where(nStations1 > nStations2, nStations1, nStations2)
        nStations_sum = nStations1 + nStations2

        nTrkLayers_min = ak.where(nTrkLayers1 < nTrkLayers2, nTrkLayers1, nTrkLayers2)
        nTrkLayers_max = ak.where(nTrkLayers1 > nTrkLayers2, nTrkLayers1, nTrkLayers2)
        nTrkLayers_sum = nTrkLayers1 + nTrkLayers2

        # Charge correlation
        q1q2 = mu1.charge * mu2.charge   # should be -1 for selected OS events

        # Dimuon impact-parameter variables: always saved, like the per-muon IPs above
        _add_block(out_dict, {
            "mu12_dxy_sum":       dxy_sum,
            "mu12_dxy_diff":      dxy_diff,
            "mu12_dxy_absdiff":   dxy_absdiff,
            "mu12_dz_sum":        dz_sum,
            "mu12_dz_diff":       dz_diff,
            "mu12_dz_absdiff":    dz_absdiff,
            "mu12_sip3d_sum":     sip_sum,
            "mu12_sip3d_diff":    sip_diff,
            "mu12_sip3d_absdiff": sip_absdiff,
            "mu12_sip3d_prod":    sip_prod,
            "mu12_sip3d_min":     sip_min,
            "mu12_sip3d_max":     sip_max,
        })

        if save_full_muon_detail:
            _add_block(out_dict, {
                # pt correlations
                "mu12_pt_sum":      pt_sum,
                "mu12_pt_diff":     pt_diff,
                "mu12_pt_absdiff":  pt_absdiff,
                "mu12_pt_prod":     pt_prod,
                "mu12_pt_ratio12":  pt_ratio12,
                "mu12_pt_ratio21":  pt_ratio21,
                "mu12_pt_min":      pt_min,
                "mu12_pt_max":      pt_max,
                "mu12_pt_asym":     pt_asym,

                # eta / |eta| correlations
                "mu12_eta_sum":      eta_sum,
                "mu12_eta_diff":     eta_diff,
                "mu12_eta_prod":     eta_prod,
                "mu12_absEta_sum":   abs_eta_sum,
                "mu12_absEta_diff":  abs_eta_diff,
                "mu12_absEta_min":   abs_eta_min,

                # isolation correlations
                "mu12_iso04_sum":      iso_sum,
                "mu12_iso04_diff":     iso_diff,
                "mu12_iso04_absdiff":  iso_absdiff,
                "mu12_iso04_prod":     iso_prod,
                "mu12_iso04_min":      iso_min,
                "mu12_iso04_max":      iso_max,
                "mu12_iso04_asym":     iso_asym,

                # track-quality correlations
                "mu12_nStations_min":      nStations_min,
                "mu12_nStations_max":      nStations_max,
                "mu12_nStations_sum":      nStations_sum,
                "mu12_nTrackerLayers_min": nTrkLayers_min,
                "mu12_nTrackerLayers_max": nTrkLayers_max,
                "mu12_nTrackerLayers_sum": nTrkLayers_sum,

                # charge correlation
                "mu12_q1q2": q1q2,
            })

        if is_mc:
            _add_block(out_dict, {
                "gjj_mass": gjj.mass,
                "n_genjets": n_genjets,
                "n_genjets_pt25_eta47": n_genjets_pt25_eta47,
                "n_genjets_pt30_eta47": n_genjets_pt30_eta47,
                "gjet1_pt" : gjet1.pt,
                "gjet1_eta" : gjet1.eta,
                "gjet1_phi" : gjet1.phi,
                "gjet1_mass" : gjet1.mass,
                "gjet2_pt" : gjet2.pt,
                "gjet2_eta" : gjet2.eta,
                "gjet2_phi" : gjet2.phi,
                "gjet2_mass" : gjet2.mass,
                "gjj_pt" : gjj.pt,
                "gjj_eta" : gjj.eta,
                "gjj_phi" : gjj.phi,
                "gjj_dEta" : gjj_dEta,
                "gjj_dPhi" : gjj_dPhi,
                "gjj_dR" : gjj_dR,
            })

        t16 = time.perf_counter()
        logger.debug(f"[timing] Fill muon and gjet variables time: {t16 - t15:.2f} seconds")
        # ------------------------------------------------------------#
        # HEMVeto study
        # ------------------------------------------------------------#
        if (self.config["switches"]["do_HemVeto"] and self.config["switches"]["do_HemVetoStudy"]):
            logger.debug("Adding HemVeto_filter and is_HemRegion for HemVetoStudy!")
            HemVeto_filter = ak.to_packed(HemVeto_filter[event_filter==True])   # used for HemVetoStudy, doesn't compute if do_hemVetoStudy is False
            is_HemRegion = ak.to_packed(is_HemRegion[event_filter==True]) # used for HemVetoStudy, doesn't compute if do_hemVetoStudy is False

            _add_block(out_dict, {
                "HemVeto_filter" : HemVeto_filter,
                "is_HemRegion" : is_HemRegion,
            })
        # ------------------------------------------------------------#
        # Loop over JEC variations and fill jet variables
        # ------------------------------------------------------------#
        logger.debug(f"pt_variations: {pt_variations}")
        for variation in pt_variations:
            jet_loop_dict = self.jet_loop(
                events,
                jets,
                dimuon,
                mu1,
                mu2,
                variation,
                weights,
                NanoAODv = NanoAODv,
                dnn_year=dnn_year,
                do_jet_horn_puid = self.config["switches"]["do_jet_horn_puid"]
            )

            _add_block(out_dict, jet_loop_dict)

        logger.debug(f"out_dict.keys() after jet loop: {out_dict.keys()}")

        t17 = time.perf_counter()
        logger.debug(f"[timing] Jet pT variations time: {t17 - t16:.2f} seconds")

        # fill in the regions
        mass = dimuon.mass
        z_peak = ((mass >= 70.0) & (mass < 110.0))
        h_peak, h_sidebands = _higgs_peak_and_sidebands(mass)
        _add_block(out_dict, {
            "z_peak" : ak.fill_none(z_peak, value=False),
            "h_sidebands" : ak.fill_none(h_sidebands, value=False),
            "h_peak" : ak.fill_none(h_peak, value=False),
        })
        t18 = time.perf_counter()
        logger.debug(f"[timing] various region (z-peak) fill time: {t18 - t17:.2f} seconds")

        # do zpt weight at the very end
        dataset = events.metadata["dataset"]
        do_zpt = ('dy' in dataset) and is_mc and self.config["switches"]["do_zpt"]
        if do_zpt:
            njets_reco = out_dict["njets_nominal"]
            njets_gen = n_genjets_pt30_eta47
            save_zpt_variations = self.config["switches"].get("save_zpt_variations", False)

            logger.debug("=======================  apply zpt weights =======================")
            # choose the config file
            if "MiNNLO" in dataset:
                zpt_cfg = self.config["new_zpt_weights_file_MiNNLO"]
            else:
                zpt_cfg = self.config["new_zpt_weights_file_aMCatNLO"]

            logger.warning(f"zpt weights: {zpt_cfg}")

            zpt_wgt_reco = getZptWgts_3region(dimuon.pt, njets_reco, "function", year, zpt_cfg, NanoAODv)
            zpt_wgt_gen  = getZptWgts_3region(dimuon.pt, njets_gen,  "function", year, zpt_cfg, NanoAODv)

            # --- save both to parquet
            zpt_block = {
                "zpt_wgt_reco": zpt_wgt_reco,
                "zpt_wgt_gen":  zpt_wgt_gen,
            }

            if save_zpt_variations:
                zpt_wgt_reco_up = getZptWgts_3region(
                    dimuon.pt, njets_reco, "function", year, zpt_cfg, NanoAODv, sigma_shift=1.0
                )
                zpt_wgt_reco_down = getZptWgts_3region(
                    dimuon.pt, njets_reco, "function", year, zpt_cfg, NanoAODv, sigma_shift=-1.0
                )
                zpt_wgt_gen_up = getZptWgts_3region(
                    dimuon.pt, njets_gen, "function", year, zpt_cfg, NanoAODv, sigma_shift=1.0
                )
                zpt_wgt_gen_down = getZptWgts_3region(
                    dimuon.pt, njets_gen, "function", year, zpt_cfg, NanoAODv, sigma_shift=-1.0
                )
                zpt_block.update(
                    {
                        "zpt_wgt_reco_up": zpt_wgt_reco_up,
                        "zpt_wgt_reco_down": zpt_wgt_reco_down,
                        "zpt_wgt_gen_up": zpt_wgt_gen_up,
                        "zpt_wgt_gen_down": zpt_wgt_gen_down,
                    }
                )

            _add_block(out_dict, zpt_block)

            _add_block(out_dict, {
                "zpt_njets_reco": njets_reco,
                "zpt_njets_gen": njets_gen,
            })

            # apply reco zpt weight to event weight
            if save_zpt_variations:
                weights.add(
                    "zpt",
                    weight=zpt_wgt_reco,
                    weightUp=zpt_wgt_reco_up,
                    weightDown=zpt_wgt_reco_down,
                )
            else:
                weights.add("zpt", weight=zpt_wgt_reco)

        t19 = time.perf_counter()
        logger.debug(f"[timing] Zpt weights time: {t19 - t18:.2f} seconds")

        logger.debug(f"weight statistics: {weights.weightStatistics.keys()}")

        # add in weights
        weight_dict = {"wgt_nominal": weights.weight()}

        if save_all_weight_variations:
            for variation in weights.variations:
                variation_name = "wgt_" + variation.replace("Up", "_up").replace("Down", "_down") # match the naming scheme of copperhead
                weight_dict[variation_name] = weights.weight(variation)

            # PDF eigenvector members. The `_up` suffix is what makes stage2's existing
            # wgt_*_up/_down discovery pick them up and fill one histogram per member
            # with no stage2 change; stage3 intercepts the `pdfMemberHessEig` prefix and
            # collapses them via Eq. (6.5) instead of emitting 100 nuisances.
            if write_pdf_members:
                nominal_weight = weights.weight()
                for k in range(PDF_N_EIGENVECTOR_MEMBERS):
                    col = f"wgt_pdfMemberHessEig{k:03d}_up"
                    if pdf_member_ratios is None:
                        weight_dict[col] = nominal_weight
                    else:
                        weight_dict[col] = nominal_weight * pdf_member_ratios[:, k]

                # The two alpha_s members, LHEPdfWeight[101] and [102]. Their own prefix
                # keeps them out of stage3's 100-member count; stage3 combines them per
                # PDF4LHC15 Eqs. (27)-(28). Unity for the gated-out samples, for the same
                # group-intersection reason as the members above.
                for j, index in enumerate(PDF_ALPHA_S_MEMBER_INDICES):
                    col = f"wgt_pdfAlphaS{index}_up"
                    if pdf_alpha_s_ratios is None:
                        weight_dict[col] = nominal_weight
                    else:
                        weight_dict[col] = nominal_weight * pdf_alpha_s_ratios[:, j]

                # Save raw w_0 for debugging precision offsets or corrupt weights.
                # Use NaN when the PDF gate skips reading it; unity would imply a measurement.
                # Omit wgt_/_up naming so stage2 does not treat it as a variation.
                _add_block(out_dict, {
                    "pdf_central_member": (
                        np.full(len(nominal_weight), np.nan)
                        if pdf_central_weight is None
                        else pdf_central_weight
                    ),
                })

        t20 = time.perf_counter()
        logger.debug(f"[timing] Weights variations time: {t20 - t19:.2f} seconds")

        # Save partial weights start -----------------------------------------
        if do_save_partial_weights:
            for weight_type in list(weights.weightStatistics.keys()):
                wgt_name = "separate_wgt_" + weight_type
                weight_dict[wgt_name] = weights.partial_weight(include=[weight_type])
                logger.debug(f"adding separate weight: {wgt_name}")
        else:
            if do_zpt:
                _add_block(out_dict, {
                    "separate_wgt_zpt": zpt_wgt_reco,
                })

        t21 = time.perf_counter()
        logger.debug(f"[timing] Weights partials time: {t21 - t20:.2f} seconds")

        _add_block(out_dict, weight_dict)
        if ps_ratios is not None:
            _add_block(out_dict, ps_ratios)

        # ------------------------------------------------------------#
        # Cutflow
        if self.isCutflow:
            # Ensure all selections exist before calling cutflow
            # Add protection for the cutflow if the selection is not in the cutflow
            logger.debug(f"selection: {self.selection}")
            # NOTE: order matters -- PackedSelection.cutflow(*names) ANDs cuts
            # cumulatively in the order given here, not in .add() registration
            # order, so this list must match the actual .add() call sequence
            # below for the per-step "individual"/"cumulative" numbers to mean
            # anything (the final fully-AND'd count doesn't depend on order,
            # but every intermediate row does). Keep in sync with the .add(...)
            # call sites as this method evolves.
            all_required_selections = [
                "TotalEntries",
                "lumi_mask",
                "LHE_cut",
                "HLT_filter",
                "event_quality_flags",
                "muon_pT_roch",
                "muon_eta",
                "muon_id",
                "muon_isGlobal_or_Tracker",
                "muon_ip",          # only with do_muon_ip_cut / do_lepton_ip_veto
                "muon_selection",
                "muon_iso",
                "trigger_match",    # do_trigger_match; otherwise leading_muon_pt
                "leading_muon_pt",
                "electron_veto",
                "HemVeto",
                "PV_npvsGood",
                "nMuons_ge2",       # with do_lepton_ip_veto: these two replace nmuons
                "nLooseMuons_eq2",
                "nmuons",
                "mm_charge",
                "jet_veto_maps",
                "dimuon_mass_window_76_106",
                "h_peak_115_135",
                "h_sidebands_110_115_135_150",
                "h_sidebands_106_115_135_150",
            ]
            # Available cuts inside PackedSelection
            try:
                available_cuts = set(self.selection.names)
            except AttributeError:
                # very old coffea versions might differ — fallback
                available_cuts = set(getattr(self.selection, "_names", []))
                logger.warning(
                    "PackedSelection.names not available (old coffea API?); "
                    f"falling back to _names, found {len(available_cuts)} cut(s)"
                )

            # Keep only the cuts that actually exist in PackedSelection (some
            # are added conditionally, e.g. per year/data-vs-MC), preserving
            # the order above -- including "TotalEntries" itself, so a future
            # rename there gets the same graceful-skip treatment as everything
            # else instead of a silent inconsistency.
            required_selections = [cut for cut in all_required_selections if cut in available_cuts]

            logger.debug(f"dynamic required_selections = {required_selections}")

            # Optional: warn about missing cuts
            missing = [cut for cut in all_required_selections if cut not in available_cuts]
            if missing:
                logger.warning(f"These requested cuts are not defined and will be skipped: {missing}")

            self.cutflow_names = required_selections
            # NOT passing weights= here: PackedSelection.cutflow(*names,
            # weights=..., weightsmodifier=...) does support a weighted
            # cutflow natively since coffea 2025.3.0 (confirmed present in the
            # installed 2026.5.0, see
            # https://coffea-hep.readthedocs.io/en/v2026.5.0/api/coffea.analysis_tools.Cutflow.html)
            # -- but there is no valid same-length weights array available at
            # this point in THIS pipeline to pass it. `events` (and hence
            # `weights`, built from `len(events)` above) gets reduced to the
            # already-selected population at `events = events[event_filter ==
            # True]`, right after every selection here is registered;
            # `self.selection`'s per-cut masks are all still sized to the
            # original, pre-reduction chunk. Confirmed by hitting exactly this
            # mismatch (IndexError, e.g. "size of axis is 148 but size of
            # corresponding boolean axis is 5420") when this was tried via
            # `scripts/update_sync_references.sh` on 2017 sync data (2026-09-09).
            self.cutflow = self.selection.cutflow(*required_selections)
            # Cutflow.print() writes straight to stdout via coffea's own Rich
            # console (not this module's logger), so this table can't be
            # attributed to a chunk when interleaved across concurrent Dask
            # workers -- log a header identifying which dataset it belongs to
            # right before it. The table itself is left as coffea's own
            # print() rather than reimplemented, since re-deriving its
            # dask-delayed materialization here would risk quietly breaking it.
            # Per-chunk table is DEBUG-only so warnings stay visible; cutflow is still saved to JSON/NPZ.
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[{dataset}] Cutflow selection stats (stdout table follows):")
                self.cutflow.print()
        t22 = time.perf_counter()
        logger.debug(f"[timing] Cutflow time: {t22 - t21:.2f} seconds")

        return out_dict, self.processed_event_count  # For METADATA of event count

    def postprocess(self, accumulator):
        """
        Arbitrary postprocess function that's required to run the processor
        """
        logger.debug(f"postprocess: {accumulator}")
        return accumulator

    def get_mass_resolution(self, dimuon, mu1,mu2, is_mc:bool, doing_BS_correction=False):
        """
        - Calculate the dimuon mass resolution based on muon pt uncertainties.
        - If `doing_BS_correction` is True, apply additional calibration from BeamSpot constraint correction
           based on the provided correction JSON file.

        Returns:
        - mass_resolution: The calculated mass resolution.
        - calibration: The calibration factor applied (1.0 if no BS correction).
        """
        muon_E = dimuon.mass / 2.0
        dpt1 = (mu1.ptErr / mu1.pt) * muon_E
        dpt2 = (mu2.ptErr / mu2.pt) * muon_E
        sigma = (dpt1 * dpt1 + dpt2 * dpt2)**0.5

        calibration = 1.0 # default: no calibration applied

        if doing_BS_correction: # apply resolution calibration from BeamSpot constraint correction
            logger.debug("Applying BeamSpot resolution calibration")

            # Load the correction set
            json_path = self.config["BS_res_calib_path"]["MC"] if is_mc else self.config["BS_res_calib_path"]["Data"]
            correction_set = get_corrset(json_path)

            # Access the specific correction by name
            correction = correction_set["BS_ebe_mass_res_calibration"]
            logger.debug(f"correction_set: {correction_set}")
            logger.debug(f"correction: {correction}")

            calibration = correction.evaluate(mu1.pt, abs(mu1.eta), abs(mu2.eta))

        return sigma, calibration

    def prepare_jets(self, events, NanoAODv=9): # analogous to add_jec_variables function in boosted higgs
        # Initialize missing fields (needed for JEC)
        logger.debug(f"prepare jets NanoAODv: {NanoAODv}")
        events["Jet", "pt_raw"] = (1 - events.Jet.rawFactor) * events.Jet.pt
        events["Jet", "mass_raw"] = (1 - events.Jet.rawFactor) * events.Jet.mass
        if NanoAODv >= 12:
            fixedGridRhoFastjetAll = events.Rho.fixedGridRhoFastjetAll
        else: # if v9
            fixedGridRhoFastjetAll = events.fixedGridRhoFastjetAll
        events["Jet", "PU_rho"] = ak.broadcast_arrays(fixedGridRhoFastjetAll, events.Jet.pt)[0] # IMPORTANT: do NOT override "rho" in jets. rho is used for something else, thus we NEED to use PU_rho
        return

    # TODO: STXS VBF cross-section uncertainty

    def jet_loop(
        self,
        events,
        jets,
        dimuon,
        mu1,
        mu2,
        variation,
        weights,
        NanoAODv = 9,
        dnn_year = None,
        do_jet_horn_puid = False,
    ):
        is_mc = events.metadata["is_mc"]
        dataset = events.metadata["dataset"]
        year = self.config["year"]
        logger.debug(f"jet_loop: dataset={dataset}, year={year}, variation={variation}")

        if (not is_mc) and variation != "nominal":
            return {}

        # apply clean jet selection
        # AN-19-124 line 465: "Jets are also cleaned w.r.t. the selected muon candidates by requiring a geometrical separation of ∆R ( j, µ ) > 0.4"
        # muon_sel_corrected_kinematics: clean against ScaRe+FSR muon eta/phi (Round-1), else raw
        mu_eta_f, mu_phi_f = ("eta", "phi") if self.config["switches"]["muon_sel_corrected_kinematics"] else ("eta_raw", "phi_raw")
        _, _, mu1_jet_dR = delta_r_V1(
            mu1[:, np.newaxis][mu_eta_f],
            jets.eta,
            mu1[:, np.newaxis][mu_phi_f],
            jets.phi,
        )
        matched_mu1_jet = mu1_jet_dR <= 0.4
        matched_mu1_jet = ak.fill_none(matched_mu1_jet, value=False)

        _, _, mu2_jet_dR = delta_r_V1(
            mu2[:, np.newaxis][mu_eta_f],
            jets.eta,
            mu2[:, np.newaxis][mu_phi_f],
            jets.phi,
        )
        matched_mu2_jet = mu2_jet_dR <= 0.4
        matched_mu2_jet = ak.fill_none(matched_mu2_jet, value=False)

        matched_mu_pass = matched_mu1_jet | matched_mu2_jet
        clean = ~matched_mu_pass
        clean = ak.fill_none(clean, value=True)

        # INFO: Below patch is basically applying the eta selection and returns new jet
        #       collection. As all other selections are already applied for the jets selection.
        #       I keep the functionality to add other selection so that in near future
        #       if ttH analysis changes anything that we can change accordingly.
        """
        Jet collection: AK4CHS (Run-2) or AK4PUPPI (Run-3)
        Jet pT> 20 GeV
        Jet |η|< 2.4 (before and including 2016) and |η|< 2.5 for 2017 onward
        Tight jet ID
        Loose PUID for jets with pT< 50 GeV (for CHS jets only), as recommended by JetMet
        Reference: https://btv-wiki.docs.cern.ch/ScaleFactors/#important-notes
        """
        # NOTE: b-tag jets are built here, BEFORE the JES/JER swap below, i.e. from nominal-pT jets.
        # The b-jet pT threshold and ID therefore never move with JES/JER, so nBtagLoose/Medium_{variation}
        # (b-veto) equal the nominal counts for every jet variation: the b-veto is not part of the
        # JES/JER migration (tagger scores do not depend on JES/JER; only the pT-threshold edge would).
        btag_selection = btag_jet_selection(jets, self.config, year, clean=clean)
        btag_jets = ak.to_packed(jets[btag_selection])

        # Select particular JEC variation
        if is_mc and (variation != "nominal"):
            fields2add = [
                "puId",
                "jetId",
                "qgl",
                "area",
                "btagDeepB",
                "btagPNetB",
                "btagUParTAK4B",
                "btagRobustParTAK4B",
                # else filled with -1 below for every JES/JER variation
                "btagPNetQvG",
                "btagUParTAK4QvG",
                # Need following when running over JEC. First two for 2022 and 2023. All below for 2024
                "genJetIdx",
                "btagDeepFlavB",
                "chHEF",
                "neHEF",
                "chEmEF",
                "neEmEF",
                "muEF",
                "chMultiplicity",
                "neMultiplicity",
                "multiplicity",
                # Needed by PySR-derived optional features in Run 3
                "hfcentralEtaStripSize",
                "hfadjacentEtaStripsSize",
                "hfsigmaEtaEta",
                "hfsigmaPhiPhi",
                "nConstituents",
                "rawFactor",
                # Needed by the PU-DNN model's feature list (see scaler.json
                # under pu_dnn_model_dir) when do_use_pu_dnn_score is enabled.
                "hfEmEF",
                "hfHEF",
                "nElectrons",
                "nMuons",
                "puIdDisc",
            ]
            jets =  get_jet_variation(jets, variation, fields2add)
            # the JES/JER shift can change the pT order: keep jets pT-sorted (jet1/jet2 = leading, as in nominal);
            # `clean` (muon cleaning, computed above) is per jet and must follow the same order
            order = ak.argsort(jets.pt, axis=1, ascending=False)
            jets = jets[order]
            clean = clean[order]

        # ------------------------------------------------------------#
        # Apply jetID and PUID
        # ------------------------------------------------------------#
        pass_jet_id = jet_id(jets, self.config, year)

        logger.debug(f"jet loop NanoAODv: {NanoAODv}")
        logger.debug(f"dnn_year: {dnn_year}")
        if self.config["switches"]["do_jet_PUID_cut"]:
            logger.debug("Applying jet PUID cut!")
            pass_jet_puid = jet_puid(jets, self.config)
        else:
            pass_jet_puid = ak.ones_like(pass_jet_id, dtype="bool")

        # ------------------------------------------------------------#
        # Select jets
        # ------------------------------------------------------------#
        if NanoAODv == 9:
            pass
        elif NanoAODv == 12:
            jets["btagPNetQvG"] = jets.btagPNetQvG if hasattr(jets, "btagPNetQvG") else ak.zeros_like(jets.pt) - 1.0
            jets["btagDeepFlavQG"] = jets.btagDeepFlavQG if hasattr(jets, "btagDeepFlavQG") else ak.zeros_like(jets.pt) - 1.0
            jets["btagRobustParTAK4QG"] = jets.btagRobustParTAK4QG if hasattr(jets, "btagRobustParTAK4QG") else ak.zeros_like(jets.pt) - 1.0
        elif NanoAODv == 15:
            jets["btagPNetQvG"] = jets.btagPNetQvG if hasattr(jets, "btagPNetQvG") else ak.zeros_like(jets.pt) - 1.0
            jets["btagDeepFlavQG"] = jets.btagDeepFlavQG if hasattr(jets, "btagDeepFlavQG") else ak.zeros_like(jets.pt) - 1.0
            jets["btagUParTAK4QvG"] = jets.btagUParTAK4QvG if hasattr(jets, "btagUParTAK4QvG") else ak.zeros_like(jets.pt) - 1.0
        else:
            raise ValueError(f"NanoAODv{NanoAODv} not supported for jet tagger fields (year {year})!")

        jet_pt_cut = (jets.pt > self.config["jet_pt_cut"])
        abs_eta = abs(jets.eta)
        is_hehf = abs_eta > 2.5
        is_he = is_hehf & (abs_eta <= 3.0)
        all_pass = ak.ones_like(pass_jet_id, dtype="bool")

        # Run-2 jet-horn tight PU ID for |eta| > 2.5
        # source: https://indico.cern.ch/event/1434807/contributions/6040633/attachments/2893077/5071932/JERC%20meeting%2009_07.pdf
        jetHorn_PUID_cut = all_pass
        if do_jet_horn_puid and is_run2(year):
            jetHorn_puid_cut = (get_puId(jets) >= 7) | (jets.pt >= 50)  # tight pu Id #FIXME: hardcoded puID
            jetHorn_PUID_cut = ~is_hehf | (jet_pt_cut & jetHorn_puid_cut)

        do_he_ptcut = self.config["switches"]["do_jet_horn_ptcut"]
        add_hehf_ptcut = self.config["switches"]["add_pt_cut_for_HE_HF_jets"]
        add_hehf_asym = self.config["switches"]["add_asymmetric_pt_cut_for_HE_HF_jets"]
        do_jet_HE_nConstCut = self.config["switches"]["do_jet_HE_nConstCut"]
        n_active = sum(bool(x) for x in [do_he_ptcut, add_hehf_ptcut, add_hehf_asym])
        if n_active > 1:
            raise ValueError(
                "Only one of "
                "do_jet_horn_ptcut, add_pt_cut_for_HE_HF_jets, "
                "add_asymmetric_pt_cut_for_HE_HF_jets can be enabled at once."
            )

        rawFactor_cut = all_pass
        if self.config["switches"]["do_reject_high_rawFactor_jets"]:
            # Official JME mitigation for a rare L2Relative asymptotic-behavior known
            # issue in Run3 (https://cms-jme-jerc.docs.cern.ch/recommendations/jes/#known-issues):
            # in a very narrow pT bin the L2Relative JEC can blow up to anomalously
            # large corrected jet pT (>10 TeV). Affects ~1e-7 to 1e-6 of events in the
            # rare samples where it occurs at all; JME's own fix is to reject jets with
            # Jet_rawFactor > 0.9.
            logger.debug("Applying Jet_rawFactor > 0.9 rejection (rare L2Relative asymptotic-JEC mitigation)")
            rawFactor_cut = jets.rawFactor <= 0.9

        # Run-3 recommendation: remove HE jets (2.5 < |eta| <= 3.0) with pT below do_he_ptcut
        # (https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetMET#Run3_recommendations)
        jetHorn_ptcut = all_pass
        if do_he_ptcut:
            logger.debug(f"Applying additional jet pT cut of {do_he_ptcut} GeV for forward region (jet horn region)!")
            jetHorn_ptcut = ~is_he | (jets.pt > do_he_ptcut)

        if add_hehf_ptcut:
            logger.debug(f"Applying additional jet pT cut of {add_hehf_ptcut} GeV for HE/HF jets!")

        # Run-3 recommendation: remove HE jets with nConstituents <= 3
        jetHorn_nConst_Cut = all_pass
        if do_jet_HE_nConstCut:
            logger.debug("Applying nConstituent cut > 3 for the HE region")
            jetHorn_nConst_Cut = ~is_he | (jets.nConstituents > 3)

        base_jet_selection = (
            jet_pt_cut
            & pass_jet_id
            & pass_jet_puid
            & clean
            & jetHorn_PUID_cut
            & jetHorn_ptcut
            & jetHorn_nConst_Cut
            & rawFactor_cut
            & (abs_eta < self.config["jet_eta_cut"])
        )

        HE_HF_ptcut = all_pass
        if add_hehf_ptcut:
            HE_HF_ptcut = ~is_hehf | (jets.pt > add_hehf_ptcut)
        elif add_hehf_asym:
            thr_lead, thr_sub = add_hehf_asym
            logger.debug(
                f"Applying asymmetric jet pT cut for HE/HF jets (|eta|>2.5): "
                f"leading>{thr_lead} GeV, subleading>{thr_sub} GeV"
            )
            # lead/sub = first/second jet, in collection order (= output jet1/jet2), among jets passing
            # every other jet cut. Removing a failing lead/sub promotes the next jet, so re-check it;
            # each pass only removes jets, 4 passes cover any realistic event.
            for _ in range(4):
                sel = base_jet_selection & HE_HF_ptcut
                rank = ak.argsort(ak.argsort(ak.where(sel, 0, 1), axis=1), axis=1)  # stable sort
                is_lead = sel & (rank == 0)
                is_sub = sel & (rank == 1)
                fails = is_hehf & ((is_lead & (jets.pt <= thr_lead)) | (is_sub & (jets.pt <= thr_sub)))
                HE_HF_ptcut = HE_HF_ptcut & ~fails

        jet_selection = base_jet_selection & HE_HF_ptcut

        jets = jets[jet_selection] # INFO: this causes huuuuge memory overflow close to 100 GB. Without it, it goes to around 20 GB

        ifpySR = self.config["switches"]["do_use_pySR_score"]
        if ifpySR:
            jets = ensure_symbolic_features(jets, self.pysr_all_features)
            pysr_dict, pysr_region = build_pysr_pu_masks(jets, self.pysr_configs)
            jets = jets[pysr_dict["jet_pysr_pu_pass"]]

        if_pu_dnn = self.config["switches"].get("do_use_pu_dnn_score", False)
        if if_pu_dnn:
            jets = add_dphi_met_jet_features(jets, events.PuppiMET.phi)
            pu_dnn_out = eval_pu_dnn(jets, self.pu_dnn_configs)
            jets = jets[pu_dnn_out["pu_dnn_pass"]]

        jets = ak.to_packed(jets)

        # apply jetpuid if not have done already
        if is_mc and (variation=="nominal") and is_run2(year) and hasattr(jets, "puId"): # INFO: Skip jet PUID for Run3 samples as they don't have puid yet
            logger.debug("Applying jet PUID scale factors and adding jetpuid wgt!")
            jetpuid_weight = get_jetpuid_weights_eta_dependent(year, jets, self.config) # FIXME
            # FIXME: we should get the weight for each jet and multiply them together.
            weights.add("jetpuid",
                    weight=jetpuid_weight,
            )
        else:
            logger.debug(f"Skipping jet PUID SFs for variation: {variation}, is_mc: {is_mc}, dnn_year: {dnn_year}")

        njets = ak.num(jets, axis=1)

        # ------------------------------------------------------------#
        # Pick VBF jet pair with different criteria
        # 1. Pick two leading jets (as we are doing it)
        # 2. Pick the two jets with highest di-jet invariant mass
        # 3. Pick the two jets with highes pseudo-rapidity gap
        # 4. Pick thw two jets wich highest di-jet invariant mass that passes the critera dEta(j, j) > 2.5
        # ------------------------------------------------------------#
        pair_dict = pick_vbf_pairs(jets)

        # ------------------------------------------------------------#
        # Fill jet-related variables
        # ------------------------------------------------------------#
        padded_jets = ak.pad_none(jets, target=4) # padd jets
        jet1, jet2 = pair_dict["lead"]

        jet_loop_out_dict = {}

        save_four_jets_kinematics = self.config["switches"]["save_four_jets_kinematics"]
        if save_four_jets_kinematics:
            jet3 = padded_jets[:,2]
            jet4 = padded_jets[:,3]

        # switch (switches yaml) or legacy config["output"]; both absent -> off
        save_alt_vbf_pairs = (
            self.config["switches"].get("save_alt_vbf_pairs", False)
            or self.config.get("output", {}).get("save_alt_vbf_pairs", False)
        )
        if variation == "nominal" and save_alt_vbf_pairs:
            for tag, (j1, j2, i1, i2) in pair_dict["pairs_by_method"].items():
                jj = j1 + j2
                jet_loop_out_dict.update({
                    f"vbf_{tag}_jet1_pt_{variation}":   j1.pt,
                    f"vbf_{tag}_jet1_eta_{variation}":  j1.eta,
                    f"vbf_{tag}_jet1_phi_{variation}":  j1.phi,
                    f"vbf_{tag}_jet2_pt_{variation}":   j2.pt,
                    f"vbf_{tag}_jet2_eta_{variation}":  j2.eta,
                    f"vbf_{tag}_jet2_phi_{variation}":  j2.phi,
                    f"vbf_{tag}_mjj_{variation}":       jj.mass,
                    f"vbf_{tag}_deta_{variation}":      np.abs(j1.eta - j2.eta),
                    # pT-order positions in the selected-jet collection (truth matching, pair identity)
                    f"vbf_{tag}_jet1_idx_{variation}":  i1,
                    f"vbf_{tag}_jet2_idx_{variation}":  i2,
                })
                if hasattr(jets, "btagPNetQvG"):
                    jet_loop_out_dict[f"vbf_{tag}_jet1_btagPNetQvG_{variation}"] = j1.btagPNetQvG
                    jet_loop_out_dict[f"vbf_{tag}_jet2_btagPNetQvG_{variation}"] = j2.btagPNetQvG
                if is_mc and variation == "nominal":
                    jet_loop_out_dict.update({
                        f"vbf_{tag}_jet1_hasMatchedGenJet_{variation}": j1.genJetIdx != -1,
                        f"vbf_{tag}_jet2_hasMatchedGenJet_{variation}": j2.genJetIdx != -1,
                    })

            jet_loop_out_dict[f"vbf_maxmjj_deta25_hasPair_{variation}"] = pair_dict["has_mjj_deta"]

            # Truth for the pair study, two definitions, each with the selected jet matched to every parton (dR<0.4),
            # via the jet's GenJet and directly:
            #   lhe : outgoing LHE partons (status 1, quark/gluon, pT>0: pT=0 rows have eta=+-1e11), every generator
            #   genq: outgoing hard-process quarks after the shower (GenPart fromHardProcess & isLastCopy, pT>0 drops
            #         the incoming partons); Pythia only - Herwig NanoAOD has no hard-process flags, so it stays empty
            if is_mc and hasattr(events, "GenJet"):
                gidx = jets.genJetIdx
                gjet = events.GenJet[ak.mask(gidx, (gidx >= 0) & (gidx < ak.num(events.GenJet, axis=1)))]
                truth = {}
                if hasattr(events, "LHEPart"):
                    lhe = events.LHEPart
                    truth["lhe"] = lhe[(lhe.status == 1) & ((np.abs(lhe.pdgId) <= 5) | (lhe.pdgId == 21)) & (lhe.pt > 0)]
                if hasattr(events, "GenPart"):
                    gp = events.GenPart
                    truth["genq"] = gp[(np.abs(gp.pdgId) <= 5) & gp.hasFlags("fromHardProcess", "isLastCopy") & (gp.pt > 0)]
                for src, partons in truth.items():
                    partons = partons[ak.argsort(partons.pt, axis=1, ascending=False)]
                    match = {}
                    for chain, obj in (("reco", jets), ("gen", gjet)):
                        cp = ak.cartesian({"q": partons, "j": obj}, axis=1, nested=True)
                        dr = cp.q.delta_r(cp.j)  # [event][parton][selected jet]; None for jets without GenJet
                        imin = ak.argmin(dr, axis=2)
                        match[chain] = ak.fill_none(ak.where(ak.min(dr, axis=2) < 0.4, imin, -1), -1)
                    jet_loop_out_dict[f"vbf_n{src}_quarks"] = ak.sum(np.abs(partons.pdgId) <= 5, axis=1)
                    padded_q = ak.pad_none(partons, 4, clip=True)
                    padded_m = {k: ak.pad_none(v, 4, clip=True) for k, v in match.items()}
                    for i in range(4):
                        for f in ("pt", "eta", "phi", "pdgId"):
                            jet_loop_out_dict[f"vbf_{src}{i + 1}_{f}"] = padded_q[:, i][f]
                        for chain in ("reco", "gen"):
                            jet_loop_out_dict[f"vbf_{src}{i + 1}_jetidx_{chain}"] = ak.fill_none(padded_m[chain][:, i], -1)

        dijet = jet1+jet2

        jj_dEta = abs(jet1.eta - jet2.eta)
        jj_dPhi = abs(jet1.delta_phi(jet2))
        mmj1_dEta = abs(dimuon.eta - jet1.eta)
        mmj2_dEta = abs(dimuon.eta - jet2.eta)

        min_dEta_filter  = ak.fill_none((mmj1_dEta < mmj2_dEta), value=True)
        mmj_min_dEta = ak.where(
            min_dEta_filter,
            mmj1_dEta,
            mmj2_dEta,
        )

        mmj1_dPhi = abs(dimuon.delta_phi(jet1))
        mmj2_dPhi = abs(dimuon.delta_phi(jet2))
        mmj1_dR = dimuon.delta_r(jet1)
        mmj2_dR = dimuon.delta_r(jet2)

        min_dPhi_filter = ak.fill_none((mmj1_dPhi < mmj2_dPhi), value=True)
        mmj_min_dPhi = ak.where(
            min_dPhi_filter,
            mmj1_dPhi,
            mmj2_dPhi,
        )

        # zeppenfeld definition in  line 1118 in the AN
        dimuon_rapidity = getRapidity(dimuon)
        jet1_rapidity = getRapidity(jet1)
        jet2_rapidity = getRapidity(jet2)
        save_four_jets_kinematics = self.config["switches"]["save_four_jets_kinematics"]
        if save_four_jets_kinematics:
            jet3_rapidity = getRapidity(jet3)
            jet4_rapidity = getRapidity(jet4)
        zeppenfeld = dimuon_rapidity - 0.5 * (jet1_rapidity + jet2_rapidity)
        zeppenfeld = zeppenfeld / np.abs(jet1_rapidity - jet2_rapidity)
        mmjj = dimuon + dijet

        rpt = mmjj.pt / (
            dimuon.pt + jet1.pt + jet2.pt
        )

        # pt_centrality formula is in eqn A.1 fron AN-19-124
        pt_centrality = dimuon.pt - abs(jet1.pt + jet2.pt)/2
        pt_centrality = pt_centrality / abs(jet1.pt - jet2.pt)

        if ifpySR:
            for region in pysr_region:
                padded_score = ak.pad_none(pysr_dict[f"jet_pysr_{region}_score"], 4)
                padded_pass  = ak.pad_none(pysr_dict[f"jet_pysr_{region}_pass"], 4)

                jet_loop_out_dict.update({
                    f"jet1_pysr_{region}_score_{variation}": padded_score[:, 0],
                    f"jet2_pysr_{region}_score_{variation}": padded_score[:, 1],
                    f"jet1_pysr_{region}_pass_{variation}":  padded_pass[:, 0],
                    f"jet2_pysr_{region}_pass_{variation}":  padded_pass[:, 1],
                })
                if save_four_jets_kinematics:
                    jet_loop_out_dict.update({
                        f"jet3_pysr_{region}_score_{variation}": padded_score[:, 2],
                        f"jet4_pysr_{region}_score_{variation}": padded_score[:, 3],
                        f"jet3_pysr_{region}_pass_{variation}":  padded_pass[:, 2],
                        f"jet4_pysr_{region}_pass_{variation}":  padded_pass[:, 3],
                    })

        jet_loop_out_dict.update({
            f"jet1_mass_{variation}": jet1.mass,
            f"jet1_pt_{variation}": jet1.pt,
            f"jet1_eta_{variation}": jet1.eta,
            f"jet1_phi_{variation}": jet1.phi,
            f"jet1_puId_{variation}": get_puId(jet1),
            # -------------------------
            f"jet2_mass_{variation}": jet2.mass,
            f"jet2_pt_{variation}": jet2.pt,
            f"jet2_eta_{variation}": jet2.eta,
            f"jet2_phi_{variation}": jet2.phi,
            f"jet2_puId_{variation}": get_puId(jet2),
            # -------------------------
            f"jj_mass_{variation}": dijet.mass,
            f"jj_mass_log_{variation}": np.log(dijet.mass),
            f"jj_dEta_{variation}": jj_dEta,
            f"jj_dPhi_{variation}": jj_dPhi,
            f"mmj_min_dEta_{variation}": mmj_min_dEta,
            f"mmj_min_dPhi_{variation}": mmj_min_dPhi,
            f"rpt_{variation}": rpt,
            f"pt_centrality_{variation}": pt_centrality,
            f"ll_zstar_log_{variation}": np.log(np.abs(zeppenfeld)),
            f"zeppenfeld_{variation}": zeppenfeld,
            f"njets_{variation}": njets,
        })

        jet_loop_out_dict.update(
            {
                f"jet1_rapidity_{variation}": jet1_rapidity,  # max rel err: 0.7394
                f"jet2_rapidity_{variation}": jet2_rapidity,  # max rel err: 0.781
                f"jj_pt_{variation}": dijet.pt,
                f"jj_eta_{variation}": dijet.eta,
                f"jj_phi_{variation}": dijet.phi,
                f"mmj1_dEta_{variation}": mmj1_dEta,
                f"mmj1_dPhi_{variation}": mmj1_dPhi,
                f"mmj1_dR_{variation}": mmj1_dR,
                f"mmj2_dEta_{variation}": mmj2_dEta,
                f"mmj2_dPhi_{variation}": mmj2_dPhi,
                f"mmj2_dR_{variation}": mmj2_dR,
                f"mmjj_pt_{variation}": mmjj.pt,
                f"mmjj_eta_{variation}": mmjj.eta,
                f"mmjj_phi_{variation}": mmjj.phi,
                f"mmjj_mass_{variation}": mmjj.mass,
            }
        )

        if save_four_jets_kinematics:
            jet_loop_out_dict.update(
                {
                    f"jet3_pt_{variation}": jet3.pt,
                    f"jet3_eta_{variation}": jet3.eta,
                    f"jet3_rapidity_{variation}": jet3_rapidity,
                    f"jet3_phi_{variation}": jet3.phi,
                    f"jet3_mass_{variation}": jet3.mass,
                    # -------------------------
                    f"jet4_pt_{variation}": jet4.pt,
                    f"jet4_eta_{variation}": jet4.eta,
                    f"jet4_rapidity_{variation}": jet4_rapidity,
                    f"jet4_phi_{variation}": jet4.phi,
                    f"jet4_mass_{variation}": jet4.mass,
                }
            )

            jet_loop_out_dict.update({
                f"jet3_puId_{variation}": get_puId(jet3),
                f"jet4_puId_{variation}": get_puId(jet4),
            })

        if is_mc and variation == "nominal":
            jet_loop_out_dict.update({
                f"jet1_hasMatchedGenJet_{variation}": jet1.genJetIdx != -1,
                f"jet2_hasMatchedGenJet_{variation}": jet2.genJetIdx != -1,
            })
            if save_four_jets_kinematics:
                jet_loop_out_dict.update({
                    f"jet3_hasMatchedGenJet_{variation}": jet3.genJetIdx != -1,
                    f"jet4_hasMatchedGenJet_{variation}": jet4.genJetIdx != -1,
                })

        if self.config["switches"].get("save_extended_jet_vars", True):
            jet_loop_out_dict.update(
                build_extended_jet_vars(jets, pair_dict, dimuon, mu1, mu2, events.PuppiMET.phi, variation)
            )

        if self.config["switches"].get("save_vbf_dnn_derived_vars", True):
            family = "nominal" if variation == "nominal" else "jet"
            jet_loop_out_dict.update(
                build_vbf_dnn_derived_vars(dimuon, mu1, mu2, jet1, jet2, variation, family, year)
            )
            # muon pT variations x nominal jets (same condition as the muon variation block)
            if (variation == "nominal" and is_mc
                    and self.config["switches"].get("save_muon_roccor_variations", True)):
                for suffix, field in MUON_PT_VARIATION_FIELDS.items():
                    if not (hasattr(mu1, field) and hasattr(mu2, field)):
                        continue
                    mu1_var = ak.with_field(mu1, mu1[field], "pt")
                    mu2_var = ak.with_field(mu2, mu2[field], "pt")
                    jet_loop_out_dict.update(build_vbf_dnn_derived_vars(
                        mu1_var + mu2_var, mu1_var, mu2_var, jet1, jet2, suffix, "mu", year
                    ))

        # ------------------------------------------------------------------
        # Add additional Jet NanoAOD variables for leading 4 jets
        # (only if the branches exist in this NanoAOD)
        # ------------------------------------------------------------------
        jet_vars = [
                "qgl", "btagPNetQvG", "btagDeepFlavQG",
                "btagRobustParTAK4QG", "btagUParTAK4QvG",
                "btagPNetB",
                "btagUParTAK4B",
                "btagDeepB",
                # Needed to check do_reject_high_rawFactor_jets' rare L2Relative
                # asymptotic-JEC mitigation (rejects Jet_rawFactor > 0.9) even when
                # that switch (do_reject_high_rawFactor_jets) is off
                "rawFactor",
                ]
        if self.config["switches"]["do_add_jet_ID_vars"]:
            jet_vars += [
                "area",
                "jetId",
                "chEmEF", "chHEF", "neEmEF", "neHEF", "muEF",
                "hfEmEF", "hfHEF",
                "muonSubtrDeltaEta", "muonSubtrDeltaPhi",
                "chMultiplicity", "neMultiplicity",
                "nConstituents", "nElectrons", "nMuons",
                "hadronFlavour", "partonFlavour",
                "hfcentralEtaStripSize", "hfadjacentEtaStripsSize",
                "hfsigmaEtaEta", "hfsigmaPhiPhi",
                "muonSubtrFactor",
                "puIdDisc",
            ]
        jets_to_process = [jet1, jet2, jet3, jet4] if save_four_jets_kinematics else [jet1, jet2]
        for i, jet in enumerate(jets_to_process, start=1):
            for var in jet_vars:
                val = getattr(jet, var, None)  # only branches that exist in this NanoAOD
                if val is not None:
                    jet_loop_out_dict[f"jet{i}_{var}_{variation}"] = val

        # ------------------------------------------------------------#
        # Fill soft activity jet variables
        # ------------------------------------------------------------#

        # Computed per JES/JER variation on purpose: the soft-activity footprint removal uses the first two
        # selected jets (dR < 0.4, AN-19-124 l.2966), and JES/JER can change which jets pass the
        # selection (hence which two come first), so these values do move with the jet variation.
        nmuons = ak.num(events.Muon, axis=1) # FIXME (I think it should be selected muons)
        for cutout in [2, 5]:
            sj_out = fill_softjets_HIG19006(events, jets, mu1, mu2, nmuons, cutout)
            jet_loop_out_dict.update({f"{key}_{variation}": val for key, val in sj_out.items()})

        # ------------------------------------------------------------#
        # Calculate QGL weights
        # ------------------------------------------------------------#
        if is_mc and (variation == "nominal") and (self.config["switches"]["do_qgl_wgt"]):
            isHerwig = "herwig" in dataset
            logger.debug("adding QGL weights!")

            qgl_wgts = qgl_weights_V2(jets, self.config, isHerwig, year)
            weights.add("qgl",
                        weight=qgl_wgts["nom"],
                        weightUp=qgl_wgts["up"],
                        weightDown=qgl_wgts["down"]
            )

        # ------------------------------------------------------------#
        # btag SF and apply btag veto
        # ------------------------------------------------------------#
        do_btag_wgt = (
            is_mc
            and (variation == "nominal")
            and self.config["switches"]["do_btag_wgt"]
        )
        if do_btag_wgt and is_run3(year):
            # BTV fixedWP SFs for the L+M working points the b-veto uses (needs MC eff maps)
            logger.debug("doing Run3 fixedWP btag wgt!")
            btag_wgt, btag_syst = btag_fixedwp_weights(btag_jets, self.config, year)
            weights.add("btag", weight=btag_wgt)
            safe_btag_wgt = ak.where(btag_wgt != 0, btag_wgt, 1.0)
            for name, (up, down) in btag_syst.items():
                weights.add(f"btagWP_{name}",
                    weight=ak.ones_like(btag_wgt),
                    weightUp=up / safe_btag_wgt,
                    weightDown=down / safe_btag_wgt,
                )
        elif do_btag_wgt:  # Run 2: deepJet shape SFs
            btag_eta_val = 2.4 if str(year).startswith("2016") else 2.5
            logger.debug("doing btag wgt!")
            bjet_sel_mask = ak.ones_like(ak.num(btag_jets, axis=1))
            btag_systs = self.config["btag_systs"]
            if "RERECO" in year:
                btag_json = BTagScaleFactor(
                    self.config["btag_sf_csv"],
                    BTagScaleFactor.RESHAPE,
                    "iterativefit,iterativefit,iterativefit",
                )
            else:
                btag_file = get_corrset(self.config["btag_sf_json"])
                available_keys = list(btag_file.keys())
                logger.debug(f"Available btag correction keys: {available_keys}")

                if "deepJet_shape" not in available_keys:
                    raise KeyError(
                        f"deepJet_shape not found in {self.config['btag_sf_json']}. "
                        f"Available keys: {available_keys}"
                    )

                btag_json=btag_file["deepJet_shape"]
                logger.debug("Using btag correction key: deepJet_shape")

            btag_wgt, btag_syst = btag_weights_jsonKeepDim(
                        self, btag_systs, btag_jets, btag_eta_val, weights, bjet_sel_mask, btag_json
            )
            weights.add("btag",
                    weight=btag_wgt,
            )
            # --- Btag weights variations --- #
            for name, bs in btag_syst.items():
                logger.debug(f"{name} value: {bs}")
                weights.add(f"btag_{name}",
                    weight=ak.ones_like(btag_wgt),
                    weightUp=bs["up"],
                    weightDown=bs["down"]
                )

        btagLoose_filter = ak.zeros_like(btag_jets.pt, dtype="bool")
        btagMedium_filter = ak.zeros_like(btag_jets.pt, dtype="bool")

        if "RERECO" in year:
            btagLoose_filter = btag_jets.btagDeepB > self.config["btag_loose_wp"]
            btagMedium_filter = btag_jets.btagDeepB > self.config["btag_medium_wp"]

        if is_run2(year) and NanoAODv == 12 and hasattr(btag_jets, "btagDeepB"):
            logger.debug("Using btagDeepB btag!")
            btagLoose_filter = btag_jets.btagDeepB > self.config["btag_loose_wp_DeepCSV"] # FIXME: check the var name and score
            btagMedium_filter = btag_jets.btagDeepB > self.config["btag_medium_wp_DeepCSV"]

        if is_run2(year) and NanoAODv == 15 and hasattr(btag_jets, "btagUParTAK4B"):
            logger.debug("Using btagUParTAK4B btag!")
            btagLoose_filter = btag_jets.btagUParTAK4B > self.config["btag_loose_wp_UParT"]
            btagMedium_filter = btag_jets.btagUParTAK4B > self.config["btag_medium_wp_UParT"]

        if is_run3(year) and "RERECO" not in year:
            # tagger by year (ParticleNet 2022-23, UParT 2024-26), same as the btag SFs above
            _, score_field, wp_suffix = btag_fixedwp_tagger(year)
            logger.debug(f"Using {score_field} btag!")
            if not hasattr(btag_jets, score_field):
                # all-False filters would silently disable the b-veto
                raise KeyError(f"b-veto: jets have no {score_field} for {year}")
            btagLoose_filter = btag_jets[score_field] > self.config[f"btag_loose_wp_{wp_suffix}"]
            btagMedium_filter = btag_jets[score_field] > self.config[f"btag_medium_wp_{wp_suffix}"]

        btagLoose_filter = ak.fill_none(btagLoose_filter, value=False)
        btagMedium_filter = ak.fill_none(btagMedium_filter, value=False)

        nBtagLoose = ak.sum(btagLoose_filter, axis=1)
        nBtagMedium = ak.sum(btagMedium_filter, axis=1)

        temp_out_dict = {
            f"nBtagLoose_{variation}": nBtagLoose,
            f"nBtagMedium_{variation}": nBtagMedium,
        }

        jet_loop_out_dict.update(temp_out_dict)

        return jet_loop_out_dict
