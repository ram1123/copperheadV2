# import numpy as np
import awkward as ak
from typing import TypeVar, Tuple
ak_array = TypeVar('ak_array')
coffea_nanoevent = TypeVar('coffea_nanoevent')


def apply_geofit(
    events: coffea_nanoevent,
    year: str,
    opposite_fsr_mask: ak_array
    ):
    """
    params:
    opposite_fsr_mask = boolean mask that is the opposite value of
    the output awkward array from fsr_recovery() function
    if we didn't do fsr_recovery b4hand, this is equivalent to events.Muons
    with False in place of muon objects
    """
    d0_BS_charge = events.Muon.dxybs * events.Muon.charge
    # print(f"apply_geofit ak.sum(opposite_fsr_mask) : {ak.sum(opposite_fsr_mask)}")
    mask = opposite_fsr_mask & (abs(d0_BS_charge) < 999999.0)
    # print(f"apply_geofit ak.sum(mask) : {ak.sum(mask)}")
    # print(f"geofit year: {year}")
    pt = events.Muon.pt
    eta = events.Muon.eta

    cuts = {
        "eta_1": (abs(eta) < 0.9),
        "eta_2": ((abs(eta) < 1.7) & (abs(eta) >= 0.9)),
        "eta_3": (abs(eta) >= 1.7),
    }

    factors = {
        "2016preVFP": {"eta_1": 411.34, "eta_2": 673.40, "eta_3": 1099.0},
        "2016postVFP": {"eta_1": 411.34, "eta_2": 673.40, "eta_3": 1099.0},
        "2016": {"eta_1": 411.34, "eta_2": 673.40, "eta_3": 1099.0},
        "2017": {"eta_1": 582.32, "eta_2": 974.05, "eta_3": 1263.4},
        "2018": {"eta_1": 650.84, "eta_2": 988.37, "eta_3": 1484.6},
        "2016_RERECO": {"eta_1": 411.34, "eta_2": 673.40, "eta_3": 1099.0},
        "2017_RERECO": {"eta_1": 582.32, "eta_2": 974.05, "eta_3": 1263.4},
        "2018_RERECO": {"eta_1": 650.84, "eta_2": 988.37, "eta_3": 1484.6},
    }
    pt_corr = ak.zeros_like(pt)
    for eta_i in ["eta_1", "eta_2", "eta_3"]:
        value = factors[year][eta_i] * d0_BS_charge * pt * pt / 10000.0
        # print(f"apply_geofit value: {value}")
        pt_corr = ak.where(cuts[eta_i], value, pt_corr) # NOTE: pt_corr == pt_roch - pt_gen


    events["Muon", "pt_gf"] = ak.where(mask, pt - pt_corr, pt)
    return mask, pt_corr # return these values for debugging purposes
