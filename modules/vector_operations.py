from typing import Tuple, TypeVar

import awkward as ak
import numpy as np
from coffea.nanoevents.methods import vector

coffea_nanoevent = TypeVar("coffea_nanoevent")
ak_array = TypeVar("ak_array")

def convert_vector_type_4d(vector, vector_name):
    """Convert an Awkward Vector of any type to a specified 4D vector type.
    Args:
        vector: Awkward Vector (e.g. ak.vector.x4, ak.vector.LorentzVector, etc.)
        vector_name: str, name of the target vector type (e.g. 'LorentzVector', 'PtEtaPhiMLorentzVector', etc.)
    Returns:
        Awkward Vector of the specified type.
    """
    new_vector = ak.zip(
        {
            "pt": vector.pt,
            "eta": vector.eta,
            "phi": vector.phi,
            "mass": vector.mass,
            "charge": vector.charge,
        },
        with_name=vector_name,
        behavior=vector.behavior,
    )
    return new_vector


def getRapidity(obj):
    if hasattr(obj, "energy") and hasattr(obj, "pz"):
        e = obj.energy
        pz = obj.pz
    else:
        px = obj.pt * np.cos(obj.phi)
        py = obj.pt * np.sin(obj.phi)
        pz = obj.pt * np.sinh(obj.eta)
        e = np.sqrt(px**2 + py**2 + pz**2 + obj.mass**2)
    rap = 0.5 * np.log((e + pz) / (e - pz))
    return rap

# helper: delta-phi in [-pi, pi]
def _delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi
    return abs(dphi)

def _mass2_kernel(t, x, y, z):
    return t * t - x * x - y * y - z * z


def p4_sum_mass(obj1, obj2):
    result_px = ak.zeros_like(obj1.pt)
    result_py = ak.zeros_like(obj1.pt)
    result_pz = ak.zeros_like(obj1.pt)
    result_e = ak.zeros_like(obj1.pt)
    for obj in [obj1, obj2]:
        # px_ = obj.pt * np.cos(obj.phi)
        # py_ = obj.pt * np.sin(obj.phi)
        # pz_ = obj.pt * np.sinh(obj.eta)
        px_ = obj.px
        py_ = obj.py
        pz_ = obj.pz
        e_ = np.sqrt(px_**2 + py_**2 + pz_**2 + obj.mass**2)
        result_px = result_px + px_
        result_py = result_py + py_
        result_pz = result_pz + pz_
        result_e = result_e + e_
    # result_pt = np.sqrt(result_px**2 + result_py**2)
    # result_eta = np.arcsinh(result_pz / result_pt)
    # result_phi = np.arctan2(result_py, result_px)
    result_mass = np.sqrt(result_e**2 - result_px**2 - result_py**2 - result_pz**2)
    return result_mass


def p4_subtract_pt(obj1, obj2):
    """
    obtain the pt vector subtraction of two variables
    """
    result_px = ak.zeros_like(obj1.pt)
    result_py = ak.zeros_like(obj1.pt)
    result_pz = ak.zeros_like(obj1.pt)
    coeff = 1.0
    for obj in [obj1, obj2]:
        # px_ = obj.pt * np.cos(obj.phi)
        # py_ = obj.pt * np.sin(obj.phi)
        # pz_ = obj.pt * np.sinh(obj.eta)
        px_ = obj.px
        py_ = obj.py
        pz_ = obj.pz
        result_px = result_px + coeff * px_
        result_py = result_py + coeff * py_
        result_pz = result_pz + coeff * pz_
        coeff = -1 * coeff  # switch coeff
    result_pt = np.sqrt(result_px**2 + result_py**2)
    return result_pt


# Dmitry's implementation of delta_r
def delta_r_V1(eta1, eta2, phi1, phi2):
    deta = abs(eta1 - eta2)
    dphi = abs(np.mod(phi1 - phi2 + np.pi, 2 * np.pi) - np.pi)
    dr = np.sqrt(deta**2 + dphi**2)
    return deta, dphi, dr


def etaFrame_variables(mu1: coffea_nanoevent, mu2: coffea_nanoevent) -> Tuple[ak_array]:
    """
    Obtain eta frame cos(theta) and phi as specified in:
    https://link.springer.com/article/10.1140/epjc/s10052-011-1600-y and
    This Eta frame values supposedly plays a similar role to CS frame in terms of physics
    sensitivity, but with better resolution. Not nocessarily believe this claim
    however.
    """
    # divide muons in terms of negative and positive charges instead of leading pT
    mu_neg = ak.where((mu1.charge < 0), mu1, mu2)
    mu_pos = ak.where((mu1.charge > 0), mu1, mu2)
    dphi = abs(mu_neg.delta_phi(mu_pos))
    theta_eta = np.arccos(np.tanh((mu_neg.eta - mu_pos.eta) / 2))
    phi_eta = np.tan((np.pi - np.abs(dphi)) / 2) * np.sin(theta_eta)
    return np.cos(theta_eta), phi_eta


def cs_variables(mu1: coffea_nanoevent, mu2: coffea_nanoevent) -> Tuple[ak_array]:
    """
    return cos(theta) and phi in collins-soper frame
    """
    dimuon = mu1 + mu2
    cos_theta_cs = getCosThetaCS(mu1, mu2, dimuon)
    phi_cs = getPhiCS(mu1, mu2, dimuon)
    return cos_theta_cs, phi_cs


def getCosThetaCS(
    mu1: coffea_nanoevent,
    mu2: coffea_nanoevent,
    dimuon: coffea_nanoevent,
) -> ak_array:
    """
    return cos(theta) in collins-soper frame
    the formula for cos(theta) is given in Eqn 1. of https://www.ciemat.es/portal.do?TR=A&IDR=1&identificador=813
    """
    dimuon_pt = dimuon.pt
    dimuon_mass = dimuon.mass
    nominator = 2 * (mu1.pz * mu2.energy - mu2.pz * mu1.energy)
    demoninator = dimuon_mass * (dimuon_mass**2 + dimuon_pt**2) ** (0.5)
    cos_theta_cs = -(
        nominator / demoninator
    )  # add negative sign to match the sign on pisa implementation at https://github.com/green-cabbage/copperhead_fork2/blob/Run3/python/math_tools.py#L152-L223
    return cos_theta_cs


def getPhiCS(
    mu1: coffea_nanoevent,
    mu2: coffea_nanoevent,
    dimuon: coffea_nanoevent,
) -> ak_array:
    """
    return phi in collins-soper frame
    the formula for phi is given in Eqn F.8 of https://people.na.infn.it/~elly/TesiAtlas/SpinCP/TestIpotesi/CollinSoperDefinition.pdf
    the implementation is heavily inspired from https://github.com/JanFSchulte/SUSYBSMAnalysis-Zprime2muAnalysis/blob/mini-AOD-2018/src/AsymFunctions.C#L1549-L1603
    """
    mu_neg = ak.where((mu1.charge < 0), mu1, mu2)
    mu_pos = ak.where((mu1.charge > 0), mu1, mu2)
    dimuon_pz = dimuon.pz
    dimuon_pt = dimuon.pt
    dimuon_mass = dimuon.mass
    beam_vec_z = ak.where(
        (dimuon_pz > 0), ak.ones_like(dimuon_pz), -ak.ones_like(dimuon_pz)
    )
    # intialize beam vector as threevector to do cross product
    # beam_vec =  ak.zip(
    #     {
    #         "x": ak.zeros_like(dimuon_pz),
    #         "y": ak.zeros_like(dimuon_pz),
    #         "z": beam_vec_z,
    #     },
    #     with_name="ThreeVector",
    #     behavior=vector.behavior,
    # )
    # logger.info(f"vector.__file__: {vector.__file__}")
    beam_vec = ak.zip(
        {
            "x": ak.zeros_like(dimuon_pz),
            "y": ak.zeros_like(dimuon_pz),
            "z": beam_vec_z,
        },
        with_name="Momentum3D",
        behavior=vector.behavior,
    )
    # apply cross product. note x,y,z of dimuon refers to its momentum, NOT its location
    # mu.px == mu.x, mu.py == mu.y and so on
    dimuon3D_vec = ak.zip(
        {"x": dimuon.x, "y": dimuon.y, "z": dimuon.z},
        with_name="Momentum3D",
        behavior=vector.behavior,
    )
    R_T = beam_vec.cross(
        dimuon3D_vec
    )  # direct cross product with dimuon doesn't work bc it's a 5D vector with x,y,z,t and charge

    R_T = R_T.unit()  # make it a unit vector
    Q_T = dimuon
    Q_coeff = (
        ((dimuon_mass * dimuon_mass + (dimuon_pt * dimuon_pt))) ** (0.5)
    ) / dimuon_mass
    delta_T_dot_R_T = (mu_neg.px - mu_pos.px) * R_T.x + (mu_neg.py - mu_pos.py) * R_T.y
    delta_R_term = delta_T_dot_R_T
    delta_R_term = (
        -delta_R_term
    )  # add negative sign to match the sign on pisa implementation at https://github.com/green-cabbage/copperhead_fork2/blob/Run3/python/math_tools.py#L152-L223
    delta_T_dot_Q_T = (mu_neg.px - mu_pos.px) * Q_T.px + (
        mu_neg.py - mu_pos.py
    ) * Q_T.py
    delta_T_dot_Q_T = (
        -delta_T_dot_Q_T
    )  # add negative sign to match the sign on pisa implementation at https://github.com/green-cabbage/copperhead_fork2/blob/Run3/python/math_tools.py#L152-L223
    delta_Q_term = delta_T_dot_Q_T
    delta_Q_term = (
        delta_Q_term / dimuon_pt
    )  # normalize since Q_T should techincally be a unit vector
    phi_cs = np.arctan2(Q_coeff * delta_R_term, delta_Q_term)
    return phi_cs
