import numpy as np
from scipy.special import erfinv, erf
from random import random
import awkward as ak
import correctionlib.schemav2 as cs

"""
This file is taken from https://gitlab.cern.ch/cms-muonPOG/muonscarekit/-/blob/master/scripts/MuonScaRe.py?ref_type=heads

Full credits go to KIT and the authors of the script

This script has been altered to match dask-awkward workflow
"""



class CrystallBall:

    def __init__(self, m, s, a, n):
        self.pi = 3.14159
        self.sqrtPiOver2 = np.sqrt(self.pi/2.0)
        self.sqrt2 = np.sqrt(2.0)
        self.m = m
        self.s = s
        self.a = a
        self.n = n
        self.fa = abs(self.a)
        self.ex = np.exp(-self.fa * self.fa/2)
        self.A  = (self.n/self.fa)**self.n * self.ex
        self.C1 = self.n/self.fa/(self.n-1) * self.ex
        self.D1 = 2 * self.sqrtPiOver2 * erf(self.fa/self.sqrt2)

        self.B = self.n/self.fa - self.fa
        self.C = (self.D1 + 2 * self.C1)/self.C1
        self.D = (self.D1 + 2 * self.C1)/2

        self.N = 1.0/self.s/(self.D1 + 2 * self.C1)
        self.k = 1.0/(self.n - 1)

        self.NA = self.N * self.A
        self.Ns = self.N * self.s
        self.NC = self.Ns * self.C1
        self.F = 1 - self.fa * self.fa/self.n
        self.G = self.s * self.n/self.fa
        self.cdfMa = self.cdf(self.m - self.a * self.s)
        self.cdfPa = self.cdf(self.m + self.a * self.s)


    def cdf(self, x):
        d = (x - self.m)/self.s
        result = ak.full_like(d, 1.0)
        
        # define different conditions
        c1a = (d<-self.a) & (self.F - self.s * d / self.G > 0)
        c1b = (d<-self.a) & (self.F - self.s * d / self.G <= 0)
        c2a = (d>self.a) & (self.F + self.s * d / self.G > 0)
        c2b = (d>self.a) & (self.F + self.s * d / self.G <= 0)

        c3 = ~c1a & ~c1b & ~c2a & ~c2b
        # For d < -a
        result = ak.where(c1a, 
                        self.NC / np.power(self.F - self.s * d / self.G, self.n - 1), 
                        result)
        result = ak.where(c1b, 
                        self.NC, 
                        result)

        # For d > a
        result = ak.where(c2a, 
                        self.NC * (self.C - np.power(self.F + self.s * d / self.G, 1 - self.n)), 
                        result)
        result = ak.where(c2b, 
                        self.NC * self.C, 
                        result)

        # For -a <= d <= a
        result = ak.where(c3, 
                        self.Ns * (self.D - self.sqrtPiOver2 * erf(-d / self.sqrt2)), 
                        result)

        return result


    def invcdf(self, u):
        result = ak.zeros_like(u)

        c1a = (u < self.cdfMa) & (self.NC/u > 0)
        c1b = (u < self.cdfMa) & (self.NC/u <= 0)
        c2a = (u > self.cdfPa) &  (self.C - u/self.NC > 0)
        c2b = (u > self.cdfPa) &  (self.C - u/self.NC <= 0)
        c3 = ~c1a & ~c1b & ~c2a & ~c2b

        # For u < cdfMa
        result = ak.where(c1a, 
                        self.m + self.G * (self.F - (self.NC / u) ** self.k), 
                        result)
        result = ak.where(c1b, 
                        self.m + self.G * self.F, 
                        result)

        # For u > cdfPa
        result = ak.where(c2a, 
                        self.m - self.G * (self.F - (self.C - u / self.NC) ** (-self.k)), 
                        result)
        result = ak.where(c2b, 
                        self.m - self.G * self.F, 
                        result)

        # For cdfMa <= u <= cdfPa
        result = ak.where(c3, 
                        self.m - self.sqrt2 * self.s * erfinv((self.D - u / self.Ns) / self.sqrtPiOver2), 
                        result)

        return result


def generateRandomVals(eta, nL, extra_rndm_seed):
    resrng = cs.Correction(
        name="resrng",
        description="Deterministic smearing value generator",
        version=1,
        inputs=[
            cs.Variable(name="seed1", type="real", description="seed1"),
            cs.Variable(name="seed2", type="real", description="seed2"),
            cs.Variable(name="seed3", type="real", description="seed3"),
        ],
        output=cs.Variable(name="rng", type="real"),
        data=cs.HashPRNG(
            nodetype="hashprng",
            inputs=["seed1", "seed2", "seed3"],
            distribution="stdflat",
        )
    )
    rand_vals = resrng.to_evaluator().evaluate(
        eta,
        nL,
        extra_rndm_seed,
    )
    # print(f"rand_vals: {rand_vals.compute()}")
    return rand_vals

def get_rndm(eta, nL, cset, extra_rndm_seed, nested=False):
    # obtain parameters from correctionlib
    # if nested:
    #     eta_f, nL_f, nmuons = ak.flatten(eta), ak.flatten(nL), ak.num(nL)
    # else:
    #     eta_f, nL_f, nmuons = eta, nL, np.ones_like(eta)
    eta_f, nL_f, nmuons = eta, nL, ak.ones_like(eta)
    
    mean_f = cset.get("cb_params").evaluate(abs(eta_f), nL_f, 0)
    sigma_f = cset.get("cb_params").evaluate(abs(eta_f), nL_f, 1)
    n_f = cset.get("cb_params").evaluate(abs(eta_f), nL_f, 2)
    alpha_f = cset.get("cb_params").evaluate(abs(eta_f), nL_f, 3)

    # get random number following the CB
    # print(nmuons)
    # rndm_f = [random() for i in nmuons for j in range(int(i))]
    rndm_f = generateRandomVals(eta, nL, extra_rndm_seed)
    # print(f"nmuons: {ak.num(nL, axis=1).compute()}")

    cb_f = CrystallBall(mean_f, sigma_f, alpha_f, n_f)

    result_f = cb_f.invcdf(rndm_f)

    # if nested:
    #     result = ak.unflatten(result_f, nmuons)
    # else:
    #     result = result_f
    result = result_f

    return result


def get_std(pt, eta, nL, cset, nested=False):
    # if nested:
    #     eta_f, nL_f, pt_f, nmuons = ak.flatten(eta), ak.flatten(nL), ak.flatten(pt), ak.num(nL)
    # else:
    #     eta_f, nL_f, pt_f, nmuons = eta, nL, pt, 1
    eta_f, nL_f, pt_f, nmuons = eta, nL, pt, 1

    # obtain parameters from correctionlib    
    param0_f = cset.get("poly_params").evaluate(abs(eta_f), nL_f, 0)
    param1_f = cset.get("poly_params").evaluate(abs(eta_f), nL_f, 1)
    param2_f = cset.get("poly_params").evaluate(abs(eta_f), nL_f, 2)

    # calculate value and return max(0, val)
    sigma_f = param0_f + param1_f * pt_f + param2_f * pt_f*pt_f
    sigma_corrected_f = ak.where(sigma_f < 0, 0, sigma_f)

    # if nested:
    #     result = ak.unflatten(sigma_corrected_f, nmuons)
    # else:
    #     result = sigma_corrected_f
    result = sigma_corrected_f

    return result


def get_k(eta, var, cset, nested=False):
    # if nested:
    #     eta_f, nmuons = ak.flatten(eta), ak.num(eta)
    # else:
    #     eta_f = eta
    eta_f = eta

    # obtain parameters from correctionlib
    k_data_f = cset.get("k_data").evaluate(abs(eta_f), var)
    k_mc_f = cset.get("k_mc").evaluate(abs(eta_f), var)

    # calculate residual smearing factor 
    # return 0 if smearing in MC already larger than in data
    k_f = ak.zeros_like(k_data_f)
    condition = k_mc_f<k_data_f
    # k_f[condition] = (k_data_f[condition]**2 - k_mc_f[condition]**2)**.5
    k_f_condition = (k_data_f**2 - k_mc_f**2)**.5
    k_f = ak.where(condition, k_f_condition, k_f)
    

    # if nested:
    #     result = ak.unflatten(k_f, nmuons)
    # else:
    #     result = k_f
    result = k_f

    return result


def filter_boundaries(pt_corr, pt, nested):
    if not nested:
        pt_corr = np.asarray(pt_corr)
        pt = np.asarray(pt)

    # Check for pt values outside the range of [26, 200]
    outside_bounds = (pt < 26) | (pt > 200)

    if nested:
        n_pt_outside = ak.sum(ak.any(outside_bounds, axis=-1))
    else:
        n_pt_outside = np.sum(outside_bounds)

    if n_pt_outside > 0:
        print(
            f"There are {n_pt_outside} events with muon pt outside of [26,200] GeV. "
            "Setting those entries to their initial value."
        )
        pt_corr = ak.where(pt>200, pt, pt_corr)
        pt_corr = ak.where(pt<26, pt, pt_corr)

    # Check for NaN entries in pt_corr
    nan_entries = np.isnan(pt_corr)

    if nested:
        n_nan = ak.sum(ak.any(nan_entries, axis=-1))
        n_nan += ak.sum(ak.is_none(pt_corr, axis=-1)) # Nan and None are considered different in awkward
    else:
        n_nan = np.sum(nan_entries)

    if n_nan > 0:
        print(
            f"There are {n_nan} nan entries in the corrected pt. "
            "This might be due to the number of tracker layers hitting boundaries. "
            "Setting those entries to their initial value."
        )
        pt_corr = ak.where(np.isnan(pt_corr), pt, pt_corr)
        pt_corr = ak.where(ak.is_none(pt_corr), pt, pt_corr)

    return pt_corr


def pt_resol(pt, eta, nL, cset, nested=False):
    """"
    Function for the calculation of the resolution correction
    Input: 
    pt - muon transverse momentum
    eta - muon pseudorapidity
    nL - muon number of tracker layers
    cset - correctionlib object

    This function should only be applied to reco muons in MC!
    """
    extra_rndm_seed = pt
    rndm = get_rndm(eta, nL, cset, extra_rndm_seed, nested)
    # rndm = ak.ones_like(pt)
    std = get_std(pt, eta, nL, cset, nested)
    k = get_k(eta, "nom", cset, nested)

    pt_corr = pt * (1 + k * std * rndm)

    pt_corr = filter_boundaries(pt_corr, pt, nested)

    return pt_corr


def pt_resol_var(pt_woresol, pt_wresol, eta, updn, cset, nested=False):
    """
    Function for the calculation of the resolution uncertainty
    Input:
    pt_woresol - muon transverse momentum without resolution correction
    pt_wresol - muon transverse momentum with resolution correction
    eta - muon pseudorapidity
    updn - uncertainty variation (up or dn)
    cset - correctionlib object
    
    This function should only be applied to reco muons in MC!
    """
    
    # if nested:
    #     eta_f, nmuons = ak.flatten(eta), ak.num(eta)
    #     pt_wresol_f, pt_woresol_f = ak.flatten(pt_wresol), ak.flatten(pt_woresol)
    # else:
    #     eta_f, nmuons = eta, 1
    #     pt_wresol_f, pt_woresol_f = pt_wresol, pt_woresol
    eta_f, nmuons = eta, 1
    pt_wresol_f, pt_woresol_f = pt_wresol, pt_woresol

    k_unc_f = cset.get("k_mc").evaluate(abs(eta_f), "stat")
    k_f = cset.get("k_mc").evaluate(abs(eta_f), "nom")

    pt_var_f = pt_wresol_f

     # Define condition and standard correction
    condition = k_f > 0
    std_x_cb = (pt_wresol_f / pt_woresol_f - 1) / k_f

    # Apply up or down variation using ak.where
    if updn == "up":
        pt_var_f = ak.where(
            condition,
            pt_woresol_f * (1 + (k_f + k_unc_f) * std_x_cb),
            pt_var_f,
        )
    elif updn == "dn":
        pt_var_f = ak.where(
            condition,
            pt_woresol_f * (1 + (k_f - k_unc_f) * std_x_cb),
            pt_var_f,
        )
    else:
        print("ERROR: updn must be 'up' or 'dn'")

    # if nested:
    #     pt_var = ak.unflatten(pt_var_f, nmuons)
    # else:
    #     pt_var = pt_var_f
    pt_var = pt_var_f

    return pt_var


def pt_scale(is_data, pt, eta, phi, charge, cset, nested=False):
    """
    Function for the calculation of the scale correction
    Input:
    is_data - flag that is True if dealing with data and False if MC
    pt - muon transverse momentum
    eta - muon pseudorapidity
    phi - muon angle
    charge - muon charge
    var - variation (standard is "nom")
    cset - correctionlib object
    
    This function should be applied to reco muons in data and MC
    """
    if is_data:
        dtmc = "data"
    else:
        dtmc = "mc"

    # if nested:
    #     eta_f, phi_f, nmuons = ak.flatten(eta), ak.flatten(phi), ak.num(eta)
    # else:
    #     eta_f, phi_f, nmuons = eta, phi, 1
    eta_f, phi_f, nmuons = eta, phi, 1
    
    a_f = cset.get("a_"+dtmc).evaluate(eta_f, phi_f, "nom")
    m_f = cset.get("m_"+dtmc).evaluate(eta_f, phi_f, "nom")

    # if nested:
    #     a, m = ak.unflatten(a_f, nmuons), ak.unflatten(m_f, nmuons)
    # else: 
    #     a, m = a_f, m_f
    a, m = a_f, m_f

    pt_corr = 1. / (m/pt + charge * a)

    pt_corr = filter_boundaries(pt_corr, pt, nested)

    return pt_corr


def pt_scale_var(pt, eta, phi, charge, updn, cset, nested=False):
    """
    Function for the calculation of the scale uncertainty
    Input:
    pt - muon transverse momentum
    eta - muon pseudorapidity
    phi - muon angle
    charge - muon charge
    updn - uncertainty variation (up or dn)
    cset - correctionlib object

    This function should be applied to reco muons in MC!
    """

    # if nested:
    #     eta_f, phi_f, nmuons = ak.flatten(eta), ak.flatten(phi), ak.num(eta)
    # else:
    #     eta_f, phi_f, pt_f, nmuons = eta, phi, pt, 1
    eta_f, phi_f, pt_f, nmuons = eta, phi, pt, 1

    stat_a_f = cset.get("a_mc").evaluate(eta_f, phi_f, "stat")
    stat_m_f = cset.get("m_mc").evaluate(eta_f, phi_f, "stat")
    stat_rho_f = cset.get("m_mc").evaluate(eta_f, phi_f, "rho_stat")

    # if nested:
    #     stat_a, stat_m, stat_rho = ak.unflatten(stat_a_f, nmuons), ak.unflatten(stat_m_f, nmuons), ak.unflatten(stat_rho_f, nmuons)
    stat_a, stat_m, stat_rho = stat_a_f, stat_m_f, stat_rho_f

    unc = pt*pt * (stat_m*stat_m / (pt*pt) + stat_a*stat_a + 2*charge*stat_rho*stat_m/pt*stat_a)**.5

    pt_var = pt

    if updn=="up":
        pt_var = pt_var + unc
    elif updn=="dn":
        pt_var = pt_var - unc

    return pt_var