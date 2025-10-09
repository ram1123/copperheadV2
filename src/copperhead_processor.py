import coffea.processor as processor
from coffea.lookup_tools import extractor
from coffea.btag_tools import BTagScaleFactor
from coffea.analysis_tools import PackedSelection
import awkward as ak
import numpy as np
from typing import Union, TypeVar, Tuple
import correctionlib
from src.corrections.rochester import apply_roccor, apply_roccorRun3
from src.corrections.fsr_recovery import fsr_recovery, fsr_recoveryV1
from src.corrections.geofit import apply_geofit
from src.corrections.jet import get_jec_factories, jet_id, jet_puid, fill_softjets, fill_softjets_HIG19006, applyHemVeto, do_jec_scale, do_jer_smear, get_jet_variation, applyUpDown, applyJetUncertaintyKinematics
# from src.corrections.weight import Weights
from src.corrections.evaluator import pu_evaluator, nnlops_weights, musf_evaluator, get_musf_lookup, lhe_weights, stxs_lookups, add_stxs_variations, add_pdf_variations,  qgl_weights_keepDim, qgl_weights_V2, btag_weights_json, btag_weights_jsonKeepDim, get_jetpuid_weights, get_jetpuid_weights_old, get_jetpuid_weights_eta_dependent
import json
from coffea.lumi_tools import LumiMask
import pandas as pd # just for debugging
import dask_awkward as dak
import dask
from coffea.analysis_tools import Weights
import copy
from coffea.nanoevents.methods import vector
import sys
from src.corrections.custom_jec import ApplyJetCorrections
from omegaconf import OmegaConf
from coffea.analysis_tools import PackedSelection

import time
import logging
from modules.utils import logger

coffea_nanoevent = TypeVar('coffea_nanoevent')
ak_array = TypeVar('ak_array')

save_path = "/depot/cms/users/yun79/results/stage1/DNN_test//2018/f0_1/data_B/0" # for debugging

"""
TODO!: add the correct btag working points for rereco samples that you will be working with when adding rereco samples
"""


# def passGoodPV_cut(events):
#     """
#     Manually apply GOdd pv cut defined as:
#     NdfPV > 4; |zPV| < 24cm; nPV > 0
#     """
#     dof_cut = events.PV.ndof > 4
#     z_cut = abs(events.PV.z) < 24
#     nPV_cut = events.PV.npvs > 0
#     out_filter = (dof_cut &
#                 z_cut &
#                 nPV_cut
#     )
#     # logger.info(f"passGoodPV_cut is any none: {ak.any(ak.is_none(out_filter)).compute()}")
#     return out_filter


def getZptWgts_3region(dimuon_pt, njets, nbins, year, config_path):
    # config_path = "./data/zpt_rewgt/fitting/zpt_rewgt_params.yaml"
    # config_path = config["new_zpt_wgt"]
    logger.info(f"zpt config file: {config_path}")
    wgt_config = OmegaConf.load(config_path)
    max_order = 5 #9
    zpt_wgt = ak.ones_like(dimuon_pt)
    jet_multiplicies = [0,1,2]
    # logger.info(f"zpt_wgt: {zpt_wgt}")

    for jet_multiplicity in jet_multiplicies:

        zpt_wgt_by_jet = ak.zeros_like(dimuon_pt)
        # zpt_wgt_by_jet = ak.ones_like(dimuon_pt) * -1 # debugging
        # first polynomial fit
        zpt_wgt_by_jet_poly = ak.zeros_like(dimuon_pt)
        for order in range(max_order + 1):  # Dynamically use max_order from the configuration
            coeff = wgt_config[str(year)][f"njet_{jet_multiplicity}"][nbins][f"f0_p{order}"]
            # logger.info(f"njet{jet_multiplicity} order {order} coeff: {coeff}")
            polynomial_term = coeff*dimuon_pt**order
            zpt_wgt_by_jet_poly = zpt_wgt_by_jet_poly + polynomial_term
            # logger.info(f"njet{jet_multiplicity} order {order} polynomial_term: {polynomial_term}")
            # logger.info(f"njet{jet_multiplicity} order {order} zpt_wgt_by_jet_poly: {zpt_wgt_by_jet_poly}")
        poly_fit_cutoff_min = wgt_config[str(year)][f"njet_{jet_multiplicity}"][nbins]["polynomial_range"]["xmin1"]
        zpt_wgt_by_jet = ak.where((poly_fit_cutoff_min >= dimuon_pt), zpt_wgt_by_jet_poly, zpt_wgt_by_jet)

        # polynomial fit
        zpt_wgt_by_jet_poly = ak.zeros_like(dimuon_pt)
        for order in range(max_order+1): # p goes from 0 to max_order
            coeff = wgt_config[str(year)][f"njet_{jet_multiplicity}"][nbins][f"f1_p{order}"]
            # logger.info(f"njet{jet_multiplicity} order {order} coeff: {coeff}")
            polynomial_term = coeff*dimuon_pt**order
            zpt_wgt_by_jet_poly = zpt_wgt_by_jet_poly + polynomial_term
            # logger.info(f"njet{jet_multiplicity} order {order} polynomial_term: {polynomial_term}")
            # logger.info(f"njet{jet_multiplicity} order {order} zpt_wgt_by_jet_poly: {zpt_wgt_by_jet_poly}")
        poly_fit_cutoff_max = wgt_config[str(year)][f"njet_{jet_multiplicity}"][nbins]["polynomial_range"]["xmax1"]
        zpt_wgt_by_jet = ak.where(((poly_fit_cutoff_min < dimuon_pt) & (poly_fit_cutoff_max >= dimuon_pt)), zpt_wgt_by_jet_poly, zpt_wgt_by_jet)

        # horizontal line beyond poly_fit_cutoff_max
        coeff = wgt_config[str(year)][f"njet_{jet_multiplicity}"][nbins][f"horizontal_c0"]
        zpt_wgt_by_jet_horizontal = ak.ones_like(dimuon_pt) * coeff
        zpt_wgt_by_jet = ak.where((poly_fit_cutoff_max < dimuon_pt), zpt_wgt_by_jet_horizontal, zpt_wgt_by_jet)
        # logger.info(f"zpt_wgt_by_jet testing: {ak.all(zpt_wgt_by_jet != -1).compute()}")
        # raise ValueError

        if jet_multiplicity != 2:
            njet_mask = njets == jet_multiplicity
        else:
            njet_mask = njets >= 2 # njet 2 is inclusive
        # logger.info(f"njet{jet_multiplicity} order  zpt_wgt_by_jet: {zpt_wgt_by_jet}")
        zpt_wgt = ak.where(njet_mask, zpt_wgt_by_jet, zpt_wgt) # if matching jet multiplicity, apply the values
        # logger.info(f"zpt_wgt after njet {jet_multiplicity}: {zpt_wgt}")

    cutOff_mask = dimuon_pt < 200 # ignore wgts from dimuon pT > 200
    zpt_wgt = ak.where(cutOff_mask, zpt_wgt, ak.ones_like(dimuon_pt))
    return zpt_wgt


def merge_zpt_wgt(yun_wgt, valerie_wgt, njets, year):
    """
    helper function that merges yun_wgt and valerie_wgt defined by jet multiplicity
    """
    val_filter_dict_run2 = {
        "2018": {0: False, 1: False, 2: True},
        "2017": {0: True, 1: False, 2: True},
        "2016postVFP": False,
        "2016preVFP": False,
    }
    val_filter_dict = val_filter_dict_run2[year]
    # logger.info(f"val_filter_dict: {val_filter_dict}")
    if val_filter_dict == False:
        return yun_wgt
    else: # divide by njet multiplicity
        val_filter = ak.zeros_like(valerie_wgt, dtype="bool")
        for njet_multiplicity_target, use_flag in val_filter_dict.items():
            # logger.info(f"njet_multiplicity_target: {njet_multiplicity_target}")
            # logger.info(f"{year} njet {njet_multiplicity_target} use_flag: {use_flag}")
            if use_flag == False: # skip
                logger.info("skipping!")
                continue
            # If true, generate a boolean 1-D array
            if njet_multiplicity_target != 2:
                use_valerie_zpt =  njets == njet_multiplicity_target
            else:
                use_valerie_zpt =  njets >= njet_multiplicity_target
            val_filter = val_filter | use_valerie_zpt

            # logger.info(f"{year} njet {njet_multiplicity_target} use_valerie_zpt: {use_valerie_zpt[:20].compute()}")
            # logger.info(f"{year} njet {njet_multiplicity_target} njets: {njets[:20].compute()}")
        # logger.info(f"{year}  val_filter: {val_filter[:20].compute()}")
        # raise ValueError
        final_filter = ak.where(val_filter, valerie_wgt, yun_wgt)
        return final_filter

def getRapidity(obj):
    px = obj.pt * np.cos(obj.phi)
    py = obj.pt * np.sin(obj.phi)
    pz = obj.pt * np.sinh(obj.eta)
    e = np.sqrt(px**2 + py**2 + pz**2 + obj.mass**2)
    rap = 0.5 * np.log((e + pz) / (e - pz))
    return rap

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
    result_mass = np.sqrt(
        result_e**2 - result_px**2 - result_py**2 - result_pz**2
    )
    result_rap = 0.5 * np.log((result_e + result_pz) / (result_e - result_pz))
    return result_mass

def p4_subtract_pt(obj1, obj2):
    """
    obtain the pt vector subtraction of two variables
    """
    result_px = ak.zeros_like(obj1.pt)
    result_py = ak.zeros_like(obj1.pt)
    result_pz = ak.zeros_like(obj1.pt)
    result_e = ak.zeros_like(obj1.pt)
    coeff = 1.0
    for obj in [obj1, obj2]:
        # px_ = obj.pt * np.cos(obj.phi)
        # py_ = obj.pt * np.sin(obj.phi)
        # pz_ = obj.pt * np.sinh(obj.eta)
        px_ = obj.px
        py_ = obj.py
        pz_ = obj.pz
        result_px = result_px + coeff*px_
        result_py = result_py + coeff*py_
        result_pz = result_pz + coeff*pz_
        coeff = -1*coeff # switch coeff
    result_pt = np.sqrt(result_px**2 + result_py**2)
    return result_pt

def testJetVector(jets):
    """
    This is a helper function in debugging observed inconsistiency in Jet variables after
    migration from coffea native vectors to hep native vectors
    params:
    jets -> nanoevent vector of Jet. IE: events.Jet
    """
    padded_jets = ak.pad_none(jets, target=2)
    # logger.info(f"type padded_jets: {type(padded_jets.compute())}")
    jet1 = padded_jets[:, 0]
    jet2 = padded_jets[:, 1]
    normal_dijet =  jet1 + jet2
    # logger.info(f"type normal_dijet: {type(normal_dijet.compute())}")
    # explicitly reinitialize the jets
    jet1_4D_vec = ak.zip({"pt":jet1.pt, "eta":jet1.eta, "phi":jet1.phi, "mass":jet1.mass}, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
    jet2_4D_vec = ak.zip({"pt":jet2.pt, "eta":jet2.eta, "phi":jet2.phi, "mass":jet2.mass}, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
    new_dijet = jet1_4D_vec + jet2_4D_vec
    target_arr = ak.fill_none(new_dijet.mass.compute(), value=-99.0)
    out_arr = ak.fill_none(normal_dijet.mass.compute(), value=-99.0)
    rel_err = np.abs((target_arr-out_arr)/target_arr)
    logger.info(f"max rel_err: {ak.max(rel_err)}")

# Dmitry's implementation of delta_r
def delta_r_V1(eta1, eta2, phi1, phi2):
    deta = abs(eta1 - eta2)
    dphi = abs(np.mod(phi1 - phi2 + np.pi, 2 * np.pi) - np.pi)
    dr = np.sqrt(deta**2 + dphi**2)
    return deta, dphi, dr


def etaFrame_variables(
        mu1: coffea_nanoevent,
        mu2: coffea_nanoevent
    ) -> Tuple[ak_array]:
    """
    Obtain eta frame cos(theta) and phi as specified in:
    https://link.springer.com/article/10.1140/epjc/s10052-011-1600-y and
    This Eta frame values supposedly plays a similar role to CS frame in terms of physics
    sensitivity, but with better resolution. Not nocessarily believe this claim
    however.
    """
    # divide muons in terms of negative and positive charges instead of leading pT
    mu_neg = ak.where((mu1.charge<0), mu1,mu2)
    mu_pos = ak.where((mu1.charge>0), mu1,mu2)
    dphi = abs(mu_neg.delta_phi(mu_pos))
    theta_eta = np.arccos(np.tanh((mu_neg.eta - mu_pos.eta) / 2))
    phi_eta = np.tan((np.pi - np.abs(dphi)) / 2) * np.sin(theta_eta)
    return np.cos(theta_eta), phi_eta

def cs_variables(
        mu1: coffea_nanoevent,
        mu2: coffea_nanoevent
    ) -> Tuple[ak_array]:
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
    ) -> ak_array :
    """
    return cos(theta) in collins-soper frame
    the formula for cos(theta) is given in Eqn 1. of https://www.ciemat.es/portal.do?TR=A&IDR=1&identificador=813
    """
    dimuon_pt = dimuon.pt
    dimuon_mass = dimuon.mass
    nominator = 2*(mu1.pz*mu2.energy - mu2.pz*mu1.energy)
    demoninator = dimuon_mass * (dimuon_mass**2 + dimuon_pt**2)**(0.5)
    cos_theta_cs = -(nominator/demoninator) # add negative sign to match the sign on pisa implementation at https://github.com/green-cabbage/copperhead_fork2/blob/Run3/python/math_tools.py#L152-L223
    return cos_theta_cs

def getPhiCS(
    mu1: coffea_nanoevent,
    mu2: coffea_nanoevent,
    dimuon: coffea_nanoevent,
    ) -> ak_array :
    """
    return phi in collins-soper frame
    the formula for phi is given in Eqn F.8 of https://people.na.infn.it/~elly/TesiAtlas/SpinCP/TestIpotesi/CollinSoperDefinition.pdf
    the implementation is heavily inspired from https://github.com/JanFSchulte/SUSYBSMAnalysis-Zprime2muAnalysis/blob/mini-AOD-2018/src/AsymFunctions.C#L1549-L1603
    """
    mu_neg = ak.where((mu1.charge<0), mu1,mu2)
    mu_pos = ak.where((mu1.charge>0), mu1,mu2)
    dimuon_pz = dimuon.pz
    dimuon_pt = dimuon.pt
    dimuon_mass = dimuon.mass
    beam_vec_z = ak.where((dimuon_pz>0), ak.ones_like(dimuon_pz), -ak.ones_like(dimuon_pz))
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
    beam_vec =  ak.zip(
        {
            "x": ak.zeros_like(dimuon_pz),
            "y": ak.zeros_like(dimuon_pz),
            "z": beam_vec_z,
        },
        with_name="Momentum3D",
        behavior=vector.behavior
    )
    # apply cross product. note x,y,z of dimuon refers to its momentum, NOT its location
    # mu.px == mu.x, mu.py == mu.y and so on
    dimuon3D_vec = ak.zip({"x":dimuon.x, "y":dimuon.y, "z":dimuon.z}, with_name="Momentum3D", behavior=vector.behavior)
    R_T = beam_vec.cross(dimuon3D_vec) # direct cross product with dimuon doesn't work bc it's a 5D vector with x,y,z,t and charge

    R_T = R_T.unit() # make it a unit vector
    Q_T = dimuon
    Q_coeff = ( ((dimuon_mass*dimuon_mass + (dimuon_pt*dimuon_pt)))**(0.5) )/dimuon_mass
    delta_T_dot_R_T = (mu_neg.px-mu_pos.px)*R_T.x + (mu_neg.py-mu_pos.py)*R_T.y
    delta_R_term = delta_T_dot_R_T
    delta_R_term = -delta_R_term # add negative sign to match the sign on pisa implementation at https://github.com/green-cabbage/copperhead_fork2/blob/Run3/python/math_tools.py#L152-L223
    delta_T_dot_Q_T = (mu_neg.px-mu_pos.px)*Q_T.px + (mu_neg.py-mu_pos.py)*Q_T.py
    delta_T_dot_Q_T = -delta_T_dot_Q_T # add negative sign to match the sign on pisa implementation at https://github.com/green-cabbage/copperhead_fork2/blob/Run3/python/math_tools.py#L152-L223
    delta_Q_term = delta_T_dot_Q_T
    delta_Q_term = delta_Q_term / dimuon_pt # normalize since Q_T should techincally be a unit vector
    phi_cs = np.arctan2(Q_coeff*delta_R_term, delta_Q_term)
    return phi_cs


class EventProcessor(processor.ProcessorABC):
    # def __init__(self, config_path: str,**kwargs):
    def __init__(self, config: dict, test_mode=False, isCutflow=False, **kwargs):
        """
        TODO: replace all of these with self.config dict variable which is taken from a
        pre-made json file
        """
        self.config = config
        self.isCutflow = isCutflow

        self.test_mode = test_mode
        dict_update = {
            # "hlt" :["IsoMu24"],
            "do_trigger_match" : True, # False
            "do_roccor" : True,# True
            "do_fsr" : True, # True
            "do_geofit" : False, # True # FIXME: Make it false for always
            "do_beamConstraint": True, # if True, override do_geofit
            "do_nnlops" : True,
            "do_pdf" : True,
        }
        self.config.update(dict_update)
        # logger.info(f"self.config: {self.config}")

        # --- Evaluator
        extractor_instance = extractor()
        year = self.config["year"]
        # Z-pT reweighting
        zpt_filename = self.config["zpt_weights_file"]
        extractor_instance.add_weight_sets([f"* * {zpt_filename}"])

        if "2016" in year:
            # self.zpt_path = "zpt_weights_all" # Valerie
            self.zpt_path = "zpt_weights/2016_value" # Dmitry
        else:
            # self.zpt_path = "zpt_weights_all" # Valerie
            self.zpt_path = "zpt_weights/2017_value" # Dmitry
        # Calibration of event-by-event mass resolution

        # add valerie Zpt
        zpt_filename = self.config["zpt_weights_file_valerie"]
        extractor_instance.add_weight_sets([f"* * {zpt_filename}"])
        self.zpt_path_valerie = "zpt_weights_all" # Valerie

        for mode in ["Data", "MC"]:
            if "2016" in year: # 2016PreVFP, 2016PostVFP, 2016_RERECO
                yearUL = "2016"
            elif ("22" in year) or ("23" in year):# FIXME: temporary solution until I can generate my own dimuon mass resolution
                yearUL = "2018"
            elif "RERECO" in year: # 2017_RERECO. 2018_RERECO
                yearUL=year.replace("_RERECO","")
            else:
                yearUL=year #Work around before there are seperate new files for pre and postVFP
            label = f"res_calib_{mode}_{yearUL}"
            file_path = self.config["res_calib_path"][mode]
            calib_str = f"{label} {label} {file_path}"
            logger.debug(f"file_path: {file_path}")
            logger.debug(f"calib_str: {calib_str}")
            extractor_instance.add_weight_sets([calib_str])

        # PU ID weights
        jetpuid_filename = self.config["jetpuid_sf_file"]
        extractor_instance.add_weight_sets([f"* * {jetpuid_filename}"])

        extractor_instance.finalize()
        self.evaluator = extractor_instance.make_evaluator()

        self.evaluator[self.zpt_path]._axes = self.evaluator[self.zpt_path]._axes[0]# this exists in Dmitry's code

        # Initialize PackedSelection
        self.selection = {}
        self.cutflow = {}

    def process(self, events: coffea_nanoevent):
        t0 = time.perf_counter()
        year = self.config["year"]
        # ReInitialize PackedSelection, otherwise processor would merge selection from previous run
        self.selection = PackedSelection()
        do_jec_unc = True #True
        """
        TODO: Once you're done with testing and validation, do LHE cut after HLT and trigger match event filtering to save computation

        Apply LHE cuts for DY sample stitching
        Basically remove events that has dilepton mass between 100 and 200 GeV
        """

        event_filter = ak.ones_like(events.event, dtype="bool") # 1D boolean array to be used to filter out bad events
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
        is_mc = events.metadata['is_mc']
        logger.debug(f"NanoAODv: {NanoAODv}")
        t1 = time.perf_counter()
        logger.info(f"[timing] Metadata read time: {t1 - t0:.2f} seconds")
        # LHE cut original start -----------------------------------------------------------------------------
        if 'dy_M-50' in dataset: # if dy_M-50, apply LHE cut
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

            # LHE_filter = ak.flatten(((LHE_dilepton_mass > 100) & (LHE_dilepton_mass < 200)))
            LHE_filter = (((LHE_dilepton_mass > 100) & (LHE_dilepton_mass < 200)))[:,0]
            # logger.info(f"LHE_filter: {LHE_filter.compute()}")
            LHE_filter = ak.fill_none(LHE_filter, value=False)
            LHE_filter = (LHE_filter== False) # we want True to indicate that we want to keep the event
            # logger.info(f"copperhead2 EventProcessor LHE_filter[32]: \n{ak.to_numpy(LHE_filter[32])}")
            # self.selection.add("LHE_cut", LHE_filter)
            event_filter = event_filter & LHE_filter
        # LHE cut original end -----------------------------------------------------------------------------

        t2 = time.perf_counter()
        logger.info(f"[timing] LHE cut time: {t2 - t1:.2f} seconds")
        """
        If the digenjet mass of the two leading jet is less than or equal to 350 GeV then keep the event.
        This genjet should be cleaned with the leptons coming from the Z-boson decay.
        """

        # --- GenJet filter for DY phase space stitching ---
        # Only apply for MC, and only if GenJet and GenPart are present
        do_GenMjjCut_forDY = False
        if (
            (is_mc and hasattr(events, "GenJet") and hasattr(events, "GenPart"))
            and ('dy_VBF_filter' in dataset  or
                 'dy_M-100To200' in dataset or
                 'dy_M-50_MiNNLO' in dataset)
            and do_GenMjjCut_forDY
            ) :
            logger.info("doing genjet filter!")
            gjets = events.GenJet
            # Select leptons from Z decay (hard process)
            gleptons = events.GenPart[
                (
                    (abs(events.GenPart.pdgId) == 13)
                    | (abs(events.GenPart.pdgId) == 11)
                    | (abs(events.GenPart.pdgId) == 15)
                )
                & events.GenPart.hasFlags('isHardProcess')
            ]
            # Clean GenJets: remove jets within dR<0.3 of any Z lepton
            gl_pair = ak.cartesian({"jet": gjets, "lepton": gleptons}, axis=1, nested=True)
            dr_gl = gl_pair["jet"].delta_r(gl_pair["lepton"])
            isolated = ak.all((dr_gl > 0.4), axis=-1)

            # if logger.isEnabledFor(logging.DEBUG):
            #     logger.debug(f"gl_pair type: {type(gl_pair)}")
            #     logger.debug(f"gl_pair: {gl_pair[:10].compute()}")

            #     temp_gl_pair_jets = gl_pair[1].compute()['jet']
            #     temp_gl_pair_leptons = gl_pair[1].compute()['lepton']
            #     logger.debug(f"gl_pair (jet): {temp_gl_pair_jets}")
            #     logger.debug(f"gl_pair (lepton): {temp_gl_pair_leptons}")
            #     logger.debug(f"gl_pair (jet.eta): {temp_gl_pair_jets.eta}")
            #     for i in range(len(temp_gl_pair_jets.eta)):
            #         logger.debug(f"gl_pair (jet.eta)[{i}]: {temp_gl_pair_jets.eta[i]}")
            #         logger.debug(f"gl_pair (jet.phi)[{i}]: {temp_gl_pair_jets.phi[i]}")
            #     logger.debug(f"gl_pair (lepton.eta): {temp_gl_pair_leptons.eta}")
            #     for i in range(len(temp_gl_pair_leptons.eta)):
            #         logger.debug(f"gl_pair (lepton.eta)[{i}]: {temp_gl_pair_leptons.eta[i]}")
            #         logger.debug(f"gl_pair (lepton.phi)[{i}]: {temp_gl_pair_leptons.phi[i]}")
            #     # logger.debug(f"gl_pair (jet.phi): {temp_gl_pair_jets.phi}")
            #     # logger.debug(f"gl_pair (lepton.phi): {temp_gl_pair_leptons.phi}")

            #     deta_gl = gl_pair["jet"].deltaeta(gl_pair["lepton"])
            #     temp_dEta_gl = deta_gl[1].compute()
            #     logger.debug(f"deta_gl type: {temp_dEta_gl}")
            #     for i in range(len(temp_dEta_gl)):
            #         logger.debug(f"deta_gl[{i}]: {temp_dEta_gl[i]}")

            #     temp_dR_gl = dr_gl[1].compute()
            #     logger.debug(f"dr_gl type: {temp_dR_gl}")
            #     logger.debug(f"dr_gl length: {len(temp_dR_gl)}")
            #     for i in range(len(temp_dR_gl)):
            #         logger.debug(f"dr_gl[{i}]: {temp_dR_gl[i]}")

            #     logger.debug(f"isolated type: {isolated[1].compute()}")
            #     for i in range(len(isolated[1].compute())):
            #         logger.debug(f"isolated[{i}]: {isolated[1].compute()[i]}")

            gjets_clean = gjets[isolated]
            # Sort by pt, pad to 2
            sorted_args = ak.argsort(gjets_clean.pt, ascending=False)
            sorted_gjets = (gjets_clean[sorted_args])
            gjets_sorted = ak.pad_none(sorted_gjets, target=2)
            gjet1 = gjets_sorted[:, 0]
            gjet2 = gjets_sorted[:, 1]
            gjj_mass = (gjet1 + gjet2).mass
            if 'dy_VBF_filter' in dataset:
                logger.info("doing VBF filter!")
                # Keep event if di-genjet mass >= 350 GeV
                genjet_filter = ak.fill_none((gjj_mass <= 350), value=False)
            elif ('dy_M-100To200' in dataset or 'dy_M-50_MiNNLO' in dataset):
                logger.info(f"doing M-100To200 filter! -- {dataset} --")
                # Keep event if di-genjet mass <= 350 GeV
                genjet_filter = ak.fill_none((gjj_mass > 350), value=False)
            event_filter = event_filter & genjet_filter

        t3 = time.perf_counter()
        logger.info(f"[timing] GenJet filter time: {t3 - t2:.2f} seconds")
        # ------------------------------------------------------------#

        # Apply HLT to both Data and MC. NOTE: this would probably be superfluous if you already do trigger matching
        HLT_filter = ak.zeros_like(event_filter, dtype="bool")  # start with 1D of Falses
        for HLT_str in self.config["hlt"]:
            logger.debug(f"HLT_str: {HLT_str}")
            # HLT_filter = HLT_filter | events.HLT[HLT_str]
            HLT_filter = HLT_filter | ak.fill_none(events.HLT[HLT_str], value=False)
        self.selection.add("HLT_filter", HLT_filter)
        event_filter = event_filter & HLT_filter

        # ------------------------------------------------------------#
        # Skimming end, filter out events and prepare for pre-selection
        # Edit: NVM; doing it this stage breaks fsr recovery
        # ------------------------------------------------------------#
        # events = events[event_filter]
        # event_filter = ak.ones_like(events.HLT.IsoMu24)

        if is_mc:
            lumi_mask = ak.ones_like(event_filter, dtype="bool")
            self.selection.add("lumi_mask", lumi_mask)

        else:
            logger.debug(f'self.config["lumimask"]: {self.config["lumimask"]}')
            lumi_info = LumiMask(self.config["lumimask"])
            lumi_mask = lumi_info(events.run, events.luminosityBlock)
            self.selection.add("lumi_mask", lumi_mask)

        t4 = time.perf_counter()
        logger.info(f"[timing] HLT and lumi mask time: {t4 - t3:.2f} seconds")
        # ------------------------------------------------------------#

        do_pu_wgt = True # True
        if self.test_mode is True: # this override should prob be replaced with something more robust in the future, or just be removed
            do_pu_wgt = False # basic override bc PU due to slight differences in implementation copperheadV1 and copperheadV2 implementation

        if do_pu_wgt:

            # obtain PU reweighting b4 event filtering, and apply it after we finalize event_filter
            logger.debug(f"year: {year}")
            if ("22" in year) or ("23" in year) or ("24" in year):
                run_campaign = 3
            else:
                run_campaign = 2
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

        # # Save raw variables before computing any corrections
        # # rochester and geofit corrects pt only, but fsr_recovery changes all vals below
        # attempt at fixing fsr issue start -------------------------------------------------------------------
        events["Muon", "pt_raw"] = ak.ones_like(events.Muon.pt) * events.Muon.pt
        events["Muon", "eta_raw"] = ak.ones_like(events.Muon.eta) * events.Muon.eta
        events["Muon", "phi_raw"] = ak.ones_like(events.Muon.phi) * events.Muon.phi
        events["Muon", "pfRelIso04_all_raw"] = ak.ones_like(events.Muon.pfRelIso04_all) * events.Muon.pfRelIso04_all
        # attempt at fixing fsr issue end ---------------------------------------------------------------

        # --------------------------------------------------------#
        # Select muons that pass pT, eta, isolation cuts,
        # muon ID and quality flags
        # Select events with 2 good muons, no electrons,
        # passing quality cuts and at least one good PV
        # --------------------------------------------------------#

        # Apply event quality flags MET filter
        evnt_qual_flg_selection = ak.ones_like(event_filter, dtype="bool")
        logger.debug("Applying event quality (MET-filter) flags")
        for evt_qual_flg in self.config["event_flags"]:
            logger.debug(f"evt_qual_flg: {evt_qual_flg}")
            evnt_qual_flg_selection = evnt_qual_flg_selection & events.Flag[evt_qual_flg]
        self.selection.add("event_quality_flags", evnt_qual_flg_selection)

        # # --------------------------------------------------------
        # apply Beam constraint b4 Rochester. We need Rochester b4 trigger matching
        # # --------------------------------------------------------

        doing_BS_correction = False # boolean that will be used for picking the correct ebe mass calibration factor
        if self.config["do_beamConstraint"] and ("bsConstrainedChi2" in events.Muon.fields): # beamConstraint overrides geofit
            logger.debug(f"doing beam constraint!")
            doing_BS_correction = True
            """
            TODO: apply dimuon mass resolution calibration factors calibrated FROM beamConstraint muons
            """
            # logger.info(f"events.Muon.fields: {events.Muon.fields}")
            BSConstraint_mask = (
                (events.Muon.bsConstrainedChi2 <30)
            )
            BSConstraint_mask = ak.fill_none(BSConstraint_mask, False)
            # comment off (~applied_fsr) cut for now
            # BSConstraint_mask = BSConstraint_mask & (~applied_fsr) # apply BSContraint on non FSR muons
            events["Muon", "pt"] = ak.where(BSConstraint_mask, events.Muon.bsConstrainedPt, events.Muon.pt)
            events["Muon", "ptErr"] = ak.where(BSConstraint_mask, events.Muon.bsConstrainedPtErr, events.Muon.ptErr)

        # logger.debug(f"muons pT: {events.Muon.pt[:5].compute()}")

        # # --------------------------------------------------------
        # # # Apply Rochester correction
        if self.config["do_roccor"]:
            # TODO make more elegant distinction between Run2 and Run3
            if "16" in year or "17" in year or "18" in year:# Run2 roccor
                logger.debug("doing Run2 rochester!")
                apply_roccor(events, self.config["roccor_file"], is_mc)
            else:
                logger.debug("doing Run3 rochester!")
                apply_roccorRun3(events, self.config["roccor_file"], is_mc)
            events["Muon", "pt"] = events.Muon.pt_roch
            # logger.info(f"df.Muon.pt after roccor: {events.Muon.pt.compute()}")

        muon_selection = (
            (events.Muon.pt_raw > self.config["muon_pt_cut"]) # pt_raw is pt b4 rochester
            & (abs(events.Muon.eta_raw) < self.config["muon_eta_cut"])
            & events.Muon[self.config["muon_id"]]
            & (events.Muon.isGlobal | events.Muon.isTracker) # Table 3.5  AN-19-124
        )
        self.selection.add("muon_pT_roch", ak.any(events.Muon.pt_roch >= self.config["muon_pt_cut"], axis=1))
        self.selection.add("muon_eta", ak.any(abs(events.Muon.eta_raw) <= self.config["muon_eta_cut"], axis=1))
        self.selection.add("muon_id", ak.any(events.Muon[self.config["muon_id"]], axis=1))
        self.selection.add("muon_isGlobal_or_Tracker", ak.any(events.Muon.isGlobal | events.Muon.isTracker, axis=1))
        self.selection.add("muon_selection", ak.any(muon_selection, axis=1))

        # calculate FSR recovery, but don't apply it until trigger matching is done
        # but apply muon iso overwrite, so base muon selection could be done
        do_fsr = self.config["do_fsr"]
        if do_fsr:
            logger.debug(f"doing fsr!")
            # applied_fsr = fsr_recovery(events)
            applied_fsr = fsr_recoveryV1(events)# testing for pt_raw inconsistency
            events["Muon", "pfRelIso04_all"] = events.Muon.iso_fsr

        # apply iso portion of base muon selection, now that possible FSR photons are integrated into pfRelIso04_all as specified in line 360 of AN-19-124
        muon_selection = muon_selection & (events.Muon.pfRelIso04_all < self.config["muon_iso_cut"])
        self.selection.add("muon_iso", ak.any(events.Muon.pfRelIso04_all < self.config["muon_iso_cut"], axis=1))
        # logger.info(f"muon_selectiont: {ak.to_dataframe(muon_selection.compute())}")

        t5 = time.perf_counter()
        logger.info(f"[timing] Muon selection time: {t5 - t4:.2f} seconds")
        # --------------------------------------------------------
        # apply tirgger match after base muon selection and Rochester correction, but b4 FSR recovery as implied in line 373 of AN-19-124
        if self.config["do_trigger_match"]:
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
            pt_threshold = self.config["muon_trigmatch_pt"] #- 0.5 # leave a little room for uncertainties

            logger.debug(f"pt_threshold: {pt_threshold}")

            pass_id = abs(events.TrigObj.id) == mu_id
            # pass_pt = events.TrigObj.pt >= pt_threshold
            # # start TrigObject matching
            # pass_filterbit_total = ak.zeros_like(events.TrigObj.filterBits, dtype="bool")
            # # grab muon candidates passing any one of the used HLTs
            # for HLT_str in self.config["hlt"]:
            #     if "IsoTkMu".lower() in HLT_str.lower():
            #         trig_filterbit = 8 # isoTkMu; source https://cms-talk.web.cern.ch/t/understanding-trigobj-filterbits-in-nanoaodv9/21646/2
            #     else:
            #         trig_filterbit = 2 # isoMu; source https://cms-talk.web.cern.ch/t/understanding-trigobj-filterbits-in-nanoaodv9/21646/2
            #     pass_filterbit = (events.TrigObj.filterBits & trig_filterbit) > 0
            #     pass_filterbit_total = pass_filterbit_total | pass_filterbit

            # trigger_cands_filter = pass_pt & pass_id & pass_filterbit_total
            pass_filterbit = (events.TrigObj.filterBits & 8) > 0
            trigger_cands_filter = pass_id & pass_filterbit
            trigger_cands = events.TrigObj[trigger_cands_filter]

            dr_threshold = self.config["muon_trigmatch_dr"]
            logger.debug(f"dr_threshold: {dr_threshold}")

            # check the first two leading muons match any of the HLT trigger objs. if neither match, reject event
            padded_muons = ak.pad_none(events.Muon[muon_selection], 2) # pad in case we have only one muon or zero in an event
            sorted_args = ak.argsort(padded_muons.pt, ascending=False)
            muons_sorted = (padded_muons[sorted_args])
            mu1 = muons_sorted[:,0]

            mu1_dr_match = mu1.delta_r(trigger_cands) <= dr_threshold

            mu1_dr_match = ak.sum(mu1_dr_match, axis=1) > 0
            mu1_dr_match = ak.fill_none(mu1_dr_match, value=False) # None is coming from the muon pad none, not trigger_cands, so this is ok
            mu1_leading_pt_match = mu1.pt_roch >= self.config["muon_leading_pt"] # apply leading pt cut for trigger matching muon
            mu1_leading_pt_match = ak.fill_none(mu1_leading_pt_match, value=False)
            mu1_trigger_match = mu1_dr_match & mu1_leading_pt_match

            mu2 = muons_sorted[:,1]
            mu2_dr_match = mu2.delta_r(trigger_cands) <= dr_threshold

            mu2_dr_match = ak.sum(mu2_dr_match, axis=1) > 0
            mu2_dr_match = ak.fill_none(mu2_dr_match, value=False) # None is coming from the muon pad none, not trigger_cands, so this is ok
            mu2_leading_pt_match = mu2.pt_roch >= self.config["muon_leading_pt"] # apply leading pt cut for trigger matching muon
            mu2_leading_pt_match = ak.fill_none(mu2_leading_pt_match, value=False)
            mu2_trigger_match = mu2_dr_match & mu2_leading_pt_match

            trigger_match = mu1_trigger_match  | mu2_trigger_match # if neither mu1 or mu2 is matched, fail trigger match
            event_filter = event_filter & trigger_match
        else:
            do_seperate_mu1_leading_pt_cut = True
            logger.warning("NO trigger match! Doing leading mu pass instead!")

        t6 = time.perf_counter()
        logger.info(f"[timing] Trigger match time: {t6 - t5:.2f} seconds")
        # --------------------------------------------------------

        # apply FSR correction, since trigger match is calculated
        if do_fsr:
            events["Muon", "pt"] = events.Muon.pt_fsr
            events["Muon", "eta"] = events.Muon.eta_fsr
            events["Muon", "phi"] = events.Muon.phi_fsr
        else:
            # if no fsr, just copy 'pt' to 'pt_fsr'
            applied_fsr = ak.zeros_like(events.Muon.pt, dtype="bool") # boolean array of Falses
            events["Muon", "pt_fsr"] = events.Muon.pt

        t6a = time.perf_counter()
        logger.info(f"[timing] FSR correction time: {t6a - t6:.2f} seconds")
        # -----------------------------------------------------------------

        if not doing_BS_correction: # apply geofit
            if self.config["do_geofit"] and ("dxybs" in events.Muon.fields):
                logger.info(f"doing geofit!")
                gf_filter, gf_pt_corr = apply_geofit(events, self.config["year"], ~applied_fsr)
                events["Muon", "pt"] = events.Muon.pt_gf # original
            else:
                logger.warning(f"doing neither beam constraint nor geofit!")

        t6b = time.perf_counter()
        logger.info(f"[timing] Geofit correction time: {t6b - t6a:.2f} seconds")
        # --------------------------------------------------------#

        muons = events.Muon[muon_selection]
        t6c = time.perf_counter()
        logger.info(f"[timing] Muon selection time: {t6c - t6b:.2f} seconds")

        # muons = ak.to_packed(events.Muon[muon_selection])

        # do the separate mu1 leading pt cut that copperheadV1 does instead of trigger matching
        if do_seperate_mu1_leading_pt_cut:
            muons_padded = ak.pad_none(muons, 2)
            sorted_args = ak.argsort(muons_padded.pt_raw, ascending=False) # since we're applying cut onver raw pt, we sort by raw pt. Sorting by reco pt gives us fewer events
            muons_sorted = (muons_padded[sorted_args])
            mu1 = muons_sorted[:,0]
            pass_leading_pt = ak.fill_none((mu1.pt_raw > self.config["muon_leading_pt"]), value=False)
            event_filter = event_filter & pass_leading_pt
        t6d = time.perf_counter()
        logger.info(f"[timing] Separate leading muon pT cut time: {t6d - t6c:.2f} seconds")
        # count muons that pass the muon selection
        nmuons = ak.num(muons, axis=1)
        # logger.debug(f"nmuons: {nmuons.compute()}")
        t6e = time.perf_counter()
        logger.info(f"[timing] Count muons time: {t6e - t6d:.2f} seconds")

        # Find opposite-sign muons
        mm_charge = ak.prod(muons.charge, axis=1) # techinally not a product of two leading pT muon charge, but (nmuons==2) cut ensures that there's only two muons

        t7 = time.perf_counter()
        logger.info(f"[timing] diMuon selection time: {t7 - t6:.2f} seconds")
        # --------------------------------------------------------#

        electron_id = self.config[f"electron_id_v{NanoAODv}"]
        logger.debug(f"electron_id: {electron_id}")
        # Veto events with good quality electrons; VBF and ggH categories need zero electrons
        ecal_gap = (1.44 < abs(events.Electron.eta)) & (1.57 > abs(events.Electron.eta)) # Source: line 460 of https://cms.cern.ch/iCMS/analysisadmin/cadilines?id=1973&ancode=EGM-17-001&tp=an&line=EGM-17-001
        electron_selection = (
            (events.Electron.pt > self.config["electron_pt_cut"])
            & (abs(events.Electron.eta) < self.config["electron_eta_cut"])
            & events.Electron[electron_id]
            & ~ecal_gap # reject electrons in ecal gap region, as specified in table 3.5 of AN-19-124
        )
        # self.selection.add("electron_pT", ak.any(events.Electron.pt > self.config["electron_pt_cut"], axis=1))
        # self.selection.add("electron_eta", ak.any(abs(events.Electron.eta) < self.config["electron_eta_cut"], axis=1))
        # self.selection.add("electron_id", ak.any(events.Electron[electron_id], axis=1))
        # self.selection.add("ecal_gap", ak.any(ecal_gap, axis=1))
        # self.selection.add("electron_selection", ak.any(electron_selection, axis=1))

        # some temporary testing code start -----------------------------------------
        # if doing_ebeMassCalib:
        #     """
        #     if obtaining results for ebe mass Calibration calculation, we want electron_veto to be turned off
        #     """
        #     electron_veto = ak.ones_like(event_filter)
        # else:
        #     electron_veto = (ak.num(events.Electron[electron_selection], axis=1) == 0)
        # some temporary testing code end -----------------------------------------

        nelectrons = ak.sum(electron_selection, axis=1)
        electron_veto = (nelectrons == 0)
        self.selection.add("electron_veto", electron_veto)
        if self.config["do_HemVeto"]:
            HemVeto_filter, _ = applyHemVeto(events.Jet, events.run, events.event, self.config, is_mc)
        else:
            HemVeto_filter = ak.ones_like(event_filter, dtype="bool")

        self.selection.add("HemVeto", HemVeto_filter == True)
        event_filter = (
                event_filter
                & lumi_mask
                & (evnt_qual_flg_selection > 0)
                # & (nmuons == 2)
                # & (mm_charge == -1)
                # & electron_veto
                & (events.PV.npvsGood > 0) # number of good primary vertex cut
                & HemVeto_filter

        )
        event_filter = event_filter & (nmuons == 2)
        self.selection.add("nmuons", nmuons==2)

        event_filter = event_filter & (mm_charge == -1)
        self.selection.add("mm_charge", mm_charge==-1)
        event_filter = event_filter & electron_veto

        t8 = time.perf_counter()
        logger.info(f"[timing] Electron selection filtering time: {t8 - t7:.2f} seconds")

        # --------------------------------------------------------#
        # Select events with muons passing leading pT cut
        # --------------------------------------------------------#

        # original start---------------------------------------------------------------
        # # Events where there is at least one muon passing
        # # leading muon pT cut
        # pass_leading_pt = muons.pt_raw > self.config["muon_leading_pt"]
        # logger.debug(f'type self.config["muon_leading_pt"] : {type(self.config["muon_leading_pt"])}')
        # logger.debug(f'type muons.pt_raw : {ak.type(muons.pt_raw.compute())}')
        # # testing -----------------------
        # # pass_leading_pt = muons.pt > self.config["muon_leading_pt"]
        # # ----------------------------------------
        # pass_leading_pt = ak.fill_none(pass_leading_pt, value=False)
        # pass_leading_pt = ak.sum(pass_leading_pt, axis=1)

        # event_filter = event_filter & (pass_leading_pt >0)
        # original end ---------------------------------------------------------------

        # better original start---------------------------------------------------------------
        # # Events where there is at least one muon passing
        # # leading muon pT cut
        # # muons_pt_raw_padded =
        # pass_leading_pt = ak.max(muons.pt_raw, axis=1) > self.config["muon_leading_pt"]
        # # testing -----------------------
        # # pass_leading_pt = muons.pt > self.config["muon_leading_pt"]
        # # ----------------------------------------
        # pass_leading_pt = ak.fill_none(pass_leading_pt, value=False)

        # event_filter = event_filter & pass_leading_pt
        # better original end ---------------------------------------------------------------

        # test start ----------------------------------------------------------------
        # # NOTE: if you want to keep this method, (which I don't btw since the original
        # # code above is conceptually more correct at this moment), you should optimize
        # # this code, bc this was just something I put together for quick testing

        # muons_padded = ak.pad_none(muons, target=2)
        # sorted_args = ak.argsort(muons_padded.pt, ascending=False) # leadinig pt is ordered by pt
        # muons_sorted = (muons_padded[sorted_args])
        # mu1 = muons_sorted[:,0]
        # pass_leading_pt = mu1.pt_raw > self.config["muon_leading_pt"]
        # pass_leading_pt = ak.fill_none(pass_leading_pt, value=False)

        # event_filter = event_filter & pass_leading_pt
        # test end -----------------------------------------------------------------------

        # calculate sum of gen weight b4 skimming off bad events
        if is_mc:
            # if True:
            if self.test_mode: # for small files local testing
                sumWeights = ak.sum(events.genWeight, axis=0) # for testing
                # logger.debug(f"small file test sumWeights: {(sumWeights.compute())}") # for testing
            else:
                sumWeights = events.metadata['sumGenWgts']
                logger.debug(f"sumWeights: {(sumWeights)}")

        events = events[event_filter==True]
        muons = muons[event_filter==True]
        nmuons = ak.to_packed(nmuons[event_filter==True])

        if is_mc and do_pu_wgt:
            for variation in pu_wgts.keys():
                pu_wgts[variation] = ak.to_packed(pu_wgts[variation][event_filter==True])
        # pass_leading_pt = ak.to_packed(pass_leading_pt[event_filter==True])

        t9 = time.perf_counter()
        logger.info(f"[timing] GEN weight and PU time: {t9 - t8:.2f} seconds")

        # --------------------------------------------------------#
        # Fill dimuon and muon variables
        # --------------------------------------------------------#

        # ---------------------------------------------------------
        # TODO: find out why we don't filter out bad events right now via
        # even_selection column, since fill muon is computationally exp
        # Last time I checked there was some errors on LHE correction shape mismatch
        # ---------------------------------------------------------

        muons_padded = ak.pad_none(muons, target=2)
        sorted_args = ak.argsort(muons_padded.pt, ascending=False)
        muons_sorted = (muons_padded[sorted_args])
        mu1 = muons_sorted[:,0]
        mu2 = muons_sorted[:,1]

        dimuon_dR = mu1.delta_r(mu2)
        dimuon_dEta = abs(mu1.eta - mu2.eta)
        dimuon_dPhi = abs(mu1.delta_phi(mu2))
        acoplanarity = 1 - dimuon_dPhi/ np.pi  # acoplanarity = 1 - delta_phi/pi
        dimuon = mu1+mu2

        uncalibrated_dimuon_ebe_mass_res, calibration = self.get_mass_resolution(dimuon, mu1, mu2, is_mc, test_mode=self.test_mode, doing_BS_correction=doing_BS_correction)
        dimuon_ebe_mass_res = uncalibrated_dimuon_ebe_mass_res * calibration
        dimuon_ebe_mass_res_rel = dimuon_ebe_mass_res/dimuon.mass
        dimuon_cos_theta_cs, dimuon_phi_cs = cs_variables(mu1,mu2)
        dimuon_cos_theta_eta, dimuon_phi_eta = etaFrame_variables(mu1,mu2)

        t10 = time.perf_counter()
        logger.info(f"[timing] Dimuon variables time: {t10 - t9:.2f} seconds")
        # skip validation for genjets for now -----------------------------------------------
        # test:
        # logger.info(dimuon_cos_theta_cs.compute())
        # logger.info(dimuon_phi_cs.compute())

        # #fill genjets

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
            # logger.debug(f"n_gleptons: {ak.num(gleptons,axis=1).compute()}")
            gl_pair = ak.cartesian({"jet": gjets, "lepton": gleptons}, axis=1, nested=True)
            dr_gl = gl_pair["jet"].delta_r(gl_pair["lepton"])
            # logger.debug(f'gl_pair["jet"]: {gl_pair["jet"].pt.compute().show(formatter=np.set_printoptions(threshold=sys.maxsize))}')
            # logger.debug(f'gl_pair["lepton"]: {gl_pair["lepton"].pt.compute().show(formatter=np.set_printoptions(threshold=sys.maxsize))}')
            # test start --------------------------------
            # _, _, dr_gl = delta_r_V1(
            #     gl_pair["jet"].eta,
            #     gl_pair["lepton"].eta,
            #     gl_pair["jet"].phi,
            #     gl_pair["lepton"].phi,
            # )
            # test end --------------------------------
            # logger.debug(f"n_gjets: {ak.num(gjets,axis=1).compute()}")
            # logger.debug(f"gl_pair: {gl_pair.compute()}")
            # logger.debug(f"dr_gl: {dr_gl.compute().show(formatter=np.set_printoptions(threshold=sys.maxsize))}")
            # logger.debug(f"gjets b4 isolation: {gjets.compute()}")
            isolated = ak.all((dr_gl > 0.3), axis=-1) # this also returns true if there's no leptons near the gjet
            # logger.debug(f"isolated: {isolated.compute()}")
            # logger.debug(f"dr_gl[isolated]: {dr_gl[isolated].compute()}")
            # original start ----------------------------------------
            # padded_iso_gjet = ak.pad_none(
            #     ak.to_packed(gjets[isolated]),
            #     target=2,
            # ) # pad with none val to ensure that events have at least two columns each event
            # sorted_args = ak.argsort(padded_iso_gjet.pt, ascending=False) # leading pt is ordered by pt
            # gjets_sorted = (padded_iso_gjet[sorted_args])
            # original end ----------------------------------------

            # same order sorting algorithm as reco jet start -----------------
            gjets = ak.to_packed(gjets[isolated])
            # logger.debug(f"gjets.pt: {gjets.pt.compute()}")
            sorted_args = ak.argsort(gjets.pt, ascending=False)
            sorted_gjets = (gjets[sorted_args])
            gjets_sorted = ak.pad_none(sorted_gjets, target=2)
            # same order sorting algorithm as reco jet end -----------------

            # logger.debug(f"gjets_sorted: {gjets_sorted.compute()}")
            gjet1 = gjets_sorted[:,0]
            gjet2 = gjets_sorted[:,1]
            # original start -----------------------------------------------
            gjj = gjet1 + gjet2
            # logger.debug(f"gjj.mass: {gjj_mass.compute().show(formatter=np.set_printoptions(threshold=sys.maxsize))}")
            # logger.debug(f"gjj.mass: {ak.sum(gjj_mass,axis=None).compute()}")
            # original end -------------------------------------------------

            # gjet1_Lvec = ak.zip({"pt":gjet1.pt, "eta":gjet1.eta, "phi":gjet1.phi, "mass":gjet1.mass}, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
            # gjet2_Lvec = ak.zip({"pt":gjet2.pt, "eta":gjet2.eta, "phi":gjet2.phi, "mass":gjet2.mass}, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
            # gjj = gjet1_Lvec + gjet2_Lvec

            gjj_dEta = abs(gjet1.eta - gjet2.eta)
            gjj_dPhi = abs(gjet1.delta_phi(gjet2))
            gjj_dR = gjet1.delta_r(gjet2)

        t11 = time.perf_counter()
        logger.info(f"[timing] GenJet variables time: {t11 - t10:.2f} seconds")

        self.prepare_jets(events, NanoAODv=NanoAODv)
        # logger.debug("test ject vector right after prepare_jets")
        # testJetVector(events.Jet)

        # ------------------------------------------------------------#
        # Apply JEC, get JEC and JER variations
        # ------------------------------------------------------------#
        # JER: https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetResolution
        # JES: https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC

        year = self.config["year"]
        jets = events.Jet
        fatJets = events.FatJet
        nfatJets = ak.num(fatJets, axis=1)
        self.jec_factories_mc, self.jec_factories_data = get_jec_factories(
            self.config["jec_parameters"],
            year
        )
        t12 = time.perf_counter()
        logger.info(f"[timing] prepare jets time: {t12 - t11:.2f} seconds")

        do_jec = True # True # FIXME: Hardcoded
        # do_jecunc = self.config["do_jecunc"]
        # do_jerunc = self.config["do_jerunc"]
        # testing
        do_jecunc = False
        do_jerunc = False
        # cache = events.caches[0]
        factory = None
        useclib = False
        jet_default = ak.pad_none(jets, target=4) # save pre jec and jer Jet for comparison
        jet1_default = jet_default[:, 0]
        jet2_default = jet_default[:, 1]
        jet3_default = jet_default[:, 2]
        jet4_default = jet_default[:, 3]

        # -----------------------------------------------------
        # pre-selection for fatjets
        # add pre-selection for fatjets before saving the information: pT > 150 GeV and |eta| < 2.4 and pass the tight jet ID, dR(j, muons) > 0.8, FatJet_particleNetWithMass_WvsQCD > 0.75
        # Save the number of fat jets that passes this conditions
        # print first 5 events, fatjet pT
        # logger.warning(f"Number of fatjets (before selection): {nfatJets[:25].compute()}")
        # logger.warning(f"FatJet pT (before selection): {fatJets.pt[:25].compute()}")

        fatjet_selection = (
            (fatJets.pt > 150)
            & (abs(fatJets.eta) < 2.4)
            & (fatJets.jetId >= 2) # tight jet ID
            & (fatJets.particleNetWithMass_WvsQCD > 0.75) # W vs QCD discriminator
        )
        fatJets = fatJets[fatjet_selection]
        nfatJets_pre = ak.num(fatJets, axis=1)
        # logger.warning(f"Number of fatjets (after selection): {nfatJets_pre[:25].compute()}")
        # logger.warning(f"FatJet pT (after pre-selection): {fatJets.pt[:25].compute()}")

        # if nfatJets_pre > 0, we apply the dR(jet, muon) > 0.8 cut and save the number of fatjets that passes this
        # here muons are mu1 and mu2, as defined above
        fatJets_dRmu1 = fatJets.delta_r(mu1)
        fatJets_dRmu2 = fatJets.delta_r(mu2)
        fatJets_dRmu1 = ak.fill_none(fatJets_dRmu1, 999) # if there's no fatjet, set dR to a large number, set it to +999 as later I am checking min of the two numbers. So, set it to large +ve number
        fatJets_dRmu2 = ak.fill_none(fatJets_dRmu2, 999)
        fatJets_dRmu = np.minimum(fatJets_dRmu1, fatJets_dRmu2)

        # logger.warning(f"dR(jet, mu1) (before dR cut): {fatJets_dRmu1[:25].compute()}")
        # logger.warning(f"dR(jet, mu2) (before dR cut): {fatJets_dRmu2[:25].compute()}")
        # logger.warning(f"mininum dR(jet, muon) (before dR cut): {fatJets_dRmu[:25].compute()}")

        fatJets = fatJets[fatJets_dRmu > 0.8]
        nfatJets_drmuon = ak.num(fatJets, axis=1)
        # logger.warning(f"FatJet pT (after dR(jet, muon) > 0.8 cut): {fatJets.pt[:25].compute()}")
        # logger.warning(f"Number of fatjets (after dR(jet, muon) > 0.8 cut): {nfatJets_drmuon[:25].compute()}")

        # keep only the leading fatjet after all the selections above
        fatJets_default = ak.pad_none(fatJets, target=1)
        fatJet1_default = fatJets_default[:, 0]

        if do_jec: # old method
            if is_mc:
                factory = self.jec_factories_mc["jec"]
            else:
                for run in self.config["jec_parameters"]["runs"]:
                    logger.debug(f"run: {run}")
                    logger.debug(f"dataset: {dataset}")
                    if run in dataset:
                        factory = self.jec_factories_data[run]
                if factory == None:
                    logger.debug("JEC factory not recognized!")
                    raise ValueError

            # -------------------------------------
            logger.debug("doing JEC + SMEARing!")
            if do_jec_unc:
                variation_l = ["nominal"] + self.config["jec_parameters"]["jec_unc_to_consider"]
            else:
                variation_l = ["nominal"]
            jets = do_jec_scale(jets, self.config, is_mc, dataset, uncs=variation_l)
            # jets = do_jec_scale(jets, self.config, is_mc, dataset)
            # print(f"jets test: {jets.pt_Absolute_up.compute()}")
            jets["mass_jec"] = jets.mass
            jets["pt_jec"] = jets.pt

            if is_mc: # JER smearing
                jets = do_jer_smear(jets, self.config, events.event, year=year)
            sorted_args = ak.argsort(jets.pt, ascending=False)
            jets = (jets[sorted_args])

            # now JER has been applied, we apply unc coeefficients to the latest value
            variation_l.remove("nominal")
            if is_mc:
                jets = applyJetUncertaintyKinematics(jets, variation_l)

            # -------------------------------------

            # testJetVector(jets)
            # logger.info(f"jets pt b4 jec: {jets.pt.compute()}")
            # -------------------------------------
            # logger.info("do old jec!")
            # jets = factory.build(jets)
            # -------------------------------------
            # logger.info(f"jets pt after jec: {jets.pt.compute()}")
            # testJetVector(jets)

        else:
            jets["mass_jec"] = jets.mass
            jets["pt_jec"] = jets.pt

        t13 = time.perf_counter()
        logger.info(f"[timing] JEC and JER time: {t13 - t12:.2f} seconds")
        # # ------------------------------------------------------------#

        # # ------------------------------------------------------------#
        # # Apply genweights, PU weights
        # # and L1 prefiring weights
        # # ------------------------------------------------------------#
        weights = Weights(None, storeIndividual=True) # none for dask awkward
        # weights = Weights(len(events))
        if is_mc:
            if "MiNNLO" in dataset: # We have spurious gen weight issue. ref: https://cms-talk.web.cern.ch/t/huge-event-weights-in-dy-powhegminnlo/8718/9
                weights.add("genWeight", weight=np.sign(events.genWeight)) # just extract the sign, not the magnitude
            else:
                weights.add("genWeight", weight=events.genWeight)
            # original initial weight start ----------------
            weights.add("genWeight_normalization", weight=ak.ones_like(events.genWeight)/sumWeights) # temporary commenting out

            cross_section = self.config["cross_sections"][dataset]
            # check if there's a year-wise different cross section
            try:
                cross_section = cross_section[year] # we assume cross_section is a map (Omegaconf)
            except:
                cross_section = cross_section # we assume this is a number
            logger.debug(f"cross_section: {cross_section}")
            logger.debug(f"type(cross_section): {type(cross_section)}")
            integrated_lumi = self.config["integrated_lumis"]
            weights.add("xsec", weight=ak.ones_like(events.genWeight)*cross_section)
            weights.add("lumi", weight=ak.ones_like(events.genWeight)*integrated_lumi)
            # original initial weight end ----------------

            if do_pu_wgt:
                logger.debug("adding PU wgts!")
                weights.add("pu_wgt", weight=pu_wgts["nom"],weightUp=pu_wgts["up"],weightDown=pu_wgts["down"])
                # logger.info(f"pu_wgts['nom']: {ak.to_numpy(pu_wgts['nom'].compute())}")
            # L1 prefiring weights
            if self.config["do_l1prefiring_wgts"] and ("L1PreFiringWeight" in events.fields):
                logger.debug("adding L1 prefiring wgts!")
                L1_nom = events.L1PreFiringWeight.Nom
                L1_up = events.L1PreFiringWeight.Up
                L1_down = events.L1PreFiringWeight.Dn
                weights.add("l1prefiring",
                    weight=L1_nom,
                    weightUp=L1_up,
                    weightDown=L1_down
                )
                # logger.info(f"L1_nom: {ak.to_numpy(L1_nom.compute())}")
        else: # data-> just add in ak ones for consistency
            weights.add("ones", weight=ak.values_astype(ak.ones_like(events.HLT.IsoMu24), "float32"))

        t14 = time.perf_counter()
        logger.info(f"[timing] Weights time: {t14 - t13:.2f} seconds")

        # ------------------------------------------------------------#
        # Calculate other event weights
        # ------------------------------------------------------------#
        jec_pars = self.config["jec_parameters"]
        # FIXME: For data (is is_mc == False) I should not add this variations.
        do_jec_unc = True
        if do_jec_unc:
            pt_variations = (
                ["nominal"]
                + applyUpDown(jec_pars["jec_unc_to_consider"])
                + jec_pars["jer_variations"]
            )
        else:
            pt_variations = ["nominal"]

        if is_mc:
            # moved nnlops reweighting outside of dak process and to run_stage1-----------------
            do_nnlops = self.config["do_nnlops"] and ("ggh" in events.metadata["dataset"])
            if do_nnlops:
                logger.debug("doing NNLOPS!")
                nnlopsw = nnlops_weights(events.HTXS.Higgs_pt, events.HTXS.njets30, self.config, events.metadata["dataset"])
                # logger.info(f"nnlopsw: {ak.to_numpy(nnlopsw.compute())}")
                weights.add("nnlops", weight=nnlopsw)
            # moved nnlops reweighting outside of dak process-----------------

            # do mu SF start -------------------------------------
            logger.debug("doing musf!")
            musf_lookup = get_musf_lookup(self.config)
            muID, muIso, muTrig = musf_evaluator(
                musf_lookup, self.config["year"], mu1, mu2
            )
            weights.add("muID",
                    weight=muID["nom"],
                    weightUp=muID["up"],
                    weightDown=muID["down"]
            )
            weights.add("muIso",
                    weight=muIso["nom"],
                    weightUp=muIso["up"],
                    weightDown=muIso["down"]
            )
            weights.add("muTrig",
                    weight=muTrig["nom"],
                    weightUp=muTrig["up"],
                    weightDown=muTrig["down"]
            )
            # do mu SF end -------------------------------------

            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
            do_lhe = (
                ("LHEScaleWeight" in events.fields)
                and ("LHEPdfWeight" in events.fields)
                and ("nominal" in pt_variations)
            )
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
                ("vbf" in dataset)
                and ("dy" not in dataset)
                and ("nominal" in pt_variations)
                and ("stage1_1_fine_cat_pTjet30GeV" in events.HTXS.fields)
            )
            # do_thu = False
            if do_thu:
                logger.info("doing THU!")
                add_stxs_variations(
                    events,
                    weights,
                    self.config,
                )

            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
            do_pdf = (
                self.config["do_pdf"]
                and ("nominal" in pt_variations)
                and (
                    "dy" in dataset
                    or "ewk" in dataset
                    or "ggh" in dataset
                    or "vbf" in dataset
                )
                and ("mg" not in dataset)
            )
            if do_pdf:
                logger.debug("doing pdf!")
                # add_pdf_variations(events, self.weight_collection, self.config, dataset)
                pdf_vars = add_pdf_variations(events, self.config, dataset)
                weights.add("pdf_2rms",
                    weight=ak.ones_like(pdf_vars["up"]),
                    weightUp=pdf_vars["up"],
                    weightDown=pdf_vars["down"]
                )
        t15 = time.perf_counter()
        logger.info(f"[timing] some GEN event weights for syst time: {t15 - t14:.2f} seconds")

        # ------------------------------------------------------------#
        # Fill Muon variables and gjet variables
        # ------------------------------------------------------------#
        if "2016" in year:
            dnn_year = 2016
        else:
            dnn_year = int(year)
        logger.debug(f"dnn_year: {dnn_year}")
        out_dict = {
            "event": events.event,
            "PV_npvs": events.PV.npvs,
            "PV_npvsGood": events.PV.npvsGood,
            "MET_pt": events.MET.pt,
            "MET_phi": events.MET.phi,
            "MET_sumEt": events.MET.sumEt,
            "mu1_pt": mu1.pt,
            "mu1_ptErr": mu1.ptErr,
            "mu2_pt": mu2.pt,
            "mu2_ptErr": mu2.ptErr,
            "mu1_pt_over_mass": mu1.pt / dimuon.mass,
            "mu2_pt_over_mass": mu2.pt / dimuon.mass,
            "mu1_eta": mu1.eta,
            "mu2_eta": mu2.eta,
            "mu1_phi": mu1.phi,
            "mu2_phi": mu2.phi,
            "mu1_charge": mu1.charge,
            "mu2_charge": mu2.charge,
            "mu1_iso": mu1.pfRelIso04_all,
            "mu2_iso": mu2.pfRelIso04_all,
            "mu1_pt_over_mu2_pt": mu1.pt / mu2.pt,
            "mu1_eta_over_mu2_eta": abs(mu1.eta) / abs(mu2.eta),
            # "mu1_pt_roch" : mu1.pt_roch,
            # "mu1_pt_fsr" : mu1.pt_fsr,
            # "mu1_pt_gf" : mu1.pt_gf,
            # "mu2_pt_roch" : mu2.pt_roch,
            # "mu2_pt_fsr" : mu2.pt_fsr,
            # "mu2_pt_gf" : mu2.pt_gf,
            "nmuons": nmuons,
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
            "dimuon_ebe_mass_res": dimuon_ebe_mass_res,
            "dimuon_ebe_mass_res_rel": dimuon_ebe_mass_res_rel,
            "uncalibrated_dimuon_ebe_mass_res": uncalibrated_dimuon_ebe_mass_res,
            "dimuon_cos_theta_cs": dimuon_cos_theta_cs,
            "dimuon_phi_cs": dimuon_phi_cs,
            "dimuon_cos_theta_eta": dimuon_cos_theta_eta,
            "dimuon_phi_eta": dimuon_phi_eta,
            "dimuon_pt_over_MET_pt": dimuon.pt / events.MET.pt,
            "dimuon_pt_over_jet1_pt": dimuon.pt / jet1_default.pt,
            "dimuon_pt_over_jet2_pt": dimuon.pt / jet2_default.pt,
            "mu1_pt_raw": mu1.pt_raw,
            "mu2_pt_raw": mu2.pt_raw,
            "mu1_pt_fsr": mu1.pt_fsr,
            "mu2_pt_fsr": mu2.pt_fsr,
            # "pass_leading_pt" : pass_leading_pt,
            "year": ak.ones_like(nmuons) * dnn_year,
            "run": events.run,
            "event": events.event,
            "luminosityBlock": events.luminosityBlock,
            "fraction": ak.ones_like(events.event) * events.metadata["fraction"],
            # add jet default kinematics here
            "jet1_default_pt_nominal": jet1_default.pt,
            "jet1_default_eta_nominal": jet1_default.eta,
            "jet1_default_phi_nominal": jet1_default.phi,
            "jet1_default_mass_nominal": jet1_default.mass,
            "jet2_default_pt_nominal": jet2_default.pt,
            "jet2_default_eta_nominal": jet2_default.eta,
            "jet2_default_phi_nominal": jet2_default.phi,
            "jet2_default_mass_nominal": jet2_default.mass,
            "jet3_default_pt_nominal": jet3_default.pt,
            "jet3_default_eta_nominal": jet3_default.eta,
            "jet3_default_phi_nominal": jet3_default.phi,
            "jet3_default_mass_nominal": jet3_default.mass,
            "jet4_default_pt_nominal": jet4_default.pt,
            "jet4_default_eta_nominal": jet4_default.eta,
            "jet4_default_phi_nominal": jet4_default.phi,
            "jet4_default_mass_nominal": jet4_default.mass,

            "nfatJets": nfatJets,
            "nfatJets_pre": nfatJets_pre,
            "nfatJets_drmuon": nfatJets_drmuon,
            # "fatjets_dRmu_gt0p8": fatJets_dRmu,


            # add fatjet1 default kinematics
            "fatJet1_default_pt_nominal": fatJet1_default.pt,
            "fatJet1_default_eta_nominal": fatJet1_default.eta,
            "fatJet1_default_phi_nominal": fatJet1_default.phi,
            "fatJet1_default_mass_nominal": fatJet1_default.mass,
            "fatJet1_default_jetId_nominal": fatJet1_default.jetId,
            "fatJet1_default_msoftdrop_nominal": fatJet1_default.msoftdrop,
            "fatJet1_default_electronIdx3SJ_nominal": fatJet1_default.electronIdx3SJ,
            "fatJet1_default_nConstituents_nominal": fatJet1_default.nConstituents,
            "fatJet1_default_tau1_nominal": fatJet1_default.tau1,
            "fatJet1_default_tau2_nominal": fatJet1_default.tau2,
            "fatJet1_default_tau3_nominal": fatJet1_default.tau3,
            "fatJet1_default_tau4_nominal": fatJet1_default.tau4,
            "fatJet1_default_btagDDBvLV2_nominal": fatJet1_default.btagDDBvLV2,
            "fatJet1_default_btagDDCvBV2_nominal": fatJet1_default.btagDDCvBV2,
            "fatJet1_default_btagDDCvLV2_nominal": fatJet1_default.btagDDCvLV2,
            "fatJet1_default_btagDeepB_nominal": fatJet1_default.btagDeepB,
            "fatJet1_default_btagHbb_nominal": fatJet1_default.btagHbb,
            "fatJet1_default_particleNetWithMass_QCD_nominal": fatJet1_default.particleNetWithMass_QCD,
            "fatJet1_default_particleNetWithMass_WvsQCD_nominal": fatJet1_default.particleNetWithMass_WvsQCD,
            "fatJet1_default_particleNetWithMass_ZvsQCD_nominal": fatJet1_default.particleNetWithMass_ZvsQCD,
            "fatJet1_default_particleNet_QCD_nominal": fatJet1_default.particleNet_QCD,
            "fatJet1_default_particleNet_XbbVsQCD_nominal": fatJet1_default.particleNet_XbbVsQCD,
            "fatJet1_default_particleNet_XccVsQCD_nominal": fatJet1_default.particleNet_XccVsQCD,
            "fatJet1_default_particleNet_XggVsQCD_nominal": fatJet1_default.particleNet_XggVsQCD,
            "fatJet1_default_particleNet_XqqVsQCD_nominal": fatJet1_default.particleNet_XqqVsQCD,
            "fatJet1_default_particleNet_massCorr_nominal": fatJet1_default.particleNet_massCorr
        }
        if is_mc:
            mc_dict = {
                # "HTXS_Higgs_pt" : events.HTXS.Higgs_pt, # for nnlops weight for ggH signal sample
                # "HTXS_njets30" : events.HTXS.njets30, # for nnlops weight for ggH signal sample
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
                "gjj_mass": gjj.mass,
                "gjj_dEta" : gjj_dEta,
                "gjj_dPhi" : gjj_dPhi,
                "gjj_dR" : gjj_dR,
            }
            out_dict.update(mc_dict)

        t16 = time.perf_counter()
        logger.info(f"[timing] Fill muon and gjet variables time: {t16 - t15:.2f} seconds")
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
                do_jec = do_jec,
                do_jecunc = do_jecunc,
                do_jerunc = do_jerunc,
                # event_match=event_match # debugging
            )

            out_dict.update(jet_loop_dict)

        logger.debug(f"out_dict.keys() after jet loop: {out_dict.keys()}")

        t17 = time.perf_counter()
        logger.info(f"[timing] Jet pT variations time: {t17 - t16:.2f} seconds")
        # logger.info(f"out_dict.keys() after jet loop: {out_dict.keys()}")

        # logger.info(f"out_dict.persist 2: {ak.zip(out_dict).persist().to_parquet(save_path)}")
        # logger.info(f"out_dict.compute 2: {ak.zip(out_dict).to_parquet(save_path)}")

        # # fill in the regions
        mass = dimuon.mass
        z_peak = ((mass > 76) & (mass < 106))
        h_sidebands =  ((mass > 110) & (mass < 115.03)) | ((mass > 135.03) & (mass < 150))
        h_peak = ((mass > 115.03) & (mass < 135.03))
        region_dict = {
            "z_peak" : ak.fill_none(z_peak, value=False),
            "h_sidebands" : ak.fill_none(h_sidebands, value=False),
            "h_peak" : ak.fill_none(h_peak, value=False),
        }
        t18 = time.perf_counter()
        logger.info(f"[timing] various region (z-peak) fill time: {t18 - t17:.2f} seconds")
        # self.selection.add("z_peak", z_peak == True)
        # self.selection.add("h_sidebands", h_sidebands == True)
        # self.selection.add("h_peak", h_peak == True)

        out_dict.update(region_dict)

        # b4 we do any filtering, we obtain the sum of gen weights for normalization
        # events["genWeight"] = ak.values_astype(events.genWeight, "float64") # increase precision or it gives you slightly different value for summing them up

        # logger.info(f"out_dict.compute 3: {ak.zip(out_dict).to_parquet(save_path)}")
        njets = out_dict[f"njets_nominal"]
        # logger.info(f"njets: {ak.to_numpy(njets.compute())}")

        # do zpt weight at the very end
        dataset = events.metadata["dataset"]
        do_zpt = ('dy' in dataset) and is_mc
        # do_zpt = False # temporary overwrite to obtain for zpt re-wgt
        if do_zpt:
            logger.info("=======================  apply zpt weights =======================")
            if "MiNNLO" in dataset: # FIXME: temporary fix for MiNNLO samples
                zpt_weight_mine_nbins100 = getZptWgts_3region(dimuon.pt, njets, 100, year, self.config["new_zpt_weights_file_MiNNLO"])
            else:
                zpt_weight_mine_nbins100 = getZptWgts_3region(dimuon.pt, njets, 100, year, self.config["new_zpt_weights_file_aMCatNLO"])

            # logger.info("========================= zpt weights using mu1 and mu2 pT =========================")
            # sf_dict = load_sf_dict("/depot/cms/users/shar1172/copperheadV2_CheckSetup/data/zpt_rewgt/fitting_mu1mu2pt/sf_data_flat.json")
            # logger.info(f"sf_dict: {sf_dict}")
            # zpt_weight_mine_nbins100 = getZptWgts_new(mu1.pt, mu2.pt, acoplanarity, njets, sf_dict)
            # logger.info(f"zpt_weight_mine_nbins100: {type(zpt_weight_mine_nbins100)}")
            # logger.info(f"zpt_weight_mine_nbins100: {(zpt_weight_mine_nbins100)}")

            # calibration = correction.evaluate(mu1.pt, abs(mu1.eta), abs(mu2.eta))
            #
            # correction_set = correctionlib.CorrectionSet.from_file("/depot/cms/private/users/shar1172/copperheadV2_CheckSetup/data/zpt_rewgt/fitting_mu1mu2pt/sf_data_correctionlib.json")
            # correction_set = correctionlib.CorrectionSet.from_file("/depot/cms/private/users/shar1172/copperheadV2_CheckSetup/data/zpt_rewgt/fitting_mu1mu2pt/sf_data_correctionlib_variable_acop.json")
            # correction = correction_set["acoplanaritySF"]
            # logger.info(f"correction_set: {correction_set}")
            # logger.info(f"correction: {correction}")
            # zpt_weight = correction.evaluate(mu1.pt, mu2.pt, njets, acoplanarity)
            # logger.info(f"zpt_weight: {zpt_weight}")

            # logger.info( f"zpt_weight_mine_nbins100 after new sf_dict: {zpt_weight_mine_nbins100.compute()}")

            zpt_weight = zpt_weight_mine_nbins100
            weights.add("zpt_wgt",
                    weight=zpt_weight,
            )

        t19 = time.perf_counter()
        logger.info(f"[timing] Zpt weights time: {t19 - t18:.2f} seconds")

        # apply vbf filter phase cut if DY test start ---------------------------------
        # if dataset == 'dy_M-100To200':
        #     vbfReverseFilter = ak.values_astype(
        #         ak.fill_none((gjj.mass <= 350), value=False),
        #         np.int32
        #     ) # any higher value should be populated by VBF filtered DY instead
        #     weights.add("vbfReverseFilter",
        #             weight=vbfReverseFilter,
        #     )
        # apply vbf filter phase cut if DY test end ---------------------------------
        logger.debug(f"weight statistics: {weights.weightStatistics.keys()}")
        # logger.debug(f"weight variations: {weights.variations}")
        wgt_nominal = weights.weight()

        # add in weights

        weight_dict = {"wgt_nominal" : wgt_nominal}

        # loop through weight variations
        for variation in weights.variations:
            wgt_variation = weights.weight(variation)
            variation_name = "wgt_" + variation.replace("Up", "_up").replace("Down", "_down") # match the naming scheme of copperhead
            weight_dict[variation_name] = wgt_variation

        t20 = time.perf_counter()
        logger.info(f"[timing] Weights variations time: {t20 - t19:.2f} seconds")

        # temporarily shut off partial weights start -----------------------------------------
        for weight_type in list(weights.weightStatistics.keys()):
            wgt_name = "separate_wgt_" + weight_type
            # logger.info(f"wgt_name: {wgt_name}")
            weight_dict[wgt_name] = weights.partial_weight(include=[weight_type])
        # temporarily shut off partial weights end -----------------------------------------
        t21 = time.perf_counter()
        logger.info(f"[timing] Weights partials time: {t21 - t20:.2f} seconds")

        # logger.info(f"out_dict.persist 5: {ak.zip(out_dict).persist().to_parquet(save_path)}")
        # logger.info(f"out_dict.compute 5: {ak.zip(out_dict).to_parquet(save_path)}")
        out_dict.update(weight_dict)

        # # Combine all cuts
        logger.info(f"selection: {self.selection}")

        # wgtcutflow = selection.cutflow("noMuon", "twoElectron", "leadPt20", weights=weights, weightsmodifier=None)
        # Ensure all selections exist before calling cutflow
        # Add protection for the cutflow if the selection is not in the cutflow

        if self.isCutflow:
            required_selections = ['TotalEntries', 'HLT_filter', 'lumi_mask', 'event_quality_flags',
                               'muon_pT_roch', 'muon_eta', 'muon_id', 'muon_isGlobal_or_Tracker', 'muon_selection', 'muon_iso', 'nmuons',
                                'mm_charge', 'electron_veto', 'HemVeto']
            # , weights=weights, weightsmodifier=None) # FIXME: weights and weightsmodifier are availalbe starting coffea: 2025.3.0
            self.cutflow = self.selection.cutflow(*required_selections)
            logger.info(f"cutflow: {self.cutflow}")
            logger.info(f"self.cutflow.logger.info(): {self.cutflow.print()}")
            # logger.info(f"self.cutflow.logger.info(): {self.cutflow.logger.info(weighted=False)}") # FIXME: weights and weightsmodifier are availalbe starting coffea: 2025.3.0
            # logger.info(f"self.cutflow.result(): {self.cutflow.result()}")
        t22 = time.perf_counter()
        logger.info(f"[timing] Cutflow time: {t22 - t21:.2f} seconds")

        return out_dict

    def postprocess(self, accumulator):
        """
        Arbitrary postprocess function that's required to run the processor
        """
        logger.info(f"postprocess: {accumulator}")
        return accumulator

    def get_mass_resolution(self, dimuon, mu1,mu2, is_mc:bool, doing_BS_correction=False, test_mode=False):
        # Returns absolute mass resolution!
        muon_E = dimuon.mass /2
        dpt1 = (mu1.ptErr / mu1.pt) * muon_E
        dpt2 = (mu2.ptErr / mu2.pt) * muon_E
        logger.debug(f"muons mass_resolution dpt1: {dpt1}")

        year = self.config["year"]
        if "2016" in year: # 2016PreVFP, 2016PostVFP, 2016_RERECO
            yearUL = "2016"
        elif ("22" in year) or ("23" in year):# temporary solution until I can generate my own dimuon mass resolution
            yearUL = "2018"
        elif "RERECO" in year: # 2017_RERECO. 2018_RERECO
            yearUL=year.replace("_RERECO","")
        else:
            yearUL = year
        if doing_BS_correction: # apply resolution calibration from BeamSpot constraint correction
            # TODO: add 2016pre and 2016post versions too
            logger.debug("Doing BS constraint correction mass calibration!")

            # Load the correction set
            json_path = self.config["BS_res_calib_path"]["MC"] if is_mc else self.config["BS_res_calib_path"]["Data"]
            correction_set = correctionlib.CorrectionSet.from_file(json_path)

            # Access the specific correction by name
            correction = correction_set["BS_ebe_mass_res_calibration"]
            logger.debug(f"correction_set: {correction_set}")
            logger.debug(f"correction: {correction}")

            calibration = correction.evaluate(mu1.pt, abs(mu1.eta), abs(mu2.eta))
        else:
            if is_mc:
                label = f"res_calib_MC_{yearUL}"
            else:
                label = f"res_calib_Data_{yearUL}"
            logger.debug(f"yearUL: {yearUL}")
            calibration =  self.evaluator[label]( # this is a coffea.dense_lookup instance
                mu1.pt,
                abs(mu1.eta),
                abs(mu2.eta) # calibration depends on year, data/mc, pt, and eta region for each muon (ie, BB, BO, OB, etc)
            )

        # return ((dpt1 * dpt1 + dpt2 * dpt2)**0.5) * calibration
        return ((dpt1 * dpt1 + dpt2 * dpt2)**0.5), calibration
        # return ((dpt1 * dpt1 + dpt2 * dpt2)**0.5) # turning calibration off for calibration factor recalculation

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

        if events.metadata["is_mc"]:
            # pt_gen is used for JEC (one of the factory name map values)
            events["Jet", "pt_gen"] =  ak.values_astype(
                ak.fill_none(events.Jet.matched_gen.pt, value=0.0),
                "float32"
            )
            events["Jet", "has_matched_gen"] = events.Jet.genJetIdx > 0
        else:
            events["Jet", "has_matched_gen"] = False

        return

    # def prepare_lookups(self):
    # JEC, JER and uncertainties
    # self.jec_factories_mc, self.jec_factories_data = get_jec_factories(
    #     self.config["jec_parameters"],
    #     self.year
    # )

    # # Muon scale factors
    # self.musf_lookup = musf_lookup(self.parameters)
    # # Pile-up reweighting
    # #self.pu_lookups = pu_lookups(self.parameters)
    # # Btag weights
    # #self.btag_csv = BTagScaleFactor(
    #     #self.parameters["btag_sf_csv"],
    #     #BTagScaleFactor.RESHAPE,
    #     #"iterativefit,iterativefit,iterativefit",
    # #)
    # self.btag_json =  correctionlib.CorrectionSet.from_file(self.parameters["btag_sf_json"],)

    # # STXS VBF cross-section uncertainty
    # self.stxs_acc_lookups, self.powheg_xsec_lookup = stxs_lookups()

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
        do_jec = False,
        do_jecunc = False,
        do_jerunc = False,
        event_match = None
    ):
        logger.debug(f'variation: {variation}')
        is_mc = events.metadata["is_mc"]
        dataset = events.metadata["dataset"]
        year = self.config["year"]
        if (not is_mc) and variation != "nominal":
            return {}

        # Find jets that have selected muons within dR<0.4 from them -> line 465 of AN-19-124

        # matched_mu_pt = jets.matched_muons.pt_fsr if "pt_fsr" in jets.matched_muons.fields else jets.matched_muons.pt
        matched_mu_pt = jets.matched_muons.pt_raw # afaik, matched muons are muons that are within dr < 0.4 to jets
        matched_mu_eta = jets.matched_muons.eta_raw
        matched_mu_iso = jets.matched_muons.pfRelIso04_all
        matched_mu_id = jets.matched_muons[self.config["muon_id"]]
        # logger.info(f'self.config["muon_id": {self.config["muon_id"]}')
        # logger.info(f"matched_mu_id: {matched_mu_id.compute()}")
        # logger.info(f"jets.matched_muons: {jets.matched_muons.compute()}")
        # AN-19-124 line 465: "Jets are also cleaned w.r.t. the selected muon candidates by requiring a geometrical separation of ∆R ( j, µ ) > 0.4"

        # --------------------------------------
        # matched_mu_pass = ( # apply the same muon selection condition from before
        #     (matched_mu_pt > self.config["muon_pt_cut"])
        #     & (abs(matched_mu_eta) < self.config["muon_eta_cut"])
        #     & (matched_mu_iso < self.config["muon_iso_cut"])
        #     & matched_mu_id
        #     & (jets.matched_muons.isGlobal | jets.matched_muons.isTracker) # Table 3.5 AN-19-124
        # )
        # # logger.info(f"matched_mu_pass: {matched_mu_pass.compute()}")
        # # logger.info(f"ak.sum(matched_mu_pass, axis=2): {ak.sum(matched_mu_pass, axis=2).compute()}")
        # matched_mu_pass = ak.sum(matched_mu_pass, axis=2) > 0 # there's at least one matched mu that passes the muon selection
        # clean = ~(ak.fill_none(matched_mu_pass, value=False))
        # # # logger.info(f"clean: {clean.compute()}")
        # # # logger.info(f"jets: {jets.compute()}")
        # --------------------------------------

        # apply clean jet selection
        # mu1_jet_dR = jets.delta_r(mu1[:, np.newaxis])
        _, _, mu1_jet_dR = delta_r_V1(
            mu1[:, np.newaxis].eta_raw,
            jets.eta,
            mu1[:, np.newaxis].phi_raw,
            jets.phi,
        )
        matched_mu1_jet = mu1_jet_dR <= 0.4
        matched_mu1_jet = ak.fill_none(matched_mu1_jet, value=False)

        # mu2_jet_dR = jets.delta_r(mu2[:, np.newaxis])
        _, _, mu2_jet_dR = delta_r_V1(
            mu2[:, np.newaxis].eta_raw,
            jets.eta,
            mu2[:, np.newaxis].phi_raw,
            jets.phi,
        )
        matched_mu2_jet = mu2_jet_dR <= 0.4
        matched_mu2_jet = ak.fill_none(matched_mu2_jet, value=False)

        matched_mu_pass = matched_mu1_jet | matched_mu2_jet
        # matched_mu_pass = ak.sum(matched_mu_pass, axis=2) > 0
        # clean = ~(ak.fill_none(matched_mu_pass, value=False))
        clean = ~matched_mu_pass
        clean = ak.fill_none(clean, value=True)

        # # Select particular JEC variation

        if is_mc and (variation != "nominal"):
            fields2add = [
                "puId",
                "jetId",
                "qgl",
                "rho",
                "area",
                "btagDeepB",
            ]
            jets =  get_jet_variation(jets, variation, fields2add)

        # if "jer" in variation: # https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution#JER_Scaling_factors_and_Uncertai
        #     logger.info("doing JER unc!")
        #     jer_mask_dict ={
        #         "jer1" : abs(jets.eta) < 1.93,
        #         "jer2" : (abs(jets.eta) > 1.93) & (abs(jets.eta) < 2.5),
        #         "jer3" : (abs(jets.eta) > 2.5) & (abs(jets.eta) < 3.0) & (jets.pt < 50),
        #         "jer4" : (abs(jets.eta) > 2.5) & (abs(jets.eta) < 3.0) & (jets.pt > 50),
        #         "jer5" : (abs(jets.eta) > 3.0) & (abs(jets.eta) < 5.0) & (jets.pt < 50),
        #         "jer6" : (abs(jets.eta) > 3.0) & (abs(jets.eta) < 5.0) & (jets.pt > 50),
        #     }
        #     jets_nominal = jets
        #     logger.info(f"JER variation: {variation}")
        #     if "_up" in variation:
        #         unc_name = variation.replace("_up", "")
        #         logger.info(f"unc_name: {unc_name}")
        #         jets_jer_up = jets['JER']['up']
        #         jer_mask = jer_mask_dict[unc_name]
        #         jets = ak.where(jer_mask, jets_jer_up, jets_nominal)
        #     elif "_down" in variation:
        #         unc_name = variation.replace("_down", "")
        #         logger.info(f"unc_name: {unc_name}")
        #         jets_jer_down = jets['JER']['down']
        #         jer_mask = jer_mask_dict[unc_name]
        #         jets = ak.where(jer_mask, jets_jer_down, jets_nominal)
        # else: # if jec uncertainty
        #     if "_up" in variation:
        #         logger.info("doing JEC unc!")
        #         unc_name = "JES_" + variation.replace("_up", "")
        #         if unc_name not in jets.fields:
        #             return
        #         jets = jets[unc_name]["up"]
        #     elif "_down" in variation:
        #         logger.info("doing JEC unc!")
        #         unc_name = "JES_" + variation.replace("_down", "")
        #         if unc_name not in jets.fields:
        #             return
        #         jets = jets[unc_name]["down"]
        #     else:
        #         jets = jets

        # if variation == "nominal":
        #     # Update pt and mass if JEC was applied
        #     if do_jec:
        #         jets["pt"] = jets["pt_jec"]
        #         jets["mass"] = jets["mass_jec"]

        # # ------------------------------------------------------------#
        # # Apply jetID and PUID
        # # ------------------------------------------------------------#

        pass_jet_id = jet_id(jets, self.config)

        logger.debug(f"jet loop NanoAODv: {NanoAODv}")
        is_2017 = "2017" in year
        if NanoAODv == 9  or NanoAODv == 12:
            pass_jet_puid = jet_puid(jets, self.config)
        else: # NanoAODv12 doesn't have Jet_PuID yet
            pass_jet_puid = ak.ones_like(pass_jet_id, dtype="bool")
        # ------------------------------------------------------------#
        # Select jets
        # ------------------------------------------------------------#
        # get QGL cut
        if NanoAODv == 9 :
            jets["qgl"] = jets.qgl
        elif ((NanoAODv == 12 or NanoAODv == 15) and year in ["2016preVFP", "2016postVFP", "2017", "2018"]): # NanoAODv12 and NanoAODv15 have qgl as a field as AK4 jets are CHS for run-2
            jets["qgl"] = jets.qgl
            jets["btagPNetQvG"] = jets.qgl
            jets["btagDeepFlavQG"] = jets.qgl
        else: # NanoAODv12
            jets["btagPNetQvG"] = jets.btagPNetQvG
            jets["btagDeepFlavQG"] = jets.btagDeepFlavQG

        jet_pt_cut = (jets.pt > self.config["jet_pt_cut"])
        # add additonal pT cut for the forward regions to reduce jet horn  ----------------------------------------------
        # source: https://nam04.safelinks.protection.outlook.com/?url=https%3A%2F%2Findico.cern.ch%2Fevent%2F1434807%2Fcontributions%2F6040633%2Fattachments%2F2893077%2F5071932%2FJERC%2520meeting%252009_07.pdf&data=05%7C02%7Cyun79%40purdue.edu%7C3d76cc7f47974533372708dd896f875a%7C4130bd397c53419cb1e58758d6d63f21%7C0%7C0%7C638817834635140303%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&sdata=fh11i5iJCGo0EQKYBdw0Df8oaesOX2hCnJ%2FU78o37%2BU%3D&reserved=0
        jetHorn_region = abs(jets.eta) > 2.5
        jetHorn_pt_cut = (jets.pt > self.config["jet_pt_cut"]) # pt cut on jethorn doesn't change
        jetHorn_puid_cut = (jets.puId >= 7) | (jets.pt >= 50) # tight pu Id #FIXME: hardcoded puID
        jetHorn_cut = jetHorn_pt_cut & jetHorn_puid_cut
        jetHorn_PUID_cut = ak.ones_like(pass_jet_puid, dtype="bool") # default value is True
        jetHorn_PUID_cut = ak.where(jetHorn_region, jetHorn_cut, jetHorn_PUID_cut)

        # add additonal pT cut for the forward regions  ----------------------------------------------
        jet_selection = (
            jet_pt_cut
            & pass_jet_id
            & pass_jet_puid
            & clean
            & jetHorn_PUID_cut
            & (abs(jets.eta) < self.config["jet_eta_cut"])
        )

        # jets = jets[jet_selection] # this causes huuuuge memory overflow close to 100 GB. Without it, it goes to around 20 GB
        jets = jets[jet_selection]
        # jets = ak.to_layout(jets)
        jets = ak.to_packed(jets)

        # apply jetpuid if not have done already
        if is_mc and (variation=="nominal"):
            jetpuid_weight = get_jetpuid_weights_eta_dependent(year, jets, self.config) # FIXME
            # now we add jetpuid_wgt
            weights.add("jetpuid_wgt",
                    weight=jetpuid_weight,
            )

        # jets = ak.where(jet_selection, jets, None)
        # muons = events.Muon
        njets = ak.num(jets, axis=1)

        # # FIXME: Out of all jets get the two jets with maximum di-jet rapidity gap
        # # if there are at least two jets
        # print the pt, eta and phi of the first 3 events for debugging
        # logger.info(f"Before: jets[:3].pt: {jets[:3].pt.compute()}")
        # logger.info(f"Before: jets[:3].eta: {jets[:3].eta.compute()}")
        # logger.info(f"Before: jets[:3].phi: {jets[:3].phi.compute()}")

        # gl_pair = ak.cartesian({"jet1": jets, "jet2": jets}, axis=1, nested=True)
        # dr_j12 = gl_pair["jet1"].delta_r(gl_pair["jet2"])
        # # print out the dr_j12 for the first 3 events for debugging
        # logger.info(f"dr_j12[:3]: {dr_j12[:3].compute()}")

        # # get pair with max dr and then sort them in pt. After that print out the pt, eta and phi of the first 3 events for debugging
        # gl_pair = gl_pair[dr_j12.argmax(axis=2)]
        # gl_pair = gl_pair[gl_pair.jet1.pt > gl_pair.jet2.pt]
        # logger.info(f"After dr_j12.argmax and pt sort: gl_pair[:3].jet1.pt: {gl_pair[:3].jet1.pt.compute()}")
        # logger.info(f"After dr_j12.argmax and pt sort: gl_pair[:3].jet1.eta: {gl_pair[:3].jet1.eta.compute()}")
        # logger.info(f"After dr_j12.argmax and pt sort: gl_pair[:3].jet1.phi: {gl_pair[:3].jet1.phi.compute()}")
        # logger.info(f"After dr_j12.argmax and pt sort: gl_pair[:3].jet2.pt: {gl_pair[:3].jet2.pt.compute()}")
        # logger.info(f"After dr_j12.argmax and pt sort: gl_pair[:3].jet2.eta: {gl_pair[:3].jet2.eta.compute()}")
        # logger.info(f"After dr_j12.argmax and pt sort: gl_pair[:3].jet2.phi: {gl_pair[:3].jet2.phi.compute()}")
        # # print dr of jet1 and jet2 for the first 3 events for debugging
        # logger.info(f"After dr_j12.argmax and pt sort: gl_pair[:3].dr: {gl_pair[:3].jet1.delta_r(gl_pair[:3].jet2).compute()}")

        # ------------------------------------------------------------#
        # Fill jet-related variables
        # ------------------------------------------------------------#
        padded_jets = ak.pad_none(jets, target=4) # padd jets
        jet1 = padded_jets[:,0]
        jet2 = padded_jets[:,1]
        jet3 = padded_jets[:,2]
        jet4 = padded_jets[:,3]

        dijet = jet1+jet2

        # jet1_4D_vec = ak.zip({"x":jet1.x, "y":jet1.y, "z":jet1.z, "E":jet1.E}, with_name="Momentum4D")
        # jet2_4D_vec = ak.zip({"x":jet2.x, "y":jet2.y, "z":jet2.z, "E":jet2.E}, with_name="Momentum4D")
        # dijet = jet1_4D_vec+jet2_4D_vec
        # logger.info(f"dijet: {dijet}")
        jj_dEta = abs(jet1.eta - jet2.eta)
        jj_dPhi = abs(jet1.delta_phi(jet2))
        # dimuon = muons[:,0] + muons[:,1]
        mmj1_dEta = abs(dimuon.eta - jet1.eta)
        mmj2_dEta = abs(dimuon.eta - jet2.eta)

        min_dEta_filter  = ak.fill_none((mmj1_dEta < mmj2_dEta), value=True)
        mmj_min_dEta = ak.where(
            min_dEta_filter,
            mmj1_dEta,
            mmj2_dEta,
        )
        # logger.info(f"mmj_min_dEta: {mmj_min_dEta.compute()}")
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
        # logger.info(f"mmj_min_dPhi: {mmj_min_dPhi.compute()}")
        # zeppenfeld definition in  line 1118 in the AN
        dimuon_rapidity = getRapidity(dimuon)
        jet1_rapidity = getRapidity(jet1)
        jet2_rapidity = getRapidity(jet2)
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
        # pt_centrality = dimuon.pt - dijet.pt/2
        # j12_subtract_pt = p4_subtract_pt(jet1, jet2) # pt of momentum vector subtraction of jet1 and jet2
        # pt_centrality = pt_centrality / j12_subtract_pt

        jet_loop_out_dict = {
            f"jet1_pt_{variation}": jet1.pt,
            f"jet1_eta_{variation}": jet1.eta,
            f"jet1_rapidity_{variation}": jet1_rapidity,  # max rel err: 0.7394
            f"jet1_phi_{variation}": jet1.phi,
            f"jet1_qgl_{variation}": jet1.qgl,  # FIXME: NanoAODv12 and NanoAODv15 have qgl as a field as AK4 jets are CHS for run-2, but not for run-3
            f"jet1_btagPNetQvG_{variation}": jet1.btagPNetQvG,
            f"jet1_btagDeepFlavQG_{variation}": jet1.btagDeepFlavQG,
            f"jet1_jetId_{variation}": jet1.jetId,
            f"jet1_puId_{variation}": jet1.puId,
            f"jet1_mass_{variation}": jet1.mass,
            f"jet1_area_{variation}": jet1.area,
            # -------------------------
            f"jet2_pt_{variation}": jet2.pt,
            f"jet2_eta_{variation}": jet2.eta,
            f"jet2_rapidity_{variation}": jet2_rapidity,  # max rel err: 0.781
            f"jet2_phi_{variation}": jet2.phi,
            f"jet2_qgl_{variation}": jet2.qgl,  # FIXME: NanoAODv12 and NanoAODv15 have qgl as a field as AK4 jets are CHS for run-2, but not for run-3
            f"jet2_btagPNetQvG_{variation}": jet2.btagPNetQvG,
            f"jet2_btagDeepFlavQG_{variation}": jet2.btagDeepFlavQG,
            f"jet2_jetId_{variation}": jet2.jetId,
            f"jet2_puId_{variation}": jet2.puId,
            f"jet2_mass_{variation}": jet2.mass,
            f"jet2_area_{variation}": jet2.area,
            # -------------------------
            f"jet3_pt_{variation}": jet3.pt,
            f"jet3_eta_{variation}": jet3.eta,
            f"jet3_rapidity_{variation}": jet3_rapidity,  # max rel err: 0.781
            f"jet3_phi_{variation}": jet3.phi,
            f"jet3_qgl_{variation}": jet3.qgl,  # FIXME: NanoAODv12 and NanoAODv15 have qgl as a field as AK4 jets are CHS for run-2, but not for run-3
            f"jet3_btagPNetQvG_{variation}": jet3.btagPNetQvG,
            f"jet3_btagDeepFlavQG_{variation}": jet3.btagDeepFlavQG,
            f"jet3_jetId_{variation}": jet3.jetId,
            f"jet3_puId_{variation}": jet3.puId,
            f"jet3_mass_{variation}": jet3.mass,
            f"jet3_area_{variation}": jet3.area,
            # -------------------------
            f"jet4_pt_{variation}": jet4.pt,
            f"jet4_eta_{variation}": jet4.eta,
            f"jet4_rapidity_{variation}": jet4_rapidity,  # max rel err: 0.781
            f"jet4_phi_{variation}": jet4.phi,
            f"jet4_qgl_{variation}": jet4.qgl,  # FIXME: NanoAODv12 and NanoAODv15 have qgl as a field as AK4 jets are CHS for run-2, but not for run-3
            f"jet4_btagPNetQvG_{variation}": jet4.btagPNetQvG,
            f"jet4_btagDeepFlavQG_{variation}": jet4.btagDeepFlavQG,
            f"jet4_jetId_{variation}": jet4.jetId,
            f"jet4_puId_{variation}": jet4.puId,
            f"jet4_mass_{variation}": jet4.mass,
            f"jet4_area_{variation}": jet4.area,
            # -------------------------
            f"jj_mass_{variation}": dijet.mass,
            # f"jj_mass_{variation}" : p4_sum_mass(jet1,jet2),
            f"jj_mass_log_{variation}": np.log(dijet.mass),
            f"jj_pt_{variation}": dijet.pt,
            f"jj_eta_{variation}": dijet.eta,
            f"jj_phi_{variation}": dijet.phi,
            f"jj_dEta_{variation}": jj_dEta,
            f"jj_dPhi_{variation}": jj_dPhi,
            f"mmj1_dEta_{variation}": mmj1_dEta,
            f"mmj1_dPhi_{variation}": mmj1_dPhi,
            f"mmj1_dR_{variation}": mmj1_dR,
            f"mmj2_dEta_{variation}": mmj2_dEta,
            f"mmj2_dPhi_{variation}": mmj2_dPhi,
            f"mmj2_dR_{variation}": mmj2_dR,
            f"mmj_min_dEta_{variation}": mmj_min_dEta,
            f"mmj_min_dPhi_{variation}": mmj_min_dPhi,
            f"mmjj_pt_{variation}": mmjj.pt,
            f"mmjj_eta_{variation}": mmjj.eta,
            f"mmjj_phi_{variation}": mmjj.phi,
            f"mmjj_mass_{variation}": mmjj.mass,
            f"rpt_{variation}": rpt,
            f"pt_centrality_{variation}": pt_centrality,
            f"zeppenfeld_{variation}": zeppenfeld,
            f"ll_zstar_log_{variation}": np.log(np.abs(zeppenfeld)),
            f"njets_{variation}": njets,
        }
        if is_mc and (variation == "nominal"):
            nominal_dict = {
                f"jet1_pt_gen_{variation}" : jet1.pt_gen,
                f"jet2_pt_gen_{variation}" : jet2.pt_gen,
            }
            jet_loop_out_dict.update(nominal_dict)

        if (variation == "nominal"):
            nominal_dict = {
                f"jet1_pt_raw_{variation}" : jet1.pt_raw,
                f"jet2_pt_raw_{variation}" : jet2.pt_raw,
                f"jet1_mass_raw_{variation}" : jet1.mass_raw,
                f"jet2_mass_raw_{variation}" : jet2.mass_raw,
                f"jet1_mass_jec_{variation}" : jet1.mass_jec,
                f"jet2_mass_jec_{variation}" : jet2.mass_jec,
                f"jet1_pt_jec_{variation}" : jet1.pt_jec,
                f"jet2_pt_jec_{variation}" : jet2.pt_jec,
            }
            jet_loop_out_dict.update(nominal_dict)

        # ------------------------------------------------------------#
        # Fill soft activity jet variables
        # ------------------------------------------------------------#

        # Effect of changes in jet acceptance should be negligible,
        # no need to calcluate this for each jet pT variation

        sj_dict = {}
        sj_dict_HIG19006 = {}
        cutouts = [2,5]
        nmuons = ak.num(events.Muon, axis=1) # FIXME (I think it should be selected muons)
        # PLEASE NOTE: SoftJET variables are all from Nominal variation despite variation names
        for cutout in cutouts:
            sj_out = fill_softjets(events, jets, mu1, mu2, nmuons, cutout) # obtain nominal softjet values
            sj_out = { # add variation even thought it's always nominal
                key+"_"+variation : val \
                for key, val in sj_out.items()
            }
            sj_dict.update(sj_out)

            sj_out_HIG19006 = fill_softjets_HIG19006(events, jets, mu1, mu2, nmuons, cutout) # obtain nominal softjet values
            sj_out_HIG19006 = { # add variation even thought it's always nominal
                key+"_"+variation : val \
                for key, val in sj_out_HIG19006.items()
            }
            sj_dict_HIG19006.update(sj_out_HIG19006)

        logger.debug(f"sj_dict.keys(): {sj_dict.keys()}")
        jet_loop_out_dict.update(sj_dict)
        jet_loop_out_dict.update(sj_dict_HIG19006)

        # ------------------------------------------------------------#
        # Apply remaining cuts
        # ------------------------------------------------------------#

        # Cut has to be defined here because we will use it in
        # b-tag weights calculation
        # vbf_cut = (dijet.mass > 400) & (jj_dEta > 2.5) & (jet1.pt > 35) # the extra jet1 pt cut is for Dmitry's Vbf cut, but that doesn't exist on AN-19-124's ggH category cut

        # vbf_cut = (dijet.mass > 400) & (jj_dEta > 2.5)
        # vbf_cut = ak.fill_none(vbf_cut, value=False)
        # jet_loop_out_dict.update({"vbf_cut": vbf_cut})

        # # ------------------------------------------------------------#
        # # Calculate QGL weights, btag SF and apply btag veto
        # # ------------------------------------------------------------#
        if is_mc and (variation == "nominal"):
            # --- QGL weights  start --- #
            isHerwig = "herwig" in dataset
            logger.debug("adding QGL weights!")

            # keep dims start -------------------------------------
            # qgl_wgts = qgl_weights_keepDim(jet1, jet2, njets, isHerwig)
            qgl_wgts = qgl_weights_V2(jets, self.config, isHerwig)
            # keep dims end -------------------------------------
            weights.add("qgl_wgt",
                        weight=qgl_wgts["nom"],
                        weightUp=qgl_wgts["up"],
                        weightDown=qgl_wgts["down"]
            )

            #     # --- QGL weights  end --- #

            #     # # --- Btag weights  start--- #
            do_btag_wgt = True # True
            if NanoAODv ==12:
                do_btag_wgt = False # temporary condition
            if do_btag_wgt:
                logger.info("doing btag wgt!")
                bjet_sel_mask = ak.ones_like(njets) #& two_jets & vbf_cut
                btag_systs = self.config["btag_systs"] #if do_btag_syst else []
                if "RERECO" in year:
                    # if True:
                    btag_json = BTagScaleFactor(
                    self.config["btag_sf_csv"],
                    BTagScaleFactor.RESHAPE,
                    "iterativefit,iterativefit,iterativefit",
                )
                else:
                    btag_file =  correctionlib.CorrectionSet.from_file(self.config["btag_sf_json"],)
                    # btag_json=btag_file["deepJet_shape"]
                    btag_json=btag_file["deepCSV_shape"]

                # keep dims start -------------------------------------
                btag_wgt, btag_syst = btag_weights_jsonKeepDim(
                            self, btag_systs, jets, weights, bjet_sel_mask, btag_json
                )
                weights.add("btag_wgt",
                        weight=btag_wgt,
                )
                # --- Btag weights variations --- #
                for name, bs in btag_syst.items():
                    logger.info(f"{name} value: {bs}")
                    weights.add(f"btag_wgt_{name}",
                        weight=ak.ones_like(btag_wgt),
                        weightUp=bs["up"],
                        weightDown=bs["down"]
                    )
                # TODO: add btag systematics by adding seperate wgts
                # keep dims end -------------------------------------
                # logger.info(f"btag_wgt: {ak.to_numpy(btag_wgt.compute())}")
                # logger.info(f"btag_syst['jes_up']: {ak.to_numpy(btag_syst['jes']['up'].compute())}")
                # logger.info(f"btag_syst['jes_down']: {ak.to_numpy(btag_syst['jes']['down'].compute())}")
            # # --- Btag weights end --- #

            # logger.info(f"weight nom b4 adding btag: {ak.to_numpy(weights.weight().compute())}")
            # adding btag wgt directly to weights doesn't work, this may
            # have to do with the fact that we use weights.weight() to
            # calculate btag_wgt, so save this separtely and apply it later
            # weights.add("btag_wgt",
            #             weight=btag_wgt
            # )
            # logger.info(f"btag_wgt: {ak.to_numpy(btag_wgt.compute())}")
            # logger.info(f"weight statistics: {weights.weightStatistics.keys()}")
            # logger.info(f"weight nom after adding btag: {ak.to_numpy(weights.weight().compute())}")

        #     # --- Btag weights variations --- #
        #     for name, bs in btag_syst.items():
        #         weights.add_weight(f"btag_wgt_{name}", bs, how="only_vars")

        # Separate from ttH and VH phase space

        if "RERECO" in year:
            btagLoose_filter = (jets.btagDeepB > self.config["btag_loose_wp"]) & (abs(jets.eta) < 2.5) # original value
            btagMedium_filter = (jets.btagDeepB > self.config["btag_medium_wp"]) & (abs(jets.eta) < 2.5)
        else: # UL
            # NOTE: maybe keep the nBtagLoose and nBtagMedium deepbFlavB as a separate variable for quick testing
            # btagLoose_filter = (jets.btagDeepFlavB > self.config["btag_loose_wp"]) & (abs(jets.eta) < 2.5)
            # btagMedium_filter = (jets.btagDeepFlavB > self.config["btag_medium_wp"]) & (abs(jets.eta) < 2.5)
            btagLoose_filter = (jets.btagDeepB > self.config["btag_loose_wp"]) & (abs(jets.eta) < 2.5)
            btagMedium_filter = (jets.btagDeepB > self.config["btag_medium_wp"]) & (abs(jets.eta) < 2.5)

        btagLoose_filter = ak.fill_none(btagLoose_filter, value=False)
        btagMedium_filter = ak.fill_none(btagMedium_filter, value=False)

        nBtagLoose = ak.sum(btagLoose_filter, axis=1)
        nBtagMedium = ak.sum(btagMedium_filter, axis=1)

        # #quick sanity check
        # logger.info(f"nBtagLoose : {nBtagLoose[:20].compute()}")
        # logger.info(f"btagLoose_filter sum : {ak.sum(btagLoose_filter, axis=1)[:20].compute()}")
        # logger.info(f"nBtagMedium : {nBtagMedium[:20].compute()}")
        # logger.info(f"btagMedium_filter sum : {ak.sum(btagMedium_filter, axis=1)[:20].compute()}")
        # raise ValueError

        # logger.info(f"nBtagLoose: {jets.btagDeepFlavB.compute()}")
        # logger.info(f"nBtagLoose: {ak.to_numpy(nBtagLoose.compute())}")
        # logger.info(f"njets: {ak.to_numpy(njets.compute())}")
        temp_out_dict = {
            f"nBtagLoose_{variation}": nBtagLoose,
            f"nBtagMedium_{variation}": nBtagMedium,
        }
        jet_loop_out_dict.update(temp_out_dict)

        # --------------------------------------------------------------#
        # Fill outputs
        # --------------------------------------------------------------#

        # variables.update({"wgt_nominal": weights.get_weight("nominal")})

        # All variables are affected by jet pT because of jet selections:
        # a jet may or may not be selected depending on pT variation.

        #     for key, val in variables.items():
        #         output.loc[:, pd.IndexSlice[key, variation]] = val

        return jet_loop_out_dict
