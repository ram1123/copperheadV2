from coffea.jetmet_tools import CorrectedJetsFactory, JECStack
# from src.corrections.custom_jec import CorrectedJetsFactory, JECStack
from coffea.lookup_tools import extractor
import numpy as np
import awkward as ak
import os
import correctionlib.schemav2 as cs
import correctionlib

import coffea.nanoevents.methods.candidate as candidate
from omegaconf import OmegaConf
import random

from modules.utils import logger
from modules.classify_year import is_run3, is_run2
from modules.correctionlib_file_cache import get_corrset, get_corr_input_names

def printCorrObjInputs(corr_obj):
    input_names = [inp.name for inp in corr_obj.inputs]
    logger.debug(f"printCorrObjInputs names: \n {input_names}")

def jec_names_and_sources(jec_pars):
    # logger.debug(f"jec_pars: {jec_pars}")
    jet_alg = jec_pars["jet_algorithm"]
    names = {}
    suffix = {
        "jec_names": [f"_{level}_{jet_alg}" for level in jec_pars["jec_levels_mc"]],
        "jec_names_data": [
            f"_{level}_{jet_alg}" for level in jec_pars["jec_levels_data"]
        ],
        "junc_names": [f"_Uncertainty_{jet_alg}"],
        "junc_names_data": [f"_Uncertainty_{jet_alg}"],
        "junc_sources": [f"_UncertaintySources_{jet_alg}"],
        "junc_sources_data": [f"_UncertaintySources_{jet_alg}"],
        "jer_names": [f"_PtResolution_{jet_alg}"],
        "jersf_names": [f"_SF_{jet_alg}"],
    }
    # logger.debug(f"JEC suffix: {suffix}")

    for key, suff in suffix.items():
        if "data" in key:
            names[key] = {}
            for run in jec_pars["runs"]:
                for tag, iruns in jec_pars["jec_data_tags"].items():
                    if run in iruns:
                        names[key].update({run: [f"{tag}{s}" for s in suff]})
        else:
            tag = jec_pars["jer_tags"] if "jer" in key else jec_pars["jec_tags"]
            names[key] = [f"{tag}{s}" for s in suff]

    return names


def jec_weight_sets(jec_pars, year):
    weight_sets = {}
    names = jec_names_and_sources(jec_pars)

    extensions = {
        "jec_names": "jec",
        "jer_names": "jr",
        "jersf_names": "jersf",
        "junc_names": "junc",
        "junc_sources": "junc",
    }

    weight_sets["jec_weight_sets_mc"] = []
    weight_sets["jec_weight_sets_data"] = []

    for opt, ext in extensions.items():
        # MC
        weight_sets["jec_weight_sets_mc"].extend(
            [f"* * data/jec/{name}.{ext}.txt" for name in names[opt]]
        )
        # Data
        if "jer" in opt:
            continue
        data = []
        for run, items in names[f"{opt}_data"].items():
            data.extend(items)
        data = list(set(data))
        weight_sets["jec_weight_sets_data"].extend(
            [f"* * data/jec/{name}.{ext}.txt" for name in data]
        )

    # return weight_sets
    return (weight_sets, names)


def get_name_map(stack):
    name_map = stack.blank_name_map
    name_map["JetPt"] = "pt"
    name_map["JetMass"] = "mass"
    name_map["JetEta"] = "eta"
    name_map["JetA"] = "area"
    name_map["ptGenJet"] = "pt_gen"
    name_map["ptRaw"] = "pt_raw"
    name_map["massRaw"] = "mass_raw"
    name_map["Rho"] = "PU_rho" # IMPORTANT: do NOT override "rho" in jets. rho is used for something else, thus we NEED to use PU_rho
    # logger.debug(f"name_map: {name_map}")
    return name_map

def get_jec_factories(jec_parameters: dict, year):
    jec_pars = jec_parameters

    weight_sets, names = jec_weight_sets(jec_pars, year)

    jec_factories = {}
    jec_factories_data = {}

    # Prepare evaluators for JEC, JER and their systematics
    jetext = extractor()
    jetext.add_weight_sets(weight_sets["jec_weight_sets_mc"])
    jetext.add_weight_sets(weight_sets["jec_weight_sets_data"])
    jetext.finalize()
    jet_evaluator = jetext.make_evaluator()

    stacks_def = {
        "jec_stack": ["jec_names"],
        "jer_stack": ["jer_names", "jersf_names"],
        "junc_stack": ["junc_names"],
    }

    stacks = {}
    for key, vals in stacks_def.items():
        stacks[key] = []
        for v in vals:
            stacks[key].extend(names[v])

    jec_input_options = {}
    jet_variations = ["jec", "junc", "jer"]

    for variation in jet_variations:
        # jec_input_options[variation] = {
        #     name: jet_evaluator[name] for name in stacks[f"{variation}_stack"]
        # }
        """
        matches names specific for jet variation with the appropriate jet evaluator
        """
        # jec_input_options[opt] = {
        #     name: jet_evaluator[name] for name in stacks[f"{opt}_stack"]
        # }
        jec_input_options[variation] ={}
        for name in stacks[f"{variation}_stack"]:
            jec_input_options[variation][name] =jet_evaluator[name]

    # logger.debug(f"jec_factories jec_input_options: \n {jec_input_options}")
    for src in names["junc_sources"]:
        for key in jet_evaluator.keys():
            if src in key:
                jec_input_options["junc"][key] = jet_evaluator[key]

    # Create separate factories for JEC, JER, JEC variations
    for variation in jet_variations:
        stack = JECStack(jec_input_options[variation])
        # logger.debug(f"jec_factories JECStack: {stack}")
        # logger.debug(f"jec_factories get_name_map(stack): {get_name_map(stack)}")
        jec_factories[variation] = CorrectedJetsFactory(get_name_map(stack), stack)

    # Create a separate factory for each data run
    for run in jec_pars["runs"]:
        jec_inputs_data = {}
        for opt in ["jec", "junc"]:
            jec_inputs_data.update(
                {name: jet_evaluator[name] for name in names[f"{opt}_names_data"][run]}
            )
        for src in names["junc_sources_data"][run]:
            for key in jet_evaluator.keys():
                if src in key:
                    jec_inputs_data[key] = jet_evaluator[key]

        jec_stack_data = JECStack(jec_inputs_data)
        jec_factories_data[run] = CorrectedJetsFactory(
            get_name_map(jec_stack_data), jec_stack_data
        )

    return jec_factories, jec_factories_data


def custom_jet_id(jets, year, jet_type="AK4PUPPI"):
    """
    https://twiki.cern.ch/twiki/bin/view/CMS/JetID13TeVUL#Preliminary_Recommendations_for

     Returns:
       pass_tight, pass_tight_lepveto
    """
    if jet_type != "AK4PUPPI":
        raise ValueError(f"Unsupported jet type: {jet_type}")
    if not is_run2(year):
        raise ValueError(f"Custom jet ID is only defined for Run 2 years. Unsupported year: {year}")

    eta = jets.eta
    aeta = abs(eta)

    neHEF = jets.neHEF
    neEmEF = jets.neEmEF
    chHEF = jets.chHEF
    chEmEF = jets.chEmEF
    muEF = jets.muEF

    chMult = jets.chMultiplicity
    neMult = jets.neMultiplicity
    nMult = chMult + neMult

    # -------------------------
    # 2016 (pre/post VFP): barrel edge is 2.4
    # -------------------------
    if "2016" in year:
        barrel_tight = (
            (aeta <= 2.4)
            & (neHEF < 0.90)
            & (neEmEF < 0.90)
            & (nMult > 1)
            & (chHEF > 0.0)
            & (chMult > 0)
        )
        transition_tight = (
            (aeta > 2.4) & (aeta <= 2.7) & (neHEF < 0.98) & (neEmEF < 0.99)
        )
        endcap_tight = (aeta > 2.7) & (aeta <= 3.0) & (neMult >= 1)
        forward_tight = (aeta > 3.0) & (aeta <= 5.0) & (neEmEF < 0.9) & (neMult > 2)

        pass_tight = barrel_tight | transition_tight | endcap_tight | forward_tight

        # LepVeto is defined only for |eta|<=2.4 in the 2016 table
        pass_tight_lepveto = ak.where(
            aeta <= 2.4,
            pass_tight & (muEF < 0.8) & (chEmEF < 0.8),
            pass_tight,
        )

    # -------------------------
    # 2017 & 2018: barrel edge is 2.6
    # -------------------------
    elif ("2017" in year) or ("2018" in year):
        barrel_tight = (
            (aeta <= 2.6)
            & (neHEF < 0.90)
            & (neEmEF < 0.90)
            & (nMult > 1)
            & (chHEF > 0.0)
            & (chMult > 0)
        )
        transition_tight = (
            (aeta > 2.6) & (aeta <= 2.7) & (neHEF < 0.90) & (neEmEF < 0.99)
        )
        endcap_tight = (aeta > 2.7) & (aeta <= 3.0) & (neHEF < 0.9999)
        forward_tight = (aeta > 3.0) & (aeta <= 5.0) & (neEmEF < 0.9) & (neMult > 2)

        pass_tight = barrel_tight | transition_tight | endcap_tight | forward_tight

        # LepVeto applies up to |eta|<=2.7 in 2017/2018 table
        pass_tight_lepveto = ak.where(
            aeta <= 2.7,
            pass_tight & (muEF < 0.8) & (chEmEF < 0.8),
            pass_tight,
        )
    else:
        raise ValueError(f"Unsupported year: {year}")

    return pass_tight, pass_tight_lepveto


def jet_id(jets, config, year=None, jet_id_key="jet_id"):
    """https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD#NanoAOD_format , jet Id is same for UL 2016,2017 and 2018

    If "jetId" is in the fields of jets, use that. Else, use custom_jet_id function as mentioned in the link:
    https://twiki.cern.ch/twiki/bin/view/CMS/JetID13p6TeV#nanoAOD_Flags
    """
    if year is None:
        raise ValueError("Year must be specified for jet ID determination.")

    pass_jet_id = ak.ones_like(jets.pt, dtype=bool)
    jet_id2use = config.get(jet_id_key, config["jet_id"])
    if hasattr(jets, "jetId") and is_run2(year):
        logger.info("Using Run2 official jet-id for the custom nanoAODv12")
        jet_id_wps = {
            "tight": jets.jetId >= 2,
            "tightFailLepVeto": jets.jetId == 2,
            "tightPassLepVeto": jets.jetId == 6,
        }
        pass_jet_id = jet_id_wps[jet_id2use]
    elif hasattr(jets, "jetId") and is_run3(year):
        # Reference: https://twiki.cern.ch/twiki/bin/view/CMS/JetID13p6TeV#nanoAOD_Flags
        logger.info("Using Run3 official jet-id for the nanoAODv12")

        abs_eta = abs(jets.eta)

        # Tight jet ID
        passJetIdTight = ak.zeros_like(jets.pt, dtype=bool)

        mask_barrel_endcap = abs_eta <= 2.7
        mask_he = (abs_eta > 2.7) & (abs_eta <= 3.0)
        mask_hf = abs_eta > 3.0

        pass_tight_bit = (jets.jetId & (1 << 1)) != 0

        passJetIdTight = (
            (mask_barrel_endcap & pass_tight_bit)
            | (mask_he & pass_tight_bit & (jets.neHEF < 0.99))
            | (mask_hf & pass_tight_bit & (jets.neEmEF < 0.4))
        )

        # TightLepVeto jet ID
        passJetIdTightLepVeto = ak.where(
            abs_eta <= 2.7,
            passJetIdTight & (jets.muEF < 0.8) & (jets.chEmEF < 0.8),
            passJetIdTight,
        )
        jet_id_wps = {
            "tight": passJetIdTight,
            "tightFailLepVeto": passJetIdTight & ~passJetIdTightLepVeto,
            "tightPassLepVeto": passJetIdTight & passJetIdTightLepVeto,
        }
        pass_jet_id = jet_id_wps[jet_id2use]
    elif is_run2(year):
        """For Run 2, use the custom jet ID based on the official tight WP definition."""
        logger.info("Using custom jet ID for Run 2!")
        pass_jetid_tight, pass_jetid_tight_lepveto = custom_jet_id(
            jets, year, jet_type="AK4PUPPI"
        )
        jet_id_wps = {
            "tight": pass_jetid_tight,
            "tightFailLepVeto": pass_jetid_tight & ~pass_jetid_tight_lepveto,
            "tightPassLepVeto": pass_jetid_tight & pass_jetid_tight_lepveto,
        }
        pass_jet_id = jet_id_wps[jet_id2use]
    elif is_run3(year):
        """For Run 3, use the correctionlib based jetID."""
        logger.info("Using correctionlib-based jet ID for Run 3!")
        jet_id_json_files = config["jet_id_json_files"]
        cset = get_corrset(jet_id_json_files)
        eval_dict = {
            "eta": jets.eta,
            "chHEF": jets.chHEF,
            "neHEF": jets.neHEF,
            "chEmEF": jets.chEmEF,
            "neEmEF": jets.neEmEF,
            "muEF": jets.muEF,
            "chMultiplicity": jets.chMultiplicity,
            "neMultiplicity": jets.neMultiplicity,
            "multiplicity": jets.chMultiplicity + jets.neMultiplicity
        }

        # Default tight for NanoAOD version 12 and above
        idTight = cset["AK4PUPPI_Tight"]
        inputsTight = [eval_dict[input.name] for input in idTight.inputs]
        idTight_value = idTight.evaluate(*inputsTight)

        # Default tight lepton veto
        idTightLepVeto = cset["AK4PUPPI_TightLeptonVeto"]
        inputsTightLepVeto = [eval_dict[input.name] for input in idTightLepVeto.inputs]
        idTightLepVeto_value = idTightLepVeto.evaluate(*inputsTightLepVeto)

        jet_id_wps = {
            "tight": idTight_value == 1,
            "tightFailLepVeto": (idTight_value == 1) & (idTightLepVeto_value == 0),
            "tightPassLepVeto": (idTight_value == 1) & (idTightLepVeto_value == 1),
        }

        pass_jet_id = jet_id_wps[jet_id2use]

    else:
        raise ValueError("Jet collection has no 'jetId' branch and is not Run 3 for correctionlib-based jet ID. Cannot determine jet ID.")

    return pass_jet_id


def btag_jet_selection(jets, config, year, clean=None):
    """Select jets passing b-tagging requirements with pt, eta, ID, and cleaning cuts.

    Parameters:
    ----------
    jets : awkward.Array
        Input jet collection with candidate behavior
    config : dict
        Configuration dictionary containing the selection criteria
    year : str
        Data taking year (determines eta cuts and jet ID criteria from config file)
    clean : awkward.Array of bool, optional
        Boolean mask to clean jets from selected muons

    Returns:
    -------
    awkward.Array of bool
        Boolean mask where True indicates jets passing all selection criteria

    Notes:
    -----
    - Run 3 uses |eta| < 2.5 for b-tagging
    - Run 2 uses |eta| < 2.4 for 2016 and |eta| < 2.5 for 2017/2018
    - The cleaning mask is applied last in the selection chain
    """
    if year is None:
        raise ValueError("Year must be specified for b-tag jet selection.")
    btag_pt_cut = jets.pt > config["btag_jet_pt_cut"]
    btag_id_cut = jet_id(jets, config, year=year, jet_id_key="btag_jet_id")
    if clean is None:
        clean = ak.ones_like(jets.pt, dtype=bool)    
    if is_run3(year):
        btag_eta_cut = abs(jets.eta) < 2.5
    elif is_run2(year):
        btag_eta_cut = abs(jets.eta) < (2.4 if str(year).startswith("2016") else 2.5)
    else:
        raise ValueError(f"The year: {year} is not supported. Please use the appropriated year from Run2 or Run3")

    return btag_pt_cut & btag_eta_cut & btag_id_cut & clean


def get_puId(jets):
    """
    Return the correct PUID field from a Jet or FatJet collection.
    Priority:
      1) puId17   (NanoAODv15 Run3)
      2) puId     (Run2/NanoAODv12)
      3) puIdDisc (fallback)
    """
    if hasattr(jets, "puId17"):
        logger.info("Using puId17 for PUID")
        return jets.puId17

    if hasattr(jets, "puId"):
        logger.info("Using puId for PUID")
        return jets.puId

    if hasattr(jets, "puIdDisc"):
        logger.info("Using puIdDisc for PUID")
        return jets.puIdDisc

    logger.warning(
        "Jet collection has no PUID branch (puId17/puId/puIdDisc). Falling back to tight-pass dummy puId=7."
    )
    return ak.full_like(jets.pt, 7, dtype=np.int8)

def jet_puid(jets, config):
    jet_puid2use = config["jet_puid"]
    year = config["year"]
    puId = get_puId(jets)
    # jet puid for standard wps are different for 2016 vs 2017,2018 as shown in https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetIDUL#Working_Points
    # only apply jet puid to jets with pt < 50, else, pass
    # as stated in https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetIDUL
    if ("2016" in year) and ("RERECO" not in year): #Only 2016 UL samples are different
        logger.debug("2016 UL exception!")
        jet_puid_wps = {
            "loose": (puId >= 1) | (jets.pt >= 50),
            "medium": (puId >= 3) | (jets.pt >= 50),
            "tight": (puId >= 7) | (jets.pt >= 50),
        }
    else: # 2017 and 2018
        jet_puid_wps = {
            "loose": (puId >= 4) | (jets.pt >= 50),
            "medium": (puId >= 6) | (jets.pt >= 50),
            "tight": (puId >= 7) | (jets.pt >= 50),
        }
    pass_jet_puid = ak.ones_like(jets.pt, dtype=bool)

    if "2017" in year: # for misreco due ot ECAL endcap noise
        # NOTE: PUID for 2017 is tight in horn region but not for other eras in Run2
        eta_window = (abs(jets.eta) > 2.6) & (abs(jets.eta) < 3.0)

        # tight puid in the noisy eta window, else loose
        pass_jet_puid = (eta_window & (puId >= 7)) | (
                (~eta_window) & jet_puid_wps["loose"]
        )
    else:
        pass_jet_puid = jet_puid_wps[jet_puid2use]

    return pass_jet_puid


def fill_softjets(events, jets, mu1, mu2, nmuons, cutoff, test_mode=False):
    if test_mode:
        logger.debug(f"jets events.SoftActivityJet.fields: {events.SoftActivityJet.fields}")
        logger.debug(f"jets cutoff: {cutoff}")
    saj = events.SoftActivityJet

    padded_jets = ak.pad_none(jets, 2)
    jet1 = padded_jets[:,0]
    jet2 = padded_jets[:,1]

    # line 2966 of AN-19-124: "The two identified muons and the charged PF candidates associated to the two leading jets in the event are not included in the soft-jet definition"
    dR_m1 = saj.delta_r(mu1)
    dR_m2 = saj.delta_r(mu2)
    dR_j1 = saj.delta_r(jet1)
    dR_j2 = saj.delta_r(jet2)
    dR_m1_filter = ak.fill_none((dR_m1 > 0.4), value=True, axis=None)
    dR_m2_filter = ak.fill_none((dR_m2 > 0.4), value=True, axis=None)
    dR_j1_filter = ak.fill_none((dR_j1 > 0.4), value=True, axis=None)
    dR_j2_filter = ak.fill_none((dR_j2 > 0.4), value=True, axis=None)

    good_saj = dR_m1_filter & dR_m2_filter & dR_j1_filter & dR_j2_filter
    softjet_cleaning = (good_saj) & (saj.pt > cutoff)
    saj_of_interest = saj[softjet_cleaning]

    ht_corrected = ak.sum(saj_of_interest.pt, axis=1)
    corrected_njets = ak.num(saj_of_interest, axis=1)

    out_dict = {
        f"nsoftjets{cutoff}_new": corrected_njets,
        f"htsoft{cutoff}_new": ht_corrected
    }
    return out_dict


def fill_softjets_HIG19006(events, jets, mu1, mu2, nmuons, cutoff, test_mode=False):
    if test_mode:
        logger.debug(
            f"jets events.SoftActivityJet.fields: {events.SoftActivityJet.fields}"
        )
        logger.debug(f"jets cutoff: {cutoff}")
    events["SoftActivityJet", "mass"] = 0
    saj = events.SoftActivityJet
    saj_Njets = events[f"SoftActivityJetNjets{cutoff}"]
    saj_HT = events[f"SoftActivityJetHT{cutoff}"]

    njets = ak.num(jets, axis=1)
    padded_jets = ak.pad_none(jets, 2)
    jet1 = padded_jets[:, 0]
    jet2 = padded_jets[:, 1]

    # nmuons = ak.num(muons, axis=1)
    # mu1 = muons[:,0]
    # mu2 = muons[:,1]
    if test_mode:
        logger.debug(f"jets njets: {njets}")
        logger.debug(f"jets saj.pt: {saj.pt}")
        logger.debug(f"jets jet1.pt: {jet1.pt}")
        logger.debug(f"jets jet2.pt: {jet2.pt}")
        logger.debug(f"jets mu1.pt: {mu1.pt}")
        logger.debug(f"jets mu2.pt: {mu2.pt}")

    # line 2966 of AN-19-124: "The two identified muons and the charged PF candidates associated to the two leading jets in the event are not included in the soft-jet definition"
    dR_m1 = saj.delta_r(mu1)
    dR_m2 = saj.delta_r(mu2)
    dR_j1 = saj.delta_r(jet1)
    dR_j2 = saj.delta_r(jet2)
    dR_m1_filter = ak.fill_none((dR_m1 < 0.4), value=False, axis=None)
    dR_m2_filter = ak.fill_none((dR_m2 < 0.4), value=False, axis=None)
    dR_j1_filter = ak.fill_none((dR_j1 < 0.4), value=False, axis=None)
    dR_j2_filter = ak.fill_none((dR_j2 < 0.4), value=False, axis=None)
    if test_mode:
        logger.debug(f"jets dR_m1_filter: {dR_m1_filter}")
        logger.debug(f"jets dR_m2_filter: {dR_m2_filter}")
        logger.debug(f"jets dR_j1_filter: {dR_j1_filter}")
        logger.debug(f"jets dR_j2_filter: {dR_j2_filter}")
    saj_to_remove = dR_m1_filter | dR_m2_filter | dR_j1_filter | dR_j2_filter
    saj_to_remove = ak.fill_none(saj_to_remove, value=False)

    footprint = saj[(saj_to_remove) & (saj.pt > cutoff)]
    footprint_sumPt = ak.sum(footprint.pt, axis=1)
    if test_mode:
        logger.debug(f"jets saj_to_remove: {saj_to_remove}")
        logger.debug(f"jets footprint_sumPt: {ak.to_numpy(footprint_sumPt)}")
    ht_corrected = saj_HT - footprint_sumPt
    footprint_njets = ak.num(footprint, axis=1)
    corrected_njets = saj_Njets - footprint_njets

    if test_mode:
        logger.debug(f"jets footprint_njets: {ak.to_numpy(footprint_njets)}")
        logger.debug(f"jets corrected_njets: {ak.to_numpy(corrected_njets)}")
        logger.debug(f"jets saj_Njets: {saj_Njets}")

    evnts_to_correct = (nmuons == 2) | (njets > 0)
    if test_mode:
        logger.debug(f"jets evnts_to_correct: {evnts_to_correct}")
        logger.debug(f"jets footprint_njets b4: {ak.to_numpy(saj_Njets)}")
        logger.debug(f"jets corrected_njets b4: {ak.to_numpy(saj_HT)}")

    saj_Njets = ak.where(evnts_to_correct, corrected_njets, saj_Njets)
    saj_HT = ak.where(evnts_to_correct, ht_corrected, saj_HT)

    if test_mode:
        logger.debug(f"jets footprint_njets after: {ak.to_numpy(saj_Njets)}")
        logger.debug(f"jets corrected_njets after: {ak.to_numpy(saj_HT)}")
    out_dict = {f"nsoftjets{cutoff}": saj_Njets, f"htsoft{cutoff}": saj_HT}
    return out_dict


def getHemVetoRunFilter(run, event_num, config, is_mc: bool, NanoAODv: int):
    """
    For data:
    return the conditions for applying HemVeto. For data, this is just
    end of data B run + full data C,D (run >= 319077).
    For MC:
    Randomly reject a given fraction of events using for MC to match HEM Vetoed jets in 2018 UL as reccommended in https://cms-talk.web.cern.ch/t/question-about-hem15-16-issue-in-2018-ultra-legacy/38654/8 (though we reject her "eventNum % 15 == 0" method of random rejection and just use random number generation)
    """
    hemvetoRatioConfig = config["HemVeto_ratio"]
    if is_mc:
        try: # see if the nanoAODV distinction exists
            logger.info(f"nanoAODv{NanoAODv}")
            prob = config["HemVeto_ratio"][f"nanoAODv{NanoAODv}"] # ratio of HemVeto applicable run / total nevents for 2018UL
        except:
            prob = config["HemVeto_ratio"]

        logger.info(f"HEMveto prob: {prob}")
        # intialize random number generator
        resrng = cs.Correction(
            name="resrng",
            description="Deterministic smearing value generator",
            version=1,
            inputs=[
                cs.Variable(name="event", type="real", description="Event number"),
            ],
            output=cs.Variable(name="rng", type="real"),
            data=cs.HashPRNG(
                    nodetype="hashprng",
                    inputs=["event"],
                    distribution="stdflat",
            )
        )
        # get random number from 0 to 1
        rand = resrng.to_evaluator().evaluate(event_num)
        # logger.debug(f"rand: {rand[:20].compute()}")
        # logger.debug(f"(rand < prob): {(rand < prob)[:20].compute()}")
        # raise ValueError
        return (rand < prob) # for prob amount of times, this is true
    else: #For data, just a simple run >= 319077 cut. Source: https://cms-talk.web.cern.ch/t/question-about-hem15-16-issue-in-2018-ultra-legacy/38654/8
        return (run >= 319077)

def applyHemVeto(jets, run, event_num, config, is_mc: bool, NanoAODv: int):
    """
    Apply HEM veto for 2018 UL as recommended on https://cms-talk.web.cern.ch/t/question-about-hem15-16-issue-in-2018-ultra-legacy/38654/5
    """
    if hasattr(jets, "jetId") and is_run2(config["year"]):
        jetId_bits = jets.jetId
    else:
        # synthesize bit-coded jetId from custom_jet_id
        # FIXME: Do we really this this? As HEM veto is there only for 2018 UL
        #               Even if we need it, we should fetch it from the function jet_id.
        tight, tightLepVeto = custom_jet_id(jets, config["year"], jet_type="AK4PUPPI")
        jetId_bits = ak.zeros_like(jets.pt, dtype=np.int8)
        jetId_bits = ak.where(tight, jetId_bits | 2, jetId_bits)
        jetId_bits = ak.where(tightLepVeto, jetId_bits | 4, jetId_bits)

    puId = get_puId(jets)
    # jet puid selection
    jet_puid_wps = {
            "loose": (puId >= 4) | (jets.pt >= 50),
            "medium": (puId >= 6) | (jets.pt >= 50),
            "tight": (puId >= 7) | (jets.pt >= 50),
    }
    jet_puid2use = config["jet_puid"]
    pass_jet_puid = jet_puid_wps[jet_puid2use]# the recommendation doesn't specify, so use PU Id that we apply



    # jets that don’t overlap with PF muon (dR < 0.2)
    jet_muon_iso_cut = (jets.muonIdx1 == -1) & (jets.muonIdx2 == -1) # Source: https://cms-talk.web.cern.ch/t/jetvetomaps-usage-for-2018ul/61981/2
    jet_em_frac_cut  = (jets.chEmEF + jets.neEmEF) < 0.9 # EM fraction cut
    pass_jet_tightID = (jetId_bits >= 2) & jet_em_frac_cut & jet_muon_iso_cut

    pass_jet_tightLepVetoID = jetId_bits ==6 # Source: https://twiki.cern.ch/twiki/bin/view/CMS/JetID13TeVUL

    pass_jet_id_total = pass_jet_tightLepVetoID | pass_jet_tightID # Source: https://cms-talk.web.cern.ch/t/question-about-hem15-16-issue-in-2018-ultra-legacy/38654/2



    loose_jet_selection =( # Source: https://cms-talk.web.cern.ch/t/question-about-hem15-16-issue-in-2018-ultra-legacy/38654/2
        (jets.pt > 15)
        & pass_jet_id_total
        & pass_jet_puid
    )
    hemveto_region = ( # "in jets with -3.2<eta<-1.3 and -1.57<phi< -0.87 " Source: https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetMET#Run2_recommendations
        (jets.eta > -3.2)
        & (jets.eta < -1.3)
        & (jets.phi > -1.57)
        & (jets.phi < -0.87)
    )

    # hemveto_run_filter = (run >= 319077)
    hemveto_run_filter = getHemVetoRunFilter(run, event_num, config, is_mc, NanoAODv)

    # combine all the conditions
    hemveto = loose_jet_selection & hemveto_region
    hemveto = ak.any(hemveto, axis=1) & hemveto_run_filter
    # we reject events if we find hemveto jets, so reverse the bool arr
    hemveto = ~hemveto
    is_HemRegion = ak.any(hemveto_region, axis=1) # eventwise arr if any jet is in the hem region
    return hemveto, is_HemRegion


def getJecDataTag(run, jec_data_tags, NanoAODv=None):
    logger.debug(f"run: {run}")
    logger.debug(f"jec_data_tags: {jec_data_tags}")
    if (not isinstance(jec_data_tags, str)) and (NanoAODv is not None): # if it's a dictionary like
        nano_key = f"nanoAODv{NanoAODv}"
        # Run2 years nest jec_data_tags by NanoAOD version; Run3 years (only one
        # NanoAOD version supported so far) are a flat {tag: [runs]} dict. Only
        # descend into the nano_key sub-dict if it's actually nested that way.
        if nano_key in jec_data_tags:
            jec_data_tags = jec_data_tags[nano_key]
    logger.debug(f"jec_data_tags: {jec_data_tags}")
    for jec_tag, jec_run_l in jec_data_tags.items():
        for jec_run in jec_run_l:
            if run == jec_run:
                logger.debug(f"found match in jec_run {jec_run}!")
                return jec_tag

    return None # return none if nothing matches

def applyUpDown(variation_base_l: list):
    """
    helper function that adds _up and _down to the variations
    """
    variation_up_l = [f"{variation}_up" for variation in variation_base_l]
    variation_down_l = [f"{variation}_down" for variation in variation_base_l]
    combined_variation_l = variation_up_l + variation_down_l
    # logger.debug(f"combined_variation_l: {combined_variation_l}")
    return combined_variation_l


def extract_values(cfg):
    """
    Recursively extract all values from an OmegaConf object into a flat list.
    """
    values = []
    for v in cfg.values():
        if isinstance(v, dict) or isinstance(v, OmegaConf):
            values.extend(extract_values(v))
        else:
            values.extend(v)

    values = list(set(values)) # make a list of unique vals
    return values
    
def getJecJerUncertainties(yaml_filename, year=None, jer_unc = False):
    """
    helper function to load the jec uncertainties
    """
    # Load YAML file
    # cfg = OmegaConf.load("/work/users/yun79/Run3/copperheadV2/configs/parameters/jec.yaml")
    cfg = OmegaConf.load(yaml_filename)
    cfg = cfg["jec_parameters"]["jec_unc_to_consider"]
    # Get list of all values
    if year is None: # extract all years
        jec_uncs = extract_values(cfg)
    else:
        jec_uncs = cfg[year]
    if jer_unc:
        jer_uncs = [f"jer{i}" for i in range(1,7)]  
    else:
        jer_uncs = []
    return jec_uncs + jer_uncs

def get_baseVariations(variation_shifts : list):
    """
    helper function that removes _up and _down and removed redundant lists
    """
    variation_base_l = [variation.replace("_up","").replace("_down","") for variation in variation_shifts]
    variation_base_l = list(set(variation_base_l)) # remove repetitions
    # logger.debug(f"variation_base_l: {variation_base_l}")
    return variation_base_l

def get_jec_sources(cset, jec_tag, jet_type="AK4PFPuppi"):
    sources = []

    prefix = f"{jec_tag}_Regrouped_"
    suffix = f"_{jet_type}"

    for key in cset.keys():
        logger.debug(f"JEC keys: {key}")
        if key.startswith(prefix) and key.endswith(suffix):
            source = key[len(prefix):-len(suffix)]
            sources.append(source)

    return sorted(sources)

def do_jec_scale(jets, events, config, is_mc, dataset, uncs=["nominal"]):
    jec_parameters = config["jec_parameters"]

    jerc_load_path = jec_parameters["jerc_load_path"]
    logger.debug(f"jerc_load_path: {jerc_load_path}")

    cset = get_corrset(jerc_load_path, config['NanoAODv'])


    if is_mc:
        jec_tag = jec_parameters["jec_tags"]
        logger.debug(f"jec_parameters: {jec_parameters}")
        logger.debug(f"jec_tag: {jec_tag}")
        logger.debug(f"NanoAODv: {config['NanoAODv']}")
        if (not isinstance(jec_tag, str)): # if it's a dictionary like
            NanoAODv = config["NanoAODv"]
            jec_tag = jec_tag[f"nanoAODv{NanoAODv}"]
    else: # data
        jec_tag = None
        for run in jec_parameters["runs"]:
            logger.debug(f"run: {run}, dataset: {dataset}")
            if run in dataset:
                jec_tag = getJecDataTag(run, jec_parameters["jec_data_tags"], NanoAODv=config["NanoAODv"])
    logger.debug(f"jec_tag: {jec_tag}")
    if jec_tag is None:
        raise ValueError("JEC tag not found!")


    algo = jec_parameters["jet_algorithm"]

    if (not is_mc) and ("run" not in jets.fields):
        jets["run"] = ak.ones_like(jets.pt_raw, dtype=np.int64) * events.run

    input_map = {
        "JetA": jets.area,
        "JetEta": jets.eta,
        "JetPt": jets.pt_raw,
        "Rho": jets.PU_rho,
        "JetPhi": jets.phi,
        "run": jets.run if "run" in jets.fields else None,
    }

    for unc in uncs: # NOTE: we assume that "nominal" is the first element list
        if (not is_mc) and "nominal" not in unc:
            continue
        if unc == "nominal":
            lvl_compound = "L1L2L3Res"
        else:
            lvl_compound = f"Regrouped_{unc}"

        logger.info(f"[MC: {is_mc}]: Applying JEC: {unc} with level: {lvl_compound}")

        key = f"{jec_tag}_{lvl_compound}_{algo}"
        logger.debug(f"jec key: {key}")
        if unc == "nominal":
            sf = cset.compound[key]
        else:
            sf = cset[key]

        inputs = []
        for inp in sf.inputs:
            if inp.name not in input_map:
                raise ValueError(f"JEC input {inp.name} not found in input_map!")
            arr = input_map.get(inp.name, None)
            if arr is None:
                raise ValueError(f"Missing required JEC input: {inp.name} for key: {key}")
            inputs.append(arr)
            logger.debug(f"JEC input {inp.name}")
            # logger.debug(f"JEC input {inp.name}: {arr[:2].compute()}")
        logger.debug(f"{unc} JEC input: {inputs}") # use this a reference to add inputs

        # inputs = get_corr_inputs(example_value_dict, sf)
        printCorrObjInputs(sf) # for debugging
        new_jec_scale = sf.evaluate(*inputs)
        # logger.debug(f"new_jec_scale: {new_jec_scale}")
        # logger.debug(f"new_jec_scale {unc}: {new_jec_scale.compute()}")

        # logger.debug("JSON result AK4: {}".format(new_jec_scale[:20].compute()))

        if unc == "nominal":
            jet_pt_jec = new_jec_scale*jets.pt_raw
            jet_mass_jec = new_jec_scale*jets.mass_raw
            jets["pt"] = jet_pt_jec
            jets["mass"] = jet_mass_jec
            jets["pt_jec"] = jet_pt_jec
            jets["mass_jec"] = jet_mass_jec
        else:
            # up
            jet_pt_jec = (1+new_jec_scale) # apply these corrections fully after JER
            jet_mass_jec = (1+new_jec_scale) # apply these corrections fully after JER
            jets[f"pt_{unc}_up"] = jet_pt_jec
            jets[f"mass_{unc}_up"] = jet_mass_jec

            # down
            jet_pt_jec = (1-new_jec_scale) # apply these corrections fully after JER
            jet_mass_jec = (1-new_jec_scale) # apply these corrections fully after JER
            jets[f"pt_{unc}_down"] = jet_pt_jec
            jets[f"mass_{unc}_down"] = jet_mass_jec
    return jets


def applyJetUncertaintyKinematics(jets, uncs):
    """
    we assume do_jec_scale function with the uncertainties have already been applied to jets nanoEvent
    """
    # grab the latest correct mass and pt
    jet_pt = jets["pt"]
    jet_mass = jets["mass"]
    # apply the jec uncertainty coeffs that you obtained previously to the latest corrected mass and pt
    for unc in uncs:
        # up
        jets[f"pt_{unc}_up"] = jets[f"pt_{unc}_up"] * jet_pt
        jets[f"mass_{unc}_up"] = jets[f"mass_{unc}_up"] * jet_mass
        # down
        jets[f"pt_{unc}_down"] = jets[f"pt_{unc}_down"] * jet_pt
        jets[f"mass_{unc}_down"] = jets[f"mass_{unc}_down"] * jet_mass
    return jets

def applyStrat1(apply_scaling, jer_smearing, jet_puId, jet_pt, jet_eta):
    is_tightPuId = (jet_puId >= 7)
    keep_jerSmear = (is_tightPuId & (jet_pt <= 50)) | (jet_pt > 50)
    keep_jerSmear = keep_jerSmear | apply_scaling # if scaling, don't change anything
    no_smearing = ak.ones_like(jer_smearing)
    return ak.where(keep_jerSmear, jer_smearing, no_smearing)


def applyStrat2(apply_scaling, jer_smearing, jet_puId, jet_pt, jet_eta):
    remove_jerSmear = (abs(jet_eta) > 2.5) & (jet_pt <= 50)
    keep_jerSmear = (~remove_jerSmear) | apply_scaling # if scaling, don't change anything
    no_smearing = ak.ones_like(jer_smearing)
    return ak.where(keep_jerSmear, jer_smearing, no_smearing)

def applyStrat1n2(apply_scaling, jer_smearing, jet_puId, jet_pt, jet_eta):
    jer_smearing1 = applyStrat1(apply_scaling, jer_smearing, jet_puId, jet_pt, jet_eta)
    jer_smearing2 = applyStrat2(apply_scaling, jer_smearing, jet_puId, jet_pt, jet_eta)
    apply_stat2 = abs(jet_eta) < 3
    return ak.where(apply_stat2, jer_smearing2, jer_smearing1)

def applyStrat1n2Revised(apply_scaling, jer_smearing, jet_puId, jet_pt, jet_eta, year:str):
    jer_smearing1 = applyStrat1(apply_scaling, jer_smearing, jet_puId, jet_pt, jet_eta)
    jer_smearing2 = applyStrat2(apply_scaling, jer_smearing, jet_puId, jet_pt, jet_eta)
    if ("2018" in year) or ("2017" in year):
    # if ("2018" in year):
        apply_stat2 = abs(jet_eta) < 3.0
    else:
        apply_stat2 = abs(jet_eta) < 2.5
    return ak.where(apply_stat2, jer_smearing2, jer_smearing1)


def applyStrat4(apply_scaling, jer_smearing):
    """Official JME ad hoc mitigation ("option b") for the poor Run3
    2.5 < |eta| < 3.0 data/MC agreement, per
    https://cms-jme-jerc.docs.cern.ch/recommendations/jer/ : the jet is
    corrected only when a matched gen-jet is found (scaling method); if no
    match is available, no stochastic smearing is applied. Unlike
    applyStrat1/applyStrat2, this drops the stochastic component
    UNCONDITIONALLY for every unmatched jet -- no PUID/pT/eta gating. The
    doc names this exact approach for H->WW, H->mumu, and VBF SUSY.
    """
    no_smearing = ak.ones_like(jer_smearing)
    return ak.where(apply_scaling, jer_smearing, no_smearing)


def apply_jer_unc(jets):
    """
    we assume do_jer_smear has been applied
    Taken from Dmitry's commented out code, with eta bins updates from https://cms-jerc.web.cern.ch/Recommendations/#run-2_1
    source:  https://github.com/green-cabbage/copperhead_fork2/blob/97a0fcd7668927b46931e6334de4bbf25d3d2031/stage1/corrections/jec.py#L212C14-L242C69
    """
    has_matchedGenJet = jets.genJetIdx != -1
    # logger.debug(f"has_matchedGenJet: {has_matchedGenJet.compute()}")
    # logger.debug(f"jets.genJetIdx: {jets.genJetIdx[:100].compute()}")
    jer_categories = {
       'jer1': (abs(jets.eta) < 1.93),
       'jer2': (abs(jets.eta) > 1.93) & (abs(jets.eta) < 2.5),
       'jer3': ((abs(jets.eta) > 2.5) &
                (abs(jets.eta) < 3.0) &
                (jets.pt < 50)),
       'jer4': ((abs(jets.eta) > 2.5) &
                (abs(jets.eta) < 3.0) &
                (jets.pt > 50)),
       'jer5': (abs(jets.eta) > 3.0) & (abs(jets.eta) < 5.0) & (jets.pt < 50),
       'jer6': (abs(jets.eta) > 3.0) & (abs(jets.eta) < 5.0) & (jets.pt > 50),
    }
    for jer_unc_name, jer_cut in jer_categories.items():
        jer_cut = jer_cut & (has_matchedGenJet)
        pt_name_up = f"pt_{jer_unc_name}_up"
        pt_name_down = f"pt_{jer_unc_name}_down"
        jer_pt_nom = jets["pt_jer_nom"] # NOTE: if I name this "pt_jer_nominal", sorting jets by jet.pt breaks
        jer_pt_up = ak.where(jer_cut, jets["pt_jer_up"], jer_pt_nom)
        jer_pt_down = ak.where(jer_cut, jets["pt_jer_down"], jer_pt_nom)
        jets[pt_name_up] = jer_pt_up
        jets[pt_name_down] = jer_pt_down

    return jets


def do_jer_smear(jets, config, event_id, syst_l=["nom", "up", "down"]):
    """
    we assume that jec has been applied (we need pt_jec and pt_raw)

    params:
    syst: nom, up and down
    """
    year = config["year"]

    jec_parameters = config["jec_parameters"]
    jerc_load_path = jec_parameters["jerc_load_path"]
    logger.debug(f"jerc_load_path: {jerc_load_path}")

    cset = get_corrset(jerc_load_path, NanoAODv=config['NanoAODv'])

    jersmear_load_path = jec_parameters["jersmear_load_path"]
    cset_jersmear = get_corrset(jersmear_load_path)
    logger.debug(f"jerc_load_path: {jerc_load_path}")
    logger.debug(f"jersmear_load_path: {jersmear_load_path}")

    # jer_tag = "Summer20UL16_JRV3_MC"
    jer_tag = jec_parameters["jer_tags"]
    # algo = "AK4PFchs"
    algo = jec_parameters["jet_algorithm"]

    #  JER scale factor key
    key = "{}_{}_{}".format(jer_tag, "ScaleFactor", algo)
    logger.info(f"key: {key}")
    sf = cset[key]
    logger.debug(f"JER SF name: {sf}")
    logger.debug(f"JER SF name: {sf.name}")
    logger.debug(f"JER SF inputs: {sf.inputs}")
    for inp in sf.inputs:
        logger.debug(f"JER SF input name: {inp.name}, type: {inp.type}, description: {inp.description}")
    jer_sf_input_names = [inp.name for inp in sf.inputs]
    logger.debug(f"JER SF input: {jer_sf_input_names}")

    # ScaleFactor payloads come in three schemas:
    #   (JetEta, systematic)          -- Run 2 UL
    #   (JetEta, JetPt, systematic)   -- Run 3 JRV1 (2022preEE/postEE, 2023, 2023BPix)
    #   (JetEta, JetPt)               -- Summer24 (2024, 2025, 2026): no "systematic"
    #                                    axis; the uncertainty is a separate correction
    #                                    "<jer_tag>_SFUncertainty_<algo>" holding a
    #                                    symmetric absolute delta, so
    #                                    SF_up/down = SF +/- SFUncertainty.
    sf_has_syst_axis = "systematic" in jer_sf_input_names
    if sf_has_syst_axis:
        sf_unc = None
        jer_sf_unc_input_names = []
    else:
        unc_key = "{}_{}_{}".format(jer_tag, "SFUncertainty", algo)
        logger.info(
            f"JER ScaleFactor has no 'systematic' axis; taking up/down from key: {unc_key}"
        )
        sf_unc = cset[unc_key]
        jer_sf_unc_input_names = [inp.name for inp in sf_unc.inputs]
        logger.debug(f"JER SFUncertainty input: {jer_sf_unc_input_names}")

    #  JER pT resolution key
    key = "{}_{}_{}".format(jer_tag, "PtResolution", algo)
    sf_ptres = cset[key]

    sf_input_names = [inp.name for inp in sf_ptres.inputs]
    logger.debug(f"JER resolution input: {sf_input_names}")

    for syst in syst_l:
        # Build ScaleFactor inputs from the payload's declared axes (see schema note
        # above), instead of hardcoding per run era.
        sf_arg_by_name = {
            "JetEta": jets.eta,
            "JetPt": jets.pt_raw,
            "systematic": syst,
        }
        printCorrObjInputs(sf)
        if sf_has_syst_axis:
            jer_sf = sf.evaluate(*(sf_arg_by_name[n] for n in jer_sf_input_names))
        else:
            jer_sf_nom = sf.evaluate(*(sf_arg_by_name[n] for n in jer_sf_input_names))
            if syst == "nom":
                jer_sf = jer_sf_nom
            elif syst in ("up", "down"):
                sf_unc_val = sf_unc.evaluate(
                    *(sf_arg_by_name[n] for n in jer_sf_unc_input_names)
                )
                jer_sf = jer_sf_nom + sf_unc_val if syst == "up" else jer_sf_nom - sf_unc_val
            else:
                raise ValueError(
                    f"Unsupported JER systematic '{syst}' for split SFUncertainty payload"
                )
        # logger.debug("JER SF : {}".format(jer_sf.compute()))

        inputs = ( # Source: https://github.com/cms-jet/JECDatabase/blob/4d736bfcc4db71a539f5e31a3b66d014df9add72/scripts/JERC2JSON/minimalDemo.py#L107C73-L107C75
            jets.eta, # == JetEta
            jets.pt_raw,
            jets.PU_rho, # == Rho
        )
        printCorrObjInputs(sf_ptres)
        # inputs = get_corr_inputs(example_value_dict, sf)
        jer_res = sf_ptres.evaluate(*inputs)
        # logger.debug("JER Res : {}".format(jer_res.compute()))

        key_jersmear = "JERSmear"
        sf_jersmear = cset_jersmear[key_jersmear]
        sf_input_names = [inp.name for inp in sf_jersmear.inputs]
        logger.debug(f"JER smear input: {sf_input_names}")

        pt_gen = ak.fill_none(jets.matched_gen.pt, value=-1.0) # if no match, fill with -1.0. Source https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/examples/jercExample.py?ref_type=heads#L45
        pt_jec = jets.pt_jec
        pt_gen_filter  = abs(pt_jec - pt_gen) < (3*pt_jec*jer_res) # Source https://github.com/cms-jet/JECDatabase/blob/4d736bfcc4db71a539f5e31a3b66d014df9add72/scripts/JERC2JSON/minimalDemo.py#L108C1-L108C66
        false_cond_val = -1*ak.ones_like(jets.pt_jec)
        pt_gen = ak.where(pt_gen_filter, pt_gen, false_cond_val)
        apply_scaling = pt_gen != -1.0
        # event_id is per-event (events.event), but every other input here is
        # per-jet. correctionlib's evaluate() only broadcasts a per-event scalar
        # against per-jet arrays via its jagged-awkward fallback path, which is
        # skipped whenever a chunk happens to have a uniform jet count across all
        # its events (the awkward array collapses to a regular/rectangular numpy
        # array, and plain numpy broadcasting -- aligned from the right -- then
        # fails: e.g. shape (2, 6) vs (2,)). Broadcast event_id to jet-level shape
        # explicitly so behavior doesn't depend on that coincidence, same as
        # PU_rho already is (see events["Jet","PU_rho"] assignment upstream).
        event_id_jets = ak.broadcast_arrays(pt_jec, event_id)[1]
        inputs = (
            pt_jec, # == JetPt
            jets.eta, # == JetEta
            pt_gen, # == GenPt
            jets.PU_rho, # == Rho
            event_id_jets, # == EventID
            jer_res, # == JERs
            jer_sf, # == JERSF

        )
        jer_smearing = sf_jersmear.evaluate(*inputs)
        if "puId" in jets.fields:
            jet_puId = jets.puId
        else: #dummy values
            jet_puId = ak.ones_like(jets.pt)
        # logger.debug("JER smearing : {}".format(jer_smearing[:20].compute()))
        # logger.debug(f"jets.pt b4 JER smear: {jets.pt[:20].compute()}")

        jer_strat = config["switches"]["jer_strat"]
        logger.info(f"jer_strat: {jer_strat}")
        if jer_strat == 0: # default Hybrid
            jer_smearing = jer_smearing # no change
        elif jer_strat == 1:
            jer_smearing = applyStrat1(apply_scaling, jer_smearing, jet_puId, pt_jec, jets.eta)
        elif jer_strat == 2:
            jer_smearing = applyStrat2(apply_scaling, jer_smearing, jet_puId, pt_jec, jets.eta)
        elif jer_strat == 3:
            jer_smearing = applyStrat1n2Revised(apply_scaling, jer_smearing, get_puId(jets), pt_jec, jets.eta, year)
        elif jer_strat == 4:
            jer_smearing = applyStrat4(apply_scaling, jer_smearing)
        else:
            raise ValueError(f"jer strategy {jer_strat} is not yet supported!")
        # jets["pt"] = jer_smearing * pt_jec # Source: https://github.com/cms-jet/JECDatabase/blob/4d736bfcc4db71a539f5e31a3b66d014df9add72/scripts/JERC2JSON/minimalDemo.py#L111
        jets[f"pt_jer_{syst}"] = jer_smearing * pt_jec  # Source: https://github.com/cms-jet/JECDatabase/blob/4d736bfcc4db71a539f5e31a3b66d014df9add72/scripts/JERC2JSON/minimalDemo.py#L111
    jets["pt"] = jets[f"pt_jer_nom"]
    # logger.debug(f"jet pt: {jets.pt[:100].compute()}")
    # logger.debug(f"jet pt_jer_up: {jets.pt_jer_up[:100].compute()}")
    # logger.debug(f"jet pt_jer_down: {jets.pt_jer_down[:100].compute()}")
    # The per-eta-bin JER-uncertainty columns (pt_jer1_up ... pt_jer6_down) are
    # only consumed when do_jer_unc drives the pt_variations loop. For a
    # nominal-only run the caller passes syst_l == ["nom"], so pt_jer_up/down
    # were never built -- skip apply_jer_unc (nothing downstream reads those
    # columns and they are not written to the skim).
    if "up" in syst_l and "down" in syst_l:
        jets = apply_jer_unc(jets)
    # for i in range(1,7):
    #     logger.debug(f"pt_jer{i}_up: {jets[f'pt_jer{i}_up'][:100].compute()}")
    #     logger.debug(f"pt_jer{i}_down: {jets[f'pt_jer{i}_down'][:100].compute()}")

    return jets


def get_jet_variation(jets_orig, variation, fields2add):
    """Generate a new jet collection with applied JEC/JER variation.

    This function creates a modified jet candidate array by applying the specified
    variation (e.g., JEC/JER up/down) to the original jets. The variation is applied
    to the transverse momentum (pt) and mass (if applicable), while preserving the
    original phi and eta. Additional fields from the original jets can be carried over.

    Parameters:
    ----------
    jets_orig : awkward.Array
        Original jet collection with candidate behavior (PtEtaPhiMCandidate)
    variation : str
        Name of the variation to apply (e.g., 'jec_unc1_up', 'jer1_up')
    fields2add : list of str
        List of additional fields to copy from original jets to the new collection

    Returns:
    -------
    awkward.Array
        New jet collection with candidate behavior, containing:
        - Modified pt and mass (depending on variation type)
        - Original phi and eta
        - All fields from fields2add

    Notes:
    -----
    - For 'jer' variations, mass is not modified and taken from original jets
    - For non-JER variations (e.g., JEC), mass is modified according to the variation
    - Uses PtEtaPhiMCandidate behavior for proper physics vector handling
    - Fields like 'puId' are handled with special fallback logic if missing
    """

    logger.debug(f"get_jet_variation variation: {variation}")
    new_jets_pt = jets_orig[f"pt_{variation}"]
    logger.debug(f"{variation} jets_orig.fields: {jets_orig.fields}")
    if "jer" in variation:
        new_jets_mass = jets_orig.mass
    else: # jec unc impacts mass, but jer uncs do not
        new_jets_mass = jets_orig[f"mass_{variation}"]

    new_jets = ak.zip( # bahviour setup source: https://mattermost.web.cern.ch/cms-exp/pl/fu9kemtazi8rznucdf57ug1xac
        {
            "x": new_jets_pt * np.cos(jets_orig.phi),
            "y": new_jets_pt * np.sin(jets_orig.phi),
            "z": new_jets_pt * np.sinh(jets_orig.eta),
            "mass": new_jets_mass,
            "charge": jets_orig.charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    ) # NOTE: if you use pt, eta, phi, or t variables to initialize, it doesn't work. It's quite finnicky in that way.
    for field in fields2add:
        if hasattr(jets_orig, field):
            new_jets[field] = getattr(jets_orig, field)
        else:
            logger.warning(f"jets_orig has no field {field}!")
            if field == "puId":
                new_jets["puId"] = get_puId(jets_orig)

    return new_jets
