from coffea.jetmet_tools import CorrectedJetsFactory, JECStack
# from src.corrections.custom_jec import CorrectedJetsFactory, JECStack
from coffea.lookup_tools import extractor
import numpy as np
import awkward as ak
import os
import correctionlib.schemav2 as cs
import correctionlib

import logging
from modules.utils import logger

# def jec_names_and_sources_yaml(jec_pars, year):
#     localdir = os.path.dirname(os.path.abspath("__file__"))
#     logger.debug(f"localdir: {localdir}")
#     OmegaConf.register_new_resolver("default_params_dir", lambda: localdir+"/data", replace=True)
#     jec_dict = OmegaConf.load(localdir+'/parameters/jets_calibration.yaml')
#     jec_fnames = jec_dict['default_jets_calibration']["factory_configuration"]["AK4PFchs"]["JES_JER_Syst"][year]
#     names = {}
#     suffix = {
#         "jec_names": [f"_{level}_AK4PFchs" for level in jec_pars["jec_levels_mc"]],
#         "junc_names": ["_Uncertainty_AK4PFchs"],
#         "junc_sources": ["_UncertaintySources_AK4PFchs"],
#         "jer_names": ["_PtResolution_AK4PFchs"],
#         "jersf_names": ["_SF_AK4PFchs"],
#     }

#     for key, suff in suffix.items():
#         if "data" in key:
#             names[key] = {}
#             for run in jec_pars["runs"]:
#                 for tag, iruns in jec_pars["jec_data_tags"].items():
#                     if run in iruns:
#                         names[key].update({run: [f"{tag}{s}" for s in suff]})
#         else:
#             tag = jec_pars["jer_tags"] if "jer" in key else jec_pars["jec_tags"]
#             names[key] = [f"{tag}{s}" for s in suff]

#     return names

def jec_names_and_sources(jec_pars):
    # logger.debug(f"jec_pars: {jec_pars}")
    jet_alg = jec_pars["jet_algorithm"]
    names = {}
    # suffix = {
    #     "jec_names": [f"_{level}_AK4PFchs" for level in jec_pars["jec_levels_mc"]],
    #     "jec_names_data": [
    #         f"_{level}_AK4PFchs" for level in jec_pars["jec_levels_data"]
    #     ],
    #     "junc_names": ["_Uncertainty_AK4PFchs"],
    #     "junc_names_data": ["_Uncertainty_AK4PFchs"],
    #     "junc_sources": ["_UncertaintySources_AK4PFchs"],
    #     "junc_sources_data": ["_UncertaintySources_AK4PFchs"],
    #     "jer_names": ["_PtResolution_AK4PFchs"],
    #     "jersf_names": ["_SF_AK4PFchs"],
    # }
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
    # jec_pars = {k: v[year] for k, v in jec_parameters.items()}
    # jec_pars = {k: v for k, v in jec_parameters.items()}

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



def jet_id(jets, config):
    # logger.debug(f"jets parameters: {parameters}")
    pass_jet_id = ak.ones_like(jets.jetId, dtype=bool)
    year = config["year"]
    if ("2016" in year) and ("RERECO" in year):  # 2016RERECO
        if "loose" in config["jet_id"]:
            pass_jet_id = jets.jetId >= 1
        elif "tight" in config["jet_id"]: # according to https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD#NanoAOD_format , jet Id is same for UL 2016,2017 and 2018
            pass_jet_id = jets.jetId >= 3
    else: # 2017RERECO, 2018RERECO, all UL and Run3
        if "loose" in config["jet_id"]:
            pass_jet_id = jets.jetId >= 1 # NOTE: for Run2 UL, loose is not specified in https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD#NanoAOD_format
        elif "tight" in config["jet_id"]: # according to https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD#NanoAOD_format , jet Id is same for UL 2016,2017 and 2018
            pass_jet_id = jets.jetId >= 2

    # logger.debug(f"pass_jet_id: {pass_jet_id[:10].compute()}")
    # test_pass_jet_id = jets.jetId >= 2
    # logger.debug(f"test_pass_jet_id: {test_pass_jet_id[:10].compute()}")
    # raise ValueError
    return pass_jet_id


def jet_puid(jets, config):
    jet_puid2use = config["jet_puid"]
    year = config["year"]
    if year=="2017_RERECO":
        logger.debug("using puId 17!")
        puId = jets.puId17
    else:
        puId = jets.puId
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
    pass_jet_puid = ak.ones_like(jets.jetId, dtype=bool)

    if "2017" in year: # for misreco due ot ECAL endcap noise
        eta_window = (abs(jets.eta) > 2.6) & (abs(jets.eta) < 3.0)
        # pass_jet_puid = (eta_window & jet_puid_wps["tight"]) | ( # tight puid in the noisy eta window, else loose
        #     (~eta_window) & jet_puid_wps[jet_puid2use]
        # )
        pass_jet_puid = (eta_window & (puId >= 7)) | (
                (~eta_window) & jet_puid_wps["loose"]
        )
    else:
        pass_jet_puid = jet_puid_wps[jet_puid2use]
        # logger.debug("else case!")
        # logger.debug(f"pass_jet_puid: {pass_jet_puid[:10].compute()}")
        # logger.debug(f"jet_puid_wps['loose']: {jet_puid_wps['loose'][:10].compute()}")
        # raise ValueError
    return pass_jet_puid


def fill_softjets(events, jets, mu1, mu2, nmuons, cutoff, test_mode=False):
    if test_mode:
        logger.debug(f"jets events.SoftActivityJet.fields: {events.SoftActivityJet.fields}")
        logger.debug(f"jets cutoff: {cutoff}")
    events["SoftActivityJet","mass"] = 0
    saj = events.SoftActivityJet
    saj_Njets = events[f"SoftActivityJetNjets{cutoff}"]
    saj_HT = events[f"SoftActivityJetHT{cutoff}"]


    njets = ak.num(jets, axis=1)
    padded_jets = ak.pad_none(jets, 2)
    jet1 = padded_jets[:,0]
    jet2 = padded_jets[:,1]

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

    evnts_to_correct = (nmuons==2) |(njets > 0)
    if test_mode:
        logger.debug(f"jets evnts_to_correct: {evnts_to_correct}")
        logger.debug(f"jets footprint_njets b4: {ak.to_numpy(saj_Njets)}")
        logger.debug(f"jets corrected_njets b4: {ak.to_numpy(saj_HT)}")

    saj_Njets = ak.where(evnts_to_correct,corrected_njets,saj_Njets)
    saj_HT = ak.where(evnts_to_correct,ht_corrected,saj_HT)

    if test_mode:
        logger.debug(f"jets footprint_njets after: {ak.to_numpy(saj_Njets)}")
        logger.debug(f"jets corrected_njets after: {ak.to_numpy(saj_HT)}")
    out_dict = {
        f"nsoftjets{cutoff}" : saj_Njets,
        f"htsoft{cutoff}" : saj_HT
    }
    return out_dict



def getHemVetoRunFilter(run, event_num, config, is_mc):
    """
    For data:
    return the conditions for applying HemVeto. For data, this is just
    end of data B run + full data C,D (run >= 319077).
    For MC:
    Randomly reject a given fraction of events using for MC to match HEM Vetoed jets in 2018 UL as reccommended in https://cms-talk.web.cern.ch/t/question-about-hem15-16-issue-in-2018-ultra-legacy/38654/8 (though we reject her "eventNum % 15 == 0" method of random rejection and just use random number generation)
    """
    if is_mc:
        prob = config["HemVeto_ratio"] # ratio of HemVeto applicable run / total nevents for 2018UL
        logger.debug(f"HEMveto prob: {prob}")
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

def applyHemVeto(jets, run, event_num, config, is_mc: bool):
    """
    Apply HEM veto for 2018 UL as recommended on https://cms-talk.web.cern.ch/t/question-about-hem15-16-issue-in-2018-ultra-legacy/38654/5
    """
    puId = jets.puId
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
    pass_jet_tightID = (jets.jetId >= 2) & jet_em_frac_cut & jet_muon_iso_cut

    pass_jet_tightLepVetoID = jets.jetId ==6 # Source: https://twiki.cern.ch/twiki/bin/view/CMS/JetID13TeVUL

    pass_jet_id_total = pass_jet_tightLepVetoID | pass_jet_tightID # Source: https://cms-talk.web.cern.ch/t/question-about-hem15-16-issue-in-2018-ultra-legacy/38654/2



    loose_jet_selection =( # Source: https://cms-talk.web.cern.ch/t/question-about-hem15-16-issue-in-2018-ultra-legacy/38654/2
        jets.pt > 15
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
    hemveto_run_filter = getHemVetoRunFilter(run, event_num, config, is_mc)

    # combine all the conditions
    hemveto = loose_jet_selection & hemveto_region
    hemveto = ak.any(hemveto, axis=1) & hemveto_run_filter
    # we reject events if we find hemveto jets, so reverse the bool arr
    hemveto = ~hemveto
    is_HemRegion = ak.any(hemveto_region, axis=1) # eventwise arr if any jet is in the hem region
    return hemveto, is_HemRegion


def getJecDataTag(run, jec_data_tags):
    logger.debug(f"run: {run}")
    logger.debug(f"jec_data_tags: {jec_data_tags}")
    for jec_tag, jec_run_l in jec_data_tags.items():
        for jec_run in jec_run_l:
            if run == jec_run:
                logger.debug(f"found match in jec_run {jec_run}!")
                return jec_tag

    return None # return none if nothing matches

def do_jec_scale(jets, config, is_mc, dataset):
    jec_parameters = config["jec_parameters"]

    jerc_load_path = jec_parameters["jerc_load_path"]
    cset = correctionlib.CorrectionSet.from_file(jerc_load_path)


    if is_mc:
        jec_tag = jec_parameters["jec_tags"]
    else: # data
        jec_tag = None
        for run in jec_parameters["runs"]:
            logger.debug(f"run: {run}")
            logger.debug(f"dataset: {dataset}")
            if run in dataset:
                jec_tag = getJecDataTag(run, jec_parameters["jec_data_tags"])
    logger.debug(f"jec_tag: {jec_tag}")
    if jec_tag is None:
        logger.debug("ERROR! JEC tag not found!")
        raise ValueError


    # algo = "AK4PFchs"
    algo = jec_parameters["jet_algorithm"]
    lvl_compound = "L1L2L3Res"

    key = "{}_{}_{}".format(jec_tag, lvl_compound, algo)
    logger.debug(f"jec key: {key}")
    sf = cset.compound[key]

    sf_input_names = [inp.name for inp in sf.inputs]
    logger.debug(f"JEC input: {sf_input_names}")


    inputs = (
        jets.area, # == JetA
        jets.eta, # == JetEta
        jets.pt_raw, # == JetPt
        jets.PU_rho, # == Rho
    )
    # inputs = get_corr_inputs(example_value_dict, sf)
    new_jec_scale = sf.evaluate(*inputs)
    # logger.debug("JSON result AK4: {}".format(new_jec_scale[:20].compute()))
    jet_pt_jec = new_jec_scale*jets.pt_raw
    jet_mass_jec = new_jec_scale*jets.mass_raw
    jets["pt"] = jet_pt_jec
    jets["mass"] = jet_mass_jec
    jets["pt_jec"] = jet_pt_jec
    jets["mass_jec"] = jet_mass_jec
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


def do_jer_smear(jets, config, syst, event_id):
    """
    we assume that jec has been applied (we need pt_jec and pt_raw)

    params:
    syst: nom, up and down
    """
    jec_parameters = config["jec_parameters"]
    jerc_load_path = jec_parameters["jerc_load_path"]
    cset = correctionlib.CorrectionSet.from_file(jerc_load_path)


    jersmear_load_path = jec_parameters["jersmear_load_path"]
    cset_jersmear = correctionlib.CorrectionSet.from_file(jersmear_load_path)


    # jer_tag = "Summer20UL16_JRV3_MC"
    jer_tag = jec_parameters["jer_tags"]
    algo = "AK4PFchs"

    # First, jet JER SF
    key = "{}_{}_{}".format(jer_tag, "ScaleFactor", algo)
    sf = cset[key]
    sf_input_names = [inp.name for inp in sf.inputs]
    logger.debug(f"JER SF input: {sf_input_names}")

    # Second, get JER resolution
    inputs = (
        jets.eta, # == JetEta
        syst, # == systematic
    )
    jer_sf = sf.evaluate(*inputs)
    # logger.debug("JER SF : {}".format(jer_sf.compute()))


    key = "{}_{}_{}".format(jer_tag, "PtResolution", algo)
    sf = cset[key]

    sf_input_names = [inp.name for inp in sf.inputs]
    logger.debug(f"JER resolution input: {sf_input_names}")

    inputs = ( # Source: https://github.com/cms-jet/JECDatabase/blob/4d736bfcc4db71a539f5e31a3b66d014df9add72/scripts/JERC2JSON/minimalDemo.py#L107C73-L107C75
        jets.eta, # == JetEta
        jets.pt_raw, # == systematic
        jets.PU_rho, # == Rho
    )
    # inputs = get_corr_inputs(example_value_dict, sf)
    jer_res = sf.evaluate(*inputs)
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
    inputs = (
        pt_jec, # == JetPt
        jets.eta, # == JetEta
        pt_gen, # == GenPt
        jets.PU_rho, # == Rho
        event_id, # == EventID
        jer_res, # == JERs
        jer_sf, # == JERSF

    )
    jer_smearing = sf_jersmear.evaluate(*inputs)
    # logger.debug("JER smearing : {}".format(jer_smearing[:20].compute()))
    # logger.debug(f"jets.pt b4 JER smear: {jets.pt[:20].compute()}")
    # jer_smearing = applyStrat1(apply_scaling, jer_smearing, jets.puId, pt_jec, jets.eta)
    # jer_smearing = applyStrat2(apply_scaling, jer_smearing, jets.puId, pt_jec, jets.eta)
    jer_smearing = applyStrat1n2(apply_scaling, jer_smearing, jets.puId, pt_jec, jets.eta)

    # print("JER smearing : {}".format(jer_smearing[:20].compute()))
    # print(f"jets.pt b4 JER smear: {jets.pt[:20].compute()}")
    jets["pt"] = jer_smearing * pt_jec # Source: https://github.com/cms-jet/JECDatabase/blob/4d736bfcc4db71a539f5e31a3b66d014df9add72/scripts/JERC2JSON/minimalDemo.py#L111
    # logger.debug(f"jets.pt after JER smear: {jets.pt[:20].compute()}")
    return jets

