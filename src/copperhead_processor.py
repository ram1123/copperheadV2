import time
from typing import Tuple, TypeVar

import awkward as ak
import coffea.processor as processor
import correctionlib
import numpy as np
from coffea.analysis_tools import PackedSelection, Weights
from coffea.btag_tools import BTagScaleFactor
from coffea.lookup_tools import extractor
from coffea.lumi_tools import LumiMask
from coffea.nanoevents.methods import vector
import dask_awkward as dak

from modules.classify_year import is_run2, is_run3
from modules.get_sample_info import get_sample_info
from modules.utils import logger
from omegaconf import OmegaConf

# from src.corrections.weight import Weights
from src.corrections.evaluator import (
    add_pdf_variations,
    add_stxs_variations,
    btag_weights_jsonKeepDim,
    get_jetpuid_weights_eta_dependent,
    get_musf_lookup,
    lhe_weights,
    musf_evaluator,
    nnlops_weights,
    pu_evaluator,
    qgl_weights_V2,
)
from src.corrections.muon_sf import add_muon_sfs_correctionlib
from src.corrections.fsr_recovery import fsr_recoveryV1
from src.corrections.geofit import apply_geofit
from src.corrections.jet import (
    applyHemVeto,
    applyJetUncertaintyKinematics,
    applyUpDown,
    custom_jet_id,
    do_jec_scale,
    do_jer_smear,
    fill_softjets_HIG19006,
    get_jec_factories,
    get_jet_variation,
    get_puId,
    jet_id,
    jet_puid,
)
from src.corrections.rochester import apply_roccor, apply_roccorRun3
from src.corrections.zpt_dnn import ZptDNNConfig, eval_zpt_torchscript_by_njet

from modules.vector_operations import (
    getRapidity,
    delta_r_V1,
    etaFrame_variables,
    cs_variables,
)

coffea_nanoevent = TypeVar('coffea_nanoevent')
ak_array = TypeVar('ak_array')


def safe_ratio(num, den, default=0.0):
    """Element-wise safe division for awkward arrays."""
    return ak.where(den != 0, num / den, default)


def getZptWgts_3region(dimuon_pt, njets, nbins, year, config_path):
    """
    Get Z pT weights based on polynomial fits in 3 regions.
    TODO: Implement the possibility to apply the zpt weights w.r.t. number of generated jets instead of reco jets.
    """
    logger.info(f"zpt config file: {config_path}")
    wgt_config = OmegaConf.load(config_path)
    zpt_wgt = ak.ones_like(dimuon_pt)
    jet_multiplicies = [0,1,2]

    for jet_multiplicity in jet_multiplicies:

        zpt_wgt_by_jet = ak.zeros_like(dimuon_pt)

        # Get cut-off regions between the polynomial fits
        poly_fit_cutoff_min = wgt_config[str(year)][f"njet_{jet_multiplicity}"][nbins]["polynomial_range"]["xmin1"]
        poly_fit_cutoff_max = wgt_config[str(year)][f"njet_{jet_multiplicity}"][nbins]["polynomial_range"]["xmax1"]

        # Get the function order
        f0_order = int(wgt_config[str(year)][f"njet_{jet_multiplicity}"][nbins]["fit_orders"]["f0_order"])
        f1_order = int(wgt_config[str(year)][f"njet_{jet_multiplicity}"][nbins]["fit_orders"]["f1_order"])

        # first polynomial fit
        zpt_wgt_by_jet_poly = ak.zeros_like(dimuon_pt)
        for order in range(f0_order + 1):  # Dynamically use max_order from the configuration
            coeff = wgt_config[str(year)][f"njet_{jet_multiplicity}"][nbins][f"f0_p{order}"]
            polynomial_term = coeff*(dimuon_pt**order) # a * x^n
            zpt_wgt_by_jet_poly = zpt_wgt_by_jet_poly + polynomial_term

        # compute the value of the first polynomial at the cutoff min
        f0_xmin = 0.0
        for order in range(f0_order + 1):
            coeff = wgt_config[str(year)][f"njet_{jet_multiplicity}"][nbins][f"f0_p{order}"]
            f0_xmin += coeff * (poly_fit_cutoff_min ** order)

        zpt_wgt_by_jet = ak.where((poly_fit_cutoff_min >= dimuon_pt), zpt_wgt_by_jet_poly, zpt_wgt_by_jet)

        # 2nd polynomial fit
        zpt_wgt_by_jet_poly = ak.zeros_like(dimuon_pt)
        for order in range(f1_order + 1):  # p goes from 0 to max_order
            coeff = wgt_config[str(year)][f"njet_{jet_multiplicity}"][nbins][f"f1_p{order}"]
            polynomial_term = coeff * (dimuon_pt**order)  # a * x^n
            zpt_wgt_by_jet_poly = zpt_wgt_by_jet_poly + polynomial_term

        # compute the value of the 2nd polynomial at the cutoff min
        f1_xmin = 0.0
        f1_xmax = 0.0
        for order in range(f1_order + 1):
            coeff = wgt_config[str(year)][f"njet_{jet_multiplicity}"][nbins][f"f1_p{order}"]
            f1_xmin += coeff * (poly_fit_cutoff_min ** order)
            f1_xmax += coeff * (poly_fit_cutoff_max ** order)

        # continuity offset so that f1(xmin)+offset == f0(xmin)
        offset = f0_xmin - f1_xmin

        zpt_wgt_by_jet = ak.where(
            ((poly_fit_cutoff_min < dimuon_pt) & (poly_fit_cutoff_max >= dimuon_pt)),
            zpt_wgt_by_jet_poly + offset,
            zpt_wgt_by_jet)

        # horizontal line beyond poly_fit_cutoff_max horizontal_c0 and horizontal_mx
        # coeff = wgt_config[str(year)][f"njet_{jet_multiplicity}"][nbins]["horizontal_c0"]
        mx = wgt_config[str(year)][f"njet_{jet_multiplicity}"][nbins]["horizontal_mx"]
        y_at_xmax = f1_xmax + offset
        coeff = y_at_xmax - mx*poly_fit_cutoff_max

        zpt_wgt_by_jet_horizontal = mx*dimuon_pt + coeff # y=mx*x + c0
        zpt_wgt_by_jet = ak.where((poly_fit_cutoff_max < dimuon_pt), zpt_wgt_by_jet_horizontal, zpt_wgt_by_jet)

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
    Returns a dict of jet1/jet2 for different pairing criteria.
    jets is the already-selected, pt-sorted Array of jets per event.
    """
    # need at least 2 jets to form a pair; combinations() will give empty for <2
    pairs = ak.combinations(jets, 2, fields=["j1", "j2"])

    # pair metrics
    mjj = (pairs.j1 + pairs.j2).mass
    deta = np.abs(pairs.j1.eta - pairs.j2.eta)

    # indices of best pairs (per event)
    idx_max_mjj = ak.argmax(mjj, axis=1, keepdims=True)
    idx_max_deta = ak.argmax(deta, axis=1, keepdims=True)

    # criterion 4: max mjj with deta > 2.5
    deta_cut = 2.5
    mjj_masked = ak.where(deta > deta_cut, mjj, -np.inf)
    idx_max_mjj_deta = ak.argmax(mjj_masked, axis=1, keepdims=True)

    # extract pairs (these are still "length-1 lists" per event because keepdims=True)
    pair_max_mjj = pairs[idx_max_mjj]
    pair_max_deta = pairs[idx_max_deta]
    pair_max_mjj_deta = pairs[idx_max_mjj_deta]

    # fallback for criterion 4 when no pair passes deta>2.5:
    # detect "all masked" events -> max is -inf
    has_pair_deta = ak.any(deta > deta_cut, axis=1)
    # flatten the chosen pair objects to scalars per event
    j1_mjj_deta = ak.firsts(pair_max_mjj_deta.j1)
    j2_mjj_deta = ak.firsts(pair_max_mjj_deta.j2)

    j1_mjj = ak.firsts(pair_max_mjj.j1)
    j2_mjj = ak.firsts(pair_max_mjj.j2)

    j1_mjj_deta = ak.where(has_pair_deta, j1_mjj_deta, j1_mjj)
    j2_mjj_deta = ak.where(has_pair_deta, j2_mjj_deta, j2_mjj)

    # criterion 1: leading-pt jets (your current method)
    padded = ak.pad_none(jets, 2)
    j1_lead = padded[:, 0]
    j2_lead = padded[:, 1]

    # criterion 2: max mjj
    j1_max_mjj = j1_mjj
    j2_max_mjj = j2_mjj

    # criterion 3: max deta
    j1_max_deta = ak.firsts(pair_max_deta.j1)
    j2_max_deta = ak.firsts(pair_max_deta.j2)

    return {
        "lead": (j1_lead, j2_lead),
        "max_mjj": (j1_max_mjj, j2_max_mjj),
        "max_deta": (j1_max_deta, j2_max_deta),
        "mjj_deta": (j1_mjj_deta, j2_mjj_deta),
        "has_mjj_deta": has_pair_deta,
    }


class EventProcessor(processor.ProcessorABC):
    def __init__(self, config: dict, test_mode=False, isCutflow=False, **kwargs):
        self.config = config
        self.isCutflow = isCutflow
        self.test_mode = test_mode

        year = self.config["year"]

        # Initialize PackedSelection
        # Reference: https://nbviewer.org/github/scikit-hep/coffea/blob/master/binder/packedselection.ipynb
        self.selection = {}
        self.cutflow = {}

    def compute_jet_veto_eventfilter(self, events, jets):
        """ apply the jet veto maps. the .gz file should be read using correctionlib and the file
        # is saved in "jet_veto_maps" field in config. Also switch to turn on/off the jet veto map
        # application is in "do_jet_veto_maps_filterEvents" field in config.
        # If any jet in the event falls into the veto map region, the whole event is vetoed.
        """
        jet_veto_maps_path = self.config.get("jet_veto_maps", None)
        logger.debug(f"jet_veto_maps_path: {jet_veto_maps_path}")
        if jet_veto_maps_path is None:
            logger.error("Jet veto maps path is not specified in the config!")
            raise ValueError("Jet veto maps path is not specified in the config!")

        # Load correction set
        cset = correctionlib.CorrectionSet.from_file(jet_veto_maps_path)
        logger.debug(f"jet_veto_maps_cset: {cset}")
        logger.debug(f"jet_veto_maps_cset keys: {list(cset.keys())}")

        input_dict = {
            "type": "jetvetomap",
            "eta": jets.eta,
            "phi": jets.phi,
        }

        jetVetoMapTag = self.config.get("jet_veto_maps_tag", None)
        logger.debug(f"Jet veto map tag from config: {jetVetoMapTag}")

        jet_veto_map = cset[jetVetoMapTag]
        inputs = [input_dict[input.name] for input in cset[jetVetoMapTag].inputs]

        # logger.debug(f"eta: {ak.to_list(jets.eta[50:56].compute())}")
        # logger.debug(f"phi: {ak.to_list(jets.phi[50:56].compute())}")

        jet_veto_mask = jet_veto_map.evaluate(*(inputs))

        # logger.debug(f"jet_veto_mask: {ak.to_list(jet_veto_mask[50:56].compute())}")

        jet_veto_eventFilter = ak.any(jet_veto_mask, axis=1)
        # logger.debug(f"jet_veto_eventFilter: {ak.to_list(jet_veto_eventFilter[50:56].compute())}")

        return jet_veto_eventFilter

    def compute_jet_veto_jetfilter(self, events, jets, PuppiMET):
        """apply the jet veto maps. the .gz file should be read using correctionlib and the file
        # is saved in "jet_veto_maps" field in config. Also switch to turn on/off the jet veto map
        # application is in "do_jet_veto_maps_filterJets" field in config.
        # If any jet in the event falls into the veto map region, then just remove that jet from the jet collection.
        # and set the MET pt to zero.
        """
        jet_veto_maps_path = self.config.get("jet_veto_maps", None)
        logger.debug(f"jet_veto_maps_path: {jet_veto_maps_path}")
        if jet_veto_maps_path is None:
            logger.error("Jet veto maps path is not specified in the config!")
            raise ValueError("Jet veto maps path is not specified in the config!")

        # Load correction set
        cset = correctionlib.CorrectionSet.from_file(jet_veto_maps_path)
        logger.debug(f"jet_veto_maps_cset: {cset}")
        logger.debug(f"jet_veto_maps_cset keys: {list(cset.keys())}")

        input_dict = {
            "type": "jetvetomap",
            "eta": jets.eta,
            "phi": jets.phi,
        }

        jetVetoMapTag = self.config.get("jet_veto_maps_tag", None)
        logger.debug(f"Jet veto map tag from config: {jetVetoMapTag}")

        jet_veto_map = cset[jetVetoMapTag]
        inputs = [input_dict[input.name] for input in cset[jetVetoMapTag].inputs]

        # logger.debug(f"eta: {ak.to_list(jets.eta[40:47].compute())}")
        # logger.debug(f"phi: {ak.to_list(jets.phi[40:47].compute())}")

        jet_veto_mask = jet_veto_map.evaluate(*(inputs))
        # logger.debug(f"jet_veto_mask: {ak.to_list(jet_veto_mask[40:47].compute())}")

        jet_veto_eventFilter = ak.any(jet_veto_mask, axis=1)
        # logger.debug(f"jet_veto_eventFilter: {ak.to_list(jet_veto_eventFilter[30:35].compute())}")

        # logger.debug(f"PuppiMET.pt after jet veto jet filter: {ak.to_list(PuppiMET.pt[30:35].compute())}")

        jets = jets[jet_veto_mask != 100.0]

        # logger.debug(f"eta: {ak.to_list(jets.eta[40:47].compute())}")

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

        # logger.debug(f"PuppiMET.pt after jet veto jet filter: {ak.to_list(PuppiMET.pt[30:35].compute())}")

        return jets, PuppiMET

    def process(self, events: coffea_nanoevent, dataset_yaml_file: str):
        t0 = time.perf_counter()
        year = self.config["year"]

        # ReInitialize PackedSelection, otherwise processor would merge selection from previous run
        self.selection = PackedSelection()

        event_filter = ak.ones_like(events.event, dtype="bool") # 1D boolean array to be used to filter out bad events
        # Debugging: Check structure of event_filter
        logger.debug(f"event_filter type: {type(event_filter)}")
        logger.debug(f"event_filter length: {len(event_filter)}")
        logger.debug(f"events length: {len(events)}")

        # For debug: if run, lumi, and event :
        # 356371,72,61849995
        # debug_run = 356371
        # debug_lumi = 72
        # debug_event = 61849995
        # debug_mask = ~((events.run == debug_run) & (events.luminosityBlock == debug_lumi) & (events.event == debug_event))
        # event_filter = event_filter & debug_mask

        # # just print muon pT for run, lumi, and event : 355870,33,39923308
        # debug_mask_2 = (events.run == debug_run) & (events.luminosityBlock == debug_lumi) & (events.event == debug_event)

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
        if "dy_M-50" in dataset and self.config["switches"]["do_remove_dy_M100to200"]:
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

            # LHE_filter = ak.flatten(((LHE_dilepton_mass > 100) & (LHE_dilepton_mass < 200)))
            LHE_filter = (((LHE_dilepton_mass > 100) & (LHE_dilepton_mass < 200)))[:,0]
            # logger.info(f"LHE_filter: {LHE_filter.compute()}")
            LHE_filter = ak.fill_none(LHE_filter, value=False)
            LHE_filter = (LHE_filter== False) # we want True to indicate that we want to keep the event
            # logger.info(f"copperhead2 EventProcessor LHE_filter[32]: \n{ak.to_numpy(LHE_filter[32])}")
            # self.selection.add("LHE_cut", LHE_filter)
            event_filter = event_filter & LHE_filter

            self.selection.add("LHE_cut", LHE_filter)

        # LHE cut original end -----------------------------------------------------------------------------

        t3 = time.perf_counter()
        logger.info(f"[timing] LHE cut time: {t3 - t1:.2f} seconds")
        # ------------------------------------------------------------#

        # ------------------------------------------------------------#
        # Step-3: Apply HLT
        # ------------------------------------------------------------#
        # Apply HLT to both Data and MC.
        # NOTE: this would probably be superfluous if you already do trigger matching
        HLT_filter = ak.zeros_like(event_filter, dtype="bool")  # start with 1D of Falses
        for HLT_str in self.config["hlt"]:
            logger.debug(f"HLT_str: {HLT_str}")
            # HLT_filter = HLT_filter | events.HLT[HLT_str]
            HLT_filter = HLT_filter | ak.fill_none(events.HLT[HLT_str], value=False)
        self.selection.add("HLT_filter", HLT_filter)
        event_filter = event_filter & HLT_filter

        t4 = time.perf_counter()
        logger.info(f"[timing] HLT and lumi mask time: {t4 - t3:.2f} seconds")
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
        # logger.debug(f"muons pT: {events.Muon.pt[:5].compute()}")

        # Save raw variables before computing any corrections
        # rochester corrects pt only, but fsr_recovery changes all vals below
        events["Muon", "pt_raw"] = ak.ones_like(events.Muon.pt) * events.Muon.pt
        events["Muon", "eta_raw"] = ak.ones_like(events.Muon.eta) * events.Muon.eta
        events["Muon", "phi_raw"] = ak.ones_like(events.Muon.phi) * events.Muon.phi
        events["Muon", "pfRelIso04_all_raw"] = ak.ones_like(events.Muon.pfRelIso04_all) * events.Muon.pfRelIso04_all

        # --------------------------------------------------------
        # Step-7: Apply Rochester correction to muon pT
        # --------------------------------------------------------
        if self.config["switches"]["do_roccor"]:
            # TODO make more elegant distinction between Run2 and Run3
            if is_run2(year):
                logger.debug("doing Run2 rochester!")
                apply_roccor(events, self.config["roccor_file"], is_mc)
            elif is_run3(year):
                logger.debug("doing Run3 rochester!")
                apply_roccorRun3(events, self.config["roccor_file"], is_mc)
            else:
                raise ValueError(f"Year {year} is neither Run2 nor Run3!")
            events["Muon", "pt"] = events.Muon.pt_roch
            # logger.info(f"df.Muon.pt after roccor: {events.Muon.pt.compute()}")
        else:
            events["Muon", "pt_roch"] = events.Muon.pt

        muon_selection = (
            (events.Muon.pt_raw > self.config["muon_pt_cut"]) # pt_raw is pt b4 rochester #FIXME: Why pt_raw
            & (abs(events.Muon.eta_raw) < self.config["muon_eta_cut"])
            & events.Muon[self.config["muon_id"]]
            & (events.Muon.isGlobal | events.Muon.isTracker) # Table 3.5  AN-19-124
        )

        # logger.info(f"Debug event muon pt after roccor: {events.Muon.pt[debug_mask_2].compute()}")
        # logger.info(f"Debug event muon pt_raw after roccor: {events.Muon.pt_raw[debug_mask_2].compute()}")
        # logger.info(f"Debug event muon pt_roch after roccor: {events.Muon.pt_roch[debug_mask_2].compute()}")
        # logger.info(f"Debug event muon eta after roccor: {events.Muon.eta[debug_mask_2].compute()}")
        # logger.info(f"Debug event muon eta after roccor: {events.Muon.eta_raw[debug_mask_2].compute()}")
        # logger.info(f"Debug event muon phi after roccor: {events.Muon.phi[debug_mask_2].compute()}")
        # logger.info(f"Debug event muon id after roccor: {events.Muon[self.config['muon_id']][debug_mask_2].compute()}")
        # logger.info(f"Debug event muon isGlobal after roccor: {events.Muon.isGlobal[debug_mask_2].compute()}")
        # logger.info(f"Debug event muon isTracker after roccor: {events.Muon.isTracker[debug_mask_2].compute()}")

        self.selection.add("muon_pT_roch", ak.any(events.Muon.pt_roch >= self.config["muon_pt_cut"], axis=1))
        self.selection.add("muon_eta", ak.any(abs(events.Muon.eta_raw) <= self.config["muon_eta_cut"], axis=1))
        self.selection.add("muon_id", ak.any(events.Muon[self.config["muon_id"]], axis=1))
        self.selection.add("muon_isGlobal_or_Tracker", ak.any(events.Muon.isGlobal | events.Muon.isTracker, axis=1))
        self.selection.add("muon_selection", ak.any(muon_selection, axis=1))

        # calculate FSR recovery, but don't apply it until trigger matching is done
        # but apply muon iso overwrite, so base muon selection could be done
        do_fsr = self.config["switches"]["do_fsr"]
        if do_fsr:
            logger.debug("doing fsr!")
            # applied_fsr = fsr_recovery(events)
            applied_fsr = fsr_recoveryV1(events)# testing for pt_raw inconsistency
            events["Muon", "pfRelIso04_all"] = events.Muon.iso_fsr

        # apply iso portion of base muon selection, now that possible FSR photons are integrated into pfRelIso04_all as specified in line 360 of AN-19-124
        muon_selection = muon_selection & (events.Muon.pfRelIso04_all < self.config["muon_iso_cut"])
        self.selection.add("muon_iso", ak.any(events.Muon.pfRelIso04_all < self.config["muon_iso_cut"], axis=1))
        # logger.info(f"muon_selectiont: {ak.to_dataframe(muon_selection.compute())}")

        # logger.info(f"Debug event muon pt after roccor: {events.Muon.pt[debug_mask_2].compute()}")
        # logger.info(f"Debug event muon pt_raw after roccor: {events.Muon.pt_raw[debug_mask_2].compute()}")
        # logger.info(f"Debug event muon pt_roch after roccor: {events.Muon.pt_roch[debug_mask_2].compute()}")
        # logger.info(f"Debug event muon eta after roccor: {events.Muon.eta[debug_mask_2].compute()}")
        # logger.info(f"Debug event muon eta after roccor: {events.Muon.eta_raw[debug_mask_2].compute()}")
        # logger.info(f"Debug event muon phi after roccor: {events.Muon.phi[debug_mask_2].compute()}")
        # logger.info(f"Debug event muon id after roccor: {events.Muon[self.config['muon_id']][debug_mask_2].compute()}")
        # logger.info(f"Debug event muon isGlobal after roccor: {events.Muon.isGlobal[debug_mask_2].compute()}")
        # logger.info(f"Debug event muon isTracker after roccor: {events.Muon.isTracker[debug_mask_2].compute()}")

        t5 = time.perf_counter()
        logger.info(f"[timing] Muon selection time: {t5 - t4:.2f} seconds")
        # --------------------------------------------------------
        # apply tirgger match after base muon selection and Rochester correction, but b4 FSR recovery as implied in line 373 of AN-19-124
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

            self.selection.add("trigger_match", trigger_match)
        else:
            do_seperate_mu1_leading_pt_cut = True
            logger.warning("NO trigger match! Doing leading mu pass instead!")

        t6 = time.perf_counter()
        logger.info(f"[timing] Trigger match time: {t6 - t5:.2f} seconds")
        # --------------------------------------------------------

        # # print the mask debug_mask_2 and trigger_match for the debug event
        # logger.info(f"After Trigger match event trigger_match: {trigger_match[debug_mask_2].compute()}")
        # logger.info(f"After Trigger match event muon pt after roccor: {events.Muon.pt[debug_mask_2].compute()}")
        # logger.info(f"After Trigger match event muon pt_raw after roccor: {events.Muon.pt_raw[debug_mask_2].compute()}")
        # logger.info(f"After Trigger match event muon pt_roch after roccor: {events.Muon.pt_roch[debug_mask_2].compute()}")
        # logger.info(f"After Trigger match event muon eta after roccor: {events.Muon.eta[debug_mask_2].compute()}")
        # logger.info(f"After Trigger match event muon eta after roccor: {events.Muon.eta_raw[debug_mask_2].compute()}")
        # logger.info(f"After Trigger match event muon phi after roccor: {events.Muon.phi[debug_mask_2].compute()}")
        # logger.info(f"After Trigger match event muon id after roccor: {events.Muon[self.config['muon_id']][debug_mask_2].compute()}")
        # logger.info(f"After Trigger match event muon isGlobal after roccor: {events.Muon.isGlobal[debug_mask_2].compute()}")
        # logger.info(f"After Trigger match event muon isTracker after roccor: {events.Muon.isTracker[debug_mask_2].compute()}")
        # logger.info(f"After Trigger match event muon pfRelIso04_all after roccor: {events.Muon.pfRelIso04_all[debug_mask_2].compute()}")
        # logger.info(f"After Trigger match event muon_selection: {muon_selection[debug_mask_2].compute()}")

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

        muons = events.Muon[muon_selection]
        t6c = time.perf_counter()
        logger.info(f"[timing] Muon selection time: {t6c - t6a:.2f} seconds")

        # muons = ak.to_packed(events.Muon[muon_selection])

        # do the separate mu1 leading pt cut that copperheadV1 does instead of trigger matching
        if do_seperate_mu1_leading_pt_cut:
            muons_padded = ak.pad_none(muons, 2)
            sorted_args = ak.argsort(muons_padded.pt_raw, ascending=False) # since we're applying cut onver raw pt, we sort by raw pt. Sorting by reco pt gives us fewer events
            muons_sorted = (muons_padded[sorted_args])
            mu1 = muons_sorted[:,0]
            pass_leading_pt = ak.fill_none((mu1.pt_raw > self.config["muon_leading_pt"]), value=False)
            event_filter = event_filter & pass_leading_pt

            self.selection.add("leading_muon_pt", pass_leading_pt)

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
        # logger.debug(f"nelectrons: {nelectrons[debug_mask_2].compute()}")
        # logger.debug(f"electron_veto: {electron_veto[debug_mask_2].compute()}")
        self.selection.add("electron_veto", electron_veto)
        if self.config["switches"]["do_HemVeto"]:
            HemVeto_filter, is_HemRegion = applyHemVeto(events.Jet, events.run, events.event, self.config, is_mc)
            if (not self.config["switches"]["do_HemVetoStudy"]): # when we are calculating HemVeto fraction for MC, we shouldn't filter out hem veto events
                logger.info("adding HemVeto!")
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
        if self.config["switches"].get("do_jet_veto_maps_filterEvents", False):
            logger.info("Applying jet veto maps!")
            jets_for_veto = events.Jet
            jet_veto_eventFilter = self.compute_jet_veto_eventfilter(events, jets_for_veto)
            event_filter = event_filter & ~jet_veto_eventFilter

            keep_after_jet_veto = ~jet_veto_eventFilter
            self.selection.add("jet_veto_maps", keep_after_jet_veto)

        # Below patch is to define dimuon mass for the cutflow before we filter out bad events
        # ------------------- Cutflow dimuon mass window: START -----------------------
        muons_padded_for_mass = ak.pad_none(muons, target=2)
        sorted_args_for_mass = ak.argsort(muons_padded_for_mass.pt, ascending=False)
        muons_sorted_for_mass = (muons_padded_for_mass[sorted_args_for_mass])
        mu1_for_mass = muons_sorted_for_mass[:,0]
        mu2_for_mass = muons_sorted_for_mass[:,1]
        dimuon_for_mass = mu1_for_mass + mu2_for_mass
        dimuon_mass_for_cutflow = ak.fill_none(dimuon_for_mass.mass, 0.0)
        dimuon_mass_window_cut = ( (dimuon_mass_for_cutflow > 76.0) & (dimuon_mass_for_cutflow < 106.0) )
        self.selection.add("dimuon_mass_window_76_106", dimuon_mass_window_cut)

        h_peak = ((dimuon_mass_for_cutflow >= 115.0) & (dimuon_mass_for_cutflow < 135.0))
        h_sidebands1 =  ((dimuon_mass_for_cutflow >= 110.0) & (dimuon_mass_for_cutflow < 115.0)) | ((dimuon_mass_for_cutflow >= 135.0) & (dimuon_mass_for_cutflow < 150.0))
        h_sidebands2 =  ((dimuon_mass_for_cutflow >= 106.0) & (dimuon_mass_for_cutflow < 115.0)) | ((dimuon_mass_for_cutflow >= 135.0) & (dimuon_mass_for_cutflow < 150.0))

        self.selection.add("h_peak_115_135", h_peak)
        self.selection.add("h_sidebands_110_115_135_150", h_sidebands1)
        self.selection.add("h_sidebands_106_115_135_150", h_sidebands2)
        # ------------------- Cutflow dimuon mass window: END -----------------------

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

            # number of gen jets
            n_genjets = ak.num(gjets, axis=1)
            # number of gen jets with pT > 25 GeV and |eta| < 4.7
            n_genjets_pt25_eta47 = ak.sum((gjets.pt > 25) & (abs(gjets.eta) < 4.7), axis=1)
            # number of gen jets with pT > 30 GeV and |eta| < 4.7
            n_genjets_pt30_eta47 = ak.sum((gjets.pt > 30) & (abs(gjets.eta) < 4.7), axis=1)

        t11 = time.perf_counter()
        logger.info(f"[timing] GenJet variables time: {t11 - t10:.2f} seconds")

        self.prepare_jets(events, NanoAODv=NanoAODv)

        # ------------------------------------------------------------#
        # Apply JEC, get JEC and JER variations
        # ------------------------------------------------------------#
        # JER: https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetResolution
        # JES: https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC

        year = self.config["year"]
        jets = events.Jet

        PuppiMET = events.PuppiMET
        if self.config["switches"].get("do_jet_veto_maps_filterJets", False):
            logger.info("Applying jet veto maps!")
            jets, PuppiMET = self.compute_jet_veto_jetfilter(events, jets, PuppiMET)

        t12 = time.perf_counter()
        logger.info(f"[timing] prepare jets time: {t12 - t11:.2f} seconds")

        factory = None
        jet_default = ak.pad_none(jets, target=4) # save pre jec and jer Jet for comparison
        jet1_default = jet_default[:, 0]
        jet2_default = jet_default[:, 1]
        do_additional_jet_vars = self.config["switches"]["do_additional_jet_vars"]
        if do_additional_jet_vars:
            jet3_default = jet_default[:, 2]
            jet4_default = jet_default[:, 3]

        # -----------------------------------------------------
        # pre-selection for fatjets
        # add pre-selection for fatjets before saving the information: pT > 150 GeV and |eta| < 2.4 and pass the tight jet ID, dR(j, muons) > 0.8, FatJet_particleNetWithMass_WvsQCD > 0.75
        # Save the number of fat jets that passes this conditions
        # print first 5 events, fatjet pT
        # logger.warning(f"Number of fatjets (before selection): {nfatJets[:25].compute()}")
        # logger.warning(f"FatJet pT (before selection): {fatJets.pt[:25].compute()}")
        do_getFatJet_vars = self.config["switches"].get("do_getFatJet_vars", False)
        if do_getFatJet_vars:
            fatJets = events.FatJet
            nfatJets = ak.num(fatJets, axis=1)
            fatjet_selection = (
                (fatJets.pt > 150)
                & (abs(fatJets.eta) < 2.4)
                & (fatJets.particleNetWithMass_WvsQCD > 0.75) # W vs QCD discriminator
            )
            if hasattr(fatJets, "jetId"):
                fatjet_selection = fatjet_selection & (fatJets.jetId >= 2)
            else:
                logger.warning("FatJets have no jetId field!")
                tight_id, _ = custom_jet_id(fatJets)
                fatjet_selection = fatjet_selection & tight_id

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

        do_jec = self.config["switches"]["do_jec"]
        do_jec_unc = self.config["switches"]["do_jec_unc"]
        do_jer_unc = self.config["switches"]["do_jer_unc"]
        if do_jec:
            logger.info("doing JEC  (+ JER for MC)!")

            # 1) JES/JER variation labels you want to carry
            if do_jec_unc:
                variation_l = ["nominal"] + self.config["jec_parameters"]["jec_unc_to_consider"]
            else:
                variation_l = ["nominal"]

            # 2) Apply JES to jets (nominal + uncertainty sources)
            jets = do_jec_scale(jets, events, self.config, is_mc, dataset, uncs=variation_l)

            # store nominal snapshot names
            jets["mass_jec"] = jets.mass
            jets["pt_jec"] = jets.pt

            logger.debug(f"year: {year}, is_mc: {is_mc}, dataset: {dataset}")

            # 3) Apply JER smearing on MC
            # if "jer" in variation: # https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution#JER_Scaling_factors_and_Uncertai
            if is_mc and self.config["switches"]["jer_strat"]:
                logger.debug("Applying JER smearing!")
                jets = do_jer_smear(jets, self.config, events.event, nanoAOD_version=NanoAODv)
            else:
                logger.warning(f"==> Not applying JER smearing. is_mc: {is_mc}, jer_strat: {self.config['switches']['jer_strat']}")

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

            logger.info(f"year: {year}, dataset_yaml_file: {dataset_yaml_file}")
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
            logger.info(f"cross_section (after k-factor): {cross_section}")

            weights.add("xsec", weight=ak.ones_like(events.genWeight)*cross_section)
            weights.add("lumi", weight=ak.ones_like(events.genWeight)*integrated_lumi)
            # original initial weight end ----------------

            if do_pu_wgt:
                logger.debug("adding PU wgts!")
                weights.add("pu_wgt", weight=pu_wgts["nom"],weightUp=pu_wgts["up"],weightDown=pu_wgts["down"])
                # logger.info(f"pu_wgts['nom']: {ak.to_numpy(pu_wgts['nom'].compute())}")
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
        do_jec_unc = self.config["switches"]["do_jec_unc"]
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
            do_nnlops = self.config["switches"]["do_nnlops"] and ("ggh" in events.metadata["dataset"])
            if do_nnlops:
                logger.debug("doing NNLOPS!")
                nnlopsw = nnlops_weights(events.HTXS.Higgs_pt, events.HTXS.njets30, self.config, events.metadata["dataset"])
                # logger.info(f"nnlopsw: {ak.to_numpy(nnlopsw.compute())}")
                weights.add("nnlops", weight=nnlopsw)
            # moved nnlops reweighting outside of dak process-----------------

            # do mu SF start -------------------------------------
            logger.debug("doing musf!")
            if is_run2(year) or is_run3(year):
                muID, muIso, muTrig = add_muon_sfs_correctionlib(mu1, mu2, self.config["muSFFileList"], year)
            else:
                raise ValueError(f"Year {year} is not recognized as Run 2 or Run 3 year for muon SFs!")
            # -----------------------------
            # push into weights (same as run2)
            # -----------------------------
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
                self.config["switches"]["do_THU"]
                and ("nominal" in pt_variations)
                and ("vbf" in dataset)
                and ("dy" not in dataset)
                and ("stage1_1_fine_cat_pTjet30GeV" in events.HTXS.fields)
            )
            if do_thu:
                logger.info("doing THU weights!")
                add_stxs_variations(
                    events,
                    weights,
                    self.config,
                )

            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
            do_pdf = (
                self.config["switches"]["do_pdf"]
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
        # if year length is > 4, then it contains "pre" or "post" or "BPix"
        if len(year) > 4:
            """For the DNN training, we want to add year as one of the input variables.

            The expected format for `year` is a string like "2016pre", "2016post", "2017", "2018", "2022pre", or "2022post".

            If the year contains "pre", it is mapped to .0 (e.g., "2016pre" -> 2016.0).
            If the year contains "post", it is mapped to .5 (e.g., "2016post" -> 2016.5).
            If the year does not match these patterns, it is converted directly to float (e.g., "2017" -> 2017.0).
            If the format is unexpected, this may raise a ValueError.
            """
            logger.warning(f"Year format contains more than 4 characters: {year}")
            dnn_year = float(year[:4])
            if "pre" in year:
                dnn_year += 0.0
            else:
                dnn_year += 0.5
            logger.warning(f"Mapped year to dnn_year: {dnn_year}")
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
        if do_additional_jet_vars:
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

                "jet3_default_pt_nominal": jet3_default.pt,
                "jet3_default_eta_nominal": jet3_default.eta,
                "jet3_default_phi_nominal": jet3_default.phi,
                "jet3_default_mass_nominal": jet3_default.mass,

                "jet4_default_pt_nominal": jet4_default.pt,
                "jet4_default_eta_nominal": jet4_default.eta,
                "jet4_default_phi_nominal": jet4_default.phi,
                "jet4_default_mass_nominal": jet4_default.mass,
            })

        # --- Extra muon variables  ----------------------
        do_additional_vars = self.config["switches"]["do_additional_vars"]
        if do_additional_vars:
            _add_block(out_dict, {
                "PV_npvs": events.PV.npvs,
                "PV_npvsGood": events.PV.npvsGood,

                "PuppiMET_pt": PuppiMET.pt,
                "PuppiMET_phi": PuppiMET.phi,
                "PuppiMET_sumEt": PuppiMET.sumEt,
                "mu1_charge": mu1.charge,
                "mu2_charge": mu2.charge,
                "mu1_iso": mu1.pfRelIso04_all,
                "mu2_iso": mu2.pfRelIso04_all,
                "mu1_pt_over_mu2_pt": safe_ratio(mu1.pt, mu2.pt),
                "mu1_eta_over_mu2_eta": safe_ratio(abs(mu1.eta), abs(mu2.eta)),
                "mu1_pt_roch" : mu1.pt_roch,
                "mu1_pt_fsr" : mu1.pt_fsr,
                # "mu1_pt_gf" : mu1.pt_gf,
                "mu2_pt_roch" : mu2.pt_roch,
                "mu2_pt_fsr" : mu2.pt_fsr,
                # "mu2_pt_gf" : mu2.pt_gf,

                # Impact parameters / beamspot / PV
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
                # "pass_leading_pt" : pass_leading_pt,
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
        eta_absdiff = abs(eta_diff)
        eta_prod    = mu1.eta * mu2.eta

        abs_eta1    = abs(mu1.eta)
        abs_eta2    = abs(mu2.eta)
        abs_eta_sum = abs_eta1 + abs_eta2
        abs_eta_diff = abs(abs_eta1 - abs_eta2)
        abs_eta_min = ak.where(abs_eta1 < abs_eta2, abs_eta1, abs_eta2)
        abs_eta_max = ak.where(abs_eta1 > abs_eta2, abs_eta1, abs_eta2)

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

        if do_additional_vars:
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
                "mu12_eta_absdiff":  eta_absdiff,
                "mu12_eta_prod":     eta_prod,
                "mu12_absEta_sum":   abs_eta_sum,
                "mu12_absEta_diff":  abs_eta_diff,
                "mu12_absEta_min":   abs_eta_min,
                "mu12_absEta_max":   abs_eta_max,

                # isolation correlations
                "mu12_iso04_sum":      iso_sum,
                "mu12_iso04_diff":     iso_diff,
                "mu12_iso04_absdiff":  iso_absdiff,
                "mu12_iso04_prod":     iso_prod,
                "mu12_iso04_min":      iso_min,
                "mu12_iso04_max":      iso_max,
                "mu12_iso04_asym":     iso_asym,

                # impact parameters
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
                "gjj_dEta" : gjj_dEta,
                "gjj_dPhi" : gjj_dPhi,
                "gjj_dR" : gjj_dR,
            })

        t16 = time.perf_counter()
        logger.info(f"[timing] Fill muon and gjet variables time: {t16 - t15:.2f} seconds")
        # ------------------------------------------------------------#
        # HEMVeto study
        # ------------------------------------------------------------#
        if (self.config["switches"]["do_HemVeto"] and self.config["switches"]["do_HemVetoStudy"]):
            logger.info("Adding HemVeto_filter and is_HemRegion for HemVetoStudy!")
            HemVeto_filter = ak.to_packed(HemVeto_filter[event_filter==True]) # used for HemVetoStudy, doesn't compute if do_hemVetoStudy is False
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
                do_jec = do_jec,
                do_jecunc = do_jec_unc,
                do_jerunc = do_jer_unc,
                # event_match=event_match # debugging
                dnn_year=dnn_year,
                do_jet_horn_puid = self.config["switches"]["do_jet_horn_puid"]
            )

            _add_block(out_dict, jet_loop_dict)

        logger.debug(f"out_dict.keys() after jet loop: {out_dict.keys()}")

        t17 = time.perf_counter()
        logger.info(f"[timing] Jet pT variations time: {t17 - t16:.2f} seconds")

        # fill in the regions
        mass = dimuon.mass
        z_peak = ((mass >= 70.0) & (mass < 110.0))
        h_sidebands =  ((mass >= 110.0) & (mass < 115.0)) | ((mass >= 135.0) & (mass < 150.0))
        h_peak = ((mass >= 115.0) & (mass < 135.0))
        _add_block(out_dict, {
            "z_peak" : ak.fill_none(z_peak, value=False),
            "h_sidebands" : ak.fill_none(h_sidebands, value=False),
            "h_peak" : ak.fill_none(h_peak, value=False),
        })
        t18 = time.perf_counter()
        logger.info(f"[timing] various region (z-peak) fill time: {t18 - t17:.2f} seconds")

        # do zpt weight at the very end
        dataset = events.metadata["dataset"]
        do_zpt = ('dy' in dataset) and is_mc and self.config["switches"]["do_zpt"]
        if do_zpt:
            njets_reco = out_dict["njets_nominal"]
            njets_gen = n_genjets_pt30_eta47

            logger.info("=======================  apply zpt weights =======================")
            whichMethod = "function" # DNN or function or both
            if whichMethod == "function" or whichMethod == "both":
                # choose the config file
                if "MiNNLO" in dataset:
                    zpt_cfg = self.config["new_zpt_weights_file_MiNNLO"]
                else:
                    zpt_cfg = self.config["new_zpt_weights_file_aMCatNLO"]

                zpt_wgt_reco = getZptWgts_3region(dimuon.pt, njets_reco, "function", year, zpt_cfg)
                zpt_wgt_gen  = getZptWgts_3region(dimuon.pt, njets_gen,  "function", year, zpt_cfg)

                # --- save both to parquet
                _add_block(out_dict, {
                    "zpt_wgt_reco": zpt_wgt_reco,
                    "zpt_wgt_gen":  zpt_wgt_gen,
                })
            if (whichMethod == "DNN" or whichMethod == "both") and str(year) == "2024": #FIXME: year is temporarily here.
                # 1) choose model family (MiNNLO vs aMCatNLO)
                # model_paths = self.config["zpt_dnn_models_aMCatNLO"]  # dict with 0j/1j/2j
                model_paths_by_cats = {
                    "0j": "/depot/cms/private/users/shar1172/copperheadV2_main/try_zpt_dnn/Run3_nanoAODv12_10Feb_FilterJetsHorn30GeV/njet0/model_ts.pt",
                    "1j": "/depot/cms/private/users/shar1172/copperheadV2_main/try_zpt_dnn/Run3_nanoAODv12_10Feb_FilterJetsHorn30GeV/njet1/model_ts.pt",
                    "2j": "/depot/cms/private/users/shar1172/copperheadV2_main/try_zpt_dnn/Run3_nanoAODv12_10Feb_FilterJetsHorn30GeV/njet2p/model_ts.pt",
                }
                scalar_paths_by_cats = {
                    "0j": "/depot/cms/private/users/shar1172/copperheadV2_main/try_zpt_dnn/Run3_nanoAODv12_10Feb_FilterJetsHorn30GeV/njet0/scaler.npz",
                    "1j": "/depot/cms/private/users/shar1172/copperheadV2_main/try_zpt_dnn/Run3_nanoAODv12_10Feb_FilterJetsHorn30GeV/njet1/scaler.npz",
                    "2j": "/depot/cms/private/users/shar1172/copperheadV2_main/try_zpt_dnn/Run3_nanoAODv12_10Feb_FilterJetsHorn30GeV/njet2p/scaler.npz",
                }

                # 2) build features
                zpt_features_reco = {
                    "mu1_pt": mu1.pt,
                    "mu2_pt": mu2.pt,
                    "mu1_eta": mu1.eta,
                    "mu2_eta": mu2.eta,
                    "acoplanarity": acoplanarity,
                    "dimuon_pt": dimuon.pt,
                    "dimuon_rapidity": getRapidity(dimuon),
                }

                cfg_base = ZptDNNConfig(
                    model_path="DUMMY",
                    feature_names=[
                        "mu1_pt","mu2_pt","mu1_eta","mu2_eta",
                        "acoplanarity","dimuon_pt","dimuon_rapidity"
                    ],
                    output_mode="logit_to_odds",
                    device="cpu",
                    clip_weight_min=0.2,
                    clip_weight_max=5.0,
                )

                zpt_wgt_reco_dnn = eval_zpt_torchscript_by_njet(zpt_features_reco, njets_reco, cfg_base, model_paths_by_cats, scalar_paths_by_cats)
                zpt_wgt_gen_dnn = eval_zpt_torchscript_by_njet(zpt_features_reco, njets_gen, cfg_base, model_paths_by_cats, scalar_paths_by_cats)

                # --- save both to parquet
                _add_block(out_dict, {
                    "zpt_wgt_reco_dnn": zpt_wgt_reco_dnn,
                    "zpt_wgt_gen_dnn": zpt_wgt_gen_dnn,
                })

            _add_block(out_dict, {
                "zpt_njets_reco": njets_reco,
                "zpt_njets_gen": njets_gen,
            })
            # apply reco zpt weight to event weight
            weights.add("zpt_wgt", weight=zpt_wgt_reco)

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
        _add_block(out_dict, weight_dict)

        # ------------------------------------------------------------#
        # Cutflow
        if self.isCutflow:
            # FIXME: weights and weightsmodifier are availalbe starting coffea: 2025.3.0
            # Ensure all selections exist before calling cutflow
            # Add protection for the cutflow if the selection is not in the cutflow
            logger.info(f"selection: {self.selection}")
            all_required_selections = [
                "TotalEntries",
                "lumi_mask",
                "LHE_cut",
                "HLT_filter",
                "event_quality_flags",
                "PV_npvsGood",
                "muon_pT_roch",
                "muon_eta",
                "muon_id",
                "muon_isGlobal_or_Tracker",
                "muon_selection",
                "muon_iso",
                "nmuons",
                "mm_charge",
                "electron_veto",
                "HemVeto",
                "trigger_match",
                "leading_muon_pt",
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

            # Start with "TotalEntries" explicitly, if you want it in the table
            required_selections = []
            if "TotalEntries" in all_required_selections:
                required_selections.append("TotalEntries")

            # Add only those cuts that actually exist in PackedSelection, preserving order
            for cut in all_required_selections:
                if cut == "TotalEntries":
                    continue
                if cut in available_cuts:
                    required_selections.append(cut)

            logger.info(f"dynamic required_selections = {required_selections}")

            # Optional: warn about missing cuts
            missing = [cut for cut in all_required_selections
                    if cut not in available_cuts and cut != "TotalEntries"]
            if missing:
                logger.warning(f"These requested cuts are not defined and will be skipped: {missing}")

            self.cutflow = self.selection.cutflow(*required_selections)
            logger.info(f"cutflow: {self.cutflow}")
            logger.info(f"self.cutflow.logger.info(): {self.cutflow.print()}")

            # logger.info(f"wgtcutflow: {wgtcutflow.print()}")

            # self.nminusone = self.selection.nminusone(*required_selections)
            # logger.info(f"self.cutflow.logger.info(): {self.nminusone.print()}")
            # logger.info(f"self.cutflow.logger.info(): {self.cutflow.logger.info(weighted=False)}") # FIXME: weights and weightsmodifier are availalbe starting coffea: 2025.3.0
            # logger.info(f"self.cutflow.result(): {self.cutflow.result()}")

            # # --- FIXME: extra info for (unweighted + weighted + efficiencies)
            # # n_total = len(events)
            # n_total = int(dak.num(events, axis=0).compute())
            # w_all  = weights.weight()
            # mask_cum = dak.ones_like(w_all, dtype=bool)

            # rows = []
            # prev_n = n_total
            # prev_w = float(dak.sum(w_all).compute())

            # for name in required_selections:
            #     # boolean mask for this single cut
            #     mask_this = self.selection.all(name)
            #     # update cumulative mask
            #     mask_cum = mask_cum & mask_this

            #     n_pass = int(ak.sum(mask_cum))
            #     w_pass = float(ak.sum(w_all[mask_cum]))

            #     eff_step     = n_pass / prev_n if prev_n > 0 else 0.0
            #     eff_step_w   = w_pass / prev_w if prev_w > 0 else 0.0
            #     eff_cum      = n_pass / n_total if n_total > 0 else 0.0
            #     eff_cum_w    = w_pass / float(ak.sum(w_all)) if ak.sum(w_all) != 0 else 0.0

            #     rows.append(
            #         dict(
            #             cut=name,
            #             n_pass=n_pass,
            #             w_pass=w_pass,
            #             eff_step=eff_step,
            #             eff_step_w=eff_step_w,
            #             eff_cum=eff_cum,
            #             eff_cum_w=eff_cum_w,
            #         )
            #     )

            #     prev_n = n_pass
            #     prev_w = w_pass

            # self.cutflow_table = pd.DataFrame(rows)
            # logger.info("\n" + str(self.cutflow_table))
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
            correction_set = correctionlib.CorrectionSet.from_file(json_path)

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
        event_match = None,
        dnn_year = None,
        do_jet_horn_puid = False,
    ):
        logger.debug(f'variation: {variation}')
        is_mc = events.metadata["is_mc"]
        dataset = events.metadata["dataset"]
        year = self.config["year"]

        # print raw pt, jec pt and jer pt
        # logger.warning(f"jets.pt_raw: {jets.pt_raw[:1].compute()}, jets.pt: {jets.pt[:1].compute()}")

        if (not is_mc) and variation != "nominal":
            return {}

        # apply clean jet selection
        # AN-19-124 line 465: "Jets are also cleaned w.r.t. the selected muon candidates by requiring a geometrical separation of ∆R ( j, µ ) > 0.4"
        _, _, mu1_jet_dR = delta_r_V1(
            mu1[:, np.newaxis].eta_raw,
            jets.eta,
            mu1[:, np.newaxis].phi_raw,
            jets.phi,
        )
        matched_mu1_jet = mu1_jet_dR <= 0.4
        matched_mu1_jet = ak.fill_none(matched_mu1_jet, value=False)

        _, _, mu2_jet_dR = delta_r_V1(
            mu2[:, np.newaxis].eta_raw,
            jets.eta,
            mu2[:, np.newaxis].phi_raw,
            jets.phi,
        )
        matched_mu2_jet = mu2_jet_dR <= 0.4
        matched_mu2_jet = ak.fill_none(matched_mu2_jet, value=False)

        matched_mu_pass = matched_mu1_jet | matched_mu2_jet
        clean = ~matched_mu_pass
        clean = ak.fill_none(clean, value=True)

        # Select particular JEC variation
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

        # ------------------------------------------------------------#
        # Apply jetID and PUID
        # ------------------------------------------------------------#
        pass_jet_id = jet_id(jets, self.config, year)

        logger.debug(f"jet loop NanoAODv: {NanoAODv}")
        logger.debug(f"dnn_year: {dnn_year}")
        if self.config["switches"]["do_jet_PUID_wgt"]:
            logger.info("Applying jet PUID!")
            pass_jet_puid = jet_puid(jets, self.config)
        else:
            pass_jet_puid = ak.ones_like(pass_jet_id, dtype="bool")
        # ------------------------------------------------------------#
        # Select jets
        # ------------------------------------------------------------#
        # get QGL cut
        if NanoAODv == 9 and is_run2(year):
            jets["qgl"] = jets.qgl
        elif is_run2(year):
            # if qgl is not present, set it to -1.0
            jets["qgl"] = jets.qgl if hasattr(jets, "qgl") else ak.zeros_like(jets.pt) - 1.0
        elif is_run3(year):
            jets["btagPNetQvG"] = jets.btagPNetQvG
            jets["btagDeepFlavQG"] = jets.btagDeepFlavQG
        else:
            raise ValueError(f"Year {year} not recognized for jet QGL assignment!")

        jet_pt_cut = (jets.pt > self.config["jet_pt_cut"])

        # add additonal pT cut for the forward regions to reduce jet horn  ----------------------------------------------
        # source: https://indico.cern.ch/event/1434807/contributions/6040633/attachments/2893077/5071932/JERC%20meeting%2009_07.pdf
        jetHorn_region = abs(jets.eta) > 2.5
        jetHorn_pt_cut = (jets.pt > self.config["jet_pt_cut"]) # pt cut on jethorn doesn't change
        if do_jet_horn_puid: # For Run-2
            jetHorn_puid_cut = (get_puId(jets) >= 7) | (
                jets.pt >= 50
            )  # tight pu Id #FIXME: hardcoded puID
            jetHorn_cut = jetHorn_pt_cut & jetHorn_puid_cut
            jetHorn_PUID_cut = ak.ones_like(pass_jet_puid, dtype="bool") # default value is True
            # jetHorn_PUID_cut = ak.where(jetHorn_region, jetHorn_cut, jetHorn_PUID_cut)
            jetHorn_region, jetHorn_cut, jetHorn_PUID_cut = ak.broadcast_arrays(
                jetHorn_region, jetHorn_cut, jetHorn_PUID_cut
            )
            jetHorn_PUID_cut = ak.where(jetHorn_region, jetHorn_cut, jetHorn_PUID_cut)
        else:
            jetHorn_PUID_cut = ak.ones_like(pass_jet_puid, dtype="bool") # default value is True

        do_jet_horn_ptcut = self.config["switches"].get("do_jet_horn_ptcut", False)
        if do_jet_horn_ptcut:
            """ Remove jets in the jet horn region with pT < 50 GeV
                 horn region: 3.0 > abs(eta) > 2.5
            """
            logger.info(f"Applying additional jet pT cut of {do_jet_horn_ptcut} GeV for forward region (jet horn region)!")
            jetHorn_region = (abs(jets.eta) > 2.5) & (abs(jets.eta) < 3.0)
            jetHorn_pt_cut = (jets.pt > do_jet_horn_ptcut) # https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetMET#Run3_recommendations

            jetHorn_ptcut = ak.ones_like(pass_jet_id, dtype="bool") # default value is True
            jetHorn_region, jetHorn_ptcut = ak.broadcast_arrays(
                jetHorn_region, jetHorn_ptcut
            )
            jetHorn_ptcut = ak.where(jetHorn_region, jetHorn_pt_cut, jetHorn_ptcut)
        else:
            jetHorn_ptcut = ak.ones_like(pass_jet_id, dtype="bool") # default value is True

        # add additonal pT cut for the forward regions  ----------------------------------------------
        jet_selection = (
            jet_pt_cut
            & pass_jet_id
            & pass_jet_puid
            & clean
            & jetHorn_PUID_cut
            & jetHorn_ptcut
            & (abs(jets.eta) < self.config["jet_eta_cut"])
        )

        jets = jets[jet_selection] # INFO: this causes huuuuge memory overflow close to 100 GB. Without it, it goes to around 20 GB
        jets = ak.to_packed(jets)

        # apply jetpuid if not have done already
        if is_mc and (variation=="nominal") and is_run2(year): # INFO: Skip jet PUID for Run3 samples as they don't have puid yet
            logger.debug("Applying jet PUID scale factors and adding jetpuid_wgt!")
            jetpuid_weight = get_jetpuid_weights_eta_dependent(year, jets, self.config) # FIXME
            # now we add jetpuid_wgt
            # FIXME: we should get the weight for each jet and multiply them together.
            weights.add("jetpuid_wgt",
                    weight=jetpuid_weight,
            )
        else:
            logger.warning(f"Skipping jet PUID SFs for variation: {variation}, is_mc: {is_mc}, dnn_year: {dnn_year}")

        # jets = ak.where(jet_selection, jets, None)
        # muons = events.Muon
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
        do_additional_jet_vars = self.config["switches"]["do_additional_jet_vars"]
        if do_additional_jet_vars:
            jet3 = padded_jets[:,2]
            jet4 = padded_jets[:,3]

        jet_loop_out_dict = {}
        if variation == "nominal":
            for tag, (j1, j2) in [
                ("lead", pair_dict["lead"]),
                ("maxmjj", pair_dict["max_mjj"]),
                ("maxdeta", pair_dict["max_deta"]),
                ("maxmjj_deta25", pair_dict["mjj_deta"]),
            ]:
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
                })

            jet_loop_out_dict[f"vbf_maxmjj_deta25_hasPair_{variation}"] = pair_dict["has_mjj_deta"]

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
        do_additional_jet_vars = self.config["switches"]["do_additional_jet_vars"]
        if do_additional_jet_vars:
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

        jet_loop_out_dict.update({
            f"jet1_pt_{variation}": jet1.pt,
            f"jet1_eta_{variation}": jet1.eta,
            f"jet1_phi_{variation}": jet1.phi,
            # f"jet1_puId_{variation}": get_puId(jet1),
            # f"jet1_hasMatchedGenJet_{variation}": jet1.genJetIdx != -1,
            # -------------------------
            f"jet2_pt_{variation}": jet2.pt,
            f"jet2_eta_{variation}": jet2.eta,
            f"jet2_phi_{variation}": jet2.phi,
            # f"jet2_puId_{variation}": get_puId(jet2),
            # f"jet2_hasMatchedGenJet_{variation}": jet2.genJetIdx != -1,
            # -------------------------
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

        if is_run2(year):
            """Additional jet variables only for Run2"""
            jet_loop_out_dict.update({
                f"jet1_qgl_{variation}": jet1.qgl,  # FIXME: NanoAODv12 and NanoAODv15 have qgl as a field as AK4 jets are CHS for run-2, but not for run-3
                f"jet2_qgl_{variation}": jet2.qgl,
            })
            if do_additional_jet_vars:
                jet_loop_out_dict.update({
                    f"jet3_qgl_{variation}": jet3.qgl,
                    f"jet4_qgl_{variation}": jet4.qgl,
                })
        elif is_run3(year):
            """Additional jet variables only for Run3"""
            jet_loop_out_dict.update({
                f"jet1_btagPNetQvG_{variation}": jet1.btagPNetQvG,
                f"jet2_btagPNetQvG_{variation}": jet2.btagPNetQvG,
            })
            if do_additional_jet_vars:
                jet_loop_out_dict.update({
                    f"jet3_btagPNetQvG_{variation}": jet3.btagPNetQvG,
                    f"jet4_btagPNetQvG_{variation}": jet4.btagPNetQvG,
                })

        if do_additional_jet_vars:
            jet_loop_out_dict.update(
                {
                    f"jet1_rapidity_{variation}": jet1_rapidity,  # max rel err: 0.7394
                    f"jet1_btagDeepFlavQG_{variation}": jet1.btagDeepFlavQG,
                    f"jet1_mass_{variation}": jet1.mass,
                    f"jet1_area_{variation}": jet1.area,
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
                    f"jet2_rapidity_{variation}": jet2_rapidity,  # max rel err: 0.781
                    f"jet2_btagPNetQvG_{variation}": jet2.btagPNetQvG,
                    f"jet2_btagDeepFlavQG_{variation}": jet2.btagDeepFlavQG,
                    f"jet2_mass_{variation}": jet2.mass,
                    f"jet2_area_{variation}": jet2.area,


                    f"jet3_pt_{variation}": jet3.pt,
                    f"jet3_eta_{variation}": jet3.eta,
                    f"jet3_rapidity_{variation}": jet3_rapidity,
                    f"jet3_phi_{variation}": jet3.phi,
                    f"jet3_btagDeepFlavQG_{variation}": jet3.btagDeepFlavQG,
                    f"jet3_mass_{variation}": jet3.mass,
                    f"jet3_area_{variation}": jet3.area,
                    # -------------------------
                    f"jet4_pt_{variation}": jet4.pt,
                    f"jet4_eta_{variation}": jet4.eta,
                    f"jet4_rapidity_{variation}": jet4_rapidity,
                    f"jet4_phi_{variation}": jet4.phi,
                    f"jet4_btagDeepFlavQG_{variation}": jet4.btagDeepFlavQG,
                    f"jet4_mass_{variation}": jet4.mass,
                    f"jet4_area_{variation}": jet4.area,
                }
            )

        do_additional_vars = self.config["switches"]["do_additional_vars"]
        if do_additional_vars:
            if hasattr(jets, "jetId"):
                jet_loop_out_dict.update({
                    f"jet1_jetId_{variation}": jet1.jetId,
                    f"jet2_jetId_{variation}": jet2.jetId,
                })
                if do_additional_jet_vars:
                    jet_loop_out_dict.update({
                        f"jet3_jetId_{variation}": jet3.jetId,
                        f"jet4_jetId_{variation}": jet4.jetId,
                    })
            if do_additional_jet_vars:
                jet_loop_out_dict.update({
                    f"jet3_puId_{variation}": get_puId(jet3),
                    f"jet4_puId_{variation}": get_puId(jet4),
                })

        # ------------------------------------------------------------------
        # Add additional Jet NanoAOD variables for leading 4 jets
        # (only if the branches exist in this NanoAOD)
        # ------------------------------------------------------------------
        extra_jet_loop_dict = {}

        # --- DeepJet (DeepFlav) taggers ---
        if "btagDeepFlavB" in jets.fields:
            extra_jet_loop_dict.update({
                f"jet1_btagDeepFlavB_{variation}":   jet1.btagDeepFlavB,
                # f"jet1_btagDeepFlavCvB_{variation}": jet1.btagDeepFlavCvB,
                # f"jet1_btagDeepFlavCvL_{variation}": jet1.btagDeepFlavCvL,
                # f"jet1_btagDeepFlavQG_{variation}":  jet1.btagDeepFlavQG,
                f"jet2_btagDeepFlavB_{variation}":   jet2.btagDeepFlavB,
                # f"jet2_btagDeepFlavCvB_{variation}": jet2.btagDeepFlavCvB,
                # f"jet2_btagDeepFlavCvL_{variation}": jet2.btagDeepFlavCvL,
                # f"jet2_btagDeepFlavQG_{variation}":  jet2.btagDeepFlavQG,
                # f"jet3_btagDeepFlavCvB_{variation}": jet3.btagDeepFlavCvB,
                # f"jet3_btagDeepFlavCvL_{variation}": jet3.btagDeepFlavCvL,
                # f"jet3_btagDeepFlavQG_{variation}":  jet3.btagDeepFlavQG,
                # f"jet4_btagDeepFlavCvB_{variation}": jet4.btagDeepFlavCvB,
                # f"jet4_btagDeepFlavCvL_{variation}": jet4.btagDeepFlavCvL,
                # f"jet4_btagDeepFlavQG_{variation}":  jet4.btagDeepFlavQG,
            })
            if do_additional_jet_vars:
                extra_jet_loop_dict.update({
                    f"jet3_btagDeepFlavB_{variation}":   jet3.btagDeepFlavB,
                    f"jet4_btagDeepFlavB_{variation}":   jet4.btagDeepFlavB,
                })

        # --- ParticleNet b-tag family ---
        # if "btagPNetB" in jets.fields:
        #     extra_jet_loop_dict.update({
        #         f"jet1_btagPNetB_{variation}":       jet1.btagPNetB,
        #         f"jet1_btagPNetCvB_{variation}":     jet1.btagPNetCvB,
        #         f"jet1_btagPNetCvL_{variation}":     jet1.btagPNetCvL,
        #         f"jet1_btagPNetTauVJet_{variation}": jet1.btagPNetTauVJet,
        #         f"jet2_btagPNetB_{variation}":       jet2.btagPNetB,
        #         f"jet2_btagPNetCvB_{variation}":     jet2.btagPNetCvB,
        #         f"jet2_btagPNetCvL_{variation}":     jet2.btagPNetCvL,
        #         f"jet2_btagPNetTauVJet_{variation}": jet2.btagPNetTauVJet,
        #         # f"jet3_btagPNetB_{variation}":       jet3.btagPNetB,
        #         # f"jet3_btagPNetCvB_{variation}":     jet3.btagPNetCvB,
        #         # f"jet3_btagPNetCvL_{variation}":     jet3.btagPNetCvL,
        #         # f"jet3_btagPNetTauVJet_{variation}": jet3.btagPNetTauVJet,
        #         # f"jet4_btagPNetB_{variation}":       jet4.btagPNetB,
        #         # f"jet4_btagPNetCvB_{variation}":     jet4.btagPNetCvB,
        #         # f"jet4_btagPNetCvL_{variation}":     jet4.btagPNetCvL,
        #         # f"jet4_btagPNetTauVJet_{variation}": jet4.btagPNetTauVJet,
        #     })

        # --- RobustParTAK4 taggers ---
        # if "btagRobustParTAK4B" in jets.fields:
        #     extra_jet_loop_dict.update({
        #         f"jet1_btagRobustParTAK4B_{variation}":  jet1.btagRobustParTAK4B,
        #         f"jet2_btagRobustParTAK4B_{variation}":  jet2.btagRobustParTAK4B,
        #         # f"jet3_btagRobustParTAK4B_{variation}":  jet3.btagRobustParTAK4B,
        #         # f"jet4_btagRobustParTAK4B_{variation}":  jet4.btagRobustParTAK4B,
        #     })

        # --- Energy fractions ---
        if "chEmEF" in jets.fields:
            extra_jet_loop_dict.update({
                f"jet1_chEmEF_{variation}": jet1.chEmEF,
                f"jet1_chHEF_{variation}":  jet1.chHEF,
                f"jet1_neEmEF_{variation}": jet1.neEmEF,
                f"jet1_neHEF_{variation}":  jet1.neHEF,
                f"jet1_muEF_{variation}":   jet1.muEF,
                f"jet2_chEmEF_{variation}": jet2.chEmEF,
                f"jet2_chHEF_{variation}":  jet2.chHEF,
                f"jet2_neEmEF_{variation}": jet2.neEmEF,
                f"jet2_neHEF_{variation}":  jet2.neHEF,
                f"jet2_muEF_{variation}":   jet2.muEF,
                # f"jet3_chEmEF_{variation}": jet3.chEmEF,
                # f"jet3_chHEF_{variation}":  jet3.chHEF,
                # f"jet3_neEmEF_{variation}": jet3.neEmEF,
                # f"jet3_neHEF_{variation}":  jet3.neHEF,
                # f"jet3_muEF_{variation}":   jet3.muEF,
                # f"jet4_chEmEF_{variation}": jet4.chEmEF,
                # f"jet4_chHEF_{variation}":  jet4.chHEF,
                # f"jet4_neEmEF_{variation}": jet4.neEmEF,
                # f"jet4_neHEF_{variation}":  jet4.neHEF,
                # f"jet4_muEF_{variation}":   jet4.muEF,
            })
        if "chMultiplicity" in jets.fields:
            extra_jet_loop_dict.update({
                f"jet1_chMultiplicity_{variation}": jet1.chMultiplicity,
                f"jet2_chMultiplicity_{variation}": jet2.chMultiplicity,
                f"jet1_neMultiplicity_{variation}": jet1.neMultiplicity,
                f"jet2_neMultiplicity_{variation}": jet2.neMultiplicity,
                # f"jet3_chMultiplicity_{variation}": jet3.chMultiplicity,
                # f"jet4_chMultiplicity_{variation}": jet4.chMultiplicity,
                # f"jet3_neMultiplicity_{variation}": jet3.neMultiplicity,
                # f"jet4_neMultiplicity_{variation}": jet4.neMultiplicity,
            })

        # # --- Multiplicities & constituents ---
        if "nConstituents" in jets.fields:
            extra_jet_loop_dict.update({
                f"jet1_nConstituents_{variation}": jet1.nConstituents,
                f"jet1_nElectrons_{variation}":    jet1.nElectrons,
                f"jet1_nMuons_{variation}":        jet1.nMuons,
                f"jet1_nSVs_{variation}":          jet1.nSVs,
                f"jet2_nConstituents_{variation}": jet2.nConstituents,
                f"jet2_nElectrons_{variation}":    jet2.nElectrons,
                f"jet2_nMuons_{variation}":        jet2.nMuons,
                f"jet2_nSVs_{variation}":          jet2.nSVs,
                # f"jet3_nConstituents_{variation}": jet3.nConstituents,
                # f"jet3_nElectrons_{variation}":    jet3.nElectrons,
                # f"jet3_nMuons_{variation}":        jet3.nMuons,
                # f"jet3_nSVs_{variation}":          jet3.nSVs,
                # f"jet4_nConstituents_{variation}": jet4.nConstituents,
                # f"jet4_nElectrons_{variation}":    jet4.nElectrons,
                # f"jet4_nMuons_{variation}":        jet4.nMuons,
                # f"jet4_nSVs_{variation}":          jet4.nSVs,
            })

        # # --- Jet–electron & jet–muon indices, SV indices ---
        # if "electronIdx1" in jets.fields:
        #     extra_jet_loop_dict.update({
        #         f"jet1_electronIdx1_{variation}": jet1.electronIdx1,
        #         f"jet1_electronIdx2_{variation}": jet1.electronIdx2,
        #         f"jet2_electronIdx1_{variation}": jet2.electronIdx1,
        #         f"jet2_electronIdx2_{variation}": jet2.electronIdx2,
        #         # f"jet3_electronIdx1_{variation}": jet3.electronIdx1,
        #         # f"jet3_electronIdx2_{variation}": jet3.electronIdx2,
        #         # f"jet4_electronIdx1_{variation}": jet4.electronIdx1,
        #         # f"jet4_electronIdx2_{variation}": jet4.electronIdx2,
        #     })

        # if "muonIdx1" in jets.fields:
        #     extra_jet_loop_dict.update({
        #         f"jet1_muonIdx1_{variation}": jet1.muonIdx1,
        #         f"jet1_muonIdx2_{variation}": jet1.muonIdx2,
        #         f"jet2_muonIdx1_{variation}": jet2.muonIdx1,
        #         f"jet2_muonIdx2_{variation}": jet2.muonIdx2,
        #         # f"jet3_muonIdx1_{variation}": jet3.muonIdx1,
        #         # f"jet3_muonIdx2_{variation}": jet3.muonIdx2,
        #         # f"jet4_muonIdx1_{variation}": jet4.muonIdx1,
        #         # f"jet4_muonIdx2_{variation}": jet4.muonIdx2,
        #     })

        # if "svIdx1" in jets.fields:
        #     extra_jet_loop_dict.update({
        #         f"jet1_svIdx1_{variation}": jet1.svIdx1,
        #         f"jet1_svIdx2_{variation}": jet1.svIdx2,
        #         f"jet2_svIdx1_{variation}": jet2.svIdx1,
        #         f"jet2_svIdx2_{variation}": jet2.svIdx2,
        #         # f"jet3_svIdx1_{variation}": jet3.svIdx1,
        #         # f"jet3_svIdx2_{variation}": jet3.svIdx2,
        #         # f"jet4_svIdx1_{variation}": jet4.svIdx1,
        #         # f"jet4_svIdx2_{variation}": jet4.svIdx2,
        #     })

        # # --- Flavour and gen matching ---
        # if "genJetIdx" in jets.fields:
        #     extra_jet_loop_dict.update({
        #         f"jet1_genJetIdx_{variation}": jet1.genJetIdx,
        #         f"jet2_genJetIdx_{variation}": jet2.genJetIdx,
        #         # f"jet3_genJetIdx_{variation}": jet3.genJetIdx,
        #         # f"jet4_genJetIdx_{variation}": jet4.genJetIdx,
        #     })

        if "hadronFlavour" in jets.fields:
            extra_jet_loop_dict.update({
                f"jet1_hadronFlavour_{variation}": jet1.hadronFlavour,
                f"jet2_hadronFlavour_{variation}": jet2.hadronFlavour,
                # f"jet3_hadronFlavour_{variation}": jet3.hadronFlavour,
                # f"jet4_hadronFlavour_{variation}": jet4.hadronFlavour,
            })

        if "partonFlavour" in jets.fields:
            extra_jet_loop_dict.update({
                f"jet1_partonFlavour_{variation}": jet1.partonFlavour,
                f"jet2_partonFlavour_{variation}": jet2.partonFlavour,
                # f"jet3_partonFlavour_{variation}": jet3.partonFlavour,
                # f"jet4_partonFlavour_{variation}": jet4.partonFlavour,
            })

        # --- HF noise variables ---
        if "hfcentralEtaStripSize" in jets.fields:
            extra_jet_loop_dict.update({
                f"jet1_hfcentralEtaStripSize_{variation}":   jet1.hfcentralEtaStripSize,
                f"jet1_hfadjacentEtaStripsSize_{variation}": jet1.hfadjacentEtaStripsSize,
                f"jet1_hfsigmaEtaEta_{variation}":           jet1.hfsigmaEtaEta,
                f"jet1_hfsigmaPhiPhi_{variation}":           jet1.hfsigmaPhiPhi,
                f"jet2_hfcentralEtaStripSize_{variation}":   jet2.hfcentralEtaStripSize,
                f"jet2_hfadjacentEtaStripsSize_{variation}": jet2.hfadjacentEtaStripsSize,
                f"jet2_hfsigmaEtaEta_{variation}":           jet2.hfsigmaEtaEta,
                f"jet2_hfsigmaPhiPhi_{variation}":           jet2.hfsigmaPhiPhi,
                # f"jet3_hfcentralEtaStripSize_{variation}":   jet3.hfcentralEtaStripSize,
                # f"jet3_hfadjacentEtaStripsSize_{variation}": jet3.hfadjacentEtaStripsSize,
                # f"jet3_hfsigmaEtaEta_{variation}":           jet3.hfsigmaEtaEta,
                # f"jet3_hfsigmaPhiPhi_{variation}":           jet3.hfsigmaPhiPhi,
                # f"jet4_hfcentralEtaStripSize_{variation}":   jet4.hfcentralEtaStripSize,
                # f"jet4_hfadjacentEtaStripsSize_{variation}": jet4.hfadjacentEtaStripsSize,
                # f"jet4_hfsigmaEtaEta_{variation}":           jet4.hfsigmaEtaEta,
                # f"jet4_hfsigmaPhiPhi_{variation}":           jet4.hfsigmaPhiPhi,
            })

        # # --- Muon subtraction factor ---
        # if "muonSubtrFactor" in jets.fields:
        #     extra_jet_loop_dict.update({
        #         f"jet1_muonSubtrFactor_{variation}": jet1.muonSubtrFactor,
        #         f"jet2_muonSubtrFactor_{variation}": jet2.muonSubtrFactor,
        #         # f"jet3_muonSubtrFactor_{variation}": jet3.muonSubtrFactor,
        #         # f"jet4_muonSubtrFactor_{variation}": jet4.muonSubtrFactor,
        #     })

        # # --- Raw factor (1 - JEC factor) ---
        # if "rawFactor" in jets.fields:
        #     extra_jet_loop_dict.update({
        #         f"jet1_rawFactor_{variation}": jet1.rawFactor,
        #         f"jet2_rawFactor_{variation}": jet2.rawFactor,
        #         # f"jet3_rawFactor_{variation}": jet3.rawFactor,
        #         # f"jet4_rawFactor_{variation}": jet4.rawFactor,
        #     })

        # Merge into main jet_loop_out_dict
        jet_loop_out_dict.update(extra_jet_loop_dict)

        # if is_mc and (variation == "nominal"):
        #     nominal_dict = {
        #         f"jet1_pt_gen_{variation}" : jet1.pt_gen,
        #         f"jet2_pt_gen_{variation}" : jet2.pt_gen,
        #     }
        #     jet_loop_out_dict.update(nominal_dict)

        # if (variation == "nominal"):
        #     nominal_dict = {
        #         f"jet1_pt_raw_{variation}" : jet1.pt_raw,
        #         f"jet2_pt_raw_{variation}" : jet2.pt_raw,
        #         f"jet1_mass_raw_{variation}" : jet1.mass_raw,
        #         f"jet2_mass_raw_{variation}" : jet2.mass_raw,
        #         f"jet1_mass_jec_{variation}" : jet1.mass_jec,
        #         f"jet2_mass_jec_{variation}" : jet2.mass_jec,
        #         f"jet1_pt_jec_{variation}" : jet1.pt_jec,
        #         f"jet2_pt_jec_{variation}" : jet2.pt_jec,
        #     }
        # jet_loop_out_dict.update(nominal_dict)

        # ------------------------------------------------------------#
        # Fill soft activity jet variables
        # ------------------------------------------------------------#

        # Effect of changes in jet acceptance should be negligible,
        # no need to calcluate this for each jet pT variation

        # sj_dict = {}
        sj_dict_HIG19006 = {}
        cutouts = [2,5]
        nmuons = ak.num(events.Muon, axis=1) # FIXME (I think it should be selected muons)
        # PLEASE NOTE: SoftJET variables are all from Nominal variation despite variation names
        for cutout in cutouts:
            # sj_out = fill_softjets(events, jets, mu1, mu2, nmuons, cutout) # obtain nominal softjet values
            # sj_out = { # add variation even thought it's always nominal
            #     key+"_"+variation : val \
            #     for key, val in sj_out.items()
            # }
            # sj_dict.update(sj_out)

            sj_out_HIG19006 = fill_softjets_HIG19006(events, jets, mu1, mu2, nmuons, cutout) # obtain nominal softjet values
            sj_out_HIG19006 = { # add variation even thought it's always nominal
                key+"_"+variation : val \
                for key, val in sj_out_HIG19006.items()
            }
            sj_dict_HIG19006.update(sj_out_HIG19006)

        # logger.debug(f"sj_dict.keys(): {sj_dict.keys()}")
        # jet_loop_out_dict.update(sj_dict)
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
        if is_mc and (variation == "nominal") and (self.config["switches"]["do_qgl_wgt"]):
            # --- QGL weights  start --- #
            isHerwig = "herwig" in dataset
            logger.debug("adding QGL weights!")

            # keep dims start -------------------------------------
            # qgl_wgts = qgl_weights_keepDim(jet1, jet2, njets, isHerwig)
            qgl_wgts = qgl_weights_V2(jets, self.config, isHerwig, dnn_year)
            # keep dims end -------------------------------------
            weights.add("qgl_wgt",
                        weight=qgl_wgts["nom"],
                        weightUp=qgl_wgts["up"],
                        weightDown=qgl_wgts["down"]
            )
            # --- QGL weights  end --- #

        if is_mc and (variation == "nominal") and (self.config["switches"]["do_btag_wgt"]):
            # --- Btag weights  start--- #
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
            # --- Btag weights end --- #

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
        if is_run3(year): # Run3: Different btagging taggers and WPs
            btagLoose_filter = (jets.btagDeepFlavB > self.config["btag_loose_wp"]) & (abs(jets.eta) < 2.5)
            btagMedium_filter = (jets.btagDeepFlavB > self.config["btag_medium_wp"]) & (abs(jets.eta) < 2.5)
        else: # UL
            # NOTE: maybe keep the nBtagLoose and nBtagMedium deepbFlavB as a separate variable for quick testing
            # btagLoose_filter = (jets.btagDeepFlavB > self.config["btag_loose_wp"]) & (abs(jets.eta) < 2.5)
            # btagMedium_filter = (jets.btagDeepFlavB > self.config["btag_medium_wp"]) & (abs(jets.eta) < 2.5)
            if hasattr(jets, "btagDeepB"):
                btagLoose_filter = (jets.btagDeepB > self.config["btag_loose_wp"]) & (abs(jets.eta) < 2.5)
                btagMedium_filter = (jets.btagDeepB > self.config["btag_medium_wp"]) & (abs(jets.eta) < 2.5)
            elif hasattr(jets, "btagDeepFlavB"):
                # FIXME: Currently the working point is used what was defined for DeepB, should be updated for DeepFlavB
                btagLoose_filter = (jets.btagDeepFlavB > self.config["btag_loose_wp"]) & (abs(jets.eta) < 2.5)
                btagMedium_filter = (jets.btagDeepFlavB > self.config["btag_medium_wp"]) & (abs(jets.eta) < 2.5)

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
