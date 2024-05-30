import coffea.processor as processor
from coffea.lookup_tools import extractor
import awkward as ak
import numpy as np
from typing import Union, TypeVar, Tuple
import correctionlib
from corrections.rochester import apply_roccor
from corrections.fsr_recovery import fsr_recovery, fsr_recoveryV1
from corrections.geofit import apply_geofit
from corrections.jet import get_jec_factories, jet_id, jet_puid, fill_softjets
from corrections.weight import Weights
from corrections.evaluator import pu_evaluator, nnlops_weights, musf_evaluator, get_musf_lookup, lhe_weights, stxs_lookups, add_stxs_variations, add_pdf_variations, qgl_weights, qgl_weights_eager, qgl_weights_keepDim, btag_weights_json, btag_weights_jsonKeepDim, get_jetpuid_weights
import json
from coffea.lumi_tools import LumiMask
import pandas as pd # just for debugging
import dask_awkward as dak
import dask
from coffea.analysis_tools import Weights
import copy
from coffea.nanoevents.methods import vector


coffea_nanoevent = TypeVar('coffea_nanoevent') 
ak_array = TypeVar('ak_array')

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
    cos_theta_cs = nominator/demoninator
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
    beam_vec =  ak.zip(
        {
            "x": ak.zeros_like(dimuon_pz),
            "y": ak.zeros_like(dimuon_pz),
            "z": beam_vec_z,
        },
        with_name="ThreeVector",
        behavior=vector.behavior,
    )
    # apply cross product. note x,y,z of dimuon refers to its momentum, NOT its location
    # mu.px == mu.x, mu.py == mu.y and so on
    R_T = beam_vec.cross(dimuon)
    # make it a unit vector
    R_T = R_T.unit
    Q_T = dimuon
    Q_coeff = ( ((dimuon_mass*dimuon_mass + (dimuon_pt*dimuon_pt)))**(0.5) )/dimuon_mass
    delta_T_dot_R_T = (mu_neg.px-mu_pos.px)*R_T.x + (mu_neg.py-mu_pos.py)*R_T.y 
    delta_R_term = delta_T_dot_R_T
    delta_T_dot_Q_T = (mu_neg.px-mu_pos.px)*Q_T.px + (mu_neg.py-mu_pos.py)*Q_T.py
    # delta_T_dot_Q_T = -delta_T_dot_Q_T
    delta_Q_term = delta_T_dot_Q_T
    delta_Q_term = delta_Q_term / dimuon_pt # normalize since Q_T should techincally be a unit vector
    phi_cs = np.arctan2(Q_coeff*delta_R_term, delta_Q_term)
    # phi_cs = np.arctan(Q_coeff*delta_R_term/ delta_Q_term)
    return phi_cs
    

class EventProcessor(processor.ProcessorABC):
    # def __init__(self, config_path: str,**kwargs):
    def __init__(self, config: dict, test_mode=False, **kwargs):
        """
        TODO: replace all of these with self.config dict variable which is taken from a
        pre-made json file
        """
        self.config = config

        self.test = test_mode# False
        dict_update = {
            # "hlt" :["IsoMu24"],
            "do_trigger_match" : False, # False
            "do_roccor" : True,# False
            "do_fsr" : True,
            "do_geofit" : True, # False
            "do_beamConstraint": True, # if True, override do_geofit
            "year" : "2018",
            "do_nnlops" : True,
            "do_pdf" : True,
        }
        self.config.update(dict_update)
        

        # --- Evaluator
        extractor_instance = extractor()
        year = self.config["year"]
        # Z-pT reweighting 
        zpt_filename = self.config["zpt_weights_file"]
        extractor_instance.add_weight_sets([f"* * {zpt_filename}"])
        if "2016" in year:
            self.zpt_path = "zpt_weights/2016_value"
        else:
            self.zpt_path = "zpt_weights_all"
        # print(f"self.zpt_path: {self.zpt_path}")
        # Calibration of event-by-event mass resolution
        for mode in ["Data", "MC"]:
            if "2016" in year:
                yearUL = "2016"
            else:
                yearUL=year #Work around before there are seperate new files for pre and postVFP
            label = f"res_calib_{mode}_{yearUL}"
            path = self.config["res_calib_path"]
            file_path = f"{path}/{label}.root"
            extractor_instance.add_weight_sets([f"{label} {label} {file_path}"])

        # PU ID weights
        jetpuid_filename = self.config["jetpuid_sf_file"]
        extractor_instance.add_weight_sets([f"* * {jetpuid_filename}"])
        
        extractor_instance.finalize()
        self.evaluator = extractor_instance.make_evaluator()

    def process(self, events: coffea_nanoevent):
        """
        TODO: Once you're done with testing and validation, do LHE cut after HLT and trigger match event filtering to save computation
        """


        if self.test:
            print(f"copperhead2 events muon pt: {ak.to_dataframe(events.Muon.pt)}")
            print(f"copperhead2 type(events): {type(events)}")
            print(f"copperhead2 events.metadata: {events.metadata}")
        """
        Apply LHE cuts for DY sample stitching
        Basically remove events that has dilepton mass between 100 and 200 GeV
        """
        event_filter = ak.ones_like(events.HLT.IsoMu24) # 1D boolean array to be used to filter out bad events
        dataset = events.metadata['dataset']
        print(f"dataset: {dataset}")
        print(f"events.metadata: {events.metadata}")
        NanoAODv = events.metadata['NanoAODv']
        print(f"NanoAODv: {NanoAODv}")
        # LHE cut original start -----------------------------------------------------------------------------
        if dataset == 'dy_M-50': # if dy_M-50, apply LHE cut
            print("doing dy_M-50 LHE cut!")
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
            # print(f"LHE_filter: {LHE_filter.compute()}")
            LHE_filter = ak.fill_none(LHE_filter, value=False) 
            LHE_filter = (LHE_filter== False) # we want True to indicate that we want to keep the event
            # print(f"copperhead2 EventProcessor LHE_filter[32]: \n{ak.to_numpy(LHE_filter[32])}")

            event_filter = event_filter & LHE_filter
        # LHE cut original end -----------------------------------------------------------------------------
        
        
# --------------------------------------------------------        
        # if self.config["do_trigger_match"]:
        #     """
        #     Apply trigger matching. We take the two leading pT reco muons and try to have at least one of the muons
        #     to be matched with the trigger object that fired our HLT. If none of the muons did it, then we reject the 
        #     event. This operation is computationally expensive, so perhaps worth considering not implementing it if 
        #     it has neglible impact
            
        #     TODO: The impact this operation has onto the statistics is supposedly very low, but I have to check that
        #     """
        #     isoMu_filterbit = 2
        #     mu_id = 13
        #     pt_threshold = 24 
        #     dr_threshold = 0.1
        #     IsoMu24_muons = (events.TrigObj.id == mu_id) &  \
        #             ((events.TrigObj.filterBits & isoMu_filterbit) == isoMu_filterbit) & \
        #             (events.TrigObj.pt > pt_threshold)
        #     #check the first two leading muons match any of the HLT trigger objs. if neither match, reject event
        #     padded_muons = ak.pad_none(events.Muon, 2) # pad in case we have only one muon or zero in an event
        #     # padded_muons = ak.pad_none(events.Muon, 4)
        #     # print(f"copperhead2 EventProcessor padded_muons: \n {padded_muons}")
        #     mu1 = padded_muons[:,0]
        #     mu2 = padded_muons[:,1]
        #     mu1_match = (mu1.delta_r(events.TrigObj[IsoMu24_muons]) < dr_threshold) & \
        #         (mu1.pt > pt_threshold)
        #     mu1_match = ak.sum(mu1_match, axis=1)
        #     mu1_match = ak.fill_none(mu1_match, value=False)
        #     mu2_match = (mu2.delta_r(events.TrigObj[IsoMu24_muons]) < dr_threshold) & \
        #         (mu2.pt > pt_threshold)
        #     mu2_match =  ak.sum(mu2_match, axis=1)
        #     mu2_match = ak.fill_none(mu2_match, value=False)

        #     trigger_match = (mu1_match >0) | (mu2_match > 0)
# --------------------------------------------------------            

            
            # if self.test:
            #     print(f"copperhead2 EventProcessor mu1_match: \n {mu1_match}")
            #     print(f"copperhead2 EventProcessor mu2_match: \n {mu2_match}")
            #     print(f"copperhead2 EventProcessor trigger_match: \n {trigger_match}")
            #     print(f"copperhead2 EventProcessor events.HLT.IsoMu24 and trigger_match mismatch count: \n {ak.sum(events.HLT.IsoMu24 != trigger_match)}")
                
            # print(f"copperhead2 EventProcessor trigger_match sum: {ak.sum(trigger_match)}")
            # print(f"copperhead2 EventProcessor trigger_non_match: \n {ak.sum(trigger_non_match)}")
            # print(f"copperhead2 EventProcessor trigger_non_match ratio: {ak.sum(trigger_non_match)/ak.sum(trigger_match)}")
            

# just reading test start --------------------------------------------------------------------------        
        #     event_filter = event_filter & trigger_match

            
        # # Apply HLT to both Data and MC. NOTE: this would probably be superfluous if you already do trigger matching
        for HLT_str in self.config["hlt"]:
            event_filter = event_filter & events.HLT[HLT_str]


        # ------------------------------------------------------------#
        # Skimming end, filter out events and prepare for pre-selection
        # Edit: NVM; doing it this stage breaks fsr recovery
        # ------------------------------------------------------------#
        # events = events[event_filter]
        # event_filter = ak.ones_like(events.HLT.IsoMu24)
        
        if events.metadata["is_mc"]:
            lumi_mask = ak.ones_like(event_filter)

        
        else:
            lumi_info = LumiMask(self.config["lumimask"])
            lumi_mask = lumi_info(events.run, events.luminosityBlock)
            if self.test:
                print(f"copperhead2 EventProcessor lumi_mask: \n {ak.to_numpy(lumi_mask)}")

        # turn off pu weights test start ---------------------------------
        # obtain PU reweighting b4 event filtering, and apply it after we finalize event_filter
        if events.metadata["is_mc"]:
            pu_wgts = pu_evaluator(
                        self.config,
                        events.Pileup.nTrueInt,
                        onTheSpot=False # use locally saved true PU dist
                        # onTheSpot=True
                )
        # turn off pu weights test end ---------------------------------
       
        # # Save raw variables before computing any corrections
        # # rochester and geofit corrects pt only, but fsr_recovery changes all vals below
        # # original -------------------------------------------------------------------
        # events["Muon", "pt_raw"] = events.Muon.pt
        # events["Muon", "eta_raw"] = events.Muon.eta
        # events["Muon", "phi_raw"] = events.Muon.phi
        # events["Muon", "pfRelIso04_all_raw"] = events.Muon.pfRelIso04_all
        # # original end ---------------------------------------------------------------

        # attempt at fixing fsr issue -------------------------------------------------------------------
        events["Muon", "pt_raw"] = ak.ones_like(events.Muon.pt) * events.Muon.pt
        events["Muon", "eta_raw"] = ak.ones_like(events.Muon.eta) * events.Muon.eta
        events["Muon", "phi_raw"] = ak.ones_like(events.Muon.phi) * events.Muon.phi
        events["Muon", "pfRelIso04_all_raw"] = ak.ones_like(events.Muon.pfRelIso04_all) * events.Muon.pfRelIso04_all
        # attempt at fixing fsr issue end ---------------------------------------------------------------
    
        
        # # --------------------------------------------------------
        # # # Apply Rochester correction
        if self.config["do_roccor"]:
            print("doing rochester!")
            apply_roccor(events, self.config["roccor_file"], events.metadata["is_mc"])
            events["Muon", "pt"] = events.Muon.pt_roch
        # FSR recovery
        if self.config["do_fsr"]:
            print(f"doing fsr!")
            # applied_fsr = fsr_recovery(events)
            applied_fsr = fsr_recoveryV1(events)# testing for pt_raw inconsistency
            events["Muon", "pt"] = events.Muon.pt_fsr
            events["Muon", "eta"] = events.Muon.eta_fsr
            events["Muon", "phi"] = events.Muon.phi_fsr
            events["Muon", "pfRelIso04_all"] = events.Muon.iso_fsr
        else:
            # if no fsr, just copy 'pt' to 'pt_fsr'
            applied_fsr = (ak.zeros_like(events.Muon.pt) != 0) # boolean array of Falses
            events["Muon", "pt_fsr"] = events.Muon.pt
            # events["Muon", "eta_fsr"] = events.Muon.eta
            # events["Muon", "phi_fsr"] = events.Muon.phi
            # events["Muon", "iso_fsr"] = events.Muon.pfRelIso04_all
        
        # Note temporary soln to copperheadV1 pt_raw being overwritten with fsr... and possibly with geofit
        # which is important for the muon selection later on
        # events["Muon", "pt_raw"] = events.Muon.pt
        # events["Muon", "eta_raw"] = events.Muon.eta
        # events["Muon", "phi_raw"] = events.Muon.phi
        
        #-----------------------------------------------------------------
        
        # apply Beam constraint or geofit or nothing if neither
        if self.config["do_beamConstraint"] and ("bsConstrainedChi2" in events.Muon.fields): # beamConstraint overrides geofit
            print(f"doing beam constraint!")
            # print(f"events.Muon.fields: {events.Muon.fields}")
            BSConstraint_mask = (
                (events.Muon.bsConstrainedChi2 <30)
            )
            BSConstraint_mask = ak.fill_none(BSConstraint_mask, False)
            # comment off (~applied_fsr) cut for now 
            # BSConstraint_mask = BSConstraint_mask & (~applied_fsr) # apply BSContraint on non FSR muons
            events["Muon", "pt"] = ak.where(BSConstraint_mask, events.Muon.bsConstrainedPt, events.Muon.pt)
            events["Muon", "ptErr"] = ak.where(BSConstraint_mask, events.Muon.bsConstrainedPtErr, events.Muon.ptErr)
        else:
            if self.config["do_geofit"] and ("dxybs" in events.Muon.fields):
                print(f"doing geofit!")
                gf_filter, gf_pt_corr = apply_geofit(events, self.config["year"], ~applied_fsr)
                events["Muon", "pt"] = events.Muon.pt_gf
            else: 
                print(f"doing neither beam constraint nor geofit!")
                pass

        ## Note temporary soln to copperheadV1 pt_raw being overwritten with fsr... and possibly with geofit
        # # which is important for the muon selection later on
        # events["Muon", "pt_raw"] = events.Muon.pt
        # events["Muon", "eta_raw"] = events.Muon.eta
        # events["Muon", "phi_raw"] = events.Muon.phi
        # # # --------------------------------------------------------
        
        # # --------------------------------------------------------#
        # # Select muons that pass pT, eta, isolation cuts,
        # # muon ID and quality flags
        # # Select events with 2 good muons, no electrons,
        # # passing quality cuts and at least one good PV
        # # --------------------------------------------------------#

        # Apply event quality flags
        evnt_qual_flg_selection = ak.ones_like(event_filter)
        for evt_qual_flg in self.config["event_flags"]:
            evnt_qual_flg_selection = evnt_qual_flg_selection & events.Flag[evt_qual_flg]

        
        # muon_id = "mediumId" if "medium" in self.config["muon_id"] else "looseId"
        # print(f"copperhead2 EventProcessor muon_id: {muon_id}")
        # original muon selection ------------------------------------------------
        muon_selection = (
            (events.Muon.pt_raw > self.config["muon_pt_cut"])
            # (events.Muon.pt > self.config["muon_pt_cut"]) # testing
            & (abs(events.Muon.eta_raw) < self.config["muon_eta_cut"])
            & (events.Muon.pfRelIso04_all < self.config["muon_iso_cut"])
            # & events.Muon[muon_id]
            & events.Muon[self.config["muon_id"]]
        )
        # original muon selection end ------------------------------------------------

        
        # # testing muon selection ------------------------------------------------
        # muon_selection = (
        #     (ak.values_astype(events.Muon.pt_raw,  "float64") > self.config["muon_pt_cut"])
        #     & (ak.values_astype(abs(events.Muon.eta_raw), "float64")  < self.config["muon_eta_cut"])
        #     & (ak.values_astype(events.Muon.pfRelIso04_all, "float64")  < self.config["muon_iso_cut"])
        #     # & events.Muon[muon_id]
        #     & events.Muon[self.config["muon_id"]]
        # )
        # # testing muon selection end ------------------------------------------------

        muons = events.Muon[muon_selection]
        # muons = ak.to_packed(events.Muon[muon_selection])

        # count muons that pass the muon selection
        nmuons = ak.num(muons, axis=1)
        # Find opposite-sign muons
        mm_charge = ak.prod(muons.charge, axis=1)
        
        electron_id = self.config[f"electron_id_v{NanoAODv}"]
        print(f"electron_id: {electron_id}")
        # Veto events with good quality electrons; VBF and ggH categories need zero electrons
        electron_selection = (
            (events.Electron.pt > self.config["electron_pt_cut"])
            & (abs(events.Electron.eta) < self.config["electron_eta_cut"])
            & events.Electron[electron_id]
        )
        
        # some temporary testing code start -----------------------------------------
        # if doing_ebeMassCalib:
        #     """
        #     if obtaining results for ebe mass Calibration calculation, we want electron_veto to be turned off
        #     """
        #     electron_veto = ak.ones_like(event_filter)
        # else:
        #     electron_veto = (ak.num(events.Electron[electron_selection], axis=1) == 0) 
        # some temporary testing code end -----------------------------------------
        
        electron_veto = (ak.num(events.Electron[electron_selection], axis=1) == 0) 

        
        event_filter = (
                event_filter
                & lumi_mask
                & (evnt_qual_flg_selection > 0)
                & (nmuons == 2)
                & (mm_charge == -1)
                & electron_veto 
                & (events.PV.npvsGood > 0) # number of good primary vertex cut

        )

        

        # --------------------------------------------------------#
        # Select events with muons passing leading pT cut
        # --------------------------------------------------------#

        # original start---------------------------------------------------------------
        # # Events where there is at least one muon passing
        # # leading muon pT cut
        # pass_leading_pt = muons.pt_raw > self.config["muon_leading_pt"]
        # print(f'type self.config["muon_leading_pt"] : {type(self.config["muon_leading_pt"])}')
        # print(f'type muons.pt_raw : {ak.type(muons.pt_raw.compute())}')
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
        # NOTE: if you want to keep this method, (which I don't btw since the original
        # code above is conceptually more correct at this moment), you should optimize
        # this code, bc this was just something I put together for quick testing
        muons_padded = ak.pad_none(muons, target=2)
        sorted_args = ak.argsort(muons_padded.pt, ascending=False) # leadinig pt is ordered by pt
        muons_sorted = (muons_padded[sorted_args])
        mu1 = muons_sorted[:,0]
        pass_leading_pt = mu1.pt_raw > self.config["muon_leading_pt"]
        pass_leading_pt = ak.fill_none(pass_leading_pt, value=False) 


        event_filter = event_filter & pass_leading_pt
        # test end -----------------------------------------------------------------------

        
        
        # calculate sum of gen weight b4 skimming off bad events
        is_mc = events.metadata["is_mc"]
        if is_mc:
            events["genWeight"] = ak.values_astype(events.genWeight, "float64") # increase precision or it gives you slightly different value for summing them up
            # small files testing start ------------------------------------------
            # sumWeights = ak.sum(events.genWeight, axis=0) # for testing
            # print(f"sumWeights: {(sumWeights.compute())}") # for testing
            # small files testing end ------------------------------------------
            # original start ----------------------------------------------
            sumWeights = events.metadata['sumGenWgts']
            print(f"sumWeights: {(sumWeights)}")
            # original end -------------------------------------------------
        # skim off bad events onto events and other related variables
        # # original -----------------------------------------------
        # events = events[event_filter==True]
        # muons = muons[event_filter==True]
        # nmuons = nmuons[event_filter==True]
        # applied_fsr = applied_fsr[event_filter==True]
        # if is_mc:
        #     for variation in pu_wgts.keys():
        #         pu_wgts[variation] = pu_wgts[variation][event_filter==True]
        # pass_leading_pt = pass_leading_pt[event_filter==True]
        # # original end -----------------------------------------------


        # to_packed testing -----------------------------------------------
        events = events[event_filter==True]
        muons = muons[event_filter==True]
        nmuons = ak.to_packed(nmuons[event_filter==True])
        applied_fsr = ak.to_packed(applied_fsr[event_filter==True])

       
        # turn off pu weights test start ---------------------------------
        if is_mc:
            for variation in pu_wgts.keys():
                pu_wgts[variation] = ak.to_packed(pu_wgts[variation][event_filter==True])
        pass_leading_pt = ak.to_packed(pass_leading_pt[event_filter==True])

        
            
        
       
        
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
        dimuon = mu1+mu2
        # fill in pd Dataframe as placeholder. Should be fine since we don't need jagged arrays
        # dimuon_ebe_mass_res = self.get_mass_resolution(events, test_mode=self.test)
        
        dimuon_ebe_mass_res = self.get_mass_resolution(dimuon, mu1, mu2, is_mc, test_mode=self.test)
        rel_dimuon_ebe_mass_res = dimuon_ebe_mass_res/dimuon.mass
        dimuon_cos_theta_cs, dimuon_phi_cs = cs_variables(mu1,mu2)
        dimuon_cos_theta_eta, dimuon_phi_eta = etaFrame_variables(mu1,mu2)

        # skip validation for genjets for now -----------------------------------------------
        # #fill genjets
        
        if events.metadata["is_mc"]:
            #fill gen jets for VBF filter on postprocess
            gjets = events.GenJet
            gleptons = events.GenPart[
                (abs(events.GenPart.pdgId) == 13)
                | (abs(events.GenPart.pdgId) == 11)
                | (abs(events.GenPart.pdgId) == 15)
            ]
            gl_pair = ak.cartesian({"jet": gjets, "lepton": gleptons}, axis=1, nested=True)
            dr_gl = gl_pair["jet"].delta_r(gl_pair["lepton"])
            isolated = ak.all((dr_gl > 0.3), axis=-1) # this also returns true if there's no leptons near the gjet
            
            # I suppose we assume there's at least two jets
            padded_iso_gjet = ak.pad_none(
                ak.to_packed(gjets[isolated]),
                target=2,
                clip=True
            ) # pad with none val to ensure that events have at least two columns each event
            sorted_args = ak.argsort(padded_iso_gjet.pt, ascending=False) # leading pt is ordered by pt
            gjets_sorted = (padded_iso_gjet[sorted_args])
            gjet1 = gjets_sorted[:,0]
            gjet2 = gjets_sorted[:,1] 
            gjj = gjet1 + gjet2
            
            gjj_dEta = abs(gjet1.eta - gjet2.eta)
            gjj_dPhi = abs(gjet1.delta_phi(gjet2))
            gjj_dR = gjet1.delta_r(gjet2)


        self.prepare_jets(events, NanoAODv=NanoAODv)


        # ------------------------------------------------------------#
        # Apply JEC, get JEC and JER variations
        # ------------------------------------------------------------#
        year = self.config["year"]
        jets = events.Jet
        self.jec_factories_mc, self.jec_factories_data = get_jec_factories(
            self.config["jec_parameters"], 
            year
        )   
        
        do_jec = True # True       
        # do_jecunc = self.config["do_jecunc"]
        # do_jerunc = self.config["do_jerunc"]
        #testing 
        do_jecunc = False
        do_jerunc = False
        is_mc = events.metadata["is_mc"]
        # cache = events.caches[0]
        factory = None
        if do_jec:
            if is_mc:
                factory = self.jec_factories_mc["jec"]
            else:
                for run in self.config["jec_parameters"]["runs"]:
                    # print(f"run: {run}")
                    if run in dataset:
                        factory = self.jec_factories_data[run]
                if factory == None:
                    print("JEC factory not recognized!")
                    raise ValueError
                
            print("do jec!")
            jets = factory.build(jets)

        else:
            jets["mass_jec"] = jets.mass
            jets["pt_jec"] = jets.pt

        
        # # TODO: only consider nuisances that are defined in run parameters
        # # Compute JEC uncertainties
        # if events.metadata["is_mc"] and do_jecunc:
        #     jets = self.jec_factories_mc["junc"].build(jets, lazy_cache=cache)
    
        # # # Compute JER uncertainties
        # # if events.metadata["is_mc"] and do_jerunc:
        # #     jets = self.jec_factories_mc["jer"].build(jets, lazy_cache=cache)
        
        # # # TODO: JER nuisances

        



        

        # # ------------------------------------------------------------#
        # # Apply genweights, PU weights
        # # and L1 prefiring weights
        # # ------------------------------------------------------------#
        weights = Weights(None) # none for dask awkward
        is_mc = events.metadata["is_mc"]
        if is_mc:
            weights.add("genWeight", weight=events.genWeight)
            # original initial weight start ----------------
            weights.add("genWeight_normalization", weight=ak.ones_like(events.genWeight)/sumWeights)
            #temporary lhe filter start -----------------
            # M105to160normalizedWeight = M105to160normalizedWeight*events.genWeight/sumWeights
            #temporary lhe filter end -----------------
            cross_section = self.config["cross_sections"][dataset]
            integrated_lumi = self.config["integrated_lumis"]
            weights.add("xsec", weight=ak.ones_like(events.genWeight)*cross_section)
            weights.add("lumi", weight=ak.ones_like(events.genWeight)*integrated_lumi)
            # original initial weight end ----------------
            # hard code to match lumi weight of valerie's code start ----
            # weights.add("lumi_weight", weight=ak.ones_like(events.genWeight)*0.012550352440399929)
            # hard code to match lumi weight of valerie's code end ----
            
            # turn off pu weights test start ---------------------------------
            print("adding PU wgts!")
            weights.add("pu", weight=pu_wgts["nom"],weightUp=pu_wgts["up"],weightDown=pu_wgts["down"])
            # turn off pu weights test end ---------------------------------
            # L1 prefiring weights
            if self.config["do_l1prefiring_wgts"] and ("L1PreFiringWeight" in events.fields):
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
        
          

        
        # ------------------------------------------------------------#
        # Calculate other event weights
        # ------------------------------------------------------------#
        pt_variations = (
            ["nominal"]
            # + jec_pars["jec_variations"]
            # + jec_pars["jer_variations"]
        )
        if events.metadata["is_mc"]:
            # moved nnlops reweighting outside of dak process and to run_stage1-----------------
            do_nnlops = self.config["do_nnlops"] and ("ggh" in events.metadata["dataset"])
            if do_nnlops:
                print("doing NNLOPS!")
                nnlopsw = nnlops_weights(events.HTXS.Higgs_pt, events.HTXS.njets30, self.config, events.metadata["dataset"])
                weights.add("nnlops", weight=nnlopsw)
                # print(f"nnlopsw: \n  {ak.to_numpy(nnlopsw.compute())}")
        #     # else:
        #     #     weights.add_weight("nnlops", how="dummy")
        #     # print(f'copperheadV1 weights.df nnlops: \n {weights.df.to_string()}')
            # moved nnlops reweighting outside of dak process-----------------
            

            #do mu SF start -------------------------------------
            print("doing musf!")
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
            #do mu SF end -------------------------------------

            
            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
            do_lhe = (
                ("LHEScaleWeight" in events.fields)
                and ("LHEPdfWeight" in events.fields)
                and ("nominal" in pt_variations)
            )
            if do_lhe:
                print("doing LHE!")
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
            # do_thu = (
            #     ("vbf" in dataset)
            #     and ("dy" not in dataset)
            #     and ("nominal" in pt_variations)
            #     and ("stage1_1_fine_cat_pTjet30GeV" in events.HTXS.fields)
            # )
            do_thu = False
            if do_thu:
                print("doing THU!")
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
                print("doing pdf!")
                # add_pdf_variations(events, self.weight_collection, self.config, dataset)
                pdf_vars = add_pdf_variations(events, self.config, dataset)
                weights.add("pdf_2rms", 
                    weight=ak.ones_like(pdf_vars["up"]),
                    weightUp=pdf_vars["up"],
                    weightDown=pdf_vars["down"]
                )

# just reading test end
# just reading part 2 start -------------------------        
        # ------------------------------------------------------------#
        # Fill Muon variables and gjet variables
        # ------------------------------------------------------------#
        out_dict = {
            "event" : events.event,
            "mu1_pt" : mu1.pt,
            "mu2_pt" : mu2.pt,
            "mu1_eta" : mu1.eta,
            "mu2_eta" : mu2.eta,
            "mu1_phi" : mu1.phi,
            "mu2_phi" : mu2.phi,
            "mu1_charge" : mu1.charge,
            "mu2_charge" : mu2.charge,
            "mu1_iso" : mu1.pfRelIso04_all,
            "mu2_iso" : mu2.pfRelIso04_all,
            "nmuons" : nmuons,
            "dimuon_mass" : dimuon.mass,
            "dimuon_pt" : dimuon.pt,
            "dimuon_eta" : dimuon.eta,
            "dimuon_phi" : dimuon.phi,
            "dimuon_dEta" : dimuon_dEta,
            "dimuon_dPhi" : dimuon_dPhi,
            "dimuon_dR" : dimuon_dR,
            "dimuon_ebe_mass_res" : dimuon_ebe_mass_res,
            "dimuon_cos_theta_cs" : dimuon_cos_theta_cs,
            "dimuon_phi_cs" : dimuon_phi_cs,
            "dimuon_cos_theta_eta" : dimuon_cos_theta_eta,
            "dimuon_phi_eta" : dimuon_phi_eta,
            "mu1_pt_raw" : mu1.pt_raw,
            "mu2_pt_raw" : mu2.pt_raw,
            "mu1_pt_fsr" : mu1.pt_fsr,
            "mu2_pt_fsr" : mu2.pt_fsr,
            "pass_leading_pt" : pass_leading_pt,
            # temporary test start ------------------------------------
            # "M105to160normalizedWeight" : M105to160normalizedWeight,
            # temporary test end ------------------------------------
        }
        if is_mc:
            mc_dict = {
                "HTXS_Higgs_pt" : events.HTXS.Higgs_pt, # for nnlops weight for ggH signal sample
                "HTXS_njets30" : events.HTXS.njets30, # for nnlops weight for ggH signal sample
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
        
        # ------------------------------------------------------------#
        # Loop over JEC variations and fill jet variables
        # ------------------------------------------------------------#
        
        for variation in pt_variations:
            jet_loop_dict = self.jet_loop(
                events, 
                jets,
                dimuon,
                mu1,
                mu2,
                variation,
                weights,
                do_jec = do_jec,
                do_jecunc = do_jecunc,
                do_jerunc = do_jerunc,
            )
                    
        out_dict.update(jet_loop_dict) 
        # print(f"out_dict.keys() after jet loop: {out_dict.keys()}")
        
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
            
        out_dict.update(region_dict) 

       
        # b4 we do any filtering, we obtain the sum of gen weights for normalization
        # events["genWeight"] = ak.values_astype(events.genWeight, "float64") # increase precision or it gives you slightly different value for summing them up
        

        njets = out_dict["njets"]
        # print(f"njets: {ak.to_numpy(njets.compute())}")

        # do zpt weight at the very end
        dataset = events.metadata["dataset"]
        is_mc = events.metadata["is_mc"]
        do_zpt = ('dy' in dataset) and is_mc
        # do_zpt = False
        if do_zpt:
            print("doing zpt weight!")
            zpt_weight =\
                     self.evaluator[self.zpt_path](dimuon.pt, njets)
            # print(f"zpt_weight: {zpt_weight.compute()}")
            # weights.add("zpt_wgt", weight=zpt_weight) # leave it outsie like btag
        

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
        print(f"weight statistics: {weights.weightStatistics.keys()}")
        weights = weights.weight()
        if "btag_wgt" in out_dict.keys():
            print("adding btag wgts!")
            weights = weights*out_dict["btag_wgt"]
        if do_zpt:
            weights = weights*zpt_weight
        # add in weights
        # weights = ak.to_packed(weights)
        out_dict.update({"weights" : weights})

    
        return out_dict
        
    def postprocess(self, accumulator):
        """
        Arbitrary postprocess function that's required to run the processor
        """
        pass

    
    def get_mass_resolution(self, dimuon, mu1,mu2, is_mc:bool, test_mode=False):
        # Returns absolute mass resolution!
        # mu1 = events.Muon[:,0]
        # mu2 = events.Muon[:,1]
        # muon_E = (mu1+mu2).mass /2
        muon_E = dimuon.mass /2
        dpt1 = (mu1.ptErr / mu1.pt) * muon_E
        dpt2 = (mu2.ptErr / mu2.pt) * muon_E
        if test_mode:
            print(f"muons mass_resolution dpt1: {dpt1}")
        if "2016" in self.config["year"]:
            yearUL = "2016"
        else:
            yearUL = self.config["year"] #Work around before there are seperate new files for pre and postVFP
        if is_mc:
            label = f"res_calib_MC_{yearUL}"
        else:
            label = f"res_calib_Data_{yearUL}"
        calibration =  self.evaluator[label]( # this is a coffea.dense_lookup instance
            mu1.pt, 
            abs(mu1.eta), 
            abs(mu2.eta) # calibration depends on year, data/mc, pt, and eta region for each muon (ie, BB, BO, OB, etc)
        )
    
        # return ((dpt1 * dpt1 + dpt2 * dpt2)**0.5) * calibration
        return ((dpt1 * dpt1 + dpt2 * dpt2)**0.5) # turning calibration off for calibration factor recalculation
    
    def prepare_jets(self, events, NanoAODv=9): # analogous to add_jec_variables function in boosted higgs
        # Initialize missing fields (needed for JEC)
        print(f"prepare jets NanoAODv: {NanoAODv}")
        events["Jet", "pt_raw"] = (1 - events.Jet.rawFactor) * events.Jet.pt
        events["Jet", "mass_raw"] = (1 - events.Jet.rawFactor) * events.Jet.mass
        if NanoAODv >= 12:
            fixedGridRhoFastjetAll = events.Rho.fixedGridRhoFastjetAll
        else: # if v9
            fixedGridRhoFastjetAll = events.fixedGridRhoFastjetAll
        events["Jet", "rho"] = ak.broadcast_arrays(fixedGridRhoFastjetAll, events.Jet.pt)[0]
    
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
        # do_jec = True, #False
        # do_jecunc = False,
        # do_jerunc = False,
        do_jec = False, 
        do_jecunc = False,
        do_jerunc = False,
    ):
        is_mc = events.metadata["is_mc"]
        if (not is_mc) and variation != "nominal":
            return
        # variables = pd.DataFrame(index=output.index)
        # print(f"variables: {variables}")

        """
        keep the below code for records, but idk if this is important or something I can get rid of 
        jet_columns = [
            "pt",
            "eta",
            "phi",
            "jetId",
            "qgl",
            "puId",
            "mass",
            "btagDeepFlavB",
            "has_matched_gen",
        ]
        if "puId17" in events.Jet.fields:
            jet_columns += ["puId17"]
        
        if is_mc:
            jet_columns += ["partonFlavour", "hadronFlavour"]
        if variation == "nominal":
            # pt_jec and mass_jec are same as pt and mass
            # if do_jec:
                # jet_columns += ["pt_jec", "mass_jec"] 
            if is_mc and do_jerunc:
                jet_columns += ["pt_orig", "mass_orig"]
        """
        # Find jets that have selected muons within dR<0.4 from them

        # matched_mu_pt = jets.matched_muons.pt_fsr
        matched_mu_pt = jets.matched_muons.pt_fsr if "pt_fsr" in jets.matched_muons.fields else jets.matched_muons.pt
        matched_mu_iso = jets.matched_muons.pfRelIso04_all
        matched_mu_id = jets.matched_muons[self.config["muon_id"]]
        matched_mu_pass = (
            (matched_mu_pt > self.config["muon_pt_cut"])
            & (matched_mu_iso < self.config["muon_iso_cut"])
            & matched_mu_id
        )
        if self.test:
            print(f"jet loop matched_mu_pass b4 : {matched_mu_pass}")
        matched_mu_pass = ak.sum(matched_mu_pass, axis=2) > 0 # there's at least one matched mu that passes the muon selection
        clean = ~(ak.fill_none(matched_mu_pass, value=False))
        
        # skip selecting particular JEC variation for now
        # # Select particular JEC variation
        # if "_up" in variation:
        #     unc_name = "JES_" + variation.replace("_up", "")
        #     if unc_name not in jets.fields:
        #         return
        #     jets = jets[unc_name]["up"][jet_columns]
        # elif "_down" in variation:
        #     unc_name = "JES_" + variation.replace("_down", "")
        #     if unc_name not in jets.fields:
        #         return
        #     jets = jets[unc_name]["down"][jet_columns]
        # else:
        #     jets = jets[jet_columns]


    #         # We use JER corrections only for systematics, so we shouldn't
    #         # update the kinematics. Use original values,
    #         # unless JEC were applied.
        """
        if is_mc and do_jerunc and not do_jec: # NOTE: I don't think this is needed anymore since jets variable is the original events.Jet if do_jec==False
            events["Jet","pt"] = jets["pt_orig"]
            events["Jet","mass"] = jets["mass_orig"]
            jets = events.Jet
        """

        # # ------------------------------------------------------------#
        # # Apply jetID and PUID
        # # ------------------------------------------------------------#

        pass_jet_id = jet_id(jets, self.config)

        
        # jet PUID disabled as it's not applicable for jets with JEC and pt> 50,
        # as stated in https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetIDUL
        pass_jet_puid = jet_puid(jets, self.config)
        # print(f"sum pass_jet_puid: {ak.to_numpy(ak.sum(pass_jet_puid, axis=1).compute())}")
        # print(f"sum pass_jet_puid total: {ak.to_numpy(ak.sum(pass_jet_puid, axis=None).compute())}")
        # print(f"pass_jet_puid: {(pass_jet_puid.compute())}")
        # # only apply jet puid to jets with pt < 50, else, pass
        # bool_filter = ak.ones_like(pass_jet_puid, dtype="bool")
        # pass_jet_puid = ak.where((jets.pt <50),pass_jet_puid, bool_filter)
        # # Jet PUID scale factors, which also takes pt < 50 into account within the function
        year = self.config["year"]
        if is_mc:  
            print("doing jet puid weights!")
            jet_puid_opt = self.config["jet_puid"]
            pt_name = "pt"
            puId = jets.puId
            jetpuid_weight = get_jetpuid_weights(
                self.evaluator, year, jets, pt_name,
                jet_puid_opt, pass_jet_puid
            )
            weights.add("jetpuid_wgt", 
                    weight=jetpuid_weight,
            )
        #     print(f"puid_weight: {puid_weight}")
        # #     weights.add("puid_wgt", weight=puid_weight)
        # # #     weights.add_weight('puid_wgt', puid_weight)

        # ------------------------------------------------------------#
        # Select jets
        # ------------------------------------------------------------#
        # apply HEM Veto, written in "HEM effect in 2018" appendix K of the main long AN
        HEMVeto = ak.ones_like(clean) == 1 # 1D array saying True
        if year == "2018":
            HEMVeto_filter = (
                (jets.pt >= 20.0)
                & (jets.eta >= -3.0)
                & (jets.eta <= -1.3)
                & (jets.phi >= -1.57)
                & (jets.phi <= -0.87)
            )
            false_arr = ak.ones_like(HEMVeto) < 0
            HEMVeto = ak.where(HEMVeto_filter, false_arr, HEMVeto)
        # print(f"HEMVeto : {HEMVeto}")
        
        # original jet_selection-----------------------------------------------
        jet_selection = (
            pass_jet_id
            & pass_jet_puid
            & (jets.qgl > -2)
            & clean
            & (jets.pt > self.config["jet_pt_cut"])
            & (abs(jets.eta) < self.config["jet_eta_cut"])
            & HEMVeto
        )
        # original jet_selection end ----------------------------------------------


        # jets = jets[jet_selection] # this causes huuuuge memory overflow close to 100 GB. Without it, it goes to around 20 GB

        jets = ak.to_packed(jets[jet_selection]) 
        # jets = jets[jet_selection]

        
        
        # print(f"jets after selection: {jets}")
        # print(f"jets._meta after selection: {str(jets._meta.compute())}")
        # print(f"jet_selection._meta: {str(jet_selection._meta.compute())}")
        # print(f"jets._meta after selection: {repr(jets._meta)}")
        # print(f"jet_selection._meta: {repr(jet_selection._meta)}")
        # print(f"dak.necessary_columns(jets.pt) after selection: {dak.necessary_columns(jets.pt)}")
        # 
        
        # jets = ak.where(jet_selection, jets, None)
        # muons = events.Muon 
        njets = ak.num(jets, axis=1)
        
        # ------------------------------------------------------------#
        # Fill jet-related variables
        # ------------------------------------------------------------#

        
        # original start ----------------------------------------
        # padded_jets = ak.pad_none(jets, target=2) 
        # # # jet1 = padded_jets[:,0]
        # # # jet2 = padded_jets[:,1]
        # # jet_flip = padded_jets.pt[:,0] < padded_jets.pt[:,1]  
        # # jet_flip = ak.fill_none(jet_flip, value=False)
        # # # take the subleading muon values if that now has higher pt after corrections
        # # jet1 = ak.where(jet_flip, padded_jets[:,1], padded_jets[:,0])
        # # jet2 = ak.where(jet_flip, padded_jets[:,0], padded_jets[:,1])
        # sorted_args = ak.argsort(padded_jets.pt, ascending=False)
        # sorted_jets = (padded_jets[sorted_args])
        # jet1 = sorted_jets[:,0]
        # jet2 = sorted_jets[:,1]
        # original end ----------------------------------------
        # test start ----------------------------------------
        
        sorted_args = ak.argsort(jets.pt, ascending=False)
        sorted_jets = (jets[sorted_args])
        jets = sorted_jets
        paddedSorted_jets = ak.pad_none(sorted_jets, target=2) 
        jet1 = paddedSorted_jets[:,0]
        jet2 = paddedSorted_jets[:,1]
        # test end ----------------------------------------

        
        dijet = jet1+jet2
        # print(f"dijet: {dijet}")
        jj_dEta = abs(jet1.eta - jet2.eta)
        jj_dPhi = abs(jet1.delta_phi(jet2))
        # dimuon = muons[:,0] + muons[:,1]
        mmj1_dEta = abs(dimuon.eta - jet1.eta)
        mmj2_dEta = abs(dimuon.eta - jet2.eta)
        mmj_min_dEta = ak.where(
            (mmj1_dEta < mmj2_dEta),
            mmj1_dEta,
            mmj2_dEta,
        )
        mmj1_dPhi = abs(dimuon.delta_phi(jet1))
        mmj2_dPhi = abs(dimuon.delta_phi(jet2))
        mmj1_dR = dimuon.delta_r(jet1)
        mmj2_dR = dimuon.delta_r(jet2)
        mmj_min_dPhi = ak.where(
            (mmj1_dPhi < mmj2_dPhi),
            mmj1_dPhi,
            mmj2_dPhi,
        )
        zeppenfeld = dimuon.eta - 0.5 * (
            jet1.eta + jet2.eta
        )
        mmjj = dimuon + dijet
        rpt = mmjj.pt / (
            dimuon.pt + jet1.pt + jet2.pt
        )

        # calculate ll_zstar
        ll_ystar = dimuon.rapidity - (jet1.rapidity + jet1.rapidity) / 2
        ll_zstar = abs(ll_ystar / (jet1.rapidity - jet1.rapidity))
    
        jet_loop_out_dict = {
            "jet1_pt" : jet1.pt,
            "jet1_eta" : jet1.eta,
            "jet1_rap" : jet1.rapidity,
            "jet1_phi" : jet1.phi,
            "jet1_qgl" : jet1.qgl,
            "jet1_jetId" : jet1.jetId,
            "jet1_puId" : jet1.puId,
            "jet2_pt" : jet2.pt,
            "jet2_eta" : jet2.eta,
            "jet1_mass" : jet1.mass,
            "jet2_mass" : jet2.mass,
            "jet1_pt_raw" : jet1.pt_raw,
            "jet2_pt_raw" : jet2.pt_raw,
            "jet1_mass_raw" : jet1.mass_raw,
            "jet2_mass_raw" : jet2.mass_raw,
            "jet1_rho" : jet1.rho,
            "jet2_rho" : jet2.rho,
            "jet1_area" : jet1.area,
            "jet2_area" : jet2.area,
            "jet1_pt_jec" : jet1.pt_jec,
            "jet2_pt_jec" : jet2.pt_jec,
            "jet1_mass_jec" : jet1.mass_jec,
            "jet2_mass_jec" : jet2.mass_jec,
            #-------------------------
            "jet2_rap" : jet2.rapidity,
            "jet2_phi" : jet2.phi,
            "jet2_qgl" : jet2.qgl,
            "jet2_jetId" : jet2.jetId,
            "jet2_puId" : jet2.puId,
            "jj_mass" : dijet.mass,
            "jj_pt" : dijet.pt,
            "jj_eta" : dijet.eta,
            "jj_phi" : dijet.phi,
            "jj_dEta" : jj_dEta,
            "jj_dPhi":  jj_dPhi,
            "mmj1_dEta" : mmj1_dEta,
            "mmj1_dPhi" : mmj1_dPhi,
            "mmj1_dR" : mmj1_dR,
            "mmj2_dEta" : mmj2_dEta,
            "mmj2_dPhi" : mmj2_dPhi,
            "mmj2_dR" : mmj2_dR,
            "mmj_min_dEta" : mmj_min_dEta,
            "mmj_min_dPhi" : mmj_min_dPhi,
            "mmjj_pt" : mmjj.pt,
            "mmjj_eta" : mmjj.eta,
            "mmjj_phi" : mmjj.phi,
            "mmjj_mass" : mmjj.mass,
            "rpt" : rpt,
            "zeppenfeld" : zeppenfeld,
            "njets" : njets,
            "ll_zstar" : ll_zstar,
            
        }
        if is_mc:
            mc_dict = {
                "jet1_pt_gen" : jet1.pt_gen,
                "jet2_pt_gen" : jet2.pt_gen,
            }
            jet_loop_out_dict.update(mc_dict)
        
        # jet_loop_out_dict = {
        #     key: ak.to_numpy(val) for key, val in jet_loop_out_dict.items()
        # }
        # jet_loop_placeholder =  pd.DataFrame(
        #     jet_loop_out_dict
        # )
        # jet_loop_placeholder.to_csv("./V2jet_loop.csv")
        
        # ------------------------------------------------------------#
        # Fill soft activity jet variables
        # ------------------------------------------------------------#

        # Effect of changes in jet acceptance should be negligible,
        # no need to calcluate this for each jet pT variation

        sj_dict = {}
        cutouts = [2,5]
        nmuons = ak.num(events.Muon, axis=1)
        if variation == "nominal":
            for cutout in cutouts:
                sj_out = fill_softjets(events, jets, mu1, mu2, nmuons, cutout, test_mode=self.test)
                sj_out = {
                    key+"_"+variation : val \
                    for key, val in sj_out.items()
                }
                sj_dict.update(sj_out)

        print(f"sj_dict.keys(): {sj_dict.keys()}")
        jet_loop_out_dict.update(sj_dict)
        

        # ------------------------------------------------------------#
        # Apply remaining cuts
        # ------------------------------------------------------------#

        # Cut has to be defined here because we will use it in
        # b-tag weights calculation
        vbf_cut = (dijet.mass > 400) & (jj_dEta > 2.5) & (jet1.pt > 35)
        vbf_cut = ak.fill_none(vbf_cut, value=False)
        jet_loop_out_dict.update({"vbf_cut": vbf_cut})

        # # ------------------------------------------------------------#
        # # Calculate QGL weights, btag SF and apply btag veto
        # # ------------------------------------------------------------#
        if is_mc and variation == "nominal":
        #     # --- QGL weights  start --- #
            isHerwig = "herwig" in dataset
            print("adding QGL weights!")
            # original start -------------------------------------
            # qgl_wgts = qgl_weights(jet1, jet2, njets, isHerwig)
            # original end -------------------------------------
            
            # keep dims start -------------------------------------
            qgl_wgts = qgl_weights_keepDim(jet1, jet2, njets, isHerwig)
            # keep dims end -------------------------------------
            weights.add("qgl", 
                        weight=qgl_wgts["nom"],
                        weightUp=qgl_wgts["up"],
                        weightDown=qgl_wgts["down"]
            )
        #     # --- QGL weights  end --- #
            

        #     # # --- Btag weights  start--- #
            do_btag_wgt = True # True
            if do_btag_wgt:
                print("doing btag wgt!")
                bjet_sel_mask = ak.ones_like(vbf_cut) #& two_jets & vbf_cut
                btag_systs = self.config["btag_systs"] #if do_btag_syst else []
                btag_json =  correctionlib.CorrectionSet.from_file(self.config["btag_sf_json"],)
                # original start -------------------------------------
                # btag_wgt, btag_syst = btag_weights_json(
                #     self, btag_systs, jets, weights, bjet_sel_mask, btag_json
                # )
                # original end -------------------------------------
                
                # keep dims start -------------------------------------
                btag_wgt, btag_syst = btag_weights_jsonKeepDim(
                            self, btag_systs, jets, weights, bjet_sel_mask, btag_json
                )
                # keep dims end -------------------------------------
                # print(f"btag_wgt: {ak.to_numpy(btag_wgt.compute())}")
                # print(f"btag_syst['jes_up']: {ak.to_numpy(btag_syst['jes']['up'].compute())}")
                # print(f"btag_syst['jes_down']: {ak.to_numpy(btag_syst['jes']['down'].compute())}")
            # # --- Btag weights end --- #

        
            # print(f"weight nom b4 adding btag: {ak.to_numpy(weights.weight().compute())}")
            # adding btag wgt directly to weights doesn't work, this may 
            # have to do with the fact that we use weights.weight() to 
            # calculate btag_wgt, so save this separtely and apply it later
            # weights.add("btag_wgt", 
            #             weight=btag_wgt
            # )
            # print(f"btag_wgt: {ak.to_numpy(btag_wgt.compute())}")
            # print(f"weight statistics: {weights.weightStatistics.keys()}")
            # print(f"weight nom after adding btag: {ak.to_numpy(weights.weight().compute())}")

        #     # --- Btag weights variations --- #
        #     for name, bs in btag_syst.items():
        #         weights.add_weight(f"btag_wgt_{name}", bs, how="only_vars")

        # Separate from ttH and VH phase space

        btagLoose_filter = (jets.btagDeepFlavB > self.config["btag_loose_wp"]) & (abs(jets.eta) < 2.5)
        nBtagLoose = ak.num(ak.to_packed(jets[btagLoose_filter]), axis=1)
        nBtagLoose = ak.fill_none(nBtagLoose, value=0)
            

        btagMedium_filter = (jets.btagDeepFlavB > self.config["btag_medium_wp"]) & (abs(jets.eta) < 2.5)
        nBtagMedium = ak.num(ak.to_packed(jets[btagMedium_filter]), axis=1)
        nBtagMedium = ak.fill_none(nBtagMedium, value=0)
            
        # print(f"nBtagLoose: {jets.btagDeepFlavB.compute()}")
        # print(f"nBtagLoose: {ak.to_numpy(nBtagLoose.compute())}")
        # print(f"njets: {ak.to_numpy(njets.compute())}")
        temp_out_dict = {
            "nBtagLoose": nBtagLoose,
            "nBtagMedium": nBtagMedium,
        }
        jet_loop_out_dict.update(temp_out_dict)
        if is_mc and do_btag_wgt:
            jet_loop_out_dict.update({
                "btag_wgt": btag_wgt
            })

        

        # --------------------------------------------------------------#
        # Fill outputs
        # --------------------------------------------------------------#

    #     variables.update({"wgt_nominal": weights.get_weight("nominal")})

    #     # All variables are affected by jet pT because of jet selections:
    #     # a jet may or may not be selected depending on pT variation.

    #     for key, val in variables.items():
    #         output.loc[:, pd.IndexSlice[key, variation]] = val

        return jet_loop_out_dict
    