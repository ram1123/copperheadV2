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
from corrections.evaluator import pu_evaluator, nnlops_weights, musf_evaluator, get_musf_lookup, lhe_weights, stxs_lookups, add_stxs_variations, add_pdf_variations, qgl_weights, qgl_weights_eager, btag_weights_json, get_jetpuid_weights
import json
from coffea.lumi_tools import LumiMask
import pandas as pd # just for debugging
import dask_awkward as dak
import dask
from coffea.analysis_tools import Weights
import copy

coffea_nanoevent = TypeVar('coffea_nanoevent') 
ak_array = TypeVar('ak_array')

def cs_variables(
        mu1: coffea_nanoevent,
        mu2: coffea_nanoevent
    ) -> Tuple[ak_array]: 
    dphi = abs(mu1.delta_phi(mu2))
    theta_cs = np.arccos(np.tanh((mu1.eta - mu2.eta) / 2))
    phi_cs = np.tan((np.pi - np.abs(dphi)) / 2) * np.sin(theta_cs)
    return np.cos(theta_cs), phi_cs



class EventProcessor(processor.ProcessorABC):
    # def __init__(self, config_path: str,**kwargs):
    def __init__(self, config: dict, test_mode=False, **kwargs):
        """
        TODO: replace all of these with self.config dict variable which is taken from a
        pre-made json file
        """
        # with open(config_path) as file:
        #     self.config = json.loads(file.read())
        self.config = config

        # self.config = json.loads(config_path)
        # print(f"copperhead proccesor self.config b4 update: \n {self.config}")
        self.test = test_mode# False
        dict_update = {
            # "hlt" :["IsoMu24"],
            "do_trigger_match" : False, # False
            "do_roccor" : True,# False
            "do_fsr" : True,
            "do_geofit" : True, # False
            "do_beamConstraint": True, # if True, override do_geofit
            "year" : "2018",
            # "rocorr_file_path" : "data/roch_corr/RoccoR2018.txt",
            "do_nnlops" : True,
            "do_pdf" : True,
        }
        self.config.update(dict_update)
        # print(f"copperhead proccesor self.config after update: \n {self.config}")
        

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
        # turn ._axes from tuple of axes to just axes
        # self.evaluator[self.zpt_path]._axes = self.evaluator[self.zpt_path]._axes[0]  
                
        
        # # prepare lookup tables for all kinds of corrections
        # self.prepare_lookups()

    def process(self, events: coffea_nanoevent):
        """
        TODO: Once you're done with testing and validation, do LHE cut after HLT and trigger match event filtering to save computation
        """
        # dataset = events.metadata['dataset']
        # cross_section = self.config["cross_sections"][dataset]
        # totalGenWgts = events.metadata['sumGenWgts']
        # integrated_lumi = self.config["integrated_lumis"][year]
        # lumi_weight = cross_section * integrated_lumi/ totalGenWgts
        # self.config["lumi_weight"] = lumi_weight
        if self.test:
            print(f"copperhead2 events muon pt: {ak.to_dataframe(events.Muon.pt)}")
            print(f"copperhead2 type(events): {type(events)}")
            print(f"copperhead2 events.metadata: {events.metadata}")
        """
        Apply LHE cuts for DY sample stitching
        Basically remove events that has dilepton mass between 100 and 200 GeV
        """
        event_filter = ak.ones_like(events.HLT.IsoMu24) # 1D boolean array to be used to filter out bad events
        if events.metadata['dataset'] == 'dy_M-50': # if dy_M-50, apply LHE cut
            print("doing LHE cut!")
            LHE_particles = events.LHEPart #has unique pdgIDs of [ 1,  2,  3,  4,  5, 11, 13, 15, 21]
            bool_filter = (abs(LHE_particles.pdgId) == 11) | (abs(LHE_particles.pdgId) == 13) | (abs(LHE_particles.pdgId) == 15)
            LHE_leptons = LHE_particles[bool_filter]

            if self.test:
                # check LHE muons maintain the same event length
                print(f"copperhead2 EventProcessor LHE_particles: {len(LHE_particles)}")
                print(f"copperhead2 EventProcessor LHE_leptons: {len(LHE_leptons)}")
                print(f"copperhead2 EventProcessor LHE_leptons.pdgId: {LHE_leptons.pdgId}")

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
        # if self.test:
        #     print(f"copperhead2 EventProcessor events.HLT.IsoMu24: \n {ak.to_numpy(events.HLT.IsoMu24)}")
        #     print(f"copperhead2 EventProcessor ak.sum(events.HLT.IsoMu24): \n {ak.sum(events.HLT.IsoMu24)}")
        #     print(f"copperhead2 EventProcessor event_filter: \n {ak.to_numpy(event_filter)}")

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
                        test=self.test
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
            print(f"doing beam constraint")
            print(f"events.Muon.fields: {events.Muon.fields}")
            BSConstraint_mask = (
                (events.Muon.bsConstrainedChi2 <30)
            )
            BSConstraint_mask = ak.fill_none(BSConstraint_mask, False)
            BSConstraint_mask = BSConstraint_mask & (~applied_fsr) # apply BSContraint on non FSR muons
            events["Muon", "pt"] = ak.where(BSConstraint_mask, events.Muon.bsConstrainedPt, events.Muon.pt)
            events["Muon", "ptErr"] = ak.where(BSConstraint_mask, events.Muon.bsConstrainedPtErr, events.Muon.ptErr)
        else:
            if self.config["do_geofit"] and ("dxybs" in events.Muon.fields):
                print(f"doing geofit")
                gf_filter, gf_pt_corr = apply_geofit(events, self.config["year"], ~applied_fsr)
                events["Muon", "pt"] = events.Muon.pt_gf
                # save gf_filter, gf_pt_corr for debugging. comment it out later
                # events["Muon", "gf_filter"] = gf_filter
                # events["Muon", "gf_pt_corr"] = gf_pt_corr
            else: 
                # print(f"doing neither beam constraint nor geofit")
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

        # print(f"sum muon_selection: {ak.sum(muon_selection).compute()}")
        
        # # testing muon selection ------------------------------------------------
        # muon_selection = (
        #     (ak.values_astype(events.Muon.pt_raw,  "float64") > self.config["muon_pt_cut"])
        #     & (ak.values_astype(abs(events.Muon.eta_raw), "float64")  < self.config["muon_eta_cut"])
        #     & (ak.values_astype(events.Muon.pfRelIso04_all, "float64")  < self.config["muon_iso_cut"])
        #     # & events.Muon[muon_id]
        #     & events.Muon[self.config["muon_id"]]
        # )
        # # testing muon selection end ------------------------------------------------

        # # count muons that pass the general cut 
        # nmuons = ak.num(events.Muon[muon_selection], axis=1)
        
        # # Find opposite-sign muons
        # mm_charge = ak.prod(events.Muon.charge[muon_selection], axis=1)
        
        muons = events.Muon[muon_selection]
        # muons = ak.to_packed(events.Muon[muon_selection])
        # count muons that pass the general cut 
        nmuons = ak.num(muons, axis=1)
        # Find opposite-sign muons
        mm_charge = ak.prod(muons.charge, axis=1)
        

        # Veto events with good quality electrons; VBF and ggH categories need zero electrons
        electron_selection = (
            (events.Electron.pt > self.config["electron_pt_cut"])
            & (abs(events.Electron.eta) < self.config["electron_eta_cut"])
            & events.Electron[self.config["electron_id"]]
        )
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
        # good_pv = (events.PV.npvsGood > 0) 
        # mm_charge_cond = (mm_charge == -1)
        # nmuon_cond  = (nmuons == 2)
        
        # print(f" events len: {ak.num(events.Muon, axis=0).compute()}")
        # print(f" event filter true len: {ak.sum((event_filter==True), axis=0).compute()}")
        
        # print(f" mm_charge_cond true len: {ak.sum((mm_charge_cond==True), axis=0).compute()}")
        # print(f" nmuon_cond true len: {ak.sum((nmuon_cond==True), axis=0).compute()}")
        # print(f" electron_veto true len: {ak.sum((electron_veto==True), axis=0).compute()}")
        # print(f" good_pv true len: {ak.sum((good_pv==True), axis=0).compute()}")
        # if self.test:
        #     # print(f"copperhead2 EventProcessor b4 leading pt cut event_filter long: \n {pd.DataFrame(ak.to_numpy(event_filter)).to_string()}")
        #     print(f"copperhead2 EventProcessor b4 leading pt cut event_filter sum:  {ak.sum(event_filter)}")
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
        # mu2 = muons_sorted[:,1]
        pass_leading_pt = mu1.pt_raw > self.config["muon_leading_pt"]
        pass_leading_pt = ak.fill_none(pass_leading_pt, value=False) 
        # print(f"pass_leading_pt: {pass_leading_pt.compute()}")


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
        # print(f"events b4 filter length: {ak.num(events.Muon.pt, axis=0).compute()}")
        # skim off bad events onto events and other related variables
        # # original -----------------------------------------------
        # events = events[event_filter==True]
        # muons = muons[event_filter==True]
        # nmuons = nmuons[event_filter==True]
        # applied_fsr = applied_fsr[event_filter==True]
        # if is_mc:
        #     for variation in pu_wgts.keys():
        #         pu_wgts[variation] = pu_wgts[variation][event_filter==True]

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
        # turn off pu weights test end ---------------------------------
        pass_leading_pt = ak.to_packed(pass_leading_pt[event_filter==True])

        # to_packed testing end -----------------------------------------------
        
            
        
        # if self.test:
            
        #     print(f"copperhead2 EventProcessor pass_leading_pt: \n {pass_leading_pt}")
        #     print(f"copperhead2 EventProcessor after leading pt cut event_filter long: \n {ak.to_dataframe(event_filter).to_string()}")
        #     print(f"copperhead2 EventProcessor after leading pt cut ak.sum(event_filter): \n {ak.sum(event_filter)}")
        #     print(f"copperhead2 EventProcessor events.Muon: \n {ak.num(events.Muon, axis=1)}")


        
        
        # --------------------------------------------------------#
        # Fill dimuon and muon variables
        # --------------------------------------------------------#

        # ---------------------------------------------------------
        # TODO: find out why we don't filter out bad events right now via
        # even_selection column, since fill muon is computationally exp
        # Last time I checked there was some errors on LHE correction shape mismatch
        # ---------------------------------------------------------

        muons_padded = ak.pad_none(muons, target=2)
        # muon_flip = muons_padded.pt[:,0] < muons_padded.pt[:,1]  
        # muon_flip = ak.fill_none(muon_flip, value=False)
        # # take the subleading muon values if that now has higher pt after corrections
        # mu1 = ak.where(muon_flip, muons_padded[:,1], muons_padded[:,0])
        # mu2 = ak.where(muon_flip, muons_padded[:,0], muons_padded[:,1])
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

        # skip validation for genjets for now -----------------------------------------------
        # #fill genjets
        
        # if events.metadata["is_mc"]:
        #     #fill gen jets for VBF filter on postprocess
        #     gjets = events.GenJet
        #     gleptons = events.GenPart[
        #         (abs(events.GenPart.pdgId) == 13)
        #         | (abs(events.GenPart.pdgId) == 11)
        #         | (abs(events.GenPart.pdgId) == 15)
        #     ]
        #     gl_pair = ak.cartesian({"jet": gjets, "lepton": gleptons}, axis=1, nested=True)
        #     dr_gl = gl_pair["jet"].delta_r(gl_pair["lepton"])
        #     isolated = ak.all((dr_gl > 0.3), axis=-1) # this also returns true if there's no leptons near the gjet
            
        #     # I suppose we assume there's at least two jets
        #     padded_iso_gjet = ak.pad_none(gjets[isolated],2) # pad with none val to ensure that events have at least two columns each event
        #     gjet1 = padded_iso_gjet[:,0]
            
        #     gjet2 = padded_iso_gjet[:,1] 
        #     gjj = gjet1 + gjet2
            
        #     gjj_dEta = abs(gjet1.eta - gjet2.eta)
        #     gjj_dPhi = abs(gjet1.delta_phi(gjet2))
        #     gjj_dR = gjet1.delta_r(gjet2)
        #     if self.test:
        #         print(f"fill_gen_jets isolated: \n {isolated}")
        #         print(f"fill_gen_jets isolated long: \n {ak.to_numpy(ak.flatten(isolated))}")
        #         print(f"fill_gen_jets gjet1: \n {gjet1}")
        #         print(f"fill_gen_jets gjj: \n {gjj}")
        #         print(f"fill_gen_jets gjj_dEta: \n {gjj_dEta}")
        #         print(f"fill_gen_jets gjj_dPhi: \n {gjj_dPhi}")
        #         print(f"fill_gen_jets gjj_dR: \n {gjj_dR}")
        # # else: # if data
        # #     # TODO fill this with None values later, this filling is for testing
        # #     # gjet1 = ak.zeros_like(events.Muon.pt[:,0])
        # #     # gjet2 = ak.zeros_like(events.Muon.pt[:,0])
        # #     padded_iso_gjet = ak.pad_none(ak.zeros_like(events.Muon.pt), 2)
        # #     gjj = ak.zeros_like(events.Muon.pt[:,0])
        # #     gjj_dEta = ak.zeros_like(events.Muon.pt[:,0])
        # #     gjj_dPhi = ak.zeros_like(events.Muon.pt[:,0])
        # #     gjj_dR = ak.zeros_like(events.Muon.pt[:,0])
        
        # if self.test:
        #     print(f"copperhead2 EventProcessor events.Jet.rho: \n {events.Jet.rho}")
        #     print(f"copperhead2 EventProcessor events.Jet.rho long: \n {ak.to_numpy(ak.flatten(events.Jet.rho))}")   
        #     print(f'copperheadV2 EventProcessor jets.pt b4 apply_jec long: \n {ak.to_numpy(ak.flatten(events.Jet.pt))}')
        #     print(f'copperheadV2 EventProcessor jets.eta b4 apply_jec long: \n {ak.to_numpy(ak.flatten(events.Jet.eta))}')
        #     print(f'copperheadV2 EventProcessor jets.phi b4 apply_jec long: \n {ak.to_numpy(ak.flatten(events.Jet.phi))}')
        #     print(f'copperheadV2 EventProcessor jets.mass b4 apply_jec long: \n {ak.to_numpy(ak.flatten(events.Jet.mass))}')
        # skip validation for genjet end --------------------------------------------

        self.prepare_jets(events)


        # ------------------------------------------------------------#
        # Apply JEC, get JEC and JER variations
        # ------------------------------------------------------------#
        
        jets = events.Jet
        self.jec_factories_mc, self.jec_factories_data = get_jec_factories(
            self.config["jec_parameters"], 
            test_mode=self.test
        )   
        # print(f"njets pre jec: {ak.to_numpy(ak.num(jets, axis=1).compute())}")
        
        do_jec = True # True 

        

        # do_jecunc = self.config["do_jecunc"]
        # do_jerunc = self.config["do_jerunc"]
        #testing 
        do_jecunc = False
        do_jerunc = False
        # cache = events.caches[0]
        if do_jec:
            if events.metadata["is_mc"]:
                factory = self.jec_factories_mc["jec"]
            else:
                for run in self.config["jec_parameters"]["runs"]:
                    if run in events.metadata["dataset"]:
                        factory = self.jec_factories_data[run]
                
            print("do jec!")
            # jets = factory.build(jets, lazy_cache=cache)
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

        


        # # if self.test:
        # #     print(f'copperheadV2 EventProcessor after apply_jec jets.pt short: \n {jets.pt}')
        # #     print(f'copperheadV2 EventProcessor after apply_jec jets.pt long: \n {ak.to_numpy(ak.flatten(jets.pt.compute()))}')
        # #     print(f'copperheadV2 EventProcessor jets.pt_jec b4 apply_jec long: \n {ak.to_numpy(ak.flatten(jets.pt_jec.compute()))}')
        # #     print(f'copperheadV2 EventProcessor after apply_jec jets.pt_orig long: \n {ak.to_numpy(ak.flatten(jets.pt_orig.compute()))}')
        # #     print(f'copperheadV2 EventProcessor after apply_jec jets.eta long: \n {ak.to_numpy(ak.flatten(jets.eta.compute()))}')
        # #     print(f'copperheadV2 EventProcessor after apply_jec jets.phi long: \n {ak.to_numpy(ak.flatten(jets.phi.compute()))}')
        # #     print(f'copperheadV2 EventProcessor jets.mass b4 apply_jec long: \n {ak.to_numpy(ak.flatten(jets.mass.compute()))}')
        # #     print(f'copperheadV2 EventProcessor jets.mass_jec b4 apply_jec long: \n {ak.to_numpy(ak.flatten(jets.mass_jec.compute()))}')
        # #     print(f'copperheadV2 EventProcessor jets.mass_orig b4 apply_jec long: \n {ak.to_numpy(ak.flatten(jets.mass_orig.compute()))}')
        # # print(f'copperheadV2 EventProcessor jets.fields: \n {jets.fields}')
        

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
            dataset = events.metadata['dataset']
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
        
        #     # For MC: initialize weight_collection
        #     weight_ones = ak.ones_like(events.Muon.pt[:,0]) # get 1D array of filtered events
        #     # print(f"weight_collection len(weight_ones):  {ak.num(weight_ones, axis=0)}")
        #     self.weight_collection = Weights(weight_ones)
        
        #     # For MC: Apply gen.weights, pileup weights, lumi weights,
        #     # apply event_filter on pu weights and then add them to weight_collection
        #     # print(f"weight_collection len(pu_wgts):  {len(pu_wgts)}")
        #     # print(f"weight_collection len(event_filter):  {len(event_filter)}")
        #     # print(f"weight_collection (pu_wgts):  {(pu_wgts)}")
        #     # print(f"weight_collection (event_filter):  {(event_filter)}")
        #     for key in pu_wgts.keys():
        #         pu_wgts[key] = pu_wgts[key][event_filter==True]
            
        #     self.weight_collection.add_weight("pu_wgt", pu_wgts, how="all")
        #     if self.test:
        #         print(f"weight_collection pu_wgt info: \n  {self.weight_collection.get_info()}")
            

            

        
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
                # print(f"lhe_ren: {lhe_ren}")
                # print(f"lhe_fac: {lhe_fac}")
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
        # out_dict = {
        #     "mu_pt" : events.Muon.pt,
        #     # "mu_eta" : events.Muon.eta,
        #     # "mu_phi" : events.Muon.phi,
        #     # "mu_iso" : events.Muon.pfRelIso04_all,
        #     # "dimuon_mass": dimuon.mass,
        #     # "dimuon_pt" : dimuon.pt,
        #     # "dimuon_eta" : dimuon.eta,
        #     # "dimuon_phi" : dimuon.phi,
        #     # "dimuon_dEta" : dimuon_dEta,
        #     # "dimuon_dPhi" : dimuon_dPhi,
        #     # "dimuon_dR" : dimuon_dR,
        #     # "dimuon_ebe_mass_res": dimuon_mass_resolution,
        #     # "dimuon_ebe_mass_res_rel": rel_dimuon_ebe_mass_res,
        #     # "mu_pt_over_mass" : events.Muon.pt / dimuon.mass,
        #     # "dimuon_cos_theta_cs" : dimuon_cos_theta_cs,
        #     # "dimuon_phi_cs" : dimuon_phi_cs,
        # }
        out_dict = {
            "mu1_pt" : mu1.pt,
            "mu2_pt" : mu2.pt,
            "mu1_eta" : mu1.eta,
            "mu2_eta" : mu2.eta,
            "mu1_phi" : mu1.phi,
            "mu2_phi" : mu2.phi,
            "mu1_charge" : mu1.charge,
            "mu2_charge" : mu2.charge,
            "nmuons" : nmuons,
            "dimuon_mass" : dimuon.mass,
            "dimuon_eta" : dimuon.eta,
            "dimuon_phi" : dimuon.phi,
            "dimuon_dEta" : dimuon_dEta,
            "dimuon_dPhi" : dimuon_dPhi,
            "dimuon_dR" : dimuon_dR,
            "dimuon_ebe_mass_res" : dimuon_ebe_mass_res,
            "dimuon_cos_theta_cs" : dimuon_cos_theta_cs,
            "dimuon_phi_cs" : dimuon_phi_cs,
            "mu1_pt_raw" : mu1.pt_raw,
            "mu2_pt_raw" : mu2.pt_raw,
            "mu1_pt_fsr" : mu1.pt_fsr,
            "mu2_pt_fsr" : mu2.pt_fsr,
            "pass_leading_pt" : pass_leading_pt,
            
        }
        if is_mc:
            mc_dict = {
                "HTXS_Higgs_pt" : events.HTXS.Higgs_pt, # for nnlops weight for ggH signal sample
                "HTXS_njets30" : events.HTXS.njets30, # for nnlops weight for ggH signal sample
                # "gjet_pt" : padded_iso_gjet.pt,
                # "gjet_eta" : padded_iso_gjet.eta,
                # "gjet_phi" : padded_iso_gjet.phi,
        #         "gjet_mass" : padded_iso_gjet.mass,
        #         "gjj_mass": gjj.mass,
        #         "gjj_pt" : gjj.pt,
        #         "gjj_eta" : gjj.eta,
        #         "gjj_phi" : gjj.phi,
        #         "gjj_dEta" : gjj_dEta,
        #         "gjj_dPhi" : gjj_dPhi,
        #         "gjj_dR" : gjj_dR,
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
            
            # # testing --------------------------------------
            # jets = self.jet_loop(
            #     events, 
            #     jets,
            #     dimuon,
            #     variation,
            #     weights,
            #     do_jec = do_jec,
            #     do_jecunc = do_jecunc,
            #     do_jerunc = do_jerunc,
            # )
            # # testing end --------------------------------------
        
        out_dict.update(jet_loop_dict) 
        
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


        # # add in the weights
        # if events.metadata["is_mc"]:
        #     if self.test:
        #         print(f"self.weight_collection.weights: {self.weight_collection.weights} ")
        #     out_dict.update({
        #        "nominal" : self.weight_collection.get_weight("nominal")
        #     })
        #     # out_dict.update(self.weight_collection.weights["nominal"])
        # just reading part 2 end ---------------------------

        # --------------------------
        
        # b4 we do any filtering, we obtain the sum of gen weights for normalization
        # events["genWeight"] = ak.values_astype(events.genWeight, "float64") # increase precision or it gives you slightly different value for summing them up
        
        # raise ValueError
        # muons = events.Muon[muon_selection]
        # nmuons = ak.num(muons, axis=1)
        # event_filter =   nmuons>=1
        # print(f"event_filter: {event_filter.compute()}")
        # events = events[event_filter]
        # muons = muons[event_filter] # update events on these too
        # nmuons = nmuons[event_filter] # update events on these too
        # applied_fsr = applied_fsr[event_filter]
        # jets = events.Jet
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
        
        # print(f"events after filter length: {ak.num(muons.pt, axis=0).compute()}")
        # weights = events.genWeight
        # dataset = events.metadata['dataset']
        # cross_section = self.config["cross_sections"][dataset]
        # integrated_lumi = self.config["integrated_lumis"]
        # if is_mc:
        #     print(f"cross_section: {(cross_section)}")
        #     print(f"integrated_lumi: {(integrated_lumi)}")
        # weights = weights*cross_section*integrated_lumi/sumWeights
        print(f"weight statistics: {weights.weightStatistics.keys()}")
        weights = weights.weight()
        if "btag_wgt" in out_dict.keys():
            print("adding btag wgts!")
            weights = weights*out_dict["btag_wgt"]
        if do_zpt:
            weights = weights*zpt_weight
        
        # print(f"weights: {ak.to_numpy(weights.compute())}")
        # weights = weights.weight("pdf_2rmsUp")
        # weights = weights.weight("LHEFacDown")
        
        # print(f"weights: {ak.num(weights, axis=0).compute()}")
        
        # print(f"nmuons: {ak.num(nmuons, axis=0).compute()}")
        # print(f"njets: {ak.num(njets, axis=0).compute()}")
        # print(f"jets: {ak.num(jets, axis=0).compute()}")
        # print(f"muons: {ak.num(muons, axis=0).compute()}")
        # muons_padded = ak.pad_none(muons, 2)
        # muon_flip = muons.pt[:,0] < muons.pt[:,1]  
        # # take the subleading muon values if that now has higher pt after corrections
        # mu1 = ak.where(muon_flip, muons_padded[:,1], muons_padded[:,0])
        # mu2 = ak.where(muon_flip, muons_padded[:,0], muons_padded[:,1])
        # out_dict = {
        #     # "mu_pt" : ak.pad_none(muons.pt, 2),
        #     # "mu_eta" : ak.pad_none(muons.eta, 2),
        #     # "mu_phi" : ak.pad_none(muons.phi, 2),
        #     # "mu_charge" : ak.pad_none(muons.charge, 2),
        #     "mu1_pt" : mu1.pt,
        #     "mu2_pt" : mu2.pt,
        #     "mu1_eta" : mu1.eta,
        #     "mu2_eta" : mu2.eta,
        #     "mu1_phi" : mu1.phi,
        #     "mu2_phi" : mu2.phi,
        #     "mu1_charge" : mu1.charge,
        #     "mu2_charge" : mu2.charge,
        #     "nmuons" : nmuons,
        #     # "jet_pt" : ak.pad_none(jets.pt, 2),
        #     # "jet_mass" : ak.pad_none(jets.mass, 2),
        #     # "jet_eta" : ak.pad_none(jets.eta, 2),
        #     # "jet_phi" : ak.pad_none(jets.phi, 2),
        #     # "jet_pt_raw" : ak.pad_none(jets.pt_raw, 2),
        #     # "jet_mass_raw" : ak.pad_none(jets.mass_raw, 2),
        #     # "jet_rho" : ak.pad_none(jets.rho, 2),
        #     # "jet_area" : ak.pad_none(jets.area, 2),
        #     # "jet_pt_gen" : ak.pad_none(jets.pt_gen, 2),
        #     # "jet_pt_jec" : ak.pad_none(jets.pt_jec, 2),
        #     # "jet_mass_jec" : ak.pad_none(jets.mass_jec, 2),
        #     # "njets" : njets,
        #     "weights" : weights,
        #     # "mu1_gf_filter" : events.Muon.gf_filter[:,0],
        #     # "mu1_gf_pt_corr" :events.Muon.gf_pt_corr[:,0],
        #     "dimuon_mass" : dimuon.mass,
        #     "dimuon_ebe_mass_res" : dimuon_ebe_mass_res,
        #     "dimuon_cos_theta_cs" : dimuon_cos_theta_cs,
        #     "dimuon_phi_cs" : dimuon_phi_cs,
        #     "HTXS_Higgs_pt" : events.HTXS.Higgs_pt,
        #     "HTXS_njets30" : events.HTXS.njets30,
        # }

        # add in weights
        out_dict.update({"weights" : weights})
        
        # if self.config["do_fsr"]:
        #     fsr_dict = {"fsr_mask" : (ak.sum(applied_fsr, axis=1) > 0)}
        #     out_dict.update(fsr_dict)
        #----------------------------
        
        return out_dict
        
    def postprocess(self, accumulator):
        """
        Arbitrary postprocess function that's required to run the processor
        """
        pass

    
    # def get_mass_resolution(self, events, test_mode=False):
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
    
        return ((dpt1 * dpt1 + dpt2 * dpt2)**0.5) * calibration
    
    def prepare_jets(self, events): # analogous to add_jec_variables function in boosted higgs
        # Initialize missing fields (needed for JEC)
        events["Jet", "pt_raw"] = (1 - events.Jet.rawFactor) * events.Jet.pt
        events["Jet", "mass_raw"] = (1 - events.Jet.rawFactor) * events.Jet.mass
        events["Jet", "rho"] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, events.Jet.pt)[0]
    
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
            "btagDeepB",
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
        # print(f"sum pass_jet_id: {ak.to_numpy(ak.sum(pass_jet_id, axis=1).compute())}")
        # print(f"sum pass_jet_id: {ak.to_numpy(ak.sum(pass_jet_id, axis=None).compute())}")
        # print(f"njets: {ak.to_numpy(ak.num(jets, axis=1).compute())}")
        
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
        # print(f"sum jet_selection: {ak.to_numpy(ak.sum(jet_selection, axis=1).compute()).shape}")
        # print(f"sum jet_selection: {ak.to_numpy(ak.sum(jet_selection, axis=1).compute())}")
        # print(f"sum jet_selection total: {ak.sum(jet_selection, axis=None).compute()}")
        # original jet_selection end ----------------------------------------------


        # jets = jets[jet_selection] # this causes huuuuge memory overflow close to 100 GB. Without it, it goes to around 20 GB
        # jets = ak.to_packed(ak.mask(jets, jet_selection))
        # ak mask leaves None in room of filtered Jets -> which isn't good
        # bc then leading jet now becomes None bc idx==0 jet is None
        
        jets = ak.to_packed(jets[jet_selection]) 
        # jets = jets[jet_selection]

        
        # #testing ------------------------
        # return jets
        # # testing ------------
        
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
        
        # print(f"number of events passing jet selection: {ak.sum((njets>0)).compute()}")

        # ------------------------------------------------------------#
        # Fill jet-related variables
        # ------------------------------------------------------------#

        
        # if self.test:
        # print(f"jet loop njets: {njets}")
        # print(f"jet loop ak.num(events, axis=0): {ak.num(events, axis=0)}")
        # print(f"jet loop jet_selection short: {jet_selection}")
        # print(f"jet loop jet_selection sum: {ak.sum(jet_selection, axis =1)}")
        # print(f"jet loop jet_selection long: {ak.to_numpy(ak.flatten(jet_selection))}")
        # print(f"jet loop jets.pt short: {jets.pt}")
        # variables["njets"] = njets

        # fill_jets(output, variables, jet1, jet2)
        #fill_jets
        # print(f"jets: {jets.compute()}")
        # jet_argmax = ak.argmax(jets.pt, axis=1)
        # jet_argmax_not_leading = ak.fill_none((jet_argmax != 0), value=False)
        # print(f"jet_argmax_not_leading: {jet_argmax_not_leading.compute()}")
        # print(f"sum jet_argmax_not_leading: {ak.sum(jet_argmax_not_leading).compute()}")
        # print(f"jet_argmax[jet_argmax_not_leading] : {jet_argmax[jet_argmax_not_leading].compute()}")
        # print(f"jets.pt[jet_argmax_not_leading] : {jets.pt[jet_argmax_not_leading].compute()}")
        # # print(f"jets.pt_jec[jet_argmax_not_leading] : {jets.pt_jec[jet_argmax_not_leading].compute()}")
        # print(f"jets.pt_raw[jet_argmax_not_leading] : {jets.pt_raw[jet_argmax_not_leading].compute()}")
        # print(f"jets.mass_raw[jet_argmax_not_leading] : {jets.mass_raw[jet_argmax_not_leading].compute()}")
        # print(f"jets.mass[jet_argmax_not_leading] : {jets.mass[jet_argmax_not_leading].compute()}")
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

        jet_loop_out_dict = {
            "jet1_pt" : jet1.pt,
            # "jet1_pt" : ak.where(jet_selection,jet1.pt,-999),
            "jet1_eta" : jet1.eta,
            "jet1_rap" : jet1.rapidity,
            "jet1_phi" : jet1.phi,
            "jet1_qgl" : jet1.qgl,
            "jet1_jetId" : jet1.jetId,
            "jet1_puId" : jet1.puId,
            "jet2_pt" : jet2.pt,
            "jet2_eta" : jet2.eta,
            #-------------------------------
            # "jet1_pt" : jet1.pt[jet_selection],
            # "jet1_eta" : jet1.eta[jet_selection],
            # "jet1_rap" : jet1.rapidity[jet_selection],
            # "jet1_phi" : jet1.phi[jet_selection],
            # "jet1_qgl" : jet1.qgl[jet_selection],
            # "jet1_jetId" : jet1.jetId[jet_selection],
            # "jet1_puId" : jet1.puId[jet_selection],
            # "jet2_pt" : jet2.pt[jet_selection],
            # "jet2_eta" : jet2.eta[jet_selection],
            #----------------------------------
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
            # # "jj_mass" : ak.where(jet_selection,dijet.mass,-999),
            # "jj_mass_log" : np.log(dijet.mass),
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
                # sj_out = fill_softjets(events, jets, muons, cutout, test_mode=self.test)
                sj_out = fill_softjets(events, jets, mu1, mu2, nmuons, cutout, test_mode=self.test)
                sj_out = {
                    key+"_"+variation : val \
                    for key, val in sj_out.items()
                }
                sj_dict.update(sj_out)
            if self.test:
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
            isHerwig = "herwig" in events.metadata['dataset']
            print("adding QGL weights!")
            qgl_wgts = qgl_weights(jet1, jet2, njets, isHerwig)
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
                btag_wgt, btag_syst = btag_weights_json(
                    self, btag_systs, jets, weights, bjet_sel_mask, btag_json
                )
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
        
    #     # print(f'self.parameters["btag_medium_wp"] : {self.parameters["btag_medium_wp"]}')
    #     # print(f'jets  \n: {jets}')
    #     # print(f'jets.btagDeepB  \n: {jets.btagDeepB}')
    #     # test = jets[
    #     #         (jets.btagDeepB > self.parameters["btag_loose_wp"])
    #     #         & (abs(jets.eta) < 2.5)
    #     #     ]
    #     # print(f'jets btag_loose_wp test \n: {test.btagDeepB[:50]}')
    #     # test = test.reset_index().groupby("entry")["subentry"]
    #     # print(f'nBtagLoose test nunique \n: {test.nunique()[:50]}')
    #     # print(f'nBtagLoose test sum \n: {test.sum()[:50]}')

        btagLoose_filter = (jets.btagDeepB > self.config["btag_loose_wp"]) & (abs(jets.eta) < 2.5)
        nBtagLoose = ak.num(jets[btagLoose_filter], axis=1)
        nBtagLoose = ak.fill_none(nBtagLoose, value=0)
            

        btagMedium_filter = (jets.btagDeepB > self.config["btag_medium_wp"]) & (abs(jets.eta) < 2.5)
        nBtagMedium = ak.num(jets[btagMedium_filter], axis=1)
        nBtagMedium = ak.fill_none(nBtagMedium, value=0)
            
        
        
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
    