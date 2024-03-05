import coffea.processor as processor
from coffea.lookup_tools import extractor
import awkward as ak
import numpy as np
from typing import Union, TypeVar, Tuple
from corrections.rochester import apply_roccor
from corrections.fsr_recovery import fsr_recovery
from corrections.geofit import apply_geofit
from corrections.jet import get_jec_factories, jet_id, jet_puid, fill_softjets
from corrections.weight import Weights
from corrections.evaluator import pu_evaluator, nnlops_weights, musf_evaluator, get_musf_lookup, lhe_weights, stxs_lookups, add_stxs_variations, add_pdf_variations, qgl_weights
import json
from coffea.lumi_tools import LumiMask
import pandas as pd # just for debugging
import dask_awkward as dak
import dask

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
            "do_roccor" : False,# False
            "do_fsr" : False,
            "do_geofit" : False, # False
            "do_beamConstraint": False,
            "year" : "2018",
            # "rocorr_file_path" : "data/roch_corr/RoccoR2018.txt",
            "do_nnlops" : False,
            "do_pdf" : False
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
        elif ("2017" in year) or ("2018" in year):
            self.zpt_path = "zpt_weights/2017_value" # use same weights for 2018
        elif "2022" in year:
            self.zpt_path = "zpt_weights/2022_value" #  hypothetical value, we don't have run3 data yet
        else:
            print(f"USER WARNING: unrecognized Zpt correction for the year {year}!")
        
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
        self.evaluator[self.zpt_path]._axes = self.evaluator[self.zpt_path]._axes[0]  
        
        self.weight_collection = None # will be initialzed later

        
        
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
        event_filter = ak.ones_like(events.HLT.IsoMu24) # 1D array to be used to filter out bad events
        # if events.metadata['dataset'] == 'dy_M-50': # if dy_M-50, apply LHE cut
        #     LHE_particles = events.LHEPart #has unique pdgIDs of [ 1,  2,  3,  4,  5, 11, 13, 15, 21]
        #     bool_filter = (abs(LHE_particles.pdgId) == 11) | (abs(LHE_particles.pdgId) == 13) | (abs(LHE_particles.pdgId) == 15)
        #     LHE_leptons = LHE_particles[bool_filter]

        #     if self.test:
        #         # check LHE muons maintain the same event length
        #         print(f"copperhead2 EventProcessor LHE_particles: {len(LHE_particles)}")
        #         print(f"copperhead2 EventProcessor LHE_leptons: {len(LHE_leptons)}")
        #         print(f"copperhead2 EventProcessor LHE_leptons.pdgId: {LHE_leptons.pdgId}")

        #     """
        #     TODO: maybe we can get faster by just indexing first and second, instead of argmax and argmins
        #     When I had a quick look, all LHE_leptons had either two or zero leptons per event, never one, 
        #     so just indexing first and second could work
        #     """
        #     max_idxs = ak.argmax(LHE_leptons.pdgId , axis=1,keepdims=True) # get idx for normal lepton
        #     min_idxs = ak.argmin(LHE_leptons.pdgId , axis=1,keepdims=True) # get idx for anti lepton
        #     LHE_lepton_barless = LHE_leptons[max_idxs]
        #     LHE_lepton_bar = LHE_leptons[min_idxs]
        #     if self.test:
        #         print(f"copperhead2 EventProcessor LHE_lepton_bar: {LHE_lepton_bar}")
        #     LHE_dilepton_mass =  (LHE_lepton_barless +LHE_lepton_bar).mass
        #     if self.test:
        #         print(f"copperhead2 EventProcessor LHE_dilepton_mass: \n{ak.to_numpy(LHE_dilepton_mass)}")
        #     LHE_filter = ak.flatten(((LHE_dilepton_mass > 100) & (LHE_dilepton_mass < 200)))
        #     LHE_filter = ak.fill_none(LHE_filter, value=False) 
        #     LHE_filter = (LHE_filter== False) # we want True to indicate that we want to keep the event
        #     # print(f"copperhead2 EventProcessor LHE_filter[32]: \n{ak.to_numpy(LHE_filter[32])}")
        #     if self.test:
        #         print(f"copperhead2 EventProcessor LHE_filter: \n{ak.to_numpy(LHE_filter)}")
        #     # event_filter = event_filter & LHE_filter
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

            
        # # Apply HLT to both Data and MC
        # for HLT_str in self.config["hlt"]:
        #     event_filter = event_filter & events.HLT[HLT_str]
        # # if self.test:
        # #     print(f"copperhead2 EventProcessor events.HLT.IsoMu24: \n {ak.to_numpy(events.HLT.IsoMu24)}")
        # #     print(f"copperhead2 EventProcessor ak.sum(events.HLT.IsoMu24): \n {ak.sum(events.HLT.IsoMu24)}")
        # #     print(f"copperhead2 EventProcessor event_filter: \n {ak.to_numpy(event_filter)}")

        # if events.metadata["is_mc"]:
        #     lumi_mask = ak.ones_like(event_filter)

        
        # else:
        #     lumi_info = LumiMask(self.config["lumimask"])
        #     lumi_mask = lumi_info(events.run, events.luminosityBlock)
        #     if self.test:
        #         print(f"copperhead2 EventProcessor lumi_mask: \n {ak.to_numpy(lumi_mask)}")



        
        # # # ------------------------------------------------------------#
        # # # Apply lumimask, genweights, PU weights
        # # # and L1 prefiring weights
        # # # ------------------------------------------------------------#
        # # weight_ones = ak.ones_like(event_filter)
        # # # weight_ones = ak.ones_like(events.Muon.pt[:,0]) # get 1D array of filtered events
        # # self.weight_collection = Weights(weight_ones)
        # # if events.metadata["is_mc"]:
        # #     # For MC: Apply gen.weights, pileup weights, lumi weights,
        # #     # L1 prefiring weights
        # #     genweight = events.genWeight
        # #     self.weight_collection.add_weight("genwgt", genweight)
        # #     print(f"weight_collection genwgt info: \n  {self.weight_collection.get_info()}")
        # #     print(f"weight_collection genwgt wgts: \n  {self.weight_collection.wgts}")
        # #     self.weight_collection.add_weight("lumi", 0.03576104036357644) # hard code for now
        # #     print(f"weight_collection lumi info: \n  {self.weight_collection.get_info()}")
        # #     print(f"weight_collection lumi wgts: \n  {self.weight_collection.wgts}")
        # #     if self.config["do_l1prefiring_wgts"] and ("L1PreFiringWeight" in df.fields):
        # #     # if True:
        # #         L1_nom = events.L1PreFiringWeight.Nom
        # #         # L1_up = events.L1PreFiringWeight.Up
        # #         # L1_down = events.L1PreFiringWeight.Dn
        # #         self.weight_collection.add_weight("l1prefiring_wgt", L1_nom)
        # #         print(f"weight_collection l1prefiring_wgt info: \n  {self.weight_collection.get_info()}")

        
        # if events.metadata["is_mc"]:
        #     #apply PU reweighting b4 event filtering, and keep pu_wgts until we finalize event_filter
        #     pu_wgts = pu_evaluator(
        #                 self.config,
        #                 events.Pileup.nTrueInt,
        #                 test=self.test
        #         )
        #     if self.test:
        #         print(f"copperhead2 EventProcessor events.Pileup.nTrueInt: \n  {ak.to_numpy(events.Pileup.nTrueInt)}")
        #         for key in pu_wgts.keys():
        #             print(f"copperhead2 EventProcessor pu_wgts {key} b4: \n  {ak.to_numpy(pu_wgts[key])}")
            

           
        
        # # # NOTE: this portion of code below is commented out bc original copperhead doesn't filter out event until the very end.
        # # # however, once everything is validated, filtering out events b4 any events would save computational time
        # # ----------------------------------------------------------------------------------
        # # # Filter out the events to ignore corrections (ie rochester, fsr recovery and geofit)
        # # print(f"copperhead2 EventProcessor len(events) b4: {len(events)}")
        # # events = events[event_filter]
        # # print(f"copperhead2 EventProcessor len(events) after: {len(events)}")
        # #-------------------------------------------------------------------------

        
        # # Save raw variables before computing any corrections
        # # rochester and geofit corrects pt only, but fsr_recovery changes all vals below
        # events["Muon", "pt_raw"] = events.Muon.pt
        # events["Muon", "eta_raw"] = events.Muon.eta
        # events["Muon", "phi_raw"] = events.Muon.phi
        # events["Muon", "pfRelIso04_all_raw"] = events.Muon.pfRelIso04_all
        # # --------------------------------------------------------
        # # # Apply Rochester correction
        # # if self.config["do_roccor"]:
        # #     apply_roccor(events, self.config["roccor_file"], events.metadata["is_mc"])
        # #     events["Muon", "pt"] = events.Muon.pt_roch
        # # # FSR recovery
        # # if self.config["do_fsr"]:
        # #     applied_fsr = fsr_recovery(events)
        # #     events["Muon", "pt"] = events.Muon.pt_fsr
        # #     events["Muon", "eta"] = events.Muon.eta_fsr
        # #     events["Muon", "phi"] = events.Muon.phi_fsr
        # #     events["Muon", "pfRelIso04_all"] = events.Muon.iso_fsr
        # # else:
        # #     events["Muon", "pt_fsr"] = events.Muon.pt
        # #     events["Muon", "eta_fsr"] = events.Muon.eta
        # #     events["Muon", "phi_fsr"] = events.Muon.phi
        # #     events["Muon", "iso_fsr"] = events.Muon.pfRelIso04_all
        # # # apply Beam constraint or geofit or nothing if neither
        # # if self.config["do_beamConstraint"] and ("bsConstrainedChi2" in events.Muon.fields): # beamConstraint overrides geofit
        # #     print(f"doing beam constraint")
        # #     print(f"events.Muon.fields: {events.Muon.fields}")
        # #     BSConstraint_mask = (
        # #         (events.Muon.bsConstrainedChi2 <30)
        # #     )
        # #     BSConstraint_mask = ak.fill_none(BSConstraint_mask, False)
        # #     events["Muon", "pt"] = ak.where(BSConstraint_mask, events.Muon.bsConstrainedPt, events.Muon.pt)
        # #     events["Muon", "ptErr"] = ak.where(BSConstraint_mask, events.Muon.bsConstrainedPtErr, events.Muon.ptErr)
        # # else:
        # #     if self.config["do_geofit"] and ("dxybs" in events.Muon.fields):
        # #         # print(f"doing geofit")
        # #         apply_geofit(events, self.config["year"], ~applied_fsr)
        # #         events["Muon", "pt"] = events.Muon.pt_gf
        # #     else: 
        # #         # print(f"doing neither beam constraint nor geofit")
        # #         pass

        # # --------------------------------------------------------
        
        # # --------------------------------------------------------#
        # # Select muons that pass pT, eta, isolation cuts,
        # # muon ID and quality flags
        # # Select events with 2 good muons, no electrons,
        # # passing quality cuts and at least one good PV
        # # --------------------------------------------------------#

        # # Apply event quality flags
        # evnt_qual_flg_selection = ak.ones_like(event_filter)
        # for evt_qual_flg in self.config["event_flags"]:
        #     evnt_qual_flg_selection = evnt_qual_flg_selection & events.Flag[evt_qual_flg]

        
        # # muon_id = "mediumId" if "medium" in self.config["muon_id"] else "looseId"
        # # print(f"copperhead2 EventProcessor muon_id: {muon_id}")
        # muon_selection = (
        #     (events.Muon.pt_raw > self.config["muon_pt_cut"])
        #     & (abs(events.Muon.eta_raw) < self.config["muon_eta_cut"])
        #     & (events.Muon.pfRelIso04_all < self.config["muon_iso_cut"])
        #     # & events.Muon[muon_id]
        #     & events.Muon[self.config["muon_id"]]
        # )
        # if self.test:
        #     print(f"copperhead2 EventProcessor evnt_qual_flg_selection long: \n {ak.to_numpy((evnt_qual_flg_selection))}")
        #     print(f"copperhead2 EventProcessor muon_selection[44]: \n {muon_selection[44]}")
        #     print(f"copperhead2 EventProcessor muon_selection: \n {muon_selection}")
        #     print(f"copperhead2 EventProcessor muon_selection long: \n {ak.to_numpy(ak.flatten(muon_selection))}")
        
        # # count muons that pass the general cut 
        # nmuons = ak.num(events.Muon[muon_selection], axis=1)
        
        # # Find opposite-sign muons
        # mm_charge = ak.prod(events.Muon.charge[muon_selection], axis=1)
        

        # # Veto events with good quality electrons; VBF and ggH categories need zero electrons
        # electron_selection = (
        #     (events.Electron.pt > self.config["electron_pt_cut"])
        #     & (abs(events.Electron.eta) < self.config["electron_eta_cut"])
        #     & events.Electron[self.config["electron_id"]]
        # )
        # electron_veto = (ak.num(events.Electron[electron_selection], axis=1) == 0)

        
        # if self.test:
        #     print(f"copperhead2 EventProcessor nmuons long: \n {pd.DataFrame(ak.to_numpy(nmuons)).to_string()}")
        #     print(f"copperhead2 EventProcessor mm_charge long: \n {ak.to_numpy(mm_charge)}")
        #     print(f'processor electron_selection : \n {((electron_selection))}')
        #     print(f'processor electron_selection long: \n {ak.to_numpy(ak.flatten(electron_selection))}')
        #     # print(f'processor electron_selection long: \n {ak.to_numpy(ak.flatten(electron_selection))}')
        #     print(f"copperhead2 EventProcessor electron_veto long: \n {pd.DataFrame(ak.to_numpy(electron_veto)).to_string()}")
        #     print(f"copperhead2 EventProcessor b4 selection ak.sum(event_filter): \n {ak.sum(event_filter)}")
        #     print(f"copperhead2 EventProcessor b4 selection ak.sum((nmuons == 2)): \n {ak.sum((nmuons == 2))}")
        #     print(f"copperhead2 EventProcessor b4 selection ak.sum(lumi_mask): \n {ak.sum(lumi_mask)}")
        #     print(f"copperhead2 EventProcessor b4 selection ak.sum((evnt_qual_flg_selection > 0)): \n {ak.sum((evnt_qual_flg_selection > 0))}")
        #     print(f"copperhead2 EventProcessor b4 selection ak.sum((mm_charge == -1): \n {ak.sum((mm_charge == -1))}")
        #     print(f"copperhead2 EventProcessor b4 selection ak.sum(electron_veto: \n {ak.sum(electron_veto)}")
        #     print(f"copperhead2 EventProcessor b4 selection ak.sum(good_pv: \n {ak.sum((events.PV.npvsGood > 0))}")


        
        # event_filter = (
        #         event_filter
        #         & lumi_mask
        #         & (evnt_qual_flg_selection > 0)
        #         & (nmuons == 2)
        #         & (mm_charge == -1)
        #         & electron_veto
        #         & (events.PV.npvsGood > 0) # number of good primary vertex cut

        # )
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
        # # --------------------------------------------------------#
        # # Select events with muons passing leading pT cut
        # # --------------------------------------------------------#

        # # Events where there is at least one muon passing
        # # leading muon pT cut
        # pass_leading_pt = events.Muon[:,:1].pt_raw > self.config["muon_leading_pt"]
        # pass_leading_pt = ak.fill_none(pass_leading_pt, value=False) 
        # pass_leading_pt = ak.sum(pass_leading_pt, axis=1)
        

        # event_filter = event_filter & (pass_leading_pt >0)
        
        
        
        # # filter out bad events since we're calculating delta_r
        # events = events[event_filter==True]

        # if self.test:
            
        #     print(f"copperhead2 EventProcessor pass_leading_pt: \n {pass_leading_pt}")
        #     print(f"copperhead2 EventProcessor after leading pt cut event_filter long: \n {ak.to_dataframe(event_filter).to_string()}")
        #     print(f"copperhead2 EventProcessor after leading pt cut ak.sum(event_filter): \n {ak.sum(event_filter)}")
        #     print(f"copperhead2 EventProcessor events.Muon: \n {ak.num(events.Muon, axis=1)}")


        # # now fill in the weights HERE
        
        # # --------------------------------------------------------#
        # # Fill dimuon and muon variables
        # # --------------------------------------------------------#

        # # ---------------------------------------------------------
        # # TODO: find out why we don't filter out bad events right now via
        # # even_selection column, since fill muon is computationally exp
        # # Last time I checked there was some errors on LHE correction shape mismatch
        # # ---------------------------------------------------------
        # mu1 = events.Muon[:,0]
        # mu2 = events.Muon[:,1]
        # dimuon_dR = mu1.delta_r(mu2)
        # dimuon_dEta = abs(mu1.eta - mu2.eta)
        # dimuon_dPhi = abs(mu1.delta_phi(mu2))
        # dimuon = mu1+mu2
        # # fill in pd Dataframe as placeholder. Should be fine since we don't need jagged arrays
        # dimuon_mass_resolution = self.mass_resolution(events, test_mode=self.test)
        # rel_dimuon_ebe_mass_res = dimuon_mass_resolution/dimuon.mass
        # dimuon_cos_theta_cs, dimuon_phi_cs = cs_variables(mu1,mu2)

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
        # self.prepare_jets(events)

        # if self.test:
        #     print(f"copperhead2 EventProcessor events.Jet.rho: \n {events.Jet.rho}")
        #     print(f"copperhead2 EventProcessor events.Jet.rho long: \n {ak.to_numpy(ak.flatten(events.Jet.rho))}")   
        #     print(f'copperheadV2 EventProcessor jets.pt b4 apply_jec long: \n {ak.to_numpy(ak.flatten(events.Jet.pt))}')
        #     print(f'copperheadV2 EventProcessor jets.eta b4 apply_jec long: \n {ak.to_numpy(ak.flatten(events.Jet.eta))}')
        #     print(f'copperheadV2 EventProcessor jets.phi b4 apply_jec long: \n {ak.to_numpy(ak.flatten(events.Jet.phi))}')
        #     print(f'copperheadV2 EventProcessor jets.mass b4 apply_jec long: \n {ak.to_numpy(ak.flatten(events.Jet.mass))}')
        
        # jets = events.Jet
        # self.jec_factories_mc, self.jec_factories_data = get_jec_factories(
        #     self.config["jec_parameters"], 
        #     test_mode=self.test
        # )
        
        # do_jec = False

        # # # We only need to reapply JEC for 2018 data
        # # # (unless new versions of JEC are released)
        # # is_data = not events.metadata["is_mc"]
        # # if is_data and ("2018" in self.config["year"]):
        # #     do_jec = True

        # # do_jecunc = self.config["do_jecunc"]
        # # do_jerunc = self.config["do_jerunc"]
        # #testing 
        # do_jecunc = False
        # do_jerunc = False
        

        # # ------------------------------------------------------------#
        # # Apply JEC, get JEC and JER variations
        # # ------------------------------------------------------------#
        # if do_jec:
        #     if events.metadata["is_mc"]:
        #         factory = self.jec_factories["jec"]
        #     else:
        #         for run in self.config["jec_parameters"]["runs"]:
        #             if run in events.metadata["dataset"]:
        #                 factory = self.jec_factories_data[run]
        #     if self.test:
        #         print("jets build")
        #     jets = factory.build(jets)
        #     if self.test :
        #         jets = jets.compute() # can't circumvent JEC only being on dask distributed
        # # TODO: only consider nuisances that are defined in run parameters
        # # Compute JEC uncertainties
        # if events.metadata["is_mc"] and do_jecunc:
        #     jets = self.jec_factories["junc"].build(jets)
    
        # # Compute JER uncertainties
        # if events.metadata["is_mc"] and do_jerunc:
        #     jets = self.jec_factories["jer"].build(jets)
        
        # # TODO: JER nuisances

        


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
        # # Apply lumimask, genweights, PU weights
        # # and L1 prefiring weights
        # # ------------------------------------------------------------#
        # if events.metadata["is_mc"]:
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
            
        #     # gen Weight
        #     genweight = events.genWeight
        #     self.weight_collection.add_weight("genwgt", genweight)
        #     if self.test:
        #         print(f"weight_collection genwgt info: \n  {self.weight_collection.get_info()}")
        #         print(f"weight_collection genwgt wgts: \n  {self.weight_collection.wgts}")
            
        #     # Lumi Weight
            
        #     dataset = events.metadata['dataset']
        #     cross_section = self.config["cross_sections"][dataset]
        #     totalGenWgts = events.metadata['sumGenWgts']
        #     integrated_lumi = self.config["integrated_lumis"]
        #     lumi_weight = cross_section * integrated_lumi/ totalGenWgts
        #     self.config["lumi_weight"] = lumi_weight

        #     # lumi_weight = 0.03576104036357644*ak.ones_like(weight_ones) # dy 50
        #     # lumi_weight = 2.955104456012521e-05 # ggh 
        #     # lumi_weight = 1.3805388208609223e-05 # vbf 
        #     lumi_weight = self.config["lumi_weight"]
        #     self.weight_collection.add_weight("lumi", lumi_weight) 
        #     if self.test:
        #         print(f"weight_collection integrated_lumi: \n  {integrated_lumi}")
        #         print(f"weight_collection lumi info: \n  {self.weight_collection.get_info()}")
        #         print(f"weight_collection lumi wgts: \n  {self.weight_collection.wgts}")
        #         print(f"weight_collection lumi_weight: \n  {lumi_weight}")
            

        #     # L1 prefiring weights
        #     if self.config["do_l1prefiring_wgts"] and ("L1PreFiringWeight" in df.fields):
        #     # if True:
        #         L1_nom = events.L1PreFiringWeight.Nom
        #         # L1_up = events.L1PreFiringWeight.Up
        #         # L1_down = events.L1PreFiringWeight.Dn
        #         self.weight_collection.add_weight("l1prefiring_wgt", L1_nom)
        #         if self.test:
        #             print(f"weight_collection l1prefiring_wgt info: \n  {self.weight_collection.get_info()}")

        
        # # ------------------------------------------------------------#
        # # Calculate other event weights
        # # ------------------------------------------------------------#
        # pt_variations = (
        #     ["nominal"]
        #     # + jec_pars["jec_variations"]
        #     # + jec_pars["jer_variations"]
        # )
        # if events.metadata["is_mc"]:
        #     """ nnlops crashes in dask awkward
        #     do_nnlops = self.config["do_nnlops"] and ("ggh" in events.metadata["dataset"])
        #     print(f"do_nnlops: {do_nnlops}")
        #     if do_nnlops:
        #     # if True:
        #         nnlopsw = nnlops_weights(events, self.config, events.metadata["dataset"])
        #         self.weight_collection.add_weight("nnlops", nnlopsw)
        #         print(f"weight_collection nnlops info: \n  {self.weight_collection.get_info()}")
        #     """
        #     # else:
        #     #     weights.add_weight("nnlops", how="dummy")
        #     # print(f'copperheadV1 weights.df nnlops: \n {weights.df.to_string()}')
            
        #     # do zpt SF
        #     do_zpt = ('dy' in dataset)
        #     do_zpt
        #     if do_zpt:
        #         zpt_weight = self.evaluator[self.zpt_path](dimuon.pt)
        #         self.weight_collection.add_weight('zpt_wgt', zpt_weight)
        #     """
            
                
        #     #do mu SF
        #     musf_lookup = get_musf_lookup(self.config)
        #     muID, muIso, muTrig = musf_evaluator(
        #         musf_lookup, self.config["year"], events.Muon
        #     )
        #     # print(f'copperheadV2 EventProcessor muTrig["nom"]: \n {ak.to_dataframe(muTrig["nom"]).to_string()}')
        #     # print(f'copperheadV2 EventProcessor muTrig["nom"]: \n {ak.to_numpy(muTrig["nom"])}')
        #     self.weight_collection.add_weight("muID", muID, how="all")
        #     if self.test:
        #         print(f"weight_collection muID info: \n  {self.weight_collection.get_info()}")
        #     self.weight_collection.add_weight("muIso", muIso, how="all")
        #     if self.test:
        #         print(f"weight_collection muIso info: \n  {self.weight_collection.get_info()}")
        #     self.weight_collection.add_weight("muTrig", muTrig, how="all")
        #     if self.test:
        #         print(f"weight_collection muTrig info: \n  {self.weight_collection.get_info()}")
        #     # self.weight_collection.add_weight("muID", muID, how="all")
        #     # self.weight_collection.add_weight("muIso", muIso, how="all")
        #     # self.weight_collection.add_weight("muTrig", muTrig, how="all") 
        #     """
        #     # --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
        #     do_lhe = (
        #         ("LHEScaleWeight" in events.fields)
        #         and ("LHEPdfWeight" in events.fields)
        #         and ("nominal" in pt_variations)
        #     )
        #     do_lhe = False
        #     if self.test:
        #         print(f"do_lhe: {do_lhe}")
        #     if do_lhe:
        #         lhe_ren, lhe_fac = lhe_weights(events, events.metadata["dataset"], self.config["year"])
        #         if self.test:
        #             print(f"weight_collection LHEFac info: \n  {self.weight_collection.get_info()}")
        #             print(f"weight_collection LHEFac info: \n  {self.weight_collection.get_info()}")
        #             # self.weight_collection.add_weight("LHERen", lhe_ren, how="only_vars")
        #             # print(f"weight_collection LHERen info: \n  {self.weight_collection.get_info()}")
        #             # self.weight_collection.add_weight("LHEFac", lhe_fac, how="only_vars")
        #             # print(f"weight_collection LHEFac info: \n  {self.weight_collection.get_info()}")
            
        #     # --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
        #     dataset = events.metadata["dataset"]
        #     do_thu = False
        #     # do_thu = (
        #     #     ("vbf" in dataset)
        #     #     and ("dy" not in dataset)
        #     #     and ("nominal" in pt_variations)
        #     #     and ("stage1_1_fine_cat_pTjet30GeV" in events.HTXS.fields)
        #     # )
        #     if do_thu:
        #         add_stxs_variations(
        #             events,
        #             self.weight_collection,
        #             self.config,
        #         )
        #     if self.test:
        #         print(f"do_thu: {do_thu}")
        #         print(f"weight_collection do_thu info: \n  {self.weight_collection.get_info()}")
        #     # --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
        #     do_pdf = False
        #     # do_pdf = (
        #     #     self.config["do_pdf"]
        #     #     and ("nominal" in pt_variations)
        #     #     and (
        #     #         "dy" in dataset
        #     #         or "ewk" in dataset
        #     #         or "ggh" in dataset
        #     #         or "vbf" in dataset
        #     #     )
        #     #     and ("mg" not in dataset)
        #     # )
        #     if do_pdf:
        #         add_pdf_variations(events, self.weight_collection, self.config, dataset)
        #     if self.test:
        #         print(f"do_pdf: {do_pdf}")
        #         print(f"weight_collection do_pdf info: \n  {self.weight_collection.get_info()}")
# just reading test end
# just reading part 2 start -------------------------        
        # ------------------------------------------------------------#
        # Fill Muon variables and gjet variables
        # ------------------------------------------------------------#
        # print(f"events.Muon.pt: {events.Muon.pt}")
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
        # if events.metadata["is_mc"]:
        #     gjet_dict = {
        #         "gjet_pt" : padded_iso_gjet.pt,
        #         "gjet_eta" : padded_iso_gjet.eta,
        #         "gjet_phi" : padded_iso_gjet.phi,
        #         "gjet_mass" : padded_iso_gjet.mass,
        #         "gjj_mass": gjj.mass,
        #         "gjj_pt" : gjj.pt,
        #         "gjj_eta" : gjj.eta,
        #         "gjj_phi" : gjj.phi,
        #         "gjj_dEta" : gjj_dEta,
        #         "gjj_dPhi" : gjj_dPhi,
        #         "gjj_dR" : gjj_dR,
        #     }
        #     out_dict.update(gjet_dict)
        
        # # ------------------------------------------------------------#
        # # Loop over JEC variations and fill jet variables
        # # ------------------------------------------------------------#
        
        # for variation in pt_variations:
        #     jet_loop_dict = self.jet_loop(
        #         events, 
        #         jets,
        #         variation,
        #         do_jec = do_jec,
        #         do_jecunc = do_jecunc,
        #         do_jerunc = do_jerunc,
        #     )

        # out_dict.update(jet_loop_dict) 
        
        # # fill in the regions
        # mass = dimuon.mass
        # z_peak = ((mass > 76) & (mass < 106))
        # h_sidebands =  ((mass > 110) & (mass < 115.03)) | ((mass > 135.03) & (mass < 150))
        # h_peak = ((mass > 115.03) & (mass < 135.03))
        # region_dict = {
        #     "z_peak" : ak.fill_none(z_peak, value=False),
        #     "h_sidebands" : ak.fill_none(h_sidebands, value=False),
        #     "h_peak" : ak.fill_none(h_peak, value=False),
        # }
        # # print(f"dimuon mass: {ak.to_numpy(mass.compute())}")
        # # if self.test:
        # #     print(f"region_dict: {region_dict}")
        # # else:
        # #     print(f"region_dict: {dask.compute(region_dict)}")
            
        # out_dict.update(region_dict) 

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
        print(f"events b4 filter length: {ak.num(events.Muon.pt, axis=0).compute()}")
        nmuons = ak.num(events.Muon.pt, axis=1)
        event_filter =   nmuons>=1
        events = events[event_filter]
        print(f"events after filter length: {ak.num(events.Muon.pt, axis=0).compute()}")
        out_dict = {
            "mu_pt" : ak.pad_none(events.Muon.pt, 2),
            "nmuons" : nmuons[event_filter],
        }
        #----------------------------
        
        return out_dict
        
    def postprocess(self, accumulator):
        """
        Arbitrary postprocess function that's required to run the processor
        """
        pass

    
    def mass_resolution(self, events, test_mode=False):
        # Returns absolute mass resolution!
        mu1 = events.Muon[:,0]
        mu2 = events.Muon[:,1]
        muon_E = (mu1+mu2).mass /2
        dpt1 = (mu1.ptErr / mu1.pt) * muon_E
        dpt2 = (mu2.ptErr / mu2.pt) * muon_E
        if test_mode:
            print(f"muons mass_resolution dpt1: {dpt1}")
        if "2016" in self.config["year"]:
            yearUL = "2016"
        else:
            yearUL = self.config["year"] #Work around before there are seperate new files for pre and postVFP
        if events.metadata["is_mc"]:
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
            # comment off pt_gen assignment bc I don't see it being used anywhere
            # events["Jet", "pt_gen"] = events.Jet.matched_gen.pt 
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
        variation,
        # do_jec = True, #False
        # do_jecunc = False,
        # do_jerunc = False,
        do_jec = False, #False
        do_jecunc = False,
        do_jerunc = False,
        # dataset,
        # mask,
        # muons,
        # mu1,
        # mu2,
        # jets,
        # weights,
        # numevents,
        # output,
    ):
        # weights = copy.deepcopy(weights)
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
        if self.test:
            print(f"jets.matched_muons: {jets.matched_muons}")
            print(f"type(jets.matched_muons): {type(jets.matched_muons)}")
            print(f"type(jets.matched_muons.pt_fsr): {type(jets.matched_muons.pt_fsr)}")
            print(f"(jets.matched_muons.pt_fsr): {(jets.matched_muons.pt_fsr)}")
            print(f"ak.to_dataframe(jets.matched_muons.pt_fsr): {ak.to_dataframe(jets.matched_muons.pt_fsr)}")
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
        
        if self.test:
            # print(f"jet loop jets.matched_muons: {jets.matched_muons}")
            print(f"jet loop matched_mu_pass after: {matched_mu_pass}")
            print(f"jet loop matched_mu_pt: {matched_mu_pt}")
            print(f"jet loop matched_mu_iso: {matched_mu_iso}")
            print(f"jet loop matched_mu_id: {matched_mu_id}")
            
            # print(f"type(matched_mu_pass): {type(matched_mu_pass)}")
            # print(f"ak.to_dataframe(matched_mu_pass): {ak.to_dataframe(matched_mu_pass)}")
            # print(f"jet loop clean: {clean}")
            print(f"jet loop clean: {ak.to_numpy(ak.flatten(clean))}")
            print(f"jet loop clean sum: {ak.to_numpy(ak.sum(clean, axis=1))}")

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

        # ------------------------------------------------------------#
        # Apply jetID and PUID
        # ------------------------------------------------------------#

        pass_jet_id = jet_id(jets, self.config)
        pass_jet_puid = jet_puid(jets, self.config)

        """
        this code has been disabled by Dmitry
        # Jet PUID scale factors
        # if is_mc and False:  # disable for now
        #     puid_weight = puid_weights(
        #         self.evaluator, self.year, jets, pt_name,
        #         jet_puid_opt, jet_puid, numevents
        #     )
        #     weights.add_weight('puid_wgt', puid_weight)
        """
        # ------------------------------------------------------------#
        # Select jets
        # ------------------------------------------------------------#
        HEMVeto = ak.ones_like(clean) == 1 # 1D array saying True
        if self.config["year"] == "2018":
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
        jet_selection = (
            pass_jet_id
            & pass_jet_puid
            & (jets.qgl > -2)
            & clean
            & (jets.pt > self.config["jet_pt_cut"])
            & (abs(jets.eta) < self.config["jet_eta_cut"])
            & HEMVeto
        )
        # print(f"jet_selection: {jet_selection}")
        # print(f"jets b4 selection: {jets}")
        # print(f"jets._meta b4 selection: {repr(jets._meta)}")
        # print(f"dak.necessary_columns(jets.pt) b4 selection: {dak.necessary_columns(jets.pt)}")
        # jets = jets[jet_selection] # this causes huuuuge memory overflow close to 100 GB. Without it, it goes to around 20 GB
        jets = ak.to_packed(ak.mask(jets, jet_selection))
        # print(f"jets after selection: {jets}")
        # print(f"jets._meta after selection: {str(jets._meta.compute())}")
        # print(f"jet_selection._meta: {str(jet_selection._meta.compute())}")
        # print(f"jets._meta after selection: {repr(jets._meta)}")
        # print(f"jet_selection._meta: {repr(jet_selection._meta)}")
        # print(f"dak.necessary_columns(jets.pt) after selection: {dak.necessary_columns(jets.pt)}")
        # 
        
        # jets = ak.where(jet_selection, jets, None)
        muons = events.Muon 
        njets = ak.num(jets, axis=1)

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
        padded_jets = ak.pad_none(jets, 2)
        jet1 = padded_jets[:,0]
        jet2 = padded_jets[:,1]
        dijet = jet1+jet2
        # print(f"dijet: {dijet}")
        jj_dEta = abs(jet1.eta - jet2.eta)
        jj_dPhi = abs(jet1.delta_phi(jet2))
        dimuon = muons[:,0] + muons[:,1]
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
        # return jet_loop_out_dict
        
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
        if variation == "nominal":
            for cutout in cutouts:
                sj_out = fill_softjets(events, jets, muons, cutout, test_mode=self.test)
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
        # # TODO: qgl seems to be only for vbf, double check that
        # if is_mc and variation == "nominal":
        #     # --- QGL weights --- #
        #     isHerwig = "herwig" in events.metadata['dataset']

        #     qgl_wgts = qgl_weights(jet1, jet2, njets, isHerwig)
        #     self.weight_collection.add_weight("qgl_wgt", qgl_wgts, how="all")
        #     # print(f"type(qgl_wgts) : \n {type(qgl_wgts)}")
        #     # print(f"qgl_wgts : \n {qgl_wgts}")

        #     # --- Btag weights --- #
        #     bjet_sel_mask = output.event_selection #& two_jets & vbf_cut
        #     btag_systs = self.config["btag_systs"]
        #     btag_json =  correctionlib.CorrectionSet.from_file(self.parameters["btag_sf_json"],)
        #     btag_wgt, btag_syst = btag_weights_json(
        #         self, btag_systs, jets, weights, bjet_sel_mask, btag_json
        #     )
        #     weights.add_weight("btag_wgt", btag_wgt)

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
        if self.test:
            print(f"jet loop nBtagLoose: {nBtagLoose}")
            print(f"jet loop nBtagMedium: {nBtagMedium}")
            print(f"jet loop jet_loop_out_dict.keys(): {jet_loop_out_dict.keys()}")
        

        # --------------------------------------------------------------#
        # Fill outputs
        # --------------------------------------------------------------#

    #     variables.update({"wgt_nominal": weights.get_weight("nominal")})

    #     # All variables are affected by jet pT because of jet selections:
    #     # a jet may or may not be selected depending on pT variation.

    #     for key, val in variables.items():
    #         output.loc[:, pd.IndexSlice[key, variation]] = val

        return jet_loop_out_dict
    