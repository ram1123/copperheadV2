import coffea.processor as processor
import awkward as ak
import numpy as np
from typing import Union, TypeVar, Tuple
from corrections.rochester import apply_roccor
from corrections.fsr_recovery import fsr_recovery
from corrections.geofit import apply_geofit
import json
coffea_nanoevent = TypeVar('coffea_nanoevent') 
import pandas as pd # just for debugging

class EventProcessor(processor.ProcessorABC):
    def __init__(self, config_path: str,**kwargs):
        """
        TODO: replace all of these with self.config dict variable which is taken from a
        pre-made json file
        """
        with open(config_path) as file:
            self.config = json.loads(file.read())

        # self.config = json.loads(config_path)
        # print(f"copperhead proccesor self.config b4 update: \n {self.config}")
        dict_update = {
            # "hlt" :["IsoMu24"],
            "do_trigger_match" : False,
            "do_roccor" : False,# True
            "do_fsr" : True,
            "do_geofit" : True,
            "year" : "2018",
            "rocorr_file_path" : "data/roch_corr/RoccoR2018.txt",
        }
        self.config.update(dict_update)
        # print(f"copperhead proccesor self.config after update: \n {self.config}")
        self.test = True
        
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
        event_filter = ak.ones_like(events.HLT.IsoMu24) # 1D array to be used to filter out bad events
        if events.metadata['dataset'] == 'dy_M-50': # if dy_M-50, apply LHE cut
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
            print(f"copperhead2 EventProcessor LHE_lepton_bar: {LHE_lepton_bar}")
            LHE_dilepton_mass =  (LHE_lepton_barless +LHE_lepton_bar).mass
            print(f"copperhead2 EventProcessor LHE_dilepton_mass: \n{ak.to_numpy(LHE_dilepton_mass)}")
            LHE_filter = ak.flatten(((LHE_dilepton_mass > 100) & (LHE_dilepton_mass < 200)))
            LHE_filter = ak.fill_none(LHE_filter, value=False) 
            LHE_filter = (LHE_filter== False) # we want True to indicate that we want to keep the event
            # print(f"copperhead2 EventProcessor LHE_filter[32]: \n{ak.to_numpy(LHE_filter[32])}")
            print(f"copperhead2 EventProcessor LHE_filter: \n{ak.to_numpy(LHE_filter)}")
            event_filter = event_filter & LHE_filter
        
        if self.config["do_trigger_match"]:
            """
            Apply trigger matching. We take the two leading pT reco muons and try to have at least one of the muons
            to be matched with the trigger object that fired our HLT. If none of the muons did it, then we reject the 
            event. This operation is computationally expensive, so perhaps worth considering not implementing it if 
            it has neglible impact
            
            TODO: The impact this operation has onto the statistics is supposedly very low, but I have to check that
            """
            # event_filter = event_filter & trigger_match
            pass # to be filled in later

        print(f"copperhead2 EventProcessor events.HLT.IsoMu24: \n {ak.to_numpy(events.HLT.IsoMu24)}")
        # Apply HLT to both Data and MC
        for HLT_str in self.config["hlt"]:
            event_filter = event_filter & events.HLT[HLT_str]
        # event_filter = event_filter & events.HLT.IsoMu24
        print(f"copperhead2 EventProcessor event_filter: \n {ak.to_numpy(event_filter)}")

        if events.metadata["is_mc"]:
            lumi_mask = ak.ones_like(event_filter)

        # ------------------------------------------------------------#
        # Apply lumimask, genweights, PU weights
        # and L1 prefiring weights
        # ------------------------------------------------------------#

        
        # NOTE: this portion of code below is commented out bc original copperhead doesn't filter out event until the very end.
        # however, once everything is validated, filtering out events b4 any events would save computational time
        """
        # Filter out the events to ignore corrections (ie rochester, fsr recovery and geofit)
        print(f"copperhead2 EventProcessor len(events) b4: {len(events)}")
        events = events[event_filter]
        print(f"copperhead2 EventProcessor len(events) after: {len(events)}")
        """

        
        # Save raw variables before computing any corrections
        # rochester and geofit corrects pt only, but fsr_recovery changes all vals below
        events["Muon", "pt_raw"] = events.Muon.pt
        events["Muon", "eta_raw"] = events.Muon.eta
        events["Muon", "phi_raw"] = events.Muon.phi
        events["Muon", "pfRelIso04_all_raw"] = events.Muon.pfRelIso04_all
        
        # Apply Rochester correction
        if self.config["do_roccor"]:
            apply_roccor(events, self.config["rocorr_file_path"], True)
            events["Muon", "pt"] = events.Muon.pt_roch
        # FSR recovery
        if self.config["do_fsr"]:
            applied_fsr = fsr_recovery(events)
            events["Muon", "pt"] = events.Muon.pt_fsr
            events["Muon", "eta"] = events.Muon.eta_fsr
            events["Muon", "phi"] = events.Muon.phi_fsr
            events["Muon", "pfRelIso04_all"] = events.Muon.iso_fsr
        # geofit
        if self.config["do_geofit"] and ("dxybs" in events.Muon.fields):
            apply_geofit(events, self.config["year"], ~applied_fsr)
            events["Muon", "pt"] = events.Muon.pt_gf


        # --------------------------------------------------------#
        # Select muons that pass pT, eta, isolation cuts,
        # muon ID and quality flags
        # Select events with 2 OS muons, no electrons,
        # passing quality cuts and at least one good PV
        # --------------------------------------------------------#

        # Apply event quality flags
        evnt_qual_flg_selection = ak.ones_like(event_filter)
        for evt_qual_flg in self.config["event_flags"]:
            evnt_qual_flg_selection = evnt_qual_flg_selection & events.Flag[evt_qual_flg]

        print(f"copperhead2 EventProcessor evnt_qual_flg_selection long: \n {ak.to_numpy((evnt_qual_flg_selection))}")
        
        # muon_id = "mediumId" if "medium" in self.config["muon_id"] else "looseId"
        # print(f"copperhead2 EventProcessor muon_id: {muon_id}")
        muon_selection = (
            (events.Muon.pt_raw > self.config["muon_pt_cut"])
            & (abs(events.Muon.eta_raw) < self.config["muon_eta_cut"])
            & (events.Muon.pfRelIso04_all < self.config["muon_iso_cut"])
            # & events.Muon[muon_id]
            & events.Muon[self.config["muon_id"]]
        )
        print(f"copperhead2 EventProcessor muon_selection[44]: \n {muon_selection[44]}")
        print(f"copperhead2 EventProcessor muon_selection: \n {muon_selection}")
        print(f"copperhead2 EventProcessor muon_selection long: \n {ak.to_numpy(ak.flatten(muon_selection))}")
        
        # count muons that pass the general cut
        nmuons = ak.num(events.Muon[muon_selection], axis=1)
        print(f"copperhead2 EventProcessor nmuons long: \n {pd.DataFrame(ak.to_numpy(nmuons)).to_string()}")
        # Find opposite-sign muons
        mm_charge = ak.prod(events.Muon.charge, axis=1)
        print(f"copperhead2 EventProcessor mm_charge long: \n {ak.to_numpy(mm_charge)}")

        # Veto events with good quality electrons; VBF and ggH categories need zero electrons
        electron_selection = (
            (events.Electron.pt > self.config["electron_pt_cut"])
            & (abs(events.Electron.eta) < self.config["electron_eta_cut"])
            & events.Electron[self.config["electron_id"]]
        )
        print(f'processor electron_selection : \n {((electron_selection))}')
        print(f'processor electron_selection long: \n {ak.to_numpy(ak.flatten(electron_selection))}')
        electron_veto = (ak.num(events.Electron[electron_selection], axis=1) == 0)
        print(f"copperhead2 EventProcessor electron_veto long: \n {pd.DataFrame(ak.to_numpy(electron_veto)).to_string()}")



        event_filter = (
                event_filter
                & lumi_mask
                & (evnt_qual_flg_selection > 0)
                & (nmuons == 2)
                & (mm_charge == -1)
                & electron_veto
                & (events.PV.npvsGood > 0) # number of good primary vertex cut

        )
        print(f"copperhead2 EventProcessor b4 leading pt cut event_filter long: \n {pd.DataFrame(ak.to_numpy(event_filter)).to_string()}")

        # --------------------------------------------------------#
        # Select events with muons passing leading pT cut
        # --------------------------------------------------------#

        # Events where there is at least one muon passing
        # leading muon pT cut
        pass_leading_pt = events.Muon[:,:1].pt_raw > self.config["muon_leading_pt"]
        pass_leading_pt = ak.fill_none(pass_leading_pt, value=False) 
        pass_leading_pt = ak.sum(pass_leading_pt, axis=1)
        print(f"copperhead2 EventProcessor pass_leading_pt: \n {pass_leading_pt}")

        event_filter = event_filter & (pass_leading_pt >0)
        
        print(f"copperhead2 EventProcessor after leading pt cut event_filter long: \n {ak.to_dataframe(event_filter)}")
        
        # filter out bad events since we're calculating delta_r
        events = events[event_filter]
        
        # --------------------------------------------------------#
        # Fill dimuon and muon variables
        # --------------------------------------------------------#

        """
        TODO: find out why we don't filter out bad events right now via
        even_selection column, since fill muon is computationally exp
        Last time I checked there was some errors on LHE correction shape mismatch
        """
        # fill_muons(self, output, mu1, mu2, is_mc)
        mu1 = events.Muon[:,0]
        mu2 = events.Muon[:,1]
        delta_r = mu1.delta_r(mu2)
        delta_eta = abs(mu1.eta -mu2.eta)
        delta_phi = abs(mu1.phi -mu2.phi)
        
        return events
        
    def postprocess(self, accumulator):
        """
        Arbitrary postprocess function that's required to run the processor
        """
        pass

