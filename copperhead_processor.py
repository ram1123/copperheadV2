import coffea.processor as processor
import awkward as ak
import numpy as np
from typing import Union, TypeVar, Tuple
from corrections.rochester import apply_roccor
from corrections.fsr_recovery import fsr_recovery
from corrections.geofit import apply_geofit
import json
coffea_nanoevent = TypeVar('coffea_nanoevent') 


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
        
    def process(self, events: coffea_nanoevent):
        """
        TODO: do LHE cut after HLT and trigger match event filtering to save computation
        """

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
        
        return events
    def postprocess(self, accumulator):
        """
        Arbitrary postprocess function that's required to run the processor
        """
        pass