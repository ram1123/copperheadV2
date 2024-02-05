import coffea.processor as processor
from coffea.lookup_tools import extractor
import awkward as ak
import numpy as np
from typing import Union, TypeVar, Tuple
from corrections.rochester import apply_roccor
from corrections.fsr_recovery import fsr_recovery
from corrections.geofit import apply_geofit
import json
from corrections.jet import get_jec_factories
from coffea.lumi_tools import LumiMask
import pandas as pd # just for debugging

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
            "do_trigger_match" : True,
            "do_roccor" : False,# True
            "do_fsr" : True,
            "do_geofit" : True,
            "year" : "2018",
            "rocorr_file_path" : "data/roch_corr/RoccoR2018.txt",
        }
        self.config.update(dict_update)
        # print(f"copperhead proccesor self.config after update: \n {self.config}")
        self.test = True# False

        # --- Evaluator
        extractor_instance = extractor()
        # Calibration of event-by-event mass resolution
        for mode in ["Data", "MC"]:
            if "2016" in self.config["year"]:
                yearstr = "2016"
            else:
                yearstr=self.config["year"] #Work around before there are seperate new files for pre and postVFP
            label = f"res_calib_{mode}_{yearstr}"
            path = self.config["res_calib_path"]
            file_path = f"{path}/{label}.root"
            extractor_instance.add_weight_sets([f"{label} {label} {file_path}"])
        extractor_instance.finalize()
        self.evaluator = extractor_instance.make_evaluator()

        # # prepare lookup tables for all kinds of corrections
        # self.prepare_lookups()

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
            if self.test:
                print(f"copperhead2 EventProcessor LHE_lepton_bar: {LHE_lepton_bar}")
            LHE_dilepton_mass =  (LHE_lepton_barless +LHE_lepton_bar).mass
            if self.test:
                print(f"copperhead2 EventProcessor LHE_dilepton_mass: \n{ak.to_numpy(LHE_dilepton_mass)}")
            LHE_filter = ak.flatten(((LHE_dilepton_mass > 100) & (LHE_dilepton_mass < 200)))
            LHE_filter = ak.fill_none(LHE_filter, value=False) 
            LHE_filter = (LHE_filter== False) # we want True to indicate that we want to keep the event
            # print(f"copperhead2 EventProcessor LHE_filter[32]: \n{ak.to_numpy(LHE_filter[32])}")
            if self.test:
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
            isoMu_filterbit = 2
            mu_id = 13
            pt_threshold = 24 
            dr_threshold = 0.1
            IsoMu24_muons = (events.TrigObj.id == mu_id) &  \
                    ((events.TrigObj.filterBits & isoMu_filterbit) == isoMu_filterbit) & \
                    (events.TrigObj.pt > pt_threshold)
            #check the first two leading muons match any of the HLT trigger objs. if neither match, reject event
            padded_muons = ak.pad_none(events.Muon, 2) # pad in case we have only one muon or zero in an event
            print(f"copperhead2 EventProcessor padded_muons: \n {padded_muons}")
            mu1 = padded_muons[:,0]
            mu2 = padded_muons[:,1]
            mu1_match = (mu1.delta_r(events.TrigObj[IsoMu24_muons]) < dr_threshold) & \
                (mu1.pt > pt_threshold)
            mu1_match = ak.sum(mu1_match, axis=1)
            mu1_match = ak.fill_none(mu1_match, value=False)
            mu2_match = (mu2.delta_r(events.TrigObj[IsoMu24_muons]) < dr_threshold) & \
                (mu2.pt > pt_threshold)
            mu2_match =  ak.sum(mu2_match, axis=1)
            mu2_match = ak.fill_none(mu2_match, value=False)
            
            trigger_match = (mu1_match >0) | (mu2_match > 0)
            if self.test:
                print(f"copperhead2 EventProcessor mu1_match: \n {mu1_match}")
                print(f"copperhead2 EventProcessor mu2_match: \n {mu2_match}")
                print(f"copperhead2 EventProcessor trigger_match: \n {trigger_match}")
            print(f"copperhead2 EventProcessor events.HLT.IsoMu24 and trigger_match mismatch count: \n {ak.sum(events.HLT.IsoMu24 != trigger_match)}")
            event_filter = event_filter & trigger_match

            
        # Apply HLT to both Data and MC
        for HLT_str in self.config["hlt"]:
            event_filter = event_filter & events.HLT[HLT_str]
        if self.test:
            print(f"copperhead2 EventProcessor events.HLT.IsoMu24: \n {ak.to_numpy(events.HLT.IsoMu24)}")
            print(f"copperhead2 EventProcessor event_filter: \n {ak.to_numpy(event_filter)}")

        if events.metadata["is_mc"]:
            lumi_mask = ak.ones_like(event_filter)
        else:
            lumi_info = LumiMask(self.config["lumimask"])
            lumi_mask = lumi_info(events.run, events.luminosityBlock)
            print(f"copperhead2 EventProcessor lumi_mask: \n {ak.to_numpy(lumi_mask)}")

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
            apply_roccor(events, self.config["rocorr_file_path"], events.metadata["is_mc"])
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

        
        # muon_id = "mediumId" if "medium" in self.config["muon_id"] else "looseId"
        # print(f"copperhead2 EventProcessor muon_id: {muon_id}")
        muon_selection = (
            (events.Muon.pt_raw > self.config["muon_pt_cut"])
            & (abs(events.Muon.eta_raw) < self.config["muon_eta_cut"])
            & (events.Muon.pfRelIso04_all < self.config["muon_iso_cut"])
            # & events.Muon[muon_id]
            & events.Muon[self.config["muon_id"]]
        )
        if self.test:
            print(f"copperhead2 EventProcessor evnt_qual_flg_selection long: \n {ak.to_numpy((evnt_qual_flg_selection))}")
            print(f"copperhead2 EventProcessor muon_selection[44]: \n {muon_selection[44]}")
            print(f"copperhead2 EventProcessor muon_selection: \n {muon_selection}")
            print(f"copperhead2 EventProcessor muon_selection long: \n {ak.to_numpy(ak.flatten(muon_selection))}")
        
        # count muons that pass the general cut
        nmuons = ak.num(events.Muon[muon_selection], axis=1)
        
        # Find opposite-sign muons
        mm_charge = ak.prod(events.Muon.charge, axis=1)
        

        # Veto events with good quality electrons; VBF and ggH categories need zero electrons
        electron_selection = (
            (events.Electron.pt > self.config["electron_pt_cut"])
            & (abs(events.Electron.eta) < self.config["electron_eta_cut"])
            & events.Electron[self.config["electron_id"]]
        )
        electron_veto = (ak.num(events.Electron[electron_selection], axis=1) == 0)

        
        if self.test:
            print(f"copperhead2 EventProcessor nmuons long: \n {pd.DataFrame(ak.to_numpy(nmuons)).to_string()}")
            print(f"copperhead2 EventProcessor mm_charge long: \n {ak.to_numpy(mm_charge)}")
            print(f'processor electron_selection : \n {((electron_selection))}')
            print(f'processor electron_selection long: \n {ak.to_numpy(ak.flatten(electron_selection))}')
        
        
            # print(f'processor electron_selection long: \n {ak.to_numpy(ak.flatten(electron_selection))}')
            print(f"copperhead2 EventProcessor electron_veto long: \n {pd.DataFrame(ak.to_numpy(electron_veto)).to_string()}")

        # if self.test:
        #     # save electrons in for plotting
        #     electrons = events.Electron[electron_selection]
        #     placeholder =  pd.DataFrame({
        #         'el_pt': ak.to_numpy(ak.flatten(electrons.pt)),
        #         'el_eta': ak.to_numpy(ak.flatten(electrons.eta)),
        #         'el_phi': ak.to_numpy(ak.flatten(electrons.phi)),
        #         'el_charge': ak.to_numpy(ak.flatten(electrons.charge)),
        #     })
        #     placeholder.to_csv("./V2electrons.csv")
        
        
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

        # Events where there is at least one muon passing
        # leading muon pT cut
        pass_leading_pt = events.Muon[:,:1].pt_raw > self.config["muon_leading_pt"]
        pass_leading_pt = ak.fill_none(pass_leading_pt, value=False) 
        pass_leading_pt = ak.sum(pass_leading_pt, axis=1)
        

        event_filter = event_filter & (pass_leading_pt >0)
        
        
        
        # filter out bad events since we're calculating delta_r
        events = events[event_filter==True]

        if self.test:
            print(f"copperhead2 EventProcessor b4 leading pt cut event_filter long: \n {pd.DataFrame(ak.to_numpy(event_filter)).to_string()}")
            print(f"copperhead2 EventProcessor pass_leading_pt: \n {pass_leading_pt}")
            print(f"copperhead2 EventProcessor after leading pt cut event_filter long: \n {ak.to_dataframe(event_filter)}")
            print(f"copperhead2 EventProcessor ak.sum(event_filter): \n {ak.sum(event_filter)}")
            print(f"copperhead2 EventProcessor events.Muon: \n {ak.num(events.Muon, axis=1)}")
        
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
        dimuon_dR = mu1.delta_r(mu2)
        dimuon_dEta = abs(mu1.eta -mu2.eta)
        dimuon_dPhi = abs(mu1.delta_phi(mu2))
        dimuon = mu1+mu2
        # fill in pd Dataframe as placeholder. Should be fine since we don't need jagged arrays
        dimuon_mass_resolution = self.mass_resolution(events)
        rel_dimuon_ebe_mass_res = dimuon_mass_resolution/dimuon.mass
        dimuon_cos_theta_cs, dimuon_phi_cs = cs_variables(mu1,mu2)

        #fill genjets
        
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
            padded_iso_gjet = ak.pad_none(gjets[isolated],2) # pad with none val to ensure that events have at least two columns each event
            gjet1 = padded_iso_gjet[:,0]
            
            gjet2 = padded_iso_gjet[:,1] 
            gjj = gjet1 + gjet2
            
            gjj_dEta = abs(gjet1.eta - gjet2.eta)
            gjj_dPhi = abs(gjet1.delta_phi(gjet2))
            gjj_dR = gjet1.delta_r(gjet2)
            if self.test:
                print(f"fill_gen_jets isolated: \n {isolated}")
                print(f"fill_gen_jets isolated long: \n {ak.to_numpy(ak.flatten(isolated))}")
                print(f"fill_gen_jets gjet1: \n {gjet1}")
                print(f"fill_gen_jets gjj: \n {gjj}")
                print(f"fill_gen_jets gjj_dEta: \n {gjj_dEta}")
                print(f"fill_gen_jets gjj_dPhi: \n {gjj_dPhi}")
                print(f"fill_gen_jets gjj_dR: \n {gjj_dR}")

        self.prepare_jets(events)

        if self.test:
            print(f"copperhead2 EventProcessor events.Jet.rho: \n {events.Jet.rho}")
            print(f"copperhead2 EventProcessor events.Jet.rho long: \n {ak.to_numpy(ak.flatten(events.Jet.rho))}")   
            print(f'copperheadV2 EventProcessor jets.pt b4 apply_jec long: \n {ak.to_numpy(ak.flatten(events.Jet.pt))}')
        print(f'copperheadV2 EventProcessor jets.eta b4 apply_jec long: \n {ak.to_numpy(ak.flatten(events.Jet.eta))}')
        print(f'copperheadV2 EventProcessor jets.phi b4 apply_jec long: \n {ak.to_numpy(ak.flatten(events.Jet.phi))}')
        
        jets = events.Jet
        self.jec_factories_mc, self.jec_factories_data = get_jec_factories(
            self.config["jec_parameters"], 
            # self.config["year"]
        )
        
        do_jec = False

        # We only need to reapply JEC for 2018 data
        # (unless new versions of JEC are released)
        is_data = not events.metadata["is_mc"]
        if is_data and ("2018" in self.config["year"]):
            do_jec = True

        # jec_pars = {k: v[self.config["year"]] for k, v in self.config["jec_parameters"].items()}
        # jec_pars = self.config["jec_parameters"]
        
        # do_jecunc = False
        # do_jerunc = False
        # pt_variations = (
        #     ["nominal"]
        #     + jec_pars["jec_variations"]
        #     + jec_pars["jer_variations"]
        # )
        # for ptvar in self.pt_variations:
        #     if ptvar in jec_pars["jec_variations"]:
        #         do_jecunc = True
        #     if ptvar in jec_pars["jer_variations"]:
        #         do_jerunc = True
        do_jecunc = self.config["do_jecunc"]
        do_jerunc = self.config["do_jerunc"]
        
        
        # apply General jet energy correction (JEC)
        # jets = apply_jec(
        #     events,
        #     jets,
        #     dataset,
        #     is_mc,
        #     self.year,
        #     self.do_jec,
        #     self.do_jecunc,
        #     self.do_jerunc,
        #     self.jec_factories,
        #     self.jec_factories_data,
        # )
        if do_jec:
            if events.metadata["is_mc"]:
                factory = self.jec_factories["jec"]
            else:
                for run in self.config["jec_parameters"]["runs"]:
                    if run in events.metadata["dataset"]:
                        factory = self.jec_factories_data[run]
            print("jets build")
            jets = factory.build(jets)
        # TODO: only consider nuisances that are defined in run parameters
        # Compute JEC uncertainties
        if events.metadata["is_mc"] and do_jecunc:
            jets = self.jec_factories["junc"].build(jets)
    
        # Compute JER uncertainties
        if events.metadata["is_mc"] and do_jerunc:
            jets = self.jec_factories["jer"].build(jets)
    
        # TODO: JER nuisances

        if self.test:
            print(f'copperheadV2 EventProcessor after apply_jec jets.pt short: \n {jets.pt}')
            print(f'copperheadV2 EventProcessor after apply_jec jets.pt long: \n {ak.to_numpy(ak.flatten(jets.pt.compute()))}')
        print(f'copperheadV2 EventProcessor after apply_jec jets.eta long: \n {ak.to_numpy(ak.flatten(jets.eta.compute()))}')
        print(f'copperheadV2 EventProcessor after apply_jec jets.phi long: \n {ak.to_numpy(ak.flatten(jets.phi.compute()))}')
            # placeholder =  pd.DataFrame({
            #     'mu1_pt': ak.to_numpy((mu1.pt)),
            #     'mu2_pt': ak.to_numpy(mu2.pt),
            #     'mu1_eta': ak.to_numpy(mu1.eta),
            #     'mu2_eta': ak.to_numpy(mu2.eta),
            #     'mu1_phi': ak.to_numpy(mu1.phi),
            #     'mu2_phi': ak.to_numpy(mu2.phi),
            #     'mu1_iso': ak.to_numpy(mu1.pfRelIso04_all),
            #     'mu2_iso': ak.to_numpy(mu2.pfRelIso04_all),
            #     'mu1_pt_over_mass': ak.to_numpy(mu1.pt/dimuon.mass),
            #     'mu2_pt_over_mass': ak.to_numpy(mu2.pt/dimuon.mass),
            #     "dimuon_mass": ak.to_numpy(dimuon.mass),
            #     "dimuon_ebe_mass_res": ak.to_numpy(dimuon_mass_resolution),
            #     "dimuon_ebe_mass_res_rel": ak.to_numpy(rel_dimuon_ebe_mass_res),
            #     "dimuon_pt": ak.to_numpy(dimuon.pt),
            #     "dimuon_pt_log": ak.to_numpy(np.log(dimuon.pt)), # np functions are compatible with ak if input is ak array 
            #     "dimuon_eta": ak.to_numpy(dimuon.eta),
            #     "dimuon_phi": ak.to_numpy(dimuon.phi),
            #     "dimuon_dEta": ak.to_numpy(dimuon_dEta),
            #     "dimuon_dPhi": ak.to_numpy(dimuon_dPhi),
            #     "dimuon_dR": ak.to_numpy(dimuon_dR),
            #     "dimuon_cos_theta_cs": ak.to_numpy(dimuon_cos_theta_cs), 
            #     "dimuon_phi_cs": ak.to_numpy(dimuon_phi_cs), 
            #     "gjj_mass":  ak.to_numpy(gjj.mass),
            #     "gjet1_mass":  ak.to_numpy(gjet1.mass),
            #     "gjet1_pt":  ak.to_numpy(gjet1.pt),
            #     "gjet1_eta":  ak.to_numpy(gjet1.eta),
            #     "gjet1_phi":  ak.to_numpy(gjet1.phi),
            #     "gjet2_mass":  ak.to_numpy(gjet2.mass),
            #     "gjet2_pt":  ak.to_numpy(gjet2.pt),
            #     "gjet2_eta":  ak.to_numpy(gjet2.eta),
            #     "gjet2_phi":  ak.to_numpy(gjet2.phi),
            #     "gjj_dEta": ak.to_numpy(gjj_dEta),
            #     "gjj_dPhi": ak.to_numpy(gjj_dPhi),
            #     "gjj_dR": ak.to_numpy(gjj_dR),
                
            # })
            # # print(f"copperhead2 EventProcessor after leading pt cut placeholder: \n {placeholder.to_string()}")
            # placeholder.to_csv("./V2placeholder.csv")
        
        out_dict = {
            "mu_pt" : events.Muon.pt,
            "mu_eta" : events.Muon.eta,
            "mu_phi" : events.Muon.phi,
            "mu_iso" : events.Muon.pfRelIso04_all,
            "dimuon_mass": dimuon.mass,
            "dimuon_pt" : dimuon.pt,
            "dimuon_eta" : dimuon.eta,
            "dimuon_phi" : dimuon.phi,
            "dimuon_dEta" : dimuon_dEta,
            "dimuon_dPhi" : dimuon_dPhi,
            "dimuon_dR" : dimuon_dR,
            "dimuon_ebe_mass_res": dimuon_mass_resolution,
            "dimuon_ebe_mass_res_rel": rel_dimuon_ebe_mass_res,
            "mu_pt_over_mass" : events.Muon.pt / dimuon.mass,
            "dimuon_cos_theta_cs" : dimuon_cos_theta_cs,
            "dimuon_phi_cs" : dimuon_phi_cs,
            "gjet_pt" : padded_iso_gjet.pt,
            "gjet_eta" : padded_iso_gjet.eta,
            "gjet_phi" : padded_iso_gjet.phi,
            "gjet_mass" : padded_iso_gjet.mass,
            "gjj_mass": gjj.mass,
            "gjj_pt" : gjj.pt,
            "gjj_eta" : gjj.eta,
            "gjj_phi" : gjj.phi,
            "gjj_dEta" : gjj_dEta,
            "gjj_dPhi" : gjj_dPhi,
            "gjj_dR" : gjj_dR,
        }
        return out_dict
        
    def postprocess(self, accumulator):
        """
        Arbitrary postprocess function that's required to run the processor
        """
        pass

    
    def mass_resolution(self, events):
        # Returns absolute mass resolution!
        mu1 = events.Muon[:,0]
        mu2 = events.Muon[:,1]
        muon_E = (mu1+mu2).mass /2
        dpt1 = (mu1.ptErr / mu1.pt) * muon_E
        dpt2 = (mu2.ptErr / mu2.pt) * muon_E
        print(f"muons mass_resolution dpt1: {dpt1}")
        if "2016" in self.config["year"]:
            yearstr = "2016"
        else:
            yearstr=self.config["year"] #Work around before there are seperate new files for pre and postVFP
        if events.metadata["is_mc"]:
            label = f"res_calib_MC_{yearstr}"
        else:
            label = f"res_calib_Data_{yearstr}"
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

        # # --- Evaluator
        # self.extractor = extractor()

        # # Z-pT reweigting (disabled)
        # zpt_filename = self.parameters["zpt_weights_file"]
        # self.extractor.add_weight_sets([f"* * {zpt_filename}"])
        # if "2016" in self.year:
        #     self.zpt_path = "zpt_weights/2016_value"
        # else:
        #     self.zpt_path = "zpt_weights/2017_value"
        # # PU ID weights
        # puid_filename = self.parameters["puid_sf_file"]
        # self.extractor.add_weight_sets([f"* * {puid_filename}"])
        # # Calibration of event-by-event mass resolution
        # for mode in ["Data", "MC"]:
        #     if "2016" in self.year:
        #         yearstr = "2016"
        #     else:
        #         yearstr=self.year #Work around before there are seperate new files for pre and postVFP
        #     label = f"res_calib_{mode}_{yearstr}"
        #     path = self.parameters["res_calib_path"]
        #     file_path = f"{path}/{label}.root"
        #     self.extractor.add_weight_sets([f"{label} {label} {file_path}"])
        # # Mass resolution - Pisa implementation
        # self.extractor.add_weight_sets(["* * data/mass_res_pisa/muonresolution.root"])
        # self.extractor.finalize()
        # self.evaluator = self.extractor.make_evaluator()
        # print(f"processor self.evaluator: {self.evaluator}")

        # self.evaluator[self.zpt_path]._axes = self.evaluator[self.zpt_path]._axes[0]

        # return

    # def jet_loop(
    #     self,
    #     variation,
    #     is_mc,
    #     df,
    #     dataset,
    #     mask,
    #     muons,
    #     mu1,
    #     mu2,
    #     jets,
    #     weights,
    #     numevents,
    #     output,
    # ):
    #     # weights = copy.deepcopy(weights)

    #     if not is_mc and variation != "nominal":
    #         return

    #     variables = pd.DataFrame(index=output.index)
    #     # print(f"variables: {variables}")
        
    #     jet_columns = [
    #         "pt",
    #         "eta",
    #         "phi",
    #         "jetId",
    #         "qgl",
    #         "puId",
    #         "mass",
    #         "btagDeepB",
    #         "has_matched_gen",
    #     ]
    #     if "puId17" in df.Jet.fields:
    #         jet_columns += ["puId17"]
    #     if is_mc:
    #         jet_columns += ["partonFlavour", "hadronFlavour"]
    #     if variation == "nominal":
    #         if self.do_jec:
    #             jet_columns += ["pt_jec", "mass_jec"]
    #         if is_mc and self.do_jerunc:
    #             jet_columns += ["pt_orig", "mass_orig"]

    #     # Find jets that have selected muons within dR<0.4 from them
    #     # print(f"jets.matched_muons: {jets.matched_muons}")
    #     # print(f"type(jets.matched_muons): {type(jets.matched_muons)}")
    #     # print(f"type(jets.matched_muons.pt_fsr): {type(jets.matched_muons.pt_fsr)}")
    #     # print(f"(jets.matched_muons.pt_fsr): {(jets.matched_muons.pt_fsr)}")
    #     # print(f"ak.to_pandas(jets.matched_muons.pt_fsr): {ak.to_pandas(jets.matched_muons.pt_fsr)}")
    #     matched_mu_pt = jets.matched_muons.pt_fsr
    #     matched_mu_iso = jets.matched_muons.pfRelIso04_all
    #     matched_mu_id = jets.matched_muons[self.parameters["muon_id"]]
    #     matched_mu_pass = (
    #         (matched_mu_pt > self.parameters["muon_pt_cut"])
    #         & (matched_mu_iso < self.parameters["muon_iso_cut"])
    #         & matched_mu_id
    #     )
    #     # print(f"matched_mu_pass: {matched_mu_pass}")
    #     # print(f"type(matched_mu_pass): {type(matched_mu_pass)}")
    #     # print(f"ak.to_pandas(matched_mu_pass): {ak.to_pandas(matched_mu_pass)}")
    #     clean = ~(
    #         ak.to_pandas(matched_mu_pass)
    #         .astype(float)
    #         .fillna(0.0)
    #         .groupby(level=[0, 1])
    #         .sum()
    #         .astype(bool)
    #     )
    #     # print(f"~clean: {~clean}")
    #     # Select particular JEC variation
    #     if "_up" in variation:
    #         unc_name = "JES_" + variation.replace("_up", "")
    #         if unc_name not in jets.fields:
    #             return
    #         jets = jets[unc_name]["up"][jet_columns]
    #     elif "_down" in variation:
    #         unc_name = "JES_" + variation.replace("_down", "")
    #         if unc_name not in jets.fields:
    #             return
    #         jets = jets[unc_name]["down"][jet_columns]
    #     else:
    #         jets = jets[jet_columns]

    #     # --- conversion from awkward to pandas --- #
    #     jets = ak.to_pandas(jets)
    #     # print(f"jets ak to pandas: \n {jets.head()}")
    #     # print(f"jets.index.nlevels : \n {jets.index.nlevels }")
        

    #     if jets.index.nlevels == 3:
    #         # sometimes there are duplicates?
    #         jets = jets.loc[pd.IndexSlice[:, :, 0], :]
    #         jets.index = jets.index.droplevel("subsubentry")

    #     if variation == "nominal":
    #         # Update pt and mass if JEC was applied
    #         if self.do_jec:
    #             jets["pt"] = jets["pt_jec"]
    #             jets["mass"] = jets["mass_jec"]

    #         # We use JER corrections only for systematics, so we shouldn't
    #         # update the kinematics. Use original values,
    #         # unless JEC were applied.
    #         if is_mc and self.do_jerunc and not self.do_jec:
    #             jets["pt"] = jets["pt_orig"]
    #             jets["mass"] = jets["mass_orig"]

    #     # ------------------------------------------------------------#
    #     # Apply jetID and PUID
    #     # ------------------------------------------------------------#

    #     pass_jet_id = jet_id(jets, self.parameters, self.year)
    #     pass_jet_puid = jet_puid(jets, self.parameters, self.year)

    #     # Jet PUID scale factors
    #     # if is_mc and False:  # disable for now
    #     #     puid_weight = puid_weights(
    #     #         self.evaluator, self.year, jets, pt_name,
    #     #         jet_puid_opt, jet_puid, numevents
    #     #     )
    #     #     weights.add_weight('puid_wgt', puid_weight)

    #     # ------------------------------------------------------------#
    #     # Select jets
    #     # ------------------------------------------------------------#
    #     jets["clean"] = clean

    #     jet_selection = (
    #         pass_jet_id
    #         & pass_jet_puid
    #         & (jets.qgl > -2)
    #         & jets.clean
    #         & (jets.pt > self.parameters["jet_pt_cut"])
    #         & (abs(jets.eta) < self.parameters["jet_eta_cut"])
    #     )

    #     jets = jets[jet_selection]

    #     # ------------------------------------------------------------#
    #     # Fill jet-related variables
    #     # ------------------------------------------------------------#

    #     njets = jets.reset_index().groupby("entry")["subentry"].nunique()
    #     variables["njets"] = njets

    #     # one_jet = (njets > 0)
    #     two_jets = njets > 1

    #     # Sort jets by pT and reset their numbering in an event
    #     jets = jets.sort_values(["entry", "pt"], ascending=[True, False])
    #     jets.index = pd.MultiIndex.from_arrays(
    #         [jets.index.get_level_values(0), jets.groupby(level=0).cumcount()],
    #         names=["entry", "subentry"],
    #     )

    #     # Select two jets with highest pT
    #     try:
    #         jet1 = jets.loc[pd.IndexSlice[:, 0], :]
    #         jet2 = jets.loc[pd.IndexSlice[:, 1], :]
    #         # print(f"jet1 b4 droplevel: \n {jet1}")
    #         jet1.index = jet1.index.droplevel("subentry")
    #         jet2.index = jet2.index.droplevel("subentry")
    #         # print(f"jet1 after droplevel: \n {jet1}")
    #     except Exception:
    #         return

    #     fill_jets(output, variables, jet1, jet2)

    #     # ------------------------------------------------------------#
    #     # Fill soft activity jet variables
    #     # ------------------------------------------------------------#

    #     # Effect of changes in jet acceptance should be negligible,
    #     # no need to calcluate this for each jet pT variation
    #     if variation == "nominal":
    #         fill_softjets(df, output, variables, 2)
    #         fill_softjets(df, output, variables, 5)

    #     # ------------------------------------------------------------#
    #     # Apply remaining cuts
    #     # ------------------------------------------------------------#

    #     # Cut has to be defined here because we will use it in
    #     # b-tag weights calculation
    #     vbf_cut = (variables.jj_mass > 400) & (variables.jj_dEta > 2.5) & (jet1.pt > 35)

    #     # ------------------------------------------------------------#
    #     # Calculate QGL weights, btag SF and apply btag veto
    #     # ------------------------------------------------------------#

    #     if is_mc and variation == "nominal":
    #         # --- QGL weights --- #
    #         isHerwig = "herwig" in dataset

    #         qgl_wgts = qgl_weights(jet1, jet2, isHerwig, output, variables, njets)
    #         weights.add_weight("qgl_wgt", qgl_wgts, how="all")
    #         # print(f"type(qgl_wgts) : \n {type(qgl_wgts)}")
    #         # print(f"qgl_wgts : \n {qgl_wgts}")

    #         # --- Btag weights --- #
    #         bjet_sel_mask = output.event_selection #& two_jets & vbf_cut

    #         btag_wgt, btag_syst = btag_weights_json(
    #             self, self.btag_systs, jets, weights, bjet_sel_mask, self.btag_json
    #         )
    #         weights.add_weight("btag_wgt", btag_wgt)

    #         # --- Btag weights variations --- #
    #         for name, bs in btag_syst.items():
    #             weights.add_weight(f"btag_wgt_{name}", bs, how="only_vars")

    #     # Separate from ttH and VH phase space
        
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
        
    #     variables["nBtagLoose"] = (
    #         jets[
    #             (jets.btagDeepB > self.parameters["btag_loose_wp"])
    #             & (abs(jets.eta) < 2.5)
    #         ]
    #         .reset_index()
    #         .groupby("entry")["subentry"]
    #         .nunique()
    #     )
    #     # print(f'variables["nBtagLoose"] : {variables["nBtagLoose"] }')
        
    #     variables["nBtagMedium"] = (
    #         jets[
    #             (jets.btagDeepB > self.parameters["btag_medium_wp"])
    #             & (abs(jets.eta) < 2.5)
    #         ]
    #         .reset_index()
    #         .groupby("entry")["subentry"]
    #         .nunique()
    #     )
    #     variables.nBtagLoose = variables.nBtagLoose.fillna(0.0)
    #     variables.nBtagMedium = variables.nBtagMedium.fillna(0.0)

    #     # --------------------------------------------------------------#
    #     # Fill outputs
    #     # --------------------------------------------------------------#

    #     variables.update({"wgt_nominal": weights.get_weight("nominal")})

    #     # All variables are affected by jet pT because of jet selections:
    #     # a jet may or may not be selected depending on pT variation.

    #     for key, val in variables.items():
    #         output.loc[:, pd.IndexSlice[key, variation]] = val

    #     print("done jet loop")
    #     # print(f'output : \n {output.head()}')
    #     # print(f'output["dimuon_mass"] : \n {output["dimuon_mass"].head()}')
    #     return output
    