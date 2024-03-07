import numpy as np
import awkward as ak
from typing import TypeVar, Tuple
ak_array = TypeVar('ak_array')
coffea_nanoevent = TypeVar('coffea_nanoevent') 


def fsr_recovery(events: coffea_nanoevent) -> ak_array:
    # print(f"fsr_recovery type(events): {type(events)}")
    fsrPhotonsToRecover = (
        (events.Muon.fsrPhotonIdx >= 0)
        & (events.Muon.matched_fsrPhoton.relIso03 < 1.8)
        & (events.Muon.matched_fsrPhoton.dROverEt2 < 0.012)
        & (events.Muon.matched_fsrPhoton.pt / events.Muon.pt < 0.4)
        & (abs(events.Muon.matched_fsrPhoton.eta) < 2.4)
    ) # loosen the condition for more yields during testing
    fsrPhotonsToRecover = ak.fill_none(fsrPhotonsToRecover, False) 
    # add mass and charge as otherwise you can't add two lorentzvectors
    events["FsrPhoton", "mass"] = 0 
    events["FsrPhoton", "charge"] = 0 
    fsr_muons = events.Muon.matched_fsrPhoton + events.Muon # None means there weren't matched fsrphotons
    fsr_iso = (events.Muon.pfRelIso04_all * events.Muon.pt - events.Muon.matched_fsrPhoton.pt) / fsr_muons.pt
   
    events["Muon", "pt_fsr"] = ak.where(fsrPhotonsToRecover, fsr_muons.pt, events.Muon.pt)
    events["Muon", "eta_fsr"] = ak.where(fsrPhotonsToRecover, fsr_muons.eta, events.Muon.eta)
    events["Muon", "phi_fsr"] = ak.where(fsrPhotonsToRecover, fsr_muons.phi, events.Muon.phi)
    events["Muon", "mass_fsr"] = ak.where(fsrPhotonsToRecover, fsr_muons.mass, events.Muon.mass)
    events["Muon", "iso_fsr"] = ak.where(fsrPhotonsToRecover, fsr_iso, events.Muon.pfRelIso04_all)

    #-----------------------------------------------------
    # #some quick test. comment them out when done:
    # assert(
    #     0 ==
    #     (ak.sum(events.Muon[~fsrPhotonsToRecover].pt_fsr!= events.Muon[~fsrPhotonsToRecover].pt)) 
    # )
    # assert(
    #     0 ==
    #     (ak.sum(events.Muon[fsrPhotonsToRecover].pt_fsr== events.Muon[fsrPhotonsToRecover].pt)) 
    # )
    #-----------------------------------------------------
    
    
    # print(f"fsrPhotonsToRecover: {fsrPhotonsToRecover.compute()}")
    # print(f"ak.sum(fsrPhotonsToRecover, axis=1: {ak.sum(fsrPhotonsToRecover, axis=1).compute()}")
    fsr_filter = ak.sum(fsrPhotonsToRecover, axis=1) > 0
    fsr_events = events[fsr_filter]
    # # testing the inconsistent idxs with copper head V1 ----------------
    # # print(f"test: {ak.num(events.Muon.pt, axis=0)}")
    # fsr_filter_computed = (fsr_filter).compute()
    # bad_idxs = ak.to_numpy(ak.zeros_like(fsr_filter_computed))
    # bad_idxs[393] = 1
    # bad_idxs[669] = 1
    # bad_idxs[1550] = 1
    # bad_idxs[2286] = 1
    # bad_idxs[4710] = 1
    # bad_idxs[6403] = 1
    # bad_idxs[6643] = 1
    # bad_idxs[7378] = 1
    # bad_idxs[8396] = 1
    # bad_idxs = bad_idxs[ak.to_numpy(fsr_filter_computed)]
    # print(f"bad_idxs len: {ak.num(bad_idxs, axis=0)}")
    #-------------------------------------------------------------------
    
    print(f"ak.num(fsr_events.Muon): {ak.num(fsr_events.Muon, axis=0).compute()}")
    print(f"ak.num(events.Muon): {ak.num(events.Muon, axis=0).compute()}")
    # print(f"muon_fsr.pt_fsr: {fsr_events.Muon.pt_fsr.compute()}")
    # print(f"muon_fsr.pt: {fsr_events.Muon.pt.compute()}")
    # print(f"bad muon_fsr.pt_fsr: {fsr_events.Muon.pt_fsr.compute()[bad_idxs]}")
    # print(f"bad muon_fsr.pt: {fsr_events.Muon.pt.compute()[bad_idxs]}")
    print(f"muon_fsr.matched_fsrPhoton.pt: \n {fsr_events.Muon.matched_fsrPhoton.pt.compute()}")
    # print(f"events.Muon.pt[mask]: {events.Muon.pt[fsrPhotonsToRecover].compute()}")
    # print(f"sum events.Muon.pt[mask]: {ak.sum(events.Muon.pt[fsrPhotonsToRecover], axis=1).compute()}")
    # print(f"num events.Muon.pt[mask]: {ak.num(events.Muon.pt[fsrPhotonsToRecover], axis=1).compute()}")
    return fsrPhotonsToRecover # return boolean filter for geofit





    