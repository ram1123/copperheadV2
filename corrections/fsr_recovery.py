import numpy as np
import awkward as ak
from typing import TypeVar, Tuple
ak_array = TypeVar('ak_array')
coffea_nanoevent = TypeVar('coffea_nanoevent') 


def fsr_recovery(events: coffea_nanoevent) -> ak_array:
    print(f"fsr_recovery type(events): {type(events)}")
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

    #some quick test. comment them out when done:
    assert(
        0 ==
        (ak.sum(events.Muon[~fsrPhotonsToRecover].pt_fsr!= events.Muon[~fsrPhotonsToRecover].pt)) 
    )
    assert(
        0 ==
        (ak.sum(events.Muon[fsrPhotonsToRecover].pt_fsr== events.Muon[fsrPhotonsToRecover].pt)) 
    )

    # print(f"fsr_recovery events.Muon.pt_fsr[fsrPhotonsToRecover]: \n {ak.to_numpy(ak.flatten(events.Muon.pt_fsr[fsrPhotonsToRecover]))}")
    # print(f"fsr_recovery events.Muon.eta_fsr[fsrPhotonsToRecover]: \n {ak.to_numpy(ak.flatten(events.Muon.eta_fsr[fsrPhotonsToRecover]))}")
    # print(f"fsr_recovery events.Muon.phi_fsr[fsrPhotonsToRecover]: \n {ak.to_numpy(ak.flatten(events.Muon.phi_fsr[fsrPhotonsToRecover]))}")
    # print(f"fsr_recovery events.Muon.mass_fsr:[fsrPhotonsToRecover] \n {ak.to_numpy(ak.flatten(events.Muon.mass_fsr[fsrPhotonsToRecover]))}")
    # print(f"fsr_recovery events.Muon.iso_fsr[fsrPhotonsToRecover]: \n {ak.to_numpy(ak.flatten(events.Muon.iso_fsr[fsrPhotonsToRecover]))}")
    # print(f"fsr_recovery ak.where(ak.sum(fsrPhotonsToRecover, axis=1)>0): \n {ak.to_numpy(ak.where(ak.sum(fsrPhotonsToRecover, axis=1)>0))}")
    # print(f"fsr_recovery ak.sum(fsrPhotonsToRecover): {ak.sum(fsrPhotonsToRecover)}")
    
    return fsrPhotonsToRecover





    