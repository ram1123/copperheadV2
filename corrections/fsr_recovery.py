import numpy as np
import awkward as ak
from typing import TypeVar, Tuple
ak_array = TypeVar('ak_array')
coffea_nanoevent = TypeVar('coffea_nanoevent') 


def fsr_recovery(events: coffea_nanoevent) -> ak_array:
    # test that no muon fsrPhoton idx is overlapping with electron photon idx------------------
    # idxs = [1,2,3] # check up to 3 assocaited photons
    # for i in idxs:
    #     mu_photon_idxs = events.Muon.fsrPhotonIdxG[events.Muon.fsrPhotonIdxG != -1][:,(i-1):i]
    #     mu_photon_idxs = ak.flatten(ak.pad_none(mu_photon_idxs,1)) # get event length array of idxs with flatten
    #     for j in idxs:
    #         el_photon_idxs = events.Electron.photonIdxG[events.Electron.photonIdxG != -1][:,(j-1):j]
    #         el_photon_idxs = ak.flatten(ak.pad_none(el_photon_idxs,1)) # get event length array of idxs with flatten
    #         # print(mu_photon_idxs.compute())
    #         flag = ak.fill_none((mu_photon_idxs==el_photon_idxs), value=False)
    #         print(f"# of {i}th muon fsrPhoton beig same as {j}th electron photon: {ak.sum(flag).compute()}")
    # fsrPhoton test end ----------------------------------------------------------------------
    # print(f"fsr_recovery type(events): {type(events)}")

    # # testing better fsrPhotonsToRecover ---------------------------------------------------------------
    # mu_fsr_delta_r = events.Muon.delta_r(events.Muon.matched_fsrPhoton)
    # fsr_abs_eta = abs(events.Muon.matched_fsrPhoton.eta)
    # fsrPhotonsToRecover = (
    #     (events.Muon.fsrPhotonIdx >= 0) # pick rows and cols that have values (-1 means no photon found)
    #     & (events.Muon.matched_fsrPhoton.relIso03 < 1.8)
    #     & (events.Muon.matched_fsrPhoton.dROverEt2 < 0.012)
    #     & (events.Muon.matched_fsrPhoton.pt / events.Muon.pt < 0.4) # suppress Zgamma -> mumu contanmination
    #     & ( ((fsr_abs_eta > 0)&(fsr_abs_eta < 1.4442)) | ((fsr_abs_eta > 1.566)&(fsr_abs_eta < 2.5)) )
    #     & ((mu_fsr_delta_r > 0.0001) & (mu_fsr_delta_r < 0.5))
    # ) 
    # # # testing better fsrPhotonsToRecover end ---------------------------------------------------------------
    
    # original fsrPhotonsToRecover ---------------------------------------------------------------
    fsrPhotonsToRecover = (
        (events.Muon.fsrPhotonIdx >= 0) # pick rows and cols that have values (-1 means no photon found)
        & (events.Muon.matched_fsrPhoton.relIso03 < 1.8)
        & (events.Muon.matched_fsrPhoton.dROverEt2 < 0.012)
        & (events.Muon.matched_fsrPhoton.pt / events.Muon.pt < 0.4) # suppress Zgamma -> mumu contanmination
        & (abs(events.Muon.matched_fsrPhoton.eta) < 2.4)
    ) 
    # original fsrPhotonsToRecover end ---------------------------------------------------------------

    
    fsrPhotonsToRecover = ak.fill_none(fsrPhotonsToRecover, False) 
    # print(f"fsrPhotonsToRecover: {fsrPhotonsToRecover.compute()}")
    # print(f"ak.sum(fsrPhotonsToRecover, axis=1): {ak.sum(fsrPhotonsToRecover, axis=1).compute()}")
    # print(f"nmuons axis: {ak.num(events.Muon, axis=1).compute()}")
    # print(f"muons: {events.Muon.pt.compute()}")

    print(f"total nmuons applied with fsrPhotons: {ak.sum(fsrPhotonsToRecover,axis=None).compute()}")
    print(f"total nmuons: {ak.sum(ak.num(events.Muon,axis=1)).compute()}")
    
    # add mass and charge as otherwise you can't add two lorentzvectors
    events["FsrPhoton", "mass"] = 0 
    events["FsrPhoton", "charge"] = 0 
    # print(f"events.Muon.matched_fsrPhoton: {events.Muon.matched_fsrPhoton.compute()}")
    fsr_muons = events.Muon.matched_fsrPhoton + events.Muon # None means there weren't matched fsrphotons
    # fsr_muons =  ak.values_astype(events.Muon.matched_fsrPhoton,  "float64")  + ak.values_astype(events.Muon,  "float64") 
    
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
    
    
    fsr_filter = (ak.sum(fsrPhotonsToRecover, axis=1) > 0)
    # print(f"fsr_filter: {fsr_filter.compute()}")
    # print(f"fsr_filter sum: {ak.sum(fsr_filter).compute()}")
    # print(f"fsr_filter num: {ak.num(fsr_filter, axis=0).compute()}")

    fsr_events = events[fsr_filter]
    # print(f"ak.num(fsr_events.Muon.pt_fsr, axis=1): {ak.num(fsr_events.Muon.pt_fsr, axis=1).compute()}")
    # print(f"ak.num(fsr_events.Muon.pt_fsr, axis=0): {ak.num(fsr_events.Muon.pt_fsr, axis=0).compute()}")
    argmax = ak.argmax(fsr_events.Muon.pt_fsr, axis=1)
    # print(f"argmax max: {(ak.max(argmax)).compute()}")
    # print(f"argmax min: {(ak.min(argmax)).compute()}")
    argmax_not_leading = argmax != 0
    # print(f"muon argmax[argmax_not_leading]: {argmax[argmax_not_leading].compute()}")
    # print(f"argmax_not_leading: {(argmax_not_leading).compute()}")
    # print(f"argmax_not_leading sum: {ak.sum(argmax_not_leading).compute()}")
    # print(f"argmax not leading muon_fsr.pt_fsr: {fsr_events.Muon.pt_fsr[argmax_not_leading].compute()}")
    argmax_not_leading_events = fsr_events[argmax_not_leading]
    # print(f"argmax_not_leading_events.Muon.fsrPhotonIdx >= 0 : \n {(argmax_not_leading_events.Muon.fsrPhotonIdx >= 0).compute()}")
    # print(f"argmax_not_leading_events.Muon.matched_fsrPhoton.relIso03 < 1.8 : \n {(argmax_not_leading_events.Muon.matched_fsrPhoton.relIso03 < 1.8).compute()}")
    # print(f"argmax_not_leading_events.Muon.matched_fsrPhoton.dROverEt2 < 0.012 : \n {(argmax_not_leading_events.Muon.matched_fsrPhoton.dROverEt2 < 0.012).compute()}")
    # print(f"argmax_not_leading_events.Muon.matched_fsrPhoton.pt / argmax_not_leading_events.Muon.pt < 0.4 : \n {(argmax_not_leading_events.Muon.matched_fsrPhoton.pt / argmax_not_leading_events.Muon.pt < 0.4 ).compute()}")
    # print(f"abs(argmax_not_leading_events.Muon.matched_fsrPhoton.eta) < 2.4 : \n {(abs(argmax_not_leading_events.Muon.matched_fsrPhoton.eta) < 2.4).compute()}")

    # print(f"ak.num(fsr_events, axis=0) : {ak.num(fsr_events, axis=0).compute()}")
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
    # print(f"ak.argmax(fsr_events.Muon.pt): {ak.argmax(fsr_events.Muon.pt, axis=1).compute()}")
    
    # print(f"leading still leading after fsr:{ak.sum(ak.argmax(fsr_events.Muon.pt_fsr, axis=1)==0).compute()}")
    # print(f"subleading leading after fsr:{ak.sum(ak.argmax(fsr_events.Muon.pt_fsr, axis=1)==1).compute()}")
    # print(f"subsubleading+ leading after fsr:{ak.sum(ak.argmax(fsr_events.Muon.pt_fsr, axis=1)>1).compute()}")
    # print(f"ak.num(fsr_events.Muon): {ak.num(fsr_events.Muon, axis=0).compute()}")
    
    # print(f"ak.num(events.Muon): {ak.num(events.Muon, axis=0).compute()}")
    # print(f"muon_fsr.pt_fsr: {fsr_events.Muon.pt_fsr.compute()}")
    # print(f"muon_fsr.pt: {fsr_events.Muon.pt.compute()}")
    # print(f"bad muon_fsr.pt_fsr: {fsr_events.Muon.pt_fsr.compute()[bad_idxs]}")
    # print(f"bad muon_fsr.pt: {fsr_events.Muon.pt.compute()[bad_idxs]}")
    # print(f"muon_fsr.matched_fsrPhoton.pt: \n {fsr_events.Muon.matched_fsrPhoton.pt.compute()}")
    # print(f"events.Muon.pt[mask]: {events.Muon.pt[fsrPhotonsToRecover].compute()}")
    # print(f"sum events.Muon.pt[mask]: {ak.sum(events.Muon.pt[fsrPhotonsToRecover], axis=1).compute()}")
    # print(f"num events.Muon.pt[mask]: {ak.num(events.Muon.pt[fsrPhotonsToRecover], axis=1).compute()}")
    return fsrPhotonsToRecover # return boolean filter for geofit





    