import dask_awkward as dak
import awkward as ak
from distributed import LocalCluster, Client, progress
import time
import numpy as np
import matplotlib.pyplot as plt
import json
import mplhep as hep
import glob


# def applyGGH_cut(events):
#     btag_cut =ak.fill_none((events.nBtagLoose >= 2), value=False) | ak.fill_none((events.nBtagMedium >= 1), value=False)
#     # vbf_cut = ak.fill_none(events.vbf_cut, value=False
#     vbf_cut = (events.jj_mass > 400) & (events.jj_dEta > 2.5)
#     # vbf_cut = (events.jj_mass > 400) & (events.jj_dEta > 2.5) & (events.jet1_pt > 35) 
#     vbf_cut = ak.fill_none(vbf_cut, value=False)
#     region = events.h_sidebands | events.h_peak
#     # region = events.h_sidebands 
#     ggH_filter = (
#         ~vbf_cut & 
#         region &
#         ~btag_cut # btag cut is for VH and ttH categories
#     )
#     return events[ggH_filter]

def applySigReg_cut(events):
    region = events.h_sidebands | events.h_peak
    # region = events.h_sidebands 
    ggH_filter = (
        region 
    )
    return events[ggH_filter]


def calculateSubCat(processed_events, score_edges):
    BDT_score = processed_events["BDT_score"]
    print(f"BDT_score :{BDT_score}")
    print(f"ak.max(BDT_score) :{ak.max(BDT_score)}")
    print(f"ak.min(BDT_score) :{ak.min(BDT_score)}")
    subCat_idx = -1*ak.ones_like(BDT_score)
    for i in range(len(score_edges) - 1):
        lo = score_edges[i]
        hi = score_edges[i + 1]
        cut = (BDT_score > lo) & (BDT_score <= hi)
        # cut = (BDT_score <= lo) & (BDT_score > hi)
        subCat_idx = ak.where(cut, i, subCat_idx)
    # print(f"subCat_idx: {subCat_idx}")
    # test if any remain has -1 value
    print(f"ak.sum(subCat_idx==-1): {ak.sum(subCat_idx==-1)}")
    processed_events["subCategory_idx"] = subCat_idx

def separateNfit(load_paths, score_edges):
    bkg_event_l = []
    sig_event_l = []
    for load_path in load_paths:
        print(f"separateNfit load_path: {load_path}")
        cols_of_interest = [
        'dimuon_mass',
        ]
        additional_fields = [
            "BDT_score",
            "wgt_nominal_total",
            "h_sidebands",
            "h_peak",
            # "nBtagLoose",
            # "nBtagMedium",
            # "jet1_pt",
            # "jj_mass",
            # "jj_dEta",
        ]
        fields2compute = cols_of_interest +  additional_fields
        # load the events by eras, compute them,
        events_sig = dak.from_parquet(f"{load_path}/processed_events_sigMC*.parquet") # ggH and VBF together
        events_sig = ak.zip({field: events_sig[field] for field in fields2compute}).compute()

        events_bkg = dak.from_parquet(f"{load_path}/processed_events_bkgMC*.parquet") # ggH and VBF together
        events_bkg = ak.zip({field: events_bkg[field] for field in fields2compute}).compute()

        # apply ggH cat cut, calculate sub Cat
        # events_sig = applyGGH_cut(events_sig) # we assume ggH category cut is already done via run_stage2
        # events_bkg = applyGGH_cut(events_bkg) # we assume ggH category cut is already done via run_stage2
        events_sig = applySigReg_cut(events_sig)
        events_bkg = applySigReg_cut(events_bkg)
        

        # calculate the subcategory

        # add to the list to concatenate later
        bkg_event_l.append(events_sig)
        sig_event_l.append(events_bkg)
        
    # concantenate the events and then seperate them
    bkg_event_total = ak.concatenate(bkg_event_l)
    sig_event_total = ak.concatenate(sig_event_l)
    print(f"bkg_event_total: {bkg_event_total}")
    print(f"sig_event_total: {sig_event_total}")






