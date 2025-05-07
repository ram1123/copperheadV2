import uproot
import awkward as ak
import matplotlib.pyplot as plt
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from distributed import LocalCluster, Client, progress

import json
import mplhep as hep
import matplotlib.pyplot as plt
plt.style.use(hep.style.CMS)
import numpy as np
from dask_gateway import Gateway
import os
import argparse
import sys
import ROOT
ROOT.gStyle.SetOptStat(0) # remove stats box
from array import array
np.set_printoptions(threshold=sys.maxsize)
import dask


def applyMuonBaseSelection(events):
    muons = events.Muon
    mm_charge = ak.prod(muons.charge, axis=1)
    muon_selection = (
        muons.mediumId
        & (muons.pt > 20 )
        & (abs(muons.eta) < 2.4)
        & (muons.pfRelIso04_all < 0.25)
        & (muons.isGlobal | muons.isTracker)
        & (events.HLT.IsoMu24)
        & (mm_charge < 0)
    )
    nmuons = ak.sum(muon_selection, axis=1)
    # print(f"nmuons len: {ak.num(nmuons, axis=0).compute()}")
    
    return events[nmuons==2]


def applyQuickSelection(events):
    """
    apply dijet mass and dimuon mass cut
    """
    # apply njet and nmuons cut first
    # start_len = ak.num(events.Muon.pt, axis=0).compute()

    # events = applyMuonBaseSelection(events)
    muons = events.Muon
    njets = ak.num(events.Jet, axis=1)
    nmuons = ak.num(muons, axis=1)
    
    # now all events have at least two jets, apply dijet dR and dijet mass cut
    padded_jets = ak.pad_none(events.Jet, target=2)
    jet1 = padded_jets[:,0]
    jet2 = padded_jets[:,1]
    dijet_dR = jet1.deltaR(jet2)
    dijet = jet1+jet2
    padded_muons = ak.pad_none(events.Muon, target=2)
    mu1 = padded_muons[:,0]
    mu2 = padded_muons[:,1]
    dimuon = mu1 + mu2
    # selection = (
    #     (nmuons >= 2)
    #     # (njets >= 2)
    #     # & (dijet.mass > 350)
    #     # & (dimuon.mass > 110)
    #     # & (dimuon.mass < 150)
    # )
    # print(f"selection len: {ak.num(selection, axis=0).compute()}")
    
    selection = ak.ones_like(nmuons, dtype="bool")
    
    events = events[selection]
    # end_len = ak.num(events.Muon.pt, axis=0).compute()
    # print(f" {end_len} events out of {start_len} events passed the selection")

    return events


def getParentID(particle, GenPart):
    has_no_parent = particle.genPartIdxMother == -1
    self_id = particle.pdgId
    parent = GenPart[particle.genPartIdxMother] 
    parent_id = parent.pdgId
    # print(f"has_no_parent: {has_no_parent.compute()}")
    # print(f"self_id: {self_id.compute()}")
    # print(f"GenPart.genPartIdxMother: {GenPart.genPartIdxMother.compute()}")
    # print(f"parent_id: {parent_id.compute()}")
    ParentID = ak.where(has_no_parent, self_id, parent_id)
    return ParentID

def getZip(events) -> ak.zip:
    """
    from events return dictionary of dimuon, muon, dijet, jet values
    we assume all events have at least two jet and two muons
    """
    jets = ak.pad_none(events.Jet, target=2)
    # jets = jets.compute()
    jet1 = jets[:,0]
    jet2 = jets[:,1]
    dijet = jet1 + jet2
    # dijet = dijet[~ak.is_none(dijet.pt)]
    muons = ak.pad_none(events.Muon, target=2)
    mu1 = muons[:,0]
    mu2 = muons[:,1]
    dimuon = mu1 + mu2
    
    gen_idx = events.Muon.genPartIdx
    muons_gen = ak.pad_none(events.GenPart[gen_idx], target=2, clip=True)
    
    mu1_gen = muons_gen[:,0]
    mu2_gen = muons_gen[:,1]
    
    jj_dEta = abs(jet1.eta - jet2.eta)
    jj_dPhi = abs(jet1.delta_phi(jet2))
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
    mmjj = dimuon + dijet
    # flatten variables for muons and jets to convert to 1 dim arrays
    muons = ak.flatten(muons)
    jets = ak.flatten(jets)
    # mmjj = mmjj[~ak.is_none(mmjj.pt)]
    # print(f"mu1.matched_gen.pt: {mu1.matched_gen.pt.compute()}")
    # print(f"mu1_gen.pt: {mu1_gen.pt.compute()}")
    # print(f"mu2_gen.pt: {mu2_gen.pt.compute()}")
    # print(f"mu1_gen.eta: {mu1_gen.eta.compute()}")
    # print(f"mu2_gen.eta: {mu2_gen.eta.compute()}")
    # print(f"events.Muon.matched_gen: {events.Muon.matched_gen.compute()}")

    LHE_part = events.LHEPart
    # selection LHE muons
    LHE_selection = (
        (abs(LHE_part.pdgId) ==13)
        # & (LHE_part.status==1) # nanoV6 doesn't have status
        
    )
    # print(f"len(LHE_part): {ak.num(LHE_part, axis=0).compute()}")
    # print(f"len(LHE_selection): {ak.num(LHE_selection, axis=0).compute()}")
    # print(f"len(LHE_selection): {LHE_selection.compute()}")
    
    # print(f"LHE_part.pdgId: {LHE_part.pdgId.compute()}")

    LHE_muon = LHE_part[LHE_selection]
    two_LHE_muons = (ak.num(LHE_muon, axis=1) == 2) & (ak.prod(LHE_muon.pdgId,axis=1) < 0 )
    LHE_muon = ak.pad_none(LHE_muon[two_LHE_muons], target=2, clip=True)
    mu1_lhe = LHE_muon[:,0]
    mu2_lhe = LHE_muon[:,1]

    has_negCharge = LHE_muon.pdgId > 0 # positive muon id is negative muon
    mu_neg_lhe = LHE_muon[has_negCharge][:,0]
    mu_pos_lhe = LHE_muon[~has_negCharge][:,0]

    genPart = events.GenPart
    # gen_selection = (
    #      (genPart.status ==1) # must be a stable. Source: https://github.com/cms-sw/cmssw/blob/b3c939c01124861dffae4f08177fbc598538c569/PhysicsTools/JetMCAlgos/src/Pythia8PartonSelector.cc#L20
    #     # (abs(genPart.pdgId) ==13)
    #     # & (genPart.status ==23) # must be an outgoing particle. Source: https://pythia.org/latest-manual/ParticleProperties.html
    # )
    gen_selection = (abs(genPart.pdgId) ==13)
    gen_muon = genPart[gen_selection]
    parent_id = getParentID(gen_muon, genPart)
    # # print(f"gen_muon.pdgId: {gen_muon.pdgId.compute()}")
    # parent_Zboson = abs(parent_id) == 23 # parent must be from Z boson. Source: https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf
    # gen_muon = gen_muon[parent_Zboson]
    # print(f"parent_id: {parent_id.compute()}")
    from_hard_process = (gen_muon.statusFlags & 2**8) > 0
    is_stable_process = (gen_muon.status ==1)
    dy_muon_filter = from_hard_process & is_stable_process & (gen_muon.pt > 20)
    gen_muon = gen_muon[dy_muon_filter]
    n_gen_muons = ak.num(gen_muon, axis=1)
    more_than_two = n_gen_muons > 2
    # print(f"more_than_two sum: {ak.sum(more_than_two).compute()}")
    two_gen_muons = (n_gen_muons == 2) & (ak.prod(gen_muon.pdgId,axis=1) < 0 )
    gen_muon = ak.pad_none(gen_muon[two_gen_muons], target=2, clip=True)
    # print(f"gen_muon.pt b4 sort: {gen_muon.pt.compute()}")
    sorted_args = ak.argsort(gen_muon.pt, ascending=False)
    gen_muon = (gen_muon[sorted_args])
    # print(f"gen_muon.pt after sort: {gen_muon.pt.compute()}")
    
    # gen_muon = ak.pad_none(gen_muon, target=2, clip=True)
    mu1_gen = gen_muon[:,0]
    mu2_gen = gen_muon[:,1]
    

    
    return_dict = {
        # "mu1_pt" : mu1.pt,
        # "mu2_pt" : mu2.pt,
        # "mu1_eta" : mu1.eta,
        # "mu2_eta" : mu2.eta,
        # "mu1_phi" : mu1.phi,
        # "mu2_phi" : mu2.phi,
        # "mu1_pt_gen" : mu1_gen.pt,
        # "mu2_pt_gen" : mu2_gen.pt,
        # "mu1_eta_gen" : mu1_gen.eta,
        # "mu2_eta_gen" : mu2_gen.eta,
        # "mu1_phi_gen" : mu1_gen.phi,
        # "mu2_phi_gen" : mu2_gen.phi,
        "mu1_pt_gen" : mu1_gen.pt,
        "mu2_pt_gen" : mu2_gen.pt,
        "mu1_eta_gen" : mu1_gen.eta,
        "mu2_eta_gen" : mu2_gen.eta,
        # "mu1_pt_lhe" : mu1_lhe.pt,
        # "mu2_pt_lhe" : mu2_lhe.pt,
        # "mu1_eta_lhe" : mu1_lhe.eta,
        # "mu2_eta_lhe" : mu2_lhe.eta,
        # "mu1_phi_lhe" : mu1_lhe.phi,
        # "mu2_phi_lhe" : mu2_lhe.phi,
        # "mu_neg_lhe_eta" : mu_neg_lhe.eta,
        # "mu_pos_lhe_eta" : mu_pos_lhe.eta,
        # "mu1_iso" : mu1.pfRelIso04_all,
        # "mu2_iso" : mu2.pfRelIso04_all,
        # "mu_pt" : events.Muon.pt,
        # "mu_eta" : events.Muon.eta,
        # "mu_phi" : events.Muon.phi,
        # "mu_iso" : events.Muon.pfRelIso04_all,
        # "dimuon_mass" : dimuon.mass,
        # "dimuon_pt" : dimuon.pt,
        # "dimuon_eta" : dimuon.eta,
        # "dimuon_rapidity" : dimuon.rapidity,
        # "dimuon_phi" : dimuon.phi,
        # "jet1_pt" : jet1.pt,
        # "jet1_eta" : jet1.eta,
        # "jet1_phi" : jet1.phi,
        # "jet2_pt" : jet2.pt,
        # "jet2_eta" : jet2.eta,
        # "jet1_mass" : jet1.mass,
        # "jet2_mass" : jet2.mass,
        # "jet_pt" : events.Jet.pt,
        # "jet_eta" : events.Jet.eta,
        # "jet_phi" : events.Jet.phi,
        # "jet_mass" : events.Jet.mass,
        # "jj_mass" : dijet.mass,
        # "jj_pt" : dijet.pt,
        # "jj_eta" : dijet.eta,
        # "jj_phi" : dijet.phi,
        # "jj_dEta" : jj_dEta,
        # "jj_dPhi":  jj_dPhi,
        # "mmj1_dEta" : mmj1_dEta,
        # "mmj1_dPhi" : mmj1_dPhi,
        # "mmj1_dR" : mmj1_dR,
        # "mmj2_dEta" : mmj2_dEta,
        # "mmj2_dPhi" : mmj2_dPhi,
        # "mmj2_dR" : mmj2_dR,
        # "mmj_min_dEta" : mmj_min_dEta,
        # "mmj_min_dPhi" : mmj_min_dPhi,
        # "mmjj_pt" : mmjj.pt,
        # "mmjj_eta" : mmjj.eta,
        # "mmjj_phi" : mmjj.phi,
        # "mmjj_mass" : mmjj.mass,
    }
    # comput zip and return
    return_dict = ak.zip(return_dict).compute()
    print(f"return_dict: {return_dict}")
    return return_dict
    
def getHist(value, binning, normalize=True):
    weights = ak.ones_like(value) # None values are propagated as None here, which is useful, bc we can just override those events with zero weights
    weights = ak.fill_none(weights, value=0)
    # print(f"number of nones: {ak.sum(ak.is_none(value))}")
    hist, edges = np.histogram(value, bins=binning, weights=weights)
    hist_w2, edges = np.histogram(value, bins=binning, weights=weights*weights)
    # normalize hist and hist_w2
    hist_orig = hist
    hist_err = np.sqrt(hist_w2)
    if normalize:
        hist = hist / np.sum(hist) 
        hist_err = hist_err /  hist_orig * hist

    # convert to numpy arrays so that we can print the full arr
    hist = ak.to_numpy(hist)
    hist_err = ak.to_numpy(hist_err)
    return hist, hist_err  

def plotTwoWay(zip_fromScratch, zip_rereco, plot_bins, save_path="./plots"):
    fields2plot = zip_fromScratch.fields
    for field in fields2plot:
        if field not in plot_bins.keys():
            continue
        binning = np.linspace(*plot_bins[field]["binning_linspace"])
        print(f"{field} binning len: {len(binning)}")
        
        fig, (ax_main, ax_ratio) = plt.subplots(2, 1, figsize=(10, 13), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        # fig, (ax_main, ax_ratio) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1]}, sharex=True)
        
        hist_fromScratch, hist_fromScratch_err = getHist(zip_fromScratch[field], binning)
        hist_rereco, hist_rereco_err = getHist(zip_rereco[field], binning)        
        print(f"{field} rel hist_fromScratch_err: {hist_fromScratch_err/hist_fromScratch}")
        
        hep.histplot(hist_fromScratch, bins=binning, 
                 histtype='errorbar', 
                label="UL private", 
                 xerr=True, 
                 yerr=(hist_fromScratch_err),
                color = "blue",
                ax=ax_main
        )
        hep.histplot(hist_rereco, bins=binning, 
                 histtype='errorbar', 
                label="RERECO central", 
                 xerr=True, 
                 yerr=(hist_rereco_err),
                color = "red",
                ax=ax_main
        )
        
        ax_main.set_ylabel("A. U.")
        ax_main.legend()
        ax_main.set_title(f"2018")

        # make ration plot of UL private / RERECO
        hist_fromScratch = ak.to_numpy(hist_fromScratch)
        hist_rereco = ak.to_numpy(hist_rereco)
        # ratio_hist = np.zeros_like(hist_fromScratch)
        # inf_filter = hist_rereco>0
        # ratio_hist[inf_filter] = hist_fromScratch[inf_filter]/  hist_rereco[inf_filter]
        
        # rel_unc_ratio = np.sqrt((hist_fromScratch_err/hist_fromScratch)**2 + (hist_rereco_err/hist_rereco)**2)
        # ratio_err = rel_unc_ratio*ratio_hist
        
        # hep.histplot(ratio_hist, 
        #              bins=binning, histtype='errorbar', yerr=ratio_err, 
        #              color='black', label= 'Ratio', ax=ax_ratio)
        
        # ax_ratio.axhline(1, color='gray', linestyle='--')
        # ax_ratio.set_xlabel( plot_bins[field].get("xlabel"))
        # ax_ratio.set_ylabel('UL / Rereco')
        # ax_ratio.set_ylim(0.5,1.5) 
        diff_hist = hist_fromScratch - hist_rereco
        rel_unc_diff = np.sqrt((hist_fromScratch_err/hist_fromScratch)**2 + (hist_rereco_err/hist_rereco)**2)
        ratio_err = np.abs(rel_unc_diff*diff_hist)
        
        hep.histplot(diff_hist, 
                     bins=binning, histtype='errorbar', yerr=ratio_err, 
                     color='black', label= 'Difference', ax=ax_ratio)
        
        ax_ratio.axhline(1, color='gray', linestyle='--')
        ax_ratio.set_xlabel( plot_bins[field].get("xlabel"))
        ax_ratio.set_ylabel('UL - Rereco')
        ax_ratio.set_ylim(-0.01, 0.01) 
        plt.tight_layout()
        
        
        # plt.show()
        save_full_path = f"{save_path}/TwoWayPrivateProd_{field}.pdf"
        plt.savefig(save_full_path)
        plt.clf()


def plotTwoWayCentral(zip_fromScratch, zip_rereco, plot_bins, save_path="./plots"):
    fields2plot = zip_fromScratch.fields
    for field in fields2plot:
        if field not in plot_bins.keys():
            continue
        binning = np.linspace(*plot_bins[field]["binning_linspace"])
        print(f"{field} binning len: {len(binning)}")
        
        fig, (ax_main, ax_ratio) = plt.subplots(2, 1, figsize=(10, 13), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        # fig, (ax_main, ax_ratio) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1]}, sharex=True)
        
        hist_fromScratch, hist_fromScratch_err = getHist(zip_fromScratch[field], binning)
        hist_rereco, hist_rereco_err = getHist(zip_rereco[field], binning)        
        print(f"{field} rel hist_fromScratch_err: {hist_fromScratch_err/hist_fromScratch}")
        
        hep.histplot(hist_fromScratch, bins=binning, 
                 histtype='errorbar', 
                label="UL central", 
                 xerr=True, 
                 yerr=(hist_fromScratch_err),
                color = "blue",
                ax=ax_main
        )
        hep.histplot(hist_rereco, bins=binning, 
                 histtype='errorbar', 
                label="RERECO central", 
                 xerr=True, 
                 yerr=(hist_rereco_err),
                color = "red",
                ax=ax_main
        )
        
        ax_main.set_ylabel("A. U.")
        ax_main.legend()
        ax_main.set_title(f"2018")

        # make ration plot of UL private / RERECO
        hist_fromScratch = ak.to_numpy(hist_fromScratch)
        hist_rereco = ak.to_numpy(hist_rereco)
        # ratio_hist = np.zeros_like(hist_fromScratch)
        # inf_filter = hist_rereco>0
        # ratio_hist[inf_filter] = hist_fromScratch[inf_filter]/  hist_rereco[inf_filter]
        
        # rel_unc_ratio = np.sqrt((hist_fromScratch_err/hist_fromScratch)**2 + (hist_rereco_err/hist_rereco)**2)
        # ratio_err = rel_unc_ratio*ratio_hist
        
        # hep.histplot(ratio_hist, 
        #              bins=binning, histtype='errorbar', yerr=ratio_err, 
        #              color='black', label= 'Ratio', ax=ax_ratio)
        
        # ax_ratio.axhline(1, color='gray', linestyle='--')
        # ax_ratio.set_xlabel( plot_bins[field].get("xlabel"))
        # ax_ratio.set_ylabel('UL / Rereco')
        # ax_ratio.set_ylim(0.5,1.5) 
        diff_hist = hist_fromScratch - hist_rereco
        rel_unc_diff = np.sqrt((hist_fromScratch_err/hist_fromScratch)**2 + (hist_rereco_err/hist_rereco)**2)
        ratio_err = np.abs(rel_unc_diff*diff_hist)
        
        hep.histplot(diff_hist, 
                     bins=binning, histtype='errorbar', yerr=ratio_err, 
                     color='black', label= 'Difference', ax=ax_ratio)
        
        ax_ratio.axhline(1, color='gray', linestyle='--')
        ax_ratio.set_xlabel( plot_bins[field].get("xlabel"))
        ax_ratio.set_ylabel('UL - Rereco')
        ax_ratio.set_ylim(-0.01, 0.01) 
        plt.tight_layout()
        
        
        # plt.show()
        save_full_path = f"{save_path}/TwoWayPrivateProd_{field}.pdf"
        plt.savefig(save_full_path)
        plt.clf()
    
def plotThreeWay(zip_fromScratch, zip_rereco, zip_ul, plot_bins, save_path="./plots"):
    fields2plot = zip_fromScratch.fields
    for field in fields2plot:
        if field not in plot_bins.keys():
            continue
        binning = np.linspace(*plot_bins[field]["binning_linspace"])
        
        fig, ax_main = plt.subplots()
        
        hist_fromScratch, hist_fromScratch_err = getHist(zip_fromScratch[field], binning)
        hist_rereco, hist_rereco_err = getHist(zip_rereco[field], binning)
        hist_UL, hist_UL_err = getHist(zip_ul[field], binning)
        
            
        hep.histplot(hist_fromScratch, bins=binning, 
                 histtype='errorbar', 
                label="UL private production", 
                 xerr=True, 
                 yerr=(hist_fromScratch_err),
                color = "blue",
                ax=ax_main
        )
        hep.histplot(hist_rereco, bins=binning, 
                 histtype='errorbar', 
                label="RERECO central production", 
                 xerr=True, 
                 yerr=(hist_rereco_err),
                color = "red",
                ax=ax_main
        )
        hep.histplot(hist_UL, bins=binning, 
                 histtype='errorbar', 
                label="UL central production", 
                 xerr=True, 
                 yerr=(hist_UL_err),
                color = "black",
                ax=ax_main
        )
        
        ax_main.set_xlabel( plot_bins[field].get("xlabel"))
        ax_main.set_ylabel("A. U.")
        # plt.title(f"{field} distribution of privately produced samples")
        plt.title(f"2018")
        # plt.legend(loc="upper right")
        plt.legend()
        # plt.show()
        save_full_path = f"{save_path}/ThreeWayPrivateProd_{field}.pdf"
        plt.savefig(save_full_path)
        plt.clf()


def plotIndividual(ak_zip, plot_bins, save_fname, save_path="./plots"):
    # fields2plot = ["mu1_eta", "mu2_eta"]
    # fields2plot = ["mu1_eta", "mu2_eta", "mu1_eta_gen", "mu2_eta_gen"]
    # fields2plot = ["mu1_eta_gen", "mu2_eta_gen"]
    # fields2plot = ["mu1_eta_gen", "mu2_eta_gen"]
    fields2plot = ["mu1_eta_lhe", "mu2_eta_lhe"]
    
    for field in fields2plot:
        field4plot_setting = field.replace("_gen","")
        if field4plot_setting not in plot_bins.keys():
            print(f"skipping {field4plot_setting}!")
            continue
        binning = np.linspace(*plot_bins[field4plot_setting]["binning_linspace"])
        
        fig, ax_main = plt.subplots()
        
        hist, hist_err = getHist(ak_zip[field], binning, normalize=False)
        
            
        hep.histplot(hist, bins=binning, 
                 histtype='step', 
                label="", 
                 xerr=True, 
                 yerr=(hist_err),
                color = "black",
                ax=ax_main,
                # flow="sum"
        )
        
        # hep.histplot(hist, bins=binning, 
        #          histtype='band', 
        #         label="", 
        #          xerr=True, 
        #          yerr=(hist_err),
        #         color = "blue",
        #         ax=ax_main
        # )
        
        ax_main.set_xlabel( plot_bins[field4plot_setting].get("xlabel"))
        # ax_main.set_ylabel("A. U.")
        ax_main.set_ylabel("Events")
        # plt.title(f"{field} distribution of privately produced samples")
        plt.title(f"2018")
        # plt.legend(loc="upper right")
        plt.legend()
        # plt.show()
        save_full_path = f"{save_path}/{save_fname}_{field}.pdf"
        plt.savefig(save_full_path)
        plt.clf()


def plotIndividualROOT(ak_zip, plot_bins, save_fname, save_path="./plots"):
    # fields2plot = ["mu1_eta", "mu2_eta"]
    # fields2plot = ["mu1_eta", "mu2_eta", "mu1_eta_gen", "mu2_eta_gen"]
    fields2plot = ["mu1_eta_gen", "mu2_eta_gen", "mu1_pt_gen", "mu2_pt_gen"]
    # fields2plot = ["mu1_eta_lhe", "mu2_eta_lhe"]
    
    for field in fields2plot:
        if "eta" in field:
            xlow, xhigh = -2, 2
        elif "pt" in field:
            xlow, xhigh = 0, 250
        # Create a histogram: name, title, number of bins, xlow, xhigh
        hist = ROOT.TH1F("MC hist", f"2018;{field};Entries", 30, xlow, xhigh)

        values = ak.to_numpy(ak_zip[field])
        weights = np.ones_like(values)
        values = array('d', values) # make the array double
        weights = array('d', weights) # make the array double
        
        print(len(values))
        hist.FillN(len(values), values, weights)
        
        # Create a canvas
        canvas = ROOT.TCanvas("canvas", "Canvas for Plotting", 800, 600)
        
        # Draw the histogram
        hist.Draw('E')

        # Create a legend
        legend = ROOT.TLegend(0.35, 0.8, 0.65, 0.93)  # (x1,y1,x2,y2) in NDC coordinates
        
        # Add entries
        legend.AddEntry(hist, f"Entries: {hist.GetEntries():.2e}", "l")  # "l" means line
        legend.Draw()
        
        # Save the canvas as a PNG
        save_full_path = f"{save_path}/{save_fname}_{field}_ROOT.pdf"
        canvas.Update()
        canvas.SaveAs(save_full_path)

def print_t_statisticROOT(hist_fromScratch, hist_rereco, field):
    h_diff = hist_fromScratch.Clone("h_diff") 
    
    # debug -------------------------------
    # for i in range(1, h_diff.GetNbinsX() + 1):
    #     bin_center = h_diff.GetBinCenter(i)
    #     content = h_diff.GetBinContent(i)
    #     error = h_diff.GetBinError(i)
    #     print(f"{field} Bin {i}: center={bin_center:.2f}, content={content:.2f}, error={error:.2f}, ")
    # print("-------------------------------------------------")
    # for i in range(1, hist_rereco.GetNbinsX() + 1):
    #     bin_center = hist_rereco.GetBinCenter(i)
    #     content = hist_rereco.GetBinContent(i)
    #     error = hist_rereco.GetBinError(i)
    #     print(f"{field} Bin {i}: center={bin_center:.2f}, content={content:.2f}, error={error:.2f}, ")
    # print("-------------------------------------------------")
    # debug -------------------------------

    h_diff.Add(hist_rereco, -1)        
    for i in range(1, h_diff.GetNbinsX() + 1):
        bin_center = h_diff.GetBinCenter(i)
        diff = h_diff.GetBinContent(i)
        err_ul = hist_fromScratch.GetBinError(i) 
        err_rereco = hist_rereco.GetBinError(i) 
        err_total = (err_ul**2 + err_rereco**2)**(1/2)
        if err_total != 0:
            t_val = diff/err_total
        else:
            t_val = -999.0
        print(f"{field} Bin {i}: center={bin_center:.2f}, diff={diff:.2f}, error={err_total:.2f}, t value={t_val:.2f}")


def plotTwoWayROOT(zip_fromScratch, zip_rereco, plot_bins, save_path="./plots", centralVsCentral=False):
    # fields2plot = ["mu1_eta_lhe", "mu2_eta_lhe", "mu_neg_lhe_eta", "mu_pos_lhe_eta"]
    # fields2plot = ["mu1_eta_gen", "mu2_eta_gen"]
    fields2plot = ["mu1_eta_gen", "mu2_eta_gen", "mu1_pt_gen", "mu2_pt_gen"]
    
    
    for field in fields2plot:
        # Create a histogram: name, title, number of bins, xlow, xhigh
        if "eta" in field:
            xlow, xhigh = -2, 2
        elif "pt" in field:
            xlow, xhigh = 0, 250
        # nbins = 30
        nbins = 60
        
        hist_fromScratch = ROOT.TH1F("hist_fromScratch", f"2018;{field};Entries", nbins, xlow, xhigh)

        values = ak.to_numpy(zip_fromScratch[field])
        weights = np.ones_like(values)
        values = array('d', values) # make the array double
        weights = array('d', weights) # make the array double
        
        print(len(values))
        hist_fromScratch.FillN(len(values), values, weights)

        hist_rereco = ROOT.TH1F("hist_rereco", f"2018;{field};Entries", nbins, xlow, xhigh)
        values = ak.to_numpy(zip_rereco[field])
        weights = np.ones_like(values)
        values = array('d', values) # make the array double
        weights = array('d', weights) # make the array double
        
        print(len(values))
        hist_rereco.FillN(len(values), values, weights)
        
        # Create a canvas
        canvas = ROOT.TCanvas("canvas", "Canvas for Plotting", 800, 600)

        # Style histograms
        hist_fromScratch.SetLineColor(ROOT.kRed)
        hist_rereco.SetLineColor(ROOT.kBlue)
        
        # hist_fromScratch.SetMarkerStyle(20)  # Add markers
        # hist_rereco.SetMarkerStyle(21)

        hist_fromScratch.SetMarkerColor(ROOT.kRed)
        hist_rereco.SetMarkerColor(ROOT.kBlue)
        
        # Draw the histogram
        
        hist_fromScratch.Draw('E')
        hist_rereco.Draw('E SAME')


        # Create a legend
        if centralVsCentral:
            legend = ROOT.TLegend(0.35, 0.1, 0.65, 0.23)  # (x1,y1,x2,y2) in NDC coordinates
        else:
            legend = ROOT.TLegend(0.35, 0.8, 0.65, 0.93)  # (x1,y1,x2,y2) in NDC coordinates
            
        # Add entries
        legend.AddEntry(hist_fromScratch, f"Private UL (Entries: {hist_fromScratch.GetEntries():.2e})", "l")  # "l" means line
        legend.AddEntry(hist_rereco, f"Central Rereco (Entries: {hist_rereco.GetEntries():.2e})", "l")
        legend.Draw()

        
        # Save the canvas as a PNG
        save_full_path = f"{save_path}/plotTwoWay_{field}_ROOT.pdf"
        canvas.Update()
        canvas.SaveAs(save_full_path)

        

        # print t statistic
        print_t_statisticROOT(hist_fromScratch, hist_rereco, field)

def print_t_statistic(zip_fromScratch, zip_rereco, plot_bins, normalize=True):
    # fields2plot = ["mu1_eta", "mu2_eta"]
    fields2plot = ["mu1_eta_lhe", "mu2_eta_lhe", "mu_neg_lhe_eta", "mu_pos_lhe_eta"]
    
    for field in fields2plot:
        # binning = np.linspace(*plot_bins[field4plot_setting]["binning_linspace"])
        binning = np.linspace(-2, 2, 31)
        # print(f"{field} binning len: {len(binning)}")
        print(f"printing {field}")
        
        hist_fromScratch, hist_fromScratch_err = getHist(zip_fromScratch[field], binning, normalize=normalize)
        hist_rereco, hist_rereco_err = getHist(zip_rereco[field], binning, normalize=normalize)    

        if normalize:
            print(f"Normalized UL private production hist: \n {hist_fromScratch}")
            print(f"Normalized UL private production hist err: \n {hist_fromScratch_err}")
            print(f"Normalized Rereco central production hist: \n {hist_rereco}")
            print(f"Normalized UL hist - Rereco hist: \n {(hist_fromScratch-hist_rereco)}")
            print(f"T-statistic: \n {(hist_fromScratch-hist_rereco)/hist_fromScratch_err}")
        else:
            print(f"UL private production hist: \n {hist_fromScratch}")
            print(f"UL private production hist err: \n {hist_fromScratch_err}")
            print(f"Rereco central production hist: \n {hist_rereco}")
            print(f"UL hist - Rereco hist: \n {(hist_fromScratch-hist_rereco)}")
            print(f"T-statistic: \n {(hist_fromScratch-hist_rereco)/hist_fromScratch_err}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-sel",
    "--selection",
    dest="selection",
    default=None,
    action="store",
    help="save path to store stage1 output files",
    )
    parser.add_argument(
    "--centralVsCentral",
    dest="centralVsCentral",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, compare centrally produced UL nanoV9 with centrally produced Rereco nanoV6 samples (dy M50)",
    )
    print("programe start")
    # ---------------------------------------------------------------
    
    client =  Client(n_workers=50,  threads_per_worker=1, processes=True, memory_limit='10 GiB') 
    # ---------------------------------------------------------------
    # gateway = Gateway(
    #     "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
    #     proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
    # )
    # cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
    # client = gateway.connect(cluster_info.name).get_client()

    # ---------------------------------------------------------------
    print(f"client: {client}")
    args = parser.parse_args()
    
    do_quick_test = True # for quick test
    # test_len = 4000
    
    # test_len = 14000
    # test_len = 400000
    # test_len = 4000000
    test_len = 8000000
    # test_len = 2*8000000

    centralVsCentral = args.centralVsCentral

    if centralVsCentral:
        files = json.load(open("dy_m50_v9.json", "r"))
    else:    
        files = json.load(open("new_UL_production.json", "r"))
    events_fromScratch = NanoEventsFactory.from_root(
        files,
        schemaclass=NanoAODSchema,
    ).events()
    if do_quick_test:
        events_fromScratch = events_fromScratch[:test_len]
    events_fromScratch = applyQuickSelection(events_fromScratch)

    print("starting UL zip!")
    zip_fromScratch = getZip(events_fromScratch)
    
    print("done UL zip!")


    if centralVsCentral:
        rereco_full_files = json.load(open("dy_m50_v6.json", "r"))
    else:
        rereco_full_files = json.load(open("rereco_central.json", "r"))
    events_rereco = NanoEventsFactory.from_root(
        rereco_full_files,
        schemaclass=NanoAODSchema,
    ).events()
    if do_quick_test:
        events_rereco = events_rereco[:test_len]

    # events_rereco = events_rereco[:test_len] # for some reason, rereco seems to get stuck when I read the full in
    events_rereco = applyQuickSelection(events_rereco)
    # print(f"events_rereco nevents: {ak.num(events_rereco, axis=0).compute()}")

    print("starting rereco zip!")
    zip_rereco = getZip(events_rereco)
    print("done rereco zip!")
    

    
    
    with open("plot_settings.json", "r") as file:
        plot_bins = json.load(file)

    save_path = "./plots"
    os.makedirs(save_path, exist_ok=True)

    # --------------------------------------------------------------
    
    # plot two way

    plotTwoWayROOT(zip_fromScratch, zip_rereco, plot_bins, save_path=save_path, centralVsCentral=centralVsCentral)
    
    # plotTwoWay(zip_fromScratch, zip_rereco, plot_bins, save_path=save_path)
    # plotTwoWayCentral(zip_fromScratch, zip_rereco, plot_bins, save_path=save_path)
    # --------------------------------------------------------------
    
    # do individual plots
    save_fname = "UL_private_prod"
    # plotIndividual(zip_fromScratch, plot_bins, save_fname, save_path=save_path)
    plotIndividualROOT(zip_fromScratch, plot_bins, save_fname, save_path=save_path)
    save_fname = "Rereco_private_prod"
    # plotIndividual(zip_rereco, plot_bins, save_fname, save_path=save_path)
    plotIndividualROOT(zip_rereco, plot_bins, save_fname, save_path=save_path) # printing T statistic is included in this funcition

    

    raise ValueError
    
    
    # # now plot three way

    # ul_central_files = json.load(open("UL_central_DY100To200.json", "r"))
    # events_ul = NanoEventsFactory.from_root(
    #     ul_central_files,
    #     schemaclass=NanoAODSchema,
    # ).events()
    # if do_quick_test:
    #     events_ul = events_ul[:test_len]
    # events_ul = applyQuickSelection(events_ul)
    
    # zip_ul = getZip(events_ul)


    
    
    # plotThreeWay(zip_fromScratch, zip_rereco, zip_ul, plot_bins, save_path=save_path)
    # ----------------------------------------------

    
    # fields2plot = zip_fromScratch.fields
    # for field in fields2plot:
    #     if field not in plot_bins.keys():
    #         continue
    #     binning = np.linspace(*plot_bins[field]["binning_linspace"])
        
    #     fig, ax_main = plt.subplots()
        
    #     hist_fromScratch, hist_fromScratch_err = getHist(zip_fromScratch[field], binning)
    #     hist_rereco, hist_rereco_err = getHist(zip_rereco[field], binning)
    #     hist_UL, hist_UL_err = getHist(zip_ul[field], binning)
        
            
    #     hep.histplot(hist_fromScratch, bins=binning, 
    #              histtype='errorbar', 
    #             label="UL private production", 
    #              xerr=True, 
    #              yerr=(hist_fromScratch_err),
    #             color = "blue",
    #             ax=ax_main
    #     )
    #     hep.histplot(hist_rereco, bins=binning, 
    #              histtype='errorbar', 
    #             label="RERECO central production", 
    #              xerr=True, 
    #              yerr=(hist_rereco_err),
    #             color = "red",
    #             ax=ax_main
    #     )
    #     hep.histplot(hist_UL, bins=binning, 
    #              histtype='errorbar', 
    #             label="UL central production", 
    #              xerr=True, 
    #              yerr=(hist_UL_err),
    #             color = "black",
    #             ax=ax_main
    #     )
        
    #     ax_main.set_xlabel( plot_bins[field].get("xlabel"))
    #     ax_main.set_ylabel("A. U.")
    #     # plt.title(f"{field} distribution of privately produced samples")
    #     plt.title(f"2018")
    #     # plt.legend(loc="upper right")
    #     plt.legend()
    #     # plt.show()
    #     save_full_path = f"./plots/PrivateProd_{field}.pdf"
    #     # save_full_path = f"./quick_plots/PrivateProd_{field}.png"
    #     plt.savefig(save_full_path)
    #     plt.clf()


    print("Success!")