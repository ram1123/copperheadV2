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





def applyQuickSelection(events):
    """
    apply dijet mass and dijet dR cut (vbf production category without the inverse btage cut applied)
    """
    return events
    # apply njet and nmuons cut first
    # start_len = ak.num(events.Muon.pt, axis=0).compute()

    njets = ak.num(events.Jet, axis=1)
    # nmuons = ak.num(events.Muon, axis=1)
    selection = (njets >= 2) # & (nmuons >= 2)
    events = events[selection]
    # now all events have at least two jets, apply dijet dR and dijet mass cut
    jet1 = events.Jet[:,0]
    jet2 = events.Jet[:,1]
    dijet_dR = jet1.deltaR(jet2)
    dijet = jet1+jet2
    selection = (
        # (dijet_dR > 2.5) 
        # & (dijet.mass > 400)
        (dijet.mass > 350)
    )

    
    events = events[selection]
    # end_len = ak.num(events.Muon.pt, axis=0).compute()
    # print(f" {end_len} events out of {start_len} events passed the selection")

    return events
    
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
    # muons = muons.compute()
    # mu1 = events.Muon[:,0]
    # mu2 = events.Muon[:,1]
    mu1 = muons[:,0]
    mu2 = muons[:,1]
    dimuon = mu1 + mu2
    # dimuon = dimuon[~ak.is_none(dimuon.pt)]
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
    return_dict = {
        "mu1_pt" : mu1.pt,
        "mu2_pt" : mu2.pt,
        "mu1_eta" : mu1.eta,
        "mu2_eta" : mu2.eta,
        "mu1_phi" : mu1.phi,
        "mu2_phi" : mu2.phi,
        "mu1_iso" : mu1.pfRelIso04_all,
        "mu2_iso" : mu2.pfRelIso04_all,
        # "mu_pt" : events.Muon.pt,
        # "mu_eta" : events.Muon.eta,
        # "mu_phi" : events.Muon.phi,
        # "mu_iso" : events.Muon.pfRelIso04_all,
        "dimuon_mass" : dimuon.mass,
        "dimuon_pt" : dimuon.pt,
        "dimuon_eta" : dimuon.eta,
        "dimuon_rapidity" : dimuon.rapidity,
        "dimuon_phi" : dimuon.phi,
        "jet1_pt" : jet1.pt,
        "jet1_eta" : jet1.eta,
        "jet1_phi" : jet1.phi,
        "jet2_pt" : jet2.pt,
        "jet2_eta" : jet2.eta,
        "jet1_mass" : jet1.mass,
        "jet2_mass" : jet2.mass,
        # "jet_pt" : events.Jet.pt,
        # "jet_eta" : events.Jet.eta,
        # "jet_phi" : events.Jet.phi,
        # "jet_mass" : events.Jet.mass,
        "jj_mass" : dijet.mass,
        "jj_pt" : dijet.pt,
        "jj_eta" : dijet.eta,
        "jj_phi" : dijet.phi,
        "jj_dEta" : jj_dEta,
        "jj_dPhi":  jj_dPhi,
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
    return return_dict
    
def getHist(value, binning):
    weights = ak.ones_like(value) # None values are propagated as None here, which is useful, bc we can just override those events with zero weights
    weights = ak.fill_none(weights, value=0)
    weights = weights / ak.sum(weights)
    # print(f"number of nones: {ak.sum(ak.is_none(value))}")
    hist, edges = np.histogram(value, bins=binning, weights=weights)
    hist_w2, edges = np.histogram(value, bins=binning, weights=weights*weights)
    # normalize hist and hist_w2
    hist_orig = hist
    hist = hist / np.sum(hist) 
    hist_err = np.sqrt(hist_w2) /  hist_orig * hist
    return hist, hist_err  

def plotTwoWay(zip_fromScratch, zip_rereco, plot_bins, save_path="./plots"):
    fields2plot = zip_fromScratch.fields
    for field in fields2plot:
        if field not in plot_bins.keys():
            continue
        binning = np.linspace(*plot_bins[field]["binning_linspace"])
        
        fig, (ax_main, ax_ratio) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        hist_fromScratch, hist_fromScratch_err = getHist(zip_fromScratch[field], binning)
        hist_rereco, hist_rereco_err = getHist(zip_rereco[field], binning)        
            
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
        
        ax_main.set_ylabel("A. U.")

        # make ration plot of UL private / RERECO
        ratio_hist = np.zeros_like(hist_fromScratch)
        inf_filter = hist_rereco>0
        ratio_hist[inf_filter] = hist_fromScratch[inf_filter]/  hist_rereco[inf_filter]
        
        rel_unc_ratio = np.sqrt((hist_fromScratch_err/hist_fromScratch)**2 + (hist_rereco_err/hist_rereco)**2)
        ratio_err = rel_unc_ratio*ratio_hist
        
        hep.histplot(ratio_hist, 
                     bins=binning, histtype='errorbar', yerr=ratio_err, 
                     color='black', label='Ratio', ax=ax_ratio)
        
        ax_ratio.axhline(1, color='gray', linestyle='--')
        ax_main.set_xlabel( plot_bins[field].get("xlabel"))
        ax_ratio.set_ylabel('Private UL / Rereco')
        ax_ratio.set_ylim(0.5,1.5) 
        # plt.title(f"{field} distribution of privately produced samples")
        plt.title(f"2018")
        # plt.legend(loc="upper right")
        plt.legend()
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

if __name__ == "__main__":
    client =  Client(n_workers=31,  threads_per_worker=1, processes=True, memory_limit='8 GiB') 
    # gateway = Gateway(
    #     "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
    #     proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
    # )
    # cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
    # client = gateway.connect(cluster_info.name).get_client()

    test_len = 4000
    # test_len = 14000
    # test_len = 40000


    files = json.load(open("new_UL_production.json", "r"))
    events_fromScratch = NanoEventsFactory.from_root(
        files,
        schemaclass=NanoAODSchema,
    ).events()
    events_fromScratch = applyQuickSelection(events_fromScratch)
    # events_fromScratch = events_fromScratch[:test_len]
    zip_fromScratch = getZip(events_fromScratch)
    


    
    rereco_full_files = json.load(open("rereco_central.json", "r"))
    events_rereco = NanoEventsFactory.from_root(
        # rereco_input_dict['dy_m105_160_vbf_amc']['files'],
        rereco_full_files,
        schemaclass=NanoAODSchema,
    ).events()
    events_rereco = applyQuickSelection(events_rereco)
    # events_rereco = events_rereco[:test_len]
    zip_rereco = getZip(events_rereco)

    ul_central_files = json.load(open("UL_central_DY100To200.json", "r"))
    events_ul = NanoEventsFactory.from_root(
        ul_central_files,
        schemaclass=NanoAODSchema,
    ).events()
    events_ul = applyQuickSelection(events_ul)
    # events_ul = events_ul[:test_len]
    zip_ul = getZip(events_ul)


    
    with open("plot_settings.json", "r") as file:
        plot_bins = json.load(file)

    save_path = "./plots/jjMassCut"
    os.makedirs(save_path, exist_ok=True) 
    plotThreeWay(zip_fromScratch, zip_rereco, zip_ul, plot_bins, save_path=save_path)
    plotTwoWay(zip_fromScratch, zip_rereco, plot_bins, save_path=save_path)
    
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