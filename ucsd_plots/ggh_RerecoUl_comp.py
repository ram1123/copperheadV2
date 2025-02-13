import dask_awkward as dak
import awkward as ak
from distributed import LocalCluster, Client, progress
import time
import numpy as np
import matplotlib.pyplot as plt
import json
import mplhep as hep
import glob
import pandas as pd

plt.style.use(hep.style.CMS)



"""
Compare ggH MC histograms to see if there's a diff in sigma
"""
import glob
import dask_awkward as dak
import awkward as ak
from distributed import LocalCluster, Client, progress
import time
import numpy as np
import pandas as pd
import ROOT as rt

def applyVBF_cutV1(events):
    btag_cut =ak.fill_none((events.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((events.nBtagMedium_nominal >= 1), value=False)
    print(f"sum btag_cut: {np.sum(btag_cut)}")
    # vbf_cut = ak.fill_none(events.vbf_cut, value=False
    vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35) 
    vbf_cut = ak.fill_none(vbf_cut, value=False)
    # region = events.h_peak 
    # region = events.h_sidebands | events.h_peak
    # region = events.h_sidebands 
    dimuon_mass = events.dimuon_mass

    VBF_filter = (
        vbf_cut & 
        # region &
        ~btag_cut # btag cut is for VH and ttH categories
    )
    trues = ak.ones_like(dimuon_mass, dtype="bool")
    falses = ak.zeros_like(dimuon_mass, dtype="bool")
    events["vbf_filter"] = ak.where(VBF_filter, trues,falses)
    print(f"sum (events.jj_mass_nominal > 400): {np.sum((events.jj_mass_nominal > 400))}")
    print(f"sum (events.jj_dEta_nominal > 2.5): {np.sum((events.jj_dEta_nominal > 2.5))}")
    print(f"sum  (events.jet1_pt_nominal > 35) : {np.sum( (events.jet1_pt_nominal > 35) )}")
    print(f"sum vbf_cut: {np.sum(vbf_cut)}")
    print(f"sum VBF_filter: {np.sum(VBF_filter)}")
    print(f"events.jj_mass_nominal: {events.jj_mass_nominal}")
    return events[VBF_filter]
    # return events

def applyGGH_cutV1(events):
    btag_cut =ak.fill_none((events.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((events.nBtagMedium_nominal >= 1), value=False)
    vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35) 
    vbf_cut = ak.fill_none(vbf_cut, value=False)
    dimuon_mass = events.dimuon_mass
    # region = ((dimuon_mass > 110) & (dimuon_mass < 115.03)) | ((dimuon_mass > 135.03) & (dimuon_mass < 150))
    # region = events.region == "h-peak"
    # region = events.region == "h-sidebands"
    ggH_filter = (
        ~vbf_cut & 
        ~btag_cut # btag cut is for VH and ttH categories
    )
    return events[ggH_filter]

def filterRegion(events, region="h-peak"):
    dimuon_mass = events.dimuon_mass
    if region =="h-peak":
        region = (dimuon_mass > 115.03) & (dimuon_mass < 135.03)
    elif region =="h-sidebands":
        region = ((dimuon_mass > 110) & (dimuon_mass < 115.03)) | ((dimuon_mass > 135.03) & (dimuon_mass < 150))
    elif region =="signal":
        region = (dimuon_mass >= 110) & (dimuon_mass <= 150.0)
    elif region =="z-peak":
        region = (dimuon_mass >= 70) & (dimuon_mass <= 110.0)

    events = events[region]
    return events

def generateRooHist(mass, events, name=""):
    dimuon_mass = ak.to_numpy(events.dimuon_mass)
    wgts = ak.to_numpy(events.wgt_nominal)
    nbins = mass.getBins()
    TH = rt.TH1D("TH", "TH", nbins, mass.getMin(), mass.getMax())
    TH.FillN(len(dimuon_mass), dimuon_mass, wgts) # fill the histograms with mass and weights 
    roohist = rt.RooDataHist(name, name, rt.RooArgSet(mass), TH)
    return roohist

def normalizeRooHist(x: rt.RooRealVar,rooHist: rt.RooDataHist) -> rt.RooDataHist :
    """
    Takes rootHistogram and returns a new copy with histogram values normalized to sum to one
    """
    x_name = x.GetName()
    THist = rooHist.createHistogram(x_name).Clone("clone") # clone it just in case
    THist.Scale(1/THist.Integral())
    print(f"THist.Integral(): {THist.Integral()}")
    normalizedHist_name = rooHist.GetName() + "_normalized"
    roo_hist_normalized = rt.RooDataHist(normalizedHist_name, normalizedHist_name, rt.RooArgSet(x), THist) 
    return roo_hist_normalized

V1_fields_2compute = [
    "wgt_nominal",
    "dimuon_mass",
]











if __name__ == "__main__":
    client =  Client(n_workers=15,  threads_per_worker=2, processes=True, memory_limit='8 GiB') 

    rereco_events = []
    rereco_years = ["2016", "2017", "2018"]
    for year in rereco_years:
        rereco_label = "rereco_yun_Dec05_btagSystFixed_JesJerUncOn"
        rereco_load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean//{rereco_label}//stage1_output/{year}/"
        file = f"{rereco_load_path}/ggh_amcPS"
        print(f"file: {file}")
    
        
        rereco_events_data = dak.from_parquet(f"{file}/*.parquet")
        
        rereco_events_data = filterRegion(rereco_events_data, region="signal")
        rereco_events_data = applyGGH_cutV1(rereco_events_data)
        rereco_events_data = ak.zip({field: rereco_events_data[field] for field in V1_fields_2compute}).compute()
        print(len(rereco_events_data))
        # print((rereco_events_data))
        rereco_events.append(rereco_events_data)
        # print(rereco_events_data)
        # raise ValueError
        
    rereco_events = ak.concatenate(rereco_events, axis=0)
    print(f"rereco_events len: {len(rereco_events)}")
    print(f"rereco_events sum wgts: {ak.sum(rereco_events.wgt_nominal)}")
    
    
    ul_events = []
    ul_years = ["2016preVFP", "2016postVFP","2017", "2018"]
    for year in ul_years:
        ul_label = "V2_Jan29_JecOn_TrigMatchFixed_2016UlJetIdFix"
        ul_load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean/{ul_label}/stage1_output/{year}/f1_0"
        file = f"{ul_load_path}/ggh_powhegPS"
        print(f"file: {file}")
        
        ul_events_data = dak.from_parquet(f"{file}/*/*.parquet")
        
        ul_events_data = filterRegion(ul_events_data, region="signal")
        ul_events_data = applyGGH_cutV1(ul_events_data)
        ul_events_data = ak.zip({field: ul_events_data[field] for field in V1_fields_2compute}).compute()
        print(len(ul_events_data))
        # print((ul_events_data))
        ul_events.append(ul_events_data)
        # print(ul_events_data)
        # raise ValueError
        
    ul_events = ak.concatenate(ul_events, axis=0)
    print(f"ul_events len: {len(ul_events)}")
    print(f"ul_events sum wgts: {ak.sum(ul_events.wgt_nominal)}")



    mass_name = "mh_ggh"
    mass = rt.RooRealVar(mass_name, mass_name, 120, 110, 150)
    nbins = 100
    mass.setBins(nbins)
    rereco_hist  = generateRooHist(mass, rereco_events, name="rereco hist")
    rereco_hist = normalizeRooHist(mass, rereco_hist)
    
    ul_hist  = generateRooHist(mass, ul_events, name="ul hist")
    ul_hist = normalizeRooHist(mass, ul_hist)
    
    
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    frame.SetTitle(f"Normalized ggH sample comparison for Run2")
    frame.SetXTitle(f"Dimuon Mass (GeV)")
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    
    name = "ggH MC sample"
    legend.AddEntry("", name, "")
    
    model_name = rereco_hist.GetName()
    rereco_hist.plotOn(frame,  rt.RooFit.DrawOption("E"), Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1), f"RERECO amc", "L")
    legend.AddEntry("", f"Sigma RERECO: {rereco_hist.sigma(mass):.5f}",  "")
    
    
    
    model_name = ul_hist.GetName()
    ul_hist.plotOn(frame,  rt.RooFit.DrawOption("E"), Name=name, LineColor=rt.kBlue)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1), f"UL powheg", "L")
    legend.AddEntry("", f"Sigma UL: {ul_hist.sigma(mass):.5f}",  "")
    
    
    frame.Draw()
    legend.Draw() 
    canvas.SetTicks(2, 2)
    canvas.Update()
    canvas.Draw()
    
    canvas.SaveAs(f"gghMC_RERECO_amc_vs_UL_powheg.pdf")



    """
    do the same, but with powheg samples. Weirdly, 2016 rereco powheg samples are non-existent, so skip 2016
    """
    
    rereco_events = []
    rereco_years = ["2017", "2018"]
    for year in rereco_years:
        rereco_load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean//rereco_yun_Dec05_btagSystFixed_JesJerUncOn//stage1_output/{year}/"
        file = f"{rereco_load_path}/ggh_powhegPS"
        print(f"file: {file}")
    
        
        rereco_events_data = dak.from_parquet(f"{file}/*.parquet")
        
        rereco_events_data = filterRegion(rereco_events_data, region="signal")
        rereco_events_data = applyGGH_cutV1(rereco_events_data)
        rereco_events_data = ak.zip({field: rereco_events_data[field] for field in V1_fields_2compute}).compute()
        print(len(rereco_events_data))
        # print((rereco_events_data))
        rereco_events.append(rereco_events_data)
        # print(rereco_events_data)
        # raise ValueError
        
    rereco_events = ak.concatenate(rereco_events, axis=0)
    print(f"rereco_events len: {len(rereco_events)}")
    print(f"rereco_events sum wgts: {ak.sum(rereco_events.wgt_nominal)}")
    
    
    ul_events = []
    ul_years = ["2017", "2018"]
    for year in ul_years:
        ul_load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean//V2_Jan29_JecOn_TrigMatchFixed_2016UlJetIdFix/stage1_output/{year}/f1_0"
        file = f"{ul_load_path}/ggh_powhegPS"
        print(f"file: {file}")
        
        ul_events_data = dak.from_parquet(f"{file}/*/*.parquet")
        
        ul_events_data = filterRegion(ul_events_data, region="signal")
        ul_events_data = applyGGH_cutV1(ul_events_data)
        ul_events_data = ak.zip({field: ul_events_data[field] for field in V1_fields_2compute}).compute()
        print(len(ul_events_data))
        # print((ul_events_data))
        ul_events.append(ul_events_data)
        # print(ul_events_data)
        # raise ValueError
        
    ul_events = ak.concatenate(ul_events, axis=0)
    print(f"ul_events len: {len(ul_events)}")
    print(f"ul_events sum wgts: {ak.sum(ul_events.wgt_nominal)}")
    
    
    mass_name = "mh_ggh"
    mass = rt.RooRealVar(mass_name, mass_name, 120, 110, 150)
    nbins = 100
    mass.setBins(nbins)
    rereco_hist  = generateRooHist(mass, rereco_events, name="rereco hist")
    rereco_hist = normalizeRooHist(mass, rereco_hist)
    
    ul_hist  = generateRooHist(mass, ul_events, name="ul hist")
    ul_hist = normalizeRooHist(mass, ul_hist)
    
    
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    frame.SetTitle(f"Normalized ggH sample comparison for 2017 and 2018")
    frame.SetXTitle(f"Dimuon Mass (GeV)")
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    
    name = "ggH MC sample"
    legend.AddEntry("", name, "")
    
    model_name = rereco_hist.GetName()
    rereco_hist.plotOn(frame,  rt.RooFit.DrawOption("E"), Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1), f"RERECO powheg", "L")
    legend.AddEntry("", f"Sigma RERECO: {rereco_hist.sigma(mass):.5f}",  "")
    
    
    
    model_name = ul_hist.GetName()
    ul_hist.plotOn(frame,  rt.RooFit.DrawOption("E"), Name=name, LineColor=rt.kBlue)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1), f"UL powheg", "L")
    legend.AddEntry("", f"Sigma UL: {ul_hist.sigma(mass):.5f}",  "")
    
    
    frame.Draw()
    legend.Draw() 
    canvas.SetTicks(2, 2)
    canvas.Update()
    canvas.Draw()
    
    canvas.SaveAs(f"gghMC_RERECO_powgheg_vs_UL_powheg.pdf")







