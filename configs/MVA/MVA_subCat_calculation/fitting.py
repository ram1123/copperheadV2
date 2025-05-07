import dask_awkward as dak
import awkward as ak
from distributed import LocalCluster, Client, progress
import time
import numpy as np
import matplotlib.pyplot as plt
import json
import mplhep as hep
import glob
from typing import Tuple, List, Dict
import ROOT as rt
import os

import pandas as pd
import subprocess
import re

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
    
def plotBkgByCoreFunc(mass:rt.RooRealVar, model_dict_by_coreFunction: Dict, rooHist_list, save_path: str):
    """
    takes the dictionary of all Bkg RooAbsPdf models grouped by same corefunctions, and plot them
    in the frame() of mass and saves the plots on a given directory path
    """
    # make the save_path directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    color_list = [
        rt.kGreen,
        rt.kBlue,
        rt.kRed,
        rt.kOrange,
        rt.kViolet,
    ]
    for core_type, coreFunction_list in model_dict_by_coreFunction.items():
        name = "Canvas"
        canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
        canvas.cd()
        frame = mass.frame()
        frame.SetTitle(f"Normalized Shape Plot of {core_type} PDFs")
        frame.SetXTitle(f"Dimuon Mass (GeV)")
        legend = rt.TLegend(0.65,0.55,0.9,0.7)
        
        # print(f"normalized_hist integral: {normalized_hist.sum(False)}")
        for ix in range(len(coreFunction_list)):
        # for ix in [0,4]:
            # apparently I have to plot invisible roo dataset for fit function plotting to work. Maybe this helps with normalization?
            color = color_list[ix]
            hist = rooHist_list[ix]
            normalized_hist = normalizeRooHist(mass, hist)
            normalized_hist.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0), Invisible=True )
            # normalized_hist.plotOn(frame, LineColor=color,MarkerColor=color)
            model = coreFunction_list[ix]
            name = model.GetName()
            print(f"index {ix} with name: {name}")
            # model.Print("v")
            model.plotOn(frame, Name=name, LineColor=color)
            legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
        frame.SetMaximum(0.0042)
        frame.Draw()
        legend.Draw()       
        canvas.SetTicks(2, 2)
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs(f"{save_path}/BkgShapePlot_{core_type}.pdf")

def plotSigBySample(mass:rt.RooRealVar, model_dict_by_sample: Dict, sigHist_list: List, save_path: str):
    """
    takes the dictionary of all Signal RooAbsPdf models grouped by same sample, and plot them
    in the frame() of mass and saves the plots on a given directory path
    """
    # make the save_path directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    color_list = [
        rt.kGreen,
        rt.kBlue,
        rt.kRed,
        rt.kOrange,
        rt.kViolet,
    ]
    for model_type, model_list in model_dict_by_sample.items():
        name = "Canvas"
        canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
        canvas.cd()
        frame = mass.frame()
        frame.SetMaximum(0.017)
        frame.SetMinimum(-0.002)
        frame.SetTitle(f"Normalized Shape Plot of {model_type} PDFs")
        frame.SetXTitle(f"Dimuon Mass (GeV)")
        legend = rt.TLegend(0.65,0.55,0.9,0.7)
        # apparently I have to plot invisible roo dataset for fit function plotting to work. Maybe this helps with normalization?
        
        # print(f"normalized_hist integral: {normalized_hist.sum(False)}")
        for ix in range(len(model_list)):
            sig_hist = sigHist_list[ix]
            normalized_hist = normalizeRooHist(mass, sig_hist)
            normalized_hist.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0), Invisible=True  )
            model = model_list[ix]
            name = model.GetName()
            color = color_list[ix]
            model.plotOn(frame, Name=name, LineColor=color)
            legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
        frame.Draw()
        legend.Draw()        
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs(f"{save_path}/ShapePlot_{model_type}.pdf")

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
    # nan_filter = np.isnan(region)
    # for field in events.fields:
    #     nan_filter = nan_filter | np.isnan(events[field])
    # nan_filter = ~nan_filter
    nan_filter = ~np.isnan(ak.to_numpy(events.wgt_nominal_total))
    none_filter = ~ak.is_none((events.wgt_nominal_total))
    # nan_filter = ~(np.isnan(events.wgt_nominal_total) | np.isnan(events.dimuon_mass)) # some nans are not None, apparently
    # nan_filter = ~np.isnan(events.wgt_nominal_total) 
    total_filter = (
        region 
        & nan_filter
        & none_filter
    )
    return events[total_filter]


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
    print(f"ak.min(subCat_idx): {ak.min(subCat_idx)}")
    print(f"ak.max(subCat_idx): {ak.max(subCat_idx)}")
    processed_events["subCategory_idx"] = subCat_idx
    return processed_events

def separateNfit(load_paths: Dict, score_edge_dict: Dict, plot_save_path:str):
    """
    keys of both load_paths and score_edge_dict refer to the era values (ie 2016preVFP, 2016postVFP, 2017 and 2018)
    and keys on load_paths and score_edge_dict are intended to be identical
    """
    bkg_event_l = []
    sig_event_l = []
    for era, load_path in load_paths.items():
        score_edges = score_edge_dict[era]
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

        events_bkg = dak.from_parquet(f"{load_path}/processed_events_bkgMC*.parquet") # all the bkg MCs
        events_bkg = ak.zip({field: events_bkg[field] for field in fields2compute}).compute()

        # apply ggH cat cut, calculate sub Cat
        # events_sig = applyGGH_cut(events_sig) # we assume ggH category cut is already done via run_stage2
        # events_bkg = applyGGH_cut(events_bkg) # we assume ggH category cut is already done via run_stage2
        events_sig = applySigReg_cut(events_sig)
        events_bkg = applySigReg_cut(events_bkg)
        

        # calculate the subcategory
        events_sig = calculateSubCat(events_sig, score_edges)
        events_bkg = calculateSubCat(events_bkg, score_edges)

        # add to the list to concatenate later
        sig_event_l.append(events_sig)
        bkg_event_l.append(events_bkg)
        
        
    # concantenate the events and then seperate them
    print(f"bkg_event_l: {bkg_event_l}")
    print(f"sig_event_l: {sig_event_l}")
    processed_eventsBackgroundMC = ak.concatenate(bkg_event_l)
    processed_eventsSignalMC = ak.concatenate(sig_event_l)
    # print(f"processed_eventsBackgroundMC: {processed_eventsBackgroundMC}")
    # print(f"processed_eventsSignalMC: {processed_eventsSignalMC}")


    # ---------------------------------------------------------------------------------------------
    # start the fitting process
    # ---------------------------------------------------------------------------------------------
    device = "cpu"
    # Create observables
    mass_name = "mh_ggh"
    mass = rt.RooRealVar(mass_name, mass_name, 120, 110, 150)
    nbins = 800
    mass.setBins(nbins)
    # mass.setRange("hiSB", 135, 150 )
    # mass.setRange("loSB", 110, 115 )
    # mass.setRange("h_peak", 115, 135 )
    # mass.setRange("full", 110, 150 )
    # fit_range = "hiSB,loSB" # we're fitting bkg only
    
    subCatIdx_name = "subCategory_idx"

    # ---------------------------------------------------------------------------------------------
    # start background channel fitting
    # ---------------------------------------------------------------------------------------------
    # subCat 0
    name = f"BWZ_Redux_a_coeff_subCat0"
    a_coeff_subCat0 = rt.RooRealVar(name,name, 0.06231018619106862,-0.1,0.1)
    name = f"BWZ_Redux_b_coeff_subCat0"
    b_coeff_subCat0 = rt.RooRealVar(name,name, -0.0001684318108879923,-0.1,0.1)
    name = f"BWZ_Redux_c_coeff_subCat0"
    c_coeff_subCat0 = rt.RooRealVar(name,name, 2.14876669663328,0,5.0)

    name = "subCat0_BWZ_Redux"
    BWZRedux_SubCat0 = rt.RooModZPdf(name, name, mass, a_coeff_subCat0, b_coeff_subCat0, c_coeff_subCat0) 

    # subCat 1
    name = f"BWZ_Redux_a_coeff_subCat1"
    a_coeff_subCat1 = rt.RooRealVar(name,name, 0.06231018619106862,-0.1,0.1)
    name = f"BWZ_Redux_b_coeff_subCat1"
    b_coeff_subCat1 = rt.RooRealVar(name,name, -0.0001684318108879923,-0.1,0.1)
    name = f"BWZ_Redux_c_coeff_subCat1"
    c_coeff_subCat1 = rt.RooRealVar(name,name, 2.14876669663328,0,5.0)

    name = "subCat1_BWZ_Redux"
    BWZRedux_SubCat1 = rt.RooModZPdf(name, name, mass, a_coeff_subCat1, b_coeff_subCat1, c_coeff_subCat1) 

    # initialze Bkg samples to fit to
    # subCat 0
    subCat_filter = (processed_eventsBackgroundMC[subCatIdx_name] == 0)
    subCat_mass_arr = processed_eventsBackgroundMC.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat0 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    data_subCat0_BWZRedux = rt.RooDataHist("subCat0_rooHist_BWZRedux","subCat0_rooHist_BWZRedux", rt.RooArgSet(mass), roo_datasetData_subCat0)

    norm_val = data_subCat0_BWZRedux.sumEntries()
    # data_subCat0_BWZRedux_norm = rt.RooRealVar(data_subCat0_BWZRedux.GetName()+"_norm","subCat0 norm",norm_val)
    # print(f"signal_subCat4 norm_val: {norm_val}")
    # data_subCat0_BWZRedux_norm.setConstant(True)

    # subCat 1
    subCat_filter = (processed_eventsBackgroundMC[subCatIdx_name] == 1)
    subCat_mass_arr = processed_eventsBackgroundMC.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat1 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    data_subCat1_BWZRedux = rt.RooDataHist("subCat1_rooHist_BWZRedux","subCat1_rooHist_BWZRedux", rt.RooArgSet(mass), roo_datasetData_subCat1)
    norm_val = data_subCat1_BWZRedux.sumEntries()
    # data_subCat1_BWZRedux_norm = rt.RooRealVar(data_subCat1_BWZRedux.GetName()+"_norm","subCat1 norm",norm_val)
    # print(f"signal_subCat4 norm_val: {norm_val}")
    # data_subCat1_BWZRedux_norm.setConstant(True)

    # apply bkg fit
    # subCat 0
    _ = BWZRedux_SubCat0.fitTo(data_subCat0_BWZRedux,  EvalBackend=device, Save=True, )
    fit_result = BWZRedux_SubCat0.fitTo(data_subCat0_BWZRedux,  EvalBackend=device, Save=True, )
    # subCat 1
    _ = BWZRedux_SubCat1.fitTo(data_subCat1_BWZRedux,  EvalBackend=device, Save=True, )
    fit_result = BWZRedux_SubCat1.fitTo(data_subCat1_BWZRedux,  EvalBackend=device, Save=True, )
    

    
    # ---------------------------------------------------------------------------------------------
    # start signal channel fitting
    # ---------------------------------------------------------------------------------------------
    
    # subCat 0
    # original start ------------------------------------------------------
    # MH_subCat0 = rt.RooRealVar("MH" , "MH", 125, 115,135)
    # MH_subCat0.setConstant(True) # this shouldn't change, I think
    # original end ------------------------------------------------------
    MH_subCat0 = rt.RooRealVar("MH" , "MH", 124.805, 120,130) # matching AN
    
    # sigma_subCat0 = rt.RooRealVar("sigma_subCat0" , "sigma_subCat0", 2, .1, 4.0)
    # alpha1_subCat0 = rt.RooRealVar("alpha1_subCat0" , "alpha1_subCat0", 2, 0.01, 65)
    # n1_subCat0 = rt.RooRealVar("n1_subCat0" , "n1_subCat0", 10, 0.01, 100)
    # alpha2_subCat0 = rt.RooRealVar("alpha2_subCat0" , "alpha2_subCat0", 2.0, 0.01, 65)
    # n2_subCat0 = rt.RooRealVar("n2_subCat0" , "n2_subCat0", 25, 0.01, 100)

    # copying parameters from official AN workspace as starting params
    sigma_subCat0 = rt.RooRealVar("sigma_subCat0" , "sigma_subCat0", 1.8228, .1, 4.0)
    alpha1_subCat0 = rt.RooRealVar("alpha1_subCat0" , "alpha1_subCat0", 1.12842, 0.01, 65)
    n1_subCat0 = rt.RooRealVar("n1_subCat0" , "n1_subCat0", 4.019960, 0.01, 100)
    alpha2_subCat0 = rt.RooRealVar("alpha2_subCat0" , "alpha2_subCat0", 1.3132, 0.01, 65)
    n2_subCat0 = rt.RooRealVar("n2_subCat0" , "n2_subCat0", 9.97411, 0.01, 100)

    # # temporary test
    # sigma_subCat0.setConstant(True)
    # alpha1_subCat0.setConstant(True)
    # n1_subCat0.setConstant(True)
    # alpha2_subCat0.setConstant(True)
    # n2_subCat0.setConstant(True)
    
    
    CMS_hmm_sigma_cat0_ggh = rt.RooRealVar("CMS_hmm_sigma_cat0_ggh" , "CMS_hmm_sigma_cat0_ggh", 0, -5 , 5 )
    CMS_hmm_sigma_cat0_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat0_ggh_fsigma = rt.RooFormulaVar("ggH_cat0_ggh_fsigma", "ggH_cat0_ggh_fsigma",'@0*(1+@1)',[sigma_subCat0, CMS_hmm_sigma_cat0_ggh])
    CMS_hmm_peak_cat0_ggh = rt.RooRealVar("CMS_hmm_peak_cat0_ggh" , "CMS_hmm_peak_cat0_ggh", 0, -5 , 5 )
    CMS_hmm_peak_cat0_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat0_ggh_fpeak = rt.RooFormulaVar("ggH_cat0_ggh_fpeak", "ggH_cat0_ggh_fpeak",'@0*(1+@1)',[MH_subCat0, CMS_hmm_peak_cat0_ggh])
    
    # n1_subCat0.setConstant(True) # freeze for stability
    # n2_subCat0.setConstant(True) # freeze for stability
    name = "signal_subCat0"
    signal_subCat0 = rt.RooCrystalBall(name,name,mass, ggH_cat0_ggh_fpeak, ggH_cat0_ggh_fsigma, alpha1_subCat0, n1_subCat0, alpha2_subCat0, n2_subCat0)

    # subCat 1
    # original start ------------------------------------------------------
    # MH_subCat1 = rt.RooRealVar("MH" , "MH", 125, 115,135)
    # MH_subCat1.setConstant(True) # this shouldn't change, I think
    # original end ------------------------------------------------------
    MH_subCat1 = rt.RooRealVar("MH" , "MH", 124.853, 120,130) # matching AN
    
    # sigma_subCat1 = rt.RooRealVar("sigma_subCat1" , "sigma_subCat1", 2, .1, 4.0)
    # alpha1_subCat1 = rt.RooRealVar("alpha1_subCat1" , "alpha1_subCat1", 2, 0.01, 65)
    # n1_subCat1 = rt.RooRealVar("n1_subCat1" , "n1_subCat1", 10, 0.01, 100)
    # alpha2_subCat1 = rt.RooRealVar("alpha2_subCat1" , "alpha2_subCat1", 2.0, 0.01, 65)
    # n2_subCat1 = rt.RooRealVar("n2_subCat1" , "n2_subCat1", 25, 0.01, 100)

    # copying parameters from official AN workspace as starting params
    sigma_subCat1 = rt.RooRealVar("sigma_subCat1" , "sigma_subCat1", 1.503280, .1, 4.0)
    alpha1_subCat1 = rt.RooRealVar("alpha1_subCat1" , "alpha1_subCat1", 1.3364, 0.01, 65)
    n1_subCat1 = rt.RooRealVar("n1_subCat1" , "n1_subCat1", 2.815022, 0.01, 100)
    alpha2_subCat1 = rt.RooRealVar("alpha2_subCat1" , "alpha2_subCat1", 1.57127749, 0.01, 65)
    n2_subCat1 = rt.RooRealVar("n2_subCat1" , "n2_subCat1", 9.99687, 0.01, 100)

    # # temporary test
    # sigma_subCat1.setConstant(True)
    # alpha1_subCat1.setConstant(True)
    # n1_subCat1.setConstant(True)
    # alpha2_subCat1.setConstant(True)
    # n2_subCat1.setConstant(True)
    
    CMS_hmm_sigma_cat1_ggh = rt.RooRealVar("CMS_hmm_sigma_cat1_ggh" , "CMS_hmm_sigma_cat1_ggh", 0, -5 , 5 )
    CMS_hmm_sigma_cat1_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat1_ggh_fsigma = rt.RooFormulaVar("ggH_cat1_ggh_fsigma", "ggH_cat1_ggh_fsigma",'@0*(1+@1)',[sigma_subCat1, CMS_hmm_sigma_cat1_ggh])
    CMS_hmm_peak_cat1_ggh = rt.RooRealVar("CMS_hmm_peak_cat1_ggh" , "CMS_hmm_peak_cat1_ggh", 0, -5 , 5 )
    CMS_hmm_peak_cat1_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat1_ggh_fpeak = rt.RooFormulaVar("ggH_cat1_ggh_fpeak", "ggH_cat1_ggh_fpeak",'@0*(1+@1)',[MH_subCat1, CMS_hmm_peak_cat1_ggh])
    
    # n1_subCat1.setConstant(True) # freeze for stability
    # n2_subCat1.setConstant(True) # freeze for stability
    name = "signal_subCat1"
    signal_subCat1 = rt.RooCrystalBall(name,name,mass, ggH_cat1_ggh_fpeak, ggH_cat1_ggh_fsigma, alpha1_subCat1, n1_subCat1, alpha2_subCat1, n2_subCat1)


    # ---------------------------------------------------
    # Define signal MC samples to fit to
    # ---------------------------------------------------

    # subCat 0
    subCat_filter = (processed_eventsSignalMC[subCatIdx_name] == 0)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat0_SigMC = ak.to_numpy(
        processed_eventsSignalMC.wgt_nominal_total[subCat_filter]
    ) # weights

    # generate a weighted histogram 
    roo_histData_subCat0_signal = rt.TH1F("subCat0_rooHist_signal", "subCat0_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat0_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat0_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat0_signal = rt.RooDataHist("subCat0_rooHist_signal", "subCat0_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat0_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat0_signal = roo_histData_subCat0_signal

    # define normalization value from signal MC event weights 
    flat_MC_SF = 1.00
    # flat_MC_SF = 0.92 # temporary flat SF to match my Data/MC agreement to that of AN's
    norm_val = np.sum(wgt_subCat0_SigMC)* flat_MC_SF 
    # norm_val = 254.528077 # quick test
    sig_norm_subCat0 = rt.RooRealVar(signal_subCat0.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat0 norm_val: {norm_val}")
    sig_norm_subCat0.setConstant(True)

    # subCat 1
    subCat_filter = (processed_eventsSignalMC[subCatIdx_name] == 1)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat1_SigMC = ak.to_numpy(
        processed_eventsSignalMC.wgt_nominal_total[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat1_signal = rt.TH1F("subCat1_rooHist_signal", "subCat1_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat1_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat1_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat1_signal = rt.RooDataHist("subCat1_rooHist_signal", "subCat1_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat1_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat1_signal = roo_histData_subCat1_signal

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat1_SigMC)* flat_MC_SF
    # norm_val = 295.214 # quick test
    sig_norm_subCat1 = rt.RooRealVar(signal_subCat1.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat1 norm_val: {norm_val}")
    sig_norm_subCat1.setConstant(True)

    # ---------------------------------------------------
    # Fit signal model individually, not simultaneous. Sigma, and left and right tails are different for each category
    # ---------------------------------------------------

    # subCat 0
    _ = signal_subCat0.fitTo(data_subCat0_signal,  EvalBackend=device, Save=True, )
    fit_result = signal_subCat0.fitTo(data_subCat0_signal,  EvalBackend=device, Save=True, )
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat0.setConstant(True)
    alpha1_subCat0.setConstant(True)
    n1_subCat0.setConstant(True)
    alpha2_subCat0.setConstant(True)
    n2_subCat0.setConstant(True)

    # unfreeze the param for datacard
    CMS_hmm_sigma_cat0_ggh.setConstant(False)
    CMS_hmm_peak_cat0_ggh.setConstant(False)

    # subCat 1
    _ = signal_subCat1.fitTo(data_subCat1_signal,  EvalBackend=device, Save=True, )
    fit_result = signal_subCat1.fitTo(data_subCat1_signal,  EvalBackend=device, Save=True, )
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat1.setConstant(True)
    alpha1_subCat1.setConstant(True)
    n1_subCat1.setConstant(True)
    alpha2_subCat1.setConstant(True)
    n2_subCat1.setConstant(True)

    # unfreeze the param for datacard
    CMS_hmm_sigma_cat1_ggh.setConstant(False)
    CMS_hmm_peak_cat1_ggh.setConstant(False)


    # -------------------------------------------------------------------------
    # do Plotting
    # -------------------------------------------------------------------------
    # plot_save_path = f"./quick_plots/"
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
        
    # -------------------------------------------------------------------------
    # do background plotting with fit and data
    # -------------------------------------------------------------------------
    
    # subCat 0
    
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat0_BWZRedux.GetName()
    data_subCat0_BWZRedux.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = BWZRedux_SubCat0.GetName()
    BWZRedux_SubCat0.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/bkgMC_plot_subCat0.pdf")

    # subCat 1
    
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat1_BWZRedux.GetName()
    data_subCat1_BWZRedux.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = BWZRedux_SubCat1.GetName()
    BWZRedux_SubCat1.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/bkgMC_plot_subCat1.pdf")
    
    
    # -------------------------------------------------------------------------
    # do signal plotting with fit and data
    # -------------------------------------------------------------------------

    # subCat 0
    
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat0_signal.GetName()
    data_subCat0_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat0.GetName()
    signal_subCat0.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/sigMC_plot_subCat0.pdf")

    # subCat 1
    
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat1_signal.GetName()
    data_subCat1_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat1.GetName()
    signal_subCat1.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/sigMC_plot_subCat1.pdf")

    # -------------------------------------------------------------------------
    # do Bkg plotting loop divided into core-function
    # -------------------------------------------------------------------------
    
    model_dict_by_coreFunction = {
        "BWZRedux" : [
            BWZRedux_SubCat0, 
            BWZRedux_SubCat1,
        ],
    }
    bkgHist_list = [ # for normalization histogram reference
        data_subCat0_BWZRedux,
        data_subCat1_BWZRedux,
    ]
    plotBkgByCoreFunc(mass, model_dict_by_coreFunction, bkgHist_list, plot_save_path)
    # -------------------------------------------------------------------------
    # do signal plotting for all sub-Cats in one plot
    # -------------------------------------------------------------------------
    sig_dict_by_sample = {
        "signal" : [
            signal_subCat0, 
            signal_subCat1,
        ]
    }
    sigHist_list = [ # for signal function normalization
        data_subCat0_signal,
        data_subCat1_signal,
    ]
    plotSigBySample(mass, sig_dict_by_sample, sigHist_list, plot_save_path)
    
    # ---------------------------------------------------
    # Save to Signal, Background and Data to Workspace
    # ---------------------------------------------------
    workspace_path = "./workspace"
    if not os.path.exists(workspace_path):
        os.makedirs(workspace_path)
    category = "ggh"
    # subCat 0 
    # background first
    fout = rt.TFile(f"{workspace_path}/workspace_bkg_cat0_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    data_subCat0_BWZRedux.SetName("data_cat0_ggh");
    BWZRedux_SubCat0.SetName("bkg_cat0_ggh_pdf");
    
    # make norm for data and mc pdf
    nevents = data_subCat0_BWZRedux.sumEntries()
    data_subCat0_BWZRedux_norm = rt.RooRealVar(data_subCat0_BWZRedux.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    BWZRedux_SubCat0_norm = rt.RooRealVar(BWZRedux_SubCat0.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    BWZRedux_SubCat0_norm.setConstant(True); 
    wout.Import(data_subCat0_BWZRedux_norm);
    wout.Import(data_subCat0_BWZRedux);
    wout.Import(BWZRedux_SubCat0_norm);
    wout.Import(BWZRedux_SubCat0);
    # wout.Print();
    wout.Write();

    # now signal
    fout = rt.TFile(f"{workspace_path}/workspace_sig_cat0_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    signal_subCat0.SetName("ggH_cat0_ggh_pdf");
    data_subCat0_signal.SetName("data_ggH_cat0_ggh");
    sig_norm_subCat0.SetName(signal_subCat0.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat0);
    wout.Import(signal_subCat0); 
    wout.Import(data_subCat0_signal); 
    # wout.Print();
    wout.Write();

    # subCat 1 
    # background first
    fout = rt.TFile(f"{workspace_path}/workspace_bkg_cat1_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    data_subCat1_BWZRedux.SetName("data_cat1_ggh");
    BWZRedux_SubCat1.SetName("bkg_cat1_ggh_pdf");
    
    # make norm for data and mc pdf
    nevents = data_subCat1_BWZRedux.sumEntries()
    data_subCat1_BWZRedux_norm = rt.RooRealVar(data_subCat1_BWZRedux.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    BWZRedux_SubCat1_norm = rt.RooRealVar(BWZRedux_SubCat1.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    BWZRedux_SubCat1_norm.setConstant(True); 
    wout.Import(data_subCat1_BWZRedux_norm);
    wout.Import(data_subCat1_BWZRedux);
    wout.Import(BWZRedux_SubCat1_norm);
    wout.Import(BWZRedux_SubCat1);
    # wout.Print();
    wout.Write();

    # now signal
    fout = rt.TFile(f"{workspace_path}/workspace_sig_cat1_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    signal_subCat1.SetName("ggH_cat1_ggh_pdf");
    data_subCat1_signal.SetName("data_ggH_cat1_ggh");
    sig_norm_subCat1.SetName(signal_subCat1.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat1);
    wout.Import(signal_subCat1); 
    wout.Import(data_subCat1_signal); 
    # wout.Print();
    wout.Write();

if __name__ == "__main__":
    years = [
        "2016preVFP",
        "2016postVFP",
        "2017",
        "2018",
    ]
    stage2_out_name = "BDT_WgtON_original_AN_BDT_Sept27"
    load_paths = {}
    BDT_thresholds = {}
    # sig_eff = 0.45
    sig_effs = np.arange(0.99, 0.0, step=-0.01)
    # sig_effs = [0.3,0.45,0.5]
    exp_signficances = {}
    for sig_eff in sig_effs:
        for year in years:
            load_path = f"/work/users/yun79/stage2_output/{stage2_out_name}/ggh/{year}"
            load_paths[year] = load_path
            BDT_df = pd.read_csv(f"BDT_threshold_{year}.csv")
            bool_filter = np.isclose(BDT_df["sig_eff"], sig_eff)
            threshold = BDT_df["BDT_threshold"][bool_filter].values
            print(f"{year} threshold: {threshold}")
            BDT_thresholds[year] = [0, threshold, 1.0]

        sig_eff_str = str(round(sig_eff,2)).replace(".", "_")
        plot_save_path = f"quick_plots/iter0/{sig_eff_str}"
        print(f"separateNfit plot_save_path: {plot_save_path}")
        separateNfit(load_paths, BDT_thresholds, plot_save_path)
        
        # combine the datacards
        with open("datacard_comb_sig_all_ggh_test.txt", "w") as text_file:
             subprocess.call(["combineCards.py", "datacard_cat0_ggh.txt", "datacard_cat1_ggh.txt"], stdout=text_file)
        
        # text2workspace
        subprocess.run(["text2workspace.py", "-m", "125", "datacard_comb_sig_all_ggh_test.txt"])
        # use combine to calculate significance
        out_text = subprocess.run([
            "combine",
            "-M",
            "Significance",
            "-d",
            "datacard_comb_sig_all_ggh_test.root",
            "-m",
            "125",
            "-n",
            "_signif_all_ggh",
            "--cminDefaultMinimizerStrategy=1",
            "-t" ,
            "-1",
            "--toysFrequentist",
            "--expectSignal",
            "1",
            "--X-rtd",
            "FITTER_NEWER_GIVE_UP",
            "--X-rtd",
            "FITTER_BOUND",
            "--setParameters",
            "pdf_index_ggh=0",
            "--cminRunAllDiscreteCombinations",
            "--setParameterRanges",
            "r=-10,10",
            "--X-rtd",
            "MINIMIZER_freezeDisassociatedParams",
            "--cminDefaultMinimizerTolerance",
            "0.01",
            "--X-rtd",
            "MINIMIZER_MaxCalls=9999999",
            "--X-rtd",
            "FAST_VERTICAL_MORPH"
            ],
            capture_output=True, text=True           
        )
        out_text = str(out_text)
        match  = re.search(r'Significance:\s*([0-9.]+)', out_text)
        if match:
            significance_value = float(match.group(1))
            exp_signficances[sig_eff] = significance_value
            print(f"for Sig eff: {sig_eff}, we get exp significance of :{significance_value}")
        else:
            exp_signficances[sig_eff] = None
            print(f"no re match for Sig eff: {sig_eff} from outtext {out_text}")
        
        out_data = {
            "sig_eff" : exp_signficances.keys(),
            "exp_signifiance" : exp_signficances.values(),
        }
        out_df = pd.DataFrame(out_data)
        out_df.to_csv("iter0_expSignificances.csv")
        

