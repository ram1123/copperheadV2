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
    
    subCatIdx_name = "subCategory_idx"

    # -------------------------------------------------------------------------
    # make directory for plotting later
    # -------------------------------------------------------------------------
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    # -------------------------------------------------------------------------
    # make directory for saving workspaces
    # -------------------------------------------------------------------------
    workspace_path = "./workspace"
    if not os.path.exists(workspace_path):
        os.makedirs(workspace_path)
    
    # ---------------------------------------------------------------------------------------------
    # start background channel fitting
    # ---------------------------------------------------------------------------------------------
    subCats = [0,1]
    bkg_fit_pdfs = []
    bkg_roohists = []
    # initialize lists so that python garbage collector doesn't delete them
    a_coeffs = []
    b_coeffs = []
    c_coeffs = []
    for subCat_idx in subCats:
        name = f"BWZ_Redux_a_coeff_subCat{subCat_idx}"
        a_coeff = rt.RooRealVar(name,name, 0.06231018619106862,-0.1,0.1)
        a_coeffs.append(a_coeff)
        name = f"BWZ_Redux_b_coeff_subCat{subCat_idx}"
        b_coeff = rt.RooRealVar(name,name, -0.0001684318108879923,-0.1,0.1)
        b_coeffs.append(b_coeff)
        name = f"BWZ_Redux_c_coeff_subCat{subCat_idx}"
        c_coeff = rt.RooRealVar(name,name, 2.14876669663328,0,5.0)
        c_coeffs.append(c_coeff)
    
        name = f"subCat{subCat_idx}_BWZ_Redux"
        BWZRedux_SubCat = rt.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff)
        

        # initialze Bkg samples to fit to
        subCat_filter = (processed_eventsBackgroundMC[subCatIdx_name] == subCat_idx)
        subCat_mass_arr = processed_eventsBackgroundMC.dimuon_mass[subCat_filter]
        subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
        roo_datasetData_subCat = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
        name = f"subCat{subCat_idx}_rooHist_BWZRedux"
        data_subCat_BWZRedux = rt.RooDataHist(name,name, rt.RooArgSet(mass), roo_datasetData_subCat)
    
   
        # apply bkg fit
        # subCat 0
        _ = BWZRedux_SubCat.fitTo(data_subCat_BWZRedux,  EvalBackend=device, Save=True, )
        fit_result = BWZRedux_SubCat.fitTo(data_subCat_BWZRedux,  EvalBackend=device, Save=True, )

        bkg_fit_pdfs.append(BWZRedux_SubCat)
        bkg_roohists.append(data_subCat_BWZRedux)

        # -------------------------------------------------------------------------
        # save fit data into workspace
        # -------------------------------------------------------------------------
        category = "ggh"
        # subCat 0 
        # background first
        fout = rt.TFile(f"{workspace_path}/workspace_bkg_cat{subCat_idx}_{category}.root","RECREATE")
        wout = rt.RooWorkspace("w","workspace")
        # matching names consistent with UCSD's naming scheme
        data_subCat_BWZRedux.SetName(f"data_cat{subCat_idx}_ggh");
        BWZRedux_SubCat.SetName(f"bkg_cat{subCat_idx}_ggh_pdf");
        
        # make norm for data and mc pdf
        nevents = data_subCat_BWZRedux.sumEntries()
        data_subCat_BWZRedux_norm = rt.RooRealVar(data_subCat_BWZRedux.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
        BWZRedux_SubCat_norm = rt.RooRealVar(BWZRedux_SubCat.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
        BWZRedux_SubCat_norm.setConstant(True); 
        wout.Import(data_subCat_BWZRedux_norm);
        wout.Import(data_subCat_BWZRedux);
        wout.Import(BWZRedux_SubCat_norm);
        wout.Import(BWZRedux_SubCat);
        # wout.Print();
        wout.Write();

        # -------------------------------------------------------------------------
        # do background plotting with fit and data
        # -------------------------------------------------------------------------
               
        name = "Canvas"
        canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
        canvas.cd()
        frame = mass.frame()
        legend = rt.TLegend(0.65,0.55,0.9,0.7)
        name = data_subCat_BWZRedux.GetName()
        data_subCat_BWZRedux.plotOn(frame, DataError="SumW2", Name=name)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
        name = BWZRedux_SubCat.GetName()
        BWZRedux_SubCat.plotOn(frame, Name=name, LineColor=rt.kGreen)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
        
        frame.Draw()
        legend.Draw()
        
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs(f"{plot_save_path}/bkgMC_plot_subCat{subCat_idx}.pdf")

    # -------------------------------------------------------------------------
    # do Bkg plotting loop divided into core-function
    # -------------------------------------------------------------------------
    
    model_dict_by_coreFunction = {
        "BWZRedux" : bkg_fit_pdfs
    }
    plotBkgByCoreFunc(mass, model_dict_by_coreFunction, bkg_roohists, plot_save_path)
    
    # ---------------------------------------------------------------------------------------------
    # start signal channel fitting
    # ---------------------------------------------------------------------------------------------
    sig_fit_pdfs = []
    sig_roohists = []
    # initialize lists so that python garbage collector doesn't delete them
    sigma_l = []
    alpha1_l = []
    n1_l = []
    alpha2_l = []
    n2_l = []
    MH_l = []
    CMS_hmm_sigma_cat_ggh_l = []
    CMS_hmm_peak_cat_ggh_l = []
    ggH_cat_ggh_fsigma_l = []
    ggH_cat_ggh_fpeak_l = []
    for subCat_idx in subCats:
        # subCat 0
        MH_subCat = rt.RooRealVar("MH" , "MH", 124.805, 120,130) # matching AN
        MH_l.append(MH_subCat)
        
        # copying parameters from official AN workspace as starting params
        sigma_subCat = rt.RooRealVar(f"sigma_subCat{subCat_idx}" , f"sigma_subCat{subCat_idx}", 1.8228, .1, 4.0)
        sigma_l.append(sigma_subCat)
        alpha1_subCat = rt.RooRealVar(f"alpha1_subCat{subCat_idx}" , f"alpha1_subCat{subCat_idx}", 1.12842, 0.01, 65)
        alpha1_l.append(alpha1_subCat)
        n1_subCat = rt.RooRealVar(f"n1_subCat{subCat_idx}" , f"n1_subCat{subCat_idx}", 4.019960, 0.01, 100)
        n1_l.append(n1_subCat)
        alpha2_subCat = rt.RooRealVar(f"alpha2_subCat{subCat_idx}" , f"alpha2_subCat{subCat_idx}", 1.3132, 0.01, 65)
        alpha2_l.append(alpha2_subCat)
        n2_subCat = rt.RooRealVar(f"n2_subCat{subCat_idx}" , f"n2_subCat{subCat_idx}", 9.97411, 0.01, 100)
        n2_l.append(n2_subCat)
    
    
    
        CMS_hmm_sigma_cat_ggh = rt.RooRealVar(f"CMS_hmm_sigma_cat{subCat_idx}_ggh" , f"CMS_hmm_sigma_cat{subCat_idx}_ggh", 0, -5 , 5 )
        CMS_hmm_sigma_cat_ggh.setConstant(True) # this is going to be param in datacard
        CMS_hmm_sigma_cat_ggh_l.append(CMS_hmm_sigma_cat_ggh)
        ggH_cat_ggh_fsigma = rt.RooFormulaVar(f"ggH_cat{subCat_idx}_ggh_fsigma", f"ggH_cat{subCat_idx}_ggh_fsigma",'@0*(1+@1)',[sigma_subCat, CMS_hmm_sigma_cat_ggh])
        ggH_cat_ggh_fsigma_l.append(ggH_cat_ggh_fsigma)
        CMS_hmm_peak_cat_ggh = rt.RooRealVar(f"CMS_hmm_peak_cat{subCat_idx}_ggh" , f"CMS_hmm_peak_cat{subCat_idx}_ggh", 0, -5 , 5 )
        CMS_hmm_peak_cat_ggh.setConstant(True) # this is going to be param in datacard
        CMS_hmm_peak_cat_ggh_l.append(CMS_hmm_peak_cat_ggh)
        ggH_cat_ggh_fpeak = rt.RooFormulaVar(f"ggH_cat{subCat_idx}_ggh_fpeak", f"ggH_cat{subCat_idx}_ggh_fpeak",'@0*(1+@1)',[MH_subCat, CMS_hmm_peak_cat_ggh])
        ggH_cat_ggh_fpeak_l.append(ggH_cat_ggh_fpeak)
    
    
        name = f"signal_subCat{subCat_idx}"
        signal_subCat = rt.RooCrystalBall(name,name,mass, ggH_cat_ggh_fpeak, ggH_cat_ggh_fsigma, alpha1_subCat, n1_subCat, alpha2_subCat, n2_subCat)


        # ---------------------------------------------------
        # Define signal MC samples to fit to
        # ---------------------------------------------------
    
        # subCat 0
        subCat_filter = (processed_eventsSignalMC[subCatIdx_name] == subCat_idx)
        subCat_mass_arr = ak.to_numpy(
            processed_eventsSignalMC.dimuon_mass[subCat_filter]
        ) # mass values
        wgt_subCat_SigMC = ak.to_numpy(
            processed_eventsSignalMC.wgt_nominal_total[subCat_filter]
        ) # weights
    
        # generate a weighted histogram 
        roo_histData_subCat_signal = rt.TH1F(f"subCat{subCat_idx}_rooHist_signal", f"subCat{subCat_idx}_rooHist_signal", nbins, mass.getMin(), mass.getMax())
           
        roo_histData_subCat_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat_SigMC) # fill the histograms with mass and weights 
        roo_histData_subCat_signal = rt.RooDataHist(f"subCat{subCat_idx}_rooHist_signal", f"subCat{subCat_idx}_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat_signal) # convert to RooDataHist with (picked same name, bc idk)
        
        data_subCat_signal = roo_histData_subCat_signal
    
        # define normalization value from signal MC event weights 
        norm_val = np.sum(wgt_subCat_SigMC)
        # norm_val = 254.528077 # quick test
        sig_norm_subCat = rt.RooRealVar(signal_subCat.GetName()+"_norm","Number of signal events",norm_val)
        print(f"signal_subCat{subCat_idx} norm_val: {norm_val}")
        sig_norm_subCat.setConstant(True)


        # ---------------------------------------------------
        # Fit signal model individually, not simultaneous. Sigma, and left and right tails are different for each category
        # ---------------------------------------------------
    
        # subCat 0
        _ = signal_subCat.fitTo(data_subCat_signal,  EvalBackend=device, Save=True, )
        fit_result = signal_subCat.fitTo(data_subCat_signal,  EvalBackend=device, Save=True, )
        sig_fit_pdfs.append(signal_subCat)
        sig_roohists.append(data_subCat_signal)
    
        # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
        sigma_subCat.setConstant(True)
        alpha1_subCat.setConstant(True)
        n1_subCat.setConstant(True)
        alpha2_subCat.setConstant(True)
        n2_subCat.setConstant(True)
    
        # unfreeze the param for datacard
        CMS_hmm_sigma_cat_ggh.setConstant(False)
        CMS_hmm_peak_cat_ggh.setConstant(False)


        # ---------------------------------------------------
        # Save to Workspace
        # ---------------------------------------------------

        fout = rt.TFile(f"{workspace_path}/workspace_sig_cat{subCat_idx}_{category}.root","RECREATE")
        wout = rt.RooWorkspace("w","workspace")
        # matching names consistent with UCSD's naming scheme
        signal_subCat.SetName(f"ggH_cat{subCat_idx}_ggh_pdf");
        data_subCat_signal.SetName(f"data_ggH_cat{subCat_idx}_ggh");
        sig_norm_subCat.SetName(signal_subCat.GetName()+"_norm"); 
        wout.Import(sig_norm_subCat);
        wout.Import(signal_subCat); 
        wout.Import(data_subCat_signal); 
        # wout.Print();
        wout.Write();



        # -------------------------------------------------------------------------
        # do signal plotting with fit and data
        # -------------------------------------------------------------------------
    
        # subCat 0
        
        name = "Canvas"
        canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
        canvas.cd()
        frame = mass.frame()
        legend = rt.TLegend(0.65,0.55,0.9,0.7)
        name = data_subCat_signal.GetName()
        data_subCat_signal.plotOn(frame, DataError="SumW2", Name=name)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
        name = signal_subCat.GetName()
        signal_subCat.plotOn(frame, Name=name, LineColor=rt.kGreen)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
        
        frame.Draw()
        legend.Draw()
        
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs(f"{plot_save_path}/sigMC_plot_subCat{subCat_idx}.pdf")
    
    print("signal subcat loop is complete!")
    # -------------------------------------------------------------------------
    # do signal plotting for all sub-Cats in one plot
    # -------------------------------------------------------------------------
    
    
    sig_dict_by_sample = {
        "signal" : sig_fit_pdfs
    }
    plotSigBySample(mass, sig_dict_by_sample, sig_roohists, plot_save_path)
    



    
    
    

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
        

