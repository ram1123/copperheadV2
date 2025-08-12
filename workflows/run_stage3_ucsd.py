import time
import numpy as np
import pickle
import awkward as ak
import dask_awkward as dak
from distributed import Client
from omegaconf import OmegaConf

from typing import Tuple, List, Dict
import ROOT as rt
import ROOT
from lib.fit_models.fit_functions import MakeFEWZxBernDof3_ucsd
import argparse
import os
import copy
import pandas as pd

def setParamsConstant(pdf, x, flag=True):
    params = pdf.getParameters(ROOT.RooArgSet(mass))
    # Iterate over the parameters and set each one constant.
    paramIter = params.createIterator()
    param = paramIter.Next()
    while param:
        print("Setting parameter '{}' constant.".format(param.GetName()) + f"with {flag}")
        param.setConstant(flag)
        param = paramIter.Next()

def normalizeFlatHist(x: rt.RooRealVar,rooHist: rt.RooDataHist) -> rt.RooDataHist :
    """
    Takes rootHistogram and returns a new copy with histogram values normalized to sum to one
    """
    x_name = x.GetName()
    # copy nbins and range from, rooHist, but make it empty, and fill with flat distribution
    THist = rooHist.createHistogram(x_name).Clone("clone") # clone it just in case
    THist.Reset()
    nEntries = 100000
    # print(f"THist.GetXaxis().GetXmin(): {THist.GetXaxis().GetXmin()}")
    # print(f"THist.GetXaxis().GetXmax(): {THist.GetXaxis().GetXmax()}")
    values = np.random.uniform(
        low=THist.GetXaxis().GetXmin(), 
        high=THist.GetXaxis().GetXmax(), 
        size=nEntries
    )
    weight = np.ones_like(values)
    THist.FillN(nEntries, values, weight)
    THist.Scale(1.0 / THist.Integral()) # normalize
    print(f"THist.Integral(): {THist.Integral()}")
    normalizedHist_name = rooHist.GetName() + "_normalized"
    roo_hist_normalized = rt.RooDataHist(normalizedHist_name, normalizedHist_name, rt.RooArgSet(x), THist) 
    return roo_hist_normalized

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
            fit_range = "hiSB,loSB"
            plot_range = "full"
            model.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range(plot_range), Name=name, LineColor=color)
            legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
        frame.SetMaximum(0.0042)
        frame.Draw()
        legend.Draw()       
        canvas.SetTicks(2, 2)
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs(f"{save_path}/simultaneousPlotTestFromTutorial_{core_type}.pdf")

def plotBkgBySubCat_normalized(mass:rt.RooRealVar, model_dict_by_subCat: Dict, save_path: str):
    """
    takes the dictionary of all Bkg RooAbsPdf models grouped by same sub-category, and plot them
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
    
    for subCat_idx, subCat_list in model_dict_by_subCat.items():
        name = "Canvas"
        canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
        canvas.cd()
        frame = mass.frame()
        frame.SetTitle(f"Normalized Shape Plot of Sub-Category {subCat_idx} PDFs")
        frame.SetXTitle(f"Dimuon Mass (GeV)")
        legend = rt.TLegend(0.65,0.55,0.9,0.7)
        # apparently I have to plot invisible roo dataset for fit function plotting to work. Maybe this helps with normalization?
        normalized_hist = normalizeRooHist(mass, roo_histData_subCat1)
        normalized_hist.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0) )
        # print(f"normalized_hist integral: {normalized_hist.sum(False)}")
        for ix in range(len(subCat_list)):
            model = subCat_list[ix]
            name = model.GetName()
            color = color_list[ix]
            model.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=name, LineColor=color)
            legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
        frame.Draw()
        legend.Draw()        
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs(f"{save_path}/simultaneousPlotTestFromTutorial_subCat{subCat_idx}.pdf")

def plotBkgBySubCat(mass:rt.RooRealVar, model_dict_by_subCat: Dict, data_dict_by_subCat:Dict, save_path: str):
    """
    takes the dictionary of all Bkg RooAbsPdf models grouped by same sub-category, and plot them
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
    max_list = [1300, 1000, 400, 300, 90]
    for subCat_idx, subCat_list in model_dict_by_subCat.items():
        name = "Canvas"
        canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
        canvas.cd()
        frame = mass.frame()
        frame.SetMaximum(max_list[subCat_idx])
        frame.SetTitle(f"SMF x Core Func of Sub-Category {subCat_idx}")
        frame.SetXTitle(f"Dimuon Mass (GeV)")
        legend = rt.TLegend(0.65,0.55,0.9,0.7)
        # apparently I have to plot invisible roo dataset for fit function plotting to work. Maybe this helps with normalization?
        data_hist = data_dict_by_subCat[subCat_idx]
        data_hist.plotOn(frame, Name=data_hist.GetName())
        for ix in range(len(subCat_list)):
            model = subCat_list[ix]
            name = model.GetName()
            color = color_list[ix]
            model.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=name, LineColor=color)
            legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")

            # add chi2 dof
            ndf = model.getParameters(ROOT.RooArgSet(mass)).getSize()
            print(model.GetName())
            print(data_hist.GetName())
            chi2_ndf = frame.chiSquare(model.GetName(), data_hist.GetName(), ndf)
            model_name = model.GetName()
            print(f"{model_name} ndf: {ndf}")
            chi2_text = model_name +" chi2/ndf = {:.3f}".format(chi2_ndf)
            legend.AddEntry("", chi2_text, "")
        
        frame.Draw()
        legend.Draw()        
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs(f"{save_path}/simultaneousPlotTestFromTutorial_subCat{subCat_idx}.pdf")
        # canvas.SaveAs(f"{save_path}/simultaneousPlotTestFromTutorial_subCat{subCat_idx}.png")


def plotDataBkgDiffBySubCat(mass:rt.RooRealVar, model_dict_by_subCat_n_corefunc: Dict, data_dict_by_subCat:Dict, save_path: str):
    """
    takes the dictionary of all Bkg RooAbsPdf models grouped by same sub-category, and plot them
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
    max_list = [1300, 1000, 400, 300, 90]
    for subCat_idx, corefunc_dict in model_dict_by_subCat_n_corefunc.items():
        for corefunc_name, corefunc in corefunc_dict.items():
            name = "Canvas"
            canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
            canvas.cd()
            frame = mass.frame()
            frame.SetMaximum(max_list[subCat_idx])
            frame.SetXTitle(f"Dimuon Mass (GeV)")
            legend = rt.TLegend(0.65,0.55,0.9,0.7)
            # apparently I have to plot invisible roo dataset for fit function plotting to work. Maybe this helps with normalization?
            data_hist = data_dict_by_subCat[subCat_idx]
            data_hist.plotOn(frame, Name=data_hist.GetName())
            # for ix in range(len(subCat_list)):
            model = corefunc
            name = model.GetName()
            model.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=name)

            resid_hist = frame.residHist()
            resid_hist.SetTitle(f"{corefunc_name} residual")
            resid_hist.Draw()
            # legend.AddEntry("", f"{corefunc_name} residual", "")
            # legend.Draw()        
            
            
            canvas.Update()
            canvas.Draw()
            canvas.SaveAs(f"{save_path}/data_bkg_diff_{corefunc_name}_subCat{subCat_idx}.pdf")


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
            # sig_hist = sigHist_list[ix]
            sig_hist = sigHist_list[0]
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
        canvas.SaveAs(f"{save_path}/simultaneousPlotTestFromTutorial_{model_type}.pdf")


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-load",
    "--load_path",
    dest="load_path",
    default=None,
    action="store",
    help="save path to store stage1 output files",
    )
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="all",
    action="store",
    help="string value of year we are calculating",
    )
    parser.add_argument(
    "-cat",
    "--category",
    dest="category",
    default="ggH",
    action="store",
    help="string value production category we're working on",
    )
    parser.add_argument(
    "-l",
    "--label",
    dest="label",
    default="",
    action="store",
    help="MVA model name to load",
    )
    args = parser.parse_args()
    # check for valid arguments
    if args.load_path == None:
        print("load path to load stage1 output is not specified!")
        raise ValueError

    category = args.category.lower()
    # load_path = "/work/users/yun79/stage2_output/ggH/test/processed_events_data.parquet"
    # load_path = f"{args.load_path}/{category}/{args.year}/processed_events_data.parquet"
    # if args.year=="all":
    #     load_path = f"{args.load_path}/{category}/*/processed_events_data.parquet"
    # elif args.year=="2016only":
    #     load_path = f"{args.load_path}/{category}/2016*/processed_events_data.parquet"
    # else:
    #     load_path = f"{args.load_path}/{category}/{args.year}/processed_events_data.parquet"

    # remove category we assume that the load_path already has category specified
    if args.year=="all":
        load_path = f"{args.load_path}/*/processed_events_data.parquet"
    elif args.year=="2016only":
        load_path = f"{args.load_path}/2016*/processed_events_data.parquet"
    else:
        load_path = f"{args.load_path}/{args.year}/processed_events_data.parquet"
    print(f"load_path: {load_path}")
    processed_eventsData = ak.from_parquet(load_path)
    print(f"processed_eventsData length: {ak.num(processed_eventsData.dimuon_mass, axis=0)}")
    print("events loaded!")

    

    # Define your list of column names
    column_list = ["year", "category", "dataset", "yield"]
    yield_df = pd.DataFrame(columns=column_list)

    
    device = "cpu"
    # device = "cuda"
    # rt.RooAbsReal.setCudaMode(True)
    # Create model for physics sample
    # -------------------------------------------------------------
    # Create observables
    # mass_name = "mh_ggh"
    # mass = rt.RooRealVar(mass_name, mass_name, 120, 110, 150)
    ucsd_workspace = rt.TFile("ucsd_workspace/workspace_bkg_cat0_ggh.root")["w"]
    mass = ucsd_workspace.obj("mh_ggh")
    mass_name = mass.GetName()
    nbins = 800
    mass.setBins(nbins)
    mass.setRange("hiSB", 135, 150 )
    mass.setRange("loSB", 110, 115 )
    mass.setRange("h_peak", 115, 135 )
    mass.setRange("full", 110, 150 )
    # fit_range = "loSB,hiSB" # we're fitting bkg only
    fit_range = "hiSB,loSB" # we're fitting bkg only
    
    subCatIdx_name = "subCategory_idx"
    # subCatIdx_name = "subCategory_idx_val"

    # Initialize BWZ Redux
    # --------------------------------------------------------------

    

    # # trying bigger range do that I don't get warning message from combine like: [WARNING] Found parameter BWZ_Redux_a_coeff at boundary (within ~1sigma)
    # # old start --------------------------------------------------
    name = f"BWZ_Redux_a_coeff"
    a_coeff = rt.RooRealVar(name,name, -0.02,-0.5,0.5)
    name = f"BWZ_Redux_b_coeff"
    b_coeff = rt.RooRealVar(name,name, -0.000111,-0.02,0.02)
    name = f"BWZ_Redux_c_coeff"
    c_coeff = rt.RooRealVar(name,name, 2.0, 0.0, 4.0)
    # # old end --------------------------------------------------

    # # AN start --------------------------------------------------
    # name = f"BWZ_Redux_a_coeff"
    # a_coeff = rt.RooRealVar(name,name, 0.06231018619106862,-0.1,0.1)
    # name = f"BWZ_Redux_b_coeff"
    # b_coeff = rt.RooRealVar(name,name, -0.0001684318108879923,-0.1,0.1)
    # name = f"BWZ_Redux_c_coeff"
    # c_coeff = rt.RooRealVar(name,name, 2.14876669663328,0,5.0)
    # # AN end --------------------------------------------------
    # a_coeff.setConstant(True)
    # b_coeff.setConstant(True)
    # c_coeff.setConstant(True)

    # subCat 0
    name = "subCat0_BWZ_Redux"
    # coreBWZRedux_SubCat0 = rt.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff) 
    ucsd_workspace = rt.TFile("ucsd_workspace/workspace_bkg_cat0_ggh.root")["w"]
    coreBWZRedux_SubCat0 = ucsd_workspace.obj("bwzr_cat_ggh_pdf") # extract from ucsd root file
     
    # Construct background pdf
    # old start --------------------------------------------------------------------
    # a0_subCat0 = rt.RooRealVar("a0_subCat0", "a0_subCat0", -0.01, -0.3, 0.3)
    # a1_subCat0 = rt.RooRealVar("a1_subCat0", "a1_subCat0", 0.5, -0.5, 0.5)
    # a2_subCat0 = rt.RooRealVar("a2_subCat0", "a2_subCat0", 0.5, -0.5, 0.5)
    # old end --------------------------------------------------------------------
    a0_subCat0 = rt.RooRealVar("a0_subCat0", "a0_subCat0",  -0.0375687, -0.3, 0.3)
    a1_subCat0 = rt.RooRealVar("a1_subCat0", "a1_subCat0",  -0.00197551, -0.5, 0.5)
    a2_subCat0 = rt.RooRealVar("a2_subCat0", "a2_subCat0", 0.00259001, -0.5, 0.5)
    

    name = "subCat0_SMF"
    subCat0_SMF = rt.RooChebychev(name, name, mass, [a0_subCat0, a1_subCat0, a2_subCat0])


    
    # Construct composite pdf
    name = "model_SubCat0_SMFxBWZRedux"
    model_subCat0_BWZRedux = rt.RooProdPdf(name, name, [coreBWZRedux_SubCat0, subCat0_SMF])


    
    # subCat 1
    name = "subCat1_BWZ_Redux"
    # coreBWZRedux_SubCat1 = rt.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff) 
    coreBWZRedux_SubCat1 = coreBWZRedux_SubCat0
    
    # Construct the background pdf
    """
    NOTE: unlike what's written on line 1721 of Run2 AN, workspace root file in 
    https://gitlab.cern.ch/cms-analysis/hig/HIG-19-006/datacards/-/blob/master/ggH/ucsd/workspace_bkg_cat1_ggh.root?ref_type=heads
    doesn't have a third degree of freedom
    """
    a0_subCat1 = rt.RooRealVar("a0_subCat1", "a0_subCat1", 0.0194933, -0.5, 0.5)
    a1_subCat1 = rt.RooRealVar("a1_subCat1", "a1_subCat1", -0.00165793, -0.5, 0.5)
    # values from AN workspace
    # a0_subCat1 = rt.RooRealVar("a0_subCat1", "a0_subCat1", 0.01949329222, -0.06, 0.06)
    # a1_subCat1 = rt.RooRealVar("a1_subCat1", "a1_subCat1", -0.001657932368, -0.06, 0.06)
    # a0_subCat1 = rt.RooRealVar("a0_subCat1", "a0_subCat1", 0.01949329222, -0.1, 0.1)
    # a1_subCat1 = rt.RooRealVar("a1_subCat1", "a1_subCat1", -0.001657932368, -0.06, 0.06)
    # a0_subCat1.setConstant(True)
    # a1_subCat1.setConstant(True)
    name =  "subCat1_SMF"
    subCat1_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat1, 
                              a1_subCat1, 
                              # a3_subCat1
                             ])
     
    # Construct the composite model
    name = "model_SubCat1_SMFxBWZRedux"
    model_subCat1_BWZRedux = rt.RooProdPdf(name, name, [coreBWZRedux_SubCat1, subCat1_SMF])

    # subCat 2
    name = "subCat2_BWZ_Redux"
    # coreBWZRedux_SubCat2 = rt.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff) 
    coreBWZRedux_SubCat2 = coreBWZRedux_SubCat0
    
    # Construct the background pdf
    a0_subCat2 = rt.RooRealVar("a0_subCat2", "a0_subCat2", 0.0446045, -0.3, 0.3)
    a1_subCat2 = rt.RooRealVar("a1_subCat2", "a1_subCat2", -3.46423e-05, -0.5, 0.5)
    # a0_subCat2 = rt.RooRealVar("a0_subCat2", "a0_subCat2", 0.04460447882, -0.001, 0.06)
    # a1_subCat2 = rt.RooRealVar("a1_subCat2", "a1_subCat2", -3.46E-05, -0.001, 0.06)
    # a0_subCat2 = rt.RooRealVar("a0_subCat2", "a0_subCat2", 0.04460447882, -0.1, 0.1)
    # a1_subCat2 = rt.RooRealVar("a1_subCat2", "a1_subCat2", -3.46E-05, -0.1, 0.1)
    # a0_subCat2.setConstant(True)
    # a1_subCat2.setConstant(True)
    name = "subCat2_SMF"
    subCat2_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat2, 
                              a1_subCat2, 
                             ])
    name = "model_SubCat2_SMFxBWZRedux"
    model_subCat2_BWZRedux = rt.RooProdPdf(name, name, [coreBWZRedux_SubCat2, subCat2_SMF])    

    # subCat 3
    name = "subCat3_BWZ_Redux"
    # coreBWZRedux_SubCat3 = rt.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff) 
    coreBWZRedux_SubCat3 = coreBWZRedux_SubCat0
    
    # Construct the background pdf
    a0_subCat3 = rt.RooRealVar("a0_subCat3", "a0_subCat3", 0.0737424, -0.3, 0.3)
    a1_subCat3 = rt.RooRealVar("a1_subCat3", "a1_subCat3", -8.78956e-06, -0.5, 0.5)
    # a0_subCat3 = rt.RooRealVar("a0_subCat3", "a0_subCat3", 0.07374242573, 0.05, 0.5)
    # a1_subCat3 = rt.RooRealVar("a1_subCat3", "a1_subCat3", -8.79E-06, -0.5, 0.5)
    # a0_subCat3 = rt.RooRealVar("a0_subCat3", "a0_subCat3", 0.07374242573, -0.06, 0.2)
    # a1_subCat3 = rt.RooRealVar("a1_subCat3", "a1_subCat3", -8.79E-06, -0.06, 0.06)
    # a0_subCat3.setConstant(True)
    # a1_subCat3.setConstant(True)
    name = "subCat3_SMF"
    subCat3_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat3, 
                              a1_subCat3, 
                             ])
    name = "model_SubCat3_SMFxBWZRedux"
    model_subCat3_BWZRedux = rt.RooProdPdf(name, name, [coreBWZRedux_SubCat3, subCat3_SMF])  

    # subCat 4
    name = "subCat4_BWZ_Redux"
    # coreBWZRedux_SubCat4 = rt.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff) 
    coreBWZRedux_SubCat4 = coreBWZRedux_SubCat0
    
    # Construct the background pdf
    a0_subCat4 = rt.RooRealVar("a0_subCat4", "a0_subCat4", 0.227473, -0.4, 0.5)
    a1_subCat4 = rt.RooRealVar("a1_subCat4", "a1_subCat4", -0.00064818, -0.4, 0.4)
    # a0_subCat4 = rt.RooRealVar("a0_subCat4", "a0_subCat4", 0.2274725556, -0.06, 0.56) # AN val
    # a1_subCat4 = rt.RooRealVar("a1_subCat4", "a1_subCat4", -0.0006481800973, -0.06, 0.06) # AN val
    # a0_subCat4.setConstant(True)
    # a1_subCat4.setConstant(True)
    name = "subCat4_SMF"
    subCat4_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat4, 
                              a1_subCat4, 
                             ])
    name = "model_SubCat4_SMFxBWZRedux"
    model_subCat4_BWZRedux = rt.RooProdPdf(name, name, [coreBWZRedux_SubCat4, subCat4_SMF])  


    # ---------------------------------------------------------------
    # Initialize Data for Bkg models to fit to
    # ---------------------------------------------------------------
     
    # do for cat idx 0
    subCat_filter = (processed_eventsData[subCatIdx_name] == 0)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat0 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat0 = rt.RooDataHist("subCat0_rooHist_BWZRedux","subCat0_rooHist_BWZRedux", rt.RooArgSet(mass), roo_datasetData_subCat0)
    data_subCat0_BWZRedux = roo_histData_subCat0

    # do for cat idx 1
    subCat_filter = (processed_eventsData[subCatIdx_name] == 1)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat1 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat1 = rt.RooDataHist("subCat1_rooHist_BWZRedux","subCat1_rooHist_BWZRedux", rt.RooArgSet(mass), roo_datasetData_subCat1)
    data_subCat1_BWZRedux = roo_histData_subCat1

    # do for cat idx 2
    subCat_filter = (processed_eventsData[subCatIdx_name] == 2)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat2 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat2 = rt.RooDataHist("subCat2_rooHist_BWZRedux","subCat2_rooHist_BWZRedux", rt.RooArgSet(mass), roo_datasetData_subCat2)
    data_subCat2_BWZRedux = roo_histData_subCat2

    # do for cat idx 3
    subCat_filter = (processed_eventsData[subCatIdx_name] == 3)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat3 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat3 = rt.RooDataHist("subCat3_rooHist_BWZRedux","subCat3_rooHist_BWZRedux", rt.RooArgSet(mass), roo_datasetData_subCat3)
    data_subCat3_BWZRedux = roo_histData_subCat3

    # do for cat idx 4
    subCat_filter = (processed_eventsData[subCatIdx_name] == 4)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat4 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat4 = rt.RooDataHist("subCat4_rooHist_BWZRedux","subCat4_rooHist_BWZRedux", rt.RooArgSet(mass), roo_datasetData_subCat4)
    data_subCat4_BWZRedux = roo_histData_subCat4




    # --------------------------------------------------------------
    # Initialize Sum Exponential
    # --------------------------------------------------------------

    # original start --------------------------------------------------
    # name = f"RooSumTwoExpPdf_a1_coeff"
    # a1_coeff = rt.RooRealVar(name,name, -0.043657,-2.0,1)
    # name = f"RooSumTwoExpPdf_a2_coeff"
    # a2_coeff = rt.RooRealVar(name,name, -0.23726,-2.0,1)
    # name = f"RooSumTwoExpPdf_f_coeff"
    # f_coeff = rt.RooRealVar(name,name, 0.9,0.0,1.0)
    # original end --------------------------------------------------

    # trying bigger range do that I don't get warning message from combine like: [WARNING] Found parameter BWZ_Redux_a_coeff at boundary (within ~1sigma)
    # # new start --------------------------------------------------
    name = f"RooSumTwoExpPdf_a1_coeff"
    a1_coeff = rt.RooRealVar(name,name, 0.00001, -0.1, 0.0)
    name = f"RooSumTwoExpPdf_a2_coeff"
    a2_coeff = rt.RooRealVar(name,name, 0.0001, -0.5, 0.0)
    name = f"RooSumTwoExpPdf_f_coeff"
    f_coeff = rt.RooRealVar(name,name, 0.9,0.0,1.0)
    # # new end --------------------------------------------------


    # AN start --------------------------------------------------
    # name = f"RooSumTwoExpPdf_a1_coeff"
    # a1_coeff = rt.RooRealVar(name,name, -0.034803252906117965,-1.0,0.0)
    # name = f"RooSumTwoExpPdf_a2_coeff"
    # a2_coeff = rt.RooRealVar(name,name, -0.1497754374262389,-1.0,0)
    # name = f"RooSumTwoExpPdf_f_coeff"
    # f_coeff = rt.RooRealVar(name,name, 0.7549173445209436,0.0,1.0)
    # AN end --------------------------------------------------
    # a1_coeff.setConstant(True)
    # a2_coeff.setConstant(True)
    # f_coeff.setConstant(True)

    # sumexp subcat
    a0_subCat0_sumExp = rt.RooRealVar("a0_subCat0_sumExp", "a0_subCat0_sumExp", -0.1, -1, 1)
    a1_subCat0_sumExp = rt.RooRealVar("a1_subCat0_sumExp", "a1_subCat0_sumExp", 0.5, -0.5, 0.5)
    a2_subCat0_sumExp = rt.RooRealVar("a2_subCat0_sumExp", "a2_subCat0_sumExp", 0.5, -0.5, 0.5)
    
    name = "subCat0_sumExp"
    # coreSumExp_SubCat0 = rt.RooSumTwoExpPdf(name, name, mass, a1_coeff, a2_coeff, f_coeff) 
    ucsd_workspace = rt.TFile("ucsd_workspace/workspace_bkg_cat0_ggh.root")["w"]
    coreSumExp_SubCat0 = ucsd_workspace.obj("exp_cat_ggh_pdf") # extract from ucsd root file
     
    name = "subCat0_SMF_sumExp"
    subCat0_SumExp_SMF = rt.RooChebychev(name, name, mass, [a0_subCat0, a1_subCat0, a2_subCat0]) # original
    # subCat0_SumExp_SMF = rt.RooChebychev(name, name, mass, [a0_subCat0_sumExp, a1_subCat0_sumExp, a2_subCat0_sumExp]) 


    
    # Construct composite pdf
    name = "model_SubCat0_SMFxSumExp"
    model_subCat0_sumExp = rt.RooProdPdf(name, name, [coreSumExp_SubCat0, subCat0_SumExp_SMF])
     
    # subCat 1
    name = "subCat1_sumExp"
    # coreSumExp_SubCat1 = rt.RooSumTwoExpPdf(name, name, mass, a1_coeff, a2_coeff, f_coeff) 
    coreSumExp_SubCat1 = coreSumExp_SubCat0
    
    a0_subCat1_sumExp = rt.RooRealVar("a0_subCat1_sumExp", "a0_subCat1_sumExp", -0.1, -1, 1)
    a1_subCat1_sumExp = rt.RooRealVar("a1_subCat1_sumExp", "a1_subCat1_sumExp", 0.5, -0.5, 0.5)
    
    name = "subCat1_SMF_sumExp"
    subCat1_SumExp_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat1, 
                              a1_subCat1, 
                             ])
    # subCat1_SumExp_SMF = rt.RooChebychev(name, name, mass, 
    #                          [a0_subCat1_sumExp, 
    #                           a1_subCat1_sumExp, 
    #                          ])
     
    # Construct the composite model
    name = "model_SubCat1_SMFxSumExp"
    model_subCat1_sumExp = rt.RooProdPdf(name, name, [coreSumExp_SubCat1, subCat1_SumExp_SMF])

    # subCat 2
    name = "subCat2_sumExp"
    # coreSumExp_SubCat2 = rt.RooSumTwoExpPdf(name, name, mass, a1_coeff, a2_coeff, f_coeff) 
    coreSumExp_SubCat2 = coreSumExp_SubCat0

    a0_subCat2_sumExp = rt.RooRealVar("a0_subCat2_sumExp", "a0_subCat2_sumExp", -0.1, -1, 1)
    a1_subCat2_sumExp = rt.RooRealVar("a1_subCat2_sumExp", "a1_subCat2_sumExp", 0.5, -0.5, 0.5)
    
    name = "subCat2_SMF_sumExp"
    subCat2_SumExp_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat2, 
                              a1_subCat2, 
                             ])
    # subCat2_SumExp_SMF = rt.RooChebychev(name, name, mass, 
    #                          [a0_subCat2_sumExp, 
    #                           a1_subCat2_sumExp, 
    #                          ])
    name = "model_SubCat2_SMFxSumExp"
    model_subCat2_sumExp = rt.RooProdPdf(name, name, [coreSumExp_SubCat2, subCat2_SumExp_SMF])    

    # subCat 3
    name = "subCat3_sumExp"
    # coreSumExp_SubCat3 = rt.RooSumTwoExpPdf(name, name, mass, a1_coeff, a2_coeff, f_coeff) 
    coreSumExp_SubCat3 = coreSumExp_SubCat0

    a0_subCat3_sumExp = rt.RooRealVar("a0_subCat3_sumExp", "a0_subCat3_sumExp", -0.1, -1, 1)
    a1_subCat3_sumExp = rt.RooRealVar("a1_subCat3_sumExp", "a1_subCat3_sumExp", 0.5, -0.5, 0.5)
    
    name = "subCat3_SMF_sumExp"
    subCat3_SumExp_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat3, 
                              a1_subCat3, 
                             ])
    # subCat3_SumExp_SMF = rt.RooChebychev(name, name, mass, 
    #                          [a0_subCat3_sumExp, 
    #                           a1_subCat3_sumExp, 
    #                          ])
    name = "model_SubCat3_SMFxSumExp"
    model_subCat3_sumExp = rt.RooProdPdf(name, name, [coreSumExp_SubCat3, subCat3_SumExp_SMF])    

    # subCat 4
    name = "subCat4_sumExp"
    # coreSumExp_SubCat4 = rt.RooSumTwoExpPdf(name, name, mass, a1_coeff, a2_coeff, f_coeff) 
    coreSumExp_SubCat4 = coreSumExp_SubCat0

    a0_subCat4_sumExp = rt.RooRealVar("a0_subCat4_sumExp", "a0_subCat4_sumExp", -0.1, -5, 5)
    a1_subCat4_sumExp = rt.RooRealVar("a1_subCat4_sumExp", "a1_subCat4_sumExp", 0.5, -0.5, 0.5)
    
    name = "subCat4_SMF_sumExp"
    subCat4_SumExp_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat4, 
                              a1_subCat4, 
                             ])
    # subCat4_SumExp_SMF = rt.RooChebychev(name, name, mass, 
    #                          [a0_subCat4_sumExp, 
    #                           a1_subCat4_sumExp, 
    #                          ])
    name = "model_SubCat4_SMFxSumExp"
    model_subCat4_sumExp = rt.RooProdPdf(name, name, [coreSumExp_SubCat4, subCat4_SumExp_SMF])    
     
    # Initialize Data for Bkg models to fit to
    # ---------------------------------------------------------------
     
    # do for cat idx 0
    subCat_filter = (processed_eventsData[subCatIdx_name] == 0)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat0_sumExp = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat0_sumExp = rt.RooDataHist("subCat0_rooHist_sumExp","subCat0_rooHist_sumExp", rt.RooArgSet(mass), roo_datasetData_subCat0_sumExp)
    data_subCat0_sumExp = roo_histData_subCat0_sumExp

    # do for cat idx 1
    subCat_filter = (processed_eventsData[subCatIdx_name] == 1)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat1_sumExp = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat1_sumExp = rt.RooDataHist("subCat1_rooHist_sumExp","subCat1_rooHist_sumExp", rt.RooArgSet(mass), roo_datasetData_subCat1_sumExp)
    data_subCat1_sumExp = roo_histData_subCat1_sumExp

    # do for cat idx 2
    subCat_filter = (processed_eventsData[subCatIdx_name] == 2)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat2_sumExp = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat2_sumExp = rt.RooDataHist("subCat2_rooHist_sumExp","subCat2_rooHist_sumExp", rt.RooArgSet(mass), roo_datasetData_subCat2_sumExp)
    data_subCat2_sumExp = roo_histData_subCat2_sumExp

    # do for cat idx 3
    subCat_filter = (processed_eventsData[subCatIdx_name] == 3)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat3_sumExp = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat3_sumExp = rt.RooDataHist("subCat3_rooHist_sumExp","subCat3_rooHist_sumExp", rt.RooArgSet(mass), roo_datasetData_subCat3_sumExp)
    data_subCat3_sumExp = roo_histData_subCat3_sumExp


    # do for cat idx 4
    subCat_filter = (processed_eventsData[subCatIdx_name] == 4)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat4_sumExp = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat4_sumExp = rt.RooDataHist("subCat4_rooHist_sumExp","subCat4_rooHist_sumExp", rt.RooArgSet(mass), roo_datasetData_subCat4_sumExp)
    data_subCat4_sumExp = roo_histData_subCat4_sumExp


    # --------------------------------------------------------------
    # Initialize FEWZxBernstein
    # --------------------------------------------------------------
    

    # # old start --------------------------------------------------
    # name = f"FEWZxBern_c1"
    # c1 = rt.RooRealVar(name,name, 0.2,-2,2)
    # name = f"FEWZxBern_c2"
    # c2 = rt.RooRealVar(name,name, 1.0,-2,2)
    # name = f"FEWZxBern_c3"
    # c3 = rt.RooRealVar(name,name, 0.1,-2,2)
    # # old end --------------------------------------------------

    # # an start --------------------------------------------------
    name = f"FEWZxBern_c1"
    c1 = rt.RooRealVar(name,name, 0.956483450832728,0.5,1.5)
    name = f"FEWZxBern_c2"
    c2 = rt.RooRealVar(name,name, 0.9607652348517792,0.5,1.5)
    name = f"FEWZxBern_c3"
    c3 = rt.RooRealVar(name,name, 0.9214633453188963,0.5,1.5)

    # name = f"FEWZxBern_c1"
    # c1 = rt.RooRealVar(name,name, 0.956483450832728,-19,10)
    # name = f"FEWZxBern_c2"
    # c2 = rt.RooRealVar(name,name, -10,-19,10)
    # name = f"FEWZxBern_c3"
    # c3 = rt.RooRealVar(name,name, 7,-19,10)
    # # an end --------------------------------------------------


    # # new start --------------------------------------------------
    # name = f"FEWZxBern_c1"
    # c1 = rt.RooRealVar(name,name, 0.956483450832728,-10,10)
    # name = f"FEWZxBern_c2"
    # c2 = rt.RooRealVar(name,name, 0.9607652348517792,-10,10)
    # name = f"FEWZxBern_c3"
    # c3 = rt.RooRealVar(name,name, 0.9214633453188963,-10,10)
    # # new end --------------------------------------------------
    
    # new start --------------------------------------------------
    # name = f"FEWZxBern_c1"
    # c1 = rt.RooRealVar(name,name, 1.00, 0.5, 1.5)
    # name = f"FEWZxBern_c2"
    # c2 = rt.RooRealVar(name,name, 1.00, 0.5, 1.5)
    # name = f"FEWZxBern_c3"
    # c3 = rt.RooRealVar(name,name, 1.00, 0.5, 1.5)

    # new end --------------------------------------------------
    # BernCoeff_list = [c1, c2, c3, c4] # we use RooBernstein, which requires n+1 parameters https://root.cern.ch/doc/master/classRooBernstein.html
    BernCoeff_list = [c1, c2, c3]
    # c1.setConstant(True)
    # c2.setConstant(True)
    # c3.setConstant(True)
    
    name = "subCat0_FEWZxBern"
    # coreFEWZxBern_SubCat0, params_FEWZxBern_SubCat0 = MakeFEWZxBernDof3(name, name, mass, BernCoeff_list) 
    # ucsd_workspace = rt.TFile("ucsd_workspace/workspace_bkg_cat0_ggh.root")["w"]
    coreFEWZxBern_SubCat0 = ucsd_workspace.obj("fewz_1j_spl_cat_ggh_pdf") # extract from ucsd root file
    # coreFEWZxBern_SubCat0, params_FEWZxBern_SubCat0 = MakeFEWZxBernDof3_ucsd(name, name, mass, BernCoeff_list) 
     
    name = "subCat0_SMF_FEWZxBern"
    subCat0_FEWZxBern_SMF = rt.RooChebychev(name, name, mass, [a0_subCat0, a1_subCat0, a2_subCat0])


    
    # Construct composite pdf
    name = "model_SubCat0_SMFxFEWZxBern"
    model_subCat0_FEWZxBern = rt.RooProdPdf(name, name, [coreFEWZxBern_SubCat0, subCat0_FEWZxBern_SMF])
     
    # subCat 1
    name = "subCat1_FEWZxBern"
    # coreFEWZxBern_SubCat1, params_FEWZxBern_SubCat1 = MakeFEWZxBernDof3(name, name, mass, c1, c2, c3) 
    coreFEWZxBern_SubCat1 = coreFEWZxBern_SubCat0
    

    name = "subCat1_SMF_FEWZxBern"
    subCat1_FEWZxBern_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat1, 
                              a1_subCat1, 
                              # a3_subCat1
                             ])
     
    # Construct the composite model
    name = "model_SubCat1_SMFxFEWZxBern"
    model_subCat1_FEWZxBern = rt.RooProdPdf(name, name, [coreFEWZxBern_SubCat1, subCat1_FEWZxBern_SMF])

    # subCat 2
    name = "subCat2_FEWZxBern"
    # coreFEWZxBern_SubCat2, params_FEWZxBern_SubCat2 = MakeFEWZxBernDof3(name, name, mass, c1, c2, c3) 
    coreFEWZxBern_SubCat2 = coreFEWZxBern_SubCat0
    
    name = "subCat2_SMF_FEWZxBern"
    subCat2_FEWZxBern_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat2, 
                              a1_subCat2, 
                             ])
    name = "model_SubCat2_SMFxFEWZxBern"
    model_subCat2_FEWZxBern = rt.RooProdPdf(name, name, [coreFEWZxBern_SubCat2, subCat2_FEWZxBern_SMF])    

    # subCat 3
    name = "subCat3_FEWZxBern"
    # coreFEWZxBern_SubCat3, params_FEWZxBern_SubCat3 = MakeFEWZxBernDof3(name, name, mass, c1, c2, c3)  
    coreFEWZxBern_SubCat3 = coreFEWZxBern_SubCat0
    
    name = "subCat3_SMF_FEWZxBern"
    subCat3_FEWZxBern_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat3, 
                              a1_subCat3, 
                             ])
    name = "model_SubCat3_SMFxFEWZxBern"
    model_subCat3_FEWZxBern = rt.RooProdPdf(name, name, [coreFEWZxBern_SubCat3, subCat3_FEWZxBern_SMF])    

    # subCat 4
    name = "subCat4_FEWZxBern"
    # coreFEWZxBern_SubCat4, params_FEWZxBern_SubCat4 = MakeFEWZxBernDof3(name, name, mass, c1, c2, c3)  
    coreFEWZxBern_SubCat4 = coreFEWZxBern_SubCat0
    
    name = "subCat4_SMF_FEWZxBern"
    subCat4_FEWZxBern_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat4, 
                              a1_subCat4, 
                             ])
    name = "model_SubCat4_SMFxFEWZxBern"
    model_subCat4_FEWZxBern = rt.RooProdPdf(name, name, [coreFEWZxBern_SubCat4, subCat4_FEWZxBern_SMF])        
     
    # Initialize Data for Bkg models to fit to
    # ---------------------------------------------------------------
    
    # do for cat idx 0
    subCat_filter = (processed_eventsData[subCatIdx_name] == 0)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat0_FEWZxBern = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat0_FEWZxBern = rt.RooDataHist("subCat0_rooHist_FEWZxBern","subCat0_rooHist_FEWZxBern", rt.RooArgSet(mass), roo_datasetData_subCat0_FEWZxBern)
    data_subCat0_FEWZxBern = roo_histData_subCat0_FEWZxBern

    # do for cat idx 1
    subCat_filter = (processed_eventsData[subCatIdx_name] == 1)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat1_FEWZxBern = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat1_FEWZxBern = rt.RooDataHist("subCat1_rooHist_FEWZxBern","subCat1_rooHist_FEWZxBern", rt.RooArgSet(mass), roo_datasetData_subCat1_FEWZxBern)
    data_subCat1_FEWZxBern = roo_histData_subCat1_FEWZxBern

    # do for cat idx 2
    subCat_filter = (processed_eventsData[subCatIdx_name] == 2)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat2_FEWZxBern = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat2_FEWZxBern = rt.RooDataHist("subCat2_rooHist_FEWZxBern","subCat2_rooHist_FEWZxBern", rt.RooArgSet(mass), roo_datasetData_subCat2_FEWZxBern)
    data_subCat2_FEWZxBern = roo_histData_subCat2_FEWZxBern

    # do for cat idx 3
    subCat_filter = (processed_eventsData[subCatIdx_name] == 3)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat3_FEWZxBern = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    # print(f"roo_datasetData_subCat3_FEWZxBern name: {roo_datasetData_subCat3_FEWZxBern.GetName()}")
    roo_histData_subCat3_FEWZxBern = rt.RooDataHist("subCat3_rooHist_FEWZxBern","subCat3_rooHist_FEWZxBern", rt.RooArgSet(mass), roo_datasetData_subCat3_FEWZxBern)
    data_subCat3_FEWZxBern = roo_histData_subCat3_FEWZxBern
    # print(f"data_subCat3_FEWZxBern name: {data_subCat3_FEWZxBern.GetName()}")


    # do for cat idx 4
    subCat_filter = (processed_eventsData[subCatIdx_name] == 4)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat4_FEWZxBern = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat4_FEWZxBern = rt.RooDataHist("subCat4_rooHist_FEWZxBern","subCat4_rooHist_FEWZxBern", rt.RooArgSet(mass), roo_datasetData_subCat4_FEWZxBern)
    data_subCat4_FEWZxBern = roo_histData_subCat4_FEWZxBern

    #----------------------------------------------------------------------------
    # Do sim fit to the core function first
    # ---------------------------------------------------------------------------

    # first generate full data (all subcats included) 
    subCat_mass_arr = processed_eventsData.dimuon_mass
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_histData_allSubCat = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_allSubCat = rt.RooDataHist("allSubCat_rooHist","allSubCat_rooHist", rt.RooArgSet(mass), roo_histData_allSubCat)
    data_allSubCat_BWZ = roo_histData_allSubCat
    data_allSubCat_sumExp = copy.deepcopy(roo_histData_allSubCat)
    data_allSubCat_FEWZxBern = copy.deepcopy(roo_histData_allSubCat)
    
    # # Define category to distinguish physics and control samples events
    # allSubCat_sample = rt.RooCategory("allSubCat_sample", "allSubCat_sample")
    # allSubCat_sample.defineType("allsubCat_BWZRedux")
    # allSubCat_sample.defineType("allsubCat_sumExp")
    # # allSubCat_sample.defineType("allsubCat_FEWZxBern")

     
    # # Construct combined dataset in (x,allSubCat_sample)
    # allSubCat_combData = rt.RooDataSet(
    #     "allSubCat_combData",
    #     "all subCat combined data",
    #     {mass},
    #     Index=allSubCat_sample,
    #     Import={
    #         "allsubCat_BWZRedux": data_allSubCat_BWZ, 
    #         "allsubCat_sumExp": data_allSubCat_sumExp,
    #         # "allsubCat_FEWZxBern": data_allSubCat_FEWZxBern,
    #     },
    # )
    # ---------------------------------------------------
    # Construct a simultaneous pdf in (x, sample)
    # -----------------------------------------------------------------------------------
     
    # allSubCat_simPdf = rt.RooSimultaneous(
    #                             "allSubCat_simPdf", 
    #                             "all cat simultaneous pdf", 
    #                             {
    #                                 "allsubCat_BWZRedux": coreBWZRedux_SubCat0, 
    #                                 "allsubCat_sumExp": coreSumExp_SubCat0,
    #                                 # "allsubCat_FEWZxBern": coreFEWZxBern_SubCat0,
    #                             }, 
    #                             allSubCat_sample,
    # )
    # ---------------------------------------------------
    # Perform a simultaneous fit
    # ---------------------------------------------------
     
    # start = time.time()

    # _ = allSubCat_simPdf.fitTo(allSubCat_combData, rt.RooFit.Range(fit_range), EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0)
    # fitResult = allSubCat_simPdf.fitTo(allSubCat_combData, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,)
    # end = time.time()
    
    # fitResult.Print()

    # fit core functions separately
    _ = coreBWZRedux_SubCat0.fitTo(data_allSubCat_BWZ, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,)
    fitResult = coreBWZRedux_SubCat0.fitTo(data_allSubCat_BWZ, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,)
    fitResult.Print()
    _ = coreSumExp_SubCat0.fitTo(data_allSubCat_sumExp, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,)
    fitResult = coreSumExp_SubCat0.fitTo(data_allSubCat_sumExp, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,)
    fitResult.Print()

    
    # freeze core pdf params
    # BWZ redux
    # a_coeff.setConstant(True)
    # b_coeff.setConstant(True)
    # c_coeff.setConstant(True)
    # ucsd_workspace.obj("bwzr_cat_ggh_coef1").setConstant(True)
    # ucsd_workspace.obj("bwzr_cat_ggh_coef2").setConstant(True)
    # ucsd_workspace.obj("bwzr_cat_ggh_coef3").setConstant(True)
    setParamsConstant(coreBWZRedux_SubCat0, mass, flag=True)
    
    # sumExp
    # a1_coeff.setConstant(True)
    # a2_coeff.setConstant(True)
    # f_coeff.setConstant(True)
    # ucsd_workspace.obj("exp_order2_cat_ggh_coef1").setConstant(True)
    # ucsd_workspace.obj("exp_order2_cat_ggh_coef2").setConstant(True)
    # ucsd_workspace.obj("exp_order2_cat_ggh_frac1").setConstant(True)
    setParamsConstant(coreSumExp_SubCat0, mass, flag=True)

    # fit FEWZxBern separately
    _ = coreFEWZxBern_SubCat0.fitTo(data_allSubCat_FEWZxBern, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,)
    fitResult = coreFEWZxBern_SubCat0.fitTo(data_allSubCat_FEWZxBern, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,)
    fitResult.Print()

    # FEWZxBern
    # c1.setConstant(True)
    # c2.setConstant(True)
    # c3.setConstant(True)
    # ucsd_workspace.obj("fewz_1j_spl_order3_bern_cat_ggh_coef1").setConstant(True)
    # ucsd_workspace.obj("fewz_1j_spl_order3_bern_cat_ggh_coef2").setConstant(True)
    # ucsd_workspace.obj("fewz_1j_spl_order3_bern_cat_ggh_coef3").setConstant(True)
    setParamsConstant(coreFEWZxBern_SubCat0, mass, flag=True)
    # c4.setConstant(True)

    # # -------------------------------------------------------------------------
    # # do background core function plotting
    # # -------------------------------------------------------------------------
    # base_path = f"./validation/figs/{args.year}/{args.label}"
    # # plot_save_path = f"./validation/figs/{args.year}"
    # plot_save_path = base_path
    # data_name = "Run2 Data" # for Legend
    
    # # BWZ redux
    # name = "Canvas"
    # canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    # canvas.cd()
    # frame = mass.frame()
    # legend = rt.TLegend(0.65,0.55,0.9,0.7)
    # fit_range = "hiSB,loSB"
    # plot_range = "full"
    # name = data_allSubCat_BWZ.GetName()
    # data_allSubCat_BWZ.plotOn(frame, DataError="SumW2", Name=name)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),data_name, "P")
    # name = coreBWZRedux_SubCat0.GetName()
    # coreBWZRedux_SubCat0.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range(plot_range), Name=name)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")

    # ndf = coreBWZRedux_SubCat0.getParameters(ROOT.RooArgSet(mass)).getSize()
    # chi2_ndf = frame.chiSquare(coreBWZRedux_SubCat0.GetName(), data_allSubCat_BWZ.GetName(), ndf)
    # print(f"ndf: {ndf}")
    # chi2_text = "chi2/ndf = {:.3f}".format(chi2_ndf)
    # legend.AddEntry("", chi2_text, "")
    
    # frame.Draw()
    # legend.Draw()
    
    # canvas.Update()
    # canvas.Draw()
    # canvas.SaveAs(f"{plot_save_path}/stage3_plot_BWZ_redux.pdf")


    # # Sum Exp
    # name = "Canvas"
    # canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    # canvas.cd()
    # frame = mass.frame()
    # legend = rt.TLegend(0.65,0.55,0.9,0.7)
    # fit_range = "hiSB,loSB"
    # plot_range = "full"
    # name = data_allSubCat_FEWZxBern.GetName()
    # # name = "Run2 Data"
    # data_allSubCat_FEWZxBern.plotOn(frame, DataError="SumW2", Name=name)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),data_name, "P")
    # name = coreSumExp_SubCat0.GetName()
    # coreSumExp_SubCat0.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range(plot_range), Name=name)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")

    # ndf = coreSumExp_SubCat0.getParameters(ROOT.RooArgSet(mass)).getSize()
    # chi2_ndf = frame.chiSquare(coreSumExp_SubCat0.GetName(), data_allSubCat_FEWZxBern.GetName(), ndf)
    # print(f"ndf: {ndf}")
    # chi2_text = "chi2/ndf = {:.3f}".format(chi2_ndf)
    # legend.AddEntry("", chi2_text, "")
    
    # frame.Draw()
    # legend.Draw()
    
    # canvas.Update()
    # canvas.Draw()
    # canvas.SaveAs(f"{plot_save_path}/stage3_plot_sumExp.pdf")

    # # FEWZ Bern
    # name = "Canvas"
    # canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    # canvas.cd()
    # frame = mass.frame()
    # legend = rt.TLegend(0.65,0.55,0.9,0.7)
    # fit_range = "hiSB,loSB"
    # plot_range = "full"
    # name = data_allSubCat_FEWZxBern.GetName()
    # # name = "Run2 Data"
    # data_allSubCat_FEWZxBern.plotOn(frame, DataError="SumW2", Name=name)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),data_name, "P")
    # name = coreFEWZxBern_SubCat0.GetName()
    # coreFEWZxBern_SubCat0.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range(plot_range), Name=name)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")

    # ndf = coreFEWZxBern_SubCat0.getParameters(ROOT.RooArgSet(mass)).getSize()
    # chi2_ndf = frame.chiSquare(coreFEWZxBern_SubCat0.GetName(), data_allSubCat_FEWZxBern.GetName(), ndf)
    # print(f"ndf: {ndf}")
    # chi2_text = "chi2/ndf = {:.3f}".format(chi2_ndf)
    # legend.AddEntry("", chi2_text, "")
    
    # frame.Draw()
    # legend.Draw()
    
    # canvas.Update()
    # canvas.Draw()
    
    # canvas.SaveAs(f"{plot_save_path}/stage3_plot_FEWZxBern.pdf")

    # # raise ValueError

    
    
    #----------------------------------------------------------------------------
    # Now do core-Pdf fitting with all SMF
    # ---------------------------------------------------------------------------
     
    # Define category to distinguish physics and control samples events
    sample = rt.RooCategory("sample", "sample")
    sample.defineType("subCat0_BWZRedux")
    sample.defineType("subCat1_BWZRedux")
    sample.defineType("subCat2_BWZRedux")
    # sample.defineType("subCat3_BWZRedux")
    # sample.defineType("subCat4_BWZRedux")
    sample.defineType("subCat0_sumExp")
    sample.defineType("subCat1_sumExp")
    sample.defineType("subCat2_sumExp")
    # sample.defineType("subCat3_sumExp")
    # sample.defineType("subCat4_sumExp")
    sample.defineType("subCat0_FEWZxBern")
    sample.defineType("subCat1_FEWZxBern")
    sample.defineType("subCat2_FEWZxBern")
    # sample.defineType("subCat3_FEWZxBern")
    # sample.defineType("subCat4_FEWZxBern")
     
    # Construct combined dataset in (x,sample)
    combData = rt.RooDataSet(
        "combData",
        "combined data",
        {mass},
        Index=sample,
        Import={
            "subCat0_BWZRedux": data_subCat0_BWZRedux, 
            "subCat1_BWZRedux": data_subCat1_BWZRedux,
            "subCat2_BWZRedux": data_subCat2_BWZRedux,
            # "subCat3_BWZRedux": data_subCat3_BWZRedux,
            # "subCat4_BWZRedux": data_subCat4_BWZRedux,
            "subCat0_sumExp": data_subCat0_sumExp, 
            "subCat1_sumExp": data_subCat1_sumExp,
            "subCat2_sumExp": data_subCat2_sumExp,
            # "subCat3_sumExp": data_subCat3_sumExp,
            # "subCat4_sumExp": data_subCat4_sumExp,
            "subCat0_FEWZxBern": data_subCat0_FEWZxBern, 
            "subCat1_FEWZxBern": data_subCat1_FEWZxBern,
            "subCat2_FEWZxBern": data_subCat2_FEWZxBern,
            # "subCat3_FEWZxBern": data_subCat3_FEWZxBern,
            # "subCat4_FEWZxBern": data_subCat4_FEWZxBern,
        },
    )
    # ---------------------------------------------------
    # Construct a simultaneous pdf in (x, sample)
    # -----------------------------------------------------------------------------------
     
    simPdf = rt.RooSimultaneous(
                                "simPdf", 
                                "simultaneous pdf", 
                                {
                                    "subCat0_BWZRedux": model_subCat0_BWZRedux, 
                                    "subCat1_BWZRedux": model_subCat1_BWZRedux,
                                    "subCat2_BWZRedux": model_subCat2_BWZRedux,
                                    # "subCat3_BWZRedux": model_subCat3_BWZRedux,
                                    # "subCat4_BWZRedux": model_subCat4_BWZRedux,
                                    "subCat0_sumExp": model_subCat0_sumExp, 
                                    "subCat1_sumExp": model_subCat1_sumExp,
                                    "subCat2_sumExp": model_subCat2_sumExp,
                                    # "subCat3_sumExp": model_subCat3_sumExp,
                                    # "subCat4_sumExp": model_subCat4_sumExp,
                                    "subCat0_FEWZxBern": model_subCat0_FEWZxBern, 
                                    "subCat1_FEWZxBern": model_subCat1_FEWZxBern,
                                    "subCat2_FEWZxBern": model_subCat2_FEWZxBern,
                                    # "subCat3_FEWZxBern": model_subCat3_FEWZxBern,
                                    # "subCat4_FEWZxBern": model_subCat4_FEWZxBern,
                                }, 
                                sample,
    )
    # ---------------------------------------------------
    # Perform a simultaneous fit
    # ---------------------------------------------------
     
    start = time.time()

    _ = simPdf.fitTo(combData, rt.RooFit.Range(fit_range), EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0,SumW2Error=True)
    fitResult = simPdf.fitTo(combData, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,SumW2Error=True)
    end = time.time()
    
    fitResult.Print()


    # ----------------------------------------------------------------
    # subcat 4 specific fit
    # ----------------------------------------------------------------
    
    # Define category to distinguish physics and control samples events
    sample = rt.RooCategory("sample", "sample")
    # sample.defineType("subCat0_BWZRedux")
    # sample.defineType("subCat1_BWZRedux")
    # sample.defineType("subCat2_BWZRedux")
    sample.defineType("subCat3_BWZRedux")
    # sample.defineType("subCat4_BWZRedux")
    # sample.defineType("subCat0_sumExp")
    # sample.defineType("subCat1_sumExp")
    # sample.defineType("subCat2_sumExp")
    sample.defineType("subCat3_sumExp")
    # sample.defineType("subCat4_sumExp")
    # sample.defineType("subCat0_FEWZxBern")
    # sample.defineType("subCat1_FEWZxBern")
    # sample.defineType("subCat2_FEWZxBern")
    sample.defineType("subCat3_FEWZxBern")
    # sample.defineType("subCat4_FEWZxBern")
     
    # Construct combined dataset in (x,sample)
    combData = rt.RooDataSet(
        "combData",
        "combined data",
        {mass},
        Index=sample,
        Import={
            # "subCat0_BWZRedux": data_subCat0_BWZRedux, 
            # "subCat1_BWZRedux": data_subCat1_BWZRedux,
            # "subCat2_BWZRedux": data_subCat2_BWZRedux,
            "subCat3_BWZRedux": data_subCat3_BWZRedux,
            # "subCat4_BWZRedux": data_subCat4_BWZRedux,
            # "subCat0_sumExp": data_subCat0_sumExp, 
            # "subCat1_sumExp": data_subCat1_sumExp,
            # "subCat2_sumExp": data_subCat2_sumExp,
            "subCat3_sumExp": data_subCat3_sumExp,
            # "subCat4_sumExp": data_subCat4_sumExp,
            # "subCat0_FEWZxBern": data_subCat0_FEWZxBern, 
            # "subCat1_FEWZxBern": data_subCat1_FEWZxBern,
            # "subCat2_FEWZxBern": data_subCat2_FEWZxBern,
            "subCat3_FEWZxBern": data_subCat3_FEWZxBern,
            # "subCat4_FEWZxBern": data_subCat4_FEWZxBern,
        },
    )
    # ---------------------------------------------------
    # Construct a simultaneous pdf in (x, sample)
    # -----------------------------------------------------------------------------------
     
    simPdf = rt.RooSimultaneous(
                                "simPdf", 
                                "simultaneous pdf", 
                                {
                                    # "subCat0_BWZRedux": model_subCat0_BWZRedux, 
                                    # "subCat1_BWZRedux": model_subCat1_BWZRedux,
                                    # "subCat2_BWZRedux": model_subCat2_BWZRedux,
                                    "subCat3_BWZRedux": model_subCat3_BWZRedux,
                                    # "subCat4_BWZRedux": model_subCat4_BWZRedux,
                                    # "subCat0_sumExp": model_subCat0_sumExp, 
                                    # "subCat1_sumExp": model_subCat1_sumExp,
                                    # "subCat2_sumExp": model_subCat2_sumExp,
                                    "subCat3_sumExp": model_subCat3_sumExp,
                                    # "subCat4_sumExp": model_subCat4_sumExp,
                                    # "subCat0_FEWZxBern": model_subCat0_FEWZxBern, 
                                    # "subCat1_FEWZxBern": model_subCat1_FEWZxBern,
                                    # "subCat2_FEWZxBern": model_subCat2_FEWZxBern,
                                    "subCat3_FEWZxBern": model_subCat3_FEWZxBern,
                                    # "subCat4_FEWZxBern": model_subCat4_FEWZxBern,
                                }, 
                                sample,
    )
    # ---------------------------------------------------
    # Perform a simultaneous fit
    # ---------------------------------------------------
     
    start = time.time()

    _ = simPdf.fitTo(combData, rt.RooFit.Range(fit_range), EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0,SumW2Error=True)
    fitResult = simPdf.fitTo(combData, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,SumW2Error=True)
    end = time.time()
    
    fitResult.Print()
    
    

    # ----------------------------------------------------------------
    # subcat 4 specific fit
    # ----------------------------------------------------------------
    
    # Define category to distinguish physics and control samples events
    sample = rt.RooCategory("sample", "sample")
    # sample.defineType("subCat0_BWZRedux")
    # sample.defineType("subCat1_BWZRedux")
    # sample.defineType("subCat2_BWZRedux")
    # sample.defineType("subCat3_BWZRedux")
    sample.defineType("subCat4_BWZRedux")
    # sample.defineType("subCat0_sumExp")
    # sample.defineType("subCat1_sumExp")
    # sample.defineType("subCat2_sumExp")
    # sample.defineType("subCat3_sumExp")
    sample.defineType("subCat4_sumExp")
    # sample.defineType("subCat0_FEWZxBern")
    # sample.defineType("subCat1_FEWZxBern")
    # sample.defineType("subCat2_FEWZxBern")
    # sample.defineType("subCat3_FEWZxBern")
    sample.defineType("subCat4_FEWZxBern")
     
    # Construct combined dataset in (x,sample)
    combData = rt.RooDataSet(
        "combData",
        "combined data",
        {mass},
        Index=sample,
        Import={
            # "subCat0_BWZRedux": data_subCat0_BWZRedux, 
            # "subCat1_BWZRedux": data_subCat1_BWZRedux,
            # "subCat2_BWZRedux": data_subCat2_BWZRedux,
            # "subCat3_BWZRedux": data_subCat3_BWZRedux,
            "subCat4_BWZRedux": data_subCat4_BWZRedux,
            # "subCat0_sumExp": data_subCat0_sumExp, 
            # "subCat1_sumExp": data_subCat1_sumExp,
            # "subCat2_sumExp": data_subCat2_sumExp,
            # "subCat3_sumExp": data_subCat3_sumExp,
            "subCat4_sumExp": data_subCat4_sumExp,
            # "subCat0_FEWZxBern": data_subCat0_FEWZxBern, 
            # "subCat1_FEWZxBern": data_subCat1_FEWZxBern,
            # "subCat2_FEWZxBern": data_subCat2_FEWZxBern,
            # "subCat3_FEWZxBern": data_subCat3_FEWZxBern,
            "subCat4_FEWZxBern": data_subCat4_FEWZxBern,
        },
    )
    # ---------------------------------------------------
    # Construct a simultaneous pdf in (x, sample)
    # -----------------------------------------------------------------------------------
     
    simPdf = rt.RooSimultaneous(
                                "simPdf", 
                                "simultaneous pdf", 
                                {
                                    # "subCat0_BWZRedux": model_subCat0_BWZRedux, 
                                    # "subCat1_BWZRedux": model_subCat1_BWZRedux,
                                    # "subCat2_BWZRedux": model_subCat2_BWZRedux,
                                    # "subCat3_BWZRedux": model_subCat3_BWZRedux,
                                    "subCat4_BWZRedux": model_subCat4_BWZRedux,
                                    # "subCat0_sumExp": model_subCat0_sumExp, 
                                    # "subCat1_sumExp": model_subCat1_sumExp,
                                    # "subCat2_sumExp": model_subCat2_sumExp,
                                    # "subCat3_sumExp": model_subCat3_sumExp,
                                    "subCat4_sumExp": model_subCat4_sumExp,
                                    # "subCat0_FEWZxBern": model_subCat0_FEWZxBern, 
                                    # "subCat1_FEWZxBern": model_subCat1_FEWZxBern,
                                    # "subCat2_FEWZxBern": model_subCat2_FEWZxBern,
                                    # "subCat3_FEWZxBern": model_subCat3_FEWZxBern,
                                    "subCat4_FEWZxBern": model_subCat4_FEWZxBern,
                                }, 
                                sample,
    )
    # ---------------------------------------------------
    # Perform a simultaneous fit
    # ---------------------------------------------------
     
    start = time.time()

    _ = simPdf.fitTo(combData, rt.RooFit.Range(fit_range), EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0,SumW2Error=True)
    fitResult = simPdf.fitTo(combData, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,SumW2Error=True)
    end = time.time()
    
    fitResult.Print()


    
    # BWZ redux
    # a_coeff.setConstant(False)
    # b_coeff.setConstant(False)
    # c_coeff.setConstant(False)
    # ucsd_workspace.obj("bwzr_cat_ggh_coef1").setConstant(False)
    # ucsd_workspace.obj("bwzr_cat_ggh_coef2").setConstant(False)
    # ucsd_workspace.obj("bwzr_cat_ggh_coef3").setConstant(False)
    setParamsConstant(coreBWZRedux_SubCat0, mass, flag=False)
    
    # sumExp
    # a1_coeff.setConstant(False)
    # a2_coeff.setConstant(False)
    # f_coeff.setConstant(False)
    # ucsd_workspace.obj("exp_order2_cat_ggh_coef1").setConstant(False)
    # ucsd_workspace.obj("exp_order2_cat_ggh_coef2").setConstant(False)
    # ucsd_workspace.obj("exp_order2_cat_ggh_frac1").setConstant(False)
    setParamsConstant(coreSumExp_SubCat0, mass, flag=False)


    # FEWZxBern
    # c1.setConstant(False)
    # c2.setConstant(False)
    # c3.setConstant(False)
    setParamsConstant(coreFEWZxBern_SubCat0, mass, flag=False)
    
    print(f"runtime: {end-start} seconds")

    
    # ---------------------------------------------------
    # Copy SMF variables separately for each core function
    # ---------------------------------------------------

    # sumExp
    
    # sumexp subcat 0
    a0_subCat0_sumExp = rt.RooRealVar("a0_subCat0_sumExp", "a0_subCat0_sumExp", a0_subCat0.getVal(), a0_subCat0.getMin(), a0_subCat0.getMax())
    a1_subCat0_sumExp = rt.RooRealVar("a1_subCat0_sumExp", "a1_subCat0_sumExp", a1_subCat0.getVal(), a1_subCat0.getMin(), a1_subCat0.getMax())
    a2_subCat0_sumExp = rt.RooRealVar("a2_subCat0_sumExp", "a2_subCat0_sumExp", a2_subCat0.getVal(), a2_subCat0.getMin(), a2_subCat0.getMax())
    
     
    name = "subCat0_SMF_sumExp"
    subCat0_SumExp_SMF = rt.RooChebychev(name, name, mass, [a0_subCat0_sumExp, a1_subCat0_sumExp, a2_subCat0_sumExp]) 

    
    # Construct composite pdf
    name = "model_SubCat0_SMFxSumExp"
    model_subCat0_sumExp = rt.RooProdPdf(name, name, [coreSumExp_SubCat0, subCat0_SumExp_SMF])
    
    # subCat 1
    name = "subCat1_sumExp"
    
    a0_subCat1_sumExp = rt.RooRealVar("a0_subCat1_sumExp", "a0_subCat1_sumExp", a0_subCat1.getVal(), a0_subCat1.getMin(), a0_subCat1.getMax())
    a1_subCat1_sumExp = rt.RooRealVar("a1_subCat1_sumExp", "a1_subCat1_sumExp", a1_subCat1.getVal(), a1_subCat1.getMin(), a1_subCat1.getMax())
    
    name = "subCat1_SMF_sumExp"
    subCat1_SumExp_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat1_sumExp, 
                              a1_subCat1_sumExp, 
                             ])
     
    # Construct the composite model
    name = "model_SubCat1_SMFxSumExp"
    model_subCat1_sumExp = rt.RooProdPdf(name, name, [coreSumExp_SubCat1, subCat1_SumExp_SMF])

    # subCat 2
    name = "subCat2_sumExp"

    a0_subCat2_sumExp = rt.RooRealVar("a0_subCat2_sumExp", "a0_subCat2_sumExp", a0_subCat2.getVal(), a0_subCat2.getMin(), a0_subCat2.getMax())
    a1_subCat2_sumExp = rt.RooRealVar("a1_subCat2_sumExp", "a1_subCat2_sumExp", a1_subCat2.getVal(), a1_subCat2.getMin(), a1_subCat2.getMax())
    
    name = "subCat2_SMF_sumExp"
    subCat2_SumExp_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat2_sumExp, 
                              a1_subCat2_sumExp, 
                             ])
    name = "model_SubCat2_SMFxSumExp"
    model_subCat2_sumExp = rt.RooProdPdf(name, name, [coreSumExp_SubCat2, subCat2_SumExp_SMF])  

    # subCat 3
    name = "subCat3_sumExp"

    a0_subCat3_sumExp = rt.RooRealVar("a0_subCat3_sumExp", "a0_subCat3_sumExp", a0_subCat3.getVal(), a0_subCat3.getMin(), a0_subCat3.getMax())
    a1_subCat3_sumExp = rt.RooRealVar("a1_subCat3_sumExp", "a1_subCat3_sumExp", a1_subCat3.getVal(), a1_subCat3.getMin(), a1_subCat3.getMax())
    
    name = "subCat3_SMF_sumExp"
    subCat3_SumExp_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat3_sumExp, 
                              a1_subCat3_sumExp, 
                             ])
    name = "model_SubCat3_SMFxSumExp"
    model_subCat3_sumExp = rt.RooProdPdf(name, name, [coreSumExp_SubCat3, subCat3_SumExp_SMF])    

    # subCat 4
    name = "subCat4_sumExp"

    a0_subCat4_sumExp = rt.RooRealVar("a0_subCat4_sumExp", "a0_subCat4_sumExp", a0_subCat4.getVal(), a0_subCat4.getMin(), a0_subCat4.getMax())
    a1_subCat4_sumExp = rt.RooRealVar("a1_subCat4_sumExp", "a1_subCat4_sumExp", a1_subCat4.getVal(), a1_subCat4.getMin(), a1_subCat4.getMax())
    
    name = "subCat4_SMF_sumExp"
    subCat4_SumExp_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat4_sumExp, 
                              a1_subCat4_sumExp, 
                             ])
    name = "model_SubCat4_SMFxSumExp"
    model_subCat4_sumExp = rt.RooProdPdf(name, name, [coreSumExp_SubCat4, subCat4_SumExp_SMF])

    # FEWZxBern----------------------------------------------------------
    # subCat 0
    a0_subCat0_FEWZxBern = rt.RooRealVar("a0_subCat0_FEWZxBern", "a0_subCat0_FEWZxBern", a0_subCat0.getVal(), a0_subCat0.getMin(), a0_subCat0.getMax())
    a1_subCat0_FEWZxBern = rt.RooRealVar("a1_subCat0_FEWZxBern", "a1_subCat0_FEWZxBern", a1_subCat0.getVal(), a1_subCat0.getMin(), a1_subCat0.getMax())
    a2_subCat0_FEWZxBern = rt.RooRealVar("a2_subCat0_FEWZxBern", "a2_subCat0_FEWZxBern", a2_subCat0.getVal(), a2_subCat0.getMin(), a2_subCat0.getMax())
     
    name = "subCat0_SMF_FEWZxBern"
    subCat0_FEWZxBern_SMF = rt.RooChebychev(name, name, mass, [a0_subCat0_FEWZxBern, a1_subCat0_FEWZxBern, a2_subCat0_FEWZxBern])


    
    # Construct composite pdf
    name = "model_SubCat0_SMFxFEWZxBern"
    model_subCat0_FEWZxBern = rt.RooProdPdf(name, name, [coreFEWZxBern_SubCat0, subCat0_FEWZxBern_SMF])
     
    # subCat 1
    name = "subCat1_FEWZxBern"
    coreFEWZxBern_SubCat1 = coreFEWZxBern_SubCat0
    
    a0_subCat1_FEWZxBern = rt.RooRealVar("a0_subCat1_FEWZxBern", "a0_subCat1_FEWZxBern", a0_subCat1.getVal(), a0_subCat1.getMin(), a0_subCat1.getMax())
    a1_subCat1_FEWZxBern = rt.RooRealVar("a1_subCat1_FEWZxBern", "a1_subCat1_FEWZxBern", a1_subCat1.getVal(), a1_subCat1.getMin(), a1_subCat1.getMax())

    name = "subCat1_SMF_FEWZxBern"
    subCat1_FEWZxBern_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat1_FEWZxBern, 
                              a1_subCat1_FEWZxBern, 
                             ])
     
    # Construct the composite model
    name = "model_SubCat1_SMFxFEWZxBern"
    model_subCat1_FEWZxBern = rt.RooProdPdf(name, name, [coreFEWZxBern_SubCat1, subCat1_FEWZxBern_SMF])

    # subCat 2
    name = "subCat2_FEWZxBern"
    coreFEWZxBern_SubCat2 = coreFEWZxBern_SubCat0
    
    a0_subCat2_FEWZxBern = rt.RooRealVar("a0_subCat2_FEWZxBern", "a0_subCat2_FEWZxBern", a0_subCat2.getVal(), a0_subCat2.getMin(), a0_subCat2.getMax())
    a1_subCat2_FEWZxBern = rt.RooRealVar("a1_subCat2_FEWZxBern", "a1_subCat2_FEWZxBern", a1_subCat2.getVal(), a1_subCat2.getMin(), a1_subCat2.getMax())

    name = "subCat2_SMF_FEWZxBern"
    subCat2_FEWZxBern_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat2_FEWZxBern, 
                              a1_subCat2_FEWZxBern, 
                             ])
    name = "model_SubCat2_SMFxFEWZxBern"
    model_subCat2_FEWZxBern = rt.RooProdPdf(name, name, [coreFEWZxBern_SubCat2, subCat2_FEWZxBern_SMF])    

    # subCat 3
    name = "subCat3_FEWZxBern"
    coreFEWZxBern_SubCat3 = coreFEWZxBern_SubCat0
    
    a0_subCat3_FEWZxBern = rt.RooRealVar("a0_subCat3_FEWZxBern", "a0_subCat3_FEWZxBern", a0_subCat3.getVal(), a0_subCat3.getMin(), a0_subCat3.getMax())
    a1_subCat3_FEWZxBern = rt.RooRealVar("a1_subCat3_FEWZxBern", "a1_subCat3_FEWZxBern", a1_subCat3.getVal(), a1_subCat3.getMin(), a1_subCat3.getMax())

    name = "subCat3_SMF_FEWZxBern"
    subCat3_FEWZxBern_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat3_FEWZxBern, 
                              a1_subCat3_FEWZxBern, 
                             ])
    name = "model_SubCat3_SMFxFEWZxBern"
    model_subCat3_FEWZxBern = rt.RooProdPdf(name, name, [coreFEWZxBern_SubCat3, subCat3_FEWZxBern_SMF])    

    # subCat 4
    name = "subCat4_FEWZxBern"
    coreFEWZxBern_SubCat4 = coreFEWZxBern_SubCat0
    
    a0_subCat4_FEWZxBern = rt.RooRealVar("a0_subCat4_FEWZxBern", "a0_subCat4_FEWZxBern", a0_subCat4.getVal(), a0_subCat4.getMin(), a0_subCat4.getMax())
    a1_subCat4_FEWZxBern = rt.RooRealVar("a1_subCat4_FEWZxBern", "a1_subCat4_FEWZxBern", a1_subCat4.getVal(), a1_subCat4.getMin(), a1_subCat4.getMax())

    name = "subCat4_SMF_FEWZxBern"
    subCat4_FEWZxBern_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat4_FEWZxBern, 
                              a1_subCat4_FEWZxBern, 
                             ])
    name = "model_SubCat4_SMFxFEWZxBern"
    model_subCat4_FEWZxBern = rt.RooProdPdf(name, name, [coreFEWZxBern_SubCat4, subCat4_FEWZxBern_SMF]) 

    
    # ---------------------------------------------------
    # Make CORE-PDF
    # ---------------------------------------------------

    # subCat 0 
    cat_subCat0 = rt.RooCategory("pdf_index_ggh","Index of Pdf which is active"); # name of category index should stay same across subCategories
    
    # // Make a RooMultiPdf object. The order of the pdfs will be the order of their index, ie for below
    # // 0 == BWZ_Redux
    # // 1 == sumExp
    # // 2 == FEWZxBern
    
    # FEWZxBern Sumexp is less dependent to dimuon mass as stated in line 1585 of RERECO AN
    # I suppose BWZredux is there bc it's the one function with overall least bias (which is why BWZredux is used if CORE-PDF is not used)
    pdf_list_subCat0 = rt.RooArgList(
        model_subCat0_BWZRedux,
        model_subCat0_sumExp,
        model_subCat0_FEWZxBern,
    )
    corePdf_subCat0 = rt.RooMultiPdf("CorePdf_subCat0","CorePdf_subCat0",cat_subCat0,pdf_list_subCat0)
    penalty = 0 # as told in https://cms-talk.web.cern.ch/t/combine-fitting-not-working-with-roomultipdf-leading-to-bad-signal-significance/44238/
    corePdf_subCat0.setCorrectionFactor(penalty) 
    nevents = roo_datasetData_subCat0.sumEntries() # these are data, so all weights are one, thus no need to sum over the weights, though ofc you can just do that too
    print(f"roo_datasetData_subCat0 sumentries: {nevents}")
    bkg_subCat0_norm = rt.RooRealVar(corePdf_subCat0.GetName()+"_norm","Background normalization value",nevents,0,3*nevents) # free floating value
    
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat0"],
        "dataset": ["data"], 
        "yield": [nevents]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)
    


    # subCat 1 
    cat_subCat1 = rt.RooCategory("pdf_index_ggh","Index of Pdf which is active"); # name of category index should stay same across subCategories
    
    # // Make a RooMultiPdf object. The order of the pdfs will be the order of their index, ie for below
    # // 0 == BWZ_Redux
    # // 1 == sumExp
    # // 2 == PowerSum
    
    # FEWZxBern Sumexp is less dependent to dimuon mass as stated in line 1585 of RERECO AN
    # I suppose BWZredux is there bc it's the one function with overall least bias (which is why BWZredux is used if CORE-PDF is not used)
    pdf_list_subCat1 = rt.RooArgList(
        model_subCat1_BWZRedux,
        model_subCat1_sumExp,
        model_subCat1_FEWZxBern,
    )
    corePdf_subCat1 = rt.RooMultiPdf("CorePdf_subCat1","CorePdf_subCat1",cat_subCat1,pdf_list_subCat1)
    penalty = 0 # as told in https://cms-talk.web.cern.ch/t/combine-fitting-not-working-with-roomultipdf-leading-to-bad-signal-significance/44238/
    corePdf_subCat1.setCorrectionFactor(penalty) 
    nevents = roo_datasetData_subCat1.sumEntries() # these are data, so all weights are one, thus no need to sum over the weights, though ofc you can just do that too
    print(f"roo_datasetData_subCat1 sumentries: {nevents}")
    bkg_subCat1_norm = rt.RooRealVar(corePdf_subCat1.GetName()+"_norm","Background normalization value",nevents,0,3*nevents) # free floating value
    
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat1"],
        "dataset": ["data"], 
        "yield": [nevents]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)
    
    
    # subCat 2 
    cat_subCat2 = rt.RooCategory("pdf_index_ggh","Index of Pdf which is active"); # name of category index should stay same across subCategories
    
    # // Make a RooMultiPdf object. The order of the pdfs will be the order of their index, ie for below
    # // 0 == BWZ_Redux
    # // 1 == sumExp
    # // 2 == PowerSum
    
    # FEWZxBern Sumexp is less dependent to dimuon mass as stated in line 1585 of RERECO AN
    # I suppose BWZredux is there bc it's the one function with overall least bias (which is why BWZredux is used if CORE-PDF is not used)
    pdf_list_subCat2 = rt.RooArgList(
        model_subCat2_BWZRedux,
        model_subCat2_sumExp,
        model_subCat2_FEWZxBern,
    )
    corePdf_subCat2 = rt.RooMultiPdf("CorePdf_subCat2","CorePdf_subCat2",cat_subCat2,pdf_list_subCat2)
    penalty = 0 # as told in https://cms-talk.web.cern.ch/t/combine-fitting-not-working-with-roomultipdf-leading-to-bad-signal-significance/44238/
    corePdf_subCat2.setCorrectionFactor(penalty) 
    nevents = roo_datasetData_subCat2.sumEntries() # these are data, so all weights are one, thus no need to sum over the weights, though ofc you can just do that too
    print(f"roo_datasetData_subCat2 sumentries: {nevents}")
    bkg_subCat2_norm = rt.RooRealVar(corePdf_subCat2.GetName()+"_norm","Background normalization value",nevents,0,3*nevents) # free floating value
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat2"],
        "dataset": ["data"], 
        "yield": [nevents]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)
    
        
    # subCat 3 
    cat_subCat3 = rt.RooCategory("pdf_index_ggh","Index of Pdf which is active"); # name of category index should stay same across subCategories
    
    # // Make a RooMultiPdf object. The order of the pdfs will be the order of their index, ie for below
    # // 0 == BWZ_Redux
    # // 1 == sumExp
    # // 2 == PowerSum
    
    # FEWZxBern Sumexp is less dependent to dimuon mass as stated in line 1585 of RERECO AN
    # I suppose BWZredux is there bc it's the one function with overall least bias (which is why BWZredux is used if CORE-PDF is not used)
    pdf_list_subCat3 = rt.RooArgList(
        model_subCat3_BWZRedux,
        model_subCat3_sumExp,
        model_subCat3_FEWZxBern,
    )
    corePdf_subCat3 = rt.RooMultiPdf("CorePdf_subCat3","CorePdf_subCat3",cat_subCat3,pdf_list_subCat3)
    penalty = 0 # as told in https://cms-talk.web.cern.ch/t/combine-fitting-not-working-with-roomultipdf-leading-to-bad-signal-significance/44238/
    corePdf_subCat3.setCorrectionFactor(penalty) 
    nevents = roo_datasetData_subCat3.sumEntries() # these are data, so all weights are one, thus no need to sum over the weights, though ofc you can just do that too
    print(f"roo_datasetData_subCat3 sumentries: {nevents}")
    bkg_subCat3_norm = rt.RooRealVar(corePdf_subCat3.GetName()+"_norm","Background normalization value",nevents,0,3*nevents) # free floating value
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat3"],
        "dataset": ["data"], 
        "yield": [nevents]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)
    

    # subCat 4
    cat_subCat4 = rt.RooCategory("pdf_index_ggh","Index of Pdf which is active"); # name of category index should stay same across subCategories
    
    # // Make a RooMultiPdf object. The order of the pdfs will be the order of their index, ie for below
    # // 0 == BWZ_Redux
    # // 1 == sumExp
    # // 2 == PowerSum
    
    # FEWZxBern Sumexp is less dependent to dimuon mass as stated in line 1585 of RERECO AN
    # I suppose BWZredux is there bc it's the one function with overall least bias (which is why BWZredux is used if CORE-PDF is not used)
    pdf_list_subCat4 = rt.RooArgList(
        model_subCat4_BWZRedux,
        model_subCat4_sumExp,
        model_subCat4_FEWZxBern,
    )
    # pdf_list_subCat4 = rt.RooArgList( # FastScan debugging
    #     model_subCat4_FEWZxBern,
    #     model_subCat4_sumExp,
    #     model_subCat4_BWZRedux,
    # )
    corePdf_subCat4 = rt.RooMultiPdf("CorePdf_subCat4","CorePdf_subCat4",cat_subCat4,pdf_list_subCat4)
    penalty = 0 # as told in https://cms-talk.web.cern.ch/t/combine-fitting-not-working-with-roomultipdf-leading-to-bad-signal-significance/44238/
    corePdf_subCat4.setCorrectionFactor(penalty) 
    nevents = roo_datasetData_subCat4.sumEntries() # these are data, so all weights are one, thus no need to sum over the weights, though ofc you can just do that too
    print(f"roo_datasetData_subCat4 sumentries: {nevents}")
    bkg_subCat4_norm = rt.RooRealVar(corePdf_subCat4.GetName()+"_norm","Background normalization value",nevents,0,3*nevents) # free floating value
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat4"],
        "dataset": ["data"], 
        "yield": [nevents]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)
    
    print(f"yield_df after Data: \n {yield_df}")

    

    # ---------------------------------------------------
    # Obtain signal MC events
    # ---------------------------------------------------

    # load_path = f"{args.load_path}/{category}/{args.year}/processed_events_signalMC.parquet"
    if args.year=="all":
        load_path = f"{args.load_path}/{category}/*/processed_events_sigMC_ggh.parquet"
        # load_path = f"{args.load_path}/{category}/*/processed_events_sigMC_ggh_amcPS.parquet"
    elif args.year=="2016only":
        load_path = f"{args.load_path}/{category}/2016*/processed_events_sigMC_ggh.parquet"
    else:
        load_path = f"{args.load_path}/{category}/{args.year}/processed_events_sigMC_ggh.parquet" # Fig 6.15 was only with ggH process, though with all 2016, 2017 and 2018
    # load_path = f"{args.load_path}/{category}/{args.year}/processed_events_sigMC*.parquet"
    if args.year=="all":
        load_path = f"{args.load_path}/*/processed_events_sigMC_ggh.parquet"
        # load_path = f"{args.load_path}/{category}/*/processed_events_sigMC_ggh_amcPS.parquet"
    elif args.year=="2016only":
        load_path = f"{args.load_path}/2016*/processed_events_sigMC_ggh.parquet"
    else:
        load_path = f"{args.load_path}/{args.year}/processed_events_sigMC_ggh.parquet"
    processed_eventsSignalMC = ak.from_parquet(load_path)
    print(f"ggH yield: {np.sum(processed_eventsSignalMC.wgt_nominal)}")
    print("signal events loaded")
    
    # ---------------------------------------------------
    # Define signal model's Doubcl Crystal Ball PDF
    # ---------------------------------------------------
    
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
    MH_subCat1 = MH_subCat0 
    
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

    # subCat 2
    # original start ------------------------------------------------------
    # MH_subCat2 = rt.RooRealVar("MH" , "MH", 125, 115,135)
    # MH_subCat2.setConstant(True) # this shouldn't change, I think
    # original end ------------------------------------------------------
    MH_subCat2 = MH_subCat0 
    
    # sigma_subCat2 = rt.RooRealVar("sigma_subCat2" , "sigma_subCat2", 2, .1, 4.0)
    # alpha1_subCat2 = rt.RooRealVar("alpha1_subCat2" , "alpha1_subCat2", 2, 0.01, 65)
    # n1_subCat2 = rt.RooRealVar("n1_subCat2" , "n1_subCat2", 10, 0.01, 100)
    # alpha2_subCat2 = rt.RooRealVar("alpha2_subCat2" , "alpha2_subCat2", 2.0, 0.01, 65)
    # n2_subCat2 = rt.RooRealVar("n2_subCat2" , "n2_subCat2", 25, 0.01, 100)

    # copying parameters from official AN workspace as starting params
    sigma_subCat2 = rt.RooRealVar("sigma_subCat2" , "sigma_subCat2", 1.36025, .1, 4.0)
    alpha1_subCat2 = rt.RooRealVar("alpha1_subCat2" , "alpha1_subCat2", 1.4173626, 0.01, 65)
    n1_subCat2 = rt.RooRealVar("n1_subCat2" , "n1_subCat2", 2.42748, 0.01, 100)
    alpha2_subCat2 = rt.RooRealVar("alpha2_subCat2" , "alpha2_subCat2", 1.629120, 0.01, 65)
    n2_subCat2 = rt.RooRealVar("n2_subCat2" , "n2_subCat2", 9.983334, 0.01, 100)

    # # temporary test
    # sigma_subCat2.setConstant(True)
    # alpha1_subCat2.setConstant(True)
    # n1_subCat2.setConstant(True)
    # alpha2_subCat2.setConstant(True)
    # n2_subCat2.setConstant(True)

    CMS_hmm_sigma_cat2_ggh = rt.RooRealVar("CMS_hmm_sigma_cat2_ggh" , "CMS_hmm_sigma_cat2_ggh", 0, -5 , 5 )
    CMS_hmm_sigma_cat2_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat2_ggh_fsigma = rt.RooFormulaVar("ggH_cat2_ggh_fsigma", "ggH_cat2_ggh_fsigma",'@0*(1+@1)',[sigma_subCat2, CMS_hmm_sigma_cat2_ggh])
    CMS_hmm_peak_cat2_ggh = rt.RooRealVar("CMS_hmm_peak_cat2_ggh" , "CMS_hmm_peak_cat2_ggh", 0, -5 , 5 )
    CMS_hmm_peak_cat2_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat2_ggh_fpeak = rt.RooFormulaVar("ggH_cat2_ggh_fpeak", "ggH_cat2_ggh_fpeak",'@0*(1+@1)',[MH_subCat2, CMS_hmm_peak_cat2_ggh])
    
    # n1_subCat2.setConstant(True) # freeze for stability
    # n2_subCat2.setConstant(True) # freeze for stability
    name = "signal_subCat2"
    signal_subCat2 = rt.RooCrystalBall(name,name,mass, ggH_cat2_ggh_fpeak, ggH_cat2_ggh_fsigma, alpha1_subCat2, n1_subCat2, alpha2_subCat2, n2_subCat2)

    # subCat 3
    # original start ------------------------------------------------------
    # MH_subCat3 = rt.RooRealVar("MH" , "MH", 125, 115,135)
    # MH_subCat3.setConstant(True) # this shouldn't change, I think
    # original end ------------------------------------------------------
    MH_subCat3 = MH_subCat0
    

    sigma_subCat3 = rt.RooRealVar("sigma_subCat3" , "sigma_subCat3", 0.1, .1, 10.0)
    alpha1_subCat3 = rt.RooRealVar("alpha1_subCat3" , "alpha1_subCat3", 2, 0.01, 200)
    n1_subCat3 = rt.RooRealVar("n1_subCat3" , "n1_subCat3", 25, 0.01, 200)
    alpha2_subCat3 = rt.RooRealVar("alpha2_subCat3" , "alpha2_subCat3", 2, 0.01, 65)
    n2_subCat3 = rt.RooRealVar("n2_subCat3" , "n2_subCat3", 25, 0.01, 200)

    # # copying parameters from official AN workspace as starting params
    # sigma_subCat3 = rt.RooRealVar("sigma_subCat3" , "sigma_subCat3", 1.25359, .1, 10.0)
    # alpha1_subCat3 = rt.RooRealVar("alpha1_subCat3" , "alpha1_subCat3", 1.4199, 0.01, 200)
    # n1_subCat3 = rt.RooRealVar("n1_subCat3" , "n1_subCat3", 2.409953, 0.01, 200)
    # alpha2_subCat3 = rt.RooRealVar("alpha2_subCat3" , "alpha2_subCat3", 1.64675, 0.01, 65)
    # n2_subCat3 = rt.RooRealVar("n2_subCat3" , "n2_subCat3", 9.670221, 0.01, 200)

    # # temporary test
    # sigma_subCat3.setConstant(True)
    # alpha1_subCat3.setConstant(True)
    # n1_subCat3.setConstant(True)
    # alpha2_subCat3.setConstant(True)
    # n2_subCat3.setConstant(True)

    CMS_hmm_sigma_cat3_ggh = rt.RooRealVar("CMS_hmm_sigma_cat3_ggh" , "CMS_hmm_sigma_cat3_ggh", 0, -5 , 5 )
    CMS_hmm_sigma_cat3_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat3_ggh_fsigma = rt.RooFormulaVar("ggH_cat3_ggh_fsigma", "ggH_cat3_ggh_fsigma",'@0*(1+@1)',[sigma_subCat3, CMS_hmm_sigma_cat3_ggh])
    CMS_hmm_peak_cat3_ggh = rt.RooRealVar("CMS_hmm_peak_cat3_ggh" , "CMS_hmm_peak_cat3_ggh", 0, -5 , 5 )
    CMS_hmm_peak_cat3_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat3_ggh_fpeak = rt.RooFormulaVar("ggH_cat3_ggh_fpeak", "ggH_cat3_ggh_fpeak",'@0*(1+@1)',[MH_subCat3, CMS_hmm_peak_cat3_ggh])
    
    # n1_subCat3.setConstant(True) # freeze for stability
    # n2_subCat3.setConstant(True) # freeze for stability
    name = "signal_subCat3"
    signal_subCat3 = rt.RooCrystalBall(name,name,mass, ggH_cat3_ggh_fpeak, ggH_cat3_ggh_fsigma, alpha1_subCat3, n1_subCat3, alpha2_subCat3, n2_subCat3)

    # subCat 4
    # original start ------------------------------------------------------
    # MH_subCat4 = rt.RooRealVar("MH" , "MH", 125, 115,135)
    # MH_subCat4.setConstant(True) # this shouldn't change, I think
    # original end ------------------------------------------------------
    MH_subCat4 = MH_subCat0
    
    # sigma_subCat4 = rt.RooRealVar("sigma_subCat4" , "sigma_subCat4", 2, .1, 4.0)
    # alpha1_subCat4 = rt.RooRealVar("alpha1_subCat4" , "alpha1_subCat4", 2, 0.01, 65)
    # n1_subCat4 = rt.RooRealVar("n1_subCat4" , "n1_subCat4", 10, 0.01, 100)
    # alpha2_subCat4 = rt.RooRealVar("alpha2_subCat4" , "alpha2_subCat4", 2.0, 0.01, 65)
    # n2_subCat4 = rt.RooRealVar("n2_subCat4" , "n2_subCat4", 25, 0.01, 100)

    # copying parameters from official AN workspace as starting params
    sigma_subCat4 = rt.RooRealVar("sigma_subCat4" , "sigma_subCat4", 1.28250, .1, 4.0)
    alpha1_subCat4 = rt.RooRealVar("alpha1_subCat4" , "alpha1_subCat4", 1.47936, 0.01, 65)
    n1_subCat4 = rt.RooRealVar("n1_subCat4" , "n1_subCat4", 2.24104, 0.01, 100)
    alpha2_subCat4 = rt.RooRealVar("alpha2_subCat4" , "alpha2_subCat4", 1.67898, 0.01, 65)
    n2_subCat4 = rt.RooRealVar("n2_subCat4" , "n2_subCat4", 8.8719, 0.01, 100)


    CMS_hmm_sigma_cat4_ggh = rt.RooRealVar("CMS_hmm_sigma_cat4_ggh" , "CMS_hmm_sigma_cat4_ggh", 0, -5 , 5 )
    CMS_hmm_sigma_cat4_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat4_ggh_fsigma = rt.RooFormulaVar("ggH_cat4_ggh_fsigma", "ggH_cat4_ggh_fsigma",'@0*(1+@1)',[sigma_subCat4, CMS_hmm_sigma_cat4_ggh])
    CMS_hmm_peak_cat4_ggh = rt.RooRealVar("CMS_hmm_peak_cat4_ggh" , "CMS_hmm_peak_cat4_ggh", 0, -5 , 5 )
    CMS_hmm_peak_cat4_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat4_ggh_fpeak = rt.RooFormulaVar("ggH_cat4_ggh_fpeak", "ggH_cat4_ggh_fpeak",'@0*(1+@1)',[MH_subCat4, CMS_hmm_peak_cat4_ggh])
    
    # n1_subCat4.setConstant(True) # freeze for stability
    # n2_subCat4.setConstant(True) # freeze for stability
    name = "signal_subCat4"
    signal_subCat4 = rt.RooCrystalBall(name,name,mass, ggH_cat4_ggh_fpeak, ggH_cat4_ggh_fsigma, alpha1_subCat4, n1_subCat4, alpha2_subCat4, n2_subCat4)
    
    
    # ---------------------------------------------------
    # Define signal MC samples to fit to for ggH
    # ---------------------------------------------------

    # subCat 0
    subCat_filter = (processed_eventsSignalMC[subCatIdx_name] == 0)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat0_SigMC = ak.to_numpy(
        processed_eventsSignalMC.wgt_nominal[subCat_filter]
    ) # weights

    # generate a weighted histogram 
    roo_histData_subCat0_signal = rt.TH1F("subCat0_rooHist_signal", "subCat0_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat0_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat0_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat0_signal = rt.RooDataHist("subCat0_rooHist_signal", "subCat0_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat0_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat0_signal = roo_histData_subCat0_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat0"],
        "dataset": ["ggH"], 
        "yield": [data_subCat0_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)

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
        processed_eventsSignalMC.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat1_signal = rt.TH1F("subCat1_rooHist_signal", "subCat1_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat1_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat1_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat1_signal = rt.RooDataHist("subCat1_rooHist_signal", "subCat1_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat1_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat1_signal = roo_histData_subCat1_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat1"],
        "dataset": ["ggH"], 
        "yield": [data_subCat1_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)
    

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat1_SigMC)* flat_MC_SF
    # norm_val = 295.214 # quick test
    sig_norm_subCat1 = rt.RooRealVar(signal_subCat1.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat1 norm_val: {norm_val}")
    sig_norm_subCat1.setConstant(True)

    # subCat 2
    subCat_filter = (processed_eventsSignalMC[subCatIdx_name] == 2)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat2_SigMC = ak.to_numpy(
        processed_eventsSignalMC.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat2_signal = rt.TH1F("subCat2_rooHist_signal", "subCat2_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat2_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat2_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat2_signal = rt.RooDataHist("subCat2_rooHist_signal", "subCat2_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat2_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat2_signal = roo_histData_subCat2_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat2"],
        "dataset": ["ggH"], 
        "yield": [data_subCat2_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat2_SigMC) * flat_MC_SF
    # norm_val = 124.0364 # quick test
    sig_norm_subCat2 = rt.RooRealVar(signal_subCat2.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat2 norm_val: {norm_val}")
    sig_norm_subCat2.setConstant(True)

    # subCat 3
    subCat_filter = (processed_eventsSignalMC[subCatIdx_name] == 3)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat3_SigMC = ak.to_numpy(
        processed_eventsSignalMC.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat3_signal = rt.TH1F("subCat3_rooHist_signal", "subCat3_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat3_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat3_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat3_signal = rt.RooDataHist("subCat3_rooHist_signal", "subCat3_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat3_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat3_signal = roo_histData_subCat3_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat3"],
        "dataset": ["ggH"], 
        "yield": [data_subCat3_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat3_SigMC)* flat_MC_SF
    # norm_val = 116.4918 # quick test
    sig_norm_subCat3 = rt.RooRealVar(signal_subCat3.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat3 norm_val: {norm_val}")
    sig_norm_subCat3.setConstant(True)
    
    # subCat 4
    subCat_filter = (processed_eventsSignalMC[subCatIdx_name] == 4)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat4_SigMC = ak.to_numpy(
        processed_eventsSignalMC.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat4_signal = rt.TH1F("subCat4_rooHist_signal", "subCat4_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat4_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat4_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat4_signal = rt.RooDataHist("subCat4_rooHist_signal", "subCat4_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat4_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat4_signal = roo_histData_subCat4_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat4"],
        "dataset": ["ggH"], 
        "yield": [data_subCat4_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)
    print(f"yield_df after ggH: {yield_df}")

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat4_SigMC)* flat_MC_SF
    # norm_val = 45.423052 # quick test
    sig_norm_subCat4 = rt.RooRealVar(signal_subCat4.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat4 norm_val: {norm_val}")
    sig_norm_subCat4.setConstant(True)
    
    # ---------------------------------------------------
    # Fit signal model individually, not simultaneous. Sigma, and left and right tails are different for each category
    # ---------------------------------------------------

    # subCat 0
    # _ = signal_subCat0.fitTo(data_subCat0_signal,  EvalBackend=device, Save=True,)
    # fit_result = signal_subCat0.fitTo(data_subCat0_signal,  EvalBackend=device, Save=True,)
    _ = signal_subCat0.fitTo(data_subCat0_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat0.fitTo(data_subCat0_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat0.setConstant(True)
    alpha1_subCat0.setConstant(True)
    n1_subCat0.setConstant(True)
    alpha2_subCat0.setConstant(True)
    n2_subCat0.setConstant(True)

    

    # subCat 1
    # _ = signal_subCat1.fitTo(data_subCat1_signal,  EvalBackend=device, Save=True, )
    # fit_result = signal_subCat1.fitTo(data_subCat1_signal,  EvalBackend=device, Save=True, )
    _ = signal_subCat1.fitTo(data_subCat1_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat1.fitTo(data_subCat1_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat1.setConstant(True)
    alpha1_subCat1.setConstant(True)
    n1_subCat1.setConstant(True)
    alpha2_subCat1.setConstant(True)
    n2_subCat1.setConstant(True)

    

    # subCat 2
    # _ = signal_subCat2.fitTo(data_subCat2_signal,  EvalBackend=device, Save=True, )
    # fit_result = signal_subCat2.fitTo(data_subCat2_signal,  EvalBackend=device, Save=True, )
    _ = signal_subCat2.fitTo(data_subCat2_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat2.fitTo(data_subCat2_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat2.setConstant(True)
    alpha1_subCat2.setConstant(True)
    n1_subCat2.setConstant(True)
    alpha2_subCat2.setConstant(True)
    n2_subCat2.setConstant(True)

    
    
    # subCat 3
    # _ = signal_subCat3.fitTo(data_subCat3_signal,  EvalBackend=device, Save=True, )
    # fit_result = signal_subCat3.fitTo(data_subCat3_signal,  EvalBackend=device, Save=True, )
    _ = signal_subCat3.fitTo(data_subCat3_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat3.fitTo(data_subCat3_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat3.setConstant(True)
    alpha1_subCat3.setConstant(True)
    n1_subCat3.setConstant(True)
    alpha2_subCat3.setConstant(True)
    n2_subCat3.setConstant(True)
    # sigma_subCat3.setConstant(False)
    # alpha1_subCat3.setConstant(False)
    # n1_subCat3.setConstant(False)
    # alpha2_subCat3.setConstant(False)
    # n2_subCat3.setConstant(False)


    # subCat 4
    # _ = signal_subCat4.fitTo(data_subCat4_signal,  EvalBackend=device, Save=True, )
    # fit_result = signal_subCat4.fitTo(data_subCat4_signal,  EvalBackend=device, Save=True, )
    _ = signal_subCat4.fitTo(data_subCat4_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat4.fitTo(data_subCat4_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat4.setConstant(True)
    alpha1_subCat4.setConstant(True)
    n1_subCat4.setConstant(True)
    alpha2_subCat4.setConstant(True)
    n2_subCat4.setConstant(True)

    # ---------------------------------------------------
    # Obtain signal MC events for VBF
    # ---------------------------------------------------

    # load_path = f"{args.load_path}/{category}/{args.year}/processed_events_signalMC.parquet"
    if args.year=="all":
        load_path = f"{args.load_path}/*/processed_events_sigMC_vbf.parquet"
        # load_path = f"{args.load_path}/{category}/*/processed_events_sigMC_qqh_amcPS.parquet"
    elif args.year=="2016only":
        load_path = f"{args.load_path}/2016*/processed_events_sigMC_vbf.parquet"
    else:
        load_path = f"{args.load_path}/{args.year}/processed_events_sigMC_vbf.parquet" # Fig 6.15 was only with qqH process, though with all 2016, 2017 and 2018
    
    processed_eventsSignalMC_vbf = ak.from_parquet(load_path)
    print(f"qqH yield: {np.sum(processed_eventsSignalMC_vbf.wgt_nominal)}")
    print("signal events loaded")
    
    # ---------------------------------------------------
    # Define vbf signal model's Doubcl Crystal Ball PDF
    # ---------------------------------------------------
    
    # subCat 0
    
    sigma_subCat0_vbf = rt.RooRealVar("sigma_subCat0_vbf" , "sigma_subCat0_vbf", 2, .1, 4.0)
    alpha1_subCat0_vbf = rt.RooRealVar("alpha1_subCat0_vbf" , "alpha1_subCat0_vbf", 2, 0.01, 65)
    n1_subCat0_vbf = rt.RooRealVar("n1_subCat0_vbf" , "n1_subCat0_vbf", 10, 0.01, 100)
    alpha2_subCat0_vbf = rt.RooRealVar("alpha2_subCat0_vbf" , "alpha2_subCat0_vbf", 2.0, 0.01, 65)
    n2_subCat0_vbf = rt.RooRealVar("n2_subCat0_vbf" , "n2_subCat0_vbf", 25, 0.01, 100)

    # # temporary test
    # sigma_subCat0_vbf.setConstant(True)
    # alpha1_subCat0_vbf.setConstant(True)
    # n1_subCat0_vbf.setConstant(True)
    # alpha2_subCat0_vbf.setConstant(True)
    # n2_subCat0_vbf.setConstant(True)
    

    qqH_cat0_ggh_fsigma = rt.RooFormulaVar("qqH_cat0_ggh_fsigma", "qqH_cat0_ggh_fsigma",'@0*(1+@1)',[sigma_subCat0_vbf, CMS_hmm_sigma_cat0_ggh])
    qqH_cat0_ggh_fpeak = rt.RooFormulaVar("qqH_cat0_qqh_fpeak", "qqH_cat0_ggh_fpeak",'@0*(1+@1)',[MH_subCat0, CMS_hmm_peak_cat0_ggh])
    
    # n1_subCat0_vbf.setConstant(True) # freeze for stability
    # n2_subCat0_vbf.setConstant(True) # freeze for stability
    name = "signal_subCat0_vbf"
    signal_subCat0_vbf = rt.RooCrystalBall(name,name,mass, qqH_cat0_ggh_fpeak, qqH_cat0_ggh_fsigma, alpha1_subCat0_vbf, n1_subCat0_vbf, alpha2_subCat0_vbf, n2_subCat0_vbf)

    # subCat 1

    
    sigma_subCat1_vbf = rt.RooRealVar("sigma_subCat1_vbf" , "sigma_subCat1_vbf", 2, .1, 4.0)
    alpha1_subCat1_vbf = rt.RooRealVar("alpha1_subCat1_vbf" , "alpha1_subCat1_vbf", 2, 0.01, 65)
    n1_subCat1_vbf = rt.RooRealVar("n1_subCat1_vbf" , "n1_subCat1_vbf", 10, 0.01, 100)
    alpha2_subCat1_vbf = rt.RooRealVar("alpha2_subCat1_vbf" , "alpha2_subCat1_vbf", 2.0, 0.01, 65)
    n2_subCat1_vbf = rt.RooRealVar("n2_subCat1_vbf" , "n2_subCat1_vbf", 25, 0.01, 100)

    # # temporary test
    # sigma_subCat1_vbf.setConstant(True)
    # alpha1_subCat1_vbf.setConstant(True)
    # n1_subCat1_vbf.setConstant(True)
    # alpha2_subCat1_vbf.setConstant(True)
    # n2_subCat1_vbf.setConstant(True)
    
    qqH_cat1_ggh_fsigma = rt.RooFormulaVar("qqH_cat1_ggh_fsigma", "qqH_cat1_ggh_fsigma",'@0*(1+@1)',[sigma_subCat1_vbf, CMS_hmm_sigma_cat1_ggh])
    qqH_cat1_ggh_fpeak = rt.RooFormulaVar("qqH_cat1_ggh_fpeak", "qqH_cat1_ggh_fpeak",'@0*(1+@1)',[MH_subCat1, CMS_hmm_peak_cat1_ggh])
    
    # n1_subCat1_vbf.setConstant(True) # freeze for stability
    # n2_subCat1_vbf.setConstant(True) # freeze for stability
    name = "signal_subCat1_vbf"
    signal_subCat1_vbf = rt.RooCrystalBall(name,name,mass, qqH_cat1_ggh_fpeak, qqH_cat1_ggh_fsigma, alpha1_subCat1_vbf, n1_subCat1_vbf, alpha2_subCat1_vbf, n2_subCat1_vbf)

    # subCat 2
   
    sigma_subCat2_vbf = rt.RooRealVar("sigma_subCat2_vbf" , "sigma_subCat2_vbf", 2, .1, 4.0)
    alpha1_subCat2_vbf = rt.RooRealVar("alpha1_subCat2_vbf" , "alpha1_subCat2_vbf", 2, 0.01, 65)
    n1_subCat2_vbf = rt.RooRealVar("n1_subCat2_vbf" , "n1_subCat2_vbf", 10, 0.01, 100)
    alpha2_subCat2_vbf = rt.RooRealVar("alpha2_subCat2_vbf" , "alpha2_subCat2_vbf", 2.0, 0.01, 65)
    n2_subCat2_vbf = rt.RooRealVar("n2_subCat2_vbf" , "n2_subCat2_vbf", 25, 0.01, 100)

    # # temporary test
    # sigma_subCat2_vbf.setConstant(True)
    # alpha1_subCat2_vbf.setConstant(True)
    # n1_subCat2_vbf.setConstant(True)
    # alpha2_subCat2_vbf.setConstant(True)
    # n2_subCat2_vbf.setConstant(True)

    qqH_cat2_ggh_fsigma = rt.RooFormulaVar("qqH_cat2_ggh_fsigma", "qqH_cat2_ggh_fsigma",'@0*(1+@1)',[sigma_subCat2_vbf, CMS_hmm_sigma_cat2_ggh])
    qqH_cat2_ggh_fpeak = rt.RooFormulaVar("qqH_cat2_ggh_fpeak", "qqH_cat2_ggh_fpeak",'@0*(1+@1)',[MH_subCat2, CMS_hmm_peak_cat2_ggh])
    
    # n1_subCat2_vbf.setConstant(True) # freeze for stability
    # n2_subCat2_vbf.setConstant(True) # freeze for stability
    name = "signal_subCat2_vbf"
    signal_subCat2_vbf = rt.RooCrystalBall(name,name,mass, qqH_cat2_ggh_fpeak, qqH_cat2_ggh_fsigma, alpha1_subCat2_vbf, n1_subCat2_vbf, alpha2_subCat2_vbf, n2_subCat2_vbf)

    # subCat 3

    sigma_subCat3_vbf = rt.RooRealVar("sigma_subCat3_vbf" , "sigma_subCat3_vbf", 0.1, .1, 10.0)
    alpha1_subCat3_vbf = rt.RooRealVar("alpha1_subCat3_vbf" , "alpha1_subCat3_vbf", 2, 0.01, 200)
    n1_subCat3_vbf = rt.RooRealVar("n1_subCat3_vbf" , "n1_subCat3_vbf", 25, 0.01, 200)
    alpha2_subCat3_vbf = rt.RooRealVar("alpha2_subCat3_vbf" , "alpha2_subCat3_vbf", 2, 0.01, 65)
    n2_subCat3_vbf = rt.RooRealVar("n2_subCat3_vbf" , "n2_subCat3_vbf", 25, 0.01, 200)


    # # temporary test
    # sigma_subCat3_vbf.setConstant(True)
    # alpha1_subCat3_vbf.setConstant(True)
    # n1_subCat3_vbf.setConstant(True)
    # alpha2_subCat3_vbf.setConstant(True)
    # n2_subCat3_vbf.setConstant(True)

    qqH_cat3_ggh_fsigma = rt.RooFormulaVar("qqH_cat3_ggh_fsigma", "qqH_cat3_ggh_fsigma",'@0*(1+@1)',[sigma_subCat3_vbf, CMS_hmm_sigma_cat3_ggh])
    qqH_cat3_ggh_fpeak = rt.RooFormulaVar("qqH_cat3_ggh_fpeak", "qqH_cat3_ggh_fpeak",'@0*(1+@1)',[MH_subCat3, CMS_hmm_peak_cat3_ggh])
    
    # n1_subCat3_vbf.setConstant(True) # freeze for stability
    # n2_subCat3_vbf.setConstant(True) # freeze for stability
    name = "signal_subCat3_vbf"
    signal_subCat3_vbf = rt.RooCrystalBall(name,name,mass, qqH_cat3_ggh_fpeak, qqH_cat3_ggh_fsigma, alpha1_subCat3_vbf, n1_subCat3_vbf, alpha2_subCat3_vbf, n2_subCat3_vbf)

    # subCat 4
    
    sigma_subCat4_vbf = rt.RooRealVar("sigma_subCat4_vbf" , "sigma_subCat4_vbf", 2, .1, 4.0)
    alpha1_subCat4_vbf = rt.RooRealVar("alpha1_subCat4_vbf" , "alpha1_subCat4_vbf", 2, 0.01, 65)
    n1_subCat4_vbf = rt.RooRealVar("n1_subCat4_vbf" , "n1_subCat4_vbf", 10, 0.01, 100)
    alpha2_subCat4_vbf = rt.RooRealVar("alpha2_subCat4_vbf" , "alpha2_subCat4_vbf", 2.0, 0.01, 65)
    n2_subCat4_vbf = rt.RooRealVar("n2_subCat4_vbf" , "n2_subCat4_vbf", 25, 0.01, 100)


    # # temporary test
    # sigma_subCat4_vbf.setConstant(True)
    # alpha1_subCat4_vbf.setConstant(True)
    # n1_subCat4_vbf.setConstant(True)
    # alpha2_subCat4_vbf.setConstant(True)
    # n2_subCat4_vbf.setConstant(True)

    qqH_cat4_ggh_fsigma = rt.RooFormulaVar("qqH_cat4_ggh_fsigma", "qqH_cat4_ggh_fsigma",'@0*(1+@1)',[sigma_subCat4_vbf, CMS_hmm_sigma_cat4_ggh])
    qqH_cat4_ggh_fpeak = rt.RooFormulaVar("qqH_cat4_ggh_fpeak", "qqH_cat4_ggh_fpeak",'@0*(1+@1)',[MH_subCat4, CMS_hmm_peak_cat4_ggh])
    
    # n1_subCat4_vbf.setConstant(True) # freeze for stability
    # n2_subCat4_vbf.setConstant(True) # freeze for stability
    name = "signal_subCat4_vbf"
    signal_subCat4_vbf = rt.RooCrystalBall(name,name,mass, qqH_cat4_ggh_fpeak, qqH_cat4_ggh_fsigma, alpha1_subCat4_vbf, n1_subCat4_vbf, alpha2_subCat4_vbf, n2_subCat4_vbf)
    
    
    # ---------------------------------------------------
    # Define signal MC samples to fit to for qqH
    # ---------------------------------------------------

    # subCat 0
    subCat_filter = (processed_eventsSignalMC_vbf[subCatIdx_name] == 0)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC_vbf.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat0_vbf_SigMC = ak.to_numpy(
        processed_eventsSignalMC_vbf.wgt_nominal[subCat_filter]
    ) # weights

    # generate a weighted histogram 
    roo_histData_subCat0_vbf_signal = rt.TH1F("subCat0_vbf_rooHist_signal", "subCat0_vbf_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat0_vbf_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat0_vbf_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat0_vbf_signal = rt.RooDataHist("subCat0_vbf_rooHist_signal", "subCat0_vbf_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat0_vbf_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat0_vbf_signal = roo_histData_subCat0_vbf_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat0"],
        "dataset": ["VBF"], 
        "yield": [data_subCat0_vbf_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)

    # define normalization value from signal MC event weights 
    flat_MC_SF = 1.00
    # flat_MC_SF = 0.92 # temporary flat SF to match my Data/MC agreement to that of AN's
    norm_val = np.sum(wgt_subCat0_vbf_SigMC)* flat_MC_SF 
    # norm_val = 254.528077 # quick test
    sig_norm_subCat0_vbf = rt.RooRealVar(signal_subCat0_vbf.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat0_vbf norm_val: {norm_val}")
    sig_norm_subCat0_vbf.setConstant(True)

    # subCat 1
    subCat_filter = (processed_eventsSignalMC_vbf[subCatIdx_name] == 1)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC_vbf.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat1_vbf_SigMC = ak.to_numpy(
        processed_eventsSignalMC_vbf.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat1_vbf_signal = rt.TH1F("subCat1_vbf_rooHist_signal", "subCat1_vbf_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat1_vbf_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat1_vbf_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat1_vbf_signal = rt.RooDataHist("subCat1_vbf_rooHist_signal", "subCat1_vbf_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat1_vbf_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat1_vbf_signal = roo_histData_subCat1_vbf_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat1"],
        "dataset": ["VBF"], 
        "yield": [data_subCat1_vbf_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat1_vbf_SigMC)* flat_MC_SF
    # norm_val = 295.214 # quick test
    sig_norm_subCat1_vbf = rt.RooRealVar(signal_subCat1_vbf.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat1_vbf norm_val: {norm_val}")
    sig_norm_subCat1_vbf.setConstant(True)

    # subCat 2
    subCat_filter = (processed_eventsSignalMC_vbf[subCatIdx_name] == 2)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC_vbf.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat2_vbf_SigMC = ak.to_numpy(
        processed_eventsSignalMC_vbf.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat2_vbf_signal = rt.TH1F("subCat2_vbf_rooHist_signal", "subCat2_vbf_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat2_vbf_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat2_vbf_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat2_vbf_signal = rt.RooDataHist("subCat2_vbf_rooHist_signal", "subCat2_vbf_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat2_vbf_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat2_vbf_signal = roo_histData_subCat2_vbf_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat2"],
        "dataset": ["VBF"], 
        "yield": [data_subCat2_vbf_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat2_vbf_SigMC) * flat_MC_SF
    # norm_val = 124.0364 # quick test
    sig_norm_subCat2_vbf = rt.RooRealVar(signal_subCat2_vbf.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat2_vbf norm_val: {norm_val}")
    sig_norm_subCat2_vbf.setConstant(True)

    # subCat 3
    subCat_filter = (processed_eventsSignalMC_vbf[subCatIdx_name] == 3)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC_vbf.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat3_vbf_SigMC = ak.to_numpy(
        processed_eventsSignalMC_vbf.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat3_vbf_signal = rt.TH1F("subCat3_vbf_rooHist_signal", "subCat3_vbf_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat3_vbf_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat3_vbf_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat3_vbf_signal = rt.RooDataHist("subCat3_vbf_rooHist_signal", "subCat3_vbf_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat3_vbf_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat3_vbf_signal = roo_histData_subCat3_vbf_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat3"],
        "dataset": ["VBF"], 
        "yield": [data_subCat3_vbf_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat3_vbf_SigMC)* flat_MC_SF
    # norm_val = 116.4918 # quick test
    sig_norm_subCat3_vbf = rt.RooRealVar(signal_subCat3_vbf.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat3_vbf norm_val: {norm_val}")
    sig_norm_subCat3_vbf.setConstant(True)
    
    # subCat 4
    subCat_filter = (processed_eventsSignalMC_vbf[subCatIdx_name] == 4)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC_vbf.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat4_vbf_SigMC = ak.to_numpy(
        processed_eventsSignalMC_vbf.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat4_vbf_signal = rt.TH1F("subCat4_vbf_rooHist_signal", "subCat4_vbf_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat4_vbf_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat4_vbf_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat4_vbf_signal = rt.RooDataHist("subCat4_vbf_rooHist_signal", "subCat4_vbf_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat4_vbf_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat4_vbf_signal = roo_histData_subCat4_vbf_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat4"],
        "dataset": ["VBF"], 
        "yield": [data_subCat4_vbf_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)
    print(f"yield_df after VBF: \n {yield_df}")

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat4_vbf_SigMC)* flat_MC_SF
    sig_norm_subCat4_vbf = rt.RooRealVar(signal_subCat4_vbf.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat4_vbf norm_val: {norm_val}")
    sig_norm_subCat4_vbf.setConstant(True)
    
    # ---------------------------------------------------
    # Fit signal model individually, not simultaneous. Sigma, and left and right tails are different for each category
    # ---------------------------------------------------

    # subCat 0
    # _ = signal_subCat0_vbf.fitTo(data_subCat0_vbf_signal,  EvalBackend=device, Save=True, )
    # fit_result = signal_subCat0_vbf.fitTo(data_subCat0_vbf_signal,  EvalBackend=device, Save=True, )
    _ = signal_subCat0_vbf.fitTo(data_subCat0_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat0_vbf.fitTo(data_subCat0_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat0_vbf.setConstant(True)
    alpha1_subCat0_vbf.setConstant(True)
    n1_subCat0_vbf.setConstant(True)
    alpha2_subCat0_vbf.setConstant(True)
    n2_subCat0_vbf.setConstant(True)


    # subCat 1
    # _ = signal_subCat1_vbf.fitTo(data_subCat1_vbf_signal,  EvalBackend=device, Save=True, )
    # fit_result = signal_subCat1_vbf.fitTo(data_subCat1_vbf_signal,  EvalBackend=device, Save=True, )
    _ = signal_subCat1_vbf.fitTo(data_subCat1_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat1_vbf.fitTo(data_subCat1_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat1_vbf.setConstant(True)
    alpha1_subCat1_vbf.setConstant(True)
    n1_subCat1_vbf.setConstant(True)
    alpha2_subCat1_vbf.setConstant(True)
    n2_subCat1_vbf.setConstant(True)



    # subCat 2
    # _ = signal_subCat2_vbf.fitTo(data_subCat2_vbf_signal,  EvalBackend=device, Save=True, )
    # fit_result = signal_subCat2_vbf.fitTo(data_subCat2_vbf_signal,  EvalBackend=device, Save=True, )
    _ = signal_subCat2_vbf.fitTo(data_subCat2_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat2_vbf.fitTo(data_subCat2_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat2_vbf.setConstant(True)
    alpha1_subCat2_vbf.setConstant(True)
    n1_subCat2_vbf.setConstant(True)
    alpha2_subCat2_vbf.setConstant(True)
    n2_subCat2_vbf.setConstant(True)


    
    # subCat 3
    # _ = signal_subCat3_vbf.fitTo(data_subCat3_vbf_signal,  EvalBackend=device, Save=True, )
    # fit_result = signal_subCat3_vbf.fitTo(data_subCat3_vbf_signal,  EvalBackend=device, Save=True, )
    _ = signal_subCat3_vbf.fitTo(data_subCat3_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat3_vbf.fitTo(data_subCat3_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat3_vbf.setConstant(True)
    alpha1_subCat3_vbf.setConstant(True)
    n1_subCat3_vbf.setConstant(True)
    alpha2_subCat3_vbf.setConstant(True)
    n2_subCat3_vbf.setConstant(True)
    # sigma_subCat3_vbf.setConstant(False)
    # alpha1_subCat3_vbf.setConstant(False)
    # n1_subCat3_vbf.setConstant(False)
    # alpha2_subCat3_vbf.setConstant(False)
    # n2_subCat3_vbf.setConstant(False)


    # subCat 4
    # _ = signal_subCat4_vbf.fitTo(data_subCat4_vbf_signal,  EvalBackend=device, Save=True, )
    # fit_result = signal_subCat4_vbf.fitTo(data_subCat4_vbf_signal,  EvalBackend=device, Save=True, )
    _ = signal_subCat4_vbf.fitTo(data_subCat4_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat4_vbf.fitTo(data_subCat4_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat4_vbf.setConstant(True)
    alpha1_subCat4_vbf.setConstant(True)
    n1_subCat4_vbf.setConstant(True)
    alpha2_subCat4_vbf.setConstant(True)
    n2_subCat4_vbf.setConstant(True)


    base_path = f"./validation/figs/{args.year}/{args.label}"
    # plot_save_path = f"./validation/figs/{args.year}"
    plot_save_path = base_path
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
        
    # -------------------------------------------------------------------------
    # Save yield_df
    # -------------------------------------------------------------------------
    summed_values = yield_df.groupby("dataset", as_index=False)["yield"].sum()
    summed_values["year"] = args.year
    summed_values["category"] = "combined"
    yield_df = pd.concat([yield_df, summed_values], ignore_index=True)
    # print(f"yield_df after all: \n {yield_df}")
    yield_df = yield_df.sort_values(by=["dataset", "category"], ascending=[False, True])
    yield_df.to_csv(f"{base_path}/yield_df.csv")
    
    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # do signal ggH plotting with fit and data
    # -------------------------------------------------------------------------
    
    # subCat 0
    print(f"data_subCat0_signal.sumEntries(): {data_subCat0_signal.sumEntries()}")
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
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat0.pdf")

    # subCat 1
    print(f"data_subCat1_signal.sumEntries(): {data_subCat1_signal.sumEntries()}")
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
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat1.pdf")

    # subCat 2
    print(f"data_subCat2_signal.sumEntries(): {data_subCat2_signal.sumEntries()}")
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat2_signal.GetName()
    data_subCat2_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat2.GetName()
    signal_subCat2.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat2.pdf")

    # subCat 3
    print(f"data_subCat3_signal.sumEntries(): {data_subCat3_signal.sumEntries()}")
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat3_signal.GetName()
    data_subCat3_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat3.GetName()
    signal_subCat3.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat3.pdf")

    # subCat 4
    print(f"data_subCat4_signal.sumEntries(): {data_subCat4_signal.sumEntries()}")
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat4_signal.GetName()
    data_subCat4_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat4.GetName()
    signal_subCat4.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat4.pdf")

    # -------------------------------------------------------------------------
    # do signal VBF plotting with fit and data
    # -------------------------------------------------------------------------
    
    # subCat 0
    print(f"data_subCat0_vbf_signal.sumEntries(): {data_subCat0_vbf_signal.sumEntries()}")
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat0_vbf_signal.GetName()
    data_subCat0_vbf_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat0_vbf.GetName()
    signal_subCat0_vbf.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat0_vbf.pdf")

    # subCat 1
    print(f"data_subCat1_vbf_signal.sumEntries(): {data_subCat1_vbf_signal.sumEntries()}")
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat1_vbf_signal.GetName()
    data_subCat1_vbf_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat1_vbf.GetName()
    signal_subCat1_vbf.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat1_vbf.pdf")

    # subCat 2
    print(f"data_subCat2_vbf_signal.sumEntries(): {data_subCat2_vbf_signal.sumEntries()}")
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat2_vbf_signal.GetName()
    data_subCat2_vbf_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat2_vbf.GetName()
    signal_subCat2_vbf.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat2_vbf.pdf")

    # subCat 3
    print(f"data_subCat3_vbf_signal.sumEntries(): {data_subCat3_vbf_signal.sumEntries()}")
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat3_vbf_signal.GetName()
    data_subCat3_vbf_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat3_vbf.GetName()
    signal_subCat3_vbf.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat3_vbf.pdf")

    # subCat 4
    print(f"data_subCat4_vbf_signal.sumEntries(): {data_subCat4_vbf_signal.sumEntries()}")
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat4_vbf_signal.GetName()
    data_subCat4_vbf_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat4_vbf.GetName()
    signal_subCat4_vbf.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat4_vbf.pdf")

    # ---------------------------------------------------
    # Save to Signal, Background and Data to Workspace
    # ---------------------------------------------------
    # workspace_path = "./workspaces"
    workspace_path = f"{base_path}/workspaces"
    if not os.path.exists(workspace_path):
        os.makedirs(workspace_path)


    # unfreeze the hmm sigma and peak b4 saving
    CMS_hmm_sigma_cat0_ggh.setConstant(False)
    CMS_hmm_peak_cat0_ggh.setConstant(False)

    CMS_hmm_sigma_cat1_ggh.setConstant(False)
    CMS_hmm_peak_cat1_ggh.setConstant(False)
    
    CMS_hmm_sigma_cat2_ggh.setConstant(False)
    CMS_hmm_peak_cat2_ggh.setConstant(False)
    
    CMS_hmm_sigma_cat3_ggh.setConstant(False)
    CMS_hmm_peak_cat3_ggh.setConstant(False)
    
    CMS_hmm_sigma_cat4_ggh.setConstant(False)
    CMS_hmm_peak_cat4_ggh.setConstant(False)
    
    # subCat 0 
    fout = rt.TFile(f"{workspace_path}/workspace_bkg_cat0_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    roo_histData_subCat0.SetName("data_cat0_ggh");
    corePdf_subCat0.SetName("bkg_cat0_ggh_pdf");
    bkg_subCat0_norm.SetName(corePdf_subCat0.GetName()+"_norm"); 
    # make norm for data
    nevents = roo_histData_subCat0.sumEntries()
    roo_histData_subCat0_norm = rt.RooRealVar(roo_histData_subCat0.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    wout.Import(roo_histData_subCat0_norm);
    wout.Import(roo_histData_subCat0);
    wout.Import(cat_subCat0);
    wout.Import(bkg_subCat0_norm);
    wout.Import(corePdf_subCat0);
    # wout.Print();
    wout.Write();

    fout = rt.TFile(f"{workspace_path}/workspace_sig_cat0_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    signal_subCat0.SetName("ggH_cat0_ggh_pdf");
    roo_histData_subCat0_signal.SetName("data_ggH_cat0_ggh");
    sig_norm_subCat0.SetName(signal_subCat0.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat0);
    wout.Import(signal_subCat0); 
    wout.Import(roo_histData_subCat0_signal); 
    
    signal_subCat0_vbf.SetName("qqH_cat0_ggh_pdf");
    roo_histData_subCat0_vbf_signal.SetName("data_qqH_cat0_ggh");
    sig_norm_subCat0_vbf.SetName(signal_subCat0_vbf.GetName()+"_norm"); 
    wout.Import(signal_subCat0_vbf);
    wout.Import(roo_histData_subCat0_vbf_signal); 
    wout.Import(sig_norm_subCat0_vbf); 
    
    # wout.Print();
    wout.Write();
    

    # subCat 1 
    fout = rt.TFile(f"{workspace_path}/workspace_bkg_cat1_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    roo_histData_subCat1.SetName("data_cat1_ggh");
    corePdf_subCat1.SetName("bkg_cat1_ggh_pdf");
    bkg_subCat1_norm.SetName(corePdf_subCat1.GetName()+"_norm");
    # make norm for data
    nevents = roo_histData_subCat1.sumEntries()
    roo_histData_subCat1_norm = rt.RooRealVar(roo_histData_subCat1.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    wout.Import(roo_histData_subCat1_norm);
    wout.Import(roo_histData_subCat1);
    wout.Import(cat_subCat1);
    wout.Import(bkg_subCat1_norm);
    wout.Import(corePdf_subCat1);
    # wout.Print();
    wout.Write();

    fout = rt.TFile(f"{workspace_path}/workspace_sig_cat1_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    signal_subCat1.SetName("ggH_cat1_ggh_pdf"); 
    roo_histData_subCat1_signal.SetName("data_ggH_cat1_ggh");
    sig_norm_subCat1.SetName(signal_subCat1.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat1);
    wout.Import(signal_subCat1); 
    wout.Import(roo_histData_subCat1_signal); 

    signal_subCat1_vbf.SetName("qqH_cat1_ggh_pdf"); 
    roo_histData_subCat1_vbf_signal.SetName("data_qqH_cat1_ggh");
    sig_norm_subCat1_vbf.SetName(signal_subCat1_vbf.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat1_vbf);
    wout.Import(signal_subCat1_vbf); 
    wout.Import(roo_histData_subCat1_vbf_signal); 
    # wout.Print();
    wout.Write();

    # subCat 2
    fout = rt.TFile(f"{workspace_path}/workspace_bkg_cat2_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    roo_histData_subCat2.SetName("data_cat2_ggh");
    corePdf_subCat2.SetName("bkg_cat2_ggh_pdf");
    bkg_subCat2_norm.SetName(corePdf_subCat2.GetName()+"_norm");
    # make norm for data
    nevents = roo_histData_subCat2.sumEntries()
    roo_histData_subCat2_norm = rt.RooRealVar(roo_histData_subCat2.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    wout.Import(roo_histData_subCat2_norm);
    wout.Import(roo_histData_subCat2);
    wout.Import(cat_subCat2);
    wout.Import(bkg_subCat2_norm);
    wout.Import(corePdf_subCat2);
    # wout.Print();
    wout.Write();

    fout = rt.TFile(f"{workspace_path}/workspace_sig_cat2_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    signal_subCat2.SetName("ggH_cat2_ggh_pdf"); 
    roo_histData_subCat2_signal.SetName("data_ggH_cat2_ggh");
    sig_norm_subCat2.SetName(signal_subCat2.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat2);
    wout.Import(signal_subCat2); 
    wout.Import(roo_histData_subCat2_signal); 

    signal_subCat2_vbf.SetName("qqH_cat2_ggh_pdf"); 
    roo_histData_subCat2_vbf_signal.SetName("data_qqH_cat2_ggh");
    sig_norm_subCat2_vbf.SetName(signal_subCat2_vbf.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat2_vbf);
    wout.Import(signal_subCat2_vbf); 
    wout.Import(roo_histData_subCat2_vbf_signal); 
    # wout.Print();
    wout.Write();


    # subCat 3
    fout = rt.TFile(f"{workspace_path}/workspace_bkg_cat3_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    roo_histData_subCat3.SetName("data_cat3_ggh");
    corePdf_subCat3.SetName("bkg_cat3_ggh_pdf");
    bkg_subCat3_norm.SetName(corePdf_subCat3.GetName()+"_norm");
    # make norm for data
    nevents = roo_histData_subCat3.sumEntries()
    roo_histData_subCat3_norm = rt.RooRealVar(roo_histData_subCat3.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    wout.Import(roo_histData_subCat3_norm);
    wout.Import(roo_histData_subCat3);
    wout.Import(cat_subCat3);
    wout.Import(bkg_subCat3_norm);
    wout.Import(corePdf_subCat3);
    # wout.Print();
    wout.Write();

    fout = rt.TFile(f"{workspace_path}/workspace_sig_cat3_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    signal_subCat3.SetName("ggH_cat3_ggh_pdf"); 
    roo_histData_subCat3_signal.SetName("data_ggH_cat3_ggh");
    sig_norm_subCat3.SetName(signal_subCat3.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat3);
    wout.Import(signal_subCat3); 
    wout.Import(roo_histData_subCat3_signal); 

    signal_subCat3_vbf.SetName("qqH_cat3_ggh_pdf"); 
    roo_histData_subCat3_vbf_signal.SetName("data_qqH_cat3_ggh");
    sig_norm_subCat3_vbf.SetName(signal_subCat3_vbf.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat3_vbf);
    wout.Import(signal_subCat3_vbf); 
    wout.Import(roo_histData_subCat3_vbf_signal); 
    # wout.Print();
    wout.Write();

    # subCat 4
    fout = rt.TFile(f"{workspace_path}/workspace_bkg_cat4_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    roo_histData_subCat4.SetName("data_cat4_ggh");
    corePdf_subCat4.SetName("bkg_cat4_ggh_pdf");
    bkg_subCat4_norm.SetName(corePdf_subCat4.GetName()+"_norm");
    # make norm for data
    nevents = roo_histData_subCat4.sumEntries()
    roo_histData_subCat4_norm = rt.RooRealVar(roo_histData_subCat4.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    wout.Import(roo_histData_subCat4_norm);
    wout.Import(roo_histData_subCat4);
    wout.Import(cat_subCat4);
    wout.Import(bkg_subCat4_norm);
    wout.Import(corePdf_subCat4);
    # wout.Print();
    wout.Write();

    fout = rt.TFile(f"{workspace_path}/workspace_sig_cat4_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    signal_subCat4.SetName("ggH_cat4_ggh_pdf"); 
    roo_histData_subCat4_signal.SetName("data_ggH_cat4_ggh");
    sig_norm_subCat4.SetName(signal_subCat4.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat4);
    wout.Import(signal_subCat4); 
    wout.Import(roo_histData_subCat4_signal); 

    signal_subCat4_vbf.SetName("qqH_cat4_ggh_pdf"); 
    roo_histData_subCat4_vbf_signal.SetName("data_qqH_cat4_ggh");
    sig_norm_subCat4_vbf.SetName(signal_subCat4_vbf.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat4_vbf);
    wout.Import(signal_subCat4_vbf); 
    wout.Import(roo_histData_subCat4_vbf_signal);
    # wout.Print();
    wout.Write();
    
    # ---------------------------------------------------
    # Group plotting start here
    # ---------------------------------------------------
    
    
    # -------------------------------------------------------------------------
    # do signal plotting for all sub-Cats in one plot
    # -------------------------------------------------------------------------
    sig_dict_by_sample = {
        "ggh_signal" : [
            signal_subCat0, 
            signal_subCat1,
            signal_subCat2,
            signal_subCat3,
            signal_subCat4,
        ],
    }
    sigHist_list = [ # for signal function normalization
        roo_histData_subCat0_signal,
        roo_histData_subCat1_signal,
        roo_histData_subCat2_signal,
        roo_histData_subCat3_signal,
        roo_histData_subCat4_signal
    ]
    plotSigBySample(mass, sig_dict_by_sample, sigHist_list, plot_save_path)

    sig_dict_by_sample = {
        "vbf_signal" : [
            signal_subCat0_vbf, 
            signal_subCat1_vbf,
            signal_subCat2_vbf,
            signal_subCat3_vbf,
            signal_subCat4_vbf,
        ]
    }
    sigHist_list = [ # for signal function normalization
        roo_histData_subCat0_vbf_signal,
        roo_histData_subCat1_vbf_signal,
        roo_histData_subCat2_vbf_signal,
        roo_histData_subCat3_vbf_signal,
        roo_histData_subCat4_vbf_signal
    ]
    plotSigBySample(mass, sig_dict_by_sample, sigHist_list, plot_save_path)
        

    # -------------------------------------------------------------------------
    # do Bkg plotting loop divided into core-function
    # -------------------------------------------------------------------------
    
    model_dict_by_coreFunction = {
        "BWZRedux" : [
            model_subCat0_BWZRedux, 
            model_subCat1_BWZRedux,
            model_subCat2_BWZRedux,
            model_subCat3_BWZRedux,
            model_subCat4_BWZRedux,
        ],
        "sumExp" : [
            model_subCat0_sumExp, 
            model_subCat1_sumExp,
            model_subCat2_sumExp,
            model_subCat3_sumExp,
            model_subCat4_sumExp,
        ],
        "FEWZxBern" : [
            model_subCat0_FEWZxBern, 
            model_subCat1_FEWZxBern,
            model_subCat2_FEWZxBern,
            model_subCat3_FEWZxBern,
            model_subCat4_FEWZxBern,
        ],
        # "FEWZxBern" : [
        #     coreFEWZxBern_SubCat0, 
        #     coreFEWZxBern_SubCat1,
        #     coreFEWZxBern_SubCat2,
        #     coreFEWZxBern_SubCat3,
        #     coreFEWZxBern_SubCat4,
        # ],
        "SMF" : [
            subCat0_SMF, 
            subCat1_SMF,
            subCat2_SMF,
            subCat3_SMF,
            subCat4_SMF,
        ],
    }
    rooHist_list = [ # for normalization histogram reference
        roo_histData_subCat0,
        roo_histData_subCat1,
        roo_histData_subCat2,
        roo_histData_subCat3,
        roo_histData_subCat4
    ]
    plotBkgByCoreFunc(mass, model_dict_by_coreFunction, rooHist_list, plot_save_path)
    

    # -------------------------------------------------------------------------
    # do Bkg plotting loop divided into Sub Categories
    # -------------------------------------------------------------------------

    model_dict_by_subCat = {
        0 : [
            model_subCat0_BWZRedux, 
            model_subCat0_sumExp,
            model_subCat0_FEWZxBern,
        ],
        1 : [
            model_subCat1_BWZRedux, 
            model_subCat1_sumExp,
            model_subCat1_FEWZxBern,
        ],
        2 : [
            model_subCat2_BWZRedux, 
            model_subCat2_sumExp,
            model_subCat2_FEWZxBern,
        ],
        3 : [
            model_subCat3_BWZRedux, 
            model_subCat3_sumExp,
            model_subCat3_FEWZxBern,
        ],
        4 : [
            model_subCat4_BWZRedux, 
            model_subCat4_sumExp,
            model_subCat4_FEWZxBern,
        ],
    }
    data_dict_by_subCat = {
        0 : roo_histData_subCat0,
        1 : roo_histData_subCat1,
        2 : roo_histData_subCat2,
        3 : roo_histData_subCat3,
        4 : roo_histData_subCat4,
    }
    plotBkgBySubCat(mass, model_dict_by_subCat, data_dict_by_subCat, plot_save_path)
    # model_dict_by_subCat_n_corefunc = {
    #     0 : {
    #         "BWZ_redux" : model_subCat0_BWZRedux, 
    #         "SumExp" : model_subCat0_sumExp,
    #         "FEWZxBern" : model_subCat0_FEWZxBern,
    #     },
    #     1 : {
    #         "BWZ_redux" : model_subCat1_BWZRedux, 
    #         "SumExp" : model_subCat1_sumExp,
    #         "FEWZxBern" : model_subCat1_FEWZxBern,
    #     },
    #     2 : {
    #         "BWZ_redux" : model_subCat2_BWZRedux, 
    #         "SumExp" : model_subCat2_sumExp,
    #         "FEWZxBern" : model_subCat2_FEWZxBern,
    #     },
    #     3 : {
    #         "BWZ_redux" : model_subCat3_BWZRedux, 
    #         "SumExp" : model_subCat3_sumExp,
    #         "FEWZxBern" : model_subCat3_FEWZxBern,
    #     },
    #     4 : {
    #         "BWZ_redux" : model_subCat4_BWZRedux, 
    #         "SumExp" : model_subCat4_sumExp,
    #         "FEWZxBern" : model_subCat4_FEWZxBern,
    #     },
    # }
    # plotDataBkgDiffBySubCat(mass, model_dict_by_subCat_n_corefunc, data_dict_by_subCat, plot_save_path)

    
    # -------------------------------------------------------------------------
    # do background core function plotting
    # -------------------------------------------------------------------------
    base_path = f"./validation/figs/{args.year}/{args.label}"
    # plot_save_path = f"./validation/figs/{args.year}"
    plot_save_path = base_path
    data_name = "Run2 Data" # for Legend
    
    # BWZ redux
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    fit_range = "hiSB,loSB"
    plot_range = "full"
    name = data_allSubCat_BWZ.GetName()
    data_allSubCat_BWZ.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),data_name, "P")
    name = coreBWZRedux_SubCat0.GetName()
    coreBWZRedux_SubCat0.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range(plot_range), Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")

    ndf = coreBWZRedux_SubCat0.getParameters(ROOT.RooArgSet(mass)).getSize()
    chi2_ndf = frame.chiSquare(coreBWZRedux_SubCat0.GetName(), data_allSubCat_BWZ.GetName(), ndf)
    print(f"ndf: {ndf}")
    chi2_text = "chi2/ndf = {:.3f}".format(chi2_ndf)
    legend.AddEntry("", chi2_text, "")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_BWZ_redux.pdf")


    # Sum Exp
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    fit_range = "hiSB,loSB"
    plot_range = "full"
    name = data_allSubCat_FEWZxBern.GetName()
    # name = "Run2 Data"
    data_allSubCat_FEWZxBern.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),data_name, "P")
    name = coreSumExp_SubCat0.GetName()
    coreSumExp_SubCat0.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range(plot_range), Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")

    ndf = coreSumExp_SubCat0.getParameters(ROOT.RooArgSet(mass)).getSize()
    chi2_ndf = frame.chiSquare(coreSumExp_SubCat0.GetName(), data_allSubCat_FEWZxBern.GetName(), ndf)
    print(f"ndf: {ndf}")
    chi2_text = "chi2/ndf = {:.3f}".format(chi2_ndf)
    legend.AddEntry("", chi2_text, "")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_sumExp.pdf")

    # FEWZ Bern
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    fit_range = "hiSB,loSB"
    plot_range = "full"
    name = data_allSubCat_FEWZxBern.GetName()
    # name = "Run2 Data"
    data_allSubCat_FEWZxBern.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),data_name, "P")
    name = coreFEWZxBern_SubCat0.GetName()
    coreFEWZxBern_SubCat0.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range(plot_range), Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")

    ndf = coreFEWZxBern_SubCat0.getParameters(ROOT.RooArgSet(mass)).getSize()
    chi2_ndf = frame.chiSquare(coreFEWZxBern_SubCat0.GetName(), data_allSubCat_FEWZxBern.GetName(), ndf)
    print(f"ndf: {ndf}")
    chi2_text = "chi2/ndf = {:.3f}".format(chi2_ndf)
    legend.AddEntry("", chi2_text, "")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_FEWZxBern.pdf")
    

    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Success! Execution time: {elapsed_time:.3f} seconds")

