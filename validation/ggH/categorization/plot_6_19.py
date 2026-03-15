import awkward as ak
import dask_awkward as dak
import argparse
import sys
import os
import numpy as np
import json
from collections import OrderedDict
import cmsstyle as CMS
import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib
plt.style.use(hep.style.CMS)
from omegaconf import OmegaConf
from modules.utils import fillSampleValues, getDimuMassBySubCat
from modules.RooWorkspaceUtils import rebinRooDataHist
from modules.GoF_utils import getGOF_KS
from modules.selection import filterRegion
from modules.fit_functions import getFEWZ_roospline, getPowerLaw
import ROOT
import ROOT as rt
import copy
import pandas as pd
import uuid

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
# Add it to sys.path
sys.path.insert(0, parent_dir)
# Now you can import your module
from src.lib.histogram.plotting import plotFig_6_13



def createUnitHistogram(name, nbins, xmin, xmax):
    hist = ROOT.TH1D(name, name, nbins, xmin, xmax)
    # Set all bin contents (1-based indexing for ROOT histograms)
    for bin_idx in range(1, nbins + 1):
        hist.SetBinContent(bin_idx, 1.0)
        hist.SetBinError(bin_idx, 0.0)

    return hist

def createUnitHistogram_roofit(x):
    name = "flat unit histogram"
    nbins = x.getBins()       # Number of bins (default or defined)
    xmin  = x.getMin()        # Lower range
    xmax  = x.getMax()        # Upper range
    print(f"nbins: {nbins}")
    print(f"xmin: {xmin}")
    print(f"xmax: {xmax}")
    th1 = createUnitHistogram(name, nbins, xmin, xmax)
    th1_rooHist = rt.RooDataHist(name, name, rt.RooArgSet(x), th1) 
    return th1_rooHist

def getColor(name):
    """
    helper function to get root colors
    """
    # color_list = [
    #     rt.kGreen,
    #     rt.kBlue,
    #     rt.kRed,
    #     rt.kOrange,
    #     rt.kViolet,
    # ]
    if "bwzredux" in name.lower():
        return rt.kRed, rt.kSolid
    elif "bwz" in name.lower() and "bern" in name.lower():
        return rt.kBlue, rt.kSolid
    elif "s-power" in name.lower():
        return rt.kCyan, rt.kDashDotted
    elif "s-exponential" in name.lower():
        return rt.kOrange, rt.kDashed 
    elif "bwz" in name.lower() and "gamma" in name.lower():
        return rt.kGreen, rt.kDotted
    elif "fewz" in name.lower() and "bern" in name.lower():
        return rt.kViolet, rt.kDashDotted
    elif "landau" in name.lower() and "bern" in name.lower():
        return rt.kGray, rt.kDashDotted
    else:
        print("Error, color not available for the function!")
        raise ValueError

def getBWZ_gamma(x):
    name = f"BWZ_a_coeff"
    a_coeff_bwz = rt.RooRealVar(name,name, -0.02,-0.5,0.5)
    name = "BWZ"
    BWZ = rt.RooModZPdf(name, name, x, a_coeff_bwz) 

    name = f"Gamma_a_coeff"
    a_coeff_gamma = rt.RooRealVar(name,name, -0.00005,-0.05,0.05)
    gamma = rt.RooGenericPdf("Gamma", "exp(@1*@0)/pow(@0,2)", rt.RooArgList(x, a_coeff_gamma))

    name = f"frac"
    frac = rt.RooRealVar(name,name, 0.5, 0.0, 1.0) 
    name = "BWZGamma"
    coreBWZGamma = rt.RooAddPdf(name, name, [BWZ, gamma], [frac])

    param_l = [
        a_coeff_bwz,
        BWZ,
        a_coeff_gamma,
        gamma,
        frac,
    ]# list of variables to return so that they don't get deleted in python functions. Otherwise the roofit pdfs don't work
    return coreBWZGamma, param_l


def getBWZxBern(x):
    name = f"BWZ_a_coeff"
    a_coeff_bwz = rt.RooRealVar(name,name, -0.02, -10, 10)

    name = f"bernstein_a0"
    a0_bern = rt.RooRealVar(name,name, 0.3, -10, 10)
    name = f"bernstein_a1"
    a1_bern = rt.RooRealVar(name,name, 0.3, -10, 10) # starting value = 1/n_coeffs
    

    name = "BWZxBernstein"
    coreBWZxBern = rt.RooModZPdf(name, name, x, a_coeff_bwz, rt.RooArgList(a0_bern,a1_bern)) 

    param_l = [
        a_coeff_bwz,
        a0_bern,
        a1_bern,
    ]# list of variables to return so that they don't get deleted in python functions. Otherwise the roofit pdfs don't work
    return coreBWZxBern, param_l


def getLandxBern(x):
    name = f"m_Z"
    m_Z = rt.RooRealVar(name,name, 91.2)
    name = f"a_coeff"
    # a_coeff = rt.RooRealVar(name,name, 0.2,0.1,3)
    a_coeff = rt.RooRealVar(name,name, 0.258087,0.0,3)
    name = "Landau"
    Landau = rt.RooLandau(name, name, x, m_Z, a_coeff) 

    name = f"bernstein_a0"
    # a0_bern = rt.RooRealVar(name,name, 1.157, 0, 2)
    # a0_bern = rt.RooRealVar(name,name, 0.3, 0, 2)
    a0_bern = rt.RooRealVar(name,name, 1.0) # first term being constant
    name = f"bernstein_a1"
    # a1_bern = rt.RooRealVar(name,name, 1.6, 0, 2)
    # a1_bern = rt.RooRealVar(name,name, 1.5, 0, 5) # starting value = 1/n_coeffs
    a1_bern = rt.RooRealVar(name,name, 1.5, 0, 5) # starting value = 1/n_coeffs
    name = f"bernstein_a2"
    a2_bern = rt.RooRealVar(name,name, 0.75, 0.0, 10) # starting value = 1/n_coeffs
    
    # bern_pol = rt.RooBernsteinFast(3)(name, name, x, rt.RooArgList(a0_bern, a1_bern, a2_bern)) 
    # name = "LandauxBernstein"
    name = "bernstein_landau"
    bern_pol = rt.RooBernstein(name, name, x, rt.RooArgList(a0_bern, a1_bern, a2_bern)) # extra parameter is needed https://root-forum.cern.ch/t/roobernstein-correction/41800

    
    name = "LandauxBernstein"
    # coreLandxBern = rt.RooProdPdf(name, name, [Landau, bern_pol])
    coreLandxBern = rt.RooProdPdf(name, name, [bern_pol, Landau])

    param_l = [
        m_Z,
        a_coeff,
        Landau,
        a0_bern,
        a1_bern,
        a2_bern,
        bern_pol,
    ]# list of variables to return so that they don't get deleted in python functions. Otherwise the roofit pdfs don't work
    return coreLandxBern, param_l


def getFEWZxBern(x):
    # roo_spline_func = getFEWZ_roospline(x, "modules/ucsd_workspace/")
    roo_spline_func = getFEWZ_roospline(x, "modules/ucsd_workspace/")

    name = "fewz_1j_spl_pdf"
    roo_spline_pdf = rt.RooWrapperPdf(name, name, roo_spline_func)

    name = f"bernstein_a0"
    a0_bern = rt.RooRealVar(name,name, 1.0) # first term being constant
    name = f"bernstein_a1"
    a1_bern = rt.RooRealVar(name,name, 1.5, 0, 5) 
    name = f"bernstein_a2"
    a2_bern = rt.RooRealVar(name,name, 0.75, 0.0, 10) 
    name = f"bernstein_a3"
    a3_bern = rt.RooRealVar(name,name, 0.75, 0.0, 10) 
    
    name = "bernstein_FEWZxBern"
    bern_pol = rt.RooBernstein(name, name, x, rt.RooArgList(a0_bern, a1_bern, a2_bern, a3_bern)) # extra parameter is needed https://root-forum.cern.ch/t/roobernstein-correction/41800

    
    name = "FEWZxBernstein"
    coreFEWZxBern = rt.RooProdPdf(name, name, [roo_spline_pdf, bern_pol])

    param_l = [
        roo_spline_func,
        roo_spline_pdf,
        a0_bern,
        a1_bern,
        a2_bern,
        a3_bern,
        bern_pol,
    ]# list of variables to return so that they don't get deleted in python functions. Otherwise the roofit pdfs don't work
    return coreFEWZxBern, param_l

def plot_6_19(dataDict_by_subCat, save_fname, nSubCats=5, apply_blind=True):
    device = "cpu"
    # nSubCats=1 # FIXME
    df_rows = []
    fitResults={}
    powerLawStarValDict = {
        0: {
            'RooSumTwoPowerLawPdf_a1_coeff': -1.9743280291018659, 'RooSumTwoPowerLawPdf_a2_coeff': -5.070611059366495, 'RooSumTwoPowerLawPdf_f_coeff': 0.9169613224723027
           }, 
        1: {
            'RooSumTwoPowerLawPdf_a1_coeff': -2.6281519073027844, 'RooSumTwoPowerLawPdf_a2_coeff': -4.460879015664051, 'RooSumTwoPowerLawPdf_f_coeff': 0.44823019493423466
           }, 
        2: {
            'RooSumTwoPowerLawPdf_a1_coeff': -2.3033162245329164, 'RooSumTwoPowerLawPdf_a2_coeff': -4.183898141656655, 'RooSumTwoPowerLawPdf_f_coeff': 0.40115183986375974
           }, 
        3: {
            'RooSumTwoPowerLawPdf_a1_coeff': -2.4924512293219796, 'RooSumTwoPowerLawPdf_a2_coeff': -3.681762231096663, 'RooSumTwoPowerLawPdf_f_coeff': 0.2902132176929161
           }, 
        4: {
            'RooSumTwoPowerLawPdf_a1_coeff': -1.3892075987583783, 'RooSumTwoPowerLawPdf_a2_coeff': -84.29141727097351, 'RooSumTwoPowerLawPdf_f_coeff': 0.9948389705318498
           }, 
        'all': {'RooSumTwoPowerLawPdf_a1_coeff': -2.1969979862085047, 'RooSumTwoPowerLawPdf_a2_coeff': -4.18629756373829, 'RooSumTwoPowerLawPdf_f_coeff': 0.6569304312671497}
    }

    ks_df = pd.DataFrame(columns=["pdf category", "region", "KS statistic", "nevents", "alpha", "pass threshold", "test pass"])
    
    
    # for target_subCat in range(nSubCats):
    for target_subCat in dataDict_by_subCat.keys():
        dataDict_target = dataDict_by_subCat[target_subCat]
        mass_name = "mh_ggh"
        mass = rt.RooRealVar(mass_name, mass_name, 120, 110, 150)
        nbins = 800
        # nbins = 120
        # nbins = 200
        mass.setBins(nbins)
        mass.setRange("hiSB", 135, 150 )
        mass.setRange("loSB", 110, 115 )
        mass.setRange("h_peak", 115, 135 )
        mass.setRange("full", 110, 150 )
        if apply_blind:
            fit_range = "hiSB,loSB" 
        else:
            fit_range = "full" 
        subCat_mass_arr  = ak.to_numpy(dataDict_target["dimuon_mass"]) # convert to numpy for rt.RooDataSet
        if apply_blind:
            dimuon_mass = subCat_mass_arr
            h_peak = (dimuon_mass > 115) & (dimuon_mass < 135)
            blind_filter = ~h_peak
            subCat_mass_arr = subCat_mass_arr[blind_filter]
        roo_datasetData = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
        roo_histData = rt.RooDataHist("rooHist_BWZRedux","rooHist_BWZRedux", rt.RooArgSet(mass), roo_datasetData)
        
        # fit BWZ redux
        name = f"BWZ_Redux_a_coeff"
        a_coeff = rt.RooRealVar(name,name, -0.02,-0.5,0.5)
        name = f"BWZ_Redux_b_coeff"
        b_coeff = rt.RooRealVar(name,name, -0.000111,-0.1,0.1)
        name = f"BWZ_Redux_c_coeff"
        c_coeff = rt.RooRealVar(name,name, 0.5,-10.0,10.0)
        name = "BWZRedux" # source: https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/blob/5ae49dd944479b79af5692ff47fd7f1d9de16e91/interface/HMuMuRooPdfs.h#L11
        coreBWZRedux = rt.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff) 
        _ = coreBWZRedux.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0)
        fitResult = coreBWZRedux.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,)
        # print(f"fitResult: {fitResult}")
        

        # fit Sum exp
        name = f"RooSumTwoExpPdf_a1_coeff"
        a1_coeff = rt.RooRealVar(name,name, 0.00001,-2.0,1)
        name = f"RooSumTwoExpPdf_a2_coeff"
        a2_coeff = rt.RooRealVar(name,name, 0.1,-2.0,1)
        name = f"RooSumTwoExpPdf_f_coeff"
        f_coeff = rt.RooRealVar(name,name, 0.9,0.0,1.0)
    
        name = "S-Exponential"
        coreSumExp = rt.RooSumTwoExpPdf(name, name, mass, a1_coeff, a2_coeff, f_coeff) 
        _ = coreSumExp.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0)
        fitResult = coreSumExp.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,)

        # fit Sum Power law
        name = f"RooSumTwoPowerLawPdf_a1_coeff"
        # a1_coeff_pow = rt.RooRealVar(name,name, 0.0,-20.0,20)
        # a1_coeff_pow = rt.RooRealVar(name,name, -2,-3.5,-1.0)
        # print(powerLawStarValDict[target_subCat])
        
        a1_coeff_pow = rt.RooRealVar(name,name, powerLawStarValDict[target_subCat][name],-3.5,-1.0)
        name = f"RooSumTwoPowerLawPdf_a2_coeff"
        # a2_coeff_pow = rt.RooRealVar(name,name, 0.0001,-20.0,20)
        if target_subCat == 4:
            a2_coeff_pow = rt.RooRealVar(name,name, powerLawStarValDict[target_subCat][name],-150.0, -50)
            # a2_coeff_pow = rt.RooRealVar(name,name, -84,-150.0, -50)
        else:
            a2_coeff_pow = rt.RooRealVar(name,name, powerLawStarValDict[target_subCat][name],-6.0, -3)
            # a2_coeff_pow = rt.RooRealVar(name,name, -4,-6.0, -3)
        name = f"RooSumTwoPowerLawPdf_f_coeff"
        # f_coeff_pow = rt.RooRealVar(name,name, 0.1,0.0,1.0)
        # f_coeff_pow = rt.RooRealVar(name,name, 0.9,0.0,1.0)
        f_coeff_pow = rt.RooRealVar(name,name, powerLawStarValDict[target_subCat][name],0.0,1.0)
    
        name = "S-Power-Law"
        # coreSumPow = rt.RooSumTwoPowerLawPdf(name, name, mass, a1_coeff_pow, a2_coeff_pow, f_coeff_pow)
        coreSumPow, param_l_powerLaw  = getPowerLaw(mass, powerLawStarValDict[target_subCat])
        # _ = coreSumPow.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0)
        _ = coreSumPow.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device,  PrintLevel=0 ,Save=True, Minos=True)
        fitResult = coreSumPow.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,)
        # _ = coreSumPow.fitTo(roo_histData, rt.RooFit.Range("full"), EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0)
        # fitResult = coreSumPow.fitTo(roo_histData, rt.RooFit.Range("full"), EvalBackend=device, PrintLevel=0 ,Save=True,)
        print(f"coreSumPow : \n")
        fitResult.Print()
        # fitResults[target_subCat] = {
        #     "RooSumTwoPowerLawPdf_a1_coeff" : a1_coeff_pow.getVal(),
        #     "RooSumTwoPowerLawPdf_a2_coeff" : a2_coeff_pow.getVal(),
        #     "RooSumTwoPowerLawPdf_f_coeff" : f_coeff_pow.getVal(),
        # }

        # fit BWZ Gamma
        coreBWZGamma, param_l_bwz_gamma = getBWZ_gamma(mass)
        _ = coreBWZGamma.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0)
        fitResult = coreBWZGamma.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,)
        print(f"coreBWZGamma : \n")
        # fitResult.Print()
        # raise ValueError

        # fit BWZ Gamma
        coreBWZxBern, param_l_bwz_bern = getBWZxBern(mass)
        _ = coreBWZxBern.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0)
        fitResult = coreBWZxBern.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,)
        # print(f"coreBWZxBern : \n")
        # fitResult.Print()

        # fit FEWZxBern
        coreFEWZxBern, param_l_fewz_bern = getFEWZxBern(mass)
        _ = coreFEWZxBern.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0)
        fitResult = coreFEWZxBern.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True)
        print(f"coreFEWZxBern : \n")
        # fitResult.Print()
        
        # fit LandxBern
        coreLandxBern, param_l_land_bern = getLandxBern(mass)
        _ = coreLandxBern.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0)
        fitResult = coreLandxBern.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True)
        # print(f"coreLandxBern : \n")
        # fitResult.Print()

        
        nfree_params = fitResult.floatParsFinal().getSize() # ndf should be consistent over all possible bkg fit functions
        print(f"nfree_params: {nfree_params}")

        # --------------------------------------------------------------------
        # plot
        # --------------------------------------------------------------------
        name = f"Canvas_{uuid.uuid4().hex}"
        canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
        canvas.cd()
        # Define upper and lower pads
        pad1 = ROOT.TPad("pad1", "Distribution", 0, 0.3, 1, 1.0)
        pad2 = ROOT.TPad("pad2", "Ratio", 0, 0.0, 1, 0.3)
        
        # Adjust margins
        pad1.SetBottomMargin(0)  # Upper plot does not need bottom margin
        pad2.SetTopMargin(0)     # Lower plot does not need top margin
        pad2.SetBottomMargin(0.3)

        pad1.SetTicks(2, 2)
        pad2.SetTicks(2, 2)
        pad1.Draw() # value plot
        pad2.Draw() # ratio plot
    
        pad1.cd()
        legend = rt.TLegend(0.65,0.55,0.9,0.7)
        frame = mass.frame()

        # plot invisible data points for pdfs could be normalized
        hist_name = roo_histData.GetName()
        roo_histData.plotOn(frame, Invisible=True, Name=hist_name,)
        # legend.AddEntry(frame.getObject(int(frame.numItems())-1),"Data", "P")
        
        # BWZRedux
        color, style = getColor(coreBWZRedux.GetName())
        name = coreBWZRedux.GetName()
        coreBWZRedux.plotOn(frame, DataError="SumW2", Name=name, LineColor=color, LineStyle=style)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")

        chi2ndf = frame.chiSquare(name, hist_name, nfree_params)
        print(f"chi^2 {name} = ", chi2ndf)
        df_rows.append({
            "BDT cat": target_subCat,
            "fit function": name,
            "chi2ndf": chi2ndf
        })
        
        # -----------------------------------------------------------
        # BWZxBern
        color, style = getColor(coreBWZxBern.GetName())
        name = coreBWZxBern.GetName()
        coreBWZxBern.plotOn(frame, DataError="SumW2", Name=name, LineColor=color, LineStyle=style)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")

        chi2ndf = frame.chiSquare(name, hist_name, nfree_params)
        print(f"chi^2 {name} = ", chi2ndf)
        df_rows.append({
            "BDT cat": target_subCat,
            "fit function": name,
            "chi2ndf": chi2ndf
        })
        # -----------------------------------------------------------
        # Sum Exp
        color, style = getColor(coreSumExp.GetName())
        name = coreSumExp.GetName()
        coreSumExp.plotOn(frame, DataError="SumW2", Name=name, LineColor=color, LineStyle=style)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")

        chi2ndf = frame.chiSquare(name, hist_name, nfree_params)
        print(f"chi^2 {name} = ", chi2ndf)
        df_rows.append({
            "BDT cat": target_subCat,
            "fit function": name,
            "chi2ndf": chi2ndf
        })
        # -----------------------------------------------------------
        # Sum Power
        color, style = getColor(coreSumPow.GetName())
        name = coreSumPow.GetName()
        coreSumPow.plotOn(frame, DataError="SumW2", Name=name, LineColor=color, LineStyle=style)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")

        chi2ndf = frame.chiSquare(name, hist_name, nfree_params)
        print(f"chi^2 {name} = ", chi2ndf)
        df_rows.append({
            "BDT cat": target_subCat,
            "fit function": name,
            "chi2ndf": chi2ndf
        })
        # -----------------------------------------------------------
        # BWZxGamma
        color, style = getColor(coreBWZGamma.GetName())
        name = coreBWZGamma.GetName()
        coreBWZGamma.plotOn(frame, DataError="SumW2", Name=name, LineColor=color, LineStyle=style)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")

        chi2ndf = frame.chiSquare(name, hist_name, nfree_params)
        print(f"chi^2 {name} = ", chi2ndf)
        df_rows.append({
            "BDT cat": target_subCat,
            "fit function": name,
            "chi2ndf": chi2ndf
        })
        # -----------------------------------------------------------
        # FEWZxBern
        color, style = getColor(coreFEWZxBern.GetName())
        name = coreFEWZxBern.GetName()
        coreFEWZxBern.plotOn(frame, DataError="SumW2", Name=name, LineColor=color, LineStyle=style)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")

        chi2ndf = frame.chiSquare(name, hist_name, nfree_params)
        print(f"chi^2 {name} = ", chi2ndf)
        df_rows.append({
            "BDT cat": target_subCat,
            "fit function": name,
            "chi2ndf": chi2ndf
        })
        # -----------------------------------------------------------
        # LandxBern
        color, style = getColor(coreLandxBern.GetName())
        name = coreLandxBern.GetName()
        coreLandxBern.plotOn(frame, DataError="SumW2", Name=name, LineColor=color, LineStyle=style)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")

        chi2ndf = frame.chiSquare(name, hist_name, nfree_params)
        print(f"chi^2 {name} = ", chi2ndf)
        df_rows.append({
            "BDT cat": target_subCat,
            "fit function": name,
            "chi2ndf": chi2ndf
        })
        # -----------------------------------------------------------
        # Data
        # roo_histData.plotOn(frame)
        target_nbins = 80
        rebin_factor = nbins // target_nbins
        roo_histData_rebinned = rebinRooDataHist(mass, roo_histData, rebin_factor)
        roo_histData_rebinned.plotOn(frame)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),"Data", "P")
        
        
        # Draw frame and legend        
        frame.Draw()
        legend.Draw()
        
        # ---------------------------------------------------------------
        # ratio plot
        # ---------------------------------------------------------------
        pad2.cd()
        ratio_frame= mass.frame()
        

        flat_unit_hist = createUnitHistogram_roofit(mass)
        # Note: DataError=None flag is needed to set our GetYaxis().SetRangeUser() to our desired values. Otherwise, the size of the error bars plays a role.
        flat_unit_hist.plotOn(ratio_frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0), Invisible=True, DataError=None)

        # BWZRedux
        flat_pdf = ROOT.RooPolynomial("BWZRedux ratio", "BWZRedux ratio", mass)
        color, style = getColor(flat_pdf.GetName())
        flat_pdf.plotOn(ratio_frame, DataError="SumW2", LineColor=color, LineStyle=style)

        # -----------------------------------------------------------
        # BWZxBern
        ratio_bwzxBern = rt.RooGenericPdf("BWZxBernstein ratio", "@0/@1", rt.RooArgList(coreBWZxBern,coreBWZRedux,))
        color, style = getColor(ratio_bwzxBern.GetName())
        ratio_bwzxBern.plotOn(ratio_frame, DataError="SumW2", LineColor=color, LineStyle=style)
        # -----------------------------------------------------------
        # Sum Exp
        ratio_sumExp = rt.RooGenericPdf("S-Exponential ratio", "@0/@1", rt.RooArgList(coreSumExp,coreBWZRedux))
        color, style = getColor(ratio_sumExp.GetName())
        ratio_sumExp.plotOn(ratio_frame, DataError="SumW2", LineColor=color, LineStyle=style)
        # -----------------------------------------------------------
        # Sum Power
        ratio_sumPow = rt.RooGenericPdf("S-Power-Law ratio", "@0/@1", rt.RooArgList(coreSumPow,coreBWZRedux))
        color, style = getColor(ratio_sumPow.GetName())
        ratio_sumPow.plotOn(ratio_frame, DataError="SumW2", LineColor=color, LineStyle=style)

        # -----------------------------------------------------------
        # BWZxGamma
        ratio_bwzGamma = rt.RooGenericPdf("BWZGamma ratio", "@0/@1", rt.RooArgList(coreBWZGamma,coreBWZRedux))
        color, style = getColor(ratio_bwzGamma.GetName())
        ratio_bwzGamma.plotOn(ratio_frame, DataError="SumW2", LineColor=color, LineStyle=style)
        # -----------------------------------------------------------
        # FEWZxBern
        ratio_FEWZxBern = rt.RooGenericPdf("FEWZxBernstein ratio", "@0/@1", rt.RooArgList(coreFEWZxBern,coreBWZRedux))
        color, style = getColor(ratio_FEWZxBern.GetName())
        ratio_FEWZxBern.plotOn(ratio_frame, DataError="SumW2", LineColor=color, LineStyle=style)
        # -----------------------------------------------------------
        # LandxBern
        ratio_landXBern = rt.RooGenericPdf("LandauxBernstein ratio", "@0/@1", rt.RooArgList(coreLandxBern,coreBWZRedux))
        color, style = getColor(ratio_landXBern.GetName())
        ratio_landXBern.plotOn(ratio_frame, DataError="SumW2", LineColor=color, LineStyle=style)

        
        
        # set ranges and label sizes after all things are plotted
        ratio_frame.GetXaxis().SetLabelSize(0.10)
        ratio_frame.SetXTitle(f"Dimuon Mass (GeV)")
        ratio_frame.GetXaxis().SetTitleSize(0.10)
        
        ratio_frame.GetYaxis().SetLabelSize(0.08)
        ratio_frame.GetYaxis().SetRangeUser(0.98, 1.02)
        ratio_frame.SetTitle("")
        ratio_frame.Draw()
        
        canvas.Update()
        canvas.Draw()
        if apply_blind:
            canvas.SaveAs(f"{save_fname}_subCat{target_subCat}_blinded.pdf")
        else:
            canvas.SaveAs(f"{save_fname}_subCat{target_subCat}_unblinded.pdf")
        canvas.Close()
        del canvas

        # --------------------------------------------------------------------
        # KS test
        # --------------------------------------------------------------------
        pdf_l = [
            coreBWZRedux,
            coreSumExp,
            coreSumPow,
            coreBWZGamma,
            coreBWZxBern,
            coreFEWZxBern,
            coreLandxBern,
        ]
        # pdf_cat_name_dict = {
        #     0: coreBWZRedux.GetName(),
        #     1: coreSumExp.GetName(),
        #     2: coreSumPow.GetName(),
        #     3: coreBWZGamma.GetName(),
        #     4: coreBWZxBern.GetName(),
        #     5: coreFEWZxBern.GetName(),
        #     6: coreLandxBern.GetName(),
        # }
        # for i in range(len(pdf_l)):
        for pdf in pdf_l:
            hist_data = roo_histData
            # pdf = pdf_l[i]
            # core_func_name = pdf_cat_name_dict[i]
            core_func_name = pdf.GetName()
            gof_test_name = f"{core_func_name}_cat{target_subCat}"
            gof_save_path = ""
            KS_dict = getGOF_KS(mass, hist_data, pdf, gof_test_name, gof_save_path)
            print(f"KS_dict: {KS_dict}")
            for region, ks_stat_dict in KS_dict.items():
                nevents = ks_stat_dict["nevents"]
                ks_stat = ks_stat_dict["ks_statistic"]
                # alpha = 0.05
                # pass_threshold = 1.358 / (nevents**(0.5))
                alpha = 0.1
                pass_threshold = 1.22385 / (nevents**(0.5))
                
                ks_df.loc[len(ks_df)] = {
                    "pdf category": gof_test_name,
                    "region": region,
                    "KS statistic": ks_stat,
                    "nevents": nevents,
                    "alpha": alpha,
                    "pass threshold": pass_threshold,
                    "test pass": ks_stat<pass_threshold,
                }

    df = pd.DataFrame(df_rows, columns=["BDT cat", "fit function", "chi2ndf"])
    df_pivot = df.pivot(index="BDT cat", columns="fit function", values="chi2ndf")
    print(df)
    print(fitResults)
    if apply_blind:
        df.to_csv(f"{save_fname}_chi2ndf_blinded.csv")
        df_pivot.to_csv(f"{save_fname}_chi2ndfByFitFunc_blinded.csv")
    else:
        df.to_csv(f"{save_fname}_chi2ndf_unblinded.csv")
        df_pivot.to_csv(f"{save_fname}_chi2ndfByFitFunc_unblinded.csv")

    # ks test 
    ks_df.to_csv(f"{save_fname}_KS_stats.csv")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-label",
    "--label",
    dest="label",
    default="",
    action="store",
    help="label",
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
    "-save",
    "--save_path",
    dest="save_path",
    default="plots",
    action="store",
    help="string value production category we're working on",
    )
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="all",
    action="store",
    help="label",
    )
    parser.add_argument(
    "-reg",
    "--region",
    dest="region",
    default="signal",
    action="store",
    help="region value to plot, available regions are: h_peak, h_sidebands, z_peak and signal (h_peak OR h_sidebands)",
    )
    parser.add_argument(
    "--unblind",
    dest="unblind",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="If true, unblind data",
    )
    parser.add_argument(
    "-base",
    "--base_path",
    dest="base_path",
    default="/depot/cms/users/yun79/hmm/copperheadV1clean",
    action="store",
    help="",
    )
    nSubCats = 5
    
    args = parser.parse_args()
    # load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean/{args.label}/{args.category}/stage2_output/*/"
    year = args.year
    if year == "all":
        year_param = "*"
    elif year == "2016":
        year_param = "2016*"
    else:
        year_param = year
    # load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean/{args.label}/{args.category}/stage2_output/{year_param}/"
    load_path =f"{args.base_path}/{args.label}/{args.category}/stage2_output/{year_param}/"
    # events = dak.from_parquet(f"{load_path}/*data.parquet")
    # print(events.fields)
    print(f"load_path : {load_path}")
    lumi_dict = {
        "2018" : 59.83,
        "2017" : 41.48,
        "2016postVFP": 19.50,
        "2016preVFP": 16.81,
        "2016": 36.3,
        # "all" : 137, # Run2
        "2022preEE": "7.9804",
        "2022postEE": "26.6717",
        "2023": "17.7940",
        "2023BPix": "9.4510",
        "2024": "108.9600",
        "all": "170.8571", # 2022 - 2024
    }
    lumi_val = lumi_dict[year]
    sample_groups = {
        "data" : "data*",
    }
    sample_dict = {
        group: {
            "wgt_nominal" : [],
            "dimuon_mass": [],
            "subCategory_idx": [],
        } for group in sample_groups.keys()
    }
    if args.region != "signal":
        print("Error, region is not signal!")
        raise ValueError
    for group, group_fname in sample_groups.items():
        full_load_path = load_path+f"*{group_fname}.parquet" 
        events = dak.from_parquet(full_load_path)
        _, events = filterRegion(events, region=args.region)
        sample_dict = fillSampleValues(events, sample_dict, group)


    # plot_setting_fname = "../../../src/lib/histogram/plot_settings_vbfCat_MVA_input.json"
    plot_setting_fname = "src/lib/histogram/plot_settings_gghCat_BDT_input.json"
    # plot_setting_fname = "plot_settings_vbfCat_MVA_input.json"
    with open(plot_setting_fname, "r") as file:
        plot_settings = json.load(file)
    # plot_var = "BDT_score"
    # binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
    # save_fname = f"plots/{args.label}_x_{args.category}/{args.year}_signal/Fig6_19"
    save_fname = f"{args.save_path}/{args.label}_x_{args.category}/{args.year}_{args.region}/Fig6_19"
    
    # status = "Private"
    status = "Simulation"
    apply_blind= not args.unblind
    sample="data"
    dataDict_by_subCat = getDimuMassBySubCat(sample_dict, sample=sample, nSubCats=nSubCats)
    print(f"sample_dict: {sample_dict}")
    print(f"dataDict_by_subCat: {dataDict_by_subCat}")
    # add "all" category
    dataDict_by_subCat["all"] = {
            "dimuon_mass" : sample_dict[sample]["dimuon_mass"],
            "wgt_nominal" : sample_dict[sample]["wgt_nominal"],
    }
    
    plot_6_19(dataDict_by_subCat, save_fname, apply_blind = apply_blind)
    # plot_6_19(dataDict_by_subCat, save_fname, apply_blind = False)
    

    