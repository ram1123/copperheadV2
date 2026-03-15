import ROOT
import ROOT as rt
from typing import Tuple, List, Dict
ROOT.gStyle.SetOptStat(0) # remove stats box
from modules.GoF_utils import chi2_ndf_manual
import pandas as pd
from modules.utils import logger
import uuid

def getFEWZ_roospline(x, root_path):
    """
    Extract RooSpline1D instance that we assume has been saved in ucsd_workspace/fewz.root
    with the name "fewz_1j_spl_order1_cat_ggh" (which we will keep)
    replace the variable that the RooSpline1D was constructed with, with our own variable "x"
    so fitTo could work with the rest of the roofit pdfs
    """
    # ucsd_spline = rt.TFile("modules/ucsd_workspace/fewz.root")["fewz_1j_spl_order1_cat_ggh"]
    ucsd_spline = rt.TFile(f"{root_path}/fewz.root")["fewz_1j_spl_order1_cat_ggh"]
    ucsd_var = ucsd_spline.getVariables()[0]
    # replace the variable with our variable
    customizer = rt.RooCustomizer(ucsd_spline, "")
    customizer.replaceArg(ucsd_var, x)
    roo_spline_func = customizer.build()
    name = "fewz_1j_spl_order1_cat_ggh"
    roo_spline_func.SetName(name)
    return roo_spline_func



def MakeFEWZxBernDof3(
        name_final:str, 
        title:str, 
        mass: rt.RooRealVar, 
        BernCoeff_list,
    ) ->Tuple[rt.RooProdPdf, Dict]:
    """
    params:
    mass = rt.RooRealVar that we will fitTo
    dof = degrees of freedom given to this model. Since the spline
    has no dof, all the dof is inserted to the Bernstein
    """
    # collect all variables that we don't want destroyed by Python once function ends
    out_dict = {}

    # name = f"BernsteinFast"
    # n_coeffs = len(BernCoeff_list)
    # bern_model = rt.RooBernsteinFast(n_coeffs)(name, name, mass, BernCoeff_list)
    name = f"Bernstein_FEWZxBern"
    bern_model = rt.RooBernstein(name, name, mass, BernCoeff_list) # we assume that we have one extra frozen parameter
    out_dict[name] = bern_model # add model to make python remember


    
    # make the spline portion
    roo_spline_func = getFEWZ_roospline(mass, "ucsd_workspace/")# extract from ucsd 's fews root file
    out_dict[roo_spline_func.GetName()] = roo_spline_func
    name = "fewz_1j_spl_pdf"
    roo_spline_pdf = rt.RooWrapperPdf(name, name, roo_spline_func)
    out_dict[name] = roo_spline_pdf # add model to make python remember  

    final_model = rt.RooProdPdf(name_final, name_final, [bern_model, roo_spline_pdf]) 
   
    return (final_model, out_dict)


def MakeFEWZxBernDof3_BernFast(
        name_final:str, 
        title:str, 
        mass: rt.RooRealVar, 
        # c1: rt.RooRealVar, c2: rt.RooRealVar, c3: rt.RooRealVar
        BernCoeff_list,
    ) ->Tuple[rt.RooProdPdf, Dict]:
    """
    params:
    mass = rt.RooRealVar that we will fitTo
    dof = degrees of freedom given to this model. Since the spline
    has no dof, all the dof is inserted to the Bernstein
    """
    # collect all variables that we don't want destroyed by Python once function ends
    out_dict = {}

    # make BernStein of order == dof
    # n_coeffs = 3
    # BernCoeff_list = [c1, c2, c3]
    # n_coeffs = 1
    # BernCoeff_list = [c1,]
    name = f"BernsteinFast"
    n_coeffs = len(BernCoeff_list)
    bern_model = rt.RooBernsteinFast(n_coeffs)(name, name, mass, BernCoeff_list)
    # bern_model = rt.RooBernstein(name, name, mass, BernCoeff_list)
    out_dict[name] = bern_model # add model to make python remember


    
    # make the spline portion

    # this ROOT files has branches full_36fb, full_xsec, full_shape -> all three has the same shape (same hist once normalized)
    FEWZ_file = rt.TFile("./data/NNLO_Bourilkov_2017.root", "READ")
    FEWZ_histo = FEWZ_file.Get("full_36fb")
    # FEWZ_histo = FEWZ_file.Get("full_shape") # 50 bins in total

    # rebin_factor = 5#5 
    rebin_factor = 2#5 
    FEWZ_histo = FEWZ_histo.Rebin(rebin_factor, "hist_rebinned")
    # FEWZ_data = rt.RooDataHist("fewzdata","fewzdata",mass,FEWZ_histo) # this is RoofitHist data
    
    
    # x_arr, y_arr = getFEWZ_vals(FEWZ_histo)
    

    # # Roospline start ---------------------------------    
    # x_arr_vec = rt.vector("double")(x_arr)
    # y_arr_vec = rt.vector("double")(y_arr)
    # name = "fewz_roospline_func"
    # roo_spline_func = rt.RooSpline(name, name, mass, x_arr_vec, y_arr_vec, order=3)
    # out_dict[name] = roo_spline_func
    # # Roospline end ------------------------------------


    
    roo_spline_func = getFEWZ_roospline(mass, "ucsd_workspace/")# extract from ucsd 's fews root file
    
    out_dict[roo_spline_func.GetName()] = roo_spline_func
    
    # Roospline1D start ---------------------------------    
    # x0 = (ctypes.c_double * len(x_arr))(*x_arr)
    # y0 = (ctypes.c_double * len(y_arr))(*y_arr)
    # n = len(x0)
    # name = "fewz_roospline_func"
    # roo_spline_func = rt.RooSpline1D(name, name, mass, n,x0, y0)
    # out_dict[name] = roo_spline_func
    # Roospline1D end ------------------------------------

    

    # turn roo_spline_func into pdf
    # RooProdPDF seems to automatically normalize the pdfs on https://root.cern.ch/doc/master/classRooProdPdf.html,
    # so no need to "normalize" rooSpline into a PDF. Also, I tried both full_36fb and full_shape bracnhes of 
    # FEWZ histograms (they have same shape, but different values), and I saw no difference in fit function plot,
    # loosely suggesting automatic normalization
    # roo_spline_pdf = roo_spline_func
    
    
    name = "fewz_1j_spl_pdf"
    roo_spline_pdf = rt.RooWrapperPdf(name, name, roo_spline_func)
    out_dict[name] = roo_spline_pdf # add model to make python remember  

    # RooWrapperPdf doesn't seem to work well with FitTo. Just freezes for a long time
    # name = "fewz_roospline_pdf"
    # roo_spline_pdf = rt.RooGenericPdf(name, "@0", rt.RooArgList(roo_spline_func))      
    # out_dict[name] = roo_spline_pdf # add model to make python remember  

    final_model = rt.RooProdPdf(name_final, name_final, [bern_model, roo_spline_pdf]) 
    # final_model = rt.RooGenericPdf(name_final, "@0*@1", rt.RooArgList(roo_spline_pdf, bern_model))  
    # final_model = bern_model
    # final_model = roo_spline_pdf
   
    return (final_model, out_dict)


def getShapeModifierHist(x, allCat_hist, subCat_hist, normalize=False, nbins=100):
    x_name = x.GetName()
    nbins_old = x.getBins()
    nbins_new = nbins
    reBinFactor = int(nbins_old/nbins_new)
    # print(f"nbins_old : {nbins_old}")
    # print(f"nbins_new : {nbins_new}")
    # print(f"reBinFactor : {reBinFactor}")
    allCat_th1 = allCat_hist.createHistogram(x_name).Clone("allCat_clone").Rebin(reBinFactor) # clone it just in case
    subCat_th1 = subCat_hist.createHistogram(x_name).Clone("subCat_clone").Rebin(reBinFactor) # clone it just in case
    subCat_th1.Divide(allCat_th1)
    if normalize:
        subCat_th1.Scale(1/subCat_th1.Integral()) # normalize to one
    rooHist_name = "shapModifier_hist"
    roo_hist_shapModifier = rt.RooDataHist(rooHist_name, rooHist_name, rt.RooArgSet(x), subCat_th1) 
    return roo_hist_shapModifier

def getPdfToHist(x, pdf, hist2copy):
    pdf_hist = hist2copy.Clone("pdf_hist")
    for i in range(1, pdf_hist.GetNbinsX()+1):
        xval = pdf_hist.GetXaxis().GetBinCenter(i)
        x.setVal(xval)
        # get uncertainty on PDF at this point from fit result
        pdf_val = pdf.getVal(ROOT.RooArgSet(x))
        pdf_hist.SetBinContent(i, pdf_val)  # ratio = 1
        pdf_hist.SetBinError(i, 0) # remove errors
    return pdf_hist

# def getRatioHist(x, hist, pdf):
#     ratio_hist = hist.Clone("ratio_hist")
#     for i in range(1, hist.GetNbinsX()+1):
#         xval = ratio_hist.GetXaxis().GetBinCenter(i)
#         x.setVal(xval)
    
#         # get uncertainty on PDF at this point from fit result
#         hist_val = ratio_hist.GetBinContent(i)
#         pdf_val = pdf.getVal(ROOT.RooArgSet(x))
#         print(f"bin {i} hist_val: {hist_val}")
#         print(f"bin {i} pdf_val: {pdf_val}")
#         ratio_hist.SetBinContent(i, hist_val/pdf_val)  # ratio = 1
#         ratio_hist.SetBinError(i, 0) # remove errors
#     return ratio_hist

def getRatioHist(x, hist, pdf):
    hist_clone = hist.Clone("ratio_hist")
    hist_pdf = getPdfToHist(x, pdf, hist)
    # normalize
    hist_clone.Scale(1/hist_clone.Integral())
    hist_pdf.Scale(1/hist_pdf.Integral())
    # for i in range(1, hist_clone.GetNbinsX()+1):
    #        # get uncertainty on PDF at this point from fit result
    #     hist_val = hist_clone.GetBinContent(i)
    #     pdf_val = hist_pdf.GetBinContent(i)
    #     print(f"bin {i} hist_val: {hist_val}")
    #     print(f"bin {i} pdf_val: {pdf_val}")
    ratio_hist = hist_clone
    ratio_hist.Divide(hist_pdf)

    # Style: black dots with error bars
    ratio_hist.SetMarkerStyle(20)        # Filled circle
    ratio_hist.SetMarkerSize(1.0)
    ratio_hist.SetMarkerColor(ROOT.kBlack)
    ratio_hist.SetLineColor(ROOT.kBlack)  # Error bars in black
    return ratio_hist


def getUnityHistBand(x, pdf, fitResult, hist2copy):
    """
    from roofit histogram, generate a histogram with value one with relative fit errors from pdf and paste them in the same TH1 format as hist2copy
    """
    h_band = hist2copy.Clone("h_band")
    for i in range(1, h_band.GetNbinsX()+1):
        xval = h_band.GetXaxis().GetBinCenter(i)
        x.setVal(xval)
    
        # get uncertainty on PDF at this point from fit result
        val = pdf.getVal(ROOT.RooArgSet(x))
        err = pdf.getPropagatedError(fitResult)
    
        # rel_err = err / val if val != 0 else 0
        rel_err = err
        h_band.SetBinContent(i, 1.0)  # ratio = 1
        h_band.SetBinError(i, rel_err)
        # print(f"bin {i} rel_err: {rel_err}")
        # print(f"bin {i} val: {val}")
        # print(f"bin {i} err: {err}")
        

    # Style
    h_band.SetFillColor(ROOT.kBlue - 9)
    h_band.SetMarkerSize(0)
    h_band.SetLineWidth(0)
    return h_band

def zero_yield_in_range(hist, x_min, x_max):
    """
    Set the yield (bin content) of a TH1 histogram to zero
    for bins whose centers lie within [x_min, x_max].

    Parameters
    ----------
    hist : ROOT.TH1
        Input histogram (e.g., TH1F)
    x_min : float
        Lower bound of x range
    x_max : float
        Upper bound of x range
    """

    for i in range(1, hist.GetNbinsX() + 1):
        x = hist.GetBinCenter(i)

        if x_min <= x <= x_max:
            hist.SetBinContent(i, 0.0)
            hist.SetBinError(i, 0.0)

    return hist

def plot_6_23(x, roo_histData_allCat, subCat_dataHists, SMF_pdf_l, fitResult, save_fname, normalize=True, nbins=100, y_range_l=None, blinded=True, fit_range="hiSB,loSB"):
    # normalize=False
    x_name = x.GetName()
    for ix in range(len(subCat_dataHists)):
        name = f"Canvas_{ix}"        
        canvas = rt.TCanvas(name, name, 800, 800) # giving a specific name for each canvas prevents segfault
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

        # Top pad start
        pad1.cd()
        legend = rt.TLegend(0.65,0.75,0.9,0.9)
        frame = x.frame()
        roo_hist_shapModifier = getShapeModifierHist(x, roo_histData_allCat, subCat_dataHists[ix], normalize=normalize, nbins=nbins)

        # plot the SMF fit function first
        roo_hist_shapModifier.plotOn(frame, Invisible=True) # Invisible plot for SMF functions to plot over
        SMF_pdf = SMF_pdf_l[ix]
        SMF_pdf.plotOn(frame, VisualizeError=(fitResult, 1), FillColor=(ROOT.kBlue - 9), Components=SMF_pdf.GetName()) # don't need the specify component name, but I guess it's good practice
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),"Uncertainty", "F")
        
        SMF_pdf.plotOn(frame, LineColor=rt.kRed)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),"Polynomial fit", "L")
        
        # plot the shape modifier data
        if blinded==True:
            roo_hist_shapModifier.plotOn(frame, ROOT.RooFit.CutRange(fit_range))
        else:
            roo_hist_shapModifier.plotOn(frame)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),"Shape Modifier", "PE")

        # plot settings
        frame.Draw()
        legend.Draw()
        # frame.GetYaxis().SetLabelSize(0.08)
        # frame.GetYaxis().SetRangeUser(0.98, 1.02)
        if normalize:
            frame.GetYaxis().SetTitle("A.U.")
        else:
            frame.GetYaxis().SetTitle("Events")
        frame.SetTitle("")
        if y_range_l is not None:
            x_min, x_max = y_range_l[ix]
            frame.GetYaxis().SetRangeUser(x_min, x_max)
        
        # Bottom pad start
        pad2.cd()
        ratio_frame= x.frame()

        # SMF_pdf_hist = SMF_pdf.createHistogram(x.GetName()).Clone("SMF_pdf_clone")
        # ratio_hist = roo_hist_shapModifier.createHistogram(x.GetName()).Clone("ratio_hist")
        # ratio_hist.Scale(SMF_pdf_hist.Integral() / ratio_hist.Integral())  # normalize
        # ratio_hist.Divide(SMF_pdf_hist)

        shapeModifier_hist = roo_hist_shapModifier.createHistogram(x.GetName())
        ratio_hist = getRatioHist(x, shapeModifier_hist, SMF_pdf)
        ratio_hist.GetYaxis().SetRangeUser(0.9, 1.1)
        ratio_hist.SetTitle("")
        ratio_hist.GetYaxis().SetTitle("Data/Pred")
        ratio_hist.GetXaxis().SetTitle("m_{\mu\mu} [GeV]")
        
            
        h_band = getUnityHistBand(x, SMF_pdf, fitResult, ratio_hist)

        # start draw
        h_band.Draw("E2") # draw h band first
        # change style to add a straight red line
        h_band_line = h_band.Clone("h_bandClone")
        h_band_line.SetLineColor(ROOT.kRed)
        h_band_line.SetLineWidth(2)
        h_band_line.SetFillStyle(0)  # No fill
        h_band_line.Draw("HIST L SAME")

        if blinded:
            blinded_x_min, blinded_x_max = 115, 135
            ratio_hist = zero_yield_in_range(ratio_hist, blinded_x_min, blinded_x_max)
        ratio_hist.Draw("E1 SAME")

        canvas.Update()
        canvas.Draw()
        canvas.SaveAs(f"{save_fname}_subCat{ix}.pdf")
        canvas.Close()
        del canvas


def getSigBkgPdf(bkg_pdf_dict, sig_pdf_dict, nSubCats=5):
    parameters = []
    sim_sigBkg_pdf = {}
    for ix in range(nSubCats):
        name = f"frac_subCat{ix}_BWZRedux"
        frac_BWZRedux = rt.RooRealVar(name,name,0.01, 0.0, 1.0) 
        name = f"frac_subCat{ix}_SumExp"
        frac_SumExp = rt.RooRealVar(name,name,0.01, 0.0, 1.0) 
        name = f"frac_subCat{ix}_FEWZxBern"
        frac_FEWZxBern = rt.RooRealVar(name,name,0.01, 0.0, 1.0) 

        frac_SumExp = frac_BWZRedux
        frac_FEWZxBern = frac_BWZRedux

        bwz_redux = bkg_pdf_dict[f"subCat{ix}_BWZRedux"]
        sum_exp = bkg_pdf_dict[f"subCat{ix}_sumExp"]
        fewzXbern = bkg_pdf_dict[f"subCat{ix}_FEWZxBern"]
        signal_ggh = sig_pdf_dict[f"signal_subCat{ix}"]
        
        name = f"sigBkg_subCat{ix}_BWZRedux"
        sigBkg_BWZRedux = rt.RooAddPdf(name, name, [signal_ggh, bwz_redux], [frac_BWZRedux])
        name = f"sigBkg_subCat{ix}_sumExp"
        sigBkg_sumExp = rt.RooAddPdf(name, name, [signal_ggh, sum_exp], [frac_SumExp])
        name = f"sigBkg_subCat{ix}_FEWZxBern"
        sigBkg_FEWZxBern = rt.RooAddPdf(name, name, [signal_ggh, fewzXbern], [frac_FEWZxBern])
        
        # parameters.append(frac)
        parameters = parameters + [
            frac_BWZRedux,
            frac_SumExp,
            frac_FEWZxBern
        ]
        
        sim_sigBkg_pdf[f"subCat{ix}_BWZRedux"] = sigBkg_BWZRedux
        sim_sigBkg_pdf[f"subCat{ix}_sumExp"] = sigBkg_sumExp
        sim_sigBkg_pdf[f"subCat{ix}_FEWZxBern"] = sigBkg_FEWZxBern

    return sim_sigBkg_pdf, parameters
        

def rebinHist(x, roofitHist,  nbins, normalize=False,):
    x_name = x.GetName()
    nbins_old = x.getBins()
    nbins_new = nbins
    reBinFactor = int(nbins_old/nbins_new)
    # print(f"nbins_old : {nbins_old}")
    # print(f"nbins_new : {nbins_new}")
    # print(f"reBinFactor : {reBinFactor}")
    roofit_th1 = roofitHist.createHistogram(x_name).Clone("subCat_clone").Rebin(reBinFactor) # clone it just in case
    if normalize:
        roofit_th1.Scale(1/roofit_th1.Integral()) # normalize to one
    rooHist_name = roofitHist.GetName() + f"rebinned_{nbins}"
    roofitHist_rebinned = rt.RooDataHist(rooHist_name, rooHist_name, rt.RooArgSet(x), roofit_th1) 
    return roofitHist_rebinned

def get_pdf_by_name(add_pdf, name):
    """
    Extracts a sub-PDF from a RooAddPdf by its name.

    Args:
        add_pdf (ROOT.RooAddPdf): The RooAddPdf instance.
        name (str): The name of the sub-PDF to extract.

    Returns:
        ROOT.RooAbsPdf or None: The extracted RooAbsPdf if found, otherwise None.
    """
    if not isinstance(add_pdf, ROOT.RooAddPdf):
        print("Warning: Input is not a RooAddPdf instance, just returning the pdf")
        return add_pdf

    # Get the list of component PDFs
    pdf_list = add_pdf.pdfList()

    # Iterate through the list and find the PDF by name
    # In PyROOT, you can iterate directly over RooArgList
    for i in range(pdf_list.getSize()):
        current_pdf = pdf_list.at(i) # Use at(i) to get the element
        if current_pdf and current_pdf.GetName() == name:
            return current_pdf

    print(f"Warning: PDF with name '{name}' not found in RooAddPdf '{add_pdf.GetName()}'.")
    return None

def get_fracFromAddPdf(add_pdf, frac_name):
    """
    Extracts the fraction (yield) RooAbsReal for a specific sub-PDF by its name
    from a RooAddPdf.
    """
    if not isinstance(add_pdf, ROOT.RooAddPdf):
        print("Error: Input is not a RooAddPdf instance.")
        return None

    coeff_list = add_pdf.coefList()

    for i in range(coeff_list.getSize()):
        current_frac = coeff_list.at(i) # Use at(i) to get the element
        if current_frac.GetName() == frac_name:
            return current_frac

    else:
        print(f"Warning: coeff with name '{frac_name}' not found in RooAddPdf '{add_pdf.GetName()}'.")
        return None


# def getResidHistBand(x, pdf, fitResult, hist2copy):
#     """
#     from roofit histogram, generate a histogram with value one with relative fit errors from pdf and paste them in the same TH1 format as hist2copy
#     """
#     h_band = hist2copy.Clone("h_band")
#     for i in range(1, h_band.GetNbinsX()+1):
#         xval = h_band.GetXaxis().GetBinCenter(i)
#         x.setVal(xval)
    
#         # get uncertainty on PDF at this point from fit result
#         val = pdf.getVal(ROOT.RooArgSet(x))
#         err = pdf.getPropagatedError(fitResult)
        
#         rel_err = err * val 
#         hist_val = hist2copy.GetBinContent(i) 
#         h_band.SetBinContent(i, 0.0)  # ratio = 1
#         h_band.SetBinError(i, rel_err*2*hist_val)
#         print(f"bin {i} rel_err: {rel_err}")
#         print(f"bin {i} val: {val}")
#         print(f"bin {i} err: {err}")
#         print(f"bin {i} hist_val: {hist_val}")
        

#     # Style
#     h_band.SetFillColor(ROOT.kOrange)
#     h_band.SetMarkerSize(0)
#     h_band.SetLineWidth(0)
#     # raise ValueError
#     return h_band


def getResidHistBand(x, pdf, fitResult, dataHist, n_sigma=1, color=rt.kGreen):
    """
    Source: https://root-forum.cern.ch/t/problems-with-errors-for-residhist/51455/5
    """
    nbins=dataHist.numEntries() # match the nbins from dataHist
    old_nbins = x.getBins()
    x.setBins(nbins) 
    # h_band = hist2copy.Clone("h_band")
    nBkg = rt.RooRealVar("nBkg", "nBkg", 5000, 0, 10000)
    binning = x.getBinning()
    h_band = dataHist.createHistogram(x.GetName()).Clone("h_band")
    for i in range(dataHist.numEntries()):
        # xval = h_band.GetXaxis().GetBinCenter(i)
        # x.setVal(xval)
        x.setRange("range_for_bin", binning.binLow(i), binning.binHigh(i))
        bkgPdfIntegral = pdf.createIntegral(x, rt.RooFit.NormSet(x), rt.RooFit.Range("range_for_bin"))
        bkgYield = rt.RooProduct("bkgYield", "bkgYield", [bkgPdfIntegral, nBkg])
        one_sigma_err = bkgYield.getPropagatedError(fitResult)
        # print(f"bin {i} dataHist->weight(): {dataHist.weight()}")
        # print(f"bin {i} dataHist->weightError(): {dataHist.weightError()}")
        # print(f"bin {i} bkgYield.getPropagatedError(fitResult): {one_sigma_err}")
        # print(f"bin {i} binning.binLow(i): {binning.binLow(i)}")
        # print(f"bin {i} binning.binHigh(i): {binning.binHigh(i)}")

        h_band.SetBinContent(i+1, 0.0) 
        h_band.SetBinError(i+1, one_sigma_err*n_sigma)

    # Style
    h_band.SetFillColor(color)
    h_band.SetMarkerSize(0)
    h_band.SetLineWidth(0)

    # convert to RooDataHist
    # x_name = x.GetName()
    # h_band = rt.RooDataHist(x_name, x_name, rt.RooArgSet(x), h_band) 

    # for i in range(h_band.numEntries()):
    #     coord = h_band.get(i)  # returns RooArgSet
    #     yval = h_band.weight(i)  # bin content
    #     yerr = h_band.weightError(i)  # bin error
    #     print(f"Bin {i}:val = {yval:.2f} ± {yerr:.2f}")


    x.setBins(old_nbins) 
    # print(f"old_nbins: {old_nbins}")
    # raise ValueError
    return h_band

# def turnRooHist2RooDataHist(roo_hist, x):
#     # extract the TH1
#     th1 = roo_hist.GetHistogram()     # a TH1*
#     print(f"th1.Integral(): {th1.Integral()}")
#     # now build a RooDataHist out of it
#     x_name = x.GetName()
#     roo_dataHist = rt.RooDataHist(
#         x_name,                # name
#         x_name, 
#         rt.RooArgList(x),       # list of observables (must match the axis of th1)
#         th1                  # the TH1 to import
#     )
#     return roo_dataHist


def plot_6_26(x, subCat_dataHists, multi_pdf_l, fitResult, save_fname, target_nbins=50, coreFuncName="", unblind=False, applyBkgCompName=True, nFit_params=3):
    # target_nbins = 100
    target_nbins = 80 # NOTE: the plots are sensitve to rebins
    x_name = x.GetName()
    sig_yield_multiply_l = [
        50,
        50,
        30,
        30,
        20
    ]
    if unblind:
        fit_range = "full"
        plot_range = "full"
    else:
        fit_range = "loSB,hiSB"
        plot_range = "full"
    df_dict = {
        "bdt_cat" : [],
        "chi2/ndf" : [],
    }
    for ix in range(len(subCat_dataHists)):
        name = f"canvas_{ix}"
        canvas = rt.TCanvas(name, name, 800, 800) # giving a specific name for each canvas prevents segfault
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

        # Top pad start
        pad1.cd()
        legend = rt.TLegend(0.55,0.65,0.9,0.9)
        frame = x.frame()
        subCat_dataHist = subCat_dataHists[ix]
         
        
        multi_pdf = multi_pdf_l[ix]
        if applyBkgCompName:
            bkg_pdf_name = f"model_SubCat{ix}_SMFx{coreFuncName}"
        else:
            bkg_pdf_name = multi_pdf.GetName()
        
        # get chi2
        if bkg_pdf_name != multi_pdf.GetName():
            comps = multi_pdf.getComponents()
            bkg_comp = comps.find(bkg_pdf_name)
        else:
            bkg_comp = multi_pdf
        chi2_regions = [(110,115), (135,150)] # h sidebands
        chi2, NDF = chi2_ndf_manual(bkg_comp, subCat_dataHist, x, chi2_regions, nFit_params)
        df_dict["bdt_cat"].append(ix)
        df_dict["chi2/ndf"].append(chi2/NDF)

        # plot
        subCat_dataHist = rebinHist(x, subCat_dataHist, target_nbins) # rebin
        subCat_dataHist.plotOn(frame, Invisible=True)
        # bkg_pdf_name = f"model_SubCat{ix}_SMFxBWZRedux"
        
        multi_pdf.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range(plot_range), Components=bkg_pdf_name, Invisible=True) 
        hresid_bkg_only = frame.residHist() # obtain residual for later

        multi_pdf.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range(plot_range), VisualizeError=(fitResult, 2), FillColor=(ROOT.kOrange), Components=bkg_pdf_name) 
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),"+/ 2\sigma", "F")
        
        multi_pdf.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range(plot_range), VisualizeError=(fitResult, 1), FillColor=(ROOT.kGreen), Components=bkg_pdf_name) 
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),"+/ 1\sigma", "F")
        
        multi_pdf.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range(plot_range), LineColor=rt.kRed, LineWidth=2, Components=bkg_pdf_name, LineStyle=rt.kDashed)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),"Fitted background", "L")
        
        # sig_frac = get_fracFromAddPdf(add_pdf, f"frac_subCat{ix}")
        # sig_frac.Print("v")
        # original_frac_val = sig_frac.getVal()
        # multipy_val = sig_yield_multiply_l[ix]
        # sig_frac.setVal(original_frac_val*multipy_val)
        # multi_pdf.plotOn(frame, LineColor=rt.kBlue, LineWidth=2, Components=f"ggH_cat{ix}_ggh_pdf")

        
        if unblind:
            multi_pdf.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range(plot_range), LineColor=rt.kRed, LineWidth=2)
            legend.AddEntry(frame.getObject(int(frame.numItems())-1),"S+B fit", "L")
            
            add_pdf = multi_pdf
            sig_frac = get_fracFromAddPdf(add_pdf, f"frac_subCat{ix}_BWZRedux")
            # sig_frac = get_fracFromAddPdf(add_pdf, f"frac_subCat{ix}_{coreFuncName}")
            # sig_frac.Print("v")
            original_frac_val = sig_frac.getVal()
            multipy_val = sig_yield_multiply_l[ix]
            sig_frac.setVal(original_frac_val*multipy_val)
            multi_pdf.plotOn(frame, LineColor=rt.kBlue, LineWidth=2, Components=f"ggH_cat{ix}_ggh_pdf")
            legend.AddEntry(frame.getObject(int(frame.numItems())-1),f"Post-fit signal x {multipy_val}, m_H = 125 GeV", "L")

        subCat_dataHist.plotOn(frame, rt.RooFit.CutRange(fit_range))
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),"Data", "PE")
        
        frame.Draw()
        legend.Draw()

        
        # print(f"subCat {ix} {coreFuncName} dataHist sumentries: {subCat_dataHist.sumEntries()}")
        # print(f"subCat {ix} {coreFuncName} signal yield: {subCat_dataHist.sumEntries()*original_frac_val}")
        # done with pad1
        
        # Bottom pad start
        pad2.cd()

        
        # get dummy frame and get blinded residual
        dummy_frame = x.frame()
        subCat_dataHist.plotOn(dummy_frame, Invisible=True)
        multi_pdf.plotOn(dummy_frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range(fit_range), Components=bkg_pdf_name) 
        # multi_pdf.plotOn(dummy_frame, Components=bkg_pdf_name) 
        subCat_dataHist.plotOn(dummy_frame, rt.RooFit.CutRange(fit_range))
        
        cut_hresid_bkg_only = dummy_frame.residHist()
        


        # get residual plot frame
        frame_resid = x.frame()
        frame_resid.addPlotable(hresid_bkg_only, "P", invisible=True)
        frame_resid.Draw() # draw invisible residual to set y range in pad2
        
        # set fraction back to normal
        if unblind:
            sig_frac.setVal(original_frac_val)

        bkg_pdf = get_pdf_by_name(multi_pdf, bkg_pdf_name)
        # print(f"bkg_pdf.GetName(): {bkg_pdf.GetName()}")
        # raise ValueError
        bkgOnly_resid_pdf = rt.RooGenericPdf("bkg_resid_pdf", "@0-@0", rt.RooArgList(bkg_pdf))
        # bkg_resid_pdf.plotOn(frame, VisualizeError=(fitResult, 2), FillColor=(ROOT.kOrange)) 
        # bkg_resid_pdf.plotOn(frame, VisualizeError=(fitResult, 1), FillColor=(ROOT.kGreen)) 
        
        bkgOnly_resid_pdf.plotOn(frame_resid, LineColor=rt.kRed, LineStyle=rt.kDashed, LineWidth=2)

        # multi_pdf.Print("V")
        # sigBkg_resid_pdf = rt.RooGenericPdf("sigBkg_resid_pdf", "@0-@1", rt.RooArgList(multi_pdf,bkg_pdf))

        sig_pdf_name = f"ggH_cat{ix}_ggh_pdf"
        sigBkg_resid_pdf = get_pdf_by_name(multi_pdf, sig_pdf_name)
        if unblind:
            sigBkg_resid_pdf.plotOn(frame_resid, LineColor=rt.kRed, LineStyle=rt.kSolid, LineWidth=2)

        # Get the Erro bands
        # h_band_sig2 = getResidHistBand(x, bkg_pdf, fitResult, subCat_dataHist, n_sigma=2, color=rt.kOrange)
        # h_band_sig1 = getResidHistBand(x, bkg_pdf, fitResult, subCat_dataHist, n_sigma=1, color=rt.kGreen)
        h_band_sig2 = getResidHistBand(x, multi_pdf, fitResult, subCat_dataHist, n_sigma=2, color=rt.kOrange)
        h_band_sig1 = getResidHistBand(x, multi_pdf, fitResult, subCat_dataHist, n_sigma=1, color=rt.kGreen)
        
        # plot the residual data points again, but visible this time
        # frame_resid.addPlotable(hresid_bkg_only, "P", )
        frame_resid.addPlotable(cut_hresid_bkg_only, "P", )
        
        
        # draw 
        h_band_sig2.Draw("E2 SAME")
        h_band_sig1.Draw("E2 SAME")
        frame_resid.Draw("SAME")
        # frame_resid.Draw()
        
        # done with pad2
        
        canvas.Update()
        canvas.Draw()
        if unblind:
            canvas.SaveAs(f"{save_fname}_{coreFuncName}_subCat{ix}_unblinded.pdf")
        else:
            canvas.SaveAs(f"{save_fname}_{coreFuncName}_subCat{ix}_blinded.pdf")
        canvas.Close()
        del canvas
    # fitResult.Print()
    return pd.DataFrame(df_dict)


# -----------------------------------------------------
# bias test
# -----------------------------------------------------

def getBWZ_gamma(x, init_param_dict):
    name = f"bwzgamma_BWZ_a_coeff"
    a_coeff_bwz = rt.RooRealVar(name,name, init_param_dict[name],-0.5,0.5)
    # a_coeff_bwz = rt.RooRealVar(name,name, init_param_dict[name],-0.05,0.05)
    name = "BWZ"
    BWZ = rt.RooModZPdf(name, name, x, a_coeff_bwz) 

    name = f"bwzgamma_Gamma_a_coeff"
    # a_coeff_gamma = rt.RooRealVar(name,name, init_param_dict[name],-0.05,0.05)
    a_coeff_gamma = rt.RooRealVar(name,name, init_param_dict[name],-0.5,0.5)
    gamma = rt.RooGenericPdf("Gamma", "exp(@1*@0)/pow(@0,2)", rt.RooArgList(x, a_coeff_gamma))

    name = f"bwzgamma_frac"
    frac = rt.RooRealVar(name,name, init_param_dict[name], 0.0, 1.0) 
    name = "BWZGamma"
    coreBWZGamma = rt.RooAddPdf(name, name, [BWZ, gamma], [frac])
    coreBWZGamma.fixCoefNormalization(rt.RooArgSet(x)) # I get this . Use RooAddPdf::fixCoefNormalization(nset) to provide a normalization set for defining uniquely RooAddPdf coefficients! error other wise
    param_l = [
        # a_coeff_bwz,
        BWZ,
        # a_coeff_gamma,
        gamma,
        # frac,
        {
            "2freeze": [a_coeff_bwz, a_coeff_gamma, frac]
        }
    ]# list of variables to return so that they don't get deleted in python functions. Otherwise the roofit pdfs don't work
    return coreBWZGamma, param_l

def getPowerLaw(x, init_param_dict):
    name = f"RooSumTwoPowerLawPdf_a1_coeff"
    a1_coeff_pow = rt.RooRealVar(name,name, init_param_dict[name],-3.5,-1.0)
    name = f"RooSumTwoPowerLawPdf_a2_coeff"
    a2_coeff_pow = rt.RooRealVar(name,name, init_param_dict[name],-10.0, -3)
    name = f"RooSumTwoPowerLawPdf_f_coeff"
    f_coeff_pow = rt.RooRealVar(name,name, init_param_dict[name],0.0,1.0)
    pow_pdf1 = rt.RooGenericPdf("pow_pdf1", "pow(@0, @1)", rt.RooArgList(x, a1_coeff_pow))
    pow_pdf2 = rt.RooGenericPdf("pow_pdf2", "pow(@0, @1)", rt.RooArgList(x, a2_coeff_pow))
    name = "S-Power-Law"
    coreSumPow = rt.RooAddPdf(name, name, [pow_pdf1, pow_pdf2], [f_coeff_pow])
    param_l = [
        a1_coeff_pow,
        a2_coeff_pow,
        f_coeff_pow,
        pow_pdf1,
        pow_pdf2,
    ]
    return coreSumPow, param_l
    

def getBWZxBern(x, init_param_dict):
    name = f"BWZxBern_a_coeff"
    a_coeff_bwz = rt.RooRealVar(name,name, init_param_dict[name], -0.03, -0.001)

    # we use RooModZPdf, so all bernstein coeffs are freely floating
    name = f"bwz_bernstein_a0" 
    a0_bern = rt.RooRealVar(name,name, init_param_dict[name], 0.15, 0.5)
    name = f"bwz_bernstein_a1"
    a1_bern = rt.RooRealVar(name,name, init_param_dict[name], 0.015, 0.05) # starting value = 1/n_coeffs

    name = "BWZxBernstein"
    coreBWZxBern = rt.RooModZPdf(name, name, x, a_coeff_bwz, rt.RooArgList(a0_bern,a1_bern)) 

    param_l = [
        # a_coeff_bwz,
        # a0_bern,
        # a1_bern,
        {
            "2freeze": [a_coeff_bwz, a0_bern, a1_bern]
        }
    ]# list of variables to return so that they don't get deleted in python functions. Otherwise the roofit pdfs don't work
    return coreBWZxBern, param_l


def getLandxBern(x, init_param_dict):
    name = f"landau_m_Z"
    m_Z = rt.RooRealVar(name,name, 91.2)
    name = f"landau_a_coeff"
    # a_coeff = rt.RooRealVar(name, name, init_param_dict[name],0.0,3)
    a_coeff = rt.RooRealVar(name, name, init_param_dict[name],1.0,3)
    name = "Landau"
    Landau = rt.RooLandau(name, name, x, m_Z, a_coeff) 

    name = f"landau_bernstein_a0"
    a0_bern = rt.RooRealVar(name,name, 1.0) # first term being constant
    name = f"landau_bernstein_a1"
    # a1_bern = rt.RooRealVar(name, name, init_param_dict[name], 0, 5) # starting value = 1/n_coeffs
    a1_bern = rt.RooRealVar(name, name, init_param_dict[name], 0.5, 5) # starting value = 1/n_coeffs
    name = f"landau_bernstein_a2"
    # a2_bern = rt.RooRealVar(name, name, init_param_dict[name], 0.0, 10) # starting value = 1/n_coeffs
    a2_bern = rt.RooRealVar(name, name, init_param_dict[name], 0.1, 5) # starting value = 1/n_coeffs

   #    --------------------  ------------  --------------------------  --------
   #      landau_a_coeff    3.2148e-06    3.6326e-06 +/-  2.14e-03  <none>
   # landau_bernstein_a1    1.2154e+00    1.2154e+00 +/-  6.50e-02  <none>
   # landau_bernstein_a2    1.2806e+00    1.2806e+00 +/-  7.49e-02  <none>
    
    # bern_pol = rt.RooBernsteinFast(3)(name, name, x, rt.RooArgList(a0_bern, a1_bern, a2_bern)) 
    # name = "LandauxBernstein"
    name = "bernstein_landau"
    bern_pol = rt.RooBernstein(name, name, x, rt.RooArgList(a0_bern, a1_bern, a2_bern)) # extra parameter is needed https://root-forum.cern.ch/t/roobernstein-correction/41800

    name = "LandauxBernstein"
    # coreLandxBern = rt.RooProdPdf(name, name, [Landau, bern_pol])
    coreLandxBern = rt.RooProdPdf(name, name, [bern_pol, Landau])

    param_l = [
        m_Z,
        # a_coeff,
        Landau,
        a0_bern,
        # a1_bern,
        # a2_bern,
        bern_pol,
        {
            "2freeze": [a_coeff, a1_bern, a2_bern]
        }
    ]# list of variables to return so that they don't get deleted in python functions. Otherwise the roofit pdfs don't work
    return coreLandxBern, param_l


def getFEWZxBern(x, init_param_dict, fewz_workspace_path="modules/ucsd_workspace/"):
    roo_spline_func = getFEWZ_roospline(x, fewz_workspace_path)

    name = "fewz_1j_spl_pdf"
    roo_spline_pdf = rt.RooWrapperPdf(name, name, roo_spline_func)

    name = f"fewz_bernstein_a0"
    a0_bern = rt.RooRealVar(name,name, 1.0) # first term being constant
    name = f"fewz_bernstein_a1"
    a1_bern = rt.RooRealVar(name,name, init_param_dict[name], 0, 5) 
    name = f"fewz_bernstein_a2"
    a2_bern = rt.RooRealVar(name,name, init_param_dict[name], 0.0, 10) 
    name = f"fewz_bernstein_a3"
    a3_bern = rt.RooRealVar(name,name, init_param_dict[name], 0.0, 10) 
    
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