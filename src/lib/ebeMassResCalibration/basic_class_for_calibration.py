"""
Collection of basic functions  for the mass resolution calibration
"""
import numpy as np
import ROOT as rt
import time
import pandas as pd
import matplotlib.pyplot as plt
import json

# surpress RooFit printout
rt.RooMsgService.instance().setGlobalKillBelow(rt.RooFit.ERROR)

def filter_region(events, region="h-peak"):
    dimuon_mass = events.dimuon_mass
    if region == "h-peak":
        region_filter = (dimuon_mass > 115.03) & (dimuon_mass < 135.03)
    elif region == "h-sidebands":
        region_filter = ((dimuon_mass > 110) & (dimuon_mass < 115.03)) | ((dimuon_mass > 135.03) & (dimuon_mass < 150))
    elif region == "signal":
        region_filter = (dimuon_mass >= 110) & (dimuon_mass <= 150.0)
    elif region == "z-peak" or region == "z_peak":
        region_filter = (dimuon_mass >= 76) & (dimuon_mass <= 106.0)
        # region_filter = (dimuon_mass >= 80) & (dimuon_mass <= 100.0)
        # region_filter = (dimuon_mass >= 70) & (dimuon_mass <= 110.0)
    return events[region_filter]


# Define the calibration categories ---
def get_calib_categories(events):
    """
    Returns a dictionary of 30 boolean masks based on muon1_eta, muon2_eta and mu1_pt.
    Eta bins:
      B: |eta| <= 0.9
      O: 0.9 < |eta| <= 1.8
      E: 1.8 < |eta| <= 2.4
    pT bins for mu1_pt:
      Bin1: (30, 45]
      Bin2: (45, 52]
      Bin3: (52, 62]
      Bin4: (62, 200]

    For the lowest pT bin, the eta combinations are merged into three groups.
    For the other bins, each of the nine eta combinations is kept separately.
    Referece: HIG-19-006, AN-19-124
    """
    BB = ((np.abs(events["mu1_eta"])<=0.9) & (np.abs(events["mu2_eta"])<=0.9))
    BO = ((np.abs(events["mu1_eta"])<=0.9) & ((np.abs(events["mu2_eta"])>0.9) & (np.abs(events["mu2_eta"]) <=1.8)))
    BE = ((np.abs(events["mu1_eta"])<=0.9) & ((np.abs(events["mu2_eta"])>1.8) & (np.abs(events["mu2_eta"]) <=2.4)))
    OB = (((np.abs(events["mu1_eta"])>0.9) & (np.abs(events["mu1_eta"])<=1.8)) & (np.abs(events["mu2_eta"])<=0.9))
    OO = (((np.abs(events["mu1_eta"])>0.9) & (np.abs(events["mu1_eta"])<=1.8)) & ((np.abs(events["mu2_eta"])>0.9) & (np.abs(events["mu2_eta"])<=1.8)))
    OE = (((np.abs(events["mu1_eta"])>0.9) & (np.abs(events["mu1_eta"])<=1.8)) & ((np.abs(events["mu2_eta"])>1.8) & (np.abs(events["mu2_eta"])<=2.4)))
    EB = (((np.abs(events["mu1_eta"])>1.8) & (np.abs(events["mu1_eta"])<=2.4)) & (np.abs(events["mu2_eta"])<=0.9))
    EO = (((np.abs(events["mu1_eta"])>1.8) & (np.abs(events["mu1_eta"])<=2.4)) & ((np.abs(events["mu2_eta"])>0.9) & (np.abs(events["mu2_eta"])<=1.8)))
    EE = (((np.abs(events["mu1_eta"])>1.8) & (np.abs(events["mu1_eta"])<=2.4)) & ((np.abs(events["mu2_eta"])>1.8) & (np.abs(events["mu2_eta"])<=2.4)))

    # pT bins for mu1_pt
    mask_30_45 = (events["mu1_pt"] > 30) & (events["mu1_pt"] <= 45)
    mask_45_52 = (events["mu1_pt"] > 45) & (events["mu1_pt"] <= 52)
    mask_52_62 = (events["mu1_pt"] > 52) & (events["mu1_pt"] <= 62)
    mask_62_200 = (events["mu1_pt"] > 62) & (events["mu1_pt"] <= 200)

    # For pT bin 30-45, group the eta combinations into three categories.
    cat_30_45_1 = mask_30_45 & (BB | OB | EB)
    cat_30_45_2 = mask_30_45 & (BO | OO | EO)
    cat_30_45_3 = mask_30_45 & (BE | OE | EE)

    # For the remaining bins, each eta combination is its own category.
    cats_45_52 = {
        "45-52_BB": mask_45_52 & BB,
        "45-52_BO": mask_45_52 & BO,
        "45-52_BE": mask_45_52 & BE,
        "45-52_OB": mask_45_52 & OB,
        "45-52_OO": mask_45_52 & OO,
        "45-52_OE": mask_45_52 & OE,
        "45-52_EB": mask_45_52 & EB,
        "45-52_EO": mask_45_52 & EO,
        "45-52_EE": mask_45_52 & EE,
    }

    cats_52_62 = {
        "52-62_BB": mask_52_62 & BB,
        "52-62_BO": mask_52_62 & BO,
        "52-62_BE": mask_52_62 & BE,
        "52-62_OB": mask_52_62 & OB,
        "52-62_OO": mask_52_62 & OO,
        "52-62_OE": mask_52_62 & OE,
        "52-62_EB": mask_52_62 & EB,
        "52-62_EO": mask_52_62 & EO,
        "52-62_EE": mask_52_62 & EE,
    }

    cats_62_200 = {
        "62-200_BB": mask_62_200 & BB,
        "62-200_BO": mask_62_200 & BO,
        "62-200_BE": mask_62_200 & BE,
        "62-200_OB": mask_62_200 & OB,
        "62-200_OO": mask_62_200 & OO,
        "62-200_OE": mask_62_200 & OE,
        "62-200_EB": mask_62_200 & EB,
        "62-200_EO": mask_62_200 & EO,
        "62-200_EE": mask_62_200 & EE,
    }

    categories = {
        "30-45_BB_OB_EB": cat_30_45_1,
        "30-45_BO_OO_EO": cat_30_45_2,
        "30-45_BE_OE_EE": cat_30_45_3
    }
    categories.update(cats_45_52)
    categories.update(cats_52_62)
    categories.update(cats_62_200)

    return categories

def generateVoigtian_plot(mass_arr, cat_idx: int, nbins, df_fit, logfile="CalibrationLog.txt", out_string=""):
    """
    params
    mass_arr: numpy arrary of dimuon mass value to do calibration fit on
    cat_idx: int index of specific calibration category the mass_arr is from
    """
    # if you want TCanvas to not crash, separate fitting and drawing
    canvas = rt.TCanvas(str(cat_idx),str(cat_idx),800, 800) # giving a specific name for each canvas prevents segfault?
    # canvas.cd()
    upper_pad = rt.TPad("upper_pad", "upper_pad", 0, 0.25, 1, 1)
    lower_pad = rt.TPad("lower_pad", "lower_pad", 0, 0, 1, 0.35)
    upper_pad.SetBottomMargin(0.14)
    lower_pad.SetTopMargin(0.00001)
    lower_pad.SetBottomMargin(0.25)
    upper_pad.Draw()
    lower_pad.Draw()
    upper_pad.cd()
    # workspace = rt.RooWorkspace("w", "w")
    mass_name = "dimuon_mass"
    # mass =  rt.RooRealVar(mass_name,"mass (GeV)",100,np.min(mass_arr),np.max(mass_arr))
    mass =  rt.RooRealVar(mass_name,"mass (GeV)",100,80,100)
    mass.setBins(nbins)
    roo_dataset = rt.RooDataSet.from_numpy({mass_name: mass_arr}, [mass]) # associate numpy arr to RooRealVar
    # workspace.Import(mass)
    frame = mass.frame(Title=f"ZCR Dimuon Mass Voigtian calibration fit for category {cat_idx}")

    # Voigtian --------------------------------------------------------------------------
    bwmZ = rt.RooRealVar("bwz_mZ" , "mZ", 91.1876, 91, 92)
    bwWidth = rt.RooRealVar("bwz_Width" , "widthZ", 2.4952, 1, 3)
    sigma = rt.RooRealVar("sigma" , "sigma", 2, 0.5, 2.5)
    bwWidth.setConstant(True)
    model1 = rt.RooVoigtian("signal" , "signal", mass, bwmZ, bwWidth, sigma)


    # # Exp x Erfc Background --------------------------------------------------------------------------
    # # exp_coeff = rt.RooRealVar("exp_coeff", "exp_coeff", 0.01, 0.00000001, 1) # positve coeff to get the peak shape we want
    # exp_coeff = rt.RooRealVar("exp_coeff", "exp_coeff", -0.1, -1, -0.00000001) # negative coeff to get the peak shape we want
    # shift = rt.RooRealVar("shift", "Offset", 85, 75, 105)
    # shifted_mass = rt.RooFormulaVar("shifted_mass", "@0-@1", rt.RooArgList(mass, shift))
    # model2_1 = rt.RooExponential("Exponential", "Exponential", shifted_mass,exp_coeff)

    # erfc_center = rt.RooRealVar("erfc_center" , "erfc_center", 91.2, 75, 105)
    # erfc_coeff = rt.RooRealVar("erfc_coeff" , "erfc_coeff", 0.1, 0, 1.5)
    # erfc_in = rt.RooFormulaVar("erfc_in", "(@0 - @2) * @1", rt.RooArgList(mass, erfc_coeff, erfc_center))
    # model2_2a = rt.RooFit.bindFunction("erfc", rt.TMath.Erfc, erfc_in) # turn TMath function to Roofit funciton
    # model2_2 = rt.RooWrapperPdf("erfc","erfc", model2_2a) # turn bound function to pdf
    # model2 = rt.RooProdPdf("bkg", "bkg", [model2_1, model2_2]) # generate ExpxErfc bkg

    # Landau Background --------------------------------------------------------------------------
    mean_landau = rt.RooRealVar("mean_landau" , "mean_landau", 95, 90, 150)
    sigma_landau = rt.RooRealVar("sigma_landau" , "sigma_landau", 2, 0.5, 8.5)
    model2 = rt.RooLandau("bkg", "bkg", mass, mean_landau, sigma_landau) # generate Landau bkg


    sigfrac = rt.RooRealVar("sigfrac", "sigfrac", 0.9, 0, 1.0)
    final_model = rt.RooAddPdf("final_model", "final_model", [model1, model2],[sigfrac])


    time_step = time.time()
    #fitting directly to unbinned dataset is slow, so first make a histogram
    roo_hist = rt.RooDataHist("data_hist","binned version of roo_dataset", rt.RooArgSet(mass), roo_dataset)  # copies binning from mass variable
    # do fitting
    rt.EnableImplicitMT()
    _ = final_model.fitTo(roo_hist, Save=True,  EvalBackend ="cpu")
    fit_result = final_model.fitTo(roo_hist, Save=True,  EvalBackend ="cpu")
    print(f"fitting elapsed time: {time.time() - time_step}")
    time.sleep(1) # rest a second for stability
    #do plotting
    roo_dataset.plotOn(frame, DataError="SumW2", Name="data_hist") # name is explicitly defined so chiSquare can find it
    # roo_hist.plotOn(frame, Name="data_hist")
    final_model.plotOn(frame, Name="final_model", LineColor=rt.kGreen)
    final_model.plotOn(frame, Components="signal", LineColor=rt.kBlue)
    final_model.plotOn(frame, Components="bkg", LineColor=rt.kRed)
    model1.paramOn(frame, Parameters=[sigma], Layout=[0.55,0.94, 0.8])
    frame.GetYaxis().SetTitle("Events")
    frame.Draw()

    #calculate chi2 and add to plot
    n_free_params = fit_result.floatParsFinal().getSize()
    print(f"n_free_params: {n_free_params}")
    chi2 = frame.chiSquare(final_model.GetName(), "data_hist", n_free_params)
    chi2 = float('%.3g' % chi2) # get upt to 3 sig fig
    print(f"chi2: {chi2}")
    latex = rt.TLatex()
    latex.SetNDC()
    latex.SetTextAlign(11)
    latex.SetTextFont(42)
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.7,0.8,f"#chi^2 = {chi2}")
    # canvas.Update()

    # obtain pull plot
    hpull = frame.pullHist("data_hist", "final_model")
    lower_pad.cd()
    frame2 = mass.frame(Title=" ")
    frame2.addPlotable(hpull, "P")
    frame2.GetYaxis().SetTitle("(Data-Fit)/ #sigma")
    frame2.GetYaxis().SetRangeUser(-5, 8)
    frame2.GetYaxis().SetTitleOffset(0.3)
    frame2.GetYaxis().SetTitleSize(0.08)
    frame2.GetYaxis().SetLabelSize(0.08)
    frame2.GetXaxis().SetLabelSize(0.08)
    frame2.GetXaxis().SetTitle("m_{#mu#mu} (GeV)")
    frame2.Draw()

    # canvas.Modified()
    canvas.Update()
    # canvas.Draw()
    print(f"sigma result for cat {cat_idx}: {sigma.getVal()} +- {sigma.getError()}")

    # save to df_fit
    # df_fit.loc[cat_idx] = [sigma.getVal(), sigma.getError()]
    # df_fit = df_fit.append({"cat_name": cat_idx, "fit_val": sigma.getVal(), "fit_err": sigma.getError()}, ignore_index=True)
    new_row = pd.DataFrame({"cat_name": cat_idx, "fit_val": sigma.getVal(), "fit_err": sigma.getError()}, index=[0])
    df_fit = pd.concat([df_fit, new_row], ignore_index=True)

    # Save the cat_idx and sigma value to a log file
    with open(logfile, "a") as f:
        f.write(f"{cat_idx} {sigma.getVal()} {sigma.getError()}\n")

    # save plot
    canvas.SaveAs(f"calibration_fitCat{cat_idx}{out_string}.pdf")
    del canvas
    # # consider script to wait a second for stability?
    # time.sleep(1)
    return df_fit

def generateBWxDCB_plot(mass_arr, cat_idx: int, nbins, df_fit = "", logfile="CalibrationLog.txt", out_string=""):
    """
    params
    mass_arr: numpy arrary of dimuon mass value to do calibration fit on
    cat_idx: int index of specific calibration category the mass_arr is from
    """
    # if you want TCanvas to not crash, separate fitting and drawing
    canvas = rt.TCanvas(str(cat_idx),str(cat_idx),800, 800) # giving a specific name for each canvas prevents segfault?
    # canvas.cd()
    upper_pad = rt.TPad("upper_pad", "upper_pad", 0, 0.25, 1, 1)
    lower_pad = rt.TPad("lower_pad", "lower_pad", 0, 0, 1, 0.35)
    upper_pad.SetBottomMargin(0.14)
    lower_pad.SetTopMargin(0.00001)
    lower_pad.SetBottomMargin(0.25)
    upper_pad.Draw()
    lower_pad.Draw()
    upper_pad.cd()
    # workspace = rt.RooWorkspace("w", "w")
    mass_name = "dimuon_mass"
    # mass =  rt.RooRealVar(mass_name,"mass (GeV)",100,np.min(mass_arr),np.max(mass_arr))
    mass =  rt.RooRealVar(mass_name,"mass (GeV)",100,80,100)
    mass.setBins(nbins)
    roo_dataset = rt.RooDataSet.from_numpy({mass_name: mass_arr}, [mass]) # associate numpy arr to RooRealVar
    # workspace.Import(mass)
    frame = mass.frame(Title=f"ZCR Dimuon Mass BWxDCB calibration fit for category {cat_idx}")

    # BWxDCB --------------------------------------------------------------------------
    bwmZ = rt.RooRealVar("bwz_mZ" , "mZ", 91.1876, 91, 92)
    bwWidth = rt.RooRealVar("bwz_Width" , "widthZ", 2.4952, 1, 3)
    # bwmZ.setConstant(True)
    bwWidth.setConstant(True)


    model1_1 = rt.RooBreitWigner("bwz", "BWZ",mass, bwmZ, bwWidth)

    """
    Note from Jan: sometimes freeze n values in DCB to be frozen (ie 1, but could be other values)
    This is because alpha and n are highly correlated, so roofit can be really confused.
    Also, given that we care about the resolution, not the actual parameter values alpha and n, we can
    put whatevere restrictions we want.
    """
    mean = rt.RooRealVar("mean" , "mean", 0, -10,10) # mean is mean relative to BW
    # mean = rt.RooRealVar("mean" , "mean", 100, 95,110) # test
    sigma = rt.RooRealVar("sigma" , "sigma", 2, .1, 4.0)
    alpha1 = rt.RooRealVar("alpha1" , "alpha1", 2, 0.01, 65)
    n1 = rt.RooRealVar("n1" , "n1", 10, 0.01, 185)
    alpha2 = rt.RooRealVar("alpha2" , "alpha2", 2.0, 0.01, 65)
    n2 = rt.RooRealVar("n2" , "n2", 25, 0.01, 385)
    # n2 = rt.RooRealVar("n2" , "n2", 114, 0.01, 385) #test 114
    n1.setConstant(True)
    n2.setConstant(True)
    model1_2 = rt.RooCrystalBall("dcb","dcb",mass, mean, sigma, alpha1, n1, alpha2, n2)

    # merge BW with DCB via convolution
    model1 = rt.RooFFTConvPdf("signal", "signal", mass, model1_1, model1_2) # BWxDCB


    mass.setBins(10000,"cache") # This nbins has nothing to do with actual nbins of mass. cache bins is representation of the variable only used in FFT
    mass.setMin("cache",50.5)
    mass.setMax("cache",130.5)

    # # Exp Background --------------------------------------------------------------------------
    # coeff = rt.RooRealVar("coeff", "coeff", 0.01, 0.00000001, 1)
    # shift = rt.RooRealVar("shift", "Offset", 85, 75, 105)
    # shifted_mass = rt.RooFormulaVar("shifted_mass", "@0-@1", rt.RooArgList(mass, shift))
    # model2 = rt.RooExponential("bkg", "bkg", shifted_mass, coeff)
    #--------------------------------------------------

    # Landau Background --------------------------------------------------------------------------
    # mean_landau = rt.RooRealVar("mean_landau" , "mean_landau", 90, 70, 200)
    # sigma_landau = rt.RooRealVar("sigma_landau" , "sigma_landau", 7, 0.5, 8.5)
    # model2 = rt.RooLandau("bkg", "bkg", mass, mean_landau, sigma_landau) # generate Landau bkg
    #-----------------------------------------------------

    # neg Exp Background --------------------------------------------------------------------------
    # coeff = rt.RooRealVar("coeff", "coeff", -0.01, -1,  -0.00000001)
    # shift = rt.RooRealVar("shift", "Offset", 70, 40, 105)
    # shifted_mass = rt.RooFormulaVar("shifted_mass", "@0-@1", rt.RooArgList(mass, shift))
    # model2 = rt.RooExponential("bkg", "bkg", shifted_mass, coeff)
    #--------------------------------------------------

    # Reverse Landau Background test--------------------------------------------------------------------------
    mean_landau = rt.RooRealVar("mean_landau" , "mean_landau", -80,  -150, -70) # 80
    mass_neg = rt.RooFormulaVar("mass_neg", "-@0", [mass])
    sigma_landau = rt.RooRealVar("sigma_landau" , "sigma_landau", 7, 0.5, 8.5)
    model2 = rt.RooLandau("bkg", "bkg", mass_neg, mean_landau, sigma_landau) # generate Landau bkg
    #-----------------------------------------------------

    # Exp x Erf Background --------------------------------------------------------------------------
    # exp_coeff = rt.RooRealVar("exp_coeff", "exp_coeff", 0.01, 0.00000001, 1) # positve coeff to get the peak shape we want
    # # exp_coeff = rt.RooRealVar("exp_coeff", "exp_coeff", -0.1, -1, -0.00000001) # negative coeff to get the peak shape we want
    # shift = rt.RooRealVar("shift", "Offset", 85, 75, 105)
    # shifted_mass = rt.RooFormulaVar("shifted_mass", "(@0 - @1)", rt.RooArgList(mass, shift))
    # model2_1 = rt.RooExponential("Exponential", "Exponential", shifted_mass,exp_coeff)

    # erf_center = rt.RooRealVar("erf_center" , "erf_center", 91.2, 75, 155)
    # erf_in = rt.RooFormulaVar("erf_in", "(@0 - @1)", rt.RooArgList(mass, erf_center))
    # model2_2a = rt.RooFit.bindFunction("erf", rt.TMath.Erf, erf_in) # turn TMath function to Roofit funciton
    # model2_2 = rt.RooWrapperPdf("erf","erf", model2_2a) # turn bound function to pdf
    # # model2 = rt.RooProdPdf("bkg", "bkg", [model2_1, model2_2]) # generate Expxerf bkg

    #-----------------------------------------------------

    # # Exp x Erf Background V2--------------------------------------------------------------------------
    # # exp_coeff = rt.RooRealVar("exp_coeff", "exp_coeff", 0.01, 0.00000001, 1) # positve coeff to get the peak shape we want
    # exp_coeff = rt.RooRealVar("exp_coeff", "exp_coeff", -0.1, -1, -0.00000001) # negative coeff to get the peak shape we want
    # shift = rt.RooRealVar("shift", "Offset", 100, 90, 150)
    # shifted_mass = rt.RooFormulaVar("shifted_mass", "@0-@1", rt.RooArgList(mass, shift))
    # model2_1 = rt.RooExponential("Exponential", "Exponential", shifted_mass,exp_coeff)
    # erfc_center = rt.RooRealVar("erfc_center" , "erfc_center", 100, 90, 150)
    # erfc_in = rt.RooFormulaVar("erfc_in", "(@0 - @1)", rt.RooArgList(mass, erfc_center))
    # # both bindPdf and RooGenericPdf work, but one may have better cuda integration over other, so leaving both options
    # # model2_2 = rt.RooFit.bindPdf("erfc", rt.TMath.Erfc, erfc_in)
    # model2_2 = rt.RooGenericPdf("erfc", "TMath::Erf(@0)+1", erfc_in)
    # model2 = rt.RooProdPdf("bkg", "bkg", rt.RooArgList(model2_1, model2_2))
    #-----------------------------------------------------



    sigfrac = rt.RooRealVar("sigfrac", "sigfrac", 0.9, 0.000001, 0.99999999)
    final_model = rt.RooAddPdf("final_model", "final_model", [model1, model2],[sigfrac])
    # final_model = model1_2



    time_step = time.time()

    #fitting directly to unbinned dataset is slow, so first make a histogram
    roo_hist = rt.RooDataHist("data_hist","binned version of roo_dataset", rt.RooArgSet(mass), roo_dataset)  # copies binning from mass variable
    # do fitting
    rt.EnableImplicitMT()
    _ = final_model.fitTo(roo_hist, Save=True,  EvalBackend ="cpu")
    fit_result = final_model.fitTo(roo_hist, Save=True,  EvalBackend ="cpu")
    print(f"fitting elapsed time: {time.time() - time_step}")
    time.sleep(1) # rest a second for stability
    #do plotting
    roo_dataset.plotOn(frame, DataError="SumW2", Name="data_hist") # name is explicitly defined so chiSquare can find it
    # roo_hist.plotOn(frame, Name="data_hist") # name is explicitly defined so chiSquare can find it
    final_model.plotOn(frame, Name="final_model", LineColor=rt.kGreen)
    final_model.plotOn(frame, Components="signal", LineColor=rt.kBlue)
    final_model.plotOn(frame, Components="bkg", LineColor=rt.kRed)
    model1.paramOn(frame, Parameters=[sigma], Layout=[0.55,0.94, 0.8])
    frame.GetYaxis().SetTitle("Events")
    frame.Draw()

    #calculate chi2 and add to plot
    n_free_params = fit_result.floatParsFinal().getSize()
    print(f"n_free_params: {n_free_params}")
    chi2 = frame.chiSquare(final_model.GetName(), "data_hist", n_free_params)
    chi2 = float('%.3g' % chi2) # get upt to 3 sig fig
    print(f"chi2: {chi2}")
    latex = rt.TLatex()
    latex.SetNDC()
    latex.SetTextAlign(11)
    latex.SetTextFont(42)
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.7,0.8,f"#chi^2 = {chi2}")
    # canvas.Update()

    # obtain pull plot
    hpull = frame.pullHist("data_hist", "final_model")
    lower_pad.cd()
    frame2 = mass.frame(Title=" ")
    frame2.addPlotable(hpull, "P")
    frame2.GetYaxis().SetTitle("(Data-Fit)/ #sigma")
    frame2.GetYaxis().SetRangeUser(-5, 8)
    frame2.GetYaxis().SetTitleOffset(0.3)
    frame2.GetYaxis().SetTitleSize(0.08)
    frame2.GetYaxis().SetLabelSize(0.08)
    frame2.GetXaxis().SetLabelSize(0.08)
    frame2.GetXaxis().SetTitle("m_{#mu#mu} (GeV)")
    frame2.Draw()

    # canvas.Modified()
    canvas.Update()
    # canvas.Draw()


    print(f"mean_landau: {mean_landau.getVal()}")
    print(f"sigma_landau: {sigma_landau.getVal()}")
    print(f"n1: {n1.getVal()}")
    print(f"n2: {n2.getVal()}")
    print(f"alpha1: {alpha1.getVal()}")
    print(f"alpha2: {alpha2.getVal()}")
    print(f"sigma result for cat {cat_idx}: {sigma.getVal()} +- {sigma.getError()}")

    # save cat_idx and sigma value to a pandas dataframe
    if not df_fit.empty:
        new_row = pd.DataFrame([{"cat_name": cat_idx, "fit_val": sigma.getVal(), "fit_err": sigma.getError()}])
        df_fit = pd.concat([df_fit, new_row], ignore_index=True)
    else:
        df_fit = pd.DataFrame([{"cat_name": cat_idx, "fit_val": sigma.getVal(), "fit_err": sigma.getError()}])


    # Save the cat_idx and sigma value to a log file
    with open(logfile, "a") as f:
        f.write(f"{cat_idx} {sigma.getVal()} {sigma.getError()}\n")

    canvas.SaveAs(f"calibration_fitCat{cat_idx}{out_string}.pdf")
    del canvas
    # consider script to wait a second for stability?
    time.sleep(1)
    return df_fit

def save_calibration_json(df_merged, json_filename="calibration_factors.json"):
    """
    Given a DataFrame (df_merged) with columns "cat_name" and "calibration_factor",
    write out a JSON file with the following multibinning structure.

    The calibration categories are assumed to be labeled as:
      "<pt_bin>_<eta1><eta2>"
    with:
      pt_bin in {"30-45", "45-52", "52-62", "62-200"}
      eta1, eta2 in {"B", "O", "E"}
    corresponding to the following bin edges:
      leading_mu_pt: [30.0, 45.0, 52.0, 62.0, 200.0]
      leading_mu_abseta: [0.0, 0.9, 1.8, 2.4]
      subleading_mu_abseta: [0.0, 0.9, 1.8, 2.4]

    The "content" field will be a flattened list of 36 calibration factors ordered as:
    For each pt_bin (in order: "30-45", "45-52", "52-62", "62-200"),
      for each leading muon eta bin (B, O, E),
      for each subleading muon eta bin (B, O, E),
      use the calibration factor from the corresponding category.
    If a category is missing, a default value of 1.0 is used.
    """
    # Define the bin edges and labels
    pt_bins = ["30-45", "45-52", "52-62", "62-200"]
    eta_bins = ["B", "O", "E"]

    calib_dict = dict(zip(df_merged["cat_name"], df_merged["calibration_factor"]))

    content = []
    # Loop over pt bins:
    for pt_bin in pt_bins:
        if pt_bin == "30-45":
            # For pt bin "30-45", we have only three merged categories.
            # Loop over all 9 (leading, subleading) combinations but choose the factor based solely on subleading muon.
            for eta1 in eta_bins:
                for eta2 in eta_bins:
                    if eta2 == "B":
                        cat_name = "30-45_BB_OB_EB"
                    elif eta2 == "O":
                        cat_name = "30-45_BO_OO_EO"
                    elif eta2 == "E":
                        cat_name = "30-45_BE_OE_EE"
                    factor = calib_dict.get(cat_name, 1.0)
                    content.append(factor)
        else:
            # For other pt bins, there are 9 cells: loop over leading eta then subleading eta.
            for eta1 in eta_bins:
                for eta2 in eta_bins:
                    cat_name = f"{pt_bin}_{eta1}{eta2}"
                    content.append(calib_dict.get(cat_name, 1.0))

    # Build the JSON structure.
    json_dict = {
        "schema_version": 2,
        "corrections": [
            {
                "name": "BS_ebe_mass_res_calibration",
                "description": "Dimuon Mass resolution calibration with BeamSpot Constraint correction applied",
                "version": 1,
                "inputs": [
                    {
                        "name": "leading_mu_pt",
                        "type": "real",
                        "description": "Transverse momentum of the leading muon (GeV)"
                    },
                    {
                        "name": "leading_mu_abseta",
                        "type": "real",
                        "description": "Absolute pseudorapidity of the leading muon"
                    },
                    {
                        "name": "subleading_mu_abseta",
                        "type": "real",
                        "description": "Absolute pseudorapidity of the subleading muon"
                    }
                ],
                "output": {
                    "name": "correction_factor",
                    "type": "real"
                },
                "data": {
                    "nodetype": "multibinning",
                    "inputs": [
                        "leading_mu_pt",
                        "leading_mu_abseta",
                        "subleading_mu_abseta"
                    ],
                    "edges": [
                        [30.0, 45.0, 52.0, 62.0, 200.0],
                        [0.0, 0.9, 1.8, 2.4],
                        [0.0, 0.9, 1.8, 2.4]
                    ],
                    "content": content,
                    "flow": "clamp"
                }
            }
        ]
    }

    with open(json_filename, "w") as f:
        json.dump(json_dict, f, indent=4)
    print(f"Calibration JSON saved to {json_filename}")


def closure_test_from_df(df, additional_string, output_plot="closure_test.pdf"):
    """
    Given a DataFrame with columns:
         cat_name, fit_val, fit_err, median_val, calibration_factor,
    produce a closure test plot that compares the fitted resolution (fit_val)
    to the median predicted resolution (median_val) for each calibration category.

    A reference line y = x is drawn to indicate perfect agreement.

    Parameters:
      df         : Pandas DataFrame with the required columns.
      output_plot: Filename for the closure test plot.

    Returns:
      The input DataFrame (unchanged).
    """
    # Check that the necessary columns exist
    required_cols = {"cat_name", "fit_val", "fit_err", "median_val", "calibration_factor"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    # Create the closure test plot.
    plt.figure(figsize=(8,6))
    plt.errorbar(df["median_val"], df["fit_val"], yerr=df["fit_err"], fmt='o', label="Categories")

    # Plot the reference y = x line.
    x_min = df["median_val"].min()
    x_max = df["median_val"].max()
    x_vals = np.linspace(0.5, x_max*1.1, 100)
    plt.plot(x_vals, x_vals, "r--", label="y = x")

    # plot the 10% dotted line for reference
    y_10 = x_vals * 1.1
    plt.plot(x_vals, y_10, "g--", label="y = 1.1x")
    y_10 = x_vals * 0.9
    plt.plot(x_vals, y_10, "g--", label="y = 0.9x")



    plt.xlabel("Predicted $\\sigma_{\\mu\\mu}$ [GeV]")
    plt.ylabel("Measured $\\sigma_{\\mu\\mu}$ [GeV]")
    plt.title("Closure Test: Measured vs. Predicted Resolution")
    plt.legend()
    output_plot = output_plot.replace(".pdf", f"_{additional_string}.pdf")
    plt.savefig(output_plot)
    plt.close()

    print(f"Closure test plot saved as {output_plot}")
    return df

def closure_test_from_calibrated_df(df_fit, df_calibrated, additional_string, output_plot="closure_test.pdf"):
    df_merged = pd.merge(df_fit, df_calibrated, on="cat_name", how="inner")
    df = df_merged
    required_cols = {"cat_name", "fit_val", "fit_err", "median_val"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    # Create the closure test plot.
    plt.figure(figsize=(8,6))
    plt.errorbar(df["median_val"], df["fit_val"], yerr=df["fit_err"], fmt='o', label="Categories")

    # Plot the reference y = x line.
    x_min = df["median_val"].min()
    x_max = df["median_val"].max()
    x_vals = np.linspace(0.5, x_max*1.1, 100)
    plt.plot(x_vals, x_vals, "r--", label="y = x")

    # plot the 10% dotted line for reference
    y_10 = x_vals * 1.1
    plt.plot(x_vals, y_10, "g--", label="y = 1.1x")
    y_10 = x_vals * 0.9
    plt.plot(x_vals, y_10, "g--", label="y = 0.9x")




    plt.xlabel("Predicted $\\sigma_{\\mu\\mu}$ [GeV]") # plt.xlabel("Median Predicted Resolution (GeV)")
    plt.ylabel("Measured $\\sigma_{\\mu\\mu}$ [GeV]") #plt.ylabel("Fitted Resolution (GeV)")
    plt.title("Closure Test: Measured vs. Predicted Resolution [2018 C]")
    plt.legend()
    output_plot = output_plot.replace(".pdf", f"_{additional_string}.pdf")
    plt.savefig(output_plot)
    # save image to png
    plt.savefig(output_plot.replace(".pdf", ".png"))
    plt.close()

    print(f"Closure test plot saved as {output_plot}")
    # return df_merged

