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
import itertools
import glob
import ROOT as rt
import ROOT

plt.style.use(hep.style.CMS)


def applyVBF_cutV1(events):
    btag_cut =ak.fill_none((events.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((events.nBtagMedium_nominal >= 1), value=False)
    vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35) 
    vbf_cut = ak.fill_none(vbf_cut, value=False)
    dimuon_mass = events.dimuon_mass
    VBF_filter = (
        vbf_cut & 
        ~btag_cut # btag cut is for VH and ttH categories
    )
    trues = ak.ones_like(dimuon_mass, dtype="bool")
    falses = ak.zeros_like(dimuon_mass, dtype="bool")
    events["vbf_filter"] = ak.where(VBF_filter, trues,falses)
    return events[VBF_filter]

def applyGGH_cutV1(events):
    btag_cut =ak.fill_none((events.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((events.nBtagMedium_nominal >= 1), value=False)
    vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35) 
    vbf_cut = ak.fill_none(vbf_cut, value=False)
    dimuon_mass = events.dimuon_mass
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

    # mu1_pt = events.mu1_pt
    # mu1ptOfInterest = (mu1_pt > 75) & (mu1_pt < 150.0)
    # events = events[region&mu1ptOfInterest]
    events = events[region]
    return events


def generateBWxDCB_plot(mass_arr, cat_idx: int, nbins, df_fit, logfile="CalibrationLog.txt"):
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
    # mass =  rt.RooRealVar(mass_name,"mass (GeV)",100,70,110)
    mass =  rt.RooRealVar(mass_name,"mass (GeV)",100,115,135)
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
    mean = rt.RooRealVar("mean" , "mean", 125, 110,135) # mean is mean relative to BW
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
    # model1 = rt.RooFFTConvPdf("signal", "signal", mass, model1_1, model1_2) # BWxDCB
    model1 =rt.RooCrystalBall("dcb","dcb",mass, mean, sigma, alpha1, n1, alpha2, n2)

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
    # df_fit = df_fit.append({"cat_name": cat_idx, "fit_val": sigma.getVal(), "fit_err": sigma.getError()}, ignore_index=True)
    new_row = pd.DataFrame([{"cat_name": cat_idx, "fit_val": sigma.getVal(), "fit_err": sigma.getError()}])
    df_fit = pd.concat([df_fit, new_row], ignore_index=True)


    # Save the cat_idx and sigma value to a log file
    with open(logfile, "a") as f:
        f.write(f"{cat_idx} {sigma.getVal()} {sigma.getError()}\n")

    canvas.SaveAs(f"calibration_fitCat{cat_idx}.pdf")
    del canvas
    # consider script to wait a second for stability?
    time.sleep(1)
    return df_fit

def generateRooHist(x, dimuon_mass, wgts, name=""):
    print("generateRooHist version 2")
    dimuon_mass = np.asarray(ak.to_numpy(dimuon_mass)).astype(np.float64) # explicit float64 format is required
    wgts = np.asarray(ak.to_numpy(wgts)).astype(np.float64) # explicit float64 format is required
    nbins = x.getBins()
    TH = rt.TH1D("TH", "TH", nbins, x.getMin(), x.getMax())
    TH.FillN(len(dimuon_mass), dimuon_mass, wgts) # fill the histograms with mass and weights 
    roohist = rt.RooDataHist(name, name, rt.RooArgSet(x), TH)
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



def fitPlot_ggh(events_bsOn, events_bsOff, save_filename, save_plot=True):
    """
    generate histogram from dimuon mass and wgt, fit DCB
    aftwards, plot the histogram and return the fit params
    as fit DCB sigma and chi2_dof
    """
    mass_name = "mh_ggh"
    mass = rt.RooRealVar(mass_name, mass_name, 120, 115, 135) # signal region
    # nbins = 100
    nbins = 80
    mass.setBins(nbins)
    dimuon_mass = ak.to_numpy(events_bsOn.dimuon_mass)
    wgt = ak.to_numpy(events_bsOn.wgt_nominal)
    hist_bsOn = generateRooHist(mass, dimuon_mass, wgt, name=f"BSC fit")
    hist_bsOn = normalizeRooHist(mass, hist_bsOn)
    print(f"fitPlot_ggh hist_bsOn: {hist_bsOn}")

    dimuon_mass = ak.to_numpy(events_bsOff.dimuon_mass)
    wgt = ak.to_numpy(events_bsOff.wgt_nominal)
    hist_bsOff = generateRooHist(mass, dimuon_mass, wgt, name=f"geofit")
    hist_bsOff = normalizeRooHist(mass, hist_bsOff)
    print(f"fitPlot_ggh hist_bsOff: {hist_bsOff}")

    # --------------------------------------------------
    # Fitting
    # --------------------------------------------------
    
    MH_bsOn = rt.RooRealVar("MH" , "MH", 125, 110, 150)
    sigma_bsOn = rt.RooRealVar("sigma" , "sigma", 1.8228, .1, 4.0)
    alpha1_bsOn = rt.RooRealVar("alpha1" , "alpha1", 1.12842, 0.01, 65)
    n1_bsOn = rt.RooRealVar("n1" , "n1", 4.019960, 0.01, 100)
    alpha2_bsOn = rt.RooRealVar("alpha2" , "alpha2", 1.3132, 0.01, 65)
    n2_bsOn = rt.RooRealVar("n2" , "n2", 9.97411, 0.01, 100)
    name = f"BSC fit"
    model_bsOn = rt.RooDoubleCBFast(name,name,mass, MH_bsOn, sigma_bsOn, alpha1_bsOn, n1_bsOn, alpha2_bsOn, n2_bsOn)

    device = "cpu"
    _ = model_bsOn.fitTo(hist_bsOn,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = model_bsOn.fitTo(hist_bsOn,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result.Print()

    MH_bsOff = rt.RooRealVar("MH" , "MH", 125, 110, 150)
    sigma_bsOff = rt.RooRealVar("sigma" , "sigma", 1.8228, .1, 4.0)
    alpha1_bsOff = rt.RooRealVar("alpha1" , "alpha1", 1.12842, 0.01, 65)
    n1_bsOff = rt.RooRealVar("n1" , "n1", 4.019960, 0.01, 100)
    alpha2_bsOff = rt.RooRealVar("alpha2" , "alpha2", 1.3132, 0.01, 65)
    n2_bsOff = rt.RooRealVar("n2" , "n2", 9.97411, 0.01, 100)
    name = f"BSC fit"
    model_bsOff = rt.RooDoubleCBFast(name,name,mass, MH_bsOff, sigma_bsOff, alpha1_bsOff, n1_bsOff, alpha2_bsOff, n2_bsOff)

    device = "cpu"
    _ = model_bsOff.fitTo(hist_bsOff,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = model_bsOff.fitTo(hist_bsOff,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result.Print()

    # ------------------------------------
    # Plotting
    # ------------------------------------
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    legend = rt.TLegend(0.6,0.60,0.9,0.9)
        

    frame = mass.frame()
    hist_bsOn.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0), Invisible=True )
    model_bsOn.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1), "ggH Powheg with BSC fit", "L")
    sigma_val = round(sigma_bsOn.getVal(), 3)
    sigma_err = round(sigma_bsOn.getError(), 3) 
    legend.AddEntry("", f"BSC sigma: {sigma_val} +- {sigma_err}", "")
    
    hist_bsOff.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0), Invisible=True )
    model_bsOff.plotOn(frame, Name=name, LineColor=rt.kBlue)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1), "ggH Powheg with geofit", "L")
    sigma_val = round(sigma_bsOff.getVal(), 3)
    sigma_err = round(sigma_bsOff.getError(), 3)

    legend.AddEntry("", f"geofit sigma: {sigma_val} +- {sigma_err}", "")

    frame.SetYTitle(f"A.U.")
    frame.SetXTitle(f"Dimuon Mass (GeV)")
    frame.SetTitle(f"")

    frame.Draw()
    legend.Draw()        
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(save_filename)

if __name__ == "__main__":
    client =  Client(n_workers=15,  threads_per_worker=2, processes=True, memory_limit='8 GiB') 
    V1_fields_2compute = [
        "wgt_nominal",
        "nBtagLoose_nominal",
        "nBtagMedium_nominal",
        "mu1_pt",
        "mu2_pt",
        "mu1_eta",
        "mu2_eta",
        "mu1_phi",
        "mu2_phi",
        "dimuon_pt",
        "dimuon_eta",
        "dimuon_phi",
        "dimuon_mass",
        "jet1_phi_nominal",
        "jet1_pt_nominal",
        "jet2_pt_nominal",
        "jet2_phi_nominal",
        "jet1_eta_nominal",
        "jet2_eta_nominal",
        "jj_mass_nominal",
        "jj_dEta_nominal",
        # "region",
        "event",
    ]
     
    # load_path = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_24Feb_BSon//stage1_output/2018/f1_0/data_A/0/*.parquet"
    load_path = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_24Feb_BSon//stage1_output/2018/f1_0/ggh_powhegPS/0/*.parquet"

    # print(f"file: {file}")
    # events_data = dak.from_parquet(f"{file}/*/*.parquet")
    events_data = dak.from_parquet(f"{load_path}")

    events_data = ak.zip({field: events_data[field] for field in V1_fields_2compute}).compute()
    # events_data = filterRegion(events_data, region="z-peak")
    events_BSon = filterRegion(events_data, region="signal")
    # events_data = applyGGH_cutV1(events_data)
    nbins =100
    df_fit = pd.DataFrame({})
    mass_arr = ak.to_numpy(events_data.dimuon_mass)
    # generateBWxDCB_plot(mass_arr, 0, nbins, df_fit)

    # same with BSC off
    
    # load_path = "/depot/cms/users/yun79/hmm/copperheadV1clean/V2_Jan29_JecOn_TrigMatchFixed_2016UlJetIdFix/stage1_output/2018/f1_0/data_A/0/*.parquet"
    # load_path = "/depot/cms/users/yun79/hmm/copperheadV1clean//rereco_yun_Dec05_btagSystFixed_JesJerUncOn//stage1_output/2018/data_A/*.parquet"
    # load_path = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_24Feb_BSoff//stage1_output/2018/f1_0/data_A/0/*.parquet"
    load_path = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_24Feb_BSoff//stage1_output/2018/f1_0/ggh_powhegPS/0/*.parquet"
    events_data = dak.from_parquet(f"{load_path}")

    events_data = ak.zip({field: events_data[field] for field in V1_fields_2compute}).compute()
    # events_data = filterRegion(events_data, region="z-peak")
    events_BSoff = filterRegion(events_data, region="signal")
    # events_data = applyGGH_cutV1(events_data)
    nbins =100
    df_fit = pd.DataFrame({})
    mass_arr = ak.to_numpy(events_data.dimuon_mass)
    # print(f"mass_arr: {mass_arr}")
    # raise ValueError
    # generateBWxDCB_plot(mass_arr, 1, nbins, df_fit)


    fitPlot_ggh(events_BSon, events_BSoff, "BSC_geofit_comparison.pdf", save_plot=True)
    
