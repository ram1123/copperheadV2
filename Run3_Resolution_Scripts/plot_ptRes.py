import dask_awkward as dak
import awkward as ak
from distributed import Client
import time
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
import ROOT as rt

plt.style.use(hep.style.CMS)

def apply_vbf_cut(events):
    btag_cut = ak.fill_none((events.nBtagLoose_nominal >= 2) | (events.nBtagMedium_nominal >= 1), value=False)
    vbf_cut = ak.fill_none((events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35), value=False)
    vbf_filter = vbf_cut & ~btag_cut
    trues = ak.ones_like(events.dimuon_mass, dtype="bool")
    falses = ak.zeros_like(events.dimuon_mass, dtype="bool")
    events["vbf_filter"] = ak.where(vbf_filter, trues, falses)
    return events[vbf_filter]

def apply_ggh_cut(events):
    btag_cut = ak.fill_none((events.nBtagLoose_nominal >= 2) | (events.nBtagMedium_nominal >= 1), value=False)
    vbf_cut = ak.fill_none((events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35), value=False)
    ggH_filter = ~vbf_cut & ~btag_cut
    return events[ggH_filter]

def filter_region(events, region="h-peak"):
    dimuon_mass = events.dimuon_mass
    if region == "h-peak":
        region_filter = (dimuon_mass > 115.03) & (dimuon_mass < 135.03)
    elif region == "h-sidebands":
        region_filter = ((dimuon_mass > 110) & (dimuon_mass < 115.03)) | ((dimuon_mass > 135.03) & (dimuon_mass < 150))
    elif region == "signal":
        region_filter = (dimuon_mass >= 110) & (dimuon_mass <= 150.0)
    elif region == "z-peak":
        region_filter = (dimuon_mass >= 70) & (dimuon_mass <= 110.0)
    return events[region_filter]

def filter_region_using_rapidity_leadMuon(events, region):
    # Function to filter events based on leading and subleading muon rapidity
    # Eta bins:
    #   B: |eta| <= 0.9
    #   O: 0.9 < |eta| <= 1.8
    #   E: 1.8 < |eta| <= 2.4

    eta_bins = {
        "B": (abs(events.mu1_eta) <= 0.9),
        "O": (abs(events.mu1_eta) > 0.9) & (abs(events.mu1_eta) <= 1.8),
        "E": (abs(events.mu1_eta) > 1.8) & (abs(events.mu1_eta) <= 2.4)
    }
    return events[eta_bins[region]]

def filter_region_using_rapidity_SubleadMuon(events, region):
    # Function to filter events based on leading and subleading muon rapidity
    # Eta bins:
    #   B: |eta| <= 0.9
    #   O: 0.9 < |eta| <= 1.8
    #   E: 1.8 < |eta| <= 2.4

    eta_bins = {
        "B": (abs(events.mu2_eta) <= 0.9),
        "O": (abs(events.mu2_eta) > 0.9) & (abs(events.mu2_eta) <= 1.8),
        "E": (abs(events.mu2_eta) > 1.8) & (abs(events.mu2_eta) <= 2.4)
    }
    return events[eta_bins[region]]

def filter_subleading_muon_pt(events, pt_low, pt_high):
    # Function to filter events based on subleading muon pt
    pt_filter = (events.mu2_pt > pt_low) & (events.mu2_pt < pt_high)
    return events[pt_filter]

def filter_region_using_rapidity(events, region):
    # Function to filter events based on leading and subleading muon rapidity
    # Eta bins:
    #   B: |eta| <= 0.9
    #   O: 0.9 < |eta| <= 1.8
    #   E: 1.8 < |eta| <= 2.4

    eta_bins = {
        "BB": (abs(events.mu1_eta) <= 0.9) & (abs(events.mu2_eta) <= 0.9),
        "BO": (abs(events.mu1_eta) <= 0.9) & (abs(events.mu2_eta) > 0.9) & (abs(events.mu2_eta) <= 1.8),
        "OB": (abs(events.mu1_eta) > 0.9) & (abs(events.mu1_eta) <= 1.8) & (abs(events.mu2_eta) <= 0.9),
        "OO": (abs(events.mu1_eta) > 0.9) & (abs(events.mu1_eta) <= 1.8) & (abs(events.mu2_eta) > 0.9) & (abs(events.mu2_eta) <= 1.8),
        "BE": (abs(events.mu1_eta) <= 0.9) & (abs(events.mu2_eta) > 1.8) & (abs(events.mu2_eta) <= 2.4),
        "EB": (abs(events.mu1_eta) > 1.8) & (abs(events.mu1_eta) <= 2.4) & (abs(events.mu2_eta) <= 0.9),
        "EO": (abs(events.mu1_eta) > 1.8) & (abs(events.mu1_eta) <= 2.4) & (abs(events.mu2_eta) > 0.9) & (abs(events.mu2_eta) <= 1.8),
        "OE": (abs(events.mu1_eta) > 0.9) & (abs(events.mu1_eta) <= 1.8) & (abs(events.mu2_eta) > 1.8) & (abs(events.mu2_eta) <= 2.4),
        "EE": (abs(events.mu1_eta) > 1.8) & (abs(events.mu1_eta) <= 2.4) & (abs(events.mu2_eta) > 1.8) & (abs(events.mu2_eta) <= 2.4)
    }
    return events[eta_bins[region]]

def generate_bwxdcb_plot(mass_arr, cat_idx, nbins, df_fit, logfile="CalibrationLog.txt"):
    """
    params
    mass_arr: numpy arrary of dimuon mass value to do calibration fit on
    cat_idx: int index of specific calibration category the mass_arr is from
    """
    # INFO: if you want TCanvas to not crash, separate fitting and drawing
    canvas = rt.TCanvas(str(cat_idx), str(cat_idx), 800, 800) # INFO: giving a specific name for each canvas prevents segfault
    upper_pad = rt.TPad("upper_pad", "upper_pad", 0, 0.25, 1, 1)
    lower_pad = rt.TPad("lower_pad", "lower_pad", 0, 0, 1, 0.35)
    upper_pad.SetBottomMargin(0.14)
    lower_pad.SetTopMargin(0.00001)
    lower_pad.SetBottomMargin(0.25)
    upper_pad.Draw()
    lower_pad.Draw()
    upper_pad.cd()

    mass = rt.RooRealVar("dimuon_mass", "mass (GeV)", 100, 115, 135)
    mass.setBins(nbins)
    roo_dataset = rt.RooDataSet.from_numpy({"dimuon_mass": mass_arr}, [mass])
    frame = mass.frame(Title=f"ZCR Dimuon Mass BWxDCB calibration fit for category {cat_idx}")

    # Defining the models Breit-Wigner and Double Crystal Ball
    # Step-1: Define the Breit-Wigner model
    bwmZ = rt.RooRealVar("bwz_mZ", "mZ", 91.1876, 91, 92)
    bwWidth = rt.RooRealVar("bwz_Width", "widthZ", 2.4952, 1, 3)
    bwWidth.setConstant(True)
    model1_1 = rt.RooBreitWigner("bwz", "BWZ", mass, bwmZ, bwWidth)

    """
    INFO: **Note from Jan** :  sometimes freeze n values in DCB to be frozen (ie 1, but could be other values)
    This is because alpha and n are highly correlated, so roofit can be really confused.
    Also, given that we care about the resolution, not the actual parameter values alpha and n, we can
    put whatevere restrictions we want.
    """

    # Step-2: Define the Double Crystal Ball model
    mean = rt.RooRealVar("mean", "mean", 125, 110, 135) # INFO: mean is mean relative to BW
    sigma = rt.RooRealVar("sigma", "sigma", 2, .1, 4.0)
    alpha1 = rt.RooRealVar("alpha1", "alpha1", 2, 0.01, 65)
    n1 = rt.RooRealVar("n1", "n1", 10, 0.01, 185)
    alpha2 = rt.RooRealVar("alpha2", "alpha2", 2.0, 0.01, 65)
    n2 = rt.RooRealVar("n2", "n2", 25, 0.01, 385)
    n1.setConstant(True)
    n2.setConstant(True)
    model1_2 = rt.RooCrystalBall("dcb", "dcb", mass, mean, sigma, alpha1, n1, alpha2, n2)

    # Step-3: Define the convolution of BW and DCB
    # model1 = rt.RooFFTConvPdf("signal", "signal", mass, model1_1, model1_2) # BWxDCB
    model1 = rt.RooCrystalBall("dcb", "dcb", mass, mean, sigma, alpha1, n1, alpha2, n2)

    mass.setBins(10000, "cache") # INFO: This nbins has nothing to do with actual nbins of mass. cache bins is representation of the variable only used in FFT
    mass.setMin("cache", 50.5)
    mass.setMax("cache", 130.5)

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

    # INFO: Reverse Landau Background test --------------------------------------------------------------------------
    mean_landau = rt.RooRealVar("mean_landau", "mean_landau", -80, -150, -70)
    mass_neg = rt.RooFormulaVar("mass_neg", "-@0", [mass])
    sigma_landau = rt.RooRealVar("sigma_landau", "sigma_landau", 7, 0.5, 8.5)
    model2 = rt.RooLandau("bkg", "bkg", mass_neg, mean_landau, sigma_landau)

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
    final_model = rt.RooAddPdf("final_model", "final_model", [model1, model2], [sigfrac])

    time_step = time.time()
    # INFO: fitting directly to unbinned dataset is slow, so first make a histogram
    roo_hist = rt.RooDataHist("data_hist", "binned version of roo_dataset", rt.RooArgSet(mass), roo_dataset)
    rt.EnableImplicitMT()
    _ = final_model.fitTo(roo_hist, Save=True, EvalBackend="cpu")
    fit_result = final_model.fitTo(roo_hist, Save=True, EvalBackend="cpu")
    print(f"fitting elapsed time: {time.time() - time_step}")
    time.sleep(1) # INFO: Rest a second for stability

    # plotting
    roo_dataset.plotOn(frame, DataError="SumW2", Name="data_hist") # Name is explicitly defined so chiSquare can find it
    final_model.plotOn(frame, Name="final_model", LineColor=rt.kGreen)
    final_model.plotOn(frame, Components="signal", LineColor=rt.kBlue)
    final_model.plotOn(frame, Components="bkg", LineColor=rt.kRed)
    model1.paramOn(frame, Parameters=[sigma], Layout=[0.55, 0.94, 0.8])
    frame.GetYaxis().SetTitle("Events")
    frame.Draw()

    #calculate chi2 and add to plot
    n_free_params = fit_result.floatParsFinal().getSize()
    chi2 = frame.chiSquare(final_model.GetName(), "data_hist", n_free_params)
    chi2 = float('%.3g' % chi2)
    print(f"Number of free params: {n_free_params}, chi2: {chi2}")
    latex = rt.TLatex()
    latex.SetNDC()
    latex.SetTextAlign(11)
    latex.SetTextFont(42)
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.7, 0.8, f"#chi^2 = {chi2}")

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

    canvas.Update()
    canvas.SaveAs(f"calibration_fitCat{cat_idx}.pdf")
    del canvas

    print(f"mean_landau: {mean_landau.getVal()}")
    print(f"sigma_landau: {sigma_landau.getVal()}")
    print(f"n1: {n1.getVal()}")
    print(f"n2: {n2.getVal()}")
    print(f"alpha1: {alpha1.getVal()}")
    print(f"alpha2: {alpha2.getVal()}")
    print(f"sigma result for cat {cat_idx}: {sigma.getVal()} +- {sigma.getError()}")
    # save cat_idx and sigma value to a pandas dataframe
    new_row = pd.DataFrame([{"cat_name": cat_idx, "fit_val": sigma.getVal(), "fit_err": sigma.getError()}])
    df_fit = pd.concat([df_fit, new_row], ignore_index=True)

    # Save the cat_idx and sigma value to a log file
    with open(logfile, "a") as f:
        f.write(f"{cat_idx} {sigma.getVal()} {sigma.getError()}\n")

    return df_fit

def generate_roo_hist(x, dimuon_mass, wgts, name=""):
    dimuon_mass = np.asarray(ak.to_numpy(dimuon_mass)).astype(np.float64) # INFO: explicit float64 format is required
    wgts = np.asarray(ak.to_numpy(wgts)).astype(np.float64) # INFO: explicit float64 format is required
    nbins = x.getBins()
    TH = rt.TH1D("TH", "TH", nbins, x.getMin(), x.getMax())
    TH.FillN(len(dimuon_mass), dimuon_mass, wgts) # INFO: Fill the histogram with mass and weights
    roohist = rt.RooDataHist(name, name, rt.RooArgSet(x), TH)
    return roohist

def normalize_roo_hist(x, roo_hist):
    """
    Takes rootHistogram and returns a new copy with histogram values normalized to sum to one
    """
    x_name = x.GetName()
    THist = roo_hist.createHistogram(x_name).Clone("clone")
    THist.Scale(1 / THist.Integral())
    normalized_hist_name = roo_hist.GetName() + "_normalized"
    roo_hist_normalized = rt.RooDataHist(normalized_hist_name, normalized_hist_name, rt.RooArgSet(x), THist)
    return roo_hist_normalized

def fit_plot_ggh(events_bs_on, events_bs_off, save_filename, save_plot=True, selection="", region=""):
    """
    generate histogram from dimuon mass and wgt, fit DCB
    aftwards, plot the histogram and return the fit params
    as fit DCB sigma and chi2_dof
    """
    if selection:
        dimuon_eta_bs_on = events_bs_on.dimuon_eta
        dimuon_eta_bs_off = events_bs_off.dimuon_eta

        if selection == "BB":
            selection_bs_on = (dimuon_eta_bs_on > 0) & (dimuon_eta_bs_on < 0.9)
            selection_bs_off = (dimuon_eta_bs_off > 0) & (dimuon_eta_bs_off < 0.9)
        elif selection == "BE":
            selection_bs_on = (dimuon_eta_bs_on > 0.9) & (dimuon_eta_bs_on < 1.5)
            selection_bs_off = (dimuon_eta_bs_off > 0.9) & (dimuon_eta_bs_off < 1.5)
        elif selection == "EE":
            selection_bs_on = (dimuon_eta_bs_on > 1.5) & (dimuon_eta_bs_on < 2.4)
            selection_bs_off = (dimuon_eta_bs_off > 1.5) & (dimuon_eta_bs_off < 2.4)

        events_bs_on = events_bs_on[selection_bs_on]
        events_bs_off = events_bs_off[selection_bs_off]

    nbins = 80
    range_min = 0.0
    range_max = 0.1
    if region in ["EB", "EE", "EO"]:
        range_max = 0.1
        nbins = 50
    elif region == "BE":
        range_max = 0.03
        nbins = 70
    else:
        range_max = 0.05

    dimuon_mass = ak.to_numpy(events_bs_on.mu2_ptErr / events_bs_on.mu2_pt)
    # dimuon_mass = ak.to_numpy(events_bs_on.dimuon_mass)
    wgt = ak.to_numpy(events_bs_on.wgt_nominal)
    hist_bs_on = rt.TH1D("hist_bs_on", f"Region: {region}", nbins, range_min, range_max)
    for mass, weight in zip(dimuon_mass, wgt):
        hist_bs_on.Fill(mass, weight)

    dimuon_mass = ak.to_numpy(events_bs_off.mu2_ptErr / events_bs_off.mu2_pt)
    # dimuon_mass = ak.to_numpy(events_bs_off.dimuon_mass)
    wgt = ak.to_numpy(events_bs_off.wgt_nominal)
    hist_bs_off = rt.TH1D("hist_bs_off", f"Region: {region}", nbins, range_min, range_max)
    for mass, weight in zip(dimuon_mass, wgt):
        hist_bs_off.Fill(mass, weight)

    # Normalize histograms
    hist_bs_on.Scale(1.0 / hist_bs_on.Integral())
    hist_bs_off.Scale(1.0 / hist_bs_off.Integral())

    canvas = rt.TCanvas("Canvas", "Canvas", 800, 800)
    canvas.cd()
    legend = rt.TLegend(0.6, 0.60, 0.9, 0.9)

    rt.gStyle.SetOptStat(0000)

    hist_bs_on.SetLineColor(rt.kGreen)
    hist_bs_on.SetMarkerColor(rt.kGreen)
    hist_bs_on.SetMarkerStyle(20)
    hist_bs_on.SetMarkerSize(0.5)
    hist_bs_on.SetLineWidth(2)
    hist_bs_on.GetYaxis().SetTitle("A.U.")
    hist_bs_on.GetXaxis().SetTitle("#Delta p_{T}/p_{T}")
    hist_bs_on.GetXaxis().SetTitleOffset(1.2)
    hist_bs_on.GetYaxis().SetTitleOffset(1.2)
    hist_bs_on.GetXaxis().SetTitleSize(0.04)
    hist_bs_on.GetYaxis().SetTitleSize(0.04)
    # draw histogram with error bars
    # hist_bs_on.Draw("E")


    hist_bs_off.SetLineColor(rt.kBlue)
    hist_bs_off.SetMarkerColor(rt.kBlue)
    hist_bs_off.SetMarkerStyle(20)
    hist_bs_off.SetMarkerSize(0.5)
    hist_bs_off.SetLineWidth(2)
    # hist_bs_off.Draw("E SAME")

    # Use TRatioPlot to plot the ratio
    rp = rt.TRatioPlot(hist_bs_on, hist_bs_off)
    rp.Draw()
    rp.GetLowerRefYaxis().SetTitle("Ratio")
    rp.GetLowerRefYaxis().SetRangeUser(0.0, 5.0)
    rp.GetLowerRefGraph().SetMinimum(0.0)
    rp.GetLowerRefGraph().SetMaximum(5.0)

    rp.GetUpperPad().cd()
    legend.AddEntry(hist_bs_on, "BSC fit", "L")
    legend.AddEntry("", f"BSC mean: {round(hist_bs_on.GetMean(), 3)}", "")
    legend.AddEntry("", f"BSC sigma: {round(hist_bs_on.GetStdDev(), 3)}", "")
    legend.AddEntry(hist_bs_off, "geofit", "L")
    legend.AddEntry("", f"geofit mean: {round(hist_bs_off.GetMean(), 3)}", "")
    legend.AddEntry("", f"geofit sigma: {round(hist_bs_off.GetStdDev(), 3)}", "")
    legend.Draw()

    canvas.Update()
    canvas.SaveAs(save_filename)

def plot_hist_var(dimuon_mass_bs_on, wgt_bs_on, dimuon_mass_bs_off, wgt_bs_off, xlabel, title, nbins, range_min, range_max, save_filename, control_region="", region = "", save_plot=True):

    # print entries in each region
    print(f"Entries in {control_region} region: {len(dimuon_mass_bs_on)}")
    print(f"Entries in {control_region} region: {len(dimuon_mass_bs_off)}")

    # if region in ["EB", "EE", "EO"]:
    #     range_max = 0.1
    #     nbins = 50
    # elif region == "BE":
    #     range_max = 0.03
    #     nbins = 70
    # else:
    #     range_max = 0.05

    hist_bs_on = rt.TH1D("hist_bs_on", f"Region: {region}", nbins, range_min, range_max)
    for mass, weight in zip(dimuon_mass_bs_on, wgt_bs_on):
        hist_bs_on.Fill(mass, weight)

    hist_bs_off = rt.TH1D("hist_bs_off", f"Region: {region}", nbins, range_min, range_max)
    for mass, weight in zip(dimuon_mass_bs_off, wgt_bs_off):
        hist_bs_off.Fill(mass, weight)

    # Normalize histograms
    hist_bs_on.Scale(1.0 / hist_bs_on.Integral())
    hist_bs_off.Scale(1.0 / hist_bs_off.Integral())

    canvas = rt.TCanvas("Canvas", "Canvas", 800, 800)
    canvas.cd()
    legend = rt.TLegend(0.6, 0.60, 0.9, 0.9)

    rt.gStyle.SetOptStat(0000)

    hist_bs_on.SetLineColor(rt.kGreen)
    hist_bs_on.SetMarkerColor(rt.kGreen)
    hist_bs_on.SetMarkerStyle(20)
    hist_bs_on.SetMarkerSize(0.5)
    hist_bs_on.SetLineWidth(2)
    hist_bs_on.GetYaxis().SetTitle("A.U.")
    hist_bs_on.GetXaxis().SetTitle(xlabel)
    hist_bs_on.GetXaxis().SetTitleOffset(1.2)
    hist_bs_on.GetYaxis().SetTitleOffset(1.2)
    hist_bs_on.GetXaxis().SetTitleSize(0.04)
    hist_bs_on.GetYaxis().SetTitleSize(0.04)
    # draw histogram with error bars
    # hist_bs_on.Draw("E")


    hist_bs_off.SetLineColor(rt.kBlue)
    hist_bs_off.SetMarkerColor(rt.kBlue)
    hist_bs_off.SetMarkerStyle(20)
    hist_bs_off.SetMarkerSize(0.5)
    hist_bs_off.SetLineWidth(2)
    # hist_bs_off.Draw("E SAME")

    # Use TRatioPlot to plot the ratio
    rp = rt.TRatioPlot(hist_bs_on, hist_bs_off)
    rp.Draw()
    rp.GetLowerRefYaxis().SetTitle("Ratio")
    if control_region == "z-peak":
        rp.GetLowerRefYaxis().SetRangeUser(0.8, 1.2)
        rp.GetLowerRefGraph().SetMinimum(0.8)
        rp.GetLowerRefGraph().SetMaximum(1.2)
    else:
        rp.GetLowerRefYaxis().SetRangeUser(0.0, 5.0)
        rp.GetLowerRefGraph().SetMinimum(0.0)
        rp.GetLowerRefGraph().SetMaximum(5.0)

    rp.GetUpperPad().cd()
    legend.AddEntry(hist_bs_on, "BSC fit", "L")
    legend.AddEntry("", f"BSC mean: {round(hist_bs_on.GetMean(), 3)}", "")
    legend.AddEntry("", f"BSC sigma: {round(hist_bs_on.GetStdDev(), 3)}", "")
    legend.AddEntry(hist_bs_off, "geofit", "L")
    legend.AddEntry("", f"geofit mean: {round(hist_bs_off.GetMean(), 3)}", "")
    legend.AddEntry("", f"geofit sigma: {round(hist_bs_off.GetStdDev(), 3)}", "")
    legend.Draw()

    canvas.Update()
    canvas.SaveAs(save_filename)

    # cleare  memory
    hist_bs_on.Delete()
    hist_bs_off.Delete()
    canvas.Close()

if __name__ == "__main__":
    client = Client(n_workers=15, threads_per_worker=2, processes=True, memory_limit='8 GiB')
    fields_to_compute = [
        "wgt_nominal",
        "mu1_pt", "mu1_ptErr", "mu1_eta", "mu1_phi",
        "mu2_pt", "mu2_ptErr",  "mu2_eta", "mu2_phi",
        "dimuon_pt", "dimuon_eta", "dimuon_phi", "dimuon_mass",
        "event"
    ]

    load_path = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run3_nanoAODv12_BSOn/stage1_output/2022preEE/f1_0/data_*/*/*.parquet"
    events_data = dak.from_parquet(load_path)
    events_data = ak.zip({field: events_data[field] for field in fields_to_compute}).compute()
    events_bs_on = filter_region(events_data, region="signal")

    load_path = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run3_nanoAODv12_BSOff/stage1_output/2022preEE/f1_0/data_*/*/*.parquet"
    events_data = dak.from_parquet(load_path)
    events_data = ak.zip({field: events_data[field] for field in fields_to_compute}).compute()
    events_bs_off = filter_region(events_data, region="signal")

    fit_plot_ggh(events_bs_on, events_bs_off, "BSC_geofit_comparison_2022PreEE_dpT_all.pdf", save_plot=True, region="Inclusive")

    for region in ["BB", "BO", "OB", "OO", "BE", "EB", "EO", "OE", "EE"]:
        events_bs_on_region = filter_region_using_rapidity(events_bs_on, region)
        events_bs_off_region = filter_region_using_rapidity(events_bs_off, region)
        fit_plot_ggh(events_bs_on_region, events_bs_off_region, f"BSC_geofit_comparison_2022PreEE_dpT_{region}.pdf", save_plot=True, region=region)
