import numpy as np
import pickle
import awkward as ak
import dask_awkward as dak
from distributed import Client
from omegaconf import OmegaConf

from typing import Tuple, List, Dict
import ROOT as rt


from quickSMFtest_functions import MakeBWZ_Redux, MakeBWZxBern, MakeSumExponential, MakeFEWZxBern, MakeBWZxBernFast, MakePowerSum



if __name__ == "__main__":
    """
    laoding stage1 output data and evalauting BDT is deletegated to run_stage2.py
    """
    # client =  Client(n_workers=31,  threads_per_worker=1, processes=True, memory_limit='4 GiB') 

    # load_path = "/work/users/yun79/stage2_output/test/processed_events.parquet"
    load_path = "/work/users/yun79/stage2_output/test/processed_events_data.parquet"
    processed_eventsData = ak.from_parquet(load_path)

    load_path = "/work/users/yun79/stage2_output/test/processed_events_signalMC.parquet"
    processed_eventsSignalMC = ak.from_parquet(load_path)

    
    print("events loaded!")
    
    # comence roofit fitting for each subcategory 
    n_subCats = 5
    poly_order_by_cat = {
        0:2,
        1:2,
        2:2,
        3:3,
        4:3,
    }
    dof = 3 # degrees of freedom for the core-functions. This should be same for all the functions
    
    # for cat_ix in range(5):
    for cat_ix in [0]:
        subCat_filter = (processed_eventsData["subCategory_idx"] == cat_ix)
        subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
        subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
        # start Root fit 
        name = "Canvas"
        canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
        canvas.cd()
        mass_name = "dimuon_mass"
        mass =  rt.RooRealVar(mass_name,mass_name,120,110,150)
        nbins = 80
        mass.setBins(nbins)
    
        # for debugging purposes -----------------
        binning = np.linspace(110, 150, nbins+1) # RooDataHist nbins are a bit different from np.histogram nbins apparently
        np_hist, _ = np.histogram(subCat_mass_arr, bins=binning)
        print(f"np_hist: {np_hist}")
        # -------------------------------------------
        
    
        # set sideband mass range after initializing dataset (idk why this order matters, but that's how it's shown here https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/tutorial2023/parametric_exercise/?h=sideband#background-modelling)
        mass.setRange("hiSB", 135, 150 )
        mass.setRange("loSB", 110, 115 )
        
        mass.setRange("h_peak", 115, 135 )
        mass.setRange("full", 110, 150 )
        # fit_range = "loSB,hiSB" # we're fitting bkg only
        fit_range = "hiSB,loSB" # we're fitting bkg only
    
        
       
        
        roo_datasetData = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
        
        # initialize the categories
        
       
        # roo_datasetData.Print()
        roo_histData = rt.RooDataHist("data_hist",f"binned version of roo_datasetData of subcat {cat_ix}", rt.RooArgSet(mass), roo_datasetData)  # copies binning from mass variable
        # roo_histData.Print()
    
        
        

        FEWZxBern, params_fewz = MakeFEWZxBern(mass, dof, roo_histData)
        # FEWZxBern_func, params_fewz = MakeFEWZxBern(mass, dof, roo_histData)
        # FEWZxBern = rt.RooGenericPdf("FEWZxBern", "Spline * Bernstein PDF", "@0", rt.RooArgList(FEWZxBern_func))
        
        # BWZxBern, params_bern = MakeBWZxBern(mass, dof)
        BWZxBern, params_bern = MakeBWZxBernFast(mass, dof)
        
        sumExp, params_exp = MakeSumExponential(mass, dof)
        BWZ_Redux, params_redux =  MakeBWZ_Redux(mass, dof)
        powerSum, params_power = MakePowerSum(mass, dof)
    
        smfVarList = []
        smf_order= poly_order_by_cat[cat_ix]
        print(f"smf_order: {smf_order}")
        for ix in range(smf_order-1): # minus one bc the normalization constraint takes off one degree of freedom
            name = f"smf_{ix}"
            smf_coeff = rt.RooRealVar(name, name, -0.005, -0.007, 0.007)
            smfVarList.append(smf_coeff)
    

        # shift = rt.RooRealVar("shift", "Offset", 125, 75, 150)
        # shift.setConstant(True)
        shifted_mass = rt.RooFormulaVar("shifted_mass", "@0-125", rt.RooArgList(mass))
        polynomial_model = rt.RooPolynomial("pol", "pol", shifted_mass, smfVarList)
        # -------------------------------------------------------
        name = f"smf x {powerSum.GetName()}"
        final_powerSum = rt.RooProdPdf(name, name, [polynomial_model,powerSum]) 
        name = f"smf x {BWZxBern.GetName()}"
        final_BWZxBern = rt.RooProdPdf(name, name, [polynomial_model,BWZxBern]) 
        # -------------------------------------------------------
        name = f"smf x {FEWZxBern.GetName()}"
        final_FEWZxBern = rt.RooProdPdf(name, name, [polynomial_model,FEWZxBern]) 
        name = f"smf x {sumExp.GetName()}"
        final_sumExp = rt.RooProdPdf(name, name, [polynomial_model,sumExp]) 
        name = f"smf x {BWZ_Redux.GetName()}"
        final_BWZ_Redux = rt.RooProdPdf(name, name, [polynomial_model,BWZ_Redux]) 
       
        
        
        rt.EnableImplicitMT()
        # _ = final_model.fitTo(roo_histData, rt.RooFit.Range(fit_range),  EvalBackend="cpu", Save=True, )
        # fit_result = final_model.fitTo(roo_histData, rt.RooFit.Range(fit_range),  EvalBackend="cpu", Save=True, )
        # _ = final_model.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )
        # fit_result = final_model.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )
        
        
        # -------------------------------------------------------
        # print("start final_powerSum !")
        # _ = final_powerSum.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )
        print("start final_BWZxBern !")
        _ = final_BWZxBern.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )
        # model = params_bern["BWZxBernFast_Bernstein_model_n_coeffs_3"]
        # model.Print()
        # print(f"model: {model}")
        # _ = model.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )
        # print("start Fewz Bern !")
        # _ = final_FEWZxBern.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )
        # -------------------------------------------------------
        print("start BWZ_Redux !")
        _ = final_BWZ_Redux.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )
        print("start sumExp !")
        _ = final_sumExp.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )
        
        # 
        
        # freeze the polynomial coefficient, and fine-tune the core functions
        for poly_coeff in smfVarList:
            poly_coeff.setConstant(True)

        # -------------------------------------------------------
        fit_result = final_powerSum.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )
        fit_result = final_BWZxBern.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )
        # fit_result = final_FEWZxBern.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )
        # -------------------------------------------------------
        fit_result = final_BWZ_Redux.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )
        fit_result = final_sumExp.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )
    
    
        
        # draw on canvas
        frame = mass.frame()
    
        # apparently I have to plot invisible roo dataset for fit function plotting to work. Maybe this helps with normalization?
        roo_datasetData.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0) )
        # final_model.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=final_model.GetName(), LineColor=rt.kGreen)
        final_BWZ_Redux.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=final_BWZ_Redux.GetName(), LineColor=rt.kGreen)
        final_sumExp.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=final_sumExp.GetName(), LineColor=rt.kBlue)
        # -------------------------------------------------------
        # final_powerSum.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=final_sumExp.GetName(), LineColor=rt.kRed)
        final_BWZxBern.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=final_sumExp.GetName(), LineColor=rt.kRed)
        # final_FEWZxBern.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=final_sumExp.GetName(), LineColor=rt.kRed)
        # -------------------------------------------------------
        dataset_name = "data"
        roo_datasetData.plotOn(frame, rt.RooFit.CutRange(fit_range),DataError="SumW2", Name=dataset_name)
        frame.Draw()
    
        # legend
        legend = rt.TLegend(0.65,0.55,0.9,0.7)
        name=final_BWZ_Redux.GetName()
        legend.AddEntry(name,name, "L")
        name=final_sumExp.GetName()
        legend.AddEntry(name,name, "L")
        # -------------------------------------------------------
        # name=final_powerSum.GetName()
        # legend.AddEntry(name,name, "L")
        name=final_BWZxBern.GetName()
        legend.AddEntry(name,name, "L")
        # name=final_FEWZxBern.GetName()
        # legend.AddEntry(name,name, "L")
        # -------------------------------------------------------
        name="data"
        legend.AddEntry(name,name, "P")
        legend.Draw()
        
    
        
        canvas.Update()
        canvas.Draw()
    
        # canvas.SaveAs(f"./quick_plots/stage3_plot_SMF_subCat{cat_ix}_{final_model.GetName()}.pdf")
        canvas.SaveAs(f"./quick_plots/stage3_plot_SMF_subCat{cat_ix}.pdf")

        # generate MutliPdf and save them into a workspace
        # trying multi pdf for like the 5th time
        cat = rt.RooCategory("pdf_index","Index of Pdf which is active");
    
        # // Make a RooMultiPdf object. The order of the pdfs will be the order of their index, ie for below
        # // 0 == BWZ_Redux
        # // 1 == sumExp
        # // 2 == PowerSum
    
        pdf_list = rt.RooArgList(
            final_BWZ_Redux,
            final_sumExp,
            # -------------------------------------------------------
            final_powerSum,
            final_BWZxBern,
            # final_FEWZxBern,
            # -------------------------------------------------------
        )
        print("just b4 roo multipdf")
        multipdf = rt.RooMultiPdf("roomultipdf","All Pdfs",cat,pdf_list)

        norm = rt.RooRealVar("roomultipdf_norm","Number of background events",1000,0,10000)

        # # inject a signal 
        # sigma = rt.RooRealVar("sigma","sigma",1.2); 
        # sigma.setConstant(True);
        # MH = rt.RooRealVar ("MH","MH",125); 
        # MH.setConstant(True)
        # signal =rt.RooGaussian ("signal","signal",mass,MH,sigma);

        #----------------------------------------------------------------------
        # fit signal
        
        subCat_filter = (processed_eventsSignalMC["subCategory_idx"] == cat_ix)
        subCat_mass_arrSigMC = processed_eventsSignalMC.dimuon_mass[subCat_filter]
        subCat_mass_arrSigMC  = ak.to_numpy(subCat_mass_arrSigMC) # convert to numpy for rt.RooDataSet
        
        # the mass range and nbins are taken from Fig 6.15 of the long AN (page 57)
        mass_name = "ggH_dimuon_mass"
        massSigMC =  rt.RooRealVar(mass_name,mass_name,125,110,140) # h peak range
        nbins = 80 # This could be wrong bc I counted by hand from the histogram
        massSigMC.setBins(nbins)
        
        roo_datasetSigMC = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arrSigMC}, [massSigMC])
        roo_datasetSigMC.SetName(f"ggH PowHeg MC subCat {cat_ix}")
        roo_histSigMC = rt.RooDataHist("SigMC_hist",f"binned version of SigMC of subcat {cat_ix}", rt.RooArgSet(massSigMC), roo_datasetSigMC)  # copies binning from mass variable
        # make roofit signal model
        mH = rt.RooRealVar("mH" , "mH", 125, 115,135)
        mH.setConstant(True) #
        sigma = rt.RooRealVar("sigma" , "sigma", 2, .1, 4.0)
        alpha1 = rt.RooRealVar("alpha1" , "alpha1", 2, 0.01, 65)
        n1 = rt.RooRealVar("n1" , "n1", 10, 0.01, 100)
        alpha2 = rt.RooRealVar("alpha2" , "alpha2", 2.0, 0.01, 65)
        n2 = rt.RooRealVar("n2" , "n2", 25, 0.01, 100)
        # n1.setConstant(True) # freeze for stability
        # n2.setConstant(True) # freeze for stability
        # dcb_name = f"ggH Signal Model subCat {cat_ix}"
        dcb_name = "signal"
        signal = rt.RooCrystalBall(dcb_name,dcb_name,massSigMC, mH, sigma, alpha1, n1, alpha2, n2)
        
        # fit signal model
        _ = signal.fitTo(roo_histSigMC,  EvalBackend="cpu", Save=True, )
        fit_result = signal.fitTo(roo_histSigMC,  EvalBackend="cpu", Save=True, )
        
        # clear canvas to plot the signal model
        canvas.Clear()
        frame = massSigMC.frame()
        roo_datasetSigMC.plotOn(frame, DataError="SumW2", Name=roo_datasetSigMC.GetName())
        signal.plotOn(frame, Name=signal.GetName(), LineColor=rt.kGreen)
        frame.Draw()
        
        
        # legend
        legend = rt.TLegend(0.65,0.55,0.9,0.7)
        name=signal.GetName()
        legend.AddEntry(name,name, "L")
        
        name=roo_datasetSigMC.GetName()
        legend.AddEntry(name,name, "P")
        legend.Draw()
        
        canvas.Update()
        canvas.Draw()
        
        canvas.SaveAs(f"./quick_plots/stage3_plot_SigMC_ggH{cat_ix}.pdf")

        # freeze other parameters b4 adding to workspace
        sigma.setConstant(True)
        alpha1.setConstant(True)
        n1.setConstant(True)
        alpha2.setConstant(True)
        n2.setConstant(True)
        #----------------------------------------------------------------------
        

        fout = rt.TFile("./workspace.root","RECREATE")
        wout = rt.RooWorkspace("workspace","workspace")
        roo_histData.SetName("data");
        wout.Import(roo_histData);
        wout.Import(cat);
        wout.Import(norm);
        wout.Import(multipdf);
        wout.Import(signal);
        wout.Print();
        wout.Write();
    
        # # make SMF plots
        # name = "Canvas"
        # canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
        # canvas.cd()
        # hist_data = rt.TH1F("hist1", "Histogram for all data", 80, 110, 150)
        # print(f"subCat_mass_arr.shape: {subCat_mass_arr.shape}")
        # hist_data.FillN(len(subCat_mass_arr), subCat_mass_arr, np.ones(len(subCat_mass_arr)))
        
        # # print(f"hist_data: {hist_data}")
        # model_hist = core_model.asTF(mass)
        # # model_hist.Draw("EP")
        
        # hist_data.Divide(model_hist)
        # # normalize
        # hist_data.Scale(1/hist_data.Integral(), "width")
        # hist_data.Draw("EP")
    
        # # # smf_hist = polynomial_model.createHistogram("smf hist", mass,  rt.RooFit.Binning(80, 110, 150))
        # # shift = 125
        # # shifted_mass_var =  rt.RooRealVar("shifted mass","mass (GeV)",120-shift,110-shift,150-shift)
        # # smf_hist = polynomial_model.createHistogram("smf hist", shifted_mass_var,  rt.RooFit.Binning(80, 110-shift, 150-shift))
        # # # normalize
        # # smf_hist.Scale(1/smf_hist.Integral(), "width")
        # # smf_hist.Draw("hist same")
        # # polynomial_model.asTF(mass).Draw("hist same")
    
        # frame = mass.frame()
        # # RooRatio("test", "test", roo_histData,)
        # # roo_histData.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0) )
        # polynomial_model.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), LineColor=rt.kGreen)
        # frame.Draw("hist same")
        
        # # polynomial_model.asTF(mass).Draw("same")
        
        # # # draw on canvas
        # # frame = mass.frame()
        # # # apparently I have to plot invisible roo dataset for fit function plotting to work. Maybe this helps with normalization?
        # # roo_datasetData.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0) )
        # # final_model.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=final_model.GetName(), LineColor=rt.kGreen)
        # # frame.Draw("hist same")
        
        # canvas.Update()
        
        
        # canvas.SaveAs(f"./quick_plots/stage3_plot_test_SMF_SMF_subCat{cat_ix}_{final_model.GetName()}.pdf")