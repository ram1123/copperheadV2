import numpy as np
import pickle
import awkward as ak
import dask_awkward as dak
from distributed import Client
from omegaconf import OmegaConf

from typing import Tuple, List, Dict
import ROOT as rt


from quickSMFtest_functions import MakeBWZ_Redux, MakeBWZxBern, MakeSumExponential, MakeFEWZxBern, MakeBWZxBernFast



if __name__ == "__main__":
    """
    laoding stage1 output data and evalauting BDT is deletegated to run_stage2.py
    """
    # client =  Client(n_workers=31,  threads_per_worker=1, processes=True, memory_limit='4 GiB') 
    # load_path = "/depot/cms/users/yun79/results/stage1/test_VBF-filter_JECon_07June2024/2018/f1_0"
    # # full_load_path = load_path+f"/data_C/*/*.parquet"
    # # full_load_path = load_path+f"/data_D/*/*.parquet"
    # # full_load_path = load_path+f"/data_*/*/*.parquet"
    # full_load_path = load_path+f"/data_A/*/*.parquet"
    # events = dak.from_parquet(full_load_path)

    # # load and obtain MVA outputs
    # events["dimuon_dEta"] = np.abs(events.mu1_pt -events.mu2_pt)
    # events["dimuon_pt_log"] = np.log(events.dimuon_pt)
    # events["jj_mass_log"] = np.log(events.jj_mass)
    # events["ll_zstar_log"] = np.log(events.ll_zstar)
    # events["mu1_pt_over_mass"] = events.mu1_pt / events.dimuon_mass
    # events["mu2_pt_over_mass"] = events.mu2_pt / events.dimuon_mass
    
    
    # training_features = [
    #     'dimuon_cos_theta_cs', 'dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR', 'dimuon_eta', 'dimuon_phi', 'dimuon_phi_cs', 'dimuon_pt', 
    #     'dimuon_pt_log', 'jet1_eta', 'jet1_phi', 'jet1_pt', 'jet1_qgl', 'jet2_eta', 'jet2_phi', 
    #     'jet2_pt', 'jet2_qgl', 'jj_dEta', 'jj_dPhi', 'jj_eta', 'jj_mass', 'jj_mass_log', 
    #     'jj_phi', 'jj_pt', 'll_zstar_log', 'mmj1_dEta', 'mmj1_dPhi', 'mmj2_dEta', 'mmj2_dPhi', 
    #     'mmj_min_dEta', 'mmj_min_dPhi', 'mmjj_eta', 'mmjj_mass', 'mmjj_phi', 'mmjj_pt', 'mu1_eta', 'mu1_iso', 
    #     'mu1_phi', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_iso', 'mu2_phi', 'mu2_pt_over_mass', 'zeppenfeld'
    # ]
    # for training_feature in training_features:
    #     if training_feature not in events.fields:
    #         print(f"mssing feature: {training_feature}")
    
    # fields2load = training_features + ["h_peak", "h_sidebands", "dimuon_mass", "wgt_nominal_total"]
    # events = events[fields2load]
    # # load data to memory using compute()
    # events = ak.zip({
    #     field : events[field] for field in events.fields
    # }).compute()


    # parameters = {
    # "models_path" : "/depot/cms/hmm/vscheure/data/trained_models/"
    # }
    # # model_name = "BDTv12_2018"
    # # model_name = "phifixedBDT_2018"
    # model_name = "BDTperyear_2018"
    
    # processed_events = evaluate_bdt(events, "nominal", model_name, parameters)

    # # load BDT score edges for subcategory divison
    # BDTedges_load_path = "../configs/MVA/ggH/BDT_edges.yaml"
    # edges = OmegaConf.load(BDTedges_load_path)
    # year = "2018"
    # edges = np.array(edges[year])

    # # Calculate the subCategory index 
    # BDT_score = processed_events["BDT_score"]
    # n_edges = len(edges)
    # BDT_score_repeat = ak.concatenate([BDT_score[:,np.newaxis] for i in range(n_edges)], axis=1)
    # # BDT_score_repeat
    # n_rows = len(BDT_score_repeat)
    # edges_repeat = np.repeat(edges[np.newaxis,:],n_rows,axis=0)
    # # edges_repeat.shape
    # edge_idx = ak.sum( (BDT_score_repeat >= edges_repeat), axis=1)
    # subCat_idx =  edge_idx - 1 # sub category index starts at zero
    # processed_events["subCategory_idx"] = subCat_idx

    load_path = "/work/users/yun79/stage2_output/test/processed_events.parquet"
    processed_events = ak.from_parquet(load_path)
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
        subCat_filter = (processed_events["subCategory_idx"] == cat_ix)
        subCat_mass_arr = processed_events.dimuon_mass[subCat_filter]
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
    
        
       
        
        roo_dataset = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
        
        # initialize the categories
        
       
        # roo_dataset.Print()
        roo_hist = rt.RooDataHist("data_hist",f"binned version of roo_dataset of subcat {cat_ix}", rt.RooArgSet(mass), roo_dataset)  # copies binning from mass variable
        # roo_hist.Print()
    
        
        

        FEWZxBern, params_fewz = MakeFEWZxBern(mass, dof, roo_hist)
        
        
        # FEWZxBern_func, params_fewz = MakeFEWZxBern(mass, dof, roo_hist)
        # FEWZxBern = rt.RooGenericPdf("FEWZxBern", "Spline * Bernstein PDF", "@0", rt.RooArgList(FEWZxBern_func))
        
        # BWZxBern, params_bern = MakeBWZxBern(mass, dof)
        BWZxBern, params_bern = MakeBWZxBernFast(mass, dof)
        sumExp, params_exp = MakeSumExponential(mass, dof)
        BWZ_Redux, params_redux =  MakeBWZ_Redux(mass, dof)
    
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
        # core_model = sumExp # BWZxBern , sumExp, BWZ_Redux
        # name = f"smf x {core_model.GetName()}"
        # final_model =  rt.RooProdPdf(name, name, [polynomial_model,core_model]) 
        name = f"smf x {BWZxBern.GetName()}"
        final_BWZxBern = rt.RooProdPdf(name, name, [polynomial_model,BWZxBern]) 
        name = f"smf x {FEWZxBern.GetName()}"
        final_FEWZxBern = rt.RooProdPdf(name, name, [polynomial_model,FEWZxBern]) 
        name = f"smf x {sumExp.GetName()}"
        final_sumExp = rt.RooProdPdf(name, name, [polynomial_model,sumExp]) 
        name = f"smf x {BWZ_Redux.GetName()}"
        final_BWZ_Redux = rt.RooProdPdf(name, name, [polynomial_model,BWZ_Redux]) 
        
        
        rt.EnableImplicitMT()
        # _ = final_model.fitTo(roo_hist, rt.RooFit.Range(fit_range),  EvalBackend="cpu", Save=True, )
        # fit_result = final_model.fitTo(roo_hist, rt.RooFit.Range(fit_range),  EvalBackend="cpu", Save=True, )
        # _ = final_model.fitTo(roo_hist, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )
        # fit_result = final_model.fitTo(roo_hist, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )
        
        
        # -------------------------------------------------------
        print("start final_BWZxBern !")
        _ = final_BWZxBern.fitTo(roo_hist, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )
        # _ = final_BWZxBern.fitTo(roo_hist, rt.RooFit.Range("full"), EvalBackend="cpu", Save=True, )
        # print("start Fewz Bern !")
        # _ = final_FEWZxBern.fitTo(roo_hist, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )
        # -------------------------------------------------------
        print("start BWZ_Redux !")
        _ = final_BWZ_Redux.fitTo(roo_hist, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )
        print("start sumExp !")
        _ = final_sumExp.fitTo(roo_hist, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )
        

        # freeze the polynomial coefficient, and fine-tune the core functions
        for poly_coeff in smfVarList:
            poly_coeff.setConstant(True)

        # -------------------------------------------------------
        fit_result = final_BWZxBern.fitTo(roo_hist, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )
        # fit_result = final_FEWZxBern.fitTo(roo_hist, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )
        # -------------------------------------------------------
        fit_result = final_BWZ_Redux.fitTo(roo_hist, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )
        fit_result = final_sumExp.fitTo(roo_hist, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )
    
    
        
        # draw on canvas
        frame = mass.frame()
    
        # apparently I have to plot invisible roo dataset for fit function plotting to work. Maybe this helps with normalization?
        roo_dataset.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0) )
        # final_model.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=final_model.GetName(), LineColor=rt.kGreen)
        final_BWZ_Redux.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=final_BWZ_Redux.GetName(), LineColor=rt.kGreen)
        final_sumExp.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=final_sumExp.GetName(), LineColor=rt.kBlue)
        # -------------------------------------------------------
        # final_FEWZxBern.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=final_sumExp.GetName(), LineColor=rt.kRed)
        final_BWZxBern.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=final_sumExp.GetName(), LineColor=rt.kRed)
        # -------------------------------------------------------
        dataset_name = "data"
        roo_dataset.plotOn(frame, rt.RooFit.CutRange(fit_range),DataError="SumW2", Name=dataset_name)
        frame.Draw()
    
        # legend
        legend = rt.TLegend(0.65,0.55,0.9,0.7)
        name=final_BWZ_Redux.GetName()
        legend.AddEntry(name,name, "L")
        name=final_sumExp.GetName()
        legend.AddEntry(name,name, "L")
        # -------------------------------------------------------
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
        # // 2 == FEWZxBern
    
        pdf_list = rt.RooArgList(
            # -------------------------------------------------------
            # final_BWZxBern,
            # params_bern["BWZxBern_Bernstein_model_n_coeffs_3"],
            BWZxBern,
            # final_FEWZxBern,
            # -------------------------------------------------------
            # final_BWZ_Redux,
            # final_sumExp,
        )
        print("just b4 roo multipdf")
        multipdf = rt.RooMultiPdf("roomultipdf","All Pdfs",cat,pdf_list)

        norm = rt.RooRealVar("roomultipdf_norm","Number of background events",1000,0,10000)

        # inject a signal 
        sigma = rt.RooRealVar("sigma","sigma",1.2); 
        sigma.setConstant(True);
        MH = rt.RooRealVar ("MH","MH",125); 
        MH.setConstant(True)
        signal =rt.RooGaussian ("signal","signal",mass,MH,sigma);

        fout = rt.TFile("./workspace.root","RECREATE")
        wout = rt.RooWorkspace("workspace","workspace")
        roo_hist.SetName("data");
        wout.Import(roo_hist);
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
        # # RooRatio("test", "test", roo_hist,)
        # # roo_hist.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0) )
        # polynomial_model.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), LineColor=rt.kGreen)
        # frame.Draw("hist same")
        
        # # polynomial_model.asTF(mass).Draw("same")
        
        # # # draw on canvas
        # # frame = mass.frame()
        # # # apparently I have to plot invisible roo dataset for fit function plotting to work. Maybe this helps with normalization?
        # # roo_dataset.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0) )
        # # final_model.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=final_model.GetName(), LineColor=rt.kGreen)
        # # frame.Draw("hist same")
        
        # canvas.Update()
        
        
        # canvas.SaveAs(f"./quick_plots/stage3_plot_test_SMF_SMF_subCat{cat_ix}_{final_model.GetName()}.pdf")