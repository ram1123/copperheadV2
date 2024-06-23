import numpy as np
import pickle
import awkward as ak
import dask_awkward as dak
from distributed import Client
from omegaconf import OmegaConf

from typing import Tuple, List, Dict
import ROOT as rt

# # functions for MVA related stuff start --------------------------------------------
# def prepare_features(df, training_features, variation="nominal", add_year=False):
#     #global training_features
#     if add_year:
#         features = training_features + ["year"]
#     else:
#         features = training_features
#     features_var = []
#     #print(features)
#     for trf in features:
#         if f"{trf}_{variation}" in df.fields:
#             features_var.append(f"{trf}_{variation}")
#         elif trf in df.fields:
#             features_var.append(trf)
#         else:
#             print(f"Variable {trf} not found in training dataframe!")
#     return features_var

    

# def evaluate_bdt(df, variation, model, parameters):

#     # filter out events neither h_peak nor h_sidebands
#     row_filter = (df.h_peak != 0) | (df.h_sidebands != 0)
#     df = df[row_filter]
    
#     # training_features = ['dimuon_cos_theta_cs', 'dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR', 'dimuon_eta', 'dimuon_phi', 'dimuon_phi_cs', 'dimuon_pt', 'dimuon_pt_log', 'jet1_eta_nominal', 'jet1_phi_nominal', 'jet1_pt_nominal', 'jet1_qgl_nominal', 'jet2_eta_nominal', 'jet2_phi_nominal', 'jet2_pt_nominal', 'jet2_qgl_nominal', 'jj_dEta_nominal', 'jj_dPhi_nominal', 'jj_eta_nominal', 'jj_mass_nominal', 'jj_mass_log_nominal', 'jj_phi_nominal', 'jj_pt_nominal', 'll_zstar_log_nominal', 'mmj1_dEta_nominal', 'mmj1_dPhi_nominal', 'mmj2_dEta_nominal', 'mmj2_dPhi_nominal', 'mmj_min_dEta_nominal', 'mmj_min_dPhi_nominal', 'mmjj_eta_nominal', 'mmjj_mass_nominal', 'mmjj_phi_nominal', 'mmjj_pt_nominal', 'mu1_eta', 'mu1_iso', 'mu1_phi', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_iso', 'mu2_phi', 'mu2_pt_over_mass', 'zeppenfeld_nominal']
#     training_features = [
#         'dimuon_cos_theta_cs', 'dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR', 'dimuon_eta', 'dimuon_phi', 'dimuon_phi_cs', 'dimuon_pt', 
#         'dimuon_pt_log', 'jet1_eta', 'jet1_phi', 'jet1_pt', 'jet1_qgl', 'jet2_eta', 'jet2_phi', 
#         'jet2_pt', 'jet2_qgl', 'jj_dEta', 'jj_dPhi', 'jj_eta', 'jj_mass', 'jj_mass_log', 
#         'jj_phi', 'jj_pt', 'll_zstar_log', 'mmj1_dEta', 'mmj1_dPhi', 'mmj2_dEta', 'mmj2_dPhi', 
#         'mmj_min_dEta', 'mmj_min_dPhi', 'mmjj_eta', 'mmjj_mass', 'mmjj_phi', 'mmjj_pt', 'mu1_eta', 'mu1_iso', 
#         'mu1_phi', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_iso', 'mu2_phi', 'mu2_pt_over_mass', 'zeppenfeld'
#     ]

    
#     # df['mu1_pt_over_mass'] = df['mu1_pt']/df['dimuon_mass']
#     # df['mu2_pt_over_mass'] = df['mu2_pt']/df['dimuon_mass']
#     # df['njets'] = ak.fill_none(df['njets'], value=0)

#     #df[df['njets_nominal']<2]['jj_dPhi_nominal'] = -1
#     none_val = -99.0
#     for field in df.fields:
#         df[field] = ak.fill_none(df[field], value= none_val)
#         inf_cond = (np.inf == df[field]) | (-np.inf == df[field]) 
#         df[field] = ak.where(inf_cond, none_val, df[field])
        
#     # print(f"df.h_peak: {df.h_peak}")
#     print(f"sum df.h_peak: {ak.sum(df.h_peak)}")
#     # overwrite dimuon mass for regions not in h_peak
#     not_h_peak = (df.h_peak ==0)
#     # df["dimuon_mass"] = ak.where(not_h_peak, 125.0,  df["dimuon_mass"])
    


#     # idk why mmj variables are overwritten something to double chekc later
#     df['mmj_min_dEta'] = df["mmj2_dEta"]
#     df['mmj_min_dPhi'] = df["mmj2_dPhi"]

#     # temporary definition of even bc I don't have it
#     if "event" not in df.fields:
#         df["event"] = np.arange(len(df.dimuon_pt))
    
#     features = prepare_features(df,training_features, variation=variation, add_year=False)
#     # features = training_features
#     #model = f"{model}_{parameters['years'][0]}"
#     # score_name = f"score_{model}_{variation}"
#     score_name = "BDT_score"

#     # df.loc[:, score_name] = 0
#     score_total = np.zeros(len(df['dimuon_pt']))
    
#     nfolds = 4
    
#     for i in range(nfolds):
#         # eval_folds are the list of test dataset chunks that each bdt is trained to evaluate
#         eval_folds = [(i + f) % nfolds for f in [3]]
#         # eval_filter = df.event.mod(nfolds).isin(eval_folds)
#         eval_filter = (df.event % nfolds ) == (np.array(eval_folds) * ak.ones_like(df.event))
#         scalers_path = f"{parameters['models_path']}/{model}/scalers_{model}_{i}.npy"
#         scalers = np.load(scalers_path, allow_pickle=True)
#         model_path = f"{parameters['models_path']}/{model}/{model}_{i}.pkl"

#         bdt_model = pickle.load(open(model_path, "rb"))
#         df_i = df[eval_filter]
#         # print(f"df_i: {len(df_i)}")
#         # print(len
#         if len(df_i) == 0:
#             continue
#         # df_i.loc[df_i.region != "h-peak", "dimuon_mass"] = 125.0
#         print(f"scalers: {scalers.shape}")
#         print(f"df_i: {df_i}")
#         df_i_feat = df_i[features]
#         # df_i_feat = np.transpose(np.array(ak.unzip(df_i_feat)))
#         df_i_feat = ak.concatenate([df_i_feat[field][:, np.newaxis] for field in df_i_feat.fields], axis=1)
#         print(f"df_i_feat[:,0]: {df_i_feat[:,0]}")
#         print(f'df_i.dimuon_cos_theta_cs: {df_i.dimuon_cos_theta_cs}')
#         # print(f"type df_i_feat: {type(df_i_feat)}")
#         # print(f"df_i_feat: {df_i_feat.shape}")
#         df_i_feat = ak.Array(df_i_feat)
#         df_i = (df_i_feat - scalers[0]) / scalers[1]
#         if len(df_i) > 0:
#             print(f"model: {model}")
#             prediction = np.array(
#                 # bdt_model.predict_proba(df_i.values)[:, 1]
#                 bdt_model.predict_proba(df_i_feat)[:, 1]
#             ).ravel()
#             print(f"prediction: {prediction}")
#             # df.loc[eval_filter, score_name] = prediction  # np.arctanh((prediction))
#             # score_total = ak.where(eval_filter, prediction, score_total)
#             score_total[eval_filter] = prediction

#     df[score_name] = score_total
#     return df

# # functions for MVA related stuff end --------------------------------------------

# # functions for Core pdf fitting (except teh actual core pdf) start --------------------------------------------
# def MakeBWZ_Redux(mass: rt.RooRealVar, order: int) ->Tuple[rt.RooAddPdf, Dict]:
#     # collect all variables that we don't want destroyed by Python once function ends
#     out_dict = {}
    
#     name = f"BWZ_Redux_a_coeff"
#     a_coeff = rt.RooRealVar(name,name, -0.00001,-0.001,0.001)
#     name = "exp_model_mass"
#     exp_model_mass = rt.RooExponential(name, name, mass, a_coeff)
    
#     mass_sq = rt.RooFormulaVar("mass_sq", "@0*@0", rt.RooArgList(mass))
#     name = f"BWZ_Redux_b_coeff"
#     b_coeff = rt.RooRealVar(name,name, -0.00001,-0.001,0.001)
    
#     name = "exp_model_mass_sq"
#     exp_model_mass_sq = rt.RooExponential(name, name, mass_sq, b_coeff)

#     # add in the variables and models
#     out_dict[a_coeff.GetName()] = a_coeff 
#     out_dict[exp_model_mass.GetName()] = exp_model_mass
#     out_dict[mass_sq.GetName()] = mass_sq
#     out_dict[b_coeff.GetName()] = b_coeff
#     out_dict[exp_model_mass_sq.GetName()] = exp_model_mass_sq
    
#     # make Z boson related stuff
#     bwWidth = rt.RooRealVar("bwWidth", "bwWidth", 2.5, 0, 30)
#     bwmZ = rt.RooRealVar("bwmZ", "bwmZ", 91.2, 90, 92)
#     bwWidth.setConstant(True)
#     bwmZ.setConstant(True)

#     # start multiplying them all
#     name = f"BWZ_Redux_c_coeff"
#     c_coeff = rt.RooRealVar(name,name, 2,-5.0,5.0)
#     BWZ_redux_main = rt.RooGenericPdf(
#         "BWZ_redux_main", "@1/ ( pow((@0-@2), @3) + 0.25*pow(@1, @3) )", rt.RooArgList(mass, bwWidth, bwmZ, c_coeff)
#     )
#     # add in the variables and models
#     out_dict[bwWidth.GetName()] = bwWidth 
#     out_dict[bwmZ.GetName()] = bwmZ 
#     out_dict[c_coeff.GetName()] = c_coeff 
#     out_dict[BWZ_redux_main.GetName()] = BWZ_redux_main 

#     name = "BWZ_Redux"
#     final_model = rt.RooProdPdf(name, name, [BWZ_redux_main, exp_model_mass, exp_model_mass_sq]) 
#     return (final_model, out_dict)

# def MakeBWZxBern(mass: rt.RooRealVar, order: int) ->Tuple[rt.RooAddPdf, Dict]:
#     """
#     params:
#     mass = rt.RooRealVar that we will fitTo
#     order = order of the sum of exponential, that we assume to be >= 2
#     """
#     # collect all variables that we don't want destroyed by Python once function ends
#     out_dict = {}


    
#     # make BernStein
#     bern_order = order-1
#     BernCoeff_list = []
#     for ix in range(bern_order):
#         name = f"Bernstein_c_{ix}"
#         if ix == 0:
#             coeff = rt.RooRealVar(name,name, 1,-5.0,5.0)
#         else:
#             coeff = rt.RooRealVar(name,name, 1,-5.0,5.0)
#         out_dict[name] = coeff # add variable to make python remember 
#         BernCoeff_list.append(coeff)
#     name = f"Bernstein_model_order_{bern_order}"
#     bern_model = rt.RooBernstein(name, name, mass, BernCoeff_list)
#     out_dict[name] = bern_model # add variable to make python remember

    
#     # make BWZ
#     bwWidth = rt.RooRealVar("bwWidth", "bwWidth", 2.5, 0, 30)
#     bwmZ = rt.RooRealVar("bwmZ", "bwmZ", 91.2, 90, 92)
#     bwWidth.setConstant(True)
#     bwmZ.setConstant(True)
#     out_dict[bwWidth.GetName()] = bwWidth 
#     out_dict[bwmZ.GetName()] = bwmZ 
    
#     name = "VanillaBW_model"
#     BWZ = rt.RooBreitWigner(name, name, mass, bwmZ,bwWidth)
#     # our BWZ model is also multiplied by exp(a* mass) as defined in the AN
#     name = "BWZ_exp_coeff"
#     expCoeff = rt.RooRealVar(name, name, -0.0, -3.0, 1.0)
#     name = "BWZ_exp_model"
#     exp_model = rt.RooExponential(name, name, mass, expCoeff)
#     # name = "BWZxExp"
#     # full_BWZ = rt.RooProdPdf(name, name, [BWZ, exp_model]) 

#     # add variables
#     out_dict[BWZ.GetName()] = BWZ 
#     out_dict[expCoeff.GetName()] = expCoeff 
#     out_dict[exp_model.GetName()] = exp_model 
#     # out_dict[full_BWZ.GetName()] = full_BWZ 
    
#     # multiply BWZ and Bernstein
#     name = f"BWZxBern_order_{order}"
#     # final_model = rt.RooProdPdf(name, name, [bern_model, full_BWZ]) 
#     final_model = rt.RooProdPdf(name, name, [bern_model, BWZ, exp_model]) 
   
#     return (final_model, out_dict)
    

# def MakeSumExponential(mass: rt.RooRealVar, order: int) ->Tuple[rt.RooAddPdf, Dict]:
#     """
#     params:
#     mass = rt.RooRealVar that we will fitTo
#     order = order of the sum of exponential, that we assume to be >= 2
#     returns:
#     rt.RooAddPdf
#     dictionary of variables with {variable name : rt.RooRealVar or rt.RooExponential} format mainly for keep python from
#     destroying these variables, but also useful in debugging
#     """
#     model_list = [] # list of RooExp models for RooAddPdf
#     a_i_list = [] # list of RooExp coeffs for RooAddPdf
#     rest_list = [] # list of rest of variables to save it from being destroyed
#     for ix in range(order):
#         name = f"S_exp_b_{ix}"
#         b_i = rt.RooRealVar(name, name, -0.5, -5.0, 1.0)
#         rest_list.append(b_i)
        
#         name = f"S_exp_model_{ix}"
#         model = rt.RooExponential(name, name, mass, b_i)
#         model_list.append(model)
        
#         if ix >0:
#             name = f"S_exp_a_{ix}"
#             a_i = rt.RooRealVar(name, name, 0.5, 0, 1.0)
#             a_i_list.append(a_i)
            
#     name = f"S_exp_order_{order}"
#     final_model = rt.RooAddPdf(name, name, model_list, a_i_list)
#     # collect all variables that we don't want destroyed by Python once function ends
#     out_dict = {}
#     for model in model_list:
#         out_dict[model.GetName()] = model
#     for a_i in a_i_list:
#         out_dict[a_i.GetName()] = a_i
#     for var in rest_list:
#         out_dict[var.GetName()] = var
#     return (final_model, out_dict)

from quickSMFtest_functions import MakeBWZ_Redux, MakeBWZxBern, MakeSumExponential,prepare_features,evaluate_bdt



if __name__ == "__main__":
    client =  Client(n_workers=31,  threads_per_worker=1, processes=True, memory_limit='4 GiB') 
    load_path = "/depot/cms/users/yun79/results/stage1/test_VBF-filter_JECon_07June2024/2018/f1_0/"
    full_load_path = load_path+f"/data_C/*/*.parquet"
    # full_load_path = load_path+f"/data_D/*/*.parquet"
    events = dak.from_parquet(full_load_path)

    # load and obtain MVA outputs
    events["dimuon_dEta"] = np.abs(events.mu1_pt -events.mu2_pt)
    events["dimuon_pt_log"] = np.log(events.dimuon_pt)
    events["jj_mass_log"] = np.log(events.jj_mass)
    events["ll_zstar_log"] = np.log(events.ll_zstar)
    events["mu1_pt_over_mass"] = events.mu1_pt / events.dimuon_mass
    events["mu2_pt_over_mass"] = events.mu2_pt / events.dimuon_mass
    
    
    training_features = [
        'dimuon_cos_theta_cs', 'dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR', 'dimuon_eta', 'dimuon_phi', 'dimuon_phi_cs', 'dimuon_pt', 
        'dimuon_pt_log', 'jet1_eta', 'jet1_phi', 'jet1_pt', 'jet1_qgl', 'jet2_eta', 'jet2_phi', 
        'jet2_pt', 'jet2_qgl', 'jj_dEta', 'jj_dPhi', 'jj_eta', 'jj_mass', 'jj_mass_log', 
        'jj_phi', 'jj_pt', 'll_zstar_log', 'mmj1_dEta', 'mmj1_dPhi', 'mmj2_dEta', 'mmj2_dPhi', 
        'mmj_min_dEta', 'mmj_min_dPhi', 'mmjj_eta', 'mmjj_mass', 'mmjj_phi', 'mmjj_pt', 'mu1_eta', 'mu1_iso', 
        'mu1_phi', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_iso', 'mu2_phi', 'mu2_pt_over_mass', 'zeppenfeld'
    ]
    for training_feature in training_features:
        if training_feature not in events.fields:
            print(f"mssing feature: {training_feature}")
    
    fields2load = training_features + ["h_peak", "h_sidebands", "dimuon_mass", "wgt_nominal_total"]
    events = events[fields2load]
    # load data to memory using compute()
    events = ak.zip({
        field : events[field] for field in events.fields
    }).compute()


    parameters = {
    "models_path" : "/depot/cms/hmm/vscheure/data/trained_models/"
    }
    # model_name = "BDTv12_2018"
    # model_name = "phifixedBDT_2018"
    model_name = "BDTperyear_2018"
    
    processed_events = evaluate_bdt(events, "nominal", model_name, parameters)

    # load BDT score edges for subcategory divison
    BDTedges_load_path = "../parameters/MVA/ggH/BDT_edges.yaml"
    edges = OmegaConf.load(BDTedges_load_path)
    year = "2018"
    edges = np.array(edges[year])

    # Calculate the subCategory index 
    BDT_score = processed_events["BDT_score"]
    n_edges = len(edges)
    BDT_score_repeat = ak.concatenate([BDT_score[:,np.newaxis] for i in range(n_edges)], axis=1)
    # BDT_score_repeat
    n_rows = len(BDT_score_repeat)
    edges_repeat = np.repeat(edges[np.newaxis,:],n_rows,axis=0)
    # edges_repeat.shape
    edge_idx = ak.sum( (BDT_score_repeat >= edges_repeat), axis=1)
    subCat_idx =  edge_idx - 1 # sub category index starts at zero
    processed_events["subCategory_idx"] = subCat_idx

    # comence roofit fitting for each subcategory 
    n_subCats = 5
    poly_order_by_cat = {
        0:2,
        1:2,
        2:2,
        3:3,
        4:3,
    }
    for cat_ix in range(5):
        subCat_filter = (processed_events["subCategory_idx"] == cat_ix)
        subCat_mass_arr = processed_events.dimuon_mass[subCat_filter]
        subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
        # start Root fit 
        name = "Canvas"
        canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
        canvas.cd()
        mass_name = "dimuon_mass"
        mass =  rt.RooRealVar(mass_name,"mass (GeV)",120,110,150)
        nbins = 80
        mass.setBins(nbins)
    
        # for debugging purposes -----------------
        binning = np.linspace(110, 150, nbins)
        np_hist, _ = np.histogram(subCat_mass_arr, bins=binning)
        print(f"np_hist: {np_hist}")
        # -------------------------------------------
        
    
        # set sideband mass range after initializing dataset (idk why this order matters, but that's how it's shown here https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/tutorial2023/parametric_exercise/?h=sideband#background-modelling)
        mass.setRange("loSB", 110, 115 )
        mass.setRange("hiSB", 135, 150 )
        mass.setRange("h_peak", 115, 135 )
        mass.setRange("full", 110, 150 )
        fit_range = "loSB,hiSB" # we're fitting bkg only
    
        
        order = 3
        BWZxBern, params_bern = MakeBWZxBern(mass, order)
        sumExp, params_exp = MakeSumExponential(mass, order)
        BWZ_Redux, params_redux =  MakeBWZ_Redux(mass, order)
        
        roo_dataset = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
        
        # initialize the categories
        
       
        # roo_dataset.Print()
        roo_hist = rt.RooDataHist("data_hist",f"binned version of roo_dataset of subcat {cat_ix}", rt.RooArgSet(mass), roo_dataset)  # copies binning from mass variable
        # roo_hist.Print()
    
        
        
    
        smfVarList = []
        smf_order= poly_order_by_cat[cat_ix]
        print(f"smf_order: {smf_order}")
        for ix in range(smf_order-1): # minus one bc the normalization constraint takes off one degree of freedom
            name = f"smf_{ix}"
            smf_coeff = rt.RooRealVar(name, name, 2.5, 0, 30)
            smfVarList.append(smf_coeff)
    
        polynomial_model = rt.RooPolynomial("pol", "pol", mass, smfVarList)
        name = "smf x model"
        final_model =  rt.RooProdPdf(name, name, [polynomial_model, sumExp]) 
        # final_model = BWZxBern
        
        rt.EnableImplicitMT()
        # _ = final_model.fitTo(roo_hist, rt.RooFit.Range(fit_range),  EvalBackend="cpu", Save=True, )
        # fit_result = final_model.fitTo(roo_hist, rt.RooFit.Range(fit_range),  EvalBackend="cpu", Save=True, )
        _ = final_model.fitTo(roo_hist, EvalBackend="cpu", Save=True, )
        fit_result = final_model.fitTo(roo_hist, EvalBackend="cpu", Save=True, )
    
    
        
        # draw on canvas
        frame = mass.frame()
    
        # apparently I have to plot invisible roo dataset for fit function plotting to work. Maybe this helps with normalization?
        roo_dataset.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0) )
        final_model.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=final_model.GetName(), LineColor=rt.kGreen)
        dataset_name = "data"
        roo_dataset.plotOn(frame, rt.RooFit.CutRange(fit_range),DataError="SumW2", Name=dataset_name)
    
    
        # legend
        legend = rt.TLegend(0.65,0.55,0.9,0.7)
        name=final_model.GetName()
        legend.AddEntry(name,name, "L")
        name="data"
        legend.AddEntry(name,name, "P")
        legend.Draw()
        
    
        frame.Draw()
        canvas.Update()
        canvas.Draw()
    
        canvas.SaveAs(f"./quick_plots/stage3_plot_SMF_subCat{cat_ix}_{final_model.GetName()}.pdf")
    
        # make SMF plots
        name = "Canvas"
        canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
        canvas.cd()
        hist_data = rt.TH1F("hist1", "Histogram for all data", 80, 110, 150)
        print(f"subCat_mass_arr.shape: {subCat_mass_arr.shape}")
        hist_data.FillN(len(mass_arr), mass_arr, np.ones(len(mass_arr)))
        
        # print(f"hist_data: {hist_data}")
        model_hist = BWZxBern.asTF(mass)
        # model_hist.Draw("EP")
        
        hist_data.Divide(model_hist)
        # normalize
        hist_data.Scale(1/hist_data.Integral(), "width")
        hist_data.Draw("EP")
    
        smf_model = polynomial_model.createHistogram("smf hist", mass,  rt.RooFit.Binning(80, 110, 150))
        # normalize
        smf_model.Scale(1/smf_model.Integral(), "width")
        smf_model.Draw("hist same")
        # polynomial_model.asTF(mass).Draw("hist same")
        
        # polynomial_model.asTF(mass).Draw("same")
        
        # # draw on canvas
        # frame = mass.frame()
        # # apparently I have to plot invisible roo dataset for fit function plotting to work. Maybe this helps with normalization?
        # roo_dataset.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0) )
        # final_model.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=final_model.GetName(), LineColor=rt.kGreen)
        # frame.Draw("hist same")
        
        canvas.Update()
        
        
        canvas.SaveAs(f"./quick_plots/stage3_plot_SMF_SMF_subCat{cat_ix}_{final_model.GetName()}.pdf")