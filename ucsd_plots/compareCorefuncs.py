import ROOT
import ROOT as rt
import os
import uproot
from typing import Tuple, List, Dict
import pandas as pd

def addRooHists(x: rt.RooRealVar,rooHist_l: List[rt.RooDataHist]) -> rt.RooDataHist :
    """
    Takes rootHistogram and returns a new copy with histogram values all added on
    """
    x_name = x.GetName()
    THist = rooHist_l[0].createHistogram(x_name).Clone("clone") # clone it just in case
    print(f"{0}th THist.Integral(): {THist.Integral()}")
    print(f"{0}th rooHist_l.sumEntries(): {rooHist_l[0].sumEntries()}")
    for ix in range(1, len(rooHist_l)):
        THist_ix = rooHist_l[ix].createHistogram(x_name).Clone("clone")
        print(f"{ix}th THist.Integral(): {THist_ix.Integral()}")
        print(f"{ix}th rooHist_l.sumEntries(): {rooHist_l[ix].sumEntries()}")
        THist.Add(THist_ix)
    combinedHist_name = f"combined category of {x_name}"
    # THist.Print("v")
    print(f"roo_hist_combined.Integral(): {THist.Integral()}")
    roo_hist_combined = rt.RooDataHist(combinedHist_name, combinedHist_name, rt.RooArgSet(x), THist) 
    print(f"roo_hist_combined.sumEntries(): {roo_hist_combined.sumEntries()}")
    roo_hist_combined.Print("v")
    return roo_hist_combined



def rebinnHist(mass, rooHist):
    x_name = mass.GetName()
    THist = rooHist.createHistogram(x_name).Clone("clone") 
    # target_nbins = 100
    rebin_factor = 8
    THist = THist.Rebin(rebin_factor, "hist_rebinned")
    rebinned_rooHist = rt.RooDataHist(rooHist.GetName(), rooHist.GetName(), rt.RooArgSet(mass), THist)
    return rebinned_rooHist

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
    

def plotCombinedCorefunc_comparison(mass:rt.RooRealVar, rooHist_list, UCSD_corefunc_dict, Purdue_corefunc_dict, save_path: str):
    """
    takes the dictionary of all Bkg RooAbsPdf models grouped by same corefunctions, and plot them
    in the frame() of mass and saves the plots on a given directory path
    """
    # make the save_path directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # color_list = [
    #     rt.kGreen,
    #     rt.kBlue,
    #     rt.kRed,
    #     rt.kOrange,
    #     rt.kViolet,
    # ]
    for corefunc_name, UCSD_corefunc in UCSD_corefunc_dict.items():
        Purdue_corefunc = Purdue_corefunc_dict[corefunc_name]
        name = "Canvas"
        canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
        canvas.cd()
        frame = mass.frame()
        frame.SetTitle(f"Combined data histograms of all categories")
        frame.SetXTitle(f"Dimuon Mass (GeV)")
        legend = rt.TLegend(0.65,0.55,0.9,0.7)
        
        combinedRooHist = addRooHists(mass, rooHist_list)
        name = "combined category data"
        combinedRooHist.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0), Invisible=True )
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    
        model_name = UCSD_corefunc.GetName()
        UCSD_corefunc.plotOn(frame, rt.RooFit.NormRange("hiSB,loSB"), rt.RooFit.Range("full"),  ROOT.RooFit.DataError(None), Name=model_name, LineColor=rt.kGreen)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1), f"UCSD {corefunc_name}", "L")
    
        model_name = Purdue_corefunc.GetName()
        Purdue_corefunc.plotOn(frame, rt.RooFit.NormRange("hiSB,loSB"), rt.RooFit.Range("full"),  ROOT.RooFit.DataError(None), Name=model_name, LineColor=rt.kBlue)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1), f"Purdue {corefunc_name}", "L")
    
        # model = UCSD_corefunc
        # data_hist = combinedRooHist
        # # add chi2
        # ndf = model.getParameters(ROOT.RooArgSet(mass)).getSize()
        # print(model.GetName())
        # print(data_hist.GetName())
        # chi2_ndf = frame.chiSquare(model.GetName(), name, ndf)
        # model_name = model.GetName()
        # print(f"{model_name} ndf: {ndf}")
        # chi2_text = model_name +" chi2/ndf = {:.3f}".format(chi2_ndf)
        # legend.AddEntry("", chi2_text, "")
        
        frame.SetMaximum(6200)
        frame.Draw()
        legend.Draw() 
        canvas.SetTicks(2, 2)
        canvas.Update()
        canvas.Draw()
    
        canvas.SaveAs(f"{save_path}/combinedRooHistData_{corefunc_name}_comparison.pdf")


# def plotCorefuncComparisonBySubCat(mass:rt.RooRealVar, model_dict_by_subCat_n_corefunc: Dict, data_dict_by_subCat:Dict, save_path: str):
#     """
#     takes the dictionary of all Bkg RooAbsPdf models grouped by same sub-category, and plot them
#     in the frame() of mass and saves the plots on a given directory path
#     """
#     # make the save_path directory if it doesn't exist
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
        
#     color_list = [
#         rt.kGreen,
#         rt.kBlue,
#         rt.kRed,
#         rt.kOrange,
#         rt.kViolet,
#     ]
#     max_list = [1300, 1000, 400, 300, 90]
#     for subCat_idx, corefunc_dict in model_dict_by_subCat_n_corefunc.items():
#         UCSD_corefunc_dict = corefunc_dict["UCSD"]
#         Purdue_corefunc_dict = corefunc_dict["Purdue"]
#         for corefunc_name, UCSD_corefunc in UCSD_corefunc_dict.items():
#             Purdue_corefunc = Purdue_corefunc_dict[corefunc_name]
#             name = "Canvas"
#             canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
#             canvas.cd()
#             frame = mass.frame()
#             # frame.SetMaximum(max_list[subCat_idx])
#             frame.SetXTitle(f"Dimuon Mass (GeV)")
#             legend = rt.TLegend(0.65,0.55,0.9,0.7)
#             # apparently I have to plot invisible roo dataset for fit function plotting to work. Maybe this helps with normalization?
#             # data_hist = data_dict_by_subCat[subCat_idx]
#             # data_hist.plotOn(frame, Name=data_hist.GetName())
#             # for ix in range(len(subCat_list)):

#             name = f"category {subCat_idx} data"
#             data_hist.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0), Invisible=True )
#             legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
        
#             model_name = UCSD_corefunc.GetName()
#             UCSD_corefunc.plotOn(frame, rt.RooFit.NormRange("hiSB,loSB"), rt.RooFit.Range("full"), Name=model_name, LineColor=rt.kGreen)
#             legend.AddEntry(frame.getObject(int(frame.numItems())-1), f"UCSD {corefunc_name} cat{subCat_idx}", "L")
        
#             model_name = Purdue_corefunc.GetName()
#             Purdue_corefunc.plotOn(frame, rt.RooFit.NormRange("hiSB,loSB"), rt.RooFit.Range("full"), Name=model_name, LineColor=rt.kBlue)
#             legend.AddEntry(frame.getObject(int(frame.numItems())-1), f"Purdue {corefunc_name} cat{subCat_idx}", "L")

#             frame.Draw()
#             legend.Draw() 
#             canvas.SetTicks(2, 2)
#             canvas.Update()
#             canvas.Draw()
#             canvas.SaveAs(f"{save_path}/bkgFitComparison_{corefunc_name}_subCat{subCat_idx}.pdf")


def plotSignalModelComparisonBySubCat(mass:rt.RooRealVar, model_dict_by_subCat: Dict, data_dict_by_subCat:Dict, save_path: str, label="ggh"):
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
    for subCat_name, sigmodel_dict in model_dict_by_subCat.items():
        UCSD_sigmodel = sigmodel_dict["UCSD"]
        Purdue_sigmodel = sigmodel_dict["Purdue"]
        data_hist = data_dict_by_subCat[subCat_name]
        name = "Canvas"
        canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
        canvas.cd()
        frame = mass.frame()
        frame.SetXTitle(f"Dimuon Mass (GeV)")
        legend = rt.TLegend(0.65,0.55,0.9,0.7)
        legend.AddEntry("", f"Category {subCat_name}", "")

        data_hist.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0), Invisible=True )
            
        model_name = UCSD_sigmodel.GetName()
        UCSD_sigmodel.plotOn(frame, Name=model_name, LineColor=rt.kGreen)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1), f"UCSD", "L")
    
        model_name = Purdue_sigmodel.GetName()
        # Purdue_sigmodel.removeStringAttribute("h_peak")
        Purdue_sigmodel.plotOn(frame,rt.RooFit.NormRange("h_peak"), rt.RooFit.Range("full"), Name=model_name, LineColor=rt.kBlue)
        # Purdue_sigmodel.plotOn(frame, Name=model_name, LineColor=rt.kBlue)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1), f"Purdue", "L")

        UCSD_sigmodel.Print("V")
        Purdue_sigmodel.Print("V")

        frame.Draw()
        legend.Draw() 
        canvas.SetTicks(2, 2)
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs(f"{save_path}/{label}_signalComparison_{subCat_name}.pdf")
        canvas.SaveAs(f"{save_path}/{label}_signalComparison_{subCat_name}.png")


def plotCorefuncComparisonByCombined_n_SubCat(mass:rt.RooRealVar, data_dict_by_subCat:Dict, save_path: str, return_df=False, label="data", normalize=False):
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
    out_df = pd.DataFrame()
    print(f"data_dict_by_subCat: {data_dict_by_subCat}")
    for subCat_idx, data_dict in data_dict_by_subCat.items():
        # print(data_dict)
        # raise ValueError
        UCSD_data = data_dict["UCSD"]
        Purdue_data = data_dict["Purdue"]
        UCSD_data = rebinnHist(mass, UCSD_data)
        Purdue_data = rebinnHist(mass, Purdue_data)

        if normalize:
            UCSD_data = normalizeRooHist(mass, UCSD_data)
            Purdue_data = normalizeRooHist(mass, Purdue_data)
        
        name = "Canvas"
        canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
        canvas.cd()
        frame = mass.frame()
        frame.SetXTitle(f"Dimuon Mass (GeV)")
        legend = rt.TLegend(0.8,0.7,0.9,0.9)
        legend.AddEntry("", f"Category: {subCat_idx}", "")
        
        name = f"UCSD {subCat_idx} data"
        UCSD_data.plotOn(frame, ROOT.RooFit.DrawOption("E"), Name=name, LineColor=rt.kGreen)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1), f"UCSD", "L")
    
        name = f"Purdue {subCat_idx} data"
        Purdue_data.plotOn(frame,  ROOT.RooFit.DrawOption("E"), Name=name, LineColor=rt.kBlue)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1), f"Purdue", "L")

        ucsd_mean_val = UCSD_data.mean(mass)
        ucsd_std_dev = UCSD_data.sigma(mass)
        purdue_mean_val = Purdue_data.mean(mass)
        purdue_std_dev = Purdue_data.sigma(mass)

        if "mc" in label.lower():
            legend.AddEntry("", f"Mean UCSD: {ucsd_mean_val:.3f}",  "")
            legend.AddEntry("", f"Mean Purdue: {purdue_mean_val:.3f}",  "")
            legend.AddEntry("", f"Sigma UCSD: {ucsd_std_dev:.3f}",  "")
            legend.AddEntry("", f"Sigma Purdue: {purdue_std_dev:.3f}",  "")

        
        # print(f"Mean: {mean_val:.3f}")
        # print(f"Standard Deviation: {std_dev:.3f}")

        
        frame.Draw()
        legend.Draw() 
        canvas.SetTicks(2, 2)
        canvas.Update()
        canvas.Draw()
        if normalize:
            canvas.SaveAs(f"{save_path}/bkgFitComparison_dataHisComparision_subCat{subCat_idx}_{label}_normalized.pdf")
            canvas.SaveAs(f"{save_path}/bkgFitComparison_dataHisComparision_subCat{subCat_idx}_{label}_normalized.png")
        else:
            canvas.SaveAs(f"{save_path}/bkgFitComparison_dataHisComparision_subCat{subCat_idx}_{label}.pdf")
            canvas.SaveAs(f"{save_path}/bkgFitComparison_dataHisComparision_subCat{subCat_idx}_{label}.png")


        # add mean and std dev values
        # df_data = {
        #     "Institution" : ["UCSD", "Purdue"],
        #     "Category" : [subCat_idx, subCat_idx],
        #     "mean": [ucsd_mean_val, purdue_mean_val],
        #     "sigma" : [ucsd_std_dev, purdue_std_dev],
        # }
        df_data = {
            "Category" : [subCat_idx],
            "UCSD_mean": [ucsd_mean_val],
            "UCSD_sigma" : [ucsd_std_dev],
            "Purdue_mean": [purdue_mean_val],
            "Purdue_sigma" : [purdue_std_dev],
        }
        out_df = pd.concat([out_df, pd.DataFrame(df_data)], ignore_index=True)

    # end of loop, return out df if requested
    if return_df: 
        return out_df
    else: 
        return None


if __name__ == "__main__":

    plot_save_path = "./plots"
    mass_name = "mh_ggh"
    mass = rt.RooRealVar(mass_name, mass_name, 120, 110, 150)
    fit_range = "hiSB,loSB"
    plot_range = "full"
    nbins = 800
    mass.setBins(nbins)
    mass.setRange("hiSB", 135, 150 )
    mass.setRange("loSB", 110, 115 )
    mass.setRange("h_peak", 115, 135 )
    mass.setRange("full", 110, 150 )
    
    rooHist_list = []
    
    for cat_ix in range(5):
        file = rt.TFile(f"../ucsd_workspace/workspace_bkg_cat{cat_ix}_ggh.root")
        rooHist_list.append(file["w"].obj(f"data_cat{cat_ix}_ggh"))
    
    fewz_core = rt.TFile(f"../ucsd_workspace/workspace_bkg_cat{cat_ix}_ggh.root")["w"].obj(f"fewz_1j_spl_cat_ggh_pdf")
    BWZ_core = rt.TFile(f"../ucsd_workspace/workspace_bkg_cat{cat_ix}_ggh.root")["w"].obj(f"bwzr_cat_ggh_pdf")
    exp_core = rt.TFile(f"../ucsd_workspace/workspace_bkg_cat{cat_ix}_ggh.root")["w"].obj(f"exp_cat_ggh_pdf")
    UCSD_corefunc_dict = {
        "BWZ_Redux" : BWZ_core,
        "SumExp" : exp_core,
        "FEWZxBern" : fewz_core,
    }
    # Purdue_corefunc_dict = {
    #     "BWZ_Redux" : rt.TFile(f"../my_workspace/workspace_bkg_cat{cat_ix}_ggh.root")["w"].obj(f"subCat0_BWZ_Redux"),
    #     "SumExp" : rt.TFile(f"../my_workspace/workspace_bkg_cat{cat_ix}_ggh.root")["w"].obj(f"subCat0_sumExp"),
    #     "FEWZxBern" : rt.TFile(f"../my_workspace/workspace_bkg_cat{cat_ix}_ggh.root")["w"].obj(f"subCat0_FEWZxBern"),
    # }
    Purdue_corefunc_dict = {
        "BWZ_Redux" : rt.TFile(f"my_workspace/workspace_bkg_cat{cat_ix}_ggh.root")["w"].obj(f"bwzr_cat_ggh_pdf"),
        "SumExp" : rt.TFile(f"my_workspace/workspace_bkg_cat{cat_ix}_ggh.root")["w"].obj(f"exp_cat_ggh_pdf"),
        "FEWZxBern" : rt.TFile(f"my_workspace/workspace_bkg_cat{cat_ix}_ggh.root")["w"].obj(f"fewz_1j_spl_cat_ggh_pdf"),
    }
    rt.TFile(f"my_workspace/workspace_bkg_cat{cat_ix}_ggh.root")["w"].Print()
    # raise ValueError
    plotCombinedCorefunc_comparison(mass, rooHist_list, UCSD_corefunc_dict, Purdue_corefunc_dict, plot_save_path)


    # ----------------------------------------------------------------------------------------------
    # do same, but by subcat, with SMF applied
    # ----------------------------------------------------------------------------------------------
    
    data_dict_by_subCat = {}
    model_dict_by_subCat_n_corefunc = {}
    for cat_ix in range(5):
        file = rt.TFile(f"../ucsd_workspace/workspace_bkg_cat{cat_ix}_ggh.root")
        data_hist = file["w"].obj(f"data_cat{cat_ix}_ggh")
        data_dict_by_subCat[cat_ix] = data_hist
    
        UCSD_corefunc_dict = {
            "BWZ_redux" : file["w"].obj(f"bkg_bwzr_cat{cat_ix}_ggh_pdf"),
            "SumExp" : file["w"].obj(f"bkg_exp_cat{cat_ix}_ggh_pdf"),
            "FEWZxBern" : file["w"].obj(f"bkg_fewz_1j_spl_cat{cat_ix}_ggh_pdf") 
        }
        Purdue_corefunc_dict = {
            "BWZ_redux" : rt.TFile(f"my_workspace/workspace_bkg_cat{cat_ix}_ggh.root")["w"].obj(f"model_SubCat{cat_ix}_SMFxBWZRedux"),
            "SumExp" : rt.TFile(f"my_workspace/workspace_bkg_cat{cat_ix}_ggh.root")["w"].obj(f"model_SubCat{cat_ix}_SMFxSumExp"),
            "FEWZxBern" : rt.TFile(f"my_workspace/workspace_bkg_cat{cat_ix}_ggh.root")["w"].obj(f"model_SubCat{cat_ix}_SMFxFEWZxBern") 
        }
        # Purdue_corefunc_dict = {
        #     "BWZ_redux" : rt.TFile(f"../my_workspace/workspace_bkg_cat{cat_ix}_ggh.root")["w"].obj(f"bkg_bwzr_cat{cat_ix}_ggh_pdf"),
        #     "SumExp" : rt.TFile(f"../my_workspace/workspace_bkg_cat{cat_ix}_ggh.root")["w"].obj(f"bkg_exp_cat{cat_ix}_ggh_pdf"),
        #     "FEWZxBern" : rt.TFile(f"../my_workspace/workspace_bkg_cat{cat_ix}_ggh.root")["w"].obj(f"bkg_fewz_1j_spl_cat{cat_ix}_ggh_pdf") 
        # }
        model_dict_by_subCat_n_corefunc[cat_ix] = {
            "UCSD" : UCSD_corefunc_dict,
            "Purdue" : Purdue_corefunc_dict,
        }

    # Purdue_model_dict_by_subCat_n_corefunc = {}
    # for cat_ix in range(5):
    #     corefunc_dict = {
    #         "BWZ_redux" : rt.TFile(f"../my_workspace/workspace_bkg_cat{cat_ix}_ggh.root")["w"].obj(f"model_SubCat{cat_ix}_SMFxBWZRedux"),
    #         "SumExp" : rt.TFile(f"../my_workspace/workspace_bkg_cat{cat_ix}_ggh.root")["w"].obj(f"model_SubCat{cat_ix}_SMFxSumExp"),
    #         "FEWZxBern" : rt.TFile(f"../my_workspace/workspace_bkg_cat{cat_ix}_ggh.root")["w"].obj(f"model_SubCat{cat_ix}_SMFxFEWZxBern") 
    #     }
    #     Purdue_model_dict_by_subCat_n_corefunc[cat_ix] = corefunc_dict
        
    # plotCorefuncComparisonBySubCat(mass, model_dict_by_subCat_n_corefunc, data_dict_by_subCat, plot_save_path)


    ucsd_rooHist_list = []
    
    for cat_ix in range(5):
        file = rt.TFile(f"../ucsd_workspace/workspace_bkg_cat{cat_ix}_ggh.root")
        ucsd_rooHist_list.append(file["w"].obj(f"data_cat{cat_ix}_ggh"))
    
    ucsd_combinedRooHist = addRooHists(mass, ucsd_rooHist_list)
    
    purdue_rooHist_list = []
    for cat_ix in range(5):
        file = rt.TFile(f"my_workspace/workspace_bkg_cat{cat_ix}_ggh.root")
        purdue_rooHist_list.append(file["w"].obj(f"data_cat{cat_ix}_ggh"))
    
    purdue_combinedRooHist = addRooHists(mass, purdue_rooHist_list)
    
    data_dict_by_combinedNsubcat = {
        "combined" : {
            "UCSD" : ucsd_combinedRooHist,
            "Purdue" : purdue_combinedRooHist
                     },
    }
    for cat_ix in range(5):
        out_dict = {}
        file = rt.TFile(f"../ucsd_workspace/workspace_bkg_cat{cat_ix}_ggh.root")
        data_hist = file["w"].obj(f"data_cat{cat_ix}_ggh")
        out_dict["UCSD"] = data_hist
        file = rt.TFile(f"my_workspace/workspace_bkg_cat{cat_ix}_ggh.root")
        data_hist = file["w"].obj(f"data_cat{cat_ix}_ggh")
        out_dict["Purdue"] = data_hist
        data_dict_by_combinedNsubcat[f"cat{cat_ix}"] = out_dict

    plotCorefuncComparisonByCombined_n_SubCat(mass, data_dict_by_combinedNsubcat, plot_save_path, label="data")
    
    plotCorefuncComparisonByCombined_n_SubCat(mass, data_dict_by_combinedNsubcat, plot_save_path, label="data", normalize=True)


    raise ValueError

    # ------------------------------------------
    # do the same with signal histogram
    # ------------------------------------------

    # ------------------------------------------
    # ggH signal
    # ------------------------------------------
    
    ucsd_rooHist_list = []
    
    for cat_ix in range(5):
        file = rt.TFile(f"../ucsd_workspace/workspace_sig_cat{cat_ix}_ggh.root")
        ucsd_rooHist_list.append(file["w"].obj(f"data_ggH_cat{cat_ix}_ggh_m125"))
    
    ucsd_combinedRooHist = addRooHists(mass, ucsd_rooHist_list)
    
    purdue_rooHist_list = []
    for cat_ix in range(5):
        file = rt.TFile(f"my_workspace/workspace_sig_cat{cat_ix}_ggh.root")
        purdue_rooHist_list.append(file["w"].obj(f"data_ggH_cat{cat_ix}_ggh"))
    
    purdue_combinedRooHist = addRooHists(mass, purdue_rooHist_list)

    data_dict_by_combinedNsubcat = {
        "combined" : {
            "UCSD" : ucsd_combinedRooHist,
            "Purdue" : purdue_combinedRooHist
                     },
    }
    for cat_ix in range(5):
        out_dict = {}
        file = rt.TFile(f"../ucsd_workspace/workspace_sig_cat{cat_ix}_ggh.root")
        data_hist = file["w"].obj(f"data_ggH_cat{cat_ix}_ggh_m125")
        out_dict["UCSD"] = data_hist
        file = rt.TFile(f"my_workspace/workspace_sig_cat{cat_ix}_ggh.root")
        data_hist = file["w"].obj(f"data_ggH_cat{cat_ix}_ggh")
        out_dict["Purdue"] = data_hist
        data_dict_by_combinedNsubcat[f"cat{cat_ix}"] = out_dict

    ggh_df = plotCorefuncComparisonByCombined_n_SubCat(mass, data_dict_by_combinedNsubcat, plot_save_path, label="ggh_MC", return_df=True)
    print(ggh_df)

    ggh_df.to_csv("ggh_df.csv")

    
    # ------------------------------------------
    # compare ggH signal fits
    # ------------------------------------------

    
    data_dict_by_subCat = {}
    model_dict_by_subCat = {}
    for ix in range(5):
        cat_ix = f"cat{ix}"
        file = rt.TFile(f"../ucsd_workspace/workspace_sig_cat{ix}_ggh.root")
        # file["w"].Print("v")
        # raise ValueError
        data_hist = file["w"].obj(f"data_ggH_cat{ix}_ggh_m125")
        # data_dict_by_subCat[cat_ix] = data_hist
        data_dict_by_subCat[cat_ix] = rt.TFile(f"my_workspace/workspace_sig_cat{ix}_ggh.root")["w"].obj(f"data_ggH_cat{ix}_ggh")
    
        UCSD_model = file["w"].obj(f"ggH_cat{ix}_ggh_pdf")
        Purdue_model = rt.TFile(f"my_workspace/workspace_sig_cat{ix}_ggh.root")["w"].obj(f"ggH_cat{ix}_ggh_pdf")
        model_dict_by_subCat[cat_ix] = {
            "UCSD" : UCSD_model,
            "Purdue" : Purdue_model,
        }
    plotSignalModelComparisonBySubCat(mass, model_dict_by_subCat, data_dict_by_subCat, plot_save_path, label="ggh")

    raise ValueError
    
    # # ------------------------------------------
    # # VBF signal
    # # ------------------------------------------
    
    # ucsd_rooHist_list = []
    
    # for cat_ix in range(5):
    #     file = rt.TFile(f"../ucsd_workspace/workspace_sig_cat{cat_ix}_ggh.root")
    #     ucsd_rooHist_list.append(file["w"].obj(f"data_qqH_cat{cat_ix}_ggh_m125"))
    
    # ucsd_combinedRooHist = addRooHists(mass, ucsd_rooHist_list)
    
    # purdue_rooHist_list = []
    # for cat_ix in range(5):
    #     file = rt.TFile(f"my_workspace/workspace_sig_cat{cat_ix}_ggh.root")
    #     purdue_rooHist_list.append(file["w"].obj(f"data_qqH_cat{cat_ix}_ggh"))
    
    # purdue_combinedRooHist = addRooHists(mass, purdue_rooHist_list)

    # data_dict_by_combinedNsubcat = {
    #     "combined" : {
    #         "UCSD" : ucsd_combinedRooHist,
    #         "Purdue" : purdue_combinedRooHist
    #                  },
    # }
    # for cat_ix in range(5):
    #     out_dict = {}
    #     file = rt.TFile(f"../ucsd_workspace/workspace_sig_cat{cat_ix}_ggh.root")
    #     data_hist = file["w"].obj(f"data_qqH_cat{cat_ix}_ggh_m125")
    #     out_dict["UCSD"] = data_hist
    #     file = rt.TFile(f"my_workspace/workspace_sig_cat{cat_ix}_ggh.root")
    #     data_hist = file["w"].obj(f"data_qqH_cat{cat_ix}_ggh")
    #     out_dict["Purdue"] = data_hist
    #     data_dict_by_combinedNsubcat[f"cat{cat_ix}"] = out_dict

    # vbf_df = plotCorefuncComparisonByCombined_n_SubCat(mass, data_dict_by_combinedNsubcat, plot_save_path, label="vbf_MC", return_df=True)
    # print(ggh_df)

    # vbf_df.to_csv("vbf_df.csv")



