import ROOT
import ROOT as rt
import os
import uproot
from typing import Tuple, List, Dict

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
        UCSD_corefunc.plotOn(frame, rt.RooFit.NormRange("hiSB,loSB"), rt.RooFit.Range("full"), Name=model_name, LineColor=rt.kGreen)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1), f"UCSD {corefunc_name}", "L")
    
        model_name = Purdue_corefunc.GetName()
        Purdue_corefunc.plotOn(frame, rt.RooFit.NormRange("hiSB,loSB"), rt.RooFit.Range("full"), Name=model_name, LineColor=rt.kBlue)
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


def plotCorefuncComparisonBySubCat(mass:rt.RooRealVar, model_dict_by_subCat_n_corefunc: Dict, data_dict_by_subCat:Dict, save_path: str):
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
        UCSD_corefunc_dict = corefunc_dict["UCSD"]
        Purdue_corefunc_dict = corefunc_dict["Purdue"]
        for corefunc_name, UCSD_corefunc in UCSD_corefunc_dict.items():
            Purdue_corefunc = Purdue_corefunc_dict[corefunc_name]
            name = "Canvas"
            canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
            canvas.cd()
            frame = mass.frame()
            # frame.SetMaximum(max_list[subCat_idx])
            frame.SetXTitle(f"Dimuon Mass (GeV)")
            legend = rt.TLegend(0.65,0.55,0.9,0.7)
            # apparently I have to plot invisible roo dataset for fit function plotting to work. Maybe this helps with normalization?
            # data_hist = data_dict_by_subCat[subCat_idx]
            # data_hist.plotOn(frame, Name=data_hist.GetName())
            # for ix in range(len(subCat_list)):

            name = f"category {subCat_idx} data"
            data_hist.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0), Invisible=True )
            legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
        
            model_name = UCSD_corefunc.GetName()
            UCSD_corefunc.plotOn(frame, rt.RooFit.NormRange("hiSB,loSB"), rt.RooFit.Range("full"), Name=model_name, LineColor=rt.kGreen)
            legend.AddEntry(frame.getObject(int(frame.numItems())-1), f"UCSD {corefunc_name} cat{subCat_idx}", "L")
        
            model_name = Purdue_corefunc.GetName()
            Purdue_corefunc.plotOn(frame, rt.RooFit.NormRange("hiSB,loSB"), rt.RooFit.Range("full"), Name=model_name, LineColor=rt.kBlue)
            legend.AddEntry(frame.getObject(int(frame.numItems())-1), f"Purdue {corefunc_name} cat{subCat_idx}", "L")

            frame.Draw()
            legend.Draw() 
            canvas.SetTicks(2, 2)
            canvas.Update()
            canvas.Draw()
            canvas.SaveAs(f"{save_path}/bkgFitComparison_{corefunc_name}_subCat{subCat_idx}.pdf")


def plotCorefuncComparisonBySubCat(mass:rt.RooRealVar, data_dict_by_subCat:Dict, save_path: str):
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
    for subCat_idx, corefunc_dict in data_dict_by_subCat.items():
        UCSD_corefunc_dict = corefunc_dict["UCSD"]
        Purdue_corefunc_dict = corefunc_dict["Purdue"]
        Purdue_corefunc = Purdue_corefunc_dict[corefunc_name]
        name = "Canvas"
        canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
        canvas.cd()
        frame = mass.frame()
        frame.SetXTitle(f"Dimuon Mass (GeV)")
        legend = rt.TLegend(0.65,0.55,0.9,0.7)
        # apparently I have to plot invisible roo dataset for fit function plotting to work. Maybe this helps with normalization?
        # data_hist = data_dict_by_subCat[subCat_idx]
        # data_hist.plotOn(frame, Name=data_hist.GetName())
        # for ix in range(len(subCat_list)):

        name = f"category {subCat_idx} data"
        data_hist.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0), Invisible=True )
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    
        model_name = UCSD_corefunc.GetName()
        UCSD_corefunc.plotOn(frame, rt.RooFit.NormRange("hiSB,loSB"), rt.RooFit.Range("full"), Name=model_name, LineColor=rt.kGreen)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1), f"UCSD {corefunc_name} cat{subCat_idx}", "L")
    
        model_name = Purdue_corefunc.GetName()
        Purdue_corefunc.plotOn(frame, rt.RooFit.NormRange("hiSB,loSB"), rt.RooFit.Range("full"), Name=model_name, LineColor=rt.kBlue)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1), f"Purdue {corefunc_name} cat{subCat_idx}", "L")

        frame.Draw()
        legend.Draw() 
        canvas.SetTicks(2, 2)
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs(f"{save_path}/bkgFitComparison_dataHisComparision_subCat{subCat_idx}.pdf")

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
        
    plotCorefuncComparisonBySubCat(mass, model_dict_by_subCat_n_corefunc, data_dict_by_subCat, plot_save_path)


    purdue_rooHist_list = []
    for cat_ix in range(5):
        file = rt.TFile(f"my_workspace/workspace_bkg_cat{cat_ix}_ggh.root")
        purdue_rooHist_list.append(file["w"].obj(f"data_cat{cat_ix}_ggh"))
    purdue_combinedRooHist = addRooHists(mass, rooHist_list)
    
    data_dict_by_combinedNsubcat = {
        "combined" : {
            "UCSD" : combinedRooHist,
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
        out_dict["Pudue"] = data_hist
        data_dict_by_combinedNsubcat[f"cat{cat_ix}"] = out_dict


