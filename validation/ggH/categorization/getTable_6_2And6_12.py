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
from modules.selection import filterRegion
from modules.fit_functions import getFEWZ_roospline
import ROOT
import ROOT as rt
import copy
import pandas as pd
import glob

# # Get the parent directory
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
# # Add it to sys.path
# sys.path.insert(0, parent_dir)
# # Now you can import your module
# from src.lib.histogram.plotting import plotFig_6_13



# def createUnitHistogram(name, nbins, xmin, xmax):
#     hist = ROOT.TH1D(name, name, nbins, xmin, xmax)
#     # Set all bin contents (1-based indexing for ROOT histograms)
#     for bin_idx in range(1, nbins + 1):
#         hist.SetBinContent(bin_idx, 1.0)
#         hist.SetBinError(bin_idx, 0.0)

#     return hist

# def createUnitHistogram_roofit(x):
#     name = "flat unit histogram"
#     nbins = x.getBins()       # Number of bins (default or defined)
#     xmin  = x.getMin()        # Lower range
#     xmax  = x.getMax()        # Upper range
#     print(f"nbins: {nbins}")
#     print(f"xmin: {xmin}")
#     print(f"xmax: {xmax}")
#     th1 = createUnitHistogram(name, nbins, xmin, xmax)
#     th1_rooHist = rt.RooDataHist(name, name, rt.RooArgSet(x), th1) 
#     return th1_rooHist

# def getColor(name):
#     # color_list = [
#     #     rt.kGreen,
#     #     rt.kBlue,
#     #     rt.kRed,
#     #     rt.kOrange,
#     #     rt.kViolet,
#     # ]
#     if "bwzredux" in name.lower():
#         return rt.kRed, rt.kSolid
#     elif "bwz" in name.lower() and "bern" in name.lower():
#         return rt.kBlue, rt.kSolid
#     elif "s-power" in name.lower():
#         return rt.kCyan, rt.kDashDotted
#     elif "s-exponential" in name.lower():
#         return rt.kOrange, rt.kDashed 
#     elif "bwz" in name.lower() and "gamma" in name.lower():
#         return rt.kGreen, rt.kDotted
#     elif "fewz" in name.lower() and "bern" in name.lower():
#         return rt.kViolet, rt.kDashDotted
#     elif "landau" in name.lower() and "bern" in name.lower():
#         return rt.kGray, rt.kDashDotted
#     else:
#         print("Error, color not available for the function!")
#         raise ValueError

# def getBWZ_gamma(x):
#     name = f"BWZ_a_coeff"
#     a_coeff_bwz = rt.RooRealVar(name,name, -0.02,-0.5,0.5)
#     name = "BWZ"
#     BWZ = rt.RooModZPdf(name, name, x, a_coeff_bwz) 

#     name = f"Gamma_a_coeff"
#     a_coeff_gamma = rt.RooRealVar(name,name, -0.00005,-0.05,0.05)
#     gamma = rt.RooGenericPdf("Gamma", "exp(@1*@0)/pow(@0,2)", rt.RooArgList(x, a_coeff_gamma))

#     name = f"frac"
#     frac = rt.RooRealVar(name,name, 0.5, 0.0, 1.0) 
#     name = "BWZGamma"
#     coreBWZGamma = rt.RooAddPdf(name, name, [BWZ, gamma], [frac])

#     param_l = [
#         a_coeff_bwz,
#         BWZ,
#         a_coeff_gamma,
#         gamma,
#         frac,
#     ]# list of variables to return so that they don't get deleted in python functions. Otherwise the roofit pdfs don't work
#     return coreBWZGamma, param_l

# def plot_6_19(dataDict_by_subCat, save_fname, nSubCats=5, apply_blind=True):
#     device = "cpu"
#     # nSubCats=1 # FIXME
#     for target_subCat in range(nSubCats):
#         dataDict_target = dataDict_by_subCat[target_subCat]
#         mass_name = "mh_ggh"
#         mass = rt.RooRealVar(mass_name, mass_name, 120, 110, 150)
#         # nbins = 800
#         nbins = 100
#         mass.setBins(nbins)
#         mass.setRange("hiSB", 135, 150 )
#         mass.setRange("loSB", 110, 115 )
#         mass.setRange("h_peak", 115, 135 )
#         mass.setRange("full", 110, 150 )
#         if apply_blind:
#             fit_range = "hiSB,loSB" 
#         else:
#             fit_range = "full" 
#         subCat_mass_arr  = ak.to_numpy(dataDict_target["dimuon_mass"]) # convert to numpy for rt.RooDataSet
#         if apply_blind:
#             dimuon_mass = subCat_mass_arr
#             h_peak = (dimuon_mass > 115) & (dimuon_mass < 135)
#             blind_filter = ~h_peak
#             subCat_mass_arr = subCat_mass_arr[blind_filter]
#         roo_datasetData = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
#         roo_histData = rt.RooDataHist("rooHist_BWZRedux","rooHist_BWZRedux", rt.RooArgSet(mass), roo_datasetData)
        
#         # fit BWZ redux
#         name = f"BWZ_Redux_a_coeff"
#         a_coeff = rt.RooRealVar(name,name, -0.02,-0.5,0.5)
#         name = f"BWZ_Redux_b_coeff"
#         b_coeff = rt.RooRealVar(name,name, -0.000111,-0.1,0.1)
#         name = f"BWZ_Redux_c_coeff"
#         c_coeff = rt.RooRealVar(name,name, 0.5,-10.0,10.0)
#         name = "BWZRedux" # source: https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/blob/5ae49dd944479b79af5692ff47fd7f1d9de16e91/interface/HMuMuRooPdfs.h#L11
#         coreBWZRedux = rt.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff) 
#         _ = coreBWZRedux.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0)
#         fitResult = coreBWZRedux.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,)
#         # print(f"fitResult: {fitResult}")

#         # fit Sum exp
#         name = f"RooSumTwoExpPdf_a1_coeff"
#         a1_coeff = rt.RooRealVar(name,name, 0.00001,-2.0,1)
#         name = f"RooSumTwoExpPdf_a2_coeff"
#         a2_coeff = rt.RooRealVar(name,name, 0.1,-2.0,1)
#         name = f"RooSumTwoExpPdf_f_coeff"
#         f_coeff = rt.RooRealVar(name,name, 0.9,0.0,1.0)
    
#         name = "S-Exponential"
#         coreSumExp = rt.RooSumTwoExpPdf(name, name, mass, a1_coeff, a2_coeff, f_coeff) 
#         _ = coreSumExp.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0)
#         fitResult = coreSumExp.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,)

#         # fit Sum Power law
#         name = f"RooSumTwoPowerLawPdf_a1_coeff"
#         a1_coeff_pow = rt.RooRealVar(name,name, 0.00001,-2.0,1)
#         name = f"RooSumTwoPowerLawPdf_a2_coeff"
#         a2_coeff_pow = rt.RooRealVar(name,name, 0.1,-2.0,1)
#         name = f"RooSumTwoPowerLawPdf_f_coeff"
#         f_coeff_pow = rt.RooRealVar(name,name, 0.9,0.0,1.0)
    
#         name = "S-Power-Law"
#         coreSumPow = rt.RooSumTwoPowerLawPdf(name, name, mass, a1_coeff_pow, a2_coeff_pow, f_coeff_pow) 
#         _ = coreSumPow.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0)
#         fitResult = coreSumPow.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,)

#         # fit BWZ Gamma
#         coreBWZGamma, param_l_bwz_gamma = getBWZ_gamma(mass)
#         _ = coreBWZGamma.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0)
#         fitResult = coreBWZGamma.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,)
#         print(f"coreBWZGamma : \n")
#         fitResult.Print()
#         # raise ValueError

#         # fit BWZ Gamma
#         coreBWZxBern, param_l_bwz_bern = getBWZxBern(mass)
#         _ = coreBWZxBern.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0)
#         fitResult = coreBWZxBern.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,)
#         # print(f"coreBWZxBern : \n")
#         # fitResult.Print()

#         # fit FEWZxBern
#         coreFEWZxBern, param_l_fewz_bern = getFEWZxBern(mass)
#         _ = coreFEWZxBern.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0)
#         fitResult = coreFEWZxBern.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True)
#         print(f"coreFEWZxBern : \n")
#         fitResult.Print()
        
#         # fit LandxBern
#         coreLandxBern, param_l_land_bern = getLandxBern(mass)
#         _ = coreLandxBern.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0)
#         fitResult = coreLandxBern.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True)
#         # print(f"coreLandxBern : \n")
#         # fitResult.Print()

        
        
        
#         # raise ValueError

#         # --------------------------------------------------------------------
#         # plot
#         # --------------------------------------------------------------------
#         name = "Canvas"
#         canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
#         canvas.cd()
#         # Define upper and lower pads
#         pad1 = ROOT.TPad("pad1", "Distribution", 0, 0.3, 1, 1.0)
#         pad2 = ROOT.TPad("pad2", "Ratio", 0, 0.0, 1, 0.3)
        
#         # Adjust margins
#         pad1.SetBottomMargin(0)  # Upper plot does not need bottom margin
#         pad2.SetTopMargin(0)     # Lower plot does not need top margin
#         pad2.SetBottomMargin(0.3)

#         pad1.SetTicks(2, 2)
#         pad2.SetTicks(2, 2)
#         pad1.Draw() # value plot
#         pad2.Draw() # ratio plot
    
#         pad1.cd()
#         legend = rt.TLegend(0.65,0.55,0.9,0.7)
#         frame = mass.frame()

#         # plot invisible data points for pdfs could be normalized
#         roo_histData.plotOn(frame, Invisible=True)
#         # legend.AddEntry(frame.getObject(int(frame.numItems())-1),"Data", "P")
        
#         # BWZRedux
#         color, style = getColor(coreBWZRedux.GetName())
#         name = coreBWZRedux.GetName()
#         coreBWZRedux.plotOn(frame, DataError="SumW2", Name=name, LineColor=color, LineStyle=style)
#         legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
#         # -----------------------------------------------------------
#         # BWZxBern
#         color, style = getColor(coreBWZxBern.GetName())
#         name = coreBWZxBern.GetName()
#         coreBWZxBern.plotOn(frame, DataError="SumW2", Name=name, LineColor=color, LineStyle=style)
#         legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
#         # -----------------------------------------------------------
#         # Sum Exp
#         color, style = getColor(coreSumExp.GetName())
#         name = coreSumExp.GetName()
#         coreSumExp.plotOn(frame, DataError="SumW2", Name=name, LineColor=color, LineStyle=style)
#         legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")

#         # -----------------------------------------------------------
#         # Sum Power
#         color, style = getColor(coreSumPow.GetName())
#         name = coreSumPow.GetName()
#         coreSumPow.plotOn(frame, DataError="SumW2", Name=name, LineColor=color, LineStyle=style)
#         legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")

#         # -----------------------------------------------------------
#         # BWZxGamma
#         color, style = getColor(coreBWZGamma.GetName())
#         name = coreBWZGamma.GetName()
#         coreBWZGamma.plotOn(frame, DataError="SumW2", Name=name, LineColor=color, LineStyle=style)
#         legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")

#         # -----------------------------------------------------------
#         # FEWZxBern
#         color, style = getColor(coreFEWZxBern.GetName())
#         name = coreFEWZxBern.GetName()
#         coreFEWZxBern.plotOn(frame, DataError="SumW2", Name=name, LineColor=color, LineStyle=style)
#         legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
        
#         # -----------------------------------------------------------
#         # LandxBern
#         color, style = getColor(coreLandxBern.GetName())
#         name = coreLandxBern.GetName()
#         coreLandxBern.plotOn(frame, DataError="SumW2", Name=name, LineColor=color, LineStyle=style)
#         legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")

#         # -----------------------------------------------------------
#         # Data
#         roo_histData.plotOn(frame)
#         legend.AddEntry(frame.getObject(int(frame.numItems())-1),"Data", "P")
        
        
#         # Draw frame and legend        
#         frame.Draw()
#         legend.Draw()
        
#         # ---------------------------------------------------------------
#         # ratio plot
#         # ---------------------------------------------------------------
#         pad2.cd()
#         ratio_frame= mass.frame()
        

#         flat_unit_hist = createUnitHistogram_roofit(mass)
#         # Note: DataError=None flag is needed to set our GetYaxis().SetRangeUser() to our desired values. Otherwise, the size of the error bars plays a role.
#         flat_unit_hist.plotOn(ratio_frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0), Invisible=True, DataError=None)

#         # BWZRedux
#         flat_pdf = ROOT.RooPolynomial("BWZRedux ratio", "BWZRedux ratio", mass)
#         color, style = getColor(flat_pdf.GetName())
#         flat_pdf.plotOn(ratio_frame, DataError="SumW2", LineColor=color, LineStyle=style)

#         # -----------------------------------------------------------
#         # BWZxBern
#         ratio_bwzxBern = rt.RooGenericPdf("BWZxBernstein ratio", "@0/@1", rt.RooArgList(coreBWZxBern,coreBWZRedux,))
#         color, style = getColor(ratio_bwzxBern.GetName())
#         ratio_bwzxBern.plotOn(ratio_frame, DataError="SumW2", LineColor=color, LineStyle=style)
#         # -----------------------------------------------------------
#         # Sum Exp
#         ratio_sumExp = rt.RooGenericPdf("S-Exponential ratio", "@0/@1", rt.RooArgList(coreSumExp,coreBWZRedux))
#         color, style = getColor(ratio_sumExp.GetName())
#         ratio_sumExp.plotOn(ratio_frame, DataError="SumW2", LineColor=color, LineStyle=style)
#         # -----------------------------------------------------------
#         # Sum Power
#         ratio_sumPow = rt.RooGenericPdf("S-Power-Law ratio", "@0/@1", rt.RooArgList(coreSumPow,coreBWZRedux))
#         color, style = getColor(ratio_sumPow.GetName())
#         ratio_sumPow.plotOn(ratio_frame, DataError="SumW2", LineColor=color, LineStyle=style)

#         # -----------------------------------------------------------
#         # BWZxGamma
#         ratio_bwzGamma = rt.RooGenericPdf("BWZGamma ratio", "@0/@1", rt.RooArgList(coreBWZGamma,coreBWZRedux))
#         color, style = getColor(ratio_bwzGamma.GetName())
#         ratio_bwzGamma.plotOn(ratio_frame, DataError="SumW2", LineColor=color, LineStyle=style)
#         # -----------------------------------------------------------
#         # FEWZxBern
#         ratio_FEWZxBern = rt.RooGenericPdf("FEWZxBernstein ratio", "@0/@1", rt.RooArgList(coreFEWZxBern,coreBWZRedux))
#         color, style = getColor(ratio_FEWZxBern.GetName())
#         ratio_FEWZxBern.plotOn(ratio_frame, DataError="SumW2", LineColor=color, LineStyle=style)
#         # -----------------------------------------------------------
#         # LandxBern
#         ratio_landXBern = rt.RooGenericPdf("LandauxBernstein ratio", "@0/@1", rt.RooArgList(coreLandxBern,coreBWZRedux))
#         color, style = getColor(ratio_landXBern.GetName())
#         ratio_landXBern.plotOn(ratio_frame, DataError="SumW2", LineColor=color, LineStyle=style)

        
        
#         # set ranges and label sizes after all things are plotted
#         ratio_frame.GetXaxis().SetLabelSize(0.10)
#         ratio_frame.SetXTitle(f"Dimuon Mass (GeV)")
#         ratio_frame.GetXaxis().SetTitleSize(0.10)
        
#         ratio_frame.GetYaxis().SetLabelSize(0.08)
#         ratio_frame.GetYaxis().SetRangeUser(0.98, 1.02)
#         ratio_frame.SetTitle("")
#         ratio_frame.Draw()
        
#         canvas.Update()
#         canvas.Draw()
#         if apply_blind:
#             canvas.SaveAs(f"{save_fname}_subCat{target_subCat}_blinded.pdf")
#         else:
#             canvas.SaveAs(f"{save_fname}_subCat{target_subCat}_unblinded.pdf")
            

def getYieldTable(sample_dict, samples2list, nSubCats=5):
    """
    take ggh and vbf samples and generate yield table by BDT category
    """
    # samples2list = ["ggh", "vbf"] # this is the column
    bdt_subCats = [f"ggH-cat{i}" for i in range(nSubCats)] # this is the rows

    # initialize empty pd df
    # Define column names and row names
    column_names = samples2list
    row_names = bdt_subCats
    
    # Create DataFrame
    df = pd.DataFrame(np.nan, index=row_names, columns=column_names)
    
    # Add a name to the row index
    df.index.name = 'Category'
    
    # Print the DataFrame
    # print(df)
    # df.to_csv("test.csv")

    # fill in the yield values.
    for sample in samples2list:
        sample_zip = sample_dict[sample]
        sample_wgt = sample_zip["wgt_nominal"]
        sample_bdt_cat = sample_zip["subCategory_idx"]
        print(f"sample_wgt: {sample_wgt}")
        if len(sample_wgt) ==0: continue
        # print(np.max(sample_bdt_cat))
        for cat in range(nSubCats):
            # print(f"{sample} cat: {cat}")
            cat_filter = sample_bdt_cat == cat
            cat_sample_wgt = sample_wgt[cat_filter]
            cat_sample_yield = np.sum(cat_sample_wgt)
            # df[sample][cat] = cat_sample_yield
            df.loc[f"ggH-cat{cat}", sample] = cat_sample_yield
            # print(f"{sample} cat{cat} yield: {cat_sample_yield}")

    print(df)
    return df

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
    # sample_groups = {
    #     "data" : "data*",
    # }
    sample_groups = {
        # "data": ["data"],
        "ggh": ["ggh"],
        "vbf": ["vbf"],
        "dy": ["dy"],
        "tt": ["tt"],
        "ewk": ["ewk"],
        "diboson": ["ww", "wz", "zz"],
        "other": ["other", "st"],
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
    for group, group_fnames in sample_groups.items():
        filelist = []
        for group_fname in group_fnames:
            full_load_path = load_path+f"*{group_fname}.parquet" 
            fname_filelist = glob.glob(full_load_path)
            filelist += fname_filelist
        # print(f"{group} filelist : {filelist}")
        # print(f"{group} filelist len: {len(filelist)}")
        
        # events = dak.from_parquet(full_load_path)
        if len(filelist) == 0: continue # empty list, then skip
        events = dak.from_parquet(filelist)
        _, events = filterRegion(events, region=args.region)
        sample_dict = fillSampleValues(events, sample_dict, group)

    # print(f"sample_dict: {sample_dict}")
    save_path = f"{args.save_path}/{args.label}_x_{args.category}/{args.year}_{args.region}/"
    samples2list = ["ggh", "vbf"]
    table_6_2 = getYieldTable(sample_dict, samples2list)
    table_6_2.to_csv(f"{save_path}/table6_2.csv")

    samples2list = ["dy",  "ewk", "tt", "diboson", "other"]
    table_6_12_abs = getYieldTable(sample_dict, samples2list)
    table_6_12_abs.to_csv(f"{save_path}/table6_12_absolute.csv")


    # Compute percentage per row
    table_6_12_percentage = table_6_12_abs.div(table_6_12_abs.sum(axis=1), axis=0) * 100 # values are relative yield values normalized over each row (BDT sub cat) in percentage
    table_6_12_percentage = table_6_12_percentage.round(1) # round to nearest tenth percent
    table_6_12_percentage.to_csv(f"{save_path}/table6_12.csv") 
    
        
    # getTable_6_12Absolute(sample_dict)
    # # normalize the values and save the pd df
    
    # # print(f"sample_dict: {sample_dict}")
    # print(f"dataDict_by_subCat: {dataDict_by_subCat}")
    
    # plot_6_19(dataDict_by_subCat, save_fname, apply_blind = apply_blind)
    

    