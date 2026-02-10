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
import ROOT
import ROOT as rt
import copy
from array import array
ROOT.gStyle.SetOptStat(0) # remove stats box
from distributed import Client


# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
# Add it to sys.path
sys.path.insert(0, parent_dir)
# Now you can import your module
from src.lib.histogram.plotting import plotFig_6_13


def getColor(name):
    # color_list = [
    #     rt.kGreen,
    #     rt.kBlue,
    #     rt.kRed,
    #     rt.kOrange,
    #     rt.kViolet,
    # ]
    if "bwzredux" in name.lower():
        return rt.kRed, rt.kSolid
    elif "bwz" in name.lower() and "bern" in name.lower():
        return rt.kBlue, rt.kSolid
    elif "s-power" in name.lower():
        return rt.kCyan, rt.kDashDotted
    elif "s-exponential" in name.lower():
        return rt.kOrange, rt.kDashed 
    elif "bwz" in name.lower() and "gamma" in name.lower():
        return rt.kGreen, rt.kDotted
    elif "fewz" in name.lower() and "bern" in name.lower():
        return rt.kViolet, rt.kDashDotted
    elif "landau" in name.lower() and "bern" in name.lower():
        return rt.kGray, rt.kDashDotted
    else:
        print("Error, color not available for the function!")
        raise ValueError

def plot_6_8(bkg_variables, bdt_edges, nbins, xmin, xmax, save_fname, normalize=True, draw_mode="HIST", cat_idx=None):        
    BDT_scores = bkg_variables["BDT_score"]
    weights = bkg_variables["wgt_nominal"]
    dimuon_mass = bkg_variables["dimuon_mass"]
    
    bdt_edges = np.array(bdt_edges)
    # build one mask per bin [edges[i], edges[i+1])
    masks = [
        (BDT_scores >= bdt_edges[i]) & (BDT_scores < bdt_edges[i+1])
        for i in range(len(bdt_edges)-1)
    ]
    color_l = [
        rt.kBlack,
        rt.kCyan,
        rt.kBlue,
        rt.kOrange,
        rt.kGreen,
        rt.kRed,
    ]
    canvas = ROOT.TCanvas("c", "c", 800, 600)
    leg = ROOT.TLegend(0.75,0.75,1.0,1.0)
    hist_l = []
    BDT_cats = enumerate(masks)
    print(f"masks: {masks}")
    for i, m in BDT_cats:
        if cat_idx is not None:
            if i != cat_idx: # skip plotting cat_idx not specified
                continue
        legend_str = f"{bdt_edges[i]:.2f} <= BDT < {bdt_edges[i+1]:.2f}"
        # bin_bdt_scores = BDT_scores[m]
        bin_dimuon_mass = dimuon_mass[m]
        bin_wgts = weights[m]

        # print(f"bin_bdt_scores: {bin_bdt_scores}")
        print(f"bin_wgts: {bin_wgts}")
        print(f"bin_dimuon_mass: {bin_dimuon_mass}")
        
        bin_dimuon_mass = array('d', bin_dimuon_mass) # make the array double
        bin_wgts = array('d', bin_wgts) # make the array double

        THist = ROOT.TH1F(f"{i}_hist", f"{i}_hist", nbins, xmin, xmax)
        THist.FillN(len(bin_dimuon_mass), bin_dimuon_mass, bin_wgts)
        leg.AddEntry(THist, legend_str,"l")
        
        # Normalize
        if normalize and (THist.Integral()>0):
            THist.Scale(1/THist.Integral())
            THist.GetYaxis().SetTitle("A.U.")
            

        
        print(f"THist.Integral(): {THist.Integral()}")
        color = color_l[i]
        THist.SetLineColor(color)
        THist.SetMarkerColor(color)
        # draw_mode = "E"
        # draw_mode = "HIST"
        if i ==0:
            THist.SetTitle("")
            THist.GetXaxis().SetTitle("m_{\mu\mu} [GeV]")
            
            THist.Draw(f"{draw_mode}")
        else:
            THist.Draw(f"{draw_mode} SAME")

        hist_l.append(THist) # add to list so that THist doesn't get garbage collected in for loop

    leg.Draw()
    canvas.SetTicks(2, 2)
    canvas.Update()
    canvas.Draw()

    if cat_idx is not None:
        save_fname_addendum = f"_cat{cat_idx}"
    else:
        save_fname_addendum = ""
    if "E" in draw_mode:
        save_fname_final = f"{save_fname}_ErrBar{save_fname_addendum}"
    else:
        save_fname_final = f"{save_fname}{save_fname_addendum}"

    # save canvas as pdf and root file
    print(f"save_fname_final: {save_fname_final}")
    canvas.SaveAs(f"{save_fname_final}.pdf")
    f = ROOT.TFile(f"{save_fname_final}.root", "RECREATE")
    d = f.mkdir("plots")
    d.cd()
    canvas.Write()   # writes inside /plots/
    f.Close()

def plot_6_8BySubCat(bkg_variables, bdt_edges, nbins, xmin, xmax, save_fname, normalize=True, draw_mode="HIST", cat_idx=None):        
    subCategory_idx = bkg_variables["subCategory_idx"]
    weights = bkg_variables["wgt_nominal"]
    dimuon_mass = bkg_variables["dimuon_mass"]
    max_cat = np.max(subCategory_idx)
    bdt_edges = np.array(bdt_edges)
    # build one mask per bin [edges[i], edges[i+1])
    masks = [
        (subCategory_idx ==i)
        for i in range(max_cat + 1)
    ]
    color_l = [
        rt.kBlack,
        rt.kCyan,
        rt.kBlue,
        rt.kOrange,
        rt.kGreen,
        rt.kRed,
    ]
    canvas = ROOT.TCanvas("c", "c", 800, 600)
    leg = ROOT.TLegend(0.75,0.75,1.0,1.0)
    hist_l = []
    print(f"masks: {masks}")
    BDT_cats = enumerate(masks)
    for i, m in BDT_cats:
        if cat_idx is not None:
            if i != cat_idx: # skip plotting cat_idx not specified
                continue
        legend_str = f"BDT category {i}"
        # bin_bdt_scores = BDT_scores[m]
        bin_dimuon_mass = dimuon_mass[m]
        bin_wgts = weights[m]

        # print(f"bin_bdt_scores: {bin_bdt_scores}")
        print(f"bin_wgts: {bin_wgts}")
        print(f"bin_dimuon_mass: {bin_dimuon_mass}")
        
        bin_dimuon_mass = array('d', bin_dimuon_mass) # make the array double
        bin_wgts = array('d', bin_wgts) # make the array double

        THist = ROOT.TH1F(f"{i}_hist", f"{i}_hist", nbins, xmin, xmax)
        THist.FillN(len(bin_dimuon_mass), bin_dimuon_mass, bin_wgts)
        leg.AddEntry(THist, legend_str,"l")
        
        # Normalize
        if normalize and (THist.Integral()>0):
            THist.Scale(1/THist.Integral())
            THist.GetYaxis().SetTitle("A.U.")
            

        
        print(f"THist.Integral(): {THist.Integral()}")
        color = color_l[i]
        THist.SetLineColor(color)
        THist.SetMarkerColor(color)
        # draw_mode = "E"
        # draw_mode = "HIST"
        if i ==0:
            THist.SetTitle("")
            THist.GetXaxis().SetTitle("m_{\mu\mu} [GeV]")
            
            THist.Draw(f"{draw_mode}")
        else:
            THist.Draw(f"{draw_mode} SAME")

        hist_l.append(THist) # add to list so that THist doesn't get garbage collected in for loop

    leg.Draw()
    canvas.SetTicks(2, 2)
    canvas.Update()
    canvas.Draw()

    if cat_idx is not None:
        save_fname_addendum = f"_cat{cat_idx}"
    else:
        save_fname_addendum = ""
    if "E" in draw_mode:
        save_fname_final = f"{save_fname}BySubCat_ErrBar{save_fname_addendum}"
    else:
        save_fname_final = f"{save_fname}BySubCat{save_fname_addendum}"

    # save canvas as pdf and root file
    canvas.SaveAs(f"{save_fname_final}.pdf")
    f = ROOT.TFile(f"{save_fname_final}.root", "RECREATE")
    d = f.mkdir("plots")
    d.cd()
    canvas.Write()   # writes inside /plots/
    f.Close()

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
    "-base",
    "--base_path",
    dest="base_path",
    default="/depot/cms/users/yun79/hmm/copperheadV1clean",
    action="store",
    help="",
    )
    client =  Client(n_workers=30,  threads_per_worker=1, processes=True, memory_limit='10 GiB') 
    
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
    # load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean/{args.label}/{args.category}/stage2_outputForFig6_7/{year_param}/"
    load_path =f"{args.base_path}/{args.label}/{args.category}/stage2_outputForFig6_7/{year_param}/"
    
    # events = dak.from_parquet(f"{load_path}/*data.parquet")
    # print(events.fields)
    bdt_edges = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.0]
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
    sample_groups = {
        "data" : "data*",
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
    # for group, group_fname in sample_groups.items():
    #     full_load_path = load_path+f"*{group_fname}.parquet" 
    #     events = dak.from_parquet(full_load_path)
    #     events = filterRegion(events, region=args.region)
    #     sample_dict = fillSampleValues(events, sample_dict, group)

    full_load_path = load_path+f"processed_events_bkgMC*.parquet" 
    bkgMC_events = dak.from_parquet(full_load_path) 
    bkgMC_BDT_score = bkgMC_events.BDT_score.compute()
    bkgMC_wgt_nominal = bkgMC_events.wgt_nominal.compute()
    bkgMC_dimuon_mass = bkgMC_events.dimuon_mass.compute()
    bkgMC_subCategory_idx = bkgMC_events.subCategory_idx.compute()

    plot_setting_fname = "../../../src/lib/histogram/plot_settings_vbfCat_MVA_input.json"
    # plot_setting_fname = "plot_settings_vbfCat_MVA_input.json"
    with open(plot_setting_fname, "r") as file:
        plot_settings = json.load(file)
    plot_var = "BDT_score"
    binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
    
    # status = "Private"
    status = "Simulation"

    
    xmin = 110
    xmax = 150
    bdt_edges = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.0]
    n_bdt_cats = len(bdt_edges) -1
    bkgMC_BDT_score = ak.to_numpy(bkgMC_BDT_score)
    bkgMC_wgt_nominal = ak.to_numpy(bkgMC_wgt_nominal)
    bkgMC_dimuon_mass = ak.to_numpy(bkgMC_dimuon_mass)

    bkg_variables = {
        "BDT_score" : bkgMC_BDT_score,
        "wgt_nominal" : bkgMC_wgt_nominal,
        "dimuon_mass" : bkgMC_dimuon_mass,
        "subCategory_idx" : bkgMC_subCategory_idx,
    }
    print(f"bkgMC_BDT_score: {bkgMC_BDT_score}")
    print(f"bkgMC_wgt_nominal: {bkgMC_wgt_nominal}")

    nbins_l = [
        100,
        50,
        25
    ]
    for nbins in nbins_l:
        save_fname = f"plots/{args.label}_x_{args.category}/{args.year}_signal/fig_6_8Nbins{nbins}/Fig6_8"
        # Make directory if it doesn't exist
        os.makedirs(os.path.dirname(save_fname), exist_ok=True)
        
        
        
        plot_6_8(bkg_variables, bdt_edges, nbins, xmin, xmax, save_fname, normalize=True)
        plot_6_8BySubCat(bkg_variables, bdt_edges, nbins, xmin, xmax, save_fname, normalize=True)
        for cat_idx in range(n_bdt_cats):
            plot_6_8(bkg_variables, bdt_edges, nbins, xmin, xmax, save_fname, normalize=True, cat_idx=cat_idx)
            plot_6_8BySubCat(bkg_variables, bdt_edges, nbins, xmin, xmax, save_fname, normalize=True, cat_idx=cat_idx)
            
        plot_6_8(bkg_variables, bdt_edges, nbins, xmin, xmax, save_fname, normalize=True, draw_mode="E")
        plot_6_8BySubCat(bkg_variables, bdt_edges, nbins, xmin, xmax, save_fname, normalize=True, draw_mode="E")
        for cat_idx in range(n_bdt_cats):
            plot_6_8(bkg_variables, bdt_edges, nbins, xmin, xmax, save_fname, normalize=True, draw_mode="E", cat_idx=cat_idx)
            plot_6_8BySubCat(bkg_variables, bdt_edges, nbins, xmin, xmax, save_fname, normalize=True, draw_mode="E", cat_idx=cat_idx)
    

    