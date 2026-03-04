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
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
# Add it to sys.path
sys.path.insert(0, parent_dir)
# Now you can import your module
from src.lib.histogram.plotting import plotFig_6_13



def tranformBDT_score(computed_zip):
    """
    helper function that changes the range from [0,1] to [-1,-1]
    """
    score_name = "BDT_score"
    BDT_score = computed_zip[score_name]
    computed_zip[score_name] = (BDT_score-0.5)*2
    return computed_zip

def tranformBDT_edges(bin_edges):
    """
    helper function that transforms bin edges from [0,1] to [-1,-1]
    """
    transformed_edges = []
    for bin_edge in bin_edges:
        if bin_edge <=0 or bin_edge>=1:
            continue
        transformed_edge = (bin_edge-0.5)*2
        transformed_edges.append(transformed_edge)
    return transformed_edges

def getHWHM(fwhm, counts, bin_centers):
    hwhm = fwhm/2
    max_ix = np.argmax(counts)
    max_center = bin_centers[max_ix]
    bin_center_l = max_center - hwhm/2
    bin_center_r = max_center + hwhm/2
    # print(f"hwhm: {hwhm:.3f}")
    # print(f"max_center: {max_center:.3f}, Left: {bin_center_l:.3f}, Right: {bin_center_r:.3f}")
    
    return bin_center_l, bin_center_r
    
def compute_hwhm_with_edges(counts, bin_edges):
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    max_count = np.max(counts)
    half_max = max_count / 2

    # Identify bins where counts are above half max
    above_half_max = counts >= half_max
    indices = np.where(above_half_max)[0]

    if len(indices) < 2:
        return None, None, None

    left_idx = indices[0]
    right_idx = indices[-1]

    fwhm = bin_centers[right_idx] - bin_centers[left_idx]
    bin_center_l, bin_center_r = getHWHM(fwhm, counts, bin_centers)
    hwhm = fwhm/2
    return hwhm, bin_center_l, bin_center_r


def getHWHM_withEdges(sample_dict, binning):
    dimuon_mass = sample_dict["dimuon_mass"]
    wgt_nominal = sample_dict["wgt_nominal"]
    hist, _ = np.histogram(dimuon_mass, bins=binning, weights=wgt_nominal)
    # print(f"hist: {hist}")
    # print(f"hist len: {len(hist)}")
    hwhm, bin_center_l, bin_center_r = compute_hwhm_with_edges(hist, binning)
    return hwhm, bin_center_l, bin_center_r

def getYield_hwhm(sample_dict, hwhm_left, hwhm_right):
    dimuon_mass = sample_dict["dimuon_mass"]
    wgt_nominal = sample_dict["wgt_nominal"]
    
    hwhm_filter = (hwhm_left <= dimuon_mass) & (hwhm_right >= dimuon_mass)
    hwhm_yield = np.sum(wgt_nominal[hwhm_filter])
    # print(f"hwhm_yield: {hwhm_yield}")
    return hwhm_yield

def getSignificanceHist(sample_dict, nSubCats=5):
    # divide the dimuon mass arrays by subcategory values
    sigDict_by_subCat = getDimuMassBySubCat(sample_dict, sample="signal", nSubCats=nSubCats)
    bkgDict_by_subCat = getDimuMassBySubCat(sample_dict, sample="background", nSubCats=nSubCats) 
    # print(f"sigDict_by_subCat: {sigDict_by_subCat}")
    # print(f"bkgDict_by_subCat: {bkgDict_by_subCat}")
    
    # calculate HWHM and the edges
    signficanceBySubCat = {}
    for subCat_target in range(nSubCats):
        binning = np.linspace(110, 150, 51) # signal fit region
        # print(f"sigDict_by_subCat[subCat_target]: {sigDict_by_subCat[subCat_target]}")
        hwhm, hwhm_left, hwhm_right = getHWHM_withEdges(sigDict_by_subCat[subCat_target], binning)
        # print(f"hwhm: {hwhm}")
        # print(f"hwhm_left: {hwhm_left}")
        # print(f"hwhm_right: {hwhm_right}")
        
        sigYield_hwhm = getYield_hwhm(sigDict_by_subCat[subCat_target], hwhm_left, hwhm_right)
        bkgYield_hwhm = getYield_hwhm(bkgDict_by_subCat[subCat_target], hwhm_left, hwhm_right)
        # print(f"sigYield_hwhm: {sigYield_hwhm}")
        # print(f"bkgYield_hwhm: {bkgYield_hwhm}")
        # sigYield_hwhm = sigYield_hwhm *3
        # bkgYield_hwhm = bkgYield_hwhm *3
        significance = sigYield_hwhm/ (bkgYield_hwhm**(0.5))
        signficanceBySubCat[subCat_target] = significance
        # print(f"subcat {subCat_target} significance: {significance}")
    print(f"signficanceBySubCat: {signficanceBySubCat}")
    # convert the dictionary with significance values to np his arrays
    significanceHist = np.zeros(nSubCats)
    for subcat, significance in signficanceBySubCat.items():
        significanceHist[subcat] = significance
        
    print(f"significanceHist: {significanceHist}")
    return significanceHist


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
        "signal" : "sigMC*",
        "background" : "bkgMC*",
    }
    sample_dict = {
        group: {
            "wgt_nominal" : [],
            "BDT_score": [],
            "dimuon_mass": [],
            "subCategory_idx": [],
        } for group in sample_groups.keys()
    }
    if args.region != "signal":
        print("Error, region is not signal!")
        raise ValueError
    for group, group_fname in sample_groups.items():
        full_load_path = load_path+f"*{group_fname}.parquet" 
        events = dak.from_parquet(full_load_path)
        _, events = filterRegion(events, region=args.region)
        sample_dict = fillSampleValues(events, sample_dict, group)
        print(f"sample_dict: {sample_dict}")

    #transform BDT score
    for group, field_dict in sample_dict.items():
        field_dict = tranformBDT_score(field_dict)
        sample_dict[group] = field_dict

    # plot_setting_fname = "../../../src/lib/histogram/plot_settings_vbfCat_MVA_input.json"
    plot_setting_fname = "src/lib/histogram/plot_settings_gghCat_BDT_input.json"
    with open(plot_setting_fname, "r") as file:
        plot_settings = json.load(file)
    plot_var = "BDT_score"
    binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
    bkg_MC = sample_dict["background"]
    sig_MC = sample_dict["signal"]
    # save_fname = "plots/Fig6_13.pdf"
    save_fname = f"plots/{args.label}_x_{args.category}/{args.year}_signal/Fig6_13.pdf"
    # print(f"sample_dict: {sample_dict}")
    print(f"binning: {binning}")
    print(f"bkg_MC: {bkg_MC}")
    print(f"sig_MC: {sig_MC}")
    # status = "Private"
    status = "Simulation"
    if year =="all": # temporarily overwrite load paths and year to get 2018 bdt edges
        # year="2018" # Run2
        year="2024"
    elif year == "2016":
        year="2016preVFP"
    # else:
        # ye/ar="2024"
    load_path =f"{args.base_path}/{args.label}/{args.category}/stage2_output/{year}/"
    
    bdt_edges = OmegaConf.load(f"{load_path}/BDT_edges.yaml")[year]
    print(f"bdt_edges b4 transform: {bdt_edges}")
    
    bdt_edges = tranformBDT_edges(bdt_edges)
    print(f"bdt_edges after transform: {bdt_edges}")
    print(f"binning: {binning}")
    # print(f"bdt_edges: {bdt_edges}")
    bdt_edges4plot = np.append(bdt_edges, [-1.0, 1.0]) # add the extreme edge values, -0.9 and 0.9
    bdt_edges4plot.sort() 
    print(f"bdt_edges4plot: {bdt_edges4plot}")
    subCatSignificance_hist = getSignificanceHist(sample_dict)
    # Make directory if it doesn't exist
    os.makedirs(os.path.dirname(save_fname), exist_ok=True)
    plotFig_6_13(
        binning, bkg_MC, sig_MC, save_fname,
        title = "", 
        x_title = plot_settings[plot_var].get("xlabel"), 
        lumi = lumi_val,
        status = status,
        bdtCat_boundaries=bdt_edges,
        significance_tuple = (subCatSignificance_hist, bdt_edges4plot),
        # ymax=0.1
    )

    