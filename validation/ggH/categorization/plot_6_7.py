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
import dask.dataframe as dd
import matplotlib.cm as cm
from modules.utils import pair_and_remove, getSqrtSOverB
from modules.RooWorkspaceUtils import hist_stddev_with_unc
from modules.selection import filterRegion
import pandas as pd

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
# Add it to sys.path
sys.path.insert(0, parent_dir)
# Now you can import your module

from src.lib.histogram.plotting import plotScatter, plot2D

def plotWgtAnnhilation(parq_path, save_dir):
    """
    wrapper function
    """
    df = dd.read_parquet(parq_path).compute()
    _, df = filterRegion(df, region="h-peak")

    # print(f"{parq_path} \n df: {df}")
    # print(f"{parq_path} \n df: {len(df)}")
    colsOfInterest = ["mu1_eta", "mu2_eta","dimuon_pt"]
    matches, remaining = pair_and_remove(df, cols=colsOfInterest)
    # print(f"matches: {matches}")
    # print(f"remaining: {remaining}")
    print(f"df: {len(df)}")
    print(f"matches: {len(matches)}")
    print(f"remaining: {len(remaining)}")
    # print(f"remaining all positive: {np.all(remaining['wgt_nominal'] >= 0)}")

    for var in variables:
        plot_var = getPlotVar(var)
        if plot_var == "dimuon_mass":
            binning = np.linspace(70, 110, 50)
        elif plot_var == "jj_mass":
            binning = np.linspace(0, 2500, 100)
        else:
            binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
        xlabel =  plot_settings[plot_var].get("xlabel")
        df_dict = {
            "DY" : df,
            "DY pair match removed" : remaining,
        }
        compareMC(df_dict, binning, var, xlabel, save_dir, applyWgt=True, do_logscale=False)




def getPlotVar(var: str):
    """
    Helper function that removes the variations in variable name if they exist
    """
    if "_nominal" in var:
        plot_var = var.replace("_nominal", "")
    else:
        plot_var = var
    return plot_var

def get_unique_colors(n, cmap_name="tab10"):

    """
    Return a list of n unique colors (as RGBA tuples) compatible with pyplot.
    
    Parameters
    ----------
    n : int
        Number of colors to generate.
    cmap_name : str, optional
        Name of a matplotlib colormap (default "tab10").
    
    Returns
    -------
    list of RGBA tuples
    """
    cmap = cm.get_cmap(cmap_name, n)  # sample 'n' distinct colors
    return [cmap(i) for i in range(n)]
    
def weighted_quantile(values, quantile, sample_weight=None):
    """
    Compute weighted quantile of given data.
    
    values : np.ndarray
        Data (e.g. BDT scores).
    quantile : float
        Quantile in [0, 1] (e.g. 0.3 for 30%).
    sample_weight : np.ndarray or None
        Weights (same length as values).
    """
    values = np.array(values)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    else:
        sample_weight = np.array(sample_weight)
    
    # sort by values
    sorter = np.argsort(values)
    values = values[sorter]
    weights = sample_weight[sorter]

    # compute cumulative normalized weights
    cumsum = np.cumsum(weights)
    cutoff = quantile * np.sum(weights)

    return values[np.searchsorted(cumsum, cutoff)]

def plot_6_7FineGrain(df, binning, var, xlabel, save_dir):
    
    # --- binning ---
    # bdt_edges = np.array([-1.00, -0.28, -0.10, 0.08, 0.23, 0.32, 0.43, 0.51, 1.00])
    n_cats = 20
    # n_cats = 12
    bdt_edges = np.linspace(-1,1,n_cats+1)
    # bdt_edges = np.array([-1.   , -0.625, -0.5  , -0.375, -0.25 , -0.125,        0.   ,  0.125,  0.25 ,  0.375,  0.5  ,  0.625,  0.75 ,  0.875,        1.   ])
    # n_cats = len(bdt_edges)-1
    
    # --- plotting ---
    # plt.figure(figsize=(7,5))
    fig, ax_main = plt.subplots()
    
    colors = get_unique_colors(n_cats, cmap_name="tab20")
    score_name =  "BDT_score"
    for (lo, hi), color in zip(zip(bdt_edges[:-1], bdt_edges[1:]), colors):
        mask = (df[score_name] > lo) & (df[score_name] <= hi)
        if not mask.any():
            continue
        wgt_var= "wgt_nominal"
        hist, bins = np.histogram(df.loc[mask, var], bins=binning, weights=df.loc[mask, wgt_var])
        # hist, bins = np.histogram(df.loc[mask, var], bins=binning, density=True)
        hist = hist / np.sum(hist)
        hep.histplot(
            hist,
            bins,
            label=f"{lo:.2f} < BDT < {hi:.2f}",
            histtype="step",
            color=color,
            ax=ax_main,
        )

    plt.xlabel(xlabel)
    plt.ylabel("A.U.")
    plt.title("")
    # plt.title("Normalized dimuon mass by BDT slice")
    plt.legend(fontsize=12, loc="best", ncol=1)
    # plt.legend(ncol=2)
    # plt.tight_layout()
    # plt.show()
    CenterOfMass = 13
    # status = "Simulation"
    # hep.cms.label(data=False, loc=0, label=status, com=CenterOfMass, ax=ax_main)
    hep.cms.label(data=False, loc=0, com=CenterOfMass, ax=ax_main)
    fig_name = f"{save_dir}/{plot_var}FineGrain.pdf"
    # fig_name = f"{plot_var}.pdf"
    plt.savefig(fig_name)

def plot_6_7BDTCatMerged(df, binning, var, xlabel, save_dir):
    
    # --- binning ---
    bdt_edges = np.array([-1.00, -0.28, -0.10, 0.08, 0.23, 0.32, 0.43, 0.51, 1.00])
    
    # --- plotting ---
    # plt.figure(figsize=(7,5))
    fig, ax_main = plt.subplots()
    
    colors = ["black","red","blue","orange","green","cyan","magenta","gray"]
    score_name =  "BDT_score"
    hist_l = []
    for (lo, hi), color in zip(zip(bdt_edges[:-1], bdt_edges[1:]), colors):
        mask = (df[score_name] > lo) & (df[score_name] <= hi)
        if not mask.any():
            continue
        wgt_var= "wgt_nominal"
        hist, bins = np.histogram(df.loc[mask, var], bins=binning, weights=df.loc[mask, wgt_var])
        hist_l.append(hist)
    # print(f"hist_l: {hist_l}")
    hist = sum(hist_l)
    # print(f"hist: {hist}")
    hist = hist/np.sum(hist)
    hep.histplot(
        hist,
        bins,
        label=f"combined BDT category",
        histtype="step",
        color=color,
        ax=ax_main,
    )

    plt.xlabel(xlabel)
    plt.ylabel("A.U.")
    plt.title("")
    # plt.title("Normalized dimuon mass by BDT slice")
    plt.legend(fontsize=12, loc="best", ncol=1)
    # plt.legend(ncol=2)
    # plt.tight_layout()
    # plt.show()
    CenterOfMass = 13
    # status = "Simulation"
    # hep.cms.label(data=False, loc=0, label=status, com=CenterOfMass, ax=ax_main)
    hep.cms.label(data=False, loc=0, com=CenterOfMass, ax=ax_main)
    fig_name = f"{save_dir}/{plot_var}_BDTCatMerged.pdf"
    # fig_name = f"{plot_var}.pdf"
    plt.savefig(fig_name)


def plot_6_7(df, binning, var, xlabel, save_dir):
    
    # --- binning ---
    bdt_edges = np.array([-1.00, -0.28, -0.10, 0.08, 0.23, 0.32, 0.43, 0.51, 1.00])
    
    # --- plotting ---
    # plt.figure(figsize=(7,5))
    fig, ax_main = plt.subplots()
    
    colors = ["black","red","blue","orange","green","cyan","magenta","gray"]
    score_name =  "BDT_score"
    for (lo, hi), color in zip(zip(bdt_edges[:-1], bdt_edges[1:]), colors):
        mask = (df[score_name] > lo) & (df[score_name] <= hi)
        if not mask.any():
            continue
        wgt_var= "wgt_nominal"
        hist, bins = np.histogram(df.loc[mask, var], bins=binning, weights=df.loc[mask, wgt_var])
        # hist, bins = np.histogram(df.loc[mask, var], bins=binning, density=True)
        hist = hist / np.sum(hist)
        hep.histplot(
            hist,
            bins,
            label=f"{lo:.2f} < BDT < {hi:.2f}",
            histtype="step",
            color=color,
            ax=ax_main,
        )

    plt.xlabel(xlabel)
    plt.ylabel("A.U.")
    plt.title("")
    # plt.title("Normalized dimuon mass by BDT slice")
    plt.legend(fontsize=12, loc="best", ncol=1)
    # plt.legend(ncol=2)
    # plt.tight_layout()
    # plt.show()
    CenterOfMass = 13
    # status = "Simulation"
    # hep.cms.label(data=False, loc=0, label=status, com=CenterOfMass, ax=ax_main)
    hep.cms.label(data=False, loc=0, com=CenterOfMass, ax=ax_main)
    fig_name = f"{save_dir}/{plot_var}.pdf"
    # fig_name = f"{plot_var}.pdf"
    plt.savefig(fig_name)


def plot_6_7BySubCat(df, binning, var, xlabel, save_dir, cat_idx=None):
    
    # --- binning --- # FIXME: copy the edges from 2024 BDT edges yaml file
    bdt_edges = np.array([ # 2018 UL subcat edges
        0.0,
        0.39617791771888733,
        0.44750669598579407,
        0.5049778819084167,
        0.5425251722335815,
        1.1,
    ])
    bdt_edges = bdt_edges*2 -1

    # --- plotting ---
    # plt.figure(figsize=(7,5))
    fig, ax_main = plt.subplots()
    
    colors = ["black","red","blue","orange","green","cyan","magenta","gray"]
    score_name =  "BDT_score"
    bdt_loop = zip(zip(bdt_edges[:-1], bdt_edges[1:]), colors)
    for idx, ((lo, hi), color) in enumerate(bdt_loop):
        if cat_idx is not None:
            if idx != cat_idx: # skip plotting cat_idx not specified
                continue
        mask = (df[score_name] > lo) & (df[score_name] <= hi)
        if not mask.any():
            continue
        wgt_var= "wgt_nominal"
        hist, bins = np.histogram(df.loc[mask, var], bins=binning, weights=df.loc[mask, wgt_var])
        hist_w2, _ = np.histogram(df.loc[mask, var], bins=binning, weights=df.loc[mask, wgt_var]*df.loc[mask, wgt_var])
        if plot_var == "dimuon_mass":
            std, std_err = hist_stddev_with_unc(hist, bins)
            label = f"{lo:.2f} < BDT < {hi:.2f}, \n stdDev = {std:.2f} ± {std_err:.2f}"
        else:
            label=f"{lo:.2f} < BDT < {hi:.2f}"

        if cat_idx is not None:
            hep.histplot(
                hist,
                bins,
                label=label,
                histtype="step",
                color=color,
                ax=ax_main,
                yerr=np.sqrt(hist_w2),
            )
        else:
            hist = hist / np.sum(hist)
            hep.histplot(
                hist,
                bins,
                label=label,
                histtype="step",
                color=color,
                ax=ax_main,
            )

    plt.xlabel(xlabel)
    plt.ylabel("A.U.")
    plt.title("")
    # plt.title("Normalized dimuon mass by BDT slice")
    plt.legend(fontsize=12, loc="best", ncol=1)
    # plt.legend(ncol=2)
    # plt.tight_layout()
    # plt.show()
    CenterOfMass = 13
    # status = "Simulation"
    # hep.cms.label(data=False, loc=0, label=status, com=CenterOfMass, ax=ax_main)
    hep.cms.label(data=False, loc=0, com=CenterOfMass, ax=ax_main)

    if cat_idx is not None:
        save_fname_addendum = f"_cat{cat_idx}"
    else:
        save_fname_addendum = ""
    fig_name = f"{save_dir}/{plot_var}BySubCat{save_fname_addendum}.pdf"
    plt.savefig(fig_name)
    fig_name = f"{save_dir}/{plot_var}BySubCat{save_fname_addendum}.png"
    plt.savefig(fig_name)

# def compareMCByEbeMass(df_dict, binning, var, xlabel, save_dir, unweighted=False, abs_wgt=False, removeNegWgt=False, applyWgt=False, do_logscale=True):
    
#     # --- plotting ---
#     # plt.figure(figsize=(7,5))
    
    
#     wgt_var= "wgt_nominal"
#     mass_res_edges = [0, 1.25, 8]
#     for njet_target in [0, 1, 2, 3, 4]:
#         plt.clf()
#         fig, ax_main = plt.subplots()
    
#         for label, df in df_dict.items():
#             if abs_wgt:
#                 wgt = abs(df[wgt_var])
#                 var_val = df[var]
#             elif removeNegWgt:
#                 is_pos = df[wgt_var] >=0 
#                 wgt = df[wgt_var][is_pos]
#                 var_val = df[var][is_pos]
#             else:
#                 wgt = df[wgt_var]
#                 var_val = df[var]
#             njet_filter = df["njets_nominal"] == njet_target
#             # print(f"njet_filter: {njet_filter}")
#             print(f"njet_filter: {np.sum(njet_filter)}")
#             var_val_njet = var_val[njet_filter]
#             wgt_njet = wgt[njet_filter]
#             hist, bins = np.histogram(var_val_njet, bins=binning, weights=wgt_njet)
#             if not applyWgt: # every other options lead to normalizing the histogram
#                 hist = hist / np.sum(hist)
#             hep.histplot(
#                 hist,
#                 bins,
#                 label=label+f" njet {njet_target}",
#                 histtype="step",
#                 ax=ax_main,
#             )
#             plt.xlabel(xlabel)
#             if applyWgt:
#                 plt.ylabel("yield")
#                 if do_logscale:
#                     plt.yscale("log")
#             else:
#                 plt.ylabel("A.U.")
#             plt.title("")
#             plt.legend(fontsize=12, loc="best", ncol=1)
#             # plt.legend(ncol=2)
#             # plt.tight_layout()
#             # plt.show()
#             CenterOfMass = 13
#             # status = "Simulation"
#             # hep.cms.label(data=False, loc=0, label=status, com=CenterOfMass, ax=ax_main)
#             hep.cms.label(data=False, loc=0, com=CenterOfMass, ax=ax_main)
#             plot_var = getPlotVar(var)
#             if unweighted:
#                 fig_name = f"{save_dir}/{plot_var}_sigMC_compUnWgtedNjet{njet_target}.pdf"
#             elif abs_wgt:
#                 fig_name = f"{save_dir}/{plot_var}_sigMC_compAbsWgtNjet{njet_target}.pdf"
#             elif removeNegWgt:
#                 fig_name = f"{save_dir}/{plot_var}_sigMC_compPosWgtNjet{njet_target}.pdf"
#             elif applyWgt:
#                 fig_name = f"{save_dir}/{plot_var}_sigMC_compXsecNormalizedNjet{njet_target}.pdf"
#             else: 
#                 fig_name = f"{save_dir}/{plot_var}_sigMC_compNjet{njet_target}.pdf"
#             plt.savefig(fig_name)


def compareMCByNjet(df_dict, binning, var, xlabel, save_dir, unweighted=False, abs_wgt=False, removeNegWgt=False, applyWgt=False, do_logscale=True):
    
    # --- plotting ---
    # plt.figure(figsize=(7,5))
    
    wgt_var= "wgt_nominal"
    for njet_target in [0, 1, 2, 3, 4]:
        hist_dict = {}
        plt.clf()
        fig, ax_main = plt.subplots()
    
        for label, df in df_dict.items():
            if abs_wgt:
                wgt = abs(df[wgt_var])
                var_val = df[var]
            elif removeNegWgt:
                is_pos = df[wgt_var] >=0 
                wgt = df[wgt_var][is_pos]
                var_val = df[var][is_pos]
            else:
                wgt = df[wgt_var]
                var_val = df[var]
            njet_filter = df["njets_nominal"] >= njet_target
            current_label = label+f", njet >={njet_target}"
            
            # if njet_target < 3:
            #     njet_filter = df["njets_nominal"] == njet_target
            #     current_label = label+f", njet ={njet_target}"
            # else:
            #     njet_filter = df["njets_nominal"] >= njet_target
            #     current_label = label+f", njet >={njet_target}"
            # print(f"njet_filter: {njet_filter}")
            print(f"njet_filter: {np.sum(njet_filter)}")
            var_val_njet = var_val[njet_filter]
            wgt_njet = wgt[njet_filter]
            # Merge bins: keep every 2nd edge
            current_bins = binning[::2]
            
            hist, bins = np.histogram(var_val_njet, bins=current_bins, weights=wgt_njet)
            # --------------------
            print(f"hist: {len(hist)}")
            print(f"bins: {len(bins)}")
            # hist_dict[label] = (hist, bins)
            # --------------------
            if not applyWgt: # every other options lead to normalizing the histogram
                hist = hist / np.sum(hist)
            hist_dict[label] = (hist, bins)
            
            hep.histplot(
                hist,
                bins,
                label=current_label,
                histtype="step",
                ax=ax_main,
            )
        
        plt.xlabel(xlabel)
        if applyWgt:
            plt.ylabel("yield")
            if do_logscale:
                plt.yscale("log")
        else:
            plt.ylabel("A.U.")
        plt.title("")
        plt.legend(fontsize=12, loc="best", ncol=1)
        # plt.legend(ncol=2)
        # plt.tight_layout()
        # plt.show()
        CenterOfMass = 13
        # status = "Simulation"
        # hep.cms.label(data=False, loc=0, label=status, com=CenterOfMass, ax=ax_main)
        hep.cms.label(data=False, loc=0, com=CenterOfMass, ax=ax_main)
        plot_var = getPlotVar(var)
        # save_dir_njet = f"{save_dir}/Njet{njet_target}"
        if unweighted:
            fig_name = f"{save_dir}/{plot_var}_sigMC_compUnWgtedNjet{njet_target}.pdf"
        elif abs_wgt:
            fig_name = f"{save_dir}/{plot_var}_sigMC_compAbsWgtNjet{njet_target}.pdf"
        elif removeNegWgt:
            fig_name = f"{save_dir}/{plot_var}_sigMC_compPosWgtNjet{njet_target}.pdf"
        elif applyWgt:
            fig_name = f"{save_dir}/{plot_var}_sigMC_compXsecNormalizedNjet{njet_target}.pdf"
        else: 
            fig_name = f"{save_dir}/{plot_var}_sigMC_compNjet{njet_target}.pdf"
        plt.savefig(fig_name)

        print(hist_dict)
        sig_counts = None
        bkg_counts = None
        bin_edges = None
        for key, value in hist_dict.items():
            if "ggH+VBF" in key:
                sig_counts, bin_edges = value
            elif "Bkg" in key:
                bkg_counts, bin_edges = value

        if (sig_counts is None) or (bkg_counts is None) or (bin_edges is None):
            continue
        save_path = f"{save_dir}/significanceScan"
        os.makedirs(save_path, exist_ok=True)
        fname = f"significanceNjet{njet_target}"
        print(f"sig_counts: {len(sig_counts)}")
        print(f"bkg_counts: {len(bkg_counts)}")
        getSqrtSOverB(bin_edges, sig_counts, bkg_counts, save_path, fname)
        


def compareMC(df_dict, binning, var, xlabel, save_dir, unweighted=False, abs_wgt=False, removeNegWgt=False, applyWgt=False, do_logscale=True):
    
    # --- plotting ---
    # plt.figure(figsize=(7,5))
    plt.clf()
    fig, ax_main = plt.subplots()
    
    wgt_var= "wgt_nominal"

    for label, df in df_dict.items():
        if abs_wgt:
            wgt = abs(df[wgt_var])
            var_val = df[var]
        elif removeNegWgt:
            is_pos = df[wgt_var] >=0 
            wgt = df[wgt_var][is_pos]
            var_val = df[var][is_pos]
        else:
            wgt = df[wgt_var]
            var_val = df[var]
        hist, bins = np.histogram(var_val, bins=binning, weights=wgt)
        if not applyWgt: # every other options lead to normalizing the histogram
            hist = hist / np.sum(hist)
        hep.histplot(
            hist,
            bins,
            label=label,
            histtype="step",
            ax=ax_main,
        )
    plt.xlabel(xlabel)
    if applyWgt:
        plt.ylabel("yield")
        if do_logscale:
            plt.yscale("log")
    else:
        plt.ylabel("A.U.")
    plt.title("")
    plt.legend(fontsize=12, loc="best", ncol=1)
    # plt.legend(ncol=2)
    # plt.tight_layout()
    # plt.show()
    CenterOfMass = 13
    # status = "Simulation"
    # hep.cms.label(data=False, loc=0, label=status, com=CenterOfMass, ax=ax_main)
    hep.cms.label(data=False, loc=0, com=CenterOfMass, ax=ax_main)
    plot_var = getPlotVar(var)
    if unweighted:
        fig_name = f"{save_dir}/{plot_var}_sigMC_compUnWgted.pdf"
    elif abs_wgt:
        fig_name = f"{save_dir}/{plot_var}_sigMC_compAbsWgt.pdf"
    elif removeNegWgt:
        fig_name = f"{save_dir}/{plot_var}_sigMC_compPosWgt.pdf"
    elif applyWgt:
        fig_name = f"{save_dir}/{plot_var}_sigMC_compXsecNormalized.pdf"
    else: 
        fig_name = f"{save_dir}/{plot_var}_sigMC_comp.pdf"
    plt.savefig(fig_name)

def getDfAndPreProcess(full_load_path):
    df = dd.read_parquet(full_load_path).compute()
    print(df.columns)
    print(full_load_path)
    _, df = filterRegion(df, region="h-peak")
    # change BDT score range from [0,1] to [-1,1]
    # print(df)
    print(df.columns)
    df["BDT_score"] = (df["BDT_score"] *2 ) -1
    # print(df.columns)
    # print(df.isna().any().any()) 
    # print(df.isna().sum().sum())
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
    "--bdt_year",
    dest="bdt_year",
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
    parser.add_argument(
    "-model",
    "--model_name",
    dest="model_name",
    default="",
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
    #     _, events = filterRegion(events, region=args.region)
    #     sample_dict = fillSampleValues(events, sample_dict, group)

    
    full_load_path = load_path+f"processed_events_sigMC*.parquet" 
    # full_load_path = load_path+f"processed_events_sigMC_ggh*.parquet" 
    df = getDfAndPreProcess(full_load_path)


    # plot_setting_fname = "../../../src/lib/histogram/plot_settings_vbfCat_MVA_input.json"
    # plot_setting_fname = "../../../src/lib/histogram/plot_settings_gghCat_BDT_input.json"
    plot_setting_fname = "../../../src/lib/histogram/plot_settings_gghCat_BDT_inputFig6_7.json"
    # plot_setting_fname = "plot_settings_vbfCat_MVA_input.json"
    with open(plot_setting_fname, "r") as file:
        plot_settings = json.load(file)

    save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/Fig6_7"
    # save_dir = f"plots/{args.category}/{args.year}signal/Fig6_7"
    os.makedirs(save_dir, exist_ok=True)
    # extract BDT inputs
    
    model_path = f"/work/users/yun79/Run2_MVA_trainer/output/bdt_{args.model_name}_{args.bdt_year}"
    training_feat_path = f"{model_path}/training_features.json"
    print(f"trainig_feat_path: {training_feat_path}")
    with open(training_feat_path, 'r') as file:
        bdt_inputs = json.load(file)
    # bdt_inputs = [
    #     'dimuon_cos_theta_cs', 
    #     'dimuon_phi_cs', 
    #     'dimuon_rapidity', 
    #     'dimuon_pt', 
    #     'jet1_eta_nominal', 
    #     # 'jet2_eta_nominal', 
    #     'jet1_pt_nominal', 
    #     'jet2_pt_nominal', 
    #     'jj_dEta_nominal', 
    #     'jj_dPhi_nominal', 
    #     'jj_mass_nominal', 
    #     # 'mmj1_dEta', 
    #     # 'mmj1_dPhi',  
    #     'mmj_min_dEta_nominal', 
    #     'mmj_min_dPhi_nominal', 
    #     'mu1_eta', 
    #     'mu1_pt_over_mass', 
    #     'mu2_eta', 
    #     'mu2_pt_over_mass', 
    #     'zeppenfeld_nominal',
    #     'njets_nominal',
    #     # 'mmj1_dEta_nominal', 
    #     # 'mmj1_dPhi_nominal',  
    #     'BDT_score',  
    # ]
    variables = bdt_inputs + ["dimuon_mass", "dimuon_ebe_mass_res"]
    # variables =  ["dimuon_mass", "dimuon_ebe_mass_res"
    print(f"varibles: {variables}")
    print(f"df.columns: {df.columns}")
    threshold_targets = [0.3, .65, .8, .95]
    variables.remove("year") # no need to print year for now
    for var in variables:
        plot_var = getPlotVar(var)
        if plot_var == "dimuon_mass":
            binning = np.linspace(115, 135, 50)
        else:
            binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
        xlabel =  plot_settings[plot_var].get("xlabel")
        thresholds = []
        for threshold_target in threshold_targets:
            threshold = weighted_quantile(df["BDT_score"], threshold_target, sample_weight=df["wgt_nominal"])
            thresholds.append(threshold)
        thresholds = np.array(thresholds)
        print("BDT score threshold (30% cumulative weight):", thresholds)
        plot_6_7(df, binning, var, xlabel, save_dir)
        plot_6_7BySubCat(df, binning, var, xlabel, save_dir)
        plot_6_7FineGrain(df, binning, var, xlabel, save_dir)
        plot_6_7BDTCatMerged(df, binning, var, xlabel, save_dir)


    
    
    # # ----------------------------------------------------
    # #  compare jet eta distribution when | y_mumu | > 1.0
    # # ----------------------------------------------------
    # save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/Fig6_7_yMuMuCut"
    # os.makedirs(save_dir, exist_ok=True)
    
    # yMuMuCut = abs(df["dimuon_rapidity"]) > 1.0
    # df_yMuMuCut = df[yMuMuCut]
    # for var in ["jet1_eta_nominal", "jet2_eta_nominal", "dimuon_rapidity"]:
    #     plot_var = getPlotVar(var)
    #     if plot_var == "dimuon_mass":
    #         binning = np.linspace(115, 135, 50)
    #     else:
    #         binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
    #     xlabel =  plot_settings[plot_var].get("xlabel")
    #     thresholds = []
    #     for threshold_target in threshold_targets:
    #         threshold = weighted_quantile(df["BDT_score"], threshold_target, sample_weight=df["wgt_nominal"])
    #         thresholds.append(threshold)
    #     thresholds = np.array(thresholds)
    #     print("BDT score threshold (30% cumulative weight):", thresholds)
    #     plot_6_7BySubCat(df_yMuMuCut, binning, var, xlabel, save_dir)
    
    # # save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/Scatter"
    # # os.makedirs(save_dir, exist_ok=True)

    # x_var = "dimuon_pt"
    # plotScatter(df, variables, x_var, save_dir)
    # # ----------------
    # save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/Hist2D"
    # os.makedirs(save_dir, exist_ok=True)
    # x_var = "dimuon_pt"
    # plot2D(df, variables, x_var, plot_settings, save_dir)
    # plot2D(df, variables, x_var, plot_settings, save_dir, inclusive=True)

    # # -----------------------
    # x_var = "dimuon_rapidity"
    # save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/Hist2D/{x_var}"
    # os.makedirs(save_dir, exist_ok=True)
    # plot2D(df, variables, x_var, plot_settings, save_dir)


    # ----------------------------------------------------
    #  Plot dy dimuon mass background
    # ----------------------------------------------------
    save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/Fig6_7_dy"
    os.makedirs(save_dir, exist_ok=True)
    full_load_path = load_path+f"processed_events_bkgMC_dy.parquet" 
    df = getDfAndPreProcess(full_load_path)
    
    for var in ["dimuon_mass"]:
        plot_var = getPlotVar(var)
        if plot_var == "dimuon_mass":
            binning = np.linspace(115, 135, 50)
        else:
            binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
        xlabel =  plot_settings[plot_var].get("xlabel")
        thresholds = []
        for threshold_target in threshold_targets:
            threshold = weighted_quantile(df["BDT_score"], threshold_target, sample_weight=df["wgt_nominal"])
            thresholds.append(threshold)
        thresholds = np.array(thresholds)
        print("BDT score threshold (30% cumulative weight):", thresholds)
        # plot_6_7(df, binning, var, xlabel, save_dir)
        plot_6_7BySubCat(df, binning, var, xlabel, save_dir)
        for idx in range(5):
            plot_6_7BySubCat(df, binning, var, xlabel, save_dir, cat_idx=idx)
            
        # plot_6_7FineGrain(df, binning, var, xlabel, save_dir)
        # plot_6_7BDTCatMerged(df, binning, var, xlabel, save_dir)
    raise ValueError


    # # # ----------------------------------------------------
    # # #  Add sigMC comparison
    # # # ----------------------------------------------------
    # # save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/sigMC_comp"
    # # os.makedirs(save_dir, exist_ok=True)
    
    # # full_load_path = load_path+f"processed_events_sigMC_ggh.parquet" 
    # # ggh_df = dd.read_parquet(full_load_path).compute()
    # # full_load_path = load_path+f"processed_events_sigMC_vbf.parquet" 
    # # vbf_df = dd.read_parquet(full_load_path).compute()
    
    # # # for var in ["dimuon_pt", "dimuon_mass"]:
    # # for var in variables:
    # #     plot_var = getPlotVar(var)
    # #     if plot_var == "dimuon_mass":
    # #         binning = np.linspace(115, 135, 50)
    # #     else:
    # #         binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
    # #     xlabel =  plot_settings[plot_var].get("xlabel")
    # #     df_dict = {
    # #         "ggH" : ggh_df,
    # #         "VBF" : vbf_df
    # #     }
    # #     compareMC(df_dict, binning, var, xlabel, save_dir)


    # # ----------------------------------------------------
    # #  Add sigMC vs bkg comparison
    # # ----------------------------------------------------
    # save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/sigBkgMC_comp"
    # os.makedirs(save_dir, exist_ok=True)
    
    # full_load_path = load_path+f"processed_events_sigMC*.parquet" 
    # sig_df = dd.read_parquet(full_load_path).compute()
    # _, sig_df = filterRegion(sig_df, region="h-peak")
    # # ------------------------------------------
    # full_load_path = load_path+f"processed_events_bkgMC*.parquet" 
    # print(full_load_path)
    # pos_filter = sig_df["dimuon_rapidity"] > 0
    # pos_wgt_sum =  sig_df.loc[pos_filter, "wgt_nominal"].sum()
    # neg_wgt_sum =  sig_df.loc[~pos_filter, "wgt_nominal"].sum()
    # # print(f"pos_wgt_sum: {pos_wgt_sum}")
    # # print(f"neg_wgt_sum: {neg_wgt_sum}")
    # bkg_df = dd.read_parquet(full_load_path).compute()
    # _, bkg_df = filterRegion(bkg_df, region="h-peak")
    # print(bkg_df.columns)

    
    
    # for var in variables:
    #     plot_var = getPlotVar(var)
    #     if plot_var == "dimuon_mass":
    #         binning = np.linspace(115, 135, 50)
    #     else:
    #         binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
    #     xlabel =  plot_settings[plot_var].get("xlabel")
    #     df_dict = {
    #         "ggH+VBF" : sig_df,
    #         "Bkg" : bkg_df,
    #     }
    #     compareMC(df_dict, binning, var, xlabel, save_dir)


    # # ----------------------------------------------------------
    # #  Add sigMC vs bkg comparison with jj mass cut
    # # ----------------------------------------------------
    # save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/sigBkgMC_comp/JJMassCutGeq400"
    # os.makedirs(save_dir, exist_ok=True)
    
    # test_filter = sig_df["jj_mass_nominal"] > 400
    # sig_df_current = sig_df[test_filter] 
    # test_filter = bkg_df["jj_mass_nominal"] > 400
    # bkg_df_current = bkg_df[test_filter] 
    # for var in ["dimuon_pt"]:
    #     plot_var = getPlotVar(var)
    #     if plot_var == "dimuon_mass":
    #         binning = np.linspace(115, 135, 50)
    #     else:
    #         binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
    #     xlabel =  plot_settings[plot_var].get("xlabel")
    #     df_dict = {
    #         "ggH+VBF, jj Mass > 400" : sig_df_current,
    #         "Bkg, jj Mass > 400" : bkg_df_current,
    #     }
    #     compareMCByNjet(df_dict, binning, var, xlabel, save_dir)


    # # ----------------------------------------------------------
    # #  Add sigMC vs bkg comparison with jj dEta cut
    # # ----------------------------------------------------
    # save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/sigBkgMC_comp/JJdEtaCutGeq2p5"
    # os.makedirs(save_dir, exist_ok=True)
    
    # test_filter = sig_df["jj_dEta_nominal"] > 2.5
    # sig_df_current = sig_df[test_filter] 
    # test_filter = bkg_df["jj_dEta_nominal"] > 2.5
    # bkg_df_current = bkg_df[test_filter] 
    # for var in ["dimuon_pt"]:
    #     plot_var = getPlotVar(var)
    #     if plot_var == "dimuon_mass":
    #         binning = np.linspace(115, 135, 50)
    #     else:
    #         binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
    #     xlabel =  plot_settings[plot_var].get("xlabel")
    #     df_dict = {
    #         "ggH+VBF, jj dEta > 2.5" : sig_df_current,
    #         "Bkg, jj dEta > 2.5" : bkg_df_current,
    #     }
    #     compareMCByNjet(df_dict, binning, var, xlabel, save_dir)
    
    # ----------------------------------------------------
    #  Add ggH vs VBF vs bkg comparison
    # ----------------------------------------------------
    save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/ggHVBF_DYMC_comp"
    os.makedirs(save_dir, exist_ok=True)
    
    full_load_path = load_path+f"processed_events_sigMC*.parquet" 
    sig_df = dd.read_parquet(full_load_path).compute()
    _, sig_df = filterRegion(sig_df, region="h-peak")
    full_load_path = load_path+f"processed_events_sigMC_ggh.parquet" 
    ggh_df = dd.read_parquet(full_load_path).compute()
    _, ggh_df = filterRegion(ggh_df, region="h-peak")
    full_load_path = load_path+f"processed_events_sigMC_vbf.parquet" 
    vbf_df = dd.read_parquet(full_load_path).compute()
    _, vbf_df = filterRegion(vbf_df, region="h-peak")
    full_load_path = load_path+f"processed_events_bkgMC_dy.parquet" 
    dy_df = dd.read_parquet(full_load_path).compute()
    _, dy_df = filterRegion(dy_df, region="h-peak")
    print(dy_df.columns)
    full_load_path = load_path+f"processed_events_bkgMC_tt.parquet" 
    dy_tt = dd.read_parquet(full_load_path).compute()
    _, dy_tt = filterRegion(dy_tt, region="h-peak")
    full_load_path = load_path+f"processed_events_bkgMC_st.parquet" 
    dy_st = dd.read_parquet(full_load_path).compute()
    _, dy_st = filterRegion(dy_st, region="h-peak")
    full_load_path = load_path+f"processed_events_bkgMC_ewk.parquet" 
    dy_ewk = dd.read_parquet(full_load_path).compute()
    _, dy_ewk = filterRegion(dy_ewk, region="h-peak")
    
    for var in variables:
        plot_var = getPlotVar(var)
        if plot_var == "dimuon_mass":
            binning = np.linspace(115, 135, 50)
        else:
            binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
        xlabel =  plot_settings[plot_var].get("xlabel")
        df_dict = {
            "ggH" : ggh_df,
            "VBF" : vbf_df,
            "DY" : dy_df,
            "TT" : dy_tt,
            # "ST" : dy_st,
            # "EWK" : dy_ewk,
        }
        compareMC(df_dict, binning, var, xlabel, save_dir)
        compareMCByNjet(df_dict, binning, var, xlabel, save_dir)
    # #     compareMC(df_dict, binning, var, xlabel, save_dir, unweighted=True)
    # #     compareMC(df_dict, binning, var, xlabel, save_dir, abs_wgt=True)
    # #     compareMC(df_dict, binning, var, xlabel, save_dir, removeNegWgt=True)
    # #     compareMC(df_dict, binning, var, xlabel, save_dir, applyWgt=True)

    raise ValueError
    

    save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/ggHVBF_TTMC_comp"
    os.makedirs(save_dir, exist_ok=True)
    for var in ["dimuon_pt"]:
        plot_var = getPlotVar(var)
        if plot_var == "dimuon_mass":
            binning = np.linspace(115, 135, 50)
        else:
            binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
        xlabel =  plot_settings[plot_var].get("xlabel")
        df_dict = {
            "ggH" : ggh_df,
            "VBF" : vbf_df,
            # "DY" : dy_df,
            "TT" : dy_tt,
            # "ST" : dy_st,
            # "EWK" : dy_ewk,
        }
        compareMC(df_dict, binning, var, xlabel, save_dir)
        compareMCByNjet(df_dict, binning, var, xlabel, save_dir)

    save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/sigMC_TTMC_comp"
    os.makedirs(save_dir, exist_ok=True)
    for var in ["dimuon_pt"]:
        plot_var = getPlotVar(var)
        if plot_var == "dimuon_mass":
            binning = np.linspace(115, 135, 50)
        else:
            binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
        xlabel =  plot_settings[plot_var].get("xlabel")
        df_dict = {
            "ggH+VBF" : sig_df,
            "TT" : dy_tt,
            # "ST" : dy_st,
            # "EWK" : dy_ewk,
        }
        compareMC(df_dict, binning, var, xlabel, save_dir)
        compareMCByNjet(df_dict, binning, var, xlabel, save_dir)
    

    # # # ----------------------------------------------------
    # # #  check DY in z peak region with no ggH cat cuts
    # # # ----------------------------------------------------
    # # # extract directly from stage1
    # # save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/DYMC_comp"
    # # os.makedirs(save_dir, exist_ok=True)

    # # if year == "all":
    # #     year_param = "*"
    # # else:
    # #     year_param = year
    # # stage1_load_path=f"/depot/cms/users/yun79/hmm/copperheadV1clean/{args.label}/stage1_output/{year_param}/f1_0/"
    # # # stage1_load_path=f"/depot/cms/users/yun79/hmm/copperheadV1clean/{args.label}/stage1_output/2018/f1_0/"

    # # full_load_path = stage1_load_path+f"dy*/*/*.parquet" 
    # # print(full_load_path)
    # # fields2load = variables + ["wgt_nominal"]
    # # dy_df = dd.read_parquet(full_load_path)[fields2load].compute()
    # # _, dy_df_zpeak = filterRegion(dy_df, region="z-only")
    # # _, dy_df_hpeak = filterRegion(dy_df, region="h-peak")
    
    
    # # print(dy_df.columns)
    # # for var in variables:
    # #     plot_var = getPlotVar(var)
    # #     if plot_var == "dimuon_mass":
    # #         binning = np.linspace(70, 110, 50)
    # #     else:
    # #         binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
    # #     xlabel =  plot_settings[plot_var].get("xlabel")
    # #     df_dict = {
    # #         "DY ($85 < m_{\mu\mu} < 95$)" : dy_df_zpeak,
    # #         "DY H peak" : dy_df_hpeak,
    # #     }
    # #     compareMC(df_dict, binning, var, xlabel, save_dir)

    # # # ----------------------------------------------------
    # # #  add DY with ggH channel cut
    # # # ----------------------------------------------------
    
    # # save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/DYMCggHCut_comp"
    # # os.makedirs(save_dir, exist_ok=True)
    # # full_load_path = load_path+f"processed_events_bkgMC_dy.parquet" 
    # # dy_df_gghCut = dd.read_parquet(full_load_path).compute()
    # # _, dy_df_gghCut = filterRegion(dy_df_gghCut, region="h-peak")
    
    # # for var in variables:
    # #     plot_var = getPlotVar(var)
    # #     if plot_var == "dimuon_mass":
    # #         binning = np.linspace(70, 110, 50)
    # #     elif plot_var == "jj_mass":
    # #         binning = np.linspace(0, 2500, 100)
    # #     else:
    # #         binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
    # #     xlabel =  plot_settings[plot_var].get("xlabel")
    # #     df_dict = {
    # #         "DY ($85 < m_{\mu\mu} < 95$)" : dy_df_zpeak,
    # #         "DY H peak" : dy_df_hpeak,
    # #         "DY H peak + ggH channel cut" : dy_df_gghCut,
    # #     }
    # #     compareMC(df_dict, binning, var, xlabel, save_dir)
    # #     compareMC(df_dict, binning, var, xlabel, save_dir, applyWgt=True)

    # # ----------------------------------------------------
    # #  compare DY sample with sample wgt annhilation
    # # ----------------------------------------------------
    
    # save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/DYMCNegWgtPair_comp"
    # os.makedirs(save_dir, exist_ok=True)
    # full_load_path = load_path+f"processed_events_bkgMC_dy.parquet" 
    # plotWgtAnnhilation(full_load_path, save_dir)
    
    # # dy_df = dd.read_parquet(full_load_path).compute()
    # # _, dy_df = filterRegion(dy_df, region="h-peak")

    # # print(f"dy_df: {dy_df}")
    # # print(f"dy_df: {len(dy_df)}")
    # # colsOfInterest = ["mu1_eta", "mu2_eta","dimuon_pt"]
    # # matches, remaining = pair_and_remove(dy_df, cols=colsOfInterest)
    # # print(f"matches: {matches}")
    # # print(f"remaining: {remaining}")
    # # print(f"matches: {len(matches)}")
    # # print(f"remaining: {len(remaining)}")

    # # for var in variables:
    # #     plot_var = getPlotVar(var)
    # #     if plot_var == "dimuon_mass":
    # #         binning = np.linspace(70, 110, 50)
    # #     elif plot_var == "jj_mass":
    # #         binning = np.linspace(0, 2500, 100)
    # #     else:
    # #         binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
    # #     xlabel =  plot_settings[plot_var].get("xlabel")
    # #     df_dict = {
    # #         "DY" : dy_df,
    # #         "DY pair match removed" : remaining,
    # #     }
    # #     compareMC(df_dict, binning, var, xlabel, save_dir, applyWgt=True, do_logscale=False)


    # # ----------------------------------------------------
    # #  compare top sample with sample wgt annhilation
    # # ----------------------------------------------------
    
    # save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/TTMCNegWgtPair_comp"
    # os.makedirs(save_dir, exist_ok=True)
    # full_load_path = load_path+f"processed_events_bkgMC_tt.parquet" 
    # plotWgtAnnhilation(full_load_path, save_dir)
    # save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/STMCNegWgtPair_comp"
    # os.makedirs(save_dir, exist_ok=True)
    # full_load_path = load_path+f"processed_events_bkgMC_st.parquet" 
    # plotWgtAnnhilation(full_load_path, save_dir)
    
    
    