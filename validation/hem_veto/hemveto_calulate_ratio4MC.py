import awkward as ak
import dask_awkward as dak
import dask
import numpy as np
import json
import argparse
import sys
import os
from distributed import Client
import time    
import tqdm
import cmsstyle as CMS
from collections import OrderedDict
import glob
import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib
import copy
from distributed import LocalCluster, Client, progress



# Add the parent directory to the system path
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")) # in order to import plotDataMC_compare
sys.path.append(main_dir)
from src.lib.histogram.plotting import plotDataMC_compare

# def plotHemVetoValidation(compute_dict, save_fname, hem_veto_on=False):
#     # Define your bin ranges
#     eta_range = (-2.4, 2.4)
#     phi_range = (-np.pi, np.pi)

#     # Example data
#     eta_jet = ak.concatenate([compute_dict["jet1_eta_nominal"], compute_dict["jet2_eta_nominal"]])
#     phi_jet = ak.concatenate([compute_dict["jet1_phi_nominal"], compute_dict["jet2_phi_nominal"]])

#     if hem_veto_on:
#         HemVeto_filter = ak.concatenate([compute_dict["HemVeto_filter"], compute_dict["HemVeto_filter"]])
#         eta_jet = eta_jet[HemVeto_filter]
#         phi_jet = phi_jet[HemVeto_filter]
    
#     eta_jet = ak.to_numpy(eta_jet)
#     phi_jet = ak.to_numpy(phi_jet)
    
#     plt.figure(figsize=(8, 6))
#     hist, xedges, yedges, img = plt.hist2d(
#         eta_jet, phi_jet,
#         bins=[50, 50],                # [number of eta bins, number of phi bins]
#         range=[eta_range, phi_range],  # [x_range, y_range]
#         cmap='viridis'
#     )
#     plt.colorbar(img, label='Counts')
#     plt.xlabel('eta_jet')
#     plt.ylabel('phi_jet')
#     plt.title('2D Histogram Heatmap')
#     # plt.show()
#     # validation plots could be compared with that from https://cms-pub-talk.web.cern.ch/t/jme-object-review-of-b2g-24-010/30422/3
#     if hem_veto_on:
#         plt.savefig(f"{save_fname}_hemVetoOn.png")
#     else:
#         plt.savefig(f"{save_fname}_hemVetoOff.png")


def getHemVetoDataRatio(compute_dict):
    """
    compute_dict: stage1 output of data A,B,C,D of 2018UL
    returns: a float point of the portion of the data that has been vetoed if any jets were within the HEM region
    """
    HemVeto_filter = compute_dict["HemVeto_filter"] # here, True means no Veto
    nevents = ak.ones_like(HemVeto_filter, dtype="bool")
    nevents = ak.sum(nevents)
    # is_HemRegion = compute_dict["is_HemRegion"]
    # is_HemRegionAndHemVeto = is_HemRegion & HemVeto_filter
    # veto_ratio = ak.sum(is_HemRegionAndHemVeto) / ak.sum(is_HemRegion)
    veto_ratio = ak.sum(HemVeto_filter) / nevents
    return (1-veto_ratio)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="2018",
    action="store",
    help="string value of year we are calculating",
    )
    parser.add_argument(
    "-data",
    "--data",
    dest="data_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of data samples represented by alphabetical letters A-H",
    )
    parser.add_argument(
    "-bkg",
    "--background",
    dest="bkg_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of bkg samples represented by shorthands: DY, TT, ST, DB (diboson), EWK",
    )
    parser.add_argument(
    "-sig",
    "--signal",
    dest="sig_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of sig samples represented by shorthands: ggH, VBF",
    )
    parser.add_argument(
    "-var",
    "--variables",
    dest="variables",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of variables to plot (ie: jet, mu, dimuon)",
    )
    parser.add_argument(
    "-load",
    "--load_path",
    dest="load_path",
    default="/depot/cms/users/yun79/results/stage1/test_full/f0_1",
    action="store",
    help="load path",
    )
    parser.add_argument(
    "-label",
    "--label",
    dest="label",
    default="",
    action="store",
    help="label",
    )
    parser.add_argument(
    "-save",
    "--save_path",
    dest="save_path",
    default="./plots/",
    action="store",
    help="save path",
    )
    parser.add_argument(
    "-lumi",
    "--lumi",
    dest="lumi",
    default="",
    action="store",
    help="string value of integrated luminosity to label",
    )
    parser.add_argument(
    "--status",
    dest="status",
    default="",
    action="store",
    help="Status of results ie Private, Preliminary, In Progress",
    )
    parser.add_argument(
    "--no_ratio",
    dest="no_ratio",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="doesn't plot Data/MC ratio",
    )
    parser.add_argument(
    "--linear_scale",
    dest="linear_scale",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, provide plots in linear scale",
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
    "--use_gateway",
    dest="use_gateway",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, uses dask gateway client instead of local",
    )
    parser.add_argument(
    "-cat",
    "--category",
    dest="category",
    default="nocat",
    action="store",
    help="define production mode category. optionsare ggh, vbf and nocat (no category cut)",
    )
    parser.add_argument(
    "-plotset",
    "--plot_setting",
    dest="plot_setting",
    default="",
    action="store",
    help="path to load json file with plott binning and xlabels",
    )

    client = Client(n_workers=32,  threads_per_worker=1, processes=True, memory_limit='10 GiB') 
    
    #---------------------------------------------------------
    # gather arguments
    args = parser.parse_args()
    available_processes = []
    # if doing VBF filter study, add the vbf filter sample to the DY group
    
    plt.style.use(hep.style.CMS)
    fields2load = [
        "HemVeto_filter",
        # "jet1_eta_nominal",
        # "jet2_eta_nominal",
        # "jet1_phi_nominal",
        # "jet2_phi_nominal",
        # "is_HemRegion",
    ]
    label = "HemVetoStudy_04Apr2025"
    year="2018"
    load_path = f"/depot/cms/users/yun79/hmm/copperheadV1clean/{label}/stage1_output/{year}/f1_0"

    # data2load = ['data_A', 'data_B', 'data_C', 'data_D']
    # # data2load = ['data_C', 'data_B']
    # events_l = []
    # for data_name in data2load:
    #     events = dak.from_parquet(f"{load_path}/{data_name}/*/*.parquet")
    #     events_l.append(events)
        
    # events = ak.concatenate(events_l)
    year = "*"
    events = dak.from_parquet(f"{load_path}/{year}/*/*.parquet")
    print(events)
    compute_dict = {
        field: ak.fill_none(events[field], value=False) for field in fields2load
    }
    compute_dict = dask.compute(compute_dict)[0]
    # compute_dict = dask.compute(ak.zip(compute_dict)) 
    print(compute_dict)

    veto_ratio = getHemVetoDataRatio(compute_dict)
    print(f"The proportion of 2018UL data events vetoed in HEM region due to bad HEM jets is {veto_ratio}")

    #---------------------------------------------------------

