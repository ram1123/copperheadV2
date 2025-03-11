import dask_awkward as dak
import awkward as ak
from distributed import Client
import time
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
import ROOT as rt

plt.style.use(hep.style.CMS)
rt.gStyle.SetOptStat(0000)

from plot_vars import plot_muon_dimuon_kinematics
from plot_vars import compare_kinematics
from plot_ptRes import *


def read_files(control_region):
    fields_to_compute = [
        "wgt_nominal",
        "mu1_pt", "mu1_ptErr", "mu1_eta", "mu1_phi",
        "mu2_pt", "mu2_ptErr",  "mu2_eta", "mu2_phi",
        "dimuon_pt", "dimuon_eta", "dimuon_phi", "dimuon_mass",
        "event"
    ]


    load_path = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run3_nanoAODv12_BSOn/stage1_output/2022preEE/f1_0/data_*/*/*.parquet"
    events_data = dak.from_parquet(load_path)
    events_data = ak.zip({field: events_data[field] for field in fields_to_compute}).compute()
    events_bs_on = filter_region(events_data, region=control_region)

    load_path = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run3_nanoAODv12_BSOff/stage1_output/2022preEE/f1_0/data_*/*/*.parquet"
    events_data = dak.from_parquet(load_path)
    events_data = ak.zip({field: events_data[field] for field in fields_to_compute}).compute()
    events_bs_off = filter_region(events_data, region=control_region)

    return events_bs_on, events_bs_off

if __name__ == "__main__":
    client = Client(n_workers=15, threads_per_worker=2, processes=True, memory_limit='8 GiB')

    # control_region = "signal"
    control_region = "z-peak"
    events_bs_on, events_bs_off = read_files(control_region)

    # print entries in each dataset
    print(f"Number of events in bs_on: {len(events_bs_on)}")
    print(f"Number of events in bs_off: {len(events_bs_off)}")

    getBasicVariables = True
    getHigherOrderVariables = True

    if getBasicVariables:
        # plot_muon_dimuon_kinematics(events_bs_on, save_prefix="kinematics_bs_on")
        # plot_muon_dimuon_kinematics(events_bs_off, save_prefix="kinematics_bs_off")

        compare_kinematics(events_bs_on, events_bs_off, "mu1_pt", "Leading Muon p_{T} [GeV]", save_filename="kinematics_comparison"+"_"+control_region)
        compare_kinematics(events_bs_on, events_bs_off, "mu1_eta", "Leading Muon #eta ", save_filename="kinematics_comparison"+"_"+control_region)
        compare_kinematics(events_bs_on, events_bs_off, "mu1_phi", "Leading Muon #phi", save_filename="kinematics_comparison"+"_"+control_region)

        compare_kinematics(events_bs_on, events_bs_off, "mu2_pt", "Sub-Leading Muon p_{T} [GeV]", save_filename="kinematics_comparison"+"_"+control_region)
        compare_kinematics(events_bs_on, events_bs_off, "mu2_eta", "Sub-Leading Muon #eta ", save_filename="kinematics_comparison"+"_"+control_region)
        compare_kinematics(events_bs_on, events_bs_off, "mu2_phi", "Sub-Leading Muon #phi", save_filename="kinematics_comparison"+"_"+control_region)

        compare_kinematics(events_bs_on, events_bs_off, "dimuon_pt", "(H#rightarrow #mu #mu) p_{T} [GeV]", save_filename="kinematics_comparison"+"_"+control_region)
        compare_kinematics(events_bs_on, events_bs_off, "dimuon_eta", "(H#rightarrow #mu #mu) #eta ", save_filename="kinematics_comparison"+"_"+control_region)
        compare_kinematics(events_bs_on, events_bs_off, "dimuon_phi", "(H#rightarrow #mu #mu) #phi", save_filename="kinematics_comparison"+"_"+control_region)
        compare_kinematics(events_bs_on, events_bs_off, "dimuon_mass", "(H#rightarrow #mu #mu)  Invariant Mass [GeV]", save_filename="kinematics_comparison"+"_"+control_region)


    if getHigherOrderVariables:
        fit_plot_ggh(events_bs_on, events_bs_off, f"BSC_geofit_comparison_2022PreEE_dpT_{control_region}_all.pdf", save_plot=True, region="Inclusive")

        for region in ["BB", "BO", "OB", "OO", "BE", "EB", "EO", "OE", "EE"]:
            events_bs_on_region = filter_region_using_rapidity(events_bs_on, region)
            events_bs_off_region = filter_region_using_rapidity(events_bs_off, region)
            fit_plot_ggh(events_bs_on_region, events_bs_off_region, f"BSC_geofit_comparison_2022PreEE_dpT_{control_region}_{region}.pdf", save_plot=True, region=region)


