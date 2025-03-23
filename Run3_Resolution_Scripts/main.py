import dask_awkward as dak
import awkward as ak
from distributed import Client
import time
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
import ROOT as rt
import os

plt.style.use(hep.style.CMS)
rt.gStyle.SetOptStat(0000)

from plot_vars import plot_muon_dimuon_kinematics
from plot_vars import compare_kinematics
from plot_vars import compare_kinematics_2D
from plot_ptRes import *


def read_files(control_region):
    fields_to_compute = [
        "wgt_nominal",
        "nBtagLoose_nominal", "nBtagMedium_nominal",
        "mu1_pt",  "mu1_eta", "mu1_phi",
        "mu2_pt",  "mu2_eta", "mu2_phi",
        "mu1_ptErr", "mu2_ptErr",
        "dimuon_pt", "dimuon_eta", "dimuon_rapidity", "dimuon_phi", "dimuon_mass",
        "dimuon_ebe_mass_res_rel",
        "jet1_pt_nominal", "jet1_eta_nominal", "jet1_phi_nominal",
        "jet2_pt_nominal", "jet2_eta_nominal", "jet2_phi_nominal",
        "jj_mass_nominal", "jj_dEta_nominal",
        "event"
    ]

    # year = 2018 or 2022preEE
    year = "2022preEE"
    # year = "2018"

    if str(year) == "2018":
        load_path_bs_on = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_24Feb_BSon/stage1_output/2018/f1_0/data_*/*/*.parquet"
        load_path_bs_off = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_24Feb_BSoff/stage1_output/2018/f1_0/data_*/*/*.parquet"
        # load_path_bs_off = "/depot/cms/users/yun79/hmm/copperheadV1clean/UpdatedDY_100_200_CrossSection_24Feb/stage1_output/2018/f1_0/data_*/*/*.parquet"
    elif str(year) == "2022preEE":
        # load_path_bs_on = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run3_nanoAODv12_BSOn/stage1_output/2022preEE/f1_0/data_*/*/*.parquet"
        load_path_bs_on = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run3_nanoAODv12_BSOn_UpdateMassCalib/stage1_output/2022preEE/f1_0/data_*/*/*.parquet"
        # load_path_bs_off = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run3_nanoAODv12_BSOff/stage1_output/2022preEE/f1_0/data_*/*/*.parquet"
        load_path_bs_off = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_12March_NoGeoNoBSC//stage1_output/2022preEE/f1_0/data_*/*/*.parquet"

    events_data = dak.from_parquet(load_path_bs_on)
    events_data = ak.zip({field: events_data[field] for field in fields_to_compute}).compute()
    events_bs_on = filter_region(events_data, region=control_region)

    events_data = dak.from_parquet(load_path_bs_off)
    events_data = ak.zip({field: events_data[field] for field in fields_to_compute}).compute()
    events_bs_off = filter_region(events_data, region=control_region)

    return events_bs_on, events_bs_off

if __name__ == "__main__":
    client = Client(n_workers=15, threads_per_worker=2, processes=True, memory_limit='8 GiB')

    # control_region = "signal"
    control_region = "z-peak"
    events_bs_on, events_bs_off = read_files(control_region)

    # Add variable: ptErr/pT for both leading and sub-leading muons
    events_bs_on = ak.with_field(events_bs_on, events_bs_on.mu1_ptErr/events_bs_on.mu1_pt, "mu1_Ratio_pTErr_pt")
    events_bs_off = ak.with_field(events_bs_off, events_bs_off.mu1_ptErr/events_bs_off.mu1_pt, "mu1_Ratio_pTErr_pt")
    events_bs_on = ak.with_field(events_bs_on, events_bs_on.mu2_ptErr/events_bs_on.mu2_pt, "mu2_Ratio_pTErr_pt")
    events_bs_off = ak.with_field(events_bs_off, events_bs_off.mu2_ptErr/events_bs_off.mu2_pt, "mu2_Ratio_pTErr_pt")

# DistributionComparer
    # print entries in each dataset
    print(f"Number of events in bs_on: {len(events_bs_on)}")
    print(f"Number of events in bs_off: {len(events_bs_off)}")

    getBasicVariables = False
    getBasicVariables_pTDipInvestigate = True
    getBasicVariables_2D = False
    getHigherOrderVariables = False
    ggH_Filter = False
    vbf_Filter = False # error range

    # get log_plots dir if it doesn't exist
    if not os.path.exists("log_plots"):
        os.makedirs("log_plots")

    if getBasicVariables_pTDipInvestigate:
        compare_kinematics(events_bs_on, events_bs_off, "mu2_pt", "Sub-Leading Muon p_{T} [GeV]", save_filename="kinematics_comparison"+"_"+control_region)
        # get 2D plots for sub-leading muon pT and eta
        compare_kinematics_2D(events_bs_on, events_bs_off, "mu2_pt", "mu2_eta", "Sub-Leading Muon p_{T} [GeV]", "Sub-Leading Muon #eta", save_filename="kinematics_comparison_2DPlots"+"_"+control_region)
        compare_kinematics_2D(events_bs_on, events_bs_off, "mu2_pt", "mu2_phi", "Sub-Leading Muon p_{T} [GeV]", "Sub-Leading Muon #phi", save_filename="kinematics_comparison_2DPlots"+"_"+control_region)
        compare_kinematics_2D(events_bs_on, events_bs_off, "mu2_eta", "mu2_phi", "Sub-Leading Muon #eta", "Sub-Leading Muon #phi", save_filename="kinematics_comparison_2DPlots"+"_"+control_region)

        # get 2D plots for leading muon pT and eta
        compare_kinematics_2D(events_bs_on, events_bs_off, "mu1_pt", "mu1_eta", "Leading Muon p_{T} [GeV]", "Leading Muon #eta", save_filename="kinematics_comparison_2DPlots"+"_"+control_region)
        compare_kinematics_2D(events_bs_on, events_bs_off, "mu1_pt", "mu1_phi", "Leading Muon p_{T} [GeV]", "Leading Muon #phi", save_filename="kinematics_comparison_2DPlots"+"_"+control_region)
        compare_kinematics_2D(events_bs_on, events_bs_off, "mu1_eta", "mu1_phi", "Leading Muon #eta", "Leading Muon #phi", save_filename="kinematics_comparison_2DPlots"+"_"+control_region)

        # get 2D plots for leading muon and sub-leading muon pT, eta and phi
        compare_kinematics_2D(events_bs_on, events_bs_off, "mu1_pt", "mu2_pt", "Leading Muon p_{T} [GeV]", "Sub-Leading Muon p_{T} [GeV]", save_filename="kinematics_comparison_2DPlots"+"_"+control_region)
        compare_kinematics_2D(events_bs_on, events_bs_off, "mu1_eta", "mu2_eta", "Leading Muon #eta", "Sub-Leading Muon #eta", save_filename="kinematics_comparison_2DPlots"+"_"+control_region)
        compare_kinematics_2D(events_bs_on, events_bs_off, "mu1_phi", "mu2_phi", "Leading Muon #phi", "Sub-Leading Muon #phi", save_filename="kinematics_comparison_2DPlots"+"_"+control_region)

        # add sub-leading muon pT cut 40-50 GeV
        events_bs_on = events_bs_on[(events_bs_on.mu2_pt > 40) & (events_bs_on.mu2_pt < 55)]
        events_bs_off = events_bs_off[(events_bs_off.mu2_pt > 40) & (events_bs_off.mu2_pt < 55)]
        # get 2D plots for sub-leading muon pT and eta
        compare_kinematics_2D(events_bs_on, events_bs_off, "mu2_pt", "mu2_eta", "Sub-Leading Muon p_{T} [GeV]", "Sub-Leading Muon #eta", save_filename="kinematics_comparison_2DPlots"+"_"+control_region+"_subleadMuPtCut40-50")
        compare_kinematics_2D(events_bs_on, events_bs_off, "mu2_pt", "mu2_phi", "Sub-Leading Muon p_{T} [GeV]", "Sub-Leading Muon #phi", save_filename="kinematics_comparison_2DPlots"+"_"+control_region+"_subleadMuPtCut40-50")

        pass

    if getBasicVariables_2D:
        compare_kinematics_2D(events_bs_on, events_bs_off, "mu1_pt", "mu1_ptErr", "Leading Muon p_{T} [GeV]", "Leading Muon p_{T} Error [GeV]", save_filename="kinematics_comparison_2DPlots"+"_"+control_region)
        compare_kinematics_2D(events_bs_on, events_bs_off, "mu1_eta", "mu1_ptErr", "Leading Muon #eta", "Leading Muon p_{T} Error [GeV]", save_filename="kinematics_comparison_2DPlots"+"_"+control_region)
        compare_kinematics_2D(events_bs_on, events_bs_off, "mu1_phi", "mu1_ptErr", "Leading Muon #phi", "Leading Muon p_{T} Error [GeV]", save_filename="kinematics_comparison_2DPlots"+"_"+control_region)

        compare_kinematics_2D(events_bs_on, events_bs_off, "mu2_pt", "mu2_ptErr", "Sub-Leading Muon p_{T} [GeV]", "Sub-Leading Muon p_{T} Error [GeV]", save_filename="kinematics_comparison_2DPlots"+"_"+control_region)
        compare_kinematics_2D(events_bs_on, events_bs_off, "mu2_eta", "mu2_ptErr", "Sub-Leading Muon #eta", "Sub-Leading Muon p_{T} Error [GeV]", save_filename="kinematics_comparison_2DPlots"+"_"+control_region)
        compare_kinematics_2D(events_bs_on, events_bs_off, "mu2_phi", "mu2_ptErr", "Sub-Leading Muon #phi", "Sub-Leading Muon p_{T} Error [GeV]", save_filename="kinematics_comparison_2DPlots"+"_"+control_region)


        compare_kinematics_2D(events_bs_on, events_bs_off, "mu1_pt", "mu1_Ratio_pTErr_pt", "Leading Muon p_{T} [GeV]", "Leading Muon #delta p_{T}/p_{T}", save_filename="kinematics_comparison_2DPlots"+"_"+control_region)
        compare_kinematics_2D(events_bs_on, events_bs_off, "mu1_eta", "mu1_Ratio_pTErr_pt", "Leading Muon #eta", "Leading Muon #delta p_{T}/p_{T}", save_filename="kinematics_comparison_2DPlots"+"_"+control_region)

        compare_kinematics_2D(events_bs_on, events_bs_off, "mu2_pt", "mu2_Ratio_pTErr_pt", "Sub-Leading Muon p_{T} [GeV]", "Sub-Leading Muon #delta p_{T}/p_{T}", save_filename="kinematics_comparison_2DPlots"+"_"+control_region)
        compare_kinematics_2D(events_bs_on, events_bs_off, "mu2_eta", "mu2_Ratio_pTErr_pt", "Sub-Leading Muon #eta", "Sub-Leading Muon #delta p_{T}/p_{T}", save_filename="kinematics_comparison_2DPlots"+"_"+control_region)

        compare_kinematics_2D(events_bs_on, events_bs_off, "mu1_phi", "mu1_Ratio_pTErr_pt", "Leading Muon #phi", "Leading Muon #delta p_{T}/p_{T}", save_filename="kinematics_comparison_2DPlots"+"_"+control_region)
        compare_kinematics_2D(events_bs_on, events_bs_off, "mu2_phi", "mu2_Ratio_pTErr_pt", "Sub-Leading Muon #phi", "Sub-Leading Muon #delta p_{T}/p_{T}", save_filename="kinematics_comparison_2DPlots"+"_"+control_region)

        for region in ["B", "O", "E"]:
            events_bs_on_region = filter_region_using_rapidity_leadMuon(events_bs_on, region)
            events_bs_off_region = filter_region_using_rapidity_leadMuon(events_bs_off, region)
            compare_kinematics_2D(events_bs_on_region, events_bs_off_region, "mu1_pt", "mu1_Ratio_pTErr_pt", "Leading Muon p_{T} [GeV]", "Leading Muon #delta p_{T}/p_{T}", save_filename="kinematics_comparison_2DPlots"+"_"+control_region+"_"+region)
            compare_kinematics_2D(events_bs_on_region, events_bs_off_region, "mu1_eta", "mu1_Ratio_pTErr_pt", "Leading Muon #eta", "Leading Muon #delta p_{T}/p_{T}", save_filename="kinematics_comparison_2DPlots"+"_"+control_region+"_"+region)
            compare_kinematics_2D(events_bs_on_region, events_bs_off_region, "mu1_phi", "mu1_Ratio_pTErr_pt", "Leading Muon #phi", "Leading Muon #delta p_{T}/p_{T}", save_filename="kinematics_comparison_2DPlots"+"_"+control_region+"_"+region)

            events_bs_on_region = filter_region_using_rapidity_SubleadMuon(events_bs_on, region)
            events_bs_off_region = filter_region_using_rapidity_SubleadMuon(events_bs_off, region)
            compare_kinematics_2D(events_bs_on_region, events_bs_off_region, "mu2_pt", "mu2_Ratio_pTErr_pt", "Sub-Leading Muon p_{T} [GeV]", "Sub-Leading Muon #delta p_{T}/p_{T}", save_filename="kinematics_comparison_2DPlots"+"_"+control_region+"_"+region)
            compare_kinematics_2D(events_bs_on_region, events_bs_off_region, "mu2_eta", "mu2_Ratio_pTErr_pt", "Sub-Leading Muon #eta", "Sub-Leading Muon #delta p_{T}/p_{T}", save_filename="kinematics_comparison_2DPlots"+"_"+control_region+"_"+region)
            compare_kinematics_2D(events_bs_on_region, events_bs_off_region, "mu2_phi", "mu2_Ratio_pTErr_pt", "Sub-Leading Muon #phi", "Sub-Leading Muon #delta p_{T}/p_{T}", save_filename="kinematics_comparison_2DPlots"+"_"+control_region+"_"+region)

    if getBasicVariables:
        compare_kinematics(events_bs_on, events_bs_off, "mu1_pt", "Leading Muon p_{T} [GeV]", save_filename="kinematics_comparison"+"_"+control_region)
        compare_kinematics(events_bs_on, events_bs_off, "mu1_ptErr", "Leading Muon p_{T} Error [GeV]", save_filename="kinematics_comparison"+"_"+control_region)
        compare_kinematics(events_bs_on, events_bs_off, "mu1_eta", "Leading Muon #eta ", save_filename="kinematics_comparison"+"_"+control_region)
        compare_kinematics(events_bs_on, events_bs_off, "mu1_phi", "Leading Muon #phi", save_filename="kinematics_comparison"+"_"+control_region)
        compare_kinematics(events_bs_on, events_bs_off, "mu1_Ratio_pTErr_pt", "Leading Muon #delta p_{T}/p_{T}", save_filename="kinematics_comparison"+"_"+control_region)

        compare_kinematics(events_bs_on, events_bs_off, "mu2_pt", "Sub-Leading Muon p_{T} [GeV]", save_filename="kinematics_comparison"+"_"+control_region)
        compare_kinematics(events_bs_on, events_bs_off, "mu2_ptErr", "Sub-Leading Muon p_{T} Error [GeV]", save_filename="kinematics_comparison"+"_"+control_region)
        compare_kinematics(events_bs_on, events_bs_off, "mu2_eta", "Sub-Leading Muon #eta ", save_filename="kinematics_comparison"+"_"+control_region)
        compare_kinematics(events_bs_on, events_bs_off, "mu2_phi", "Sub-Leading Muon #phi", save_filename="kinematics_comparison"+"_"+control_region)
        compare_kinematics(events_bs_on, events_bs_off, "mu2_Ratio_pTErr_pt", "Sub-Leading Muon #delta p_{T}/p_{T}", save_filename="kinematics_comparison"+"_"+control_region)

        compare_kinematics(events_bs_on, events_bs_off, "dimuon_pt", " p_{T} (#mu #mu) [GeV]", save_filename="kinematics_comparison"+"_"+control_region)
        compare_kinematics(events_bs_on, events_bs_off, "dimuon_eta", " #eta (#mu #mu) ", save_filename="kinematics_comparison"+"_"+control_region)
        compare_kinematics(events_bs_on, events_bs_off, "dimuon_phi", " #phi (#mu #mu)", save_filename="kinematics_comparison"+"_"+control_region)
        compare_kinematics(events_bs_on, events_bs_off, "dimuon_mass", "  Invariant Mass (#mu #mu) [GeV]", save_filename="kinematics_comparison"+"_"+control_region)
        compare_kinematics(events_bs_on, events_bs_off, "dimuon_rapidity", " Rapidity (#mu #mu) ", save_filename="kinematics_comparison"+"_"+control_region)
        compare_kinematics(events_bs_on, events_bs_off, "dimuon_ebe_mass_res_rel", " Relative Mass Resolution (#mu #mu) ", save_filename="kinematics_comparison"+"_"+control_region)

        for region in ["B", "O", "E"]:
            events_bs_on_region = filter_region_using_rapidity_leadMuon(events_bs_on, region)
            events_bs_off_region = filter_region_using_rapidity_leadMuon(events_bs_off, region)
            compare_kinematics(events_bs_on_region, events_bs_off_region, "mu1_ptErr", "Leading Muon p_{T} Error [GeV]", save_filename="kinematics_comparison"+"_"+control_region+"_"+region)
            compare_kinematics(events_bs_on_region, events_bs_off_region, "mu1_Ratio_pTErr_pt", "Leading Muon #delta p_{T}/p_{T}", save_filename="kinematics_comparison"+"_"+control_region+"_"+region)

            events_bs_on_region = filter_region_using_rapidity_SubleadMuon(events_bs_on, region)
            events_bs_off_region = filter_region_using_rapidity_SubleadMuon(events_bs_off, region)
            compare_kinematics(events_bs_on_region, events_bs_off_region, "mu2_ptErr", "Sub-Leading Muon p_{T} Error [GeV]", save_filename="kinematics_comparison"+"_"+control_region+"_"+region)
            compare_kinematics(events_bs_on_region, events_bs_off_region, "mu2_Ratio_pTErr_pt", "Sub-Leading Muon #delta p_{T}/p_{T}", save_filename="kinematics_comparison"+"_"+control_region+"_"+region)

        for region in ["BB", "BO", "OB", "OO", "BE", "EB", "EO", "OE", "EE"]:
            events_bs_on_region = filter_region_using_rapidity(events_bs_on, region)
            events_bs_off_region = filter_region_using_rapidity(events_bs_off, region)
            compare_kinematics(events_bs_on_region, events_bs_off_region, "dimuon_ebe_mass_res_rel", " Relative Mass Resolution (#mu #mu) ", save_filename="kinematics_comparison"+"_"+control_region+"_"+region)
            compare_kinematics(events_bs_on_region, events_bs_off_region, "dimuon_mass", " Invariant Mass (#mu #mu) [GeV]", save_filename="kinematics_comparison"+"_"+control_region+"_"+region)

    if ggH_Filter:
        # apply apply_ggh_cut(events)
        events_bs_on = apply_ggh_cut(events_bs_on)
        events_bs_off = apply_ggh_cut(events_bs_off)

        # plot events_bs_on.dimuon_mass, events_bs_on.dimuon_pt: Pass this info to the fit_plot_ggh function
        wgt_bs_on = ak.to_numpy(events_bs_on.wgt_nominal)
        wgt_bs_off = ak.to_numpy(events_bs_off.wgt_nominal)

        dimuon_mass_bs_on = ak.to_numpy(events_bs_on.dimuon_mass)
        dimuon_mass_bs_off = ak.to_numpy(events_bs_off.dimuon_mass)
        plot_hist_var(dimuon_mass_bs_on, wgt_bs_on, dimuon_mass_bs_off, wgt_bs_off, "M_{#mu#mu} [GeV]", "Dimuon Invariant Mass", 100, 70, 110 ,f"dimuon_mass_{control_region}_ggH.pdf", control_region)

        for region in ["BB", "BO", "OB", "OO", "BE", "EB", "EO", "OE", "EE"]:
            events_bs_on_region = filter_region_using_rapidity(events_bs_on, region)
            events_bs_off_region = filter_region_using_rapidity(events_bs_off, region)

            dimuon_mass_bs_on_region = ak.to_numpy(events_bs_on_region.dimuon_mass)
            dimuon_mass_bs_off_region = ak.to_numpy(events_bs_off_region.dimuon_mass)

            wgt_bs_on = ak.to_numpy(events_bs_on_region.wgt_nominal)
            wgt_bs_off = ak.to_numpy(events_bs_off_region.wgt_nominal)

            plot_hist_var(dimuon_mass_bs_on_region, wgt_bs_on, dimuon_mass_bs_off_region, wgt_bs_off, "M_{#mu#mu} [GeV]", f"Dimuon Invariant Mass : {region}", 100, 70, 110 ,f"dimuon_mass_{control_region}_ggH_{region}.pdf", control_region)

    if vbf_Filter:
        # apply apply_vbf_cut(events)
        events_bs_on = apply_vbf_cut(events_bs_on)
        events_bs_off = apply_vbf_cut(events_bs_off)

        # plot events_bs_on.dimuon_mass, events_bs_on.dimuon_pt: Pass this info to the fit_plot_ggh function
        wgt_bs_on = ak.to_numpy(events_bs_on.wgt_nominal)
        wgt_bs_off = ak.to_numpy(events_bs_off.wgt_nominal)

        dimuon_mass_bs_on = ak.to_numpy(events_bs_on.dimuon_mass)
        dimuon_mass_bs_off = ak.to_numpy(events_bs_off.dimuon_mass)
        plot_hist_var(dimuon_mass_bs_on, wgt_bs_on, dimuon_mass_bs_off, wgt_bs_off, "M_{#mu#mu} [GeV]", "Dimuon Invariant Mass", 100, 70, 110 ,f"dimuon_mass_{control_region}_VBF.pdf", control_region)

        for region in ["BB", "BO", "OB", "OO", "BE", "EB", "EO", "OE", "EE"]:
            events_bs_on_region = filter_region_using_rapidity(events_bs_on, region)
            events_bs_off_region = filter_region_using_rapidity(events_bs_off, region)

            dimuon_mass_bs_on_region = ak.to_numpy(events_bs_on_region.dimuon_mass)
            dimuon_mass_bs_off_region = ak.to_numpy(events_bs_off_region.dimuon_mass)

            wgt_bs_on = ak.to_numpy(events_bs_on_region.wgt_nominal)
            wgt_bs_off = ak.to_numpy(events_bs_off_region.wgt_nominal)

            plot_hist_var(dimuon_mass_bs_on_region, wgt_bs_on, dimuon_mass_bs_off_region, wgt_bs_off, "M_{#mu#mu} [GeV]", f"Dimuon Invariant Mass : {region}", 100, 70, 110 ,f"dimuon_mass_{control_region}_VBF_{region}.pdf", control_region)


    if getHigherOrderVariables:
        # # plot events_bs_on.dimuon_mass, events_bs_on.dimuon_pt: Pass this info to the fit_plot_ggh function
        # wgt_bs_on = ak.to_numpy(events_bs_on.wgt_nominal)
        # wgt_bs_off = ak.to_numpy(events_bs_off.wgt_nominal)

        # dimuon_mass_bs_on = ak.to_numpy(events_bs_on.dimuon_mass)
        # dimuon_mass_bs_off = ak.to_numpy(events_bs_off.dimuon_mass)
        # plot_hist_var(dimuon_mass_bs_on, wgt_bs_on, dimuon_mass_bs_off, wgt_bs_off, "M_{#mu#mu} [GeV]", "Dimuon Invariant Mass", 100, 70, 110 ,f"dimuon_mass_{control_region}.pdf", control_region)

        # for region in ["BB", "BO", "OB", "OO", "BE", "EB", "EO", "OE", "EE"]:
        #     events_bs_on_region = filter_region_using_rapidity(events_bs_on, region)
        #     events_bs_off_region = filter_region_using_rapidity(events_bs_off, region)

        #     dimuon_mass_bs_on_region = ak.to_numpy(events_bs_on_region.dimuon_mass)
        #     dimuon_mass_bs_off_region = ak.to_numpy(events_bs_off_region.dimuon_mass)

        #     wgt_bs_on = ak.to_numpy(events_bs_on_region.wgt_nominal)
        #     wgt_bs_off = ak.to_numpy(events_bs_off_region.wgt_nominal)

        #     plot_hist_var(dimuon_mass_bs_on_region, wgt_bs_on, dimuon_mass_bs_off_region, wgt_bs_off, "M_{#mu#mu} [GeV]", f"Dimuon Invariant Mass : {region}", 100, 70, 110 ,f"dimuon_mass_{control_region}_{region}.pdf", control_region)


        # ------------------- get pTErr/pT -------------------

        # fit_plot_ggh(events_bs_on, events_bs_off, f"BSC_geofit_comparison_2022PreEE_Mu2_{control_region}_all.pdf", save_plot=True, region="Inclusive")
        # for region in ["B", "O", "E"]:
        #     events_bs_on_region = filter_region_using_rapidity_leadMuon(events_bs_on, region)
        #     events_bs_off_region = filter_region_using_rapidity_leadMuon(events_bs_off, region)
        #     fit_plot_ggh(events_bs_on_region, events_bs_off_region, f"BSC_geofit_comparison_2022PreEE_Mu2_{control_region}_{region}.pdf", save_plot=True, region=region)
        pass
