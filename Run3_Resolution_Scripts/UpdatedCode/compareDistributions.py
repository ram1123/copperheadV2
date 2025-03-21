# import dask_awkward as dak

from modules.DistributionCompare import DistributionCompare

# Example main usage
if __name__ == "__main__":
    varlist_file = "varlist.yaml"

    year = "2022preEE"  # or "2018"
    control_region = "z-peak"  # or "signal"
    directoryTag = "March21_Data_DiMuon_RapidityBins"

    input_paths_labels_1 = { # Compare with data 2022preEE
        "BSC ON": "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run3_nanoAODv12_BSOn_UpdateMassCalib/stage1_output/2022preEE/f1_0/data_*/*/*.parquet",
        "BSC OFF": "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_12March_NoGeoNoBSC//stage1_output/2022preEE/f1_0/data_*/*/*.parquet",
    }

    input_paths_labels_2 = { # Compare with data 2022preEE
        "BSC ON": "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run3_nanoAODv12_BSOn_UpdateMassCalib/stage1_output/2022preEE/f1_0/data_*/*/*.parquet",
        "GeoFit (2018)": "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run3_nanoAODv12_BSOff/stage1_output/2022preEE/f1_0/data_*/*/*.parquet",
    }

    input_paths_labels_3 = { # Compare with MC 2022preEE
        "BSC ON": "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run3_nanoAODv12_BSOn/stage1_output/2022preEE/f1_0/ggh_powhegPS/0/*.parquet",
        "BSC OFF": "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_12March_NoGeoNoBSC//stage1_output/2022preEE/f1_0/ggh_powhegPS/0/*.parquet"
    }

    input_paths_labels_4 = { # Compare with MC 2022preEE
        "BSC ON": "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run3_nanoAODv12_BSOn/stage1_output/2022preEE/f1_0/ggh_powhegPS/0/*.parquet",
        "GeoFit (2018)": "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run3_nanoAODv12_BSOff/stage1_output/2022preEE/f1_0/data_*/*/*.parquet"
    }

    input_paths_labels = input_paths_labels_2

    fields_to_load = [
        "wgt_nominal",
        "nBtagLoose_nominal", "nBtagMedium_nominal",
        "mu1_pt",  "mu1_eta", "mu1_phi",
        "mu2_pt",  "mu2_eta", "mu2_phi",
        "mu1_ptErr", "mu2_ptErr",
        "dimuon_pt", "dimuon_eta", "dimuon_rapidity", "dimuon_phi", "dimuon_mass",
        "dimuon_ebe_mass_res_rel",
        # "jet1_pt_nominal", "jet1_eta_nominal", "jet1_phi_nominal",
        # "jet2_pt_nominal", "jet2_eta_nominal", "jet2_phi_nominal",
        # "jj_mass_nominal", "jj_dEta_nominal",
        "event"
    ]

    comparer = DistributionCompare(year, input_paths_labels, fields_to_load, control_region, directoryTag, varlist_file)
    comparer.load_data()
    # Add new variable: ptErr/pT for both leading and sub-leading muons
    comparer.add_new_variable()

    LeadingMuon_Variables = ["mu1_pt", "mu1_ptErr", "ratio_pTErr_pt_mu1", "mu1_eta", "mu1_phi"]
    SubleadingMuon_Variables = ["mu2_pt", "mu2_ptErr", "ratio_pTErr_pt_mu2", "mu2_eta", "mu2_phi"]
    Dimuon_Variables = ["dimuon_pt", "dimuon_eta", "dimuon_phi", "dimuon_mass", "dimuon_ebe_mass_res_rel"]

    # variables_to_plot = ["mu1_pt"]
    variables_to_plot = LeadingMuon_Variables + SubleadingMuon_Variables + Dimuon_Variables
    # comparer.compare_all(variables_to_plot)

    # # Get 2D histograms for all combinations of leading  muon variables
    # for i, var1 in enumerate(LeadingMuon_Variables):
    #     for var2 in LeadingMuon_Variables[i+1:]:
    #         comparer.compare_2D(var1, var2)

    # # Get 2D histograms for all combinations of subleading  muon variables
    # for i, var1 in enumerate(SubleadingMuon_Variables):
    #     for var2 in SubleadingMuon_Variables[i+1:]:
    #         comparer.compare_2D(var1, var2)

    # # Get 2D histograms for all combinations of leading and subleading muon variables
    # for var1 in LeadingMuon_Variables:
    #     for var2 in SubleadingMuon_Variables:
    #         comparer.compare_2D(var1, var2)

    # # Get 2D histograms for all combinations of leading and dimuon variables
    # for var1 in LeadingMuon_Variables:
    #     for var2 in Dimuon_Variables:
    #         comparer.compare_2D(var1, var2)
    # # Get 2D histograms for all combinations of subleading and dimuon variables
    # for var1 in SubleadingMuon_Variables:
    #     for var2 in Dimuon_Variables:
    #         comparer.compare_2D(var1, var2)

    # # Get 2D histograms for all combinations of dimuon variables
    # for i, var1 in enumerate(Dimuon_Variables):
    #     for var2 in Dimuon_Variables[i+1:]:
    #         comparer.compare_2D(var1, var2)


    # Plots in different regions
    single_Muon_regions = ['B', 'O', 'E']
    double_Muon_regions = ['BB', 'BO', 'BE', 'OB', 'OO', 'OE', 'EB', 'EO', 'EE']

    for region in single_Muon_regions:
        print(f"Region: {region}")
        filtered_events = {}
        # Filter events based on leading muon pseudorapidity and make comparisons
        for label, events in comparer.events.items():
            filtered_events[label] = comparer.filter_eta1(events, region)
        comparer.compare_all(LeadingMuon_Variables, events_dict=filtered_events, suffix=region)

        # Filter events based on sub-leading muon pseudorapidity and make comparisons
        for label, events in comparer.events.items():
            filtered_events[label] = comparer.filter_eta2(events, region)
        comparer.compare_all(SubleadingMuon_Variables, events_dict=filtered_events, suffix=region)

    for region in double_Muon_regions:
        print(f"Region: {region}")
        filtered_events = {}
        # Filter events based on both leading and subleading muon pseudorapidity and make comparisons
        for label, events in comparer.events.items():
            filtered_events[label] = comparer.filter_eta(events, region)
        comparer.compare_all(variables_to_plot, events_dict=filtered_events, suffix=region)

        # get 2D histograms for all combinations of variables defined in the list variables_to_plot
        for i, var1 in enumerate(Dimuon_Variables):
            for var2 in Dimuon_Variables[i+1:]:
                comparer.compare_2D(var1, var2, events_dict=filtered_events, suffix=region)
