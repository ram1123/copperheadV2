# import dask_awkward as dak

from modules.DistributionCompare import DistributionCompare

# Example main usage
if __name__ == "__main__":
    varlist_file = "varlist.yaml"

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
        "GeoFit (2018)": "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run3_nanoAODv12_BSOff/stage1_output/2022preEE/f1_0/ggh_powhegPS/*/*.parquet"
    }

    year = "2022preEE"  # or "2018"
    control_region = "z-peak"  # z-peak or "signal"
    directoryTag = f"March21_{control_region}_massResolution"
    input_paths_labels = input_paths_labels_1

    fields_to_load = [
        "wgt_nominal",
        # "nBtagLoose_nominal", "nBtagMedium_nominal",
        # "mu1_pt",  "mu1_eta", "mu1_phi",
        # "mu2_pt",  "mu2_eta", "mu2_phi",
        # "mu1_ptErr", "mu2_ptErr",
        # "dimuon_pt", "dimuon_eta", "dimuon_rapidity", "dimuon_phi",
        "dimuon_mass", "mu1_eta", "mu2_eta",
        # "dimuon_ebe_mass_res_rel",
        # "jet1_pt_nominal", "jet1_eta_nominal", "jet1_phi_nominal",
        # "jet2_pt_nominal", "jet2_eta_nominal", "jet2_phi_nominal",
        # "jj_mass_nominal", "jj_dEta_nominal",
        "event"
    ]

    comparer = DistributionCompare(year, input_paths_labels, fields_to_load, control_region, directoryTag, varlist_file)
    comparer.load_data()

    comparer.fit_dimuonInvariantMass()

    double_Muon_regions = ['BB', 'BO', 'BE', 'OB', 'OO', 'OE', 'EB', 'EO', 'EE']
    for region in double_Muon_regions:
        print(f"Region: {region}")
        filtered_events = {}
        # Filter events based on both leading and subleading muon pseudorapidity and make comparisons
        for label, events in comparer.events.items():
            filtered_events[label] = comparer.filter_eta(events, region)

        comparer.fit_dimuonInvariantMass(events_dict=filtered_events, suffix=region)
