# import dask_awkward as dak

from DistributionCompare import DistributionCompare

# Example main usage
if __name__ == "__main__":
    year = "2022preEE"  # or "2018"
    control_region = "z-peak"  # or "signal"
    directoryTag = "March20"
    varlist_file = "varlist.yaml"

    input_paths_labels = {
        "BSC ON": "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run3_nanoAODv12_BSOn_UpdateMassCalib/stage1_output/2022preEE/f1_0/data_*/*/*.parquet",
        "BSC OFF": "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_12March_NoGeoNoBSC//stage1_output/2022preEE/f1_0/data_*/*/*.parquet",
        # "GeoFit": "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run3_nanoAODv12_BSOff/stage1_output/2022preEE/f1_0/data_*/*/*.parquet",

    }

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
    comparer.compare_all(variables_to_plot)
    exit()

    # comparer.compare_2D("mu1_pt", "mu2_pt")

    # Get 2D histograms for all combinations of leading  muon variables
    for i, var1 in enumerate(LeadingMuon_Variables):
        for var2 in LeadingMuon_Variables[i+1:]:
            comparer.compare_2D(var1, var2)

    # Get 2D histograms for all combinations of subleading  muon variables
    for i, var1 in enumerate(SubleadingMuon_Variables):
        for var2 in SubleadingMuon_Variables[i+1:]:
            comparer.compare_2D(var1, var2)

    # Get 2D histograms for all combinations of leading and subleading muon variables
    for var1 in LeadingMuon_Variables:
        for var2 in SubleadingMuon_Variables:
            comparer.compare_2D(var1, var2)

    # Get 2D histograms for all combinations of leading and dimuon variables
    for var1 in LeadingMuon_Variables:
        for var2 in Dimuon_Variables:
            comparer.compare_2D(var1, var2)
    # Get 2D histograms for all combinations of subleading and dimuon variables
    for var1 in SubleadingMuon_Variables:
        for var2 in Dimuon_Variables:
            comparer.compare_2D(var1, var2)

    # Get 2D histograms for all combinations of dimuon variables
    for i, var1 in enumerate(Dimuon_Variables):
        for var2 in Dimuon_Variables[i+1:]:
            comparer.compare_2D(var1, var2)
