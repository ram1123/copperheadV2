"""If you want to skip/run certain samples while running stage1, list them here."""

samples_to_run = None  # if not None, only run these samples (overrides samples_to_skip)

# samples_to_run = [
#     # "dyTo2L_M-50_incl",
#     "vbf_powheg_dipole",
#     "data_B",
#     # "data_C",
#     # "data_D",
#     # "data_E",
#     # "data_F",
# ]

# samples_to_skip = None  # if not None, skip these samples

samples_to_skip = [
    # "data_A",
    # "data_B",
    # "data_C",
    # "data_D",
    # "dy_M-50_aMCatNLO",
    # "dy_M-100To200_aMCatNLO",
    # "ttjets_dl",
    # "dy_M-100To200_MiNNLO",
    # "dy_M-50_MiNNLO",
    # "dy_VBF_filter",
    "dyTo2L_M-50_0j",
    "dyTo2L_M-50_1j",
    "dyTo2L_M-50_2j",
    "dyTo2Mu_MLL_10To50",
    "dyTo2Mu_MLL_50To120",
    "dyTo2Mu_MLL_120To200",
    "dyTo2Mu_M-105To160",
    "ewk_lljj",
    "ewk_lljj_mll50_mjj120",
    # "ewk_mmjj_mll_105_160",
    # "ewk_mmjj_mll_105_160_mjj120",
    "ewk_lljj",
    "ttjets_fh",
    "tt_inclusive",
    "st_tw_top",
    "st_tw_antitop",
    "ggh_amcatnlo",
    "vbf_aMCatNLO",
]
