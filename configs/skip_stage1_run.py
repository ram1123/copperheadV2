"""Allow/deny list of samples for stage-1 (consumed by run_stage1.should_process_dataset).

Precedence:
  1. --sync runs ignore both lists.
  2. samples_to_run non-empty -> ONLY these samples run (samples_to_skip is ignored).
  3. samples_to_skip non-empty -> these samples are skipped. Only applied with --skipSamples,
     which run_analysis_pipeline.sh / Snakemake ALWAYS pass (common_workflow.sh build_stage1_cmd).
  4. Otherwise every sample in the dataset YAML runs.

Names must match the sample keys in the dataset YAML (configs/datasets/*.yaml); a misspelled name
is silently ignored. Revert both lists to empty before a full production run.
"""

# Empty -> no allow-list.
samples_to_run = [
    # "vbf_powheg_dipole",
    # "dyTo2L_M-50_incl",
    # "dyTo2Mu_M-50_aMCatNLO",
    # "dyTo2L_M-50_aMCatNLO",
    # "ttjets_dl",
    # "data_B",
    # "data_C",
    # "data_D",
    # "data_E",
    # "data_F",
    # "data_H",
]

# Empty -> skip nothing.
samples_to_skip = [
    # --- data ---
    # "data_A",
    # "data_B",
    # "data_C",
    # "data_D",
    # "data_E",
    # "data_F",
    # "data_G",
    # "data_H",
    # "data_I",
    # --- DY ---
    # "dy_M-50_MiNNLO",
    # "dy_M-100To200_MiNNLO",
    # "dy_M-100To200_aMCatNLO",
    # "dy_VBF_filter",
    "dy_M-50_aMCatNLO",
    "dyTo2L_M-50_0j",
    "dyTo2L_M-50_1j",
    "dyTo2L_M-50_2j",
    "dyTo2Mu_MLL_10To50",
    "dyTo2Mu_MLL_50To120",
    "dyTo2Mu_MLL_120To200",
    "dyTo2Mu_M-105To160",
    # --- EWK Z ---
    # "ewk_mmjj_mll_105_160",
    "ewk_lljj",
    "ewk_lljj_mll50_mjj120",
    "ewk_mmjj_mll_105_160_mjj120",
    # --- top ---
    # "ttjets_dl",
    "ttjets_fh",
    "tt_inclusive",
    "st_tw_top",
    "st_tw_antitop",
    # --- diboson ---
    "zz",
    # --- signal ---
    # "vbf_aMCatNLO",
    "ggh_amcatnlo",
]
