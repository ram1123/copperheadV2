import numpy as np


def pre_scaling_clean(df):
    """
    Apply physics-motivated sentinels BEFORE scaling.
    Assumes df is a pandas DataFrame.
    """

    njet = df["njets_nominal"].values  # or nJets if that is your column

    # ---- Dimuon quantities (just clean non-finite) ----
    dimuon_feats = [
        "dimuon_mass",
        "dimuon_pt",
        "dimuon_rapidity",
        "dimuon_ebe_mass_res",
        "dimuon_ebe_mass_res_rel",
        "dimuon_cos_theta_cs",
        "dimuon_phi_cs",
    ]
    for f in dimuon_feats:
        df[f] = np.where(np.isfinite(df[f]), df[f], 0.0)

    # ---- Jet-1 ----
    df["jet1_pt_nominal"] = np.where(njet >= 1, df["jet1_pt_nominal"], 0.0)
    df["jet1_eta_nominal"] = np.where(njet >= 1, df["jet1_eta_nominal"], -5.0)
    df["jet1_phi_nominal"] = np.where(njet >= 1, df["jet1_phi_nominal"], -5.0)

    # ---- Jet-2 ----
    df["jet2_pt_nominal"] = np.where(njet >= 2, df["jet2_pt_nominal"], 0.0)
    df["jet2_eta_nominal"] = np.where(njet >= 2, df["jet2_eta_nominal"], -5.0)
    df["jet2_phi_nominal"] = np.where(njet >= 2, df["jet2_phi_nominal"], -5.0)

    # ---- Dijet system ----
    df["jj_mass_nominal"] = np.where(njet >= 2, df["jj_mass_nominal"], 0.0)
    df["jj_dEta_nominal"] = np.where(njet >= 2, df["jj_dEta_nominal"], 0.0)

    # ---- Soft activity ----
    df["nsoftjets5_nominal"] = np.where(
        np.isfinite(df["nsoftjets5_nominal"]), df["nsoftjets5_nominal"], 0.0
    )
    df["htsoft2_nominal"] = np.where(
        np.isfinite(df["htsoft2_nominal"]), df["htsoft2_nominal"], 0.0
    )

    # ---- Correlation-driven observables ----
    df["rpt_nominal"] = np.where(njet >= 2, df["rpt_nominal"], 0.0)

    df["ll_zstar_log_nominal"] = np.where(
        njet >= 2, df["ll_zstar_log_nominal"], df["dimuon_rapidity"]
    )

    df["mmj_min_dEta_nominal"] = np.where(
        njet >= 1, df["mmj_min_dEta_nominal"], df["dimuon_rapidity"]
    )

    df["pt_centrality_nominal"] = np.where(
        njet >= 2, df["pt_centrality_nominal"], -200.0
    )
    # remove infinities if any
    df["pt_centrality_nominal"] = np.where(
        np.isfinite(df["pt_centrality_nominal"]),
        df["pt_centrality_nominal"],
        -200.0,
    )
    # if pt_centrality < -200, set to -200
    df["pt_centrality_nominal"] = np.where(
        df["pt_centrality_nominal"] < -200.0, -200.0, df["pt_centrality_nominal"]
    )
    df["pt_centrality_nominal"] = np.where(
        df["pt_centrality_nominal"] > 200.0, 200.0, df["pt_centrality_nominal"]
    )

    # # add remove infinities if any for all features
    # # follow the values used in pre-scaling cleaning above
    # for col in df.columns:
    #     if col in ["year", "njets_nominal"]:
    #         continue
    #     if "eta" in col or "phi" in col:
    #         sentinel = -5.0
    #     elif "pt_centrality" in col:
    #         sentinel = -20.0
    #     elif "nsoftjets5" in col:
    #         sentinel = 0.0
    #     else:
    #         sentinel = 0.0
    #     df[col] = np.where(np.isfinite(df[col]), df[col], sentinel)

    # # ---- Final global non-finite safety net ----
    # for col in df.columns:
    #     if col in ["year", "njet"]:
    #         continue
    #     df[col] = np.where(np.isfinite(df[col]), df[col], 0.0)

    return df
