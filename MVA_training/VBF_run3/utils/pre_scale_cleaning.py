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


def _transformer_feature_sentinel(feature):
    """Return the raw-space Transformer missing-value sentinel for one feature.

    The sentinel is filled *before* the existing train-fold mean/std
    standardisation.  It is deliberately outside the physical domain whenever
    the observable has a bounded or non-negative domain; no clipping is applied
    to the resulting standardised value.
    """

    name = feature.lower()
    if "upart" in name and "qvg" in name:
        return -1.0
    if name.startswith(("jet1_", "jet2_")) and (
        "eta" in name or "phi" in name
    ):
        return -5.0
    if name == "rpt_nominal":
        return -1.0
    if name == "ll_zstar_log_nominal":
        return -20.0
    if name in {"jj_deta_nominal", "mmj_min_deta_nominal"}:
        return -1.0
    if "pt_centrality" in name:
        return -200.0
    return 0.0


def pre_scaling_clean_transformer(df, training_features=None):
    """Clean the inclusive 0/1/2+ jet phase space used by Transformer selection.

    This is intentionally separate from :func:`pre_scaling_clean`: the legacy
    cut-based training path keeps its historical cleaning byte-for-byte, while
    this function implements the sentinel contract needed when 0- and 1-jet
    events are admitted by a Transformer score.
    """

    if "njets_nominal" not in df.columns:
        raise KeyError("Transformer cleaning requires 'njets_nominal'.")

    features = list(training_features or df.columns)
    njet = np.asarray(df["njets_nominal"], dtype=np.float64)
    njet = np.where(np.isfinite(njet), njet, 0.0)
    df["njets_nominal"] = njet

    for feature in features:
        if feature not in df.columns or feature.startswith("year_"):
            continue

        values = np.asarray(df[feature], dtype=np.float64)
        sentinel = _transformer_feature_sentinel(feature)

        if feature.startswith("jet1_"):
            values = np.where((njet >= 1) & np.isfinite(values), values, sentinel)
        elif feature.startswith("jet2_"):
            values = np.where((njet >= 2) & np.isfinite(values), values, sentinel)
        elif feature.startswith("jj_") or feature in {
            "rpt_nominal",
            "ll_zstar_log_nominal",
        }:
            values = np.where(
                (njet >= 2) & np.isfinite(values), values, sentinel
            )
        elif feature == "mmj_min_dEta_nominal":
            values = np.where(
                (njet >= 1) & np.isfinite(values), values, sentinel
            )
        elif feature == "pt_centrality_nominal":
            values = np.where(
                (njet >= 2) & np.isfinite(values), values, -200.0
            )
            # This is raw-observable protection inherited from the historical
            # cleaner, not clipping of a standardised DNN input.
            values = np.clip(values, -200.0, 200.0)
        else:
            values = np.where(np.isfinite(values), values, sentinel)

        df[feature] = values

    # Assert the contract at the preprocessing boundary, before means/stds can
    # turn an unnoticed NaN/Inf into invalid scaler and model artifacts.
    remaining = [
        feature
        for feature in features
        if feature in df.columns
        and not np.isfinite(np.asarray(df[feature], dtype=np.float64)).all()
    ]
    if remaining:
        raise ValueError(
            "Transformer sentinel cleaning left non-finite training features: "
            f"{remaining}"
        )

    return df
