import numpy as np
import pandas as pd

DEFAULT_FEATURE_VARS = [
    # Energy fractions
    ("chEmEF", "chEmEF"),
    ("chHEF", "chHEF"),
    ("neEmEF", "neEmEF"),
    ("neHEF", "neHEF"),
    ("muEF", "muEF"),

    # Constituents
    ("nConstituents", "nConstituents"),
    ("nElectrons", "nElectrons"),
    ("nMuons", "nMuons"),
    ("nSVs", "nSVs"),

    # HF noise vars
    ("hfcentralEtaStripSize", "hfcentralEtaStripSize"),
    ("hfadjacentEtaStripsSize", "hfadjacentEtaStripsSize"),
    ("hfsigmaEtaEta", "hfsigmaEtaEta"),
    ("hfsigmaPhiPhi", "hfsigmaPhiPhi"),

    # Additional vars
    ("mass", "mass"),
    ("area", "area"),
    ("rawFactor", "rawFactor"),
    ("muonSubtrFactor", "muonSubtrFactor"),

    # # Additional vars: Only available in v15
    # ("chMultiplicity", "chMultiplicity"),
    # ("neMultiplicity", "neMultiplicity"),
    # ("hfEmEF", "hfEmEF"),
    # ("hfHEF", "hfHEF"),
    # ("muonSubtrDeltaEta", "muonSubtrDeltaEta"),
    # ("muonSubtrDeltaPhi", "muonSubtrDeltaPhi"),
    # ("puId", "puId"),
]

FEATURE_TO_RAW_VAR = {feat_name: varname for feat_name, varname in DEFAULT_FEATURE_VARS}
DERIVED_FEATURE_REQUIREMENTS = {
    "logpt": ["pt"],
    "lognC": ["nConstituents"],
    "hf_sigma_sum": ["hfsigmaEtaEta", "hfsigmaPhiPhi"],
    "hf_strip_sum": ["hfcentralEtaStripSize", "hfadjacentEtaStripsSize"],
    "hf_ratio": ["hfsigmaEtaEta", "hfsigmaPhiPhi"],
    # "hf_energy_asym": ["hfEmEF", "hfHEF"],
    "hf_strip_ratio": ["hfcentralEtaStripSize", "hfadjacentEtaStripsSize"],
}


def col(prefix: str, var: str, variation: str) -> str:
    # prefix like "jet1_" or "vbf_lead_jet1_"
    return f"{prefix}{var}_{variation}"


def required_parquet_columns(requested_features, prefixes, variation: str):
    needed_raw = {"pt", "eta", "hasMatchedGenJet"}
    for feature in requested_features:
        if feature in FEATURE_TO_RAW_VAR:
            needed_raw.add(FEATURE_TO_RAW_VAR[feature])
        if feature in DERIVED_FEATURE_REQUIREMENTS:
            needed_raw.update(DERIVED_FEATURE_REQUIREMENTS[feature])

    columns = []
    for prefix in prefixes:
        for raw_var in sorted(needed_raw):
            columns.append(col(prefix, raw_var, variation))
    return columns

def build_df_from_prefix(
    df_in: pd.DataFrame, prefix: str, variation: str, jidx: int
) -> pd.DataFrame:
    df = pd.DataFrame()

    pt = (
        pd.to_numeric(df_in[col(prefix, "pt", variation)], errors="coerce")
        .astype(np.float32)
        .to_numpy()
    )
    eta = (
        pd.to_numeric(df_in[col(prefix, "eta", variation)], errors="coerce")
        .astype(np.float32)
        .to_numpy()
    )

    y_hs_raw = (
        pd.to_numeric(df_in[col(prefix, "hasMatchedGenJet", variation)], errors="coerce")
        .to_numpy()
    )

    # --------------------------------------------------
    # Jet existence mask:
    # keep only jets with valid kinematics and pt > 0
    # --------------------------------------------------
    exists = np.isfinite(pt) & np.isfinite(eta) & (pt > 0)

    # if label is missing for non-existing jets, this also protects you
    good_label = exists & np.isfinite(y_hs_raw)
    y_hs = np.full_like(pt, np.nan, dtype=np.float32)
    y_hs[good_label] = y_hs_raw[good_label].astype(bool).astype(np.float32)    

    aeta = np.abs(eta)

    df["pt"] = pt
    df["eta"] = eta
    df["aeta"] = aeta
    df["y_hs"] = y_hs

    # jet order index (0..3)
    df["jidx"] = np.full_like(pt, float(jidx), dtype=np.float32)

    if "__sample_name" in df_in.columns:
        df["sample_name"] = df_in["__sample_name"].to_numpy()
    if "__process_group" in df_in.columns:
        df["process_group"] = df_in["__process_group"].to_numpy()

    for feat_name, varname in DEFAULT_FEATURE_VARS:
        cname = col(prefix, varname, variation)
        if cname in df_in.columns:
            df[feat_name] = pd.to_numeric(df_in[cname], errors="coerce").astype(
                np.float32
            )

    df = df.replace([np.inf, -np.inf], np.nan)

    # --------------------------------------------------
    # Drop non-existing jets here
    # --------------------------------------------------
    df = df[exists].copy()

    return df



def concat_prefixes(df_in: pd.DataFrame, prefixes, variation: str) -> pd.DataFrame:
    dfs = []
    for jidx, p in enumerate(prefixes):   # jidx = 0..3
        dfs.append(build_df_from_prefix(df_in, p, variation, jidx=jidx))
    df = pd.concat(dfs, axis=0, ignore_index=True)
    # drop rows missing the core label/pt/eta
    df = df.dropna(subset=["pt", "aeta", "y_hs"])
    return df


def derive_features(df):
    df = df.copy()

    if "pt" in df:
        df["logpt"] = np.log(df["pt"] + 1.0)

    if "nConstituents" in df:
        df["lognC"] = np.log(df["nConstituents"] + 1.0)

    if {"hfsigmaEtaEta", "hfsigmaPhiPhi"} <= set(df.columns):
        df["hf_sigma_sum"] = df["hfsigmaEtaEta"] + df["hfsigmaPhiPhi"]

    if {"hfcentralEtaStripSize","hfadjacentEtaStripsSize"} <= set(df.columns):
        df["hf_strip_sum"] = (
            df["hfcentralEtaStripSize"] +
            df["hfadjacentEtaStripsSize"]
        )

    if {"hfsigmaEtaEta", "hfsigmaPhiPhi"} <= set(df.columns):
        denom = df["hfsigmaEtaEta"] + df["hfsigmaPhiPhi"] + 1e-6
        df["hf_ratio"] = df["hfsigmaEtaEta"] / denom

    # df["hf_energy_asym"] = df["hfEmEF"] - df["hfHEF"]
    
    df["hf_strip_ratio"] = np.where(
        df["hfadjacentEtaStripsSize"] > 0,
        df["hfcentralEtaStripSize"] / df["hfadjacentEtaStripsSize"],
        0.0
    )

    return df


def drop_constant_columns(df, cols):
    keep = []
    for c in cols:
        if c in df and np.nanstd(df[c]) > 0:
            keep.append(c)
    return keep


def cleanup_sentinels(df: "pd.DataFrame") -> "pd.DataFrame":
    """Replace sentinel HF noise values with NaN for training stability."""
    pass
