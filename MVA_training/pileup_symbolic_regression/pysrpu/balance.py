import numpy as np
import pandas as pd

def balance_hs_pu(df, seed=1, min_train=500):
    hs = df[df["y_hs"] > 0.5]
    pu = df[df["y_hs"] <= 0.5]
    n = min(len(hs), len(pu))
    if n < min_train:
        return None
    hs = hs.sample(n=n, random_state=seed)
    pu = pu.sample(n=n, random_state=seed+1)
    return pd.concat([hs, pu], ignore_index=True).sample(frac=1, random_state=seed+2)

def finalize_feature_columns(df, requested):
    df = df.copy()
    for f in requested:
        if f.startswith("isnan_"):
            base = f.replace("isnan_", "")
            if base in df.columns:
                df[f] = df[base].isna().astype(np.float32)
    cols = [f for f in requested if f in df.columns]
    # drop constant columns per-region to avoid trivial formulas
    keep = []
    for c in cols:
        v = df[c].to_numpy()
        if np.nanstd(v) > 0:
            keep.append(c)
    return df.dropna(subset=["y_hs"] + keep), keep