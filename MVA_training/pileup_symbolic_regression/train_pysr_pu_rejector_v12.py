#!/usr/bin/env python3
"""
PySR PU-jet rejector training for NanoAODv12-style flat parquet columns.

- Uses your boolean label: <prefix>hasMatchedGenJet_nominal (True=HS/real, False=PU/fake)
- Trains a symbolic score S(x) with PySR (regression-to-probability y in {0,1})
- Derives regional thresholds (barrel / HE / HF) for low-pt jets (pt in [pt_min, pt_turnoff))
  at a fixed HS efficiency target (e.g. 0.98)
- Optionally train on multiple jet "prefixes" (jet1, vbf_lead_jet1, vbf_maxmjj_jet1, ...)

Outputs:
- prints best symbolic equation (sympy)
- writes thresholds to JSON
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

from pysr import PySRRegressor


# -----------------------
# Helpers
# -----------------------
def col(prefix: str, var: str, variation: str) -> str:
    # prefix like "jet1_" or "vbf_lead_jet1_"
    return f"{prefix}{var}_{variation}"

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def region_id(aeta):
    # 0 barrel, 1 HE, 2 HF
    rid = np.zeros_like(aeta, dtype=np.int32)
    rid[(aeta >= 2.5) & (aeta < 3.0)] = 1
    rid[aeta >= 3.0] = 2
    return rid

def safe_log1p(x):
    return np.log1p(np.clip(x, 0, None))


def threshold_for_hs_eff(scores_hs, hs_eff_target):
    # want P(score > thr | HS) = hs_eff_target
    # thr = quantile at (1 - hs_eff_target)
    q = 1.0 - hs_eff_target
    return float(np.quantile(scores_hs, q))


# -----------------------
# Feature building
# -----------------------
DEFAULT_FEATURE_VARS = [
    # Always include pt and abs(eta) for kinematics/region dependence
    # ("pt", "pt"),
    # ("aeta", "aeta"),

    # PU ID (discrete-ish, but useful)
    # ("puId", "puId"),

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
]

def build_df_from_prefix(df_in: pd.DataFrame, prefix: str, variation: str, jidx: int) -> pd.DataFrame:
    df = pd.DataFrame()

    pt = pd.to_numeric(df_in[col(prefix, "pt", variation)], errors="coerce").astype(np.float32).to_numpy()
    eta = pd.to_numeric(df_in[col(prefix, "eta", variation)], errors="coerce").astype(np.float32).to_numpy()
    aeta = np.abs(eta)

    y_hs = pd.to_numeric(df_in[col(prefix, "hasMatchedGenJet", variation)], errors="coerce").fillna(0).astype(bool).to_numpy().astype(np.float32)

    df["pt"] = pt
    df["aeta"] = aeta
    df["y_hs"] = y_hs

    # jet order index (0..3)
    df["jidx"] = np.full_like(pt, float(jidx), dtype=np.float32)

    for feat_name, varname in DEFAULT_FEATURE_VARS:
        if feat_name in ["pt", "aeta"]:
            continue
        cname = col(prefix, varname, variation)
        if cname in df_in.columns:
            df[feat_name] = pd.to_numeric(df_in[cname], errors="coerce").astype(np.float32)

    # derived
    if "hfadjacentEtaStripsSize" in df.columns and "hfcentralEtaStripSize" in df.columns:
        df["hf_ratio"] = df["hfadjacentEtaStripsSize"] / (df["hfcentralEtaStripSize"] + 1.0)
    if "hfsigmaEtaEta" in df.columns and "hfsigmaPhiPhi" in df.columns:
        df["hf_sigma_sum"] = df["hfsigmaEtaEta"] + df["hfsigmaPhiPhi"]

    df["logpt"] = safe_log1p(df["pt"].to_numpy())
    if "nConstituents" in df.columns:
        df["lognC"] = safe_log1p(df["nConstituents"].to_numpy())

    df["rid"] = region_id(df["aeta"].to_numpy()).astype(np.float32)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def concat_prefixes(df_in: pd.DataFrame, prefixes, variation: str) -> pd.DataFrame:
    dfs = []
    for jidx, p in enumerate(prefixes):   # jidx = 0..3
        dfs.append(build_df_from_prefix(df_in, p, variation, jidx=jidx))
    df = pd.concat(dfs, axis=0, ignore_index=True)
    # drop rows missing the core label/pt/eta
    df = df.dropna(subset=["pt", "aeta", "y_hs"])
    return df


# -----------------------
# PySR training
# -----------------------
def train_pysr(X, y, w=None, niterations=300, seed=123):
    model = PySRRegressor(
        niterations=niterations,
        population_size=200,
        maxsize=18,                      # keep formula readable
        model_selection="best",
        verbosity=1,
        random_state=seed,
        # deterministic=True,
        # procs=0,   

        # Physics-like operators
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["abs", "sqrt", "log1p", "tanh"],
        # unary_operators=["sqrt","log1p","tanh"],   # no abs        

        # Penalize divisions a bit (they can get wild)
        complexity_of_operators={"/": 3, "log1p": 2, "tanh": 2, "sqrt": 2, "abs": 1},
        # complexity_of_operators={"/": 3, "log1p": 2, "tanh": 2, "sqrt": 2},   # no abs  

        # Regression loss to 0/1 labels
        # elementwise_loss="loss(x, y) = (x - y)^2",
        elementwise_loss="loss(yhat, y) = log(1 + exp(- (2y - 1) * yhat))"
    )
    if w is None:
        model.fit(X, y)
    else:
        model.fit(X, y, weights=w)
    return model


# -----------------------
# Thresholds
# -----------------------
def region_mask_from_aeta(aeta: np.ndarray, region: str) -> np.ndarray:
    if region == "HE":
        return (aeta >= 2.5) & (aeta < 3.0)
    if region == "HF":
        return aeta >= 3.0
    raise ValueError(region)

def threshold_and_direction(hs_scores, pu_scores, hs_eff_target):
    # Option 1: keep high scores (pass if score > thr_hi)
    thr_hi = np.quantile(hs_scores, 1.0 - hs_eff_target)
    pu_rej_hi = float((pu_scores <= thr_hi).mean())

    # Option 2: keep low scores (pass if score < thr_lo)
    thr_lo = np.quantile(hs_scores, hs_eff_target)
    pu_rej_lo = float((pu_scores >= thr_lo).mean())

    if pu_rej_lo > pu_rej_hi:
        return float(thr_lo), "keep_low", pu_rej_lo
    else:
        return float(thr_hi), "keep_high", pu_rej_hi

def derive_thresholds(df: pd.DataFrame, score: np.ndarray,
                      hs_eff_target=0.98,
                      pt_min=25.0, pt_turnoff=80.0):
    pt = df["pt"].to_numpy()
    aeta = df["aeta"].to_numpy()
    y_hs = (df["y_hs"].to_numpy() > 0.5)

    lowpt = (pt >= pt_min) & (pt < pt_turnoff)
    

    out = {
        "meta": {
            "hs_eff_target": hs_eff_target,
            "pt_min": pt_min,
            "pt_turnoff": pt_turnoff,
        }
    }

    for region in ["HE", "HF"]:
        mreg = region_mask_from_aeta(aeta, region) & lowpt

        hs_scores = score[mreg & y_hs]
        pu_scores = score[mreg & (~y_hs)]

        if len(hs_scores) < 200 or len(pu_scores) < 200:
            out[region] = None
            continue

        thr, direction, pu_rej = threshold_and_direction(hs_scores, pu_scores, hs_eff_target)
        out[region] = {
            "threshold": thr,
            "direction": direction,
            "pu_rej_est": pu_rej,
            "n_hs": int(len(hs_scores)),
            "n_pu": int(len(pu_scores)),
        }

    q = 1.0 - hs_eff_target
    thr_hi = float(np.quantile(hs_scores, q))
    thr_lo = float(np.quantile(hs_scores, hs_eff_target))
    print(region, "Nhs/Npu", len(hs_scores), len(pu_scores),
        "thr_hi", thr_hi, "thr_lo", thr_lo,
        "hs_unique", len(np.unique(np.round(hs_scores, 6))))

    return out


def train_one_region(df_region: pd.DataFrame, region: str, outdir: str, args, feature_cols_base):
    """
    Train PySR on a region-sliced dataframe and write region-specific artifacts.
    """
    df = df_region.copy()

    # balance HS/PU inside the region
    hs = df[df["y_hs"] > 0.5]
    pu = df[df["y_hs"] <= 0.5]
    n = min(len(hs), len(pu))
    if n < 500:
        print(f"[WARN] Too few samples in {region}: n={n}. Skipping.")
        return None

    hs = hs.sample(n=n, random_state=1)
    pu = pu.sample(n=n, random_state=1)
    df = pd.concat([hs, pu], ignore_index=True).sample(frac=1, random_state=2).reset_index(drop=True)

    # features (start from base list prepared in main)
    feature_cols = list(feature_cols_base)

    # numeric + missing flags
    for c_ in feature_cols + ["y_hs"]:
        df[c_] = pd.to_numeric(df[c_], errors="coerce").astype(np.float32)

    miss_cols = [c_ for c_ in feature_cols if df[c_].isna().any()]
    for c_ in miss_cols:
        df[f"isnan_{c_}"] = df[c_].isna().astype(np.float32)
        df[c_] = df[c_].fillna(0.0)

    feature_cols = feature_cols + [f"isnan_{c_}" for c_ in miss_cols]

    # final cleanup
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols + ["y_hs"]).copy()

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["y_hs"].to_numpy(dtype=np.float32)

    print(f"\n[{region}] HS fraction: {float(y.mean()):.4f}  N={len(y)}  nfeat={len(feature_cols)}")

    model = train_pysr(X, y, niterations=args.niterations)

    score = model.predict(X).astype(np.float32)

    # thresholds for this region only
    thr = derive_thresholds(
        df, score,
        hs_eff_target=args.hs_eff,
        pt_min=args.pt_min,
        pt_turnoff=args.pt_turnoff
    )
    # keep only this region key in output JSON (cleaner)
    thr_region = {"meta": thr["meta"], region: thr[region]}

    # save
    eq = str(model.sympy())

    with open(os.path.join(outdir, f"best_equation_{region}.txt"), "w") as f:
        f.write(eq + "\n")

    with open(os.path.join(outdir, f"features_used_{region}.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(os.path.join(outdir, f"thresholds_{region}.json"), "w") as f:
        json.dump(thr_region, f, indent=2)

    print(f"[{region}] best eq: {eq}")
    print(f"[{region}] thr: {json.dumps(thr_region, indent=2)}")

    return {"region": region, "equation": eq, "features": feature_cols, "thresholds": thr_region}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input parquet")
    ap.add_argument("-o", "--outdir", default="pysr_pu_rejector_out")
    ap.add_argument("--variation", default="nominal")

    # pick which jet definitions to train on
    ap.add_argument("--n-jets", type=int, default=4, help="Use first N leading jets (1..4)")
    ap.add_argument("--jet-prefix-base", default="jet", help="Base prefix: jet -> jet1,jet2,...")

    # training controls
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--niterations", type=int, default=300)
    ap.add_argument("--hs-eff", type=float, default=0.98)
    ap.add_argument("--pt-min", type=float, default=25.0)
    ap.add_argument("--pt-turnoff", type=float, default=80.0)

    ap.add_argument("--structure-only", action="store_true",
                    help="If set, drop pt/aeta/logpt/rid/jidx from training features")

    args = ap.parse_args()
    ensure_dir(args.outdir)

    df_in = pd.read_parquet(args.input)
    if args.max_rows is not None:
        df_in = df_in.head(args.max_rows)

    n_jets = max(1, min(args.n_jets, 4))
    prefixes = [f"{args.jet_prefix_base}{i}" for i in range(1, n_jets + 1)]
    prefixes = [p + "_" for p in prefixes]

    # sanity check required columns exist for each prefix
    missing = []
    for p in prefixes:  # p is already "jet1_"
        for req in ["pt", "eta", "hasMatchedGenJet"]:
            cname = col(p, req, args.variation)
            # print(f"{cname}")
            if cname not in df_in.columns:
                missing.append(cname)
    if missing:
        raise KeyError("Missing required columns:\n  " + "\n  ".join(missing[:50]))

    # build training dataframe
    df = concat_prefixes(df_in, prefixes, args.variation)

    # low-pt focus (optional but recommended for your horn issue)
    # df = df[(df["pt"] >= args.pt_min) & (df["pt"] < args.pt_turnoff)].copy()
    lowpt = (df["pt"] >= args.pt_min) & (df["pt"] < args.pt_turnoff)
    hehf  = (df["aeta"] >= 2.5)  # HE+HF only
    df = df[lowpt & hehf].copy()    

    df_he = df[(df["aeta"] >= 2.5) & (df["aeta"] < 3.0)].copy()
    df_hf = df[(df["aeta"] >= 3.0)].copy()

    # df=

    # Feature list (as you already do)
    feature_cols = [c for c in df.columns if c not in ["y_hs"]]
    drop = {"hadronFlavour", "partonFlavour"}  # keep
    drop |= {"nConstituents", "lognC"}
    feature_cols = [c for c in feature_cols if c not in drop]   

    # --- (A) Make sure everything is numeric float32
    for c_ in feature_cols + ["y_hs"]:
        df[c_] = pd.to_numeric(df[c_], errors="coerce").astype(np.float32)


    SENTINELS = {
        "hfsigmaEtaEta": -1.0,
        "hfsigmaPhiPhi": -1.0,
        "hf_sigma_sum": -2.0,
    }
    for k, s in SENTINELS.items():
        if k in df.columns:
            df.loc[df[k] <= s + 1e-6, k] = np.nan


    # --- (B) Add missingness flags for columns that have NaNs
    miss_cols = [c_ for c_ in feature_cols if df[c_].isna().any()]
    for c_ in miss_cols:
        df[f"isnan_{c_}"] = df[c_].isna().astype(np.float32)
        # Fill NaNs with a neutral value (0 is OK if you have isnan_* flag)
        df[c_] = df[c_].fillna(0.0)

    # Update feature list to include missingness indicators
    feature_cols = feature_cols + [f"isnan_{c_}" for c_ in miss_cols]

    if args.structure_only:
        drop |= {"pt", "aeta", "logpt", "rid", "jidx"}
    feature_cols = [c for c in feature_cols if c not in drop]    

    # --- (C) Final safety: drop any residual inf/nan (should be none)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols + ["y_hs"]).copy()

    hs = df[df["y_hs"] > 0.5]
    pu = df[df["y_hs"] <= 0.5]
    n = min(len(hs), len(pu))

    hs = hs.sample(n=n, random_state=1)
    pu = pu.sample(n=n, random_state=1)
    df = pd.concat([hs, pu], ignore_index=True).sample(frac=1, random_state=2).reset_index(drop=True)

    # Build arrays
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["y_hs"].to_numpy(dtype=np.float32)

    print("HS fraction:", y.mean(), "N:", len(y))
    print("HS fraction:", float(y.mean()), "N:", len(y))
    for i, name in enumerate(feature_cols):
        v = X[:, i]
        print(i, name, float(np.min(v)), float(np.max(v)), float(np.mean(v)))
    # print("x18 feature:", feature_cols[18])
    # print("x20 feature:", feature_cols[20])
    # Train PySR
    model = train_pysr(X, y, niterations=args.niterations)

    # score
    score = model.predict(X).astype(np.float32)

    for reg in ["HE", "HF"]:
        m = region_mask_from_aeta(df["aeta"].to_numpy(), reg)
        yhs = df["y_hs"].to_numpy().astype(bool)
        hs_s = score[m & yhs]
        pu_s = score[m & (~yhs)]
        if len(hs_s) > 0 and len(pu_s) > 0:
            print(reg, "mean score HS/PU:", float(hs_s.mean()), float(pu_s.mean()),
                "med:", float(np.median(hs_s)), float(np.median(pu_s)))


    for reg in ["HE","HF"]:
        m = region_mask_from_aeta(df["aeta"].to_numpy(), reg)
        s = score[m]
        print(reg, "unique score values:", len(np.unique(np.round(s, 6))), "N:", len(s))

    # thresholds (HE/HF) at fixed HS efficiency
    thr = derive_thresholds(df, score,
                            hs_eff_target=args.hs_eff,
                            pt_min=args.pt_min,
                            pt_turnoff=args.pt_turnoff)

    # Save outputs
    eq = str(model.sympy())
    with open(os.path.join(args.outdir, "best_equation.txt"), "w") as f:
        f.write(eq + "\n")

    with open(os.path.join(args.outdir, "thresholds_HE_HF.json"), "w") as f:
        json.dump(thr, f, indent=2)

    with open(os.path.join(args.outdir, "features_used.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Print summary
    print("\n=== FEATURES USED ===")
    print(feature_cols)
    print("\n=== BEST EQUATION (sympy) ===")
    print(eq)
    print("\n=== THRESHOLDS (HE/HF) ===")
    print(json.dumps(thr, indent=2))


if __name__ == "__main__":
    main()