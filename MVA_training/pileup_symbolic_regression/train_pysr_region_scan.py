#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pandas as pd

from pysr import PySRRegressor

"""
Example commands:
python MVA_training/pileup_symbolic_regression/train_pysr_region_scan.py \
  -i /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn30GeV_Feb23_tightPassLepVeto_NoJER_AddVars_v2/stage1_output/2022postEE/f1_0/dyTo2L_M-50_incl/0/part135.parquet \
  -o validation/pysr_region_scan_Eff98 \
  --variation nominal \
  --n-jets 4 \
  --pt-min 25 \
  --pt-turnoff 80 \
  --hs-eff 0.98 \
  --niterations 300 \
  --population-size 400 \
  --maxsize 22

python MVA_training/pileup_symbolic_regression/train_pysr_region_scan.py \
  -i /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn30GeV_Feb23_tightPassLepVeto_NoJER_AddVars_v2/stage1_output/2022postEE/f1_0/dyTo2L_M-50_incl/0/part135.parquet \
  -o validation/pysr_region_scan_Eff95 \
  --variation nominal \
  --n-jets 4 \
  --pt-min 25 \
  --pt-turnoff 80 \
  --hs-eff 0.95 \
  --niterations 300 \
  --population-size 400 \
  --maxsize 22

python MVA_training/pileup_symbolic_regression/train_pysr_region_scan.py \
  -i /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn30GeV_Feb23_tightPassLepVeto_NoJER_AddVars_v2/stage1_output/2022postEE/f1_0/dyTo2L_M-50_incl/0/part135.parquet \
  -o validation/pysr_region_scan_Eff90 \
  --variation nominal \
  --n-jets 4 \
  --pt-min 25 \
  --pt-turnoff 80 \
  --hs-eff 0.90 \
  --niterations 300 \
  --population-size 400 \
  --maxsize 22

python MVA_training/pileup_symbolic_regression/train_pysr_region_scan.py \
  -i /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn30GeV_Feb23_tightPassLepVeto_NoJER_AddVars_v2/stage1_output/2022postEE/f1_0/dyTo2L_M-50_incl/0/part135.parquet \
  -o validation/pysr_region_scan_Eff80_ValidJetsInfoOnly \
  --variation nominal \
  --n-jets 4 \
  --pt-min 25 \
  --pt-turnoff 80 \
  --hs-eff 0.80 \
  --niterations 300 \
  --population-size 400 \
  --maxsize 22 

python MVA_training/pileup_symbolic_regression/train_pysr_region_scan.py \
  -i /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn30GeV_Feb23_tightPassLepVeto_NoJER_AddVars_v2/stage1_output/2022postEE/f1_0/dyTo2L_M-50_incl/0/part135.parquet \
  -o validation/pysr_region_scan_Eff80_pTMax50 \
  --variation nominal \
  --n-jets 4 \
  --pt-min 25 \
  --pt-turnoff 50 \
  --hs-eff 0.80 \
  --niterations 300 \
  --population-size 400 \
  --maxsize 22   

python MVA_training/pileup_symbolic_regression/train_pysr_region_scan.py \
  -i /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn30GeV_Feb23_tightPassLepVeto_NoJER_AddVars_v2/stage1_output/2022postEE/f1_0/dyTo2L_M-50_incl/0/part135.parquet \
  -o validation/pysr_region_scan_Eff80_pTMax50_nJ2 \
  --variation nominal \
  --n-jets 2 \
  --pt-min 25 \
  --pt-turnoff 50 \
  --hs-eff 0.80 \
  --niterations 300 \
  --population-size 400 \
  --maxsize 22   

python MVA_training/pileup_symbolic_regression/train_pysr_region_scan.py \
  -i /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn30GeV_Feb23_tightPassLepVeto_NoJER_AddVars_v2/stage1_output/2022postEE/f1_0/dyTo2L_M-50_incl/0/part135.parquet \
  -o validation/pysr_region_scan_Eff80 \
  --variation nominal \
  --n-jets 4 \
  --pt-min 25 \
  --pt-turnoff 80 \
  --hs-eff 0.80 \
  --niterations 300 \
  --population-size 400 \
  --maxsize 22     

python MVA_training/pileup_symbolic_regression/train_pysr_region_scan.py \
  -i /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn30GeV_Feb23_tightPassLepVeto_NoJER_AddVars_v2/stage1_output/2022postEE/f1_0/dyTo2L_M-50_incl/0/part135.parquet \
  -o validation/pysr_region_scan_Eff75 \
  --variation nominal \
  --n-jets 4 \
  --pt-min 25 \
  --pt-turnoff 80 \
  --hs-eff 0.75 \
  --niterations 300 \
  --population-size 400 \
  --maxsize 22  

time python MVA_training/pileup_symbolic_regression/train_pysr_region_scan.py \
  -i /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn30GeV_Feb23_tightPassLepVeto_NoJER_AddVars_v2/stage1_output/2022postEE/f1_0/dyTo2L_M-50_incl/0/part135.parquet \
  -o validation/pysr_region_scan_Eff50 \
  --variation nominal \
  --n-jets 4 \
  --pt-min 25 \
  --pt-turnoff 80 \
  --hs-eff 0.50 \
  --niterations 300 \
  --population-size 400 \
  --maxsize 22
"""


# ============================================================
# Helpers
# ============================================================
def col(prefix: str, var: str, variation: str) -> str:
    # prefix like "jet1_" or "vbf_lead_jet1_"
    return f"{prefix}{var}_{variation}"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def region_mask_from_aeta(aeta: np.ndarray, region: str) -> np.ndarray:
    if region == "HE":
        return (aeta >= 2.5) & (aeta < 3.0)
    if region == "HF":
        return aeta >= 3.0
    raise ValueError(region)


def safe_log1p(x):
    return np.log1p(np.clip(x, 0, None))


def region_id(aeta):
    # 0 barrel, 1 HE, 2 HF
    rid = np.zeros_like(aeta, dtype=np.int32)
    rid[(aeta >= 2.5) & (aeta < 3.0)] = 1
    rid[aeta >= 3.0] = 2
    return rid


def threshold_and_direction(hs_scores, pu_scores, hs_eff_target):
    if len(hs_scores) == 0 or len(pu_scores) == 0:
        return None, None, None

    # keep high
    thr_hi = np.quantile(hs_scores, 1.0 - hs_eff_target)
    pu_rej_hi = float((pu_scores < thr_hi).mean())

    # keep low
    thr_lo = np.quantile(hs_scores, hs_eff_target)
    pu_rej_lo = float((pu_scores > thr_lo).mean())

    if pu_rej_lo > pu_rej_hi:
        return float(thr_lo), "keep_low", pu_rej_lo
    else:
        return float(thr_hi), "keep_high", pu_rej_hi


def derive_thresholds(
    df: pd.DataFrame,
    score: np.ndarray,
    hs_eff_target=0.98,
    pt_min=25.0,
    pt_turnoff=80.0,
    regions=("HE", "HF"),
):
    pt = df["pt"].to_numpy()
    aeta = df["aeta"].to_numpy()
    y_hs = df["y_hs"].to_numpy() > 0.5

    lowpt = (pt >= pt_min) & (pt < pt_turnoff)

    out = {
        "meta": {
            "hs_eff_target": hs_eff_target,
            "pt_min": pt_min,
            "pt_turnoff": pt_turnoff,
        }
    }

    for region in regions:
        mreg = region_mask_from_aeta(aeta, region) & lowpt

        hs_scores = score[mreg & y_hs]
        pu_scores = score[mreg & (~y_hs)]

        if hs_scores.size < 200 or pu_scores.size < 200:
            out[region] = None
            continue

        thr, direction, pu_rej = threshold_and_direction(
            hs_scores, pu_scores, hs_eff_target
        )
        out[region] = {
            "threshold": thr,
            "direction": direction,
            "pu_rej_est": pu_rej,
            "n_hs": int(hs_scores.size),
            "n_pu": int(pu_scores.size),
        }

    return out


# ============================================================
# Feature sets
# ============================================================
HE_FEATURE_SETS = {
    "HE_comp": [
        "chEmEF", "chHEF", "neEmEF", "neHEF", "muEF",
    ],
    "HE_comp_occ": [
        "chEmEF", "chHEF", "neEmEF", "neHEF", "muEF",
        "nElectrons", "nMuons",
        "nSVs", "nConstituents",
        "lognC",
    ],
    # "HE_comp_occ_pt": [
    #     "chEmEF", "chHEF", "neEmEF", "neHEF", "muEF",
    #     "nElectrons", "nMuons",
    #     "nSVs", "nConstituents",
    #     "lognC",
    #     "pt", "logpt",
    # ],
    "HE_comp_v2": [
        "chEmEF", "chHEF", "neEmEF", "neHEF", "muEF",
        "nConstituents",
    ],    
}

HF_FEATURE_SETS = {
    "HF_core": [
        "chEmEF", "chHEF", "neEmEF", "neHEF", "muEF",
        "hfcentralEtaStripSize", "hfadjacentEtaStripsSize",
        "hfsigmaEtaEta", "hfsigmaPhiPhi",
        "hf_ratio", "hf_sigma_sum",
    ],
    "HF_core_occ": [
        "chEmEF", "chHEF", "neEmEF", "neHEF", "muEF",
        "hfcentralEtaStripSize", "hfadjacentEtaStripsSize",
        "hfsigmaEtaEta", "hfsigmaPhiPhi",
        "hf_ratio", "hf_sigma_sum",
        "nElectrons", "nMuons", "nSVs",
    ],
    # "HF_core_occ_nan": [
    #     "chEmEF", "chHEF", "neEmEF", "neHEF", "muEF",
    #     "hfcentralEtaStripSize", "hfadjacentEtaStripsSize",
    #     "hfsigmaEtaEta", "hfsigmaPhiPhi",
    #     "hf_ratio", "hf_sigma_sum",
    #     "nElectrons", "nMuons", "nSVs",
    #     "isnan_hfsigmaEtaEta", "isnan_hfsigmaPhiPhi", "isnan_hf_sigma_sum",
    # ],
    # "HF_v2": ["isnan_hfsigmaEtaEta", "isnan_hfsigmaPhiPhi", "isnan_hf_sigma_sum"],
    # "HF_v3": [
    #     "chEmEF", "chHEF", "neEmEF", "neHEF", "muEF",
    #     "hfcentralEtaStripSize", "hfadjacentEtaStripsSize",
    #     "hfsigmaEtaEta", "hfsigmaPhiPhi",
    #     "hf_ratio", "hf_sigma_sum",
    #     "pt", "logpt",
    # ],
    "HF_v4": [
        "hfcentralEtaStripSize", "hfadjacentEtaStripsSize", 
        "hfsigmaEtaEta", "hfsigmaPhiPhi", 
        "hf_ratio", "hf_sigma_sum",
    ],
    # "HF_v5": [
    #     "hfcentralEtaStripSize", "hfadjacentEtaStripsSize", 
    #     "hfsigmaEtaEta", "hfsigmaPhiPhi", 
    #     "hf_ratio", "hf_sigma_sum",
    #     "pt", "logpt",
    # ],
}

REGION_PLANS = {
    "HE": HE_FEATURE_SETS,
    "HF": HF_FEATURE_SETS,
}


# ============================================================
# Build flat jet rows
# ============================================================
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
]


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

    for feat_name, varname in DEFAULT_FEATURE_VARS:
        cname = col(prefix, varname, variation)
        if cname in df_in.columns:
            df[feat_name] = pd.to_numeric(df_in[cname], errors="coerce").astype(
                np.float32
            )

    # derived
    if (
        "hfadjacentEtaStripsSize" in df.columns
        and "hfcentralEtaStripSize" in df.columns
    ):
        df["hf_ratio"] = df["hfadjacentEtaStripsSize"] / (
            df["hfcentralEtaStripSize"] + 1.0
        )

    if "hfsigmaEtaEta" in df.columns and "hfsigmaPhiPhi" in df.columns:
        df["hf_sigma_sum"] = df["hfsigmaEtaEta"] + df["hfsigmaPhiPhi"]

    if "nConstituents" in df.columns:
        df["lognC"] = safe_log1p(df["nConstituents"].to_numpy())

    df["logpt"] = safe_log1p(df["pt"].to_numpy())
    df["rid"] = region_id(df["aeta"].to_numpy()).astype(np.float32)

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


# ============================================================
# Region preprocessing
# ============================================================
def select_region(df, region, pt_min=25.0, pt_turnoff=80.0):
    lowpt = (df["pt"] >= pt_min) & (df["pt"] < pt_turnoff)
    if region == "HE":
        reg = (df["aeta"] >= 2.5) & (df["aeta"] < 3.0)
    elif region == "HF":
        reg = df["aeta"] >= 3.0
    else:
        raise ValueError(region)
    return df[lowpt & reg].copy()


def apply_sentinel_cleanup(df):
    sentinels = {
        "hfsigmaEtaEta": -1.0,
        "hfsigmaPhiPhi": -1.0,
        "hf_sigma_sum": -2.0,
    }
    for key, sval in sentinels.items():
        if key in df.columns:
            df.loc[df[key] <= sval + 1e-6, key] = np.nan
    return df


def balance_hs_pu(df, seed=1):
    hs = df[df["y_hs"] > 0.5]
    pu = df[df["y_hs"] <= 0.5]
    n = min(len(hs), len(pu))
    if n == 0:
        return None
    hs = hs.sample(n=n, random_state=seed)
    pu = pu.sample(n=n, random_state=seed)
    return (
        pd.concat([hs, pu], ignore_index=True)
        .sample(frac=1, random_state=seed + 1)
        .reset_index(drop=True)
    )


def finalize_feature_columns(df, requested_features):
    df = df.copy()

    # create requested isnan_* flags
    for f in requested_features:
        if f.startswith("isnan_"):
            base = f.replace("isnan_", "")
            if base in df.columns:
                df[f] = df[base].isna().astype(np.float32)

    feature_cols = [f for f in requested_features if f in df.columns]

    # numeric cast
    for f in feature_cols:
        df[f] = pd.to_numeric(df[f], errors="coerce").astype(np.float32)

    return df, feature_cols


# ============================================================
# PySR
# ============================================================
def train_pysr(
    X,
    y,
    niterations=300,
    population_size=400,
    maxsize=22,
    seed=123,
    elementwise_loss="loss(yhat, y) = log(1 + exp(- (2y - 1) * yhat))",
):
    model = PySRRegressor(
        niterations=niterations,
        population_size=population_size,
        maxsize=maxsize,
        model_selection="best",
        verbosity=1,
        random_state=seed,
        deterministic=True,
        parallelism="multiprocessing",
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["abs", "sqrt", "log1p", "tanh"],
        complexity_of_operators={"/": 3, "log1p": 2, "tanh": 2, "sqrt": 2, "abs": 1},
        elementwise_loss=elementwise_loss,
    )
    model.fit(X, y)
    return model


# ============================================================
# Scan runner
# ============================================================
def train_region_feature_scan(df_all, outdir, args):
    summary = []

    for region, feature_sets in REGION_PLANS.items():
        df_reg = select_region(df_all, region, args.pt_min, args.pt_turnoff)
        df_reg = apply_sentinel_cleanup(df_reg)

        for tag, requested_features in feature_sets.items():
            work = df_reg.copy()
            work = balance_hs_pu(work)
            if work is None or len(work) < args.min_train:
                print(f"[WARN] Skip {region}/{tag}: too few balanced events")
                continue

            work, feature_cols = finalize_feature_columns(work, requested_features)
            if len(feature_cols) == 0:
                print(f"[WARN] Skip {region}/{tag}: empty feature list after filtering")
                continue

            work = work.replace([np.inf, -np.inf], np.nan).copy()
            work = work.dropna(subset=["y_hs"] + feature_cols).copy()            

            X = work[feature_cols].to_numpy(dtype=np.float32)
            y = work["y_hs"].to_numpy(dtype=np.float32)

            region_dir = os.path.join(outdir, region, tag)
            ensure_dir(region_dir)

            print(
                f"\n[{region}/{tag}] N={len(y)} nfeat={len(feature_cols)} HSfrac={y.mean():.3f}"
            )
            print("Feature index map:")
            for i, f in enumerate(feature_cols):
                v = X[:, i]
                print(
                    f"  x{i}: {f}  min={float(np.min(v)):.4g} max={float(np.max(v)):.4g} mean={float(np.mean(v)):.4g}"
                )

            model = train_pysr(
                X,
                y,
                niterations=args.niterations,
                population_size=args.population_size,
                maxsize=args.maxsize,
                seed=args.seed,
            )

            print(f"Model equation: {model.equations}")

            # In your train_pysr function or immediately after fit
            best_row = model.get_best()
            print(f"Selected Complexity: {best_row['complexity']}, Loss: {best_row['loss']}")

            score = model.predict(X).astype(np.float32)

            thr = derive_thresholds(
                work,
                score,
                hs_eff_target=args.hs_eff,
                pt_min=args.pt_min,
                pt_turnoff=args.pt_turnoff,
                regions=(region,),
            )

            eq = str(model.sympy())

            with open(
                os.path.join(region_dir, f"best_equation_{region}.txt"), "w"
            ) as f:
                f.write(eq + "\n")

            with open(
                os.path.join(region_dir, f"features_used_{region}.json"), "w"
            ) as f:
                json.dump(feature_cols, f, indent=2)

            with open(os.path.join(region_dir, f"thresholds_{region}.json"), "w") as f:
                json.dump(thr, f, indent=2)

            rec = {
                "region": region,
                "tag": tag,
                "equation": eq,
                "n_train": int(len(y)),
                "n_features": int(len(feature_cols)),
                "feature_cols": feature_cols,
                "threshold_info": thr.get(region, None),
            }
            summary.append(rec)

            print(f"[{region}/{tag}] best eq: {eq}")
            print(f"[{region}/{tag}] threshold info: {json.dumps(thr, indent=2)}")

    with open(os.path.join(outdir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input parquet")
    ap.add_argument("-o", "--outdir", default="validation/pysr_region_scan")
    ap.add_argument("--variation", default="nominal")
    ap.add_argument("--n-jets", type=int, default=4)
    ap.add_argument("--jet-prefix-base", default="jet")

    ap.add_argument("--pt-min", type=float, default=25.0)
    ap.add_argument("--pt-turnoff", type=float, default=80.0)
    ap.add_argument("--hs-eff", type=float, default=0.98)

    ap.add_argument("--niterations", type=int, default=300)
    ap.add_argument("--population-size", type=int, default=400)
    ap.add_argument("--maxsize", type=int, default=22)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--max-rows", type=int, default=None)

    args = ap.parse_args()
    ensure_dir(args.outdir)

    df_in = pd.read_parquet(args.input)
    if args.max_rows is not None:
        df_in = df_in.head(args.max_rows)

    n_jets = max(1, min(args.n_jets, 4))
    prefixes = [f"{args.jet_prefix_base}{i}_" for i in range(1, n_jets + 1)]

    missing = []
    for p in prefixes:
        for req in ["pt", "eta", "hasMatchedGenJet"]:
            cname = col(p, req, args.variation)
            if cname not in df_in.columns:
                missing.append(cname)
    if missing:
        raise KeyError("Missing required columns:\n  " + "\n  ".join(missing[:50]))

    df_all = concat_prefixes(df_in, prefixes, args.variation)

    summary = train_region_feature_scan(df_all, args.outdir, args)

    print("\n================ SUMMARY ================\n")
    for rec in summary:
        reg = rec["region"]
        tag = rec["tag"]
        thr = rec["threshold_info"]
        pu_rej = None if thr is None else thr.get("pu_rej_est", None)
        print(
            f"{reg:>2} | {tag:<18} | n={rec['n_train']:>6} | nfeat={rec['n_features']:>2} | PUrej@HS{int(args.hs_eff*100)} = {pu_rej}"
        )


if __name__ == "__main__":
    main()
