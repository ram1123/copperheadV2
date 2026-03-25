#!/usr/bin/env python3
import os
import json
import shutil
import argparse
import numpy as np
import pandas as pd

"""
python MVA_training/pileup_symbolic_regression/rescan_pysr_thresholds.py \
  -i /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn_Mar19_tightPassLepVeto_NoJER_pySR/stage1_output/2022postEE/compacted/dyTo2L_M-50_incl/0/part038.parquet \
  --model-dir validation/pysr_best \
  --outdir validation/pysr_best_threshold_scan_decorrelate_pT \
  --suffix _EffScan \
  --variation nominal \
  --n-jets 4 \
  --pt-min 25 \
  --pt-turnoff 80 \
  --hs-eff-list 0.98 0.95 0.90 0.85 0.80 0.75 0.50 \
  --use-he-eta-bins 
"""


def col(prefix: str, var: str, variation: str) -> str:
    return f"{prefix}{var}_{variation}"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def region_mask(aeta: np.ndarray, region: str) -> np.ndarray:
    if region == "HE":
        return (aeta >= 2.5) & (aeta < 3.0)
    if region == "HF":
        return aeta >= 3.0
    raise ValueError(region)


def safe_log1p(x):
    return np.log1p(np.clip(x, 0, None))


def infer_region_from_aeta(aeta: np.ndarray) -> np.ndarray:
    rid = np.zeros_like(aeta, dtype=np.int32)
    rid[(aeta >= 2.5) & (aeta < 3.0)] = 1
    rid[aeta >= 3.0] = 2
    return rid


def compute_score_from_sympy_x(expr_str: str, X: np.ndarray) -> np.ndarray:
    loc = {
        "np": np,
        "abs": np.abs,
        "Abs": np.abs,
        "sqrt": np.sqrt,
        "tanh": np.tanh,
        "log1p": safe_log1p,
        "log": lambda x: np.log(np.clip(x, 1e-6, None)),
        "sqrt_abs": lambda x: np.sqrt(np.abs(x)),
        "log1p_abs": lambda x: np.log1p(np.abs(x)),
    }
    for i in range(X.shape[1]):
        loc[f"x{i}"] = X[:, i]

    out = eval(expr_str, {"__builtins__": {}}, loc)
    out = np.asarray(out, dtype=np.float32)
    if out.ndim == 0:
        out = np.full(X.shape[0], float(out), dtype=np.float32)
    return out


def build_jet_rows(df_in: pd.DataFrame, prefixes, variation: str):
    dfs = []
    df_in = df_in.reset_index(drop=True)

    for jidx, p in enumerate(prefixes):
        need = [
            col(p, "pt", variation),
            col(p, "eta", variation),
            col(p, "hasMatchedGenJet", variation),
        ]
        if any(c not in df_in.columns for c in need):
            continue

        pt = pd.to_numeric(df_in[col(p, "pt", variation)], errors="coerce").astype(np.float32).to_numpy()
        eta = pd.to_numeric(df_in[col(p, "eta", variation)], errors="coerce").astype(np.float32).to_numpy()
        y_hs_raw = pd.to_numeric(
            df_in[col(p, "hasMatchedGenJet", variation)], errors="coerce"
        ).to_numpy()

        exists = np.isfinite(pt) & np.isfinite(eta) & (pt > 0)
        good_label = exists & np.isfinite(y_hs_raw)

        y_hs = np.full_like(pt, np.nan, dtype=np.float32)
        y_hs[good_label] = y_hs_raw[good_label].astype(bool).astype(np.float32)

        out = pd.DataFrame({
            "evt_idx": np.arange(len(df_in)),
            "pt": pt,
            "eta": eta,
            "aeta": np.abs(eta),
            "y_hs": y_hs,
            "jidx": np.full_like(pt, jidx, dtype=np.float32),
            "prefix": p,
        })

        dfs.append(out[exists].copy())

    if not dfs:
        return pd.DataFrame()

    dfj = pd.concat(dfs, ignore_index=True)
    dfj = dfj.replace([np.inf, -np.inf], np.nan)
    dfj = dfj.dropna(subset=["pt", "eta", "aeta", "y_hs"])
    return dfj


def build_feature_matrix(df_in: pd.DataFrame, dfj: pd.DataFrame, feature_cols, variation: str):
    evt_idx = dfj["evt_idx"].to_numpy()
    n = len(dfj)
    X = np.zeros((n, len(feature_cols)), dtype=np.float32)

    pt = dfj["pt"].to_numpy(np.float32)
    eta = dfj["eta"].to_numpy(np.float32)
    aeta = dfj["aeta"].to_numpy(np.float32)
    jidx = dfj["jidx"].to_numpy(np.float32)
    rid = infer_region_from_aeta(aeta).astype(np.float32)

    feat_index = {f: i for i, f in enumerate(feature_cols)}

    for k, f in enumerate(feature_cols):
        if f == "pt":
            X[:, k] = pt
            continue
        if f == "eta":
            X[:, k] = eta
            continue
        if f == "aeta":
            X[:, k] = aeta
            continue
        if f == "logpt":
            X[:, k] = safe_log1p(pt)
            continue
        if f == "rid":
            X[:, k] = rid
            continue
        if f == "jidx":
            X[:, k] = jidx
            continue
        if f.startswith("isnan_"):
            continue
        if f in ["hf_ratio", "hf_sigma_sum", "lognC"]:
            continue
        X[:, k] = np.nan

    raw_feats = [
        f for f in feature_cols
        if (not f.startswith("isnan_"))
        and f not in ["pt", "eta", "aeta", "logpt", "rid", "jidx", "hf_ratio", "hf_sigma_sum", "lognC"]
    ]

    for p in dfj["prefix"].unique():
        mask = (dfj["prefix"] == p).to_numpy()
        if not mask.any():
            continue
        rows = evt_idx[mask]

        for f in raw_feats:
            cname = col(p, f, variation)
            out_idx = feat_index[f]
            if cname in df_in.columns:
                vals = pd.to_numeric(df_in.loc[rows, cname], errors="coerce").astype(np.float32).to_numpy()
                X[mask, out_idx] = vals
            else:
                X[mask, out_idx] = np.nan

    if "nConstituents" in feat_index and "lognC" in feat_index:
        X[:, feat_index["lognC"]] = safe_log1p(X[:, feat_index["nConstituents"]])

    if all(v in feat_index for v in ["hfadjacentEtaStripsSize", "hfcentralEtaStripSize", "hf_ratio"]):
        adj = X[:, feat_index["hfadjacentEtaStripsSize"]]
        cen = X[:, feat_index["hfcentralEtaStripSize"]]
        X[:, feat_index["hf_ratio"]] = adj / (cen + 1.0)

    if all(v in feat_index for v in ["hfsigmaEtaEta", "hfsigmaPhiPhi", "hf_sigma_sum"]):
        se = X[:, feat_index["hfsigmaEtaEta"]]
        sp = X[:, feat_index["hfsigmaPhiPhi"]]
        X[:, feat_index["hf_sigma_sum"]] = se + sp

    for k, f in enumerate(feature_cols):
        if f.startswith("isnan_"):
            base = f.replace("isnan_", "")
            if base in feat_index:
                X[:, k] = np.isnan(X[:, feat_index[base]]).astype(np.float32)
            else:
                X[:, k] = 0.0

    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    return X


def threshold_and_direction(hs_scores, pu_scores, hs_eff_target):
    thr_hi = np.quantile(hs_scores, 1.0 - hs_eff_target)
    pu_rej_hi = float((pu_scores < thr_hi).mean())

    thr_lo = np.quantile(hs_scores, hs_eff_target)
    pu_rej_lo = float((pu_scores > thr_lo).mean())

    if pu_rej_lo > pu_rej_hi:
        return float(thr_lo), "keep_low", pu_rej_lo
    else:
        return float(thr_hi), "keep_high", pu_rej_hi


def scan_thresholds_for_region(dfj_region, score_region, hs_eff_list, region, pt_min, pt_turnoff):
    y_hs = dfj_region["y_hs"].to_numpy().astype(bool)
    pt = dfj_region["pt"].to_numpy()
    lowpt = (pt >= pt_min) & (pt < pt_turnoff) & np.isfinite(score_region)

    hs_scores = score_region[lowpt & y_hs]
    pu_scores = score_region[lowpt & (~y_hs)]

    out = {
        "meta": {
            "region": region,
            "pt_min": pt_min,
            "pt_turnoff": pt_turnoff,
            "n_hs_total": int(hs_scores.size),
            "n_pu_total": int(pu_scores.size),
            "mode": "global",
        }
    }

    for hs_eff in hs_eff_list:
        key = f"{hs_eff:.3f}".rstrip("0").rstrip(".")
        if hs_scores.size < 10 or pu_scores.size < 10:
            out[key] = None
            continue

        thr, direction, pu_rej = threshold_and_direction(hs_scores, pu_scores, hs_eff)
        out[key] = {
            "hs_eff_target": float(hs_eff),
            "threshold": thr,
            "direction": direction,
            "pu_rej_est": pu_rej,
            "n_hs": int(hs_scores.size),
            "n_pu": int(pu_scores.size),
        }

    return out


def derive_he_eta_binned_thresholds(score, aeta, y_hs, hs_eff_list, eta_bins=None):
    if eta_bins is None:
        eta_bins = [2.5, 2.7, 2.85, 3.0]

    out = {
        "meta": {
            "region": "HE",
            "mode": "eta_binned",
            "eta_bins": [float(x) for x in eta_bins],
        }
    }

    for hs_eff in hs_eff_list:
        key = f"{hs_eff:.3f}".rstrip("0").rstrip(".")
        out[key] = {"bins": []}

        for lo, hi in zip(eta_bins[:-1], eta_bins[1:]):
            m = (aeta >= lo) & (aeta < hi) & np.isfinite(score)
            hs_scores = score[m & y_hs]
            pu_scores = score[m & (~y_hs)]

            if len(hs_scores) < 20 or len(pu_scores) < 20:
                continue

            thr, direction, pu_rej = threshold_and_direction(hs_scores, pu_scores, hs_eff)

            out[key]["bins"].append({
                "eta_min": float(lo),
                "eta_max": float(hi),
                "threshold": float(thr),
                "direction": str(direction),
                "pu_rej_est": float(pu_rej),
                "n_hs": int(len(hs_scores)),
                "n_pu": int(len(pu_scores)),
            })

    return out


def hs_eff_key(hs_eff: float) -> str:
    return f"{hs_eff:.3f}".rstrip("0").rstrip(".")


def load_region_artifacts(model_dir, region, hs_eff):
    eq_path = os.path.join(model_dir, f"best_equation_{region}.txt")
    feat_path = os.path.join(model_dir, f"features_used_{region}.json")
    thr_path = os.path.join(model_dir, f"thresholds_{region}.json")

    if not os.path.exists(eq_path) or not os.path.exists(feat_path) or not os.path.exists(thr_path):
        return None

    expr = open(eq_path).read().strip()
    feats = json.load(open(feat_path))
    thrj = json.load(open(thr_path))

    key = hs_eff_key(hs_eff)
    if key in thrj:
        info = thrj[key]
        if info is None:
            raise ValueError(f"{region}: threshold entry for hs-eff={hs_eff} is None")
        thr = float(info["threshold"])
        direction = str(info["direction"])
        return expr, feats, thr, direction

    if region in thrj:
        info = thrj[region]
        if info is None:
            raise ValueError(f"{region}: old-format threshold entry is None")
        thr = float(info["threshold"])
        direction = str(info["direction"])
        return expr, feats, thr, direction

    raise KeyError(
        f"Could not find threshold for region={region}, hs-eff={hs_eff} in {thr_path}. "
        f"Available keys: {list(thrj.keys())}"
    )


def maybe_copy(src, dst):
    if os.path.exists(src):
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input parquet")
    ap.add_argument("--model-dir", required=True, help="Directory with saved HE/HF equations and feature jsons")
    ap.add_argument("--outdir", required=True, help="Directory to save copied model files and rescanned thresholds")
    ap.add_argument("--suffix", default="", help='Suffix added before file extension, e.g. "_EffScan"')
    ap.add_argument("--variation", default="nominal")
    ap.add_argument("--n-jets", type=int, default=4)
    ap.add_argument("--jet-prefix-base", default="jet")
    ap.add_argument("--pt-min", type=float, default=25.0)
    ap.add_argument("--pt-turnoff", type=float, default=80.0)
    ap.add_argument("--hs-eff", type=float, default=0.90,
                    help="Existing working point to read from old threshold json if needed")
    ap.add_argument(
        "--hs-eff-list",
        nargs="+",
        type=float,
        default=[0.98, 0.95, 0.90, 0.80],
        help="List of target HS efficiencies",
    )
    ap.add_argument("--use-he-eta-bins", action="store_true",
                    help="Also write eta-binned HE thresholds")
    ap.add_argument("--he-eta-bins", nargs="+", type=float,
                    default=[2.5, 2.7, 2.85, 3.0],
                    help="HE |eta| bin edges for eta-binned thresholds")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    df_in = pd.read_parquet(args.input).reset_index(drop=True)

    prefixes = [f"{args.jet_prefix_base}{i}_" for i in range(1, max(1, min(args.n_jets, 4)) + 1)]
    dfj = build_jet_rows(df_in, prefixes, args.variation)
    dfj = dfj[(dfj["pt"] >= args.pt_min) & (dfj["pt"] < args.pt_turnoff)].copy()

    if dfj.empty:
        raise RuntimeError("No jets found after selection.")

    summary = []

    regions_to_run = []

    if os.path.exists(os.path.join(args.model_dir, "best_equation_HE.txt")):
        regions_to_run.append("HE")

    if os.path.exists(os.path.join(args.model_dir, "best_equation_HF.txt")):
        regions_to_run.append("HF")

    if len(regions_to_run) == 0:
        raise RuntimeError("No valid HE/HF models found in model-dir.")

    print("\nRegions detected:")
    for r in regions_to_run:
        print(f"  - {r}")

    for region in regions_to_run:
        model = load_region_artifacts(args.model_dir, region, args.hs_eff)
        if model is None:
            print(f"[INFO] Missing complete artifacts for {region}, skipping.")
            continue

        eq, feats, old_thr, old_dir = model

        mask = region_mask(dfj["aeta"].to_numpy(), region)
        dfj_reg = dfj.loc[mask].copy()

        if dfj_reg.empty:
            print(f"[WARN] No {region} jets after selection. Skipping.")
            continue

        X = build_feature_matrix(df_in, dfj_reg, feats, args.variation)
        valid = np.all(np.isfinite(X), axis=1) & np.isfinite(dfj_reg["y_hs"].to_numpy())

        dfj_reg = dfj_reg.loc[valid].copy()
        X = X[valid]

        if len(dfj_reg) == 0:
            print(f"[WARN] No valid {region} jets after feature cleaning. Skipping.")
            continue

        score = compute_score_from_sympy_x(eq, X)

        # global thresholds
        thr_out = scan_thresholds_for_region(
            dfj_reg,
            score,
            args.hs_eff_list,
            region,
            args.pt_min,
            args.pt_turnoff,
        )

        # HE eta-binned thresholds
        eta_binned_out = None
        if region == "HE" and args.use_he_eta_bins:
            y_hs_reg = dfj_reg["y_hs"].to_numpy().astype(bool)
            aeta_reg = dfj_reg["aeta"].to_numpy()
            eta_binned_out = derive_he_eta_binned_thresholds(
                score=score,
                aeta=aeta_reg,
                y_hs=y_hs_reg,
                hs_eff_list=args.hs_eff_list,
                eta_bins=args.he_eta_bins,
            )

        eq_path = os.path.join(args.model_dir, f"best_equation_{region}.txt")
        feat_path = os.path.join(args.model_dir, f"features_used_{region}.json")

        eq_out = os.path.join(args.outdir, f"best_equation_{region}{args.suffix}.txt")
        feat_out = os.path.join(args.outdir, f"features_used_{region}{args.suffix}.json")
        thr_outfile = os.path.join(args.outdir, f"thresholds_{region}{args.suffix}.json")

        maybe_copy(eq_path, eq_out)
        maybe_copy(feat_path, feat_out)

        with open(thr_outfile, "w") as f:
            json.dump(thr_out, f, indent=2)

        if eta_binned_out is not None:
            thr_eta_outfile = os.path.join(args.outdir, f"thresholds_{region}_etaBinned{args.suffix}.json")
            with open(thr_eta_outfile, "w") as f:
                json.dump(eta_binned_out, f, indent=2)
        else:
            thr_eta_outfile = None

        summary.append({
            "region": region,
            "equation": eq,
            "old_threshold": old_thr,
            "old_direction": old_dir,
            "n_features": len(feats),
            "n_jets_used": int(len(dfj_reg)),
            "threshold_file": os.path.basename(thr_outfile),
            "eta_binned_threshold_file": None if thr_eta_outfile is None else os.path.basename(thr_eta_outfile),
        })

        print(f"[{region}] equation: {eq}")
        print(f"[{region}] old wp: thr={old_thr}, dir={old_dir}")
        print(f"[{region}] jets used: {len(dfj_reg)}")
        print(f"[{region}] saved global: {thr_outfile}")
        if thr_eta_outfile is not None:
            print(f"[{region}] saved eta-binned: {thr_eta_outfile}")

    with open(os.path.join(args.outdir, f"rescan_summary{args.suffix}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()