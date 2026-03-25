#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pandas as pd
from rich import print
import ROOT

ROOT.gROOT.SetBatch(True)

"""
python MVA_training/pileup_symbolic_regression/validate_pysr_region_models_newTHPattern.py \
  -i /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn_Mar19_tightPassLepVeto_NoJER_pySR/stage1_output/2022postEE/compacted/dyTo2L_M-50_incl/0/part038.parquet \
  --pysr-out validation/pysr_best_threshold_scan \
  -o validation/pysr_best_threshold_scan/plots_hs_eff_80 \
  --variation nominal \
  --n-jets 4 \
  --hs-eff 0.80

python MVA_training/pileup_symbolic_regression/validate_pysr_region_models_newTHPattern.py \
  -i /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn_Mar19_tightPassLepVeto_NoJER_pySR/stage1_output/2022postEE/compacted/dyTo2L_M-50_incl/0/part038.parquet \
  --pysr-out validation/pysr_best_threshold_scan_etadependent \
  -o validation/pysr_best_threshold_scan_etadependent/plots_hs_eff_80 \
  --variation nominal \
  --n-jets 4 \
  --hs-eff 0.80  

python MVA_training/pileup_symbolic_regression/validate_pysr_region_models_newTHPattern.py   \
    -i /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn_Mar19_tightPassLepVeto_NoJER_pySR/stage1_output/2022postEE/compacted/dyTo2L_M-50_incl/0/part038.parquet \
    --pysr-out validation/pysr_best_threshold_scan_v2   \
    -o validation/pysr_best_threshold_scan_v2/plots_hs_eff_85   \
    --variation nominal   --n-jets 4   --hs-eff 0.85   

python MVA_training/pileup_symbolic_regression/validate_pysr_region_models_newTHPattern.py   \
    -i /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn_Mar19_tightPassLepVeto_NoJER_pySR/stage1_output/2022postEE/compacted/dyTo2L_M-50_incl/0/part038.parquet \
    --pysr-out validation/pysr_best_threshold_scan_decorrelate_pT   \
    -o validation/pysr_best_threshold_scan_decorrelate_pT/plots_hs_eff_85   \
    --variation nominal   --n-jets 4   --hs-eff 0.85       
"""


# ============================================================
# Basic helpers
# ============================================================
def col(prefix: str, var: str, variation: str) -> str:
    return f"{prefix}{var}_{variation}"


def hs_eff_key(hs_eff: float) -> str:
    return f"{hs_eff:.3f}".rstrip("0").rstrip(".")


def region_mask(aeta: np.ndarray, region: str) -> np.ndarray:
    if region == "HE":
        return (aeta >= 2.5) & (aeta < 3.0)
    if region == "HF":
        return aeta >= 3.0
    raise ValueError(region)


def safe_log1p(x):
    return np.log1p(np.clip(x, 0, None))


def infer_range(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (0.0, 1.0)
    lo = np.percentile(x, 1)
    hi = np.percentile(x, 99)
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    return float(lo), float(hi)


def save_canvas(c, outpath):
    c.SaveAs(outpath)
    print(f"[Saved] {outpath}")


# ============================================================
# Load trained region artifacts
# ============================================================
def load_region_artifacts(pysr_out, region, hs_eff):
    eq_path = os.path.join(pysr_out, f"best_equation_{region}_EffScan.txt")
    feat_path = os.path.join(pysr_out, f"features_used_{region}_EffScan.json")
    thr_path = os.path.join(pysr_out, f"thresholds_{region}_EffScan.json")
    # thr_path = os.path.join(pysr_out, f"thresholds_{region}_etaBinned_EffScan.json")

    if not os.path.exists(eq_path) or not os.path.exists(feat_path) or not os.path.exists(thr_path):
        return None

    expr = open(eq_path).read().strip()
    feats = json.load(open(feat_path))
    thrj = json.load(open(thr_path))

    key = hs_eff_key(hs_eff)

    # New rescanned global format
    if key in thrj:
        info = thrj[key]
        if info is None:
            raise ValueError(f"{region}: threshold entry for hs-eff={hs_eff} is None")

        # global threshold entry
        if isinstance(info, dict) and ("threshold" in info) and ("direction" in info):
            thr = float(info["threshold"])
            direction = str(info["direction"])
            return expr, feats, thr, direction

        # eta-binned entry accidentally stored here
        if isinstance(info, dict) and ("bins" in info):
            return expr, feats, None, None
    
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


def load_he_eta_binned_thresholds(pysr_out, hs_eff):
    key = hs_eff_key(hs_eff)

    candidates = [
        os.path.join(pysr_out, "thresholds_HE_etaBinned.json"),
    ]

    # also allow any suffixed eta-binned file
    for fname in sorted(os.listdir(pysr_out)):
        if fname.startswith("thresholds_HE_etaBinned") and fname.endswith(".json"):
            full = os.path.join(pysr_out, fname)
            if full not in candidates:
                candidates.append(full)

    for path in candidates:
        if not os.path.exists(path):
            continue

        data = json.load(open(path))
        if key in data and data[key] is not None:
            return data[key]

    return None


def apply_wp(score, thr, direction):
    score = np.asarray(score)
    if direction == "keep_high":
        return score >= thr
    if direction == "keep_low":
        return score <= thr
    raise ValueError(direction)


def apply_he_eta_binned_wp(score, aeta, thr_json):
    score = np.asarray(score)
    aeta = np.asarray(aeta)
    pass_mask = np.ones_like(score, dtype=bool)

    if thr_json is None or "bins" not in thr_json:
        return pass_mask

    for b in thr_json["bins"]:
        lo = b["eta_min"]
        hi = b["eta_max"]
        thr = b["threshold"]
        direction = b["direction"]
        m = (aeta >= lo) & (aeta < hi) & np.isfinite(score)

        if direction == "keep_high":
            pass_mask[m] = score[m] >= thr
        elif direction == "keep_low":
            pass_mask[m] = score[m] <= thr
        else:
            raise ValueError(f"Unknown direction: {direction}")

    return pass_mask


def pu_rejection_fixed_wp(pu_scores, thr, direction):
    pu_scores = np.asarray(pu_scores)
    if direction == "keep_high":
        return float((pu_scores < thr).mean())
    if direction == "keep_low":
        return float((pu_scores > thr).mean())
    raise ValueError(direction)


def pu_rejection_fixed_wp_eta_binned(pu_scores, pu_aeta, thr_json):
    pu_scores = np.asarray(pu_scores)
    pu_aeta = np.asarray(pu_aeta)
    keep = np.ones_like(pu_scores, dtype=bool)

    if thr_json is None or "bins" not in thr_json:
        return np.nan

    for b in thr_json["bins"]:
        lo = b["eta_min"]
        hi = b["eta_max"]
        thr = b["threshold"]
        direction = b["direction"]
        m = (pu_aeta >= lo) & (pu_aeta < hi) & np.isfinite(pu_scores)

        if direction == "keep_high":
            keep[m] = pu_scores[m] >= thr
        elif direction == "keep_low":
            keep[m] = pu_scores[m] <= thr
        else:
            raise ValueError(f"Unknown direction: {direction}")

    return float((~keep).mean())


def hs_eff_fixed_wp_eta_binned(hs_scores, hs_aeta, thr_json):
    hs_scores = np.asarray(hs_scores)
    hs_aeta = np.asarray(hs_aeta)
    keep = np.ones_like(hs_scores, dtype=bool)

    if thr_json is None or "bins" not in thr_json:
        return np.nan

    for b in thr_json["bins"]:
        lo = b["eta_min"]
        hi = b["eta_max"]
        thr = b["threshold"]
        direction = b["direction"]
        m = (hs_aeta >= lo) & (hs_aeta < hi) & np.isfinite(hs_scores)

        if direction == "keep_high":
            keep[m] = hs_scores[m] >= thr
        elif direction == "keep_low":
            keep[m] = hs_scores[m] <= thr
        else:
            raise ValueError(f"Unknown direction: {direction}")

    return float(keep.mean())


# ============================================================
# Score evaluation
# ============================================================
def compute_score_from_sympy_x(expr_str: str, X: np.ndarray) -> np.ndarray:
    loc = {
        "np": np,
        "abs": np.abs,
        "Abs": np.abs,
        "sqrt": np.sqrt,
        "sqrt_abs": lambda x: np.sqrt(np.abs(x)),
        "log": lambda x: np.log(np.clip(x, 1e-6, None)),
        "log1p": safe_log1p,
        "log1p_abs": lambda x: np.log1p(np.abs(x)),
        "tanh": np.tanh,
    }
    for i in range(X.shape[1]):
        loc[f"x{i}"] = X[:, i]

    out = eval(expr_str, {"__builtins__": {}}, loc)
    out = np.asarray(out, dtype=np.float32)
    if out.ndim == 0:
        out = np.full(X.shape[0], float(out), dtype=np.float32)
    return out


# ============================================================
# Build flat jet rows
# ============================================================
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
        y_hs = (
            pd.to_numeric(df_in[col(p, "hasMatchedGenJet", variation)], errors="coerce")
            .fillna(0).astype(bool).to_numpy()
        )

        out = pd.DataFrame({
            "evt_idx": np.arange(len(df_in)),
            "pt": pt,
            "eta": eta,
            "aeta": np.abs(eta),
            "y_hs": y_hs.astype(np.int8),
            "jidx": np.full_like(pt, jidx, dtype=np.float32),
            "prefix": p,
        })
        dfs.append(out)

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

    rid = np.zeros_like(aeta, dtype=np.float32)
    rid[(aeta >= 2.5) & (aeta < 3.0)] = 1.0
    rid[aeta >= 3.0] = 2.0

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

    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


# ============================================================
# ROOT plot helpers
# ============================================================
def make_tgraph_finite(x, y, name, title, xlab, ylab):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() == 0:
        return None
    g = ROOT.TGraph(int(m.sum()), x[m], y[m])
    g.SetName(name)
    g.SetTitle(f"{title};{xlab};{ylab}")
    g.SetLineWidth(2)
    g.SetMarkerStyle(20)
    g.SetMarkerSize(1.0)
    return g


def make_hist(name, title, nbins, xmin, xmax):
    h = ROOT.TH1F(name, title, nbins, xmin, xmax)
    h.SetStats(0)
    h.Sumw2()
    return h


def fill_hist(h, arr):
    for v in arr:
        if np.isfinite(v):
            h.Fill(float(v))


def normalize_hist(h):
    if h.Integral() > 0:
        h.Scale(1.0 / h.Integral())


# ============================================================
# Plot 1: score distributions
# ============================================================
def plot_score_real_fake(score, y_hs, outpath, region):
    good = np.isfinite(score)
    score = score[good]
    y_hs = y_hs[good]

    xmin, xmax = infer_range(score)
    h_real = make_hist("h_real", f"{region} score;score;Normalized entries", 50, xmin, xmax)
    h_fake = make_hist("h_fake", f"{region} score;score;Normalized entries", 50, xmin, xmax)

    fill_hist(h_real, score[y_hs])
    fill_hist(h_fake, score[~y_hs])

    normalize_hist(h_real)
    normalize_hist(h_fake)

    h_real.SetLineColor(ROOT.kBlue + 1)
    h_real.SetLineWidth(2)
    h_fake.SetLineColor(ROOT.kRed + 1)
    h_fake.SetLineWidth(2)

    ymax = max(h_real.GetMaximum(), h_fake.GetMaximum()) * 1.25

    c = ROOT.TCanvas("c_score", "", 850, 700)
    h_real.SetMaximum(ymax)
    h_real.Draw("hist")
    h_fake.Draw("hist same")

    leg = ROOT.TLegend(0.62, 0.76, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(h_real, "Real (HS)", "l")
    leg.AddEntry(h_fake, "Fake (PU)", "l")
    leg.Draw()

    save_canvas(c, outpath)


# ============================================================
# Plot 2/3: fixed WP performance vs pt
# ============================================================
def plot_wp_performance_vs_pt(dfj, score, outdir,
                              he_cfg=None, hf_cfg=None,
                              min_hs=30, min_pu=30):
    bins = np.array([25, 30, 35, 40, 50, 60, 70, 80], dtype=np.float32)
    centers = 0.5 * (bins[:-1] + bins[1:])

    y_hs = dfj["y_hs"].to_numpy().astype(bool)
    aeta = dfj["aeta"].to_numpy()
    pt = dfj["pt"].to_numpy()

    if he_cfg is not None:
        pu_rej, hs_eff = [], []
        mreg = region_mask(aeta, "HE")

        for lo, hi in zip(bins[:-1], bins[1:]):
            m = mreg & (pt >= lo) & (pt < hi) & np.isfinite(score)

            hs_scores = score[m & y_hs]
            hs_aeta   = aeta[m & y_hs]
            pu_scores = score[m & (~y_hs)]
            pu_aeta   = aeta[m & (~y_hs)]

            if len(hs_scores) < min_hs or len(pu_scores) < min_pu:
                hs_eff.append(np.nan)
                pu_rej.append(np.nan)
                continue

            if he_cfg.get("eta_binned", False):
                hs_eff.append(hs_eff_fixed_wp_eta_binned(hs_scores, hs_aeta, he_cfg["thr_json"]))
                pu_rej.append(pu_rejection_fixed_wp_eta_binned(pu_scores, pu_aeta, he_cfg["thr_json"]))
            else:
                hs_eff.append(float(apply_wp(hs_scores, he_cfg["thr"], he_cfg["dir"]).mean()))
                pu_rej.append(pu_rejection_fixed_wp(pu_scores, he_cfg["thr"], he_cfg["dir"]))

        c1 = ROOT.TCanvas("c_purej_HE", "", 850, 700)
        g1 = make_tgraph_finite(
            centers, pu_rej,
            "g_purej_HE",
            "PU rejection vs pT (HE) [fixed WP]",
            "jet p_{T} [GeV]", "PU rejection"
        )
        if g1:
            g1.Draw("APL")
            g1.GetYaxis().SetRangeUser(0.0, 1.0)
        save_canvas(c1, os.path.join(outdir, "pu_rejection_vs_pt_HE.pdf"))

        c2 = ROOT.TCanvas("c_hseff_HE", "", 850, 700)
        g2 = make_tgraph_finite(
            centers, hs_eff,
            "g_hseff_HE",
            "HS efficiency vs pT (HE) [fixed WP]",
            "jet p_{T} [GeV]", "HS efficiency"
        )
        if g2:
            g2.Draw("APL")
            g2.GetYaxis().SetRangeUser(0.0, 1.0)
        save_canvas(c2, os.path.join(outdir, "hs_eff_vs_pt_HE.pdf"))

    if hf_cfg is not None:
        pu_rej, hs_eff = [], []
        mreg = region_mask(aeta, "HF")

        for lo, hi in zip(bins[:-1], bins[1:]):
            m = mreg & (pt >= lo) & (pt < hi) & np.isfinite(score)
            hs_scores = score[m & y_hs]
            pu_scores = score[m & (~y_hs)]

            if len(hs_scores) < min_hs or len(pu_scores) < min_pu:
                hs_eff.append(np.nan)
                pu_rej.append(np.nan)
                continue

            hs_eff.append(float(apply_wp(hs_scores, hf_cfg["thr"], hf_cfg["dir"]).mean()))
            pu_rej.append(pu_rejection_fixed_wp(pu_scores, hf_cfg["thr"], hf_cfg["dir"]))

        c1 = ROOT.TCanvas("c_purej_HF", "", 850, 700)
        g1 = make_tgraph_finite(
            centers, pu_rej,
            "g_purej_HF",
            "PU rejection vs pT (HF) [fixed WP]",
            "jet p_{T} [GeV]", "PU rejection"
        )
        if g1:
            g1.Draw("APL")
            g1.GetYaxis().SetRangeUser(0.0, 1.0)
        save_canvas(c1, os.path.join(outdir, "pu_rejection_vs_pt_HF.pdf"))

        c2 = ROOT.TCanvas("c_hseff_HF", "", 850, 700)
        g2 = make_tgraph_finite(
            centers, hs_eff,
            "g_hseff_HF",
            "HS efficiency vs pT (HF) [fixed WP]",
            "jet p_{T} [GeV]", "HS efficiency"
        )
        if g2:
            g2.Draw("APL")
            g2.GetYaxis().SetRangeUser(0.0, 1.0)
        save_canvas(c2, os.path.join(outdir, "hs_eff_vs_pt_HF.pdf"))


# ============================================================
# Stacked real/fake + total
# ============================================================
def plot_stacked_total_real_fake(values, y_hs, outpath, title, xlab, nbins=50, xmin=None, xmax=None, logy=False):
    values = np.asarray(values)
    y_hs = np.asarray(y_hs).astype(bool)
    good = np.isfinite(values)
    values = values[good]
    y_hs = y_hs[good]

    if xmin is None or xmax is None:
        xmin, xmax = infer_range(values)

    h_real = make_hist("h_real", f"{title};{xlab};Entries", nbins, xmin, xmax)
    h_fake = make_hist("h_fake", f"{title};{xlab};Entries", nbins, xmin, xmax)
    h_tot  = make_hist("h_tot",  f"{title};{xlab};Entries", nbins, xmin, xmax)

    for v in values[y_hs]:
        h_real.Fill(float(v))
        h_tot.Fill(float(v))
    for v in values[~y_hs]:
        h_fake.Fill(float(v))
        h_tot.Fill(float(v))

    h_real.SetFillColor(ROOT.kAzure - 2)
    h_real.SetLineColor(ROOT.kAzure - 2)
    h_fake.SetFillColor(ROOT.kOrange)
    h_fake.SetLineColor(ROOT.kOrange + 1)

    h_tot.SetMarkerStyle(20)
    h_tot.SetMarkerSize(0.9)
    h_tot.SetLineColor(ROOT.kBlack)
    h_tot.SetMarkerColor(ROOT.kBlack)

    stack = ROOT.THStack("stack", f"{title};{xlab};Entries")
    stack.Add(h_fake)
    stack.Add(h_real)

    # ============================================================
    # Canvas with 2 pads
    # ============================================================
    c = ROOT.TCanvas("c_stack", "", 850, 800)

    pad1 = ROOT.TPad("pad1", "", 0, 0.30, 1, 1)
    pad2 = ROOT.TPad("pad2", "", 0, 0.00, 1, 0.30)

    pad1.SetBottomMargin(0.02)
    pad2.SetTopMargin(0.00)
    pad2.SetBottomMargin(0.30)

    pad1.Draw()
    pad2.Draw()

    # ============================================================
    # Upper pad (stack)
    # ============================================================
    pad1.cd()
    if logy:
        pad1.SetLogy()

    stack.Draw("hist")
    h_tot.Draw("E1 same")

    ymax = max(stack.GetMaximum(), h_tot.GetMaximum()) * (20.0 if logy else 1.35)
    stack.SetMaximum(ymax)
    if logy:
        stack.SetMinimum(1e-1)

    leg = ROOT.TLegend(0.57, 0.72, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(h_tot, "Total jets", "lep")
    leg.AddEntry(h_real, "Real (HS)", "f")
    leg.AddEntry(h_fake, "Fake (PU)", "f")
    leg.Draw()

    # ============================================================
    # Ratio pad (Fake / Real)
    # ============================================================
    pad2.cd()

    h_ratio = h_real.Clone("h_ratio")
    h_ratio.Reset()

    for i in range(1, h_real.GetNbinsX() + 1):
        r = h_real.GetBinContent(i)
        f = h_fake.GetBinContent(i)

        if r > 0:
            val = f/(r+f)
            h_ratio.SetBinContent(i, val)
            h_ratio.SetBinError(i, 0.0)
        else:
            h_ratio.SetBinContent(i, 0.0)

    h_ratio.SetTitle("")
    h_ratio.GetYaxis().SetTitle("Fake/Total")
    h_ratio.GetXaxis().SetTitle(xlab)

    h_ratio.SetMarkerStyle(20)
    h_ratio.SetMarkerSize(0.8)

    h_ratio.GetYaxis().SetNdivisions(505)
    h_ratio.GetYaxis().SetTitleSize(0.09)
    h_ratio.GetYaxis().SetTitleOffset(0.5)
    h_ratio.GetYaxis().SetLabelSize(0.08)

    h_ratio.GetXaxis().SetTitleSize(0.10)
    h_ratio.GetXaxis().SetLabelSize(0.08)

    h_ratio.SetMinimum(0.0)
    h_ratio.SetMaximum(1.0)  # adjust if needed

    h_ratio.Draw("E1")

    # reference line at 1
    line = ROOT.TLine(xmin, 1.0, xmax, 1.0)
    line.SetLineStyle(2)
    line.Draw()

    # ============================================================
    save_canvas(c, outpath)


def plot_stacked_before_after(values, y_hs, pass_mask, outbase, title, xlab, nbins=50, xmin=None, xmax=None, logy=False):
    if xmin is None or xmax is None:
        xmin, xmax = infer_range(np.asarray(values)[np.isfinite(values)])

    plot_stacked_total_real_fake(
        values, y_hs,
        outpath=f"{outbase}_before_log.pdf" if logy else f"{outbase}_before.pdf",
        title=f"{title} (before)",
        xlab=xlab,
        nbins=nbins, xmin=xmin, xmax=xmax, logy=logy
    )

    plot_stacked_total_real_fake(
        np.asarray(values)[pass_mask],
        np.asarray(y_hs)[pass_mask],
        outpath=f"{outbase}_after_log.pdf" if logy else f"{outbase}_after.pdf",
        title=f"{title} (after)",
        xlab=xlab,
        nbins=nbins, xmin=xmin, xmax=xmax, logy=logy
    )


# ============================================================
# Physics safety plots
# ============================================================
def plot_jet_eta_before_after_mask(dfj, pass_mask, outdir):
    eta = dfj["eta"].to_numpy()

    h_before = make_hist("h_eta_before", "Jet #eta before/after PU cut;jet #eta;Entries", 60, -5.0, 5.0)
    h_after  = make_hist("h_eta_after",  "Jet #eta before/after PU cut;jet #eta;Entries", 60, -5.0, 5.0)

    fill_hist(h_before, eta[np.isfinite(eta)])
    fill_hist(h_after, eta[pass_mask & np.isfinite(eta)])


    h_before.SetLineColor(ROOT.kBlack)
    h_before.SetLineWidth(2)
    h_after.SetLineColor(ROOT.kRed + 1)
    h_after.SetLineWidth(2)
    h_after.SetLineStyle(2)

    # ============================================================
    # Canvas with ratio pad
    # ============================================================
    c = ROOT.TCanvas("c_eta_ba", "", 850, 800)

    pad1 = ROOT.TPad("pad1", "", 0, 0.30, 1, 1)
    pad2 = ROOT.TPad("pad2", "", 0, 0.00, 1, 0.30)

    pad1.SetBottomMargin(0.02)
    pad2.SetTopMargin(0.05)
    pad2.SetBottomMargin(0.30)

    pad1.Draw()
    pad2.Draw()

    # ============================================================
    # Top pad
    # ============================================================
    pad1.cd()

    ymax = max(h_before.GetMaximum(), h_after.GetMaximum()) * 1.25
    h_before.SetMaximum(ymax)

    h_before.Draw("hist")
    h_after.Draw("hist same")

    leg = ROOT.TLegend(0.62, 0.78, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(h_before, "Before cut", "l")
    leg.AddEntry(h_after, "After cut", "l")
    leg.Draw()

    # ============================================================
    # Ratio pad (After / Before)
    # ============================================================
    pad2.cd()

    h_ratio = h_after.Clone("h_ratio")
    h_ratio.Reset()

    for i in range(1, h_before.GetNbinsX() + 1):
        b = h_before.GetBinContent(i)
        a = h_after.GetBinContent(i)

        if b > 0:
            # h_ratio.SetBinContent(i, a / b)
            h_ratio.SetBinContent(i, (b-a)/b)
            h_ratio.SetBinError(i, 0.0)  # keep simple
        else:
            h_ratio.SetBinContent(i, 0.0)

    h_ratio.SetTitle("")
    h_ratio.GetYaxis().SetTitle("(Before-After)/Before")
    h_ratio.GetXaxis().SetTitle("jet #eta")

    h_ratio.SetMarkerStyle(20)
    h_ratio.SetMarkerSize(0.8)

    h_ratio.GetYaxis().SetNdivisions(505)
    h_ratio.GetYaxis().SetTitleSize(0.09)
    h_ratio.GetYaxis().SetTitleOffset(0.5)
    h_ratio.GetYaxis().SetLabelSize(0.08)

    h_ratio.GetXaxis().SetTitleSize(0.10)
    h_ratio.GetXaxis().SetLabelSize(0.08)

    h_ratio.SetMinimum(0.0)
    h_ratio.SetMaximum(1.0)  # important for efficiency plots

    h_ratio.Draw("E1")

    # reference line at 1
    line = ROOT.TLine(-5.0, 1.0, 5.0, 1.0)
    line.SetLineStyle(2)
    line.Draw()

    # ============================================================
    save_canvas(c, os.path.join(outdir, "jet_eta_before_after.pdf"))


def autodetect_col(df_in, variation, keys):
    cols = list(df_in.columns)
    for k in keys:
        for c in cols:
            if (k in c.lower()) and c.endswith(f"_{variation}"):
                return c
    return None


def plot_mjj_deta_before_after(df_in, variation, outdir,
                               vbf_base, score_func_for_twojets,
                               he_cfg, hf_cfg,
                               pt_min=25, pt_turnoff=80,
                               mjj_col=None, deta_col=None):
    p1 = f"{vbf_base}1_"
    p2 = f"{vbf_base}2_"

    needed = [
        col(p1, "pt", variation), col(p1, "eta", variation),
        col(p2, "pt", variation), col(p2, "eta", variation),
    ]
    for c in needed:
        if c not in df_in.columns:
            print(f"[WARN] Missing {c}. Skipping VBF plot2.")
            return

    if mjj_col is None:
        mjj_col = autodetect_col(df_in, variation, ["mjj", "jj_m", "dijet_m"])
    if deta_col is None:
        deta_col = autodetect_col(df_in, variation, ["deta", "jj_deta", "dijet_deta"])
    if mjj_col is None or deta_col is None:
        print("[WARN] Could not auto-detect mjj/deta columns. Skipping VBF plot2.")
        return

    pt1 = pd.to_numeric(df_in[col(p1, "pt", variation)], errors="coerce").astype(np.float32).to_numpy()
    eta1 = pd.to_numeric(df_in[col(p1, "eta", variation)], errors="coerce").astype(np.float32).to_numpy()
    pt2 = pd.to_numeric(df_in[col(p2, "pt", variation)], errors="coerce").astype(np.float32).to_numpy()
    eta2 = pd.to_numeric(df_in[col(p2, "eta", variation)], errors="coerce").astype(np.float32).to_numpy()

    aeta1 = np.abs(eta1)
    aeta2 = np.abs(eta2)

    lowpt1 = (pt1 >= pt_min) & (pt1 < pt_turnoff)
    lowpt2 = (pt2 >= pt_min) & (pt2 < pt_turnoff)

    s1, s2 = score_func_for_twojets(df_in, variation)

    def pass_one(aeta, lowpt, s):
        he = region_mask(aeta, "HE") & lowpt & np.isfinite(s)
        hf = region_mask(aeta, "HF") & lowpt & np.isfinite(s)
        pm = np.ones_like(s, dtype=bool)

        if he_cfg is not None:
            if he_cfg.get("eta_binned", False):
                pm[he] = apply_he_eta_binned_wp(s[he], aeta[he], he_cfg["thr_json"])
            else:
                pm[he] = apply_wp(s[he], he_cfg["thr"], he_cfg["dir"])

        if hf_cfg is not None:
            pm[hf] = apply_wp(s[hf], hf_cfg["thr"], hf_cfg["dir"])

        return pm

    pass_evt = pass_one(aeta1, lowpt1, s1) & pass_one(aeta2, lowpt2, s2)

    mjj_all = pd.to_numeric(df_in[mjj_col], errors="coerce").astype(np.float32).to_numpy()
    deta_all = pd.to_numeric(df_in[deta_col], errors="coerce").astype(np.float32).to_numpy()
    good = np.isfinite(mjj_all) & np.isfinite(deta_all)

    h_mjj_b = make_hist("h_mjj_b", "M_{jj} before/after PU cut;M_{jj} [GeV];Normalized entries", 60, 0, 3000)
    h_mjj_a = make_hist("h_mjj_a", "M_{jj} before/after PU cut;M_{jj} [GeV];Normalized entries", 60, 0, 3000)
    h_de_b  = make_hist("h_de_b",  "#Delta#eta_{jj} before/after PU cut;#Delta#eta_{jj};Normalized entries", 50, 0, 8)
    h_de_a  = make_hist("h_de_a",  "#Delta#eta_{jj} before/after PU cut;#Delta#eta_{jj};Normalized entries", 50, 0, 8)

    fill_hist(h_mjj_b, mjj_all[good])
    fill_hist(h_mjj_a, mjj_all[good & pass_evt])
    fill_hist(h_de_b, deta_all[good])
    fill_hist(h_de_a, deta_all[good & pass_evt])

    for h in [h_mjj_b, h_mjj_a, h_de_b, h_de_a]:
        normalize_hist(h)
        h.SetLineWidth(2)
    h_mjj_b.SetLineColor(ROOT.kBlack)
    h_mjj_a.SetLineColor(ROOT.kRed + 1)
    h_mjj_a.SetLineStyle(2)
    h_de_b.SetLineColor(ROOT.kBlack)
    h_de_a.SetLineColor(ROOT.kRed + 1)
    h_de_a.SetLineStyle(2)

    c1 = ROOT.TCanvas("c_mjj", "", 850, 700)
    ymax = max(h_mjj_b.GetMaximum(), h_mjj_a.GetMaximum()) * 1.25
    h_mjj_b.SetMaximum(ymax)
    h_mjj_b.Draw("hist")
    h_mjj_a.Draw("hist same")
    leg = ROOT.TLegend(0.62, 0.78, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(h_mjj_b, "Before cut", "l")
    leg.AddEntry(h_mjj_a, "After cut", "l")
    leg.Draw()
    save_canvas(c1, os.path.join(outdir, "vbf_mjj_before_after.pdf"))

    c2 = ROOT.TCanvas("c_deta", "", 850, 700)
    ymax2 = max(h_de_b.GetMaximum(), h_de_a.GetMaximum()) * 1.25
    h_de_b.SetMaximum(ymax2)
    h_de_b.Draw("hist")
    h_de_a.Draw("hist same")
    leg2 = ROOT.TLegend(0.62, 0.78, 0.88, 0.88)
    leg2.SetBorderSize(0)
    leg2.SetFillStyle(0)
    leg2.AddEntry(h_de_b, "Before cut", "l")
    leg2.AddEntry(h_de_a, "After cut", "l")
    leg2.Draw()
    save_canvas(c2, os.path.join(outdir, "vbf_deta_before_after.pdf"))


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--outdir", default="validation/pysr_validate")
    ap.add_argument("--variation", default="nominal")
    ap.add_argument("--pysr-out", required=True)
    ap.add_argument("--n-jets", type=int, default=4)
    ap.add_argument("--jet-prefix-base", default="jet")
    ap.add_argument("--pt-min", type=float, default=25.0)
    ap.add_argument("--pt-turnoff", type=float, default=80.0)
    ap.add_argument("--vbf-jet-base", default="vbf_maxmjj_jet")
    ap.add_argument("--mjj-col", default=None)
    ap.add_argument("--deta-col", default=None)
    ap.add_argument("--hs-eff", type=float, default=0.90,
                    help="Working point to load from rescanned thresholds json, e.g. 0.98, 0.95, 0.90")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    he_model = load_region_artifacts(args.pysr_out, "HE", args.hs_eff)
    hf_model = load_region_artifacts(args.pysr_out, "HF", args.hs_eff)
    try:
        he_eta_binned = load_he_eta_binned_thresholds(args.pysr_out, args.hs_eff)
    except Exception:
        he_eta_binned = None

    if he_model is None and hf_model is None:
        raise RuntimeError(f"No HE/HF model artifacts found in {args.pysr_out}")

    if he_model is not None:
        expr_he, feats_he, thr_he, dir_he = he_model
        print("[HE] hs-eff:", args.hs_eff, "direction/thr:", dir_he, thr_he,
              "nfeat:", len(feats_he), "expr:", expr_he)
    else:
        expr_he, feats_he, thr_he, dir_he = None, None, None, None
        print("[HE] not found, skipping.")

    if he_eta_binned is not None:
        print("[HE eta-binned] using eta-dependent thresholds")

    if hf_model is not None:
        expr_hf, feats_hf, thr_hf, dir_hf = hf_model
        print("[HF] hs-eff:", args.hs_eff, "direction/thr:", dir_hf, thr_hf,
              "nfeat:", len(feats_hf), "expr:", expr_hf)
    else:
        expr_hf, feats_hf, thr_hf, dir_hf = None, None, None, None
        print("[HF] not found, skipping.")

    df_in = pd.read_parquet(args.input).reset_index(drop=True)

    n_jets = max(1, min(args.n_jets, 4))
    prefixes = [f"{args.jet_prefix_base}{i}_" for i in range(1, n_jets + 1)]
    dfj = build_jet_rows(df_in, prefixes, args.variation)
    dfj = dfj[(dfj["pt"] >= args.pt_min) & (dfj["pt"] < args.pt_turnoff)].copy()

    aeta = dfj["aeta"].to_numpy()
    y_hs = dfj["y_hs"].to_numpy().astype(bool)
    m_he = region_mask(aeta, "HE")
    m_hf = region_mask(aeta, "HF")

    score = np.full(len(dfj), np.nan, dtype=np.float32)
    pass_mask = np.ones(len(dfj), dtype=bool)

    if he_model is not None and m_he.any():
        Xhe = build_feature_matrix(df_in, dfj.loc[m_he].copy(), feats_he, args.variation)
        score_he = compute_score_from_sympy_x(expr_he, Xhe)
        score[m_he] = score_he

        if he_eta_binned is not None:
            print("[HE] applying eta-binned thresholds")
            pass_mask[m_he] = apply_he_eta_binned_wp(score_he, aeta[m_he], he_eta_binned)
        elif (thr_he is not None) and (dir_he is not None):
            print("[HE] applying global threshold")
            pass_mask[m_he] = apply_wp(score_he, thr_he, dir_he)
        else:
            raise RuntimeError(
                "HE model exists, but neither eta-binned nor global HE thresholds were found."
            )

    if hf_model is not None and m_hf.any():
        Xhf = build_feature_matrix(df_in, dfj.loc[m_hf].copy(), feats_hf, args.variation)
        score_hf = compute_score_from_sympy_x(expr_hf, Xhf)
        score[m_hf] = score_hf
        pass_mask[m_hf] = apply_wp(score_hf, thr_hf, dir_hf)

    print("score finite fraction:", np.isfinite(score).mean(), "N=", len(score))

    if he_model is not None and m_he.any():
        plot_score_real_fake(
            score[m_he], y_hs[m_he],
            os.path.join(args.outdir, "score_real_fake_HE.pdf"), "HE"
        )

    if hf_model is not None and m_hf.any():
        plot_score_real_fake(
            score[m_hf], y_hs[m_hf],
            os.path.join(args.outdir, "score_real_fake_HF.pdf"), "HF"
        )

    he_cfg = None
    if he_model is not None:
        if he_eta_binned is not None:
            he_cfg = {"eta_binned": True, "thr_json": he_eta_binned}
        else:
            he_cfg = {"eta_binned": False, "thr": thr_he, "dir": dir_he}

    hf_cfg = None if hf_model is None else {"thr": thr_hf, "dir": dir_hf}

    plot_wp_performance_vs_pt(
        dfj, score, args.outdir,
        he_cfg=he_cfg,
        hf_cfg=hf_cfg,
    )

    for region, mask in [("HE", m_he), ("HF", m_hf)]:
        if not mask.any():
            continue

        vals_eta = dfj.loc[mask, "eta"].to_numpy()
        vals_pt = dfj.loc[mask, "pt"].to_numpy()
        vals_score = score[mask]
        ys = y_hs[mask]
        pm = pass_mask[mask]

        plot_stacked_before_after(
            vals_pt, ys, pm,
            os.path.join(args.outdir, f"stack_{region}_pt"),
            f"{region} jet p_{{T}}", "jet p_{T} [GeV]",
            nbins=40, xmin=args.pt_min, xmax=args.pt_turnoff, logy=False
        )
        plot_stacked_before_after(
            vals_eta, ys, pm,
            os.path.join(args.outdir, f"stack_{region}_eta"),
            f"{region} jet #eta", "jet #eta",
            nbins=80, xmin=-5.0, xmax=5.0, logy=False
        )
        xmin_s, xmax_s = infer_range(vals_score[np.isfinite(vals_score)])
        plot_stacked_before_after(
            vals_score, ys, pm,
            os.path.join(args.outdir, f"stack_{region}_score"),
            f"{region} score", "score",
            nbins=50, xmin=xmin_s, xmax=xmax_s, logy=False
        )

    plot_jet_eta_before_after_mask(dfj, pass_mask, args.outdir)

    def score_for_two_vbf_jets(df_in, variation):
        n = len(df_in)
        p1 = f"{args.vbf_jet_base}1_"
        p2 = f"{args.vbf_jet_base}2_"

        def score_one(prefix, jidx_val):
            tmp = pd.DataFrame({
                "pt": pd.to_numeric(df_in[col(prefix, "pt", variation)], errors="coerce").astype(np.float32),
                "eta": pd.to_numeric(df_in[col(prefix, "eta", variation)], errors="coerce").astype(np.float32),
            })
            tmp["aeta"] = np.abs(tmp["eta"])
            tmp["y_hs"] = 0
            tmp["jidx"] = float(jidx_val)
            tmp["prefix"] = prefix
            tmp["evt_idx"] = np.arange(n, dtype=np.int64)

            aeta_tmp = tmp["aeta"].to_numpy()
            mm_he = region_mask(aeta_tmp, "HE")
            mm_hf = region_mask(aeta_tmp, "HF")

            s = np.full(n, np.nan, dtype=np.float32)
            if he_model is not None and mm_he.any():
                Xhe = build_feature_matrix(df_in, tmp.loc[mm_he].copy(), feats_he, variation)
                s[mm_he] = compute_score_from_sympy_x(expr_he, Xhe)

            if hf_model is not None and mm_hf.any():
                Xhf = build_feature_matrix(df_in, tmp.loc[mm_hf].copy(), feats_hf, variation)
                s[mm_hf] = compute_score_from_sympy_x(expr_hf, Xhf)

            return s

        return score_one(p1, 0), score_one(p2, 1)

    plot_mjj_deta_before_after(
        df_in, args.variation, args.outdir,
        vbf_base=args.vbf_jet_base,
        score_func_for_twojets=score_for_two_vbf_jets,
        he_cfg=he_cfg,
        hf_cfg=hf_cfg,
        pt_min=args.pt_min, pt_turnoff=args.pt_turnoff,
        mjj_col=args.mjj_col, deta_col=args.deta_col
    )


if __name__ == "__main__":
    main()