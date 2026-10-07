#!/usr/bin/env python3
"""
Data-driven (CWoLa) PU/HS tagger for forward jets, trained on Z(mumu) + 1-jet data.

Instead of gen-matching labels, two data mixtures defined by the Z-jet balance are the
labels (Classification Without Labels, Metodiev/Nachman/Thaler, arXiv:1708.02949):
  - HS-enriched (label 1): |dphi(Z, jet)| > 2.8 and 0.5 < pT(jet)/pT(Z) < 1.5
  - PU-enriched (label 0): |dphi(Z, jet)| < 1.5
The optimal mixture classifier is the optimal HS/PU classifier only if the jet features
are independent of the balance within each class. Hence:
  - no balance/MET/muon-geometry/pT-scale features (see FORBIDDEN_FEATURES),
  - jets close to a Z muon are removed (FSR/muon overlap sits at small dphi only),
  - (pT, |eta|, nPV) are reweighted to a common shape between the two mixtures.

MC validation hook: the identical procedure is repeated on DY MC and scored against
gen-matching labels, next to a fully supervised model (same features/selection) and
puIdDisc. The data model's working point is set on DY MC gen-matched HS jets.

Output layout mirrors train_pu_dnn.py (<out>/<region>/{model_torchscript.pt,
scaler.json, summary_<region>.json}) so src/corrections/pu_dnn.py can load it for
HEpos/HEneg/HFpos/HFneg regions.

Example command:
--------------------
B=/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterJets_Sep22_tightPassLepVeto_DefaultjetPt25GeV/stage1_output/2024/compacted
python MVA_training/pileup_dnn/train_pu_cwola.py \
  --data "$B/data_*/*/*.parquet" \
  --mc "$B/dyTo2Mu_M-50_aMCatNLO/*/*.parquet" \
  -o validation/pu_cwola/run2024_baseline \
  --regions HE HF
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_curve

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))
import train_pu_dnn as base  # noqa: E402  (reuse loaders, MLP, scaler, training loop)
from modules.git_utils import get_git_commit, get_git_state  # noqa: E402


CWOLA_FEATURES = [
    "chEmEF", "chHEF", "neEmEF", "neHEF",
    "chMultiplicity", "neMultiplicity", "nConstituents",
    "area", "massOverPt",
    "hfadjacentEtaStripsSize", "hfcentralEtaStripSize",
    "hfsigmaEtaEta", "hfsigmaPhiPhi", "hfEmEF", "hfHEF",
]

# Correlated with the balance labels (MET/muon geometry, pT scale), or an MC-trained
# discriminant that itself uses pT/eta (puIdDisc, kept only as a ROC baseline).
FORBIDDEN_FEATURES = {
    "dphiZJet", "balance", "minDPhiMetJet", "maxDPhiMetJet", "absDPhiMetJet",
    "minDRJetMuon", "minAbsDEtaJetMuon", "pt", "logpt", "eta", "aeta", "mass",
    "rawFactor", "puIdDisc", "muEF", "nMuons",
    "muonSubtrFactor", "muonSubtrDeltaEta", "muonSubtrDeltaPhi",
}

EVENT_VARS = [
    "event", "dimuon_mass", "dimuon_pt", "dimuon_phi", "njets_nominal", "PV_npvs",
    "mu1_eta", "mu1_phi", "mu2_eta", "mu2_phi", "wgt_nominal",
]
JET_VARS = ["pt", "eta", "phi", "mass", "hasMatchedGenJet", "puIdDisc"] + [
    f for f in CWOLA_FEATURES if f != "massOverPt"
]

REGION_ETA_RANGE = {
    "HE": (2.5, 3.0), "HEpos": (2.5, 3.0), "HEneg": (2.5, 3.0),
    "HF": (3.0, 5.191), "HFpos": (3.0, 5.191), "HFneg": (3.0, 5.191),
}
MIX_HS, MIX_PU, MIX_MID = 1, 0, -1


# ------------------------------------------------------------------ selection / IO
def add_selection_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Z+jet selection")
    g.add_argument("--zmass", nargs=2, type=float, default=[76.0, 106.0])
    g.add_argument("--min-z-pt", type=float, default=30.0)
    g.add_argument("--pt-min", type=float, default=25.0)
    g.add_argument("--pt-max", type=float, default=50.0)
    g.add_argument("--min-abs-eta", type=float, default=2.5, help="Read only jets beyond this |eta|.")
    g.add_argument("--hs-dphi-min", type=float, default=2.8)
    g.add_argument("--hs-balance", nargs=2, type=float, default=[0.5, 1.5])
    g.add_argument("--pu-dphi-max", type=float, default=1.5)
    g.add_argument(
        "--min-dr-jet-muon", type=float, default=0.8,
        help="Drop jets near a Z muon: FSR/muon overlap only populates the small-dphi mixture.",
    )


def selection_config(args: Namespace) -> dict:
    keys = ("zmass", "min_z_pt", "pt_min", "pt_max", "min_abs_eta", "hs_dphi_min",
            "hs_balance", "pu_dphi_max", "min_dr_jet_muon")
    return {k: getattr(args, k) for k in keys}


def _read_one(path: str, sel: dict) -> pd.DataFrame | None:
    pf = pq.ParquetFile(path)
    names = set(pf.schema.names)
    jet_cols = [f"jet1_{v}_nominal" for v in JET_VARS if f"jet1_{v}_nominal" in names]
    if "jet1_pt_nominal" not in jet_cols:
        return None
    df = pf.read(columns=[c for c in EVENT_VARS if c in names] + jet_cols).to_pandas()
    df = df.rename(columns={c: c[len("jet1_"):-len("_nominal")] for c in jet_cols})
    keep = (
        (df["njets_nominal"] == 1)
        & df["dimuon_mass"].between(*sel["zmass"])
        & (df["dimuon_pt"] > sel["min_z_pt"])
        & (df["pt"] >= sel["pt_min"])
        & (df["pt"] < sel["pt_max"])
        & (df["eta"].abs() >= sel["min_abs_eta"])
    )
    df = df[keep].copy()
    sample = base.infer_sample_name(path)
    df["__sample_name"] = sample
    df["__sample_group"] = base.infer_sample_group(sample)
    return df


def load_zjet(patterns: list[str], sel: dict, max_files: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    paths = base.expand_inputs(patterns, use_glob=True)
    if max_files and len(paths) > max_files:
        paths = paths[:: len(paths) // max_files][:max_files]
    with ThreadPoolExecutor(max_workers=min(32, len(paths))) as pool:
        frames = [f for f in pool.map(lambda p: _read_one(p, sel), paths) if f is not None]
    if not frames:
        raise ValueError(f"No Z+1jet rows read from {patterns}")
    return add_zjet_vars(pd.concat(frames, ignore_index=True), sel), paths


def add_zjet_vars(df: pd.DataFrame, sel: dict) -> pd.DataFrame:
    df = base.cleanup_numeric(df)
    pt, eta, phi = (df[c].to_numpy(np.float32) for c in ("pt", "eta", "phi"))
    df["aeta"] = np.abs(eta)
    df["jidx"] = np.float32(0)
    df["dphiZJet"] = np.abs(base.delta_phi(phi, df["dimuon_phi"].to_numpy(np.float32)))
    df["balance"] = pt / df["dimuon_pt"].to_numpy(np.float32)
    df["massOverPt"] = df["mass"].to_numpy(np.float32) / pt
    drs = [
        base.delta_r(eta, phi, df[f"mu{i}_eta"].to_numpy(np.float32), df[f"mu{i}_phi"].to_numpy(np.float32))
        for i in (1, 2) if f"mu{i}_eta" in df.columns
    ]
    df["minDRJetMuon"] = np.nanmin(np.vstack(drs), axis=0) if drs else np.inf
    df = df[df["minDRJetMuon"] > sel["min_dr_jet_muon"]].copy()

    hs = (df["dphiZJet"] > sel["hs_dphi_min"]) & df["balance"].between(*sel["hs_balance"], inclusive="neither")
    pu = df["dphiZJet"] < sel["pu_dphi_max"]
    df["mix"] = np.select([hs, pu], [MIX_HS, MIX_PU], MIX_MID).astype(np.int8)
    df["w_phys"] = df["wgt_nominal"].astype(np.float32) if "wgt_nominal" in df.columns else np.float32(1.0)
    df["y_gen"] = (
        (df["hasMatchedGenJet"] > 0.5).astype(np.int8) if "hasMatchedGenJet" in df.columns else np.int8(-1)
    )
    ev = pd.to_numeric(df["event"], errors="coerce").fillna(0).astype(np.int64).abs() % 10
    df["is_test"] = ev.to_numpy() < 2  # same event-modulo split as base.split_by_event_or_random
    df["is_heldout"] = ev.to_numpy() < 4  # val + test: never used for gradient updates
    return df


def region_mask(df: pd.DataFrame, region: str) -> np.ndarray:
    return base.region_mask_eta(df["eta"].to_numpy(np.float32), region)


def eta_edges(region: str) -> np.ndarray:
    lo, hi = REGION_ETA_RANGE[region]
    inner = [e for e in base.ETA_RING_EDGES_FINE if lo < e < hi]
    return np.asarray([lo] + inner + [hi], dtype=np.float64)


# ------------------------------------------------------------------ weights / training
def decorrelation_weights(
    df: pd.DataFrame, label: np.ndarray, base_w: np.ndarray, region: str, args: Namespace
) -> np.ndarray:
    """Reweight both classes to their mean (pT, |eta|[, nPV]) shape, then equalize class sums.

    Bins populated by only one class get weight 0 so no class-exclusive phase space remains.
    """
    axes = [("pt", np.asarray(base.DEFAULT_PT_BINS, dtype=np.float64)), ("aeta", eta_edges(region))]
    if not args.no_npv_decorrelation and "PV_npvs" in df.columns:
        npv = df["PV_npvs"].to_numpy(np.float64)
        q = np.unique(np.nanquantile(npv, np.linspace(0, 1, args.npv_bins + 1)))
        q[0], q[-1] = -np.inf, np.inf
        axes.append(("PV_npvs", q))
    idx = np.zeros(len(df), dtype=np.int64)
    nbins = 1
    for col, edges in axes:
        b = np.clip(np.digitize(df[col].to_numpy(np.float64), edges) - 1, 0, len(edges) - 2)
        idx = idx * (len(edges) - 1) + b
        nbins *= len(edges) - 1

    label = np.asarray(label).astype(bool)
    w = np.abs(np.asarray(base_w, dtype=np.float64))
    hists = []
    for cls in (label, ~label):
        h = np.bincount(idx[cls], weights=w[cls], minlength=nbins)
        hists.append(h / max(h.sum(), 1e-12))
    both = (hists[0] > 0) & (hists[1] > 0)
    target = np.where(both, 0.5 * (hists[0] + hists[1]), 0.0)
    out = np.zeros_like(w)
    for cls, h in zip((label, ~label), hists):
        scale = np.divide(target, h, out=np.zeros_like(target), where=h > 0)
        scale = np.where(both, np.clip(scale, 1.0 / args.max_reweight, args.max_reweight), 0.0)
        out[cls] = w[cls] * scale[idx[cls]]
    for cls in (label, ~label):
        s = out[cls].sum()
        if s > 0:
            out[cls] /= s  # equal total weight per class
    return (out / out[out > 0].mean()).astype(np.float32)


def fit_model(df: pd.DataFrame, y: np.ndarray, w: np.ndarray, features: list[str], outdir: Path, args: Namespace):
    """Run base.train_one_model with precomputed weights (its own balancing switched off)."""
    work = df.copy()
    work["y_hs"] = np.asarray(y, dtype=np.int8)
    work["__cwola_w"] = w
    work = work[work["__cwola_w"] > 0]
    targs = Namespace(**vars(args))
    targs.use_weights = True
    targs.weight_col = "__cwola_w"
    targs.weight_clip = 1e30
    targs.no_sample_balance = True
    targs.no_class_balance = True
    targs.sample_balance_groups = []
    targs.pt_decorrelation_mode = "none"
    targs.pt_decorrelation_bins = []
    outdir.mkdir(parents=True, exist_ok=True)
    model, scaler, metrics, _ = base.train_one_model(work, features, outdir, targs)
    n_test_expected = int(work["is_test"].sum())
    if metrics["n_test"] != n_test_expected:
        print(f"[WARN] {outdir}: split fell back to random ({metrics['n_test']} vs {n_test_expected} test jets)")
    return model, scaler, metrics


def score(model, scaler: base.Scaler, df: pd.DataFrame, device, batch_size: int) -> np.ndarray:
    if len(df) == 0:
        return np.zeros(0, dtype=np.float32)
    return base.predict_scores(model, scaler.transform(df), device, batch_size)


def pick_features(df: pd.DataFrame, requested: list[str], allowed: set[str] = frozenset()) -> list[str]:
    bad = sorted(set(requested) & (FORBIDDEN_FEATURES - allowed))
    if bad:
        raise ValueError(f"Features correlated with the CWoLa labels are not allowed: {bad}")
    keep = []
    for f in requested:
        if f not in df.columns:
            continue
        v = df[f].to_numpy(np.float32)
        v = v[np.isfinite(v)]
        if len(v) and np.std(v) > 1e-8:
            keep.append(f)
    return keep


# ------------------------------------------------------------------ evaluation
def roc(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """HS efficiency / PU rejection curve; score sign flipped automatically if AUC < 0.5."""
    finite = np.isfinite(s)
    y, s, w = y[finite], s[finite], w[finite]
    auc = base.safe_auc(y, s, w)
    if np.isfinite(auc) and auc < 0.5:
        s, auc = -s, 1.0 - auc
    fpr, tpr, _ = roc_curve(y, s, sample_weight=w)
    return tpr, 1.0 - fpr, auc


def pu_rej_at(tpr: np.ndarray, rej: np.ndarray, hs_eff: float) -> float:
    order = np.argsort(tpr)
    return float(np.interp(hs_eff, tpr[order], rej[order]))


def mixture_summary(df: pd.DataFrame, is_mc: bool) -> dict:
    out = {}
    for name, code in (("HS_enriched", MIX_HS), ("PU_enriched", MIX_PU), ("middle", MIX_MID)):
        m = df["mix"].to_numpy() == code
        row = {"n_jets": int(m.sum())}
        if is_mc:
            w = df["w_phys"].to_numpy(np.float64)[m]
            g = df["y_gen"].to_numpy()[m] == 1
            row["sumw"] = float(w.sum())
            row["gen_hs_purity"] = float(w[g].sum() / w.sum()) if w.sum() else float("nan")
        out[name] = row
    return out


def score_correlations(df: pd.DataFrame, col: str) -> dict:
    """Within-gen-class correlation of the score with selection variables (want ~0)."""
    out = {}
    w = np.abs(df["w_phys"].to_numpy(np.float64))
    for cls_name, cls in (("gen_hs", 1), ("gen_pu", 0)):
        m = df["y_gen"].to_numpy() == cls
        out[cls_name] = {
            v: base.weighted_corr(df[col].to_numpy()[m], df[v].to_numpy()[m], w[m])
            for v in ("pt", "aeta", "PV_npvs", "dphiZJet", "balance") if v in df.columns
        }
    return out


def plot_roc(curves: dict, wp: tuple[float, float] | None, region: str, title: str, outbase: Path, formats: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for label, (tpr, rej, auc, style) in curves.items():
        ax.plot(tpr, rej, style, lw=1.8, label=f"{label} (AUC={auc:.3f})")
    if wp is not None:
        ax.plot([wp[0]], [wp[1]], "k*", ms=12, label=f"data-CWoLa WP: eff={wp[0]:.2f}, rej={wp[1]:.2f}")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("HS efficiency (DY MC, gen-matched)")
    ax.set_ylabel("PU rejection (DY MC, gen-unmatched)")
    ax.set_title(f"{region}: DY MC test split vs gen labels\n{title}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    base.save_plot(fig, outbase, formats)


def plot_score_datamc(data: pd.DataFrame, mc: pd.DataFrame, col: str, region: str, outbase: Path, formats: list[str]) -> None:
    bins = np.linspace(0, 1, 26)
    centers = 0.5 * (bins[1:] + bins[:-1])
    panels = [("HS-enriched", data["mix"] == MIX_HS, mc["mix"] == MIX_HS),
              ("PU-enriched", data["mix"] == MIX_PU, mc["mix"] == MIX_PU),
              ("all selected", np.ones(len(data), bool), np.ones(len(mc), bool))]
    fig, axes = plt.subplots(2, 3, figsize=(15, 5.8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for k, (title, md, mm) in enumerate(panels):
        md, mm = np.asarray(md), np.asarray(mm)
        hd, _ = np.histogram(data.loc[md, col], bins=bins)
        w = mc.loc[mm, "w_phys"].to_numpy(np.float64)
        g = mc.loc[mm, "y_gen"].to_numpy() == 1
        s = mc.loc[mm, col].to_numpy()
        h_hs, _ = np.histogram(s[g], bins=bins, weights=w[g])
        h_pu, _ = np.histogram(s[~g], bins=bins, weights=w[~g])
        norm = hd.sum() / max((h_hs + h_pu).sum(), 1e-12)
        h_hs, h_pu = h_hs * norm, h_pu * norm
        ax = axes[0, k]
        ax.hist([centers, centers], bins=bins, weights=[h_pu, h_hs], stacked=True,
                label=["MC gen-unmatched (PU)", "MC gen-matched (HS)"], color=["#d62728", "#1f77b4"], alpha=0.6)
        ax.errorbar(centers, hd, yerr=np.sqrt(hd), fmt="ko", ms=3, label="Data (held-out)")
        ax.set_title(f"{region} {title}")
        ax.set_ylabel("Jets (MC shape-normalized to data)")
        if k == 0:
            ax.legend(fontsize=8)
        tot = h_hs + h_pu
        ratio = np.divide(hd, tot, out=np.full_like(tot, np.nan), where=tot > 0)
        axes[1, k].errorbar(centers, ratio, yerr=np.divide(np.sqrt(hd), tot, out=np.zeros_like(tot), where=tot > 0), fmt="ko", ms=3)
        axes[1, k].axhline(1, color="gray", lw=1)
        axes[1, k].set_ylim(0.5, 1.5)
        axes[1, k].set_xlabel("data-CWoLa score")
        axes[1, k].set_ylabel("Data/MC")
    base.save_plot(fig, outbase, formats)


# ------------------------------------------------------------------ main
def parse_args() -> Namespace:
    p = argparse.ArgumentParser(description="Train a data-driven CWoLa PU/HS forward-jet tagger on Z+1jet events.")
    p.add_argument("--data", nargs="+", required=True, help="Data parquet globs (stage-1 compacted).")
    p.add_argument("--mc", nargs="+", required=True, help="DY MC parquet globs, used for validation vs gen labels.")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--regions", nargs="+", default=["HE", "HF"], choices=list(REGION_ETA_RANGE))
    p.add_argument("--features", nargs="+", default=CWOLA_FEATURES)
    p.add_argument(
        "--include-puIdDisc", action="store_true",
        help="Add puIdDisc as an input. Its pT/eta dependence is handled by the mixture reweighting.",
    )
    p.add_argument("--hs-eff", type=float, default=0.80, help="Working point, set on DY MC gen-matched HS jets.")
    p.add_argument("--no-npv-decorrelation", action="store_true")
    p.add_argument("--npv-bins", type=int, default=5)
    p.add_argument("--max-reweight", type=float, default=10.0)
    p.add_argument("--max-files-data", type=int, default=None)
    p.add_argument("--max-files-mc", type=int, default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=11)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--hidden", nargs="+", type=int, default=[64, 64, 32])
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--plot-format", nargs="+", default=["png", "pdf"])
    add_selection_args(p)
    return p.parse_args()


def train_region(region: str, data: pd.DataFrame, mc: pd.DataFrame, out: Path, args: Namespace) -> dict:
    rdir = out / region
    d = data[region_mask(data, region)].copy()
    m = mc[region_mask(mc, region)].copy()
    requested = args.features + (["puIdDisc"] if args.include_puIdDisc and "puIdDisc" not in args.features else [])
    feats = pick_features(d, requested, {"puIdDisc"} if args.include_puIdDisc else set())
    print(f"[{region}] data jets={len(d)}, DY MC jets={len(m)}, features={feats}")
    device = base.choose_device(args.device)

    # 1) the deliverable: CWoLa on data mixtures
    dl = d[d["mix"] != MIX_MID]
    y_dl = (dl["mix"] == MIX_HS).to_numpy()
    w_dl = decorrelation_weights(dl, y_dl, np.ones(len(dl)), region, args)
    model_d, scaler_d, met_d = fit_model(dl, y_dl, w_dl, feats, rdir, args)
    d["score_data_cwola"] = score(model_d, scaler_d, d, device, args.batch_size)
    m["score_data_cwola"] = score(model_d, scaler_d, m, device, args.batch_size)

    # 2) validation hook: identical CWoLa procedure on DY MC mixtures
    ml = m[m["mix"] != MIX_MID]
    y_ml = (ml["mix"] == MIX_HS).to_numpy()
    w_ml = decorrelation_weights(ml, y_ml, np.abs(ml["w_phys"].to_numpy()), region, args)
    model_mc, scaler_mc, met_mc = fit_model(ml, y_ml, w_ml, feats, rdir / "mc_cwola", args)
    m["score_mc_cwola"] = score(model_mc, scaler_mc, m, device, args.batch_size)

    # 3) reference: fully supervised on gen labels, same jets/features/decorrelation
    y_gen = (m["y_gen"] == 1).to_numpy()
    w_sup = decorrelation_weights(m, y_gen, np.abs(m["w_phys"].to_numpy()), region, args)
    model_s, scaler_s, met_s = fit_model(m, y_gen, w_sup, feats, rdir / "mc_supervised", args)
    m["score_mc_supervised"] = score(model_s, scaler_s, m, device, args.batch_size)

    # Evaluate on the MC test split (no MC model trained on it), all selected jets incl. "middle".
    mt = m[m["is_test"]]
    yt = (mt["y_gen"] == 1).to_numpy().astype(int)
    wt = np.abs(mt["w_phys"].to_numpy(np.float64))
    # "matched" reweights HS/PU test jets to a common (pT, |eta|, nPV) shape, so taggers that
    # exploit pT (puIdDisc) are compared at fixed kinematics like the decorrelated models.
    weightings = {
        "inclusive": (wt, "inclusive kinematics (physical weights)"),
        "matched": (decorrelation_weights(mt, yt.astype(bool), wt, region, args),
                    "HS/PU matched in pT, |eta|, nPV"),
    }
    thr, direction, hs_eff, pu_rej = base.threshold_and_direction(
        mt["score_data_cwola"].to_numpy(), yt, args.hs_eff, wt
    )
    rej_at_wp = {}
    for wname, (wv, title) in weightings.items():
        curves = {}
        for label, col, style in (
            ("data-CWoLa (deliverable)", "score_data_cwola", "b-"),
            ("DY-MC CWoLa (validation)", "score_mc_cwola", "g--"),
            ("DY-MC supervised, gen labels", "score_mc_supervised", "k:"),
            ("puIdDisc (baseline)", "puIdDisc", "r-."),
        ):
            if col not in mt.columns:
                continue
            tpr, rej, auc = roc(yt, mt[col].to_numpy(np.float64), wv)
            curves[label] = (tpr, rej, auc, style)
            rej_at_wp.setdefault(wname, {})[col] = {
                "auc": auc, f"pu_rejection_at_hs_eff_{args.hs_eff:.2f}": pu_rej_at(tpr, rej, args.hs_eff)
            }
        wp = (hs_eff, pu_rej) if wname == "inclusive" else None
        plot_roc(curves, wp, region, title, rdir / f"roc_{region}_{wname}", args.plot_format)
    dh = d[d["is_heldout"]]
    plot_score_datamc(dh, m, "score_data_cwola", region, rdir / f"score_datamc_{region}", args.plot_format)

    mix_auc = {
        "data_model_on_data_test": base.safe_auc(
            (d.loc[d["is_test"] & (d["mix"] != MIX_MID), "mix"] == MIX_HS).to_numpy().astype(int),
            d.loc[d["is_test"] & (d["mix"] != MIX_MID), "score_data_cwola"].to_numpy(),
        ),
        "mc_model_on_mc_test": base.safe_auc(
            (mt.loc[mt["mix"] != MIX_MID, "mix"] == MIX_HS).to_numpy().astype(int),
            mt.loc[mt["mix"] != MIX_MID, "score_mc_cwola"].to_numpy(),
            np.abs(mt.loc[mt["mix"] != MIX_MID, "w_phys"].to_numpy()),
        ),
    }

    keep_cols = ["event", "pt", "eta", "aeta", "PV_npvs", "dphiZJet", "balance", "mix", "y_gen", "w_phys",
                 "is_test", "puIdDisc", "score_data_cwola", "score_mc_cwola", "score_mc_supervised"]
    base.write_table(mt[[c for c in keep_cols if c in mt.columns]], rdir / "mc_test_scores.parquet")
    base.write_table(dh[[c for c in keep_cols if c in dh.columns]], rdir / "data_heldout_scores.parquet")

    summary = {
        "region": region,
        "model_type": "dnn_cwola",
        "model_path": str(rdir / "model_torchscript.pt"),
        "features": feats,
        "threshold": float(thr),
        "direction": direction,
        "hs_efficiency": float(hs_eff),
        "pu_rejection": float(pu_rej),
        "wp_defined_on": "DY MC test split, gen-matched HS jets",
        "pt_min": float(args.pt_min),
        "pt_turnoff": float(args.pt_max),
        "pt_decorrelation_mode": "pt_abseta" + ("" if args.no_npv_decorrelation else "_npv") + "_reweight",
        "selection": selection_config(args),
        "mixtures_data": mixture_summary(d, is_mc=False),
        "mixtures_dy_mc": mixture_summary(m, is_mc=True),
        "mixture_auc": mix_auc,
        "gen_label_performance_dy_mc_test": rej_at_wp,
        "score_correlations_dy_mc_test": score_correlations(mt, "score_data_cwola"),
        "training": {
            "data_cwola": {k: met_d[k] for k in ("n_train", "n_val", "n_test", "best_epoch", "val_auc", "test_auc")},
            "mc_cwola": {k: met_mc[k] for k in ("n_train", "n_val", "n_test", "best_epoch", "val_auc", "test_auc")},
            "mc_supervised": {k: met_s[k] for k in ("n_train", "n_val", "n_test", "best_epoch", "val_auc", "test_auc")},
        },
    }
    (rdir / f"summary_{region}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    base.set_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    git_state = get_git_state(out)
    git_commit = get_git_commit()
    (out / "git_state.json").write_text(json.dumps({"git_commit": git_commit, "git_state": git_state}, indent=2))

    sel = selection_config(args)
    data, data_paths = load_zjet(args.data, sel, args.max_files_data)
    mc, mc_paths = load_zjet(args.mc, sel, args.max_files_mc)
    print(f"Loaded Z+1jet forward jets: data={len(data)}, DY MC={len(mc)}")
    (out / "inputs.json").write_text(json.dumps({
        "data": args.data, "mc": args.mc, "n_data_files": len(data_paths), "n_mc_files": len(mc_paths),
        "data_files": data_paths, "mc_files": mc_paths, "git_commit": git_commit, "git_state": git_state,
    }, indent=2))
    (out / "selection.json").write_text(json.dumps(sel, indent=2))

    summaries = [train_region(r, data, mc, out, args) for r in args.regions]
    for s in summaries:
        (out / f"summary_{s['region']}.json").write_text(json.dumps(s, indent=2))
    (out / "summary_all.json").write_text(json.dumps(summaries, indent=2))
    print(f"Done. Wrote {len(summaries)} region summaries to {out}")


if __name__ == "__main__":
    main()
