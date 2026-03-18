#!/usr/bin/env python3
import os
import glob
import argparse
from pathlib import Path

import numpy as np
import awkward as ak
import dask_awkward as dak
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from modules.trials import get_stage1_path
from modules.utils import logger
from modules import selection

from modules.dask_utils import close_dask_client, get_dask_client


# ============================================================
# Validation Helpers
# ============================================================
def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

@torch.no_grad()
def predict_logits_numpy(model, X_np, batch_size=262144, device=None):
    """Return logits for a numpy feature matrix."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    X_t = torch.from_numpy(X_np.astype(np.float32))
    out = np.empty((X_np.shape[0],), dtype=np.float32)

    for i in range(0, X_np.shape[0], batch_size):
        xb = X_t[i:i+batch_size].to(device, non_blocking=True)
        logits = model(xb).squeeze(1).detach().cpu().numpy().astype(np.float32)
        out[i:i+batch_size] = logits
    return out

def weighted_accuracy(y_true, y_prob, w, thr=0.5):
    y_pred = (y_prob >= thr).astype(np.int32)
    correct = (y_pred == y_true.astype(np.int32)).astype(np.float32)
    return float(np.sum(w * correct) / max(np.sum(w), 1e-12))

def weighted_confusion_matrix(y_true, y_prob, w, thr=0.5):
    """Returns [[TN, FP],[FN, TP]] with weights."""
    y = y_true.astype(np.int32)
    p = (y_prob >= thr).astype(np.int32)

    tn = np.sum(w[(y == 0) & (p == 0)])
    fp = np.sum(w[(y == 0) & (p == 1)])
    fn = np.sum(w[(y == 1) & (p == 0)])
    tp = np.sum(w[(y == 1) & (p == 1)])
    return np.array([[tn, fp], [fn, tp]], dtype=np.float64)

def plot_loss_acc(history, outdir):
    os.makedirs(outdir, exist_ok=True)
    ep = np.arange(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure()
    plt.plot(ep, history["train_loss"], label="train")
    plt.plot(ep, history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("weighted BCE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_vs_epoch.png"), dpi=160)
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(ep, history["train_acc"], label="train")
    plt.plot(ep, history["val_acc"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("weighted accuracy (thr=0.5)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "acc_vs_epoch.png"), dpi=160)
    plt.close()

# def plot_roc(y_true, y_prob, w, outdir, tag="val"):
#     os.makedirs(outdir, exist_ok=True)
#     fpr, tpr, _ = roc_curve(y_true, y_prob, sample_weight=w)
#     roc_auc = auc(fpr, tpr)

#     plt.figure()
#     plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
#     plt.xlabel("FPR")
#     plt.ylabel("TPR")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(outdir, f"roc_{tag}.png"), dpi=160)
#     plt.close()
#     return float(roc_auc)


# def plot_roc(y_true, y_prob, w, outdir, tag="val"):
#     os.makedirs(outdir, exist_ok=True)

#     # Most robust: directly compute AUC (handles weights safely)
#     roc_auc = roc_auc_score(y_true, y_prob, sample_weight=w)

#     # For the curve, roc_curve is fine, but may produce a non-monotonic fpr due to numerics.
#     fpr, tpr, _ = roc_curve(y_true, y_prob, sample_weight=w)

#     # Enforce monotonic x for plotting + trapezoid sanity
#     order = np.argsort(fpr)
#     fpr = fpr[order]
#     tpr = tpr[order]
#     # Make tpr non-decreasing too (optional but stabilizes weird jaggedness)
#     tpr = np.maximum.accumulate(tpr)

#     plt.figure()
#     plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
#     plt.xlabel("FPR")
#     plt.ylabel("TPR")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(outdir, f"roc_{tag}.png"), dpi=160)
#     plt.close()

#     return float(roc_auc)

def weighted_roc_curve_simple(y_true, y_score, w):
    """
    Robust weighted ROC computed by sorting scores (no sklearn).
    Returns fpr, tpr arrays that are monotonic by construction.
    """
    y = y_true.astype(np.int32)
    s = y_score.astype(np.float64)
    w = w.astype(np.float64)

    # sort by score descending
    order = np.argsort(-s)
    y = y[order]
    w = w[order]

    # total pos/neg weight
    w_pos = w[y == 1].sum()
    w_neg = w[y == 0].sum()
    w_pos = max(w_pos, 1e-15)
    w_neg = max(w_neg, 1e-15)

    # cumulative TP/FP weights
    tp = np.cumsum(w * (y == 1))
    fp = np.cumsum(w * (y == 0))

    tpr = tp / w_pos
    fpr = fp / w_neg

    # add (0,0) and (1,1)
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])

    # enforce monotonic non-decreasing fpr (should already be, but keep safe)
    order2 = np.argsort(fpr)
    fpr = fpr[order2]
    tpr = tpr[order2]
    tpr = np.maximum.accumulate(tpr)

    return fpr, tpr


def weighted_auc_trapz(fpr, tpr):
    # fpr is sorted; compute trapezoid integral
    return float(np.trapz(tpr, fpr))


def plot_roc(y_true, y_prob, w, outdir, tag="val"):
    os.makedirs(outdir, exist_ok=True)

    fpr, tpr = weighted_roc_curve_simple(y_true, y_prob, w)
    roc_auc = weighted_auc_trapz(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"roc_{tag}.png"), dpi=160)
    plt.close()

    return roc_auc


def plot_confusion(cm, outdir, tag="val"):
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.xticks([0, 1], ["Pred DY(0)", "Pred Data(1)"])
    plt.yticks([0, 1], ["True DY(0)", "True Data(1)"])
    plt.colorbar()
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, f"{v:.2e}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"confusion_{tag}.png"), dpi=160)
    plt.close()

def plot_score_distributions(y_true, y_prob, w, outdir, tag="val"):
    """Plots p(Data|x) distributions for DY and Data."""
    os.makedirs(outdir, exist_ok=True)
    bins = np.linspace(0, 1, 51)

    plt.figure()
    for cls, name in [(0, "DY (label=0)"), (1, "Data (label=1)")]:
        mask = (y_true == cls)
        hist, _ = np.histogram(y_prob[mask], bins=bins, weights=w[mask], density=True)
        centers = 0.5 * (bins[:-1] + bins[1:])
        plt.plot(centers, hist, label=name)
    plt.xlabel("p(Data|x)")
    plt.ylabel("weighted density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"score_dist_{tag}.png"), dpi=160)
    plt.close()

def plot_calibration_curve(y_true, y_prob, w, outdir, tag="val", n_bins=20):
    """
    Weighted reliability diagram: compares predicted prob vs empirical fraction.
    Useful because you use p/(1-p) as density-ratio.
    """
    os.makedirs(outdir, exist_ok=True)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    prob_mean = np.zeros(n_bins, dtype=np.float64)
    frac_pos  = np.zeros(n_bins, dtype=np.float64)
    tot_w     = np.zeros(n_bins, dtype=np.float64)

    for b in range(n_bins):
        m = (bin_ids == b)
        if not np.any(m):
            continue
        wb = w[m].astype(np.float64)
        tot = wb.sum()
        tot_w[b] = tot
        prob_mean[b] = np.sum(wb * y_prob[m]) / max(tot, 1e-12)
        frac_pos[b]  = np.sum(wb * y_true[m]) / max(tot, 1e-12)

    # keep only non-empty
    m = tot_w > 0
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.scatter(prob_mean[m], frac_pos[m])
    plt.xlabel("mean predicted p(Data|x)")
    plt.ylabel("empirical fraction(Data)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"calibration_{tag}.png"), dpi=160)
    plt.close()

def plot_weight_distribution(y_true, y_prob, w_evt,
                             outdir, tag="val",
                             clip=None, eps=1e-6):
    """
    Plots DY weight distribution:
      w = p/(1-p)
    Only for DY events (label=0).

    Parameters:
      y_true : true labels
      y_prob : predicted p(Data|x)
      w_evt  : original event weights (used only for histogram weighting)
    """
    import os
    os.makedirs(outdir, exist_ok=True)

    # Only DY events (label=0)
    mask = (y_true == 0)
    if mask.sum() == 0:
        print("No DY events found for weight plot.")
        return

    p = np.clip(y_prob[mask], eps, 1 - eps)
    w_dnn = p / (1.0 - p)

    if clip is not None:
        w_dnn = np.clip(w_dnn, clip[0], clip[1])

    w_hist = w_evt[mask]

    # ---- Linear scale ----
    plt.figure()
    bins = np.linspace(0, min(10, w_dnn.max()), 100)
    plt.hist(w_dnn, bins=bins, weights=w_hist,
             histtype="step", density=True)
    plt.xlabel("DNN weight w = p/(1-p)")
    plt.ylabel("Weighted density")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"weight_dist_linear_{tag}.png"), dpi=160)
    plt.close()

    # ---- Log scale ----
    plt.figure()
    bins = np.logspace(-2, np.log10(max(10, w_dnn.max())), 100)
    plt.hist(w_dnn, bins=bins, weights=w_hist,
             histtype="step", density=True)
    plt.xscale("log")
    plt.xlabel("DNN weight w = p/(1-p)")
    plt.ylabel("Weighted density")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"weight_dist_log_{tag}.png"), dpi=160)
    plt.close()

    # ---- Print diagnostics ----
    print(f"\n[Weight diagnostics - {tag}]")
    print(f"  min   = {w_dnn.min():.4f}")
    print(f"  p1    = {np.percentile(w_dnn,1):.4f}")
    print(f"  median= {np.median(w_dnn):.4f}")
    print(f"  mean  = {w_dnn.mean():.4f}")
    print(f"  p99   = {np.percentile(w_dnn,99):.4f}")
    print(f"  max   = {w_dnn.max():.4f}")
    print(f"  frac(w>5)  = {(w_dnn>5).mean():.6f}")
    print(f"  frac(w>10) = {(w_dnn>10).mean():.6f}")


def plot_data_mc_before_after(
    data_vals: np.ndarray,
    dy_vals: np.ndarray,
    w_data: np.ndarray,
    w_dy: np.ndarray,
    w_dy_dnn: np.ndarray,
    outpath: str,
    title: str,
    bins=60,
    range=None,
    logy=False,
):
    import os
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    # histograms
    hD, edges = np.histogram(data_vals, bins=bins, range=range, weights=w_data)
    hB, _     = np.histogram(dy_vals,   bins=edges,              weights=w_dy)
    hA, _     = np.histogram(dy_vals,   bins=edges,              weights=w_dy * w_dy_dnn)

    centers = 0.5 * (edges[:-1] + edges[1:])

    # ratio
    eps = 1e-12
    rB = hD / np.maximum(hB, eps)
    rA = hD / np.maximum(hA, eps)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7.2, 6.8))

    # top: shapes
    ax1 = plt.subplot(2, 1, 1)
    ax1.step(centers, hD, where="mid", label="Data", linewidth=1.5)
    ax1.step(centers, hB, where="mid", label="DY (before)", linewidth=1.5)
    ax1.step(centers, hA, where="mid", label="DY (after)", linewidth=1.5)
    ax1.set_ylabel("Weighted events")
    ax1.set_title(title)
    ax1.legend()
    if logy:
        ax1.set_yscale("log")

    # bottom: ratios
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.step(centers, rB, where="mid", label="Data/DY(before)")
    ax2.step(centers, rA, where="mid", label="Data/DY(after)")
    ax2.axhline(1.0, linestyle="--")
    ax2.set_xlabel(title)
    ax2.set_ylabel("Ratio")
    ax2.set_ylim(0.5, 1.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

# ============================================================
# Helpers
# ============================================================
def list_good_parquet_files(parquet_dir: str):
    import glob, os
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        files = sorted(glob.glob(os.path.join(parquet_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files under {parquet_dir}")

    good = []
    bad = []
    for f in files:
        try:
            pq.ParquetFile(f)  # schema read only
            good.append(f)
        except Exception as e:
            bad.append((f, repr(e)))

    if bad:
        logger.warning(f"Found {len(bad)} bad parquet files in {parquet_dir}. Example: {bad[0]}")
    if not good:
        raise RuntimeError(f"All parquet files failed in {parquet_dir}")
    return good

def discover_sample_dirs(load_path: Path, data_glob="data_*", dy_glob="dy*"):
    data_dirs = sorted([p for p in load_path.glob(data_glob) if p.is_dir()])
    dy_dirs   = sorted([p for p in load_path.glob(dy_glob)   if p.is_dir()])
    return data_dirs, dy_dirs


def find_first_existing_field(fields: list[str], candidates: list[str]) -> str:
    fs = set(fields)
    for c in candidates:
        if c in fs:
            return c
    raise KeyError(f"None of these fields exist: {candidates}\nAvailable fields: {sorted(list(fs))[:80]} ...")


def compute_acoplanarity_dak(mu1_phi, mu2_phi):
    # aco = 1 - |Δφ|/π, with Δφ in [0,π]
    dphi = np.abs(mu1_phi - mu2_phi)
    dphi = dak.where(dphi > np.pi, 2.0 * np.pi - dphi, dphi)
    return 1.0 - (dphi / np.pi)


def filter_region_zpeak_dak(events_dak: dak.Array) -> dak.Array:
    """
    Apply ONLY selection.filterRegion(..., region="z-peak") on each partition.
    This assumes selection.filterRegion takes awkward arrays.
    """
    def _f(part: ak.Array) -> ak.Array:
        # returns (mask, filtered_events) in your code
        _, out = selection.filterRegion(part, region="z-peak")
        return out

    # meta is optional; dask_awkward often infers, but we keep it simple
    return dak.map_partitions(_f, events_dak)


def dak_to_numpy_1d(x_dak: dak.Array) -> np.ndarray:
    """
    Compute a 1D dask_awkward array -> numpy float32, nan-safe.
    """
    x = x_dak.compute()
    x = ak.to_numpy(x).astype(np.float32, copy=False)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def filter_njet_bin_dak(events_dak: dak.Array, njet_bin: str) -> dak.Array:
    if njet_bin == "incl":
        return events_dak

    head = events_dak[:1].compute()
    fields = list(head.fields)
    njet_name = find_first_existing_field(fields, ["njets_nominal", "njets", "Njets"])

    njet = events_dak[njet_name]
    if njet_bin == "0":
        return events_dak[njet == 0]
    if njet_bin == "1":
        return events_dak[njet == 1]
    if njet_bin == "2p":
        return events_dak[njet >= 2]
    raise ValueError(njet_bin)

def build_features_and_weights_dak(
    events_dak: dak.Array,
    is_data: bool,
    feature_names=None,
    weight_field_candidates=("wgt_nominal", "weight", "evt_weight", "genWeight"),
):
    """
    events_dak is already filtered to z-peak.
    Returns:
      X: (N, n_features) numpy float32
      w: (N,)            numpy float32

    Default features (as requested):
      ["mu1_pt","mu2_pt","mu1_eta","mu2_eta","acoplanarity","dimuon_pt","dimuon_rapidity"]
    """
    if feature_names is None:
        feature_names = [
            "mu1_pt",
            "mu2_pt",
            "mu1_eta",
            "mu2_eta",
            "acoplanarity",
            "dimuon_pt",
            "dimuon_rapidity",
        ]

    # Inspect fields from a tiny computed head (cheap)
    head = events_dak[:1].compute()
    fields = list(head.fields)

    # ---- Resolve aliases for muon vars ----
    mu1_pt_name  = find_first_existing_field(fields, ["mu1_pt", "Muon1_pt", "lead_mu_pt", "muon1_pt"])
    mu2_pt_name  = find_first_existing_field(fields, ["mu2_pt", "Muon2_pt", "sublead_mu_pt", "muon2_pt"])
    mu1_eta_name = find_first_existing_field(fields, ["mu1_eta", "Muon1_eta", "lead_mu_eta", "muon1_eta"])
    mu2_eta_name = find_first_existing_field(fields, ["mu2_eta", "Muon2_eta", "sublead_mu_eta", "muon2_eta"])

    # ---- acoplanarity: use stored if present; else compute from phi ----
    if "acoplanarity" in fields:
        aco_dak = events_dak["acoplanarity"]
    else:
        mu1_phi_name = find_first_existing_field(fields, ["mu1_phi", "Muon1_phi", "lead_mu_phi", "muon1_phi"])
        mu2_phi_name = find_first_existing_field(fields, ["mu2_phi", "Muon2_phi", "sublead_mu_phi", "muon2_phi"])
        aco_dak = compute_acoplanarity_dak(events_dak[mu1_phi_name], events_dak[mu2_phi_name])

    # ---- dimuon pt aliases ----
    dimuon_pt_name = None
    for c in ["dimuon_pt", "dimuonPt", "mm_pt", "mmpt", "dimuon_pT"]:
        if c in fields:
            dimuon_pt_name = c
            break
    if dimuon_pt_name is None:
        raise KeyError(
            "Requested 'dimuon_pt' but could not find it. "
            f"Available fields: {sorted(fields)[:120]} ..."
        )

    # ---- dimuon rapidity aliases ----
    dimuon_y_name = None
    for c in ["dimuon_rapidity", "dimuon_y", "dimuonRapidity", "mm_y", "mmy", "dimuonRap"]:
        if c in fields:
            dimuon_y_name = c
            break
    if dimuon_y_name is None:
        raise KeyError(
            "Requested 'dimuon_rapidity' but could not find it. "
            f"Available fields: {sorted(fields)[:120]} ..."
        )

    # ---- Map canonical feature names -> dak arrays ----
    feature_map = {
        "mu1_pt": events_dak[mu1_pt_name],
        "mu2_pt": events_dak[mu2_pt_name],
        "mu1_eta": events_dak[mu1_eta_name],
        "mu2_eta": events_dak[mu2_eta_name],
        "acoplanarity": aco_dak,
        "dimuon_pt": events_dak[dimuon_pt_name],
        "dimuon_rapidity": events_dak[dimuon_y_name],
    }

    # ---- Stack features in the requested order ----
    cols = []
    for name in feature_names:
        if name not in feature_map:
            raise KeyError(
                f"Requested feature '{name}' not available. "
                f"Valid keys: {list(feature_map.keys())}"
            )
        cols.append(dak_to_numpy_1d(feature_map[name]))

    X = np.stack(cols, axis=1).astype(np.float32)

    # ---- Event weights ----
    if is_data:
        w = np.ones((X.shape[0],), dtype=np.float32)
    else:
        wname = None
        for c in weight_field_candidates:
            if c in fields:
                wname = c
                break
        if wname is None:
            raise KeyError(
                f"No DY weight field found. Tried {weight_field_candidates}. "
                f"Available fields: {sorted(fields)[:120]} ..."
            )
        # w = dak_to_numpy_1d(events_dak[wname]).reshape(-1).astype(np.float32)

        w_nom = dak_to_numpy_1d(events_dak[wname]).reshape(-1).astype(np.float32)
        w_nom = np.abs(w_nom)

        # Remove ZpT reweighting
        if "separate_wgt_zpt_wgt" in fields:
            w_zpt = dak_to_numpy_1d(
                events_dak["separate_wgt_zpt_wgt"]
            ).reshape(-1).astype(np.float32)

            # Protect against bad values
            eps = 1e-12
            w_zpt = np.nan_to_num(w_zpt, nan=1.0, posinf=1.0, neginf=1.0)
            w_zpt = np.where(np.abs(w_zpt) < eps, 1.0, w_zpt)

            w = w_nom / w_zpt

            logger.info(
                f"Removed ZpT weight: w = {wname} / separate_wgt_zpt_wgt"
            )
        else:
            logger.warning(
                "Field 'separate_wgt_zpt_wgt' not found — using nominal weight as-is."
            )
            w = w_nom


    return X, w

# ============================================================
# PyTorch bits (same as before)
# ============================================================
class StandardScalerNP:
    def __init__(self, eps: float = 1e-12):
        self.mean_ = None
        self.std_ = None
        self.eps = eps

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0).astype(np.float32)
        self.std_ = X.std(axis=0).astype(np.float32)
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean_, "std": self.std_, "eps": float(self.eps)}


class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).view(-1, 1)
        self.w = torch.from_numpy(w.astype(np.float32)).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden=(64, 64, 32), dropout=0.10):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [
                nn.Linear(d, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def weighted_val_loss(model, loader, device):
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    tot, totw = 0.0, 0.0
    for X, y, w in loader:
        X, y, w = X.to(device), y.to(device), w.to(device)
        logits = model(X)
        loss_evt = bce(logits, y) * w
        tot += loss_evt.sum().item()
        totw += w.sum().item()
    return tot / max(totw, 1e-12)


def balance_effective_class_weights(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = w.copy().astype(np.float32)
    s_data = w[y == 1].sum()
    s_dy   = w[y == 0].sum()
    if s_data > 0 and s_dy > 0:
        w[y == 0] *= (s_data / s_dy)
    return w


def train_model(X_tr, y_tr, w_tr, X_va, y_va, w_va,
                outdir, lr=1e-3, batch=4096, epochs=40, patience=6,
                hidden=(64, 64, 32), dropout=0.10, seed=123):
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device is cuda?: {device}")

    logger.info("Loading the data using dataloader")
    tr_loader = DataLoader(NumpyDataset(X_tr, y_tr, w_tr), batch_size=batch, shuffle=True)
    va_loader = DataLoader(NumpyDataset(X_va, y_va, w_va), batch_size=batch, shuffle=False)

    logger.info("Model...")
    model = MLP(in_dim=X_tr.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    best = float("inf")
    bad = 0
    best_path = os.path.join(outdir, "model_best_state.pt")

    logger.info("Starting to train")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for ep in range(1, epochs + 1):
        model.train()
        tot, totw = 0.0, 0.0

        for X, y, w in tr_loader:
            X, y, w = X.to(device), y.to(device), w.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(X)
            loss_evt = bce(logits, y) * w
            loss = loss_evt.sum() / torch.clamp(w.sum(), min=1.0)
            loss.backward()
            opt.step()

            tot += loss_evt.sum().item()
            totw += w.sum().item()

        tr_loss = tot / max(totw, 1e-12)
        va_loss = weighted_val_loss(model, va_loader, device)


        if va_loss < best - 1e-6:
            best = va_loss
            bad = 0
            torch.save({"model_state": model.state_dict()}, best_path)
        else:
            bad += 1
            if bad >= patience:
                logger.info(f"Early stopping at epoch {ep} (best val_loss={best:.6f})")
                break


        # after computing tr_loss and va_loss (each epoch)
        logits_tr = predict_logits_numpy(model, X_tr, device=device)
        p_tr = sigmoid_np(logits_tr)
        train_acc = weighted_accuracy(y_tr, p_tr, w_tr, thr=0.5)

        logits_va = predict_logits_numpy(model, X_va, device=device)
        p_va = sigmoid_np(logits_va)
        val_acc = weighted_accuracy(y_va, p_va, w_va, thr=0.5)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        logger.info(f"epoch {ep:03d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f} "
                    f"| train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

    # ckpt = torch.load(best_path, map_location="cpu")
    ckpt = torch.load(best_path, map_location="cpu", weights_only=True)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, history


def save_artifacts(model, scaler: StandardScalerNP, feature_order: list[str], outdir: str):
    torch.save(
        {"model_state": model.state_dict(), "scaler": scaler.state_dict(), "feature_order": feature_order},
        os.path.join(outdir, "model_and_scaler.pt"),
    )
    example = torch.zeros(1, len(feature_order), dtype=torch.float32)
    ts = torch.jit.trace(model.cpu(), example)
    ts.save(os.path.join(outdir, "model_ts.pt"))
    np.savez(
        os.path.join(outdir, "scaler.npz"),
        mean=scaler.mean_.astype(np.float32),
        std=scaler.std_.astype(np.float32),
        feature_order=np.array(feature_order),
    )
    logger.info(f"Saved to {outdir}/model_ts.pt and {outdir}/scaler.npz")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--year", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--data-glob", default="data_*")
    parser.add_argument("--dy-glob", default="dyTo2Mu_M-50_aMCatNLO")

    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--batch", default=4096, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--patience", default=6, type=int)
    parser.add_argument("--seed", default=123, type=int)

    parser.add_argument("--max-events", default=None, type=int,
                        help="Optional cap on total events (after z-peak) for quick tests.")

    parser.add_argument(
        "--njet-bin", default="incl",
        choices=["incl", "0", "1", "2p"],
        help="Train model in njet bin: 0, 1, or 2p (>=2)"
    )
    parser.add_argument("--use-gateway", action="store_true", help="Use Dask gateway")
    parser.add_argument("--cluster-index", default=1, type=int, help="Cluster index for Dask gateway")

    args = parser.parse_args()
    client = get_dask_client(args.use_gateway, cluster_index=args.cluster_index)
    feature_order = [
            "mu1_pt",
            "mu2_pt",
            "mu1_eta",
            "mu2_eta",
            "acoplanarity",
            "dimuon_pt",
            "dimuon_rapidity"]
    stage1_dir = Path(get_stage1_path())
    load_path = stage1_dir / args.year / "f1_0"
    logger.info(f"Using LOAD_PATH: {load_path}")

    data_dirs, dy_dirs = discover_sample_dirs(load_path, data_glob=args.data_glob, dy_glob=args.dy_glob)
    logger.info(f"Found data dirs: {[d.name for d in data_dirs]}")
    logger.info(f"Found dy dirs:   {[d.name for d in dy_dirs]}")

    X_data_list, w_data_list = [], []
    for d in data_dirs:
        logger.info(f"Reading DATA: {d}")
        files = list_good_parquet_files(str(d))
        ev = dak.from_parquet(files)          # recursive inside directory
        ev = filter_region_zpeak_dak(ev)       # ONLY z-peak
        ev = filter_njet_bin_dak(ev, args.njet_bin)
        Xd, wd = build_features_and_weights_dak(ev, is_data=True, feature_names=feature_order)
        X_data_list.append(Xd)
        w_data_list.append(wd)

    X_dy_list, w_dy_list = [], []
    for d in dy_dirs:
        logger.info(f"Reading DY: {d}")
        files = list_good_parquet_files(str(d))
        ev = dak.from_parquet(files)
        ev = filter_region_zpeak_dak(ev)       # ONLY z-peak
        ev = filter_njet_bin_dak(ev, args.njet_bin)
        Xm, wm = build_features_and_weights_dak(ev, is_data=False, feature_names=feature_order)
        X_dy_list.append(Xm)
        w_dy_list.append(wm)

    X_data = np.concatenate(X_data_list, axis=0)
    X_dy   = np.concatenate(X_dy_list, axis=0)
    w_data = np.concatenate(w_data_list, axis=0)
    w_dy   = np.concatenate(w_dy_list, axis=0)

    neg_frac = np.mean(w_dy < 0)
    logger.info(f"DY weights: min={w_dy.min():.3e}, max={w_dy.max():.3e}, neg_frac={neg_frac:.6f}")
    logger.info(f"DY sum(w)={w_dy.sum():.6e}, sum(|w|)={np.abs(w_dy).sum():.6e}")

    logger.info(f"Z-peak events: data={X_data.shape[0]} dy={X_dy.shape[0]}")

    # Optional cap (useful for fast iterations)
    if args.max_events is not None:
        n = int(args.max_events)
        X_data, w_data = X_data[:n], w_data[:n]
        X_dy, w_dy     = X_dy[:n],   w_dy[:n]
        logger.info(f"After max-events cap: data={X_data.shape[0]} dy={X_dy.shape[0]}")

    # Labels and merge
    logger.info("label the data and merge")
    y_data = np.ones((X_data.shape[0],), dtype=np.float32)
    y_dy   = np.zeros((X_dy.shape[0],), dtype=np.float32)

    X = np.concatenate([X_data, X_dy], axis=0).astype(np.float32)
    y = np.concatenate([y_data, y_dy], axis=0).astype(np.float32)
    w = np.concatenate([w_data, w_dy], axis=0).astype(np.float32)

    # Balance class priors (recommended)
    logger.info("blance the classes")
    w = balance_effective_class_weights(y, w)

    # Shuffle & split
    logger.info("shuffle and split")
    rng = np.random.default_rng(args.seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    n_train = int(0.8 * len(idx))
    tr, va = idx[:n_train], idx[n_train:]

    X_raw_va = X[va]

    logger.info("transform the data such that it has mean = 0 and std 1")
    scaler = StandardScalerNP().fit(X[tr])
    X_tr = scaler.transform(X[tr])
    X_va = scaler.transform(X[va])

    os.makedirs(args.outdir, exist_ok=True)
    logger.info("Training the model")
    model, history = train_model(
        X_tr, y[tr], w[tr],
        X_va, y[va], w[va],
        outdir=args.outdir,
        lr=args.lr, batch=args.batch, epochs=args.epochs,
        patience=args.patience, seed=args.seed,
    )

    # ---- define aliases once (fixes y_va/y_tr undefined) ----
    y_tr = y[tr]
    w_tr = w[tr]
    y_va = y[va]
    w_va = w[va]

    checks_dir = os.path.join(args.outdir, "training_checks")
    os.makedirs(checks_dir, exist_ok=True)

    plot_loss_acc(history, checks_dir)

    # final predictions on val (best model)
    logits_va = predict_logits_numpy(model, X_va)
    p_va = sigmoid_np(logits_va)

    plot_weight_distribution(
        y_va,
        p_va,
        w_va,
        checks_dir,
        tag="val",
        clip=None   # or (0.1,10) if you want to see clipped behavior
    )

    roc_auc = plot_roc(y_va, p_va, w_va, checks_dir, tag="val")
    cm = weighted_confusion_matrix(y_va, p_va, w_va, thr=0.5)
    plot_confusion(cm, checks_dir, tag="val")
    plot_score_distributions(y_va, p_va, w_va, checks_dir, tag="val")
    plot_calibration_curve(y_va, p_va, w_va, checks_dir, tag="val")

    logger.info(f"Final val AUC = {roc_auc:.4f}")
    logger.info(f"Weighted confusion matrix (val, thr=0.5):\n{cm}")

    logits_tr = predict_logits_numpy(model, X_tr)
    p_tr = sigmoid_np(logits_tr)
    plot_roc(y_tr, p_tr, w_tr, checks_dir, tag="train")
    plot_score_distributions(y_tr, p_tr, w_tr, checks_dir, tag="train")

    plot_weight_distribution(
        y_tr,
        p_tr,
        w_tr,
        checks_dir,
        tag="train"
    )

    eps = 1e-6
    p_clip = np.clip(p_va, eps, 1-eps)
    w_dy = p_clip / (1 - p_clip)

    # look only at DY events in validation
    w_dy_only = w_dy[y_va == 0]
    logger.info(f"w_dy(val DY): min={w_dy_only.min():.3f}, median={np.median(w_dy_only):.3f}, "
                f"mean={w_dy_only.mean():.3f}, max={w_dy_only.max():.3f}, frac>10={(w_dy_only>10).mean():.4f}")

    # Build DNN weights on validation DY events
    eps = 1e-6
    p_clip = np.clip(p_va, eps, 1 - eps)
    w_dnn_all = p_clip / (1.0 - p_clip)
    # w_dnn_all = np.clip(w_dnn_all, 0.1, 10.0)  # recommended
    w_dnn_all = np.clip(w_dnn_all, 0.2, 5.0)  # recommended

    # split val arrays by class
    mask_data = (y_va == 1)
    mask_dy   = (y_va == 0)

    # val, DY only
    sum_before = np.sum(w_va[mask_dy])
    sum_after  = np.sum(w_va[mask_dy] * w_dnn_all[mask_dy])
    logger.info(f"DY sumw before={sum_before:.6e}, after={sum_after:.6e}, ratio={sum_after/sum_before:.4f}")

    # Pick columns by index in your feature_order
    i_mu1_pt = feature_order.index("mu1_pt")
    i_mu2_pt = feature_order.index("mu2_pt")
    i_aco    = feature_order.index("acoplanarity")
    i_mmpt   = feature_order.index("dimuon_pt")
    i_mmy    = feature_order.index("dimuon_rapidity")

    plots_dir = os.path.join(checks_dir, "closure_plots")

    # dimuon_pt
    plot_data_mc_before_after(
        data_vals=X_raw_va[mask_data, i_mmpt],
        dy_vals  =X_raw_va[mask_dy,   i_mmpt],
        w_data=w_va[mask_data],
        w_dy  =w_va[mask_dy],
        w_dy_dnn=w_dnn_all[mask_dy],
        outpath=os.path.join(plots_dir, "closure_dimuon_pt.png"),
        title="dimuon_pt (Z-CR, val)",
        bins=80,
        range=(0, 200),
    )

    # acoplanarity
    plot_data_mc_before_after(
        data_vals=X_raw_va[mask_data, i_aco],
        dy_vals  =X_raw_va[mask_dy,   i_aco],
        w_data=w_va[mask_data],
        w_dy  =w_va[mask_dy],
        w_dy_dnn=w_dnn_all[mask_dy],
        outpath=os.path.join(plots_dir, "closure_acoplanarity.png"),
        title="acoplanarity (Z-CR, val)",
        bins=120,
        range=(0, 0.2),
    )

    # dimuon_rapidity
    plot_data_mc_before_after(
        data_vals=X_raw_va[mask_data, i_mmy],
        dy_vals  =X_raw_va[mask_dy,   i_mmy],
        w_data=w_va[mask_data],
        w_dy  =w_va[mask_dy],
        w_dy_dnn=w_dnn_all[mask_dy],
        outpath=os.path.join(plots_dir, "closure_dimuon_rapidity.png"),
        title="dimuon_rapidity (Z-CR, val)",
        bins=60,
        range=(-2.5, 2.5),
    )
    logger.info("saving the training artifacts")
    save_artifacts(model, scaler, feature_order, args.outdir)
    logger.info("Done.")

    close_dask_client()

if __name__ == "__main__":
    main()
