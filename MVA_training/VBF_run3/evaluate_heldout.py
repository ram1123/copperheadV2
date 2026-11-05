#!/usr/bin/env python3
"""
Compare trained VBF DNNs on the held-out folds of one preprocess_dnn.py dataset: fold i is scored by each model's
fold-i network (--split evaluation: never trained on it; validation: early-stopped on it). Features are re-scaled
with each model's own scalers, so models from another preprocessing work too.
Reports per model: weighted AUC; binned AMS (|w| and signed, train_dnn.binned_ams) per fold and pooled; S, B, top
background processes and the cut-based-VBF share of the signal above each --top-effs signal efficiency.

Usage (CPU, inference only):
  python MVA_training/VBF_run3/evaluate_heldout.py --config configs/dnn_run2_vbf.yaml \
      --data-dir dnn/trained_models/<label>/<years>_h-peak_vbf_all --out heldout.json \
      --model NEW=dnn/trained_models/<label>/<years>_h-peak_vbf_all:trained_best_optuna_<tag> \
      --model OLD=<other model dir>:<its training tag>
"""
import argparse
import json
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import configs.categories.category_cuts as category_cuts  # noqa: E402
from train_dnn import binned_ams, load_config  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="DNN config; metrics.ams sets the AMS binning")
    p.add_argument("--data-dir", required=True, help="preprocess_dnn.py output whose folds are scored")
    p.add_argument("--model", action="append", required=True, metavar="LABEL=MODEL_DIR:TRAINING_TAG",
                   help="model dir holds scalers_<i>.npz and training_features.json; "
                        "<model dir>/<training tag>/fold<i>/best_torchscript.pt are the networks")
    p.add_argument("--split", choices=("evaluation", "validation"), default="evaluation")
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--top-effs", default="0.05,0.10,0.20", type=lambda s: [float(x) for x in s.split(",")],
                   help="signal-efficiency working points of the top_eff<x> report")
    p.add_argument("--out", required=True, help="output JSON")
    return p.parse_args(argv)


def unscaled_fold(data_dir: Path, split: str, fold: int) -> pd.DataFrame:
    df = pd.read_parquet(data_dir / f"data_df_{split}_{fold}.parquet")
    sc = np.load(data_dir / f"scalers_{fold}.npz", allow_pickle=True)
    raw = df[list(sc["final_features"])].astype(np.float64).copy()
    scaled = list(sc["features"])
    raw[scaled] = raw[scaled].values * sc["std"] + sc["mean"]
    raw["_label"], raw["_w"], raw["_process"], raw["_fold"] = df.label.values, df.wgt_nominal.values, df.process.values, fold
    return raw


def score(raw: pd.DataFrame, model_dir: Path, tag: str, fold: int) -> np.ndarray:
    sc = np.load(model_dir / f"scalers_{fold}.npz", allow_pickle=True)
    features = json.load(open(model_dir / "training_features.json"))
    x = raw.copy()
    scaled = list(sc["features"])
    x[scaled] = (x[scaled].values - sc["mean"]) / sc["std"]
    net = torch.jit.load(str(model_dir / tag / f"fold{fold}" / "best_torchscript.pt"), map_location="cpu").eval()
    with torch.no_grad():
        return net(torch.from_numpy(x[features].values.astype(np.float32))).reshape(-1).numpy()


def ams_pair(y, s, w, **kw):
    return {"ams_absWgt": binned_ams(y, s, w, **kw), "ams": binned_ams(y, s, w, signed=True, **kw)}


def vbf_category_mask(frame: pd.DataFrame) -> np.ndarray:
    """The cut-based VBF category cuts of configs/categories/categories.yml ("vbf"), from category_cuts."""
    names = yaml.safe_load(open(REPO / "configs" / "categories" / "categories.yml"))["vbf"]
    cuts = {c.name: c for c in vars(category_cuts).values() if isinstance(c, type) and "Custom" in c.__name__}
    events = ak.Array({c: frame[c].to_numpy() for c in frame.columns if c.endswith("_nominal")})
    mask = np.ones(len(frame), dtype=bool)
    for name in names:
        mask &= ak.to_numpy(cuts[name].filterCategory(events))
    return mask


def top_eff(e: pd.DataFrame, col: str, eff: float) -> dict:
    s = e.sort_values(col, ascending=False)
    sig_cum = np.cumsum(np.where(s._label == 1, s._w, 0.0))
    top = s.iloc[: np.searchsorted(sig_cum, eff * sig_cum[-1]) + 1]
    sig, bkg = top[top._label == 1], top[top._label == 0]
    cut = vbf_category_mask(sig)
    return {"S": float(sig._w.sum()), "B": float(bkg._w.sum()),
            "B_top_processes": {k: float(v) for k, v in bkg.groupby("_process")._w.sum().nlargest(3).items()},
            "sig_frac_passing_cutbased": float(sig._w[cut].sum() / sig._w.sum())}


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    ams_kw = dict(n_bins=cfg.ams_n_bins, bmin=cfg.ams_bmin, relmax=cfg.ams_relmax)
    pooled_kw = dict(ams_kw, bmin=cfg.ams_bmin * args.n_folds)
    models = {}
    for spec in args.model:
        label, rest = spec.split("=", 1)
        model_dir, tag = rest.rsplit(":", 1)
        models[label] = (Path(model_dir), tag)

    parts = []
    for fold in range(args.n_folds):
        raw = unscaled_fold(Path(args.data_dir), args.split, fold)
        for label, (model_dir, tag) in models.items():
            raw[label] = score(raw, model_dir, tag, fold)
        parts.append(raw)
    e = pd.concat(parts, ignore_index=True)

    summary = {"config": args.config, "data_dir": args.data_dir, "split": args.split,
               "ams_bins": {"per_fold": ams_kw, "pooled": pooled_kw}, "models": {}}
    for label in models:
        folds = [ams_pair(f._label.values, f[label].values, f._w.values, **ams_kw) for _, f in e.groupby("_fold")]
        r = {"auc_w": float(roc_auc_score(e._label, e[label], sample_weight=np.abs(e._w))),
             "per_fold": folds,
             "mean_of_folds": {k: float(np.mean([f[k] for f in folds])) for k in ("ams_absWgt", "ams")},
             "pooled": ams_pair(e._label.values, e[label].values, e._w.values, **pooled_kw)}
        for eff in args.top_effs:
            r[f"top_eff{eff:.2f}"] = top_eff(e, label, eff)
        summary["models"][label] = r

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"{args.split} folds of {args.data_dir}")
    print(f"{'model':24s} {'AUCw':>6s} {'AMS|w| pooled':>13s} {'AMS pooled':>10s} {'AMS|w| folds':>12s} {'AMS folds':>9s}"
          f" {f'B @{args.top_effs[0]:.0%} S':>8s}")
    for label, r in summary["models"].items():
        print(f"{label:24s} {r['auc_w']:6.3f} {r['pooled']['ams_absWgt']:13.3f} {r['pooled']['ams']:10.3f}"
              f" {r['mean_of_folds']['ams_absWgt']:12.3f} {r['mean_of_folds']['ams']:9.3f} {r[f'top_eff{args.top_effs[0]:.2f}']['B']:8.2f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
