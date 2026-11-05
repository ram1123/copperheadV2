#!/usr/bin/env python3
"""
Add the out-of-fold transformer score to stage1 parquet, MC and data alike, as a copy in a sibling subdirectory
(or under --out-root) with score_field (ggH probability; VBF: score < cutoff) and fold_field, both named in
configs/MVA/transformer_categorizer.yaml. --mass-window 110,150 keeps lo <= dimuon_mass < hi. Whole trees: ray_score.py.

Usage, from the repo root:
  python -m MVA_training.transformer_binary.score --stage1 <stage1_output> --years 2017 \
      --models <dir>/fold{fold}/refit/model.pt --in-subdir compacted --out-subdir compacted_<tag> \
      [--out-root <stage1_output>] [--mass-window 110,150]
"""

import argparse
import glob
import os

import awkward as ak
import numpy as np
import torch

from MVA_training.transformer_binary.features import EVENT_ID, MODEL_INPUTS, build_inputs, read_columns
from MVA_training.transformer_binary.folds import N_FOLDS, YEAR_VOCABULARIES, event_folds, run_period
from MVA_training.transformer_binary.model import BinaryHmumuTransformer
from modules.selection import transformer_config
from modules.utils import logger

SCORE_FIELD, FOLD_FIELD = transformer_config()["score_field"], transformer_config()["fold_field"]


def load_model(path, device, n_years=None):
    """Our checkpoints ({'model', 'hparams'}) and b-hive's ({'model_state_dict'}) both load here."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model" in checkpoint:
        model = BinaryHmumuTransformer(**checkpoint["hparams"])
        model.load_state_dict(checkpoint["model"])
    else:  # b-hive Sep08 production layout: default architecture, Run-2 year vocabulary
        model = BinaryHmumuTransformer(n_years=n_years or len(YEAR_VOCABULARIES["run2"]))
        model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def load_fold_models(pattern, device):
    """`pattern` contains '{fold}', e.g. <dir>/fold{fold}/refit/model.pt."""
    # no TF32: it moves scores by ~1e-3 on A100s (not T4s)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return [load_model(pattern.format(fold=fold), device) for fold in range(N_FOLDS)]


@torch.inference_mode()
def predict(model, global_features, year_idx, objects, device, batch_size=8192):
    scores = []
    for start in range(0, len(year_idx), batch_size):
        batch = (
            torch.from_numpy(a[start : start + batch_size]).to(device)
            for a in (global_features, year_idx, objects)
        )
        scores.append(torch.sigmoid(model(*batch)).reshape(-1).float().cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)


def out_of_fold_scores(models, global_features, year_idx, objects, fold, device):
    score = np.full(len(fold), np.nan, dtype=np.float32)
    for k, model in enumerate(models):
        in_fold = fold == k
        if in_fold.any():
            score[in_fold] = predict(model, global_features[in_fold], year_idx[in_fold], objects[in_fold], device)
    if not np.all((score >= 0) & (score <= 1)):
        raise RuntimeError("transformer score outside [0, 1]")
    return score


def score_columns(columns, year, sample, models, device):
    """(score, fold) for a dict of stage1 columns holding MODEL_INPUTS and EVENT_ID."""
    global_features, year_idx, objects = build_inputs(columns, year)
    fold = event_folds(year_idx, sample, columns["run"], columns["luminosityBlock"], columns["event"])
    return out_of_fold_scores(models, global_features, year_idx, objects, fold, device), fold


def score_file(path, target, year, sample, models, device, mass_window=None):
    """
    Copy `path` to `target` with the score and fold columns (rows outside mass_window dropped).
    Written with awkward, as stage1 does, so the parquet's awkward metadata matches the columns.
    """
    columns = read_columns([path], MODEL_INPUTS + EVENT_ID + ("dimuon_mass",))
    events = ak.from_parquet(path)
    if mass_window is not None:
        keep = (columns["dimuon_mass"] >= mass_window[0]) & (columns["dimuon_mass"] < mass_window[1])
        events = events[keep]
        columns = {name: values[keep] for name, values in columns.items()}
    score, fold = score_columns(columns, year, sample, models, device)
    events = ak.with_field(events, score, SCORE_FIELD)
    events = ak.with_field(events, fold, FOLD_FIELD)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = f"{target}.tmp{os.getpid()}"
    ak.to_parquet(events, tmp)
    os.chmod(tmp, 0o664)
    os.replace(tmp, target)
    return len(events)


def sample_targets(stage1, year, sample, in_subdir, out_subdir, out_root=None):
    """[(input parquet, scored output path)] for one sample; the output mirrors the input layout."""
    in_dir = os.path.join(stage1, year, in_subdir, sample)
    out_dir = os.path.join(out_root or stage1, year, out_subdir, sample)
    files = sorted(glob.glob(os.path.join(in_dir, "**", "*.parquet"), recursive=True))
    return [(path, os.path.join(out_dir, os.path.relpath(path, in_dir))) for path in files]


def score_sample(stage1, year, sample, models, device, in_subdir, out_subdir, out_root=None, mass_window=None):
    targets = sample_targets(stage1, year, sample, in_subdir, out_subdir, out_root)
    for path, target in targets:
        if not os.path.exists(target):
            score_file(path, target, year, sample, models, device, mass_window)
    logger.info(f"{year}/{sample}: scored {len(targets)} files into {out_subdir}")


def check_run_period(models, years):
    period = "run3" if models[0].hparams["n_years"] == len(YEAR_VOCABULARIES["run3"]) else "run2"
    if {run_period(y) for y in years} != {period}:
        raise ValueError("models and --years belong to different run periods")


def parse_window(text):
    lo, hi = (float(x) for x in text.split(","))
    return lo, hi


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1", required=True)
    parser.add_argument("--years", required=True, type=lambda s: s.split(","))
    parser.add_argument("--models", required=True, help="checkpoint path with a {fold} placeholder")
    parser.add_argument("--samples", default=None, type=lambda s: s.split(","), help="default: every sample")
    parser.add_argument("--in-subdir", default="compacted")
    parser.add_argument("--out-subdir", required=True)
    parser.add_argument("--out-root", default=None, help="stage1_output directory for the scored copy (default: --stage1)")
    parser.add_argument("--mass-window", default=None, type=parse_window, help="lo,hi: keep lo <= dimuon_mass < hi")
    args = parser.parse_args()
    if args.in_subdir == args.out_subdir and args.out_root in (None, args.stage1):
        parser.error("the scored copy would overwrite the input; change --out-subdir or --out-root")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_fold_models(args.models, device)
    check_run_period(models, args.years)
    for year in args.years:
        samples = args.samples or sorted(os.listdir(os.path.join(args.stage1, year, args.in_subdir)))
        for sample in samples:
            score_sample(args.stage1, year, sample, models, device, args.in_subdir, args.out_subdir, args.out_root, args.mass_window)


if __name__ == "__main__":
    main()
