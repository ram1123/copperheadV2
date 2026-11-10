#!/usr/bin/env python3
"""
Train one fold of the binary ggH-vs-VBF transformer (b-hive Sep08 recipe):
  * events: ggH (label 1) / VBF (label 0) signal from transformer_categorizer.yaml, h-peak [115, 135), b-tag veto
  * folds: model k never trains on fold k (folds.event_folds)
  * "selection": train minus a ~10% inner-validation split, early-stop on it -> best epoch
  * "refit": retrain on all training folds for that many epochs; the last epoch is the model
  * loss: |w|-weighted BCE, each class normalised to 1 with the same year mixture
Checkpoints every epoch, so a pre-empted job resumes.

Usage, from the repo root:
  python -m MVA_training.transformer_binary.train --stage1 <stage1_output> --years 2016postVFP,2016preVFP,2017,2018 \
      --fold 0 --stage selection --out <dir>
  python -m MVA_training.transformer_binary.train ... --stage refit --epochs <best_epoch+1>
"""

import argparse
import hashlib
import json
import os
import random
import time

import numpy as np
import torch

from MVA_training.transformer_binary.features import load_signal
from MVA_training.transformer_binary import config
from MVA_training.transformer_binary.folds import YEAR_VOCABULARIES, inner_validation_mask, run_period
from MVA_training.transformer_binary.model import BinaryHmumuTransformer
from modules.utils import logger

DEFAULTS = dict(config.section("training"))  # configs/MVA/transformer_categorizer.yaml: training


def save_group_readable(obj, path):
    """Atomic torch.save that leaves the file readable by the group (b-hive's mkstemp left 0600)."""
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(obj, tmp)
    os.chmod(tmp, 0o664)
    os.replace(tmp, path)


def write_json(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as stream:
        json.dump(obj, stream, indent=2, default=str)
    os.chmod(tmp, 0o664)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------- data


def load_training_data(stage1, years, cache_dir):
    """All selected signal events of `years`, cached as one .npz per (stage1, years)."""
    key = hashlib.sha1(f"{os.path.realpath(stage1)}|{','.join(years)}".encode()).hexdigest()[:12]
    cache = os.path.join(cache_dir, f"signal_{key}.npz")
    if os.path.exists(cache):
        with np.load(cache, allow_pickle=False) as stored:
            return {name: stored[name] for name in stored.files}
    logger.info(f"Building training-data cache {cache} from {stage1} for {years}")
    data = load_signal(stage1, years)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = f"{cache}.tmp{os.getpid()}.npz"
    np.savez(tmp, **data)
    os.chmod(tmp, 0o664)
    os.replace(tmp, cache)
    return data


def class_year_weights(data, selected, target_mixture):
    """|w| rescaled so that, within `selected`, each (class, year) cell sums to target_mixture[year]."""
    weights = np.zeros(len(selected), dtype=np.float64)
    for label in (0.0, 1.0):
        for year_idx, fraction in target_mixture.items():
            cell = selected & (data["is_ggH"] == label) & (data["year_idx"] == year_idx)
            total = data["abs_weight"][cell].sum()
            if not total > 0:
                raise ValueError(f"empty class/year cell: is_ggH={label}, year_idx={year_idx}")
            weights[cell] = data["abs_weight"][cell] * (fraction / total)
    return weights


def target_year_mixture(data):
    """Mean of the ggH and VBF |w| year fractions over the full selected sample."""
    fractions = []
    for label in (0.0, 1.0):
        is_class = data["is_ggH"] == label
        per_year = {y: data["abs_weight"][is_class & (data["year_idx"] == y)].sum() for y in np.unique(data["year_idx"])}
        total = sum(per_year.values())
        fractions.append({y: v / total for y, v in per_year.items()})
    mixture = {y: 0.5 * (fractions[0][y] + fractions[1][y]) for y in fractions[0]}
    norm = sum(mixture.values())
    return {int(y): v / norm for y, v in mixture.items()}


def make_splits(data, fold, stage, inner_fraction=0.10):
    """(train_mask, val_mask) of this fold and stage. The evaluation fold is never touched."""
    outer_train = data["fold"] != fold
    inner_val = np.zeros(len(outer_train), dtype=bool)
    for sample in np.unique(data["sample"]):
        is_sample = data["sample"] == sample
        inner_val[is_sample] = inner_validation_mask(
            data["year_idx"][is_sample], str(sample), data["run"][is_sample],
            data["luminosityBlock"][is_sample], data["event"][is_sample], fraction=inner_fraction,
        )
    validation = outer_train & inner_val
    training = outer_train & ~inner_val if stage == "selection" else outer_train
    return training, validation


class Tensors:
    """One split on `device` in shuffled mini-batches (shuffled on the CPU generator: same batches on any device)."""

    def __init__(self, data, mask, weights, device="cpu"):
        index = np.flatnonzero(mask)
        self.tensors = tuple(
            torch.from_numpy(array).to(device)
            for array in (
                data["global_features"][index], data["year_idx"][index], data["objects"][index],
                data["is_ggH"][index], weights[index].astype(np.float32),
            )
        )

    def __len__(self):
        return len(self.tensors[0])

    def batches(self, batch_size, device, generator=None):
        order = torch.randperm(len(self), generator=generator) if generator is not None else torch.arange(len(self))
        for start in range(0, len(self), batch_size):
            idx = order[start : start + batch_size].to(self.tensors[0].device)
            yield tuple(t[idx].to(device, non_blocking=True) for t in self.tensors)


# ---------------------------------------------------------------------------- training


def make_scheduler(optimizer, n_epochs, lr_decay_factor):
    """b-hive's epoch_lin_decay: constant, then geometric decay to lr_decay_factor over the last 30%."""
    decay_epochs = max(1, int(n_epochs * 0.3))
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(range(n_epochs - decay_epochs, n_epochs)),
        gamma=lr_decay_factor ** (1.0 / decay_epochs),
    )


def weighted_bce(logits, label, weight):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float().reshape(-1), label, reduction="none")
    return (loss * weight).sum(), weight.sum()


def run_epoch(model, split, cfg, device, optimizer=None, scaler=None, generator=None):
    """One pass over `split`; trains if an optimizer is given. Returns the weighted mean loss."""
    training = optimizer is not None
    model.train(training)
    numerator = torch.zeros((), device=device)
    denominator = torch.zeros((), device=device)
    with torch.set_grad_enabled(training):
        for global_features, year_idx, objects, label, weight in split.batches(cfg["batch_size"], device, generator):
            with torch.autocast(device.type, enabled=cfg["mixed_precision"] and device.type == "cuda"):
                batch_num, batch_den = weighted_bce(model(global_features, year_idx, objects), label, weight)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(batch_num / batch_den).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            numerator += batch_num.detach()
            denominator += batch_den.detach()
    return float(numerator / denominator)


def train_fold(data, fold, stage, out_dir, cfg, device, on_epoch=None):
    """Train or resume one fold/stage; returns the _SUCCESS summary. `on_epoch(history, best)` runs after each epoch."""
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o2775)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    generator = torch.Generator().manual_seed(cfg["seed"] + 1000 * fold)

    mixture = target_year_mixture(data)
    train_mask, val_mask = make_splits(data, fold, stage)
    # inputs (~1 GB) stay on the GPU if it has room (T4), else stream from host memory (5 GB A100 slices)
    big_gpu = device.type == "cuda" and torch.cuda.get_device_properties(device).total_memory >= 8 * 2**30
    data_device = device if big_gpu else "cpu"
    train = Tensors(data, train_mask, class_year_weights(data, train_mask, mixture), data_device)
    val = Tensors(data, val_mask, class_year_weights(data, val_mask, mixture), data_device)
    n_epochs = cfg["max_epochs"] if stage == "selection" else cfg["epochs"]
    patience = cfg["patience"] if stage == "selection" else 0
    logger.info(f"fold {fold} {stage}: {len(train)} train / {len(val)} val events, up to {n_epochs} epochs")

    model = BinaryHmumuTransformer(
        n_years=cfg["n_years"], num_enc=cfg["num_enc"], num_head=cfg["num_head"],
        embed_dim=cfg["embed_dim"], dropout=cfg["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], betas=tuple(cfg["betas"]), eps=cfg["eps"], weight_decay=cfg["weight_decay"]
    )
    scheduler = make_scheduler(optimizer, n_epochs, cfg["lr_decay_factor"])
    scaler = torch.amp.GradScaler(device.type, enabled=cfg["mixed_precision"] and device.type == "cuda")
    history = {"train_loss": [], "val_loss": [], "lr": [], "seconds": []}
    best = {"epoch": -1, "val_loss": float("inf")}

    last_path = os.path.join(out_dir, "last_epoch.pt")
    if os.path.exists(last_path):
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state["cfg"] != cfg:
            raise RuntimeError(f"{last_path} was written with a different configuration; use a new --out")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        generator.set_state(state["generator"].cpu())  # map_location moved the RNG states to the device
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        history, best = state["history"], state["best"]
        logger.info(f"resuming fold {fold} {stage} after epoch {len(history['train_loss'])}")

    for epoch in range(len(history["train_loss"]), n_epochs):
        if stage == "selection" and epoch - best["epoch"] > patience:
            break
        start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss = run_epoch(model, train, cfg, device, optimizer, scaler, generator)
        with torch.inference_mode():
            val_loss = run_epoch(model, val, cfg, device)
        scheduler.step()
        for key, value in zip(history, (train_loss, val_loss, lr, time.time() - start)):
            history[key].append(value)
        if val_loss < best["val_loss"]:
            best = {"epoch": epoch, "val_loss": val_loss}
            if stage == "selection":
                save_group_readable({"model": model.state_dict(), "hparams": model.hparams, "cfg": cfg}, os.path.join(out_dir, "best_model.pt"))
        save_group_readable(
            {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "generator": generator.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None,
                "history": history, "best": best, "cfg": cfg,
            },
            last_path,
        )
        logger.info(f"fold {fold} {stage} epoch {epoch + 1}/{n_epochs}: train {train_loss:.5f} val {val_loss:.5f} lr {lr:.2e} ({history['seconds'][-1]:.0f}s)")
        if on_epoch is not None:
            on_epoch(history, best)

    if stage == "refit":
        # the production model is the last epoch, not the best one: the epoch count was chosen in selection
        save_group_readable({"model": model.state_dict(), "hparams": model.hparams, "cfg": cfg}, os.path.join(out_dir, "model.pt"))
    summary = {
        "fold": fold, "stage": stage, "completed_epochs": len(history["train_loss"]),
        "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
        "refit_epochs": best["epoch"] + 1 if stage == "selection" else None,
        "n_train": len(train), "n_val": len(val), "year_mixture": mixture, "history": history, "cfg": cfg,
    }
    write_json(summary, os.path.join(out_dir, "_SUCCESS.json"))
    return summary


def build_cfg(args):
    cfg = dict(DEFAULTS)
    cfg.update(json.loads(args.hparams) if args.hparams else {})
    periods = {run_period(y) for y in args.years}
    if len(periods) != 1:
        raise ValueError("train one run period at a time")
    cfg.update(
        stage1=os.path.realpath(args.stage1), years=list(args.years), seed=args.seed, epochs=args.epochs,
        mixed_precision=not args.no_mixed_precision, n_years=len(YEAR_VOCABULARIES[periods.pop()]),
    )
    cfg["betas"] = list(cfg["betas"])
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1", required=True, help="stage1_output directory (<year>/compacted/<sample>/...)")
    parser.add_argument("--years", required=True, type=lambda s: s.split(","))
    parser.add_argument("--fold", required=True, type=int, choices=range(4))
    parser.add_argument("--stage", required=True, choices=("selection", "refit"))
    parser.add_argument("--epochs", type=int, default=None, help="refit epoch count (best_epoch + 1 of the selection)")
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--hparams", default=None, help="JSON overriding DEFAULTS, e.g. from the HPO")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", default=None, help="default: <out>/../data_cache")
    parser.add_argument("--no-mixed-precision", action="store_true")
    args = parser.parse_args()
    if args.stage == "refit" and not args.epochs:
        parser.error("--stage refit needs --epochs")

    cfg = build_cfg(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_training_data(args.stage1, args.years, args.cache_dir or os.path.join(os.path.dirname(os.path.abspath(args.out)), "data_cache"))
    summary = train_fold(data, args.fold, args.stage, args.out, cfg, device)
    logger.info(json.dumps({k: v for k, v in summary.items() if k not in ("history", "cfg")}))


if __name__ == "__main__":
    main()
