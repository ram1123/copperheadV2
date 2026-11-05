#!/usr/bin/env python3
"""
Ray Tune HPO of the binary transformer: each trial is a train.py selection run of one fold (default 0), reporting its
inner-validation loss per epoch; ASHA stops weak trials, Optuna TPE proposes new ones. Trial 1 is train.DEFAULTS.

Usage, from the repo root with the AF global environment's python:
  python -m MVA_training.transformer_binary.hpo submit --stage1 <stage1_output> \
      --years 2016postVFP,2016preVFP,2017,2018 --cache-dir <dir> --out <dir> --num-samples 80
Results: <out>/hpo_results.csv and <out>/best_hparams.json (train.py --hparams, ray_train.py "hparams").
"""

import argparse
import json
import os
import shlex

from MVA_training.transformer_binary import config
from MVA_training.transformer_binary.folds import YEAR_VOCABULARIES, run_period

# configs/MVA/transformer_categorizer.yaml: hpo
SEARCH_SPACE = {name: tuple(spec) for name, spec in config.section("hpo")["search_space"].items()}
METRIC = config.section("hpo")["metric"]
OPTIONS = ("stage1", "years", "cache_dir", "out", "fold", "seed", "num_samples", "max_epochs", "gpus_per_trial", "max_concurrent")


def param_space():
    from ray import tune

    make = {"loguniform": tune.loguniform, "uniform": tune.uniform, "choice": tune.choice}
    return {name: make[kind](*args) for name, (kind, *args) in SEARCH_SPACE.items()}


def trial(config, stage1, years, cache_dir, out, fold, seed, max_epochs):
    """One Ray Tune trial: a selection run of `fold` with `config` overriding train.DEFAULTS."""
    import torch
    from ray import tune

    from MVA_training.transformer_binary import train

    cfg = dict(train.DEFAULTS)
    cfg.update(config)
    cfg.update(
        stage1=os.path.realpath(stage1), years=list(years), seed=seed, epochs=None, mixed_precision=True,
        n_years=len(YEAR_VOCABULARIES[run_period(years[0])]), max_epochs=max_epochs,
    )
    cfg["betas"] = list(cfg["betas"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = train.load_training_data(stage1, years, cache_dir)

    def report(history, best):
        tune.report({
            METRIC: best["val_loss"], "best_epoch": best["epoch"] + 1, "val_loss": history["val_loss"][-1],
            "train_loss": history["train_loss"][-1], "epoch_seconds": history["seconds"][-1],
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        })

    trial_dir = os.path.join(out, "trials", tune.get_context().get_trial_id())
    train.train_fold(data, fold, "selection", trial_dir, cfg, device, on_epoch=report)


def run(args):
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search import ConcurrencyLimiter
    from ray.tune.search.optuna import OptunaSearch

    from MVA_training.transformer_binary.train import DEFAULTS, load_training_data, write_json

    ray.init()
    load_training_data(args.stage1, args.years, args.cache_dir)  # build the cache once, before the trials
    baseline = {name: DEFAULTS[name] for name in SEARCH_SPACE}
    search = OptunaSearch(metric=METRIC, mode="min", points_to_evaluate=[baseline], seed=args.seed)
    trainable = tune.with_resources(
        tune.with_parameters(trial, stage1=args.stage1, years=args.years, cache_dir=args.cache_dir, out=args.out,
                             fold=args.fold, seed=args.seed, max_epochs=args.max_epochs),
        {"cpu": 2, "gpu": args.gpus_per_trial},
    )
    experiment = os.path.join(args.out, "tune")
    if tune.Tuner.can_restore(experiment):
        tuner = tune.Tuner.restore(experiment, trainable, resume_errored=True)
        # a restored run keeps its old limits: apply this run's to new and resumed trials
        tuner._local_tuner._tune_config.search_alg.set_max_concurrency(args.max_concurrent)
        tuner._local_tuner._tune_config.max_concurrent_trials = args.max_concurrent
    else:
        tuner = tune.Tuner(
            trainable,
            param_space=param_space(),
            tune_config=tune.TuneConfig(
                metric=METRIC, mode="min", num_samples=args.num_samples,
                search_alg=ConcurrencyLimiter(search, max_concurrent=args.max_concurrent),
                scheduler=ASHAScheduler(time_attr="training_iteration", max_t=args.max_epochs, grace_period=min(4, args.max_epochs), reduction_factor=2),
            ),
            run_config=tune.RunConfig(name="tune", storage_path=args.out, failure_config=tune.FailureConfig(max_failures=5)),
        )
    results = tuner.fit()
    frame = results.get_dataframe()
    frame.sort_values(METRIC).to_csv(os.path.join(args.out, "hpo_results.csv"), index=False)
    best = results.get_best_result(metric=METRIC, mode="min")
    write_json({**best.config, "_best_val_loss": best.metrics[METRIC], "_best_epoch": best.metrics["best_epoch"],
                "_baseline": baseline}, os.path.join(args.out, "best_hparams.json"))
    print(json.dumps(best.config), best.metrics[METRIC], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("submit", "run"))
    parser.add_argument("--stage1", required=True)
    parser.add_argument("--years", required=True, type=lambda s: s.split(","))
    parser.add_argument("--cache-dir", required=True, help="train.py's data cache directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--num-samples", type=int, default=80, help="trials in total")
    parser.add_argument("--max-epochs", type=int, default=64, help="ASHA's max_t; the selection run also stops on patience")
    parser.add_argument("--gpus-per-trial", type=float, default=1.0)
    parser.add_argument("--max-concurrent", type=int, default=32)
    parser.add_argument("--max-workers", type=int, default=24, help="submit: GPU workers")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--ram-gb", type=int, default=32)
    args = parser.parse_args()
    args.out = os.path.abspath(args.out)

    if args.command == "run":
        run(args)
        return
    from MVA_training.transformer_binary.ray_train import submit_job

    forwarded = ["run"]
    for name in OPTIONS:
        value = getattr(args, name)
        forwarded += [f"--{name.replace('_', '-')}", ",".join(value) if isinstance(value, list) else str(value)]
    submit_job(f"python -m MVA_training.transformer_binary.hpo {shlex.join(forwarded)}", args.max_workers, args.cpus, args.ram_gb)


if __name__ == "__main__":
    main()
