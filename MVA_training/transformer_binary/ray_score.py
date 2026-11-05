#!/usr/bin/env python3
"""
Score a whole stage1 tree with score.py on the AF Ray cluster (file batches, 0.25 GPU per task).
Existing outputs are skipped; the run ends with <out-root>/<out-subdir>_scoring_summary.json.

Usage, from the repo root with the AF global environment's python:
  python -m MVA_training.transformer_binary.ray_score submit --stage1 <stage1_output> \
      --years 2016postVFP,2016preVFP,2017,2018 --models '<dir>/refit/fold{fold}_seed123456/model.pt' \
      --out-root <stage1_output of the copy> --out-subdir compacted --mass-window 110,150
"""

import argparse
import json
import os
import shlex
import sys
import time

from MVA_training.transformer_binary.score import parse_window, sample_targets

SCORE_OPTIONS = ("stage1", "years", "models", "samples", "in_subdir", "out_subdir", "out_root", "mass_window", "files_per_task")


def score_batch(batch, models_pattern, mass_window):
    """Body of one Ray task: [(year, sample, input, target)] scored with one set of models."""
    import torch

    from MVA_training.transformer_binary.score import load_fold_models, score_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_fold_models(models_pattern, device)
    rows = 0
    for year, sample, path, target in batch:
        if not os.path.exists(target):
            rows += score_file(path, target, year, sample, models, device, mass_window)
    return {"files": len(batch), "rows_written": rows}


def run(args):
    import ray
    import torch

    from MVA_training.transformer_binary.score import check_run_period, load_fold_models

    check_run_period(load_fold_models(args.models, torch.device("cpu")), args.years)
    batches = []
    for year in args.years:
        samples = args.samples or sorted(os.listdir(os.path.join(args.stage1, year, args.in_subdir)))
        for sample in samples:
            todo = [(year, sample, path, target)
                    for path, target in sample_targets(args.stage1, year, sample, args.in_subdir, args.out_subdir, args.out_root)
                    if not os.path.exists(target)]
            batches += [todo[i : i + args.files_per_task] for i in range(0, len(todo), args.files_per_task)]
    print(f"{sum(map(len, batches))} files to score in {len(batches)} tasks", flush=True)

    ray.init()
    task = ray.remote(num_gpus=0.25, num_cpus=2, max_retries=3)(score_batch)
    pending = {task.remote(batch, args.models, args.mass_window): batch for batch in batches}
    summary = {"files": 0, "rows_written": 0, "failed": []}
    while pending:
        done, _ = ray.wait(list(pending), num_returns=1)
        batch = pending.pop(done[0])
        try:
            result = ray.get(done[0])
            summary["files"] += result["files"]
            summary["rows_written"] += result["rows_written"]
        except Exception as error:  # keep going; a resubmission retries only what is missing
            summary["failed"].append({"first_file": batch[0][2], "n_files": len(batch), "error": repr(error)})
            print(f"[failed] {batch[0][2]} (+{len(batch) - 1}): {error!r}", flush=True)
        if len(pending) % 50 == 0:
            print(f"{len(pending)} tasks left; {summary['files']} files done", flush=True)
    out_root = args.out_root or args.stage1
    summary.update({"finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": {k: getattr(args, k) for k in SCORE_OPTIONS}})
    with open(os.path.join(out_root, f"{args.out_subdir}_scoring_summary.json"), "w") as stream:
        json.dump(summary, stream, indent=2)
    if summary["failed"]:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("submit", "run"))
    parser.add_argument("--stage1", required=True)
    parser.add_argument("--years", required=True, type=lambda s: s.split(","))
    parser.add_argument("--models", required=True, help="checkpoint path with a {fold} placeholder")
    parser.add_argument("--samples", default=None, type=lambda s: s.split(","), help="default: every sample")
    parser.add_argument("--in-subdir", default="compacted")
    parser.add_argument("--out-subdir", required=True)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--mass-window", default=None, type=parse_window)
    parser.add_argument("--files-per-task", type=int, default=20)
    parser.add_argument("--max-workers", type=int, default=8, help="submit: GPU workers")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--ram-gb", type=int, default=32)
    args = parser.parse_args()
    if args.in_subdir == args.out_subdir and args.out_root in (None, args.stage1):
        parser.error("the scored copy would overwrite the input; change --out-subdir or --out-root")

    if args.command == "run":
        run(args)
        return
    from MVA_training.transformer_binary.ray_train import submit_job

    forwarded = ["run"]
    for name in SCORE_OPTIONS:
        value = getattr(args, name)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        forwarded += [f"--{name.replace('_', '-')}", str(value)]
    submit_job(f"python -m MVA_training.transformer_binary.ray_score {shlex.join(forwarded)}", args.max_workers, args.cpus, args.ram_gb)


if __name__ == "__main__":
    main()
