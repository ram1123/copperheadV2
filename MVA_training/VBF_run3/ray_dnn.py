#!/usr/bin/env python3
"""
Run the VBF DNN HPO (hpo_optuna.py) and training (train_dnn.py) as GPU tasks on the AF Ray cluster.
  hpo    --workers tasks share one Optuna journal study in --out-dir (same --seed, different sampler seeds);
         a final pass writes optuna_best.json
  train  one task per fold (train_dnn.py --folds k)
Tasks run this checkout's scripts from /work when they start; each takes --gpus-per-task GPU and --memory-gb RAM.

Usage, from the repo root with the AF global environment's python:
  python MVA_training/VBF_run3/ray_dnn.py submit hpo --config configs/dnn_run2_vbf.yaml \
      --data-dir <preprocessed dir> --out-dir <dir>/hpo_optuna/<label> --n-trials 100 --workers 8
  python MVA_training/VBF_run3/ray_dnn.py submit train --config configs/dnn_run2_vbf.yaml \
      --data-dir <preprocessed dir> --out-dir <dir>/trained_best_optuna_<label> \
      --optuna-best-json <dir>/hpo_optuna/<label>/optuna_best.json
"""

import argparse
import math
import os
import shlex
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(REPO, "MVA_training", "VBF_run3")


def run_script(argv, cpus):
    """Body of one Ray task: one of this directory's scripts, unchanged, in a subprocess."""
    env = dict(os.environ, PYTHONPATH=REPO, OMP_NUM_THREADS=str(cpus), MKL_NUM_THREADS=str(cpus))
    done = subprocess.run([sys.executable, *argv], cwd=REPO, env=env, capture_output=True, text=True)
    return {"argv": argv, "returncode": done.returncode, "stdout": done.stdout[-3000:], "stderr": done.stderr[-3000:]}


def hpo_commands(args):
    storage = f"journal:{os.path.join(os.path.abspath(args.out_dir), 'optuna_journal.log')}"
    common = [os.path.join(SCRIPTS, "hpo_optuna.py"), "--config", args.config, "--data-dir", args.data_dir,
              "--out-dir", args.out_dir, "--storage", storage, "--seed", str(args.seed)]
    if args.folds:
        common += ["--folds", args.folds]
    if args.objective:
        common += ["--objective", args.objective]
    if args.label_smoothing_range:
        common += ["--label-smoothing-range", args.label_smoothing_range]
    per_worker = math.ceil(args.n_trials / args.workers)
    workers = [common + ["--n-trials", str(per_worker), "--sampler-seed", str(args.seed + 1000 * (i + 1))]
               for i in range(args.workers)]
    return workers, common + ["--n-trials", "0"]


def train_commands(args):
    common = [os.path.join(SCRIPTS, "train_dnn.py"), "--config", args.config, "--data-dir", args.data_dir,
              "--out-dir", args.out_dir]
    if args.optuna_best_json:
        common += ["--optuna-best-json", args.optuna_best_json]
    if args.es_monitor:
        common += ["--es-monitor", args.es_monitor]
    return [common + ["--folds", str(k)] for k in args.folds.split(",")], None


def run(args):
    import ray

    os.makedirs(args.out_dir, exist_ok=True)
    tasks, final = hpo_commands(args) if args.step == "hpo" else train_commands(args)
    ray.init()
    remote = ray.remote(num_gpus=args.gpus_per_task, num_cpus=args.cpus_per_task, memory=int(args.memory_gb * 2**30),
                        max_retries=3)(run_script)
    results = ray.get([remote.remote(argv, args.cpus_per_task) for argv in tasks])
    if final is not None:  # optuna_best.json from the complete study, not from one worker's view
        results.append(run_script(final, args.cpus_per_task))
    for r in results:
        print(f"[{'ok' if r['returncode'] == 0 else 'FAILED'}] {shlex.join(r['argv'][-4:])}\n{r['stderr'][-1500:]}", flush=True)
    if any(r["returncode"] for r in results):
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("submit", "run"))
    parser.add_argument("step", choices=("hpo", "train"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--folds", default=None, help="hpo: folds per trial (default: all); train: folds to train")
    parser.add_argument("--n-trials", type=int, default=100, help="hpo: trials in total")
    parser.add_argument("--workers", type=int, default=8, help="hpo: parallel Optuna workers")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--optuna-best-json", default=None, help="train")
    parser.add_argument("--objective", default=None, help="hpo: hpo_optuna.py --objective")
    parser.add_argument("--label-smoothing-range", default=None, help="hpo: hpo_optuna.py --label-smoothing-range LO,HI")
    parser.add_argument("--es-monitor", default=None, help="train: train_dnn.py --es-monitor")
    parser.add_argument("--gpus-per-task", type=float, default=0.25)
    parser.add_argument("--cpus-per-task", type=int, default=2)
    parser.add_argument("--memory-gb", type=float, default=9.0, help="RAM reserved per task")
    parser.add_argument("--max-workers", type=int, default=22, help="submit: Ray GPU workers")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--ram-gb", type=int, default=32)
    args = parser.parse_args()
    if args.step == "train" and not args.folds:
        args.folds = "0,1,2,3"
    for name in ("config", "data_dir", "out_dir", "optuna_best_json"):
        if getattr(args, name):
            setattr(args, name, os.path.abspath(getattr(args, name)))

    if args.mode == "run":
        run(args)
        return
    sys.path.insert(0, REPO)
    from MVA_training.transformer_binary.ray_train import submit_job

    forwarded = ["run", args.step]
    for name in ("config", "data_dir", "out_dir", "folds", "n_trials", "workers", "seed", "optuna_best_json",
                 "objective", "label_smoothing_range", "es_monitor", "gpus_per_task", "cpus_per_task", "memory_gb"):
        value = getattr(args, name)
        if value is not None:
            forwarded += [f"--{name.replace('_', '-')}", str(value)]
    submit_job(f"python MVA_training/VBF_run3/ray_dnn.py {shlex.join(forwarded)}", args.max_workers, args.cpus, args.ram_gb)


if __name__ == "__main__":
    main()
