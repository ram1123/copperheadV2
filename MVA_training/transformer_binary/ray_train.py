#!/usr/bin/env python3
"""
Parallel transformer trainings on the AF Ray cluster, one GPU each (across folds/seeds, not data-parallel).
Tasks resume from their last epoch after pre-emption.

Plan JSON:
  {"stage1": "<stage1_output>", "years": ["2017", ...], "out": "<dir on /work or /depot>",
   "hparams": {...overrides of train.DEFAULTS...},
   "jobs": [{"fold": 0, "stage": "selection+refit", "seed": 123456}, ...]}
`stage`: "selection", "refit" (needs "epochs") or "selection+refit". Outputs: <out>/<stage>/fold<k>_seed<s>/.

Usage, from the repo root with the AF global environment's python:
  python -m MVA_training.transformer_binary.ray_train submit --plan plan.json [--max-workers 4]
  python -m MVA_training.transformer_binary.ray_train status --job <id>
"""

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAY_GATEWAY = "http://ray-train-gateway:8265"


def job_dir(out, stage, fold, seed):
    return os.path.join(out, stage, f"fold{fold}_seed{seed}")


def run_one(plan, job):
    """Body of one Ray task: selection and/or refit for one fold and seed."""
    import torch

    from MVA_training.transformer_binary import train

    args = argparse.Namespace(
        stage1=plan["stage1"], years=plan["years"], seed=job["seed"], epochs=job.get("epochs"),
        hparams=json.dumps(plan.get("hparams", {})), no_mixed_precision=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = train.load_training_data(plan["stage1"], plan["years"], os.path.join(plan["out"], "data_cache"))
    result = {"job": job, "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    if job["stage"] in ("selection", "selection+refit"):
        summary = train.train_fold(data, job["fold"], "selection", job_dir(plan["out"], "selection", job["fold"], job["seed"]), train.build_cfg(args), device)
        result["selection"] = {k: summary[k] for k in ("best_epoch", "best_val_loss", "refit_epochs", "completed_epochs")}
        args.epochs = summary["refit_epochs"]
    if job["stage"] in ("refit", "selection+refit"):
        summary = train.train_fold(data, job["fold"], "refit", job_dir(plan["out"], "refit", job["fold"], job["seed"]), train.build_cfg(args), device)
        result["refit"] = {k: summary[k] for k in ("completed_epochs", "n_train")}
    return result


def run(plan_path):
    """Job entrypoint on the Ray cluster: one GPU task per plan entry."""
    import ray

    with open(plan_path) as stream:
        plan = json.load(stream)
    ray.init()
    # build the shared data cache once, before the GPU tasks start reading it
    from MVA_training.transformer_binary.train import load_training_data

    load_training_data(plan["stage1"], plan["years"], os.path.join(plan["out"], "data_cache"))
    task = ray.remote(num_gpus=1, max_retries=5, retry_exceptions=False)(run_one)
    pending = {task.remote(plan, job): job for job in plan["jobs"]}
    results = []
    while pending:
        done, _ = ray.wait(list(pending), num_returns=1)
        job = pending.pop(done[0])
        try:
            results.append(ray.get(done[0]))
            print(f"[done] {json.dumps(results[-1])}", flush=True)
        except Exception as error:  # keep the other trainings running; report at the end
            results.append({"job": job, "error": repr(error)})
            print(f"[failed] {job}: {error!r}", flush=True)
    with open(os.path.join(plan["out"], f"ray_results_{time.strftime('%Y%m%dT%H%M%S')}.json"), "w") as stream:
        json.dump(results, stream, indent=2)
    if any("error" in r for r in results):
        sys.exit(1)


def submit_job(command, max_workers, cpus, ram_gb):
    """
    Submit `command` (run from the repo root) to the user's AF Ray cluster; returns the job id.
    Jobs with the same worker settings share one cluster.
    """
    from ray.job_submission import JobSubmissionClient

    client = JobSubmissionClient(
        RAY_GATEWAY,
        headers={
            "af-max-workers": str(max_workers),
            "af-gpus-per-worker": "1",
            "af-cpus-per-worker": str(cpus),
            "af-ram-per-worker-gb": str(ram_gb),
        },
    )
    # the cluster sees /work, so the code runs from this checkout instead of being shipped
    job_id = client.submit_job(
        entrypoint=f"cd {REPO} && {command}",
        runtime_env={"env_vars": {"PYTHONPATH": REPO, "PYTHONUNBUFFERED": "1"}},
    )
    print(job_id)
    return job_id


def submit(plan_path, max_workers, cpus, ram_gb):
    command = f"python -m MVA_training.transformer_binary.ray_train run --plan {os.path.abspath(plan_path)}"
    return submit_job(command, max_workers, cpus, ram_gb)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--plan", required=True)
    p_submit.add_argument("--max-workers", type=int, default=4, help="GPU workers (one GPU each)")
    p_submit.add_argument("--cpus", type=int, default=8)
    p_submit.add_argument("--ram-gb", type=int, default=32)
    p_run = sub.add_parser("run")
    p_run.add_argument("--plan", required=True)
    p_status = sub.add_parser("status")
    p_status.add_argument("--job", required=True)
    args = parser.parse_args()

    if args.command == "submit":
        submit(args.plan, args.max_workers, args.cpus, args.ram_gb)
    elif args.command == "run":
        run(args.plan)
    else:
        from ray.job_submission import JobSubmissionClient

        client = JobSubmissionClient(RAY_GATEWAY)
        print(client.get_job_status(args.job))
        print(client.get_job_logs(args.job)[-5000:])


if __name__ == "__main__":
    main()
