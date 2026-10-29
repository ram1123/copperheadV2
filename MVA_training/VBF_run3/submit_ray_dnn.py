#!/usr/bin/env python3
"""Submit/status/wait/stop four-fold VBF-DNN training on Purdue AF Ray.

The submitted driver starts four independent Ray tasks.  Each task trains one
cross-validation fold and reserves one GPU; this is fold-level parallelism, not
distributed data-parallel training of one model.  ``--cluster-workers`` requests
an exact-size cluster and may be larger than four when several four-fold jobs
share it.  ``--max-cluster-workers`` is an opt-in elastic alternative.
"""

import argparse
import json
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAY_ADDRESS = "http://ray-train-gateway:8265"
DEFAULT_RAY_ENV = "/work/pixi/global/.pixi/envs/default"


def require_ray_visible(
    path: str | Path, name: str, *, must_exist: bool = True
) -> Path:
    resolved = Path(path).expanduser().resolve(strict=must_exist)
    text = str(resolved)
    if not (
        text == "/work"
        or text.startswith("/work/")
        or text == "/depot/cms"
        or text.startswith("/depot/cms/")
    ):
        raise ValueError(
            f"{name} must be Ray-visible under /work or /depot/cms: {resolved}"
        )
    return resolved


def validate_preprocessed_data(data_dir: Path) -> None:
    required = [data_dir / "preprocess_manifest.json"]
    for split in ("train", "validation", "evaluation"):
        required.extend(
            data_dir / f"data_df_{split}_{fold}.parquet" for fold in range(4)
        )
    required.extend(data_dir / f"scalers_{fold}.npz" for fold in range(4))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Four-fold preprocessing output is incomplete; missing: "
            + ", ".join(missing)
        )


def validate_four_fold_config(config: Path) -> None:
    with open(config, "r") as stream:
        payload = yaml.safe_load(stream) or {}
    n_folds = int(((payload.get("data") or {}).get("cv") or {}).get("n_folds", 0))
    if n_folds != 4:
        raise ValueError(f"Ray training requires data.cv.n_folds=4, got {n_folds}.")


def ray_headers(args: argparse.Namespace) -> dict[str, str]:
    max_cluster_workers = getattr(args, "max_cluster_workers", None)
    cluster_workers = getattr(args, "cluster_workers", None)
    if max_cluster_workers is not None and cluster_workers is not None:
        raise ValueError(
            "--cluster-workers and --max-cluster-workers are mutually exclusive."
        )
    if max_cluster_workers is not None:
        max_cluster_workers = int(max_cluster_workers)
        if not 1 <= max_cluster_workers <= 16:
            raise ValueError("--max-cluster-workers must be in [1, 16].")
    else:
        # Preserve the historical contract: no worker-shape option means an
        # exact four-worker cluster, and --cluster-workers remains exact-size.
        cluster_workers = 4 if cluster_workers is None else int(cluster_workers)
        if not 4 <= cluster_workers <= 64:
            raise ValueError("--cluster-workers must be in [4, 64].")
    if not 1 <= args.cpus_per_worker <= 16:
        raise ValueError("--cpus-per-worker must be in [1, 16].")
    if not 1 <= args.ram_per_worker_gb <= 64:
        raise ValueError("--ram-per-worker-gb must be in [1, 64].")
    if args.min_gpu_memory_gb < 5:
        raise ValueError("--min-gpu-memory-gb must be at least 5.")
    ray_env = require_ray_visible(args.ray_env, "Ray environment")
    headers = {
        "af-gpus-per-worker": "1",
        "af-min-gpu-memory-gb": str(args.min_gpu_memory_gb),
        "af-cpus-per-worker": str(args.cpus_per_worker),
        "af-ram-per-worker-gb": str(args.ram_per_worker_gb),
        "af-env": str(ray_env),
    }
    if max_cluster_workers is not None:
        headers["af-max-workers"] = str(max_cluster_workers)
    else:
        headers["af-n-workers"] = str(cluster_workers)
    return headers


def build_entrypoint(args: argparse.Namespace) -> tuple[list[str], Path, Path]:
    config = require_ray_visible(args.config, "DNN config")
    data_dir = require_ray_visible(args.data_dir, "preprocessed data")
    out_dir = require_ray_visible(args.out_dir, "training output", must_exist=False)
    validate_four_fold_config(config)
    validate_preprocessed_data(data_dir)

    command = [
        "python",
        # Execute a thin driver which imports train_dnn.  Executing train_dnn
        # itself, even with ``python -m``, makes its functions live in
        # ``__main__``; Ray then cloudpickles their full torch globals and fails
        # on the non-serializable CudnnModule object.
        "-m",
        "MVA_training.VBF_run3.ray_train_driver",
        "--config",
        str(config),
        "--data-dir",
        str(data_dir),
        "--out-dir",
        str(out_dir),
        "--ray-folds",
        "--ray-cpus-per-fold",
        str(args.cpus_per_worker),
        "--ray-max-retries",
        str(args.max_retries),
        "--log-level",
        args.log_level,
    ]
    if args.optuna_best_json:
        best = require_ray_visible(args.optuna_best_json, "Optuna best JSON")
        command.extend(["--optuna-best-json", str(best)])
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.init_from:
        init_from = require_ray_visible(args.init_from, "warm-start directory")
        command.extend(["--init-from", str(init_from)])
    return command, data_dir, out_dir


def submit(args: argparse.Namespace) -> None:
    try:
        from ray.job_submission import JobSubmissionClient
    except ImportError as error:
        raise RuntimeError(
            "Run this command with /work/pixi/global/.pixi/envs/default/bin/python."
        ) from error

    if Path.cwd().resolve() != REPOSITORY_ROOT:
        raise RuntimeError(
            f"Run submit from the repository root: cd {REPOSITORY_ROOT}"
        )
    if args.wait and args.poll_seconds < 5:
        raise ValueError("--poll-seconds must be at least 5.")
    command, data_dir, out_dir = build_entrypoint(args)
    headers = ray_headers(args)
    plan: dict[str, Any] = {
        "strategy": "four-independent-folds-one-gpu-each",
        "entrypoint": shlex.join(command),
        "config": str(Path(args.config).resolve()),
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "headers": headers,
        "runtime_working_dir": ".",
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        print("Dry run only: no state was written and no Ray job was submitted.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    submission_path = out_dir / "ray_submission.json"
    if submission_path.exists() and not args.force_submit:
        raise RuntimeError(
            f"Submission record already exists: {submission_path}. Use a new "
            "--out-dir, or pass --force-submit after checking its job status."
        )

    client = JobSubmissionClient(RAY_ADDRESS, headers=dict(headers))
    job_id = client.submit_job(
        entrypoint=plan["entrypoint"],
        runtime_env={
            "working_dir": ".",
            "env_vars": {
                "MPLCONFIGDIR": "/tmp/vbfdnn_mpl",
                "XDG_CACHE_HOME": "/tmp/vbfdnn_xdg",
                "PYTHONPYCACHEPREFIX": "/tmp/vbfdnn_pycache",
            },
        },
    )
    record = {
        "kind": "copperhead_vbfdnn_ray_submission",
        "schema_version": 1,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        **plan,
    }
    with open(submission_path, "w") as stream:
        json.dump(record, stream, indent=2)
    print(f"Submitted four-fold VBF-DNN Ray job: {job_id}")
    print(f"Submission record: {submission_path}")
    if args.wait:
        wait_for_job(client, job_id, args.poll_seconds)


def wait_for_job(client: Any, job_id: str, poll_seconds: int) -> None:
    """Block until Ray reaches a terminal state, preserving workflow semantics."""

    if poll_seconds < 5:
        raise ValueError("--poll-seconds must be at least 5.")
    terminal = {"SUCCEEDED", "FAILED", "STOPPED"}
    last_status = None
    while True:
        status_value = client.get_job_status(job_id)
        status_name = str(getattr(status_value, "value", status_value)).upper()
        if status_name != last_status:
            print(f"Ray job {job_id}: {status_name}", flush=True)
            last_status = status_name
        if status_name in terminal:
            break
        time.sleep(poll_seconds)

    if status_name != "SUCCEEDED":
        logs = client.get_job_logs(job_id)
        if logs:
            print(logs)
        raise RuntimeError(f"Ray job {job_id} ended with status {status_name}.")


def job_id_from_args(args: argparse.Namespace) -> str:
    if args.job_id:
        return args.job_id
    if args.submission:
        with open(args.submission, "r") as stream:
            return str(json.load(stream)["job_id"])
    raise ValueError("Provide --job-id or --submission.")


def status(args: argparse.Namespace) -> None:
    from ray.job_submission import JobSubmissionClient

    client = JobSubmissionClient(RAY_ADDRESS)
    job_id = job_id_from_args(args)
    print(client.get_job_status(job_id))
    if args.logs:
        print(client.get_job_logs(job_id))


def wait(args: argparse.Namespace) -> None:
    """Wait for an already-submitted job and propagate its terminal status."""

    from ray.job_submission import JobSubmissionClient

    client = JobSubmissionClient(RAY_ADDRESS)
    wait_for_job(client, job_id_from_args(args), args.poll_seconds)


def stop(args: argparse.Namespace) -> None:
    from ray.job_submission import JobSubmissionClient

    client = JobSubmissionClient(RAY_ADDRESS)
    job_id = job_id_from_args(args)
    print(f"stop requested for {job_id}: {client.stop_job(job_id)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Submit four-GPU training.")
    submit_parser.add_argument("--config", required=True)
    submit_parser.add_argument("--data-dir", required=True)
    submit_parser.add_argument("--out-dir", required=True)
    submit_parser.add_argument("--optuna-best-json", default="")
    submit_parser.add_argument("--init-from", default="")
    submit_parser.add_argument("--seed", type=int, default=None)
    submit_parser.add_argument("--epochs", type=int, default=None)
    submit_parser.add_argument("--log-level", default="INFO")
    submit_parser.add_argument("--ray-env", default=DEFAULT_RAY_ENV)
    worker_group = submit_parser.add_mutually_exclusive_group()
    worker_group.add_argument(
        "--cluster-workers",
        type=int,
        default=None,
        help=(
            "Exact number of one-GPU workers in the shared Ray cluster. One model "
            "uses four; use 16 to run four four-fold models concurrently. The "
            "default remains an exact four-worker cluster."
        ),
    )
    worker_group.add_argument(
        "--max-cluster-workers",
        type=int,
        default=None,
        help=(
            "Elastic cluster upper bound (1-16), sent as af-max-workers. The "
            "cluster starts with as many workers as are currently free; folds can "
            "run sequentially when fewer than four GPUs are available. Mutually "
            "exclusive with --cluster-workers."
        ),
    )
    submit_parser.add_argument("--cpus-per-worker", type=int, default=8)
    submit_parser.add_argument("--ram-per-worker-gb", type=int, default=32)
    submit_parser.add_argument("--min-gpu-memory-gb", type=int, default=16)
    submit_parser.add_argument("--max-retries", type=int, default=2)
    submit_parser.add_argument("--dry-run", action="store_true")
    submit_parser.add_argument("--force-submit", action="store_true")
    submit_parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for all four folds; return nonzero if the Ray job fails.",
    )
    submit_parser.add_argument("--poll-seconds", type=int, default=20)
    submit_parser.set_defaults(func=submit)

    for name, function in (("status", status), ("wait", wait), ("stop", stop)):
        job_parser = subparsers.add_parser(name)
        job_parser.add_argument("--job-id", default="")
        job_parser.add_argument("--submission", default="")
        if name == "status":
            job_parser.add_argument("--logs", action="store_true")
        if name == "wait":
            job_parser.add_argument("--poll-seconds", type=int, default=20)
        job_parser.set_defaults(func=function)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
