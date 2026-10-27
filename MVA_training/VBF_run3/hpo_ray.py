#!/usr/bin/env python3
"""
Parallel Optuna HPO + final training for the Run-3 VBF DNN on the Purdue AF Ray GPU cluster.

Same objective, search space, pruner and outputs as hpo_optuna.py (optuna_best.json,
optuna_best_override.yaml, optuna_trials.csv, optuna_plots/), and the same final training as
train_dnn.py --optuna-best-json, so stage-2 finds the models exactly as after run_dnn.sh.
  hpo    the driver starts N long-lived Optuna workers (N = AF_MAX_WORKERS x --trials-per-gpu),
         one per GPU, sharing one study through a journal file in --out-dir. Workers past the first
         GPU can be reclaimed by the AF at any time: Ray restarts a lost worker when a GPU is free,
         and the driver marks its abandoned trial FAIL once its heartbeat is older than --stale-min,
         so the study still ends with --n-trials more COMPLETE/PRUNED trials (workers stop starting
         trials once finished + running reaches that; two racing workers may add one extra).
         TPE runs with constant_liar=True so concurrent workers spread out.
  train  one GPU task per fold: train_dnn.py --folds <f> --optuna-best-json <out-dir>/optuna_best.json.
Systematic-aware DNN: the adversarial_cli.py flags (--use_adversarial --adversarial-lambda ...) are
forwarded to every HPO trial and final training (docs/vbf_dnn_systematic_aware.md).
Ray (global env) only schedules GPUs: every GPU task runs the training in a subprocess with
--train-python, by default the repo's CVMFS pixi env (torch 2.5) that run_dnn.sh uses, so the
saved torchscript loads in stage-2.

Sub-commands:
  submit  run in the AF session: submits "hpo_ray.py run ..." as a Ray job, then by default
          follows its logs into <out-dir>/ray_job_<id>.log until it ends. Ctrl-C stops following,
          not the job. The job runs this repo in place from /work (git provenance as in a local
          run), so don't edit train_dnn.py / hpo_optuna.py while it runs.
  run     the Ray job entrypoint (driver); worker: one Optuna worker (started by run).

How to run (repo root, from an AF session WITHOUT a GPU; needs the preprocess_dnn.py fold
parquets in --data-dir, e.g. from run_analysis_pipeline.sh -m dnn_pre; all paths must be on /work
or /depot, which the cluster sees). Ray is not in the repo's pixi envs: submit with the AF global
env, which the cluster also runs (the cluster needs the submitter's Python + Ray):
    /work/pixi/global/.pixi/envs/default/bin/python MVA_training/VBF_run3/hpo_ray.py submit \
      --config ... --data-dir ... --out-dir ... [--stages hpo,train --train-out-dir ...] \
      [--max-workers 15] [--trials-per-gpu 1] [--dry-run] [--no-follow]

Example (25 trials on up to 15 GPUs, then the 4 folds; dirs as run_analysis_pipeline.sh -m dnn uses):
    D=dnn/trained_models/Run3_nanoAODv15_FilterEvents_Oct08_tightPassLepVeto_OfficialRecomendation_IPAdditionalLepVeto_Systematics_LumiSplit/2024_h-peak_vbf_all
    /work/pixi/global/.pixi/envs/default/bin/python MVA_training/VBF_run3/hpo_ray.py submit \
      --config configs/dnn_run3_vbf.yaml --data-dir $D \
      --out-dir $D/hpo_optuna/v1_multifold_025Trials --folds 0,1,2,3 --n-trials 25 \
      --stages hpo,train --train-out-dir $D/trained_best_optuna_v1_multifold_025Trials --max-workers 15
    # systematic-aware: D from preprocess_dnn.py --include-systematic-variations, and add
    #   --use_adversarial --adversarial-lambda 0.1 --ram-per-worker-gb 64
"""
import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from adversarial_cli import add_adversarial_args, adversarial_argv

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

JOURNAL_NAME = "optuna_journal.log"
HEARTBEAT_ATTR = "ray_heartbeat"
FINISHED_STATES = ("COMPLETE", "PRUNED")
DEFAULT_TRAIN_PYTHON = (
    "/cvmfs/cms-af.opensciencegrid.org/paf/pixi/copperheadV2/.pixi/envs/default/bin/python"
)


# --------------------------------------------------------------------------------------
# Shared CLI
# --------------------------------------------------------------------------------------
def add_hpo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True, help="HPO dir (journal, optuna_best.json, plots).")
    p.add_argument(
        "--folds", default=None, help="HPO folds, comma list (e.g. 0,1,2). Default: all cfg folds."
    )
    p.add_argument("--n-trials", type=int, default=50, help="Trials to ADD to the study.")
    p.add_argument("--study-name", default="vbf_dnn_hpo")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument(
        "--trials-per-gpu",
        type=int,
        default=1,
        help="Optuna workers sharing one GPU worker pod (the DNN is small and CPU-bound).",
    )
    p.add_argument(
        "--cpus-per-worker", type=int, default=8, help="Cores of each Ray worker pod."
    )
    p.add_argument(
        "--stale-min",
        type=float,
        default=15.0,
        help="A RUNNING trial with no heartbeat for this long is marked FAIL (worker lost).",
    )
    p.add_argument("--stages", default="hpo", help="Comma list of hpo,train.")
    p.add_argument("--train-out-dir", default=None, help="train_dnn.py --out-dir (train stage).")
    p.add_argument(
        "--train-python",
        default=DEFAULT_TRAIN_PYTHON,
        help="Interpreter that runs the trainings on the GPU workers.",
    )
    add_adversarial_args(p)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Parallel Optuna HPO + training for train_dnn.py on the AF Ray cluster")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("submit", help="Submit as a Ray job (run in the AF session).")
    add_hpo_args(ps)
    ps.add_argument("--max-workers", type=int, default=4, help="af-max-workers (GPU pods).")
    ps.add_argument("--ram-per-worker-gb", type=int, default=32)
    ps.add_argument("--min-gpu-memory-gb", type=int, default=None)
    ps.add_argument("--address", default="http://ray-train-gateway:8265")
    ps.add_argument("--no-follow", action="store_true", help="Return after submitting.")
    ps.add_argument("--dry-run", action="store_true", help="Print the job, don't submit.")

    pr = sub.add_parser("run", help="Ray job entrypoint (driver).")
    add_hpo_args(pr)

    pw = sub.add_parser("worker", help="One Optuna worker (started by run, in --train-python).")
    add_hpo_args(pw)
    pw.add_argument("--worker-idx", type=int, required=True)
    pw.add_argument("--target-finished", type=int, required=True)
    return p


def _stages(args: argparse.Namespace) -> set:
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    if not stages or stages - {"hpo", "train"}:
        raise SystemExit(f"--stages must be a comma list of hpo,train, got {args.stages!r}")
    if "train" in stages and not args.train_out_dir:
        raise SystemExit("--stages train needs --train-out-dir")
    return stages


def _pass_through_args(args: argparse.Namespace) -> list:
    out = [
        "--config", args.config,
        "--data-dir", args.data_dir,
        "--out-dir", args.out_dir,
        "--n-trials", str(args.n_trials),
        "--study-name", args.study_name,
        "--seed", str(args.seed),
        "--trials-per-gpu", str(args.trials_per_gpu),
        "--cpus-per-worker", str(args.cpus_per_worker),
        "--stale-min", str(args.stale_min),
        "--stages", args.stages,
        "--train-python", args.train_python,
    ]
    if args.folds is not None:
        out += ["--folds", args.folds]
    if args.train_out_dir:
        out += ["--train-out-dir", args.train_out_dir]
    return out + adversarial_argv(args)


# --------------------------------------------------------------------------------------
# submit (AF session)
# --------------------------------------------------------------------------------------
def _abs_on_cluster_storage(path: str, what: str) -> str:
    real = os.path.realpath(path)
    if not real.startswith(("/work/", "/depot/", "/cvmfs/")):
        raise SystemExit(f"{what} {path} -> {real} is not on /work, /depot or /cvmfs; the Ray cluster can't see it.")
    return real


def submit(args: argparse.Namespace) -> None:
    from ray.job_submission import JobStatus, JobSubmissionClient

    stages = _stages(args)
    args.config = _abs_on_cluster_storage(args.config, "--config")
    args.data_dir = _abs_on_cluster_storage(args.data_dir, "--data-dir")
    args.out_dir = _abs_on_cluster_storage(args.out_dir, "--out-dir")
    if args.train_out_dir:
        args.train_out_dir = _abs_on_cluster_storage(args.train_out_dir, "--train-out-dir")
    if not os.access(args.train_python, os.X_OK):
        raise SystemExit(f"--train-python {args.train_python} is not executable")
    args.train_python = _abs_on_cluster_storage(args.train_python, "--train-python")
    if not (Path(args.data_dir) / "data_df_train_0.parquet").exists():
        raise SystemExit(f"No fold parquets in {args.data_dir}; run preprocess_dnn.py first.")
    if args.use_adversarial:
        manifest = Path(args.data_dir) / "preprocess_manifest.json"
        if not (json.loads(manifest.read_text()).get("systematic_variations") or {}).get("enabled"):
            raise SystemExit(f"--use_adversarial needs {args.data_dir} from preprocess_dnn.py --include-systematic-variations.")
    if stages == {"train"} and not (Path(args.out_dir) / "optuna_best.json").exists():
        raise SystemExit(f"--stages train without hpo needs {args.out_dir}/optuna_best.json")

    # The cluster must run the submitter's Python + Ray, so use this interpreter, not PATH's python.
    python = _abs_on_cluster_storage(sys.executable, "interpreter")
    entrypoint = shlex.join([python, str(HERE / "hpo_ray.py"), "run", *_pass_through_args(args)])

    headers = {
        "af-max-workers": str(args.max_workers),
        "af-cpus-per-worker": str(args.cpus_per_worker),
        "af-ram-per-worker-gb": str(args.ram_per_worker_gb),
    }
    if args.min_gpu_memory_gb is not None:
        headers["af-min-gpu-memory-gb"] = str(args.min_gpu_memory_gb)

    # Run the repo in place (/work is visible to the cluster) rather than a shipped copy, so
    # train_dnn.py's git provenance (git runs in the cwd) is recorded as in a local run.
    runtime_env = {"env_vars": {"PYTHONPATH": str(REPO)}}

    print(f"[hpo_ray] headers:    {headers}")
    print(f"[hpo_ray] entrypoint: {entrypoint}")
    if args.dry_run:
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = JobSubmissionClient(args.address, headers=headers)
    job_id = client.submit_job(entrypoint=entrypoint, runtime_env=runtime_env)
    print(f"[hpo_ray] submitted job {job_id}", flush=True)
    if args.no_follow:
        return

    log_path = out_dir / f"ray_job_{job_id}.log"
    print(f"[hpo_ray] following logs -> {log_path} (Ctrl-C stops following, not the job)", flush=True)

    async def _follow() -> None:
        with open(log_path, "a") as f:
            async for chunk in client.tail_job_logs(job_id):
                sys.stdout.write(chunk)
                f.write(chunk)
                f.flush()

    try:
        asyncio.run(_follow())
    except KeyboardInterrupt:
        print(f"\n[hpo_ray] stopped following; job {job_id} keeps running.")
        return
    status = client.get_job_status(job_id)
    print(f"[hpo_ray] job {job_id} ended: {status}", flush=True)
    if status != JobStatus.SUCCEEDED:
        sys.exit(1)


# --------------------------------------------------------------------------------------
# Optuna storage helpers (driver and worker)
# --------------------------------------------------------------------------------------
def _journal_storage(out_dir: Path):
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    path = str(out_dir / JOURNAL_NAME)
    # Open-lock: the journal is appended from several pods over a shared filesystem.
    return JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))


def _n_finished(study) -> int:
    import optuna

    states = tuple(getattr(optuna.trial.TrialState, s) for s in FINISHED_STATES)
    return len(study.get_trials(deepcopy=False, states=states))


def _n_running(study) -> int:
    import optuna

    return len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)))


# --------------------------------------------------------------------------------------
# worker (GPU pod, --train-python)
# --------------------------------------------------------------------------------------
def worker(args: argparse.Namespace) -> None:
    """One long-lived Optuna worker: run trials until the study has --target-finished of them."""
    import optuna
    from hpo_optuna import make_pruner, objective_factory
    from train_dnn import adversarial_from_args, load_config, set_seed

    out_dir = Path(args.out_dir)
    base_cfg = load_config(args.config)
    set_seed(args.seed)
    folds = (
        list(range(base_cfg.n_folds))
        if args.folds is None
        else [int(x) for x in args.folds.split(",") if x.strip()]
    )

    study = optuna.load_study(
        study_name=args.study_name,
        storage=_journal_storage(out_dir),
        sampler=optuna.samplers.TPESampler(
            seed=args.seed + 1000 + args.worker_idx, multivariate=False, constant_liar=True
        ),
        pruner=make_pruner(),
    )
    adv, variation_spec = adversarial_from_args(args, args.data_dir)
    objective = objective_factory(
        base_cfg=base_cfg,
        data_dir=args.data_dir,
        out_dir=str(out_dir),
        folds=folds,
        seed=args.seed,
        adv=adv,
        variation_spec=variation_spec,
    )

    # Heartbeat so the driver can tell a trial whose pod was reclaimed from a slow one.
    current = {"trial": None}
    stop = threading.Event()

    def _beat() -> None:
        while not stop.wait(60.0):
            t = current["trial"]
            if t is not None:
                try:
                    t.set_user_attr(HEARTBEAT_ATTR, time.time())
                except Exception:
                    pass  # trial finished between the read and the write

    def objective_with_heartbeat(trial: optuna.Trial) -> float:
        trial.set_user_attr(HEARTBEAT_ATTR, time.time())
        trial.set_user_attr("ray_worker", args.worker_idx)
        current["trial"] = trial
        try:
            return objective(trial)
        finally:
            current["trial"] = None

    consecutive_fail = {"n": 0}

    def _n_claimed(study: optuna.Study) -> int:
        return _n_finished(study) + _n_running(study)

    def _stop_when_done(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.FAIL:
            consecutive_fail["n"] += 1
        else:
            consecutive_fail["n"] = 0
        # Count running trials too, so N workers don't overshoot the target by up to N-1.
        if consecutive_fail["n"] >= 3 or _n_claimed(study) >= args.target_finished:
            study.stop()

    threading.Thread(target=_beat, daemon=True).start()
    try:
        while _n_finished(study) < args.target_finished:
            if _n_claimed(study) >= args.target_finished:
                # Others' running trials cover the target; wait in case one of them is reaped.
                time.sleep(30.0)
                continue
            study.optimize(
                objective_with_heartbeat,
                callbacks=[_stop_when_done],
                catch=(Exception,),
                gc_after_trial=True,
            )
            if consecutive_fail["n"] >= 3:
                raise SystemExit(
                    f"[worker {args.worker_idx}] 3 consecutive FAILED trials; see the trial errors above."
                )
    finally:
        stop.set()
    print(f"[worker {args.worker_idx}] done", flush=True)


# --------------------------------------------------------------------------------------
# run (Ray driver)
# --------------------------------------------------------------------------------------
def run_on_gpu(cmd: list) -> None:
    """Ray task body: run one training command; Ray has already pinned the GPU and thread count."""
    env = dict(os.environ)
    # Only the repo, never the global env's site-packages, on the training interpreter's path.
    env["PYTHONPATH"] = str(REPO)
    env.pop("PYTHONHOME", None)
    print(f"[gpu task] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')} {shlex.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=str(REPO))


def _reap_stale_trials(study, storage, stale_s: float) -> int:
    import optuna

    now = time.time()
    n = 0
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        last = t.user_attrs.get(HEARTBEAT_ATTR)
        if last is None and t.datetime_start is not None:
            last = t.datetime_start.timestamp()
        if last is not None and now - float(last) > stale_s:
            try:
                storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
                n += 1
            except Exception:
                pass  # finished meanwhile
    return n


def _wait_all(ray, refs: list, what: str, on_tick=None, finished=None) -> list:
    errors = []
    pending = list(refs)
    while pending:
        done, pending = ray.wait(pending, num_returns=1, timeout=60.0)
        for ref in done:
            try:
                ray.get(ref)
            except Exception as e:
                errors.append(e)
                print(f"[driver] {what} task failed: {e}", flush=True)
        if on_tick is not None:
            on_tick(len(pending))
        if pending and finished is not None and finished():
            # Queued tasks still waiting for a GPU (held by other jobs) would only start and exit.
            for ref in pending:
                ray.cancel(ref, force=True)
            print(f"[driver] {what}: target reached, cancelled {len(pending)} idle/queued task(s)", flush=True)
            break
    return errors


def run_hpo(args: argparse.Namespace, ray, gpu_task, n_gpus: int) -> None:
    import optuna
    from hpo_optuna import make_pruner, summarize_study

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = _journal_storage(out_dir)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=False),
        pruner=make_pruner(),
        load_if_exists=True,
    )
    # Like hpo_optuna.py: --n-trials is added on top of what the study already holds.
    target = _n_finished(study) + int(args.n_trials)
    k = max(1, int(args.trials_per_gpu))
    n_workers = min(n_gpus * k, int(args.n_trials))
    print(
        f"[driver] hpo: study '{args.study_name}' in {out_dir / JOURNAL_NAME}: {_n_finished(study)} "
        f"finished, target {target}; {n_workers} Optuna workers ({n_gpus} GPU(s) x {k})",
        flush=True,
    )

    task = gpu_task.options(num_gpus=1.0 / k, num_cpus=max(1, int(args.cpus_per_worker) // k))
    refs = []
    for i in range(n_workers):
        cmd = [args.train_python, str(HERE / "hpo_ray.py"), "worker", *_pass_through_args(args),
               "--worker-idx", str(i), "--target-finished", str(target)]
        refs.append(task.remote(cmd))

    def _tick(n_pending: int) -> None:
        n_reaped = _reap_stale_trials(study, storage, 60.0 * float(args.stale_min))
        if n_reaped:
            print(f"[driver] marked {n_reaped} stale RUNNING trial(s) FAIL", flush=True)
        n_running = len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)))
        print(f"[driver] hpo: {_n_finished(study)}/{target} trials finished, {n_running} running, "
              f"{n_pending} workers alive", flush=True)

    def _target_met() -> bool:
        return _n_finished(study) >= target and _n_running(study) == 0

    errors = _wait_all(ray, refs, "hpo worker", _tick, _target_met)
    _reap_stale_trials(study, storage, 0.0)  # nothing can still be running
    if _n_finished(study) == 0:
        raise SystemExit("[driver] no finished trials")
    summarize_study(study, out_dir, args.seed)
    print(f"[driver] hpo: best value {study.best_value:.6f} (trial {study.best_trial.number})", flush=True)
    if errors:
        raise SystemExit(f"[driver] {len(errors)} hpo worker(s) failed; summary written anyway")


def run_train(args: argparse.Namespace, ray, gpu_task) -> None:
    import yaml

    with open(args.config) as f:
        n_folds = int(yaml.safe_load(f)["data"]["cv"]["n_folds"])
    best_json = Path(args.out_dir) / "optuna_best.json"
    train_out = Path(args.train_out_dir)
    train_out.mkdir(parents=True, exist_ok=True)
    print(f"[driver] train: {n_folds} folds in parallel -> {train_out} (best: {best_json})", flush=True)

    task = gpu_task.options(num_gpus=1, num_cpus=max(1, int(args.cpus_per_worker)))
    refs = []
    for f in range(n_folds):
        cmd = [args.train_python, str(HERE / "train_dnn.py"), "--config", args.config, "--data-dir",
               args.data_dir + "/", "--out-dir", str(train_out), "--folds", str(f),
               "--optuna-best-json", str(best_json), *adversarial_argv(args)]
        refs.append(task.remote(cmd))
    errors = _wait_all(ray, refs, "train fold",
                       lambda n: print(f"[driver] train: {n_folds - n}/{n_folds} folds done", flush=True))
    if errors:
        raise SystemExit(f"[driver] {len(errors)} training fold(s) failed")

    # Each fold process wrote manifest.json with only its own fold; record all of them.
    manifest = train_out / "manifest.json"
    if manifest.exists():
        with open(manifest) as f:
            m = json.load(f)
        m.setdefault("cv", {})["folds_run"] = "all"
        m["cv"]["parallel_folds_on_ray"] = True
        with open(manifest, "w") as f:
            json.dump(m, f, indent=2)
    missing = [f for f in range(n_folds) if not (train_out / f"fold{f}" / "best_torchscript.pt").exists()]
    if missing:
        raise SystemExit(f"[driver] train: no best_torchscript.pt for folds {missing}")
    print("[driver] train: all folds done", flush=True)


def run(args: argparse.Namespace) -> None:
    import ray

    stages = _stages(args)
    ray.init()
    n_gpus = int(os.environ.get("AF_MAX_WORKERS", "1"))
    # max_retries=-1: a task whose pod is reclaimed restarts when a GPU frees up.
    gpu_task = ray.remote(max_retries=-1)(run_on_gpu)
    if "hpo" in stages:
        run_hpo(args, ray, gpu_task, n_gpus)
    if "train" in stages:
        run_train(args, ray, gpu_task)


def main() -> None:
    args = build_argparser().parse_args()
    {"submit": submit, "run": run, "worker": worker}[args.cmd](args)


if __name__ == "__main__":
    main()
