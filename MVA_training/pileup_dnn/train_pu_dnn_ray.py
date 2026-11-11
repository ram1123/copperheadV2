#!/usr/bin/env python3
"""
Train the PU-DNN on the AF Ray cluster: one GPU job per eta region, then merge the per-region
outputs into one model directory that src/corrections/pu_dnn.py can load.

Each region job runs `train_pu_dnn.py <train args> --regions <R> --flatten-per-file --device cuda
-o <out>/jobs/<R>` with the copperheadV2 CVMFS interpreter (same torch as stage-1, so the
TorchScript model loads there). Ray runs at most --max-workers jobs at once (1 GPU each). While the
cluster is busy under other headers the gateway refuses the submission; it is retried every
--retry-min minutes for up to --max-wait-h hours. A failed region is resubmitted up to
--max-attempts times. Regions whose jobs/<R>/summary_<R>.json already exists are skipped (restartable).
After all regions succeed: <out>/<R> -> jobs/<R>/<R>, summary_<R>.json copied, summary_all.json and
sample_composition.json ({region: composition}) written, inputs.json/git_state.json/git_diff.patch
copied from the first region job. Status lines: <out>/ray_status.log. TERM/HUP stops the Ray jobs.

How to run (repo root, AF session WITHOUT a GPU; output dir on /work or /depot; launch detached):
    /work/pixi/global/.pixi/envs/default/bin/python MVA_training/pileup_dnn/train_pu_dnn_ray.py \
        -o <out_dir> [--max-workers 3] [--dry-run | --merge-only] -- <train_pu_dnn.py args without -o/--regions>

Example (2024 Train-1 retrain on the Oct28 ntuples):
    B=/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterJets_Oct28_tightPassLepVeto_DefaultjetPt25GeV_JetIDVars_LumiSplit/stage1_output/2024/compacted
    MIN_PROXY_H=0 EXPECTED_H=30 bash .claude/skills/run-copperheadV2-develop/detach_job.sh pudnn_train_ray_2024 \
      /work/pixi/global/.pixi/envs/default/bin/python MVA_training/pileup_dnn/train_pu_dnn_ray.py \
      -o /work/projects/hmm/shar1172/pu_dnn/run2024_dy_top_ewk_Train1_Oct29_ray -- \
      -i "$B/dyTo2Mu_M-50_aMCatNLO/*/*.parquet" "$B/ttjets_*/*/*.parquet" "$B/ewk_*/*/*.parquet" --use-glob \
      --features minDPhiMetJet maxDPhiMetJet puIdDisc chEmEF chHEF neEmEF neHEF muEF chMultiplicity \
        neMultiplicity nConstituents nElectrons nMuons hfadjacentEtaStripsSize hfcentralEtaStripSize \
        hfsigmaEtaEta hfsigmaPhiPhi hfEmEF hfHEF
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = "MVA_training/pileup_dnn/train_pu_dnn.py"
DEFAULT_TRAIN_PYTHON = "/cvmfs/cms-af.opensciencegrid.org/paf/pixi/copperheadV2/.pixi/envs/default/bin/python"
REGIONS = ("HEpos", "HEneg", "HFpos", "HFneg")
SHARED_FILES = ("inputs.json", "git_state.json", "git_diff.patch")


class _Terminated(Exception):
    pass


def _raise_terminated(signum, _frame):
    raise _Terminated(f"signal {signum}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("-o", "--out", required=True, help="Model output dir (absolute, on /work or /depot).")
    p.add_argument("--regions", nargs="+", default=list(REGIONS), choices=REGIONS)
    p.add_argument("--max-workers", type=int, default=3, help="af-max-workers = max concurrent GPUs.")
    p.add_argument("--cpus-per-worker", type=int, default=8)
    p.add_argument("--ram-per-worker-gb", type=int, default=64)
    p.add_argument("--address", default="http://ray-train-gateway:8265")
    p.add_argument("--train-python", default=DEFAULT_TRAIN_PYTHON)
    p.add_argument("--retry-min", type=float, default=5.0, help="Wait between refused submissions.")
    p.add_argument("--max-wait-h", type=float, default=48.0, help="Give up submitting after this long.")
    p.add_argument("--max-attempts", type=int, default=3, help="Submissions per region.")
    p.add_argument("--poll-s", type=int, default=60)
    p.add_argument("--dry-run", action="store_true", help="Print headers and entrypoints only.")
    p.add_argument("--merge-only", action="store_true", help="Only merge existing jobs/<R> outputs.")
    p.add_argument("train_args", nargs=argparse.REMAINDER, help="After '--': train_pu_dnn.py args.")
    args = p.parse_args()
    if args.train_args and args.train_args[0] == "--":
        args.train_args = args.train_args[1:]
    for forbidden in ("-o", "--output", "--regions", "--device"):
        if forbidden in args.train_args:
            raise SystemExit(f"{forbidden} is set per region job by this script; drop it from the train args.")
    args.out = Path(args.out).resolve()
    if not str(args.out).startswith(("/work/", "/depot/")):
        raise SystemExit(f"--out {args.out} must be on /work or /depot (Ray cannot see other volumes).")
    return args


def log(out: Path, msg: str) -> None:
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(out / "ray_status.log", "a") as f:
        f.write(line + "\n")


def region_done(out: Path, region: str) -> bool:
    return (out / "jobs" / region / f"summary_{region}.json").is_file()


def entrypoint(args: argparse.Namespace, region: str) -> str:
    cmd = [
        args.train_python, TRAIN_SCRIPT, *args.train_args,
        "--regions", region, "--flatten-per-file", "--device", "cuda",
        "-o", str(args.out / "jobs" / region),
    ]
    # cwd = repo so git_state.json records this checkout, as in a local run
    return f"cd {shlex.quote(str(REPO))} && OMP_NUM_THREADS={args.cpus_per_worker} {shlex.join(cmd)}"


def submit(client, args: argparse.Namespace, region: str, deadline: float) -> str:
    runtime_env = {"env_vars": {"PYTHONPATH": str(REPO)}}
    while True:
        try:
            return client.submit_job(
                entrypoint=entrypoint(args, region),
                runtime_env=runtime_env,
                entrypoint_num_gpus=1,
                entrypoint_num_cpus=args.cpus_per_worker,
            )
        except RuntimeError as e:
            # Refused while the cluster runs something under other headers, or no GPU free yet.
            if time.time() > deadline:
                raise
            log(args.out, f"WAIT {region} submit refused, retry in {args.retry_min:g} min: {str(e)[:300]}")
            time.sleep(args.retry_min * 60)


def save_job_log(client, args: argparse.Namespace, region: str, job_id: str) -> None:
    try:
        text = client.get_job_logs(job_id)
    except Exception as e:  # logs vanish with an idle-removed cluster
        text = f"(could not fetch logs: {e})\n"
    path = args.out / "jobs" / region / f"ray_job_{job_id}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run_jobs(args: argparse.Namespace) -> None:
    from ray.job_submission import JobStatus, JobSubmissionClient

    headers = {
        "af-max-workers": str(args.max_workers),
        "af-cpus-per-worker": str(args.cpus_per_worker),
        "af-ram-per-worker-gb": str(args.ram_per_worker_gb),
    }
    todo = [r for r in args.regions if not region_done(args.out, r)]
    log(args.out, f"headers={headers} todo={todo} skip(done)={[r for r in args.regions if r not in todo]}")
    if args.dry_run:
        for region in todo:
            print(f"[{region}] {entrypoint(args, region)}")
        return
    if not todo:
        return

    client = JobSubmissionClient(args.address, headers=headers)
    deadline = time.time() + args.max_wait_h * 3600
    attempts = {r: 0 for r in todo}
    running: dict[str, str] = {}
    failed: list[str] = []

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _raise_terminated)
    try:
        pending = list(todo)
        while pending or running:
            while pending:
                region = pending.pop(0)
                attempts[region] += 1
                job_id = submit(client, args, region, deadline)
                running[region] = job_id
                log(args.out, f"START {region} job={job_id} attempt={attempts[region]}")
            time.sleep(args.poll_s)
            for region, job_id in list(running.items()):
                status = client.get_job_status(job_id)
                if not status.is_terminal():
                    continue
                del running[region]
                save_job_log(client, args, region, job_id)
                if status == JobStatus.SUCCEEDED and region_done(args.out, region):
                    log(args.out, f"OK {region} job={job_id}")
                elif attempts[region] < args.max_attempts:
                    log(args.out, f"RETRY {region} job={job_id} status={status}")
                    pending.append(region)
                else:
                    log(args.out, f"FAIL {region} job={job_id} status={status}")
                    failed.append(region)
    except _Terminated as e:
        log(args.out, f"got {e}: stopping {running}")
        for job_id in running.values():
            try:
                client.stop_job(job_id)
            except Exception as err:
                log(args.out, f"stop_job {job_id} failed: {err}")
        sys.exit(143)
    if failed:
        raise SystemExit(f"Regions failed: {failed}")


def merge(args: argparse.Namespace) -> None:
    missing = [r for r in args.regions if not region_done(args.out, r)]
    if missing:
        raise SystemExit(f"Cannot merge, regions not done: {missing}")
    summaries, composition, states = [], {}, {}
    for region in args.regions:
        job_dir = args.out / "jobs" / region
        link = args.out / region
        if not link.exists():
            link.symlink_to(Path("jobs") / region / region)
        shutil.copy2(job_dir / f"summary_{region}.json", args.out / f"summary_{region}.json")
        summaries.append(json.loads((job_dir / f"summary_{region}.json").read_text()))
        comp = job_dir / "sample_composition.json"
        if comp.is_file():
            composition[region] = json.loads(comp.read_text())
        states[region] = json.loads((job_dir / "git_state.json").read_text())
    if len({json.dumps(s, sort_keys=True) for s in states.values()}) != 1:
        log(args.out, f"WARNING: git_state differs between region jobs: {states}")
    first = args.out / "jobs" / args.regions[0]
    for name in SHARED_FILES:
        if (first / name).is_file():
            shutil.copy2(first / name, args.out / name)
    (args.out / "summary_all.json").write_text(json.dumps(summaries, indent=2))
    (args.out / "sample_composition.json").write_text(json.dumps(composition, indent=2))
    log(args.out, f"MERGED {[s['region'] for s in summaries]} -> {args.out}/summary_all.json")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not os.access(args.train_python, os.X_OK):
        raise SystemExit(f"--train-python {args.train_python} is not executable")
    if not args.merge_only:
        run_jobs(args)
        if args.dry_run:
            return
    merge(args)


if __name__ == "__main__":
    main()
