#!/usr/bin/env python3
"""Create / stop / inspect a Dask Gateway cluster, non-interactively.

The 2017 fast-iteration chain cycles the cluster several times per lambda: a
fresh one for the compact step, down again while ``scan_bins_for_dnn.py`` runs
its own 64-process *local* client, and a fresh one for Stage-2.  Holding a
gateway cluster through a multi-hour training or through the local scan is pure
waste on a shared facility, and the gateway culls idle clusters anyway.

Settings follow ``dask_cluster_params.txt`` (worker_cores=2, worker_memory=25).

Creation must go through the Options object with a minimal ``env``; passing
these as ``new_cluster()`` kwargs, or including extra keys such as
CPLUS_INCLUDE_PATH, fails server-side with ``ValueError('PATH')``.
``shutdown_on_close=False`` keeps the cluster alive after this process exits.

    dask_cluster.py create --workers 40   # waits until workers are up
    dask_cluster.py status
    dask_cluster.py stop
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone

from dask_gateway import Gateway

ADDRESS = "http://dask-gateway-k8s.geddes.rcac.purdue.edu/"
PROXY = "traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786"
REPO = "/work/users/yun79/sideHustle2/copperheadV2"
PIXI_PROJECT = "/cvmfs/cms-af.opensciencegrid.org/paf/pixi/copperheadV2"


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}", flush=True)


def gateway() -> Gateway:
    return Gateway(ADDRESS, proxy_address=PROXY)


def worker_env() -> dict:
    return {
        "PATH": f"{PIXI_PROJECT}/.pixi/envs/default/bin/:/usr/local/sbin:/usr/local/bin:"
                "/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": os.environ.get("HOME", "/home/yun79"),
        "USER": os.environ.get("USER", "yun79"),
        "LOGNAME": os.environ.get("LOGNAME", "yun79"),
        "PYTHONPATH": REPO,
        "X509_USER_PROXY": f"{REPO}/voms_proxy.txt",
        "XRD_REQUESTTIMEOUT": "300",
    }


def n_live(cluster) -> int:
    try:
        return len(cluster.get_client().scheduler_info()["workers"])
    except Exception:
        return 0


def cmd_status(args) -> int:
    g = gateway()
    clusters = g.list_clusters()
    if not clusters:
        print("0")
        log("no clusters")
        return 1
    live = n_live(g.connect(clusters[0].name))
    print(live)
    log(f"{len(clusters)} cluster(s); {clusters[0].name} has {live} worker(s)")
    return 0


def cmd_create(args) -> int:
    g = gateway()
    existing = g.list_clusters()
    if existing and not args.force:
        live = n_live(g.connect(existing[0].name))
        log(f"cluster {existing[0].name} already exists with {live} worker(s); reusing")
        if live >= args.min_workers:
            return 0
        log("existing cluster is under-populated; scaling it instead of creating a new one")
        cluster = g.connect(existing[0].name)
    else:
        opts = g.cluster_options()
        opts.pixi_project = PIXI_PROJECT
        opts.pixi_env = "default"
        opts.worker_cores = args.cores
        opts.worker_memory = args.memory
        opts.env = worker_env()
        cluster = g.new_cluster(opts, shutdown_on_close=False)
        log(f"created {cluster.name} (cores={args.cores} memory={args.memory}GB)")

    cluster.scale(args.workers)
    log(f"scaled to {args.workers}; waiting for >= {args.min_workers}")

    deadline = time.time() + args.timeout
    live = 0
    while time.time() < deadline:
        live = n_live(cluster)
        if live >= args.min_workers:
            log(f"ready with {live} worker(s)")
            return 0
        time.sleep(15)

    log(f"TIMEOUT after {args.timeout}s with only {live} worker(s)")
    return 0 if live > 0 else 2


def cmd_stop(args) -> int:
    g = gateway()
    clusters = g.list_clusters()
    if not clusters:
        log("nothing to stop")
        return 0
    for info in clusters:
        g.stop_cluster(info.name)
        log(f"stopped {info.name}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create", help="create (or top up) a cluster and wait for workers")
    c.add_argument("--cores", type=float, default=2, help="worker_cores")
    c.add_argument("--memory", type=float, default=25, help="worker_memory, GiB")
    c.add_argument("--workers", type=int, default=40, help="target worker count")
    c.add_argument("--min-workers", type=int, default=10,
                   help="return once this many are up, rather than waiting for all")
    c.add_argument("--timeout", type=int, default=900)
    c.add_argument("--force", action="store_true",
                   help="create a new cluster even if one already exists")
    c.set_defaults(func=cmd_create)

    s = sub.add_parser("stop", help="stop every cluster this user owns")
    s.set_defaults(func=cmd_stop)

    q = sub.add_parser("status", help="print the live worker count on stdout")
    q.set_defaults(func=cmd_status)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
