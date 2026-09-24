"""
===============================================================================
Dask Gateway Cluster Management Script
===============================================================================

This script provisions, monitors, and cleans up Dask Gateway clusters on 
Kubernetes/Slurm backends. It handles proxy paths, environment variable setup, 
and provides commands for lifecycle management.

PRE-REQUISITES & ENVIRONMENT SETUP
----------------------------------
Before executing this script, ensure your Grid proxy is active. You can 
initialize it directly in your terminal:

    voms-proxy-init -voms cms -rfc -valid 192:00 --out $(pwd)/voms_proxy.txt
    export X509_USER_PROXY=$(pwd)/voms_proxy.txt

Alternatively, source your environment setup script:

    ./enter_pixi.sh

USAGE EXAMPLES
--------------
1. Create a new cluster (cleans up any existing active clusters first):
    python manage_dask.py --create

2. Retrieve active dashboard links:
    python manage_dask.py --dashboard

3. Recreate a cluster (shutdown existing + create new):
    python manage_dask.py --recreate

4. Shutdown all active clusters without spinning up a new one:
    python manage_dask.py --shutdown-all

===============================================================================
"""
import argparse
import os
import sys
import time
from pathlib import Path
from dask_gateway import Gateway


def setup_environment():
    username = os.environ.get("USER") or os.environ.get("USERNAME")
    if not username:
        print("Error: Username not found in environment variables.")
        sys.exit(1)

    cwd = str(Path.cwd())
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        f"{current_pythonpath}:{cwd}" if current_pythonpath else cwd
    )
    os.environ["X509_USER_PROXY"] = f"{cwd}/voms_proxy.txt"
    os.environ["XRD_REQUESTTIMEOUT"] = "300"


def shutdown_cluster(gateway, cluster_name, attempts=3):
    if attempts <= 0:
        return

    try:
        cluster = gateway.connect(cluster_name)
        status = str(getattr(cluster, "status", "")).upper()

        if status in ("CLOSED", "STOPPED"):
            print(f"Cluster {cluster_name} is closed.")
            return

        cluster.shutdown()
    except Exception:
        pass

    time.sleep(2)
    shutdown_cluster(gateway, cluster_name, attempts - 1)


def shutdown_all_clusters(gateway):
    clusters = gateway.list_clusters()
    if not clusters:
        print("No active clusters found.")
        return

    print(f"Shutting down {len(clusters)} active cluster(s)...")
    for cluster_info in clusters:
        shutdown_cluster(gateway, cluster_info.name)


def create_cluster(gateway, scale_workers=59):
    print("Creating new cluster...")
    cluster = gateway.new_cluster(
        pixi_project="/cvmfs/cms-af.opensciencegrid.org/paf/pixi/copperheadV2",
        worker_cores=2,
        worker_memory=20,
        env=dict(os.environ),
    )
    cluster.scale(scale_workers)
    print(f"Cluster provisioned: {cluster.name}")
    print(f"Dashboard Link: {cluster.dashboard_link}")
    return cluster


def print_dashboard_links(gateway):
    clusters = gateway.list_clusters()
    if not clusters:
        print("No active clusters found to retrieve dashboard links.")
        return

    for cluster_info in clusters:
        cluster = gateway.connect(cluster_info.name)
        print(f"Cluster: {cluster_info.name}")
        print(f"Dashboard Link: {cluster.dashboard_link}\n")


def main():
    setup_environment()
    gateway = Gateway()

    parser = argparse.ArgumentParser(
        description="Manage Dask Gateway Clusters"
    )
    parser.add_argument(
        "--create", action="store_true", help="Create a new cluster"
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Get dashboard link(s) for existing cluster(s)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Shutdown existing clusters and create a new one",
    )
    parser.add_argument(
        "--shutdown-all",
        action="store_true",
        help="Shutdown all existing clusters",
    )

    args = parser.parse_args()

    if args.recreate or args.create:
        shutdown_all_clusters(gateway)
        create_cluster(gateway)
    elif args.dashboard:
        print_dashboard_links(gateway)
    elif args.shutdown_all:
        shutdown_all_clusters(gateway)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
