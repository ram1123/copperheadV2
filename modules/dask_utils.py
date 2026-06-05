# ------------------------------------------------------------------
# Dask client helper
# ------------------------------------------------------------------
import os
import sys

from modules.utils import logger


def configure_worker_runtime(extra_env=None, repo_root=None):
    """
    Ensure gateway workers can import the live checkout and inherit runtime env.
    """
    if extra_env:
        os.environ.update(extra_env)

    if repo_root:
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        current_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath_parts = [part for part in current_pythonpath.split(os.pathsep) if part]
        if repo_root not in pythonpath_parts:
            pythonpath_parts.insert(0, repo_root)
            os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

        try:
            os.chdir(repo_root)
        except OSError:
            pass

    return {
        "cwd": os.getcwd(),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "syspath0": sys.path[0] if sys.path else "",
    }

def get_dask_client(
    use_gateway: bool = False,
    n_workers: int = 12,
    threads_per_worker: int = 1,
    memory_limit: str = "10 GiB",
    cluster_index: int = 0,
):
    """
    Create or reuse a Dask client (local or gateway).
    """
    if use_gateway:
        from dask_gateway import Gateway
        logger.info("Creating new Dask Gateway client")
        gateway = Gateway(
            "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
            proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
        )
        clusters = gateway.list_clusters()
        if not clusters:
            raise RuntimeError("No Dask Gateway clusters available")
        client = gateway.connect(clusters[cluster_index].name).get_client()
        xrd_env = {
            "XRD_REQUESTTIMEOUT": "900",
            "XRD_STREAMTIMEOUT": "900",
            "XRD_CONNECTIONRETRY": "16",
            "XRD_REDIRECTLIMIT": "16",
            "XRD_CONNECTIONWINDOW": "120",
            "XRD_TIMEOUTRESOLUTION": "5",
        }
        runtime_info = client.run(
            configure_worker_runtime,
            extra_env=xrd_env,
            repo_root=os.getcwd(),
        )
        logger.info("Configured worker runtime: %s", runtime_info)
    else:
        from distributed import Client
        logger.info("Creating new local Dask client")
        client = Client(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            processes=True,
            memory_limit=memory_limit,
            dashboard_address=None,
        )

    logger.info(f"Created Dask client: {client}")
    return client


def close_dask_client():
    """Close the current Dask client if it exists."""
    from distributed import Client
    try:
        client = Client.current()
        client.close()
        logger.info("Closed Dask client")
    except ValueError:
        logger.info("No Dask client to close")
