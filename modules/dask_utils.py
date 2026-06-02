# ------------------------------------------------------------------
# Dask client helper
# ------------------------------------------------------------------
from modules.utils import logger

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
        client.run(lambda env=xrd_env: __import__("os").environ.update(env))
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
