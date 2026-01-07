# ----------------------------------------------------------------------
# Dask client helper
# ----------------------------------------------------------------------
from distributed import Client
from dask_gateway import Gateway

def get_dask_client(
    n_workers: int = 12,
    threads_per_worker: int = 1,
    memory_limit: str = "10 GiB",
) -> Client:
    """Create or reuse a local Dask client."""
    try:
        client = Client.current()
        print(f"Reusing existing Dask client: {client}")
        return client
    except ValueError:
        client = Client(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            processes=True,
            memory_limit=memory_limit,
        )
        print(f"Created new Dask client: {client}")
        return client

def close_dask_client():
    """Close the current Dask client if it exists."""
    try:
        client = Client.current()
        client.close()
        print("Closed Dask client.")
    except ValueError:
        print("No Dask client to close.")

def get_dask_gateway_client():
    """Create or reuse a Dask Gateway client."""
    try:
        print("Attempting to get existing Dask Gateway client...")
        client = Client.current()
        print(f"Reusing existing Dask client: {client}")
        return client
    except ValueError:
        print("No existing Dask client found. Creating a new Dask Gateway client...")
        gateway = Gateway()
        cluster_info = gateway.list_clusters()[0]  # get the first cluster by default. There only should be one anyways
        client = gateway.connect(cluster_info.name).get_client()
        print(f"Created new Dask Gateway client: {client}")
        return client
