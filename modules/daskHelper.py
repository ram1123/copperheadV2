# ----------------------------------------------------------------------
# Dask client helper
# ----------------------------------------------------------------------
from distributed import Client

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
