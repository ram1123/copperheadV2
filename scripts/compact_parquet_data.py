import os
import dask_awkward as dak
import argparse
from distributed import Client
import logging
from modules.utils import logger


def ensure_compacted(year, sample, load_path, compacted_dir):
    compacted_path = os.path.join(compacted_dir, sample)
    print(f"Checking compacted dataset: {compacted_path}")

    if not os.path.exists(compacted_path):
        print(f"Compacted dataset not found. Creating at {compacted_path}")

        orig_path = os.path.join(load_path, sample)
        if not os.path.exists(orig_path):
            print(f"Original data not found at {orig_path}. Skipping.")
            return

        print(f"Reading data from {orig_path}")
        inFile = dak.from_parquet(orig_path)

        target_chunksize = 250_000
        inFile = inFile.repartition(rows_per_partition=target_chunksize)

        print(f"Writing compacted data to {compacted_path}")
        inFile.to_parquet(compacted_path)
        print(f"Dataset successfully compacted.")
    else:
        print(f"Compacted dataset already exists at {compacted_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compacts parquet datasets.")
    parser.add_argument("-y", "--year", required=True, help="Year of the dataset")
    parser.add_argument("-l", "--load_path", required=True, help="Path to the original dataset")
    parser.add_argument("-c", "--compacted_dir", default="", help="Path to store the compacted dataset")
    parser.add_argument("--use_gateway", action="store_true", help="Use Dask Gateway client")

    args = parser.parse_args()

    if args.use_gateway:
        from dask_gateway import Gateway
        gateway = Gateway(
            "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
            proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
        )
        cluster_info = gateway.list_clusters()[0]  # get the first cluster by default. There only should be one anyways
        client = gateway.connect(cluster_info.name).get_client()
        logger.info("Gateway Client created")
    else:
        client = Client(n_workers=64, threads_per_worker=1, processes=True, memory_limit='10 GiB')
        logger.info("Local scale Client created")

    # append /stage1_output/2018/f1_0 to load path
    args.load_path = os.path.join(args.load_path, f"stage1_output/{args.year}/f1_0")

    if not args.compacted_dir:
        print("No compacted directory provided, using default.")
        args.compacted_dir = (args.load_path).replace("f1_0", "compacted_ch250k")
        print(f"Compacted directory set to: {args.compacted_dir}")

    samples = os.listdir(args.load_path)
    for sample in samples:
        print(f"Processing sample: {sample}")
        ensure_compacted(args.year, sample, args.load_path, args.compacted_dir)
