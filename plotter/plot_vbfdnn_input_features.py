import argparse
from pathlib import Path

import dask_awkward as dak
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

from modules.trials import get_stage1_path
from modules.dask_utils import get_dask_client, close_dask_client
from configs.dnn_features import FEATURES

def load_feature(parquet_glob, column):
    arr = dak.from_parquet(parquet_glob, columns=[column])[column]
    arr = arr.compute()
    arr = ak.fill_none(arr, np.nan)
    arr = ak.to_numpy(arr)
    arr = arr[np.isfinite(arr)]
    return arr


def plot_feature(values, cfg, outpath):
    plt.figure(figsize=(6, 5))
    plt.hist(
        values,
        bins=cfg["bins"],
        range=cfg["range"],
        histtype="step",
        linewidth=2,
    )
    plt.xlabel(cfg["title"])
    plt.ylabel("Events")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.yscale("log")
    plt.savefig(outpath.with_suffix(".log.pdf"))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="2018")
    parser.add_argument("--process", default="dy")
    parser.add_argument("--use-gateway", action="store_true")
    parser.add_argument("--outdir", default="./validation/dnn_input_features")
    args = parser.parse_args()

    client = get_dask_client(args.use_gateway)

    stage1_dir = Path(get_stage1_path())
    parquet_glob = (
        stage1_dir
        / args.year
        / "compacted"
        / args.process
        / "0"
        / "*.parquet"
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading: {parquet_glob}")

    for feat, cfg in FEATURES.items():
        print(f"[INFO] Plotting {feat}")
        values = load_feature(str(parquet_glob), cfg["column"])
        print(f"  entries = {values.size}")
        plot_feature(values, cfg, outdir / f"{feat}.pdf")

    close_dask_client()
    print("[INFO] Done")

if __name__ == "__main__":
    main()
