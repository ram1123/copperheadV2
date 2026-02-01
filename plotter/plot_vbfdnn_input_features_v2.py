#!/usr/bin/env python3

import argparse
from pathlib import Path

import dask_awkward as dak
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

from rich import print

from modules.trials import get_stage1_path
from modules.dask_utils import get_dask_client, close_dask_client

plt.style.use(hep.style.CMS)

# -------------------------------------------------
# FEATURES CONFIG
# -------------------------------------------------
from configs.dnn_features import FEATURES


def main(args):
    client = get_dask_client(args.use_gateway)

    stage1_dir = get_stage1_path()
    load_path = Path(stage1_dir) / args.year / "compacted" / args.process / "0" / "*.parquet"
    print(f"[INFO] Reading: {load_path}")

    # args.output + process
    out_path = Path(args.output)
    out_path = out_path.with_name(f"{out_path.stem}_{args.process}{out_path.suffix}")
    print(f"[INFO] Output: {out_path}")


    feat_items = list(FEATURES.items())
    n = len(feat_items)
    ncols = args.ncols
    nrows = int(np.ceil(n / ncols))

    fig_w = 4.2 * ncols
    fig_h = 3.3 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)

    for i, (name, cfg) in enumerate(feat_items):
        ax = axes[i]
        col = cfg["column"]

        try:
            arr = dak.from_parquet(str(load_path), columns=[col])[col].compute()
            arr = arr[~ak.is_none(arr)]
            values = ak.to_numpy(arr)

            if values.size == 0:
                ax.text(0.5, 0.5, f"{name}\n(empty)", ha="center", va="center")
                ax.set_axis_off()
                print(f"[SKIP] {name}: empty")
                continue

            ax.hist(
                values,
                bins=cfg["bins"],
                range=cfg["range"],
                histtype="step",
                lw=1.6,
            )
            ax.set_title(name, fontsize=10)
            ax.set_xlabel(cfg["title"], fontsize=9)
            ax.set_ylabel("Events", fontsize=9)
            ax.tick_params(axis="both", labelsize=8)

            print(f"[OK] {name}")

        except Exception as e:
            ax.text(0.5, 0.5, f"{name}\nFAIL", ha="center", va="center")
            ax.set_axis_off()
            print(f"[FAIL] {name}: {e}")

    # turn off unused pads
    for j in range(n, len(axes)):
        axes[j].set_axis_off()

    # hep.cms.label(ax=axes[0], data=True, label="Private work", com="13")
    fig.suptitle(f"Input feature distributions ({args.year})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)

    close_dask_client()
    print(f"[DONE] Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="2018")
    parser.add_argument("--process", default="dy")
    parser.add_argument("--output", default="all_features_onepage.pdf")
    parser.add_argument("--use-gateway", action="store_true")
    parser.add_argument("--ncols", type=int, default=4, help="subplot columns")
    args = parser.parse_args()

    main(args)
