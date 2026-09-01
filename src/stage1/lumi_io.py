import json
import os

import awkward as ak
import numpy as np

from modules.utils import logger


def write_processed_lumis(run_arr, lumi_arr, save_path, dataset_name, file_idx):
    """
    Write the unique (run, luminosityBlock) pairs actually read in this chunk to
    a small per-shard JSON, right alongside the cutflow shard files written by
    write_cutflow_outputs() (src/stage1/cutflow_io.py).

    It collects per-chunk record of which lumi sections were actually run over, 
    so the whole dataset's shards can later be merged + range-compressed into 
    one golden-JSON-format file usable directly with `brilcalc lumi -i <file>` 
    to get the true processed luminosity.

    We can compute the processed lumi and compare it with golden json lumi. 
    This will help us to ensure that we are not missing any data events.
    """
    runs = np.asarray(ak.to_numpy(run_arr))
    lumis = np.asarray(ak.to_numpy(lumi_arr))
    if runs.size == 0:
        return

    pairs = np.unique(np.c_[runs, lumis], axis=0)

    # group lumis by run: {"<run>": [lumi, lumi, ...]} (unsorted-free, unique)
    per_run: dict[str, list[int]] = {}
    for run_val, lumi_val in pairs:
        per_run.setdefault(str(int(run_val)), []).append(int(lumi_val))
    for lumi_list in per_run.values():
        lumi_list.sort()

    base_name = f"processedlumis_{dataset_name}_{file_idx}"
    json_path = os.path.join(save_path, f"{base_name}.json")
    try:
        with open(json_path, "w") as handle:
            json.dump(per_run, handle)
        logger.info(f"Processed-lumi shard saved to {json_path}")
    except Exception as err:
        logger.error(f"Processed-lumi shard save failed: {err}")
