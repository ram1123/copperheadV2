#!/usr/bin/env python3
"""
Merge the per-chunk cutflow .npz shards stage-1 writes for every completed
chunk (--isCutflow, see src/copperhead_processor.py's cutflow block and
src/stage1/cutflow_io.py::write_cutflow_outputs) into one combined cutflow
for a whole sample/dataset, and print it as a table.

Each shard's .npz holds coffea's Cutflow.to_npz() output: `labels` (cut
names, prefixed with a synthetic "initial" entry for the pre-any-cut
baseline -- confirmed on real 2026-09 output: 21 labels for 20 registered
cuts), `nevonecut` (event count passing that cut ALONE), `nevcutflow`
(event count passing ALL cuts up to and including that one -- the
"cumulative" column), plus `masksonecut`/`maskscutflow` (the underlying
per-event boolean arrays, NOT merged here -- concatenating them across a
whole dataset's chunks would be both huge and not particularly useful post-
merge; only the summary counts are combined).

Chunks are disjoint subsets of events, so simple element-wise summation of
`nevonecut`/`nevcutflow` across all of a sample's shards gives the correct
whole-dataset cutflow (cumulative-AND is computed per chunk, and summing a
per-chunk cumulative count across disjoint chunks equals the same cumulative
count over their union).

Usage
-----
    python scripts/merge_cutflow_npz_file.py <dir_or_glob> [<dir_or_glob> ...] [-o OUT_JSON] [--out-npz OUT_NPZ]

<dir_or_glob> is either a directory (searched recursively for
cutflow_*.npz) or a glob pattern matching .npz files directly, e.g.:

    # one sample, all its chunks
    python scripts/merge_cutflow_npz_file.py \\
        /work/projects/hmm/<user>/hmm_ntuples/copperheadV1clean/<label>/stage1_output/<year>/f1_0/data_C/0 \\
        -o cutflow_data_C_merged.json

    # every data era in one year (glob across sample dirs)
    python scripts/merge_cutflow_npz_file.py \\
        "/work/projects/hmm/<user>/hmm_ntuples/copperheadV1clean/<label>/stage1_output/<year>/f1_0/data_*/0" \\
        -o cutflow_data_merged.json
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def find_npz_files(target: str) -> list[str]:
    if os.path.isdir(target):
        return sorted(glob.glob(os.path.join(target, "**", "cutflow_*.npz"), recursive=True))
    return sorted(glob.glob(target))


def merge_cutflows(paths: list[str]):
    """Returns (labels, nevonecut, nevcutflow, n_files_merged)."""
    if not paths:
        raise RuntimeError("No .npz files to merge.")

    labels = None
    sum_nevonecut = None
    sum_nevcutflow = None

    for path in paths:
        data = np.load(path, allow_pickle=True)
        this_labels = data["labels"]
        this_nevonecut = data["nevonecut"].astype(np.int64)
        this_nevcutflow = data["nevcutflow"].astype(np.int64)

        if labels is None:
            labels = this_labels
            sum_nevonecut = this_nevonecut.copy()
            sum_nevcutflow = this_nevcutflow.copy()
        else:
            if not np.array_equal(this_labels, labels):
                raise RuntimeError(
                    f"Cut label/order mismatch in {path}: {list(this_labels)} != {list(labels)} "
                    "-- shards from different scenarios/switch configs shouldn't be merged together."
                )
            sum_nevonecut += this_nevonecut
            sum_nevcutflow += this_nevcutflow

    return labels, sum_nevonecut, sum_nevcutflow, len(paths)


def print_cutflow(labels, nevonecut, nevcutflow):
    total = int(nevonecut[0])  # "initial", before any cut
    print(f"Cutflow stats ({total} events before any cut):")
    for name, individual, cumulative in zip(labels, nevonecut, nevcutflow):
        ind_eff = 100.0 * individual / total if total else float("nan")
        cum_eff = 100.0 * cumulative / total if total else float("nan")
        print(
            f"  {name:28s} individual={individual:<10d} ({ind_eff:5.1f}%)   "
            f"cumulative={cumulative:<10d} ({cum_eff:5.1f}%)"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", help="directory (searched recursively) or glob pattern per sample/group")
    ap.add_argument("-o", "--out-json", help="write the merged cutflow as JSON here")
    ap.add_argument("--out-npz", help="also write the merged arrays back out as a .npz")
    args = ap.parse_args()

    paths = []
    for target in args.targets:
        found = find_npz_files(target)
        if not found:
            print(f"[WARN] no cutflow_*.npz found for target: {target}")
        paths.extend(found)

    labels, nevonecut, nevcutflow, n_files = merge_cutflows(paths)
    print(f"Merged {n_files} shard(s):")
    for p in paths:
        print(f"  {p}")
    print()
    print_cutflow(labels, nevonecut, nevcutflow)

    if args.out_json:
        import json
        combined = {
            str(name): {"individual": int(ind), "cumulative": int(cum)}
            for name, ind, cum in zip(labels, nevonecut, nevcutflow)
        }
        with open(args.out_json, "w") as handle:
            json.dump(combined, handle, indent=2)
        print(f"\nWrote merged JSON to {args.out_json}")

    if args.out_npz:
        np.savez(args.out_npz, labels=labels, nevonecut=nevonecut, nevcutflow=nevcutflow)
        print(f"Wrote merged npz to {args.out_npz}")


if __name__ == "__main__":
    main()
