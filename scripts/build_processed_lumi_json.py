#!/usr/bin/env python3
"""
Merge the per-chunk "processed lumi" shard files stage-1 writes for data
samples (``src/stage1/lumi_io.py::write_processed_lumis``, called from
``src/stage1/runner_adapter.py`` for every successfully completed chunk) into
one golden-JSON-format ``processedLumis.json``: which lumi
sections were actually run over. Feed the result to ``brilcalc lumi -i
<file>`` for the true processed luminosity, or pass ``--golden-json`` here to
directly check it against the certified lumimask (are we actually complete?).

``run_stage1.py`` calls ``build_report()`` automatically at the end of every
full (non-test-mode) run, so normally you don't need to run this by hand --
it's also runnable standalone (e.g. to re-report after resuming a partially
failed run, or to point --golden-json at a different mask after the fact).

Each shard file is ``{"<run>": [lumi, lumi, ...]}`` (unsorted, uncompressed),
written next to that chunk's parquet output at
``<stage1_dir>/f<fraction>/<dataset>/<idx>/processedlumis_<dataset>_<shard>.json``
(mirroring the existing ``cutflow_<dataset>_<shard>.json`` shard files). This
script unions all shards per dataset (and overall), sorts + range-compresses
the lumis into ``[[start, end], ...]`` pairs, and writes:

  <stage1_dir>/_status/processedLumis.json               (combined, CRAB-style name; everything actually read)
  <stage1_dir>/_status/processedLumis_certified.json     (processed & golden, if --golden-json given -- the
                                                            subset that's actually certified; this is what to
                                                            hand brilcalc / eyeball against the golden JSON)
  <stage1_dir>/_status/processed_lumis/<dataset>.json    (per-dataset breakdown)
  <stage1_dir>/_status/processed_lumis/missing.json      (golden \\ processed, if --golden-json given and non-empty)
  <stage1_dir>/_status/processed_lumis/unexpected.json   (processed \\ golden, if --golden-json given and non-empty)

Usage
-----
    python scripts/build_processed_lumi_json.py <stage1_dir> [--golden-json PATH]

<stage1_dir> is a directory like ``.../stage1_output/2024`` (i.e. the
``start_save_path`` passed to ``dataset_loop`` in run_stage1.py) -- it's walked
recursively for ``processedlumis_*.json`` shard files. --golden-json points at
the certified lumimask for that year (see configs/parameters/lumi.yaml).

Caveat on --golden-json "missing" lumis: a real gap can also mean this
specific dataset/primary-dataset legitimately has no data for those runs/lumis
(trigger/PD/detector availability changes across a run), not necessarily a
processing failure -- cross-check against job_status before treating "missing"
as "stage-1 failed on this".
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict


def find_shard_files(root: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root, "**", "processedlumis_*.json"), recursive=True))


def dataset_name_from_shard_path(path: str) -> str:
    """
    Shard files live at .../f<fraction>/<dataset>/<idx>/processedlumis_....json
    (see getSavePath() in run_stage1.py), so the dataset name is the directory
    two levels above the file. Falls back to "unknown" if the layout doesn't
    match (e.g. someone pointed this at a non-standard directory).
    """
    idx_dir = os.path.dirname(path)
    dataset_dir = os.path.dirname(idx_dir)
    name = os.path.basename(dataset_dir)
    return name if name else "unknown"


def compress_lumis(lumis) -> list[list[int]]:
    """Iterable of (possibly unsorted, possibly duplicate) lumi numbers -> contiguous [start, end] ranges (golden-JSON style)."""
    ordered = sorted(set(lumis))
    ranges: list[list[int]] = []
    start = prev = None
    for lumi in ordered:
        if start is None:
            start = prev = lumi
        elif lumi == prev + 1:
            prev = lumi
        else:
            ranges.append([start, prev])
            start = prev = lumi
    if start is not None:
        ranges.append([start, prev])
    return ranges


def expand_ranges(ranges) -> set[int]:
    """[[start, end], ...] -> the set of individual lumi numbers it covers."""
    out: set[int] = set()
    for start, end in ranges:
        out.update(range(start, end + 1))
    return out


def merge_shards(shard_paths: list[str]) -> dict[str, set[int]]:
    """run (str) -> set of lumi sections, unioned across all given shard files."""
    merged: dict[str, set[int]] = defaultdict(set)
    for path in shard_paths:
        try:
            with open(path) as handle:
                per_run = json.load(handle)
        except (OSError, json.JSONDecodeError) as err:
            print(f"[WARN] skipping unreadable shard {path}: {err}")
            continue
        for run, lumis in per_run.items():
            merged[run].update(lumis)
    return merged


def to_golden_json(merged: dict[str, set[int]]) -> dict:
    return {run: compress_lumis(lumis) for run, lumis in sorted(merged.items(), key=lambda kv: int(kv[0]))}


def summarize(name: str, merged: dict[str, set[int]]) -> str:
    n_runs = len(merged)
    n_lumis = sum(len(v) for v in merged.values())
    n_ranges = sum(len(compress_lumis(v)) for v in merged.values())
    return f"{name}: {n_runs} runs, {n_lumis} lumi sections, {n_ranges} ranges"


def compare_to_golden(processed: dict[str, set[int]], golden_json_path: str):
    """
    Returns (missing, unexpected): both {run: set(lumis)}, dropping empty runs.
    missing    = golden \\ processed  (certified lumis we did NOT process)
    unexpected = processed \\ golden  (processed lumis outside the certified mask --
                 usually means the wrong lumimask was used, or a stale one)

    Also returns `certified` = processed & golden (the intersection): the
    subset of processedLumis.json that's actually within the certified mask --
    this is the number that matters for "did I process what the golden JSON
    says I should have", and what you'd hand to brilcalc for the luminosity
    that's both processed AND certified.
    """
    with open(golden_json_path) as handle:
        golden_raw = json.load(handle)
    golden = {run: expand_ranges(ranges) for run, ranges in golden_raw.items()}

    missing: dict[str, set[int]] = {}
    unexpected: dict[str, set[int]] = {}
    certified: dict[str, set[int]] = {}

    for run, lumis in golden.items():
        diff = lumis - processed.get(run, set())
        if diff:
            missing[run] = diff
    for run, lumis in processed.items():
        diff = lumis - golden.get(run, set())
        if diff:
            unexpected[run] = diff
        overlap = lumis & golden.get(run, set())
        if overlap:
            certified[run] = overlap

    return missing, unexpected, certified


def build_report(stage1_dir: str, golden_json_path: str | None = None, out_dir: str | None = None) -> str:
    """
    Do the full merge (+ optional golden-JSON comparison) for one stage1_dir and
    write the output files described in the module docstring. Returns a short
    human-readable summary string (also safe to log from run_stage1.py).
    Never raises for "no shards found" (e.g. an MC-only run) -- just reports it.
    """
    stage1_dir = os.path.normpath(stage1_dir)
    shard_paths = find_shard_files(stage1_dir)
    lines = []
    if not shard_paths:
        return (f"[processed-lumi report] no processedlumis_*.json shards found under {stage1_dir} "
                "(expected if this run was MC-only, or ran before this feature existed)")

    out_dir = out_dir or os.path.join(stage1_dir, "_status")
    per_dataset_dir = os.path.join(out_dir, "processed_lumis")
    os.makedirs(per_dataset_dir, exist_ok=True)

    by_dataset: dict[str, list[str]] = defaultdict(list)
    for path in shard_paths:
        by_dataset[dataset_name_from_shard_path(path)].append(path)

    overall: dict[str, set[int]] = defaultdict(set)
    lines.append(f"{len(shard_paths)} shard files, {len(by_dataset)} datasets")
    for dataset, paths in sorted(by_dataset.items()):
        merged = merge_shards(paths)
        for run, lumis in merged.items():
            overall[run].update(lumis)
        out_path = os.path.join(per_dataset_dir, f"{dataset}.json")
        with open(out_path, "w") as handle:
            json.dump(to_golden_json(merged), handle, indent=2, sort_keys=True)
        lines.append(f"  {summarize(dataset, merged)}")

    combined_path = os.path.join(out_dir, "processedLumis.json")
    with open(combined_path, "w") as handle:
        json.dump(to_golden_json(overall), handle, indent=2, sort_keys=True)
    lines.append(summarize("TOTAL", overall))
    lines.append(f"processedLumis.json written to: {combined_path}")

    if golden_json_path:
        if os.path.exists(golden_json_path):
            missing, unexpected, certified = compare_to_golden(overall, golden_json_path)
            n_missing = sum(len(v) for v in missing.values())
            n_unexpected = sum(len(v) for v in unexpected.values())
            n_golden = sum(len(expand_ranges(r)) for r in json.load(open(golden_json_path)).values())
            coverage = 100.0 * (n_golden - n_missing) / n_golden if n_golden else float("nan")
            lines.append(f"vs. golden JSON ({golden_json_path}): {coverage:.3f}% of certified lumis processed "
                         f"({n_missing} missing, {n_unexpected} unexpected/outside-mask)")

            # processedLumis.json filtered down to only the golden-certified lumis
            # (processed & golden) -- this is the file to feed brilcalc for the
            # official processed-and-certified luminosity, and to directly
            # eyeball against the golden JSON to confirm completeness.
            certified_path = os.path.join(out_dir, "processedLumis_certified.json")
            with open(certified_path, "w") as handle:
                json.dump(to_golden_json(certified), handle, indent=2, sort_keys=True)
            lines.append(f"processedLumis_certified.json (processed ∩ golden) written to: {certified_path}")

            if missing:
                missing_path = os.path.join(per_dataset_dir, "missing.json")
                with open(missing_path, "w") as handle:
                    json.dump(to_golden_json(missing), handle, indent=2, sort_keys=True)
                lines.append(f"  missing lumis (golden minus processed) written to: {missing_path}")
            if unexpected:
                unexpected_path = os.path.join(per_dataset_dir, "unexpected.json")
                with open(unexpected_path, "w") as handle:
                    json.dump(to_golden_json(unexpected), handle, indent=2, sort_keys=True)
                lines.append(f"  unexpected lumis (processed outside golden) written to: {unexpected_path}")
        else:
            lines.append(f"[WARN] --golden-json {golden_json_path} not found, skipping comparison")
    else:
        lines.append(f"Feed processedLumis.json to brilcalc for the processed luminosity, e.g.:\n"
                     f"  brilcalc lumi -i {combined_path} -u /pb --normtag <normtag.json>")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage1_dirs", nargs="+", help="one or more stage1_output/<year> directories to walk")
    ap.add_argument("--golden-json", help="certified lumimask to compare against "
                                          "(see configs/parameters/lumi.yaml for the path per year)")
    ap.add_argument("--out", help="write outputs here instead of '<stage1_dir>/_status/' "
                                  "(shared across all stage1_dirs if multiple are given)")
    args = ap.parse_args()

    for stage1_dir in args.stage1_dirs:
        print(f"\n=== {stage1_dir} ===")
        print(build_report(stage1_dir, golden_json_path=args.golden_json, out_dir=args.out))


if __name__ == "__main__":
    main()
