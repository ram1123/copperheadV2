#!/usr/bin/env python3
"""
Verify data completeness at the lumisection level: compare the exact
(run, luminosityBlock) coverage of the files stage1 actually processed
against the officially certified golden JSON.

This needs no brilcalc/network access -- both inputs (the golden JSON, and
the run/luminosityBlock branches read directly from our own processed
files) are read locally. If every certified lumisection is present in at
least one processed file, the dataset is lumi-complete: nothing that should
have been included per the golden JSON was silently missed (e.g. a file
missing from DAS/rucio at prestage time, like the stale-cache issue found
for vbf_aMCatNLO/vbf_powheg this session).

Usage:
    python scripts/verify_lumi_coverage.py \\
        --prestage-json prestage_output/processor_samples_2025_NanoAODv15.json \\
        --golden-json data/lumimasks/Cert_Collisions2025_391658_398903_Golden.json \\
        [--sample data_B data_C ...]   # default: every data_* sample in the prestage JSON
"""
import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import uproot


def read_run_lumis(fname):
    with uproot.open(f"{fname}:Events", timeout=60) as tree:
        runs = tree["run"].array(library="np")
        lumis = tree["luminosityBlock"].array(library="np")
    return set(zip(runs.tolist(), lumis.tolist()))


def compress_ls_to_ranges(sorted_ls):
    """Compress a sorted list of int lumisections into [[start, end], ...]
    runs of consecutive integers -- the CMS golden-JSON lumimask format."""
    if not sorted_ls:
        return []
    ranges = []
    start = prev = sorted_ls[0]
    for ls in sorted_ls[1:]:
        if ls == prev + 1:
            prev = ls
        else:
            ranges.append([start, prev])
            start = prev = ls
    ranges.append([start, prev])
    return ranges


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--prestage-json", required=True)
    parser.add_argument("--golden-json", required=True)
    parser.add_argument(
        "--sample", nargs="*", default=None,
        help="Specific data_* sample name(s) to check (default: all data_* samples in the prestage JSON)",
    )
    parser.add_argument("--max-workers", type=int, default=30)
    parser.add_argument(
        "--cache", default=None,
        help="Path to save/load the scanned run->lumisection coverage as JSON, "
             "so re-analyzing (e.g. against a different golden JSON, or the "
             "diagnostics below) doesn't require re-scanning every file.",
    )
    parser.add_argument(
        "--dump-missing", default=None,
        help="Write the full missing (run, ls) list to this path instead of "
             "only printing the first 20 runs.",
    )
    parser.add_argument(
        "--dump-processed-json", default=None,
        help="Write the processed files' own (run, luminosityBlock) coverage as a "
             "standard CMS lumimask JSON ({run: [[startLS,endLS],...]}), independent "
             "of the golden-JSON comparison above. Feed it straight to brilcalc (e.g. "
             "via scripts/compute_lumi_from_golden_json.sh <this file>) to get the "
             "actual luminosity of what stage1 processed, as a second, brilcalc-based "
             "cross-check alongside the lumisection-coverage check this script already does.",
    )
    args = parser.parse_args()

    with open(args.golden_json) as f:
        golden = json.load(f)

    if args.cache and os.path.exists(args.cache):
        print(f"Loading cached run/lumisection coverage from {args.cache}")
        with open(args.cache) as f:
            cached = json.load(f)
        processed_run_ls = {int(run): set(lss) for run, lss in cached.items()}
        fails = []
    else:
        with open(args.prestage_json) as f:
            prestage = json.load(f)

        if args.sample:
            samples = args.sample
        else:
            samples = sorted(k for k, v in prestage.items() if v.get("metadata", {}).get("is_mc") is False)

        if not samples:
            raise SystemExit("[ERROR] No data_* samples found (or specified) to check")

        print(f"Checking lumi coverage for samples: {samples}")

        all_files = []
        for s in samples:
            if s not in prestage:
                raise SystemExit(f"[ERROR] Sample {s} not found in prestage JSON")
            all_files.extend(prestage[s]["files"].keys())
        print(f"Total files to scan: {len(all_files)}")

        processed_run_ls = defaultdict(set)
        fails = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futs = {ex.submit(read_run_lumis, f): f for f in all_files}
            done = 0
            for fut in as_completed(futs):
                done += 1
                try:
                    for run, ls in fut.result():
                        processed_run_ls[run].add(ls)
                except Exception as e:
                    fails.append((futs[fut], str(e)))
                if done % 50 == 0 or done == len(all_files):
                    print(f"  scanned {done}/{len(all_files)} files...")

        if fails:
            print(f"\n[WARNING] {len(fails)} file(s) failed to read (excluded from coverage):")
            for f, e in fails[:20]:
                print(f"  {f}: {e}")

        if args.cache:
            with open(args.cache, "w") as f:
                json.dump({str(run): sorted(lss) for run, lss in processed_run_ls.items()}, f)
            print(f"Cached run/lumisection coverage to {args.cache}")

    if processed_run_ls:
        print(
            f"\nProcessed files cover runs {min(processed_run_ls)}-{max(processed_run_ls)} "
            f"({len(processed_run_ls)} distinct runs)."
        )

    if args.dump_processed_json:
        processed_json = {
            str(run): compress_ls_to_ranges(sorted(lss))
            for run, lss in sorted(processed_run_ls.items())
        }
        with open(args.dump_processed_json, "w") as f:
            json.dump(processed_json, f)
        print(
            f"\nWrote processed-files lumi JSON ({len(processed_json)} runs) to "
            f"{args.dump_processed_json}"
        )
        print(
            f"  -> compute its luminosity via: "
            f"bash scripts/compute_lumi_from_golden_json.sh {args.dump_processed_json}"
        )

    # Compare against golden JSON
    total_certified_ls = 0
    total_covered_ls = 0
    missing = []  # (run, ls) pairs certified but not covered
    for run_str, ranges in golden.items():
        run = int(run_str)
        covered_for_run = processed_run_ls.get(run, set())
        for start, end in ranges:
            for ls in range(start, end + 1):
                total_certified_ls += 1
                if ls in covered_for_run:
                    total_covered_ls += 1
                else:
                    missing.append((run, ls))

    print()
    print(f"Certified lumisections (golden JSON): {total_certified_ls}")
    print(f"Covered by processed files:           {total_covered_ls}")
    print(f"Missing:                               {len(missing)}")

    if missing:
        missing_by_run = defaultdict(list)
        for run, ls in missing:
            missing_by_run[run].append(ls)

        max_processed_run = max(processed_run_ls) if processed_run_ls else None
        min_processed_run = min(processed_run_ls) if processed_run_ls else None
        beyond_range = sum(1 for run in missing_by_run if max_processed_run is not None and run > max_processed_run)
        before_range = sum(1 for run in missing_by_run if min_processed_run is not None and run < min_processed_run)
        within_range = len(missing_by_run) - beyond_range - before_range
        print(
            f"\nOf {len(missing_by_run)} run(s) with missing lumisections: "
            f"{beyond_range} are entirely beyond the highest processed run ({max_processed_run}), "
            f"{before_range} are entirely before the lowest processed run ({min_processed_run}), "
            f"{within_range} fall within the processed run range (these are the concerning ones)."
        )

        if args.dump_missing:
            with open(args.dump_missing, "w") as f:
                json.dump(sorted(missing), f)
            print(f"Full missing (run, ls) list written to {args.dump_missing}")

        print("\nMissing lumisections by run (showing up to 20 runs):")
        for run in sorted(missing_by_run)[:20]:
            lss = sorted(missing_by_run[run])
            print(f"  run {run}: {len(lss)} LS missing, e.g. {lss[:10]}{'...' if len(lss) > 10 else ''}")
        raise SystemExit(1)

    print("\nOK: every certified lumisection in the golden JSON is covered by the processed files.")


if __name__ == "__main__":
    main()
