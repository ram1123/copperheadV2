#!/usr/bin/env python3

"""Validate compacted stage-1 outputs against the corresponding f1_0 outputs.

This scripts checks the number of entries in the output parquet file in `f1_0` and in `compacted` 
directory. if they are same that means compact is success else not.

Run it with:
    
    python scripts/compact_sanity_check.py --base /path/to/stage1_output

"""

import argparse
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq


def dir_size_bytes(path: Path) -> int:
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())


def parquet_files(path: Path) -> list[Path]:
    return sorted(path.rglob("*.parquet"))


def count_parquet_rows(path: Path) -> int:
    total_rows = 0
    for parquet_path in parquet_files(path):
        total_rows += pq.ParquetFile(parquet_path).metadata.num_rows
    return total_rows


def validate_sample(sample: str, f1_dir: Path, cmp_dir: Path) -> bool:
    if not cmp_dir.is_dir():
        print(f"Missing compacted sample: {sample}")
        return False

    size_f1 = dir_size_bytes(f1_dir)
    size_cmp = dir_size_bytes(cmp_dir)
    ratio = (size_cmp / size_f1) if size_f1 else 0.0

    print(f"{sample} : f1={size_f1} cmp={size_cmp} ratio={ratio}")

    cmp_parquets = parquet_files(cmp_dir)
    if not cmp_parquets:
        print(f"No parquet found in {sample}")
        shutil.rmtree(cmp_dir)
        return False

    if ratio >= 0.9:
        return True

    print(f"Size mismatch for {sample} -> comparing parquet entries before deleting")
    rows_f1 = count_parquet_rows(f1_dir)
    rows_cmp = count_parquet_rows(cmp_dir)
    print(f"{sample} : rows_f1={rows_f1} rows_cmp={rows_cmp}")

    if rows_f1 == rows_cmp:
        print(f"Entry counts match for {sample} despite size mismatch; keeping compacted output")
        return True

    print(f"Entry mismatch for {sample} -> deleting compacted")
    shutil.rmtree(cmp_dir)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate compacted stage1 outputs against f1_0.")
    parser.add_argument("--base", required=True, help="Path containing f1_0 and compacted directories.")
    args = parser.parse_args()

    base = Path(args.base)
    dir_f1 = base / "f1_0"
    dir_cmp = base / "compacted"

    if not dir_f1.is_dir():
        print(f"Missing f1_0 directory: {dir_f1}", file=sys.stderr)
        return 2
    if not dir_cmp.is_dir():
        print(f"Missing compacted directory: {dir_cmp}", file=sys.stderr)
        return 2

    print("Checking compact consistency...")

    failed = False
    for f1_dir in sorted(path for path in dir_f1.iterdir() if path.is_dir()):
        if not validate_sample(f1_dir.name, f1_dir, dir_cmp / f1_dir.name):
            failed = True

    if failed:
        print("ERROR: Compact validation failed")
        return 1

    print("Compact validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
