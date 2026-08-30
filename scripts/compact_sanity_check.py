#!/usr/bin/env python3

"""Validate compacted stage-1 outputs against the corresponding f1_0 outputs.

This scripts checks the number of entries in the output parquet file in `f1_0` and in `compacted`
directory. if they are same that means compact is success else not.

Run it with:

    python scripts/compact_sanity_check.py --base /path/to/stage1_output

"""

import argparse
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from modules.utils import logger


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
        logger.error(f"[{sample}] missing compacted output: {cmp_dir}")
        return False

    size_f1 = dir_size_bytes(f1_dir)
    size_cmp = dir_size_bytes(cmp_dir)
    ratio = (size_cmp / size_f1) if size_f1 else 0.0

    logger.info(f"[{sample}] size: f1_0={size_f1:,}B compacted={size_cmp:,}B ratio={ratio:.3f}")

    cmp_parquets = parquet_files(cmp_dir)
    if not cmp_parquets:
        logger.warning(f"[{sample}] no parquet files found under {cmp_dir} -- deleting")
        shutil.rmtree(cmp_dir)
        return False

    if ratio >= 0.9:
        logger.debug(f"[{sample}] OK (size ratio {ratio:.3f} >= 0.9)")
        return True

    logger.warning(
        f"[{sample}] size ratio {ratio:.3f} < 0.9 -- comparing parquet row counts before deciding"
    )
    rows_f1 = count_parquet_rows(f1_dir)
    rows_cmp = count_parquet_rows(cmp_dir)
    logger.info(f"[{sample}] rows: f1_0={rows_f1:,} compacted={rows_cmp:,}")

    if rows_f1 == rows_cmp:
        logger.info(
            f"[{sample}] row counts match despite size mismatch -- keeping compacted output"
        )
        return True

    logger.error(
        f"[{sample}] row count mismatch (f1_0={rows_f1:,} vs compacted={rows_cmp:,}) "
        f"-- deleting {cmp_dir}"
    )
    shutil.rmtree(cmp_dir)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate compacted stage1 outputs against f1_0.")
    parser.add_argument("--base", required=True, help="Path containing f1_0 and compacted directories.")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO). Use DEBUG to also see per-sample OK lines.",
    )
    args = parser.parse_args()

    logger.setLevel(args.log_level)

    base = Path(args.base)
    dir_f1 = base / "f1_0"
    dir_cmp = base / "compacted"

    if not dir_f1.is_dir():
        logger.error(f"Missing f1_0 directory: {dir_f1}")
        return 2
    if not dir_cmp.is_dir():
        logger.error(f"Missing compacted directory: {dir_cmp}")
        return 2

    samples = sorted(path for path in dir_f1.iterdir() if path.is_dir())
    logger.info(f"Checking compact consistency for {len(samples)} sample(s) under {base}")

    failed_samples = []
    for idx, f1_dir in enumerate(samples, start=1):
        sample = f1_dir.name
        logger.debug(f"[{idx}/{len(samples)}] {sample}")
        if not validate_sample(sample, f1_dir, dir_cmp / sample):
            failed_samples.append(sample)

    logger.info(
        f"Summary: {len(samples)} checked, {len(samples) - len(failed_samples)} passed, "
        f"{len(failed_samples)} failed"
    )

    if failed_samples:
        logger.error(f"Compact validation FAILED for: {', '.join(failed_samples)}")
        return 1

    logger.info("Compact validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
