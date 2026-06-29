#!/usr/bin/env python3

"""Minimal validation of compacted stage-1 Parquet against f1_0.

Example:
    python scripts/validate_compacted_parquet_mwe.py
"""

import argparse
import hashlib
import math
import numbers
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_BASE = Path(
    "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/"
    "Run2_NanoV15_forVBFChannel_June24_2026_jetUnc/stage1_output/2018"
)
DEFAULT_COLUMNS = [
    "run",
    "luminosityBlock",
    "event",
    "mu1_pt",
    "mu2_pt",
    "dimuon_mass",
    "dimuon_pt",
]
STATS_KEYS = ["mean", "median", "min", "max"]


def part_number(path: Path) -> int | None:
    match = re.fullmatch(r"part(\d+)\.parquet", path.name)
    return int(match.group(1)) if match else None


def parquet_files(path: Path, *, numeric_part_order: bool = False) -> list[Path]:
    files = list(path.rglob("*.parquet"))
    if numeric_part_order:
        return sorted(
            files,
            key=lambda file_path: (
                part_number(file_path) is None,
                part_number(file_path) if part_number(file_path) is not None else file_path.name,
            ),
        )
    return sorted(files)


def parquet_row_count(files: list[Path]) -> int:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in files)


def parquet_schema_names(files: list[Path]) -> list[str]:
    if not files:
        return []
    return pq.ParquetFile(files[0]).schema_arrow.names


def hash_batch(batch: pa.RecordBatch, digest) -> None:
    columns = batch.schema.names
    py_columns = [batch.column(column_name).to_pylist() for column_name in columns]
    for row_idx in range(batch.num_rows):
        for column in py_columns:
            digest.update(repr(column[row_idx]).encode())
            digest.update(b"\x1f")
        digest.update(b"\x1e")


def parquet_content_hash(files: list[Path], columns: list[str], batch_size: int) -> str:
    digest = hashlib.sha256()
    schema = pq.ParquetFile(files[0]).schema_arrow
    for column_name in columns:
        digest.update(column_name.encode())
        digest.update(str(schema.field(column_name).type).encode())

    for parquet_path in files:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
            hash_batch(batch, digest)
    return digest.hexdigest()


def is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def is_scalar_number(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def median(values: list[numbers.Real]) -> float:
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[midpoint])
    return float((sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0)


def summarize_numeric_columns(
    files: list[Path], columns: list[str], batch_size: int
) -> tuple[dict[str, dict[str, float]], set[str]]:
    values_by_column = {column: [] for column in columns}
    non_numeric_columns = set()

    for parquet_path in files:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
            for column_name in batch.schema.names:
                if column_name in non_numeric_columns:
                    continue
                values = batch.column(column_name).to_pylist()
                for value in values:
                    if is_missing(value):
                        continue
                    if not is_scalar_number(value):
                        non_numeric_columns.add(column_name)
                        values_by_column[column_name] = []
                        break
                    values_by_column[column_name].append(value)

    summaries = {}
    for column_name, values in values_by_column.items():
        if column_name in non_numeric_columns or not values:
            continue
        summaries[column_name] = {
            "mean": float(math.fsum(values) / len(values)),
            "median": median(values),
            "min": float(min(values)),
            "max": float(max(values)),
        }
    return summaries, non_numeric_columns


def assert_stats_match(
    f1_stats: dict[str, dict[str, float]],
    compacted_stats: dict[str, dict[str, float]],
    rtol: float,
    atol: float,
) -> None:
    if set(f1_stats) != set(compacted_stats):
        raise AssertionError(
            "Numeric stats columns do not match: "
            f"only_f1={sorted(set(f1_stats) - set(compacted_stats))}, "
            f"only_compacted={sorted(set(compacted_stats) - set(f1_stats))}"
        )

    mismatches = []
    for column_name in sorted(f1_stats):
        for stat_name in STATS_KEYS:
            f1_value = f1_stats[column_name][stat_name]
            compacted_value = compacted_stats[column_name][stat_name]
            if not math.isclose(f1_value, compacted_value, rel_tol=rtol, abs_tol=atol):
                mismatches.append(
                    f"{column_name}.{stat_name}: f1_0={f1_value} "
                    f"compacted={compacted_value}"
                )

    if mismatches:
        raise AssertionError(
            "Numeric summary stats do not match:\n" + "\n".join(mismatches)
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate one compacted stage-1 Parquet sample against f1_0."
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--sample", default="data_A")
    parser.add_argument("--chunk", default="0")
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--columns", nargs="+", default=DEFAULT_COLUMNS)
    parser.add_argument(
        "--check-stats",
        action="store_true",
        help="Also compare mean, median, min, and max for the columns passed to --columns.",
    )
    parser.add_argument("--stats-rtol", type=float, default=1e-12)
    parser.add_argument("--stats-atol", type=float, default=1e-9)
    args = parser.parse_args()

    f1_dir = args.base / "f1_0" / args.sample / args.chunk
    compacted_dir = args.base / "compacted" / args.sample / args.chunk

    f1_files = parquet_files(f1_dir)
    compacted_files = parquet_files(compacted_dir, numeric_part_order=True)
    if not f1_files:
        raise FileNotFoundError(f"No f1_0 parquet files found under {f1_dir}")
    if not compacted_files:
        raise FileNotFoundError(f"No compacted parquet files found under {compacted_dir}")

    f1_schema = parquet_schema_names(f1_files)
    compacted_schema = parquet_schema_names(compacted_files)
    missing_columns = [col for col in args.columns if col not in f1_schema]
    if missing_columns:
        raise ValueError(f"Requested columns missing in f1_0 schema: {missing_columns}")
    missing_columns = [col for col in args.columns if col not in compacted_schema]
    if missing_columns:
        raise ValueError(f"Requested columns missing in compacted schema: {missing_columns}")

    print(f"f1_0 path:      {f1_dir}")
    print(f"compacted path: {compacted_dir}")
    print(f"f1_0 files:      {len(f1_files)}")
    print(f"compacted files: {len(compacted_files)}")

    f1_rows = parquet_row_count(f1_files)
    compacted_rows = parquet_row_count(compacted_files)
    print(f"f1_0 rows:      {f1_rows}")
    print(f"compacted rows: {compacted_rows}")
    if f1_rows != compacted_rows:
        raise AssertionError("Row counts do not match")

    if set(f1_schema) != set(compacted_schema):
        only_f1 = sorted(set(f1_schema) - set(compacted_schema))
        only_compacted = sorted(set(compacted_schema) - set(f1_schema))
        raise AssertionError(
            "Schema fields do not match: "
            f"only_f1={only_f1}, only_compacted={only_compacted}"
        )
    print(f"schema fields:  {len(f1_schema)} fields match")

    print(f"hash columns:   {', '.join(args.columns)}")
    f1_hash = parquet_content_hash(f1_files, args.columns, args.batch_size)
    compacted_hash = parquet_content_hash(compacted_files, args.columns, args.batch_size)
    print(f"f1_0 hash:      {f1_hash}")
    print(f"compacted hash: {compacted_hash}")
    if f1_hash != compacted_hash:
        raise AssertionError("Selected-column content hash does not match")

    if args.check_stats:
        stats_columns = args.columns

        missing_columns = [col for col in stats_columns if col not in f1_schema]
        if missing_columns:
            raise ValueError(f"Stats columns missing in f1_0 schema: {missing_columns}")
        missing_columns = [col for col in stats_columns if col not in compacted_schema]
        if missing_columns:
            raise ValueError(
                f"Stats columns missing in compacted schema: {missing_columns}"
            )

        print(f"stats columns:  {', '.join(stats_columns)}")
        f1_stats, f1_non_numeric = summarize_numeric_columns(
            f1_files, stats_columns, args.batch_size
        )
        compacted_stats, compacted_non_numeric = summarize_numeric_columns(
            compacted_files, stats_columns, args.batch_size
        )
        skipped_columns = sorted(f1_non_numeric | compacted_non_numeric)
        if skipped_columns:
            print(f"stats skipped non-numeric columns: {', '.join(skipped_columns)}")
        assert_stats_match(
            f1_stats,
            compacted_stats,
            rtol=args.stats_rtol,
            atol=args.stats_atol,
        )
        for column_name in sorted(f1_stats):
            stats = f1_stats[column_name]
            print(
                f"stats {column_name}: "
                f"mean={stats['mean']} median={stats['median']} "
                f"min={stats['min']} max={stats['max']}"
            )

    print("Validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
