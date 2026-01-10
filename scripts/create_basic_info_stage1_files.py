#!/usr/bin/env python3
"""
Create a CSV manifest for all stage1 parquet files.

Assumed path pattern (recommended, but parsing is robust):
/depot/cms/hmm/.../{label}/stage1_output/{year}/f1_0/{sample}/file.parquet

What it writes (CSV):
year, f1_tag, sample, rel_path, file_name, size_bytes, adler32, mtime_unix, ctime_unix, mtime_iso, host

Notes:
- On Linux, ctime is *metadata change time* (not true creation time).
- mtime is usually the more useful “file last modified” time.
"""

import os
import sys
import csv
import time
import socket
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator, Optional, Dict, Tuple

from tqdm import tqdm


def iter_parquet_files(base_path: str) -> Iterator[str]:
    """Yield all .parquet files under base_path."""
    for dirpath, _, filenames in os.walk(base_path):
        for fn in filenames:
            if fn.lower().endswith(".parquet"):
                yield os.path.join(dirpath, fn)


def safe_iso(ts: float) -> str:
    """Convert unix timestamp to ISO string (local time)."""
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    except Exception:
        return ""


def compute_adler32(file_path: str, timeout: int = 120) -> Optional[str]:
    """Compute xrdadler32 checksum. Returns None on failure."""
    try:
        res = subprocess.run(
            ["xrdadler32", file_path],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip() or res.stdout.strip())
        # Typical output: "<adler32>  <filename>"
        return res.stdout.strip().split()[0]
    except Exception as e:
        sys.stderr.write(f"[checksum-fail] {file_path}: {e}\n")
        return None


def parse_stage1_components(
    file_path: str, base_path: str
) -> Optional[Tuple[str, str, str, str]]:
    """
    Returns (year, f1_tag, sample, rel_path) or None if cannot parse.
    Parsing strategy:
      - rel_path = file relative to base_path
      - find first path part that matches f1_*
      - year is the part immediately before f1_*
      - sample is the part immediately after f1_*
    """
    try:
        rel = Path(file_path).relative_to(base_path)
    except Exception as e:
        sys.stderr.write(f"[parse-fail] not under base_path: {file_path} ({e})\n")
        return None

    parts = rel.parts
    f1_idx = None
    for i, p in enumerate(parts):
        if p.startswith("f1_") or p.startswith("compacted"):
            f1_idx = i
            break

    if f1_idx is None or f1_idx - 1 < 0 or f1_idx + 1 >= len(parts):
        sys.stderr.write(f"[parse-fail] cannot find year/f1/sample in: {rel}\n")
        return None

    year = parts[f1_idx - 1]
    f1_tag = parts[f1_idx]
    sample = parts[f1_idx + 1]
    return year, f1_tag, sample, str(rel)


def get_basic_stat(file_path: str) -> Optional[os.stat_result]:
    try:
        return os.stat(file_path)
    except Exception as e:
        sys.stderr.write(f"[stat-fail] {file_path}: {e}\n")
        return None


def process_one(
    file_path: str,
    base_path: str,
    do_checksum: bool,
    checksum_timeout: int,
    host: str,
) -> Optional[Dict[str, object]]:
    parsed = parse_stage1_components(file_path, base_path)
    if not parsed:
        return None
    year, f1_tag, sample, rel_path = parsed

    st = get_basic_stat(file_path)
    if st is None:
        return None

    adler = compute_adler32(file_path, timeout=checksum_timeout) if do_checksum else ""

    row = {
        "year": year,
        "f1_tag": f1_tag,
        "sample": sample,
        "rel_path": rel_path,
        "file_name": os.path.basename(file_path),
        "size_bytes": st.st_size,
        "adler32": adler,
        "mtime_unix": int(st.st_mtime),
        "ctime_unix": int(st.st_ctime),
        "mtime_iso": safe_iso(st.st_mtime),
        "host": host,
        "full_path": file_path,  # keep for convenience
    }
    return row


def main(
    base_path: str,
    output_dir: str,
    label: str,
    workers: int,
    no_checksum: bool,
    checksum_timeout: int,
):
    base_path = os.path.abspath(base_path)
    os.makedirs(output_dir, exist_ok=True)

    host = socket.gethostname()

    # Output filename: prefer user-provided label; else use parent of stage1_output if possible
    if not label:
        p = Path(base_path)
        # if base_path ends with ".../stage1_output", parent is label directory
        label = p.parent.name if p.name == "stage1_output" else p.name

    print(f"Input base path: {base_path}")
    print(f"Output directory: {output_dir}")
    print(f"Label: {label}")

    out_csv = os.path.join(output_dir, f"stage1_basic_info_{label}.csv")
    failed_txt = os.path.join(output_dir, f"stage1_basic_info_{label}.failed.txt")
    print(f"Output CSV: {out_csv}")
    print(f"Failed list: {failed_txt}")

    fields = [
        "year",
        "f1_tag",
        "sample",
        "rel_path",
        "file_name",
        "size_bytes",
        "adler32",
        "mtime_unix",
        "ctime_unix",
        "mtime_iso",
        "host",
        "full_path",
    ]

    files = list(iter_parquet_files(base_path))
    print(f"Found {len(files)} parquet files under: {base_path}")
    if not files:
        print(f"No parquet files found under: {base_path}")
        return

    do_checksum = not no_checksum

    # Stream rows to CSV as futures complete (no big in-memory list)
    n_ok = 0
    n_fail = 0
    n_ck_fail = 0

    print(f"Starting processing with {workers} workers...")
    with open(out_csv, "w", newline="") as f_csv, open(failed_txt, "w") as f_fail:
        writer = csv.DictWriter(f_csv, fieldnames=fields)
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(
                    process_one, fp, base_path, do_checksum, checksum_timeout, host
                )
                for fp in files
            ]

            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing files",
                unit="file",
            )
            for fut in pbar:
                try:
                    row = fut.result()
                except Exception as e:
                    # Should be rare, but catch hard failures
                    n_fail += 1
                    f_fail.write(f"EXCEPTION\t{e}\n")
                    continue

                if row is None:
                    n_fail += 1
                    # We don't necessarily know which file caused None at this point.
                    # Write a marker; detailed errors were already printed to stderr.
                    f_fail.write("PARSE_OR_STAT_FAIL\t(see stderr logs)\n")
                    continue

                # If checksum requested but failed, keep row (adler32=None) and record failure
                if do_checksum and (row["adler32"] is None):
                    f_fail.write(f"CHECKSUM_FAIL\t{row['full_path']}\n")
                    n_ck_fail += 1
                writer.writerow(row)
                n_ok += 1

                if (n_ok + n_fail) % 200 == 0:
                    pbar.set_postfix({
                        "ok": n_ok,
                        "fail": n_fail,
                        "ck_fail": n_ck_fail
                    })

    print(f"Saved {n_ok} rows to: {out_csv}")
    if n_fail > 0:
        print(f"Some failures were recorded in: {failed_txt}  (count={n_fail})")
    if n_ck_fail > 0:
        print(f"Some checksum failures were recorded in: {failed_txt}  (count={n_ck_fail})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create CSV manifest for stage1 parquet files."
    )
    parser.add_argument(
        "-p",
        "--path",
        default="/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output",
        help="Stage1 output base path (ends with .../stage1_output)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=".",
        help="Output directory for CSV + failed list",
    )
    parser.add_argument(
        "-l",
        "--label",
        default="",
        help="Label used in output filename (optional). If empty, inferred from path.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=32,
        help="Number of parallel workers (suggest 16-32 on login nodes).",
    )
    parser.add_argument(
        "--no-checksum",
        action="store_true",
        help="Skip xrdadler32 computation (much faster).",
    )
    parser.add_argument(
        "--checksum-timeout",
        type=int,
        default=211,
        help="Timeout (seconds) per xrdadler32 call.",
    )

    args = parser.parse_args()
    main(
        base_path=args.path,
        output_dir=args.output,
        label=args.label,
        workers=args.workers,
        no_checksum=args.no_checksum,
        checksum_timeout=args.checksum_timeout,
    )

# python scripts/create_basic_info_stage1_files.py -p /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar_DEPOT
# python scripts/create_basic_info_stage1_files.py -p /eos/purdue/store/user/rasharma/hmm/reducedNtuples/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar_EOS
