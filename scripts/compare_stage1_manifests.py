#!/usr/bin/env python3
"""
Compare two stage1 manifest CSV files produced by create_basic_info_stage1_files.py

Match key: (year, f1_tag, sample, rel_path)
Compare: adler32 only

Outputs (in --outdir):
  - compare_summary.txt
  - mismatched_adler32.csv
  - missing_in_A.csv   (present in B, absent in A)
  - missing_in_B.csv   (present in A, absent in B)

Usage:
  python scripts/compare_stage1_manifests.py \
    -a stage1_basic_info_LABEL_DEPOT.csv \
    -b stage1_basic_info_LABEL_EOS.csv \
    -o compare_out

Notes:
- If adler32 is empty or missing in either file, it's treated as "missing checksum".
"""

import os
import csv
import argparse
from typing import Dict, Tuple, Optional


Key = Tuple[str, str, str, str]  # (year, f1_tag, sample, rel_path)


def get_field(row: dict, name: str) -> str:
    v = row.get(name, "")
    return (v if v is not None else "").strip()


def make_key(row: dict) -> Optional[Key]:
    year = get_field(row, "year")
    f1_tag = get_field(row, "f1_tag")
    sample = get_field(row, "sample")
    rel_path = get_field(row, "rel_path")
    if not (year and f1_tag and sample and rel_path):
        return None
    return (year, f1_tag, sample, rel_path)


def load_manifest(csv_path: str) -> Dict[Key, dict]:
    """
    Load CSV into dict keyed by (year,f1_tag,sample,rel_path).
    Keeps the last occurrence if duplicates exist.
    """
    data: Dict[Key, dict] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = make_key(row)
            if k is None:
                continue
            data[k] = row
    return data


def write_csv(path: str, rows: list, fieldnames: list):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main(a_csv: str, b_csv: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    A = load_manifest(a_csv)
    B = load_manifest(b_csv)

    keysA = set(A.keys())
    keysB = set(B.keys())

    common = keysA & keysB
    onlyA = keysA - keysB
    onlyB = keysB - keysA

    mismatches = []
    missing_checksum = []

    for k in common:
        a_adler = get_field(A[k], "adler32")
        b_adler = get_field(B[k], "adler32")

        if (not a_adler) or (not b_adler):
            missing_checksum.append({
                "year": k[0],
                "f1_tag": k[1],
                "sample": k[2],
                "rel_path": k[3],
                "adler32_A": a_adler,
                "adler32_B": b_adler,
                "full_path_A": get_field(A[k], "full_path"),
                "full_path_B": get_field(B[k], "full_path"),
            })
            continue

        if a_adler != b_adler:
            mismatches.append({
                "year": k[0],
                "f1_tag": k[1],
                "sample": k[2],
                "rel_path": k[3],
                "adler32_A": a_adler,
                "adler32_B": b_adler,
                "full_path_A": get_field(A[k], "full_path"),
                "full_path_B": get_field(B[k], "full_path"),
            })

    # Missing rows outputs
    missing_in_A_rows = []
    for k in sorted(onlyB):
        missing_in_A_rows.append({
            "year": k[0],
            "f1_tag": k[1],
            "sample": k[2],
            "rel_path": k[3],
            "adler32_B": get_field(B[k], "adler32"),
            "full_path_B": get_field(B[k], "full_path"),
        })

    missing_in_B_rows = []
    for k in sorted(onlyA):
        missing_in_B_rows.append({
            "year": k[0],
            "f1_tag": k[1],
            "sample": k[2],
            "rel_path": k[3],
            "adler32_A": get_field(A[k], "adler32"),
            "full_path_A": get_field(A[k], "full_path"),
        })

    # Write outputs
    write_csv(
        os.path.join(outdir, "mismatched_adler32.csv"),
        mismatches,
        ["year", "f1_tag", "sample", "rel_path", "adler32_A", "adler32_B", "full_path_A", "full_path_B"],
    )
    write_csv(
        os.path.join(outdir, "missing_checksum.csv"),
        missing_checksum,
        ["year", "f1_tag", "sample", "rel_path", "adler32_A", "adler32_B", "full_path_A", "full_path_B"],
    )
    write_csv(
        os.path.join(outdir, "missing_in_A.csv"),
        missing_in_A_rows,
        ["year", "f1_tag", "sample", "rel_path", "adler32_B", "full_path_B"],
    )
    write_csv(
        os.path.join(outdir, "missing_in_B.csv"),
        missing_in_B_rows,
        ["year", "f1_tag", "sample", "rel_path", "adler32_A", "full_path_A"],
    )

    # Summary
    summary_path = os.path.join(outdir, "compare_summary.txt")
    with open(summary_path, "w") as s:
        s.write(f"A: {a_csv}\n")
        s.write(f"B: {b_csv}\n\n")
        s.write(f"Rows in A (unique keys): {len(A)}\n")
        s.write(f"Rows in B (unique keys): {len(B)}\n")
        s.write(f"Common keys            : {len(common)}\n")
        s.write(f"Only in A              : {len(onlyA)}\n")
        s.write(f"Only in B              : {len(onlyB)}\n")
        s.write(f"Adler32 mismatches     : {len(mismatches)}\n")
        s.write(f"Missing checksum (any) : {len(missing_checksum)}\n")

    print("Done.")
    print(f"Summary: {summary_path}")
    print(f"Mismatches: {os.path.join(outdir, 'mismatched_adler32.csv')}")
    print(f"Missing checksum: {os.path.join(outdir, 'missing_checksum.csv')}")
    print(f"Missing in A: {os.path.join(outdir, 'missing_in_A.csv')}")
    print(f"Missing in B: {os.path.join(outdir, 'missing_in_B.csv')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare two stage1 manifest CSV files (key match; compare adler32).")
    ap.add_argument("-a", "--csv-a", required=True, help="Manifest CSV A")
    ap.add_argument("-b", "--csv-b", required=True, help="Manifest CSV B")
    ap.add_argument("-o", "--outdir", default="compare_out", help="Output directory")
    args = ap.parse_args()

    main(args.csv_a, args.csv_b, args.outdir)
