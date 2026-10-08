#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd


def load_cutflow(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def compare_cutflows(reference: dict, candidate: dict) -> pd.DataFrame:
    all_keys = sorted(set(reference) | set(candidate))
    rows = []

    for key in all_keys:
        ref_entry = reference.get(key)
        cand_entry = candidate.get(key)

        if ref_entry == cand_entry:
            continue

        rows.append(
            {
                "cut": key,
                "reference_cumulative": None if ref_entry is None else ref_entry.get("cumulative"),
                "candidate_cumulative": None if cand_entry is None else cand_entry.get("cumulative"),
                "reference_individual": None if ref_entry is None else ref_entry.get("individual"),
                "candidate_individual": None if cand_entry is None else cand_entry.get("individual"),
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Compare two cutflow JSON files.")
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("-o", "--out", required=True, help="Output CSV path for differences.")
    args = parser.parse_args()

    reference = load_cutflow(args.reference)
    candidate = load_cutflow(args.candidate)
    diff_df = compare_cutflows(reference, candidate)

    out_path = Path(args.out)
    if diff_df.empty:
        out_path.touch()
    else:
        diff_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
