#!/usr/bin/env python3

"""
Sync or compare dimuon variables between directories of parquet files.
Usage:
    python sync_parquet_dimuon.py <dir1> [<dir2>] [--out OUTPUT] [--tolerance TOLERANCE]

Example:
    time python ./sync_parquet_dimuon.py  /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2017/f1_0/data_D/0 /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_28Nov_HEMVetoFix_NoSyst_V2/stage1_output/2017/f1_0/data_D/0
"""

import argparse
import glob
from pathlib import Path
from typing import List, Optional

import dask_awkward as dak
import awkward as ak
import pandas as pd

from distributed import Client

from modules import selection

# ----------------------------------------------------------------------
# Columns
# ----------------------------------------------------------------------
KEY_VARS = ["run", "luminosityBlock", "event"]

V1_FIELDS_2COMPUTE: List[str] = [
    # "gjj_mass",  # handled dynamically if needed
    "separate_wgt_genWeight",
    "separate_wgt_genWeight_normalization",
    "separate_wgt_xsec",
    "separate_wgt_lumi",
    "separate_wgt_pu_wgt",
    "separate_wgt_l1prefiring",
    "separate_wgt_muID",
    "separate_wgt_muIso",
    "separate_wgt_muTrig",
    "separate_wgt_LHERen",
    "separate_wgt_LHEFac",
    "separate_wgt_pdf_2rms",
    "separate_wgt_jetpuid_wgt",
    "separate_wgt_qgl_wgt",
    "separate_wgt_zpt_wgt",
    "separate_wgt_ones",
    "wgt_nominal",
    "nBtagLoose_nominal",
    "nBtagMedium_nominal",
    "dimuon_pt",
    "dimuon_eta",
    "dimuon_mass",
    "jet1_phi_nominal",
    "jet1_pt_nominal",
    "jet2_pt_nominal",
    "jet2_phi_nominal",
    "jet1_eta_nominal",
    "jet2_eta_nominal",
    "jj_mass_nominal",
    "jj_dEta_nominal",
    "event",
    "njets_nominal",
    # "nfatJets_drmuon",
    # "MET_pt",
]

DIMUON_VARS = list(V1_FIELDS_2COMPUTE)

# ----------------------------------------------------------------------
# Dask client helper
# ----------------------------------------------------------------------
def get_dask_client(
    n_workers: int = 12,
    threads_per_worker: int = 1,
    memory_limit: str = "10 GiB",
) -> Client:
    """Create or reuse a local Dask client."""
    try:
        client = Client.current()
        print(f"Reusing existing Dask client: {client}")
        return client
    except ValueError:
        client = Client(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            processes=True,
            memory_limit=memory_limit,
        )
        print(f"Created new Dask client: {client}")
        return client


# ----------------------------------------------------------------------
# Parquet discovery
# ----------------------------------------------------------------------
def find_parquet_pattern(directory: str) -> str:
    """
    Build a glob pattern for parquet files under a stage-1-like directory.

    Tries:
      dir/*/*.parquet
      dir/*.parquet

    Returns the first pattern that matches at least one file.
    Raises if nothing is found.
    """
    directory = str(directory)
    patterns = [
        f"{directory}/*/*.parquet",
        f"{directory}/*.parquet",
    ]

    for pat in patterns:
        if glob.glob(pat):
            return pat

    raise FileNotFoundError(f"No parquet files found under '{directory}'")


# ----------------------------------------------------------------------
# Load + optional selection
# ----------------------------------------------------------------------
def load_dir_to_df(
    directory: str,
    category: Optional[str] = None,
    region: Optional[str] = None,
    process: str = "data",
) -> pd.DataFrame:
    """
    Load all parquet files from a directory into a pandas DataFrame
    using dask_awkward, with optional selection.applyRegionCatCuts.

    Columns kept: run, luminosityBlock, event, dimuon_pt, dimuon_mass, dimuon_eta
    (and V1_FIELDS_2COMPUTE, only those that exist).
    """
    pattern = find_parquet_pattern(directory)
    print(f"[INFO] Reading parquet pattern: {pattern}")

    # Combine and de-duplicate column list (keep order: keys → dimuon → extra)
    cols = []
    for c in KEY_VARS + DIMUON_VARS + V1_FIELDS_2COMPUTE:
        if c not in cols:
            cols.append(c)

    # dask_awkward lazy collection
    events_lazy = dak.from_parquet(pattern)

    # Restrict to columns that actually exist
    available = [c for c in cols if c in events_lazy.fields]
    missing = [c for c in cols if c not in events_lazy.fields]

    if missing:
        print(f"[WARNING] Missing columns in {directory}: {missing}")

    events_lazy = events_lazy[available]

    # Materialize to awkward Array
    events = events_lazy.compute()

    # Optional selection
    if category is not None and region is not None:
        print(
            f"[INFO] Applying selection: category={category}, "
            f"region={region}, process={process}"
        )
        events = selection.applyRegionCatCuts(
            events,
            category=category,
            region_name=region,
            process=process,
            variation="nominal",
            do_vbf_filter_study=False,
            do_VH_veto=False,
        )
    else:
        print("[INFO] No selection applied (category/region not both provided).")

    # Convert to pandas (awkward v2: no ak.to_pandas)
    df = pd.DataFrame(ak.to_list(events))

    # Keep only our desired columns (those that exist)
    keep_cols = [c for c in cols if c in df.columns]
    df = df[keep_cols]

    print(f"[INFO] Loaded {len(df)} rows from {directory}\n")
    return df


# ----------------------------------------------------------------------
# Single-dir dump
# ----------------------------------------------------------------------
def dump_single_dir_sync(df: pd.DataFrame, out_path: Path) -> None:
    """
    Save a text file with:
    event,run,luminosityBlock,dimuon_pt,dimuon_mass,dimuon_eta
    """
    cols = KEY_VARS + DIMUON_VARS
    cols = [c for c in cols if c in df.columns]

    # Reorder so event,run,lumi come in that order
    ordered = ["event", "run", "luminosityBlock"]
    for c in DIMUON_VARS:
        if c in cols:
            ordered.append(c)

    ordered = [c for c in ordered if c in df.columns]

    df_out = df[ordered].copy()
    df_out.to_csv(out_path, index=False)
    print(f"[INFO] Wrote {len(df_out)} rows to {out_path}")


# ----------------------------------------------------------------------
# Two-dir comparison
# ----------------------------------------------------------------------
def compare_two_dirs(
    dir1: str,
    dir2: str,
    out_path: Path,
    tolerance: float = 0.0,
    category: Optional[str] = None,
    region: Optional[str] = None,
    process: str = "data",
) -> None:
    """
    Compare two directories of parquet files by (run, luminosityBlock, event).

    For matching events, compare dimuon variables and write only mismatches.

    Output columns:
      run, luminosityBlock, event,
      dimuon_pt_1, dimuon_pt_2, delta_dimuon_pt,
      dimuon_mass_1, dimuon_mass_2, delta_dimuon_mass,
      dimuon_eta_1, dimuon_eta_2, delta_dimuon_eta
    """
    print(f"[INFO] Loading directory 1: {dir1}")
    df1 = load_dir_to_df(dir1, category=category, region=region, process=process)

    print(f"[INFO] Loading directory 2: {dir2}")
    df2 = load_dir_to_df(dir2, category=category, region=region, process=process)

    # Index by (run, lumi, event)
    if not set(KEY_VARS).issubset(df1.columns):
        raise RuntimeError(f"Missing key columns in dir1 DataFrame: {KEY_VARS}")
    if not set(KEY_VARS).issubset(df2.columns):
        raise RuntimeError(f"Missing key columns in dir2 DataFrame: {KEY_VARS}")

    df1 = df1.set_index(KEY_VARS)
    df2 = df2.set_index(KEY_VARS)

    common_idx = df1.index.intersection(df2.index)
    only1 = df1.index.difference(df2.index)
    only2 = df2.index.difference(df1.index)

    print(f"[INFO] Common events: {len(common_idx)}")
    print(f"[INFO] Events only in dir1: {len(only1)}")
    print(f"[INFO] Events only in dir2: {len(only2)}")

    if len(common_idx) == 0:
        print("[WARNING] No common events found; nothing to compare.")
        return

    c1 = df1.loc[common_idx]
    c2 = df2.loc[common_idx]

    rows: List[dict] = []

    for idx in common_idx:
        row1 = c1.loc[idx]
        row2 = c2.loc[idx]

        record = {
            "run": idx[0],
            "luminosityBlock": idx[1],
            "event": idx[2],
        }

        mismatch = False

        for var in DIMUON_VARS:
            if var not in row1 or var not in row2:
                continue

            v1 = row1[var]
            v2 = row2[var]
            delta = v2 - v1

            record[f"{var}_1"] = v1
            record[f"{var}_2"] = v2
            record[f"delta_{var}"] = delta

            if abs(delta) > tolerance:
                mismatch = True

        if mismatch:
            rows.append(record)

    if not rows:
        print("[INFO] No mismatches found (within tolerance).")
        return

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)
    print(f"[INFO] Wrote {len(df_out)} mismatching events to {out_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Sync / compare dimuon variables between directories of parquet files."
    )
    parser.add_argument(
        "dirs",
        nargs="+",
        help="One or two directories containing parquet files.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="sync_0_vs_3_dimuon_diff.txt",
        help="Output txt/csv file path. If not given, derived from directory name(s).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Absolute tolerance for comparing dimuon variables (default: 0).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="vbf",
        help="Category label for selection.applyRegionCatCuts (default: 'vbf').",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="h-sidebands",
        help="Region name for selection.applyRegionCatCuts (default: 'h-sidebands').",
    )
    parser.add_argument(
        "--process",
        type=str,
        default="data",
        help="Process name passed to selection.applyRegionCatCuts (default: 'data').",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    dirs = args.dirs

    get_dask_client()

    if len(dirs) == 1:
        directory = dirs[0]
        if args.out is None:
            out_path = Path(directory.rstrip("/")).name + "_sync.txt"
            out_path = Path(out_path)
        else:
            out_path = Path(args.out)

        df = load_dir_to_df(
            directory,
            category=args.category,
            region=args.region,
            process=args.process,
        )
        dump_single_dir_sync(df, out_path)

    elif len(dirs) == 2:
        dir1, dir2 = dirs

        if args.out is None:
            name1 = Path(dir1.rstrip("/")).name
            name2 = Path(dir2.rstrip("/")).name
            out_name = f"sync_{name1}_vs_{name2}_dimuon_diff.txt"
            out_path = Path(out_name)
        else:
            out_path = Path(args.out)

        compare_two_dirs(
            dir1=dir1,
            dir2=dir2,
            out_path=out_path,
            tolerance=args.tolerance,
            category=args.category,
            region=args.region,
            process=args.process,
        )

    else:
        raise SystemExit("Please provide one or two directories.")


if __name__ == "__main__":
    main()
