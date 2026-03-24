#!/usr/bin/env python3

"""
Sync or compare dimuon variables between directories of parquet files.
Usage:
    python sync_parquet_dimuon.py <dir1> [<dir2>] [--out OUTPUT] [--tolerance TOLERANCE]

Example:
    time python ./scripts/sync_parquet_dimuon.py  /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2017/f1_0/data_D/0 /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_28Nov_HEMVetoFix_NoSyst_V2/stage1_output/2017/f1_0/data_D/0

    time python ./scripts/sync_parquet_dimuon.py  /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_Peking_sync/stage1_output/2022preEE/f1_0/data_C/0/

"""

import argparse
import glob
from pathlib import Path
from typing import List, Optional

import dask_awkward as dak
import awkward as ak
import pandas as pd
import json

from distributed import Client
from multiprocessing import Pool
from functools import partial
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# ----------------------------------------------------------------------
# Columns
# ----------------------------------------------------------------------
KEY_VARS = ["run", "luminosityBlock", "event"]

SYNCVARLIST: List[str] = [
    # event id
    "run",
    "luminosityBlock",
    "event",
    # leptons
    "mu1_pt",
    "mu1_eta",
    "mu1_phi",
    "mu2_pt",
    "mu2_eta",
    "mu2_phi",
    # dimuon
    "dimuon_mass",
    "dimuon_pt",
    "dimuon_eta",
    "dimuon_phi",
    # jets
    "jet1_pt_nominal",
    "jet1_eta_nominal",
    "jet1_phi_nominal",
    "jet2_pt_nominal",
    "jet2_eta_nominal",
    "jet2_phi_nominal",
    "jj_mass_nominal",
    "jj_dEta_nominal",
    # optional weights (keep if you need selection to work)
    "wgt_nominal",
    "nBtagLoose_nominal",
    "nBtagMedium_nominal",
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
]

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
    directory = Path(directory).resolve().as_posix()

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
    for c in SYNCVARLIST:
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

    # # Optional selection
    # if category is not None and region is not None:
    #     print(
    #         f"[INFO] Applying selection: category={category}, "
    #         f"region={region}, process={process}"
    #     )
    #     events = selection.applyRegionCatCuts(
    #         events,
    #         category=category,
    #         region_name=region,
    #         process=process,
    #         variation="nominal",
    #         do_vbf_filter_study=False,
    #         do_VH_veto=False,
    #     )
    # else:
    #     print("[INFO] No selection applied (category/region not both provided).")

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
def dump_single_dir_sync_to_CSV(df: pd.DataFrame, out_path: Path) -> None:
    """
    Save a text file with:
    event,run,luminosityBlock,dimuon_pt,dimuon_mass,dimuon_eta
    """
    cols = KEY_VARS + SYNCVARLIST
    cols = [c for c in cols if c in df.columns]

    # Reorder so event,run,lumi come in that order
    ordered = ["event", "run", "luminosityBlock"]
    for c in SYNCVARLIST:
        if c in cols:
            ordered.append(c)

    ordered = [c for c in ordered if c in df.columns]

    df_out = df[ordered].copy()
    df_out.to_csv(out_path, index=False)
    print(f"[INFO] Wrote {len(df_out)} rows to {out_path}")


def dump_single_dir_sync(df: pd.DataFrame, out_path: Path) -> None:
    """
    Save a text file with one event per line in format:

    run:lumi:event,mu1_pt,mu1_eta,mu1_phi,mu2_pt,mu2_eta,mu2_phi,
    dimuon_mass,dimuon_pt,dimuon_eta,dimuon_phi,
    jet1_pt_nominal,jet1_eta_nominal,jet1_phi_nominal,
    jet2_pt_nominal,jet2_eta_nominal,jet2_phi_nominal,
    jj_mass_nominal,jj_dEta_nominal,...

    Missing values are written as -100.00
    """
    missing = [c for c in SYNCVARLIST if c not in df.columns]
    required = [c for c in SYNCVARLIST if c not in KEY_VARS and c in df.columns]

    if missing:
        print(f"[WARNING] Missing columns for sync dump: {missing}")

    df2 = df.copy()

    for c in required:
        df2[c] = df2[c].fillna(-100.0)

    with open(out_path, "w") as f:
        for _, row in df2.iterrows():
            run = int(row["run"])
            lumi = int(row["luminosityBlock"])
            event = int(row["event"])

            values = []
            for c in required:
                v = row[c]
                if pd.isna(v):
                    v = -100.0
                values.append(f"{float(v):.2f}")

            line = f"{run}:{lumi}:{event}," + ",".join(values)
            f.write(line + "\n")

    print(f"[INFO] Wrote {len(df2)} lines to {out_path}")


def process_common_idxs(common_idx, c1, c2, tolerance, use_rel_error=False):
    """
    helper function that speeds looking up any difference in common idxs
    within a set tolerance
    """

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

        for var in SYNCVARLIST:
            if var not in row1 or var not in row2:
                continue

            v1 = row1[var]
            v2 = row2[var]
            delta = v2 - v1

            record[f"{var}_1"] = v1
            record[f"{var}_2"] = v2
            record[f"delta_{var}"] = delta

            if use_rel_error:
                if v2 == 0:
                    err = abs(delta/v1)
                else:
                    err = abs(delta/v2)
            else:
                err = abs(delta)

            if err > tolerance:
                mismatch = True
                print(f"[Info]: Surpass err tol on {var}")

        if mismatch:
            rows.append(record)
    return rows

def chunk_list(lst, chunk_size):
  """Split a list into chunks of specified size."""
  return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


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


    # ---------------------------
    chunk_size = 10_000
    common_idx_chunks = chunk_list(common_idx, chunk_size)
    print(f"[INFO] len(common_idx_chunks): {len(common_idx_chunks)}")
    # apply parallelization
    inputs = [(c1, c2, common_idx_chunk, tolerance) for common_idx_chunk in common_idx_chunks]
    partial_process_common_idxs = partial(process_common_idxs, c1=c1, c2=c2, tolerance=tolerance, use_rel_error=False)
    with Pool() as pool:
        rows = pool.map(partial_process_common_idxs, common_idx_chunks)
        # flatten rows
        import itertools
        rows = list(itertools.chain.from_iterable(rows))

    if not rows:
        print("[INFO] No mismatches found (within tolerance).")
        # return
    else:
        df_out = pd.DataFrame(rows)
        df_out.to_csv(out_path, index=False)
        print(f"[INFO] Wrote {len(df_out)} mismatching events to {out_path}")
    # ---------------------------

    # plotComparison = True
    # # ---------------------------
    # # plot common variables
    # # ---------------------------
    # # test_size=1000
    # # c1 = c1[:test_size]
    # # c2 = c2[:test_size]
    # if plotComparison:
    #     variables = ["mu1_pt", "mu2_pt", "mu1_eta", "mu2_eta"]
    #     weight_field = "wgt_nominal"
        
    #     bin_edges_dict = {
    #         # "mu1_pt":  np.array([0, 10, 20, 30, 40, 50, 75, 100, 150, 200]),
    #         # "mu2_pt":  np.array([0, 10, 20, 30, 40, 50, 75, 100, 150, 200]),
    #         # "mu1_eta": np.array([-2.4, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.4]),
    #         # "mu2_eta": np.array([-2.4, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.4]),s
    #         "mu1_pt":  np.linspace(0, 120, 50),
    #         "mu2_pt":  np.linspace(0, 120, 50),
    #         "mu1_eta": np.linspace(-2.4, 2.4, 50),
    #         "mu2_eta": np.linspace(-2.4, 2.4, 50),
    #     }
    #     save_dir = "./common_idx"
    #     os.makedirs(save_dir, exist_ok=True)

    #     for var in variables:
    #         bin_edges = bin_edges_dict[var]
        
    #         counts_v12 = weighted_histogram_flat(c1, var, weight_field, bin_edges)
    #         counts_v15 = weighted_histogram_flat(c2, var, weight_field, bin_edges)
    #         plot_overlay(bin_edges, counts_v12, counts_v15, var, save_dir, normalized=False)

    #     save_dir = "./as_is"
    #     os.makedirs(save_dir, exist_ok=True)
    #     for var in variables:
    #         bin_edges = bin_edges_dict[var]
        
    #         counts_v12 = weighted_histogram_flat(df1, var, weight_field, bin_edges)
    #         counts_v15 = weighted_histogram_flat(df2, var, weight_field, bin_edges)
    #         print(f"{var} counts_v12: {counts_v12}")
    #         print(f"{var} counts_v15: {counts_v15}")
    #         plot_overlay(bin_edges, counts_v12, counts_v15, var, save_dir, normalized=False)
        
    #     save_dir = "./as_is_normToUnity"
    #     os.makedirs(save_dir, exist_ok=True)
    #     for var in variables:
    #         bin_edges = bin_edges_dict[var]
        
    #         counts_v12 = weighted_histogram_flat(df1, var, weight_field, bin_edges)
    #         counts_v15 = weighted_histogram_flat(df2, var, weight_field, bin_edges)
    #         plot_overlay(bin_edges, counts_v12, counts_v15, var, save_dir, normalized=True)

def parse_sync_txt(path: str) -> pd.DataFrame:
    """
    Parse sync txt lines in format:

    run:lumi:event,val1,val2,...

    The variable order is taken from SYNCVARLIST (excluding KEY_VARS).
    If a line has fewer values than SYNCVARLIST expects, the missing ones
    are filled with -100.0. If it has more, extras are ignored.

    Returns
    -------
    pd.DataFrame
        Indexed by (run, luminosityBlock, event).
    """
    value_cols = [c for c in SYNCVARLIST if c not in KEY_VARS]

    # Read all non-empty lines
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        raise RuntimeError(f"No non-empty lines found in {path}")

    bad = 0
    records = []

    for iline, line in enumerate(lines, start=1):
        parts = line.split(",")

        if len(parts) < 2:
            bad += 1
            print(f"[WARNING] malformed line {iline} in {path}: too few fields")
            continue

        # Parse run:lumi:event
        try:
            run_str, lumi_str, evt_str = parts[0].split(":")
            run = int(run_str)
            lumi = int(lumi_str)
            evt = int(evt_str)
        except Exception:
            bad += 1
            print(f"[WARNING] failed to parse run:lumi:event on line {iline}: {parts[0]}")
            continue

        raw_vals = parts[1:]

        # Optional one-time info
        if iline == 1:
            print(
                f"[INFO] Detected {len(raw_vals)} value columns in {path}; "
                f"SYNCVARLIST expects {len(value_cols)}"
            )

        row = {
            "run": run,
            "luminosityBlock": lumi,
            "event": evt,
        }

        # Fill from SYNCVARLIST order, pad missing with -100.0
        for i, col in enumerate(value_cols):
            if i < len(raw_vals):
                try:
                    row[col] = float(raw_vals[i])
                except Exception:
                    row[col] = -100.0
            else:
                row[col] = -100.0

        records.append(row)

    if bad:
        print(f"[WARNING] {bad} malformed lines skipped in {path}")

    if not records:
        raise RuntimeError(f"No valid rows parsed from {path}")

    df = pd.DataFrame.from_records(records)
    df = df.set_index(KEY_VARS).sort_index()
    return df

def weighted_histogram_flat(dak_array, value_field, weight_field, bin_edges):
    values = (dak_array[value_field])
    # weights = ak.to_numpy(dak_array[weight_field])
    weights = np.ones_like(values)

    mask = np.isfinite(values) & np.isfinite(weights)
    values = values[mask]
    weights = weights[mask]

    counts, _ = np.histogram(values, bins=bin_edges, weights=weights)
    return counts

def plot_overlay(bin_edges, counts_v12, counts_v15, var_name, save_dir, normalized=False):
    y12 = counts_v12.astype(float)
    y15 = counts_v15.astype(float)

    if normalized:
        if y12.sum() > 0:
            y12 = y12 / y12.sum()
        if y15.sum() > 0:
            y15 = y15 / y15.sum()

    plt.figure(figsize=(7, 5))
    plt.stairs(y12, bin_edges, label="v12", linewidth=1)
    plt.stairs(y15, bin_edges, label="v15", linewidth=1)

    plt.xlabel(var_name)
    plt.ylabel("Normalized entries" if normalized else "Weighted entries")
    plt.title(f"{var_name}: v12 vs v15")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{save_dir}/{var_name}.png")


def compare_two_sync_txt(
    txt1: str,
    txt2: str,
    out_path: Path,
    tolerance: float = 0.0,
) -> None:
    """
    Compare two sync txt dumps by (run,luminosityBlock,event).
    Writes mismatches to out_path as CSV with *_1, *_2, delta_* columns.
    """
    print(f"[INFO] Loading sync txt 1: {txt1}")
    df1 = parse_sync_txt(txt1)

    print(f"[INFO] Loading sync txt 2: {txt2}")
    df2 = parse_sync_txt(txt2)

    common_idx = df1.index.intersection(df2.index)
    only1 = df1.index.difference(df2.index)
    only2 = df2.index.difference(df1.index)

    print(f"[INFO] Common events: {len(common_idx)}")
    print(f"[INFO] Events only in txt1: {len(only1)}")
    print(f"[INFO] Events only in txt2: {len(only2)}")

    # write only1 and only2 to separate files
    if len(only1) > 0:
        only1_path = out_path.parent / f"only_in_{Path(txt1).stem}.txt"
        df1.loc[only1].reset_index().to_csv(only1_path, index=False)
        print(f"[INFO] Wrote {len(only1)} events only in txt1 to {only1_path}")
    if len(only2) > 0:
        only2_path = out_path.parent / f"only_in_{Path(txt2).stem}.txt"
        df2.loc[only2].reset_index().to_csv(only2_path, index=False)
        print(f"[INFO] Wrote {len(only2)} events only in txt2 to {only2_path}")

    if len(common_idx) == 0:
        print("[WARNING] No common events found; nothing to compare.")
        return

    c1 = df1.loc[common_idx]
    c2 = df2.loc[common_idx]

    # Variables to compare (everything except keys)
    vars_to_check = [c for c in c1.columns if c in c2.columns]

    rows = []
    for idx in common_idx:
        r1 = c1.loc[idx]
        r2 = c2.loc[idx]

        mismatch = False
        rec = {"run": idx[0], "luminosityBlock": idx[1], "event": idx[2]}

        for v in vars_to_check:
            v1 = float(r1[v])
            v2 = float(r2[v])
            d = v2 - v1
            rec[f"{v}_1"] = v1
            rec[f"{v}_2"] = v2
            rec[f"delta_{v}"] = d
            if abs(d) > tolerance:
                mismatch = True

        if mismatch:
            rows.append(rec)

    if not rows:
        print("[INFO] No mismatches found (within tolerance).")
        return

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[INFO] Wrote {len(rows)} mismatching events to {out_path}")

    return


def compare_two_cutflow_json(
    json1: str,
    json2: str,
    out_path: Path,
    tolerance: float = 0.0,
) -> None:
    """
    Compare two cutflow JSON files of format:

    {
        "CutName": {
            "cumulative": 1000,
            "individual": 1000
        },
        ...
    }

    Writes only mismatches to out_path.
    """
    print(f"[INFO] Loading cutflow json 1: {json1}")
    with open(json1, "r") as f:
        d1 = json.load(f)

    print(f"[INFO] Loading cutflow json 2: {json2}")
    with open(json2, "r") as f:
        d2 = json.load(f)

    all_cuts = sorted(set(d1.keys()) | set(d2.keys()))
    rows = []

    for cut in all_cuts:
        rec = {"cut": cut}

        cut1 = d1.get(cut)
        cut2 = d2.get(cut)

        if cut1 is None:
            rec["status"] = "missing_in_json1"
            rec["cumulative_1"] = None
            rec["cumulative_2"] = cut2.get("cumulative")
            rec["delta_cumulative"] = None
            rec["individual_1"] = None
            rec["individual_2"] = cut2.get("individual")
            rec["delta_individual"] = None
            rows.append(rec)
            continue

        if cut2 is None:
            rec["status"] = "missing_in_json2"
            rec["cumulative_1"] = cut1.get("cumulative")
            rec["cumulative_2"] = None
            rec["delta_cumulative"] = None
            rec["individual_1"] = cut1.get("individual")
            rec["individual_2"] = None
            rec["delta_individual"] = None
            rows.append(rec)
            continue

        c1 = float(cut1.get("cumulative", 0.0))
        c2 = float(cut2.get("cumulative", 0.0))
        i1 = float(cut1.get("individual", 0.0))
        i2 = float(cut2.get("individual", 0.0))

        dc = c2 - c1
        di = i2 - i1

        if abs(dc) > tolerance or abs(di) > tolerance:
            rec["status"] = "different"
            rec["cumulative_1"] = c1
            rec["cumulative_2"] = c2
            rec["delta_cumulative"] = dc
            rec["individual_1"] = i1
            rec["individual_2"] = i2
            rec["delta_individual"] = di
            rows.append(rec)

    if not rows:
        print("[INFO] No cutflow mismatches found (within tolerance).")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[INFO] Wrote {len(df)} cutflow mismatches to {out_path}")

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
        default=None,
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
        default="nocat",
        help="Category label for selection.applyRegionCatCuts (default: 'nocat').",
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
        print("[INFO] Single directory provided: dumping sync txt file.")
        directory = dirs[0]
        print(f"[INFO] Loading directory: {directory}")
        if args.out is None:
            out_path = Path(directory.rstrip("/")).name + "_sync.txt"
            out_path = Path(out_path)
        else:
            out_path = Path(args.out)

        print(f"[INFO] Output path: {out_path}")
        df = load_dir_to_df(
            directory,
            category=args.category,
            region=args.region,
            process=args.process,
        )
        dump_single_dir_sync(df, out_path)

    elif len(dirs) == 2:
        file1, file2 = dirs

        out_path = Path(args.out) if args.out else Path("sync_txt_diff.txt")

        # If both are text dumps -> compare text files
        if (str(file1).endswith(".txt") ) and (str(file2).endswith(".txt")):
            compare_two_sync_txt(
                txt1=file1,
                txt2=file2,
                out_path=out_path,
                tolerance=args.tolerance,
            )
            return

        # If both are cutflow json files -> compare json files
        if str(file1).endswith(".json") and str(file2).endswith(".json"):
            compare_two_cutflow_json(
                json1=file1,
                json2=file2,
                out_path=out_path,
                tolerance=args.tolerance,
            )
            return

        # Otherwise treat as directories (existing behavior)
        dir1, dir2 = file1, file2

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
