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
import os
from pathlib import Path
from typing import List, Optional

import json
import pandas as pd
import pyarrow.parquet as pq

from modules.dask_utils import close_dask_client, get_dask_client


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
    "separate_wgt_pu",
    "separate_wgt_l1prefiring",
    "separate_wgt_muID",
    "separate_wgt_muIso",
    "separate_wgt_muTrig",
    "separate_wgt_LHERen",
    "separate_wgt_LHEFac",
    "separate_wgt_pdf_2rms",
    "separate_wgt_jetpuid",
    "separate_wgt_btag",
    "separate_wgt_qgl",
    "separate_wgt_zpt",
    "separate_wgt_zpt_wgt",
    "separate_wgt_ones",
    # "zpt_wgt_reco",
    # "zpt_wgt_gen",
]

TXT_COMPARE_EXCLUDED_VARS = {
    "wgt_nominal",
    "separate_wgt_genWeight",
    "separate_wgt_genWeight_normalization",
    "separate_wgt_xsec",
    "separate_wgt_lumi",
    "separate_wgt_pu",
    "separate_wgt_l1prefiring",
    "separate_wgt_muID",
    "separate_wgt_muIso",
    "separate_wgt_muTrig",
    "separate_wgt_LHERen",
    "separate_wgt_LHEFac",
    "separate_wgt_pdf_2rms",
    "separate_wgt_jetpuid",
    "separate_wgt_btag",
    "separate_wgt_qgl",
    "separate_wgt_zpt",
    "separate_wgt_zpt_wgt",
    "separate_wgt_ones",
    "zpt_wgt_reco",
    "zpt_wgt_gen",
}

WEIGHT_LIKE_VARS = {
    "wgt_nominal",
    "nBtagLoose_nominal",
    "nBtagMedium_nominal",
    "separate_wgt_genWeight",
    "separate_wgt_genWeight_normalization",
    "separate_wgt_xsec",
    "separate_wgt_lumi",
    "separate_wgt_pu",
    "separate_wgt_l1prefiring",
    "separate_wgt_muID",
    "separate_wgt_muIso",
    "separate_wgt_muTrig",
    "separate_wgt_LHERen",
    "separate_wgt_LHEFac",
    "separate_wgt_pdf_2rms",
    "separate_wgt_jetpuid",
    "separate_wgt_btag",
    "separate_wgt_qgl",
    "separate_wgt_zpt",
    "separate_wgt_zpt_wgt",
    "separate_wgt_ones",
    "zpt_wgt_reco",
    "zpt_wgt_gen",
}

MUON_DIMUON_VARS = {
    "mu1_pt", "mu1_eta", "mu1_phi",
    "mu2_pt", "mu2_eta", "mu2_phi",
    "dimuon_mass", "dimuon_pt", "dimuon_eta", "dimuon_phi",
}

JET_VARS = {
    "jet1_pt_nominal", "jet1_eta_nominal", "jet1_phi_nominal",
    "jet2_pt_nominal", "jet2_eta_nominal", "jet2_phi_nominal",
    "jj_mass_nominal", "jj_dEta_nominal",
}


def _is_data_sync_source(label: str) -> bool:
    label_l = str(label).lower()
    name_l = Path(str(label)).name.lower()
    return ("_data_" in label_l) or name_l.startswith("data") or "data_" in name_l


def _summary_path(out_path: Path) -> Path:
    return out_path.with_name(f"summary_{out_path.stem}.txt")


def _classify_variable_group(variable: str) -> str:
    if variable in WEIGHT_LIKE_VARS:
        return "weights"
    if variable in MUON_DIMUON_VARS:
        return "muon_dimuon"
    if variable in JET_VARS:
        return "jet_jj"
    return "other"


def _write_summary_report(
    out_path: Path,
    common_events: int,
    only1_count: int,
    only2_count: int,
    mismatch_count: int,
    variable_stats: list[dict],
    comparison_label: str,
) -> None:
    summary_path = _summary_path(out_path)
    lines = [
        f"comparison: {comparison_label}",
        f"common_events: {common_events}",
        f"only_in_1: {only1_count}",
        f"only_in_2: {only2_count}",
        f"mismatching_events: {mismatch_count}",
        "",
        "top_changed_variables:",
    ]

    if variable_stats:
        for row in variable_stats:
            lines.append(
                f"- {row['variable']}: count={row['count']}, "
                f"group={row['group']}, max_abs_delta={row['max_abs_delta']}, "
                f"mean_abs_delta={row['mean_abs_delta']}"
            )
    else:
        lines.append("- none")

    summary_path.write_text("\n".join(lines) + "\n")
    print(f"[INFO] Wrote summary report to {summary_path}")


def _build_diff_output(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    variables: list[str],
    tolerance: float,
    out_path: Path,
    common_idx,
    only1_count: int,
    only2_count: int,
    comparison_label: str,
) -> None:
    if not variables:
        print("[WARNING] No shared variables available to compare.")
        _write_summary_report(
            out_path=out_path,
            common_events=len(common_idx),
            only1_count=only1_count,
            only2_count=only2_count,
            mismatch_count=0,
            variable_stats=[],
            comparison_label=comparison_label,
        )
        return

    left = left_df[variables].astype(float).add_suffix("_1")
    right = right_df[variables].astype(float).add_suffix("_2")
    merged = left.join(right, how="inner")

    mismatch_mask = pd.Series(False, index=merged.index)
    mismatch_by_var = {}
    max_abs_delta_by_var = {}
    mean_abs_delta_by_var = {}

    for var in variables:
        delta_col = (merged[f"{var}_2"] - merged[f"{var}_1"]).round(6)
        merged[f"delta_{var}"] = delta_col
        var_mask = delta_col.abs() > tolerance
        mismatch_by_var[var] = int(var_mask.sum())
        if mismatch_by_var[var] > 0:
            abs_delta = delta_col[var_mask].abs()
            max_abs_delta_by_var[var] = round(float(abs_delta.max()), 6)
            mean_abs_delta_by_var[var] = round(float(abs_delta.mean()), 6)
        else:
            max_abs_delta_by_var[var] = 0.0
            mean_abs_delta_by_var[var] = 0.0
        mismatch_mask |= var_mask

    variable_stats = [
        {
            "variable": var,
            "count": mismatch_by_var[var],
            "group": _classify_variable_group(var),
            "max_abs_delta": max_abs_delta_by_var[var],
            "mean_abs_delta": mean_abs_delta_by_var[var],
        }
        for var in variables
        if mismatch_by_var[var] > 0
    ]
    variable_stats.sort(key=lambda row: (-row["count"], -row["max_abs_delta"], row["variable"]))

    if not bool(mismatch_mask.any()):
        print("[INFO] No mismatches found (within tolerance).")
        _write_summary_report(
            out_path=out_path,
            common_events=len(common_idx),
            only1_count=only1_count,
            only2_count=only2_count,
            mismatch_count=0,
            variable_stats=[],
            comparison_label=comparison_label,
        )
        return

    mismatches = merged.loc[mismatch_mask].copy()
    delta_cols = [f"delta_{var}" for var in variables]
    changed_lists = []
    changed_groups = []
    n_changed = []
    max_abs_delta = []
    physics_category = []

    for _, row in mismatches[delta_cols].iterrows():
        changed = []
        groups = set()
        max_delta = 0.0
        for var in variables:
            delta = row[f"delta_{var}"]
            if abs(delta) > tolerance:
                changed.append(var)
                groups.add(_classify_variable_group(var))
                max_delta = max(max_delta, abs(float(delta)))
        changed_lists.append(",".join(changed))
        changed_groups.append(",".join(sorted(groups)))
        n_changed.append(len(changed))
        max_abs_delta.append(round(max_delta, 6))
        if groups == {"weights"}:
            physics_category.append("weight_only")
        elif "weights" in groups:
            physics_category.append("kinematics_and_weights")
        else:
            physics_category.append("kinematics_only")

    mismatches["changed_columns"] = changed_lists
    mismatches["changed_groups"] = changed_groups
    mismatches["physics_category"] = physics_category
    mismatches["n_changed"] = n_changed
    mismatches["max_abs_delta"] = max_abs_delta

    key_df = mismatches.reset_index()[KEY_VARS + ["_sync_instance"]]
    compact = key_df.copy()
    compact["physics_category"] = mismatches["physics_category"].to_numpy()
    compact["changed_groups"] = mismatches["changed_groups"].to_numpy()
    compact["changed_columns"] = mismatches["changed_columns"].to_numpy()
    compact["n_changed"] = mismatches["n_changed"].to_numpy()
    compact["max_abs_delta"] = mismatches["max_abs_delta"].to_numpy()

    for var in variables:
        mask = mismatches[f"delta_{var}"].abs() > tolerance
        compact[f"{var}_1"] = mismatches[f"{var}_1"].where(mask).round(6).to_numpy()
        compact[f"{var}_2"] = mismatches[f"{var}_2"].where(mask).round(6).to_numpy()
        compact[f"delta_{var}"] = mismatches[f"delta_{var}"].where(mask).round(6).to_numpy()

    compact = compact.sort_values(
        ["n_changed", "max_abs_delta", "physics_category", "changed_columns"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    compact.to_csv(out_path, index=False, float_format="%.6f")
    print(f"[INFO] Wrote {len(compact)} mismatching events to {out_path}")

    _write_summary_report(
        out_path=out_path,
        common_events=len(common_idx),
        only1_count=only1_count,
        only2_count=only2_count,
        mismatch_count=len(compact),
        variable_stats=variable_stats,
        comparison_label=comparison_label,
    )


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
    use_gateway: bool = False,
) -> pd.DataFrame:
    """
    Load all parquet files from a directory into a pandas DataFrame
    using awkward parquet readers, with optional selection.applyRegionCatCuts.

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

    parquet_files = sorted(glob.glob(pattern))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files matched pattern '{pattern}'")

    # Restrict to columns that actually exist
    available_fields = set(pq.ParquetFile(parquet_files[0]).schema.names)
    available = [c for c in cols if c in available_fields]
    missing = [c for c in cols if c not in available]

    if missing:
        print(f"[WARNING] Missing columns in {directory}: {missing}")

    # Load channel column too (not in SYNCVARLIST) so we can filter mm-only
    extra_cols = ["channel"] if "channel" in available_fields else []

    # Load all parquet files and concatenate into a single DataFrame
    dfs = []
    for pf in parquet_files:
        table = pq.read_table(pf, columns=available + extra_cols)
        dfs.append(table.to_pandas())
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=available + extra_cols)

    # Keep only mm (dimuon) events when channel column is present; this script
    # is the H→µµ sync tool and reference files contain mm events only.
    if "channel" in df.columns:
        df = df[df["channel"] == 0].copy()
        df.drop(columns=["channel"], inplace=True)

    # Keep only our desired columns (those that exist)
    keep_cols = [c for c in cols if c in df.columns]
    df = df[keep_cols]

    print(f"[INFO] Loaded {len(df)} rows from {directory}\n")
    return df


def _load_parquet_file_to_df(parquet_file: str, columns: list[str]) -> pd.DataFrame:
    table = pq.read_table(parquet_file, columns=columns)
    return table.to_pandas()


def _with_occurrence_index(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Make repeated (run, lumi, event) keys comparable by appending a stable
    occurrence counter within each key group.
    """
    missing = [c for c in KEY_VARS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing key columns in {label}: {missing}")

    sort_cols = [c for c in KEY_VARS if c in df.columns]
    sort_cols.extend(c for c in SYNCVARLIST if c not in KEY_VARS and c in df.columns)
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    duplicate_mask = df.duplicated(KEY_VARS, keep=False)
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        print(
            f"[INFO] Found {duplicate_count} rows with duplicated event keys in {label}; with key columns {KEY_VARS}."
            "Appending occurrence index to make them comparable."
            "comparing them with an occurrence index."
        )

    df = df.copy()
    df["_sync_instance"] = df.groupby(KEY_VARS, sort=False).cumcount()
    return df.set_index(KEY_VARS + ["_sync_instance"]).sort_index()


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
    use_gateway: bool = False,
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
    df1 = load_dir_to_df(
        dir1,
        category=category,
        region=region,
        process=process,
        use_gateway=use_gateway,
    )

    print(f"[INFO] Loading directory 2: {dir2}")
    df2 = load_dir_to_df(
        dir2,
        category=category,
        region=region,
        process=process,
        use_gateway=use_gateway,
    )

    df1 = _with_occurrence_index(df1, f"dir1 ({dir1})")
    df2 = _with_occurrence_index(df2, f"dir2 ({dir2})")

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

    shared_vars = [var for var in SYNCVARLIST if var in c1.columns and var in c2.columns]
    if _is_data_sync_source(dir1) or _is_data_sync_source(dir2):
        shared_vars = [var for var in shared_vars if var not in WEIGHT_LIKE_VARS]

    _build_diff_output(
        left_df=c1,
        right_df=c2,
        variables=shared_vars,
        tolerance=tolerance,
        out_path=out_path,
        common_idx=common_idx,
        only1_count=len(only1),
        only2_count=len(only2),
        comparison_label=f"{dir1} vs {dir2}",
    )


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
    return _with_occurrence_index(df, path)


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

    # For data sync txt comparisons, skip weight-like variables
    skip_weight_like = _is_data_sync_source(txt1) or _is_data_sync_source(txt2)
    if skip_weight_like:
        vars_to_check = [
            c for c in c1.columns
            if c in c2.columns and c not in TXT_COMPARE_EXCLUDED_VARS
        ]
        skipped = [c for c in c1.columns if c in c2.columns and c in TXT_COMPARE_EXCLUDED_VARS]
        if skipped:
            print(f"[INFO] Skipping weight-like sync columns for data txt comparison: {skipped}")
    else:
        vars_to_check = [c for c in c1.columns if c in c2.columns]

    _build_diff_output(
        left_df=c1,
        right_df=c2,
        variables=vars_to_check,
        tolerance=tolerance,
        out_path=out_path,
        common_idx=common_idx,
        only1_count=len(only1),
        only2_count=len(only2),
        comparison_label=f"{txt1} vs {txt2}",
    )
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
        if out_path.exists():
            out_path.unlink()
        return

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
        default=0.1,
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
    parser.add_argument(
        "--use_gateway",
        dest="use_gateway",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, read parquet files in parallel on a Dask Gateway client using ak.from_parquet on workers.",
    )
    parser.add_argument(
        "--cluster_index",
        dest="cluster_index",
        default=0,
        type=int,
        help="Index of the Dask Gateway cluster to connect to when --use_gateway is enabled.",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    dirs = args.dirs
    client = None

    if args.use_gateway:
        print(f"[INFO] Connecting to Dask Gateway cluster index {args.cluster_index}")
        client = get_dask_client(use_gateway=True, cluster_index=args.cluster_index)
        print(f"[INFO] Connected Dask client: {client}")

    try:
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
                use_gateway=args.use_gateway,
            )
            dump_single_dir_sync(df, out_path)

        elif len(dirs) == 2:
            file1, file2 = (str(item).strip() for item in dirs)

            def normalize_input_path(path: str) -> str:
                cleaned = path.strip()
                if cleaned.endswith(".jsonn"):
                    candidate = cleaned[:-1]
                    if os.path.exists(candidate):
                        print(f"[DEBUG] repaired suspicious .jsonn suffix: {cleaned!r} -> {candidate!r}")
                        return candidate
                return cleaned

            file1 = normalize_input_path(file1)
            file2 = normalize_input_path(file2)

            out_path = Path(args.out) if args.out else Path("sync_txt_diff.txt")

            print(f"[DEBUG] normalized input 1: {file1!r}")
            print(f"[DEBUG] normalized input 2: {file2!r}")

            # If both are text dumps -> compare text files
            if (str(file1).endswith(".txt")) and (str(file2).endswith(".txt")):
                compare_two_sync_txt(
                    txt1=file1,
                    txt2=file2,
                    out_path=out_path,
                    tolerance=args.tolerance,
                )
                return

            # If both are cutflow json files -> compare json files
            if str(file1).endswith(".json") and str(file2).endswith(".json"):
                print("[DEBUG] entering cutflow JSON comparison branch")
                compare_two_cutflow_json(
                    json1=file1,
                    json2=file2,
                    out_path=out_path,
                    tolerance=args.tolerance,
                )
                return

            # Otherwise treat as directories (existing behavior)
            print("[DEBUG] falling back to directory/parquet comparison branch")
            dir1, dir2 = file1, file2

            compare_two_dirs(
                dir1=dir1,
                dir2=dir2,
                out_path=out_path,
                tolerance=args.tolerance,
                category=args.category,
                region=args.region,
                process=args.process,
                use_gateway=args.use_gateway,
            )

        else:
            raise SystemExit("Please provide one or two directories.")
    finally:
        if client is not None:
            close_dask_client()


if __name__ == "__main__":
    main()