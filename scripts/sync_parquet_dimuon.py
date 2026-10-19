#!/usr/bin/env python3

"""
Sync or compare dimuon variables between directories of parquet files.
Usage:
    python sync_parquet_dimuon.py <dir1> [<dir2>] [--out OUTPUT] [--tolerance TOLERANCE]

Example:
    time python ./scripts/sync_parquet_dimuon.py  /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2017/f1_0/data_D/0 /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_28Nov_HEMVetoFix_NoSyst_V2/stage1_output/2017/f1_0/data_D/0

    time python ./scripts/sync_parquet_dimuon.py  /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_Peking_sync/stage1_output/2022preEE/f1_0/data_C/0/

"""

import json
import argparse
import glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import awkward as ak
import numpy as np
import pandas as pd


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
    "separate_wgt_ones",
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
    "separate_wgt_ones",
}


HEADER_KEY_FIELD = "run:lumi:event"

DEFAULT_REL_TOLERANCE = 1e-3


def _is_data_sync_source(label: str) -> bool:
    label_l = str(label).lower()
    name_l = Path(str(label)).name.lower()
    return ("_data_" in label_l) or name_l.startswith("data") or "data_" in name_l


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
def _open_virtual(path: str):
    # Lazy import: coffea takes seconds to import and txt/json modes never need it
    from coffea.nanoevents import BaseSchema, NanoEventsFactory

    return NanoEventsFactory.from_parquet(
        path, schemaclass=BaseSchema, mode="virtual"
    ).events()


def _read_columns(path: str, columns: List[str]) -> pd.DataFrame:
    """Materialize only `columns` of one parquet file; missing values become NaN."""
    events = _open_virtual(path)
    data = {}
    for c in columns:
        arr = ak.to_numpy(events[c], allow_missing=True)
        if np.ma.isMaskedArray(arr):
            arr = np.ma.filled(arr.astype("float64"), np.nan)
        data[c] = arr
    return pd.DataFrame(data)


def load_dir_to_df(directory: str) -> pd.DataFrame:
    """
    Load the SYNCVARLIST columns (those that exist) of all parquet files in a
    directory into a pandas DataFrame, using coffea virtual arrays.
    """
    pattern = find_parquet_pattern(directory)
    print(f"[INFO] Reading parquet pattern: {pattern}")

    # Combine and de-duplicate column list (keep order: keys → dimuon → extra)
    cols = []
    for c in SYNCVARLIST:
        if c not in cols:
            cols.append(c)

    files = sorted(glob.glob(pattern))

    # Restrict to columns that actually exist
    first_fields = set(_open_virtual(files[0]).fields)
    available = [c for c in cols if c in first_fields]
    missing = [c for c in cols if c not in first_fields]

    if missing:
        print(f"[WARNING] Missing columns in {directory}: {missing}")

    # Virtual arrays only read the accessed columns; files are independent, so
    # read them in parallel (parquet decoding releases the GIL).
    with ThreadPoolExecutor(max_workers=min(8, len(files))) as pool:
        per_file = list(pool.map(lambda f: _read_columns(f, available), files))

    df = pd.concat(per_file, ignore_index=True)

    print(f"[INFO] Loaded {len(df)} rows from {directory}\n")
    return df


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
        )

    df = df.copy()
    df["_sync_instance"] = df.groupby(KEY_VARS, sort=False).cumcount()
    return df.set_index(KEY_VARS + ["_sync_instance"]).sort_index()


# ----------------------------------------------------------------------
# Single-dir dump
# ----------------------------------------------------------------------
def dump_single_dir_sync(df: pd.DataFrame, out_path: Path) -> None:
    """
    Save a text file whose first line names the columns, then one event per line:

    run:lumi:event,mu1_pt,mu1_eta,mu1_phi,mu2_pt,mu2_eta,mu2_phi,
    dimuon_mass,dimuon_pt,dimuon_eta,dimuon_phi,
    jet1_pt_nominal,jet1_eta_nominal,jet1_phi_nominal,
    jet2_pt_nominal,jet2_eta_nominal,jet2_phi_nominal,
    jj_mass_nominal,jj_dEta_nominal,...

    Only columns this sample actually has are written; the header records which
    ones, so a reader never has to infer a variable from its position. NaNs
    within a written column are still dumped as -100.00.
    """
    missing = [c for c in SYNCVARLIST if c not in df.columns]
    required = [c for c in SYNCVARLIST if c not in KEY_VARS and c in df.columns]

    if missing:
        print(f"[WARNING] Columns absent from this sample, not written: {missing}")

    # float() then %.2f, as the per-row loop did, so integer columns also print as N.00
    df2 = df[required].astype("float64").fillna(-100.0)
    keys = (
        df["run"].astype("int64").astype(str)
        + ":" + df["luminosityBlock"].astype("int64").astype(str)
        + ":" + df["event"].astype("int64").astype(str)
    )
    # Per-column str.format is several times faster than pandas' float_format
    columns = [keys.tolist()] + [
        list(map("{:.2f}".format, df2[c].to_numpy().tolist())) for c in required
    ]

    with open(out_path, "w") as f:
        f.write(HEADER_KEY_FIELD + "," + ",".join(required) + "\n")
        f.writelines(",".join(row) + "\n" for row in zip(*columns))

    print(f"[INFO] Wrote {len(df2)} lines to {out_path}")


def _mismatch_table(
    c1: pd.DataFrame, c2: pd.DataFrame, variables: List[str], tolerance: float
) -> pd.DataFrame:
    """
    Side-by-side *_1, *_2, delta_* table for rows of two index-aligned frames
    where any variable differs by more than the relative tolerance.
    """
    out = {}
    mismatch = np.zeros(len(c1), dtype=bool)
    for var in variables:
        v1 = c1[var].to_numpy(dtype="float64")
        v2 = c2[var].to_numpy(dtype="float64")
        out[f"{var}_1"] = v1
        out[f"{var}_2"] = v2
        out[f"delta_{var}"] = v2 - v1
        # NaN compares False, so NaN never counts as a mismatch
        mismatch |= np.abs(v2 - v1) > tolerance * np.maximum(np.abs(v1), np.abs(v2))

    table = pd.DataFrame(out, index=c1.index).reset_index()
    return table[mismatch]


# ----------------------------------------------------------------------
# Two-dir comparison
# ----------------------------------------------------------------------
def compare_two_dirs(
    dir1: str,
    dir2: str,
    out_path: Path,
    tolerance: float = DEFAULT_REL_TOLERANCE,
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
    df1 = load_dir_to_df(dir1)

    print(f"[INFO] Loading directory 2: {dir2}")
    df2 = load_dir_to_df(dir2)

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

    variables = [v for v in SYNCVARLIST if v in c1.columns and v in c2.columns]
    df_out = _mismatch_table(c1, c2, variables, tolerance)

    if df_out.empty:
        print("[INFO] No mismatches found (within tolerance).")
        return

    df_out.to_csv(out_path, index=False)
    print(f"[INFO] Wrote {len(df_out)} mismatching events to {out_path}")


def parse_sync_txt(path: str) -> pd.DataFrame:
    """
    Parse sync txt lines in format:

    run:lumi:event,val1,val2,...

    The first line must name the columns. Values are matched to those names,
    never to a position in SYNCVARLIST, so a variable that one sample does not
    have simply does not appear instead of shifting everything after it.

    Returns
    -------
    pd.DataFrame
        Indexed by (run, luminosityBlock, event).
    """
    # Read all non-empty lines
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        raise RuntimeError(f"No non-empty lines found in {path}")

    header = lines[0].split(",")
    if header[0] != HEADER_KEY_FIELD:
        raise RuntimeError(
            f"{path} has no header line. Regenerate it with the current "
            f"scripts/sync_parquet_dimuon.py; column names are no longer inferred "
            f"from position."
        )
    value_cols = header[1:]
    body = pd.Series(lines[1:], dtype=object)
    line_no = np.arange(1, len(body) + 1)

    n_fields = body.str.count(",").to_numpy() + 1
    too_few = n_fields < 2
    for iline in line_no[too_few]:
        print(f"[WARNING] malformed line {iline} in {path}: too few fields")
    wrong = ~too_few & (n_fields - 1 != len(value_cols))
    if wrong.any():
        iline = int(line_no[wrong][0])
        raise RuntimeError(
            f"line {iline} of {path} has {int(n_fields[wrong][0]) - 1} values but the header "
            f"names {len(value_cols)} columns"
        )
    bad = int(too_few.sum())
    body = body[~too_few]
    line_no = line_no[~too_few]

    records = {}
    # Header-only files leave body empty, where split(expand=True) has no column 0
    if not body.empty:
        fields = body.str.split(",", expand=True)
        key = fields[0]
        key_ok = key.str.fullmatch(r"\s*[+-]?\d+\s*:\s*[+-]?\d+\s*:\s*[+-]?\d+\s*").to_numpy(dtype=bool)
        for iline, raw_key in zip(line_no[~key_ok], key[~key_ok]):
            print(f"[WARNING] failed to parse run:lumi:event on line {iline}: {raw_key}")
        bad += int((~key_ok).sum())

        fields = fields[key_ok]
        if len(fields):
            key_parts = fields[0].str.split(":", expand=True)
            for i, name in enumerate(KEY_VARS):
                records[name] = key_parts[i].str.strip().astype("int64").to_numpy()
            for i, col in enumerate(value_cols, start=1):
                raw = fields[i].str.strip()
                vals = pd.to_numeric(raw, errors="coerce")
                # float() failures became -100.0; a literal "nan" stays NaN
                unparsable = vals.isna() & (raw.str.lower() != "nan")
                records[col] = vals.mask(unparsable, -100.0).to_numpy(dtype="float64")

    if bad:
        print(f"[WARNING] {bad} malformed lines skipped in {path}")

    if not records:
        raise RuntimeError(f"No valid rows parsed from {path}")

    df = pd.DataFrame(records)
    return _with_occurrence_index(df, path)


def compare_two_sync_txt(
    txt1: str,
    txt2: str,
    out_path: Path,
    tolerance: float = DEFAULT_REL_TOLERANCE,
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

    one_sided = sorted(set(c1.columns) ^ set(c2.columns))
    if one_sided:
        print(
            f"[WARNING] Not compared, present in only one file: {one_sided}. "
            f"This is a column set difference, not a value difference."
        )

    df_out = _mismatch_table(c1, c2, vars_to_check, tolerance)

    if df_out.empty:
        print("[INFO] No mismatches found (within tolerance).")
        return

    df_out.to_csv(out_path, index=False)
    print(f"[INFO] Wrote {len(df_out)} mismatching events to {out_path}")
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
        default=DEFAULT_REL_TOLERANCE,
        help=(
            "Relative tolerance for comparing dimuon variables "
            f"(default: {DEFAULT_REL_TOLERANCE:g}); the value is enforced to be positive. "
            "Cutflow counts are always exact."
        ),
    )
    args = parser.parse_args()
    args.tolerance = abs(args.tolerance) # A negative tolerance is meaningless here; treat it as its magnitude.
    return args


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    dirs = args.dirs

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
        df = load_dir_to_df(directory)
        dump_single_dir_sync(df, out_path)

    elif len(dirs) == 2:
        file1, file2 = dirs

        out_path = Path(args.out) if args.out else Path("sync_txt_diff.txt")

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
            compare_two_cutflow_json(
                json1=file1,
                json2=file2,
                out_path=out_path,
            )
            return

        # Otherwise treat as directories (existing behavior)
        dir1, dir2 = file1, file2
        compare_two_dirs(
            dir1=dir1,
            dir2=dir2,
            out_path=out_path,
            tolerance=args.tolerance,
        )

    else:
        raise SystemExit("Please provide one or two directories.")


if __name__ == "__main__":
    main()