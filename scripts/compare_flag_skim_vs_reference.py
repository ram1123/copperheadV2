"""Check that one flag-skim (do_flag_skim) stage-1 ntuple can replace the per-scenario ntuples.

For each sample: the flag-skim rows passing a scenario flag (e.g. pass_sel_official) must be exactly
the reference ntuple's events (run, luminosityBlock, event), and every reference column must exist
in the flag skim with the same values. Re-running stage-1 itself gives float noise and the odd jet
crossing a threshold under a JES shift, so the verdict allows <= --max-frac of rows per column beyond --rtol
(the exact per-column counts are still in the report).
It also reports on-disk sizes and the yields of each flag. Memory stays bounded for large samples
(e.g. 2024 DY, 74M rows x 2288 columns): events are matched on the keys only, then the columns are
compared in batches of --col-batch, aligned by key.

How to run (from the repo root, default pixi env):
    ./run_in_pixi.sh default python scripts/compare_flag_skim_vs_reference.py \
        --new <flag-skim stage1_output/<year>> [--year-samples <s1,s2> | (all reference samples)] \
        --ref <name>=<flag column>=<reference stage1_output/<year>> [--ref ...] [--out <report.json>] \
        [--col-batch 40] [--rtol 1e-3] [--max-frac 1e-3] [--workers 16]

Example:
    ./run_in_pixi.sh default python scripts/compare_flag_skim_vs_reference.py \
        --new /work/projects/hmm/shar1172/hmm_ntuples/flag_skim_test/Run3_nanoAODv12_FlagSkimTest_Oct28/stage1_output/2022preEE \
        --year-samples data_D,vbf_powheg_dipole \
        --ref Oct26_official=pass_sel_official=/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterEvents_Oct26_tightPassLepVeto_OfficialRecomendation_Systematics_LumiSplit/stage1_output/2022preEE \
        --out validation/flag_skim/compare_2022preEE.json
"""
import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

KEYS = ["run", "luminosityBlock", "event"]
WORKERS = 16


def parquet_files(stage1_dir, sample):
    files = sorted(glob.glob(os.path.join(stage1_dir, "compacted", sample, "*", "*.parquet")))
    if not files:
        files = sorted(glob.glob(os.path.join(stage1_dir, "f1_0", sample, "*", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet for {sample} under {stage1_dir}")
    return files


def disk_bytes(files):
    return sum(os.path.getsize(f) for f in files)


def column_bytes(files):
    """Compressed on-disk bytes per column, summed over files and row groups."""
    out = {}
    for f in files:
        md = pq.ParquetFile(f).metadata
        for rg in range(md.num_row_groups):
            for c in range(md.num_columns):
                col = md.row_group(rg).column(c)
                name = col.path_in_schema.split(".")[0]
                out[name] = out.get(name, 0) + col.total_compressed_size
    return out


def read(files, columns):
    # file order (hence row order) is fixed, so row positions from one read index every later read
    with ThreadPoolExecutor(WORKERS) as ex:
        tables = list(ex.map(lambda f: pq.read_table(f, columns=columns), files))
    return pa.concat_tables(tables).to_pandas()


def compare_values(a, b, rtol):
    """(n entries differing at all, n beyond rtol, max |a-b|, max |a-b|/|a|); NaN == NaN, None as NaN."""
    if a.dtype == object or b.dtype == object:
        a, b = a.astype("float64"), b.astype("float64")
    if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
        a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
        diff = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        if not diff.any():
            return 0, 0, 0.0, 0.0
        ad = np.abs(a[diff] - b[diff])
        rd = ad / np.maximum(np.abs(a[diff]), 1e-300)
        rd = np.where(np.isnan(rd), np.inf, rd)  # NaN on one side only
        return int(diff.sum()), int((rd > rtol).sum()), float(np.nanmax(np.where(np.isnan(ad), np.inf, ad))), float(rd.max())
    n = int(np.sum(np.asarray(a) != np.asarray(b)))
    return n, n, float("nan"), float("nan")


def compare_sample(new_dir, sample, refs, col_batch, rtol, max_frac):
    t0 = time.time()
    new_files = parquet_files(new_dir, sample)
    new_cols = pq.read_schema(new_files[0]).names
    new_colbytes = column_bytes(new_files)
    res = {"new": {"files": len(new_files), "bytes": disk_bytes(new_files), "n_columns": len(new_cols)}, "refs": {}}
    flag_cols = [c for c in new_cols if c.startswith(("pass_", "pair_", "nMuons_", "mulist", "nLooseVeto", "nElectrons_",
                                                      "extraMu_", "extraEle_", "muIDIso_sf_", "primary_is_"))]
    res["new"]["flag_columns"] = len(flag_cols)
    res["new"]["flag_bytes"] = sum(new_colbytes.get(c, 0) for c in flag_cols)

    # yields of every boolean flag (+ weighted for MC)
    bool_cols = [c for c in flag_cols if c.startswith("pass_") or "_OS" in c or "_trigMatch_" in c or "isPrimary" in c
                 or c == "primary_is_official_pair"]
    new_small = read(new_files, KEYS + bool_cols + (["wgt_nominal"] if "wgt_nominal" in new_cols else []))
    res["new"]["n_rows"] = len(new_small)
    res["new"]["yields"] = {c: int(new_small[c].sum()) for c in bool_cols}
    if "wgt_nominal" in new_small:
        res["new"]["weighted_yields"] = {c: float(new_small.loc[new_small[c], "wgt_nominal"].sum()) for c in bool_cols}

    for name, flag, ref_dir in refs:
        ref_files = parquet_files(ref_dir, sample)
        ref_cols = pq.read_schema(ref_files[0]).names
        missing = [c for c in ref_cols if c not in new_cols]
        common = [c for c in ref_cols if c in new_cols and c not in KEYS]
        ref_k = read(ref_files, KEYS)
        ref_k["_ri"] = np.arange(len(ref_k))
        sel = new_small[flag].to_numpy()
        new_k = new_small.loc[sel, KEYS].copy()
        new_k["_ni"] = np.flatnonzero(sel)
        r = {"flag": flag, "ref_files": len(ref_files), "ref_bytes": disk_bytes(ref_files), "ref_rows": len(ref_k),
             "new_rows_passing_flag": len(new_k), "missing_columns_in_new": missing,
             "duplicate_keys_ref": int(ref_k.duplicated(KEYS).sum()), "duplicate_keys_new": int(new_k.duplicated(KEYS).sum())}
        m = ref_k.merge(new_k, on=KEYS, how="outer", indicator=True)
        r["only_in_ref"] = int((m["_merge"] == "left_only").sum())
        r["only_in_new"] = int((m["_merge"] == "right_only").sum())
        both = m[m["_merge"] == "both"]
        r["matched_rows"] = len(both)
        ri, ni = both["_ri"].to_numpy(dtype=np.int64), both["_ni"].to_numpy(dtype=np.int64)
        del m, both, ref_k, new_k

        diffs, beyond = {}, {}
        for i in range(0, len(common), col_batch):
            batch = common[i:i + col_batch]
            ref_b = read(ref_files, batch)
            new_b = read(new_files, batch)
            for c in batch:
                n, nb, mad, mrd = compare_values(ref_b[c].iloc[ri].reset_index(drop=True), new_b[c].iloc[ni].reset_index(drop=True), rtol)
                if n:
                    diffs[c] = {"n_diff": n, "n_beyond_rtol": nb, "max_abs": mad, "max_rel": mrd}
                    if nb > max_frac * len(ri):
                        beyond[c] = nb
            del ref_b, new_b
        r["columns_compared"] = len(common)
        r["columns_with_differences"] = diffs
        r["columns_beyond_tolerance"] = beyond
        if flag == "pass_sel_lepIPVeto" and "pair_mediumID_IP_isPrimary" in new_small:
            r["rows_where_lepIPVeto_pair_is_not_primary"] = int((~new_small.loc[sel, "pair_mediumID_IP_isPrimary"]).sum())
        r["verdict"] = "PASS" if (r["only_in_ref"] == r["only_in_new"] == 0 and not missing and not beyond
                                  and r["duplicate_keys_ref"] == r["duplicate_keys_new"] == 0) else "FAIL"
        res["refs"][name] = r
        print(f"[{sample}] {name}: {r['verdict']} rows ref={r['ref_rows']} new={r['new_rows_passing_flag']} "
              f"only_ref={r['only_in_ref']} only_new={r['only_in_new']} missing={len(missing)} "
              f"cols={len(common)} diff={len(diffs)} beyond_tol={len(beyond)} ({time.time() - t0:.0f} s)", flush=True)
    return res


def main():
    global WORKERS
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--new", required=True)
    p.add_argument("--year-samples", default="", help="comma-separated; default = all samples of the first --ref")
    p.add_argument("--ref", action="append", required=True, help="<name>=<flag column>=<reference stage1_output/<year>>")
    p.add_argument("--out", default=None)
    p.add_argument("--col-batch", type=int, default=40)
    p.add_argument("--rtol", type=float, default=1e-3, help="relative difference above which a value counts as a mismatch")
    p.add_argument("--max-frac", type=float, default=1e-3, help="allowed fraction of matched rows per column beyond --rtol")
    p.add_argument("--workers", type=int, default=WORKERS, help="threads reading parquet files")
    a = p.parse_args()
    WORKERS = a.workers
    refs = [tuple(r.split("=", 2)) for r in a.ref]
    if a.year_samples:
        samples = a.year_samples.split(",")
    else:
        d = os.path.join(refs[0][2], "compacted")
        samples = sorted(s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s)))
    report = {}
    for s in samples:
        report[s] = compare_sample(a.new, s, refs, a.col_batch, a.rtol, a.max_frac)
        if a.out:  # rewrite after every sample so a long run leaves partial results
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            with open(a.out, "w") as f:
                json.dump(report, f, indent=1)
    print(json.dumps({s: {n: r["verdict"] for n, r in v["refs"].items()} for s, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
