#!/usr/bin/env python3
"""
Repair mismatched DEPOT vs Purdue EOS files based on mismatched_adler32.csv.

Input CSV expected columns:
  year,f1_tag,sample,rel_path,adler32_A,adler32_B,full_path_A,full_path_B

Key:
  (year,f1_tag,sample,rel_path) defines the file. We trust full_path_A (DEPOT) as source.

Copy:
  gfal-copy <DEPOT local path> -> root://eos.cms.rcac.purdue.edu//store/user/...

Recheck:
  xrdfs root://eos.cms.rcac.purdue.edu query checksum /store/user/...

Outputs in --outdir:
  - repair_ok.csv
  - repair_still_mismatch.csv
  - repair_copy_failed.csv
  - repair_bad_input.csv
  - repair_summary.txt

Notes:
- Uses EOS metadata checksum query (fast).
- Keep workers modest on EOS (4–8).
"""

import os
import sys
import csv
import time
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Tuple

EOS_HOST = "root://eos.cms.rcac.purdue.edu"


def run_cmd(cmd: List[str], timeout: int) -> Tuple[int, str, str]:
    p = subprocess.run(
        cmd, capture_output=True, text=True, check=False, timeout=timeout
    )
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def eos_query_adler32(store_path: str, timeout: int = 60) -> Optional[str]:
    """
    Query EOS checksum via metadata:
      xrdfs root://eos.cms.rcac.purdue.edu query checksum /store/user/...
    Returns hex string or None.
    """
    try:
        rc, out, err = run_cmd(
            ["xrdfs", EOS_HOST, "query", "checksum", store_path],
            timeout=timeout,
        )
        if rc != 0:
            raise RuntimeError(err or out)

        toks = out.split()
        # expected: "adler32 <hex>"
        if len(toks) >= 2 and toks[0].lower() == "adler32":
            return toks[1].strip()
        return toks[-1].strip() if toks else None
    except Exception as e:
        sys.stderr.write(f"[eos-checksum-fail] {store_path}: {e}\n")
        return None


def compute_local_adler32(file_path: str, timeout: int = 120) -> Optional[str]:
    """Compute adler32 by reading file bytes (xrdadler32)."""
    try:
        rc, out, err = run_cmd(["xrdadler32", file_path], timeout=timeout)
        if rc != 0:
            raise RuntimeError(err or out)
        return out.split()[0]
    except Exception as e:
        sys.stderr.write(f"[local-checksum-fail] {file_path}: {e}\n")
        return None


def store_path_from_full_path_b(full_path_b: str) -> Optional[str]:
    """
    Convert full_path_B to a /store/... path understood by Purdue EOS host.

    Supported inputs:
      - /eos/purdue/store/user/...  -> /store/user/...
      - /store/user/...            -> /store/user/...
      - root://...//store/user/... -> /store/user/...
    """
    s = (full_path_b or "").strip()
    if not s:
        return None

    if s.startswith("root://"):
        # split on '//store/'
        idx = s.find("//store/")
        if idx != -1:
            return s[idx + 1 :]  # keep leading /store/...
        # sometimes root://.../store/...
        idx = s.find("/store/")
        if idx != -1:
            return s[idx:]
        return None

    if s.startswith("/eos/purdue/store/"):
        return s.replace("/eos/purdue", "", 1)  # -> /store/...
    if s.startswith("/store/"):
        return s

    # If your manifest uses /eos/store/... (rare)
    if s.startswith("/eos/store/"):
        return s.replace("/eos", "", 1)

    return None


def gfal_copy_one(
    src: str, dst_store_path: str, timeout: int = 7200, retries: int = 2
) -> bool:
    """
    Copy src(local path) -> Purdue EOS using gfal-copy.

    Destination URL:
      root://eos.cms.rcac.purdue.edu//store/user/...
    """
    dst_url = f"{EOS_HOST}{dst_store_path}"  # EOS_HOST already includes root://...

    # Create destination directory (best-effort)
    dst_dir = str(Path(dst_store_path).parent)
    try:
        run_cmd(["xrdfs", EOS_HOST, "mkdir", "-p", dst_dir], timeout=60)
    except Exception:
        pass

    for attempt in range(1, retries + 2):
        try:
            cmd = [
                "gfal-copy",
                "-f",  # overwrite
                "-t",
                str(timeout),  # long timeout
                src,
                dst_url,
            ]
            rc, out, err = run_cmd(cmd, timeout=timeout + 60)
            if rc == 0:
                return True
            sys.stderr.write(
                f"[copy-fail] attempt {attempt}\n  src={src}\n  dst={dst_url}\n  {err or out}\n"
            )
        except Exception as e:
            sys.stderr.write(
                f"[copy-exception] attempt {attempt} src={src} dst={dst_url}: {e}\n"
            )

        time.sleep(5 * attempt)

    return False


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


def repair_one_row(
    row: Dict[str, str],
    checksum_timeout: int,
    copy_timeout: int,
    copy_retries: int,
    force_recompute_src: bool,
) -> Dict[str, str]:
    src = row.get("full_path_A", "").strip()
    dst_full = row.get("full_path_B", "").strip()

    dst_store = store_path_from_full_path_b(dst_full)
    if not src or not dst_store:
        row["status"] = "bad_input_paths"
        row["expected_adler32"] = ""
        row["eos_adler32_new"] = ""
        row["dst_store_path"] = dst_store or ""
        return row

    # expected checksum from CSV or recompute from src
    expected = row.get("adler32_A", "").strip()
    if force_recompute_src or not expected:
        expected = compute_local_adler32(src, timeout=max(120, checksum_timeout)) or ""
    row["expected_adler32"] = expected
    row["dst_store_path"] = dst_store

    # copy
    ok = gfal_copy_one(src, dst_store, timeout=copy_timeout, retries=copy_retries)
    if not ok:
        row["status"] = "copy_failed"
        row["eos_adler32_new"] = ""
        return row

    # recheck eos checksum (metadata)
    new_eos = eos_query_adler32(dst_store, timeout=checksum_timeout) or ""
    row["eos_adler32_new"] = new_eos

    if expected and new_eos and expected.lower() == new_eos.lower():
        row["status"] = "repair_ok"
    else:
        row["status"] = "still_mismatch"

    return row


def write_csv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Repair mismatched files DEPOT->Purdue EOS using gfal-copy and recheck checksum."
    )
    ap.add_argument(
        "-i", "--input", required=True, help="Path to mismatched_adler32.csv"
    )
    ap.add_argument("-o", "--outdir", default="repair_out", help="Output directory")
    ap.add_argument(
        "-w",
        "--workers",
        type=int,
        default=6,
        help="Parallel workers (4–8 recommended)",
    )
    ap.add_argument(
        "--checksum-timeout",
        type=int,
        default=60,
        help="Seconds for EOS checksum query",
    )
    ap.add_argument(
        "--copy-timeout", type=int, default=7200, help="Seconds for gfal-copy timeout"
    )
    ap.add_argument(
        "--copy-retries", type=int, default=2, help="Retries for gfal-copy failures"
    )
    ap.add_argument(
        "--force-recompute-src",
        action="store_true",
        help="Recompute src checksum instead of using adler32_A",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rows = read_csv_rows(args.input)
    if not rows:
        print(f"No rows found in {args.input}")
        return

    # Output columns: keep original + our new fields
    base_fields = list(rows[0].keys())
    extra = ["dst_store_path", "expected_adler32", "eos_adler32_new", "status"]
    fieldnames = base_fields + [c for c in extra if c not in base_fields]

    ok_rows, still_rows, fail_rows, bad_rows = [], [], [], []

    print(f"EOS host: {EOS_HOST}")
    print(f"Loaded {len(rows)} mismatched entries from: {args.input}")
    print(f"Repair workers={args.workers}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                repair_one_row,
                row,
                args.checksum_timeout,
                args.copy_timeout,
                args.copy_retries,
                args.force_recompute_src,
            )
            for row in rows
        ]

        done = 0
        for fut in as_completed(futs):
            done += 1
            out = fut.result()
            st = out.get("status", "")

            if st == "repair_ok":
                ok_rows.append(out)
            elif st == "still_mismatch":
                still_rows.append(out)
            elif st == "copy_failed":
                fail_rows.append(out)
            else:
                bad_rows.append(out)

            if done % 200 == 0 or done == len(futs):
                print(
                    f"[progress] {done}/{len(futs)} ok={len(ok_rows)} still={len(still_rows)} fail={len(fail_rows)} bad={len(bad_rows)}"
                )

    out_ok = os.path.join(args.outdir, "repair_ok.csv")
    out_still = os.path.join(args.outdir, "repair_still_mismatch.csv")
    out_fail = os.path.join(args.outdir, "repair_copy_failed.csv")
    out_bad = os.path.join(args.outdir, "repair_bad_input.csv")
    out_sum = os.path.join(args.outdir, "repair_summary.txt")

    write_csv(out_ok, ok_rows, fieldnames)
    write_csv(out_still, still_rows, fieldnames)
    write_csv(out_fail, fail_rows, fieldnames)
    write_csv(out_bad, bad_rows, fieldnames)

    with open(out_sum, "w") as f:
        f.write(f"Input: {args.input}\n")
        f.write(f"EOS host: {EOS_HOST}\n")
        f.write(f"Total: {len(rows)}\n\n")
        f.write(f"repair_ok: {len(ok_rows)}\n")
        f.write(f"still_mismatch: {len(still_rows)}\n")
        f.write(f"copy_failed: {len(fail_rows)}\n")
        f.write(f"bad_input_paths: {len(bad_rows)}\n")

    print("Done.")
    print(f"OK:    {out_ok}")
    print(f"Still: {out_still}")
    print(f"Fail:  {out_fail}")
    print(f"Bad:   {out_bad}")
    print(f"Sum:   {out_sum}")


if __name__ == "__main__":
    main()
