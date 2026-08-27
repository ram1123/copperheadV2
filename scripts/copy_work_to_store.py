#!/usr/bin/env python3
"""
Copy a local directory tree (e.g. under /work) to Purdue EOS /store, with
adler32 checksum verification and retry-on-mismatch.

Workflow:
  1. Walk --src recursively; compute xrdadler32 for every file -> manifest.csv
  2. xrdcp each file to root://eos.cms.rcac.purdue.edu/<dst-store-path>
  3. Query the EOS checksum (xrdfs ... query checksum) and compare to the
     source checksum computed in step 1
  4. On mismatch: xrdfs rm the bad destination file, then retry the copy
  5. Up to --max-retries attempts (default 3) per file; anything still
     mismatched/failed after that is recorded in copy_failed.csv

Outputs, written locally to --outdir and then also xrdcp'd to
<dst>/_transfer_logs/ on EOS so the logs travel with the data:
  - manifest_src.csv      (rel_path, full_path_src, size_bytes, adler32_src)
  - copy_ok.csv
  - copy_failed.csv
  - copy_summary.txt

By default the copy is incremental (rsync-like): a file already at the destination with a
matching adler32 checksum is skipped. Use --force to always re-copy, or --dry-run to preview
what would be copied/skipped without transferring anything.

Example:

  python scripts/copy_work_to_store.py \
      --src /work/projects/hmm/shar1172/hmm_ntuples \
      --dst /eos/purdue/store/user/rasharma/hmm/reducedNtuples \
      --outdir copy_out
"""

import os
import sys
import csv
import time
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Tuple, Dict

EOS_HOST = "root://eos.cms.rcac.purdue.edu"
TRANSFER_LOGS_DIRNAME = "_transfer_logs"


def run_cmd(cmd: List[str], timeout: int) -> Tuple[int, str, str]:
    p = subprocess.run(
        cmd, capture_output=True, text=True, check=False, timeout=timeout
    )
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def compute_local_adler32(file_path: str, timeout: int = 120) -> Optional[str]:
    """Compute adler32 of a local file via xrdadler32."""
    try:
        rc, out, err = run_cmd(["xrdadler32", file_path], timeout=timeout)
        if rc != 0:
            raise RuntimeError(err or out)
        return out.split()[0]
    except Exception as e:
        sys.stderr.write(f"[local-checksum-fail] {file_path}: {e}\n")
        return None


def eos_query_adler32(store_path: str, timeout: int = 60) -> Optional[str]:
    """Query EOS checksum metadata: xrdfs <host> query checksum <path>."""
    try:
        rc, out, err = run_cmd(
            ["xrdfs", EOS_HOST, "query", "checksum", store_path], timeout=timeout
        )
        if rc != 0:
            raise RuntimeError(err or out)
        toks = out.split()
        if len(toks) >= 2 and toks[0].lower() == "adler32":
            return toks[1].strip()
        return toks[-1].strip() if toks else None
    except Exception as e:
        sys.stderr.write(f"[eos-checksum-fail] {store_path}: {e}\n")
        return None


def eos_rm(store_path: str, timeout: int = 60) -> None:
    """Best-effort removal of a bad/partial destination file."""
    try:
        run_cmd(["xrdfs", EOS_HOST, "rm", store_path], timeout=timeout)
    except Exception as e:
        sys.stderr.write(f"[eos-rm-fail] {store_path}: {e}\n")


def eos_mkdir_p(store_dir: str, timeout: int = 60) -> None:
    try:
        run_cmd(["xrdfs", EOS_HOST, "mkdir", "-p", store_dir], timeout=timeout)
    except Exception:
        pass


def xrdcp_copy_once(src: str, dst_url: str, timeout: int) -> bool:
    """xrdcp has no wall-clock timeout flag; enforced via the subprocess timeout."""
    cmd = ["xrdcp", "-f", "-N", src, dst_url]
    rc, out, err = run_cmd(cmd, timeout=timeout + 60)
    if rc != 0:
        sys.stderr.write(f"[copy-fail] src={src} dst={dst_url}\n  {err or out}\n")
    return rc == 0


def build_manifest(src_root: str) -> List[Dict[str, str]]:
    """Recursively list files under src_root and compute their adler32."""
    src_root = os.path.abspath(src_root)
    files = [p for p in Path(src_root).rglob("*") if p.is_file()]

    rows: List[Dict[str, str]] = []
    for p in sorted(files):
        rel = str(p.relative_to(src_root))
        rows.append(
            {
                "rel_path": rel,
                "full_path_src": str(p),
                "size_bytes": str(p.stat().st_size),
                "adler32_src": "",
            }
        )
    return rows


def copy_and_verify_one(
    row: Dict[str, str],
    dst_prefix: str,
    max_retries: int,
    copy_timeout: int,
    checksum_timeout: int,
    skip_existing: bool = True,
    dry_run: bool = False,
) -> Dict[str, str]:
    src = row["full_path_src"]
    rel = row["rel_path"]
    dst_store = dst_prefix.rstrip("/") + "/" + rel
    # XRootD URLs need a double slash before an absolute path (root://host//path),
    # otherwise the path is parsed as relative and the server rejects it.
    dst_url = f"{EOS_HOST}/{dst_store}"

    row["dst_store_path"] = dst_store

    expected = row.get("adler32_src") or compute_local_adler32(src, timeout=max(120, checksum_timeout))
    row["adler32_src"] = expected or ""
    if not expected:
        row["status"] = "src_checksum_failed"
        row["attempts"] = "0"
        row["adler32_dst"] = ""
        return row

    # rsync-like short-circuit: if the destination already exists with a matching
    # checksum, skip the transfer entirely (missing or changed files still copy).
    if skip_existing:
        existing = eos_query_adler32(dst_store, timeout=checksum_timeout)
        if existing and existing.lower() == expected.lower():
            row["status"] = "would_skip_unchanged" if dry_run else "skipped_unchanged"
            row["attempts"] = "0"
            row["adler32_dst"] = existing
            return row

    if dry_run:
        row["status"] = "would_copy"
        row["attempts"] = "0"
        row["adler32_dst"] = ""
        return row

    eos_mkdir_p(str(Path(dst_store).parent))

    status = "copy_failed"
    got = ""
    attempt = 0
    for attempt in range(1, max_retries + 1):
        ok = xrdcp_copy_once(src, dst_url, timeout=copy_timeout)
        if not ok:
            time.sleep(3 * attempt)
            continue

        got = eos_query_adler32(dst_store, timeout=checksum_timeout) or ""
        if got and got.lower() == expected.lower():
            status = "ok"
            break

        status = "checksum_mismatch"
        eos_rm(dst_store, timeout=checksum_timeout)
        time.sleep(3 * attempt)

    row["status"] = status
    row["attempts"] = str(attempt)
    row["adler32_dst"] = got
    return row


def write_csv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def upload_logs_to_eos(local_paths: List[str], dst_prefix: str, timeout: int = 120) -> None:
    """Best-effort xrdcp of local log/manifest files into <dst_prefix>/_transfer_logs/."""
    logs_store_dir = dst_prefix.rstrip("/") + "/" + TRANSFER_LOGS_DIRNAME
    eos_mkdir_p(logs_store_dir)

    for local_path in local_paths:
        if not os.path.exists(local_path):
            continue
        dst_store = logs_store_dir + "/" + os.path.basename(local_path)
        dst_url = f"{EOS_HOST}/{dst_store}"
        ok = xrdcp_copy_once(local_path, dst_url, timeout=timeout)
        if ok:
            print(f"[log-upload] {local_path} -> {dst_store}")
        else:
            sys.stderr.write(f"[log-upload-fail] {local_path} -> {dst_store}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Copy a local directory to Purdue EOS /store with adler32 verification and retries."
    )
    ap.add_argument("-s", "--src", required=True, help="Local source directory (e.g. under /work)")
    ap.add_argument(
        "-d",
        "--dst",
        required=True,
        help="Destination EOS store path prefix, e.g. /store/user/<cern-username>/some_dir",
    )
    ap.add_argument("-o", "--outdir", default="copy_out", help="Output directory for manifest/logs")
    ap.add_argument("-w", "--workers", type=int, default=6, help="Parallel workers (4-8 recommended)")
    ap.add_argument("--max-retries", type=int, default=3, help="Max copy attempts per file")
    ap.add_argument("--copy-timeout", type=int, default=7200, help="Seconds for xrdcp timeout")
    ap.add_argument("--checksum-timeout", type=int, default=60, help="Seconds for checksum queries")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Always re-copy every file, even if the destination already has a matching checksum "
        "(default: rsync-like — skip files that already exist at the destination unchanged)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied/skipped without transferring anything or uploading logs",
    )
    args = ap.parse_args()
    skip_existing = not args.force

    # Accept either the POSIX EOS mount or the bare /store path (same file, per
    # modules/xrootd_utils.py's normalize_paths) and use the /store form throughout.
    if args.dst.startswith("/eos/purdue/store/"):
        args.dst = args.dst[len("/eos/purdue"):]
    if not args.dst.startswith("/store/"):
        sys.exit(f"--dst must start with /store/ or /eos/purdue/store/ (got {args.dst!r})")

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Scanning source: {args.src}")
    rows = build_manifest(args.src)
    if not rows:
        print("No files found under source directory; nothing to do.")
        return
    print(f"Found {len(rows)} files.")

    manifest_path = os.path.join(args.outdir, "manifest_src.csv")

    print(f"EOS host: {EOS_HOST}")
    print(f"Destination prefix: {args.dst}")
    print(f"Workers={args.workers}  max_retries={args.max_retries}  skip_existing={skip_existing}"
          f"{'  DRY-RUN' if args.dry_run else ''}")

    ok_rows, skipped_rows, fail_rows = [], [], []
    FAIL_STATUSES = {"copy_failed", "checksum_mismatch", "src_checksum_failed"}
    SKIP_STATUSES = {"skipped_unchanged", "would_skip_unchanged"}
    fieldnames = [
        "rel_path",
        "full_path_src",
        "size_bytes",
        "adler32_src",
        "dst_store_path",
        "adler32_dst",
        "status",
        "attempts",
    ]

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                copy_and_verify_one,
                row,
                args.dst,
                args.max_retries,
                args.copy_timeout,
                args.checksum_timeout,
                skip_existing,
                args.dry_run,
            )
            for row in rows
        ]

        done = 0
        for fut in as_completed(futs):
            done += 1
            out = fut.result()
            st = out.get("status", "")
            if st in FAIL_STATUSES:
                fail_rows.append(out)
            elif st in SKIP_STATUSES:
                skipped_rows.append(out)
            else:
                ok_rows.append(out)

            if done % 50 == 0 or done == len(futs):
                print(
                    f"[progress] {done}/{len(futs)} ok={len(ok_rows)} "
                    f"skipped={len(skipped_rows)} failed={len(fail_rows)}"
                )

    # manifest reflects the (now checksum-populated) source listing
    ok_path = os.path.join(args.outdir, "copy_ok.csv")
    skipped_path = os.path.join(args.outdir, "copy_skipped.csv")
    fail_path = os.path.join(args.outdir, "copy_failed.csv")
    write_csv(manifest_path, ok_rows + skipped_rows + fail_rows, fieldnames)
    write_csv(ok_path, ok_rows, fieldnames)
    write_csv(skipped_path, skipped_rows, fieldnames)
    write_csv(fail_path, fail_rows, fieldnames)

    summary_path = os.path.join(args.outdir, "copy_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Source: {args.src}\n")
        f.write(f"Destination prefix: {args.dst}\n")
        f.write(f"EOS host: {EOS_HOST}\n")
        f.write(f"Dry run: {args.dry_run}\n")
        f.write(f"Total files: {len(rows)}\n")
        f.write(f"Copied OK: {len(ok_rows)}\n")
        f.write(f"Skipped (already up to date): {len(skipped_rows)}\n")
        f.write(f"Failed: {len(fail_rows)}\n")

    print("Done.")
    print(f"Manifest: {manifest_path}")
    print(f"OK:       {ok_path} ({len(ok_rows)})")
    print(f"Skipped:  {skipped_path} ({len(skipped_rows)})")
    print(f"Failed:   {fail_path} ({len(fail_rows)})")
    print(f"Summary:  {summary_path}")

    if args.dry_run:
        print("Dry run: no files were transferred and no logs were uploaded to EOS.")
        if fail_rows:
            sys.exit(1)
        return

    logs_store_dir = args.dst.rstrip("/") + "/" + TRANSFER_LOGS_DIRNAME
    print(f"Uploading logs to EOS: {logs_store_dir}")
    upload_logs_to_eos(
        [manifest_path, ok_path, skipped_path, fail_path, summary_path],
        args.dst,
        timeout=args.checksum_timeout,
    )

    if fail_rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
