---
title: Copy /work to /store (EOS)
---

# Copy /work to /store (EOS)

The script [scripts/copy_work_to_store.py](../scripts/copy_work_to_store.py) copies a local
directory tree (typically under `/work`) to Purdue EOS `/store`, verifying every file with an
Adler-32 checksum and retrying on failure/mismatch. It behaves like an incremental `rsync`: by
default it skips any file that already exists at the destination with a matching checksum, and
only transfers files that are new or changed.

## What it does

1. Recursively scans `--src` and computes `xrdadler32` for every file.
2. For each file, queries the destination's checksum (`xrdfs ... query checksum`). If it already
   matches the source, the file is skipped — nothing is transferred.
3. Otherwise, copies the file with `xrdcp -f` to
   `root://eos.cms.rcac.purdue.edu/<dst-store-path>`.
4. Re-queries the destination checksum and compares it to the source.
5. On mismatch, removes the bad destination file (`xrdfs rm`) and retries the copy.
6. Retries up to `--max-retries` times (default 3) per file; anything still bad after that is
   recorded as failed.
7. Writes a manifest and per-status CSVs locally to `--outdir`, then uploads copies of those logs
   to `<dst>/_transfer_logs/` on EOS so a record of the transfer travels with the data.

## Basic usage

```bash
python scripts/copy_work_to_store.py \
    --src /work/users/<user>/some_output_dir \
    --dst /store/user/<cern-username>/some_output_dir \
    --outdir copy_out
```

`--dst` accepts either the bare `/store/...` path or the POSIX EOS mount form
`/eos/purdue/store/...` — the latter is normalized to `/store/...` internally.

Re-running the same command later only copies files that are new or have changed; unchanged
files are skipped (see [Incremental (rsync-like) behavior](#incremental-rsync-like-behavior)).

## Important options

- `-w, --workers`: parallel workers (default 6; 4-8 recommended so as not to hammer EOS)
- `--max-retries`: max copy attempts per file on checksum mismatch/copy failure (default 3)
- `--copy-timeout`: seconds for the `xrdcp` timeout (default 7200)
- `--checksum-timeout`: seconds for checksum queries (default 60)
- `--force`: always re-copy every file, even if the destination already has a matching checksum
- `--dry-run`: show what would be copied/skipped without transferring anything or uploading logs

## Incremental (rsync-like) behavior

By default the script only transfers files that are missing at the destination or whose content
changed (checksum mismatch). This makes it cheap to re-run after a partial failure or after the
source directory picked up new/updated files — only the delta is copied.

```bash
# Preview what would be copied/skipped, without touching EOS
python scripts/copy_work_to_store.py --src /work/users/<user>/some_dir \
    --dst /store/user/<cern-username>/some_dir --dry-run

# Force a full re-copy and re-verification of every file
python scripts/copy_work_to_store.py --src /work/users/<user>/some_dir \
    --dst /store/user/<cern-username>/some_dir --force
```

## Output

Written to `--outdir` (and mirrored to `<dst>/_transfer_logs/` on EOS after a real run):

- `manifest_src.csv` — every source file with its computed `adler32_src`, destination path,
  `adler32_dst`, `status`, and number of `attempts`
- `copy_ok.csv` — files copied and verified this run
- `copy_skipped.csv` — files already up to date at the destination (skipped)
- `copy_failed.csv` — files that never matched after `--max-retries` attempts
- `copy_summary.txt` — counts for the run

The script exits non-zero if `copy_failed.csv` is non-empty (dry runs included, if any file's
source checksum could not even be computed).

## External tools

The runtime environment must provide:

- `xrdadler32`
- `xrdcp`
- `xrdfs`

A valid VOMS proxy (`voms-proxy-init -voms cms`) or equivalent X509 credential
(`X509_USER_PROXY`) is required to write to EOS over `xrdcp`.

## Notes

- `--dst` must resolve to a `/store/...` path; anything else is rejected.
- Destination directories are created on demand via `xrdfs mkdir -p`.
- `xrdcp` has no wall-clock timeout flag of its own; `--copy-timeout` is enforced through the
  underlying `subprocess` call.
