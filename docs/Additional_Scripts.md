---
title: Additional Scripts
---

# Investigate Pickle File

This script is used to load and inspect Python pickle (*.pkl) files.

**Usage Example**

```bash
python load_pkl.py path/to/file.pkl
```


# Check for missing branches

This script checks for the missing branches in the root files, present in the
directory, and prints the missing branches.

**Usage Example:**

```bash
python scripts/check_missing_branch.py
```

# Processed-lumi tracking (`processedLumis.json`)

Stage-1 keeps a record of exactly which lumi sections were actually run over,
so you can confirm completeness against the golden JSON afterwards instead of
just trusting the input dataset's nominal luminosity.

**How it works**

- For every successfully completed **data** chunk (never MC -- `run`/
  `luminosityBlock` aren't meaningful there), `run_stage1.py` writes a small
  per-chunk shard file next to that chunk's parquet output:
  ```
  <save_path>/stage1_output/<year>/f<fraction>/<dataset>/<idx>/processedlumis_<dataset>_<shard>.json
  ```
  This is built from the chunk's raw *input* events, before any selection --
  a lumi section counts as "processed" once its events were successfully read
  and run over, independent of how many passed analysis cuts. (Implementation:
  `src/stage1/lumi_io.py`, wired into `src/stage1/runner_adapter.py`.)
- Once **every** dataset in that `run_stage1.py` invocation has finished, all
  shards under `<save_path>/stage1_output/<year>` are merged, deduplicated, and
  range-compressed into golden-JSON format (`{"<run>": [[start,end], ...]}`)
  automatically -- no extra flag needed. This also checks the merged result
  against that year's certified lumimask (`configs/parameters/lumi.yaml`) and
  logs a `[processed-lumi report]` coverage summary.
- Written to `<save_path>/stage1_output/<year>/_status/`:

  | File | Contents |
  | --- | --- |
  | `processedLumis.json` | every lumi actually read/processed, regardless of golden-JSON status |
  | `processedLumis_certified.json` | `processed ∩ golden` -- the subset that's also certified; this is what to hand `brilcalc` or eyeball against the golden JSON |
  | `processed_lumis/<dataset>.json` | per-dataset breakdown |
  | `processed_lumis/missing.json` | `golden \ processed` -- certified lumis not processed (if non-empty) |
  | `processed_lumis/unexpected.json` | `processed \ golden` -- processed lumis outside the certified mask, e.g. a PromptReco dataset that includes non-golden runs (if non-empty) |

  Note: the merged files only appear once the *whole* invocation finishes (all
  datasets, not just the one you care about) -- while a run is still in
  progress, only the per-chunk shard files above exist yet.

**Caveat**: a "missing" lumi doesn't automatically mean stage-1 failed on it
-- a specific dataset/primary-dataset can legitimately have no events for some
certified runs/lumis (trigger/PD/detector availability changes across a run).
Cross-check against `job_status.jsonl` before assuming a gap means a
processing failure.

**Manual / standalone use** -- re-report mid-run, after resuming a partially
failed run, or against a different lumimask:

```bash
python scripts/build_processed_lumi_json.py <stage1_output_dir> [--golden-json PATH]
```

Feed the combined file to `brilcalc` for the actual processed luminosity:

```bash
brilcalc lumi -i <stage1_output_dir>/_status/processedLumis_certified.json -u /pb --normtag <normtag.json>
```