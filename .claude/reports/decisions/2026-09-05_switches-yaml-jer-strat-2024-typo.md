# stage-1 crash: `switches.yaml` jer_strat key typo for 2024

**Date:** 2026-09-05
**Type:** Decision/fix (production blocker)
**Scope:** `configs/parameters/switches.yaml`

## Symptom

```
bash run_analysis_pipeline.sh -c configs/datasets/dataset_nanoAODv15_run3.yaml -v 15 \
  -l Run3_nanoAODv15_FilterJets_Aug30_tightPassLepVeto_DefaultjetPt25GeV_PUDNN -y 2024 -m 1 -k -i 0
```
crashed immediately (before any dataset processing) with:
```
omegaconf.errors.ConfigKeyError: Missing key 2024
    full_key: switches.jer_strat.2024
```

## Root cause

`configs/parameters/switches.yaml:366`: `"202-1": -1` under `switches.jer_strat` -- a typo for
`"2024"` (stray `-` in place of `4`). Confirmed as the only such typo in the file (grepped for
similar malformed year-like keys across all of `configs/parameters/*.yaml`; every other switch has
a proper `"2024"` entry).

## Fix

Changed the key `"202-1"` -> `"2024"`, value unchanged (`-1`) -- matches the value already set for
the surrounding Run3 years (2022preEE/2022postEE/2023/2023BPix/2025/2026 are all also `-1` in this
snapshot of the file), so this is a pure key-typo fix, not a value/physics change.

## Verification

Re-ran the user's exact failing command. Confirmed via live log output: config loads cleanly,
connects to the user's already-running Dask Gateway cluster (98 processes/196 threads/0.96 TiB),
and begins processing `data_C` for 2024 (past the point where it previously crashed). Full 27-dataset
run left running in background at the time of writing; not yet confirmed to finish without a
later, unrelated error.
