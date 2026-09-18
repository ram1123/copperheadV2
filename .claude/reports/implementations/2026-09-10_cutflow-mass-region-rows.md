# Cutflow: mass-region rows read as `cumulative=0` — diagnosis + minimal fix

**Date:** 2026-09-10
**Type:** Implementation (coordinated: physics-reviewer + main agent)
**Scope:** `src/copperhead_processor.py` (cutflow block only)
**Trigger:** user saw `h_sidebands_110_115_135_150` / `h_sidebands_106_115_135_150` / `h_peak_115_135`
with `cumulative=0` in the merged 2025 data C–G cutflow and suspected a processor bug.

## Diagnosis — NOT an event-processing bug

- `event_filter` (`src/copperhead_processor.py:761-1267`) contains **no dimuon-mass cut**. `events =
  events[event_filter == True]` (`:1290`) therefore keeps the full mass spectrum, and stage-1 parquet
  is intentionally un-skimmed on mass. Verified empirically earlier (2026-09-10_get-yields-2025-data.md):
  683,717 sideband events, 786,345 h-peak events on disk for 2025 data C–G.
- This is load-bearing (physics-reviewer): Z-pT reweighting is derived in 70–110 straight from stage-1
  output (`src/copperhead/zpt_rewgt/derive/save_SF_rootFiles.py:111` → `filterRegion("z-peak")`,
  `modules/selection.py:46`); the Z peak is also the mass-scale/resolution calibration and HEM-veto
  study sample. **`event_filter` must not gain a mass cut.**

## The actual defect — cutflow display

- `all_required_selections` (`:2481-2505`) lists the four mass windows **consecutively** at the tail.
  `PackedSelection.cutflow(*names)` ANDs cumulatively in argument order. The windows are mutually
  exclusive, so `76-106 ∧ 115-135 = {}` → `cumulative=0` for every row after the first. Expected math,
  not a bug — but misleading.
- The `individual` (`nevonecut`) counts for those rows were computed against the **raw pre-`event_filter`
  chunk**, so `h_peak_115_135` individual read 911,827 vs the true selected count 786,345, and also
  counted same-sign / ≥3-muon events (the window masks had no `nmuons==2` / `mm_charge` requirement).

## Fix applied (option "Minimal + document"; user chose to KEEP the 106-edge sideband)

`src/copperhead_processor.py`, "Cutflow dimuon mass window" block:

- Added `event_filter_bool = (event_filter == True)` and AND-ed it into all four registered masks
  (`dimuon_mass_window_76_106`, `h_peak_115_135`, `h_sidebands_110_115_135_150`,
  `h_sidebands_106_115_135_150`). Now each row's `individual` = "events passing the FULL sequential
  selection **and** landing in that window":
  - `dimuon_mass_window_76_106` individual → 74,092,153 (was 82,116,803)
  - `h_peak_115_135` individual → 786,345 (was 911,827)
  - `h_sidebands_110_115_135_150` individual → 683,717
  - `h_sidebands_106_115_135_150` individual → 1,192,577 (683,717 + the 106–110 slice 508,860)
- Block comment + `all_required_selections` comment now state explicitly: these four are
  mutually-exclusive alternative **regions**, not sequential cuts; **read `individual`, not
  `cumulative`**; `cumulative=0` after the first is by construction.
- No change to `event_filter`, `cutflow_io.py`, `runner_adapter.py`, or `merge_cutflow_npz_file.py`.

## Not changed (deliberately, per user + reviewer)

- `h_sidebands_106_115_135_150` kept as a wider-sideband cross-check row (used nowhere else in the
  repo; only the `_110_` variant matches `modules/selection.py:48` and the parquet region columns).
- Cutflow Z window `76-106` left narrower than the downstream `z_peak` `70-110`
  (`src/copperhead_processor.py:2281`); boundary-style inconsistency (`>76 & <106` vs `>=115 & <135`)
  left as-is. Both cosmetic, now noted in the block comment.
- The cutflow uses a padded-pair mass from the corrected `muons` collection (BS/Rochester/FSR
  applied) — same construction as the downstream `dimuon`, so no correction mismatch; only the
  pre- vs post-`event_filter` timing differs (physics-reviewer confirmed).

## Verification

- `ast.parse` on `src/copperhead_processor.py` — OK.
- Integration: ran `run_analysis_pipeline.sh -c configs/datasets/sync_dataset_nanoAODv12.yaml -v 12
  -l label_output -y 2017 -m 1 -z -S test/output` (only the stage-1 step — the `cp`-into-`test/reference/`
  steps of `update_sync_references.sh` were NOT run; `test/reference/` untouched). Processed 59 real
  `data_B` chunks through the modified cutflow code, exit 0 per chunk, no traceback; then terminated
  manually (the `data_B` stream resolves to the full `SingleMuon_Run2017B` test dataset, not a small
  sync subset — a pre-existing sync-harness quirk, see below). Results on the 59 chunks:
  - Schema intact: every `cutflow_*.json` still has `{cumulative, individual}` int pairs per row.
  - `dimuon_mass_window_76_106`: `individual == cumulative` on every chunk (aggregate 203,564 / 203,564)
    — correct: it's now `mask & event_filter`, and it's the first mass row after `jet_veto_maps`, so
    its standalone count already equals the fully-chained count.
  - `h_peak_115_135` / `h_sidebands_110_115_135_150` / `h_sidebands_106_115_135_150`: **non-zero**
    `individual` (aggregate 2,119 / 1,871 / 3,255; the `_106_` one largest, as expected since it
    also holds the 106-110 slice), `cumulative = 0` (documented, by construction).
  - Pre-fix, these `individual` values would have been larger (counted vs the raw chunk, including
    same-sign / >=3-muon events); the merged 2025 C-G numbers move e.g. `h_sidebands_110_...`
    791,912 -> ~683,717 and `h_peak_115_135` 911,827 -> ~786,345, matching the parquet histogram
    in 2026-09-10_get-yields-2025-data.md.
- `test/reference/*_cutflow_*.json` will change on the next `scripts/update_sync_references.sh`
  (the three mass-region rows' `individual` values shift down; their `cumulative` stays 0). CI sync
  diff must be refreshed when this is intentionally committed.

## Unrelated observation (NOT caused by this change, NOT investigated)

During the test, some `data_B` chunks showed `lumi_mask: pass = 0` cascading every later cumulative
row to 0 (including the mass rows). `lumi_mask` is registered at `src/copperhead_processor.py:808`
and is untouched here; other `data_B` chunks in the same run had a normal non-zero `lumi_mask`. It
looks like a golden-JSON / run-coverage gap for parts of the local `SingleMuon_Run2017B` test
dataset, plus the fact that `run_analysis_pipeline.sh -y 2017` iterates data streams `B C D E F`
while `sync_dataset_nanoAODv12.yaml` only defines 2017 `data_D` (FIXME comment there:
"replaced data_B -> data_D"). Flagged for the user; out of scope for the cutflow-display fix.
