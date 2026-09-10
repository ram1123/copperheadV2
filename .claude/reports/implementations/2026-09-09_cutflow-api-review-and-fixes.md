# Cutflow implementation review, fixes, and a self-correction

**Date:** 2026-09-09
**Type:** Implementation
**Scope:** `src/copperhead_processor.py`, `src/stage1/cutflow_io.py`, `src/stage1/runner_adapter.py`,
`scripts/update_sync_references.sh`.

## Background

User asked how to get a cutflow table from stage-1 (`--isCutflow`), then asked to cross-check the
cutflow implementation against the installed coffea version (2026.5.0) for improvements.

## Findings from the review (all confirmed via live introspection of the installed coffea, not
just training knowledge -- corroborated against
https://coffea-hep.readthedocs.io/en/v2026.5.0/api/coffea.analysis_tools.Cutflow.html, which the
user separately linked)

1. **Bug**: `all_required_selections` (`src/copperhead_processor.py`) was in a different order than
   the actual `self.selection.add(...)` call sequence -- `PackedSelection.cutflow(*names)` ANDs
   cumulatively in argument order, not registration order, so several intermediate rows
   (`PV_npvsGood`, `nmuons`, `mm_charge`, `electron_veto`, `HemVeto`, `trigger_match`,
   `leading_muon_pt`) reported efficiencies relative to the wrong preceding population. Fixed:
   reordered to match the real `.add()` sequence, with line-number comments pinning each entry.
2. **Improvement, believed unblocked**: `PackedSelection.cutflow(*names, weights=..., 
   weightsmodifier=...)` is supported since coffea 2025.3.0 (confirmed in 2026.5.0). Initially
   implemented by passing the processor's existing `weights` (`coffea.analysis_tools.Weights`)
   object through, deleting an abandoned ~40-line manual weighted-cutflow loop.
3. **Robustness**: `write_cutflow_outputs` (`src/stage1/cutflow_io.py`) read private
   `cutflow._names`/`_nevcutflow`/`_nevonecut`. Switched to the public `cutflow.result()`
   (`CutflowResult`/`ExtendedCutflowResult` namedtuple) -- required threading a new
   `selection_names` parameter through (`.result()` doesn't carry names, only whoever called
   `.cutflow(*names)` has them), added as a new `self.cutflow_names` attribute on `EventProcessor`.
4. **Minor nits**: fixed a confusing `logger.info(f"...{self.cutflow.print()}")` (logs `None`,
   `.print()` already writes to stdout); routed `"TotalEntries"` through the same
   `available_cuts` membership check as every other cut.

## Bug introduced by Finding 2, caught by testing -- reverted

A synthetic unit test (hand-built `PackedSelection`+`Weights`, exact expected-value verification)
passed cleanly, but that test used a same-length population for both -- it couldn't catch a real
integration problem. Running `scripts/update_sync_references.sh` (2017 sync data) surfaced it
immediately: `IndexError: boolean index did not match indexed array along axis 0; size of axis is
148 but size of corresponding boolean axis is 5420`, in `self.selection.cutflow(*names,
weights=weights)`.

**Root cause**: `src/copperhead_processor.py:1290` does `events = events[event_filter == True]`
right after every selection is registered, reducing the working population to the
already-(partially-)selected subset. `Weights(len(events), ...)` is constructed *after* that line,
so it's sized to the reduced population (148 events in this test chunk) -- while
`self.selection`'s per-cut masks are all still sized to the original, pre-reduction chunk (5420).
There is no valid same-length weights array available anywhere in this pipeline for a
full-population cutflow; the pipeline is deliberately structured to only compute (expensive)
per-event weights for events that already survived the bulk of the selection.

**Fix**: reverted Finding 2 -- `self.selection.cutflow(*required_selections)`, no `weights=`.
Finding 3's `.result()` change needed no further edit: `hasattr(result, "wgtevcutflow")` already
gates the weighted-key output, so it now correctly and silently produces the older
cumulative/individual-only schema.

## Separate, pre-existing bug found and fixed: `scripts/update_sync_references.sh`

Unrelated to the cutflow content changes above (confirmed: it only surfaced once 2017 processing
itself succeeded end-to-end, i.e. after the weights= revert). The script's final `cp` step hardcoded
`cutflow_${sample}_0.json` as the source filename, but the actual filename embeds the input file's
UUID and entry range (`_build_shard_id` in `src/stage1/runner_adapter.py`), e.g.
`cutflow_data_B_3B7EF8F0-...-NanoAOD_0_5420.json`. `.github/workflows/sync-stage1.yml` already
handles this correctly via a `find_cutflow_file()` glob helper; `update_sync_references.sh` never
got the same fix. Ported the same glob-based lookup into the script (destination filename in
`test/reference/` unchanged -- CI's own diff step still expects the plain `_0.json` name).

## Verification

- Synthetic functional test (`PackedSelection`+`Weights`, hand-computed expected cumulative/weighted
  counts) verified the `.result()`-based JSON writer logic including the weighted path -- useful for
  catching an off-by-one in `.result()`'s indexing (`nevcutflow`/`nevonecut` are `len(names)+1` long,
  index 0 = before any cut), but did **not** catch the length-mismatch integration bug above (same
  synthetic population used for both selection and weights, unlike the real pipeline).
- `bash scripts/update_sync_references.sh` (default years: 2017, 2022preEE), run to completion twice
  in immediate succession after each fix, both times exit code 0. Regenerated
  `test/reference/{2017,2022preEE}_cutflow_{data,dy,vbf}_*_0.json` and the corresponding
  `*_eventKinematics.txt` files. Inspected `2017_cutflow_data_B_0.json` directly: `cumulative` is
  monotonically non-increasing through the full reordered sequence (5420 -> ... -> 148 -> 131 -> 0),
  no stray `*_weighted` keys, conditionally-unregistered cuts (`LHE_cut`, `leading_muon_pt`,
  `jet_veto_maps`, absent for this data sample/scenario) cleanly omitted rather than erroring.
- Not yet done: `test/reference/*_eventKinematics.txt` were also regenerated as a side effect of
  running the script (same command touches both); not diffed line-by-line against their prior
  content here since the cutflow-scoped changes shouldn't affect event kinematics at all (no
  selection/physics logic changed, only cutflow bookkeeping) -- worth a final look before committing.

## Lesson

A synthetic/isolated unit test can validate a function's own logic (and did catch a real off-by-one
here) but cannot substitute for an actual integration run when the bug lives in how two different
parts of a *real* pipeline relate to each other (population sizes, in this case) -- the isolated
test's inputs were, by construction, already consistent with each other.

## 2026-09-10: merged full-dataset cutflow, 2025 data eras C-G

Label `Run3_nanoAODv15_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation_Cutflow`
(the `_Cutflow` sibling of the production label -- i.e. a `-z/--isCutflow` stage-1 run).
Merged across all per-chunk `cutflow_*.npz` shards for data_C..data_G with the CLI-rewritten
`scripts/merge_cutflow_npz_file.py` (see 2026-09-10 registry row for that rewrite):

```
python scripts/merge_cutflow_npz_file.py \
  /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation_Cutflow/stage1_output/2025/f1_0/data_[C-G]/0/ \
  -o cutflow_2025_dataCToG_merged.json
```

`data_[C-G]` bracket-glob excludes `data_B`. Output `cutflow_2025_dataCToG_merged.json`.

| cut | individual | (eff) | cumulative | (eff) |
|---|---:|---:|---:|---:|
| initial | 4,215,197,165 | 100.0% | 4,215,197,165 | 100.0% |
| TotalEntries | 4,215,197,165 | 100.0% | 4,215,197,165 | 100.0% |
| lumi_mask | 4,061,432,120 | 96.4% | 4,061,432,120 | 96.4% |
| HLT_filter | 1,902,595,144 | 45.1% | 1,846,338,454 | 43.8% |
| event_quality_flags | 4,170,921,041 | 98.9% | 1,840,812,129 | 43.7% |
| muon_pT_roch | 2,753,867,036 | 65.3% | 1,800,567,461 | 42.7% |
| muon_eta | 3,993,613,387 | 94.7% | 1,798,825,686 | 42.7% |
| muon_id | 3,720,856,190 | 88.3% | 1,764,352,057 | 41.9% |
| muon_isGlobal_or_Tracker | 3,948,263,626 | 93.7% | 1,764,352,057 | 41.9% |
| muon_selection | 2,452,822,831 | 58.2% | 1,751,543,676 | 41.6% |
| muon_iso | 2,922,081,251 | 69.3% | 1,618,372,404 | 38.4% |
| trigger_match | 1,407,385,877 | 33.4% | 1,361,993,701 | 32.3% |
| electron_veto | 4,210,921,965 | 99.9% | 1,360,491,667 | 32.3% |
| HemVeto | 4,215,197,165 | 100.0% | 1,360,491,667 | 32.3% |
| PV_npvsGood | 4,186,259,099 | 99.3% | 1,360,491,667 | 32.3% |
| nmuons | 96,179,407 | 2.3% | 87,334,174 | 2.1% |
| mm_charge | 866,261,688 | 20.6% | 87,072,956 | 2.1% |
| jet_veto_maps | 3,882,453,918 | 92.1% | 83,718,768 | 2.0% |
| dimuon_mass_window_76_106 | 82,116,803 | 1.9% | 74,092,153 | 1.8% |
| h_peak_115_135 | 911,827 | 0.0% | 0 | 0.0% |
| h_sidebands_110_115_135_150 | 791,912 | 0.0% | 0 | 0.0% |
| h_sidebands_106_115_135_150 | 1,365,615 | 0.0% | 0 | 0.0% |

Notes:
- The three trailing `h_*` rows have `cumulative = 0` because they are OR-alternatives applied
  *after* `dimuon_mass_window_76_106` in the argument order -- an event in the 76-106 Z window
  is not in 115-135 or the sidebands, so the running AND collapses to 0. `individual` is the
  standalone count in each window and is the meaningful number there (911,827 in 115-135, etc.).
- Cross-check against 2026-09-10_get-yields-2025-data.md (same label, non-`_Cutflow` production
  run, direct `dimuon_mass` count on the compacted parquet): `jet_veto_maps` cumulative
  **83,718,768** == "total rows on disk" there; `dimuon_mass_window_76_106` cumulative
  **74,092,153** == the `76 < m_mm < 106` count there; `h_peak_115_135` individual **911,827**
  vs the parquet `[115,135)` bin 786,345 -- the cutflow value is *before* the `jet_veto_maps`
  (and full dimuon) reduction that the parquet has, hence larger. All consistent.
- `initial` 4.215e9 is the raw NanoAOD event count read for 2025 data C-G before any cut.
