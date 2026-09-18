# `run_stats_pipeline_VBF.sh -m 9` run for 2022preEE-2026 + Run3 — 3 real issues found and fixed

**Date:** 2026-09-10
**Branch:** `Week_June10`
**Requested:** run
`bash run_stats_pipeline_VBF.sh -c configs/datasets/dataset_nanoAODv15_run3.yaml -v 15 -l Run3_nanoAODv15_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation -y "2022preEE,2022postEE,2023,2023BPix,2024,2025,2026,Run3" -m 9`,
fix any error, log output. Label `..._OfficialRecomendation`.

## Outcome

**Success.** Final run: exit 0, ~3m43s, produced real significance numbers for every year
including 2025/2026 and the Run3 combination. Summary CSV:
`.../stage3_datacards_Sep09_2026/score_.../vbf_significance_summary_Sep09_2026.csv`:

| year | significance | stat-only |
|---|---|---|
| 2022preEE | 0.255 | 0.387 |
| 2022postEE | 0.465 | 0.615 |
| 2023 | 0.288 | 0.618 |
| 2023BPix | 0.308 | 0.526 |
| 2024 | 0.735 | 1.236 |
| 2025 | 0.635 | 1.316 |
| 2026 | 0.463 | 0.581 |
| Run3 (=2022preEE+2022postEE+2023+2023BPix+2024 only, see below) | 1.271 | 2.187 |

## Issues found and fixed (in order encountered)

1. **`run_stats_pipeline_VBF.sh`'s `usage()` docstring was stale vs its own `case` block.**
   `git log` showed this file is committed/clean (not in-progress edit noise) — HEAD commit
   `4b57a44 "update the running scripts"` (ram1123, Aug 25) renumbered the case block after an
   earlier session's fix (moved `combine_vbf_summary` 9->6, `combine_vbf_impacts` 6->7,
   `combine_vbf_lhscan` 7->8, `combine_vbf_all` 8->9, `combine_vbf_limit` 11->10, and rewrote mode
   `11|vbf_limit` to combine significance+limit instead of rebuilding stage2/stage3) but never
   updated the docstring. Fixed: synced `usage()` to the actual (kept, not reverted) case
   numbering; same fix applied to `CLAUDE.md`'s mode table, which described the pre-renumbering
   layout. **Consequence this mattered for:** under the *current* numbering, `-m 9` means
   `combine_vbf_all` (build card/workspace + real significance fits + collect summary) — a much
   heavier operation than the old docs implied ("collect summary only", which is now `-m 6`).
2. **Date-rollover: `-o`/save_postfix not passed, defaulted to today's date.** First run attempt
   failed immediately: `ERROR: Missing VBF SR/SB datacards for 2022preEE`. Root cause:
   `save_postfix` defaults to `$(date +%b%d_%Y)`; the session's "today" rolled over to 2026-09-10
   between when stage3 was actually built (2026-09-09, dir `stage3_datacards_Sep09_2026`) and when
   this command ran, so the undated invocation looked for a `Sep10_2026` directory that doesn't
   exist. Not a code bug — fixed by adding `-o Sep09_2026` to match the existing stage3 output
   rather than rebuilding stage3 under a new date.
3. **`collect_vbf_significance_summary` / `collect_vbf_limit_summary` silently drop 2025 and 2026.**
   Both functions (`common_workflow.sh`) had a hardcoded `ordered_years=(2022preEE 2022postEE 2023
   2023BPix 2024 Run3)`. The significance fits for 2025/2026 genuinely ran and their
   `HMuMu_13TeV_{2025,2026}_prefitsignificance*.log` files existed on disk, but the summary
   collector's year list never checked for them, so the first full run's CSV silently omitted both
   years with no warning. Fixed: added `2025 2026` to both functions' `ordered_years` (now
   `2022preEE 2022postEE 2023 2023BPix 2024 2025 2026 Run3`). Verified by rerunning mode `6`
   (cheap, log-file-only, no refit) — CSV now includes both years with the correct already-computed
   values. `collect_vbf_limit_summary` (mode 10/11, not exercised by this `-m 9` run) has the
   identical fix applied but not independently re-verified this pass — same code path, same
   evidence, low risk.

## Known limitation, not fixed (flagged, not a bug to silently patch)

`ensure_vbf_card Run3` (`common_workflow.sh`, pre-existing, unrelated to this session's changes)
only combines 2022preEE/2022postEE/2023/2023BPix/2024 into the `Run3` pseudo-year card — 2025 and
2026 are excluded from that combination by design. So even though the `-y` list included
`2025,2026,Run3`, the `Run3` row in the summary is the same 5-year-only combination as before; it
does not represent "all of Run 3 through 2026". This is a real analysis-scope decision (whether/how
to fold in the still-preliminary 2025/2026 data into the Run3 combined result) that needs the
analysis owner's input, not something to change unilaterally.

## Files changed

- `run_stats_pipeline_VBF.sh` — `usage()` docstring synced to the actual case-block numbering.
- `CLAUDE.md` — VBF stats pipeline mode table updated to match.
- `common_workflow.sh` — `ordered_years` in both `collect_vbf_significance_summary` and
  `collect_vbf_limit_summary` extended to include `2025 2026`.

## Verification

- `bash -n` clean on both shell files.
- Full `-m 9` run for all 8 years: exit 0, real Combine `Significance` fits completed for every
  year (verified via `.log` and `higgsCombine*.root` files on disk), no tracebacks (the many
  "Error"/"Analyzing bin errors" hits in the log are `text2workspace.py`'s own MC-stat diagnostic
  printing, not failures).
- Re-ran mode `6` after the `ordered_years` fix and confirmed the regenerated CSV now includes 2025
  and 2026 with the values already computed by the full run (no refit needed).
