# VBF Statistical Pipeline — `run_stats_pipeline_VBF.sh`

Responsible: **analysis-specific driver around Combine** — the blinding convention
(Asimov-only, expected limit) is common HEP practice; the mode numbering and year
combination scope are this repo's own.

## Stored sources

| # | Source | Location | Snapshot / verified |
|---|--------|----------|---------------------|
| C1 | Driver script | `run_stats_pipeline_VBF.sh` | 2026‑09‑17 |
| C2 | Helper functions | `common_workflow.sh` (`run_vbf_significance`, `run_vbf_impacts`, `run_vbf_lhscan`, `collect_vbf_significance_summary`, `collect_vbf_limit_summary`, `ensure_vbf_card`) | 2026‑09‑17 |
| C3 | Mode numbering, verified against the live script (superseded an earlier, different numbering — see the registry) | project `CLAUDE.md` "Common commands" section; commit `4b57a44` | 2026‑09‑17 |
| C4 | A real production run + 3 bug fixes found while running it | `.claude/reports/registry.md`, 2026‑09‑10 decision entry | 2026‑09‑10 |

Classification tags: **[Analysis-specific]**, **[Implementation]**, **[Verify]**.

Requires the `combine` pixi environment (`./enter_pixi.sh combine`) — a different
ROOT pin than `default`. Runs after `stage3` has produced datacards
(`template-fit.md` / `parametric-fit.md`).

---

## 1. Modes (`-m`, C1/C3 numbering)

| Mode | Action |
|------|--------|
| 4 | Build VBF card/workspace. `-y` accepts a single year or a pseudo-year (`Run3`/`Run2`/`Run2Run3`) to combine already-built per-year cards into one. |
| 5 | Significance. |
| 6 | Collect significance summaries only. |
| 7 | Impacts. |
| 8 | Likelihood scan. |
| 9 | Card/workspace + significance + collect summary. |
| 10 | Expected 95% CL limit (`AsymptoticLimits --run blind`) + collect limit summary. |
| 11 | Mode 9 + mode 10 combined (card, workspace, significance, summary, expected limit, limit summary) — does **not** rebuild stage2/stage3; run those separately first. |
| 12 | Combine the already-built `jj_both_central` + `jj_non_central` cards for `-y` into one two-channel card (exact partition of the VBF selection, per `event-selection`'s §6 — recovers the unrestricted phase space without an `all`-phase-space stage2/3 rerun), then significance + summary + limit + limit summary on the combined card. Requires mode 9/11 already run under **both** `JJ_ETA_REGION` values first. |
| 13 | Same jj-region combination as 12, but runs impacts instead. |

`-y` accepts a comma-separated list; each mode's steps run once **per year** in that
list — pass a single pseudo-year (e.g. `Run3`) rather than a list of individual years
to get one combined result instead of several per-year ones.

Typical:

```bash
./enter_pixi.sh combine
bash run_stats_pipeline_VBF.sh -m 11 -y Run3 -l label_for_ntuple
```

---

## 2. Blinding — never fits real data

This analysis is blinded. Concretely:

- **Impacts** (modes 7/13) run **twice per year**: Asimov `r=1` (signal injected) and
  Asimov `r=0` (background-only) — never an "observed" scenario.
- **Limits** (modes 10/11/12) are the **expected** limit only
  (`AsymptoticLimits --run blind`), never an observed limit.
- Any request for an "observed" significance, limit, or impact from this pipeline
  should be flagged — this pipeline structurally does not produce one.

---

## 3. Known findings (C4)

From the 2026‑09‑10 mode‑9 production run across all years (2022preEE–2026 + `Run3`):

- `usage()`'s docstring had drifted from the actually-committed mode numbering
  (`4b57a44`) — synced, kept the renumbering (did not revert it).
- `-o` (output-date postfix) defaults to *today* if not passed — a run spanning a
  date rollover without `-o` will look for stage3 output under the wrong date and
  fail with "Missing VBF SR/SB datacards."
- `collect_vbf_significance_summary`/`collect_vbf_limit_summary` (C2) had a
  **hardcoded year list** that omitted 2025/2026 — their logs existed on disk but
  were silently dropped from the summary CSV. Fixed; **re-check this hardcoded list
  any time a new year is added to the analysis.**
- `ensure_vbf_card Run3` **by design** only combines 2022preEE–2024 into the `Run3`
  pseudo-year card — 2025/2026 are excluded from that combined row. This is a
  documented analysis-scope decision, not a bug; don't "fix" it without confirming
  the scope has actually changed.

---

## 4. Review checklist

1. Correct mode for the request (card-only vs full chain vs jj-region combination —
   modes 12/13 need both `JJ_ETA_REGION` values already run).
2. `-y` uses a pseudo-year for a combined result, not a comma-list of individual
   years (which would produce several per-year results instead).
3. `-o` matches the actual stage3 output date/postfix, especially across a date
   rollover.
4. Any "observed" framing is rejected — this pipeline only ever produces Asimov
   impacts and expected (blind) limits.
5. If a new year was added to the analysis, `collect_vbf_significance_summary`/
   `collect_vbf_limit_summary`'s year list and `ensure_vbf_card`'s `Run3` scope were
   both re-checked.

## Last verified

- Local source review: 2026‑09‑17
