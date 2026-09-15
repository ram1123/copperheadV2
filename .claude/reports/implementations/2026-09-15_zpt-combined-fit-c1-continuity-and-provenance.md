# Z-pT combined-fit C1 continuity fix, provenance metadata, pixi tooling

Date: 2026-09-15
Type: Implementation (main agent only, no sub-agents — task was well-scoped
enough that delegation wasn't warranted per the coordinate skill's own
guidance not to delegate merely because agents are available)

## Context

Continuation of the same-week Z-pT reweighting stability work
(`src/copperhead/zpt_rewgt/derive/{save_SF_rootFiles,do_f_test,get_polyFit,poly_utils}.py`).
User flagged `2022postEE/njet1`'s goodness-of-fit plot: a visible "kink"
around x≈100 and poorly-fit data points near 5 GeV and 30 GeV.

## Root cause (diagnosed, not guessed)

Wrote a standalone diagnostic script reusing `perform_fits()` directly
against the real 2022postEE njet1 histogram. Found two independent issues in
`make_combined_function_reduced()` / `build_final_piecewise_coefficients()`:

1. **No C1 (slope) continuity at xmax1.** The tail's slope was independently
   fit (`fit_flat_line` on `[xmax1, 200]`) and only nudged by a small
   `delta_tail_slope` MINUIT parameter — unrelated to f1's own slope at
   xmax1. f1's fitted value at xmax1 (≈1.077) didn't match the neighboring
   tail data (≈1.045-1.05); forcing C0-only continuity there produced a
   visible corner and a −4.7σ pull immediately past it.
2. **No independent tilt for f0.** Only `common_shift` (shared, uniform)
   applied to f0's whole domain — f1 already had its own `mid_tilt`. Since
   f1 has more bins/weight in the joint chi2 minimization, MINUIT settled on
   a `common_shift` (~+1.5%) that was net-negative for f0's own,
   already-good independent fit, showing up as systematic ~3σ negative pulls
   across the entire 0-20 GeV region (the "5 GeV" complaint) and comparable
   positive pulls in 20-50 GeV (the "30 GeV" complaint).

## Fix

`get_polyFit.py`: reduced-refit went from 3 free parameters to 4
(`common_shift, low_tilt, mid_tilt, delta_tail_slope`). Tail slope is now
`f1_prime(xmax1) + mid_tilt + delta_tail_slope` (analytic derivative of f1's
own fitted polynomial) instead of an independently-fit flat line's slope —
exact C1 continuity at `delta_tail_slope=0`, with the MINUIT fit still free
to deviate if the data supports it. `f0` gets its own `low_tilt` (pivoted at
xmin1, so it doesn't disturb the existing exact C0 continuity there),
mirroring `f1`'s `mid_tilt`. Error propagation for `tail_slope`/`tail_intercept`
updated to match (diagonal-quadrature approximation, consistent with the
rest of the function's existing rigor level — not full covariance).

After the fix, re-scanned candidate `xmax1` values for this one fit under
the corrected math (the old scan results, from before the fix, no longer
applied) — 95 beat the previously-chosen 100 (full-range max|pull| 3.2 vs
3.8, chi2/ndf 2.3 vs 2.7). `bin_definitions.py`: `2022postEE.njet1`
`[20, 100] → [20, 95]`.

**This is a general code fix** (not scoped to one year/njet), so re-ran step
2 (`get_polyFit.py` only — F-test/step 1 unaffected since orders/ranges are
unchanged elsewhere) for all 7×3=21 (year, njet) combinations to apply it
everywhere. All 21 succeeded; spot-checked several previously-"clean" plots
(e.g. 2022preEE/njet0) for regressions — none found.

## Provenance metadata (separate user ask, same session)

`save_SF_rootFiles.py` (step 0) now writes `provenance.json` per year
(produced_at/by, git commit+dirty via `modules/git_utils.get_git_state`,
`run_label`, resolved `stage1_base_path`, `sample_config`,
`matched_dy_processes`/`missing_dy_processes`, matched `data_*` dirs).
`get_polyFit.py` (step 2) folds that into a `metadata` block per year in the
final `zpt_rewgt_params_*.yaml` (sibling key to `njet_0/1/2` — confirmed
harmless, `copperhead_processor.py` only ever accesses
`wgt_config[f"njet_{n}"]` by exact key), plus its own step2 timestamp/user/
git state.

Naming bug caught by user mid-implementation: `args.dy_sample` is the
*output-directory tag* the analyst chose (`--dy_sample` CLI flag), **not**
the physical DY MC sample — that's `matched_dy_processes`, resolved
independently from `sample_config` and unrelated to the tag string. Renamed
`metadata["dy_sample"] → metadata["dy_sample_label"]` in both scripts.

Second bug found while fixing the first: `OmegaConf.merge(existing, new)` is
additive — renaming/removing a field doesn't remove the old key from
already-written YAML, so the first rename-and-rerun left both `dy_sample`
and `dy_sample_label` present simultaneously. Fixed by explicitly deleting
`existing[year]["metadata"]` before merging, for every year the current run
touches — makes `metadata` a full replace instead of a deep-merge (the
`njet_*` blocks keep their existing merge semantics, unaffected).

**Known gap, not backfilled**: `step0` provenance sub-block is only present
going forward — populating it for years whose step 0 already ran (all 7,
before this code existed) requires rerunning `save_SF_rootFiles.py`, which
takes ~10-12 min/year (~70-80 min total for all 7). Not done this session;
offer to the user if wanted.

## Reusable pixi tooling (user ask: stop re-typing the same heredoc)

Session repeatedly hand-wrote the same ~15-line `cd cvmfs; pixi run -e
default bash -lc '...'` wrapper (WORKDIR/PYTHONPATH/X509_USER_PROXY/
cmsset_default setup) for every non-interactive python invocation, because
the existing `enter_pixi.sh` always ends in `exec bash` (interactive-only —
hangs given a command instead of a TTY). Added two scripts:

- `run_in_pixi.sh` (repo root, sibling to `enter_pixi.sh`): same env setup,
  non-interactive — `./run_in_pixi.sh <env> <command> [args...]`, exits with
  the command's status. Command/args passed via `bash -c '...' bash "$@"`,
  not string-interpolated, so arbitrary quoting in the wrapped command is
  safe.
- `scripts/zpt_loop.sh`: loops `run_analysis_pipeline.sh`'s `zpt_fit*` modes
  over a space-separated year list × njet list in one call (replaces the
  double-`for` heredocs used throughout this session). `-m/-y/-n/-c/-v/-l/-o`
  flags forward directly to `run_analysis_pipeline.sh`; DY sample is not a
  flag (it's hardcoded inside `common_workflow.sh`'s `run_zpt_fit()`, not a
  CLI-configurable value).

Verified end-to-end: `./run_in_pixi.sh default bash scripts/zpt_loop.sh -m
zpt_fit2 -y "2022preEE 2022postEE 2023 2023BPix 2024 2025 2026" -n "0 1 2"
...` — the exact 21-fit metadata-cleanup rerun above — ran clean as one
command.

## Verification

- Diagnostic scripts confirmed the root cause numerically before any code
  change (gap between f0/f1 at xmin1, f1's slope/value at xmax1 vs the
  independent tail fit) — not guessed.
- Re-ran the fixed fit for 2022postEE/njet1, visually confirmed: no more
  visible kink, all pulls within ~±3.1 (was up to −11.7σ across several
  earlier range attempts, −4.7σ at the previously-shipped xmax1=100).
- Ran all 21 (year, njet) step-2 refits twice more (field rename, then
  merge-cleanup) — both batches exit 0, no `FAILED:` markers.
- Confirmed `dy_sample`/`dy_sample_label` duplicate-key bug both reproduced
  and fixed (before: both keys present; after: only `dy_sample_label`).
- Spot-checked a previously-clean plot (2022preEE/njet0) post-fix for
  regressions from the general combined-fit change — none.
- `zpt_summary.md`/`zpt_summary.html` NOT regenerated this pass (chi2/p-value
  shown there is f1's local-fit value only, unaffected by the combined-refit
  change everywhere except the one range that changed, 2022postEE/njet1,
  which was already regenerated into the docs earlier in the session).

## Open / not done

- `step0` provenance backfill for existing years (~70-80 min, not run).
- The goodness-of-fit plot's displayed χ²/ndf and p-value are still f1's
  *local* sub-fit values, not the true full-range combined-fit quality (a
  pre-existing display quirk, noted but out of scope this session — the
  pull panel is the reliable full-range diagnostic).
