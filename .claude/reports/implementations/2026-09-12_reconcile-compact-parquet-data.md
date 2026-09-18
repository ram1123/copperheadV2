# Reconcile scripts/compact_parquet_data.py (Critical finding C1 from Week_June10_MergeWithMaster merge review)

- Date: 2026-09-12
- Type: Implementation
- Status: Resolved (C1 only; other findings from the merge review are untouched)
- Applicable era: N/A (software integration, not a physics selection change)
- NanoAOD campaign: N/A
- Relevant files: `scripts/compact_parquet_data.py`, callers `plotter/validation_plotter_unified.py`, `common_workflow.sh` (`build_compact_cmd`)

## Question

A prior merge review (`Week_June10` rebased onto `main`) found that `scripts/compact_parquet_data.py`
had been destroyed by the rebase — its entire content (1000 lines on `main`, 538 on `Week_June10`,
both individually valid) was replaced by a single stray line of unrelated YAML (`"2025": true`, a
`switches.yaml` conflict-resolution fragment that leaked into this file). User instruction: keep
`main`'s batched/parallelized processing engine and `Week_June10`'s file-size-based compaction sizing
and logging; reconcile them into one working file.

## Evidence

- `git show main:scripts/compact_parquet_data.py` (1000 lines) — added `ensure_compacted_scaled`/
  `add_dnn_score_scaled`/`compact_and_add_dnn_score_scaled` (batch many samples in one Dask
  round-trip instead of per-sample blocking calls), `modules.job_status.JobStatus`-based
  resumability (`--rerun`, done/running/failed markers), and one-hot year-feature synthesis in
  `add_dnn_score` (`year_onehot_features`) needed by `run_stage2_vbf.py`'s year-onehot DNN model.
  Sizing was a hardcoded per-sample-name row-count heuristic (10k for `vbf_powheg*`, 1k for
  `top`/`ttjets`, else 30k).
- `git show Week_June10:scripts/compact_parquet_data.py` (538 lines) — added
  `DEFAULT_TARGET_MB_PER_FILE=250.0`/`_get_file_rows_and_bytes` (derives per-sample row-count budget
  from actual average uncompressed bytes/row instead of a fixed row count), `ensure_compacted`'s
  status-string return convention, elapsed-time logging, `tqdm` progress, `--target-mb-per-file` CLI.
- Both source versions are individually valid Python; the corruption was purely a rebase artifact
  (confirmed identical/individually-parseable on `main` and `Week_June10` before the rebase).

## Findings

- `plotter/validation_plotter_unified.py:483` calls `ensure_compacted(y, process, year_load_path,
  compacted_path_DNN)` with exactly 4 positional args and no client — this is the only external
  cross-module caller of the non-batched `ensure_compacted`, so that function's first-4-positional
  signature had to stay stable.
- `common_workflow.sh`'s `build_compact_cmd` invokes the script's CLI with `-y`, `--input_path`,
  `--log-level`, and optionally `-m`/`--model_tag`/`--add_dnn_score`/`--fix_dimuon_mass`/
  `--save_postfix` — never `--rerun` or `--target-mb-per-file`, so both need safe defaults.
- Dropping `main`'s hardcoded `vbf_powheg`/`top`/`ttjets` row-count special-casing entirely is safe:
  an independent code-review pass grepped the merged tree for those substrings and found no other
  use (only an unrelated pre-existing commented-out debug filter), so the heuristic was purely a
  sizing proxy that file-size-based sizing supersedes, not coupled to a different code path.

## Decision or outcome

Rewrote `scripts/compact_parquet_data.py` combining both sides (see git diff for the full merge):
kept `main`'s `_scaled` batch-processing family and `JobStatus` resumability and one-hot year-feature
support verbatim; replaced `main`'s hardcoded per-sample-name row-count sizing with `Week_June10`'s
byte-size-derived sizing (`_get_file_rows_and_bytes`, `DEFAULT_TARGET_MB_PER_FILE`) in *both*
`ensure_compacted` and `ensure_compacted_scaled`; kept `Week_June10`'s status-string convention and
elapsed-time logging; added `tqdm` only in the `client is None` sequential fallback branches (the
Dask-client branches submit everything then do one `wait()`/`gather()`, with no natural per-item loop
to wrap). `write_compacted_group` and all metadata-fetch helpers were consolidated onto the 3-tuple
`(path, rows, bytes)` shape used by both call sites.

## Verification

- Command: `python3 -c "import ast; ast.parse(open('scripts/compact_parquet_data.py').read())"` →
  parses cleanly.
- Command: `import scripts.compact_parquet_data` and `import plotter.validation_plotter_unified` in
  the project's real pixi `default` env (`/cvmfs/cms-af.opensciencegrid.org/paf/pixi/copperheadV2/
  .pixi/envs/default/bin/python`, `CONDA_OVERRIDE_CUDA=12.4` workaround per `docs/known_issues.md`) →
  both succeed (previously both failed with `SyntaxError`/`ImportError` through this file). `import
  run_plotter` still fails, but now only on the separate, already-known `run_bulk_validation` missing
  symbol (Critical finding C3 from the merge review) — confirms this fix didn't touch or mask that
  issue.
- Command: functional smoke test — wrote synthetic parquet files for two fake samples with very
  different per-row byte footprints (5 vs 50 float columns) and ran `ensure_compacted_scaled(...,
  client=None, target_mb_per_file=0.05)`. Result: sampleA (~49 B/row) got 1076 rows/file target,
  sampleB (~412 B/row) got 127 rows/file — confirms per-sample byte-size derivation works; both
  samples processed in one batched pass; output row totals matched input exactly (8000/6000), zero
  row loss.
- Independent verification pass (`code-reviewer` subagent, read-only): confirmed `write_compacted_group`'s
  3-tuple unpacking, `group_parquet_files`'s tuple-shape-agnostic slicing, the `rows_bytes_by_path`
  dict construction/lookup, CLI/caller compatibility, and `tqdm` placement are all correct. Found two
  findings, both verified by me against the original source and confirmed **pre-existing, not
  introduced by this reconciliation**:
  - `ensure_compacted_scaled`'s "already exists" backfill check (`elif glob.glob(...)`) only checks
    that *some* parquet file exists, not that the expected count is present — a crash-interrupted
    compaction could be silently accepted as done on the next run, permanently dropping rows. Verified
    byte-for-byte identical to `main`'s original code (lines 288-296) — not something introduced here.
    Its sibling `add_dnn_score_scaled` already has the stricter `len(existing) >= len(parquet_files)`
    check, so this is a pre-existing asymmetry within `main`'s own code.
  - The plain `ensure_compacted`'s "already exists" check is even weaker (`os.path.exists`, no file
    check at all) — this is `Week_June10`'s original design, deliberately left as-is since
    `plotter/validation_plotter_unified.py` depends on its exact behavior.
  - Low-severity: potential `ZeroDivisionError` if a parquet file's row-group metadata ever reports
    `total_byte_size == 0` for a non-empty file — pre-existing in `Week_June10`'s original sizing
    formula, not introduced here.

## CMS sources

N/A (software/pipeline-integration task, no physics selection changed).

## Remaining work

- Not fixed (out of scope for this task, flagged for the user to decide): the two "already exists"
  resumability gaps above are real, pre-existing correctness risks (silent row loss on a
  crash-then-resume) inherited unchanged from `main`/`Week_June10` respectively. Recommend a targeted
  follow-up if/when resumability after a crash is exercised in production.
- Other Critical findings from the original merge review (C2 `preprocess_dnn.py` bracket bug, C3
  `validation_plotter_unified.py` missing `run_bulk_validation`, C4 `train_dnn.py` adversarial
  `InMemoryBatchLoader` incompatibility) are still open and untouched by this session.
