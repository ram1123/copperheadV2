# Merge-safety review: Week_June10_MergeWithMaster vs main / Week_June10

- Date: 2026-09-12
- Type: Investigation
- Status: Open (4 Critical findings; C1 subsequently fixed, see
  `implementations/2026-09-12_reconcile-compact-parquet-data.md`)
- Applicable era: N/A (software integration + a few physics-config switches)
- NanoAOD campaign: N/A
- Relevant files: repo `/work/users/shar1172/CopperHead_Temp_MergeMineWithMaster_11Sep2026`
  (branches `main`, `Week_June10`, `Week_June10_MergeWithMaster`)

## Question

Is `Week_June10_MergeWithMaster` safe to merge into `main`? Review-only: no edits, no commits.

## Evidence

- No merge commit exists: `Week_June10_MergeWithMaster` is `main`'s tip (`be053d09`) with
  `Week_June10`'s 51 unique commits rebased on top (`HEAD` has a single parent). `main` is an
  ancestor of the branch; the original `Week_June10` tip (`88d51bf`) is not (rewritten SHAs).
  3 now-empty `Week_June10` commits (CI/pixi CUDA-scoping fixes already on `main`) were auto-dropped
  by the rebase — verified harmless (identical content across all three refs).
- Common ancestor of `main`/`Week_June10`: `5d73ffaa37675a1b4765e0ba3e8271d4f538f228`. 19 files were
  touched by both sides since then — the conflict zone. Reviewed via 3-way diff
  (`base->main`, `base->Week_June10`, `main->final`, `Week_June10->final`) per file, cross-checked
  against real imports in the project's pixi `default`/`ci` envs (not just `ast.parse`).
- `git grep` for conflict markers: clean. `git diff --check`: clean of merge artifacts (only
  pre-existing trailing-whitespace warnings). Full-repo `ast.parse` sweep of all 238 tracked `.py`
  files found 7 syntax errors: 2 confirmed rebase-introduced (below), 5 confirmed pre-existing
  identically on base/main/Week_June10 (`Run3_Resolution_Scripts/config/ntuples_path.py`,
  `configs/MVA/MVA_subCat_calculation/sigEffScore.py`, `plotter/stitchingValidation.py`,
  `src/lib/categorizer.py` — an abandoned stub, `validation/DY_VBFFilter_production/
  VBF_filter_comparison.py`) — out of scope.

## Findings

**Critical, rebase-introduced (all confirmed via real `import`, not just static parse):**

- **C1** `scripts/compact_parquet_data.py` — entire content (1000 lines on `main` / 538 on
  `Week_June10`, both individually valid) replaced by one stray line `"2025": true` (a
  `switches.yaml`-conflict fragment that leaked in). Breaks `-m 1a/compact` pipeline stage and
  cascades into `plotter/validation_plotter_unified.py` (imports `ensure_compacted`). **Fixed
  2026-09-12**, see implementation report.
- **C2** `MVA_training/VBF_run3/preprocess_dnn.py` — `main`'s new `--files-per-chunk` argparse
  block and `Week_June10`'s new `--jj-eta-region` block got spliced together, losing a closing
  paren/comma (`'(' was never closed`). Feature content of both sides is otherwise complete and
  non-overlapping once the bracket is fixed. Not yet fixed.
- **C3** `plotter/validation_plotter_unified.py` — `main`'s entire coffea-Runner rewrite
  (`ValidationHistProcessor`, `run_bulk_validation`, `JobStatus`-based resumability) was dropped;
  merged file is byte-identical to `Week_June10`'s tip, no conflict recorded. Breaks
  `run_plotter.py:7,100` (`ImportError: cannot import name 'run_bulk_validation'`). Not yet fixed.
- **C4** `MVA_training/VBF_run3/train_dnn.py` — `main`'s `--use_adversarial` batch loop does
  `xvb = batch[3].to(device)`; `Week_June10`'s new `InMemoryBatchLoader` (`make_dataloader`, called
  unconditionally) always yields a 3-tuple and never passes `ds.xvar` through. No git conflict
  flagged this since both diffs applied cleanly — a logical incompatibility between two
  independently-correct features. `IndexError` on first adversarial-training batch. Not yet fixed.

**Medium:**

- `configs/parameters/switches.yaml`: `do_zpt` disabled (`false`) for all Run3 years — `true` on
  `main` alone and at the common ancestor; `Week_June10` set it `false` and the merge correctly kept
  `Week_June10`'s value (no merge defect), but this needs physics sign-off before the next
  stage-2/3 production run (turns off Z-pT reweighting for Run3 DY MC).
- `src/lib/histogram/plotting.py`: `main`'s per-bin "values table"/binning-edges dump in the `.txt`
  summary output was dropped (same drop-one-side pattern as C3, but no consumer reads those specific
  lines — cosmetic, not a crash).

**Clean (both sides' intended changes fully and correctly reconciled, no findings):**
`modules/selection.py`, `src/corrections/jet.py`, `src/copperhead_processor.py`,
`configs/dnn_run3_vbf.yaml`, `run_stage2_vbf.py`, `configs/samples/samples.yaml`,
`stage3/make_datacards.py`, `scripts/sync_parquet_dimuon.py`, `test/reference/switches.yaml`
(its apparent "staleness" vs. production `switches.yaml` is by design — CI intentionally overwrites
production switches with this frozen snapshot before the sync test).

## Decision or outcome

Verdict at review time: **NOT READY TO MERGE**. Software: FAIL. Physics: PARTIALLY VALIDATED (core
stage-1 physics path reconciled cleanly at the code level; no cutflow rerun attempted, so no
numeric regression proven or ruled out). C1 has since been fixed (see implementation report);
C2/C3/C4 and the `do_zpt` sign-off remain open.

## Verification

- Command: `git log --oneline`, `git merge-base`, `git merge-base --is-ancestor`, `git cherry` —
  established rebase structure.
- Command: `git diff --name-only <base> main|Week_June10` + `comm -12` — found the 19-file conflict
  zone.
- Command: full-repo `ast.parse` sweep (238 `.py` files) — 7 syntax errors, triaged above.
- Command: `git grep` for conflict markers, `git diff --check` — both clean.
- Command: real `import` via the project's pixi `default`/`ci` envs
  (`/cvmfs/cms-af.opensciencegrid.org/paf/pixi/copperheadV2/.pixi/envs/{default,ci}/bin/python`,
  `CONDA_OVERRIDE_CUDA=12.4` workaround) — confirmed live `SyntaxError`/`ImportError` for C1/C2/C3;
  confirmed clean import for `src.copperhead_processor`, `modules.selection`,
  `src.corrections.jet`, `run_stage2_vbf`, `MVA_training.VBF_run3.train_dnn`,
  `stage3.make_datacards`.
- Two parallel specialist sub-agent reviews (`physics-reviewer`, `code-reviewer`) covered the
  remaining conflict-zone files via prepared 3-way diffs; their two most load-bearing claims
  (C4's `train_dnn.py` incompatibility, C3's `run_plotter.py` import chain) were independently
  re-verified directly by the main session before being reported as confirmed.

## CMS sources

N/A for the merge-mechanics findings. `do_zpt` (Medium finding) is analysis-specific and needs the
Run3 Z-pT-correction-readiness decision from whoever owns that switch, not a CMS/POG recommendation.

## Remaining work

- C2, C3, C4 not yet fixed.
- `do_zpt`=false for Run3 and the pre-existing `do_met_xy_correction` comment/value mismatch on
  `main` need explicit sign-off/confirmation, not a code fix.
- No numeric/cutflow validation was performed (would need identical inputs/definitions per repo
  convention); CI's stage-1 sync test would not have caught any of C1/C3/C4 regardless (none of
  those files are in its coverage).
