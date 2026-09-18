# Fix M1/M2 and pre-existing broken/orphaned files from the merge review

- Date: 2026-09-12
- Type: Implementation
- Status: Resolved
- Applicable era: N/A (software integration; no physics selection changed)
- NanoAOD campaign: N/A
- Relevant files: `src/lib/histogram/plotting.py`,
  `Run3_Resolution_Scripts/config/ntuples_path.py`,
  `configs/MVA/MVA_subCat_calculation/sigEffScore.py`,
  `plotter/stitchingValidation.py`, `src/lib/categorizer.py`,
  `validation/DY_VBFFilter_production/VBF_filter_comparison.py`,
  `plotter/get2Dplots.py`, `plotter/readParquetUsingRDataFrame.py`,
  `docs/known_issues.md`

## Question

Follow-up fixes requested after the `Week_June10_MergeWithMaster` merge review
and the C1-C4 fixes: (1) M1 — the `.txt` summary output should have "all
information one needs" (left to my judgment); (2) M2 — confirm `do_zpt` stays
`false` for Run3 years; (3)/(4) confirm `do_qgl_wgt`/`do_met_xy_correction`
match `main`; (5) confirm `test/reference/switches.yaml` matches `main`; (6)/(7)
fix the 5 pre-existing broken `.py` files and the 2 broken `applyRegionCatCuts`
imports (both previously flagged as pre-existing/out-of-scope for the merge
review), and document all of it under `docs/`.

## Evidence

- M1: `src/lib/histogram/plotting.py`'s current `.txt` writer (from
  `Week_June10`) has yields/fractions, signal separation power, and Data/MC
  agreement (integrated ratio, per-bin ratio stats, chi2/ndof) but dropped
  `main`'s binning-edges line and full bin-by-bin table. `main`'s version also
  had a `Data/MC ratio (Sum ratio_hist)` scalar metric that `Week_June10`'s own
  code comment documents as always-NaN (any bin with zero data/MC poisons the
  whole `np.sum`) -- deliberately replaced by the nan-aware stats already
  present.
- M2/3/4: diffed current `configs/parameters/switches.yaml` against `main`'s
  version directly -- `do_qgl_wgt` and `do_met_xy_correction` already match
  `main` exactly for every year `main` defines (2025/2026 are `Week_June10`
  additions, not conflicts); `do_zpt` is `false` for Run3 in the current file
  vs. `true` on `main` -- this is `Week_June10`'s deliberate change, already
  correctly preserved through the merge.
- M5: diffed `test/reference/switches.yaml` against `main`'s version -- the
  only difference is a `do_use_pu_dnn_score` block `Week_June10` added, with an
  explicit comment explaining it forces the switch `false` in the CI reference
  regardless of production `switches.yaml`, since the `ci` pixi env has no
  pytorch and the PU-DNN model files aren't committed. Production
  `switches.yaml` also currently has this `false` for every year, so dropping
  the block would be a no-op today but removes a real future safety net.
- Full-repo `ast.parse` sweep (238 tracked `.py` files) confirmed the 5 broken
  files were the only remaining syntax errors, and separately confirmed (before
  touching them) that all 5 were identically broken on `base`/`main`/
  `Week_June10` -- pre-existing, not introduced by this merge.
- `plotter/get2Dplots.py`/`plotter/readParquetUsingRDataFrame.py` both import
  `applyRegionCatCuts` from `plotter.validation_plotter_unified`, which has
  never defined that name on any of `base`/`main`/`Week_June10`. The real
  implementation is `modules.selection.applyRegionCatCuts`, whose signature
  (`category`, `region_name`, `process`, `variation`, `do_vbf_filter_study`,
  `jj_eta_region`, `njets_selection`, `year`) already matches both scripts'
  keyword names except `njets` (should be `njets_selection`) and the missing
  required `variation` argument.

## Decision or outcome

- **M1**: added back `main`'s `Binning (N bins): [...]` line and full
  bin-by-bin table (Data/MC-sample-groups/signal/ratio per bin), but reusing
  `Week_June10`'s already-fixed, NaN-safe `ratio_hist` for the per-bin ratio
  column instead of reintroducing `main`'s broken always-NaN scalar metric.
  The `.txt` now has every piece of information either side ever wrote, minus
  the one line that was a documented bug.
- **M2/3/4**: no code change -- confirmed via direct diff against `main` that
  these are already correctly resolved exactly as instructed.
- **M5**: asked the user directly given the real (if currently latent) CI-safety
  tradeoff; user chose to keep the `do_use_pu_dnn_score` safety-net block. No
  code change.
- **6 pre-existing broken files**: fixed each to at least parse/import cleanly,
  documented in `docs/known_issues.md` under a new section, being explicit
  about which fixes were purely mechanical (dedent/re-indent) vs. which files
  remain non-functional beyond parsing (didn't invent missing business logic
  for already-abandoned/dead code -- `sigEffScore.py`, `src/lib/categorizer.py`).
  See the docs entry for the file-by-file breakdown.
- **7 broken `applyRegionCatCuts` imports**: repointed both scripts at
  `modules.selection.applyRegionCatCuts`, fixed the `njets`->`njets_selection`
  keyword mismatch, added the required `variation="nominal"` argument (matching
  every other caller in the repo). Documented in `docs/known_issues.md`.

## Verification

- M1: functional smoke test -- called `plotDataMC_compare` with synthetic
  data/bkg/sig histograms and inspected the resulting `.txt`; confirmed every
  section present (yields, signal separation power, Data/MC agreement,
  binning-edges line, full bin-by-bin table with correct NaN-safe ratios).
- 6/7: `ast.parse` on each fixed file individually, then a full-repo sweep of
  all 238 tracked `.py` files -- 0 syntax errors remain (down from 7 at the
  start of this session). `git diff --check` on the whole tree clean of
  anything but pre-existing trailing-whitespace warnings.
- `plotter/get2Dplots.py`/`plotter/readParquetUsingRDataFrame.py`: beyond
  `ast.parse`, used `inspect.signature(applyRegionCatCuts).bind(...)` with each
  script's exact call-site keyword arguments against the real function -- both
  bind successfully with no `TypeError`, confirming the fix is call-compatible,
  not just import-resolvable.

## CMS sources

N/A (software-integration/doc task; no physics selection invented or changed --
explicitly avoided completing `sigEffScore.py`'s or `categorizer.py`'s
unfinished algorithms rather than guessing at intended physics logic, and
`VBF_filter_comparison.py`'s incomplete GenPart "hard process" flag line was
removed rather than completed with an invented flag expression).

## Remaining work

- None from this batch. All items the user listed are closed.
- Still open from earlier reports: the `ensure_compacted_scaled` resumability
  gap (High, `implementations/2026-09-12_reconcile-compact-parquet-data.md`)
  and the empty-process-list silent-warning gap (Medium,
  `implementations/2026-09-12_fix-c2-c3-c4-merge-review.md`) were flagged, not
  fixed, pending user decision.
