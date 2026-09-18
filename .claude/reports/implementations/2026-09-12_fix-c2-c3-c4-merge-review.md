# Fix C2/C3/C4 from the Week_June10_MergeWithMaster merge review

- Date: 2026-09-12
- Type: Implementation
- Status: Resolved (all three fixed and independently verified); one Medium
  observability gap and one Low pre-existing-dead-code item flagged, not fixed
- Applicable era: N/A (software integration + ML training pipeline)
- NanoAOD campaign: N/A
- Relevant files: `MVA_training/VBF_run3/preprocess_dnn.py`,
  `plotter/validation_plotter_unified.py`, `MVA_training/VBF_run3/train_dnn.py`

## Question

Fix the three remaining Critical findings from
`investigations/2026-09-12_merge-review-week-june10-mergewithmaster.md`, per the
user's explicit per-finding instructions:
- C2: "you got the correct informations to fix it" (i.e. apply the diagnosed fix).
- C3: "Keep like main. Which MC samples it is reading is better in the
  Week_June10 branch."
- C4: "Need systematic-adversarial training like main. For better
  GPU-utilization we need to use InMemoryBatchLoader from Week_June10."

## Evidence

- C2: confirmed (again) that both source branches individually parse fine and
  the corruption was purely the rebase splicing `main`'s `--files-per-chunk`
  block into `Week_June10`'s new `--jj-eta-region` block without a closing
  paren/comma.
- C3: `main`'s `group_dict` (a ~130-line hardcoded Python literal) had zero
  entries for 2025/2026 (grep-confirmed); `Week_June10`'s
  `build_group_dict_for_year(year, sample_config_path)` reads the same
  group->processes mapping from `configs/samples/samples.yaml` via
  `modules.sample_config.get_bkg_sig_dicts`/`get_data_processes` (the same
  helpers `scripts/get_yields.py` uses), which already covers 2025/2026.
- C4: `main`'s adversarial batch loop does `xvb = batch[3].to(device)  # [B, V, F]`
  when `adv is not None`; `Week_June10`'s `InMemoryBatchLoader` (added for GPU
  utilization, see `implementations/2026-09-11_vectorized-in-memory-dataloader.md`)
  always yielded a plain 3-tuple, never touching `ParquetDataset`'s
  `xvar`/`var_slot_idx`/`var_feat_idx`/`n_variations` (only populated on
  `ds_train`, only when `variation_spec` is passed). No git conflict ever
  flagged this since both diffs applied cleanly — a pure logical incompatibility.

## Decision or outcome

- **C2**: closed `--files-per-chunk`'s `add_argument(...)` call (added missing
  `),`) before `--jj-eta-region`'s block starts.
- **C3**: rewrote the file as `main`'s full architecture (`ValidationHistProcessor`,
  `run_bulk_validation`, `resolve_year_context`, `_run_validation_scope`, etc.),
  but replaced the hardcoded `group_dict`/`parseGroupProcesses` with
  `Week_June10`'s `build_group_dict_for_year` (ported, driven by a new fixed
  `ALL_GROUP_NAMES` list instead of a separately hardcoded tuple, since the Hist
  category axis must stay static/year-independent). Threaded a
  `sample_config_path`/`sample_config` parameter through
  `resolve_year_context`/`_run_validation_scope`/`run_bulk_validation`, defaulted
  to `"configs/samples/samples.yaml"` so `run_plotter.py`'s existing call (which
  passes no such arg) keeps working unchanged; the standalone `__main__` CLI path
  passes `args.sample_config` (already provided for free by
  `cli.common_argparser.build_common_parser()`).
- **C4**: gave `InMemoryBatchLoader` optional `xvar`/`var_slot_idx`/`var_feat_idx`/
  `n_variations` params; when `n_variations > 0`, `__iter__` builds
  `xvb = xb.unsqueeze(1).repeat(1, n_variations, 1)` (real, independently-writable
  tensor, not an `.expand()` view) then does one vectorized fancy-index assignment
  per batch (`xvb[:, var_slot_idx, var_feat_idx] = xvar[idx]`), yielding a 4-tuple;
  otherwise yields the original 3-tuple unchanged. `make_dataloader()` passes
  these through only when `getattr(ds, "n_variations", 0) > 0` (true only for
  `ds_train` when `--use_adversarial` is on; `ds_val`/`ds_eval` never carry them).

## Verification

- C2: `ast.parse` clean; real pixi-env module exec succeeds; built the actual
  argparser and parsed `--files-per-chunk 10 --jj-eta-region jj_both_central
  --include-systematic-variations` together — both values round-trip correctly.
- C3: `ast.parse` clean; real pixi-env import succeeds (`run_bulk_validation` and
  `build_group_dict_for_year` both present); `import run_plotter` succeeds
  end-to-end (previously failed on the missing `run_bulk_validation` symbol);
  called `build_group_dict_for_year` against the real `samples.yaml` for
  2018/2022preEE/2025 — all three resolve complete, sensible per-year group
  dicts with zero missing groups (2025 in particular, which `main`'s hardcoded
  dict never covered at all).
- C4: bit-exact numerical test — built a synthetic `ParquetDataset` with a real
  multi-variation `variation_spec`, compared the new vectorized batch
  construction against calling the original per-row `ParquetDataset.__getitem__`
  for the same indices: `np.allclose` True, max abs diff `0.0`. Full-epoch
  iteration covers every row exactly once. A dataset built with
  `variation_spec=None` still yields plain 3-tuples (non-adversarial path
  provably untouched). End-to-end toy `nn.Linear` forward pass replicating the
  exact `xvb[:, lo:hi, :]` chunk/reshape/forward pattern from
  `train_one_fold`'s adversarial branch ran without shape errors.
- Independent `code-reviewer` verification pass (read-only) confirmed all of the
  above plus additionally checked: no other argparse-block damage near the C2
  splice site; `ALL_GROUP_NAMES` matches every group name actually live in
  `run_plotter.py`'s/the CLI's defaults (no missing group); no leftover bare
  `group_dict` references anywhere in the C3 file; `.repeat()` (not `.expand()`)
  correctly gives independent storage safe for the in-place write; pin_memory
  ordering (write-then-pin) is correct; `xvar`'s dtype is forced to match `x`'s
  dtype at `ParquetDataset.__init__` time so no dtype mismatch is possible; no
  other caller of `InMemoryBatchLoader`/`make_dataloader` exists anywhere in the
  repo that could break from the new keyword-only params.
- I independently resolved the review's one open suspicion myself: grepped
  `main`'s original `group_dict` literal (saved from the earlier merge review) —
  it never had `DY_MINNLO`/`DY_AMCATNLO` as top-level keys (only `DATA, DY,
  DYVBF, EWK, TOP, VV, ggH, VBF`), so `generate_combo_plots`'s inverse-variance
  DY_MINNLO/DY_AMCATNLO merge code was already dead/vestigial in `main` before
  this rewrite -- not a statistical-combination regression introduced here.

## CMS sources

N/A (software/ML-pipeline task; no physics selection changed).

## Remaining work

- **Not fixed, flagged for the user** (Medium, `plotter/validation_plotter_unified.py`
  `resolve_year_context` around the `bkg_sample_upper in group_dict_year` /
  `.extend(...)` lines): a background group that resolves to an empty process
  list for a given year (a deliberate `samples.yaml` empty override, or a
  fallback that happens to be empty) is silently absent from the plot with no
  warning logged -- unlike a genuinely-unknown group name, which does warn.
  This is an inherent, expected consequence of switching from a hardcoded
  literal (which raised `KeyError` on any unconfigured year) to samples.yaml's
  own non-raising fallback semantics (the same convention `scripts/get_yields.py`
  already uses) -- not a wrong-physics-numbers bug, just reduced loudness on a
  real misconfiguration. Not fixed since it wasn't part of the three requested
  fixes; a one-line `logger.warning` after the `.extend()` would close it if
  wanted.
- **Not fixed, informational only** (Low, same file): `configs/samples/samples.yaml`
  defines a `VVV` background group not present in `ALL_GROUP_NAMES` or in
  `run_plotter.py`'s default `background_samples` list -- pre-existing (already
  excluded before this rewrite), not a regression, just worth knowing if `VVV`
  MC should eventually be plotted.
- Other Critical items from the original merge review are already closed by
  this session's two implementation passes (C1 in
  `implementations/2026-09-12_reconcile-compact-parquet-data.md`, C2/C3/C4 here).
  The Medium `do_zpt`/Run3 physics sign-off from the original review is still
  open and untouched.
