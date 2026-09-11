# Vectorized in-memory DataLoader for `train_dnn.py`/`hpo_optuna.py`

**Date:** 2026-09-11
**Status:** Done, verified with a correctness test suite + a realistic-scale benchmark

## Motivation

User reported the `jj_non_central` HPO run taking far longer per trial than the earlier
`jj_both_central` run (~126 min/trial avg vs ~11-16 min/trial). Live profiling of the running
process (PID 507714, confirmed attached to the GPU via `nvidia-smi --query-compute-apps`)
showed **8% GPU utilization, 214 MiB / 15360 MiB GPU memory used, ~1.9 CPU cores** — a
classic CPU-bound-dataloading signature, not GPU-compute-bound. `jj_non_central`'s fold0
train set is 3,114,701 rows vs `jj_both_central`'s 476,066 (6.5x), which only partly explains
an 8-11x wall-time increase.

## Root cause

`ParquetDataset.__getitem__` (pre-change) fetched **one row at a time** via a Python-level
function call (`torch.from_numpy(self.x[idx])`), which `torch.utils.data.DataLoader` then
collated into batches. HPO deliberately runs with `num_workers=0` (`hpo_optuna.py`, with its
own comment explaining multiprocessing had caused hangs on this batch system for long HPO
loops), so every epoch did O(n_rows) individual Python calls, single-threaded, with zero
overlap with GPU compute — the classic "starved GPU" pattern. For `jj_non_central` that's
~3.1M calls/epoch vs ~476k for `jj_both_central`, disproportionately amplifying the true
bottleneck.

## Fix

`MVA_training/VBF_run3/train_dnn.py`: replaced the `ParquetDataset` + `torch.utils.data.DataLoader`
combo with a new `InMemoryBatchLoader` class. Since the whole fold is already materialized in
RAM (`_load_fold_data_cached`, `@lru_cache`d across HPO trials), there is no I/O left for
worker processes to overlap — the fix is to stop doing per-row Python calls at all:

- `InMemoryBatchLoader.__init__` converts `x`/`y`/`w` numpy arrays to `torch.Tensor` once
  (zero-copy `torch.from_numpy`).
- `__iter__` draws one `torch.randperm(n)` (shuffle) or `torch.arange(n)` (no shuffle) per
  epoch, then produces each batch with a single vectorized fancy-index
  (`self.x[idx]`/`self.y[idx]`/`self.w[idx]`) — no per-row Python loop.
- `pin_memory`, if requested, is applied **per yielded batch**, not to the whole tensor —
  advanced/fancy indexing always returns a fresh, unpinned tensor, so pinning the
  dataset-sized parent tensor once would never have propagated to what actually gets
  returned per batch (caught by the test suite below, initially implemented incorrectly).
- `make_dataloader()` now builds this instead of a `DataLoader`; same function signature
  (`num_workers`/`prefetch_factor` accepted but unused/`del`eted, with a comment explaining
  why) so `train_one_fold()`'s call sites needed zero changes.
- `evaluate()`'s type hint updated (`DataLoader` -> `InMemoryBatchLoader`); unused
  `DataLoader` import removed.

This is a full replacement, not a conditional fallback — every current caller's dataset is a
fully in-memory `ParquetDataset` (there's exactly one ingestion path), so there was no
remaining use case for `torch.utils.data.DataLoader`'s worker-process machinery.

## Verification

**Correctness** (`test_inmemory_loader.py`, run inside the pixi `default` env against a
synthetic fold):
- `shuffle=False`: iterated batches concatenate back to the *exact* original row order and
  values (bit-for-bit match against the source DataFrame).
- `shuffle=True`: each epoch's batches are a full permutation of the original label multiset
  (no duplicates, no misses — verified by sorted-array comparison) and two consecutive
  epochs produce different orders (confirms reshuffling actually happens, not accidentally
  reusing a cached permutation).
- `len(loader)` matches `ceil(n / batch_size)`.
- `pin_memory=True` actually produces pinned tensors per batch (`xb.is_pinned()`) — this
  caught the parent-tensor-pinning bug described above during development.
- Edge cases: `batch_size > n` (single batch of size n), `n` not a multiple of `batch_size`
  (last partial batch correctly included, matching `drop_last=False`).
- `ruff check` on the changed file: one pre-existing finding (`F841` unused `weights` at
  `train_dnn.py:1149`, in the main training loop's loss computation) — confirmed via
  `git diff` hunk boundaries to be **outside** this change entirely; not touched, flagged to
  the user as an FYI only (it's a loss-weighting question, not a dataloader one, and changing
  it would need physics sign-off).

**Speed** (`bench_loader.py`, realistic scale: 3,114,701 rows x 23 features, matching the
real `jj_non_central` fold0-train exactly, `batch_size=2048`, on the session's GPU):

| | time (1 epoch, data-iteration only) | throughput |
|---|---|---|
| OLD (`DataLoader(ParquetDataset)`, `num_workers=0`) | 35.14 s | 88,649 rows/s |
| NEW (`InMemoryBatchLoader`) | 0.92 s | 3,378,943 rows/s |

**38.1x faster** for the data-iteration portion alone (no model forward/backward — this
isolates exactly the fixed bottleneck). This should let GPU compute become the actual
bottleneck for training wall-time, which is where further speedups would need to target if
this run is still not fast enough (bigger batch sizes, fewer/larger-only hidden layers, mixed
precision already enabled via `amp.enable: true`).

## Not done / left as-is

- Whole-dataset GPU residency (moving `x`/`y`/`w` to the GPU once instead of per-batch
  `.to(device)`) was considered but not implemented — the profiled bottleneck was the
  per-row Python loop, not the host-device transfer itself, and pre-loading a much bigger
  full-Run3 dataset onto a T4's 15 GB (or a smaller MIG slice on other sessions) risks OOM
  for larger years/categories. Worth revisiting only if the GPU-transfer step becomes the
  new bottleneck after this fix.
- The `run_analysis_pipeline.sh -m dnn_hpo` follow-up ideas from the earlier speed
  discussion (fewer folds during HPO, restricting the `batch_size` search space, tuning
  `MedianPruner`'s `n_startup_trials`) were **not** applied — those were offered as
  additional, independent levers; this report covers only the implementation the user asked
  for ("implement #1").
- Did not touch the user's currently-running `jj_non_central` HPO job — editing this `.py`
  file has no effect on an already-running process (Python doesn't hot-reload); this fix
  only benefits the *next* fresh invocation of `hpo_optuna.py`/`train_dnn.py`.
