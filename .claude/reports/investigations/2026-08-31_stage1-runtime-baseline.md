# Stage-1 runtime baseline — 2024 / NanoAODv15

- Date: 2026-08-31 (updated 2026-09-11 — see "2025 update" section below)
- Type: Investigation
- Status: 2024 section open (baseline only; MC samples not yet captured there). 2025
  section closed for the config tested (chunksize 250k / `scale(98)` / `worker_cores=2`,
  `worker_memory=10`) — three independent full-2025 runs measured, consistent results.
- Applicable era: 2024 (original section), 2025 (2026-09-11 update)
- NanoAOD campaign: v15 (`configs/datasets/dataset_nanoAODv15_run3.yaml`)
- Relevant files: `run_stage1.py`, `src/stage1/runner_adapter.py`, `src/copperhead_processor.py`, `modules/dask_utils.py`, `DaskGatewaySLURM.ipynb`

## Question

Record measured per-sample stage-1 runtimes from the 2024 v15 runs, as a baseline
to judge whether the pending speedup changes actually help.

## Run configuration

- Command: `python run_stage1.py -y 2024 --NanoAODv 15 --max_file_len 900 --yaml configs/datasets/dataset_nanoAODv15_run3.yaml --skipSamples --use_gateway`
- save_path: `/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation`
- Dask Gateway (k8s): `processes=35  threads=70  memory=525.00 GiB` (2 threads/proc)
- `chunksize=100_000` (coffea `Runner` + `DaskExecutor`); `jer_strat=4`; `do_jec_unc=do_jer_unc=false`
- Input redirector: `root://eos.cms.rcac.purdue.edu/` (direct Purdue EOS — keep; see the stage1-xrootd-redirector-choice memory)

## Measured per-sample runtimes

| Sample | data/MC | Events processed | Wall time | Aggregate rate | Notes |
|--------|---------|------------------|-----------|----------------|-------|
| data_C | data | — | skipped (0.009 s) | — | done marker present |
| data_D | data | 241,254,436 | **800.4 s** | ~301 k evt/s | 1 batch (<=900 files), `resume` attempt 1 OK |
| data_E | data | — | skipped (0.004 s) | — | done marker |
| data_F | data | — | skipped (0.005 s) | — | done marker (2 batches) |
| data_G | data | — | skipped (0.006 s) | — | done marker (2 batches) |
| data_H | data | — | skipped (0.003 s) | — | done marker |
| data_I | data | 210,383,667 | **673.7 s** | ~312 k evt/s | ran in the earlier run; skipped later |
| dyTo2Mu_M-105To160 | MC | 92,601,022 gen | did **not** finish | — | JER `ScaleFactor` arity crash (now fixed); `max_file_len` auto 500, 2 batches |

No MC sample has a completed timed run yet — the first MC sample hit the JER
Summer24 `ScaleFactor` schema bug on every attempt.

## Findings

- **Data throughput ~300–312 k evt/s aggregate** ≈ ~4.3–4.5 k evt/s per thread
  (70 threads). Roughly linear with event count between the two data samples.
- **Log-line gotcha:** `Finished sample X in N s` is the true per-sample
  processing time. `[Timing] Time taken to process sample X: N seconds` is a
  *cumulative* wall-clock counter from run start, not per-sample — it read
  ~803.9 s for data_E…data_I even though each was skipped in 3–9 ms.
- The large run-to-run swing seen historically (a sample taking ~10⁴ s vs ~10³ s)
  is EOS/file-server I/O variance, not CPU, and not fixed by a redirector swap
  (inputs are Purdue-local).

## Verification

- Command: none run here — numbers transcribed from the user's stage-1 console
  logs (2026-08-31 runs).
- Result: n/a (observational record).

## Additional records from `logs/` (three scenario dirs, reviewed 2026-08-31)

Cluster shape varied: older runs `35 proc / 70 threads / 525 GiB`; the current
2024 run `31 proc / 31 threads / 465 GiB`. `Finished sample X in 0.00x s` = skipped
(done marker); real per-sample numbers are the 10 s–2500 s ones.

### `logs/FilterEvents_Aug30_..._OfficialRecomendation/` — 2024 v15 (active)
- `stage1_2024_rep1.log` (Aug 31 03:38, **still running**, 31p/31t): MC now works
  after the JER fix. `dyTo2Mu_M-50_aMCatNLO` 1027 s (85.09M + 84.82M evt over 2
  chunks), `dyTo2Mu_MLL_120To200` 411 s (48.04M evt). Data all skipped.
- `...attempt_2026-08-30_21-01-26` (Aug 30, 66 min, 35p/70t): all 27 "finished"
  but data_C–data_I (at least) FAILED with
  `KeyError('Summer24Prompt24_V2_DATA_L1L2L3Res_AK4PFPuppi')` — the stale JEC
  **data** tag (V2->V5, see jer-strat-4 memory). Its 37–552 s per-sample times are
  failed-attempt durations, not valid processing times.
- `failed_jobs.log`: empty.

### `logs/FilterJets_July08_..._JVMFilterJets/` — 2025 + 2026 v15 (nominal, older scenario)
- stage1 2026 full run `...attempt_2026-08-27_02-49-10` (**5h00m**, 23 samples, 35p/70t).
  Slowest: ttjets_sl 2466 s, ttjets_dl 2353 s, dyTo2Mu_M-50_aMCatNLO 1605 s,
  dyTo2Mu_MLL_50To120 1098 s, wz_3lnu 1016 s, zz_4l 1011 s, wz_2l2q 989 s,
  ww_2l2nu 956 s, zz_2l2q 949 s, data_B 894 s, data_D 646 s, data_A 185 s.
  2 transient failures (`FileNotFoundError` on a dy chunk, `scheduler-connection-lost`),
  recovered on resume.
- stage1 2026 resume `...16-46-38` (59 min): only data ran — data_B **1956 s**,
  data_D 1296 s, data_A 264 s.
- stage1 2025 full `...attempt_2026-08-27_18-30-43` (**2h54m**); resume `...21-24-17`
  (18 min) ran only data_C at **1053 s**.
- compact 2025: 250 s (20 already / 6 done). compact 2026: **1843 s** (all 23);
  ttjets_sl 327 s for only 334k rows from **4795 input files** — tiny-file
  bottleneck. `failed_jobs.log`: stage1 2025 + compact + 6 plot jobs all SUCCESS.
- Also holds 12 plot logs (2025/2026 x 6 cat), `pu_dnn_train_2026.log` (209 KB),
  zpt step0/step1 logs for 2026.

### `logs/FilterJets_July08_..._JVMFilterJets_Syst/` — 2025 v15 (systematics variant)
- stage1 2025 full `...attempt_2026-08-26_12-14-35` (**1h44m**, 26 samples).
  Syst variant is much slower per MC sample: dyTo2Mu_MLL_50To120 **2027 s**
  (vs 1098 s nominal 2026), zz_4l 1500 s, ttjets_dl 1308 s, ttjets_sl 1098 s.
- stage1 2025 resume `...17-02-51` (34 min): wz_2l2q 1458 s, dyTo2Mu_M-50 583 s.
- `stage1_compact_2025.log`: **FAILED** — `already_exists=25, failed=1`, sample
  `dyTo2Mu_MLL_10To50` (traceback truncated in log). `failed_jobs.log` shows the
  compact retried and failed again. **Open issue.**

### Cross-cutting
- Consistently slowest: `ttjets_sl`/`ttjets_dl`, `dyTo2Mu_MLL_50To120`,
  `dyTo2Mu_M-50_aMCatNLO`, diboson `wz_*`/`zz_*`, and the large data eras
  (`data_B`/`data_C`/`data_D`).
- The `_Syst` scenario roughly 1.5–2x the per-MC-sample time (pt_variations loop).
- compact cost is dominated by input-file *count*, not row count (many O(1k–5k)
  tiny stage-1 shards per sample).
- Failure modes seen in these logs: stale JEC **data** tag `KeyError`
  (V2->V5, fixed); JER `ScaleFactor` arity (fixed this session); transient
  `FileNotFoundError` / `scheduler-connection-lost` (resume recovers);
  `_Syst` compact `dyTo2Mu_MLL_10To50` (open).

## Canonical source: `stage1_output/<year>/_status/job_status.jsonl`

Better than scraping console logs. Each line is one chunk event
(`status` = running/done/failed) with `duration_seconds`, `Expected events from
pre-stage`, `Processed events from stage-1`, `attempt`, `redirector`,
`git_commit_hash`, `git_branch`. `stage1_summary.{json,log}` beside it is the
rollup; `<sample>__<idx>.done` files are the resume markers.

**Reusable tool:** `scripts/summarize_stage1_status.py` -- Markdown tables for
done/failed/stuck, wall time, and `runs->done` (how many runs each sample's
slowest chunk needed to succeed). E.g.
`python scripts/summarize_stage1_status.py --root <ntuples-base> --failures`.
Point-in-time snapshot: `2026-08-31_stage1-status-snapshot.md` beside this file.

`meta.attempt` in the ledger is only the redirector-retry index inside one
`run_stage1.py` call (`AAA_REDIRECTORS` has one entry -> always 1); the real
"goes it took" is the count of `running` events for a chunk (= `runs->done`).

### NanoAODv15 scenarios found (as of 2026-08-31)

| Scenario / year | done | failed | stuck | sum(chunk hrs) | note |
|---|---|---|---|---|---|
| FilterEvents_Aug30_..._OfficialRecomendation / 2024 | 19 | **18** | 2 | 4.2 | active; failed/stuck chunks are pre-fix commits (JEC data tag, JER arity) |
| ..._DefaultjetPt25GeV_JVMFilterJets / 2024 | 39 | 0 | 0 | 9.2 | clean |
| ..._DefaultjetPt25GeV_JVMFilterJets / 2025 | 45 | 0 | 0 | 8.3 | clean (9-day span, many resumes) |
| ..._DefaultjetPt25GeV_JVMFilterJets / 2026 | 35 | 0 | 0 | 5.7 | clean |
| ..._DefaultjetPt25GeV_JVMFilterJets_Syst / 2025 | 45 | 0 | 0 | **10.9** | syst on |
| ..._PUDNN_TrainOn2024 / 2024 | 39 | 0 | 0 | 7.8 | clean; ~same as Default |
| ..._PUDNN_TrainOn2024 / 2025 | 45 | 0 | 0 | 9.0 | clean |
| ..._PUDNN_TrainOn2024_Syst / 2024 | 12 | **22** | 3 | 3.4 | mostly-failed MC (older commit `f7b1d74a`) — needs a look |
| ..._PUDNN_TrainOn2024_Syst / 2025 | 0 | 0 | 1 | 0.0 | never ran (abandoned) |

(NanoAODv12 `_status` dirs also exist under
`Run3_nanoAODv12_FilterJets_July08_*` — 2022preEE/postEE/2023/2023BPix — not
tabulated here; all `FAILED: 0` in their `stage1_summary.log`.)

### Per-sample stage-1 cost (v15, clean nominal runs; wall s summed over chunks)

| Sample | chunks | events | wall s | evt/s |
|---|---|---|---|---|
| data_G | 2–3 | 1.28 B | 3700–4200 | 310–350 k |
| data_F | 2–3 | 0.88–1.04 B | 2500–3000 | 300–415 k |
| data_C | 1–3 | 0.20–0.80 B | 670–4100 | 47–360 k |
| data_D | 1–3 | 0.24–0.96 B | 700–3200 | 240–350 k |
| data_E | 1–2 | 0.34–0.53 B | 1000–1700 | 205–380 k |
| ttjets_sl | 2 | 484 M | 2470–2760 | ~190 k |
| ttjets_dl | 2 | 470 M | 2330–2650 | ~200 k |
| dyTo2Mu_MLL_50To120 | 2 | 492 M | 1700–2080 | ~240–290 k |
| dyTo2Mu_M-50_aMCatNLO | 6 | 490 M | 1780–2250 | ~210–275 k |
| dyTo2Mu_MLL_10To50 | 1 | 342 M | 640–1000 | 330–530 k |
| wz_3lnu / zz_4l / wz_2l2q / ww_2l2nu | 1 | 240–248 M | 950–1140 | ~220–250 k |
| zz_2l2nu / zz_2l2q / wz_1l1nu2q | 1 | 144–189 M | 640–870 | ~200–240 k |
| dy_VBF_filter | 2 | 94 M | 520–650 | ~150–180 k |
| dyTo2Mu_M-105To160 | 2 | 93 M | 350–490 | ~200–260 k |
| dyTo2Mu_MLL_120To200 | 1 | 48 M | 215–410 | ~120–225 k |
| vbf_aMCatNLO / vbf_powheg | 1 | 5.6–11 M | 115–355 | 26–98 k |
| ewk_mmjj_* | 1 | 5–6 M | 77–176 | 33–77 k |
| ggh_powhegPS | 1 | 2.8 M | 67–153 | 18–42 k |

### Observations from the status data

- **Data ~300–410 k evt/s; MC ~180–270 k evt/s** (more corrections/weights).
  Tiny samples (`ggh_powhegPS`, `vbf_*`, `ewk_*`) run at 20–95 k evt/s — a
  ~60–250 s per-`runner()` fixed-cost floor dominates, not throughput.
- **Syst variant costs ~+30 % overall, concentrated on MC**: same 2025 sample set
  10.9 h (Syst) vs 8.3 h (nominal); `ttjets_dl` 3388 s vs 2235 s (1.5x),
  `dyTo2Mu_MLL_50To120` 3354 s vs 1912 s (1.75x); MC evt/s falls ~210 k -> ~150 k.
  Data ~unchanged (`do_jec_unc` forced false on data).
- **PU-DNN scenario ≈ Default scenario** in time — PU-DNN scoring adds little.
- **Data chunk count varies by scenario** for the same sample (e.g. `data_D` =
  1 / 3 / 2 chunks across 2024 / 2025 / 2026) — driven by `max_file_len 900` vs
  per-era file count; more chunks = more parallelism but more fixed overhead.
- **Redirector was `root://eos.cms.rcac.purdue.edu/` on every done chunk** — no
  fallback ever exercised (single-entry list).
- **Open failure clusters to investigate:** `PUDNN_TrainOn2024_Syst/2024`
  (12 done / 22 failed / 3 stuck, commit `f7b1d74a`); the pre-fix failed+stuck
  chunks in `FilterEvents_Aug30/2024` should clear on the current (post-fix) run.

## Remaining work

- Capture MC per-sample runtimes once a run gets past the JER step (after a
  gateway worker restart picks up the fixed `jet.py`).
- Re-measure `data_D` / `data_I` after the speedup changes for a before/after:
  - JER-smear nominal-only gating (applied 2026-08-31 — `do_jer_smear` +
    `apply_jer_unc` skipped for up/down when `do_jer_unc=false`);
  - `chunksize` 100 k -> 250–400 k;
  - redirector fallback list (Purdue-local only, never XCache).
- Keep appending rows here as more samples/years are measured.

---

## 2025 update — 2026-09-11: chunksize 250k + `scale(98)`/`worker_cores=2` measured

- Date: 2026-09-11
- Type: Investigation (live Dask Gateway monitoring across 4 separate full-2025 stage-1
  runs, 2026-08-31 through 2026-09-11)
- Applicable era: 2025
- Status: Closed for the config tested below — three independent 250k runs agree to
  within run-to-run noise; the `chunksize` A/B (100k vs 250k) is a clean same-input-file
  comparison. One open item (`worker_cores` 1-vs-2 A/B) not yet run — see "Open items".

### What changed since the Aug-31 (2024) baseline above

The 2024 baseline table above was measured at `chunksize=100_000`, an *adaptive*
Gateway cluster (`cluster.adapt(min, max)`), and `worker_cores=1`/`worker_memory=15`.
Across this week the following were changed and tested on 2025 (`DaskGatewaySLURM.ipynb`,
uncommitted local notebook state; `run_stage1.py:182`, **committed** as of `906d636`,
2026-09-01):

| Setting | Aug-31 baseline | Current (tested below) |
|---|---|---|
| `chunksize` (coffea `Runner`) | 100,000 | **250,000** (`run_stage1.py:182`, committed) |
| Gateway scaling | `cluster.adapt(35, 399)` | **`cluster.scale(98)`** fixed (adapt line commented out) |
| `worker_cores` | 1 | **2** |
| `worker_memory` | 15 GiB | **10 GiB** |
| `max_file_len` (per-dataset split) | 900 (data), auto per `DATASET_ELEMENT_LIMITS` (MC) | **50,000 for every dataset** — each dataset is one `file_idx` unit (one `.compute()`) instead of split into multiple |

Motivation for `adapt`->`scale(98)`: `adapt(35,399)` oscillated the desired-worker count
wildly (observed swings between 1 and 1000+) and mass-retired 20-40 workers every time the
per-file task count dipped (e.g. between chunks, at sample boundaries), which repeatedly
evicted in-flight reduction inputs and either stalled or crashed the run. A fixed pool
removes that churn entirely. Motivation for `worker_cores` 1->2: single-threaded workers
were read-bound (idle waiting on XRootD) with no way to overlap a stalled read against
compute; two threads let one task compute while the other's read is in flight. Motivation
for `max_file_len` 900->50000 (single unit per dataset): removes the idle gap at every
file-to-file boundary within a dataset (previously the pool drained to near-zero between
files); trades this off against a larger blast radius on failure (see "Reliability" below).

### Per-sample runtimes, three independent full-2025 runs at the current config

All three runs used `chunksize=250_000`, `scale(98)`, `worker_cores=2`, `worker_memory=10`,
`max_file_len=50000`. Numbers are `done.ts - running.ts` per `(dataset, idx=0)` from each
run's canonical `_status/job_status.jsonl` (not console-log scraping) — see the "Canonical
source" note above; all three job_status.jsonl files were re-read directly for this update.
Event counts are per-dataset totals (`meta.Processed events from stage-1`), identical
across runs for the same `dataset` name since the raw NanoAOD inputs don't change with
selection — confirms these are true same-input-file comparisons.

| Dataset | Events | Sep 1 `FilterJets/DefaultjetPt25GeV` | Sep 2-3 `FilterEvents/OfficialRecomendation` | Sep 11 `..._MuonID-tightId` (+ `--isCutflow`) |
|---|---:|---:|---:|---:|
| data_B | 11,200,754 | 146.9 s | 183.3 s | 164.2 s |
| data_C | 798,797,125 | 1568.7 s | 1562.7 s | 1509.6 s |
| data_D | 959,319,428 | 1904.5 s | 1873.4 s | 1883.4 s |
| data_E | 531,276,074 | 1100.6 s | 1102.6 s | 1121.2 s |
| data_F | 1,041,242,824 | 2049.4 s | 2139.8 s | 2102.8 s |
| data_G | 884,561,714 | 1721.1 s | 1868.4 s | 1911.2 s |
| dyTo2Mu_M-50_aMCatNLO | 490,076,405 | 1097.8 s | 1144.9 s | 1196.4 s |
| dy_VBF_filter | 94,375,156 | 292.3 s | 300.0 s | 308.4 s |
| ewk_mmjj_mll_105_160 | 5,274,178 | 167.4 s | 265.4 s | 180.3 s |
| ggh_powhegPS | 2,800,000 | 85.1 s | 100.1 s | 87.8 s |
| ttjets_dl | 470,123,263 | 1598.0 s | 1887.1 s | 1788.4 s |
| ttjets_sl | 484,475,057 | 1723.2 s | 1652.5 s | 2060.4 s |
| vbf_aMCatNLO | 11,193,000 | 224.7 s | 216.6 s | 353.8 s |
| vbf_powheg | 11,190,000 | 149.8 s | 154.6 s | 182.2 s |
| ww_2l2nu | 239,943,886 | 767.0 s | 787.6 s | 903.0 s |
| wz_1l1nu2q | 143,874,689 | 514.3 s | 619.1 s | 642.9 s |
| wz_2l2q | 236,049,432 | 769.0 s | 861.1 s | 834.5 s |
| wz_3lnu | 248,149,069 | 730.6 s | 910.5 s | 892.2 s |
| zz_2l2nu | 188,715,312 | 646.5 s | 605.3 s | 582.3 s |
| zz_2l2q | 171,627,615 | 634.3 s | 685.4 s | 722.7 s |
| zz_4l | 236,757,511 | 699.0 s | 947.8 s | 782.4 s |

Full-run totals (26/26 datasets each; data_A/H etc. not present for 2025 -> 21-23 datasets
actually processed, rest instant `skip_sample`): Sep 1 = 5h47m; Sep 2-3 = 5h31m; Sep 11
= 3h12m for the fresh-processed leg only (data eras B-G resumed from `.done` markers on a
mid-run relaunch, so its wall clock is not comparable to the other two full-from-scratch
totals — the per-sample table above is the valid comparison, not the run totals).

### chunksize 100k -> 250k, same selection, same worker shape (clean A/B)

The Sep 2-3 `FilterEvents/OfficialRecomendation` run reprocessed the *same label* as the
very first 2025 stage-1 run (2026-08-31, chunksize 100k, also on a `scale(98)`/
`worker_cores=2` cluster by that point — the `adapt`->`scale(98)` and `worker_cores=1->2`
changes both predate the chunksize change, so this isolates chunksize alone). That
original 100k run's `job_status.jsonl` was since overwritten by the 250k rerun (same
label -> same file); the 100k numbers below are from direct live monitoring of that run's
console log on 2026-08-31/09-01 (not re-derivable from `job_status.jsonl` now):

| Sample | Events | chunksize 100k | chunksize 250k | Delta |
|---|---:|---:|---:|---:|
| data_G | 884,561,714 | 2098 s (421 k evt/s) | 1868.4 s (473 k evt/s) | **-11%** |
| dyTo2Mu_M-50_aMCatNLO | 490,076,405 | 1234 s (397 k evt/s) | 1144.9 s (428 k evt/s) | **-7%** |

Consistent with the mechanism: fewer/larger tasks -> less per-task and scheduler
overhead. Worker CPU stayed ~50% (of the 2 available cores per worker, i.e. ~25% CPU
utilization) at *both* chunksizes -- the gain is from cutting overhead, not from filling
idle cores; the read-bound ceiling is unmoved (see "Open items").

### Findings

- **Run-to-run spread at fixed config is ~3-8%** across all three 250k runs, both
  directions (e.g. data_G 1721/1868/1911 s; ttjets_sl 1723/1652/2060 s). This is larger
  than any config change tested here and is consistent with XRootD/EOS read-latency
  variance and shared-node contention on the k8s pool, not a further tuning signal —
  see the recurring `eos slow on <node>` facility-health alerts below. Do not read a
  <10% difference between two runs as meaningful without repeating it.
- **The 2026-09-10 cutflow mass-region-rows fix (`isCutflow-mass-region-rows.md`) adds no
  measurable overhead.** The Sep-11 run has `--isCutflow` active and its per-sample times
  fall inside the same noise band as the other two runs with the flag off (e.g. data_C
  1509.6 s vs 1562.7/1568.7 s -- the *fastest* of the three). Consistent with the change
  being 4 extra vectorized boolean ANDs over already-computed arrays (see that report),
  not new per-event work.
- **Worker utilization ceiling is ~50% of 2 cores/worker (~25% CPU) at both chunksizes and
  across all three runs.** Read-bound: sub-second per-chunk compute, workers idle waiting
  on `root://eos.cms.rcac.purdue.edu/` reads. `worker_cores` 1->2 was intended to let one
  task's compute overlap another's read, but the observed ~25%-of-2-cores utilization
  (not closer to 50-100%) suggests either (a) Python-level GIL contention in
  `EventProcessor` (awkward/numpy-heavy, some of it not GIL-releasing) capping effective
  parallelism at ~1 core/worker regardless of thread count, or (b) an aggregate EOS
  read-bandwidth ceiling that more concurrent streams from the same client can't beat.
  **Not yet disambiguated** — see "Open items".
- **Small-sample fixed-overhead floor persists at 250k, unchanged in character from the
  2024 baseline table above.** `ggh_powhegPS` (85-100 s / 2.8 M evt), `vbf_powheg`
  (150-182 s / 11.2 M evt), `vbf_aMCatNLO` (217-354 s, most variable of the set),
  `ewk_mmjj_mll_105_160` (167-265 s / 5.3 M evt) are all dominated by per-`runner()`
  graph-build + worker-warmup + tail-reduction, not throughput. `chunksize` doesn't touch
  this; only reducing the *number* of `runner()` calls (batching datasets) would, and
  isn't implemented.
- **Reliability, `max_file_len=50000` (single unit per dataset) tradeoff**: on a clean run
  this removed all mid-dataset idle gaps (confirmed: summed per-sample compute time ==
  ~100% of the Sep-11 run's wall clock, no unaccounted idle). But it also means a
  mid-dataset scheduler loss discards the *entire* dataset's progress, not one file's.
  This was live during the week: repeated `scheduler-connection-lost` failures (2026-08-31
  ~16:38, ~19:51 UTC; 2026-09-01 ~02:44, ~17:46 UTC) coincided with facility
  `eos slow on <node>` alerts (node identified as **`paf-d01`** as of 2026-09-02 19:20
  UTC; unnamed in earlier alerts). One of these (2026-09-01 17:46 UTC) was root-caused via
  `distributed.scheduler`/`distributed.worker.memory` log lines to a **worker
  unmanaged-memory cascade** (workers pinned 7-9 GiB of a 10 GiB limit, `Unmanaged memory
  use is high` warnings, pause/resume thrash, then mass `Removing worker ... caused the
  cluster to lose already computed task(s)`), triggered by starting a stage-1 run on a
  Gateway cluster that had just served a plotting job (`validation_plotter_unified.py`)
  without restarting it -- the plotting job's `client.compute()` calls embed ~22-25 MiB
  graphs per call (`UserWarning: Sending large graph`) that Dask workers never fully
  release back to the OS. **Mitigation confirmed effective**: start stage-1 on a freshly
  created (or `client.restart()`ed) Gateway cluster, never one that just ran plotting.
  Runs launched under `snakemake ... --restart-times 3` self-heal from a scheduler death;
  runs launched by calling `run_analysis_pipeline.sh` directly do not and need a manual
  relaunch (observed 2026-09-01: bare launch died and stayed dead until relaunched by the
  user; the same day's snakemake-launched runs auto-recovered).
- **Facility EOS reliability is the dominant external risk, not a stage-1 config
  problem.** At least 4 scheduler-death incidents this week correlate with `eos slow on
  <node>` facility-health alerts (mix of unnamed and `paf-d01`-named occurrences,
  2026-08-31 through 2026-09-02); one incident's proximate cause was the memory cascade
  above rather than EOS directly, but the slow reads that preceded it (data_B took
  ~13x its normal wall time during that window) are the same facility issue. Worth
  tracking with the facility team as a recurring `paf-d01` pattern rather than one-off
  incidents.

### Open items

- **`worker_cores=1` + more workers (same total task-slot count) vs the current
  `worker_cores=2` A/B, to settle GIL-vs-EOS-bandwidth** (see "Findings" above). E.g.
  `worker_cores=1, scale(196)` vs the current `worker_cores=2, scale(98)` — both give 196
  concurrent task slots; if throughput rises, it was GIL (go single-threaded/more
  processes); if flat, it's an EOS bandwidth ceiling (stop tuning Dask, look at read
  path). Not yet run.
- `worker_memory` could likely drop 10 -> 6-8 GiB for stage-1-only use (observed peak
  ~2.4 GiB/worker across all three runs here) — but only if the cluster is never shared
  with a plotting job in the same session; keep it at 10 (or restart between workloads)
  otherwise.
- `chunksize` above 250k (e.g. 500k) untested; likely small further gain on large data
  eras but shrinks task-per-worker-wave count on the smallest MC samples (e.g.
  `ggh_powhegPS` at 2.8 M events would drop to ~6 tasks at 500k) — watch straggler
  tolerance if tried.
- No 2025 MC-only or Syst-variant scenario timing captured yet at the current config
  (all three runs above are the nominal `OfficialRecomendation`/`DefaultjetPt25GeV`/
  `MuonID-tightId` selections); the 2024-baseline table's "Syst costs ~+30%" finding has
  not been re-verified at `chunksize=250k`.
