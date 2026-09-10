# Stage-1 runtime baseline — 2024 / NanoAODv15

- Date: 2026-08-31
- Type: Investigation
- Status: Open (baseline only; MC samples not yet captured)
- Applicable era: 2024
- NanoAOD campaign: v15 (`configs/datasets/dataset_nanoAODv15_run3.yaml`)
- Relevant files: `run_stage1.py`, `src/stage1/runner_adapter.py`, `src/copperhead_processor.py`, `modules/dask_utils.py`

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
