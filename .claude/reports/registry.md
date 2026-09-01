# CMS Analysis Work Registry

This file is a concise index of durable project work. It is not a conversation
log.

| Date | Type | Topic | Outcome | Report |
|---|---|---|---|---|
| 2026-08-31 | Investigation | Muon-selection implementation review (read-only) vs stored muon guidance | Open — 10 confirmed defects (cutflow/mask mismatch, loose trigger matching + hardcoded `filterBits & 8`, missing IP cuts, `is_run2` RERECO misclassification, dead correction config, chunk-biased 2017 trigger SF, `pt_raw` thresholds, MuonScaRe 26 GeV floor, `doing_BS_correction` decoupling, era-asymmetric systematics) + `[Verify]` list + `muons.md` gaps; no code changed | investigations/2026-08-31_muon-selection-review.md |
| 2026-08-31 | Investigation | Stage-1 runtime baseline + status tooling (v15, multi-scenario) | Open — per-sample cost from `_status/job_status.jsonl` (data ~300-410k evt/s, MC ~180-270k, Syst ~+30%); added `scripts/summarize_stage1_status.py` (done/failed/stuck + wall + runs-to-succeed as Markdown). Open failure clusters: `FilterEvents_Aug30/2024` (pre-fix JER/JEC, clearing on current run), `PUDNN_TrainOn2024_Syst/2024` (bitwise_and TypeError, commit f7b1d74a) | investigations/2026-08-31_stage1-runtime-baseline.md (+ _stage1-status-snapshot.md) |