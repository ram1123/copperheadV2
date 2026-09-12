# get_yields.py — 2025 data yields + `--data-only` flag + mass-window count

**Date:** 2026-09-10
**Type:** Implementation / run
**Scope:** `scripts/get_yields.py`, `scripts/count_dimuon_mass_window.py` (new), `docs/GetYields.md`
**Label:** `Run3_nanoAODv15_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation`, year 2025, `cutbased`.

## What the user hit

Their command used `--input <base>/` (bare ntuple root, no `/stage1_output`). `resolve_input_layout()`
then fails the year-dir test (`<base>/2025` doesn't exist — only `<base>/stage1_output/2025` does),
falls through to the "sample_root" branch, and sets `load_path = <base>`. Every per-process glob
`<base>/data*/**/*.parquet` then matches nothing → **all yields 0.0, no error raised**. Their
earlier `yield_10Sep_2025_summary.csv` / `yield_20May_cutbased_0p92522.csv` (13:42) were all zeros
for exactly this reason.

**Fix:** point `--input` at `<base>/stage1_output`. It then resolves as the `stage1` layout and
`get_compacted_path()` picks the sibling `.../2025/compacted` dir (which exists).

## Change made

Two opt-in flags added to `get_yields.py` (defaults preserve prior behavior exactly):

- `--data-only` (`store_true`): `processes` becomes just the data glob, skipping all MC. Faster,
  and avoids any 2025 MC sample that fails to load blocking the data result (2025 MC is the known
  shared 2024-conditions Summer24 set with unverified `vbf_*` normalization — see
  2026-09-09_2025-2026-full-crosscheck.md).
- `--data-glob` (default `"data*"`): sample-dir glob for the data process, replacing the previously
  hardcoded `"data*"` at both the `--data-only` and normal `processes` build sites. `glob.glob`
  bracket classes work, so `--data-glob 'data_[C-G]'` drops `data_B`. The row's `sample` field then
  literally reads `data_[C-G]`, still classified as data everywhere (`"data" in sample.lower()`).

(The docstring example block and `ALL_YEARS` 2025/2026 entries also in the diff were pre-existing
uncommitted user edits, left as-is.)

## Command run (exit 0, ~4 min, local Dask)

```
python scripts/get_yields.py \
  --input /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation/stage1_output \
  -y 2025 --categorizer cutbased --data-only \
  --output-csv yield_10Sep_2025_cutbased_dataonly.csv \
  --summary-output-csv yield_10Sep_2025_summary.csv
```

## Result — 2025 data (raw counts = yield; data is unweighted)

| category | region | total | 0j | 1j | ≥2j |
|---|---|---|---|---|---|
| nocat | h-peak (m∈115–135) | 788140 | 476989 | 185547 | 125604 |
| nocat | h-sidebands (110–115 ∪ 135–150) | 685323 | 414373 | 161199 | 109751 |
| ggh   | h-peak | 725668 | 476988 | 185482 | 63198 |
| ggh   | h-sidebands | 630030 | 414373 | 161135 | 54522 |
| vbf   | h-peak | 8305 | 0 | 0 | 8305 |
| vbf   | h-sidebands | 7222 | 0 | 0 | 7222 |

Outputs: `yield_10Sep_2025_cutbased_dataonly.csv` (per-sample, 6 rows),
`yield_10Sep_2025_summary.csv` (summary; MC columns all 0 by construction with `--data-only`).

### Re-run excluding data_B (`--data-glob 'data_[C-G]'`, exit 0, ~4 min)

| category | region | total (C-G) | 0j | 1j | >=2j | data_B removed |
|---|---|---|---|---|---|---|
| nocat | h-peak | 786345 | 475927 | 185110 | 125308 | -1795 |
| nocat | h-sidebands | 683717 | 413398 | 160812 | 109507 | -1606 |
| ggh   | h-peak | 724003 | 475926 | 185045 | 63032 | -1665 |
| ggh   | h-sidebands | 628540 | 413398 | 160748 | 54394 | -1490 |
| vbf   | h-peak | 8285 | 0 | 0 | 8285 | -20 |
| vbf   | h-sidebands | 7210 | 0 | 0 | 7210 | -12 |

Outputs: `yield_10Sep_2025_cutbased_dataonly_noB.csv`, `yield_10Sep_2025_summary_noB.csv`.
data_B is ~0.2% of 2025, consistent with an early/short era.

## Follow-up: events after only a dimuon-mass window, no ggH/VBF/region cut

User wanted the plain event count in `70 < m_mm < 110` (and `76 < m_mm < 106`) with no
category/region selection. `get_yields.py` can't do this — its regions are hard-coded to the
Higgs windows. Key fact: **the compacted stage-1 parquet is not mass-window filtered** — it
carries the full `dimuon_mass` spectrum ([0.211, 32977] GeV on disk for 2025 C–G) after the
baseline dimuon selection only. So it's a direct count over the `dimuon_mass` column.

New script: **`scripts/count_dimuon_mass_window.py`** (pyarrow column-only read, no Dask,
~10 s/year; `--input`/`-y` resolve like `get_yields.py`, `--samples` glob, `--mass-min/max`,
`--breakdown`). Documented in `docs/GetYields.md`.

### 2025 data counts (no category, no region)

| mass window (GeV) | eras C–G | data_B | full 2025 (B–G) |
|---|---|---|---|
| 70 < m_mm < 110 | **75,960,754** | 179,468 | 76,140,222 |
| 76 < m_mm < 106 | 74,092,153 | 175,085 | 74,267,238 |

Total rows on disk (C–G, no window): 83,718,768.

### Full m_mm distribution, 2025 data C–G (raw counts)

```
[     0,     60) :  4,201,627
[    60,     70) :  1,421,618
[    70,     76) :  1,359,741
[    76,    106) : 74,092,153
[   106,    110) :    508,860
[   110,    115) :    412,255      -> get_yields nocat h-sb (with [135,150))
[   115,    135) :    786,345      -> get_yields nocat h-peak (exact match)
[   135,    150) :    271,462
[   150,    200) :    391,710
[   200,    inf) :    272,997
```

Cross-check: `[115,135) = 786,345` and `[110,115)+[135,150) = 683,717` reproduce the
`get_yields.py` "nocat h-peak" / "nocat h-sidebands" rows above exactly — validates the
direct-count method.

## Merged stage-1 cutflow, 2025 data eras C–G

From the `..._OfficialRecomendation_Cutflow` sibling label (a `-z/--isCutflow` stage-1 run),
all per-chunk `cutflow_*.npz` shards for data_C..data_G merged with the CLI-rewritten
`scripts/merge_cutflow_npz_file.py`:

```
python scripts/merge_cutflow_npz_file.py \
  /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation_Cutflow/stage1_output/2025/f1_0/data_[C-G]/0/ \
  -o cutflow_2025_dataCToG_merged.json
```

| cut | individual | (eff) | cumulative | (eff) |
|---|---:|---:|---:|---:|
| initial | 4,215,197,165 | 100.0% | 4,215,197,165 | 100.0% |
| TotalEntries | 4,215,197,165 | 100.0% | 4,215,197,165 | 100.0% |
| lumi_mask | 4,061,432,120 | 96.4% | 4,061,432,120 | 96.4% |
| HLT_filter | 1,902,595,144 | 45.1% | 1,846,338,454 | 43.8% |
| event_quality_flags | 4,170,921,041 | 98.9% | 1,840,812,129 | 43.7% |
| muon_pT_roch | 2,753,867,036 | 65.3% | 1,800,567,461 | 42.7% |
| muon_eta | 3,993,613,387 | 94.7% | 1,798,825,686 | 42.7% |
| muon_id | 3,720,856,190 | 88.3% | 1,764,352,057 | 41.9% |
| muon_isGlobal_or_Tracker | 3,948,263,626 | 93.7% | 1,764,352,057 | 41.9% |
| muon_selection | 2,452,822,831 | 58.2% | 1,751,543,676 | 41.6% |
| muon_iso | 2,922,081,251 | 69.3% | 1,618,372,404 | 38.4% |
| trigger_match | 1,407,385,877 | 33.4% | 1,361,993,701 | 32.3% |
| electron_veto | 4,210,921,965 | 99.9% | 1,360,491,667 | 32.3% |
| HemVeto | 4,215,197,165 | 100.0% | 1,360,491,667 | 32.3% |
| PV_npvsGood | 4,186,259,099 | 99.3% | 1,360,491,667 | 32.3% |
| nmuons | 96,179,407 | 2.3% | 87,334,174 | 2.1% |
| mm_charge | 866,261,688 | 20.6% | 87,072,956 | 2.1% |
| jet_veto_maps | 3,882,453,918 | 92.1% | 83,718,768 | 2.0% |
| dimuon_mass_window_76_106 | 82,116,803 | 1.9% | 74,092,153 | 1.8% |
| h_peak_115_135 | 911,827 | 0.0% | 0 | 0.0% |
| h_sidebands_110_115_135_150 | 791,912 | 0.0% | 0 | 0.0% |
| h_sidebands_106_115_135_150 | 1,365,615 | 0.0% | 0 | 0.0% |

- `individual` = the cut applied alone to the raw 4.215e9; `cumulative` = running AND in this
  argument order. The three trailing `h_*` rows go to `cumulative = 0` because they are
  OR-alternatives applied after `dimuon_mass_window_76_106` (a 76–106 event is not also in
  115–135 or the sidebands) — read their `individual` column instead.
- Ties the two halves of this report together: `jet_veto_maps` cumulative **83,718,768** =
  "total rows on disk" above; `dimuon_mass_window_76_106` cumulative **74,092,153** = the
  `76 < m_mm < 106` count above. `h_peak_115_135` individual 911,827 > parquet `[115,135)`
  786,345 because the cutflow value is before the `jet_veto_maps` + full-dimuon reduction the
  parquet already has.
- Full writeup with the same table: 2026-09-09_cutflow-api-review-and-fixes.md (2026-09-10 section).

## Caveats / not verified

- **h-peak is the blinded signal window (115–135 GeV).** These are stage-1 parquet counts; if the
  intent is a blinded look, use the h-sidebands rows only.
- vbf category is ≥2j by definition (jj-mass / jj-dEta cut) — hence 0 in 0j/1j, as expected.
- MC yields for 2025 were not computed here; a full data+MC run needs `--input .../stage1_output`
  (same fix) **without** `--data-only`, and should be sanity-checked against the 2025 MC caveats
  in 2026-09-09_2025-2026-full-crosscheck.md.
- `separate_wgt_zpt` division in `weighted_yield_for_mask` / `get_yield` is a no-op for data.
