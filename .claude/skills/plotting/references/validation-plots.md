# Control / Validation Plotting — `run_plotter.py` & `validation_plotter_unified.py`

Responsible: **analysis-specific implementation**, not a CMS recommendation.

## Stored sources

| # | Source | Location | Snapshot / verified |
|---|--------|----------|---------------------|
| C1 | Driver / user config | `run_plotter.py` — `LOAD_PATH`, `SAVE_ROOT`, `years`, `categories` are edited in-file, not read from a shared config | 2026‑09‑17 |
| C2 | Core plotting engine | `plotter/validation_plotter_unified.py` — `ValidationHistProcessor` (coffea `ProcessorABC`), `run_bulk_validation()` | 2026‑09‑17 |
| C3 | Process-group source of truth | `configs/samples/samples.yaml`, read via `get_bkg_sig_dicts`/`get_data_processes` (`modules/sample_config.py`) — same helpers `scripts/get_yields.py` uses | 2026‑09‑17 |
| C4 | Category/region source | `modules/selection.py` (`applyRegionCatCuts`) — see the `event-selection` skill | 2026‑09‑17 |
| R1 | Known stale duplicate | `.claude/reports/registry.md` / `investigations/2026-09-03_plot_real_fake_jets_stack_review.md` | 2026‑09‑03 |

Classification tags: **[Analysis-specific]**, **[Implementation]**, **[Verify]**.

---

## 1. What to edit before running (README.md, C1)

Both files have **in-file** configuration, not a shared YAML — always check and edit
these before trusting a plot:

- `run_plotter.py`: `LOAD_PATH` (stage-1 output location), `SAVE_ROOT`/`SAVE_TAG`
  (output dir), `years` (currently commented to Run 2 or Run 3 — only one list is
  active at a time), `categories` (`["nocat","vbf","ggh","bJetVeto"]`).
  `--dry-run` / `--force` / `--debug` are supported CLI flags (`--force` bypasses the
  `_status` done-markers and recomputes everything).
- `plotter/validation_plotter_unified.py`: the dataset/process lists near the top of
  the file (see the line range the `README.md` points at for the current version) —
  must be updated for the processes actually being compared.

---

## 2. How it's structured (C2, C3, C4)

- `FULL_REGIONS = ["z-peak", "signal", "h-peak", "h-sidebands"]`,
  `FULL_CHANNELS = ["nocat", "vbf", "ggh", "bJetVeto"]` — static Hist-axis category
  lists; which values actually get filled is controlled separately by the `regions`
  list passed in and by the single `category` a given call asks for. These match the
  mass-window and category definitions in the `event-selection` skill exactly — this
  script is one of `applyRegionCatCuts`'s consumers, not an independent definition.
- Process groups (`DATA, DY, DYVBF, EWK, TOP, VV, ggH, VBF`) are resolved per-year
  from `configs/samples/samples.yaml` via `build_group_dict_for_year()` — the same
  YAML and the same helper functions `scripts/get_yields.py` uses, so group
  definitions stay consistent across yield tables and plots.
- `bkg_MC_order = ["VV", "EWK", "TOP", "DY", "DYVBF"]` — fixed stacking order for
  background histograms.
- `ZPT_POSTFIX_BY_OPTION = {"default": ..., "no_zpt": ..., "dnn_zpt": ...}` — the
  three Z-pT-weight comparison modes (cross-ref `corrections/zpt-reweighting.md`):
  used to produce the with/without/DNN-method comparison plots that pipeline's
  validation step calls for.
- `run_bulk_validation()` runs one Dask pass per **scope** — the product of
  `jj_eta_region`, `vbf_filter_study`, and `region_list` values requested, since each
  of those three changes which files/processes get read or which regions get filled
  (unlike year/category/njets/zpt_option, which share one Hist pass).

---

## 3. Known stale/duplicate script (R1)

`plotter/plot_real_fake_jets_stack.py` is an **earlier, less complete snapshot** of
`MVA_training/pileup_symbolic_regression/plot_real_fake_jets_stack.py` (confirmed by
diff — same core logic and column-naming). The `MVA_training` copy is the one
actually documented (with real example invocations) in its own `README.md`; the
`plotter/` copy isn't referenced anywhere else in the repo. The `MVA_training`
version additionally has `--region` (inclusive/central/HE/HF/HEpos/HEneg/HFpos/
HFneg), `--apply-cleaning`, a Fake/Total ratio subpad, and glob-based multi-file
input. **Use the `MVA_training/pileup_symbolic_regression/` copy** unless there's a
specific reason to keep the two separate — and check for other such duplicates before
assuming a script under `plotter/` is the maintained one.

Most of the variables that script plots (`chEmEF`, `chHEF`, `neEmEF`, `neHEF`,
`muEF`, `*Multiplicity`, `nConstituents`, `nElectrons`, `nMuons`, `hadronFlavour`,
`partonFlavour`, `hf*`) are **only written to stage-1 parquet when
`do_add_jet_ID_vars: true`** — confirm that switch was on for the run being plotted,
or most of the requested columns simply won't exist.

---

## 4. Review checklist

1. `run_plotter.py`'s `LOAD_PATH`/`SAVE_ROOT`/`years`/`categories` actually match the
   request (they're commented in/out by hand, easy to leave on a stale value).
2. `validation_plotter_unified.py`'s dataset/process list at the top was updated for
   the processes being compared.
3. Category/region terms used match `event-selection`'s definitions exactly (this
   script consumes `applyRegionCatCuts`, doesn't redefine categories itself).
4. If plotting `do_add_jet_ID_vars`-gated columns, that switch was on for the
   stage-1 run being read.
5. Before adding a new plotting script, check `plotter/` and the relevant
   `MVA_training/*/` subfolder for an existing one doing the same thing.

## Last verified

- Local source review: 2026‑09‑17
