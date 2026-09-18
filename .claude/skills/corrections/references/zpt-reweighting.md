# Z-pT Reweighting — DY Dimuon-pT Mismodeling Correction

Responsible: **analysis-specific** — Z/DY pT-spectrum mismodeling corrections are
standard practice across CMS Z/H→µµ analyses, but the fitted parameters here are this
analysis's own derivation; there is no POG number to check against.

## Stored sources

| # | Source | Location | Snapshot / verified |
|---|--------|----------|---------------------|
| C1 | Switch | `configs/parameters/switches.yaml` (`do_zpt`) | 2026‑09‑17 |
| C2 | Application | `src/copperhead_processor.py`: `getZptWgts_3region`, `_eval_zpt_old_quadrature`, `_zpt_combined_param_envelope_full_cov`, `_zpt_combined_param_envelope_diagonal`, and the call site (`'dy' in dataset and is_mc and switches.do_zpt`) | 2026‑09‑18 |
| C3 | Alternative method (DNN) | `src/corrections/zpt_dnn.py`; trainer `MVA_training/zpt_reweight/train_dnn_zpt_reweight_dak.py` | 2026‑09‑17 |
| C4 | Derivation pipeline | `src/copperhead/zpt_rewgt/derive/{save_SF_rootFiles.py,do_f_test.py,get_polyFit.py,bin_definitions.py,poly_utils.py,sample_resolution.py}`, `dctr_zpt_reweighting/` | 2026‑09‑18 |
| C5 | Fitted-parameter payloads | `data/zpt_rewgt/*.yaml` (several tagged variants); selected via `configs/parameters/SF_filelist.yaml` (`new_zpt_weights_file_aMCatNLO`, `new_zpt_weights_file_MiNNLO`) | 2026‑09‑17 |
| C6 | Procedure docs | `docs/ZpT_reweight.md`, `README.md` ("Z-pT reweighting" section) | 2026‑09‑17 |

Classification tags: **[Analysis‑specific]**, **[Implementation]**, **[Verify]**.

---

## 1. What it corrects and where it's switched on

Drell‑Yan MC generators mismodel the dimuon‑pT spectrum relative to data; this weight
reweights DY MC event‑by‑event to match the observed data/DY ratio, derived separately
per jet‑multiplicity bin.

- `do_zpt` (C1): **`true` for Run 2 (2016preVFP–2018)**, **`false` for every Run 3 year
  (2022preEE–2026) in the base `switches.yaml`** — turned on for a Run 3 production via
  the `nominal_zpt` switches profile, not by editing the base file directly.
- Applied only when `'dy' in dataset AND is_mc` (C2) — never to data, never to
  non‑DY MC.
- Computed at the very end of the event loop, after the dimuon and jet collections
  (and `njets`) are final.

---

## 2. Method selection — check the code, not just the switch

`whichMethod` in `copperhead_processor.py`'s zpt block is a **hardcoded local
variable** (`"function"`), not read from `switches.yaml` or any config. Two methods
exist:

- **"function"** (the one currently wired in): the piecewise‑polynomial fit described
  below.
- **"DNN"**: `src/corrections/zpt_dnn.py` + `MVA_training/zpt_reweight/
  train_dnn_zpt_reweight_dak.py` (C3) — present in the repo but **not** the active
  path unless `whichMethod` is edited. Confirm which one a given run actually used by
  reading the code at the commit in question, not by assuming "function" is permanent.

---

## 3. "function" method — piecewise polynomial per jet multiplicity

`getZptWgts_3region(dimuon_pt, njets, nbins, year, config_path, NanoAODv, sigma_shift)`
(C2):

- Three jet‑multiplicity regions: `njet_0`, `njet_1`, `njet_2` (**inclusive `≥2`**).
- Each region: a 2‑piece fit — polynomial `f0` (order `f0_order`) below
  `polynomial_range.xmin1`, polynomial `f1` (order `f1_order`) plus a vertical `offset`
  between `xmin1` and `xmax1`, then a **linear extrapolation** beyond `xmax1` (slope
  `horizontal_mx`, matched to `f1` at `xmax1` for continuity).
- The weight is **capped to 1.0 for `dimuon_pt ≥ 200 GeV`** (`cutOff_mask`) — no
  correction applied in that region regardless of what the fit would extrapolate to.
- Both a **reco‑njets** weight (`zpt_wgt_reco`) and a **gen‑njets** weight
  (`zpt_wgt_gen`, using `n_genjets_pt30_eta47`) are computed and saved to parquet.
  **[Verify]** which one (or whether both) actually enters the nominal event weight
  downstream — not traced in this review.
- Config file selection: `MiNNLO` in the dataset name → `new_zpt_weights_file_MiNNLO`;
  otherwise → `new_zpt_weights_file_aMCatNLO` (C5, per year/NanoAOD‑version via
  `_load_zpt_config_section`). `data/zpt_rewgt/` holds several parallel tagged YAMLs
  (PySR variants, `IncDY_aMCatNLO_*` production tags, etc.) — **confirm the payload
  path in `SF_filelist.yaml` for the target year points at the intended one**; picking
  the wrong tagged file is an easy, silent mistake.

### Uncertainty (`sigma_shift`)

- Nominal: `sigma_shift = 0.0`.
- 3-tier fallback in `getZptWgts_3region`, most- to least-correct, degrading only as
  far as the payload YAML's own saved fields require:
  1. `_zpt_combined_param_envelope_full_cov` (preferred) — propagates the 4 combined-
     refit parameters (`common_shift`/`low_tilt`/`mid_tilt`/`delta_tail_slope`)
     through their **full covariance matrix** (`sigma(x)^2 = g(x)^T Cov g(x)`), not an
     independence assumption. Needs `combined_fit_covariance` in the YAML
     (`get_polyFit.py` saves it as of the Chebyshev-basis numerical-stability fix).
  2. `_zpt_combined_param_envelope_diagonal` — same 4 parameters, but combined in
     quadrature as if independent (ignores their real, generally small but nonzero,
     correlations). Used when `combined_fit_covariance` is absent but the 4 `*_err`
     fields are present (an older `get_polyFit.py` output).
  3. `_eval_zpt_old_quadrature` — the coarsest, per-coefficient (14 `f0_pN`/`f1_pN`)
     independent-quadrature envelope, used when even the 4 combined-fit fields are
     absent.
- **A config produced by an older derivation silently gets a coarser tier** — check
  which one actually ran for the payload in use (log/inspect which fields exist in
  the YAML for the njet bin being reviewed).
- Gated by `save_zpt_variations` (only computed/saved if set).

---

## 4. Derivation pipeline (C4, C6)

1. **`save_SF_rootFiles.py`** — reads stage‑1 (compacted if available, else `f1_0`)
   parquet, strips any already‑applied Z‑pT weight from DY, builds `Data`, `DY`, and
   `Data/DY` histograms per jet‑multiplicity bin, saves to ROOT workspaces.
2. **`do_f_test.py`** — F‑test to choose the polynomial order per region.
3. **`get_polyFit.py`** — fits the final piecewise function, produces goodness‑of‑fit
   plots, writes the fitted parameters (+ the 4 combined‑refit uncertainty fields) to
   the YAML consumed by `getZptWgts_3region`.
4. Validation: run `plotter/validation_plotter_unified.py` twice — nominal and with
   `--remove_zpt_weights` — and compare `inclusive`/`0`/`1`/`2`‑jet distributions.

Re‑deriving only needs DY reprocessed under a new `run_tag` (not a full stage‑1
rerun); `scripts/splice_sample_across_years.py` splices the new DY output into an
existing label's tree so downstream steps don't need path changes (`docs/ZpT_reweight.md`).

DY sample choice by era (per `docs/ZpT_reweight.md`): Run 2 usually `MiNNLO` or
`aMCatNLO`; Run 3 usually `INCamcatnloFXFX`; 2024 has a special‑cased
`dyTo2Mu_M-50_aMCatNLO` sample.

---

## 5. Review checklist

1. `do_zpt` actually on for the era being reviewed (base `switches.yaml` vs a profile
   like `nominal_zpt`) — don't assume from the profile name alone.
2. Applied to DY MC only.
3. `whichMethod` in the code (not the switches) confirms "function" vs "DNN" for the
   run being reviewed.
4. `new_zpt_weights_file_{aMCatNLO,MiNNLO}` in `SF_filelist.yaml` points at the
   intended tagged YAML for the year/NanoAOD version.
5. `zpt_wgt_reco` vs `zpt_wgt_gen` — confirmed which feeds the nominal weight.
6. Uncertainty method (combined‑envelope vs old‑quadrature fallback) matches what the
   payload's YAML actually supports; `save_zpt_variations` on if variations are needed.
7. Validation done with/without the weight, split by njets, before trusting a new
   derivation.

---

## 6. Evidence summary

| Item | Nature | Source | Established? |
|------|--------|--------|--------------|
| `do_zpt` on for Run 2, off (base) for Run 3 | implementation | C1 | yes |
| DY‑MC‑only application | implementation | C2 | yes |
| Method selection is hardcoded, not a switch | implementation | C2 | yes — flagged |
| Piecewise‑polynomial definition, 200 GeV cap | analysis‑specific | C2 | yes |
| reco vs gen njets weight — which is used downstream | implementation | — | **[Verify]** |
| Uncertainty envelope method + fallback | analysis‑specific | C2 | yes |
| Derivation pipeline steps | analysis‑specific | C4, C6 | yes |

## Last verified

- Local source review: 2026‑09‑18
