# ggH BDT — Analysis Discriminant (H→µµ ggH channel)

Responsible: **analysis‑specific MVA — no CMS POG recommendation applies.** This file
documents architecture, inputs, training and application so a review can check
consistency, not compliance.

> Naming note: the **ggH** channel uses a **BDT**; the **VBF** channel uses a **DNN**
> (see `vbf-dnn.md`). This matches `CLAUDE.md` and the repo layout. If a request says
> "DNN for ggH / BDT for VBF" it has them swapped.

## Stored sources

| # | Source | Location | Snapshot / verified |
|---|--------|----------|---------------------|
| C1 | Trainer (git submodule → `github.com/ram1123/Run2_MVA_trainer`) | `MVA_training/ggH_BDT/` — `my_trainer_withWeight_gpu.py`, `plot_roc_byFoldNYear.py`, `get_vbf_bins.py`, `modules/variables.py`, `modules/workflow.py` | 2026‑08‑31 |
| C2 | Stage‑2 application | `run_stage2.py` (`evaluate_bdt`, `load_or_create_bdt_edges`), `src/lib/MVA_functions.py` (`evaluate_bdt`) | 2026‑08‑31 |
| C3 | Subcategory edges | `configs/MVA/ggH/BDT_edgeCalculator.ipynb`; `BDT_edges.yaml` (produced/loaded at stage‑2) | 2026‑08‑31 |
| C4 | Procedure docs | `docs/ggH_BdtCategoryEdgeCalculation.md`, `docs/ggH_steps_after_stage1*.md`, `docs/ggH_dataCardGeneration.md`, `docs/ggH_validation.md` | 2026‑08‑31 |

Classification tags: **[Analysis‑specific]**, **[Implementation]**, **[Verify]**.

---

## 1. Model

- Gradient‑boosted decision tree. The stage‑2 loader is
  `bdt_model = pickle.load(open(model_path, "rb"))` (`src/lib/MVA_functions.py` L105) —
  a **pickled** classifier object. Confirm the framework/version (XGBoost vs sklearn
  `GradientBoosting`) from the submodule's `modules/workflow.py:classifier_train` and
  pin it. **[Verify]**
- Per‑year models: `{model_base_path}/output/bdt_{model_name}_{year}`.
- **2016preVFP + 2016postVFP are merged to `"2016"` for training/inference**
  (`run_stage2.py` L197–203).
- k‑fold training (`plot_roc_byFoldNYear.py`); MC event weights used
  (`my_trainer_withWeight_*`), with negative‑weight handling
  (`PairNAnnhilateNegWgt_inChunks`) and mass‑decorrelation reweighting
  (`reweightMassToFlat`, `reweightMassToTargetDist_workflow`).
- ggH‑channel event selection applied before training (`apply_gghChannelSelection`).

---

## 2. Input features (`MVA_training/ggH_BDT/modules/variables.py:training_features`)

```
dimuon_cos_theta_cs, dimuon_phi_cs, dimuon_rapidity, dimuon_pt,
jet1_eta, jet2_eta, jet1_pt, jet2_pt,
jj_dEta, jj_dPhi, jj_mass,
mmj_min_dEta, mmj_min_dPhi,
mu1_eta, mu1_pt_over_mass, mu2_eta, mu2_pt_over_mass,
zeppenfeld, njets, year
```

`dimuon_mass` is deliberately **not** an input (mass‑decorrelated discriminant).

Training samples (`training_samples`): background = DY (aMC@NLO + inclusive), tt
(dl/sl/fh), single‑top tW, WW/WZ/ZZ, EWK mµµjj; signal = ggH (powheg+PS) **and** VBF
(powheg) — VBF is included in the signal class.

---

## 3. Application (stage‑2)

- `run_stage2.py` → `evaluate_bdt(events, variation, model_name, training_features,
  parameters)` — run once per systematic `variation`; `training_features` order must
  match training.
- The BDT score is divided into ggH **subcategories** using score edges from
  `BDT_edges.yaml` (`load_or_create_bdt_edges`; a **dummy** edges file is auto‑created
  if none exists — a real one must be produced by `BDT_edgeCalculator.ipynb`, C3).
- Subcategorised templates feed the ggH datacards (`stage2/ggH_datacard/`,
  `docs/ggH_dataCardGeneration.md`).

---

## 4. Review checklist

1. Model framework/version pinned; `pickle` payload provenance = a known submodule
   commit (check `git_state.json` if present).
2. Training‑year mapping correct (2016 halves merged).
3. Stage‑2 `training_features` list **and order** identical to training.
4. `BDT_edges.yaml` is a real computed file, not the auto dummy.
5. Systematic variations each re‑evaluate the BDT with the varied inputs.
6. Signal‑class composition (ggH + VBF) understood when interpreting the ggH category.
7. Mass decorrelation validated (no sculpting of `dimuon_mass` in background).

---

## 5. Evidence summary

| Item | Nature | Source | Established? |
|------|--------|--------|--------------|
| BDT architecture / framework | analysis MVA | C1, C2 | pickled classifier; **framework/version [Verify]** |
| Input feature list | analysis MVA | C1 | yes (enumerated §2) |
| Per‑year models, 2016 merge | implementation | C2 | yes |
| Subcategory edges | analysis MVA | C3 | procedure documented; edges file must be real |
| Any CMS‑POG recommendation | — | — | **none — analysis‑specific discriminant** |

## Last verified

- Local source review: 2026‑08‑31
- Submodule commit / framework version: not pinned in this file — check
  `MVA_training/ggH_BDT` submodule state
