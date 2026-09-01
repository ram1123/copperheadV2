# VBF DNN — Analysis Discriminant (H→µµ VBF channel)

Responsible: **analysis‑specific MVA — no CMS POG recommendation applies.** This file
documents architecture, inputs, training and application so a review can check
consistency, not compliance.

> Naming note: the **VBF** channel uses a **DNN**; the **ggH** channel uses a **BDT**
> (see `ggh-bdt.md`). This matches `CLAUDE.md` and the repo layout.

## Stored sources

| # | Source | Location | Snapshot / verified |
|---|--------|----------|---------------------|
| C1 | Run 3 trainer | `MVA_training/VBF_run3/` — `preprocess_dnn.py`, `train_dnn.py`, `hpo_optuna.py`, `scan_bins_for_dnn.py`, `utils/{pre_scale_cleaning,scaling_helper}.py` | 2026‑08‑31 |
| C2 | Run 2 legacy trainer | `MVA_training/VBF_run2_legacy/` (`dnn_train*.py`, `dnn_preprocessor.py`, `getModel.py`) | 2026‑08‑31 |
| C3 | Stage‑2 application | `run_stage2_vbf.py` (`DNNWrapper(torch_wrapper)`, `load_torchscript_model`, k‑fold `model_paths`), `src/lib/MVA_functions.py` (`evaluate_dnn`) | 2026‑08‑31 |
| C4 | Config + feature list | `configs/MVA/MVA_subCat_calculation/` (`dnn_features.py`, `dnn_run{2,3}_vbf.yaml`); `docs/DNN_VBF.md` | 2026‑08‑31 |
| C5 | Preprocessing validation | `plotter/plot_vbfdnn_input_features_compare.py`, `plotter/dnn_preprocessing_validation.py` | 2026‑08‑31 |

Classification tags: **[Analysis‑specific]**, **[Implementation]**, **[Verify]**.

---

## 1. Model

- Binary MLP (PyTorch), `BCEWithLogitsLoss` (`returns_logits = true`), optional weighted
  loss (`cfg.data.weights.use_in_loss`). Optuna hyper‑parameter optimisation
  (`hpo_optuna.py`).
- **k‑fold**: `preprocess_dnn.py` writes per‑fold `data_df_{train,validation,evaluation}_{i}.parquet`;
  `train_dnn.py` writes `best.pt` / `last.pt` per fold, exported to **TorchScript** for
  deployment.
- Config `dnn_run3_vbf.yaml` holds the feature list, standardisation and weight options.

---

## 2. Input features (`docs/DNN_VBF.md`)

- Dimuon: `dimuon_mass`, `dimuon_pt`, `dimuon_pt_log`, `dimuon_rapidity`,
  `dimuon_cos_theta_cs`, `dimuon_phi_cs`
- EBE mass resolution: `dimuon_ebe_mass_res`, `dimuon_ebe_mass_res_rel`
- Jets: `jet1_pt/eta/phi`, `jet2_pt/eta/phi`, `jet1_qgl`, `jet2_qgl`
- Dijet: `jj_mass`, `jj_mass_log`, `jj_dEta`
- Soft activity: `htsoft2`, `nsoftjets5`
- VBF topology: `rpt`, `ll_zstar_log`, `mmj_min_dEta`, `pt_centrality`
- Period: `year`

Standardised to mean 0 / std 1 **except `year` and `nsoftjets5`** (kept raw) —
`docs/DNN_VBF.md`.

---

## 3. Application (stage‑2 VBF)

`run_stage2_vbf.py`:

- Load one TorchScript model per fold once per worker (`_MODEL_CACHE`,
  `load_torchscript_model`); run under `torch.inference_mode()`,
  `fold_logits = model(inputs)`.
- Score: `dnn_score = sigmoid(logits)`; the discriminant filled into histograms is
  `arctanh(clip(dnn_score, 0.0, 0.999999))` — stretches the `[0,1)` score for binning.
- CLI: `--model_tag` (must match the training label), `--model_path`.
- `use_nominal_dnn_features_for_systs` (**default true**): shape systematic variations
  are evaluated with the **nominal** DNN inputs — a documented approximation.
- DNN score → VBF subcategory bin edges (`scan_bins_for_dnn.py`,
  `configs/MVA/MVA_subCat_calculation/`); subcategorised templates feed the VBF
  datacards and `run_stats_pipeline_VBF.sh`.

---

## 4. Review checklist

1. Fold assignment at inference matches the training fold split (event‑number parity).
2. Feature list, **order**, and scaler (`scaler`/`.npz`) identical to
   `dnn_run3_vbf.yaml`; `year` and `nsoftjets5` left unstandardised on both sides.
3. `--model_tag` / `--model_path` point at the intended training.
4. Score transform (`sigmoid` → `arctanh(clip(...,0.999999))`) applied consistently in
   binning and in the edge scan.
5. `use_nominal_dnn_features_for_systs` setting is the intended one; if true, the
   approximation is acceptable for the systematics in play.
6. Bin edges from `scan_bins_for_dnn.py` are current for this model.
7. Preprocessing validated (C5): post‑preprocessing features ≈ mean 0 / std 1.

---

## 5. Evidence summary

| Item | Nature | Source | Established? |
|------|--------|--------|--------------|
| MLP architecture, loss, k‑fold, TorchScript export | analysis MVA | C1, C3 | yes |
| Input feature list | analysis MVA | C4 | yes (enumerated §2) |
| Score transform (sigmoid → arctanh/clip) | implementation | C3 | yes |
| Systematics use nominal DNN inputs (default) | implementation | C3 | yes — documented approximation |
| Subcategory bin edges | analysis MVA | C4 | procedure exists; edges must match the model |
| Any CMS‑POG recommendation | — | — | **none — analysis‑specific discriminant** |

## Last verified

- Local source review: 2026‑08‑31
- Model tag / training label: not pinned here — set via `--model_tag` at stage‑2
