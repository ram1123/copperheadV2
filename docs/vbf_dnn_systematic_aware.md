# Systematic-aware VBF DNN (adversarial loss)

The nominal VBF DNN (`MVA_training/VBF_run3/`, see [DNN_VBF.md](DNN_VBF.md)) is trained on nominal
inputs only. The systematic-aware option adds a loss term that also evaluates the network on the
shape-varied inputs (JES/JER/muon scale and resolution). It pushes the varied scores towards the
nominal score, so the DNN score shifts less under those systematics. The aim is smaller shape
uncertainties in the datacard, possibly at some cost in nominal separation.

This is an analysis-specific method, not a CMS POG recommendation. Before a model trained this way
is used for results, its λ and its effect on the expected significance and the impacts have to be
validated.

## Method

For each training batch the loss is

```
L = L_BCE(p_nom, y) + λ · Σ_i [ 2 · L_BCE(p_i, p_nom) + L_BCE(p_i, y) ]
```

- `p_nom` is the score on the nominal inputs.
- `p_i` is the score with the inputs of variation `i`. Up and down count as separate variations.
- `L_BCE` is the training loss, with the same event weights and `pos_weight` as the nominal term.
- The first term inside the sum (consistency) pulls `p_i` towards `p_nom`.
- The second term (label) keeps `p_i` discriminating.

Implementation details:

- The variation forwards run with the model in `eval()` mode, so dropout and BatchNorm statistics
  are untouched. Gradients still flow through them.
- Only the train split carries the variation inputs. Validation and evaluation metrics, early
  stopping and the HPO objective use the nominal score (`val_auc_weighted`), so their numbers stay
  comparable with a nominal training.
- The exported model is an ordinary MLP with the same inputs. Stage-2 applies it exactly like a
  nominal model.

The code is in `MVA_training/VBF_run3/train_dnn.py` (`AdversarialConfig`, `adversarial_penalty`,
`load_variation_spec`). The CLI flags are in `MVA_training/VBF_run3/adversarial_cli.py` and are
shared by `train_dnn.py`, `hpo_optuna.py` and `hpo_ray.py`.

### Variations

`preprocess_dnn.py --include-systematic-variations` writes, for each training feature that a
variation shifts, one extra column per variation (`<feature>__<variation>`). It also writes
a `systematic_variations` block to `preprocess_manifest.json`. The variation list is the one
stage-2 uses (`modules.systematics.stage2_shape_variations`). Year-decorrelated JES sources
(`Absolute_2024`, …) are mapped onto one year-agnostic slot each (`Absolute_yearDecor`).

| `--variation-set` | Variations                                                     |
| ----------------- | -------------------------------------------------------------- |
| `full` (default)  | Every stage-2 shape variation. Run 3 has 40: the JES sources (regrouped and `Total`), `jer1`–`jer6`, `mu_scale`, `mu_resol`, each up and down. |
| `sweep`           | `Total` and `mu_roccor`, up and down. Run 3 has no `mu_roccor`, so on Run 3 this set is only JES `Total` up and down. |

Things to keep in mind:

- `full` contains both the split JES sources and `Total`, so JES gets more weight in the sum than
  JER and the muon terms.
- Weight-only systematics (b-tag, pileup, …) do not change the inputs, so this loss does not
  touch them.

### Flags (`train_dnn.py`, `hpo_optuna.py`, `hpo_ray.py`)

| Flag | Meaning |
| ---- | ------- |
| `--use_adversarial` | Turn the term on. `--data-dir` must hold variation-augmented parquets. |
| `--adversarial-lambda λ` | Weight of the sum. The default of 0 computes and logs the penalty without training on it (calibration probe). |
| `--adversarial-label-only` | Keep only `Σ_i L_BCE(p_i, y)`: step 2 of the three-step schedule. |
| `--adversarial-high-score-cut C` | Apply the consistency term only to events with `arctanh(p_nom) > C`. This changes the λ scale. |
| `--adversarial-mask-label-term` | Apply the cut above to the label term as well. |
| `--adversarial-detach-nominal` | Stop the gradient through `p_nom` in the consistency term. |
| `--adversarial-consistency-smoothing inherit\|none` | Label smoothing inside the consistency term. |
| `--adversarial-variation-chunk N` | Process the variations N at a time if GPU memory runs out. |
| `--init-from DIR` (`train_dnn.py` only) | Warm start each fold from `DIR/fold<i>/best.pt`, for example a converged λ=0 model. |

## Running it through the pipeline

`run_analysis_pipeline.sh -m dnn|dnn_pre|dnn_hpo|dnn_train` and `scripts/run_dnn_trainings.sh`
read these environment variables:

| Variable | Effect |
| -------- | ------ |
| `DNN_SYST_AWARE=1` | Preprocessing adds `--include-systematic-variations`; HPO and training add `--use_adversarial`. |
| `DNN_VARIATION_SET` | `full` (default) or `sweep`. |
| `DNN_ADV_LAMBDA` | λ. Required for `dnn`, `dnn_hpo` and `dnn_train` when `DNN_SYST_AWARE=1`. |
| `DNN_ADV_EXTRA_ARGS` | Further flags from the table above, passed through as-is. |
| `RAY_RAM_PER_WORKER_GB` | RAM per Ray GPU worker (default 32). The variation inputs are held in RAM, see below. |

With `DNN_SYST_AWARE=1`, everything goes to `dnn/trained_models/<label>_SystAware/...` instead of
`<label>/...`. `preprocess_dnn.py` refuses to write variation columns into a directory that
already holds nominal fold parquets. Different λ values reuse the same preprocessed parquets and
are told apart by `HPO_LABEL` (and `TRAIN_LABEL`). For example, `HPO_LABEL=v1_multifold_100Trials_lam0p05`
gives `trained_best_optuna_v1_multifold_100Trials_lam0p05`.

For stage-2 or compact `-D`, set `DNN_SYST_AWARE=1` and the same `HPO_LABEL` to load the
systematic-aware model, or point `MODEL_LABEL` at `<label>_SystAware` directly. The bin scan for
`configs/MVA/VBF/dnn_binning.yaml` has to be redone for the new model, as for any new model.

### Storage and memory

- `/work/users` is over quota for DNN outputs. Before preprocessing, create the label directory on
  `/work/projects` and symlink it (see the commands below). The variation columns make the fold
  parquets about 10–15 times larger than the nominal ones. The variation columns are written for
  every split.
- The DNN keeps each fold fully in RAM, and an HPO worker caches all its folds. For the largest
  region (`jj_one_fwd25_one_central`, about 1M training events per fold), the variation inputs of
  4 folds come to several tens of GB. Use `RAY_RAM_PER_WORKER_GB=64`. `jj_both_central` and
  `jj_both_fwd25` are 3–4 times smaller. The inclusive `all` region (about 3M events per fold) is
  unlikely to fit with `full`.
- GPU memory is not a constraint: data stays on the CPU and is copied per batch.

## Choosing λ (calibration probe)

No λ has been calibrated for Run 3 yet. Before an HPO, run one epoch with λ=0 and
`--use_adversarial`. It logs `nominal_loss` and `penalty` per epoch without changing the
training. Run the same probe without `--use_adversarial` to get the time per epoch without the
term. Pick λ so that `λ·penalty` is a modest fraction of `nominal_loss`, then compare a few λ
values on expected significance and impacts:

```bash
D=dnn/trained_models/<label>_SystAware/<years>_h-peak_vbf_<region>
./run_in_pixi.sh default python MVA_training/VBF_run3/train_dnn.py --config configs/dnn_run3_vbf.yaml \
    --data-dir $D --out-dir $D/probe_lam0 --folds 0 --epochs 1 --use_adversarial --adversarial-lambda 0
```

## Caveats

- The HPO objective is the nominal validation AUC at a fixed λ. The HPO therefore does not tune λ
  itself, and a larger λ always lowers the objective. Compare λ values with stage-2/3 significance
  and impacts, not with the HPO value.
- `optuna_best.json` does not record λ. Keep λ in `HPO_LABEL` so the final training (which reads
  `DNN_ADV_LAMBDA` again) cannot silently use a different λ from the HPO.
  `train_dnn.py` writes its full command line to `manifest.json`.
