#!/usr/bin/env python3
"""
python MVA_training/VBF_new/hpo_optuna.py \
  --config configs/dnn_run3_vbf.yaml \
  --data-dir dnn/trained_models/test_removeClip/2022postEE_h-peak_vbf \
  --out-dir dnn/trained_models/test_removeClip/2022postEE_h-peak_vbf/hpo_optuna \
  --folds 0,1,2 \
  --n-trials 50
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np
import optuna
import yaml

matplotlib.use("Agg")  # important on batch nodes
import matplotlib.pyplot as plt
from modules.utils import logger
from optuna.visualization.matplotlib import (
    plot_edf,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)

# Import your functions/classes from train_dnn.py
from train_dnn import (
    TrainConfig,
    load_config,
    set_seed,
    train_one_fold,
)


def _to_figure(obj: Any):
    """
    Optuna matplotlib plots may return:
      - matplotlib.figure.Figure
      - matplotlib.axes.Axes
      - ndarray/list/tuple of Axes
    Convert any of these into a (fig, axes_list).
    """
    # Figure
    if hasattr(obj, "savefig") and hasattr(obj, "get_axes"):
        fig = obj
        axes = list(fig.get_axes())
        return fig, axes

    # Single Axes
    if hasattr(obj, "figure") and hasattr(obj, "plot"):
        ax = obj
        fig = ax.figure
        return fig, [ax]

    # Array/list of Axes (or nested)
    if isinstance(obj, (list, tuple, np.ndarray)):
        flat: list[Any] = []
        if isinstance(obj, np.ndarray):
            flat = list(obj.ravel())
        else:
            # flatten one level
            for x in obj:
                if isinstance(x, (list, tuple, np.ndarray)):
                    flat.extend(list(np.array(x).ravel()))
                else:
                    flat.append(x)

        axes = [a for a in flat if hasattr(a, "figure")]
        if axes:
            fig = axes[0].figure
            return fig, axes

    raise TypeError(f"Don't know how to convert object to Figure: type={type(obj)}")


def _safe_save_plot(obj: Any, outpath: Path) -> None:
    """
    Save a matplotlib object returned by Optuna plotting utilities.
    """
    fig, _axes = _to_figure(obj)
    try:
        fig.tight_layout()
    except Exception:
        # some axes layouts are weird; don't die
        pass
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def save_study_plots(study, out_dir: str) -> None:
    """
    Save Optuna study summary plots robustly.
    Important: some plots need >=2 completed trials.
    """

    out = Path(out_dir) / "optuna_plots"
    out.mkdir(parents=True, exist_ok=True)

    n_complete = sum(t.state.name == "COMPLETE" for t in study.trials)
    if n_complete < 1:
        logger.warning("[optuna] No COMPLETE trials. Skipping plots.")
        return

    # Always safe with 1 trial
    try:
        _safe_save_plot(plot_optimization_history(study), out / "opt_history.png")
    except Exception as e:
        logger.warning("[optuna] opt_history failed: %s", e)

    # These are much more meaningful with >=2 trials
    if n_complete < 2:
        logger.warning(
            "[optuna] Only %d complete trial(s). Skipping importance/slice/parallel/edf.",
            n_complete,
        )
        return

    try:
        _safe_save_plot(plot_param_importances(study), out / "param_importance.png")
    except Exception as e:
        logger.warning("[optuna] param_importance failed: %s", e)

    try:
        _safe_save_plot(plot_slice(study), out / "slice.png")
    except Exception as e:
        logger.warning("[optuna] slice failed: %s", e)

    try:
        _safe_save_plot(
            plot_parallel_coordinate(study), out / "parallel_coordinate.png"
        )
    except Exception as e:
        logger.warning("[optuna] parallel_coordinate failed: %s", e)

    try:
        _safe_save_plot(plot_edf(study), out / "edf.png")
    except Exception as e:
        logger.warning("[optuna] edf failed: %s", e)

    logger.info("[optuna] Saved plots to: %s", str(out))


def suggest_hidden(trial: optuna.Trial) -> list[int]:
    n_layers = trial.suggest_int("n_layers", 2, 7)
    first_width = trial.suggest_categorical("first_width", (128, 256, 384, 512, 768, 1024, 1280, 1536, 2048))
    shrink = trial.suggest_float("shrink_factor", 0.4, 0.75)

    hidden: list[int] = []
    w = float(first_width)
    for _ in range(n_layers):
        hidden.append(int(round(w)))
        w = max(16.0, w * shrink)

    return hidden


def suggest_hparams(trial: optuna.Trial, cfg: TrainConfig) -> Dict[str, Any]:
    """
    Define the search space. Keep it realistic for your VBF DNN:
    - LR, weight_decay, dropout, hidden sizes, activation, batch_norm, label_smoothing, grad clip.
    - You can expand later (scheduler params, betas, batch size, etc.).
    """

    # ---- architecture choices ----
    activation = trial.suggest_categorical("activation", ["relu", "gelu", "silu"])
    batch_norm = trial.suggest_categorical("batch_norm", [False, True])

    hidden = suggest_hidden(trial)
    trial.set_user_attr("hidden", hidden)
    hidden_key = "_".join(str(x) for x in hidden)
    trial.set_user_attr("hidden_key", hidden_key)

    # dropout: either one value broadcasted, or per-layer
    drop_mode = trial.suggest_categorical("drop_mode", ["one", "per_layer"])
    if drop_mode == "one":
        p = trial.suggest_float("dropout", 0.0, 0.35)
        dropout = [p] * len(hidden)
    else:
        dropout = [
            trial.suggest_float(f"dropout_l{i}", 0.0, 0.5) for i in range(len(hidden))
        ]

    # ---- optimizer / regularization ----
    lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True)

    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.05)
    grad_clip_norm = trial.suggest_float("grad_clip_norm", 0.0, 5.0)

    # (optional) training dynamics
    # keep batch size fixed unless you want to benchmark GPU memory/time carefully
    batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])

    return dict(
        activation=activation,
        batch_norm=batch_norm,
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        grad_clip_norm=grad_clip_norm,
        batch_size=batch_size,
    )


def apply_hparams(cfg: TrainConfig, hp: Dict[str, Any]) -> TrainConfig:
    """Return a new TrainConfig with overrides."""
    new_cfg = replace(
        cfg,
        activation=hp["activation"],
        batch_norm=hp["batch_norm"],
        hidden=hp["hidden"],
        dropout=hp["dropout"],
        lr=float(hp["lr"]),
        weight_decay=float(hp["weight_decay"]),
        label_smoothing=float(hp["label_smoothing"]),
        grad_clip_norm=float(hp["grad_clip_norm"]),
        es_monitor="val_auc_weighted",  # always monitor AUC for HPO
        es_mode="max",  # max if auc, min if loss
        batch_size=int(hp.get("batch_size", cfg.batch_size)),
        # Speed-ups for HPO:
        plots_enable=False,
        save_predictions=False,
        save_history=False,
        save_best=False,
        save_last=False,
    )
    return new_cfg


def objective_factory(
    base_cfg: TrainConfig,
    data_dir: str,
    out_dir: str,
    folds: List[int],
    seed: int,
):
    out_dir = str(Path(out_dir))

    def objective(trial: optuna.Trial) -> float:
        # determinism-ish per trial
        set_seed(seed + trial.number)

        hp = suggest_hparams(trial, base_cfg)
        cfg = apply_hparams(base_cfg, hp)

        # isolate outputs per trial (optional)
        trial_dir = Path(out_dir) / "optuna_trials" / f"trial_{trial.number:05d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        scores = []
        for f in folds:
            score = train_one_fold(
                f,
                cfg,
                data_dir=data_dir,
                out_dir=str(trial_dir),
                trial=trial,  # enables pruning per epoch
                trial_step_offset=f * cfg.epochs,  # unique step across folds
            )
            scores.append(float(score))

        # average across folds
        return float(np.mean(scores))

    return objective


def _reconstruct_hidden_from_params(params: dict) -> list[int]:
    n_layers = int(params["n_layers"])
    first_width = int(params["first_width"])
    shrink = float(params["shrink_factor"])

    hidden: list[int] = []
    w = float(first_width)
    for _ in range(n_layers):
        hidden.append(int(round(w)))
        w = max(16.0, w * shrink)
    return hidden


def write_yaml_override(path: Path, best_trial: optuna.trial.FrozenTrial) -> None:
    params = best_trial.params

    # ------------------
    # Hidden layers
    # ------------------
    hidden = best_trial.user_attrs.get("hidden")
    if hidden is None:
        hidden = _reconstruct_hidden_from_params(params)

    hidden = [int(x) for x in hidden]
    n_layers = len(hidden)

    # ------------------
    # Dropout
    # ------------------
    drop_mode = params.get("drop_mode", "one")

    if drop_mode == "one":
        p = float(params.get("dropout", 0.0))
        dropout = [p] * n_layers
    elif drop_mode == "per_layer":
        dropout = [float(params.get(f"dropout_l{i}", 0.0)) for i in range(n_layers)]
    else:
        raise ValueError(f"Unknown drop_mode: {drop_mode}")

    # ------------------
    # YAML structure
    # ------------------
    override = {
        "meta": {
            "optuna_trial": int(best_trial.number),
            "objective_value": float(best_trial.value),
        },
        "model": {
            "hidden": hidden,
            "activation": str(params["activation"]),
            "dropout": dropout,
            "batch_norm": bool(params["batch_norm"]),
        },
        "training": {
            "batch_size": int(params.get("batch_size", 2048)),
            "optimizer": {
                "lr": float(params["lr"]),
                "weight_decay": float(params["weight_decay"]),
            },
            "loss": {
                "bce_logits": {
                    "label_smoothing": float(params["label_smoothing"]),
                }
            },
            "regularization": {
                "gradient_clip_norm": float(params["grad_clip_norm"]),
            },
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(override, f, sort_keys=False)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Optuna HPO for train_dnn.py")
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--folds", default=None, help="Comma list (e.g. 0,1,2). Default: all cfg folds."
    )
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--timeout-min", type=int, default=None)
    p.add_argument("--study-name", default="vbf_dnn_hpo")
    p.add_argument("--storage", default=None, help="e.g. sqlite:////path/to/study.db")
    p.add_argument("--seed", type=int, default=12345)
    return p


def main() -> None:
    args = build_argparser().parse_args()

    base_cfg = load_config(args.config)
    set_seed(args.seed)

    if args.folds is None:
        folds = list(range(base_cfg.n_folds))
    else:
        folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,
    )

    objective = objective_factory(
        base_cfg=base_cfg,
        data_dir=args.data_dir,
        out_dir=str(out_dir),
        folds=folds,
        seed=args.seed,
    )

    timeout = None if args.timeout_min is None else int(args.timeout_min) * 60

    logger.info(
        "Starting Optuna study '%s' (trials=%d, folds=%s)",
        args.study_name,
        args.n_trials,
        folds,
    )
    study.optimize(
        objective, n_trials=args.n_trials, timeout=timeout, gc_after_trial=True
    )

    save_study_plots(study, out_dir)

    df_trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df_trials.to_csv(out_dir / "optuna_trials.csv", index=False)

    best = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }

    with open(out_dir / "optuna_best.json", "w") as f:
        json.dump(best, f, indent=2)

    write_yaml_override(out_dir / "optuna_best_override.yaml", study.best_trial)

    logger.info("Best value: %.6f", study.best_value)
    logger.info("Best params saved to optuna_best.json and optuna_best_override.yaml")


if __name__ == "__main__":
    main()
