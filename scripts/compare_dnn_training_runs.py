#!/usr/bin/env python3
"""
Compare VBF DNN trainings (e.g. serial hpo_optuna.py vs parallel hpo_ray.py on the AF Ray cluster):
per jj region, HPO cost (trials by state, best objective, wall time, summed/mean trial time) and the
final 4-fold model quality (val/eval AUC and weighted AUC, KS overtraining, best epoch), read from
what the trainings already wrote: the Optuna study (optuna_study.db or optuna_journal.log),
optuna_best.json and trained_best_optuna_<HPO label>/fold*/metrics_summary.json. Final-training
wall time = manifest.json mtime - optuna_best.json mtime (the -m dnn chain trains right after HPO).
Writes comparison.md + comparison.json to --out-dir.

How to run (repo root; needs optuna, so the default pixi env):
    ./run_in_pixi.sh default python scripts/compare_dnn_training_runs.py \
      --run NAME=MODEL_BASE_DIR:HPO_LABEL [--run ...] --regions r1,r2 --out-dir validation/...

Example (Aug30 serial 50 trials vs Oct23 Ray 25 trials, full Run 3):
    ./run_in_pixi.sh default python scripts/compare_dnn_training_runs.py \
      --run "Aug30 serial=dnn/trained_models/Run3_nanoAODv12_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation_Systematics/2022preEE-2022postEE-2023-2023BPix-2024-2025-2026:v1_multifold_050Trials" \
      --run "Oct23 Ray=dnn/trained_models/Run3_nanoAODv15_FilterEvents_Oct23_tightPassLepVeto_OfficialRecomendation_IPAdditionalLepVeto_Systematics_LumiSplit/2022preEE-2022postEE-2023-2023BPix-2024-2025-2026:v1_multifold_025Trials" \
      --regions jj_both_central,jj_both_fwd25,jj_one_fwd25_one_central,all \
      --out-dir validation/dnn_ray_vs_serial/Oct23_vs_Aug30
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_study(hpo_dir: Path):
    journal = hpo_dir / "optuna_journal.log"
    db = hpo_dir / "optuna_study.db"
    if journal.exists():
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend

        storage = JournalStorage(JournalFileBackend(str(journal)))
    elif db.exists():
        storage = f"sqlite:///{db.resolve()}"
    else:
        return None, None
    names = optuna.get_all_study_names(storage)
    return optuna.load_study(study_name=names[0], storage=storage), ("journal" if journal.exists() else "sqlite")


def hpo_summary(hpo_dir: Path) -> dict:
    study, kind = load_study(hpo_dir)
    if study is None:
        return {"error": f"no study in {hpo_dir}"}
    trials = study.trials
    states = {}
    for t in trials:
        states[t.state.name] = states.get(t.state.name, 0) + 1
    done = [t for t in trials if t.datetime_start and t.datetime_complete]
    durs = np.array([(t.datetime_complete - t.datetime_start).total_seconds() for t in done])
    start = min(t.datetime_start for t in done)
    end = max(t.datetime_complete for t in done)
    complete = [t for t in trials if t.state.name == "COMPLETE"]
    return {
        "storage": kind,
        "n_trials": len(trials),
        "states": states,
        "best_value": max(t.value for t in complete) if complete else None,
        "best_trial": max(complete, key=lambda t: t.value).number if complete else None,
        "wall_h": (end - start).total_seconds() / 3600.0,
        # Union of trial intervals: excludes idle gaps between resumed serial runs.
        "active_wall_h": _active_seconds(done) / 3600.0,
        "trial_time_sum_h": float(durs.sum()) / 3600.0,
        "trial_time_mean_min": float(durs.mean()) / 60.0,
        "max_parallel": _max_parallel(done),
    }


def _active_seconds(done) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted((t.datetime_start, t.datetime_complete) for t in done):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += (cur_e - cur_s).total_seconds()
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += (cur_e - cur_s).total_seconds()
    return total


def _max_parallel(done) -> int:
    events = sorted([(t.datetime_start, 1) for t in done] + [(t.datetime_complete, -1) for t in done])
    cur = best = 0
    for _, d in events:
        cur += d
        best = max(best, cur)
    return best


def train_summary(train_dir: Path, hpo_dir: Path) -> dict:
    folds = sorted(train_dir.glob("fold*/metrics_summary.json"))
    if not folds:
        return {"error": f"no fold metrics in {train_dir}"}
    rows = []
    for p in folds:
        with open(p) as f:
            m = json.load(f)
        rows.append({
            "fold": p.parent.name,
            "val_auc_w": m["val"]["auc_w"], "eval_auc_w": m["eval"]["auc_w"],
            "val_auc": m["val"]["auc"], "eval_auc": m["eval"]["auc"],
            "ks_sig": m["ks_overtraining"]["sig"], "ks_bkg": m["ks_overtraining"]["bkg"],
            "best_epoch": m["best_epoch"],
        })
    out = {"folds": rows}
    for k in ("val_auc_w", "eval_auc_w", "val_auc", "eval_auc", "ks_sig", "ks_bkg", "best_epoch"):
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
    manifest, best = train_dir / "manifest.json", hpo_dir / "optuna_best.json"
    if manifest.exists() and best.exists():
        out["train_wall_h"] = (os.path.getmtime(manifest) - os.path.getmtime(best)) / 3600.0
    return out


def fmt(x, nd=4):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", action="append", required=True, help="NAME=MODEL_BASE_DIR:HPO_LABEL (repeatable)")
    p.add_argument("--regions", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    runs = []
    for spec in args.run:
        name, rest = spec.split("=", 1)
        base, label = rest.rsplit(":", 1)
        runs.append((name, base, label))
    regions = [r for r in args.regions.split(",") if r]

    result = {}
    for name, base, label in runs:
        result[name] = {}
        for r in regions:
            d = Path(f"{base}_h-peak_vbf_{r}")
            hpo_dir = d / "hpo_optuna" / label
            train_dir = d / f"trained_best_optuna_{label}"
            result[name][r] = {"dir": str(d), "hpo": hpo_summary(hpo_dir), "train": train_summary(train_dir, hpo_dir)}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "comparison.json", "w") as f:
        json.dump({"runs": [{"name": n, "base": b, "hpo_label": lab} for n, b, lab in runs], "result": result}, f, indent=2)

    lines = ["# VBF DNN training comparison", ""]
    for name, base, label in runs:
        lines.append(f"- **{name}**: `{base}_h-peak_vbf_<region>`, HPO label `{label}`")
    lines += ["", "## HPO", "",
              "| region | run | trials (C/P/F) | best obj (val AUC_w) | first-to-last [h] | active wall [h] | sum trial time [h] | mean trial [min] | max parallel |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in regions:
        for name, _, _ in runs:
            h = result[name][r]["hpo"]
            if "error" in h:
                lines.append(f"| {r} | {name} | {h['error']} | | | | | | |")
                continue
            s = h["states"]
            cpf = f"{s.get('COMPLETE', 0)}/{s.get('PRUNED', 0)}/{s.get('FAIL', 0)}"
            lines.append(f"| {r} | {name} | {cpf} | {fmt(h['best_value'])} | {fmt(h['wall_h'], 2)} | "
                         f"{fmt(h['active_wall_h'], 2)} | {fmt(h['trial_time_sum_h'], 2)} | {fmt(h['trial_time_mean_min'], 1)} | {h['max_parallel']} |")
    lines += ["", "## Final model (4 folds: mean [min, max])", "",
              "| region | run | val AUC_w | eval AUC_w | eval AUC | KS bkg | KS sig | best epoch | train wall [h] |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in regions:
        for name, _, _ in runs:
            t = result[name][r]["train"]
            if "error" in t:
                lines.append(f"| {r} | {name} | {t['error']} | | | | | | |")
                continue

            def mm(k, nd=4):
                return f"{t[k]['mean']:.{nd}f} [{t[k]['min']:.{nd}f}, {t[k]['max']:.{nd}f}]"
            lines.append(f"| {r} | {name} | {mm('val_auc_w')} | {mm('eval_auc_w')} | {mm('eval_auc')} | "
                         f"{mm('ks_bkg', 3)} | {mm('ks_sig', 3)} | {mm('best_epoch', 0)} | {fmt(t.get('train_wall_h'), 2)} |")
    with open(out / "comparison.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out}/comparison.md and comparison.json")


if __name__ == "__main__":
    main()
