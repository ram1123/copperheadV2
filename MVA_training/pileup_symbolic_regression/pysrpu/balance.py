import numpy as np
import pandas as pd

def balance_hs_pu(df, seed=1, min_train=500, max_per_class=4000):
    hs = df[df["y_hs"] > 0.5]
    pu = df[df["y_hs"] <= 0.5]
    n = min(len(hs), len(pu), max_per_class)
    if n < min_train:
        return None
    hs = hs.sample(n=n, random_state=seed)
    pu = pu.sample(n=n, random_state=seed+1)
    return pd.concat([hs, pu], ignore_index=True).sample(frac=1, random_state=seed+2)


def summarize_process_label_counts(df: pd.DataFrame) -> list[dict]:
    if "process_group" not in df.columns:
        return []
    counts = (
        df.assign(label=np.where(df["y_hs"] > 0.5, "HS", "PU"))
        .groupby(["process_group", "label"])
        .size()
        .reset_index(name="count")
        .sort_values(["process_group", "label"])
    )
    return counts.to_dict(orient="records")


def summarize_process_rows(df: pd.DataFrame) -> list[dict]:
    if "process_group" not in df.columns:
        return []
    counts = (
        df.groupby("process_group")
        .size()
        .reset_index(name="count")
        .sort_values("process_group")
    )
    return counts.to_dict(orient="records")


def balance_hs_pu_by_process(
    df: pd.DataFrame,
    seed: int = 1,
    min_train: int = 500,
    max_per_process_class: int = 2000,
    equalize_processes: bool = True,
):
    if "process_group" not in df.columns:
        return balance_hs_pu(df, seed=seed, min_train=min_train, max_per_class=max_per_process_class), {
            "mode": "global_hs_pu",
            "before": [],
            "after": [],
        }

    work = df.copy()
    work["label_name"] = np.where(work["y_hs"] > 0.5, "HS", "PU")
    grouped = work.groupby(["process_group", "label_name"])
    counts = grouped.size().unstack(fill_value=0)
    eligible = counts[(counts.get("HS", 0) > 0) & (counts.get("PU", 0) > 0)].copy()

    if eligible.empty:
        return None, {
            "mode": "process_balanced",
            "before": summarize_process_label_counts(df),
            "after": [],
            "reason": "no process group has both HS and PU examples",
        }

    if equalize_processes:
        target = int(min(eligible["HS"].min(), eligible["PU"].min(), max_per_process_class))
    else:
        target = int(max_per_process_class)

    sampled = []
    after_rows = []
    for idx, process_group in enumerate(sorted(eligible.index)):
        proc_df = work[work["process_group"] == process_group]
        hs = proc_df[proc_df["label_name"] == "HS"]
        pu = proc_df[proc_df["label_name"] == "PU"]

        n_hs = min(len(hs), target)
        n_pu = min(len(pu), target)
        if equalize_processes:
            n_take = min(n_hs, n_pu)
            n_hs = n_take
            n_pu = n_take

        if min(n_hs, n_pu) <= 0:
            continue

        hs_take = hs.sample(n=n_hs, random_state=seed + 10 * idx + 1)
        pu_take = pu.sample(n=n_pu, random_state=seed + 10 * idx + 2)
        sampled.extend([hs_take, pu_take])
        after_rows.append({"process_group": process_group, "label": "HS", "count": int(n_hs)})
        after_rows.append({"process_group": process_group, "label": "PU", "count": int(n_pu)})

    if not sampled:
        return None, {
            "mode": "process_balanced",
            "before": summarize_process_label_counts(df),
            "after": [],
            "reason": "sampling produced no rows",
        }

    out = pd.concat(sampled, ignore_index=True)
    if len(out) < min_train:
        return None, {
            "mode": "process_balanced",
            "before": summarize_process_label_counts(df),
            "after": after_rows,
            "reason": f"balanced sample too small ({len(out)} < {min_train})",
        }

    out = out.drop(columns=["label_name"]).sample(frac=1, random_state=seed + 999).reset_index(drop=True)
    return out, {
        "mode": "process_balanced",
        "equalize_processes": bool(equalize_processes),
        "max_per_process_class": int(max_per_process_class),
        "before": summarize_process_label_counts(df),
        "after": after_rows,
        "n_total": int(len(out)),
    }

def finalize_feature_columns(df, requested):
    df = df.copy()
    for f in requested:
        if f.startswith("isnan_"):
            base = f.replace("isnan_", "")
            if base in df.columns:
                df[f] = df[base].isna().astype(np.float32)
    cols = [f for f in requested if f in df.columns]
    # drop constant columns per-region to avoid trivial formulas
    keep = []
    for c in cols:
        v = df[c].to_numpy()
        if np.nanstd(v) > 0:
            keep.append(c)
    return df.dropna(subset=["y_hs"] + keep), keep
