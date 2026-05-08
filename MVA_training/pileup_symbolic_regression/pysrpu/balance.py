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
    pt_bins=None,
):
    if "process_group" not in df.columns:
        return balance_hs_pu(df, seed=seed, min_train=min_train, max_per_class=max_per_process_class), {
            "mode": "global_hs_pu",
            "before": [],
            "after": [],
        }

    work = df.copy()
    work["label_name"] = np.where(work["y_hs"] > 0.5, "HS", "PU")
    if pt_bins is not None:
        work["pt_bin"] = pd.cut(
            work["pt"],
            bins=pt_bins,
            include_lowest=True,
            right=False,
            duplicates="drop",
        )
    else:
        work["pt_bin"] = "all"

    grouped = work.groupby(["process_group", "label_name"])
    counts = grouped.size().unstack(fill_value=0)
    eligible_groups = counts[(counts.get("HS", 0) > 0) & (counts.get("PU", 0) > 0)].copy()

    if eligible_groups.empty:
        return None, {
            "mode": "process_balanced",
            "before": summarize_process_label_counts(df),
            "after": [],
            "reason": "no process group has both HS and PU examples",
        }

    per_process_avail = {}
    for process_group in sorted(eligible_groups.index):
        proc_df = work[work["process_group"] == process_group]
        bin_counts = (
            proc_df.groupby(["pt_bin", "label_name"], observed=False)
            .size()
            .unstack(fill_value=0)
        )
        avail = {}
        for pt_bin, row in bin_counts.iterrows():
            n_pair = int(min(row.get("HS", 0), row.get("PU", 0)))
            if n_pair > 0:
                avail[pt_bin] = n_pair
        if avail:
            per_process_avail[process_group] = avail

    if not per_process_avail:
        return None, {
            "mode": "process_balanced",
            "before": summarize_process_label_counts(df),
            "after": [],
            "reason": "no process group has overlapping HS/PU statistics in any pt bin",
        }

    def allocate_by_bin(avail_map, target_total):
        total_avail = sum(avail_map.values())
        if total_avail <= target_total:
            return dict(avail_map)

        allocations = {k: 0 for k in avail_map}
        raw = {k: target_total * v / total_avail for k, v in avail_map.items()}
        for k, val in raw.items():
            allocations[k] = min(avail_map[k], int(np.floor(val)))

        assigned = sum(allocations.values())
        if assigned < target_total:
            for k, _ in sorted(
                raw.items(),
                key=lambda item: (item[1] - np.floor(item[1])),
                reverse=True,
            ):
                if assigned >= target_total:
                    break
                if allocations[k] < avail_map[k]:
                    allocations[k] += 1
                    assigned += 1
        return allocations

    process_totals = {
        process_group: sum(avail.values())
        for process_group, avail in per_process_avail.items()
    }
    if equalize_processes:
        target = int(min(min(process_totals.values()), max_per_process_class))
    else:
        target = None

    sampled = []
    after_rows = []
    for idx, process_group in enumerate(sorted(per_process_avail)):
        proc_df = work[work["process_group"] == process_group]
        target_total = target if equalize_processes else min(
            process_totals[process_group], max_per_process_class
        )
        allocations = allocate_by_bin(per_process_avail[process_group], target_total)
        process_taken = 0

        for bin_idx, (pt_bin, n_take) in enumerate(sorted(allocations.items(), key=lambda item: str(item[0]))):
            if n_take <= 0:
                continue
            bin_df = proc_df[proc_df["pt_bin"] == pt_bin]
            hs = bin_df[bin_df["label_name"] == "HS"]
            pu = bin_df[bin_df["label_name"] == "PU"]
            if min(len(hs), len(pu)) < n_take:
                n_take = min(len(hs), len(pu))
            if n_take <= 0:
                continue
            hs_take = hs.sample(n=n_take, random_state=seed + 100 * idx + 2 * bin_idx + 1)
            pu_take = pu.sample(n=n_take, random_state=seed + 100 * idx + 2 * bin_idx + 2)
            sampled.extend([hs_take, pu_take])
            process_taken += n_take

        if process_taken <= 0:
            continue
        after_rows.append({"process_group": process_group, "label": "HS", "count": int(process_taken)})
        after_rows.append({"process_group": process_group, "label": "PU", "count": int(process_taken)})

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
        "stratify_pt_bins": pt_bins is not None,
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
