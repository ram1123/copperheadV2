import os
from glob import glob
import json
import numpy as np
import yaml
import pandas as pd

from .io import expand_inputs, load_parquet
from .features import (
    derive_features,
    drop_constant_columns,
    build_df_from_prefix,
    col,
    concat_prefixes,
    required_parquet_columns,
)
from .regions import region_mask_eta
from .balance import (
    balance_hs_pu,
    balance_hs_pu_by_process,
    summarize_process_label_counts,
    summarize_process_rows,
)
from .model import train_pysr, safe_predict
from .thresholds import threshold_and_direction
from .metrics import compute_wp_vs_pt
from .plots import plot_wp_vs_pt, plot_score_real_fake, plot_stacked_before_after, plot_jet_eta_before_after_mask


def load_features_from_yaml(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if "regions" not in cfg:
        raise ValueError("YAML must contain 'regions' key")

    return cfg["regions"]


def expected_process_groups(df):
    if "__process_group" not in df.columns:
        return []
    return sorted(str(x) for x in df["__process_group"].dropna().unique())


def process_report_rows(region, stage, rows):
    out = []
    for row in rows:
        row_out = {"region": region, "stage": stage}
        row_out.update(row)
        out.append(row_out)
    return out


def write_process_report(output_dir, rows):
    if not rows:
        return
    report_df = pd.DataFrame(rows)
    report_df.to_csv(os.path.join(output_dir, "process_balance_report.csv"), index=False)


def build_columns_to_load(features_yaml_path: str):
    features_by_region = load_features_from_yaml(features_yaml_path)
    requested_features = sorted(
        {
            feature
            for region_features in features_by_region.values()
            for feature in region_features
        }
    )
    prefixes = [f"jet{i}_" for i in range(1, 4 + 1)]
    columns = required_parquet_columns(
        requested_features=requested_features,
        prefixes=prefixes,
        variation="nominal",
    )
    return features_by_region, prefixes, columns


def run_training(args):

    os.makedirs(args.output, exist_ok=True)

    # -------------------------------
    # 1) Load data
    # -------------------------------
    paths = expand_inputs(args.input, args.use_glob)
    if not paths:
        raise FileNotFoundError("No parquet inputs were resolved from --input.")
    if args.max_files is not None:
        paths = paths[: args.max_files]
    print(f"Resolved {len(paths)} parquet inputs for training.")
    features_by_region, prefixes, columns_to_load = build_columns_to_load(args.features_yaml)
    print(f"Loading {len(columns_to_load)} parquet columns for training.")

    df = load_parquet(
        paths,
        use_pyarrow=args.use_pyarrow,
        columns=columns_to_load,
        max_rows=args.max_rows,
        attach_source_meta=True,
    )

    # -------------------------------
    # 2) Derive features
    # -------------------------------
    # required columns check
    required = ["eta", "pt", "hasMatchedGenJet"]
    missing = []
    for p in prefixes:
        for req in required:
            cname = col(p, req, "nominal")
            if cname not in df.columns:
                missing.append(cname)
    if missing:
        raise KeyError("Missing required columns:\n  " + "\n  ".join(missing[:50]))
    df = concat_prefixes(df, prefixes, "nominal")
    df["y_hs"] = df["y_hs"].astype(bool)

    df = derive_features(df)

    # ------------------------------------------------
    # Apply pT selection
    # ------------------------------------------------
    pt = df["pt"].values

    turnon_mask = (pt >= args.pt_min) & (pt < args.pt_turnoff)

    df = df[turnon_mask]

    # -------------------------------
    # 3) Region loop
    # -------------------------------
    regions = ["HEpos", "HEneg", "HFpos", "HFneg"]

    summary_all = []
    process_report = []
    expected_groups = expected_process_groups(df)
    
    for region in regions:

        print(f"\n==== Region: {region} ====")

        mask_region = region_mask_eta(df["eta"].values, region)
        df_region = df[mask_region].copy()

        before_process_rows = summarize_process_rows(df_region)
        before_process_label_rows = summarize_process_label_counts(df_region)
        present_before = {row["process_group"] for row in before_process_rows}
        missing_before = sorted(set(expected_groups) - present_before)
        if missing_before:
            print(
                f"WARNING: Region {region} is missing process groups before balancing: {', '.join(missing_before)}"
            )

        process_report.extend(process_report_rows(region, "before_rows", before_process_rows))
        process_report.extend(
            process_report_rows(region, "before_labels", before_process_label_rows)
        )

        if len(df_region) < args.min_train:
            print("Too few events, skipping.")
            continue

        # -------------------------------
        # 4) Balance
        # -------------------------------
        if args.balance_processes:
            df_bal, balance_summary = balance_hs_pu_by_process(
                df_region,
                seed=args.seed,
                min_train=args.min_train,
                max_per_process_class=args.max_per_process_class,
                equalize_processes=(not args.no_equalize_processes),
            )
        else:
            df_bal = balance_hs_pu(
                df_region,
                seed=args.seed,
                min_train=args.min_train,
                max_per_class=args.max_per_class,
            )
            balance_summary = {
                "mode": "global_hs_pu",
                "n_total": int(len(df_bal)) if df_bal is not None else 0,
            }

        if df_bal is None:
            print("Balance failed, skipping.")
            if balance_summary:
                print(json.dumps(balance_summary, indent=2))
            continue
        print(json.dumps(balance_summary, indent=2))

        after_process_rows = summarize_process_rows(df_bal)
        after_process_label_rows = summarize_process_label_counts(df_bal)
        present_after = {row["process_group"] for row in after_process_rows}
        missing_after = sorted(set(expected_groups) - present_after)
        if missing_after:
            print(
                f"WARNING: Region {region} is missing process groups after balancing: {', '.join(missing_after)}"
            )

        process_report.extend(process_report_rows(region, "after_rows", after_process_rows))
        process_report.extend(
            process_report_rows(region, "after_labels", after_process_label_rows)
        )

        # -------------------------------
        # 5) Select features
        # -------------------------------
        feature_cols = features_by_region[region]
        print(f"feature_cols: {feature_cols}")
        feature_cols = drop_constant_columns(df_bal, feature_cols)

        if len(feature_cols) == 0:
            print("No valid features left.")
            continue

        X = df_bal[feature_cols].values
        y = df_bal["y_hs"].values

        # -------------------------------
        # 6) Train PySR
        # -------------------------------
        model = train_pysr(
            X,
            y,
            niterations=args.niterations,
            population_size=args.population_size,
            maxsize=args.maxsize,
            seed=args.seed,
        )

        # -------------------------------
        # 7) Evaluate on FULL region
        # -------------------------------
        X_full = df_region[feature_cols].values
        score_full = safe_predict(model, X_full)

        y_full = df_region["y_hs"].values
        y_mask = y_full.astype(bool)

        hs_scores = score_full[y_mask]
        pu_scores = score_full[~y_mask]

        if len(hs_scores) == 0 or len(pu_scores) == 0:
            print("No HS or PU events in region.")
            continue

        thr, direction, pu_rej = threshold_and_direction(
            hs_scores,
            pu_scores,
            args.hs_eff,
        )

        if thr is None:
            print("Threshold failed.")
            continue

        if direction == "keep_high":
            pass_mask = score_full > thr
        else:
            pass_mask = score_full < thr

        # -------------------------------
        # 8) Save region summary
        # -------------------------------
        result = {
            "region": region,
            "equation": str(model.get_best()["equation"]),
            "features": feature_cols,
            "threshold": float(thr),
            "direction": direction,
            "pu_rejection": float(pu_rej),
            "n_region": int(len(df_region)),
            "pt_min": args.pt_min,
            "pt_turnoff": args.pt_turnoff,
            "valid_pt_range": {
                "min": args.pt_min,
                "max": args.pt_turnoff,
            },
            "balance_summary": balance_summary,
            "missing_process_groups_before": missing_before,
            "missing_process_groups_after": missing_after,
        }

        summary_all.append(result)

        with open(
            os.path.join(args.output, f"summary_{region}.json"),
            "w",
        ) as f:
            json.dump(result, f, indent=2)

    # -------------------------------
    # 9) Save global summary
    # -------------------------------
    with open(
        os.path.join(args.output, "summary_all.json"),
        "w",
    ) as f:
        json.dump(summary_all, f, indent=2)

    write_process_report(args.output, process_report)

    print("\nTraining complete.")


def safe_log1p(x):
    return np.log1p(np.clip(x, 0, None))


def evaluate_equation(df, equation, feature_cols):
    """
    Evaluate PySR equation with correct x0, x1 mapping.
    """

    X = df[feature_cols].values

    local_dict = {}

    # map x0, x1, ...
    for i in range(X.shape[1]):
        local_dict[f"x{i}"] = X[:, i]

    safe_dict = {
        "np": np,
        "abs": np.abs,
        "Abs": np.abs,
        "sqrt": np.sqrt,
        "sqrt_abs": lambda x: np.sqrt(np.abs(x)),
        "log": lambda x: np.log(np.clip(x, 1e-6, None)),
        "log1p": safe_log1p,
        "log1p_abs": lambda x: np.log1p(np.abs(x)),
        "tanh": np.tanh,
        "log_abs": lambda x: np.log(np.abs(x) + 1e-12),
    }

    return eval(equation, safe_dict, local_dict)



def run_validation(args):

    print("Running validation...")

    os.makedirs(args.output, exist_ok=True)

    # ---------------------------
    # Load dataset
    # ---------------------------
    paths = expand_inputs(args.input, args.use_glob)
    if not paths:
        raise FileNotFoundError("No parquet inputs were resolved from --input.")
    if args.max_files is not None:
        paths = paths[: args.max_files]
    print(f"Resolved {len(paths)} parquet inputs for validation.")
    _, prefixes, columns_to_load = build_columns_to_load(args.features_yaml)
    print(f"Loading {len(columns_to_load)} parquet columns for validation.")

    df = load_parquet(
        paths,
        use_pyarrow=args.use_pyarrow,
        columns=columns_to_load,
        max_rows=args.max_rows,
        attach_source_meta=True,
    )

    # required columns check
    required = ["eta", "pt", "hasMatchedGenJet"]
    missing = []
    for p in prefixes:
        for req in required:
            cname = col(p, req, "nominal")
            if cname not in df.columns:
                missing.append(cname)
    if missing:
        raise KeyError("Missing required columns:\n  " + "\n  ".join(missing[:50]))
    df = concat_prefixes(df, prefixes, "nominal")
    df["y_hs"] = df["y_hs"].astype(bool)

    print("\n==== DEBUG y_hs after concat_prefixes ====")

    # if "y_hs" not in df.columns:
    #     print("y_hs column NOT FOUND")
    # else:
    #     y = df["y_hs"]

    #     print("dtype:", y.dtype)
    #     print("unique values (first 20):", np.unique(y)[:20])
    #     print("min:", np.nanmin(y))
    #     print("max:", np.nanmax(y))
    #     print("value counts:")
    #     print(y.value_counts(dropna=False))
    #     print("=========================================\n")
        

    df = derive_features(df)

    # ------------------------------------------------
    # Apply pT selection
    # ------------------------------------------------
    pt = df["pt"].values

    turnon_mask = (pt >= args.pt_min) & (pt < args.pt_turnoff)

    df = df[turnon_mask]


    # ---------------------------
    # Loop over saved summaries
    # ---------------------------
    summary_files = [
        f for f in glob(os.path.join(args.output, "summary_*.json"))
        if not f.endswith("summary_all.json")
    ]    

    for sfile in summary_files:

        with open(sfile) as f:
            summary = json.load(f)

        region = summary["region"]
        equation = summary["equation"]
        feature_cols = summary["features"]
        threshold = summary["threshold"]
        direction = summary["direction"]

        print(f"\nValidating region: {region}")

        mask = region_mask_eta(df["eta"].values, region)
        df_region = df[mask].copy()

        if len(df_region) == 0:
            print("No events in region.")
            continue

        score = evaluate_equation(df_region, equation, feature_cols)
        y_hs = df_region["y_hs"].values

        if direction == "keep_high":
            pass_mask = score > threshold
        else:
            pass_mask = score < threshold

        # pT performance
        hs_eff_vs_pt, pu_rej_vs_pt = compute_wp_vs_pt(
            df_region,
            pass_mask,
            args.pt_bins,
        )

        if args.make_plots:
            # --- Eff/Rej vs pT
            plot_wp_vs_pt(
                df_region,
                pass_mask,
                args.pt_bins,
                args.output,
                f"{region}_validation",
            )
        
            # --- Score distribution real vs fake
            plot_score_real_fake(
                score, y_hs,
                os.path.join(args.output, f"score_real_fake_{region}.pdf"), region
            )

            plot_jet_eta_before_after_mask(df_region, pass_mask, args.output, region)

            # ============================================================
            # New: Stacked before/after plots
            # ============================================================

            # 1 Score before/after
            plot_stacked_before_after(
                score,
                y_hs,
                pass_mask,
                outbase=os.path.join(args.output, f"stack_score_{region}"),
                title=f"{region} score",
                xlab="Score",
                nbins=50,
                logy=False
            )

            # 2 pT sculpting check
            plot_stacked_before_after(
                df_region["pt"].values,
                y_hs,
                pass_mask,
                outbase=os.path.join(args.output, f"stack_pt_{region}"),
                title=f"{region} pT",
                xlab="Jet p_{T} [GeV]",
                nbins=50,
                logy=False
            )

            # 3 eta sculpting check
            plot_stacked_before_after(
                df_region["eta"].values,
                y_hs,
                pass_mask,
                outbase=os.path.join(args.output, f"stack_eta_{region}"),
                title=f"{region} eta",
                xlab="Jet eta",
                nbins=50,
                logy=False
            )

    print("Validation complete.")


def run_rescan(args):

    print("Running threshold rescan...")

    os.makedirs(args.output, exist_ok=True)

    # ---------------------------
    # Load dataset
    # ---------------------------
    paths = expand_inputs(args.input, args.use_glob)
    if not paths:
        raise FileNotFoundError("No parquet inputs were resolved from --input.")
    if args.max_files is not None:
        paths = paths[: args.max_files]
    print(f"Resolved {len(paths)} parquet inputs for rescan.")
    _, prefixes, columns_to_load = build_columns_to_load(args.features_yaml)
    print(f"Loading {len(columns_to_load)} parquet columns for rescan.")

    df = load_parquet(
        paths,
        use_pyarrow=args.use_pyarrow,
        columns=columns_to_load,
        max_rows=args.max_rows,
        attach_source_meta=True,
    )

    df = concat_prefixes(df, [f"jet{i}_" for i in range(1, 5)], "nominal")
    df["y_hs"] = df["y_hs"].astype(bool)

    df = derive_features(df)

    # ------------------------------------------------
    # Apply pT selection
    # ------------------------------------------------
    pt = df["pt"].values

    turnon_mask = (pt >= args.pt_min) & (pt < args.pt_turnoff)

    df = df[turnon_mask]

    # Define HS-eff scan grid
    if args.hs_scan is None:
        hs_grid = np.linspace(0.6, 0.98, 20)
    else:
        hs_grid = args.hs_scan

    summary_files = [
        f for f in glob(os.path.join(args.output, "summary_*.json"))
        if not f.endswith("summary_all.json")
    ]

    for sfile in summary_files:

        with open(sfile) as f:
            summary = json.load(f)

        region = summary["region"]
        equation = summary["equation"]
        feature_cols = summary["features"]

        print(f"\nRescanning region: {region}")

        mask = region_mask_eta(df["eta"].values, region)
        df_region = df[mask].copy()

        if len(df_region) == 0:
            continue

        score = evaluate_equation(df_region, equation, feature_cols)

        y = df_region["y_hs"].values
        hs_scores = score[y]
        pu_scores = score[~y]

        scan_results = []

        for hs_eff_target in hs_grid:

            thr, direction, pu_rej = threshold_and_direction(
                hs_scores,
                pu_scores,
                hs_eff_target,
            )

            if thr is None:
                continue

            scan_results.append({
                "hs_eff_target": float(hs_eff_target),
                "threshold": float(thr),
                "direction": direction,
                "pu_rejection": float(pu_rej),
            })

        # Save scan table
        scan_outfile = os.path.join(
            args.output,
            f"rescan_{region}.json"
        )

        with open(scan_outfile, "w") as f:
            json.dump(scan_results, f, indent=2)

        print(f"Saved scan grid → {scan_outfile}")

    print("Rescan complete.")
