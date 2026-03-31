import os
from glob import glob
import json
import numpy as np
import yaml

from .io import expand_inputs, load_parquet
from .features import derive_features, drop_constant_columns, build_df_from_prefix, col, concat_prefixes
from .regions import region_mask_eta
from .balance import balance_hs_pu
from .model import train_pysr, safe_predict
from .thresholds import threshold_and_direction
from .metrics import compute_wp_vs_pt
from .plots import plot_wp_vs_pt


def load_features_from_yaml(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if "features" not in cfg:
        raise ValueError("YAML must contain 'features' key")

    return cfg["features"]


def run_training(args):

    os.makedirs(args.output, exist_ok=True)

    # -------------------------------
    # 1) Load data
    # -------------------------------
    paths = expand_inputs(args.input, args.use_glob)

    df = load_parquet(
        paths,
        use_pyarrow=args.use_pyarrow,
        columns=None,
        max_rows=args.max_rows,
    )

    # -------------------------------
    # 2) Derive features
    # -------------------------------
    # required columns check
    required = ["eta", "pt", "hasMatchedGenJet"]
    prefixes = [f"jet{i}_" for i in range(1, 4 + 1)]
    missing = []
    for p in prefixes:
        for req in required:
            cname = col(p, req, "nominal")
            if cname not in df.columns:
                missing.append(cname)
    if missing:
        raise KeyError("Missing required columns:\n  " + "\n  ".join(missing[:50]))
    df = concat_prefixes(df, prefixes, "nominal")

    df = derive_features(df)

    # -------------------------------
    # 3) Region loop
    # -------------------------------
    regions = ["HEpos", "HEneg", "HFpos", "HFneg"]

    summary_all = []

    for region in regions:

        print(f"\n==== Region: {region} ====")

        mask_region = region_mask_eta(df["eta"].values, region)
        df_region = df[mask_region].copy()

        if len(df_region) < args.min_train:
            print("Too few events, skipping.")
            continue

        # -------------------------------
        # 4) Balance
        # -------------------------------
        df_bal = balance_hs_pu(
            df_region,
            seed=args.seed,
            min_train=args.min_train,
        )

        if df_bal is None:
            print("Balance failed, skipping.")
            continue

        # -------------------------------
        # 5) Select features
        # -------------------------------
        feature_cols = load_features_from_yaml(args.features_yaml)
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

        hs_scores = score_full[y_full > 0.5]
        pu_scores = score_full[y_full <= 0.5]

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
        # 8) pT performance
        # -------------------------------
        pt_bins = args.pt_bins
        hs_eff_vs_pt, pu_rej_vs_pt = compute_wp_vs_pt(
            df_region,
            pass_mask,
            pt_bins,
        )

        # optional ROOT plot
        if args.make_plots:
            plot_wp_vs_pt(
                df_region,
                pass_mask,
                pt_bins,
                args.output,
                f"{region}",
            )

        # -------------------------------
        # 9) Save region summary
        # -------------------------------
        result = {
            "region": region,
            "equation": str(model.get_best()["equation"]),
            "features": feature_cols,
            "threshold": float(thr),
            "direction": direction,
            "pu_rejection": float(pu_rej),
            "n_region": int(len(df_region)),
        }

        summary_all.append(result)

        with open(
            os.path.join(args.output, f"summary_{region}.json"),
            "w",
        ) as f:
            json.dump(result, f, indent=2)

    # -------------------------------
    # 10) Save global summary
    # -------------------------------
    with open(
        os.path.join(args.output, "summary_all.json"),
        "w",
    ) as f:
        json.dump(summary_all, f, indent=2)

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
    }

    return eval(equation, safe_dict, local_dict)



def run_validation(args):

    print("Running validation...")

    os.makedirs(args.output, exist_ok=True)

    # ---------------------------
    # Load dataset
    # ---------------------------
    paths = expand_inputs(args.input, args.use_glob)

    df = load_parquet(
        paths,
        use_pyarrow=args.use_pyarrow,
        columns=None,
        max_rows=args.max_rows,
    )

    # required columns check
    required = ["eta", "pt", "hasMatchedGenJet"]
    prefixes = [f"jet{i}_" for i in range(1, 4 + 1)]
    missing = []
    for p in prefixes:
        for req in required:
            cname = col(p, req, "nominal")
            if cname not in df.columns:
                missing.append(cname)
    if missing:
        raise KeyError("Missing required columns:\n  " + "\n  ".join(missing[:50]))
    df = concat_prefixes(df, prefixes, "nominal")

    df = derive_features(df)

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
            plot_wp_vs_pt(
                df_region,
                pass_mask,
                args.pt_bins,
                args.output,
                f"{region}_validation",
            )

    print("Validation complete.")


def run_rescan(args):

    print("Running threshold rescan...")

    os.makedirs(args.output, exist_ok=True)

    # ---------------------------
    # Load dataset
    # ---------------------------
    paths = expand_inputs(args.input, args.use_glob)

    df = load_parquet(
        paths,
        use_pyarrow=args.use_pyarrow,
        columns=None,
        max_rows=args.max_rows,
    )

    df = concat_prefixes(df, [f"jet{i}_" for i in range(1, 5)], "nominal")
    df = derive_features(df)

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
        hs_scores = score[y > 0.5]
        pu_scores = score[y <= 0.5]

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