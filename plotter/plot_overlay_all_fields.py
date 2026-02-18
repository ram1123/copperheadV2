#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import awkward as ak
import dask_awkward as dak
import matplotlib.pyplot as plt
import numpy as np
from modules.dask_utils import close_dask_client, get_dask_client

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
# Sample can be single file or glob pattern
SAMPLES: Dict[str, str] = {
    # "VBF": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_02Feb_FilterJets/"
    # "stage1_output/2022postEE/f1_0/vbf_powheg_dipole/0/*.parquet",
    # "EWK": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_02Feb_FilterJets/"
    # "stage1_output/2022postEE/f1_0/ewk_lljj/0/*.parquet",
    # "DY": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_02Feb_FilterJets/"
    # "stage1_output/2022postEE/f1_0/dyTo2L_M-50_incl/0/*.parquet",
    # "TT": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_02Feb_FilterJets/"
    # "stage1_output/2022postEE/f1_0/ttjets_sl/0/*.parquet",
    "new_JETID": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterJetsHorn30GeV_JetIDFix/stage1_output/2024/f1_0/vbf_powheg/0/*.parquet",
    "old_JETID": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_15Feb_FilterJetsHorn30GeV/stage1_output/2024/f1_0/vbf_powheg/0/*.parquet",
    # "new_JETID": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn30GeV_JetIDFix/stage1_output/2022postEE/f1_0/vbf_powheg_dipole/0/*.parquet",
    # "old_JETID": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_15Feb_FilterJetsHorn30GeV/stage1_output/2022postEE/f1_0/vbf_powheg_dipole/0/*.parquet",
    # "new_JETID": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn30GeV_JetIDFix/stage1_output/2023/f1_0/vbf_powheg/0/*.parquet",
    # "old_JETID": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_15Feb_FilterJetsHorn30GeV/stage1_output/2023/f1_0/vbf_powheg/0/*.parquet",
}

OUTDIR = Path(
    "validation/sanity_checks/plot_overlay_all_fields/CompareOldVsNewJetID/2024_vbf_powheg_NewJETIDandMuonID"
)
NBINS = 60

MAX_POINTS_FOR_RANGE = 200_000
MAX_POINTS_FOR_HIST = 600_000
SENTINEL = -999.0


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _is_numeric_leaf(field_arr: dak.Array) -> bool:
    try:
        t = ak.type(field_arr._meta)  # noqa: SLF001
        while hasattr(t, "content"):
            t = t.content
        return isinstance(t, ak.types.NumpyType) and t.primitive in (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float16",
            "float32",
            "float64",
            "bool",
        )
    except Exception:
        return False


def _safe_name(s: str) -> str:
    return (
        s.replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace("(", "_")
        .replace(")", "_")
    )


def _stride_sample(arr: np.ndarray, max_points: int) -> np.ndarray:
    if arr.size <= max_points:
        return arr
    step = max(1, arr.size // max_points)
    return arr[::step]


def _flatten_numeric_to_numpy(x: dak.Array, max_points: int) -> np.ndarray:
    """
    Convert a dak.Array field to a flat 1D numpy array (drops None, flattens jagged),
    filters finite and SENTINEL, and returns at most `max_points` values.
    """
    # Drop None
    x = x[~ak.is_none(x)]

    # If jagged/list, flatten
    try:
        t = ak.type(x._meta)  # noqa: SLF001
        if "var *" in str(t) or "list" in str(t):
            x = ak.flatten(x, axis=None)
    except Exception:
        try:
            x = ak.flatten(x, axis=None)
        except Exception:
            pass

    x_ak = x.compute()
    if x_ak is None:
        return np.array([], dtype=np.float64)

    try:
        x_ak = ak.flatten(x_ak, axis=None)
    except Exception:
        pass

    out = ak.to_numpy(x_ak)

    # MaskedArray safety
    if np.ma.isMaskedArray(out):
        out = out.compressed()
    else:
        out = np.asarray(out)

    # Finite + remove sentinel
    m = np.isfinite(out)
    if SENTINEL is not None:
        m = m & (out != SENTINEL)
    out = out[m]

    if out.size == 0:
        return np.array([], dtype=np.float64)

    out = _stride_sample(out, max_points)
    return out.astype(np.float64, copy=False)


def _auto_range(arrs: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """
    Determine plotting range from the union of samples using robust percentiles.
    """
    merged = []
    for a in arrs.values():
        if a is not None and a.size:
            merged.append(_stride_sample(a, MAX_POINTS_FOR_RANGE))
    if not merged:
        return (0.0, 1.0)

    allv = np.concatenate(merged)
    allv = allv[np.isfinite(allv)]
    if allv.size == 0:
        return (0.0, 1.0)

    lo = np.percentile(allv, 1.0)
    hi = np.percentile(allv, 99.0)

    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.min(allv))
        hi = float(np.max(allv))
        if lo == hi:
            lo -= 0.5
            hi += 0.5

    return float(lo), float(hi)


def _save_overlay(arrs: Dict[str, np.ndarray], field: str, outdir: Path) -> None:
    """
    Overlay plot for one variable: sample lists, normalized to unity.
    """
    # Drop empty ones (but keep at least 2 for a comparison)
    arrs = {k: v for k, v in arrs.items() if v is not None and v.size > 0}
    if len(arrs) < 1:
        return

    lo, hi = _auto_range(arrs)
    name = _safe_name(field)

    # Linear
    fig, ax = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    for label, arr in arrs.items():
        ax.hist(
            arr,
            bins=NBINS,
            range=(lo, hi),
            histtype="step",
            linewidth=1.6,
            density=True, # NOTE: NORMALIZED TO UNITY
            label=f"{label} (N={arr.size})",
        )
    ax.set_xlabel(field)
    ax.set_ylabel("Unit-normalized density")
    ax.set_title(field)
    ax.grid(True)
    ax.legend()
    fig.savefig(outdir / f"{name}.pdf")
    plt.close(fig)

    # Log-y
    fig, ax = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    for label, arr in arrs.items():
        ax.hist(
            arr,
            bins=NBINS,
            range=(lo, hi),
            histtype="step",
            linewidth=1.6,
            density=True, # NOTE: NORMALIZED TO UNITY
            label=f"{label} (N={arr.size})",
        )
    ax.set_xlabel(field)
    ax.set_ylabel("Unit-normalized density")
    ax.set_title(f"{field} (log-y)")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()
    fig.savefig(outdir / f"{name}_log.pdf")
    plt.close(fig)


def _parse_vars_arg(vars_arg: Optional[List[str]]) -> List[str]:
    """
    Accept:
      --vars a b c
      --vars a,b,c
      --vars a,b c   (mixed)
    """
    if not vars_arg:
        return []
    out: List[str] = []
    for item in vars_arg:
        if not item:
            continue
        parts = [p.strip() for p in item.split(",") if p.strip()]
        out.extend(parts)
    # preserve order but unique
    seen = set()
    uniq = []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def _read_vars_file(path: Optional[str]) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--vars-file not found: {p}")
    vals = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        vals.append(s)
    # unique keep order
    seen = set()
    uniq = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot all or selected numeric parquet fields."
    )
    parser.add_argument(
        "--outdir",
        default=str(OUTDIR),
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot all numeric fields (default if --vars/--vars-file not provided).",
    )
    parser.add_argument(
        "--vars",
        nargs="*",
        default=None,
        help="List of variables to plot. Accepts space-separated and/or comma-separated.",
    )
    parser.add_argument(
        "--vars-file",
        default=None,
        help="Text file with variables to plot (one per line; # comments allowed).",
    )
    parser.add_argument(
        "--use-gateway",
        action="store_true",
        help="Use Dask gateway for distributed execution (default: False).",
    )
    parser.add_argument(
        "--cluster-index",
        type=int,
        default=1,
        help="Dask gateway cluster index passed to get_dask_client().",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    client = get_dask_client(use_gateway=args.use_gateway, cluster_index=args.cluster_index)
    print("[INFO] Dask client:", client)

    # Load all samples as dask-awkward arrays
    events = {}
    for k, path in SAMPLES.items():
        print(f"[INFO] Reading {k}: {path}")
        events[k] = dak.from_parquet(path)

    # Common fields across samples
    fields_sets = {k: set(v.fields) for k, v in events.items()}
    common_fields = sorted(set.intersection(*fields_sets.values()))
    print(f"[INFO] Common top-level fields: {len(common_fields)}")

    # Keep numeric leaves
    numeric_fields = []
    for f in common_fields:
        ok = True
        for k in events:
            try:
                x = events[k][f]
            except Exception:
                ok = False
                break
            if not _is_numeric_leaf(x):
                ok = False
                break
        if ok:
            numeric_fields.append(f)

    print(f"[INFO] Numeric common fields available: {len(numeric_fields)}")

    # --- Selection logic ---
    requested = []
    requested.extend(_read_vars_file(args.vars_file))
    requested.extend(_parse_vars_arg(args.vars))

    # If user explicitly gives vars, use them; else plot all (default)
    if requested and not args.all:
        numeric_set = set(numeric_fields)
        chosen = [v for v in requested if v in numeric_set]
        missing = [v for v in requested if v not in numeric_set]

        if missing:
            print("[WARN] Requested variables not found (or non-numeric / not common):")
            for m in missing:
                print("   -", m)

        fields_to_plot = chosen
        print(f"[INFO] Plotting requested variables: {len(fields_to_plot)}")
    else:
        fields_to_plot = numeric_fields
        print(f"[INFO] Plotting ALL numeric variables: {len(fields_to_plot)}")

    print("[INFO] Writing plots to:", outdir.resolve())

    n_done = 0
    for f in fields_to_plot:
        arrs = {}
        for label in events:
            try:
                arrs[label] = _flatten_numeric_to_numpy(
                    events[label][f], max_points=MAX_POINTS_FOR_HIST
                )
            except Exception as e:
                print(f"[WARN] {label}: failed field {f}: {e}")
                arrs[label] = np.array([], dtype=np.float64)

        _save_overlay(arrs, f, outdir)

        n_done += 1
        if n_done % 25 == 0:
            print(f"[INFO] Plotted {n_done}/{len(fields_to_plot)} fields...")

    print(f"[INFO] Done. Plotted: {n_done} fields.")
    close_dask_client()


if __name__ == "__main__":
    main()
