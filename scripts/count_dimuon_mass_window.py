#!/usr/bin/env python3
"""
count_dimuon_mass_window.py
---------------------------
Count events in a dimuon-invariant-mass window directly from the stage-1
(compacted) parquet output -- WITHOUT any ggH/VBF category or region
selection.

Why this exists
---------------
`scripts/get_yields.py` only reports the Higgs regions (h-peak 115-135,
h-sidebands 110-115 & 135-150) because its `applyRegionCatCuts` hard-codes
them. The compacted stage-1 parquet itself is NOT mass-window filtered -- it
holds the full `dimuon_mass` spectrum after the baseline dimuon selection
(two good opposite-charge muons, trigger, etc.). So a plain Z-window count,
or any custom window, is just a count over the `dimuon_mass` column.

Only the `dimuon_mass` column is read, so this is fast (~10 s for a full
year of data) and needs no Dask.

Usage
-----
    python scripts/count_dimuon_mass_window.py --input <stage1_output> -y <year> [options]

    --input PATH        stage1_output dir (resolves per-year + prefers
                        `compacted/` over `f1_0/`, like get_yields.py), or a
                        directory holding <sample>/**/*.parquet directly.
    -y, --year YEAR     data-taking year partition (e.g. 2025).
    --samples GLOB ...   sample-dir glob(s) under the resolved path
                        (default: data*). e.g. 'data_[C-G]' drops data_B.
    --mass-min FLOAT    window lower edge in GeV (default: 70).
    --mass-max FLOAT    window upper edge in GeV (default: 110).
    --field NAME        mass branch to cut on (default: dimuon_mass).
    --breakdown         also print the fixed mass-bin distribution table.

Examples
--------
    # 2025 data, Z window 70-110, excluding data_B
    python scripts/count_dimuon_mass_window.py \
        --input /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation/stage1_output \
        -y 2025 --samples 'data_[C-G]' --mass-min 70 --mass-max 110 --breakdown

    # 76-106 window, all data eras
    python scripts/count_dimuon_mass_window.py --input .../stage1_output -y 2025 \
        --samples 'data*' --mass-min 76 --mass-max 106
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# repo root on path so `modules` imports work when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.utils import get_compacted_path  # noqa: E402

# fixed bins for the --breakdown table (GeV); last edge is +inf
BREAKDOWN_EDGES = [0, 60, 70, 76, 106, 110, 115, 135, 150, 200, np.inf]


def resolve_load_path(input_path: Path, year: str) -> str:
    """Mirror get_yields.py: <input>/<year>/f1_0 -> prefer sibling compacted/."""
    year_dir = input_path / year
    if year_dir.is_dir():
        return str(get_compacted_path(year_dir / "f1_0"))
    if input_path.is_dir():
        return str(input_path)
    raise FileNotFoundError(f"Cannot resolve input for year {year}: {input_path}")


def collect_files(load_path: str, sample_globs: list[str]) -> list[str]:
    files: list[str] = []
    for pattern in sample_globs:
        hits = glob.glob(os.path.join(load_path, pattern, "**", "*.parquet"), recursive=True)
        if not hits:
            print(f"[WARN] no parquet files for sample glob '{pattern}' under {load_path}")
        files.extend(hits)
    return sorted(set(files))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", dest="input_path", required=True)
    ap.add_argument("-y", "--year", required=True)
    ap.add_argument("--samples", nargs="+", default=["data*"])
    ap.add_argument("--mass-min", type=float, default=70.0)
    ap.add_argument("--mass-max", type=float, default=110.0)
    ap.add_argument("--field", default="dimuon_mass")
    ap.add_argument("--breakdown", action="store_true")
    args = ap.parse_args()

    load_path = resolve_load_path(Path(args.input_path), args.year)
    print(f"Resolved load path: {load_path}")
    files = collect_files(load_path, args.samples)
    print(f"{len(files)} parquet file(s) across samples {args.samples}")
    if not files:
        sys.exit("No parquet files found.")

    total = 0
    in_window = 0
    mn, mx = np.inf, -np.inf
    hist = np.zeros(len(BREAKDOWN_EDGES) - 1, dtype=np.int64)
    finite_edges = np.array(BREAKDOWN_EDGES[:-1] + [1e12])

    for f in files:
        m = pq.read_table(f, columns=[args.field]).column(args.field).to_numpy()
        m = m[~np.isnan(m)]
        total += m.size
        in_window += int(((m > args.mass_min) & (m < args.mass_max)).sum())
        if m.size:
            mn, mx = min(mn, float(m.min())), max(mx, float(m.max()))
        if args.breakdown:
            hist += np.histogram(m, bins=finite_edges)[0]

    print()
    print(f"field                    : {args.field}")
    print(f"rows on disk (no window)  : {total}")
    print(f"{args.field} range on disk : [{mn:.3f}, {mx:.3f}]")
    print(f"count {args.mass_min:g} < {args.field} < {args.mass_max:g} : {in_window}")

    if args.breakdown:
        print("\nmass distribution (GeV bins):")
        for lo, hi, c in zip(BREAKDOWN_EDGES[:-1], BREAKDOWN_EDGES[1:], hist):
            hi_s = "inf" if np.isinf(hi) else f"{hi:g}"
            print(f"  [{lo:>6g}, {hi_s:>6}) : {c}")


if __name__ == "__main__":
    main()
