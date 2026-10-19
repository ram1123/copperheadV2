#!/usr/bin/env python3
"""
Summarize stage-1 cutflows for every sample of one year as a sample x cut table,
read from the merged per-sample cutflow_merged_<sample>.json files stage-1 writes
under stage1_output/<year>/cutflow/. Uses the "cumulative" counts (events passing
all cuts up to and including that one).

The dimuon mass-window cuts (Z window, H peak, H sidebands) are dropped from the
table: they are mutually exclusive, so their cumulative counts are not meaningful.

--format:
    counts   raw cumulative event counts (default)
    percent  % of events left w.r.t. the "initial" count of that sample
    both     "count (percent%)" in each cell

The table is always saved as both .csv and .tex, by default to
validation/cutflow/<label>/<year>/cutflow_<year>_<format>.{csv,tex} (override with --csv/--tex).

How to run (repo root, needs only pandas):
    ./run_in_pixi.sh default python scripts/cutflow_summary.py <dir> [--format counts|percent|both] [--csv OUT.csv] [--tex OUT.tex]

<dir> may be the cutflow/ directory itself, stage1_output/<year>/, or its compacted/ subdirectory.

Example:
    ./run_in_pixi.sh default python scripts/cutflow_summary.py \
        /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterEvents_Oct08_tightPassLepVeto_DefaultjetPt25GeV_PUDNN_JERStrat4_Systematics_LumiSplit/stage1_output/2024/compacted \
        --format both
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# mutually exclusive mass regions -> cumulative AND across them is always 0
MASS_WINDOW_CUTS = [
    "dimuon_mass_window_76_106",
    "h_peak_115_135",
    "h_sidebands_110_115_135_150",
    "h_sidebands_106_115_135_150",
]


def resolve_cutflow_dir(path):
    path = os.path.abspath(path.rstrip("/"))
    for cand in (path, os.path.join(path, "cutflow"), os.path.join(os.path.dirname(path), "cutflow")):
        if glob.glob(os.path.join(cand, "cutflow_merged_*.json")):
            return cand
    sys.exit(f"No cutflow_merged_*.json found in or next to {path}")


def load_cumulative(cutflow_dir):
    cols, order = {}, []
    for f in sorted(glob.glob(os.path.join(cutflow_dir, "cutflow_merged_*.json"))):
        sample = os.path.basename(f)[len("cutflow_merged_"):-len(".json")]
        with open(f) as handle:
            cf = json.load(handle)
        for cut in cf:
            if cut not in order:
                order.append(cut)
        cols[sample] = {cut: v["cumulative"] for cut, v in cf.items()}
    # rows = samples, columns = cuts in processing order
    order = [cut for cut in order if cut not in MASS_WINDOW_CUTS]
    return pd.DataFrame(cols).reindex(order).T


def format_table(counts, fmt):
    if fmt == "counts":
        return counts
    pct = counts.div(counts["initial"], axis=0) * 100
    if fmt == "percent":
        return pct.round(3)
    return counts.astype("Int64").astype(str) + " (" + pct.map(lambda x: f"{x:.3f}") + "%)"


def write_tex(table, path, caption, source):
    def esc(text):
        return str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")

    lines = [
        f"% source: {source}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "r" * len(table.columns) + "}",
        r"\hline",
        "Sample & " + " & ".join(esc(c) for c in table.columns) + r" \\",
        r"\hline",
    ]
    for sample, row in table.iterrows():
        lines.append(esc(sample) + " & " + " & ".join(esc(v) for v in row) + r" \\")
    lines += [r"\hline", r"\end{tabular}}", rf"\caption{{{esc(caption)}}}", r"\end{table}", ""]
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", help="cutflow/ dir, stage1_output/<year>/ dir, or its compacted/ dir")
    parser.add_argument("--format", choices=["counts", "percent", "both"], default="counts")
    parser.add_argument("--csv", help="CSV output path (default: validation/cutflow/<label>/<year>/cutflow_<year>_<format>.csv)")
    parser.add_argument("--tex", help="LaTeX output path (default: same as --csv default with .tex)")
    args = parser.parse_args()

    cutflow_dir = resolve_cutflow_dir(args.directory)
    table = format_table(load_cumulative(cutflow_dir), args.format)

    print(f"# cutflow dir: {cutflow_dir}")
    with pd.option_context("display.width", 100000, "display.max_columns", None, "display.max_rows", None):
        print(table.to_string())

    # cutflow_dir = .../<label>/stage1_output/<year>/cutflow
    year_dir = os.path.dirname(cutflow_dir)
    year = os.path.basename(year_dir)
    label = os.path.basename(os.path.dirname(os.path.dirname(year_dir)))
    stem = os.path.join(REPO_ROOT, "validation", "cutflow", label, year, f"cutflow_{year}_{args.format}")
    csv_path = args.csv or stem + ".csv"
    tex_path = args.tex or stem + ".tex"
    for out in (csv_path, tex_path):
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    table.to_csv(csv_path)
    print(f"# wrote {csv_path}")
    write_tex(table, tex_path, f"Stage-1 cumulative cutflow, {year} ({args.format})", cutflow_dir)
    print(f"# wrote {tex_path}")


if __name__ == "__main__":
    main()
