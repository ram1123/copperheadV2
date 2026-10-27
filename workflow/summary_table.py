#!/usr/bin/env python3
"""
Condense `snakemake --summary` into a status report for the workflow/Snakefile DAG.

Sections (each printed only if the DAG has such jobs):
  1. Progress per step: bar + done/total over individual output files.
  2. Per-year steps (stage1, compact, plots, zpt, mass calibration, PU DNN).
  3. Year x jj-region grid for the stage2 -> stage2plot -> stage3 -> s3validate chain,
     one symbol per step in that order.
  4. Year-less steps per region (VBF DNN training incl. the per-region dnn_chain, Run3 plot, stats,
     impacts, lhscan, summary).
A cell aggregates every job of that kind (stage1 reps, zpt njets bins, plot variants) and shows
the WORST status, so one missing job is never hidden by a finished one. Output files matching no
known pattern are listed on stderr, so a renamed .done file in the Snakefile shows up here.
Symbols are coloured ASCII by default (+ ok, ~ stale, ! incomplete, o missing, - not in DAG) since
many terminals draw emoji 2 cells wide but advance 1, smearing the columns; --emoji opts back in.
Colour is off when stdout is not a terminal, with NO_COLOR set, or with --no-color.
Note: a job that is running right now still shows as missing (its .done is written at the end).

How to run (repo root, `default` pixi env for snakemake; reads the summary on stdin). Pass the
same target and --config as the run being checked; --summary is read-only and safe mid-run:
    ./run_in_pixi.sh default bash -c 'snakemake -s workflow/Snakefile [target] --summary [--config ...] | python workflow/summary_table.py [--list] [--all-columns] [--emoji] [--no-color]'

Example (list the not-yet-done jobs too):
    ./run_in_pixi.sh default bash -c 'snakemake -s workflow/Snakefile --summary --config run_tag=FilterEvents_Sep22_tightPassLepVeto_OfficialRecomendation | python workflow/summary_table.py --list'
"""
import argparse
import os
import re
import sys
import unicodedata
from collections import Counter

YEAR = r"(?P<year>20\d\d(?:preVFP|postVFP|preEE|postEE|BPix)?(?:_20\d\d)?)"
REG = r"(?:_(?P<region>jj_\w+))?"

# (step key, header, basename regex without ".done")
YEAR_STEPS = [
    ("stage1", "stage1", rf"stage1_{YEAR}_rep\d+"),
    ("compact", "compact", rf"stage1_compact_{YEAR}"),
    ("pu_dnn", "puDNN", rf"pu_dnn_train_{YEAR}"),
    ("plots_nocat", "plots(nocat)", rf"plots_{YEAR}_nocat_.+"),
    ("plots_vbf", "plots(vbf)", rf"plots_{YEAR}_vbf_.+"),
    ("plots_ggh", "plots(ggh)", rf"plots_{YEAR}_ggh_.+"),
    ("zpt0", "zpt0", rf"zpt_step0_{YEAR}"),
    ("zpt1", "zpt1", rf"zpt_step1_{YEAR}_njets\d+"),
    ("zpt2", "zpt2", rf"zpt_step2_{YEAR}_njets\d+"),
    ("massData", "massData", rf"mass_calibration_Data_{YEAR}"),
    ("massDataCl", "massDataCl", rf"mass_calibration_Data_Closure_{YEAR}"),
    ("massMC", "massMC", rf"mass_calibration_MC_{YEAR}"),
    ("massMCCl", "massMCCl", rf"mass_calibration_MC_Closure_{YEAR}"),
]
CHAIN_STEPS = [
    ("stage2", "stage2", rf"stage2_{YEAR}{REG}"),
    ("stage2_plot", "stage2plot", rf"stage2_plot_{YEAR}{REG}"),
    ("stage3", "stage3", rf"stage3_{YEAR}{REG}"),
    ("stage3_validate", "s3validate", rf"stage3_validate_{YEAR}{REG}"),
]
GLOBAL_STEPS = [
    ("dnn_pre", "dnnPre", r"dnn_pre"),
    ("dnn_hpo", "dnnHPO", r"dnn_hpo"),
    ("dnn_train", "dnnTrain", r"dnn_train"),
    # per-region chain (Snakefile dnn_chain: true)
    ("dnn_region", "dnnRegion", rf"dnn_region{REG}"),
    ("dnn_scoredump", "scoreDump", rf"dnn_scoredump{REG}"),
    ("dnn_binscan", "binScan", rf"dnn_binscan{REG}"),
    ("plot_run3", "plot(Run3)", rf"stage2_plot_Run3{REG}"),
    ("stats", "stats", rf"stats{REG}"),
    ("impacts", "impacts", rf"impacts{REG}"),
    ("lhscan", "lhscan", rf"lhscan{REG}"),
    ("summary", "summary", r"summary"),
]
ALL_STEPS = YEAR_STEPS + CHAIN_STEPS + GLOBAL_STEPS
PATTERNS = [(key, re.compile(pat + r"\.done")) for key, _, pat in ALL_STEPS]

YEAR_ORDER = ["2016preVFP", "2016postVFP", "2017", "2018", "2022preEE", "2022postEE",
              "2023", "2023BPix", "2024", "2025", "2026", "2025_2026"]

OK, STALE, INCOMPLETE, MISSING, UNKNOWN, ABSENT = "✅", "🔁", "⚠️", "⏳", "?", "–"
# worst -> best; unknown statuses rank below everything so they are never hidden
RANK = {UNKNOWN: -1, MISSING: 0, INCOMPLETE: 1, STALE: 2, OK: 3}
ASCII = {OK: "+", STALE: "~", INCOMPLETE: "!", MISSING: "o", UNKNOWN: "?", ABSENT: "-"}
COLOR = {OK: "32", STALE: "36", INCOMPLETE: "1;31", MISSING: "33", UNKNOWN: "35", ABSENT: "2"}
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
USE_EMOJI, USE_COLOR = False, False
INDENT, GAP = "  ", "   "


def paint(text, code):
    return f"\x1b[{code}m{text}\x1b[0m" if USE_COLOR else text


def show(sym):
    return sym if USE_EMOJI else paint(ASCII[sym], COLOR[sym])


def legend():
    items = [(OK, "up to date"), (STALE, "rerun pending"), (INCOMPLETE, "incomplete"),
             (MISSING, "missing/running"), (UNKNOWN, "unknown status"), (ABSENT, "not in DAG")]
    return "Legend: " + GAP.join(f"{show(s)} {t}" for s, t in items)


def status_to_symbol(status: str, plan: str) -> str:
    if status == "missing":
        return MISSING
    if status == "incomplete":  # older snakemake only
        return INCOMPLETE
    if plan == "update pending":
        return STALE
    if status in ("ok", "removed temp file", "file local to node", "pipe file", "service file"):
        return OK
    return UNKNOWN


def worst(old, new):
    return new if old is None or RANK[new] < RANK[old] else old


def short_region(region):
    if region is None:
        return "all"
    if region.startswith("jj_combined"):
        return "combined"
    return region.removeprefix("jj_")


def parse_summary(text: str):
    """Return (records, unmatched); a record is (step, year, region, symbol, basename)."""
    records, unmatched = [], []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 6 or parts[0] == "output_file":
            continue
        out, _date, rule, _log, status, plan = (p.strip() for p in parts[:6])
        name = os.path.basename(out)  # not the full path: run_tag may contain a year
        hit = next(((k, m) for k, p in PATTERNS if (m := p.fullmatch(name))), None)
        if hit is None:
            unmatched.append(f"{rule}: {name}")
            continue
        key, m = hit
        gd = m.groupdict()
        records.append((key, gd.get("year"), short_region(gd.get("region")), status_to_symbol(status, plan), name))
    return records, unmatched


def width(s: str) -> int:
    w = 0
    for ch in ANSI_RE.sub("", s):
        if ch == "️":  # emoji presentation selector: preceding char renders 2 wide
            w += 1
        elif not unicodedata.combining(ch):
            w += 2 if unicodedata.east_asian_width(ch) in "WF" else 1
    return w


def pad(s, n):
    return s + " " * (n - width(s))


def print_grid(title, header, rows):
    widths = [max(width(r[i]) for r in [header] + rows) for i in range(len(header))]

    def fmt(r):
        return INDENT + GAP.join(pad(c, widths[i]) for i, c in enumerate(r)).rstrip()

    print(title)
    line = fmt(header)
    print(line)
    print(INDENT + "-" * (width(line) - len(INDENT)))
    for r in rows:
        print(fmt(r))
    print()


def year_key(y):
    return (YEAR_ORDER.index(y) if y in YEAR_ORDER else len(YEAR_ORDER), y)


def ordered_regions(regions):
    seen = list(dict.fromkeys(regions))
    return sorted(seen, key=lambda r: (r != "all", r == "combined", seen.index(r)))


def cells(records, steps, row_of, col_of):
    keys = {k for k, _, _ in steps}
    table = {}
    for key, year, region, sym, _ in records:
        if key in keys:
            cell = (row_of(key, year, region), col_of(key, year, region))
            table[cell] = worst(table.get(cell), sym)
    return table


def print_progress(records):
    counts = {}
    for key, _, _, sym, _ in records:
        counts.setdefault(key, Counter())[sym] += 1
    if not counts:
        return
    total = Counter(sym for *_, sym, _ in records)
    n = sum(total.values())
    print(f"Progress: {total[OK]}/{n} jobs up to date ({100 * total[OK] / n:.0f}%)")
    names = {k: d for k, d, _ in ALL_STEPS}
    label_w = max(width(names[k]) for k in counts)
    for key, _, _ in ALL_STEPS:
        c = counts.get(key)
        if not c:
            continue
        tot, bar_n = sum(c.values()), 20
        filled = round(bar_n * c[OK] / tot)
        rest = "  ".join(f"{show(s)} {c[s]}" for s in (STALE, INCOMPLETE, MISSING, UNKNOWN) if c[s])
        print(f"  {pad(names[key], label_w)}  {paint('#' * filled, COLOR[OK])}{paint('.' * (bar_n - filled), COLOR[ABSENT])}  {c[OK]:>3}/{tot:<3}  {rest}".rstrip())
    print()


def print_year_steps(records, all_columns):
    table = cells(records, YEAR_STEPS, lambda k, y, r: y, lambda k, y, r: k)
    if not table:
        return
    steps = [s for s in YEAR_STEPS if all_columns or any(c == s[0] for _, c in table)]
    years = sorted({y for y, _ in table}, key=year_key)
    print_grid("Per-year steps", ["Year"] + [d for _, d, _ in steps],
               [[y] + [show(table.get((y, k), ABSENT)) for k, _, _ in steps] for y in years])


def print_chain(records):
    table = cells(records, CHAIN_STEPS, lambda k, y, r: (y, r), lambda k, y, r: k)
    if not table:
        return
    years = sorted({y for (y, _), _ in table}, key=year_key)
    regions = ordered_regions(r for (_, r), _ in table)
    rows = []
    for y in years:
        row = [y]
        for r in regions:
            row.append(" ".join(pad(show(table.get(((y, r), k), ABSENT)), 2 if USE_EMOJI else 1) for k, _, _ in CHAIN_STEPS))
        rows.append(row)
    print_grid("Per year x jj region, each cell = stage2 stage2plot stage3 s3validate", ["Year"] + regions, rows)


def print_global(records, all_columns):
    table = cells(records, GLOBAL_STEPS, lambda k, y, r: r, lambda k, y, r: k)
    if not table:
        return
    steps = [s for s in GLOBAL_STEPS if all_columns or any(c == s[0] for _, c in table)]
    regions = ordered_regions(r for r, _ in table)
    print_grid("Combined-year steps per jj region", ["Region"] + [d for _, d, _ in steps],
               [[r] + [show(table.get((r, k), ABSENT)) for k, _, _ in steps] for r in regions])


def print_list(records):
    pending = [(sym, name) for *_, sym, name in records if sym != OK]
    if not pending:
        print("All jobs up to date.")
        return
    for sym in (UNKNOWN, INCOMPLETE, STALE, MISSING):
        names = [n for s, n in pending if s == sym]
        if names:
            print(f"{show(sym)} ({len(names)}):")
            for n in names:
                print(f"  {n}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="also list every job that is not up to date")
    ap.add_argument("--all-columns", action="store_true", help="keep columns with no job in this DAG")
    ap.add_argument("--emoji", action="store_true", help="emoji symbols (only if your terminal sizes them right)")
    ap.add_argument("--no-color", action="store_true", help="plain output without ANSI colours")
    args = ap.parse_args()
    global USE_EMOJI, USE_COLOR
    USE_EMOJI = args.emoji
    USE_COLOR = not args.emoji and not args.no_color and sys.stdout.isatty() and not os.environ.get("NO_COLOR")

    records, unmatched = parse_summary(sys.stdin.read())
    if not records:
        print("No known workflow outputs in the input (was it `snakemake --summary` output?).")
    print_progress(records)
    print_year_steps(records, args.all_columns)
    print_chain(records)
    print_global(records, args.all_columns)
    if args.list:
        print_list(records)
    print(legend())
    if unmatched:
        print(f"WARNING: {len(unmatched)} output file(s) match no known pattern:", file=sys.stderr)
        for u in unmatched:
            print(f"  {u}", file=sys.stderr)


if __name__ == "__main__":
    main()
