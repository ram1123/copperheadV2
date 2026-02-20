#!/usr/bin/env python3
import sys
import re
from collections import OrderedDict

COLS = [
    ("stage1", "stage1"),
    ("stage1Compact", "compact"),
    ("plots_nocat", "plots(nocat)"),
    ("plots_vbf", "plots(vbf)"),
    ("plots_ggh", "plots(ggh)"),
    ("zpt0", "zpt0"),
    ("zpt1", "zpt1"),
    ("zpt2", "zpt2"),
    ("MassCalibrationData", "massData"),
    ("MassCalibrationMC", "massMC"),
    ("MassCalibrationDataClosure", "massDataCl"),
    ("MassCalibrationMCClosure", "massMCCl"),
]

# status precedence: worse -> better
RANK = {"⏳": 0, "🔁": 1, "⚠️": 2, "✅": 3}

def status_to_symbol(status: str, plan: str) -> str:
    # If snakemake says it needs update, show 🔁 even if file exists
    if plan.strip() == "update pending":
        return "🔁"
    if status == "ok":
        return "✅"
    if status == "incomplete":
        return "⚠️"
    if status == "missing":
        return "⏳"
    return "⏳"

def bump(best: str, new: str) -> str:
    return new if RANK.get(new, 0) > RANK.get(best, 0) else best

def parse_summary(text: str):
    lines = [l.rstrip("\n") for l in text.splitlines()]
    entries = []
    for l in lines:
        if not l.strip():
            continue
        if l.startswith("Building DAG of jobs"):
            continue
        if l.startswith("output_file") or l.startswith("----"):
            continue

        parts = l.split("\t")
        if len(parts) < 6:
            continue
        output_file, date, rule, log, status, plan = parts[:6]
        entries.append((output_file, rule, status, plan))

    # capture year anywhere in the output filename (covers ..._2023_0.done too)
    year_re = re.compile(r"(2016preVFP|2016postVFP|2017|2018|2022preEE|2022postEE|2023BPix|2023|2024)")

    # store per-year per-key symbol; for zpt1/zpt2 we will aggregate across folds automatically by "bump"
    table = OrderedDict()

    def setcell(year, key, sym):
        if year not in table:
            table[year] = {}
        table[year][key] = bump(table[year].get(key, "⏳"), sym)

    for out, rule, status, plan in entries:
        ym = year_re.search(out)
        if not ym:
            continue
        year = ym.group(1)

        sym = status_to_symbol(status, plan)

        if rule == "stage1":
            setcell(year, "stage1", sym)
        elif rule == "stage1Compact":
            setcell(year, "stage1Compact", sym)
        elif rule == "plots":
            m = re.search(rf"plots_{re.escape(year)}_(nocat|vbf|ggh)\.done$", out)
            if m:
                setcell(year, f"plots_{m.group(1)}", sym)
        elif rule == "zpt0":
            setcell(year, "zpt0", sym)
        elif rule == "zpt1":
            # matches both zpt_step1_YEAR.done and zpt_step1_YEAR_0.done, etc.
            setcell(year, "zpt1", sym)
        elif rule == "zpt2":
            setcell(year, "zpt2", sym)
        elif rule == "MassCalibrationData":
            setcell(year, "MassCalibrationData", sym)
        elif rule == "MassCalibrationMC":
            setcell(year, "MassCalibrationMC", sym)
        elif rule == "MassCalibrationDataClosure":
            setcell(year, "MassCalibrationDataClosure", sym)
        elif rule == "MassCalibrationMCClosure":
            setcell(year, "MassCalibrationMCClosure", sym)

    return table

def print_table(table: OrderedDict):
    header = ["Year"] + [disp for _, disp in COLS]
    rows = [header]

    # stable ordering: Run2 then Run3
    order = ["2016preVFP","2016postVFP","2017","2018","2022preEE","2022postEE","2023","2023BPix","2024"]
    years = [y for y in order if y in table] + [y for y in table.keys() if y not in order]

    for y in years:
        row = [y]
        for key, _ in COLS:
            row.append(table[y].get(key, "⏳"))
        rows.append(row)

    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]

    def fmt(r):
        return "  ".join(str(r[i]).ljust(widths[i]) for i in range(len(r)))

    print(fmt(rows[0]))
    print("-" * len(fmt(rows[0])))
    for r in rows[1:]:
        print(fmt(r))

def main():
    text = sys.stdin.read()
    table = parse_summary(text)
    print_table(table)

if __name__ == "__main__":
    main()