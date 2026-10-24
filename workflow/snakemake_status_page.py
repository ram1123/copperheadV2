#!/usr/bin/env python3
"""
Live, auto-updating browser page for the Snakemake runs of this repo (workflow/Snakefile).

Every --loop seconds it finds the snakemake processes running from the repo root (their open
.snakemake/log/*.snakemake.log via /proc/<pid>/fd) plus the most recent finished runs, parses each
snakemake log (job stats, started / finished / failed / restarted jobs, "N of M steps done") and the
per-job rule logs (last line + tail), and writes validation/job_status/snakemake/status.json. The
index.html next to it polls that JSON every 15 s: run tabs, per-rule progress, the year x jj-region
grid of the stage2 -> stage2plot -> stage3 -> s3validate chain, a filterable job table (click a job
for its log tail) and recent events. Each stage1 job also gets a per-sample chunk table (state, tries,
start/end, wall, events processed/expected, error) from run_stage1.py's _status/job_status.jsonl
ledger, plus the prestage JSON's samples not started yet. If the run has stats / impacts / lhscan jobs, the page also shows
the per-year expected significance + limit of each jj region (read live from the per-year combine logs
in the card dir) and the impact / likelihood-scan plots (copied into the page dir, as the web server
only sees validation/job_status). Read-only: it never touches the jobs. Times in US Eastern.
Jobs not yet started appear only in the per-rule counts (snakemake logs them when they start).

workflow/Snakefile's `onstart:` calls `--ensure`, so every real `snakemake` run (not --dry-run /
--summary) starts the page generator + the validation/job_status web server if they are not up yet,
both detached via .claude/skills/run-copperheadV2-develop/detach_job.sh, and prints the link.
The generator exits after --idle-exit minutes with no snakemake running; the next run restarts it.
Disable per run with `--config status_page=False`.

How to run (repo root; stdlib only, global env interpreter, no pixi/proxy needed):
    /work/pixi/global/.pixi/envs/default/bin/python workflow/snakemake_status_page.py [--ensure | --loop 15 | (once)]
  Page: https://cms.geddes.rcac.purdue.edu/user/<you>/proxy/8765/snakemake/ (JupyterHub login protects it)

Example (start page + server by hand and print the link; what onstart does):
    /work/pixi/global/.pixi/envs/default/bin/python workflow/snakemake_status_page.py --ensure
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "workflow"))
from summary_table import CHAIN_STEPS, GLOBAL_STEPS, PATTERNS, YEAR_STEPS, ordered_regions, short_region, year_key  # noqa: E402

LOG_DIR = REPO / ".snakemake" / "log"
OUT_DIR = REPO / "validation" / "job_status" / "snakemake"
SERVE_DIR = REPO / "validation" / "job_status"
DETACH = REPO / ".claude" / "skills" / "run-copperheadV2-develop" / "detach_job.sh"
GLOBAL_PY = "/work/pixi/global/.pixi/envs/default/bin/python"
PORT = 8765
ET = ZoneInfo("America/New_York")
TAIL_LINES = 40

TS_RE = re.compile(r"^\[(\w{3} \w{3} +\d+ \d\d:\d\d:\d\d \d{4})\]$")
START_RE = re.compile(r"^(?:local)?(?:rule|checkpoint) (\S+):$")
FIELD_RE = re.compile(r"^    (output|log|jobid|wildcards): (.*)$")
FINISHED_RE = re.compile(r"^Finished jobid: (\d+)")
ERROR_RE = re.compile(r"^Error in rule (\S+):$")
RESTART_RE = re.compile(r"^Trying to restart job (\d+)\.")
STEPS_RE = re.compile(r"^(\d+) of (\d+) steps \((\d+)%\) done")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

# stats results: same regexes as common_workflow.sh extract_significance_value / extract_expected_limit
FILES_DIR = OUT_DIR / "files"
STATS_KEYS = ("stats", "impacts", "lhscan")
STATS_YEARS = ["2022preEE", "2022postEE", "2023", "2023BPix", "2024", "2025", "2026", "2025_2026", "Run3"]
HEADER_RE = re.compile(r"^  (Save path|Output append|Label|Years|VBF filter study): (.*)$")
SIG_RE = re.compile(r"Significance:\s*([-+0-9.eE][0-9.eE+-]*)")
LIM_RE = re.compile(r"Expected\s*50\.0%:\s*r\s*<\s*([-+0-9.eE][0-9.eE+-]*)")
RAW_REGION_RE = re.compile(r"_(jj_\w+)\.done$")

# stage-1 chunk detail: run_stage1.py logs its output dir + command; modules/job_status.py writes the ledger
SAVE_PATH_RE = re.compile(r"start_save_path:\s*(\S+)")
S1_CMD_RE = re.compile(r"^Command: (.*run_stage1\.py.*)$")
S1_ORDER = {"running": 0, "failed": 1, "killed": 2, "pending": 3, "done": 4, "skipped": 5}
S1_SKIP_RE = re.compile(r"Skipping year=\S+, dataset=(\S+)")


def utc(ts):
    return datetime.strptime(" ".join(ts.split()), "%a %b %d %H:%M:%S %Y").replace(tzinfo=timezone.utc)


def et(dt):
    return dt.astimezone(ET).strftime("%a %H:%M:%S %Z") if dt else ""


def dur(sec):
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def running_snakemakes():
    """{log path: {pid, cmd}} for snakemake processes started from this repo."""
    found = {}
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            argv = [a.decode(errors="replace") for a in argv if a]
            if not any(Path(a).name == "snakemake" for a in argv[:2]):
                continue
            if os.readlink(f"/proc/{pid}/cwd") != str(REPO):
                continue
            for fd in os.listdir(f"/proc/{pid}/fd"):
                target = os.readlink(f"/proc/{pid}/fd/{fd}")
                if target.endswith(".snakemake.log"):
                    found[target] = {"pid": int(pid), "cmd": " ".join(argv[1:] if Path(argv[0]).name.startswith("python") else argv)}
        except OSError:
            continue
    return found


def tail(path, n=TAIL_LINES):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 64 * 1024))
            text = f.read().decode(errors="replace")
    except OSError:
        return []
    # progress bars redraw with \r: keep only the last state of each line
    lines = [ANSI_RE.sub("", ln.split("\r")[-1]).rstrip() for ln in text.split("\n")]
    return [ln for ln in lines if ln.strip()][-n:]


def classify(name):
    for key, pat in PATTERNS:
        m = pat.fullmatch(name)
        if m:
            gd = m.groupdict()
            return key, gd.get("year"), short_region(gd.get("region"))
    return None, None, None


def parse_log(path):
    text = Path(path).read_text(errors="replace")
    if "Job stats:" not in text:
        return None  # dry-run / --summary / nothing-to-do logs
    lines = text.splitlines()
    rule_total, jobs, events = {}, {}, []
    steps = (0, 0, 0)
    last_ts, cur, in_stats = None, None, False
    for ln in lines:
        if ln.startswith("Job stats:"):
            in_stats, rule_total = True, {}
            continue
        if in_stats:
            parts = ln.split()
            if len(parts) == 2 and parts[1].isdigit():
                if parts[0] == "total":
                    in_stats = False
                else:
                    rule_total[parts[0]] = int(parts[1])
            continue
        m = TS_RE.match(ln)
        if m:
            last_ts, cur = utc(m.group(1)), None
            continue
        m = START_RE.match(ln)
        if m and last_ts:
            cur = {"rule": m.group(1), "start": last_ts}
            continue
        m = FIELD_RE.match(ln)
        if m and cur is not None and "rule" in cur:
            key, val = m.groups()
            cur[key] = val
            if key == "jobid":
                jid = int(val)
                prev = jobs.get(jid)
                job = {"jobid": jid, "rule": cur["rule"], "output": cur.get("output", ""), "log": cur.get("log", ""),
                       "wildcards": "", "status": "running", "start": cur["start"], "end": None,
                       "attempt": prev["attempt"] + 1 if prev else 1}
                jobs[jid] = job
                cur["job"] = job
                events.append((cur["start"], "start", job))
            elif key == "wildcards" and "job" in cur:
                cur["job"]["wildcards"] = val
            continue
        m = FINISHED_RE.match(ln)
        if m and int(m.group(1)) in jobs:
            job = jobs[int(m.group(1))]
            job["status"], job["end"] = "done", last_ts
            events.append((last_ts, "done", job))
            continue
        m = ERROR_RE.match(ln)
        if m:
            cur = {"error_rule": m.group(1)}
            continue
        if cur is not None and "error_rule" in cur and ln.startswith("    jobid: "):
            jid = int(ln.split(":")[1])
            if jid in jobs:
                jobs[jid]["status"], jobs[jid]["end"] = "failed", last_ts
                events.append((last_ts, "failed", jobs[jid]))
            cur = None
            continue
        m = RESTART_RE.match(ln)
        if m and int(m.group(1)) in jobs:
            jobs[int(m.group(1))]["status"] = "restarting"
            continue
        m = STEPS_RE.match(ln)
        if m:
            steps = tuple(int(x) for x in m.groups())
    ended = "Complete log" in text
    failed_exit = any(s in text for s in ("Will exit after finishing", "Exiting because"))
    return {"rule_total": rule_total, "jobs": jobs, "events": events, "steps": steps,
            "ended": ended, "failed_exit": failed_exit, "mtime": os.path.getmtime(path)}


def build_run(path, proc, now, served):
    p = parse_log(path)
    if p is None:
        return None
    jobs = sorted(p["jobs"].values(), key=lambda j: j["start"])
    if proc:
        state = "running"
    elif p["failed_exit"] or any(j["status"] == "failed" for j in jobs) and p["steps"][2] < 100:
        state = "failed"
    elif p["steps"][2] == 100 or (p["ended"] and p["steps"][0] == p["steps"][1]):
        state = "finished"
    else:
        state = "stopped"  # killed, or pod restarted
    if not proc:  # nothing can still be running in a dead snakemake
        for j in jobs:
            if j["status"] in ("running", "restarting"):
                j["status"] = "killed" if state != "finished" else "done"

    run_tag = next((m.group(1) for j in jobs if (m := re.match(r"state/([^/]+)/", j["output"]))), "")
    cmd = proc["cmd"] if proc else ""
    if not run_tag:
        run_tag = (re.search(r"run_tag=(\S+)", cmd) or [None, ""])[1]

    rules = {}
    for rule, total in p["rule_total"].items():
        rules[rule] = {"rule": rule, "total": total, "done": 0, "running": 0, "failed": 0}
    for j in jobs:
        r = rules.setdefault(j["rule"], {"rule": j["rule"], "total": 0, "done": 0, "running": 0, "failed": 0})
        if j["status"] == "done":
            r["done"] += 1
        elif j["status"] in ("running", "restarting"):
            r["running"] += 1
        elif j["status"] in ("failed", "killed"):
            r["failed"] += 1
    for r in rules.values():
        r["total"] = max(r["total"], r["done"] + r["running"] + r["failed"])
        r["pending"] = r["total"] - r["done"] - r["running"] - r["failed"]

    job_rows, chain, year_cells, global_cells = [], {}, {}, {}
    chain_keys = {k for k, _, _ in CHAIN_STEPS}
    year_keys = {k for k, _, _ in YEAR_STEPS}
    rank = {"failed": 0, "killed": 0, "restarting": 1, "running": 2, "done": 3}
    stat_jobs = {}
    for j in jobs:
        name = os.path.basename(j["output"].split(",")[0].strip())
        key, year, region = classify(name)
        end = j["end"] or now
        lines = tail(REPO / j["log"]) if j["log"] and j["status"] != "done" else []
        job_rows.append({
            "jobid": j["jobid"], "rule": j["rule"], "name": name, "wildcards": j["wildcards"], "log": j["log"],
            "status": j["status"], "attempt": j["attempt"], "start": et(j["start"]), "end": et(j["end"]),
            "wall": dur((end - j["start"]).total_seconds()), "start_ts": j["start"].timestamp(),
            "last": lines[-1][:200] if lines else "", "tail": lines,
        })
        if key is None:
            continue
        if key in STATS_KEYS and j["log"]:  # latest attempt wins
            raw = RAW_REGION_RE.search(name)
            stat_jobs.setdefault(raw.group(1) if raw else "all", {})[key] = j
        if key in chain_keys:
            table, cell = chain, f"{year}|{region}|{key}"
        elif key in year_keys:
            table, cell = year_cells, f"{year}|{key}"
        else:
            table, cell = global_cells, f"{region}|{key}"
        old = table.get(cell)
        if old is None or rank.get(j["status"], 0) < rank.get(old, 0):
            table[cell] = j["status"]

    def axis(cells, idx, order=None):
        vals = list(dict.fromkeys(c.split("|")[idx] for c in cells))
        return sorted(vals, key=order) if order else ordered_regions(vals)

    events = [{"time": et(t), "ts": t.timestamp(), "kind": k, "rule": j["rule"], "jobid": j["jobid"],
               "name": os.path.basename(j["output"].split(",")[0].strip())} for t, k, j in p["events"]][-60:][::-1]
    done, total, pct = p["steps"]
    if not total:
        total = sum(r["total"] for r in rules.values())
    started = utc_from_logname(path)
    return {
        "id": Path(path).name, "log": os.path.relpath(path, REPO), "state": state, "pid": proc["pid"] if proc else None,
        "cmd": cmd, "run_tag": run_tag, "started": et(started), "started_ts": started.timestamp() if started else 0,
        "updated": et(datetime.fromtimestamp(p["mtime"], timezone.utc)),
        "elapsed": dur(((now if proc else datetime.fromtimestamp(p["mtime"], timezone.utc)) - started).total_seconds()) if started else "",
        "steps_done": done, "steps_total": total, "pct": pct if total else 0,
        "rules": list(rules.values()), "jobs": job_rows[::-1], "events": events,
        "chain": {"years": axis(chain, 0, year_key), "regions": axis(chain, 1),
                  "steps": [[k, d] for k, d, _ in CHAIN_STEPS], "cells": chain},
        "year_steps": {"years": axis(year_cells, 0, year_key),
                       "steps": [[k, d] for k, d, _ in YEAR_STEPS if any(c.endswith("|" + k) for c in year_cells)],
                       "cells": year_cells},
        "global": {"regions": axis(global_cells, 0),
                   "steps": [[k, d] for k, d, _ in GLOBAL_STEPS if any(c.endswith("|" + k) for c in global_cells)],
                   "cells": global_cells},
        "stats": stats_results(stat_jobs, global_cells, served) if stat_jobs else [],
        "stage1": [s for s in (stage1_details(j, j["status"] in ("running", "restarting"))
                               for j in {j["log"]: j for j in jobs if j["rule"] == "stage1" and j["log"]}.values()) if s],
    }


def flag(cmd, name):
    m = re.search(rf"(?:^|\s){re.escape(name)}\s+(\S+)", cmd)
    return m.group(1) if m else None


def num(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def stage1_details(job, alive):
    """Per-sample chunk progress of one stage1 rule job, from its _status/job_status.jsonl ledger
    (cumulative over runs into the same label: chunks finished before this job are flagged `earlier`)."""
    save, cmd = None, ""
    try:
        with open(REPO / job["log"], errors="replace") as f:
            for _, ln in zip(range(300), f):
                ln = ANSI_RE.sub("", ln)
                cmd = cmd or (m.group(1) if (m := S1_CMD_RE.match(ln.strip())) else "")
                # the rich logger wraps the long path onto the next lines, leaving "run_stage1.py:NNN" here
                if (m := SAVE_PATH_RE.search(ln)) and m.group(1).startswith("/"):
                    save = m.group(1)
                if cmd:
                    break
    except OSError:
        return None
    # run_stage1.py: start_save_path = <--save_path>/stage1_output/<year>
    if cmd and flag(cmd, "--save_path") and flag(cmd, "-y"):
        save = f"{flag(cmd, '--save_path')}/stage1_output/{flag(cmd, '-y')}"
    if not save:
        return None
    sdir, t0 = Path(save) / "_status", job["start"].timestamp()
    chunks = {}
    try:
        with open(sdir / "job_status.jsonl", errors="replace") as f:
            for ln in f:
                try:
                    rec = json.loads(ln)
                except ValueError:
                    continue
                c = chunks.setdefault((rec["dataset"], rec["idx"]), {"tries": 0, "rec": None, "start": None})
                if rec["status"] == "running":
                    c["start"] = rec["ts"]
                    c["tries"] += rec["ts"] >= t0
                c["rec"] = rec
    except OSError:
        pass
    year, nano, tag = flag(cmd, "-y"), flag(cmd, "--NanoAODv"), flag(cmd, "--prestage-tag")
    samples = []
    if year and nano:
        try:
            samples = list(json.load(open(REPO / "prestage_output" / f"processor_samples_{year}_NanoAODv{nano}{'_' + tag if tag else ''}.json")))
        except (OSError, ValueError):
            pass
    skipped = set()  # samples run_stage1.py skips (configs/skip_stage1_run.py), never "pending"
    try:
        with open(REPO / job["log"], errors="replace") as f:
            for ln in f:
                if "Skipping year=" in ln and (m := S1_SKIP_RE.search(ANSI_RE.sub("", ln))):
                    skipped.add(m.group(1).rstrip(","))
    except OSError:
        pass
    now, rows = time.time(), []
    for (ds, idx), c in chunks.items():
        rec, meta = c["rec"], c["rec"].get("meta") or {}
        status = rec["status"]
        if status == "running" and not alive:
            status = "killed"
        end = rec["ts"] if rec["status"] != "running" else None
        start = c["start"]
        rows.append({
            "sample": ds, "idx": idx, "status": status, "tries": c["tries"], "earlier": rec["ts"] < t0,
            "start": et(datetime.fromtimestamp(start, timezone.utc)) if start else "",
            "end": et(datetime.fromtimestamp(end, timezone.utc)) if end else "",
            "wall": (dur(rec["duration_seconds"]) if rec.get("duration_seconds") else
                     "" if status == "killed" else dur((end or now) - start) if start else ""),
            "expected": num(meta.get("Expected events from pre-stage")), "processed": num(meta.get("Processed events from stage-1")),
            "redirector": meta.get("redirector") or "", "error": (rec.get("error") or "")[:1500],
        })
    seen = {r["sample"] for r in rows}
    rows += [{"sample": s, "idx": "", "status": "skipped" if s in skipped else "pending", "tries": 0, "earlier": False, "start": "", "end": "", "wall": "",
              "expected": None, "processed": None, "redirector": "", "error": ""} for s in samples if s not in seen]
    rows.sort(key=lambda r: (S1_ORDER.get(r["status"], 9), r["sample"], str(r["idx"])))
    counts = {k: sum(r["status"] == k for r in rows) for k in S1_ORDER}
    return {"year": year or "", "dir": str(sdir), "n_samples": len(samples), "counts": counts, "chunks": rows,
            "summary": (sdir / "stage1_summary.log").exists() and et(datetime.fromtimestamp((sdir / "stage1_summary.log").stat().st_mtime, timezone.utc))}


def log_header(path):
    """{field: value} from the run_stats_pipeline_VBF.sh parameter block at the top of a stats job log."""
    try:
        with open(REPO / path, errors="replace") as f:
            head = [next(f, "") for _ in range(40)]
    except OSError:
        return {}
    return dict(m.groups() for ln in head if (m := HEADER_RE.match(ln.rstrip("\n"))))


def card_dir(hdr, region):
    """= common_workflow.sh vbf_card_dir() for this job's region."""
    postfix = hdr["Output append"]
    if hdr.get("VBF filter study", "0") == "1":
        postfix += "_vbf_filter_study"
    if region != "all":
        postfix += f"_{region}"
    return Path(hdr["Save path"]) / f"stage3_datacards_{postfix}" / f"score_{hdr['Label']}"


def first_value(path, regex):
    try:
        m = regex.search(path.read_text(errors="replace"))
    except OSError:
        return None
    return m.group(1) if m else None


def mirror(src, served):
    """Copy a plot into the served page dir (only when it changed); return its URL relative to index.html."""
    st = src.stat()
    dst = FILES_DIR / hashlib.sha1(str(src.parent).encode()).hexdigest()[:12] / src.name
    try:
        same = dst.stat().st_size == st.st_size and dst.stat().st_mtime == st.st_mtime
    except OSError:
        same = False
    if not same:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst.with_suffix(dst.suffix + ".tmp"))
        os.replace(dst.with_suffix(dst.suffix + ".tmp"), dst)
    served.add(dst)
    return f"{dst.relative_to(OUT_DIR)}?v={int(st.st_mtime)}"


def stats_results(stat_jobs, cells, served):
    """Per jj region: significance / limit per year and the impact + scan plots of the run's stats jobs.

    stat_jobs: {raw region: {step key: job}}; a result older than its job's start is flagged stale
    (left over from an earlier run into the same card dir)."""
    out = []
    for region, by_key in stat_jobs.items():
        hdrs = {k: log_header(j["log"]) for k, j in by_key.items()}
        # any of the region's stats job logs carries the same paths; Years differs (stats vs scan years)
        hdr = next((h for h in hdrs.values() if {"Save path", "Output append", "Label"} <= h.keys()), None)
        if hdr is None:
            continue
        cdir, save_postfix = card_dir(hdr, region), hdr["Output append"]
        short = short_region(region)

        def stale(path, key):
            return path.stat().st_mtime < by_key[key]["start"].timestamp()

        entry = {"region": short, "card_dir": str(cdir), "steps": {k: cells.get(f"{short}|{k}") for k in by_key},
                 "significance": [], "impacts": [], "lhscan": []}
        if "stats" in by_key:
            years = hdrs["stats"].get("Years", "").split() or STATS_YEARS
            for year in sorted(set(years), key=lambda y: (STATS_YEARS.index(y) if y in STATS_YEARS else 99, y)):
                stem = cdir / f"HMuMu_13TeV_{year}"
                files = {"sig": (f"{stem}_prefitsignificance.log", SIG_RE),
                         "sig_stat": (f"{stem}_prefitsignificance_StatOnly.log", SIG_RE),
                         "lim": (f"{stem}_expectedlimit.log", LIM_RE),
                         "lim_stat": (f"{stem}_expectedlimit_StatOnly.log", LIM_RE)}
                row, newest = {"year": year, "stale": False}, None
                for col, (path, regex) in files.items():
                    path = Path(path)
                    if not path.exists() or stale(path, "stats"):
                        row[col] = None
                        row["stale"] |= path.exists()
                        continue
                    row[col] = first_value(path, regex)
                    newest = max(newest or 0, path.stat().st_mtime)
                row["updated"] = et(datetime.fromtimestamp(newest, timezone.utc)) if newest else ""
                entry["significance"].append(row)
        if "impacts" in by_key:
            for year in hdrs["impacts"].get("Years", "Run3").split():
                for tag, label in (("r1", "Asimov r=1"), ("r0", "Asimov r=0")):
                    pdf = cdir / f"impacts_{year}_{save_postfix}_{tag}.pdf"
                    ok = pdf.exists()
                    entry["impacts"].append({"year": year, "tag": label, "stale": ok and stale(pdf, "impacts"),
                                             "url": mirror(pdf, served) if ok else None,
                                             "updated": et(datetime.fromtimestamp(pdf.stat().st_mtime, timezone.utc)) if ok else ""})
        if "lhscan" in by_key:
            for year in hdrs["lhscan"].get("Years", "Run3").split():
                png, pdf = (cdir / f"lh_scan_{year}_{save_postfix}{ext}" for ext in (".png", ".pdf"))
                ok = png.exists()
                entry["lhscan"].append({"year": year, "stale": ok and stale(png, "lhscan"),
                                        "url": mirror(png, served) if ok else None,
                                        "pdf": mirror(pdf, served) if pdf.exists() else None,
                                        "updated": et(datetime.fromtimestamp(png.stat().st_mtime, timezone.utc)) if ok else ""})
        out.append(entry)
    return sorted(out, key=lambda e: ordered_regions([x["region"] for x in out]).index(e["region"]))


def utc_from_logname(path):
    try:
        return datetime.strptime(Path(path).name.split(".")[0], "%Y-%m-%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def collect(max_runs, days):
    now = datetime.now(timezone.utc)
    procs = running_snakemakes()
    recent = sorted(LOG_DIR.glob("*.snakemake.log"), key=os.path.getmtime, reverse=True)
    recent = [str(p) for p in recent if now.timestamp() - os.path.getmtime(p) < days * 86400]
    paths = list(dict.fromkeys(list(procs) + recent))
    runs, served = [], set()
    for path in paths:
        if len(runs) >= max_runs and path not in procs:
            continue
        try:
            run = build_run(path, procs.get(path), now, served)
        except (OSError, ValueError):
            continue
        if run:
            runs.append(run)
    runs.sort(key=lambda r: (r["state"] != "running", -r["started_ts"]))
    if FILES_DIR.exists():  # drop plot copies no run on the page refers to any more
        for f in FILES_DIR.rglob("*"):
            if f.is_file() and f not in served:
                f.unlink(missing_ok=True)
    return {"generated": et(now), "generated_ts": now.timestamp(), "n_running": len(procs), "runs": runs}


def write_atomic(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def page_url():
    prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX") or f"/user/{os.environ.get('NB_USER') or os.environ.get('USER')}/"
    return f"https://cms.geddes.rcac.purdue.edu{prefix}proxy/{PORT}/snakemake/"


def alive(name):
    """True if detach_job.sh job <name> is still running (its wrapper pid is alive)."""
    try:
        pid = int((REPO / "logs" / "detached" / f"{name}.pid").read_text().split()[0])
        os.kill(pid, 0)
        return True
    except (OSError, ValueError, IndexError):
        return False


def port_in_use(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def ensure():
    py = GLOBAL_PY if os.path.exists(GLOBAL_PY) else sys.executable
    env = dict(os.environ, MIN_PROXY_H="0")
    launch = []
    if not alive("snakemake_status_page"):
        launch.append(["snakemake_status_page", py, str(Path(__file__).relative_to(REPO)), "--loop", "15"])
    if not port_in_use(PORT):
        launch.append(["status_page_server", py, "-m", "http.server", str(PORT), "--bind", "127.0.0.1",
                       "--directory", str(SERVE_DIR.relative_to(REPO))])
    for args in launch:
        subprocess.run(["bash", str(DETACH), *args], cwd=REPO, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    print(f"[status page] {page_url()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loop", type=int, default=0, help="rewrite status.json every N seconds (0 = once)")
    ap.add_argument("--ensure", action="store_true", help="start generator + web server if not running, print the link")
    ap.add_argument("--max-runs", type=int, default=8, help="finished runs to keep on the page")
    ap.add_argument("--days", type=float, default=3, help="only finished runs whose log changed within this many days")
    ap.add_argument("--idle-exit", type=float, default=120, help="with --loop: exit after this many minutes with no snakemake running")
    args = ap.parse_args()
    if args.ensure:
        ensure()
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_atomic(OUT_DIR / "index.html", (REPO / "workflow" / "snakemake_status_page.html").read_text())
    idle_since = time.time()
    while True:
        data = collect(args.max_runs, args.days)
        write_atomic(OUT_DIR / "status.json", json.dumps(data, separators=(",", ":")))
        if data["n_running"]:
            idle_since = time.time()
        if not args.loop or time.time() - idle_since > args.idle_exit * 60:
            break
        time.sleep(args.loop)
    if not args.loop:
        print(f"wrote {OUT_DIR / 'status.json'} ({len(data['runs'])} runs, {data['n_running']} running) -> {page_url()}")


if __name__ == "__main__":
    main()
