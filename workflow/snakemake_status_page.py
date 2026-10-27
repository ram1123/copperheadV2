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
PROGRESS_RE = re.compile(r"^\s*\d+%\|")

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


try:
    import yaml  # global env has it; without it the page just skips the run-config card
except ImportError:
    yaml = None


def run_config(cmd):
    """The run's effective Snakemake config: workflow/config.yaml < each --configfile < --config k=v."""
    if yaml is None or not cmd:
        return {}
    args, cfg = cmd.split(), {}
    files = [REPO / "workflow" / "config.yaml"]
    overrides, mode = [], None
    for a in args:
        if a.startswith("--configfile"):
            mode = "file"
            if "=" in a:
                files.append(REPO / a.split("=", 1)[1])
        elif a == "--config":
            mode = "kv"
        elif a.startswith("-"):
            mode = None
        elif mode == "file":
            files.append(REPO / a)
        elif mode == "kv" and "=" in a:
            overrides.append(a)
    for f in files:
        try:
            cfg.update(yaml.safe_load(open(f)) or {})
        except (OSError, yaml.YAMLError):
            pass
    for kv in overrides:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = yaml.safe_load(v)
        except yaml.YAMLError:
            cfg[k] = v
    return cfg


def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            if (hit := find_key(v, key)) is not None:
                return hit
    return None


def config_summary(cfg):
    """The settings that tell runs apart, for the run header (selection, inputs, DNN, outputs)."""
    if not cfg:
        return None
    years = cfg.get("years") or []
    veto = None
    if cfg.get("switches_yaml"):
        try:
            per_year = find_key(yaml.safe_load(open(REPO / cfg["switches_yaml"])), "do_lepton_ip_veto") or {}
            vals = {bool(per_year.get(y, False)) for y in years}
            veto = "ON" if vals == {True} else "off" if vals == {False} else "mixed: " + ", ".join(y for y in years if per_year.get(y))
        except (OSError, yaml.YAMLError, AttributeError):
            pass
    pt = cfg.get("prestage_tag") or ""
    regions = cfg.get("jj_eta_regions") or [cfg.get("jj_eta_region") or "all"]
    return {
        "run_tag": cfg.get("run_tag", ""), "label": cfg.get("stage23_label") or cfg.get("stage1_label") or "",
        "years": years, "switches": cfg.get("switches_yaml") or "", "lepton_ip_veto": veto,
        "prestage_tag": ", ".join(f"{y}: {t}" for y, t in pt.items()) if isinstance(pt, dict) else pt,
        "regions": regions, "combine": cfg.get("jj_combine_regions") or [], "postfix": cfg.get("save_postfix") or "(date of start)",
        "with_variations": bool(cfg.get("with_variations")),
        "dnn": (f"chain, {cfg.get('dnn_hpo_trials', 50)} trials, {cfg.get('dnn_hpo_backend') or 'local'}" if cfg.get("dnn_chain")
                else f"model {cfg.get('dnn_model_label') or '(own label)'}"),
        "reference": cfg.get("status_reference") or None,
    }


PROXY_FILE = REPO / "voms_proxy.txt"
_proxy_cache = {"t": 0, "v": None}


def proxy_status():
    """Seconds left on voms_proxy.txt (the proxy run_in_pixi.sh exports), checked every 5 min."""
    if time.time() - _proxy_cache["t"] < 300:
        return _proxy_cache["v"]
    v = None
    try:
        out = subprocess.run(["voms-proxy-info", "-file", str(PROXY_FILE), "-timeleft"], capture_output=True, text=True, timeout=30)
        left = int(out.stdout.strip().splitlines()[-1])
        v = {"left_s": left, "left": dur(max(left, 0)), "expires": et(datetime.fromtimestamp(time.time() + left, timezone.utc)), "file": str(PROXY_FILE)}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        pass
    _proxy_cache.update(t=time.time(), v=v)
    return v


OPTUNA_CREATE, OPTUNA_SET_STATE = 4, 6  # optuna JournalStorage op codes
OPTUNA_STATES = {1: "complete", 2: "pruned", 3: "fail"}
DNN_STATUS_RE = re.compile(r"^\[([^\]]+)\] (START|OK|FAIL) (\S+) (\S+) y=(\S+) h=(\S+)(?: wall=(\d+)s)?")


def dnn_progress(cfg):
    """Per jj region of a dnn_chain run: HPO trials from the optuna journal, trained folds, and the
    START/OK/FAIL lines of run_dnn_trainings.sh's status log."""
    if not cfg.get("dnn_chain"):
        return []
    label = cfg.get("stage23_label") or cfg.get("stage1_label")
    if not label:
        return []
    years = cfg.get("dnn_years") or cfg.get("years") or []
    trials = int(cfg.get("dnn_hpo_trials", 50))
    hpo = f"v1_multifold_{trials:03d}Trials"
    regions = cfg.get("jj_eta_regions") or [cfg.get("jj_eta_region") or "all"]
    status = {}
    try:
        for ln in open(REPO / "logs" / cfg.get("run_tag", "default") / "dnn_status.log", errors="replace"):
            m = DNN_STATUS_RE.match(ln)
            if m and m.group(3) == label and m.group(6) == hpo:
                status[m.group(4)] = {"state": m.group(2), "time": m.group(1), "wall": dur(int(m.group(7))) if m.group(7) else ""}
    except OSError:
        pass
    out = []
    for r in regions:
        d = REPO / "dnn" / "trained_models" / label / f"{'-'.join(years)}_h-peak_vbf_{r}"
        created, ended = 0, {}
        try:
            for ln in open(d / "hpo_optuna" / hpo / "optuna_journal.log", errors="replace"):
                try:
                    rec = json.loads(ln)
                except ValueError:
                    continue
                if rec.get("op_code") == OPTUNA_CREATE:
                    created += 1
                elif rec.get("op_code") == OPTUNA_SET_STATE and rec.get("state") in OPTUNA_STATES:
                    k = OPTUNA_STATES[rec["state"]]
                    ended[k] = ended.get(k, 0) + 1
        except OSError:
            pass
        folds = sum((d / f"trained_best_optuna_{hpo}" / f"fold{f}" / "best_torchscript.pt").exists() for f in range(4))
        out.append({"region": r, "target": trials, "created": created, "finished": sum(ended.values()), **ended,
                    "folds": folds, "preprocessed": (d / "training_features.pkl").exists(), "status": status.get(r)})
    return out


# stage-1 errors: the cause is at the END of a long WorkItem repr (usermeta holds thousands of PDF sums)
ERROR_TYPES = [
    ("worker killed (OOM / segfault)", re.compile(r"KilledWorker|MemoryError|out-of-memory|OOMKilled", re.I)),
    ("XRootD / file access", re.compile(r"XRootD|\[ERROR\] Server|Operation expired|FileNotFoundError|No such file|"
                                        r"Auth(entication)? failed|OSError|KeyInFileError|DeserializationError", re.I)),
    ("timeout", re.compile(r"TimeoutError|timed out", re.I)),
]
USERMETA_RE = re.compile(r"usermeta=\{.*\}, preload=", re.S)


def error_type(err):
    return next((name for name, rx in ERROR_TYPES if rx.search(err)), "other") if err else ""


def short_error(err):
    return USERMETA_RE.sub("usermeta={…}, preload=", err) if err else ""


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


# YEAR_STEPS key -> Snakefile rule, where they differ (plots_<cat> are handled from plot_categories)
YEAR_STEP_RULES = {"compact": "stage1Compact", "pu_dnn": "pu_dnn_train", "massData": "MassCalibrationData",
                   "massDataCl": "MassCalibrationDataClosure", "massMC": "MassCalibrationMC", "massMCCl": "MassCalibrationMCClosure"}


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
                # a restart (--restart-times) reuses the jobid: keep the earlier attempts so run time sums them all
                attempts = prev["attempts"] + [{"start": prev["start"], "end": prev["end"] or cur["start"],
                                                 "status": "failed" if prev["status"] == "restarting" else prev["status"]}] if prev else []
                job = {"jobid": jid, "rule": cur["rule"], "output": cur.get("output", ""), "log": cur.get("log", ""),
                       "wildcards": "", "status": "running", "start": cur["start"], "end": None,
                       "attempt": prev["attempt"] + 1 if prev else 1, "attempts": attempts}
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
                # it died with snakemake: count it until the log's last write, not until now
                j["end"] = j["end"] or datetime.fromtimestamp(p["mtime"], timezone.utc)

    run_tag = next((m.group(1) for j in jobs if (m := re.match(r"state/([^/]+)/", j["output"]))), "")
    cmd = proc["cmd"] if proc else ""
    if not run_tag:
        run_tag = (re.search(r"run_tag=(\S+)", cmd) or [None, ""])[1]
    cfg = run_config(cmd)

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
        # summed over every attempt of this job (earlier ones end where Snakemake restarted it)
        wall_s = sum((a["end"] - a["start"]).total_seconds() for a in j["attempts"]) + (end - j["start"]).total_seconds()
        lines = tail(REPO / j["log"]) if j["log"] and j["status"] != "done" else []
        info = [ln for ln in lines if not PROGRESS_RE.match(ln)]  # skip bare progress-bar redraws ("0%|")
        try:
            log_age = (now.timestamp() - os.path.getmtime(REPO / j["log"])) if j["log"] and j["status"] in ("running", "restarting") else None
        except OSError:
            log_age = None
        job_rows.append({
            "jobid": j["jobid"], "rule": j["rule"], "name": name, "wildcards": j["wildcards"], "log": j["log"],
            "status": j["status"], "attempt": j["attempt"], "start": et(j["start"]), "end": et(j["end"]),
            "wall": dur(wall_s), "start_ts": j["start"].timestamp(),
            "first_start_ts": (j["attempts"][0]["start"] if j["attempts"] else j["start"]).timestamp(),
            "end_ts": j["end"].timestamp() if j["end"] else None, "wall_s": wall_s,
            "attempt_wall": dur((end - j["start"]).total_seconds()),
            "attempt_list": [{"start": et(a["start"]), "end": et(a["end"]), "wall": dur((a["end"] - a["start"]).total_seconds()), "status": a["status"]}
                             for a in j["attempts"]] + [{"start": et(j["start"]), "end": et(j["end"]), "wall": dur((end - j["start"]).total_seconds()), "status": j["status"]}],
            "last": (info or lines or [""])[-1][:200], "tail": lines, "log_age_s": log_age,
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

    # rows/columns of the grids: what the run config plans, plus whatever already ran (so the grids
    # are complete from the start instead of growing as jobs start)
    cfg_years = [str(y) for y in cfg.get("years") or []]
    if {"2025", "2026"} <= set(cfg_years):
        cfg_years.append("2025_2026")  # merged stage-3 year
    cfg_regions = [short_region(None if r == "all" else r) for r in cfg.get("jj_eta_regions") or ([cfg["jj_eta_region"]] if cfg.get("jj_eta_region") else [])]
    if cfg.get("jj_combine_regions"):
        cfg_regions.append("combined")
    plot_cats = cfg.get("plot_categories") or []

    def axis(cells, idx, order=None, planned=()):
        vals = list(dict.fromkeys(list(planned) + [c.split("|")[idx] for c in cells]))
        return sorted(vals, key=order) if order else ordered_regions(vals)

    def year_step_planned(k):
        if k.startswith("plots_"):
            return "plots" in rules and k.removeprefix("plots_") in plot_cats
        return YEAR_STEP_RULES.get(k, k) in rules

    events = [{"time": et(t), "ts": t.timestamp(), "kind": k, "rule": j["rule"], "jobid": j["jobid"],
               "name": os.path.basename(j["output"].split(",")[0].strip())} for t, k, j in p["events"]][-60:][::-1]
    done, total, pct = p["steps"]
    if not total:
        total = sum(r["total"] for r in rules.values())
    started = utc_from_logname(path)
    stage1 = [s for s in (stage1_details(j, j["status"] in ("running", "restarting"))
                          for j in {j["log"]: j for j in jobs if j["rule"] == "stage1" and j["log"]}.values()) if s]
    # a stage-1 job's log only shows progress bars: say which chunks it is on, from its ledger
    by_log = {s["log"]: s for s in stage1}
    for row in job_rows:
        s = by_log.get(row["log"])
        if s and row["status"] in ("running", "restarting"):
            run = [f"{c['sample']}#{c['idx']}" for c in s["chunks"] if c["status"] == "running"]
            row["last"] = (f"running {', '.join(run[:3])}{' …' if len(run) > 3 else ''}" if run else "between chunks") + \
                f" · {s['counts']['done']}/{len(s['chunks']) - s['counts'].get('skipped', 0)} chunks done"
            row["log_age_s"] = min(x for x in (row["log_age_s"], s.get("ledger_age_s")) if x is not None) if row["log_age_s"] is not None else s.get("ledger_age_s")
    stage2_years = [y for y in cfg_years if y != "2025_2026"]
    return {
        "id": Path(path).name, "log": os.path.relpath(path, REPO), "state": state, "pid": proc["pid"] if proc else None,
        "cmd": cmd, "run_tag": run_tag, "started": et(started), "started_ts": started.timestamp() if started else 0,
        "updated": et(datetime.fromtimestamp(p["mtime"], timezone.utc)),
        "elapsed": dur(((now if proc else datetime.fromtimestamp(p["mtime"], timezone.utc)) - started).total_seconds()) if started else "",
        "steps_done": done, "steps_total": total, "pct": pct if total else 0,
        "rules": list(rules.values()), "jobs": job_rows[::-1], "events": events,
        "chain": {"years": axis(chain, 0, year_key, cfg_years if "stage2" in rules else ()),
                  "regions": axis(chain, 1, planned=[r for r in cfg_regions if r != "combined"] if "stage2" in rules else ()),
                  "steps": [[k, d] for k, d, _ in CHAIN_STEPS], "cells": chain},
        "year_steps": {"years": axis(year_cells, 0, year_key, stage2_years),
                       "steps": [[k, d] for k, d, _ in YEAR_STEPS if year_step_planned(k) or any(c.endswith("|" + k) for c in year_cells)],
                       "cells": year_cells},
        # a step planned in this run (rule in the job stats) gets its column before its first job starts
        "global": {"regions": axis(global_cells, 0, planned=cfg_regions if any(k in rules for k, _, _ in GLOBAL_STEPS) else ()),
                   "steps": [[k, d] for k, d, _ in GLOBAL_STEPS if k in rules or any(c.endswith("|" + k) for c in global_cells)],
                   "cells": global_cells},
        "stats": stats_results(stat_jobs, global_cells, served, cfg.get("status_reference")) if stat_jobs else [],
        "config": config_summary(cfg),
        "dnn": dnn_progress(cfg),
        "stage1": stage1,
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
    samples, sample_events = [], {}
    if year and nano:
        try:
            prestage = json.load(open(REPO / "prestage_output" / f"processor_samples_{year}_NanoAODv{nano}{'_' + tag if tag else ''}.json"))
            samples = list(prestage)
            for s, v in prestage.items():  # events per sample, for the ETA of samples not processed yet
                md = v.get("metadata") or {}
                sample_events[s] = num(md.get("nGenEvts")) or num(md.get("data_entries"))
        except (OSError, ValueError, AttributeError):
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
    try:
        ledger_age = now - os.path.getmtime(sdir / "job_status.jsonl")
    except OSError:
        ledger_age = None
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
            "redirector": meta.get("redirector") or "", "error": short_error(rec.get("error") or "")[-1500:],
            "error_type": error_type(rec.get("error") or "") if status in ("failed", "killed") else "",
        })
    seen = {r["sample"] for r in rows}
    rows += [{"sample": s, "idx": "", "status": "skipped" if s in skipped else "pending", "tries": 0, "earlier": False, "start": "", "end": "", "wall": "",
              "expected": None if s in skipped else sample_events.get(s), "processed": None, "redirector": "", "error": "", "error_type": ""}
             for s in samples if s not in seen]
    # alphabetical by sample, chunks in numeric order (status is its own column)
    rows.sort(key=lambda r: (r["sample"].lower(), r["idx"] if isinstance(r["idx"], int) else -1, str(r["idx"])))
    counts = {k: sum(r["status"] == k for r in rows) for k in S1_ORDER}
    # ETA from this job's event throughput (done chunks since it started) over the events still to do
    left = [r for r in rows if r["status"] not in ("done", "skipped")]
    ev_left = sum((r["expected"] if r["expected"] is not None else sample_events.get(r["sample"])) or 0 for r in left)
    fresh = [r for r in rows if r["status"] == "done" and not r["earlier"]]
    ev_done = sum(r["processed"] or r["expected"] or 0 for r in fresh)
    eta = None
    if alive and left and len(fresh) >= 3 and ev_done and now > t0:
        secs = ev_left / (ev_done / (now - t0))
        eta = {"left": dur(secs), "at": et(datetime.fromtimestamp(now + secs, timezone.utc)), "ts": now + secs}
    events = {"processed": sum(r["processed"] or 0 for r in rows if r["status"] == "done"),
              "total": sum(r["processed"] or 0 for r in rows if r["status"] == "done") + ev_left}
    return {"year": year or "", "dir": str(sdir), "n_samples": len(samples), "counts": counts, "chunks": rows,
            "eta": eta, "events": events, "alive": alive, "log": job["log"], "ledger_age_s": ledger_age,
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


def stats_results(stat_jobs, cells, served, reference=None):
    """Per jj region: significance / limit per year and the impact + scan plots of the run's stats jobs.

    stat_jobs: {raw region: {step key: job}}; a result older than its job's start is flagged stale
    (left over from an earlier run into the same card dir). reference ({label, postfix}, the run config's
    status_reference) adds that run's values for the same region/year as ref_* columns."""
    out = []
    for region, by_key in stat_jobs.items():
        hdrs = {k: log_header(j["log"]) for k, j in by_key.items()}
        # any of the region's stats job logs carries the same paths; Years differs (stats vs scan years)
        hdr = next((h for h in hdrs.values() if {"Save path", "Output append", "Label"} <= h.keys()), None)
        if hdr is None:
            continue
        cdir, save_postfix = card_dir(hdr, region), hdr["Output append"]
        short = short_region(region)
        ref_dir = None
        if isinstance(reference, dict) and reference.get("label") and reference.get("postfix"):
            ref_hdr = dict(hdr, **{"Save path": str(Path(hdr["Save path"]).parent / reference["label"]),
                                   "Label": reference["label"], "Output append": reference["postfix"]})
            ref_dir = card_dir(ref_hdr, region)

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
                if ref_dir:
                    for col, (path, regex) in files.items():
                        p = ref_dir / Path(path).name
                        row["ref_" + col] = first_value(p, regex) if p.exists() else None
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
    return {"generated": et(now), "generated_ts": now.timestamp(), "n_running": len(procs), "runs": runs,
            "proxy": proxy_status()}


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
