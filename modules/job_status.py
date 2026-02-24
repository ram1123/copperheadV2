from datetime import datetime
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional


class JobStatus:
    """
    File-based resumability markers + an append-only JSONL ledger.

    Creates marker files:
      <dataset>__<idx>.running / .done / .fail

    And writes all transitions to:
      job_status.jsonl
    """

    def __init__(self, status_dir: str):
        self.dir = Path(status_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ledger = self.dir / "job_status.jsonl"

    def _paths(self, dataset: str, idx: int):
        base = f"{dataset}__{idx}"
        return (
            self.dir / f"{base}.running",
            self.dir / f"{base}.done",
            self.dir / f"{base}.fail",
        )

    def should_run(self, dataset: str, idx: int) -> bool:
        _, done, fail = self._paths(dataset, idx)
        if done.exists():
            # if it was later marked failed, allow rerun
            if fail.exists() and fail.stat().st_mtime > done.stat().st_mtime:
                return True
            return False
        return True

    def _append_ledger(self, rec: Dict[str, Any]) -> None:
        with self.ledger.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def mark_running(
        self, dataset: str, idx: int, meta: Optional[Dict[str, Any]] = None
    ) -> None:
        running, _, _ = self._paths(dataset, idx)

        ts = time.time()
        running.write_text(str(ts))  # store raw float timestamp

        self._append_ledger(
            {
                "ts": ts,
                "dataset": dataset,
                "idx": idx,
                "status": "running",
                "meta": meta or {},
            }
        )

    def mark_done(
        self, dataset: str, idx: int, meta: Optional[Dict[str, Any]] = None
    ) -> None:
        running, done, _ = self._paths(dataset, idx)

        end_ts = time.time()

        duration = None
        if running.exists():
            try:
                start_ts = float(running.read_text().strip())
                duration = end_ts - start_ts
            except Exception:
                duration = None

        done.write_text(str(end_ts))
        running.unlink(missing_ok=True)

        self._append_ledger(
            {
                "ts": end_ts,
                "dataset": dataset,
                "idx": idx,
                "status": "done",
                "meta": meta or {},
                "duration_seconds": duration,
            }
        )

    def mark_failed(
        self,
        dataset: str,
        idx: int,
        err: Exception,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        running, _, fail = self._paths(dataset, idx)
        fail.write_text(str(datetime.fromtimestamp(time.time())))
        running.unlink(missing_ok=True)
        self._append_ledger(
            {
                "ts": datetime.fromtimestamp(time.time()),
                "dataset": dataset,
                "idx": idx,
                "status": "failed",
                "meta": meta or {},
                "error": "".join(
                    traceback.format_exception_only(type(err), err)
                ).strip(),
            }
        )


def write_stage1_summary(
    status_dir: str, out_json_path: str, logger=None
) -> Dict[str, Any]:
    """
    Read job_status.jsonl (append-only ledger) and write a final summary:
      - out_json_path (JSON)
      - out_json_path with suffix .log (human-readable)

    Returns the summary dict.
    """
    status_dir_p = Path(status_dir)
    ledger = status_dir_p / "job_status.jsonl"
    if not ledger.exists():
        if logger:
            logger.warning(f"[summary] ledger not found: {ledger}")
        return {"n_done": 0, "n_failed": 0, "done": [], "failed": []}

    last = {}  # (dataset, idx) -> last record
    with ledger.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec.get("dataset"), rec.get("idx"))
            last[key] = rec

    done = []
    failed = []

    for (ds, idx), rec in last.items():
        st = rec.get("status")
        if st == "done":
            done.append({"dataset": ds, "idx": idx, "meta": rec.get("meta", {})})
        elif st == "failed":
            failed.append(
                {
                    "dataset": ds,
                    "idx": idx,
                    "error": rec.get("error", ""),
                    "meta": rec.get("meta", {}),
                }
            )

    summary = {
        "n_done": len(done),
        "n_failed": len(failed),
        "done": sorted(done, key=lambda r: (r["dataset"], r["idx"])),
        "failed": sorted(failed, key=lambda r: (r["dataset"], r["idx"])),
    }

    out_json = Path(out_json_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)

    out_log = out_json.with_suffix(".log")
    with out_log.open("w") as f:
        f.write("Stage1 summary\n")
        f.write(f"DONE: {summary['n_done']}  FAILED: {summary['n_failed']}\n\n")

        f.write("FAILED:\n")
        for r in summary["failed"]:
            redir = (r.get("meta") or {}).get("redirector", "")
            f.write(f"- {r['dataset']}[{r['idx']}]  {redir}  {r['error']}\n")

        f.write("\nDONE:\n")
        for r in summary["done"]:
            redir = (r.get("meta") or {}).get("redirector", "")
            path = (r.get("meta") or {}).get("path", "")
            dur = r.get("duration_seconds", None)

            if dur:
                mins = int(dur // 60)
                secs = int(dur % 60)
                dur_str = f"{mins}m {secs}s"
            else:
                dur_str = "unknown"

            f.write(f"- {r['dataset']}[{r['idx']}]  {dur_str}  {redir}  {path}\n")

    if logger:
        logger.info(f"[summary] wrote {out_log}")
        logger.info(f"[summary] wrote {out_json}")

    return summary
