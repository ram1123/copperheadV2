#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import re
from pathlib import Path
from datetime import datetime

DEFAULT_TAGS = ["TODO", "FIXME", "XXX", "HACK", "BUG", "OPTIMIZE", "DEPRECATED", "NOTE"]

DEFAULT_IGNORE_DIRS = {
    ".git", ".github", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", ".venv", "venv", "build", "dist", "_site", "site", "node_modules",
    ".ipynb_checkpoints"
}

DEFAULT_EXTS = {".py", ".sh", ".bash", ".zsh", ".yml", ".yaml", ".toml", ".md", ".rst", ".txt", ".ipynb"}


def iter_files(root: Path, exts: set[str], ignore_dirs: set[str]):
    for dirpath, dirnames, filenames in os.walk(root):
        # prune ignored dirs in-place
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in exts:
                yield p


def main():
    ap = argparse.ArgumentParser(description="Collect TODO/FIXME/etc markers from repository.")
    ap.add_argument("--root", default=".", help="Repo root (default: .)")
    ap.add_argument("--out", default="docs/TODO_SUMMARY.md", help="Output markdown file")
    ap.add_argument("--tags", default=",".join(DEFAULT_TAGS), help="Comma-separated tags")
    ap.add_argument("--exts", default=",".join(sorted(DEFAULT_EXTS)), help="Comma-separated extensions")
    ap.add_argument("--ignore-dirs", default=",".join(sorted(DEFAULT_IGNORE_DIRS)), help="Comma-separated dir names to ignore")
    ap.add_argument("--max-per-file", type=int, default=200, help="Safety cap per file")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = (root / args.out).resolve()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    exts = {e.strip() if e.strip().startswith(".") else f".{e.strip()}"
            for e in args.exts.split(",") if e.strip()}
    ignore_dirs = {d.strip() for d in args.ignore_dirs.split(",") if d.strip()}

    # Match lines like: TODO:, TODO(...):, FIXME - ..., etc. (case-insensitive)
    tag_re = r"|".join(re.escape(t) for t in tags)
    pattern = re.compile(rf"(?P<tag>{tag_re})\b\s*(?:\([^)]+\))?\s*[:\-]?\s*(?P<msg>.*)$", re.IGNORECASE)

    results = {}  # file -> list of (lineno, tag, msg, line)
    total = 0

    for p in iter_files(root, exts, ignore_dirs):
        rel = p.relative_to(root)

        # skip generated output or huge binaries pretending to be text
        try:
            text = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue

        hits = []
        for i, line in enumerate(text, start=1):
            m = pattern.search(line)
            if not m:
                continue

            tag = m.group("tag").upper()
            msg = (m.group("msg") or "").strip()
            hits.append((i, tag, msg, line.rstrip()))
            total += 1

            if len(hits) >= args.max_per_file:
                hits.append((i, "NOTE", f"Truncated after {args.max_per_file} matches in this file.", ""))
                break

        if hits:
            results[str(rel)] = hits

    # write markdown
    out.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines = []
    lines.append(f"# TODO Summary\n")
    lines.append(f"_Generated: {now}_\n")
    lines.append(f"Found **{total}** annotations across **{len(results)}** files.\n")

    if not results:
        lines.append("No TODO/FIXME/etc markers found.\n")
    else:
        # sort by file name
        for fname in sorted(results.keys()):
            lines.append(f"## `{fname}`\n")
            lines.append("| Line | Tag | Message |")
            lines.append("|---:|:---:|---|")
            for lineno, tag, msg, raw in results[fname]:
                msg_show = msg if msg else raw.strip()
                msg_show = msg_show.replace("|", "\\|")
                lines.append(f"| {lineno} | `{tag}` | {msg_show} |")
            lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out} (items={total})")


if __name__ == "__main__":
    main()
