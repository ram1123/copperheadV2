#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

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
    # pattern = re.compile(rf"(?P<tag>{tag_re})\b\s*(?:\([^)]+\))?\s*[:\-]?\s*(?P<msg>.*)$", re.IGNORECASE)
    pattern = re.compile(rf"(?P<tag>{tag_re})\b\s*(?:\([^)]+\))?\s*[:\-]?\s*(?P<msg>.*)$")

    # results = {}  # file -> list of (lineno, tag, msg, line)
    results = defaultdict(lambda: defaultdict(list))
    # results[tag][file] -> list of (lineno, msg)

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
            line_no_nl = line.rstrip()

            if "#" not in line_no_nl:
                continue

            comment = line_no_nl.split("#", 1)[1].strip()
            if not comment:
                continue

            m = pattern.search(comment)
            if not m:
                continue

            tag = m.group("tag")  # case-sensitive, already uppercase
            msg = (m.group("msg") or "").strip()

            results[tag][str(rel)].append((i, msg))
            total += 1

            if len(results[tag][str(rel)]) >= args.max_per_file:
                results[tag][str(rel)].append(
                    (i, f"Truncated after {args.max_per_file} matches.")
                )
                break

        if hits:
            results[str(rel)] = hits

    # write markdown
    out.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines = []
    lines.append("# TODO Summary\n")
    lines.append(f"_Generated: {now}_\n")
    lines.append(f"Found **{total}** annotations.\n")

    if not results:
        lines.append("No TODO/FIXME/etc markers found.\n")
    else:
        for tag in tags:
            if tag not in results:
                continue

            lines.append(f"## {tag}\n")

            for fname in sorted(results[tag].keys()):
                lines.append(f"### `{fname}`\n")
                lines.append("| Line | Message |")
                lines.append("|---:|---|")

                for lineno, msg in results[tag][fname]:
                    msg_show = msg.replace("|", "\\|")
                    github_base = (
                        "https://github.com/ram1123/copperheadV2/blob/dev_docs"
                    )
                    src_link = f"{github_base}/{fname}#L{lineno}"
                    lines.append(f"| [{lineno}]({src_link}) | {msg_show} |")

                lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out} (items={total})")


if __name__ == "__main__":
    main()
