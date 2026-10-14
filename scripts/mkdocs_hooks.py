"""MkDocs build hook: turn relative links to repo files outside docs/ into GitHub links.

Docs link code as e.g. [get_yields.py](../scripts/get_yields.py), which works when browsing the
repo on GitHub but 404s on the built site (only docs/ is published). At build time this hook
rewrites such links to https://github.com/<repo>/blob/<ref>/<path>, where <ref> is the branch
being built (GITHUB_REF_NAME in CI, else the local git branch, else "main"). Links to repo paths
that don't exist are reported as build warnings. The Markdown sources are not modified.

How to run: not run directly; mkdocs loads it via `hooks:` in mkdocs.yml.

Example:
    mkdocs build      # or: mkdocs serve
"""

import logging
import os
import posixpath
import re
import subprocess

log = logging.getLogger("mkdocs.hooks.github_links")

_LINK_RE = re.compile(r"\]\((?!https?://|mailto:|#)([^)\s]+)((?:\s+\"[^\"]*\")?)\)")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")

_repo_root = None
_blob_base = None


def _git_ref(repo_root):
    ref = os.environ.get("GITHUB_REF_NAME")
    if ref:
        return ref
    try:
        ref = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "--abbrev-ref", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        ref = ""
    return ref if ref and ref != "HEAD" else "main"


def on_config(config):
    global _repo_root, _blob_base
    _repo_root = os.path.dirname(os.path.abspath(config["config_file_path"]))
    repo_url = (config.get("repo_url") or "").rstrip("/")
    _blob_base = f"{repo_url}/blob/{_git_ref(_repo_root)}" if repo_url else None
    return config


def _rewrite(match, page_dir, docs_rel, src):
    target, title = match.group(1), match.group(2)
    path, _, frag = target.partition("#")
    resolved = posixpath.normpath(posixpath.join(page_dir, path)) if path else ""
    if not resolved.startswith("../"):
        return match.group(0)  # stays inside docs/: mkdocs validates it
    repo_path = posixpath.normpath(posixpath.join(docs_rel, page_dir, path))
    if repo_path.startswith("../"):
        return match.group(0)
    if not os.path.exists(os.path.join(_repo_root, repo_path)):
        log.warning("Doc file '%s' links to '%s', which does not exist in the repo.", src, repo_path)
    url = f"{_blob_base}/{repo_path}" + (f"#{frag}" if frag else "")
    return f"]({url}{title})"


def on_page_markdown(markdown, page, config, files):
    if not _blob_base:
        return markdown
    src = page.file.src_uri
    page_dir = posixpath.dirname(src)
    docs_rel = posixpath.relpath(os.path.abspath(config["docs_dir"]), _repo_root).replace(os.sep, "/")
    out, in_fence = [], False
    for line in markdown.split("\n"):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        elif not in_fence:
            line = _LINK_RE.sub(lambda m: _rewrite(m, page_dir, docs_rel, src), line)
        out.append(line)
    return "\n".join(out)
