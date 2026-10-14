"""Generate the "API Reference" pages of the docs site from the Python sources.

Run by the mkdocs-gen-files plugin at build time: for every .py file under API_ROOTS it writes a
virtual page api/<path>.md containing a mkdocstrings `:::` directive, plus api/SUMMARY.md, which
mkdocs-literate-nav turns into the sidebar tree. Nothing is written to disk.

Folders without __init__.py (e.g. src/corrections) are namespace packages, which griffe can't reach
through their parent package; such a folder is loaded as a top-level package from its parent dir
(that dir must be listed under mkdocstrings `paths` in mkdocs.yml) and the page heading still shows
the real import path (src.corrections.jet).

How to run: not run directly; mkdocs runs it via `plugins: gen-files` in mkdocs.yml.

Example:
    mkdocs serve      # then open "Reference > API Reference"
"""

from pathlib import Path

import mkdocs_gen_files

API_ROOTS = ["src", "modules", "cli"]
SKIP_PARTS = {"__pycache__"}

nav = mkdocs_gen_files.Nav()


def griffe_target(path):
    """Return (identifier griffe can load, real dotted import path) for a .py file."""
    parts = list(path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    real = ".".join(parts)
    # Walk up from the file's folder while folders are regular packages.
    pkg_dir = path.parent
    while pkg_dir != Path(".") and (pkg_dir / "__init__.py").exists():
        pkg_dir = pkg_dir.parent
    if pkg_dir == Path("."):
        return real, real
    # pkg_dir is a namespace folder: load it as a top-level package from its parent dir.
    start = len(pkg_dir.parts) - 1
    return ".".join(parts[start:]), real


for path in sorted(p for root in API_ROOTS for p in Path(root).rglob("*.py")):
    if SKIP_PARTS & set(path.parts) or any(part.startswith(".") for part in path.parts):
        continue  # caches and editor/Jupyter checkpoint copies
    identifier, real = griffe_target(path)
    doc_parts = list(path.with_suffix("").parts)
    if doc_parts[-1] == "__init__":
        doc_parts = doc_parts[:-1]
        doc_path = Path("api", *doc_parts, "index.md")
    else:
        doc_path = Path("api", *doc_parts).with_suffix(".md")
    nav[tuple(doc_parts)] = doc_path.relative_to("api").as_posix()

    with mkdocs_gen_files.open(doc_path, "w") as fd:
        fd.write(f"# {real}\n\n")
        fd.write(f"Source: `{path.as_posix()}`\n\n")
        fd.write(f"::: {identifier}\n    options:\n      show_root_heading: false\n")
    mkdocs_gen_files.set_edit_path(doc_path, Path("..", path))

# Landing page; listed first so section-index uses it (not modules/index.md) as the section page.
with mkdocs_gen_files.open("api/index.md", "w") as fd:
    fd.write(
        "# API Reference\n\n"
        "Generated at build time from the docstrings and signatures in "
        + ", ".join(f"`{r}/`" for r in API_ROOTS)
        + ", one page per module. Undocumented functions and classes are listed too, with their "
        "source code. Edit the docstrings in the code to improve these pages.\n"
    )

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as fd:
    fd.write("* [Overview](index.md)\n")
    fd.writelines(nav.build_literate_nav())
