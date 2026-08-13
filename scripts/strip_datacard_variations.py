#!/usr/bin/env python3
"""Drop shape nuisances from Stage-3 datacards, keeping only a chosen few.

Why this exists
---------------
The adversarial DNN training is only ever shown a *subset* of the shape
variations -- the sweep set is JEC ``Total`` plus ``mu_roccor``, up and down.
Stage-2/Stage-3, however, always build templates for every discovered variation
(13 shape nuisances per year in the Run-2 setup).  The pre-fit significance
therefore carries eleven systematics the network was never trained against,
which dilutes the very effect the study is trying to measure.

Running this immediately before ``produce_combine_cards.py``/``combine`` makes
the pre-fit vs stat-only gap reflect exactly the variations the loss targets.

What it touches
---------------
Only lines whose *type* column begins with ``shape`` (``shape``, ``shapeN``,
``shape?``).  ``lnN`` normalisation nuisances and the ``autoMCStats`` line are
left alone -- they are not shape variations and the DNN cannot decorrelate
against them.  ``kmax`` is ``*`` in these cards, so no counter needs updating.

Year suffixes are handled: a nuisance is kept when its name with any trailing
year token removed is in the keep set, so ``--keep Total,mu_roccor`` retains
both ``Total`` and ``mu_roccor2017`` while dropping ``Absolute``,
``Absolute_2017``, ``RelativeSample_2017`` and the rest.

Reversibility
-------------
The untouched card is saved once as ``<name>.txt.full``.  Every run rewrites the
card *from that backup*, so the script is idempotent and ``--restore`` puts the
directory back exactly as Stage-3 produced it.

Examples
--------
    python scripts/strip_datacard_variations.py <stage3_dir> --years 2017
    python scripts/strip_datacard_variations.py <stage3_dir> --dry-run
    python scripts/strip_datacard_variations.py <stage3_dir> --restore
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

#: Sources the sweep-phase adversarial training decorrelates against; see
#: ``modules.systematics.SWEEP_VARIATION_SOURCES``.
DEFAULT_KEEP = ("Total", "mu_roccor")

#: Trailing data-taking-year token, with or without a leading underscore:
#: ``Absolute_2017`` -> ``Absolute``, ``mu_roccor2017`` -> ``mu_roccor``,
#: ``RelativeSample_2016APV`` -> ``RelativeSample``.
YEAR_SUFFIX_RE = re.compile(r"_?(?:19|20)\d{2}[A-Za-z]*$")

BACKUP_SUFFIX = ".full"


def base_name(nuisance: str) -> str:
    """Nuisance name with any trailing year token removed."""
    return YEAR_SUFFIX_RE.sub("", nuisance)


def is_shape_line(line: str) -> Tuple[bool, str]:
    """(True, nuisance_name) if this is a shape-nuisance row."""
    parts = line.split()
    if len(parts) < 2:
        return False, ""
    if not parts[1].startswith("shape"):
        return False, ""
    return True, parts[0]


def strip_text(text: str, keep: Sequence[str]) -> Tuple[str, List[str], List[str]]:
    """Return (new_text, kept_shape_nuisances, dropped_shape_nuisances)."""
    keep_set = set(keep)
    out, kept, dropped = [], [], []
    for line in text.splitlines(keepends=True):
        shape, name = is_shape_line(line)
        if not shape:
            out.append(line)
            continue
        if base_name(name) in keep_set:
            kept.append(name)
            out.append(line)
        else:
            dropped.append(name)
    return "".join(out), kept, dropped


def card_files(directory: Path, years: Sequence[str] | None) -> List[Path]:
    """Per-region Stage-3 datacards, optionally filtered to given years.

    Deliberately excludes the ``HMuMu_13TeV_*.txt`` products of
    ``combineCards.py``: those are built *from* these files, so stripping has to
    happen before that step, not after.
    """
    if years:
        found: List[Path] = []
        for year in years:
            found += sorted(directory.glob(f"datacard_vbf_*_{year}.txt"))
        return found
    return sorted(directory.glob("datacard_vbf_*.txt"))


def restore(files: Sequence[Path]) -> int:
    n = 0
    for path in files:
        backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
        if backup.exists():
            shutil.copyfile(backup, path)
            print(f"  restored {path.name} from {backup.name}")
            n += 1
    return n


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("datacard_dir", help="Stage-3 dir holding datacard_vbf_<REGION>_<YEAR>.txt")
    ap.add_argument("--keep", default=",".join(DEFAULT_KEEP),
                    help="comma list of shape sources to keep, year suffixes ignored")
    ap.add_argument("--years", default=None,
                    help="comma list of years; default: every datacard in the directory")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    ap.add_argument("--restore", action="store_true",
                    help=f"copy each <card>{BACKUP_SUFFIX} back over its card and exit")
    args = ap.parse_args(argv)

    directory = Path(args.datacard_dir)
    if not directory.is_dir():
        print(f"not a directory: {directory}", file=sys.stderr)
        return 2

    years = [y.strip() for y in args.years.split(",") if y.strip()] if args.years else None
    files = card_files(directory, years)
    if not files:
        print(f"no datacard_vbf_*.txt found in {directory}"
              + (f" for years {years}" if years else ""), file=sys.stderr)
        return 2

    if args.restore:
        print(f"restoring {len(files)} datacard(s) in {directory}")
        return 0 if restore(files) else 2

    keep = [k.strip() for k in args.keep.split(",") if k.strip()]
    print(f"{'would strip' if args.dry_run else 'stripping'} {len(files)} datacard(s) "
          f"in {directory}\n  keeping shape sources: {', '.join(keep)}")

    for path in files:
        backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
        # Always re-derive from the pristine card so repeated runs are idempotent
        # and a changed --keep widens the set rather than only narrowing it.
        source = backup if backup.exists() else path
        text = source.read_text()
        new_text, kept, dropped = strip_text(text, keep)

        print(f"  {path.name}: keep {len(kept)} {kept} | drop {len(dropped)} {dropped}")
        if args.dry_run:
            continue
        if not backup.exists():
            shutil.copyfile(path, backup)
        path.write_text(new_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
