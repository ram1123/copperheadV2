#!/usr/bin/env python3
"""
splice_sample_across_years.py

Preserve an existing sample's stage-1 output -- across every year it exists
for -- by renaming it aside, then symlink in a replacement from a
differently-labeled reprocessing run. E.g. splicing a freshly-reprocessed DY
sample into an existing "official" label's tree, per year, so every
downstream step (compact, stage2/3, z-pT derivation, ...) that expects ONE
label directory per year keeps working unmodified, while the old sample
stays on disk for comparison rather than being lost.

Handles BOTH levels a sample can exist at -- f1_0/<sample> (raw stage-1
output) and compacted/<sample> -- for every year present, automatically
picking nanoAODv12 vs nanoAODv15 label naming per year the same way
workflow/Snakefile's YEAR_META / label_for_year() does. Nothing outside the
resolved per-year, per-level path pairs is ever touched.

YEAR_META below is a MIRROR of workflow/Snakefile's own copy -- not imported,
since the Snakefile isn't an importable module (it's parsed by Snakemake's
own DSL via `configfile:`/`rule`) -- kept in sync by hand. Same pattern
already used for modules/selection.py's PAIR_JJ_ETA_REGIONS, which documents
its own Snakefile mirror the same way.

Usage:
    python scripts/splice_sample_across_years.py \\
        --save-root /work/projects/hmm/<user>/hmm_ntuples/copperheadV1clean \\
        --old-run-tag FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation \\
        --new-run-tag FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation_DYReRun \\
        --sample dyTo2Mu_M-50_aMCatNLO \\
        [--years 2024,2025,2026] [--yes]

Years default to auto-discovery: every YEAR_META year whose OLD label
already has f1_0/<sample> on disk. Pass --years explicitly to override.

Dry-run by default (prints the full plan for every year/level, changes
nothing) -- pass --yes to actually apply it.
"""
import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Mirror of workflow/Snakefile's YEAR_META -- (era, nanoAODv) per year.
YEAR_META = {
    "2016preVFP": ("run2", "15"),
    "2016postVFP": ("run2", "15"),
    "2017": ("run2", "15"),
    "2018": ("run2", "15"),
    "2022preEE": ("run3", "12"),
    "2022postEE": ("run3", "12"),
    "2023": ("run3", "12"),
    "2023BPix": ("run3", "12"),
    "2024": ("run3", "15"),
    "2025": ("run3", "15"),
    "2026": ("run3", "15"),
}

# Levels a sample can exist at under <label>/stage1_output/<year>/. f1_0 is
# raw stage-1 output and is required on both sides; compacted is optional --
# see splice_one_level's skip-not-error handling below.
LEVELS = ("f1_0", "compacted")


def label_for(run_tag: str, year: str) -> str:
    """Mirror of workflow/Snakefile's label_for_year(), parameterized by run_tag
    instead of reading the global RUN_TAG."""
    try:
        era, nano = YEAR_META[year]
    except KeyError:
        raise SystemExit(f"Unknown year '{year}' -- not in YEAR_META, update the mirror if this is new.")
    era_tag = "Run2" if era == "run2" else "Run3"
    return f"{era_tag}_nanoAODv{nano}_{run_tag}"


def splice_one_level(old_path: Path, new_path: Path, *, level: str, year: str, apply: bool) -> None:
    if not new_path.is_dir():
        if level == "f1_0":
            print(f"[ERROR][{year}][{level}] new path does not exist: {new_path}", file=sys.stderr)
            sys.exit(1)
        print(f"[skip][{year}][{level}] new path does not exist yet, nothing to splice: {new_path}")
        return

    if not old_path.is_dir():
        print(f"[ERROR][{year}][{level}] old path does not exist: {old_path}", file=sys.stderr)
        sys.exit(1)
    if old_path.is_symlink():
        print(
            f"[ERROR][{year}][{level}] old path is already a symlink -- refusing to touch it "
            f"(looks like this level was already spliced): {old_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Absolute so the symlink stays valid regardless of which directory it's
    # later read from (a relative target resolves relative to the symlink's
    # own location, not the caller's cwd).
    new_path_abs = new_path.resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = old_path.parent / f"{old_path.name}_preRerun_{timestamp}"
    if backup_path.exists():
        print(f"[ERROR][{year}][{level}] backup destination already exists: {backup_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[{year}][{level}] Plan:")
    print(f"    mv  '{old_path}'  '{backup_path}'")
    print(f"    ln -s  '{new_path_abs}'  '{old_path}'")

    if not apply:
        return

    shutil.move(str(old_path), str(backup_path))
    old_path.symlink_to(new_path_abs, target_is_directory=True)

    print(f"[{year}][{level}] Done. Old data preserved at: {backup_path}")
    print(f"[{year}][{level}] {old_path} -> {os.readlink(old_path)}")


def discover_years(save_root: Path, old_run_tag: str, sample: str) -> list[str]:
    """Every YEAR_META year whose OLD label already has f1_0/<sample> on disk."""
    found = []
    for year in YEAR_META:
        f1_0_path = save_root / label_for(old_run_tag, year) / "stage1_output" / year / "f1_0" / sample
        if f1_0_path.is_dir():
            found.append(year)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--save-root", required=True,
        help="e.g. /work/projects/hmm/<user>/hmm_ntuples/copperheadV1clean",
    )
    parser.add_argument("--old-run-tag", required=True, help="run_tag of the label to preserve/splice into")
    parser.add_argument("--new-run-tag", required=True, help="run_tag of the label to splice FROM")
    parser.add_argument("--sample", required=True, help="sample directory name, e.g. dyTo2Mu_M-50_aMCatNLO")
    parser.add_argument(
        "--years", default=None,
        help="Comma-separated years to restrict to (default: auto-discover -- every "
             "YEAR_META year whose OLD label has f1_0/<sample> on disk).",
    )
    parser.add_argument("--yes", action="store_true", help="Actually apply (default: dry-run only).")
    args = parser.parse_args()

    save_root = Path(args.save_root)

    if args.years:
        years = [y.strip() for y in args.years.split(",") if y.strip()]
        unknown = [y for y in years if y not in YEAR_META]
        if unknown:
            raise SystemExit(f"Unknown year(s) not in YEAR_META: {unknown}")
    else:
        years = discover_years(save_root, args.old_run_tag, args.sample)
        if not years:
            raise SystemExit(
                f"No years found with an existing <old_label>/stage1_output/<year>/f1_0/"
                f"{args.sample} under {save_root} for run_tag={args.old_run_tag!r} -- "
                "pass --years explicitly if this is unexpected."
            )
        print(f"Auto-discovered years: {years}")

    print(f"{'APPLYING' if args.yes else 'DRY RUN'} -- sample={args.sample}")
    print(f"  old run_tag: {args.old_run_tag}")
    print(f"  new run_tag: {args.new_run_tag}")
    print()

    for year in years:
        old_base = save_root / label_for(args.old_run_tag, year) / "stage1_output" / year
        new_base = save_root / label_for(args.new_run_tag, year) / "stage1_output" / year

        for level in LEVELS:
            splice_one_level(
                old_base / level / args.sample,
                new_base / level / args.sample,
                level=level, year=year, apply=args.yes,
            )
        print()

    if not args.yes:
        print("Dry run only -- nothing changed. Re-run with --yes to apply.")


if __name__ == "__main__":
    main()
