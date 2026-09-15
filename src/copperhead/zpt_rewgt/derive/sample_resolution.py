"""Resolve which DY MC sample(s) a given year's Z-pT derivation actually reads.

Filesystem/YAML-only (globs + a sample-config lookup) - no parquet reading -
so this is cheap enough to call from get_polyFit.py just for provenance,
without paying save_SF_rootFiles.py's full dask read.
"""
import glob
import os

from modules.sample_config import get_sample_dict


def resolve_dy_processes(year, sample_config_path):
    bkg_dict = get_sample_dict(
        yaml_path=sample_config_path,
        section="background",
        year=str(year),
        selected_groups=["DY"],
    )
    dy_processes = bkg_dict.get("DY", [])
    if not dy_processes:
        raise ValueError(
            f"No DY processes resolved from sample config '{sample_config_path}' for year '{year}'."
        )
    return dy_processes


def collect_process_paths(base_path, process_names):
    parquet_paths = []
    matched_processes = []
    missing_processes = []
    for process in process_names:
        pattern = f"{base_path}/{process}/*/*.parquet"
        found = sorted(glob.glob(pattern))
        if found:
            parquet_paths.extend(found)
            matched_processes.append(process)
        else:
            missing_processes.append(process)
    if not parquet_paths:
        raise RuntimeError(
            f"No parquet files found for DY processes {process_names} under base path: {base_path}"
        )
    return parquet_paths, matched_processes, missing_processes


def resolve_stage1_base_path(input_path, year):
    """Same 'compacted' -> 'f1_0' fallback save_SF_rootFiles.py uses."""
    base_path = f"{input_path}/stage1_output/{year}/compacted"
    if not os.path.exists(base_path):
        base_path = base_path.replace("compacted", "f1_0")
    return base_path
