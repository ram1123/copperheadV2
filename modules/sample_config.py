from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml


def _as_str_year(year: Any) -> str:
    return str(year)


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"YAML not found: {path}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Top-level YAML must be a dict, got {type(cfg)}")
    return cfg


def _get_groups(cfg: Dict[str, Any], section: str) -> Dict[str, Any]:
    sec = cfg.get(section, {})
    if not isinstance(sec, dict):
        raise ValueError(f"Section '{section}' must be a dict, got {type(sec)}")
    groups = sec.get("groups", {})
    if not isinstance(groups, dict):
        raise ValueError(f"'{section}.groups' must be a dict, got {type(groups)}")
    return groups


def _resolve_group(group_cfg: Dict[str, Any], year: str) -> List[str]:
    """
    Rule:
      - if processes_per_year has this year -> return that list (override)
      - else -> return default processes
    """
    default_procs = group_cfg.get("processes", []) or []
    per_year = group_cfg.get("processes_per_year", {}) or {}

    if not isinstance(default_procs, list):
        raise ValueError(f"'processes' must be a list, got {type(default_procs)}")
    if not isinstance(per_year, dict):
        raise ValueError(f"'processes_per_year' must be a dict, got {type(per_year)}")

    # Normalize per_year keys (YAML may load 2024 as int)
    per_year_norm = {str(k): v for k, v in per_year.items()}

    if year in per_year_norm:
        procs = per_year_norm[year] or []
    else:
        procs = default_procs

    if not isinstance(procs, list):
        raise ValueError(f"Resolved processes must be a list, got {type(procs)}")
    for p in procs:
        if not isinstance(p, str):
            raise ValueError(f"Process names must be strings, got {type(p)} in {procs}")
    return procs


def get_sample_dict(
    yaml_path: str | Path,
    section: str,
    year: str,
    selected_groups: Sequence[str] | None = None,
) -> Dict[str, List[str]]:
    """
    Returns a dict like:
      {"DY": [...], "TT": [...], ...}

    If selected_groups is None, returns all groups in that section.
    """
    cfg = _load_yaml(yaml_path)
    groups = _get_groups(cfg, section)
    year = _as_str_year(year)

    if selected_groups is None:
        selected_groups = list(groups.keys())

    out: Dict[str, List[str]] = {}
    for g in selected_groups:
        if g not in groups:
            raise KeyError(
                f"Group '{g}' not found in section '{section}'. Available: {list(groups.keys())}"
            )
        gcfg = groups[g]
        if not isinstance(gcfg, dict):
            raise ValueError(
                f"Group '{section}.groups.{g}' must be a dict, got {type(gcfg)}"
            )
        out[g] = _resolve_group(gcfg, year)

    return out


def get_bkg_sig_dicts(
    yaml_path: str | Path,
    year: str,
    bkg_groups: Sequence[str] | None = None,
    sig_groups: Sequence[str] | None = None,
) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Convenience: returns (bkg_sample_dict, sig_sample_dict)
    """
    bkg = get_sample_dict(
        yaml_path, section="background", year=year, selected_groups=bkg_groups
    )
    sig = get_sample_dict(
        yaml_path, section="signal", year=year, selected_groups=sig_groups
    )

    # Join the two dicts
    combined = {**bkg, **sig}

    return bkg, sig, combined


def get_data_processes(yaml_path: str | Path, year: str) -> List[str]:
    """
    Returns the list of data process names (e.g. ["data_C", "data_D"]) for a
    given year from the top-level "data" section of the sample config YAML.

    Unlike background/signal groups, this is a flat {year: [processes]} map
    (no "processes"/"processes_per_year" default+override split) since every
    year's set of data-taking eras is genuinely distinct. Also accepts the
    "run2"/"run3" aggregate keys used for combined-year runs.
    """
    cfg = _load_yaml(yaml_path)
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError(f"'data' section must be a dict, got {type(data_cfg)}")

    year = _as_str_year(year)
    if year not in data_cfg:
        raise KeyError(
            f"Year '{year}' not found in 'data' section. Available: {list(data_cfg.keys())}"
        )
    procs = data_cfg[year] or []
    if not isinstance(procs, list):
        raise ValueError(f"'data.{year}' must be a list, got {type(procs)}")
    return list(procs)


def get_grouping_dict(
    yaml_path: str | Path,
    year: str,
) -> Dict[str, str]:
    """
    Returns:
      parameters["grouping"] style dict:
        process_name -> plotting group label
    """
    cfg = _load_yaml(yaml_path)
    year = _as_str_year(year)

    grouping: Dict[str, str] = {}

    # ---- Data (always)
    grouping["data"] = "Data"

    # ---- Background
    bkg_groups = _get_groups(cfg, "background")
    for group_name, group_cfg in bkg_groups.items():
        procs = _resolve_group(group_cfg, year)

        print(f"Group '{group_name}' processes: {procs}")

        # Merge TT + ST into TT+ST (your convention)
        if group_name in {"TT", "ST"}:
            label = "TT+ST"
        else:
            label = group_name

        for p in procs:
            grouping[p] = label

    # ---- Signal
    print("\nProcessing signal groups...")
    sig_groups = _get_groups(cfg, "signal")
    for group_name, group_cfg in sig_groups.items():
        procs = _resolve_group(group_cfg, year)

        print(f"Group '{group_name}' processes: {procs}")

        # Map to physics-style labels
        if group_name in {"GGH"}:
            print("  Mapping to 'ggH_hmm'")
            label = "ggH_hmm"
        elif group_name in {"VBF"}:
            print("  Mapping to 'qqH_hmm'")
            label = "qqH_hmm"
        else:
            # e.g. HIGGS → keep group name
            print(f"  Mapping to '{group_name}'")
            label = group_name

        for p in procs:
            grouping[p] = label
            print(f"  Mapping process '{p}' to label '{label}'")

    return grouping


def get_all_dicts(
    yaml_path: str | Path,
    year: str,
):
    """
    Convenience function.

    Returns:
      bkg_sample_dict, sig_sample_dict, grouping_dict
    """
    bkg = get_sample_dict(yaml_path, "background", year)
    sig = get_sample_dict(yaml_path, "signal", year)
    grouping = get_grouping_dict(yaml_path, year)
    return bkg, sig, grouping


def main():
    yaml_path = "configs/samples/samples.yaml"
    year = "2023BPix"

    bkg_sample_dict, sig_sample_dict, combined_sample_dict = get_bkg_sig_dicts(
        yaml_path=yaml_path,
        year=year,
    )

    # print("Background samples:")
    # print(bkg_sample_dict)
    # # for group, samples in bkg_sample_dict.items():
    # # print(f"  {group}: {samples}")

    # print("\nSignal samples:")
    # print(sig_sample_dict)
    # # for group, samples in sig_sample_dict.items():
    # # print(f"  {group}: {samples}")

    # print("\nCombined samples:")
    print(combined_sample_dict)

    # list of all samples (processes) across background and signal
    all_samples = set()
    for samples in combined_sample_dict.values():
        all_samples.update(samples)
    print("\nAll samples (processes) across background and signal:")
    print(all_samples)
    # bkg, sig, grouping = get_all_dicts(yaml_path, year)

    # print("Background samples:")
    # print(bkg)

    # print("\nSignal samples:")
    # print(sig)

    # print("\nGrouping dict:")
    # print(grouping)
    # for proc, label in grouping.items():
    #     print(f"  {proc} -> {label}")

if __name__ == "__main__":
    main()
