from pathlib import Path


DY_GROUPS = {"DY", "DYVBF"}
# stage-2 file category -> datacard process, per DY split scheme (run_stage2_vbf.py --dy_matched_jets)
MATCHED_JET_GROUPS_BY_SCHEME = {
    "gen_2j": {
        "matched01J": "DY_matched01J",
        "matched2J": "DY_matched2J",
    },
    "reco_012": {
        "recoMatched0J": "DY_recoMatched0J",
        "recoMatched1J": "DY_recoMatched1J",
        "recoMatched2J": "DY_recoMatched2J",
    },
}
MATCHED_JET_GROUPS = MATCHED_JET_GROUPS_BY_SCHEME["gen_2j"]
# every split DY process name, for lists that must cover any scheme (e.g. LHE/PDF decorrelation)
ALL_MATCHED_DY_PROCESSES = tuple(
    process for groups in MATCHED_JET_GROUPS_BY_SCHEME.values() for process in groups.values()
)


def stage2_histogram_directory(
    global_path, var_name, global_path_postfix, no_variations, year
):
    directory_name = var_name
    if global_path_postfix:
        directory_name += f"_{global_path_postfix}"
        if no_variations:
            directory_name += "_NoSyst"
    return Path(global_path) / "stage2_histograms" / directory_name / str(year)


def detect_dy_split_scheme(directory):
    """The DY split scheme the stage-2 histograms in `directory` were made with, or None."""
    found = [
        scheme
        for scheme, groups in MATCHED_JET_GROUPS_BY_SCHEME.items()
        if any(any(Path(directory).glob(f"*_{c}_hist.pkl")) for c in groups)
    ]
    if len(found) > 1:
        raise ValueError(f"{directory} mixes DY split schemes {found}; rerun stage-2 into a clean directory.")
    return found[0] if found else None


def has_matched_jet_histograms(directory):
    return detect_dy_split_scheme(directory) is not None


def split_dy_grouping(grouping, keep_sample_groups=False, scheme="gen_2j"):
    """Map the matched-jet DY histograms to the scheme's DY_* processes, or with
    keep_sample_groups back to their sample group (DY, DYVBF) so they stay separate."""
    split_grouping = {}
    for dataset, group in grouping.items():
        if group not in DY_GROUPS:
            split_grouping[dataset] = group
            continue

        for filename_category, matched_group in MATCHED_JET_GROUPS_BY_SCHEME[scheme].items():
            split_grouping[f"{dataset}_{filename_category}"] = (
                group if keep_sample_groups else matched_group
            )

    return split_grouping
