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
# run_stage3_vbf.py --dy_scheme merge01: reco_012 histograms with 0J + 1J summed into one process
MERGE01_GROUPS = {
    "recoMatched0J": "DY_recoMatched01J",
    "recoMatched1J": "DY_recoMatched01J",
    "recoMatched2J": "DY_recoMatched2J",
}
# every split DY process name, for lists that must cover any scheme (e.g. LHE/PDF decorrelation)
ALL_MATCHED_DY_PROCESSES = tuple(dict.fromkeys(
    process
    for groups in (*MATCHED_JET_GROUPS_BY_SCHEME.values(), MERGE01_GROUPS)
    for process in groups.values()
))


def dy_process_map(scheme, merge01=False):
    """stage-2 file category -> datacard process for the stage-2 `scheme` (merge01: reco_012 only)."""
    if merge01:
        if scheme != "reco_012":
            raise ValueError(f"--dy_scheme merge01 needs reco_012 stage-2 histograms, got {scheme}.")
        return MERGE01_GROUPS
    return MATCHED_JET_GROUPS_BY_SCHEME[scheme]


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


def split_dy_grouping(grouping, keep_sample_groups=False, scheme="gen_2j", merge01=False):
    """Map the matched-jet DY histograms to the scheme's DY_* processes, or with
    keep_sample_groups back to their sample group (DY, DYVBF) so they stay separate."""
    split_grouping = {}
    for dataset, group in grouping.items():
        if group not in DY_GROUPS:
            split_grouping[dataset] = group
            continue

        for filename_category, matched_group in dy_process_map(scheme, merge01).items():
            split_grouping[f"{dataset}_{filename_category}"] = (
                group if keep_sample_groups else matched_group
            )

    return split_grouping
