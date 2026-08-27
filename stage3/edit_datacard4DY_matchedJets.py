from pathlib import Path


DY_GROUPS = {"DY", "DYVBF"}
MATCHED_JET_GROUPS = {
    "matched01J": "DY_matched01J",
    "matched2J": "DY_matched2J",
}


def stage2_histogram_directory(
    global_path, var_name, global_path_postfix, no_variations, year
):
    directory_name = var_name
    if global_path_postfix:
        directory_name += f"_{global_path_postfix}"
        if no_variations:
            directory_name += "_NoSyst"
    return Path(global_path) / "stage2_histograms" / directory_name / str(year)


def has_matched_jet_histograms(directory):
    return any(Path(directory).glob("*matched*_hist.pkl"))


def split_dy_grouping(grouping):
    split_grouping = {}
    for dataset, group in grouping.items():
        if group not in DY_GROUPS:
            split_grouping[dataset] = group
            continue

        for filename_category, matched_group in MATCHED_JET_GROUPS.items():
            split_grouping[f"{dataset}_{filename_category}"] = matched_group

    return split_grouping
