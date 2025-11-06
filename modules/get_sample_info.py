import yaml

def get_sample_info(yaml_file, sample_name, year_key="2022preEE"):
    """
    Retrieve info (cross section or lumi) for a given sample and year.

    Works for both MC (e.g. 'dyTo2L_M-50_0j', 'tt_inclusive') and Data ('data_C', 'data_D').

    Parameters
    ----------
    yaml_file : str
        Path to the YAML configuration file.
    sample_name : str
        Sample key (e.g. 'dyTo2L_M-50_0j', 'tt_inclusive', 'data_C').
    year_key : str, optional
        Year or era key (default '2022preEE').

    Returns
    -------
    dict or None
        Dictionary with relevant fields (cross section or lumi, kfactor, references, etc.).
    """
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    year_block = data.get("years", {}).get(year_key, {})
    total_lumi = year_block.get("Data", {}).get("total_lumi_pb", None)

    for process_group, samples in year_block.items():
        if not isinstance(samples, dict):
            continue

        for sample, info in samples.items():
            if sample == sample_name:
                result = {
                    "year": year_key,
                    "process_group": process_group,
                    "sample": sample,
                    "total_lumi_pb": total_lumi,
                }

                # --- Data samples ---
                if process_group.lower() == "data":
                    result.update({
                        "lumi_pb": info.get("lumi_pb"),
                        "datasets": info.get("datasets", []),
                        "das_request": info.get("das_request"),
                        "golden_json": info.get("golden_json"),
                        "references": info.get("references", {}),
                    })
                    return result

                # --- MC samples ---
                result.update({
                    "cross_section_pb": info.get("cross_section_pb"),
                    "cross_section_source": info.get("cross_section_source"),
                    "kfactor_value": info.get("kfactor", {}).get("value"),
                    "kfactor_source": info.get("kfactor", {}).get("source"),
                    "filter_efficiency": info.get("filter_efficiency"),
                    "references": info.get("references", []),
                    "notes": info.get("notes"),
                })
                return result

    print(f"[Warning] Sample '{sample_name}' not found in year '{year_key}'.")
    return None


# # Example usage:
# result = get_sample_info("./configs/datasets/dataset_nanoAODv12_run3.yaml", "dyTo2L_M-50_0j", "2022preEE")
# if result:
#     print(f"Year: {result['year']}")
#     print(f"Process: {result['process_group']}")
#     print(f"Sample: {result['sample']}")
#     print(f"Cross section: {result['cross_section_pb']} pb")
#     print(f"Source: {result['cross_section_source']}")
#     print(f"k-factor: {result['kfactor_value']} ({result['kfactor_source']})")
#     print(f"total lumi: {result['total_lumi_pb']} pb")

# print("\n-----------------------\n")
# # For Data:
# result = get_sample_info("./configs/datasets/dataset_nanoAODv12_run3.yaml", "data_C", "2022preEE")
# if result:
#     print(f"Year: {result['year']}")
#     print(f"Process: {result['process_group']}")
#     print(f"Sample: {result['sample']}")
#     print(f"Total lumi: {result['total_lumi_pb']} pb")
