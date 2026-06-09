import json
import os


from modules.utils import logger


def write_cutflow_outputs(cutflow, save_path, dataset_name, file_idx):
    base_name = f"cutflow_{dataset_name}_{file_idx}"
    npz_path = os.path.join(save_path, f"{base_name}.npz")
    json_path = os.path.join(save_path, f"{base_name}.json")

    npz_result = cutflow.to_npz(npz_path)
    if hasattr(npz_result, "compute"):
        npz_result.compute()
    logger.info(f"NPZ saved: {npz_path}")

    try:
        combined_data = {}
        for i, name in enumerate(cutflow._names):
            cumulative = cutflow._nevcutflow[i]
            individual = cutflow._nevonecut[i]
            combined_data[name] = {
                "cumulative": cumulative.item() if hasattr(cumulative, "item") else cumulative,
                "individual": individual.item() if hasattr(individual, "item") else individual,
            }

        with open(json_path, "w") as handle:
            json.dump(combined_data, handle, indent=4)

        logger.info(f"JSON saved to {json_path}")
    except Exception as err:
        logger.error(f"JSON save failed: {err}")
