import json
import os
import uuid

import awkward as ak
import numpy as np

from modules.utils import logger
from src.copperhead_processor import EventProcessor


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


class RunnerStage1Processor(EventProcessor):
    """
    Thin runner-friendly wrapper around the analysis processor.

    Each runner chunk writes its parquet shard directly to disk and returns only
    tiny metadata so the runner can merge results efficiently.
    """

    def __init__(self, config, save_path, dataset_yaml_file, test_mode=False, isCutflow=False):
        super().__init__(config, test_mode=test_mode, isCutflow=isCutflow)
        self.save_path = save_path
        self.dataset_yaml_file = dataset_yaml_file

    def process(self, events):
        out_collections, processed_event_count = super().process(
            events,
            dataset_yaml_file=self.dataset_yaml_file,
        )

        out_collections["fraction"] = (
            events.metadata["fraction"] * ak.ones_like(out_collections["event"])
        )
        skim_zip = ak.zip(out_collections, depth_limit=1)

        parquet_name = f"part_{uuid.uuid4().hex}.parquet"
        parquet_path = os.path.join(self.save_path, parquet_name)
        ak.to_parquet(skim_zip, parquet_path)

        result = {
            "processed_event_count": int(np.asarray(processed_event_count).item()),
            "written_files": 1,
        }
        if self.isCutflow:
            result["cutflow"] = self.cutflow
        return {events.metadata["dataset"]: result}
