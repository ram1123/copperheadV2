import json
import os

from modules.utils import logger


def write_cutflow_outputs(cutflow, selection_names, save_path, dataset_name, file_idx):
    """
    Write a per-chunk cutflow shard to JSON (+ the full arrays to .npz).

    Uses coffea.analysis_tools.Cutflow's public `.result()` API rather than
    its private `_names`/`_nevcutflow`/`_nevonecut` attributes -- more robust
    against internal coffea changes, and it's also how weighted yields become
    available: if `cutflow` was built with `weights=...` (see
    src/copperhead_processor.py's cutflow block), `.result()` additionally
    carries `wgtevonecut`/`wgtevcutflow`, which get included as
    "*_weighted" JSON fields automatically.

    `selection_names` must be the exact, in-order list of names passed to
    `PackedSelection.cutflow(*selection_names, ...)` -- `.result()` returns
    the per-cut arrays but not the names themselves, so the caller (which
    already has that list) has to supply it. See
    https://coffea-hep.readthedocs.io/en/v2026.5.0/api/coffea.analysis_tools.Cutflow.html
    """
    base_name = f"cutflow_{dataset_name}_{file_idx}"
    npz_path = os.path.join(save_path, f"{base_name}.npz")
    json_path = os.path.join(save_path, f"{base_name}.json")

    npz_result = cutflow.to_npz(npz_path)
    if hasattr(npz_result, "compute"):
        npz_result.compute()
    logger.info(f"NPZ saved: {npz_path}")

    def _scalar(x):
        return x.item() if hasattr(x, "item") else x

    try:
        result = cutflow.result()
        has_weighted = hasattr(result, "wgtevcutflow")

        # nevcutflow/nevonecut (and the weighted wgtev* equivalents) have
        # len(selection_names)+1 elements: index 0 is the count before any
        # selection is applied (redundant with the "TotalEntries" row, which
        # is always index 0 in selection_names too -- see
        # copperhead_processor.py's cutflow block), index i+1 is after
        # selection_names[i]. Only the named rows are written here, matching
        # the previous output shape.
        combined_data = {}
        for i, name in enumerate(selection_names):
            entry = {
                "cumulative": _scalar(result.nevcutflow[i + 1]),
                "individual": _scalar(result.nevonecut[i + 1]),
            }
            if has_weighted:
                entry["cumulative_weighted"] = _scalar(result.wgtevcutflow[i + 1])
                entry["individual_weighted"] = _scalar(result.wgtevonecut[i + 1])
            combined_data[name] = entry

        with open(json_path, "w") as handle:
            json.dump(combined_data, handle, indent=4)

        logger.info(f"JSON saved to {json_path}")
    except Exception as err:
        logger.error(f"JSON save failed: {err}")
