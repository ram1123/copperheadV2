import uproot
import numpy as np
import json
import glob
import os

# Directory with ROOT files
input_dir = "plots_WS_DYMiNNLO_30Mar2025acoplanarity_custombin"
output_json = "sf_data_flat.json"

sf_dict = {}

# Loop over files
for filepath in glob.glob(f"{input_dir}/*.root"):
    filename = os.path.basename(filepath)
    parts = filename.replace(".root", "").split("_")

    pt1_index = parts.index("mu1")
    pt2_index = parts.index("mu2")
    njet = int(parts[pt1_index - 1][-1])

    def parse_range(index):
        if "gt" in parts[index + 1]:
            return float(parts[index + 1].replace("gt", "")), float("inf")
        else:
            return float(parts[index + 1]), float(parts[index + 2])

    pt1_low, pt1_high = parse_range(pt1_index)
    pt2_low, pt2_high = parse_range(pt2_index)

    if pt1_high == float("inf") and pt2_high == float("inf"):
        hist_key = f"hist_SF_mu1_gt{int(pt1_low)}_mu2_gt{int(pt2_low)}"
    elif pt1_high == float("inf"):
        hist_key = f"hist_SF_mu1_gt{int(pt1_low)}_mu2_{int(pt2_low)}_{int(pt2_high)}"
    elif pt2_high == float("inf"):
        hist_key = f"hist_SF_mu1_{int(pt1_low)}_{int(pt1_high)}_mu2_gt{int(pt2_low)}"
    else:
        hist_key = f"hist_SF_mu1_{int(pt1_low)}_{int(pt1_high)}_mu2_{int(pt2_low)}_{int(pt2_high)}"

    try:
        with uproot.open(filepath) as file:
            if hist_key not in file:
                print(f"{hist_key} not found in {filename}")
                exit()

            hist = file[hist_key]
            bin_edges = hist.axes[0].edges()
            sf_values = hist.values()
            sf_errors = hist.errors()

            acop_bins = [
                [float(bin_edges[i]), float(bin_edges[i + 1]), float(sf_values[i]), float(sf_errors[i])]
                for i in range(len(sf_values))
            ]

            # Create stringified key (JSON-safe)
            key = f"({pt1_low},{pt1_high})-({pt2_low},{pt2_high})-{njet}"
            sf_dict[key] = acop_bins

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Save to JSON
with open(output_json, "w") as f_json:
    json.dump(sf_dict, f_json, indent=2)

print(f"\nFlattened SF dictionary saved as: {output_json}")
