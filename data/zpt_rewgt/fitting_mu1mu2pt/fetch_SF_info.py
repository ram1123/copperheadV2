import uproot
import numpy as np
import json
import yaml
import re
import glob
import os

# Directory with ROOT files
input_dir = "plots_WS_DYMiNNLO_30Mar2025acoplanarity_custombin"
output_json = "sf_data.json"
output_yaml = "sf_data.yaml"

all_data = []

# Loop over files
for filepath in glob.glob(f"{input_dir}/*.root"):
    filename = os.path.basename(filepath)

    #fetch njet, pt1, pt2 from filename acoplanarity_2017_njet1_mu1_gt50_mu2_40_50.root
    test = filename.split("_")
    # print(test)
    # get position of 'mu1'
    pt1_index = test.index("mu1")
    pt2_index = test.index("mu2")
    njet_index = pt1_index - 1
    njet = test[njet_index][-1] # Final
    # print(f"njet: {njet}")

    # print(f"pt1_index: {test[pt1_index+1]}, pt2_index: {test[pt2_index+1]}")
    if "gt" in test[pt1_index + 1]:
        pt1_low = float(test[pt1_index + 1].split("gt")[1])
        pt1_high = 1e17
    else:
        pt1_low = float(test[pt1_index+1])
        pt1_high = float(test[pt1_index + 2])

    if "gt" in test[pt2_index + 1]:
        pt2_low = float(test[pt2_index + 1].split("gt")[1].replace(".root", ""))
        pt2_high = 1e17
    else:
        pt2_low = float(test[pt2_index+1])
        pt2_high = float(test[pt2_index + 2].replace(".root", ""))

    print(f"{filename:50}: {njet:1}, pt1_low: {pt1_low:4}, pt1_high: {pt1_high:5}, pt2_low: {pt2_low:4}, pt2_high: {pt2_high:5}")

    if pt1_high == 1e17 and pt2_high == 1e17:
        hist_key = f"hist_SF_mu1_gt{int(pt1_low)}_mu2_gt{int(pt2_low)}"
    elif pt1_high == 1e17:
        hist_key = f"hist_SF_mu1_gt{int(pt1_low)}_mu2_{int(pt2_low)}_{int(pt2_high)}"
    elif pt2_high == 1e17:
        hist_key = f"hist_SF_mu1_{int(pt1_low)}_{int(pt1_high)}_mu2_gt{int(pt2_low)}"
    else:
        hist_key = f"hist_SF_mu1_{int(pt1_low)}_{int(pt1_high)}_mu2_{int(pt2_low)}_{int(pt2_high)}"
    # print(f"hist_key: {hist_key}")

    try:
        with uproot.open(filepath) as file:
            if hist_key not in file:
                print(f"{hist_key} not found in {filename}")
                exit()

            hist = file[hist_key]
            bin_edges = hist.axes[0].edges()
            sf_values = hist.values()
            sf_errors = hist.errors()

            # print(f"bin_edges: {bin_edges}")
            # print(f"sf_values: {sf_values}")

            for i in range(len(sf_values)):
                entry = {
                    "njet": int(njet),
                    "pt1_low": pt1_low,
                    "pt1_high": pt1_high,
                    "pt2_low": pt2_low,
                    "pt2_high": pt2_high,
                    "bin_low": float(bin_edges[i]),
                    "bin_high": float(bin_edges[i + 1]),
                    "sf": float(sf_values[i]),
                    "sf_err": float(sf_errors[i]),
                }
                all_data.append(entry)
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Save to JSON
with open(output_json, "w") as f_json:
    json.dump(all_data, f_json, indent=2)

# Save to YAML
with open(output_yaml, "w") as f_yaml:
    yaml.dump(all_data, f_yaml, sort_keys=False)

print(f"\nFinished. Files saved: {output_json}, {output_yaml}")
