import uproot
import numpy as np
import json
import glob
import os
from correctionlib import schemav2 as schema

# Directory with ROOT files
# input_dir = "plots_WS_DYMiNNLO_30Mar2025acoplanarity_custombin"
input_dir = "plots_WS_DYMiNNLO_30Mar2025acoplanarity"
output_json = "sf_data_correctionlib.json"

all_entries = []

def parse_range(index, parts):
    if "gt" in parts[index + 1]:
        return float(parts[index + 1].replace("gt", "")), float("inf")
    else:
        return float(parts[index + 1]), float(parts[index + 2])

# Loop over files
for filepath in glob.glob(f"{input_dir}/*.root"):
    filename = os.path.basename(filepath)
    parts = filename.replace(".root", "").split("_")

    pt1_index = parts.index("mu1")
    pt2_index = parts.index("mu2")
    njet = int(parts[pt1_index - 1][-1])

    pt1_low, pt1_high = parse_range(pt1_index, parts)
    pt2_low, pt2_high = parse_range(pt2_index, parts)

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

            for i in range(len(sf_values)):
                entry = {
                    "pt1": [pt1_low, pt1_high],
                    "pt2": [pt2_low, pt2_high],
                    "njet": njet,
                    "acop_low": float(bin_edges[i]),
                    "acop_high": float(bin_edges[i + 1]),
                    "sf": float(sf_values[i]),
                    "sf_err": float(sf_errors[i]),
                }
                all_entries.append(entry)

    except Exception as e:
        print(f"Error processing {filename}: {e}")


acop_bins = sorted(set((e["acop_low"], e["acop_high"]) for e in all_entries))
pt1_bins = sorted(set([e["pt1"][0] for e in all_entries] + [e["pt1"][1] for e in all_entries]))
if pt1_bins[-1] != float("inf"):
    pt1_bins.append(float("inf"))

pt2_bins = sorted(set([e["pt2"][0] for e in all_entries] + [e["pt2"][1] for e in all_entries]))
if pt2_bins[-1] != float("inf"):
    pt2_bins.append(float("inf"))

njet_bins = sorted(set(e["njet"] for e in all_entries))
njet_bins.append(100)  # Make [2, 100] to cover njet ≥ 2

def clean_inf(val):
    return 1e6 if (val == float("inf") or val == float("Infinity")) else val

# Clean acoplanarity bin edges
acop_bin_edges = sorted(set([b[0] for b in acop_bins] + [acop_bins[-1][1]]))
acop_bin_edges = [round(clean_inf(b), 6) for b in acop_bin_edges]

# Clean other axes
pt1_bins_clean = [clean_inf(x) for x in pt1_bins]
pt2_bins_clean = [clean_inf(x) for x in pt2_bins]
njet_bins_clean = [clean_inf(x) for x in njet_bins]


print("== Summary Check ==")
print("pt1 bins:", len(pt1_bins_clean) - 1)
print("pt2 bins:", len(pt2_bins_clean) - 1)
print("njet bins:", len(njet_bins_clean) - 1)
print("acop bins:", len(acop_bin_edges) - 1)
print("Expected total content size:",
      (len(pt1_bins_clean)-1) *
      (len(pt2_bins_clean)-1) *
      (len(njet_bins_clean)-1) *
      (len(acop_bin_edges)-1))


# len(row) == len(acop_bin_edges) - 1

# print(f"==> len(row): {len(row)}")
print(f"==> len(acop_bin_edges): {len(acop_bin_edges)}")
# Flatten content
content = []
for pt1_idx in range(len(pt1_bins_clean) - 1):
    for pt2_idx in range(len(pt2_bins_clean) - 1):
        for njet_idx in range(len(njet_bins_clean) - 1):
            njet = njet_bins_clean[njet_idx]
            row = []
            for i in range(len(acop_bin_edges) - 1):
                low = acop_bin_edges[i]
                high = acop_bin_edges[i + 1]
                match = next(
                    (e for e in all_entries
                     if e["pt1"] == [pt1_bins[pt1_idx], pt1_bins[pt1_idx + 1]]
                     and e["pt2"] == [pt2_bins[pt2_idx], pt2_bins[pt2_idx + 1]]
                     and e["njet"] == njet
                     and abs(e["acop_low"] - low) < 1e-6
                     and abs(e["acop_high"] - high) < 1e-6),
                    None
                )
                row.append(match["sf"] if match else 1.0)
            content.extend(row)

print("Actual content size:", len(content))

# Build correction
correction = schema.Correction(
    name="acoplanaritySF",
    description="Scale factor from acoplanarity-based bins",
    version=1,
    inputs=[
        schema.Variable(name="pt1", type="real"),
        schema.Variable(name="pt2", type="real"),
        schema.Variable(name="njet", type="int"),
        schema.Variable(name="acoplanarity", type="real"),
    ],
    output=schema.Variable(name="sf", type="real"),
    data=schema.MultiBinning(
        nodetype="multibinning",
        inputs=["pt1", "pt2", "njet", "acoplanarity"],
        edges=[pt1_bins_clean, pt2_bins_clean, njet_bins_clean, acop_bin_edges],
        content=content,
        flow="clamp"
    )
)


# Wrap in CorrectionSet
cset = schema.CorrectionSet(
    schema_version=2,
    corrections=[correction]
)

# Write JSON
with open(output_json, "w") as f:
    json.dump(cset.model_dump(), f, indent=2)

print(f"\nSaved correctionlib-compatible JSON using schemav2: {output_json}")

# cross-check value for 25, 25, 0, 0.5 using evaluate fo correctionlib
# print(correction.to_evaluator().evaluate(25., 25., 0, 0.999))
