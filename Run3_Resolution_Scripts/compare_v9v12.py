import yaml
import os
import sys

from modules.DistributionCompare import DistributionCompare

config_path = "config/plot_config_nanoV12vsV9.yaml"

base = os.path.dirname(os.path.abspath(__file__))
config_full_path = os.path.join(base, config_path)

with open(config_full_path, "r") as f:
    config = yaml.safe_load(f)

year = config["year"]
directoryTag = config["directoryTag"]
input_paths_labels = config["input_paths_labels"]
fields_to_load = config["fields_to_load"]
control_region = config["control_region"]

muons = config["variables"]["muon"]
all_vars = muons[0]

print(f"variables to compare: {all_vars}")
print(f"variables to compare: {type(all_vars)}")

comparer = DistributionCompare(
    year = year,
    input_paths_labels = input_paths_labels,
    fields = fields_to_load,
    directoryTag = directoryTag,
    varlist = all_vars
)
comparer.load_data()

if config["plot_types"]["plot_1D"]:
    comparer.compare_all(all_vars)
