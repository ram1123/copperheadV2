import yaml
import os

from modules.DistributionCompare import DistributionCompare

config_path = "config/plot_config_nanoV12vsV9.yaml"
varlist_path = "config/varlist_Stage1Parquet.yaml"

base = os.path.dirname(os.path.abspath(__file__))
config_full_path = os.path.join(base, config_path)

with open(config_full_path, "r") as f:
    config = yaml.safe_load(f)

year = config["year"]
directoryTag = config["directoryTag"]
input_paths_labels = config["input_parquet_paths_labels"]
fields_to_load = config["fields_to_load"]
control_region = config["control_region"]


comparer = DistributionCompare(
    year = year,
    input_paths_labels = input_paths_labels,
    fields = fields_to_load,
    directoryTag = directoryTag,
    varlist = f"{base}/{varlist_path}",
    control_region = control_region,
)
comparer.load_data()

# Get all vars from varlist
all_vars = comparer.all_vars()
if 'default' in all_vars:
    all_vars.remove('default')

print(f"Comparing variables: {all_vars}")
if config["plot_types"]["plot_1D"]:
    comparer.compare_all(all_vars)
