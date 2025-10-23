import yaml
from modules.DistributionCompare import DistributionCompare

with open("config/plot_config_nanoV12vsV9.yaml") as f:
    config = yaml.safe_load(f)

year = config["year"]
directoryTag = config["directoryTag"]
input_paths_labels = config["input_paths_labels"]
fields_to_load = config["fields_to_load"]
control_region = config["control_region"]

muons = config["variables"]["muon"]
all_vars = muons

comparer = DistributionCompare(
    year = year,
    input_paths_labels = input_paths_labels,
    fields = fields_to_load,
    directoryTag = directoryTag,
    varlist_file = "config/varlist_nano.yaml"
)
comparer.load_data()

if config["plot_types"]["plot_1D"]:
    comparer.compare_all(all_vars)
