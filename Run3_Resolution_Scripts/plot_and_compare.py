import yaml
from modules.DistributionCompare import DistributionCompare

with open("config/plot_config.yaml") as f:
    config = yaml.safe_load(f)

year = config["year"]
control_region = config["control_region"]
directoryTag = config["directoryTag"]
input_paths_labels = config["input_paths_labels"]
fields_to_load = config["fields_to_load"]

lead_vars = config["variables"]["leading_muon"]
sublead_vars = config["variables"]["subleading_muon"]
dimuon_vars = config["variables"]["dimuon"]

all_vars = lead_vars + sublead_vars + dimuon_vars

comparer = DistributionCompare(year, input_paths_labels, fields_to_load, control_region, directoryTag, "config/varlist.yaml")
comparer.load_data()
comparer.add_new_variable()

if config["plot_types"]["plot_1D"]:
    comparer.compare_all(all_vars)

if config["plot_types"]["plot_2D"]:
    for i, var1 in enumerate(all_vars):
        for var2 in all_vars[i+1:]:
            comparer.compare_2D(var1, var2)

if config["plot_types"]["fit_z_peak"] and control_region in ["z-peak", "z_peak"]:
    comparer.fit_dimuonInvariantMass_DCBXBW(suffix="Inclusive")
    comparer.fit_dimuonInvariantMass_DCBXBW_Unbinned(suffix="Inclusive")

    # Plot in double-muon regions (eta1 ⊗ eta2)
    for region in ["BB", "BO", "BE", "OB", "OO", "OE", "EB", "EO", "EE"]:
        print(f"Double-muon region: {region}")
        filtered = {label: comparer.filter_eta(events, region) for label, events in comparer.events.items()}
        comparer.fit_dimuonInvariantMass_DCBXBW(events_dict=filtered, suffix=region)
        comparer.fit_dimuonInvariantMass_DCBXBW_Unbinned(events_dict=filtered, suffix=region)


if config["plot_types"]["fit_signal"] and control_region == "signal":
    # comparer.fit_dimuonInvariantMass_DCBXBW()
    comparer.fit_dimuonInvariantMass_DCB_Unbinned()

    # Plot in double-muon regions (eta1 ⊗ eta2)
    for region in ["BB", "BO", "BE", "OB", "OO", "OE", "EB", "EO", "EE"]:
        print(f"Double-muon region: {region}")
        filtered = {label: comparer.filter_eta(events, region) for label, events in comparer.events.items()}
        # comparer.fit_dimuonInvariantMass_DCBXBW(events_dict=filtered, suffix=region)
        comparer.fit_dimuonInvariantMass_DCB_Unbinned(events_dict=filtered, suffix=region)

# ----
for var in config.get("plot_1D", []):
    comparer.compare(var)

for var_pair in config.get("plot_2D", []):
    comparer.compare_2D(var_pair[0], var_pair[1])

# Plot in single-muon regions (eta1 or eta2)
single_regions = config.get("regions_plot", {}).get("leading_muon", {})
if single_regions.get("enable", False):
    for region in ["B", "O", "E"]:
        print(f"leading-muon region: {region}")
        filtered = {label: comparer.filter_eta1(events, region) for label, events in comparer.events.items()}
        comparer.compare_all(single_regions["variables"], events_dict=filtered, suffix=f"lead_{region}")

single_regions = config.get("regions_plot", {}).get("subleading_muon", {})
if single_regions.get("enable", False):
    for region in ["B", "O", "E"]:
        print(f"Subleading-muon region: {region}")
        filtered = {label: comparer.filter_eta2(events, region) for label, events in comparer.events.items()}
        comparer.compare_all(single_regions["variables"], events_dict=filtered, suffix=f"subl_{region}")

# Plot in double-muon regions (eta1 ⊗ eta2)
double_regions = config.get("regions_plot", {}).get("double_muon", {})
if double_regions.get("enable", False):
    for region in ["BB", "BO", "BE", "OB", "OO", "OE", "EB", "EO", "EE"]:
        print(f"Double-muon region: {region}")
        filtered = {label: comparer.filter_eta(events, region) for label, events in comparer.events.items()}
        comparer.compare_all(double_regions["variables"], events_dict=filtered, suffix=region)

        for var1, var2 in double_regions.get("plot_2D", []):
            comparer.compare_2D(var1, var2, events_dict=filtered, suffix=region)
