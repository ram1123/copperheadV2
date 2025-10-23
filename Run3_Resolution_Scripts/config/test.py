import yaml
with open("plot_config_nanoV12vsV9.yaml", "r") as f:
    config = yaml.safe_load(f)

print(config["variables"]["muon"])
print(config["variables"]["muon"][0])
print(config["variables"]["muon"][1])
print(config["variables"]["muon"][2])

print(config["variables"]["muon"][0].keys())
