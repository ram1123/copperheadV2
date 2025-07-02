#!/usr/bin/env python3
"""
Generate Data/MC comparison plots for the VBF H-sidebands region using Dask and Parquet input.

Requirements:
  - dask[dataframe]
  - distributed
  - numpy
  - matplotlib
  - pyarrow

Adjust LOAD_PATH and group_dict as needed for your directory structure.
"""

from dask.distributed import Client
import dask.dataframe as dd
import dask.array as da
import numpy as np
import matplotlib.pyplot as plt

# if args.use_gateway:
from dask_gateway import Gateway
gateway = Gateway(
    "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
    proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
)
cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
client = gateway.connect(cluster_info.name).get_client()
# else: # use local cluster
# client = Client(n_workers=15,  threads_per_worker=1, processes=True, memory_limit='30 GiB')
# logger.info("Local scale Client created")

# --- Configuration ---
years = ["2016", "2017", "2018"]
LOAD_PATH = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/" \
            "stage1_output/{year}/f1_0/{sample}/*/*.parquet"

bkg_MC_order = ["OTHER", "EWK", "VV", "TOP", "DYVBF", "GGH", "VBF"]
group_dict = {
    "DATA": ["data_A", "data_B", "data_C", "data_D", "data_E", "data_F", "data_G", "data_H"],
    "DYVBF": ["dy_VBF_filter_NewZWgt"],
    "TOP": ["ttjets_dl", "ttjets_sl", "st_tw_top", "st_tw_antitop", "st_t_top", "st_t_antitop"],
    "EWK": ["ewk_lljj_mll50_mjj120"],
    "VV": ["ww_2l2nu", "wz_3lnu", "wz_2l2q", "wz_1l1nu2q", "zz"],
    "OTHER": ["www", "wwz", "wzz", "zzz"],
    "GGH": ["ggh_powhegPS"],
    "VBF": ["vbf_powheg_dipole"]
}

# Define sideband histogram binning
mass_bins = np.linspace(110, 150, 81)  # 0.5 GeV bins

def apply_vbf_h_sidebands(df):
    """Apply VBF category and H-sidebands region cuts to a Dask DataFrame."""
    m = df["dimuon_mass"]
    # H-sidebands: 110–115.03 or 135.03–150
    sideband = ((m > 110) & (m < 115.03)) | ((m > 135.03) & (m < 150))
    # VBF tag: mjj > 400 GeV, |∆η_jj| > 2.5, leading jet pT > 35 GeV
    vbf = (df["jj_mass_nominal"] > 400) & \
          (df["jj_dEta_nominal"] > 2.5) & \
          (df["jet1_pt_nominal"] > 35)
    return df[sideband & vbf]

# Read and histogram data
data_paths = [LOAD_PATH.format(year=yr, sample=s) for yr in years for s in group_dict["DATA"]]
df_data = dd.read_parquet(data_paths)
df_data = apply_vbf_h_sidebands(df_data)
arr_data = df_data["dimuon_mass"].to_dask_array(lengths=True)
hist_data = da.histogram(arr_data, bins=mass_bins)[0].compute()


# Read and histogram each MC group
hist_mc = {}
for grp in bkg_MC_order:
    print(f"Processing MC group: {grp}")
    samples = group_dict.get(grp, [])
    if not samples:
        continue
    paths = [LOAD_PATH.format(year=yr, sample=s) for yr in years for s in samples]
    df_mc = dd.read_parquet(paths)
    df_mc = apply_vbf_h_sidebands(df_mc)
    arr_mc = df_mc["dimuon_mass"].to_dask_array(lengths=True)
    hist_mc[grp] = da.histogram(arr_mc, bins=mass_bins)[0].compute()
    # hist_mc[grp] = df_mc["dimuon_mass"] \
    #     .to_dask_array(lengths=True) \
    #     .map_blocks(np.histogram, bins=mass_bins, drop_axis=1) \
    #     .sum(axis=0) \
    #     .compute()

# Plotting
plt.figure(figsize=(10,7))
bin_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])
plt.step(bin_centers, hist_data, where="mid", label="Data", color="black")

# Stack MC histograms
bottom = np.zeros_like(bin_centers)
colors = plt.cm.tab20.colors
for i, grp in enumerate(bkg_MC_order):
    print(f"Plotting MC group: {grp}")
    vals = hist_mc.get(grp, np.zeros_like(bin_centers))
    plt.bar(bin_centers, vals, bottom=bottom, width=mass_bins[1]-mass_bins[0],
            label=grp, color=colors[i % len(colors)], alpha=0.7)
    bottom += vals

plt.xlabel("Dimuon mass [GeV]")
plt.ylabel("Events / bin")
plt.title("VBF H-sidebands: Data vs MC")
plt.yscale("log")
plt.legend(fontsize="small", ncol=2)
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
# plt.show()
plt.savefig("vbf_h_sidebands_comparison.png", dpi=300)

# Close the Dask client
client.close()
