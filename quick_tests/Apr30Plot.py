import dask_awkward as dak
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import dask

import os
import sys

from modules.selection import filterRegion

from pathlib import Path

def get_matching_directories(pattern):
    """Returns a unique, sorted list of directories matching the parquet pattern."""
    # We use Path("/") as the anchor to handle the absolute path
    # glob() handles the wildcards like 'data_*' and '0'
    unique_dirs = {p.parent for p in Path("/").glob(pattern.lstrip("/"))}
    
    return sorted([str(d) for d in unique_dirs])

# ------------------------------------------------------------
# Main reusable plotting function
# ------------------------------------------------------------

def plot_variable_from_parquets(
    variable,
    bin_edges,
    parquet_paths,
    weight_variable="wgt_nominal",
    normalize=False,
    region="h-peak",
    xlabel=None,
    ylabel=None,
    title=None,
    output_name=None,
):
    """
    Plot a weighted histogram for one variable from multiple flat parquet samples.

    Parameters
    ----------
    variable : str
        Name of the variable to load from parquet, e.g. "jet1_eta_nominal".

    bin_edges : array-like
        Histogram bin edges.

    parquet_paths : dict
        Dictionary of sample labels and parquet paths, e.g.
        {
            "Run2 NanoV12": "/path/to/run2_nanov12/*.parquet",
            "Run2 NanoV15": "/path/to/run2_nanov15/*.parquet",
            "Run3": "/path/to/run3/*.parquet",
        }

    weight_variable : str
        Name of the weight variable. Default is "wgt_nominal".

    normalize : bool
        If True, normalize each histogram so that np.sum(hist_values) == 1.

    xlabel, ylabel, title, output_name : str
        Optional plot labels and output filename.
    """

    bin_edges = np.asarray(bin_edges)

    hists = {}

    for label, parquet_path in parquet_paths.items():
        print(f"Processing {label}: {parquet_path}")
        print(f"Reading directories for {label} from: {get_matching_directories(parquet_path)}")

        events = dak.from_parquet(parquet_path)
        _, events = filterRegion(events, region=region)
        # events = events[:10_000] # take the first 10k events
        values = events[variable]
        weights = events[weight_variable]

        values_np, weights_np = dask.compute(values, weights)

        values_np = ak.to_numpy(values_np)
        weights_np = ak.to_numpy(weights_np)

        mask = np.isfinite(values_np) & np.isfinite(weights_np)
        values_np = values_np[mask]
        weights_np = weights_np[mask]

        if normalize:
            weights_np = weights_np/np.sum(weights_np)
        # print(f"np.sum(weights_np): {np.sum(weights_np)}")
        hist_values, _ = np.histogram(
            values_np,
            bins=bin_edges,
            weights=weights_np,
        )
        hist_values_w2, _ = np.histogram(
            values_np,
            bins=bin_edges,
            weights=weights_np*weights_np,
        )

        

        hists[label] = [hist_values, hist_values_w2]

    hep.style.use("CMS")

    plt.figure(figsize=(8, 6))

    for label, hist_values_l in hists.items():
        hist_values, hist_values_w2 = hist_values_l
        hep.histplot(
            hist_values,
            bins=bin_edges,
            label=label,
            histtype="step",
            # histtype="errorbar",
            yerr=np.sqrt(hist_values_w2),
            linewidth=2,
        )
        # print(f"{label} hist_values: {hist_values.sum()}")
        
    plt.xlabel(xlabel if xlabel is not None else variable)

    if ylabel is not None:
        plt.ylabel(ylabel)
    elif normalize:
        plt.ylabel("Normalized events")
    else:
        plt.ylabel("Weighted events")

    if title is not None:
        plt.title(title)

    plt.legend(fontsize=10)
    plt.tight_layout()

    if output_name is not None:
        # Extract directory path
        directory = os.path.dirname(output_name)
        
        # Create directory if it doesn't exist
        if directory:
            os.makedirs(directory, exist_ok=True)
        plt.savefig(output_name, dpi=200)

    plt.show()

    return hists, bin_edges




# ------------------------------------------------------------
# Sample replacement rules
# ------------------------------------------------------------

sample_pattern_dict = {
    "data": "data_*",
    "DY": "dy*",
    "VBF": "vbf_powheg*",
    "ggH": "ggh_powhegPS",
}


def replace_compacted_sample(path, sample_pattern):
    """
    Replace the sample folder after /compacted/.

    Example:
        .../compacted/data_*/0/*.parquet
    becomes:
        .../compacted/dy*/0/*.parquet
        .../compacted/vbf_powheg*/0/*.parquet
        .../compacted/ggh_powhegPS/0/*.parquet

    Also works for paths like:
        .../compacted/data_C/0/*.parquet
    """

    before, after = path.split("/compacted/", 1)
    sample_folder, rest = after.split("/", 1)

    return f"{before}/compacted/{sample_pattern}/{rest}"


def make_paths_for_sample(base_parquet_paths, sample):
    """
    Convert data parquet paths into data, DY, VBF, or ggH paths.
    """

    sample_pattern = sample_pattern_dict[sample]

    return {
        label: replace_compacted_sample(path, sample_pattern)
        for label, path in base_parquet_paths.items()
    }


def run_plots_for_block(
    desc,
    base_parquet_paths,
    variables,
    bin_edges,
    region,
    normalize=True,
):
    """
    For a given comparison block, make separate plots for:
        data, DY, VBF, ggH

    Output format:
        {region}/{desc}/{sample}/{var}_normalized.pdf

    Notes
    -----
    plot_variable_from_parquets() is intentionally unchanged.

    The bin_edges argument can be either:
        1. A single array-like object used for every sample.
        2. A dictionary with sample-specific bin edges, e.g.
           {
               "data": np.linspace(-4.7, 4.7, 101),
               "DY":   np.linspace(-4.7, 4.7, 101),
               "VBF":  np.linspace(-4.7, 4.7, 61),
               "ggH":  np.linspace(-4.7, 4.7, 61),
           }

    In np.linspace(start, stop, num=N), N is the number of bin edges,
    so the number of bins is N - 1.
    """

    for sample in sample_pattern_dict:
        parquet_paths = make_paths_for_sample(
            base_parquet_paths=base_parquet_paths,
            sample=sample,
        )

        # Pick the binning for this sample. This keeps
        # plot_variable_from_parquets() unchanged, because each call still
        # receives a single bin_edges array.
        if isinstance(bin_edges, dict):
            if sample not in bin_edges:
                raise KeyError(
                    f"Missing bin_edges entry for sample '{sample}'. "
                    f"Available keys: {list(bin_edges.keys())}"
                )
            sample_bin_edges = bin_edges[sample]
        else:
            sample_bin_edges = bin_edges

        for var in variables:
            var_title = var_title_dict[var]

            output_name = f"{region}/{desc}/{sample}/{var}_normalized.pdf"

            hists, bins = plot_variable_from_parquets(
                variable=var,
                bin_edges=sample_bin_edges,
                parquet_paths=parquet_paths,
                normalize=normalize,
                region=region,
                xlabel=var_title,
                # title=f"{var_title} distribution",
                output_name=output_name,
            )

# -------------------------------------------------------------
# run23Comp
# -------------------------------------------------------------

run23Comp_paths = {
    "Run2(2017) CHS": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl/stage1_output/2017/compacted/data_*/0/*.parquet",
    "Run2(2017) PUPPI": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_Apr14_2026_UpdatedBtagWp/stage1_output/2017/compacted/data_*/0/*.parquet",
    "Run3(2024) PUPPI": "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterJetsHorn25GeV_Apr09_tightPassLepVeto_NoJER_v2/stage1_output/2024/compacted/data_C/0/*.parquet",
}

# -------------------------------------------------------------
# run23Comp extended
# -------------------------------------------------------------

run23CompExt_paths = {
    "Run2(2017) CHS": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl/stage1_output/2017/compacted/data_*/0/*.parquet",
    "Run2(2017) PUPPI": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_Apr14_2026_UpdatedBtagWp/stage1_output/2017/compacted/data_*/0/*.parquet",
    "Run3(2022postEE) PUPPI": "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn25GeV_Apr09_tightPassLepVeto_NoJER_v2/stage1_output/2022postEE/compacted/data_*/0/*.parquet",
    "Run3(2024) PUPPI": "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterJetsHorn25GeV_Apr09_tightPassLepVeto_NoJER_v2/stage1_output/2024/compacted/data_C/0/*.parquet",
}

# -------------------------------------------------------------
# run2NanoV12
# -------------------------------------------------------------

run2NanoV12_paths = {
    "Run2(2018) CHS": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl/stage1_output/2018/compacted/data_*/0/*.parquet",
    "Run2(2017) CHS": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl/stage1_output/2017/compacted/data_*/0/*.parquet",
    "Run2(2016postVFP) CHS": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl/stage1_output/2016postVFP/compacted/data_*/0/*.parquet",
    "Run2(2016preVFP) CHS": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl/stage1_output/2016preVFP/compacted/data_*/0/*.parquet",
}


# -------------------------------------------------------------
# run2NanoV15
# -------------------------------------------------------------

run2NanoV15_paths = {
    "Run2(2018) PUPPI": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_Apr14_2026_UpdatedBtagWp/stage1_output/2018/compacted/data_*/0/*.parquet",
    "Run2(2017) PUPPI": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_Apr14_2026_UpdatedBtagWp/stage1_output/2017/compacted/data_*/0/*.parquet",
    "Run2(2016postVFP) PUPPI": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_Apr14_2026_UpdatedBtagWp/stage1_output/2016postVFP/compacted/data_*/0/*.parquet",
    "Run2(2016preVFP) PUPPI": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_Apr14_2026_UpdatedBtagWp/stage1_output/2016preVFP/compacted/data_*/0/*.parquet",
}


# -------------------------------------------------------------
# run3
# -------------------------------------------------------------

run3_paths = {
    "Run3(2024) PUPPI": "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterJetsHorn25GeV_Apr09_tightPassLepVeto_NoJER_v2/stage1_output/2024/compacted/data_*/0/*.parquet",
    "Run3(2023BPix) PUPPI": "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn25GeV_Apr09_tightPassLepVeto_NoJER_v2/stage1_output/2023BPix/compacted/data_*/0/*.parquet",
    "Run3(2023) PUPPI": "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn25GeV_Apr09_tightPassLepVeto_NoJER_v2/stage1_output/2023/compacted/data_*/0/*.parquet",
    "Run3(2022postEE) PUPPI": "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn25GeV_Apr09_tightPassLepVeto_NoJER_v2/stage1_output/2022postEE/compacted/data_*/0/*.parquet",
    "Run3(2022preEE) PUPPI": "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn25GeV_Apr09_tightPassLepVeto_NoJER_v2/stage1_output/2022preEE/compacted/data_*/0/*.parquet",
}

if __name__ == "__main__":
    # ------------------------------------------------------------
    # Start dask client
    # ------------------------------------------------------------
    from distributed import Client
    client =  Client(n_workers=30,  threads_per_worker=1, processes=True, memory_limit='30 GiB') 

    # ------------------------------------------------------------
    # Refactored usage section
    # Keep plot_variable_from_parquets() unchanged above this point
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Eta plots
    # ------------------------------------------------------------
    # You can define different binning for each physics sample.
    # Note: np.linspace(start, stop, num=N) gives N bin edges = N - 1 bins.
    eta_bin_edges_by_sample = {
        "data": np.linspace(-4.7, 4.7, num=101),  # 100 bins
        "DY":   np.linspace(-4.7, 4.7, num=101),  # 100 bins
        "VBF":  np.linspace(-4.7, 4.7, num=61),   # 60 bins
        "ggH":  np.linspace(-4.7, 4.7, num=61),   # 60 bins
        # "data": np.linspace(-4.7, 4.7, num=100),  # 100 bins
        # "DY":   np.linspace(-4.7, 4.7, num=100),  # 100 bins
        # "VBF":  np.linspace(-4.7, 4.7, num=100),   # 60 bins
        # "ggH":  np.linspace(-4.7, 4.7, num=100),   # 60 bins
    }

    region = "h-sidebands"
    # region = "z-peak"

    variables = ["jet1_eta_nominal", "jet2_eta_nominal"]

    var_title_dict = {
        "jet1_eta_nominal": r"Leading jet $\eta$",
        "jet2_eta_nominal": r"Sub-leading jet $\eta$",
    }

    # Use all samples for the eta plots.
    sample_pattern_dict = {
        "data": "data_*",
        "DY": "dy*",
        "VBF": "vbf_powheg*",
        "ggH": "ggh_powhegPS",
    }

    # -------------------------------------------------------------
    # Run all eta plotting blocks
    # -------------------------------------------------------------

    plot_blocks = {
        "run23Comp": run23Comp_paths,
        "run23Comp_Ext": run23CompExt_paths,
        "run2NanoV12": run2NanoV12_paths,
        "run2NanoV15": run2NanoV15_paths,
        "run3": run3_paths,
    }


    for desc, base_parquet_paths in plot_blocks.items():
        run_plots_for_block(
            desc=desc,
            base_parquet_paths=base_parquet_paths,
            variables=variables,
            bin_edges=eta_bin_edges_by_sample,
            region=region,
            normalize=True,
        )

    # ------------------------------------------------------------
    # Repeat for dimuon mass plot
    # ------------------------------------------------------------

    mass_bin_edges_by_sample = {
        "data": np.linspace(70, 110, num=201),  # 100 bins
        "DY":   np.linspace(70, 110, num=201),   # 80 bins
        # "data": np.linspace(70, 110, num=100),  # 100 bins
        # "DY":   np.linspace(70, 110, num=100),   # 80 bins
    }
    region="z-peak"
    variables=["dimuon_mass"]
    var_title_dict = {
        "dimuon_mass" : r"$m_{\mu\mu}$",
    }

    sample_pattern_dict = { # only plot DY and data
        "data": "data_*",
        "DY": "dy*",
    }


    plot_blocks = {
        "run23Comp": run23Comp_paths,
        "run23Comp_Ext": run23CompExt_paths,
        "run2NanoV12": run2NanoV12_paths,
        "run2NanoV15": run2NanoV15_paths,
        "run3": run3_paths,
    }


    for desc, base_parquet_paths in plot_blocks.items():
        run_plots_for_block(
            desc=desc,
            base_parquet_paths=base_parquet_paths,
            variables=variables,
            bin_edges=mass_bin_edges_by_sample,
            region=region,
            normalize=True,
        )