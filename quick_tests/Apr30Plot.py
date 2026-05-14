import dask_awkward as dak
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import dask
import pandas as pd

import os
import sys
from pathlib import Path

# from modules.selection import filterRegion
def filterRegion(events, region="h-peak"):
    if isinstance(events, pd.DataFrame):
        fields = events.columns
    else: # awkward zip
        fields = events.fields  
    if "dimuon_mass" not in fields:
        raise ValueError("dimuon_mass not found in events fields for region selection.")
    dimuon_mass = events["dimuon_mass"]
    z_peak = (dimuon_mass >= 70.0) & (dimuon_mass < 110.0)
    h_peak = (dimuon_mass >= 115.0) & (dimuon_mass < 135.0)
    h_sidebands = ((dimuon_mass >= 110.0) & (dimuon_mass < 115.0)) | (
        (dimuon_mass >= 135.0) & (dimuon_mass < 150.0)
    )
    if region == "z-peak":
        mask = z_peak
    elif region == "h-peak":
        mask = h_peak
    elif region == "h-sidebands":
        mask = h_sidebands
    elif region == "signal":
        mask = h_sidebands | h_peak
    elif region == "full":
        mask = z_peak | h_sidebands | h_peak
    else:
        raise ValueError(
            f"Invalid region selection: {region}. Valid options are: 'z-peak', 'h-peak', 'h-sidebands', 'signal', 'full'."
        )

    return mask, events[mask]


# Define the calibration categories ---
def get_calib_categories(events):
    """
    Returns a dictionary of 30 boolean masks based on muon1_eta, muon2_eta and mu1_pt.
    Eta bins:
      B: |eta| <= 0.9
      O: 0.9 < |eta| <= 1.8
      E: 1.8 < |eta| <= 2.4
    pT bins for mu1_pt:
      Bin1: (30, 45]
      Bin2: (45, 52]
      Bin3: (52, 62]
      Bin4: (62, 200]

    For the lowest pT bin, the eta combinations are merged into three groups.
    For the other bins, each of the nine eta combinations is kept separately.
    Referece: HIG-19-006, AN-19-124
    """
    BB = ((np.abs(events["mu1_eta"])<=0.9) & (np.abs(events["mu2_eta"])<=0.9))
    BO = ((np.abs(events["mu1_eta"])<=0.9) & ((np.abs(events["mu2_eta"])>0.9) & (np.abs(events["mu2_eta"]) <=1.8)))
    BE = ((np.abs(events["mu1_eta"])<=0.9) & ((np.abs(events["mu2_eta"])>1.8) & (np.abs(events["mu2_eta"]) <=2.4)))
    OB = (((np.abs(events["mu1_eta"])>0.9) & (np.abs(events["mu1_eta"])<=1.8)) & (np.abs(events["mu2_eta"])<=0.9))
    OO = (((np.abs(events["mu1_eta"])>0.9) & (np.abs(events["mu1_eta"])<=1.8)) & ((np.abs(events["mu2_eta"])>0.9) & (np.abs(events["mu2_eta"])<=1.8)))
    OE = (((np.abs(events["mu1_eta"])>0.9) & (np.abs(events["mu1_eta"])<=1.8)) & ((np.abs(events["mu2_eta"])>1.8) & (np.abs(events["mu2_eta"])<=2.4)))
    EB = (((np.abs(events["mu1_eta"])>1.8) & (np.abs(events["mu1_eta"])<=2.4)) & (np.abs(events["mu2_eta"])<=0.9))
    EO = (((np.abs(events["mu1_eta"])>1.8) & (np.abs(events["mu1_eta"])<=2.4)) & ((np.abs(events["mu2_eta"])>0.9) & (np.abs(events["mu2_eta"])<=1.8)))
    EE = (((np.abs(events["mu1_eta"])>1.8) & (np.abs(events["mu1_eta"])<=2.4)) & ((np.abs(events["mu2_eta"])>1.8) & (np.abs(events["mu2_eta"])<=2.4)))

    # pT bins for mu1_pt
    mask_30_45 = (events["mu1_pt"] > 30) & (events["mu1_pt"] <= 45)
    mask_45_52 = (events["mu1_pt"] > 45) & (events["mu1_pt"] <= 52)
    mask_52_62 = (events["mu1_pt"] > 52) & (events["mu1_pt"] <= 62)
    mask_62_200 = (events["mu1_pt"] > 62) & (events["mu1_pt"] <= 200)

    # For pT bin 30-45, group the eta combinations into three categories.
    # cat_30_45_1 = mask_30_45 & (BB | OB | EB)
    # cat_30_45_2 = mask_30_45 & (BO | OO | EO)
    # cat_30_45_3 = mask_30_45 & (BE | OE | EE)
    cats_30_45 = {
        "30-45_BB": mask_30_45 & BB,
        "30-45_BO": mask_30_45 & BO,
        "30-45_BE": mask_30_45 & BE,
        "30-45_OB": mask_30_45 & OB,
        "30-45_OO": mask_30_45 & OO,
        "30-45_OE": mask_30_45 & OE,
        "30-45_EB": mask_30_45 & EB,
        "30-45_EO": mask_30_45 & EO,
        "30-45_EE": mask_30_45 & EE,
    }

    # For the remaining bins, each eta combination is its own category.
    cats_45_52 = {
        "45-52_BB": mask_45_52 & BB,
        "45-52_BO": mask_45_52 & BO,
        "45-52_BE": mask_45_52 & BE,
        "45-52_OB": mask_45_52 & OB,
        "45-52_OO": mask_45_52 & OO,
        "45-52_OE": mask_45_52 & OE,
        "45-52_EB": mask_45_52 & EB,
        "45-52_EO": mask_45_52 & EO,
        "45-52_EE": mask_45_52 & EE,
    }

    cats_52_62 = {
        "52-62_BB": mask_52_62 & BB,
        "52-62_BO": mask_52_62 & BO,
        "52-62_BE": mask_52_62 & BE,
        "52-62_OB": mask_52_62 & OB,
        "52-62_OO": mask_52_62 & OO,
        "52-62_OE": mask_52_62 & OE,
        "52-62_EB": mask_52_62 & EB,
        "52-62_EO": mask_52_62 & EO,
        "52-62_EE": mask_52_62 & EE,
    }

    cats_62_200 = {
        "62-200_BB": mask_62_200 & BB,
        "62-200_BO": mask_62_200 & BO,
        "62-200_BE": mask_62_200 & BE,
        "62-200_OB": mask_62_200 & OB,
        "62-200_OO": mask_62_200 & OO,
        "62-200_OE": mask_62_200 & OE,
        "62-200_EB": mask_62_200 & EB,
        "62-200_EO": mask_62_200 & EO,
        "62-200_EE": mask_62_200 & EE,
    }

    # categories = {
    #     "30-45_BB_OB_EB": cat_30_45_1,
    #     "30-45_BO_OO_EO": cat_30_45_2,
    #     "30-45_BE_OE_EE": cat_30_45_3
    # }
    categories = cats_30_45
    # categories.update(cats_30_45)
    categories.update(cats_45_52)
    categories.update(cats_52_62)
    categories.update(cats_62_200)

    return categories

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


def get_bin_edges_for_sample(bin_edges, sample):
    """Return the bin edges for a given physics sample.

    bin_edges can be either a single array-like object, or a dictionary
    keyed by sample names like "data", "DY", "VBF", and "ggH".
    """

    if isinstance(bin_edges, dict):
        if sample not in bin_edges:
            raise KeyError(
                f"Missing bin_edges entry for sample '{sample}'. "
                f"Available keys: {list(bin_edges.keys())}"
            )
        return bin_edges[sample]

    return bin_edges


def get_calib_category_names():
    """Return the calibration category names produced by get_calib_categories()."""

    pt_bins = ["30-45", "45-52", "52-62", "62-200"]
    eta_bins = ["BB", "BO", "BE", "OB", "OO", "OE", "EB", "EO", "EE"]

    return [f"{pt_bin}_{eta_bin}" for pt_bin in pt_bins for eta_bin in eta_bins]


def plot_variable_from_parquets_calib_category(
    variable,
    bin_edges,
    parquet_paths,
    calib_category,
    weight_variable="wgt_nominal",
    normalize=False,
    region="h-peak",
    xlabel=None,
    ylabel=None,
    title=None,
    output_name=None,
):
    """
    Plot a weighted histogram after applying one pt-eta calibration category.

    This intentionally mirrors plot_variable_from_parquets(), but adds one
    extra selection step using get_calib_categories(events). The original
    plot_variable_from_parquets() function is left unchanged.
    """

    bin_edges = np.asarray(bin_edges)
    hists = {}

    for label, parquet_path in parquet_paths.items():
        print(f"Processing {label}, category {calib_category}: {parquet_path}")
        print(f"Reading directories for {label} from: {get_matching_directories(parquet_path)}")

        events = dak.from_parquet(parquet_path)
        _, events = filterRegion(events, region=region)

        calib_categories = get_calib_categories(events)
        if calib_category not in calib_categories:
            raise KeyError(
                f"Unknown calib_category '{calib_category}'. "
                f"Available categories: {list(calib_categories.keys())}"
            )

        events = events[calib_categories[calib_category]]

        values = events[variable]
        weights = events[weight_variable]

        values_np, weights_np = dask.compute(values, weights)

        values_np = ak.to_numpy(values_np)
        weights_np = ak.to_numpy(weights_np)

        mask = np.isfinite(values_np) & np.isfinite(weights_np)
        values_np = values_np[mask]
        weights_np = weights_np[mask]

        if normalize:
            weight_sum = np.sum(weights_np)
            if weight_sum > 0:
                weights_np = weights_np / weight_sum

        hist_values, _ = np.histogram(
            values_np,
            bins=bin_edges,
            weights=weights_np,
        )
        hist_values_w2, _ = np.histogram(
            values_np,
            bins=bin_edges,
            weights=weights_np * weights_np,
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
            yerr=np.sqrt(hist_values_w2),
            linewidth=2,
        )

    plt.xlabel(xlabel if xlabel is not None else variable)

    if ylabel is not None:
        plt.ylabel(ylabel)
    elif normalize:
        plt.ylabel("Normalized events")
    else:
        plt.ylabel("Weighted events")

    # NOTE: no titles in plots for now
    # if title is not None:
        # plt.title(title)

    plt.legend(fontsize=10)
    plt.tight_layout()

    if output_name is not None:
        directory = os.path.dirname(output_name)
        if directory:
            os.makedirs(directory, exist_ok=True)
        plt.savefig(output_name, dpi=200)

    plt.show()

    return hists, bin_edges


def run_plots_for_block(
    desc,
    base_parquet_paths,
    variables,
    bin_edges,
    region,
    normalize=True,
    output_subdir=None,
):
    """
    For a given comparison block, make separate plots for each sample.

    Default output format:
        {region}/{desc}/{sample}/{var}_normalized.pdf

    If output_subdir is provided:
        {region}/{desc}/{sample}/{output_subdir}/{var}_normalized.pdf

    plot_variable_from_parquets() is intentionally unchanged.
    """

    for sample in sample_pattern_dict:
        parquet_paths = make_paths_for_sample(
            base_parquet_paths=base_parquet_paths,
            sample=sample,
        )

        sample_bin_edges = get_bin_edges_for_sample(bin_edges, sample)

        for var in variables:
            var_title = var_title_dict[var]

            if output_subdir is None:
                output_name = f"{region}/{desc}/{sample}/{var}_normalized.pdf"
            else:
                output_name = f"{region}/{desc}/{sample}/{output_subdir}/{var}_normalized.pdf"

            hists, bins = plot_variable_from_parquets(
                variable=var,
                bin_edges=sample_bin_edges,
                parquet_paths=parquet_paths,
                normalize=normalize,
                region=region,
                xlabel=var_title,
                output_name=output_name,
            )


def run_dimuon_mass_calib_category_plots_for_block(
    desc,
    base_parquet_paths,
    bin_edges,
    region,
    normalize=True,
    categories=None,
):
    """
    Make dimuon_mass plots split by pt-eta calibration category.

    Output format:
        {region}/{desc}/{sample}/{pt_eta_cat}/dimuon_mass_normalized.pdf
    """

    variable = "dimuon_mass"
    var_title = var_title_dict[variable]

    if categories is None:
        categories = get_calib_category_names()

    for sample in sample_pattern_dict:
        parquet_paths = make_paths_for_sample(
            base_parquet_paths=base_parquet_paths,
            sample=sample,
        )
        sample_bin_edges = get_bin_edges_for_sample(bin_edges, sample)

        for calib_category in categories:
            output_name = f"{region}/{desc}/{sample}/{calib_category}/{variable}_normalized.pdf"

            hists, bins = plot_variable_from_parquets_calib_category(
                variable=variable,
                bin_edges=sample_bin_edges,
                parquet_paths=parquet_paths,
                calib_category=calib_category,
                normalize=normalize,
                region=region,
                xlabel=var_title,
                title=f"{var_title}: {calib_category}",
                output_name=output_name,
            )



# ------------------------------------------------------------
# Cached / preloaded dask-awkward Record workflow
# ------------------------------------------------------------

def make_event_records_for_sample(
    base_parquet_paths,
    sample,
    region,
    columns=None,
    persist_records=True,
):
    """
    Build a dictionary of already-loaded dask-awkward Records for one physics sample.

    The output replaces the parquet path dictionary used by plot_variable_from_parquets().
    Each value is a dask-awkward Record that has already had filterRegion() applied.

    filterRegion() is applied before any dask.compute() call, so only the selected
    region is materialized later. If persist_records=True, the region-filtered
    Records are persisted once and reused by the inclusive and pt-eta category plots.
    """

    parquet_paths = make_paths_for_sample(
        base_parquet_paths=base_parquet_paths,
        sample=sample,
    )

    event_records = {}

    for label, parquet_path in parquet_paths.items():
        print(f"Loading {label}, sample={sample}: {parquet_path}")
        print(f"Reading directories for {label} from: {get_matching_directories(parquet_path)}")

        if columns is None:
            events = dak.from_parquet(parquet_path)
        else:
            events = dak.from_parquet(parquet_path, columns=columns)

        # Apply the region selection while events is still lazy.
        _, events = filterRegion(events, region=region)

        # With a distributed Client, this keeps the filtered graph/results in memory.
        # This avoids rereading/recomputing the parquet files for every category plot.
        if persist_records:
            events = dask.persist(events)[0]

        event_records[label] = events

    return event_records


def compute_hist_from_record(
    events,
    variable,
    bin_edges,
    weight_variable="wgt_nominal",
    normalize=False,
):
    """Compute a weighted np.histogram from an already-filtered dask-awkward Record."""

    values = events[variable]
    weights = events[weight_variable]

    values_np, weights_np = dask.compute(values, weights)

    values_np = ak.to_numpy(values_np)
    weights_np = ak.to_numpy(weights_np)

    mask = np.isfinite(values_np) & np.isfinite(weights_np)
    values_np = values_np[mask]
    weights_np = weights_np[mask]

    if normalize:
        weight_sum = np.sum(weights_np)
        if weight_sum > 0:
            weights_np = weights_np / weight_sum

    hist_values, _ = np.histogram(
        values_np,
        bins=bin_edges,
        weights=weights_np,
    )
    hist_values_w2, _ = np.histogram(
        values_np,
        bins=bin_edges,
        weights=weights_np * weights_np,
    )

    return hist_values, hist_values_w2


def plot_hist_dict(
    hists,
    bin_edges,
    variable,
    normalize=False,
    xlabel=None,
    ylabel=None,
    title=None,
    output_name=None,
):
    """Plot precomputed np.histogram arrays with mplhep.histplot()."""

    hep.style.use("CMS")
    plt.figure(figsize=(8, 6))

    for label, hist_values_l in hists.items():
        hist_values, hist_values_w2 = hist_values_l
        hep.histplot(
            hist_values,
            bins=bin_edges,
            label=label,
            histtype="step",
            yerr=np.sqrt(hist_values_w2),
            linewidth=2,
        )

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
        directory = os.path.dirname(output_name)
        if directory:
            os.makedirs(directory, exist_ok=True)
        plt.savefig(output_name, dpi=200)

    plt.show()
    plt.close()


def plot_variable_from_records(
    variable,
    bin_edges,
    event_records,
    weight_variable="wgt_nominal",
    normalize=False,
    xlabel=None,
    ylabel=None,
    title=None,
    output_name=None,
):
    """
    Plot from already-loaded dask-awkward Records.

    event_records format:
        {"Run2(2017) CHS": events_record, "Run2(2017) PUPPI": events_record, ...}
    """

    bin_edges = np.asarray(bin_edges)
    hists = {}

    for label, events in event_records.items():
        print(f"Computing histogram for {label}: {variable}")
        hists[label] = compute_hist_from_record(
            events=events,
            variable=variable,
            bin_edges=bin_edges,
            weight_variable=weight_variable,
            normalize=normalize,
        )

    plot_hist_dict(
        hists=hists,
        bin_edges=bin_edges,
        variable=variable,
        normalize=normalize,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        output_name=output_name,
    )

    return hists, bin_edges


def plot_variable_from_records_calib_category(
    variable,
    bin_edges,
    event_records,
    calib_category,
    weight_variable="wgt_nominal",
    normalize=False,
    xlabel=None,
    ylabel=None,
    title=None,
    output_name=None,
):
    """
    Plot one pt-eta calibration category from already-loaded Records.

    The calibration category mask is applied lazily to the already region-filtered
    Records. dask.compute() only happens after both selections are applied.
    """

    bin_edges = np.asarray(bin_edges)
    hists = {}

    for label, events in event_records.items():
        print(f"Computing histogram for {label}: {variable}, category={calib_category}")

        calib_categories = get_calib_categories(events)
        if calib_category not in calib_categories:
            raise KeyError(
                f"Unknown calib_category '{calib_category}'. "
                f"Available categories: {list(calib_categories.keys())}"
            )

        selected_events = events[calib_categories[calib_category]]

        hists[label] = compute_hist_from_record(
            events=selected_events,
            variable=variable,
            bin_edges=bin_edges,
            weight_variable=weight_variable,
            normalize=normalize,
        )

    plot_hist_dict(
        hists=hists,
        bin_edges=bin_edges,
        variable=variable,
        normalize=normalize,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        output_name=output_name,
    )

    return hists, bin_edges


def run_dimuon_mass_plots_for_block_cached_records(
    desc,
    base_parquet_paths,
    bin_edges,
    region,
    normalize=True,
    categories=None,
    persist_records=True,
):
    """
    Make inclusive and pt-eta category dimuon_mass plots using cached Records.

    For each {desc, sample}, parquet files are converted to dask-awkward Records once,
    filterRegion() is applied immediately, and the resulting Records are reused for:
        1. inclusive dimuon_mass
        2. all pt-eta calibration categories

    Outputs:
        {region}/{desc}/{sample}/inclusive/dimuon_mass_normalized.pdf
        {region}/{desc}/{sample}/{pt_eta_cat}/dimuon_mass_normalized.pdf
    """

    variable = "dimuon_mass"
    var_title = var_title_dict[variable]

    if categories is None:
        categories = get_calib_category_names()

    # Only read columns needed for dimuon_mass inclusive/category plots.
    columns = sorted({
        variable,
        "wgt_nominal",
        "mu1_eta",
        "mu2_eta",
        "mu1_pt",
    })

    for sample in sample_pattern_dict:
        sample_bin_edges = get_bin_edges_for_sample(bin_edges, sample)

        # Build and optionally persist the region-filtered Records once.
        event_records = make_event_records_for_sample(
            base_parquet_paths=base_parquet_paths,
            sample=sample,
            region=region,
            columns=columns,
            persist_records=persist_records,
        )

        # Inclusive plot.
        inclusive_output_name = f"{region}/{desc}/{sample}/inclusive/{variable}_normalized.pdf"
        plot_variable_from_records(
            variable=variable,
            bin_edges=sample_bin_edges,
            event_records=event_records,
            normalize=normalize,
            xlabel=var_title,
            output_name=inclusive_output_name,
        )

        # pt-eta category plots.
        for calib_category in categories:
            category_output_name = f"{region}/{desc}/{sample}/{calib_category}/{variable}_normalized.pdf"
            plot_variable_from_records_calib_category(
                variable=variable,
                bin_edges=sample_bin_edges,
                event_records=event_records,
                calib_category=calib_category,
                normalize=normalize,
                xlabel=var_title,
                title=f"{var_title}: {calib_category}",
                output_name=category_output_name,
            )


def run_plots_for_block_cached_records(
    desc,
    base_parquet_paths,
    variables,
    bin_edges,
    region,
    normalize=True,
    output_subdir=None,
    persist_records=True,
):
    """
    Make ordinary variable plots using cached / preloaded dask-awkward Records.

    This is the cached-record equivalent of run_plots_for_block(). For each
    {desc, sample}, parquet files are read once, filterRegion() is applied while
    the collection is still lazy, and the resulting region-filtered Records are
    reused for all variables in `variables`, e.g. jet1_eta_nominal and
    jet2_eta_nominal.

    Default output format:
        {region}/{desc}/{sample}/{var}_normalized.pdf

    If output_subdir is provided:
        {region}/{desc}/{sample}/{output_subdir}/{var}_normalized.pdf
    """

    # Only read columns needed for these plots and the region selection.
    columns = sorted(set(list(variables) + ["wgt_nominal", "dimuon_mass"]))

    for sample in sample_pattern_dict:
        sample_bin_edges = get_bin_edges_for_sample(bin_edges, sample)

        # Build and optionally persist the region-filtered Records once per
        # {desc, sample}. These same Records are reused for all variables.
        event_records = make_event_records_for_sample(
            base_parquet_paths=base_parquet_paths,
            sample=sample,
            region=region,
            columns=columns,
            persist_records=persist_records,
        )

        for var in variables:
            var_title = var_title_dict[var]

            if output_subdir is None:
                output_name = f"{region}/{desc}/{sample}/{var}_normalized.pdf"
            else:
                output_name = f"{region}/{desc}/{sample}/{output_subdir}/{var}_normalized.pdf"

            plot_variable_from_records(
                variable=var,
                bin_edges=sample_bin_edges,
                event_records=event_records,
                normalize=normalize,
                xlabel=var_title,
                output_name=output_name,
            )



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
    # Define the parquet paths
    # -------------------------------------------------------------
    
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


    # for desc, base_parquet_paths in plot_blocks.items():
    #     # Cached workflow for jet eta plots:
    #     # - dak.from_parquet() is called once per {desc, sample, label}
    #     # - filterRegion() is applied before dask.compute()
    #     # - the region-filtered dask-awkward Records are reused for
    #     #   jet1_eta_nominal and jet2_eta_nominal
    #     run_plots_for_block_cached_records(
    #         desc=desc,
    #         base_parquet_paths=base_parquet_paths,
    #         variables=variables,
    #         bin_edges=eta_bin_edges_by_sample,
    #         region=region,
    #         normalize=True,
    #         persist_records=True,
    #     )

    # ------------------------------------------------------------
    # Repeat for dimuon mass plot
    # ------------------------------------------------------------

    # -------------------------------------------------------------
    # Define the parquet paths
    # -------------------------------------------------------------
    
    # -------------------------------------------------------------
    # run23Comp
    # -------------------------------------------------------------
    
    run23Comp_paths = {
        "Run2(2017) NanoAODv12": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl/stage1_output/2017/compacted/data_*/0/*.parquet",
        "Run2(2017) NanoAODv15": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_Apr14_2026_UpdatedBtagWp/stage1_output/2017/compacted/data_*/0/*.parquet",
        "Run3(2024) NanoAODv15": "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterJetsHorn25GeV_Apr09_tightPassLepVeto_NoJER_v2/stage1_output/2024/compacted/data_C/0/*.parquet",
    }
    
    # -------------------------------------------------------------
    # run23Comp extended
    # -------------------------------------------------------------
    
    run23CompExt_paths = {
        "Run2(2017) NanoAODv12": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl/stage1_output/2017/compacted/data_*/0/*.parquet",
        "Run2(2017) NanoAODv15": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_Apr14_2026_UpdatedBtagWp/stage1_output/2017/compacted/data_*/0/*.parquet",
        "Run3(2022postEE) NanoAODv12": "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn25GeV_Apr09_tightPassLepVeto_NoJER_v2/stage1_output/2022postEE/compacted/data_*/0/*.parquet",
        "Run3(2024) NanoAODv15": "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterJetsHorn25GeV_Apr09_tightPassLepVeto_NoJER_v2/stage1_output/2024/compacted/data_C/0/*.parquet",
    }
    
    # -------------------------------------------------------------
    # run2NanoV12
    # -------------------------------------------------------------
    
    run2NanoV12_paths = {
        "Run2(2018) NanoAODv12": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl/stage1_output/2018/compacted/data_*/0/*.parquet",
        "Run2(2017) NanoAODv12": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl/stage1_output/2017/compacted/data_*/0/*.parquet",
        "Run2(2016postVFP) NanoAODv12": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl/stage1_output/2016postVFP/compacted/data_*/0/*.parquet",
        "Run2(2016preVFP) NanoAODv12": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl/stage1_output/2016preVFP/compacted/data_*/0/*.parquet",
    }
    
    
    # -------------------------------------------------------------
    # run2NanoV15
    # -------------------------------------------------------------
    
    run2NanoV15_paths = {
        "Run2(2018) NanoAODv15": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_Apr14_2026_UpdatedBtagWp/stage1_output/2018/compacted/data_*/0/*.parquet",
        "Run2(2017) NanoAODv15": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_Apr14_2026_UpdatedBtagWp/stage1_output/2017/compacted/data_*/0/*.parquet",
        "Run2(2016postVFP) NanoAODv15": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_Apr14_2026_UpdatedBtagWp/stage1_output/2016postVFP/compacted/data_*/0/*.parquet",
        "Run2(2016preVFP) NanoAODv15": "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_Apr14_2026_UpdatedBtagWp/stage1_output/2016preVFP/compacted/data_*/0/*.parquet",
    }
    
    
    # -------------------------------------------------------------
    # run3
    # -------------------------------------------------------------
    
    run3_paths = {
        "Run3(2024) NanoAODv15": "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterJetsHorn25GeV_Apr09_tightPassLepVeto_NoJER_v2/stage1_output/2024/compacted/data_*/0/*.parquet",
        "Run3(2023BPix) NanoAODv12": "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn25GeV_Apr09_tightPassLepVeto_NoJER_v2/stage1_output/2023BPix/compacted/data_*/0/*.parquet",
        "Run3(2023) NanoAODv12": "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn25GeV_Apr09_tightPassLepVeto_NoJER_v2/stage1_output/2023/compacted/data_*/0/*.parquet",
        "Run3(2022postEE) NanoAODv12": "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn25GeV_Apr09_tightPassLepVeto_NoJER_v2/stage1_output/2022postEE/compacted/data_*/0/*.parquet",
        "Run3(2022preEE) NanoAODv12": "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn25GeV_Apr09_tightPassLepVeto_NoJER_v2/stage1_output/2022preEE/compacted/data_*/0/*.parquet",
    }

    
    

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
        # Cached workflow:
        # - dak.from_parquet() is called once per {desc, sample, label}
        # - filterRegion() is applied before dask.compute()
        # - the region-filtered dask-awkward Records are reused for inclusive
        #   and all pt-eta category dimuon_mass plots
        run_dimuon_mass_plots_for_block_cached_records(
            desc=desc,
            base_parquet_paths=base_parquet_paths,
            bin_edges=mass_bin_edges_by_sample,
            region=region,
            normalize=True,
            persist_records=True,
        )