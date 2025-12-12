#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import csv
import logging
import itertools
from pathlib import Path
from typing import List, Tuple

import argparse

import dask_awkward as dak
import awkward as ak

from distributed import Client

# Your modules
from modules import selection
from modules.trials import get_stage1_path


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logger = logging.getLogger("get_yields")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)


# ----------------------------------------------------------------------
# Fields to read
# ----------------------------------------------------------------------
V1_FIELDS_2COMPUTE: List[str] = [
    # "gjj_mass",  # handled dynamically
    "wgt_nominal",
    "nBtagLoose_nominal",
    "nBtagMedium_nominal",
    # "mu1_pt",
    # "mu2_pt",
    # "mu1_eta",
    # "mu2_eta",
    # "mu1_phi",
    # "mu2_phi",
    "dimuon_pt",
    "dimuon_eta",
    # "dimuon_phi",
    "dimuon_mass",
    "jet1_phi_nominal",
    "jet1_pt_nominal",
    "jet2_pt_nominal",
    "jet2_phi_nominal",
    "jet1_eta_nominal",
    "jet2_eta_nominal",
    "jj_mass_nominal",
    "jj_dEta_nominal",
    "event",
    "njets_nominal",
    # "nfatJets_drmuon",
    # "MET_pt",
]


# ----------------------------------------------------------------------
# Dask client helper
# ----------------------------------------------------------------------
def get_dask_client(
    n_workers: int = 12,
    threads_per_worker: int = 1,
    memory_limit: str = "10 GiB",
) -> Client:
    """Create or reuse a local Dask client."""
    try:
        client = Client.current()
        logger.info("Reusing existing Dask client: %s", client)
        return client
    except ValueError:
        client = Client(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            processes=True,
            memory_limit=memory_limit,
        )
        logger.info("Created new Dask client: %s", client)
        return client


# ----------------------------------------------------------------------
# Core yield function
# ----------------------------------------------------------------------
def get_yield(
    process: str,
    load_path: str,
    do_vbf_filter_study: bool,
    year: str,
    do_VH_veto: bool,
    category: str,
    region: str,
) -> Tuple[str, str, str, str, float]:
    """
    Compute total yield for a given process (glob pattern) and configuration.
    """
    print(f"{'=' * 5} {process} {'=' * 5}")

    # Fresh copy of the fields list
    fields = list(V1_FIELDS_2COMPUTE)

    # Dynamically include/exclude gjj_mass
    if "data" in process:
        # ensure gjj_mass is NOT in the list for data
        if "gjj_mass" in fields:
            fields.remove("gjj_mass")
    else:
        # ensure gjj_mass IS in the list for MC
        if "gjj_mass" not in fields:
            fields.append("gjj_mass")

    # glob over all matching sample dirs
    parquet_pattern = os.path.join(load_path, process, "*", "*.parquet")
    parquet_files = glob.glob(parquet_pattern)

    if not parquet_files:
        logger.warning("No parquet files found for '%s' under '%s'", process, load_path)
        return process, category, region, year, 0.0

    # dask_awkward read per process
    events_lazy = dak.from_parquet(parquet_pattern, columns=fields)

    # Materialize into awkward.Array
    events = events_lazy.compute()

    # Apply region / category cuts
    events = selection.applyRegionCatCuts(
        events,
        category=category,
        region_name=region,
        process=process,
        variation="nominal",
        do_vbf_filter_study=do_vbf_filter_study,
        do_VH_veto=do_VH_veto,
    )

    # Sum nominal weights
    wgts = ak.fill_none(events["wgt_nominal"], value=1.0)
    total_integral = float(ak.sum(wgts))

    print(f"\t==> Total Yield: {total_integral:.3f}")
    return process, category, region, year, total_integral


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main() -> None:
    # Environment setup
    cwd = str(Path.cwd())
    print(f"PWD: {cwd}")

    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + f":{cwd}"
    os.environ["X509_USER_PROXY"] = f"{cwd}/voms_proxy.txt"
    os.environ["XRD_REQUESTTIMEOUT"] = "2400"

    print(f"PYTHONPATH: {os.environ['PYTHONPATH']}")

    # Get base stage1 directory from your trials helper
    stage1_dir = get_stage1_path()  # default = "current"
    load_path_template = str(Path(stage1_dir) / "{year}" / "f1_0")
    print(f"Using LOAD_PATH template: {load_path_template}")

    # Physics / config toggles
    do_VH_veto = False
    do_vbf_filter_study = True

    # regions = ["z-peak", "h-sidebands"]
    # categories = ["ggh", "vbf"]
    regions = ["h-sidebands"]
    categories = ["vbf"]


    args = parse_args()

    # Define all possible years we want to support
    ALL_YEARS = [
        "2018",
        "2017",
        "2016preVFP",
        "2016postVFP",
        # "2022preEE",
        # "2022postEE",
        # "2023",
        # "2023BPix",
        # "2024",
    ]

    if args.years == ["all"]:
        years = ALL_YEARS
    else:
        years = args.years

    info = []

    suffix = "_VHVeto" if do_VH_veto else ""

    # Use parent of stage1_output as tag for output file name
    dataset_tag = Path(stage1_dir).parent.name
    tagYear = "_".join(years)
    outfile = f"yield_{dataset_tag}{suffix}_{tagYear}_New.csv"
    print(f"Will write yields to: {outfile}")

    # Start Dask client
    get_dask_client()

    # Loop over (category, year, region)
    for category, year, region in itertools.product(categories, years, regions):
        print(f"\n\n***** {year} *****")
        load_path = load_path_template.format(year=year)

        # Common kwargs for all processes
        common_kwargs = dict(
            load_path=load_path,
            year=year,
            category=category,
            region=region,
            do_vbf_filter_study=do_vbf_filter_study,
            do_VH_veto=do_VH_veto,
        )

        processes = [
            # "data_A",
            # "data_B",
            # "data_C",
            # "data_D",
            # "data*",
            # "vbf_powheg",
            "vbf_powheg_dipole",
            # "ggh_powhegPS",
            # # "dyTo2L_M-50_incl",
            # # "dyTo2L_M-50_incl_XSDYTurbo",
            # "dy_VBF_filter",
            # "dy_M-50_aMCatNLO",
            # "dy_M-100To200_aMCatNLO",
            # "dy_M-50_MiNNLO",
            # "dy_M-100To200_MiNNLO",
            # "ttjets_*",
            # "st_t_*",
            # "st_tW_*",
            # "w*_*",
            # "zz_*",
            # # "ewk_lljj",
            # "ewk_lljj_mll50_mjj120",
        ]

        for proc in processes:
            info.append(get_yield(proc, **common_kwargs))

    # Write CSV
    with open(outfile, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["sample", "category", "region", "year", "yield"])
        writer.writerows(info)

    print(f"\nWrote {len(info)} rows to {outfile}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute yields for H→μμ stage1 outputs for one or more trials."
    )

    parser.add_argument(
        "--years",
        nargs="+",
        default=["2018"],
        help=(
            "Years to run over. Examples:\n"
            "  --years 2018\n"
            "  --years 2018 2017 2016preVFP\n"
            "  --years all\n"
            "If not provided, defaults to ['2018']."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
