#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import itertools
import os
from pathlib import Path
from typing import Any, Dict, List

import awkward as ak
import dask_awkward as dak
import pandas as pd

# Your modules
from modules import selection
from modules.dask_utils import close_dask_client, get_dask_client
from modules.trials import get_stage1_path
from modules.utils import logger

# ----------------------------------------------------------------------
# Fields to read
# ----------------------------------------------------------------------
V1_FIELDS_2COMPUTE: List[str] = [
    # "gjj_mass",  # handled dynamically
    "wgt_nominal",
    "separate_wgt_zpt_wgt",
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
) -> Dict[str, Any]:
    """
    Compute total yield for a given process (glob pattern) and configuration.
    Returns a dict row for pandas DataFrame.
    """
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
        return {
            "sample": process,
            "category": category,
            "region": region,
            "year": year,
            "raw_events": 0,
            "yield": 0.0,
        }

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

    n_raw = int(ak.num(events, axis=0))

    # weights
    if "data" in process.lower():
        # data: yield is raw count
        yield_ = float(n_raw)
    else:
        # MC: expected yield
        wgts = ak.fill_none(events["wgt_nominal"], 0.0)

        # remove Zpt weight if you want "no-zpt"
        if "separate_wgt_zpt_wgt" in events.fields:
            wgts = wgts / ak.fill_none(events["separate_wgt_zpt_wgt"], 1.0)

        yield_ = float(ak.sum(wgts))

    print(f"{process:30}    {n_raw:10}  {yield_:10.3f}")

    return {
        "sample": process,
        "category": category,
        "region": region,
        "year": year,
        "raw_events": n_raw,
        "yield": yield_,
    }


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
    do_vbf_filter_study = False

    # regions = ["signal", "h-sidebands"]  # "z-peak",
    # regions = ["signal"]  # "z-peak",
    # regions = ["h-sidebands", "z-peak"]
    regions = [
        "h-sidebands",
        "h-peak"
        # "signal",
    ]

    # categories = ["ggh", "vbf"]
    categories = ["nocat", "vbf", "ggh"]
    # categories = ["nocat"]

    args = parse_args()

    # Define all possible years we want to support
    ALL_YEARS = [
        # "2018",
        # "2017",
        # "2016preVFP",
        # "2016postVFP",
        "2022preEE",
        "2022postEE",
        "2023",
        "2023BPix",
        "2024",
    ]

    years = ALL_YEARS if args.years == ["all"] else args.years

    suffix = "_VHVeto" if do_VH_veto else ""

    # Use parent of stage1_output as tag for output file name
    dataset_tag = Path(stage1_dir).parent.name
    tagYear = "_".join(years)
    outfile = f"yield_{dataset_tag}{suffix}_{tagYear}.csv"
    print(f"Will write yields to: {outfile}")

    # Start Dask client
    get_dask_client()

    rows: List[Dict[str, Any]] = []

    # Loop over (category, year, region)
    for category, year, region in itertools.product(categories, years, regions):
        print("-" * 60)
        print(f"\n\nCategory: {category} | Year: {year} | Region: {region}")
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
            # "dy_VBF_filter",
            "data*",
            "ggh_powhegPS",
            "vbf_powheg",
            "vbf_powheg_dipole",
            "dyTo2L_M-50_incl",
            "dyTo2Mu_M-50_aMCatNLO",
            "ttjets_*",
            "st_*",
            "ewk_*",
            "w*_*",
            "zz_*",
        ]

        print("-" * 60)
        print(f"{'Process':30}    {'Raw events':10}  {'Yield':10}")
        print("-" * 60)

        for proc in processes:
            if "data" in proc and region == "h-peak":
                continue
            rows.append(get_yield(proc, **common_kwargs))

    df = pd.DataFrame(rows)

    # Write CSV
    df.to_csv(outfile, index=False)
    print(f"\nWrote {len(df)} rows to {outfile}")

    # Print sum of MC vs data per (year,cat,region)
    df2 = df.copy()
    df2["is_data"] = df2["sample"].str.contains("data", case=False, regex=False)
    summary = (
        df2.groupby(["year", "category", "region"], as_index=False)
        .apply(
            lambda g: pd.Series({
                "data_yield": g.loc[g["is_data"], "yield"].sum(),
                "mc_yield": g.loc[~g["is_data"], "yield"].sum(),
                "ratio": (
                    g.loc[g["is_data"], "yield"].sum()
                    / g.loc[~g["is_data"], "yield"].sum()
                    if g.loc[~g["is_data"], "yield"].sum() > 0
                    else float("inf")
                ),
            })
        )
        .reset_index(drop=True)
    )
    print("\n=== Data vs MC yield summary ===")
    print(summary.to_string(index=False))

    # add  summary to the output CSV as well
    summary_outfile = f"yield_summary_{dataset_tag}{suffix}_{tagYear}.csv"
    summary.to_csv(summary_outfile, index=False)
    print(f"\nWrote summary to {summary_outfile}")
    close_dask_client()


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
