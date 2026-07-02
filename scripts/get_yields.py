#!/usr/bin/env python3
"""
Example:
python scripts/get_yields.py \
    --input /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl/stage1_output \
    -y 2017

time python scripts/get_yields.py \
    --input /depot/cms/users/hu1027/Hmumu_Categorization_Transformer_output/May20_2026Test_pairbias4_nominal_samples_transformer_scored/  \
    -y 2017 \
    --categorizer transformer-score \
    --output-csv yield_20May_transformer_0p92522.csv \
    --summary-output-csv yield_20May_transformer_0p92522_summary.csv
    

time python scripts/get_yields.py \
    --input /depot/cms/users/hu1027/Hmumu_Categorization_Transformer_output/May20_2026Test_pairbias4_nominal_samples_transformer_scored/  \
    -y 2017 \
    --categorizer cutbased \
    --output-csv yield_20May_cutbased_0p92522.csv \
    --summary-output-csv yield_20May_cutbased_0p92522_summary.csv
"""

import argparse
import glob
import itertools
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import awkward as ak
import dask_awkward as dak
import pandas as pd

# Your modules
from modules import selection
from modules.dask_utils import close_dask_client, get_dask_client
from modules.utils import get_compacted_path, logger
from modules.sample_config import get_bkg_sig_dicts
from cli.common_argparser import build_common_parser

# ----------------------------------------------------------------------
# Fields to read
# ----------------------------------------------------------------------
V1_FIELDS_2COMPUTE: List[str] = [
    # "gjj_mass",  # handled dynamically
    "wgt_nominal",
    "separate_wgt_zpt",
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

TRANSFORMER_SCORE_FIELDS: List[str] = [
    "transf_vbf_score",
    "transf_vbf_logit",
    "transf_ggh_score",
    "transf_ggh_logit",
    "transf_bkg_score",
    "transf_bkg_logit",
]


def order_processes(processes: List[str]) -> List[str]:
    """
    Keep process order in the CSV as:
    data -> ggH -> VBF -> backgrounds
    """
    data_processes = []
    ggh_processes = []
    vbf_processes = []
    background_processes = []

    for process in sorted(processes):
        process_lower = process.lower()
        if "data" in process_lower:
            data_processes.append(process)
        elif process_lower.startswith("ggh"):
            ggh_processes.append(process)
        elif process_lower.startswith("vbf"):
            vbf_processes.append(process)
        else:
            background_processes.append(process)

    return data_processes + ggh_processes + vbf_processes + background_processes


def safe_divide(num: float, den: float) -> float:
    if den == 0:
        return float("inf") if num > 0 else 0.0
    return num / den


def safe_s_over_sqrt_sb(signal_yield: float, background_yield: float) -> float:
    denom = signal_yield + background_yield
    if denom <= 0:
        return 0.0
    return signal_yield / (denom ** 0.5)


def get_selected_njets(events) -> ak.Array:
    for field in ("njets_nominal", "njets"):
        if field in events.fields:
            return ak.fill_none(events[field], -999)
    raise KeyError("Neither 'njets_nominal' nor 'njets' found in selected events.")


def weighted_yield_for_mask(events, mask, process: str) -> float:
    if "data" in process.lower():
        return float(ak.sum(ak.values_astype(mask, "int64")))

    wgts = ak.fill_none(events["wgt_nominal"], 0.0)
    if "separate_wgt_zpt" in events.fields:
        wgts = wgts / ak.fill_none(events["separate_wgt_zpt"], 1.0)
    return float(ak.sum(wgts[mask]))


def sum_for_type(group: pd.DataFrame, sample_type: str, column: str) -> float:
    return group.loc[group["sample_type"] == sample_type, column].sum()


def resolve_input_layout(input_path: Path, years: List[str]) -> Tuple[str, Dict[str, Path]]:
    """
    Support both of these layouts:

    1. Legacy stage1 layout:
       <input>/<year>/f1_0/<sample>/<chunk>/*.parquet

    2. Direct sample-root layout:
       <input>/<sample>/**/*.parquet
    """
    year_dirs = {
        year: get_compacted_path(input_path / year / "f1_0")
        for year in years
        if (input_path / year).is_dir()
    }
    if year_dirs:
        missing_years = [year for year in years if year not in year_dirs]
        if missing_years:
            raise FileNotFoundError(
                "Input path looks like a year-organized stage1 directory, but these "
                f"years are missing: {missing_years}"
            )
        return "stage1", year_dirs

    if input_path.is_dir():
        parquet_files = glob.glob(str(input_path / "*" / "**" / "*.parquet"), recursive=True)
        if parquet_files:
            return "sample_root", {year: input_path for year in years}

    raise FileNotFoundError(
        "Unable to resolve input layout. Expected either "
        f"'<input>/<year>/f1_0/.../*.parquet' or '<input>/<sample>/**/*.parquet'. "
        f"Got: {input_path}"
    )


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
    categorizer: str,
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
    if categorizer == "transformer-score":
        for field in TRANSFORMER_SCORE_FIELDS:
            if field not in fields:
                fields.append(field)

    # glob over all matching sample dirs
    parquet_pattern = os.path.join(load_path, process, "**", "*.parquet")
    parquet_files = glob.glob(parquet_pattern, recursive=True)

    if not parquet_files:
        logger.warning("No parquet files found for '%s' under '%s'", process, load_path)
        return {
            "sample": process,
            "category": category,
            "region": region,
            "year": year,
            "categorizer": categorizer,
            "raw_events": 0,
            "raw_events_0j": 0,
            "raw_events_1j": 0,
            "raw_events_2plusj": 0,
            "yield": 0.0,
            "yield_0j": 0.0,
            "yield_1j": 0.0,
            "yield_2plusj": 0.0,
        }

    # dask_awkward read per process
    events_lazy = dak.from_parquet(sorted(parquet_files), columns=fields)

    # Materialize into awkward.Array
    events = events_lazy.compute()

    # Apply region / category cuts
    if categorizer == "transformer-score":
        events = selection.applyRegionCatCutsByScore(
            events,
            category=category,
            region_name=region,
            process=process,
            variation="nominal",
            do_vbf_filter_study=do_vbf_filter_study,
            year=year,
            do_VH_veto=do_VH_veto,
        )
    else:
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
    njets_selected = get_selected_njets(events)
    njet0_mask = njets_selected == 0
    njet1_mask = njets_selected == 1
    njet2plus_mask = njets_selected >= 2
    raw_events_0j = int(ak.sum(ak.values_astype(njet0_mask, "int64")))
    raw_events_1j = int(ak.sum(ak.values_astype(njet1_mask, "int64")))
    raw_events_2plusj = int(ak.sum(ak.values_astype(njet2plus_mask, "int64")))

    # weights
    if "data" in process.lower():
        # data: yield is raw count
        yield_ = float(n_raw)
    else:
        # MC: expected yield
        wgts = ak.fill_none(events["wgt_nominal"], 0.0)

        # remove Zpt weight if you want "no-zpt"
        if "separate_wgt_zpt" in events.fields:
            wgts = wgts / ak.fill_none(events["separate_wgt_zpt"], 1.0)

        yield_ = float(ak.sum(wgts))

    yield_0j = weighted_yield_for_mask(events, njet0_mask, process)
    yield_1j = weighted_yield_for_mask(events, njet1_mask, process)
    yield_2plusj = weighted_yield_for_mask(events, njet2plus_mask, process)

    print(f"{process:30}    {n_raw:10}  {yield_:10.3f}")

    return {
        "sample": process,
        "category": category,
        "region": region,
        "year": year,
        "categorizer": categorizer,
        "raw_events": n_raw,
        "raw_events_0j": raw_events_0j,
        "raw_events_1j": raw_events_1j,
        "raw_events_2plusj": raw_events_2plusj,
        "yield": yield_,
        "yield_0j": yield_0j,
        "yield_1j": yield_1j,
        "yield_2plusj": yield_2plusj,
    }


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main() -> None:
    parser = build_common_parser()
    parser.add_argument(
        "--categorizer",
        default="cutbased",
        choices=["cutbased", "transformer-score"],
        help=(
            "Categorization strategy for ggH/VBF yields. "
            "'cutbased' uses the existing jj-mass/jj-dEta selection, while "
            "'transformer-score' tags events using transf_vbf_score and "
            "transf_ggh_score."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional output CSV path override for the per-sample yields table.",
    )
    parser.add_argument(
        "--summary-output-csv",
        default="",
        help="Optional output CSV path override for the summary table.",
    )
    args = parser.parse_args()

    # Environment setup
    cwd = str(Path.cwd())
    print(f"PWD: {cwd}")

    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + f":{cwd}"
    os.environ["X509_USER_PROXY"] = f"{cwd}/voms_proxy.txt"
    os.environ["XRD_REQUESTTIMEOUT"] = "2400"

    print(f"PYTHONPATH: {os.environ['PYTHONPATH']}")

    if args.input_path == "":
        raise ValueError(
            "Missing input path. Please provide --input /path/to/stage1_output"
        )

    stage1_dir = Path(args.input_path)
    print(f"Using input base path: {stage1_dir}")

    # Physics / config toggles
    do_VH_veto = False
    do_vbf_filter_study = args.do_vbf_filter_study

    # regions = ["signal", "h-sidebands"]  # "z-peak",
    # regions = ["signal"]  # "z-peak",
    # regions = ["h-sidebands", "z-peak"]
    regions = [
        "h-sidebands",
        "h-peak"
        # "signal",
    ]

    # categories = ["vbf", "ggh"]
    # categories = ["vbf"]
    categories = ["nocat", "vbf", "ggh"]
    # categories = ["nocat"]


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

    years_arg_explicit = any(
        token == "--years" or token.startswith("--years=") for token in sys.argv[1:]
    )
    if years_arg_explicit:
        years = ALL_YEARS if args.years == ["all"] else args.years
    else:
        years = [args.year]
    input_layout, load_paths_by_year = resolve_input_layout(stage1_dir, years)
    print(f"Resolved input layout: {input_layout}")
    for year in years:
        print(f"Resolved load path for {year}: {load_paths_by_year[year]}")

    suffix = "_VHVeto" if do_VH_veto else ""
    if do_vbf_filter_study:
        suffix += "_vbf_filter_study"
    categorizer_tag = args.categorizer.replace("-", "_")

    # For legacy stage1_output inputs, keep the historical tag naming.
    dataset_tag = (
        stage1_dir.parent.name if input_layout == "stage1" else stage1_dir.name
    )
    tagYear = "_".join(years)
    default_outfile = f"yield_{dataset_tag}_{categorizer_tag}{suffix}_{tagYear}_VBFgT0p966_V2.csv"
    # default_outfile = f"yield_{dataset_tag}_{categorizer_tag}{suffix}_{tagYear}.csv"
    outfile = args.output_csv if args.output_csv else default_outfile
    print(f"Will write yields to: {outfile}")

    # Start Dask client
    get_dask_client(args.use_gateway, cluster_index=args.cluster_index)

    rows: List[Dict[str, Any]] = []

    # Loop over (category, year, region)
    for category, year, region in itertools.product(categories, years, regions):
        print("-" * 60)
        print(f"\n\nCategory: {category} | Year: {year} | Region: {region}")
        load_path = str(load_paths_by_year[year])

        # Common kwargs for all processes
        common_kwargs = dict(
            load_path=load_path,
            year=year,
            category=category,
            region=region,
            do_vbf_filter_study=do_vbf_filter_study,
            do_VH_veto=do_VH_veto,
            categorizer=args.categorizer,
        )

        _, _, combined_sample_dict = get_bkg_sig_dicts(
            yaml_path=args.sample_config,
            year=year,
        )

        processes = set()
        for group_name, group_samples in combined_sample_dict.items():
            if group_name == "DYVBF" and not do_vbf_filter_study:
                continue
            processes.update(group_samples)
        processes.add("data*")
        processes = order_processes(list(processes))
        print(f"Processes to compute yields for (total {len(processes)}):")
        print(processes)

        print("-" * 60)
        print(f"{'Process':30}    {'Raw events':10}  {'Yield':10}")
        print("-" * 60)

        for proc in processes:
            # if "data" in proc and region == "h-peak":
                # continue
            rows.append(get_yield(proc, **common_kwargs))

    df = pd.DataFrame(rows)

    # Write CSV
    df.to_csv(outfile, index=False)
    print(f"\nWrote {len(df)} rows to {outfile}")

    # Print sum of data / signal / background per (year,cat,region)
    df2 = df.copy()
    df2["is_data"] = df2["sample"].str.contains("data", case=False, regex=False)
    signal_samples_by_year: Dict[str, set[str]] = {}
    ggh_samples_by_year: Dict[str, set[str]] = {}
    vbf_samples_by_year: Dict[str, set[str]] = {}
    background_samples_by_year: Dict[str, set[str]] = {}
    for year in years:
        bkg_sample_dict, sig_sample_dict, _ = get_bkg_sig_dicts(
            yaml_path=args.sample_config,
            year=year,
        )
        signal_samples_by_year[year] = set(
            sample for samples in sig_sample_dict.values() for sample in samples
        )
        ggh_samples_by_year[year] = set(sig_sample_dict.get("GGH", []))
        vbf_samples_by_year[year] = set(sig_sample_dict.get("VBF", []))
        background_samples_by_year[year] = set(
            sample for samples in bkg_sample_dict.values() for sample in samples
        )

    def classify_sample(row: pd.Series) -> str:
        sample = row["sample"]
        year = row["year"]
        if "data" in sample.lower():
            return "data"
        if sample in signal_samples_by_year.get(year, set()):
            return "signal"
        if sample in background_samples_by_year.get(year, set()):
            return "background"
        return "other"

    df2["sample_type"] = df2.apply(classify_sample, axis=1)
    summary = (
        df2.groupby(["year", "category", "region", "categorizer"], as_index=False)
        .apply(
            lambda g: pd.Series({
                "data_yield": g.loc[g["sample_type"] == "data", "yield"].sum(),
                "ggh_yield": g.loc[
                    g["sample"].isin(ggh_samples_by_year.get(g.name[0], set())), "yield"
                ].sum(),
                "vbf_yield": g.loc[
                    g["sample"].isin(vbf_samples_by_year.get(g.name[0], set())), "yield"
                ].sum(),
                "signal_yield": g.loc[g["sample_type"] == "signal", "yield"].sum(),
                "background_yield": g.loc[g["sample_type"] == "background", "yield"].sum(),
                "mc_yield": g.loc[
                    g["sample_type"].isin(["signal", "background"]), "yield"
                ].sum(),
                "data_minus_bkg": (
                    g.loc[g["sample_type"] == "data", "yield"].sum()
                    - g.loc[g["sample_type"] == "background", "yield"].sum()
                ),
                "data_minus_mc": (
                    g.loc[g["sample_type"] == "data", "yield"].sum()
                    - g.loc[g["sample_type"].isin(["signal", "background"]), "yield"].sum()
                ),
                "s_over_sqrt_s_plus_b": safe_s_over_sqrt_sb(
                    g.loc[g["sample_type"] == "signal", "yield"].sum(),
                    g.loc[g["sample_type"] == "background", "yield"].sum(),
                ),
                "s_over_b": safe_divide(
                    g.loc[g["sample_type"] == "signal", "yield"].sum(),
                    g.loc[g["sample_type"] == "background", "yield"].sum(),
                ),
                "signal_purity": safe_divide(
                    g.loc[g["sample_type"] == "signal", "yield"].sum(),
                    g.loc[g["sample_type"] == "signal", "yield"].sum()
                    + g.loc[g["sample_type"] == "background", "yield"].sum(),
                ),
                "raw_events_0j": g["raw_events_0j"].sum(),
                "raw_events_1j": g["raw_events_1j"].sum(),
                "raw_events_2plusj": g["raw_events_2plusj"].sum(),
                "yield_0j": g["yield_0j"].sum(),
                "yield_1j": g["yield_1j"].sum(),
                "yield_2plusj": g["yield_2plusj"].sum(),
                "frac_0j": safe_divide(g["yield_0j"].sum(), g["yield"].sum()),
                "frac_1j": safe_divide(g["yield_1j"].sum(), g["yield"].sum()),
                "frac_2plusj": safe_divide(g["yield_2plusj"].sum(), g["yield"].sum()),
                "data_yield_0j": sum_for_type(g, "data", "yield_0j"),
                "data_yield_1j": sum_for_type(g, "data", "yield_1j"),
                "data_yield_2plusj": sum_for_type(g, "data", "yield_2plusj"),
                "signal_yield_0j": sum_for_type(g, "signal", "yield_0j"),
                "signal_yield_1j": sum_for_type(g, "signal", "yield_1j"),
                "signal_yield_2plusj": sum_for_type(g, "signal", "yield_2plusj"),
                "background_yield_0j": sum_for_type(g, "background", "yield_0j"),
                "background_yield_1j": sum_for_type(g, "background", "yield_1j"),
                "background_yield_2plusj": sum_for_type(g, "background", "yield_2plusj"),
                "data_frac_0j": safe_divide(
                    sum_for_type(g, "data", "yield_0j"),
                    g.loc[g["sample_type"] == "data", "yield"].sum(),
                ),
                "data_frac_1j": safe_divide(
                    sum_for_type(g, "data", "yield_1j"),
                    g.loc[g["sample_type"] == "data", "yield"].sum(),
                ),
                "data_frac_2plusj": safe_divide(
                    sum_for_type(g, "data", "yield_2plusj"),
                    g.loc[g["sample_type"] == "data", "yield"].sum(),
                ),
                "signal_frac_0j": safe_divide(
                    sum_for_type(g, "signal", "yield_0j"),
                    g.loc[g["sample_type"] == "signal", "yield"].sum(),
                ),
                "signal_frac_1j": safe_divide(
                    sum_for_type(g, "signal", "yield_1j"),
                    g.loc[g["sample_type"] == "signal", "yield"].sum(),
                ),
                "signal_frac_2plusj": safe_divide(
                    sum_for_type(g, "signal", "yield_2plusj"),
                    g.loc[g["sample_type"] == "signal", "yield"].sum(),
                ),
                "background_frac_0j": safe_divide(
                    sum_for_type(g, "background", "yield_0j"),
                    g.loc[g["sample_type"] == "background", "yield"].sum(),
                ),
                "background_frac_1j": safe_divide(
                    sum_for_type(g, "background", "yield_1j"),
                    g.loc[g["sample_type"] == "background", "yield"].sum(),
                ),
                "background_frac_2plusj": safe_divide(
                    sum_for_type(g, "background", "yield_2plusj"),
                    g.loc[g["sample_type"] == "background", "yield"].sum(),
                ),
                "ggh_fraction_of_signal": safe_divide(
                    g.loc[
                        g["sample"].isin(ggh_samples_by_year.get(g.name[0], set())), "yield"
                    ].sum(),
                    g.loc[g["sample_type"] == "signal", "yield"].sum(),
                ),
                "vbf_fraction_of_signal": safe_divide(
                    g.loc[
                        g["sample"].isin(vbf_samples_by_year.get(g.name[0], set())), "yield"
                    ].sum(),
                    g.loc[g["sample_type"] == "signal", "yield"].sum(),
                ),
                "data_over_mc": safe_divide(
                    g.loc[g["sample_type"] == "data", "yield"].sum(),
                    g.loc[g["sample_type"].isin(["signal", "background"]), "yield"].sum(),
                ),
                "data_over_bkg": safe_divide(
                    g.loc[g["sample_type"] == "data", "yield"].sum(),
                    g.loc[g["sample_type"] == "background", "yield"].sum(),
                ),
            })
        )
        .reset_index(drop=True)
    )
    print("\n=== Data / Signal / Background yield summary ===")
    print(summary.to_string(index=False))

    # add  summary to the output CSV as well
    default_summary_outfile = (
        f"yield_summary_{dataset_tag}_{categorizer_tag}{suffix}_{tagYear}_VBFgT0p966_V2.csv"
        # f"yield_summary_{dataset_tag}_{categorizer_tag}{suffix}_{tagYear}.csv"
    )
    summary_outfile = (
        args.summary_output_csv if args.summary_output_csv else default_summary_outfile
    )
    summary.to_csv(summary_outfile, index=False)
    print(f"\nWrote summary to {summary_outfile}")
    close_dask_client()



if __name__ == "__main__":
    main()
