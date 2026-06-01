"""
Common ArgumentParser for copperhead framework.

This function defines the full, shared CLI.
"""
import argparse
import logging
from pathlib import Path

from modules.classify_year import is_run2, is_run3


def build_common_parser() -> argparse.ArgumentParser:
    """Build and return the common ArgumentParser used by copperhead scripts."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-y",
        "--year",
        dest="year",
        default="2018",
        action="store",
        help=(
            "year value. The options are: "
            "2016preVFP, 2016postVFP, 2017, 2018, "
            "2022preEE, 2022postEE, 2023, 2023BPix, 2024"
        ),
    )
    parser.add_argument(
        "--years",
        dest="years",
        default=["2018"],
        nargs="*",
        type=str,
        action="store",
        help=(
            "list of year values. The options are: "
            "2016preVFP, 2016postVFP, 2017, 2018, "
            "2022preEE, 2022postEE, 2023, 2023BPix, 2024"
        ),
    )
    parser.add_argument(
        "-l",
        "--label",
        dest="label",
        help="Run label: directory under plot_path to find inputs",
    )
    parser.add_argument(
        "-input",
        "--input_path",
        dest="input_path",
        default="",
        action="store",
        help="input path to read stage1 output files",
    )
    parser.add_argument(
        "-save",
        "--save_path",
        dest="save_path",
        default="validation",
        action="store",
        help="save path to store stage1 output files",
    )
    parser.add_argument(
        "--save_postfix",
        type=str,
        default="",
        help="String to append to output filenames",
    )
    parser.add_argument(
        "--yaml",
        dest="dataset_yaml_file",
        default="configs/datasets/dataset.yaml",
        help="Path of yaml file containing the dataset names",
    )
    parser.add_argument(
        "--sample-config",
        dest="sample_config",
        default="configs/samples/samples.yaml",
        help="Path to the sample configuration YAML file",
    )
    parser.add_argument(
        "-frac",
        "--change_fraction",
        dest="fraction",
        default=None,
        action="store",
        help="change fraction of steps of the data",
    )
    parser.add_argument(
        "-data",
        "--data",
        dest="data_samples",
        default=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        nargs="*",
        type=str,
        action="store",
        help="list of data samples represented by alphabetical letters A-H",
    )
    parser.add_argument(
        "-bkg",
        "--background",
        dest="bkg_samples",
        default=[],
        nargs="*",
        type=str,
        action="store",
        help="list of bkg samples represented by shorthands: DY, TT, ST, DB (diboson), EWK",
    )
    parser.add_argument(
        "-sig",
        "--signal",
        dest="sig_samples",
        default=["VBF","ggH"],
        nargs="*",
        type=str,
        action="store",
        help="list of sig samples represented by shorthands: ggH, VBF",
    )
    parser.add_argument(
        "--use_gateway",
        dest="use_gateway",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, uses dask gateway client instead of local",
    )
    parser.add_argument(
        "--cluster_index",
        dest="cluster_index",
        default=0,
        type=int,
        help="Index of the Dask Gateway cluster to connect to",
    )
    parser.add_argument(
        "-aod_v",
        "--NanoAODv",
        type=int,
        dest="NanoAODv",
        default=9,
        choices=[9, 12, 15],
        help="version number of NanoAOD samples we're working with. Currently, 9, 12, and 15 are supported",
    )

    parser.add_argument(
        "--log-level",
        default=logging.INFO,
        type=lambda x: getattr(logging, x),
        help="Configure the logging level.",
    )

    parser.add_argument(
        "-dy_sample",
        "--dy_sample",
        dest="dy_sample",
        default="MiNNLO",
        # choices=["MiNNLO", "aMCatNLO", "VBF_filter", "powheg", "amcatnloFXFX", "INCamcatnloFXFX"],
        action="store",
        help="For zpt reweighting, choose the type of DY samples to use",
    )
    parser.add_argument(
        "--vbf_filter_study",
        dest="do_vbf_filter_study",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable the DY vs DY-VBF-filter study selection.",
    )    
    return parser


def resolve_dataset_yaml_file(dataset_yaml_file: str, year: str, NanoAODv: int) -> str:
    """
    Resolve to the versioned dataset config files that actually exist in this repo.
    """
    candidate = Path(dataset_yaml_file)
    if candidate.exists():
        return str(candidate)

    run_period = "run3" if is_run3(year) else "run2"

    fallback = Path(f"configs/datasets/dataset_nanoAODv{NanoAODv}_{run_period}.yaml")
    if fallback.exists():
        return str(fallback)

    return dataset_yaml_file
