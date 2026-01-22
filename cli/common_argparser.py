"""
Common ArgumentParser for copperhead framework.

This function defines the full, shared CLI.
"""
import argparse
import logging


def build_common_parser() -> argparse.ArgumentParser:
    """Build and return the common ArgumentParser used by copperhead scripts."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-y",
        "--year",
        dest="year",
        default="2018",
        action="store",
        help="year value. The options are: 2016preVFP, 2016postVFP, 2017, 2018",
    )

    parser.add_argument(
        "--yaml",
        dest="dataset_yaml_file",
        default="configs/datasets/dataset.yaml",
        help="Path of yaml file containing the dataset names",
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
        default=[],
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
        default=[],
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
        "-save",
        "--save_path",
        dest="save_path",
        default=None,
        action="store",
        help="save path to store stage1 output files",
    )

    parser.add_argument(
        "-aod_v",
        "--NanoAODv",
        type=int,
        dest="NanoAODv",
        default=9,
        choices=[9, 12, 15],
        help="version number of NanoAOD samples we're working with. currently, only 9 and 12 are supported",
    )

    parser.add_argument(
        "--log-level",
        default=logging.ERROR,
        type=lambda x: getattr(logging, x),
        help="Configure the logging level.",
    )

    return parser
