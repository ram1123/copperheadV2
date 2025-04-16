"""
Introduction:

Commands:
python validation/zpt_rewgt/validation.py -y 2017 --label "March25_NanoAODv9_WithUpdatedZptWgt" --in /depot/cms/users/shar1172/hmm/copperheadV1clean/March25_NanoAODv9_WithUpdatedZptWgt
python validation/zpt_rewgt/validation.py -y 2016preVFP --label "March25_NanoAODv9_WithUpdatedZptWgt" --in /depot/cms/users/shar1172/hmm/copperheadV1clean/March25_NanoAODv9_WithUpdatedZptWgt
python validation/zpt_rewgt/validation.py -y 2018 --label "April09_NanoV12" --in /depot/cms/users/shar1172/hmm/copperheadV1clean/April09_NanoV12 --use_gateway

"""

import os
import sys
import argparse

import logging
from modules.utils import logger
from modules.utils import ifPathExists

def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-y", "--year",
        dest="year",
        default="2017",
        action="store",
        help="year value. The options are: 2016preVFP, 2016postVFP, 2017, 2018"
    )

    parser.add_argument(
        "--label",
        dest="inLable",
        help="path of input directory containing Data, Bkg and signal"
    )

    parser.add_argument(
        "--in",
        dest="InFilePath",
        help="path of input directory containing Data, Bkg and signal"
    )

    parser.add_argument(
        "-data", "--data",
        dest="data_samples",
        default=['A', 'B', 'C', 'D','E', 'F','H', 'G'],
        nargs="*",
        type=str,
        action="store",
        help="list of data samples represented by alphabetical letters A-H"
    )

    parser.add_argument(
        "-bkg", "--background",
        dest="bkg_samples",
        default=['DY','TT','ST','VV','EWK', 'OTHER'],
        nargs="*",
        type=str,
        action="store",
        help="list of bkg samples represented by shorthands: DY, TT, ST, DB (diboson), EWK"
    )

    parser.add_argument(
        "-sig", "--signal",
        dest="sig_samples",
        default=[],
        # default=['ggH', 'VBF'],
        nargs="*",
        type=str,
        action="store",
        help="list of sig samples represented by shorthands: ggH, VBF"
    )

    parser.add_argument(
        "--use_gateway",
        dest="use_gateway",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, uses dask gateway client instead of local"
    )

    parser.add_argument(
        "--xcache",
        dest="xcache",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, uses xcache root file paths"
    )

    parser.add_argument(
        "-aod_v", "--NanoAODv",
        type=int,
        dest="NanoAODv",
        default=9,
        choices=[9, 12],
        help="version number of NanoAOD samples we're working with. currently, only 9 and 12 are supported"
    )

    parser.add_argument(
        "--run2_rereco",
        dest="run2_rereco",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, uses skips bad files when calling preprocessing"
    )

    parser.add_argument(
        "--log-level",
        default=logging.INFO,
        type=lambda x: getattr(logging, x),
        help="Configure the logging level."
    )

    parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, enables debug mode"
    )

    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()

    logger.setLevel(args.log_level)
    data_l = args.data_samples
    bkg_l = args.bkg_samples
    sig_l = args.sig_samples
    label = args.inLable
    year = args.year

    # vars2plot = ['dimuon', 'mu', 'jet', 'dijet']
    vars2plot = ['dimuon', 'mu', 'jet']
    # vars2plot = ['dimuon']
    # vars2plot = ['mu', 'jet', 'dijet']

    lumi_dict = {
        "2018" : 59.83,
        "2017" : 41.48,
        "2016postVFP": 19.5,
        "2016preVFP": 16.81,
        "2022preEE" : 8.000,
    }
    lumi = lumi_dict[year]

    status = "Private_Work"
    # region = "z-peak"
    region = "signal"
    load_path = os.path.join(args.InFilePath, f"stage1_output/{year}/f1_0/")
    ifPathExists(load_path)

    plot_setting = "./validation/zpt_rewgt/plot_settings_Zpt_reWgt.json"
    ifPathExists(plot_setting)
    # if not os.path.exists(load_path):
        # logger.error(f"Path: {load_path} does not exits")
        # sys.exit()
    # else:
        # logger.info(f"Path of ntuples: {load_path}")


    # if not os.path.exists(plot_setting):
    #     logger.error(f"File: {plot_setting} does not exits.")
    #     sys.exit()
    # else:
    #     logger.debug(f"Plot configuration file, {plot_setting}, exists.")

    keep_zpt_on = True # dummy value


    vars2plot = ' '.join(vars2plot)
    data_l = ' '.join(data_l)
    bkg_l = ' '.join(bkg_l)
    sig_l = ' '.join(sig_l)

    # data_l = ['A', 'B', 'C', 'D']

    logger.info(f"data: {data_l}")
    logger.info(f"background: {bkg_l}")
    logger.info(f"signal: {sig_l}")

    njets = [-1, 0, 1, 2]
    categories = ["vbf", "ggh", "nocat"]
    WithZpT = ["yes_zpt", "no_zpt"]

    if args.debug:
        logger.warning("Running in DEBUG mode")
        njets = [-1]
        categories = ["nocat"]
        WithZpT = ["yes_zpt"]

    logger.debug(f"nJets: {njets}")
    logger.debug(f"categories: {categories}")
    logger.debug(f"WithZpT: {WithZpT}")
    # sys.exit()

    for njet in njets:
        for cat in categories:
            if cat == "vbf" and njet < 2:
                continue
            for zpt_name in WithZpT:
                command = f"python validation/zpt_rewgt/zpt_validation_plotter.py -y {year} --load_path {load_path}  -var {vars2plot} --data {data_l} --background {bkg_l} --signal {sig_l} --lumi {lumi} --status {status} -cat {cat} -reg {region} --label {label} --plot_setting {plot_setting} --jet_multiplicity {njet} --zpt_wgt_name {zpt_name}"
                if args.use_gateway:
                    command += " --use_gateway"
                logger.info(command)
                os.system(command)


if __name__ == "__main__":
    main()
