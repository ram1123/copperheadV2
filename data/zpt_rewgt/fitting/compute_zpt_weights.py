import ROOT
import os
import argparse
import logging
from omegaconf import OmegaConf

from modules.utils import logger
from zpt_fitting_utils import poly_fit_ranges
from zpt_fitting_utils import perform_f_test
from zpt_fitting_utils import run_goodness_of_fit

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--run_label", type=str, required=True, help="Run label")
parser.add_argument("--years", type=str, nargs="+", required=True, help="Year")
parser.add_argument("--njet", type=int, nargs="+", default=[0, 1, 2], help="Number of jets")
parser.add_argument("--input_path", type=str, required=True, help="Input path")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--outAppend", type=str, default="", help="Append to output file name")
parser.add_argument("--nbins", type=int, default=501, help="Number of bins")
args = parser.parse_args()

# Set logging level
logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

# Setup paths
inDirectory = f"./plots_WS_{args.run_label}{args.outAppend}"
save_path = f"{inDirectory}/GOF_{args.run_label}{args.outAppend}"
os.makedirs(save_path, exist_ok=True)

optimized_order = {}

for year in args.years:
    save_dict = {} # To store the final results of fitting
    for njet in args.njet:
        input_file = f"{inDirectory}/{year}_njet{njet}.root"
        if not os.path.exists(input_file):
            logger.error(f"File {input_file} not found!")
            exit(1)

        file = ROOT.TFile(input_file, "READ")
        workspace = file.Get("zpt_Workspace") # FIXME: Hardcoded workspace name
        if not workspace:
            logger.error(f"Workspace not found in {input_file}!")
            file.Close()
            exit(1)

        hist_SF = workspace.obj("hist_SF").Clone(f"hist_SF_clone_{year}_njet{njet}")
        # file.Close()

        # print content of the hist_SF
        # hist_SF.Print("all")

        fit_xmin, fit_xmax = poly_fit_ranges[year][f"njet{njet}"]
        outTextFile = open(f"{save_path}/fTest_results_{year}_njet{njet}.txt", "w")

        logger.info(f"Running F-test for {year} njet{njet} with {args.nbins} bins")
        order_info, txtfileOut = perform_f_test(hist_SF, fit_xmin, fit_xmax, args.nbins, save_path, year, njet)
        optimized_order.update(order_info)
        outTextFile.write(txtfileOut)

        logger.debug(f"Optimized Orders: {optimized_order}")
        logger.debug(f"key: {year}, {njet}, {args.nbins}")
        key = (year, njet, args.nbins)
        if key not in optimized_order:
            logger.warning(f"Optimized order not found for {key}")
            continue
        order = optimized_order[key]
        # continue

        chi2_dof, p_value, save_dict_local = run_goodness_of_fit(hist_SF, fit_xmin, fit_xmax, order, args.nbins, year, njet, save_path)
        save_dict.update(save_dict_local)

        logger.info(f"{year} njet{njet}: Optimized Order: {order}, Chi2/DOF: {chi2_dof:.3f}, P-Value: {p_value:.5f}")
        outTextFile.close()

    yaml_path = f"{save_path}/zpt_rewgt_params.yaml"
    if os.path.isfile(yaml_path): # if yaml exists, append to existing config (values with same keys will be overwirtten
        config = OmegaConf.load(yaml_path)
        config = OmegaConf.merge(config, save_dict)
    else:
        config = OmegaConf.create(save_dict)
    OmegaConf.save(config=config, f=yaml_path)

