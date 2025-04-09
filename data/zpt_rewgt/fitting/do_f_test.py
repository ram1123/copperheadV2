import ROOT
from scipy.stats import f
import os
import argparse
import logging
import yaml
from collections import defaultdict
from modules.utils import logger
from omegaconf import OmegaConf


# define cut ranges to do polynomial fits. pt ranges beyond that point we fit with a constant
poly_fit_ranges = {
    "2018" : {
        "njet0" : [0, 85],
        "njet1" : [0, 50],
        "njet2" : [0, 50],
    },
    "2017" : {
        "njet0" : [0, 75],
        "njet1" : [0, 100],
        "njet2" : [0, 65],
    },
    "2016postVFP" : {
        "njet0" : [0, 50],
        "njet1" : [0, 80],
        "njet2" : [0, 30],
    },
    "2016preVFP" : {
        "njet0" : [0, 70],
        "njet1" : [0, 55],
        "njet2" : [0, 55],
    },
}

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--run_label", type=str, help="Run label", required=True)
parser.add_argument("--years", type=str, nargs="+", help="Year", required=True)
parser.add_argument("--njet", type=int, nargs="+", default=[0, 1, 2], help="Number of jets")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--outAppend", type=str, default="", help="Append to output file name")
parser.add_argument("--nbins", type=str, default="CustomBins", help="Number of bins")
parser.add_argument("-save", "--plot_path", dest="plot_path", default="plots", action="store", help="save path to store plots")
args = parser.parse_args()

logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

year = args.years[0]
run_label = args.run_label
inPath = f"{args.plot_path}/{run_label}/{year}"
save_path = f"{args.plot_path}/{run_label}/{year}/fTest_{args.outAppend}"
os.makedirs(save_path, exist_ok=True)

optimized_orders = {}
save_fit_config = defaultdict(dict)  # For YAML output

def perform_f_test(hist_SF, fit_xmin, fit_xmax, target_nbins, outTextFile, outTextFile_keys, year, njet):
    for order in range(2, 8):
        order_low, order_high = order, order + 1

        polynomial_expr_low = " + ".join([f"[{i}]*x**{i}" for i in range(order_low + 1)])
        fit_func_low = ROOT.TF1(f"poly{order}", polynomial_expr_low, 0, fit_xmax)
        _ = hist_SF.Fit(fit_func_low, "L", xmin=fit_xmin, xmax=fit_xmax)
        _ = hist_SF.Fit(fit_func_low, "L", xmin=fit_xmin, xmax=fit_xmax)
        fit_low = hist_SF.Fit(fit_func_low, "", xmin=fit_xmin, xmax=fit_xmax)
        chi2_low = fit_func_low.GetChisquare()
        ndf_low = fit_func_low.GetNDF()

        polynomial_expr_high = " + ".join([f"[{i}]*x**{i}" for i in range(order_high + 1)])
        fit_func_high = ROOT.TF1(f"poly{order}", polynomial_expr_high, 0, fit_xmax)
        _ = hist_SF.Fit(fit_func_high, "L", xmin=fit_xmin, xmax=fit_xmax)
        _ = hist_SF.Fit(fit_func_high, "L", xmin=fit_xmin, xmax=fit_xmax)
        fit_high = hist_SF.Fit(fit_func_high, "", xmin=fit_xmin, xmax=fit_xmax)

        chi2_high = fit_func_high.GetChisquare()
        ndf_high = fit_func_high.GetNDF()

        delta_chi2 = chi2_low - chi2_high
        delta_dof = -(ndf_high - ndf_low) # Negative sign because the order_high is greater than order_low
        f_statistic = (delta_chi2 / chi2_high) * (ndf_high / delta_dof) if delta_dof != 0 and chi2_high != 0 else 0
        p_value = 1 - f.cdf(f_statistic, delta_dof, ndf_high)

        # Log results
        if ndf_low == 0 or ndf_high == 0:
            logger.error("NDF is zero!")
            logger.debug(f"Order {order_low}: χ² = {chi2_low:.2f}, NDF = {ndf_low}")
            logger.debug(f"Order {order_high}: χ² = {chi2_high:.2f}, NDF = {ndf_high}")
        else:
            logger.debug(f"Order {order_low}: χ² = {chi2_low:.2f}, NDF = {ndf_low}, χ²/NDF = {chi2_low/ndf_low:.3f}")
            logger.debug(f"Order {order_high}: χ² = {chi2_high:.2f}, NDF = {ndf_high}, χ²/NDF = {chi2_high/ndf_high:.3f}")
        logger.debug(f"F-statistic: {f_statistic:.3f}, p-value: {p_value:.5f}")

        # Decision based on p-value
        if p_value < 0.05:
            logger.info(f"Significant improvement with polynomial order {order_high} over {order_low}.")
            outTextFile.write(f"{year} njet{njet} {target_nbins} bins: Higher-order {order_high} polynomial significantly improves the fit over {order_low}. chi2_low: {chi2_low/ndf_low} vs chi2_high: {chi2_high/ndf_high}\n")
            outTextFile_keys.write(f"{year} {njet} {target_nbins} {order_high} {order_low}\n")

            key = (year, njet, target_nbins)
            if key not in optimized_orders:
                optimized_orders[key] = order_high
                save_fit_config[year][f"njet{njet}"] = {
                    "order": order_high,
                    "fit_range": [fit_xmin, fit_xmax]
                }
              # Save the fit plot with the best function
                canvas = ROOT.TCanvas("canvas", f"Fit SF {year} njet{njet}", 800, 600)
                hist_SF.SetLineColor(ROOT.kBlue)
                hist_SF.SetMinimum(0.25)
                hist_SF.SetMaximum(4)
                hist_SF.Draw()
                fit_func_high.SetLineColor(ROOT.kRed)
                fit_func_high.Draw("SAME")

                # Add formula text
                formula_string = f"Polynomial (order {order_high}): " #+ polynomial_expr_high
                text_box_formula = ROOT.TPaveText(0.15, 0.78, 0.85, 0.88, "NDC")
                text_box_formula.SetFillColor(0)
                text_box_formula.SetTextAlign(12)
                text_box_formula.SetTextFont(42)
                text_box_formula.SetTextSize(0.03)
                text_box_formula.AddText(formula_string)
                text_box_formula.Draw()

                canvas.SaveAs(f"{save_path}/fit_best_{year}_njet{njet}_{target_nbins}.pdf")

            elif optimized_orders[key] == order_low:
                optimized_orders[key] = order_high

for njet in args.njet:
    input_file = f"{inPath}/{year}_njet{njet}.root"
    logger.info(f"Processing {input_file}")

    if not os.path.exists(input_file):
        logger.error(f"File {input_file} not found!")
        exit()

    file = ROOT.TFile(input_file, "READ")
    workspace = file.Get("zpt_Workspace")
    if not workspace:
        logger.error(f"Workspace not found in {input_file}!")
        exit()

    fit_xmin, fit_xmax = poly_fit_ranges[year][f"njet{njet}"]

    for target_nbins in [args.nbins]:
        hist_data = workspace.obj("hist_data").Clone("hist_data_clone")
        hist_dy = workspace.obj("hist_dy").Clone("hist_dy_clone")
        orig_nbins = hist_data.GetNbinsX()
        rebin_coeff = int(int(orig_nbins)/int(target_nbins))
        # Rebin histograms
        hist_data = hist_data.Rebin(rebin_coeff, "rebinned hist_data")
        hist_dy = hist_dy.Rebin(rebin_coeff, "rebinned hist_dy")

        # Compute Scale Factor (SF)
        hist_SF = hist_data.Clone("hist_SF")
        hist_SF.Divide(hist_dy)

        # Run F-test
        outTextFile = open(f"{save_path}/fTest_results_{year}_njet{njet}_nbins{args.nbins}_UpdatedCode.txt", "w")
        outTextFile_keys = open(f"{save_path}/fTest_results_{year}_njet{njet}_nbins{args.nbins}_keys.txt", "w")
        perform_f_test(hist_SF, fit_xmin, fit_xmax, target_nbins, outTextFile, outTextFile_keys, year, njet)
        outTextFile.close()
        outTextFile_keys.close()
        file.Close()

# Save config to YAML
yaml_path = f"{save_path}/zpt_fit_config.yaml"

# Load existing config if it exists
if os.path.isfile(yaml_path):
    config = OmegaConf.load(yaml_path)
    config = OmegaConf.merge(config, dict(save_fit_config))
else:
    config = OmegaConf.create(dict(save_fit_config))
OmegaConf.save(config=config, f=yaml_path)

logger.info("Saved optimized fit config to zpt_fit_config.yaml")
