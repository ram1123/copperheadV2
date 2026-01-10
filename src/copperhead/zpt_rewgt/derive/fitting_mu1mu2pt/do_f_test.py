"""
F-test
"""
import ROOT
from scipy.stats import f
import os
import argparse
import logging

from modules.utils import logger

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
        "njet0" : [0, 70],
        "njet1" : [0, 45],
        "njet2" : [0, 50],
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
# parser.add_argument("--input_path", type=str, help="Input path", required=True)
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--outAppend", type=str, default="", help="Append to output file name")
parser.add_argument("--nbins", type=str, default="CustomBins", help="Number of bins")
args = parser.parse_args()

# Set logging level
logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

year = args.years[0]
run_label = args.run_label
inDirectory = f"./plots_WS_{args.run_label}{args.outAppend}"
save_path = f"{inDirectory}/plots_fTest_{args.run_label}{args.outAppend}"
os.makedirs(save_path, exist_ok=True)

optimized_orders = {}  # Dictionary to store final orders per (year, njet, bins)

# Function to perform polynomial fitting and F-test
def perform_f_test(hist_SF, fit_xmin, fit_xmax, target_nbins, outTextFile, outTextFile_keys, year, njet):
    """
    Perform F-test to compare polynomial fits and determine the best polynomial order.
    """
    logger.info(f"Starting F-test for {year}, njet{njet}, {target_nbins} bins.")

    for order in range(2, 8):
        order_low, order_high = order, order + 1

        # Fit with lower-order polynomial
        polynomial_expr = " + ".join([f"[{i}]*x**{i}" for i in range(order_low + 1)])
        # Define the TF1 function with the generated expression
        fit_func_low = ROOT.TF1(f"poly{order}", polynomial_expr, -5, 5)
        _ = hist_SF.Fit(fit_func_low, "L ", xmin=fit_xmin, xmax=fit_xmax)
        _ = hist_SF.Fit(fit_func_low, "L ", xmin=fit_xmin, xmax=fit_xmax)
        fit_low = hist_SF.Fit(fit_func_low, "", xmin=fit_xmin, xmax=fit_xmax)
        chi2_low = fit_func_low.GetChisquare()
        ndf_low = fit_func_low.GetNDF()

        # Fit with the higher-order polynomial
        polynomial_expr = " + ".join([f"[{i}]*x**{i}" for i in range(order_high + 1)])
        # Define the TF1 function with the generated expression
        fit_func_high = ROOT.TF1(f"poly{order}", polynomial_expr, -5, 5)
        _ = hist_SF.Fit(fit_func_high, "L ", xmin=fit_xmin, xmax=fit_xmax)
        _ = hist_SF.Fit(fit_func_high, "L ", xmin=fit_xmin, xmax=fit_xmax)
        fit_high = hist_SF.Fit(fit_func_high, "", xmin=fit_xmin, xmax=fit_xmax)

        # fit_high = hist_SF.Fit(fit_func_high, "SQ0", "", fit_xmin, fit_xmax)
        chi2_high = fit_func_high.GetChisquare()
        ndf_high = fit_func_high.GetNDF()

        # Compute F-statistic and p-value
        delta_chi2 = chi2_low - chi2_high
        delta_dof = -(ndf_high - ndf_low) # Negative sign because the order_high is greater than order_low
        if delta_dof == 0 or chi2_high == 0:
            f_statistic = 0
        else:
            f_statistic = (delta_chi2 / chi2_high) * (ndf_high / delta_dof)
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
            # print(f"Key: {key}")
            # print(f"Optimized orders: {optimized_orders}, {order_low}, {order_high}")

            # If not in dictionary, set initial order
            if key not in optimized_orders:
                # print("Setting initial order")
                optimized_orders[key] = order_high
            else:
                # print("Updating order")
                # print(f"Current order: {optimized_orders[key]}")
                # Ensure the sequence is continuous (e.g., 6->7->8 but not skipping 8->9)
                if optimized_orders[key] == order_low:
                    optimized_orders[key] = order_high

            # break  # Stop increasing order if we found a significant improvement


for njet in args.njet:
    outTextFile = open(f"{inDirectory}/fTest_results_{year}_njet{njet}_nbins{args.nbins}_UpdatedCode.txt", "w")
    outTextFile_keys = open(f"{inDirectory}/fTest_results_{year}_njet{njet}_nbins{args.nbins}_keys.txt", "w")
    # input_file = f"{inDirectory}/{year}_njet{njet}.root"
    # file = ROOT.TFile(f"{year}_njet{njet}.root", "READ")
    # input_file = f"{inDirectory}/{year}_njet{njet}_nbins{args.nbins}.root"
    input_file = f"{year}_njet{njet}.root"
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

    for target_nbins in [args.nbins]: # INFO: What is the purpose of this target_nbins?
        # hist_data = workspace.obj("hist_data"+"_njet"+str(njet)+"_nbins"+str(args.nbins)).Clone("hist_data_clone")
        # hist_dy = workspace.obj("hist_dy"+"_njet"+str(njet)+"_nbins"+str(args.nbins)).Clone("hist_dy_clone")
        hist_data = workspace.obj("hist_data").Clone("hist_data_clone")
        hist_dy = workspace.obj("hist_dy").Clone("hist_dy_clone")
        orig_nbins = hist_data.GetNbinsX()
        rebin_coeff = int(int(orig_nbins)/int(target_nbins))

        # Rebin histograms
        # logger.debug(f"rebin_coeff: {rebin_coeff}")
        # hist_data_rebinned = hist_data.Rebin(rebin_coeff, f"hist_data_rebinned_{target_nbins}")
        # hist_dy_rebinned = hist_dy.Rebin(rebin_coeff, f"hist_dy_rebinned_{target_nbins}")
        hist_data = hist_data.Rebin(rebin_coeff, "rebinned hist_data")
        hist_dy = hist_dy.Rebin(rebin_coeff, "rebinned hist_dy")

        # Compute Scale Factor (SF)
        # hist_SF = hist_data_rebinned.Clone(f"hist_SF_{target_nbins}")
        # hist_SF.Divide(hist_dy_rebinned)
        hist_SF = hist_data.Clone("hist_SF")
        hist_SF.Divide(hist_dy)

        # Run F-test
        perform_f_test(hist_SF, fit_xmin, fit_xmax, target_nbins, outTextFile, outTextFile_keys, year, njet)

        # Save plots
        canvas = ROOT.TCanvas(f"canvas_{target_nbins}", f"SF Histogram {year} njet{njet}", 800, 600)
        hist_data.SetLineColor(ROOT.kRed)
        hist_dy.SetLineColor(ROOT.kBlue)
        # Change the plot title
        hist_data.SetTitle(f"njet {njet} {target_nbins} bins Data and DY")
        hist_data.Draw()

        hist_dy.Draw("SAME")
        # Add a legend
        legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)  # Legend coordinates (x1, y1, x2, y2)
        legend.AddEntry(hist_data, "Data", "l")  # "l" means line style
        legend.AddEntry(hist_dy, "DY", "l")
        legend.Draw()
        canvas.SetLogy(1)
        canvas.Update()
        canvas.SaveAs(f"{save_path}/dataDy_{year}_njet{njet}_{target_nbins}.png")
        canvas.SaveAs(f"{save_path}/dataDy_{year}_njet{njet}_{target_nbins}.pdf")

        # Plot SF histogram
        canvas.Clear()
        canvas.SetLogy(0)

        hist_SF.SetTitle(f"Scale Factor (SF) Histogram - {year} njet{njet} ({target_nbins} bins)")
        hist_SF.SetMinimum(0.5)  # Set the lower bound of the Y-axis
        hist_SF.SetMaximum(4)  # Set the upper bound of the Y-axis
        hist_SF.Draw()
        canvas.Draw()
        canvas.SaveAs(f"{save_path}/SF_{year}_njet{njet}_{target_nbins}.png")

    file.Close()
    outTextFile.close()
    outTextFile_keys.close()

# Print results
for (year, njet, bins), order in optimized_orders.items():
    logger.info(f"Optimized order for {year} njet{njet} {bins} bins: {order}")

