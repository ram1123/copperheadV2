import os
import ROOT
from scipy.stats import f
import logging

from modules.utils import logger

global_fit_xmax = 200

# define cut ranges to do polynomial fits. pt ranges beyond that point we fit with a constant
poly_fit_ranges = {
    "2018" : {
        "njet0" : [0, 30],
        "njet1" : [0, 30],
        "njet2" : [0, 20],
    },
    "2017" : {
        "njet0" : [0, 70],
        "njet1" : [0, 55],
        "njet2" : [0, 60],
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

# Function to perform polynomial fitting and F-test
def perform_f_test(hist_SF, fit_xmin, fit_xmax, target_nbins, save_path, year, njet):
    """
    Perform F-test to compare polynomial fits and determine the best polynomial order.
    """
    outTextFile = open(f"{save_path}/fTest_results_{year}_njet{njet}_NEW.txt", "w")
    optimized_orders = {} # Dictionary to store final orders per (year, njet, bins)
    txtfileOut = ""
    # save the hist_SF as png
    # canvas = ROOT.TCanvas("canvas", f"{target_nbins} bins SF hist", 800, 800)
    # hist_SF.SetTitle(f"njet {njet}, {target_nbins} bins SF")
    # hist_SF.SetLineColor(ROOT.kBlue)
    # hist_SF.SetMinimum(0.25)  # Set the lower bound of the Y-axis
    # hist_SF.SetMaximum(4)  # Set the upper bound of the Y-axis
    # hist_SF.Draw()
    # canvas.SaveAs(f"{save_path}/{year}_njet{njet}_{target_nbins}_SF_hist_DEBUG.png")
    # canvas.SaveAs(f"{save_path}/{year}_njet{njet}_{target_nbins}_SF_hist_DEBUG.pdf")
    logger.info(f"Starting F-test for {year}, njet{njet}, {target_nbins} bins.")
    # exit()
    for order in range(2, 11):
        order_low, order_high = order, order + 1

        # Fit with lower-order polynomial
        polynomial_expr = " + ".join([f"[{i}]*x**{i}" for i in range(order_low + 1)])
        # Define the TF1 function with the generated expression
        fit_func_low = ROOT.TF1(f"poly{order}", polynomial_expr, -5, 5) # FIXME: Range should be fit_xmin, fit_xmax
        _ = hist_SF.Fit(fit_func_low, "L ", xmin=fit_xmin, xmax=fit_xmax)
        _ = hist_SF.Fit(fit_func_low, "L ", xmin=fit_xmin, xmax=fit_xmax)
        fit_low = hist_SF.Fit(fit_func_low, "", xmin=fit_xmin, xmax=fit_xmax)
        chi2_low, ndf_low = fit_func_low.GetChisquare(), fit_func_low.GetNDF()

        # Fit with the higher-order polynomial
        polynomial_expr = " + ".join([f"[{i}]*x**{i}" for i in range(order_high + 1)])
        # Define the TF1 function with the generated expression
        fit_func_high = ROOT.TF1(f"poly{order}", polynomial_expr, -5, 5) # FIXME: Range
        _ = hist_SF.Fit(fit_func_high, "L ", xmin=fit_xmin, xmax=fit_xmax)
        _ = hist_SF.Fit(fit_func_high, "L ", xmin=fit_xmin, xmax=fit_xmax)
        fit_high = hist_SF.Fit(fit_func_high, "", xmin=fit_xmin, xmax=fit_xmax)
        chi2_high, ndf_high = fit_func_high.GetChisquare(), fit_func_high.GetNDF()

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
            txtfileOut = f"{year} njet{njet} {target_nbins} bins: Higher-order {order_high} polynomial significantly improves the fit over {order_low}. chi2_low: {chi2_low/ndf_low} vs chi2_high: {chi2_high/ndf_high}\n"
            outTextFile.write(f"{year} njet{njet} {target_nbins} bins: Higher-order {order_high} polynomial significantly improves the fit over {order_low}. chi2_low: {chi2_low/ndf_low} vs chi2_high: {chi2_high/ndf_high}\n")
            logger.error(f"{year} njet{njet} {target_nbins} bins: Higher-order {order_high} polynomial significantly improves the fit over {order_low}. chi2_low: {chi2_low/ndf_low} vs chi2_high: {chi2_high/ndf_high}\n")

            key = (year, njet, target_nbins)
            logger.debug(f"key: {key}")
            if key not in optimized_orders:
                logger.debug(f"Setting optimized order for {key} to {order_high}")
                optimized_orders[key] = order_high
            else:
                logger.debug("Updating optimized order")
                logger.debug(f"Old order: {optimized_orders[key]}")
                if optimized_orders[key] == order_low:
                    optimized_orders[key] = order_high
                logger.debug(f"New order: {optimized_orders[key]}")
            logger.debug(f"Optimized Orders: {optimized_orders}")

    outTextFile.close()
    return optimized_orders, txtfileOut

def run_goodness_of_fit(hist_SF, fit_xmin, fit_xmax, order, target_nbins, year, njet, save_path):
    """Run the goodness-of-fit test using the optimized polynomial order."""
    out_dict_by_nbin = {}
    out_dict_by_year = {}
    save_dict = {}


    # Fit with the lower-order polynomial
    polynomial_expr = " + ".join([f"[{i}]*x**{i}" for i in range(order + 1)])
    # Define the TF1 function with the generated expression
    fit_func = ROOT.TF1(f"poly{order}", polynomial_expr, -5, 5)     # FIXME: Range
    _ = hist_SF.Fit(fit_func, "L S", xmin=fit_xmin, xmax=fit_xmax)
    _ = hist_SF.Fit(fit_func, "L S", xmin=fit_xmin, xmax=fit_xmax)
    fit_results = hist_SF.Fit(fit_func, "L S", xmin=fit_xmin, xmax=fit_xmax)

    # fit straight line
    horizontal_line = ROOT.TF1("horizontal_line", "[0]", fit_xmax, global_fit_xmax)
    fit_results = hist_SF.Fit(horizontal_line, "S R+", xmin=fit_xmax, xmax=global_fit_xmax)

    chi2, ndf = fit_func.GetChisquare(), fit_func.GetNDF()
    chi2_dof = chi2/ndf
    p_value = ROOT.TMath.Prob(chi2, ndf)

    canvas = ROOT.TCanvas("canvas", f"{target_nbins} bins SF hist", 800, 800)
    canvas.Divide(1, 2)  # Split canvas into 2 rows
    canvas.cd(1)
    hist_SF.SetTitle(f"{order} order poly, njet {njet}, {target_nbins} bins SF")
    hist_SF.SetLineColor(ROOT.kBlue)
    hist_SF.SetMinimum(0.25)  # Set the lower bound of the Y-axis
    hist_SF.SetMaximum(4)  # Set the upper bound of the Y-axis
    hist_SF.Draw()
    fit_func.SetLineColor(ROOT.kRed)  # Change color for each fit
    fit_func.Draw("SAME")  # Draw the fit function on the same canvas
    horizontal_line.SetLineColor(ROOT.kGreen)  # Change color for each fit
    horizontal_line.Draw("SAME")  # Draw the fit function on the same canvas

    # Add a legend
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)  # Legend coordinates (x1, y1, x2, y2)
    legend.AddEntry(hist_SF, "hist_SF", "l")  # "l" means line style
    legend.AddEntry(fit_func, "poly fit", "l")
    legend.AddEntry(horizontal_line, "horizontal line fit", "l")
    legend.Draw()

    # Add a text box using TPaveText
    text_box = ROOT.TPaveText(0.0, 0.7, 0.4, 0.9, "NDC")  # NDC for normalized coordinates (0-1 range)
    text_box.SetFillColor(0)  # Transparent background
    text_box.SetBorderSize(1)  # Border thickness
    text_box.AddText("Fit Results:")
    text_box.AddText(f"chi2 / DOF = {float('%.4g' % chi2_dof)}")
    text_box.AddText(f"P value = {float('%.4g' % p_value)}")
    text_box.Draw()

    # Update the canvas
    canvas.Update()

    # Calculate the pull distribution
    hist = hist_SF
    pull_hist = ROOT.TH1D("pull_hist", "Pull Distribution", hist.GetNbinsX(), hist.GetXaxis().GetXmin(), hist.GetXaxis().GetXmax())
    for i in range(1, hist.GetNbinsX() + 1):
        data = hist.GetBinContent(i)
        error = hist.GetBinError(i)
        x_value = hist.GetBinCenter(i)
        if x_value <= fit_xmax :
            fit_value = fit_func.Eval(x_value)
        else:
            fit_value = horizontal_line.Eval(x_value)
        if error > 0:  # Avoid division by zero
            pull = (data - fit_value) / error
            pull_hist.SetBinContent(i, pull)

    # Draw the pull distribution on the bottom pad
    canvas.cd(2)
    ROOT.gPad.SetPad(0, 0, 1, 0.4)  # Bottom pad takes 40% of the canvas
    ROOT.gPad.SetGrid()
    pull_hist.SetMarkerStyle(20)
    pull_hist.SetTitle("Pull Distribution;X-axis;Pull")
    pull_hist.Draw("P")  # "P" for marker plot


    # Save the canvas
    canvas.SaveAs(f"{save_path}/{year}_njet{njet}_{target_nbins}_order{order}_goodnessOfFit.png")
    canvas.SaveAs(f"{save_path}/{year}_njet{njet}_{target_nbins}_order{order}_goodnessOfFit.pdf")

    # ---------------------------------------------------
    # save the fit coeffs
    # ---------------------------------------------------

    # first make a baseline dict
    # max_order = 9
    max_order = 5
    param_dict = {}
    for ix in range(max_order+1):
        param_dict[f"p{ix}"] = 0.0
        param_dict[f"p{ix}_err"] = 0.0

    num_params = fit_func.GetNpar()  # Number of parameters in the fit
    logger.debug("Fitted parameters and uncertainties:")
    for ix in range(num_params):
        param_value = fit_func.GetParameter(ix)  # Fitted parameter value
        param_error = fit_func.GetParError(ix)  # Fitted parameter uncertainty
        logger.debug(f"{year} njet {njet}  Parameter {i}: {param_value:} ± {param_error:}")
        logger.debug(f"{year} njet {njet}  Parameter {i} rel err: {param_error/ param_value *100} % ")
        param_dict[f"p{ix}"] = param_value
        param_dict[f"p{ix}_err"] = param_error

    # add straight_line constant poarameter
    param_dict[f"horizontal_c0"] = horizontal_line.GetParameter(0)
    # add xrange
    param_dict["polynomial_range"] = {
        "x_min" : fit_xmin,
        "x_max" : fit_xmax
    }

    out_dict_by_nbin[target_nbins] = param_dict
    out_dict_by_year[f"njet_{njet}"] = out_dict_by_nbin

    save_dict[year] = out_dict_by_year

    return chi2 / ndf, p_value, save_dict
