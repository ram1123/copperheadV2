import array
import os

import ROOT
import yaml
from cli.common_argparser import build_common_parser
from modules.utils import logger
from omegaconf import OmegaConf

# Run in batch mode and disable statistics box
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)


def parse_arguments():
    parser = build_common_parser()
    parser.add_argument(
        "--njet", type=int, nargs="+", default=[0, 1, 2],
        help="Jet multiplicities to loop over"
    )
    return parser.parse_args()

def eval_polynomial(coeffs, xval):
    return sum(coeff * (xval ** idx) for idx, coeff in enumerate(coeffs))


def make_combined_function_reduced(f0_coeffs, f1_coeffs, xmin, xmax, base_tail_slope):
    """
    Builds a reduced-parameter piecewise function:
    - frozen low-range polynomial, plus a common vertical shift
    - frozen mid-range polynomial, plus a small linear tilt around xmin
    - linear tail beyond xmax, with only the slope adjusted in the final refit

    Free parameters:
    - par[0]: common vertical shift applied to low/mid regions
    - par[1]: extra mid-range tilt multiplying (x - xmin)
    - par[2]: delta on the tail slope relative to the local tail fit
    """
    def func(x, par):
        xx = x[0]
        common_shift = par[0]
        mid_tilt = par[1]
        tail_slope = base_tail_slope + par[2]

        low_xmin = eval_polynomial(f0_coeffs, xmin) + common_shift
        mid_xmin_raw = eval_polynomial(f1_coeffs, xmin)
        mid_shift = low_xmin - mid_xmin_raw

        def eval_mid(xmid):
            return eval_polynomial(f1_coeffs, xmid) + mid_shift + mid_tilt * (xmid - xmin)

        if xx < 0.0:
            return 0.0
        elif xx <= xmin:
            return eval_polynomial(f0_coeffs, xx) + common_shift
        elif xx < xmax:
            return eval_mid(xx)
        else:
            y_at_xmax = eval_mid(xmax)
            tail_intercept = y_at_xmax - tail_slope * xmax
            return tail_slope * xx + tail_intercept

    return func

def rebin_histogram(hist, edges):
    """
    Rebins a TH1 histogram into variable-width bins defined by 'edges'.
    Returns the rebinned clone.
    """
    nbins = len(edges) - 1
    xbins = array.array('d', edges)
    name = hist.GetName() + f"_rebinned_{nbins}"
    rebinned = hist.Rebin(nbins, name, xbins)
    return rebinned


def make_confidence_band(hist_sf, fit_result, confidence_level, name):
    band = hist_sf.Clone(name)
    band.SetDirectory(0)
    band.Reset("ICESM")
    ROOT.TVirtualFitter.GetFitter().GetConfidenceIntervals(band, confidence_level)
    return band

def fit_polynomial(hist_sf, order, xmin, xmax, fit_opts="L S Q"):
    """
    Fits a polynomial of degree 'order' to hist_sf between [xmin, xmax].
    Returns the TF1 polynomial object.
    """
    expr = " + ".join(f"[{i}]*x**{i}" for i in range(order + 1))
    func = ROOT.TF1(f"poly{order}", expr, xmin, xmax)
    hist_sf.Fit(func, fit_opts, "", xmin, xmax)
    hist_sf.Fit(func, fit_opts, "", xmin, xmax)
    hist_sf.Fit(func, "L S R", "", xmin, xmax)
    return func

def fit_flat_line(hist_sf, xmin, xmax, fit_opts="L I S R"):
    """
    Fits a constant line to hist_sf between [xmin, xmax].
    Returns the TF1 object for that line.
    """
    func = ROOT.TF1("flat_line", "[0]*x + [1]", xmin, xmax)
    hist_sf.Fit(func, fit_opts, "", xmin, xmax)
    return func


def build_final_piecewise_coefficients(f0, order0, f1, order1, f_flat, f_comb, xmin1, xmax1):
    """
    Convert the reduced-parameter combined refit back into the full set of
    piecewise coefficients expected by stage1.
    """
    common_shift = f_comb.GetParameter(0)
    common_shift_err = f_comb.GetParError(0)
    mid_tilt = f_comb.GetParameter(1)
    mid_tilt_err = f_comb.GetParError(1)
    delta_tail_slope = f_comb.GetParameter(2)
    delta_tail_slope_err = f_comb.GetParError(2)

    f0_coeffs = [f0.GetParameter(i) for i in range(order0 + 1)]
    f0_errors = [f0.GetParError(i) for i in range(order0 + 1)]
    f1_coeffs = [f1.GetParameter(i) for i in range(order1 + 1)]
    f1_errors = [f1.GetParError(i) for i in range(order1 + 1)]

    final_f0_coeffs = list(f0_coeffs)
    final_f0_errors = list(f0_errors)
    final_f0_coeffs[0] += common_shift
    final_f0_errors[0] = (final_f0_errors[0] ** 2 + common_shift_err ** 2) ** 0.5

    low_xmin_nominal = eval_polynomial(f0_coeffs, xmin1)
    mid_xmin_nominal = eval_polynomial(f1_coeffs, xmin1)
    continuity_shift = low_xmin_nominal - mid_xmin_nominal + common_shift

    final_f1_coeffs = list(f1_coeffs)
    final_f1_errors = list(f1_errors)
    final_f1_coeffs[0] += continuity_shift - mid_tilt * xmin1
    final_f1_coeffs[1] += mid_tilt
    final_f1_errors[0] = (final_f1_errors[0] ** 2 + common_shift_err ** 2 + (xmin1 * mid_tilt_err) ** 2) ** 0.5
    final_f1_errors[1] = (final_f1_errors[1] ** 2 + mid_tilt_err ** 2) ** 0.5

    final_tail_slope = f_flat.GetParameter(0) + delta_tail_slope
    final_tail_slope_err = (f_flat.GetParError(0) ** 2 + delta_tail_slope_err ** 2) ** 0.5
    y_at_xmax = eval_polynomial(final_f1_coeffs, xmax1)
    tail_intercept = y_at_xmax - final_tail_slope * xmax1
    tail_intercept_err = (
        f_flat.GetParError(1) ** 2
        + common_shift_err ** 2
        + ((xmax1 - xmin1) * mid_tilt_err) ** 2
        + (xmax1 * delta_tail_slope_err) ** 2
    ) ** 0.5

    return {
        "f0_coeffs": final_f0_coeffs,
        "f0_errors": final_f0_errors,
        "f1_coeffs": final_f1_coeffs,
        "f1_errors": final_f1_errors,
        "tail_slope": final_tail_slope,
        "tail_slope_err": final_tail_slope_err,
        "tail_intercept": tail_intercept,
        "tail_intercept_err": tail_intercept_err,
        "common_shift": common_shift,
        "mid_tilt": mid_tilt,
        "delta_tail_slope": delta_tail_slope,
    }

def perform_fits(hist_sf, order0, xmin0, xmax0, order1, xmin1, xmax1, global_xmax):
    """
    Runs the three-step fits: 1) poly(order0) on [0, xmax0], 2) poly(order1) on [xmin1, xmax1],
    3) flat line on [xmax1, global_xmax]. Then creates and fits the combined TF1 over [0, global_xmax].
    Returns all TF1s: (f0, f1, f_flat, f_combined).
    """
    logger.info(f"Performing piecewise fits with orders {order0} and {order1}")

    # 1) Low-range fit
    logger.debug(f"Fitting low range: 0 to {xmax0} with order {order0}")
    f0 = fit_polynomial(hist_sf, order0, 0.0, xmax0, fit_opts="L S Q")

    # 2) Mid-range fit
    logger.debug(f"Fitting mid range: {xmin1} to {xmax1} with order {order1}")
    f1 = fit_polynomial(hist_sf, order1, xmin1, xmax1, fit_opts="L I S Q")

    # 3) High-range flat fit
    logger.debug(f"Fitting high range: {xmax1} to {global_xmax} with flat line")
    f_flat = fit_flat_line(hist_sf, xmax1, global_xmax, fit_opts="L I S R")
    # f_flat = fit_polynomial(hist_sf, order1, xmax1, global_xmax, fit_opts="L I S R")

    # Build reduced-parameter combined TF1 using the stable local fits as anchors.
    f0_coeffs = [f0.GetParameter(i) for i in range(order0 + 1)]
    f1_coeffs = [f1.GetParameter(i) for i in range(order1 + 1)]
    base_tail_slope = f_flat.GetParameter(0)
    logger.debug("Creating reduced-parameter combined function with 3 parameters")

    comb_func = make_combined_function_reduced(
        f0_coeffs=f0_coeffs,
        f1_coeffs=f1_coeffs,
        xmin=xmin1,
        xmax=xmax1,
        base_tail_slope=base_tail_slope,
    )
    logger.debug("Prepared reduced-parameter combined function for fitting")

    f_combined = ROOT.TF1("f_combined", comb_func, 0.0, global_xmax, 3)
    f_combined.SetParName(0, "common_shift")
    f_combined.SetParName(1, "mid_tilt")
    f_combined.SetParName(2, "delta_tail_slope")
    f_combined.SetParameter(0, 0.0)
    f_combined.SetParameter(1, 0.0)
    f_combined.SetParameter(2, 0.0)
    f_combined.SetParLimits(0, -0.5, 0.5)
    f_combined.SetParLimits(1, -0.02, 0.02)
    f_combined.SetParLimits(2, -0.02, 0.02)

    # Perform final reduced refit
    final_fit = hist_sf.Fit(f_combined, "L I S R", "", 0.0, global_xmax)
    final_fit = hist_sf.Fit(f_combined, "L I S R", "", 0.0, global_xmax)
    final_fit = hist_sf.Fit(f_combined, "L I S R", "", 0.0, global_xmax)
    logger.debug(f"Final fit result: {final_fit}")

    return f0, f1, f_flat, f_combined, final_fit

def plot_sf_and_pulls(hist_sf, f0, f1, f_flat, f_combined, fit_result,
                      xmin0, xmax0, xmin1, xmax1, global_xmax,
                      year, njet, nbins, save_dir):
    """
    Creates a two-panel canvas: upper panel shows SF vs x with all fit lines, lower panel shows pull distribution.
    Saves .pdf, .png, and .root in save_dir.
    """
    # Compute chi2/ndf and p-value for the mid-range fit f1
    chi2 = f1.GetChisquare()
    ndf = f1.GetNDF() if f1.GetNDF() > 0 else 1
    chi2ndf = chi2 / ndf
    pval = ROOT.TMath.Prob(chi2, ndf)

    # Set up canvas
    canv = ROOT.TCanvas(f"c_{year}_nj{njet}", "SF & Pulls", 800, 800)
    canv.Divide(1, 2)

    # --- Upper pad: SF histogram and fits ---
    canv.cd(1)
    # Force X-axis range from 0 to global_xmax and draw full axis
    hist_sf.GetXaxis().SetRangeUser(0.0, global_xmax)
    hist_sf.SetTitle(f"Year {year}, njet={njet}, bins={nbins}")
    hist_sf.SetLineColor(ROOT.kBlue)
    # Draw only the axis first to fix the range
    hist_sf.Draw("axis")

    band95 = None
    band68 = None
    if fit_result and int(fit_result.Status()) == 0:
        band95 = make_confidence_band(hist_sf, fit_result, 0.95, f"band95_{year}_{njet}")
        band95.SetFillColorAlpha(ROOT.kAzure - 9, 0.35)
        band95.SetLineColor(ROOT.kAzure - 9)
        band95.SetLineWidth(0)
        band95.SetMarkerSize(0)

        band68 = make_confidence_band(hist_sf, fit_result, 0.68, f"band68_{year}_{njet}")
        band68.SetFillColorAlpha(ROOT.kOrange - 2, 0.45)
        band68.SetLineColor(ROOT.kOrange - 2)
        band68.SetLineWidth(0)
        band68.SetMarkerSize(0)

    # Draw the fit function across the full x-range
    f_combined.SetRange(0.0, global_xmax)
    f_combined.SetNpx(5000)   # or 10000 if you want it super smooth
    f_combined.SetLineColor(ROOT.kRed)

    # Finally draw the histogram and the combined fit
    hist_sf.GetListOfFunctions().Clear()  # remove attached
    hist_sf.Draw("axis")
    if band95:
        band95.Draw("E3 SAME")
    if band68:
        band68.Draw("E3 SAME")
    hist_sf.Draw("same E")
    f_combined.Draw("SAME")
    ROOT.gPad.Update()


    txt = ROOT.TPaveText(0.4, 0.7, 0.7, 0.9, "NDC")
    # Legend
    if year == "2018":
        if njet == 0:
            leg = ROOT.TLegend(0.0, 0.7, 0.4, 0.9)
        elif njet == 1:
            leg = ROOT.TLegend(0.7, 0.1, 0.9, 0.3)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
        else:
            leg = ROOT.TLegend(0.7, 0.1, 0.9, 0.3)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
    elif year == "2017":
        if njet == 0:
            leg = ROOT.TLegend(0.7, 0.1, 0.9, 0.3)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
        elif njet == 1:
            leg = ROOT.TLegend(0.7, 0.1, 0.9, 0.3)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
        else:
            leg = ROOT.TLegend(0.7, 0.1, 0.9, 0.3)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
    elif year == "2016postVFP":
        if njet == 0:
            leg = ROOT.TLegend(0.0, 0.7, 0.4, 0.9)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
        elif njet == 1:
            leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
        else:
            leg = ROOT.TLegend(0.7, 0.1, 0.9, 0.3)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
    elif year == "2016preVFP":
        if njet == 0:
            leg = ROOT.TLegend(0.0, 0.7, 0.4, 0.9)
        elif njet == 1:
            leg = ROOT.TLegend(0.7, 0.1, 0.9, 0.3)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
        else:
            leg = ROOT.TLegend(0.7, 0.1, 0.9, 0.3)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
    elif year == "2022preEE":
        if njet == 2 or njet == 1:
            leg = ROOT.TLegend(0.7, 0.1, 0.9, 0.3)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
        else:
            leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
    elif year == "2022postEE":
        if njet == 2 or njet == 1:
            leg = ROOT.TLegend(0.7, 0.1, 0.9, 0.3)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
        else:
            leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
    elif year == "2023":
        if njet == 2 or njet == 1 or njet == 0:
            leg = ROOT.TLegend(0.7, 0.1, 0.9, 0.3)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
        else:
            leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
    elif year == "2023BPix":
        if njet == 2 or njet == 1 or njet == 0:
            leg = ROOT.TLegend(0.7, 0.1, 0.9, 0.3)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
        else:
            leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
    else:
        leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
        txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
    leg.AddEntry(hist_sf, "Data / DY MC SF", "l")
    # leg.AddEntry(f0, f"Poly(order={len(f0.GetParameters())-1})", "l")
    # leg.AddEntry(f1, f"Poly(order={len(f1.GetParameters())-1})", "l")
    # leg.AddEntry(f_flat, "Flat Above xmax", "l")
    leg.AddEntry(f_combined, "Combined Fit", "l")
    if band68:
        leg.AddEntry(band68, "68% fit band", "f")
    if band95:
        leg.AddEntry(band95, "95% fit band", "f")
    leg.Draw()

    # Text box with fit stats
    txt.SetFillColor(0)
    txt.SetBorderSize(1)
    txt.AddText("Fit Results:")
    txt.AddText(f"chi2/ndf = {chi2ndf:.5f}")
    txt.AddText(f"p-value = {pval:.3g}")
    txt.Draw()

    # --- Lower pad: Pull distribution ---
    canv.cd(2)
    ROOT.gPad.SetGrid()

    nbins_hist = hist_sf.GetNbinsX()
    xmin_hist = hist_sf.GetXaxis().GetXmin()
    xmax_hist = hist_sf.GetXaxis().GetXmax()

    # Generate pull histogram with same binning as hist_sf
    pull_hist = hist_sf.Clone("pull") # clone for copying binning
    pull_hist.SetTitle("Pull;Bin Center;(Data-Fit)/Error")
    pull_hist.Reset("ICES") # reset
    pull_hist.GetListOfFunctions().Clear() # remove the red line
    for i in range(1, nbins_hist + 1):
        data_val = hist_sf.GetBinContent(i)
        err = hist_sf.GetBinError(i)
        xval = hist_sf.GetBinCenter(i)
        fit_val = f_combined.Eval(xval) if err > 0 else 0.0
        pull = (data_val - fit_val) / err if err > 0 else 0.0
        pull_hist.SetBinContent(i, pull)

    pull_hist.SetMarkerStyle(20)
    pull_hist.Draw("P")

    # Save the canvas
    for ext in ("pdf", "png", "root"):
        canv.SaveAs(f"{save_dir}/{year}_njet{njet}_goodnessOfFit.{ext}")

def main():
    args = parse_arguments()
    run_label = args.label
    out_append = args.save_postfix
    # Determine which years to process
    if args.year.lower() == "all":
        years = ["2018", "2017", "2016postVFP", "2016preVFP"]
    else:
        years = [args.year]

    save_dict = {}
    global_fit_xmax = 200.0

    for year in years:
        in_dir = f"{args.save_path}/zpt_rewgt/{run_label}/{args.dy_sample}/{year}"
        save_dir = f"{in_dir}/gof_{out_append}"
        os.makedirs(save_dir, exist_ok=True)

        # Load the fit configuration YAML
        cfg_path = f"{in_dir}/fTest_{out_append}/zpt_fit_config.yaml"
        with open(cfg_path, "r") as cfg_file:
            fit_config = yaml.safe_load(cfg_file)

        year_dict = {}
        for njet in args.njet:
            key = f"njet{njet}"
            cfg = fit_config[year][key]

            order0 = cfg["f0"]["order"]
            xmin0, xmax0 = cfg["f0"]["fit_range"]
            order1 = cfg["f1"]["order"]
            xmin1, xmax1 = cfg["f1"]["fit_range"]
            edges = cfg["f0"]["bin_edges"]

            # Open the ROOT file and retrieve histograms
            in_file = ROOT.TFile(os.path.join(in_dir, f"{year}_njet{njet}.root"), "READ")
            workspace = in_file.Get("zpt_Workspace")

            # Clone data and DY MC histograms
            h_data = workspace.obj("hist_data").Clone("h_data_clone")
            h_dy   = workspace.obj("hist_dy").Clone("h_dy_clone")

            # Rebin both histograms with custom edges
            h_data_rebinned = rebin_histogram(h_data, edges)
            h_dy_rebinned   = rebin_histogram(h_dy, edges)
            nbins_new = h_data_rebinned.GetNbinsX()

            # Compute Scale Factor (SF) histogram = Data / DY MC
            h_SF = h_data_rebinned.Clone("h_SF")
            h_SF.Divide(h_dy_rebinned)

            # Removed previous call to h_SF.GetXaxis().SetRangeUser(0.0, global_fit_xmax)

            # Perform the piecewise fits
            f0, f1, f_flat, f_comb, fit_result = perform_fits(
                h_SF, order0, xmin0, xmax0, order1, xmin1, xmax1, global_fit_xmax
            )

            # Plot the SF and pull distributions
            plot_sf_and_pulls(
                h_SF, f0, f1, f_flat, f_comb, fit_result,
                xmin0, xmax0, xmin1, xmax1, global_fit_xmax,
                year, njet, nbins_new, save_dir
            )


            # Collect fit parameters for output
            max_order = 10
            params_dict = {f"f0_p{i}": 0.0 for i in range(max_order+1)}
            params_dict.update({f"f0_p{i}_err": 0.0 for i in range(max_order+1)})
            params_dict.update({f"f1_p{i}": 0.0 for i in range(max_order+1)})
            params_dict.update({f"f1_p{i}_err": 0.0 for i in range(max_order+1)})

            logger.debug(f"order0: {order0}, order1: {order1}")
            for i in range(f_comb.GetNpar()):
                logger.debug(f"f_comb parameter {i}: {f_comb.GetParameter(i)} +/- {f_comb.GetParError(i)}")

            final_piecewise = build_final_piecewise_coefficients(
                f0=f0,
                order0=order0,
                f1=f1,
                order1=order1,
                f_flat=f_flat,
                f_comb=f_comb,
                xmin1=xmin1,
                xmax1=xmax1,
            )

            for i in range(order0 + 1):
                params_dict[f"f0_p{i}"] = final_piecewise["f0_coeffs"][i]
                params_dict[f"f0_p{i}_err"] = final_piecewise["f0_errors"][i]
                logger.debug(
                    f"f0 parameter {i}: {final_piecewise['f0_coeffs'][i]} "
                    f"(local={f0.GetParameter(i)}) +/- {final_piecewise['f0_errors'][i]}"
                )

            for i in range(order1 + 1):
                params_dict[f"f1_p{i}"] = final_piecewise["f1_coeffs"][i]
                params_dict[f"f1_p{i}_err"] = final_piecewise["f1_errors"][i]
                logger.debug(
                    f"f1 parameter {i}: {final_piecewise['f1_coeffs'][i]} "
                    f"(local={f1.GetParameter(i)}) +/- {final_piecewise['f1_errors'][i]}"
                )

            logger.debug(
                f"horizontal_mx: {final_piecewise['tail_slope']} "
                f"(local={f_flat.GetParameter(0)}) +/- {final_piecewise['tail_slope_err']}"
            )
            logger.debug(
                f"horizontal_c0: {final_piecewise['tail_intercept']} "
                f"(local={f_flat.GetParameter(1)}) +/- {final_piecewise['tail_intercept_err']}"
            )
            logger.debug(
                f"combined adjustments: common_shift={final_piecewise['common_shift']}, "
                f"mid_tilt={final_piecewise['mid_tilt']}, "
                f"delta_tail_slope={final_piecewise['delta_tail_slope']}"
            )

            params_dict["horizontal_mx"] = final_piecewise["tail_slope"]
            params_dict["horizontal_c0"] = final_piecewise["tail_intercept"]
            params_dict["polynomial_range"] = {"xlow": 0.0, "xmin1": xmin1, "xmax1": xmax1, "xhigh": global_fit_xmax}
            params_dict["total_bins"] = nbins_new
            params_dict["fit_orders"] = {"f0_order": order0, "f1_order": order1}
            bin_array = array.array("d", edges)
            params_dict["bin_edges"] = bin_array.tolist()

            year_dict[f"njet_{njet}"] = {"function": params_dict}
            print(f"Using custom binning with {nbins_new} bins: {edges}")

        save_dict[year] = year_dict

    # Merge with existing YAML or create fresh
    # print(f"Saving fit parameters to YAML: \n{save_dict}")
    # ------------------------------------------------------------------
    # Save YAML with top-level keys = years
    # ------------------------------------------------------------------
    in_dir_yaml = f"{args.save_path}/zpt_rewgt/{run_label}/{args.dy_sample}/"
    os.makedirs(in_dir_yaml, exist_ok=True)
    yaml_path = f"{in_dir_yaml}/zpt_rewgt_params_{args.dy_sample}.yaml"

    new_cfg = OmegaConf.create(save_dict)

    if os.path.isfile(yaml_path):
        existing = OmegaConf.load(yaml_path)
        merged = OmegaConf.merge(existing, new_cfg)  # merge year-by-year (and njet-by-njet)
    else:
        merged = new_cfg

    # Convert to sorted YAML string first
    sorted_yaml = OmegaConf.to_yaml(merged, sort_keys=True)

    # Save the sorted string to the file
    with open(yaml_path, "w") as f:
        f.write(sorted_yaml)

    print(f"Saved fit parameters to {yaml_path}")

if __name__ == "__main__":
    main()
