import os
import argparse
import yaml
import array
import numpy as np
import ROOT
from scipy.stats import f
from omegaconf import OmegaConf

# Run in batch mode and disable statistics box
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

# from do_f_test import define_custom_binning
from bin_definitions import define_custom_binning

def parse_arguments():
    parser = argparse.ArgumentParser(description="Perform goodness-of-fit for Z pT SFs")
    parser.add_argument(
        "-l", "--label", dest="label", required=True,
        help="Run label: directory under plot_path to find inputs"
    )
    parser.add_argument(
        "-y", "--year", dest="year", default="all",
        help="Year to process (or 'all' for 2016preVFP, 2016postVFP, 2017, 2018)"
    )
    parser.add_argument(
        "-save", "--plot_path", dest="plot_path", default="plots",
        help="Base directory where plots and inputs live"
    )
    parser.add_argument(
        "--nbins", type=str, default="CustomBins",
        help="Binning key (only 'CustomBins' is handled)"
    )
    parser.add_argument(
        "--njet", type=int, nargs="+", default=[0, 1, 2],
        help="Jet multiplicities to loop over"
    )
    parser.add_argument(
        "--outAppend", type=str, default="",
        help="String to append to output filenames"
    )
    parser.add_argument(
        "-dy_sample", "--dy_sample", dest="dy_sample",
        default="MiNNLO",
        choices=["MiNNLO", "aMCatNLO", "VBF_filter"],
        action="store",
        help="choose the type of DY samples to use for Zpt reweighting",
    )
    return parser.parse_args()

def make_combined_function(order0, order, xmin, xmax):
    """
    Builds a piecewise function:
    - Polynomial of degree 'order0' from 0 to xmin
    - Polynomial of degree 'order' from xmin to xmax, shifted to ensure continuity
    - Linear function y = m·x + c beyond xmax, matched to the value and slope at xmax
    """
    def func(x, par):
        xx = x[0]
        # if xx <= 0:
        #     return 0.0

        # Polynomial f0 up to xmin
        f0_xmin = sum(par[i] * (xmin**i) for i in range(order0 + 1))

        # Polynomial f1 up to xmin and xmax
        f1_coeffs = [par[order0 + 1 + i] for i in range(order + 1)]
        f1_xmin = sum(f1_coeffs[i] * (xmin**i) for i in range(order + 1))
        f1_xmax = sum(f1_coeffs[i] * (xmax**i) for i in range(order + 1))

        # Evaluate derivative of f1 at xmax to compute slope for linear extension
        df1_xmax = sum(i * f1_coeffs[i] * (xmax**(i - 1)) for i in range(1, order + 1))

        if 0 <= xx <= xmin:
            return sum(par[i] * (xx**i) for i in range(order0 + 1))
        elif xx < xmax:
            return sum(f1_coeffs[i] * (xx**i) for i in range(order + 1)) + (f0_xmin - f1_xmin)
        else:
            # Straight line y = m·x + c, passing through (xmax, f_combined(xmax))
            m = df1_xmax  # slope from derivative
            y_at_xmax = f1_xmax + (f0_xmin - f1_xmin)
            c = y_at_xmax - m * xmax
            return m * xx + c

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

def fit_polynomial(hist_sf, order, xmin, xmax, fit_opts="L S Q"):
    """
    Fits a polynomial of degree 'order' to hist_sf between [xmin, xmax].
    Returns the TF1 polynomial object.
    """
    expr = " + ".join(f"[{i}]*x**{i}" for i in range(order + 1))
    func = ROOT.TF1(f"poly{order}", expr, xmin, xmax)
    hist_sf.Fit(func, fit_opts, "", xmin, xmax)
    hist_sf.Fit(func, fit_opts, "", xmin, xmax)
    result = hist_sf.Fit(func, "L S R", "", xmin, xmax)
    return func

def fit_flat_line(hist_sf, xmin, xmax, fit_opts="L I S R"):
    """
    Fits a constant line to hist_sf between [xmin, xmax].
    Returns the TF1 object for that line.
    """
    func = ROOT.TF1("flat_line", "[0]*x + [1]", xmin, xmax)
    result = hist_sf.Fit(func, fit_opts, "", xmin, xmax)
    return func

def perform_fits(hist_sf, order0, xmin0, xmax0, order1, xmin1, xmax1, global_xmax):
    """
    Runs the three-step fits: 1) poly(order0) on [0, xmax0], 2) poly(order1) on [xmin1, xmax1],
    3) flat line on [xmax1, global_xmax]. Then creates and fits the combined TF1 over [0, global_xmax].
    Returns all TF1s: (f0, f1, f_flat, f_combined).
    """
    # 1) Low-range fit
    f0 = fit_polynomial(hist_sf, order0, 0.0, xmax0, fit_opts="L S Q")

    # 2) Mid-range fit
    f1 = fit_polynomial(hist_sf, order1, xmin1, xmax1, fit_opts="L I S Q")

    # 3) High-range flat fit
    f_flat = fit_flat_line(hist_sf, xmax1, global_xmax, fit_opts="L I S R")
    # f_flat = fit_polynomial(hist_sf, order1, xmax1, global_xmax, fit_opts="L I S R")

    # Build combined TF1
    npar = (order0 + 1) + (order1 + 1) + 2  # coefficients: f0, f1, flat_c
    comb_func = make_combined_function(order0, order1, xmin1, xmax1)
    f_combined = ROOT.TF1("f_combined", comb_func, 0.0, global_xmax, npar)

    # Gather initial parameters from f0, f1, f_flat
    params = []
    for i in range(order0 + 1):
        params.append(f0.GetParameter(i))
    for i in range(order1 + 1):
        params.append(f1.GetParameter(i))
    params.append(f_flat.GetParameter(0))
    f_combined.SetParameters(*params)

    # Ensure the function is zero below x=0
    # (the user-defined func handles this internally)

    # Perform final fit
    hist_sf.Fit(f_combined, "L I S R", "", 0.0, global_xmax)
    hist_sf.Fit(f_combined, "L I S R", "", 0.0, global_xmax)
    hist_sf.Fit(f_combined, "L I S R", "", 0.0, global_xmax)

    return f0, f1, f_flat, f_combined

def plot_sf_and_pulls(hist_sf, f0, f1, f_flat, f_combined,
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

    # Draw the fit function across the full x-range
    f_combined.SetRange(0.0, global_xmax)
    f_combined.SetLineColor(ROOT.kRed)
    f_combined.Draw("SAME")

    # Finally draw the histogram itself (with error bars)
    hist_sf.Draw("same E")

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
    else:
        leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    leg.AddEntry(hist_sf, "Data / DY MC SF", "l")
    #leg.AddEntry(f0, f"Poly(order={len(f0.GetParameters())-1})", "l")
    #leg.AddEntry(f1, f"Poly(order={len(f1.GetParameters())-1})", "l")
    #leg.AddEntry(f_flat, "Flat Above xmax", "l")
    leg.AddEntry(f_combined, "Combined Fit", "l")
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
    ROOT.gPad.SetPad(0, 0, 1, 0.4)
    ROOT.gPad.SetGrid()

    nbins_hist = hist_sf.GetNbinsX()
    xmin_hist = hist_sf.GetXaxis().GetXmin()
    xmax_hist = hist_sf.GetXaxis().GetXmax()

    pull_hist = ROOT.TH1D("pull", "Pull;Bin Center;(Data-Fit)/Error", nbins_hist, xmin_hist, xmax_hist)
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
        canv.SaveAs(f"{save_dir}/{year}_njet{njet}_{nbins}_goodnessOfFit.{ext}")

def main():
    args = parse_arguments()
    run_label = args.label
    plot_base = args.plot_path
    out_append = args.outAppend

    # Determine which years to process
    if args.year.lower() == "all":
        years = ["2018", "2017", "2016postVFP", "2016preVFP"]
    else:
        years = [args.year]

    save_dict = {}
    global_fit_xmax = 200.0

    for year in years:
        in_dir = f"{args.plot_path}/zpt_rewgt/{run_label}/{args.dy_sample}/{year}"
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

            # Open the ROOT file and retrieve histograms
            in_file = ROOT.TFile(os.path.join(in_dir, f"{year}_njet{njet}.root"), "READ")
            workspace = in_file.Get("zpt_Workspace")

            # Clone data and DY MC histograms
            h_data = workspace.obj("hist_data").Clone("h_data_clone")
            h_dy   = workspace.obj("hist_dy").Clone("h_dy_clone")

            # Rebin both histograms with custom edges
            edges = define_custom_binning()
            h_data_rebinned = rebin_histogram(h_data, edges)
            h_dy_rebinned   = rebin_histogram(h_dy, edges)
            nbins_new = h_data_rebinned.GetNbinsX()

            # Compute Scale Factor (SF) histogram = Data / DY MC
            h_SF = h_data_rebinned.Clone("h_SF")
            h_SF.Divide(h_dy_rebinned)

            # Removed previous call to h_SF.GetXaxis().SetRangeUser(0.0, global_fit_xmax)

            # Perform the piecewise fits
            f0, f1, f_flat, f_comb = perform_fits(
                h_SF, order0, xmin0, xmax0, order1, xmin1, xmax1, global_fit_xmax
            )

            # Plot the SF and pull distributions
            plot_sf_and_pulls(
                h_SF, f0, f1, f_flat, f_comb,
                xmin0, xmax0, xmin1, xmax1, global_fit_xmax,
                year, njet, nbins_new, save_dir
            )


            # Collect fit parameters for output
            max_order = 5
            params_dict = {f"f0_p{i}": 0.0 for i in range(max_order+1)}
            params_dict.update({f"f0_p{i}_err": 0.0 for i in range(max_order+1)})
            params_dict.update({f"f1_p{i}": 0.0 for i in range(max_order+1)})
            params_dict.update({f"f1_p{i}_err": 0.0 for i in range(max_order+1)})

            for i in range(order0+1):
                params_dict[f"f0_p{i}"] = f0.GetParameter(i)
                params_dict[f"f0_p{i}_err"] = f0.GetParError(i)
            for i in range(order1+1):
                params_dict[f"f1_p{i}"] = f1.GetParameter(i)
                params_dict[f"f1_p{i}_err"] = f1.GetParError(i)

            params_dict["horizontal_mx"] = f_flat.GetParameter(0)
            params_dict["horizontal_c0"] = f_flat.GetParameter(1)
            params_dict["polynomial_range"] = {"xmin1": xmin1, "xmax1": xmax1}

            year_dict[f"njet_{njet}"] = {nbins_new: params_dict}
            print(f"Using custom binning with {nbins_new} bins: {edges}")

        save_dict[year] = year_dict

    # Merge with existing YAML or create fresh
    in_dir_yaml = f"{args.plot_path}/zpt_rewgt/{run_label}/{args.dy_sample}/"
    os.makedirs(in_dir_yaml, exist_ok=True)
    yaml_path = f"{in_dir_yaml}/zpt_rewgt_params_{args.dy_sample}.yaml"
    if os.path.isfile(yaml_path):
        existing = OmegaConf.load(yaml_path)
        merged = OmegaConf.merge(existing, {"gof_results": save_dict})
    else:
        merged = OmegaConf.create({"gof_results": save_dict})
    OmegaConf.save(merged, yaml_path)
    print(f"Saved fit parameters to {yaml_path}")

if __name__ == "__main__":
    main()
