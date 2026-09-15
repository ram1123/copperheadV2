import array
import json
import os
from datetime import datetime, timezone

import ROOT
import yaml
import poly_utils
from cli.common_argparser import build_common_parser
from modules.git_utils import get_git_state
from modules.utils import logger
from omegaconf import OmegaConf
from sample_resolution import (
    collect_process_paths,
    resolve_dy_processes,
    resolve_stage1_base_path,
)

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


def eval_polynomial_derivative(coeffs, xval):
    return sum(idx * coeff * (xval ** (idx - 1)) for idx, coeff in enumerate(coeffs) if idx >= 1)


def make_combined_function_reduced(f0_coeffs, f1_coeffs, xmin, xmax):
    """
    Builds a reduced-parameter piecewise function with exact C0 (value) continuity
    at both xmin and xmax, AND exact C1 (slope) continuity at xmax by construction:
    the tail's slope/intercept are derived analytically from f1's own fit at xmax
    (not from an unrelated, independently-fit flat-line tail), so a delta_tail_slope
    of 0 already joins smoothly instead of leaving a visible corner. f0 gets its own
    tilt (mirroring f1's), so a single shared vertical shift no longer has to serve
    both regions at once - the previous single-shift design otherwise pulls whichever
    region has the most fit weight (usually f1/tail) at the expense of the other.

    Free parameters:
    - par[0]: common_shift - vertical shift applied to the low-range polynomial
              (propagated into the mid-range join via continuity)
    - par[1]: low_tilt - extra low-range tilt multiplying (x - xmin); vanishes at
              x=xmin, so xmin-continuity is unaffected by its value
    - par[2]: mid_tilt - extra mid-range tilt multiplying (x - xmin)
    - par[3]: delta_tail_slope - small deviation from f1's own analytic slope at xmax
    """
    f1_prime_xmax = eval_polynomial_derivative(f1_coeffs, xmax)

    def func(x, par):
        xx = x[0]
        common_shift = par[0]
        low_tilt = par[1]
        mid_tilt = par[2]
        delta_tail_slope = par[3]

        def eval_low(xlow):
            return eval_polynomial(f0_coeffs, xlow) + common_shift + low_tilt * (xlow - xmin)

        low_xmin = eval_low(xmin)
        mid_xmin_raw = eval_polynomial(f1_coeffs, xmin)
        mid_shift = low_xmin - mid_xmin_raw

        def eval_mid(xmid):
            return eval_polynomial(f1_coeffs, xmid) + mid_shift + mid_tilt * (xmid - xmin)

        tail_slope = f1_prime_xmax + mid_tilt + delta_tail_slope

        if xx < 0.0:
            return 0.0
        elif xx <= xmin:
            return eval_low(xx)
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


def make_confidence_band(global_xmax, confidence_level, name, npoints=400):
    """
    Confidence band as a fine TGraphErrors spanning [0, global_xmax], not a
    clone of hist_sf: TH1's "E3" fill only spans the first-to-last bin
    *center*, not the true axis edges, so a wide last bin (e.g. the [170,200]
    tail bin, center 185) leaves a visible gap at the very end - a fine grid
    of points decouples the band's resolution from the (necessarily coarse,
    sparse-stats) tail binning. Draw with option "3 SAME".
    """
    xs = array.array('d', [global_xmax * i / (npoints - 1) for i in range(npoints)])
    ys = array.array('d', [0.0] * npoints)
    graph = ROOT.TGraphErrors(npoints, xs, ys)
    graph.SetName(name)
    ROOT.TVirtualFitter.GetFitter().GetConfidenceIntervals(graph, confidence_level)
    return graph

def fit_polynomial(hist_sf, order, xmin, xmax, name):
    """
    Fits a polynomial of degree 'order' to hist_sf between [xmin, xmax] using a
    numerically stable, centered/scaled Chebyshev basis (see poly_utils.py),
    then converts back to plain monomial-in-x coefficients + covariance.
    Returns the poly_utils.fit_chebyshev_poly() result dict.
    """
    return poly_utils.fit_chebyshev_poly(hist_sf, order, xmin, xmax, name)

def fit_flat_line(hist_sf, xmin, xmax, fit_opts="S R Q"):
    """
    Fits a straight line (slope, intercept) to hist_sf between [xmin, xmax]
    using a proper chi-square fit against the histogram's own bin errors.
    Returns the TF1 object for that line.
    """
    func = ROOT.TF1("flat_line", "[0]*x + [1]", xmin, xmax)
    hist_sf.Fit(func, fit_opts, "", xmin, xmax)
    return func


def build_final_piecewise_coefficients(f0_coeffs, f0_errors, order0, f1_coeffs, f1_errors, order1, f_flat, f_comb, xmin1, xmax1):
    """
    Convert the reduced-parameter combined refit back into the full set of
    piecewise coefficients expected by stage1. f0_coeffs/f0_errors and
    f1_coeffs/f1_errors are the monomial-in-x coefficients (and their
    uncertainties) coming from poly_utils.fit_chebyshev_poly()'s basis
    conversion - not read off a monomial-parametrized TF1 directly. f_flat (the
    independent tail-only fit) is unused here - the tail slope/intercept are
    now derived analytically from f1 at xmax1 for exact C1 continuity there,
    see make_combined_function_reduced().
    """
    common_shift = f_comb.GetParameter(0)
    common_shift_err = f_comb.GetParError(0)
    low_tilt = f_comb.GetParameter(1)
    low_tilt_err = f_comb.GetParError(1)
    mid_tilt = f_comb.GetParameter(2)
    mid_tilt_err = f_comb.GetParError(2)
    delta_tail_slope = f_comb.GetParameter(3)
    delta_tail_slope_err = f_comb.GetParError(3)

    final_f0_coeffs = list(f0_coeffs)
    final_f0_errors = list(f0_errors)
    final_f0_coeffs[0] += common_shift - low_tilt * xmin1
    final_f0_coeffs[1] += low_tilt
    final_f0_errors[0] = (final_f0_errors[0] ** 2 + common_shift_err ** 2 + (xmin1 * low_tilt_err) ** 2) ** 0.5
    final_f0_errors[1] = (final_f0_errors[1] ** 2 + low_tilt_err ** 2) ** 0.5

    low_xmin_nominal = eval_polynomial(f0_coeffs, xmin1)
    mid_xmin_nominal = eval_polynomial(f1_coeffs, xmin1)
    continuity_shift = low_xmin_nominal - mid_xmin_nominal + common_shift

    final_f1_coeffs = list(f1_coeffs)
    final_f1_errors = list(f1_errors)
    final_f1_coeffs[0] += continuity_shift - mid_tilt * xmin1
    final_f1_coeffs[1] += mid_tilt
    final_f1_errors[0] = (final_f1_errors[0] ** 2 + common_shift_err ** 2 + (xmin1 * mid_tilt_err) ** 2) ** 0.5
    final_f1_errors[1] = (final_f1_errors[1] ** 2 + mid_tilt_err ** 2) ** 0.5

    # Tail slope: f1's own analytic derivative at xmax1 (guarantees C1 continuity
    # when delta_tail_slope == 0) plus the small MINUIT-fitted correction.
    f1_prime_xmax = eval_polynomial_derivative(f1_coeffs, xmax1)
    final_tail_slope = f1_prime_xmax + mid_tilt + delta_tail_slope
    # Diagonal (uncorrelated) quadrature approximation, consistent with the rest
    # of this function's error propagation.
    f1_prime_xmax_err = sum(
        (idx * (xmax1 ** (idx - 1)) * err) ** 2 for idx, err in enumerate(f1_errors) if idx >= 1
    ) ** 0.5
    final_tail_slope_err = (f1_prime_xmax_err ** 2 + mid_tilt_err ** 2 + delta_tail_slope_err ** 2) ** 0.5

    y_at_xmax = eval_polynomial(final_f1_coeffs, xmax1)
    tail_intercept = y_at_xmax - final_tail_slope * xmax1
    y_at_xmax_err = sum((xmax1 ** idx * err) ** 2 for idx, err in enumerate(final_f1_errors)) ** 0.5
    tail_intercept_err = (y_at_xmax_err ** 2 + (xmax1 * final_tail_slope_err) ** 2) ** 0.5

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
        "low_tilt": low_tilt,
        "mid_tilt": mid_tilt,
        "delta_tail_slope": delta_tail_slope,
    }

def perform_fits(hist_sf, order0, xmin0, xmax0, order1, xmin1, xmax1, global_xmax):
    """
    Runs the three-step fits: 1) poly(order0) on [0, xmax0], 2) poly(order1) on [xmin1, xmax1],
    3) flat line on [xmax1, global_xmax]. Then creates and fits the combined TF1 over [0, global_xmax].
    Returns (f0_result, f1_result, f_flat, f_combined, final_fit), where f0_result/f1_result are the
    dicts from poly_utils.fit_chebyshev_poly() (Chebyshev TF1 + converted monomial coeffs/errors).
    """
    logger.info(f"Performing piecewise fits with orders {order0} and {order1}")

    # 1) Low-range fit
    logger.debug(f"Fitting low range: 0 to {xmax0} with order {order0}")
    f0_result = fit_polynomial(hist_sf, order0, 0.0, xmax0, "f0_local")

    # 2) Mid-range fit
    logger.debug(f"Fitting mid range: {xmin1} to {xmax1} with order {order1}")
    f1_result = fit_polynomial(hist_sf, order1, xmin1, xmax1, "f1_local")

    # 3) High-range flat fit - kept only as an independent diagnostic reference
    # (e.g. to sanity-check the analytic tail slope below); it is no longer the
    # tail's baseline, since matching an independently-fit line's value but not
    # its slope is exactly what produced the visible "kink" at xmax1.
    logger.debug(f"Fitting high range: {xmax1} to {global_xmax} with flat line")
    f_flat = fit_flat_line(hist_sf, xmax1, global_xmax)

    # Build reduced-parameter combined TF1 using the stable local fits as anchors.
    f0_coeffs = list(f0_result["coeffs_x"])
    f1_coeffs = list(f1_result["coeffs_x"])
    logger.debug("Creating reduced-parameter combined function with 4 parameters")

    comb_func = make_combined_function_reduced(
        f0_coeffs=f0_coeffs,
        f1_coeffs=f1_coeffs,
        xmin=xmin1,
        xmax=xmax1,
    )
    logger.debug("Prepared reduced-parameter combined function for fitting")

    f_combined = ROOT.TF1("f_combined", comb_func, 0.0, global_xmax, 4)
    f_combined.SetParName(0, "common_shift")
    f_combined.SetParName(1, "low_tilt")
    f_combined.SetParName(2, "mid_tilt")
    f_combined.SetParName(3, "delta_tail_slope")
    f_combined.SetParameter(0, 0.0)
    f_combined.SetParameter(1, 0.0)
    f_combined.SetParameter(2, 0.0)
    f_combined.SetParameter(3, 0.0)
    # These 4 parameters are small corrections around the (already stable)
    # local anchor fits, so generous - not razor-tight - bounds are enough to
    # keep MIGRAD from running away; overly tight limits (as before) push the
    # minimum onto a bound, where MINUIT's internal boundary transform makes
    # HESSE/MINOS errors unreliable (often artificially huge or asymmetric).
    f_combined.SetParLimits(0, -2.0, 2.0)
    f_combined.SetParLimits(1, -0.5, 0.5)
    f_combined.SetParLimits(2, -0.5, 0.5)
    f_combined.SetParLimits(3, -0.5, 0.5)

    # Perform final reduced refit: a single proper chi2 fit against hist_sf's
    # own bin errors (not the Poisson log-likelihood option "L", which is not
    # appropriate for an already-computed Data/MC ratio histogram).
    final_fit = hist_sf.Fit(f_combined, "S R Q", "", 0.0, global_xmax)
    if final_fit and int(final_fit.Status()) != 0:
        final_fit = hist_sf.Fit(f_combined, "S R Q", "", 0.0, global_xmax)
    logger.debug(f"Final fit result: {final_fit}")

    return f0_result, f1_result, f_flat, f_combined, final_fit

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
    pad1 = canv.cd(1)
    # Force X-axis range from 0 to global_xmax
    hist_sf.GetXaxis().SetRangeUser(0.0, global_xmax)
    hist_sf.SetTitle(f"Year {year}, njet={njet}, bins={nbins}")
    hist_sf.SetLineColor(ROOT.kBlue)
    hist_sf.SetMarkerColor(ROOT.kBlue)
    hist_sf.SetMarkerStyle(20)
    hist_sf.SetMarkerSize(0.6)
    hist_sf.Draw("axis")
    pad1.Update()
    ymin_auto = pad1.GetUymin()
    ymax_auto = pad1.GetUymax()

    band95 = None
    band68 = None
    if fit_result and int(fit_result.Status()) == 0:
        band95 = make_confidence_band(global_xmax, 0.95, f"band95_{year}_{njet}")
        band95.SetFillColorAlpha(ROOT.kAzure - 9, 0.35)
        band95.SetLineColor(ROOT.kAzure - 9)
        band95.SetLineWidth(0)
        band95.SetMarkerSize(0)

        band68 = make_confidence_band(global_xmax, 0.68, f"band68_{year}_{njet}")
        band68.SetFillColorAlpha(ROOT.kOrange - 2, 0.45)
        band68.SetLineColor(ROOT.kOrange - 2)
        band68.SetLineWidth(0)
        band68.SetMarkerSize(0)

    # Draw the fit function across the full x-range
    f_combined.SetRange(0.0, global_xmax)
    f_combined.SetNpx(5000)   # or 10000 if you want it super smooth
    f_combined.SetLineColor(ROOT.kRed)

    # Rebuild the pad's frame explicitly at exactly [0, global_xmax] - bypasses
    # TH1's automatic (padded) frame sizing entirely, unlike SetRangeUser or
    # SetNdivisions(optimize=False), neither of which affected the actual
    # rendered frame edge when tried here.
    hist_sf.GetListOfFunctions().Clear()  # remove attached
    pad1.Clear()
    frame = pad1.DrawFrame(0.0, ymin_auto, global_xmax, ymax_auto)
    frame.SetTitle(hist_sf.GetTitle())
    frame.GetXaxis().SetTitle(hist_sf.GetXaxis().GetTitle())
    frame.GetYaxis().SetTitle(hist_sf.GetYaxis().GetTitle())
    if band95:
        band95.Draw("3 SAME")
    if band68:
        band68.Draw("3 SAME")
    # "P" (points at bin centers) instead of plain "E" (which also draws a
    # connecting step outline - i.e. a flat horizontal segment across each
    # bin's full width). The tail bins are tens of GeV wide, so that flat
    # segment visibly diverges from the smoothly-varying fit curve/band,
    # looking like a sharp discontinuity that isn't actually there (the fit
    # itself and its confidence band are smooth - verified numerically).
    hist_sf.Draw("same P E1")
    f_combined.Draw("SAME")
    pad1.Update()

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
        if njet == 2 or njet == 1 or njet == 0:
            leg = ROOT.TLegend(0.7, 0.1, 0.9, 0.3)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
        else:
            leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
    elif year == "2023":
        if njet == 2 or njet == 0:
            leg = ROOT.TLegend(0.7, 0.1, 0.9, 0.3)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
        else:
            leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
            txt = ROOT.TPaveText(0.4, 0.7, 0.7, 0.9, "NDC")
    elif year == "2023BPix":
        if njet == 2:
            leg = ROOT.TLegend(0.7, 0.1, 0.9, 0.3)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
        else:
            leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
            txt = ROOT.TPaveText(0.4, 0.7, 0.7, 0.9, "NDC")
    elif year == "2024":
        if njet == 2  or njet == 1 or njet == 0:
            leg = ROOT.TLegend(0.7, 0.1, 0.9, 0.3)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
        else:
            leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
            txt = ROOT.TPaveText(0.4, 0.7, 0.7, 0.9, "NDC")      
    elif year == "2025":
        if njet == 2  or njet == 1 or njet == 0:
            leg = ROOT.TLegend(0.7, 0.1, 0.9, 0.3)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
        else:
            leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
            txt = ROOT.TPaveText(0.4, 0.7, 0.7, 0.9, "NDC")    
    elif year == "2026":
        if njet == 1 or njet == 0:
            leg = ROOT.TLegend(0.7, 0.1, 0.9, 0.3)
            txt = ROOT.TPaveText(0.4, 0.1, 0.7, 0.3, "NDC")
        else:
            leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
            txt = ROOT.TPaveText(0.4, 0.7, 0.7, 0.9, "NDC")                              
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
    pad2 = canv.cd(2)
    pad2.SetGrid()

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
    pull_hist.SetMarkerColor(ROOT.kBlack)  # pull_hist is a hist_sf clone; keep its own look, not hist_sf's blue
    pull_hist.GetXaxis().SetRangeUser(0.0, global_xmax)
    pull_hist.Draw("P")
    pad2.Update()
    # Rebuild the frame at exactly [0, global_xmax] - see the upper panel's
    # comment for why (TH1's automatic frame sizing pads past the request).
    ymin_auto = pad2.GetUymin()
    ymax_auto = pad2.GetUymax()
    pad2.Clear()
    pad2.SetGrid()
    frame2 = pad2.DrawFrame(0.0, ymin_auto, global_xmax, ymax_auto)
    frame2.SetTitle(pull_hist.GetTitle())
    frame2.GetXaxis().SetTitle(pull_hist.GetXaxis().GetTitle())
    frame2.GetYaxis().SetTitle(pull_hist.GetYaxis().GetTitle())
    pull_hist.Draw("P SAME")

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

    # Provenance for the fit step itself (produced once per script run, not
    # per year) - who ran get_polyFit.py, when, and against which git state.
    # Folded per-year below alongside step 0's own provenance.json (sample
    # resolution), when present, so the final YAML carries a full trail.
    in_dir_yaml = f"{args.save_path}/zpt_rewgt/{run_label}/{args.dy_sample}/"
    os.makedirs(in_dir_yaml, exist_ok=True)
    fit_git_state = get_git_state(in_dir_yaml)

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
            f0_result, f1_result, f_flat, f_comb, fit_result = perform_fits(
                h_SF, order0, xmin0, xmax0, order1, xmin1, xmax1, global_fit_xmax
            )

            # Plot the SF and pull distributions (f0/f1 TF1s are the Chebyshev-
            # parametrized fits - same curve/chi2/ndf as the monomial form)
            plot_sf_and_pulls(
                h_SF, f0_result["tf1"], f1_result["tf1"], f_flat, f_comb, fit_result,
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
                f0_coeffs=list(f0_result["coeffs_x"]),
                f0_errors=list(f0_result["errors_x"]),
                order0=order0,
                f1_coeffs=list(f1_result["coeffs_x"]),
                f1_errors=list(f1_result["errors_x"]),
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
                    f"(local={f0_result['coeffs_x'][i]}) +/- {final_piecewise['f0_errors'][i]}"
                )

            for i in range(order1 + 1):
                params_dict[f"f1_p{i}"] = final_piecewise["f1_coeffs"][i]
                params_dict[f"f1_p{i}_err"] = final_piecewise["f1_errors"][i]
                logger.debug(
                    f"f1 parameter {i}: {final_piecewise['f1_coeffs'][i]} "
                    f"(local={f1_result['coeffs_x'][i]}) +/- {final_piecewise['f1_errors'][i]}"
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
                f"low_tilt={final_piecewise['low_tilt']}, "
                f"mid_tilt={final_piecewise['mid_tilt']}, "
                f"delta_tail_slope={final_piecewise['delta_tail_slope']}"
            )

            params_dict["horizontal_mx"] = final_piecewise["tail_slope"]
            params_dict["horizontal_mx_err"] = final_piecewise["tail_slope_err"]
            params_dict["horizontal_c0"] = final_piecewise["tail_intercept"]
            params_dict["horizontal_c0_err"] = final_piecewise["tail_intercept_err"]
            params_dict["polynomial_range"] = {"xlow": 0.0, "xmin1": xmin1, "xmax1": xmax1, "xhigh": global_fit_xmax}
            params_dict["total_bins"] = nbins_new
            params_dict["fit_orders"] = {"f0_order": order0, "f1_order": order1}
            bin_array = array.array("d", edges)
            params_dict["bin_edges"] = bin_array.tolist()

            year_dict[f"njet_{njet}"] = {"function": params_dict}
            print(f"Using custom binning with {nbins_new} bins: {edges}")

        metadata = {
            "step2_derived_at": datetime.now(timezone.utc).isoformat(),
            "step2_derived_by": os.getenv("USER", "unknown"),
            "step2_git_commit": fit_git_state["commit"],
            "step2_git_dirty": fit_git_state["dirty"],
            "step2_git_diff_file": fit_git_state["diff_file"],
            "run_label": run_label,
            # Output-directory tag for this derivation run, not the physical
            # DY MC sample(s) actually used - see dy_mc_samples below.
            "dy_sample_label": args.dy_sample,
            "save_postfix": out_append,
        }

        # The actual DY MC sample(s) read from the stage1 output, resolved
        # the same way save_SF_rootFiles.py does (sample_resolution.py) -
        # just the cheap glob/YAML lookup, no parquet read, so this doesn't
        # need step0 to have been rerun with provenance capture. Requires
        # --input_path (the stage1 output base) to have been passed through;
        # degrades to a logged note, not a crash, if it wasn't.
        if args.input_path:
            try:
                stage1_base_path = resolve_stage1_base_path(args.input_path, year)
                dy_processes = resolve_dy_processes(year, args.sample_config)
                _, matched_dy_processes, missing_dy_processes = collect_process_paths(
                    stage1_base_path, dy_processes
                )
                metadata["dy_mc_samples"] = {
                    "stage1_base_path": stage1_base_path,
                    "sample_config": args.sample_config,
                    "matched": matched_dy_processes,
                    "missing": missing_dy_processes,
                }
            except Exception as exc:
                logger.warning(f"Could not resolve actual DY MC sample(s) for {year}: {exc}")
        else:
            logger.debug(
                "No --input_path given; cannot resolve the actual DY MC sample(s) "
                "for metadata.dy_mc_samples (dy_sample_label is only the output-dir tag)."
            )

        # save_SF_rootFiles.py (step 0) writes this alongside the per-year ROOT
        # files: when it ran, by whom, and the same Data/DY resolution above
        # (redundant with dy_mc_samples but captured at step0 time). Fold it
        # in here so the final YAML carries the full trail even though step 0
        # and step 2 run as separate processes.
        prov_path = f"{in_dir}/provenance.json"
        if os.path.isfile(prov_path):
            with open(prov_path) as prov_file:
                metadata["step0"] = json.load(prov_file)
        else:
            logger.debug(f"No step0 provenance.json found at {prov_path}")
        year_dict["metadata"] = metadata

        save_dict[year] = year_dict

    # Merge with existing YAML or create fresh
    # print(f"Saving fit parameters to YAML: \n{save_dict}")
    # ------------------------------------------------------------------
    # Save YAML with top-level keys = years
    # ------------------------------------------------------------------
    yaml_path = f"{in_dir_yaml}/zpt_rewgt_params_{args.dy_sample}.yaml"

    new_cfg = OmegaConf.create(save_dict)

    if os.path.isfile(yaml_path):
        existing = OmegaConf.load(yaml_path)
        # `metadata` must fully replace, not deep-merge, for any year this run
        # touches - otherwise a renamed/removed field (e.g. dy_sample ->
        # dy_sample_label) lingers forever alongside its replacement, since
        # OmegaConf.merge only adds/overwrites keys, never drops them.
        for year in save_dict:
            if year in existing and "metadata" in existing[year]:
                del existing[year]["metadata"]
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
