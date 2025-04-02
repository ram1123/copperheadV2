import ROOT
from scipy.stats import f
from omegaconf import OmegaConf
import os
import argparse

# dictionary of all orders of polynomial from f-test
f_orders = { # to recalculate these, re-run f-test on do_f_test.py
    "2018" : {
        "njet0" : 5,
        "njet1" : 5, #4
        "njet2" : 4,
    },
    "2017" : {
        "njet0" : 8,
        "njet1" : 5,
        "njet2" : 4,
    },
    "2016postVFP" : {
        "njet0" : 5,
        "njet1" : 4,
        "njet2" : 4,
    },
    "2016preVFP" : {
        "njet0" : 4,
        "njet1" : 4,
        "njet2" : 4,
    },
}

poly_fit_ranges = {
    "2018" : {
        "njet0" : [0, 95],
        "njet1" : [0, 90], #50
        "njet2" : [0, 90],
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
global_fit_xmax = 200


if __name__ == "__main__":
    """
    This file is meant to define the Zpt histogram binning for zpt fitting
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-l",
    "--label",
    dest="label",
    default=None,
    action="store",
    help="save path to store stage1 output files",
    )
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="all",
    action="store",
    help="string value of year we are calculating",
    )
    # years = ["2018", "2017", "2016postVFP", "2016preVFP"]
    # years = ["2018",]
    args = parser.parse_args()
    run_label = args.label
    if args.year == "all":
        # years =  ["2018", "2017","2016postVFP","2016preVFP"]
        years =  ["2017","2016postVFP","2016preVFP"]
    else:
        years = [args.year]
    # nbins = [50, 100]
    nbins = [100]
    jet_multiplicities = [0,1,2]

    save_dict = {} # to match the config setup, it's

    test_val = 110


    for year in years:

        out_dict_by_year = {}

        for njet in jet_multiplicities:
        # for njet in [2]:
            file = ROOT.TFile(f"{year}_njet{njet}.root", "READ")
            save_path = f"./plots/{run_label}"
            workspace = file.Get("zpt_Workspace")
            # target_nbins = 50
            # for target_nbins in [25, 50, 100, 250]:
            print(f"{year} njet{njet}------------------------------------------------------------------------------------------------------")
            order = f_orders[year][f"njet{njet}"]
            out_dict_by_nbin = {}
            for target_nbins in nbins:
                fit_xmin, fit_xmax = poly_fit_ranges[year][f"njet{njet}"]
                print(f"{year} njet{njet} fit_range: {fit_xmin}, {fit_xmax}")
                # hist_data.GetXaxis().SetRangeUser(*fit_range)
                # hist_dy.GetXaxis().SetRangeUser(*fit_range)


                hist_data = workspace.obj("hist_data").Clone("hist_data_clone")
                hist_dy = workspace.obj("hist_dy").Clone("hist_dy_clone")
                orig_nbins = hist_data.GetNbinsX()
                rebin_coeff = int(orig_nbins/target_nbins)
                print(f"rebin_coeff: {rebin_coeff}")
                hist_data = hist_data.Rebin(rebin_coeff, "rebinned hist_data")
                hist_dy = hist_dy.Rebin(rebin_coeff, "rebinned hist_dy")

                hist_SF = hist_data.Clone("hist_SF")
                hist_SF.Divide(hist_dy)

                # Fit with the lower-order polynomial
                polynomial_expr = " + ".join([f"[{i}]*x**{i}" for i in range(order + 1)])
                polynomial_func = ROOT.TF1(f"poly{order}", polynomial_expr, -5, 5)
                # Define the TF1 function with the generated expression
                fit_func = polynomial_func
                _ = hist_SF.Fit(fit_func, "L S", xmin=fit_xmin, xmax=fit_xmax)
                _ = hist_SF.Fit(fit_func, "L S", xmin=fit_xmin, xmax=fit_xmax)
                fit_results = hist_SF.Fit(fit_func, "L S", xmin=fit_xmin, xmax=fit_xmax)



                # fit straight line
                horizontal_line = ROOT.TF1("horizontal_line", "[0]", fit_xmax, global_fit_xmax)
                fit_results = hist_SF.Fit(horizontal_line, "S R+", xmin=fit_xmax, xmax=global_fit_xmax)

                # good setup

                chi2 = fit_func.GetChisquare()
                ndf = fit_func.GetNDF()
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
                # horizontal_line.SetLineColor(ROOT.kGreen)  # Change color for each fit
                horizontal_line.Draw("SAME")  # Draw the fit function on the same canvas
                # polynomial_func2.Draw("SAME")  # Draw the fit function on the same canvas


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
                # canvas.SaveAs(f"{save_path}/{year}_njet{njet}_{target_nbins}_order{order}_goodnessOfFit.pdf")

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
                print("Fitted parameters and uncertainties:")
                for ix in range(num_params):
                    param_value = fit_func.GetParameter(ix)  # Fitted parameter value
                    param_error = fit_func.GetParError(ix)  # Fitted parameter uncertainty
                    # print(f"{year} njet {njet}  Parameter {i}: {param_value:} ± {param_error:}")
                    # print(f"{year} njet {njet}  Parameter {i} rel err: {param_error/ param_value *100} % ")
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
        # yaml_path = "./zpt_rewgt_params.yaml"
        # yaml_path = "./zpt_rewgt_params_amcnlo.yaml"
        yaml_path = "./zpt_rewgt_params_minnlo.yaml"
        if os.path.isfile(yaml_path): # if yaml exists, append to existing config (values with same keys will be overwirtten
            config = OmegaConf.load(yaml_path)
            config = OmegaConf.merge(config, save_dict)
        else:
            config = OmegaConf.create(save_dict)
        OmegaConf.save(config=config, f=yaml_path)
