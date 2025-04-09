import ROOT
from scipy.stats import f
from omegaconf import OmegaConf
import os
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--label", dest="label", default=None, action="store", help="save path to store stage1 output files")
parser.add_argument("-y", "--year", dest="year", default="all", action="store", help="string value of year we are calculating")
parser.add_argument("-save", "--plot_path", dest="plot_path", default="plots", action="store", help="save path to store plots")
parser.add_argument("--nbins", type=str, default="CustomBins", help="Number of bins")
parser.add_argument("--njet", type=int, nargs="+", default=[0, 1, 2], help="Number of jets")
parser.add_argument("--outAppend", type=str, default="", help="Append to output file name")
args = parser.parse_args()

run_label = args.label
if args.year == "all":
    years = ["2018", "2017", "2016postVFP", "2016preVFP"]
else:
    years = [args.year]

save_dict = {}
global_fit_xmax = 200

for year in years:
    inPath = f"{args.plot_path}/{run_label}/{year}"
    save_path = f"{args.plot_path}/{run_label}/{year}/gof_{args.outAppend}"
    os.makedirs(save_path, exist_ok=True)

    # Load YAML config with optimized fit order and range
    fit_config_path = f"{args.plot_path}/{run_label}/{year}/fTest_{args.outAppend}/zpt_fit_config.yaml"
    with open(fit_config_path, "r") as f:
        fit_config = yaml.safe_load(f)

    out_dict_by_year = {}

    for njet in args.njet:
        jet_key = f"njet{njet}"
        order = fit_config[year][jet_key]["order"]
        fit_xmin, fit_xmax = fit_config[year][jet_key]["fit_range"]

        file = ROOT.TFile(f"{inPath}/{year}_njet{njet}.root", "READ")
        workspace = file.Get("zpt_Workspace")

        out_dict_by_nbin = {}
        for target_nbins in [args.nbins]:
            hist_data = workspace.obj("hist_data").Clone("hist_data_clone")
            hist_dy = workspace.obj("hist_dy").Clone("hist_dy_clone")
            orig_nbins = hist_data.GetNbinsX()
            rebin_coeff = int(orig_nbins / int(target_nbins))
            print(f"rebin_coeff: {rebin_coeff}")
            hist_data = hist_data.Rebin(rebin_coeff, "rebinned hist_data")
            hist_dy = hist_dy.Rebin(rebin_coeff, "rebinned hist_dy")

            hist_SF = hist_data.Clone("hist_SF")
            hist_SF.Divide(hist_dy)

            # Fit with the lower-order polynomial
            polynomial_expr = " + ".join([f"[{i}]*x**{i}" for i in range(order + 1)])
            # Define the TF1 function with the generated expression
            fit_func = ROOT.TF1(f"poly{order}", polynomial_expr, 0, fit_xmax)
            _ = hist_SF.Fit(fit_func, "L S", xmin=fit_xmin, xmax=fit_xmax)
            _ = hist_SF.Fit(fit_func, "L S", xmin=fit_xmin, xmax=fit_xmax)
            fit_results = hist_SF.Fit(fit_func, "L S", xmin=fit_xmin, xmax=fit_xmax)

            # Fit straight line beyond fit_xmax
            horizontal_line = ROOT.TF1("horizontal_line", "[0]", fit_xmax, global_fit_xmax)
            fit_results = hist_SF.Fit(horizontal_line, "S R+", xmin=fit_xmax, xmax=global_fit_xmax)

            # Calculate chi2 and p-value
            chi2 = fit_func.GetChisquare()
            ndf = fit_func.GetNDF()
            chi2_dof = chi2 / ndf
            p_value = ROOT.TMath.Prob(chi2, ndf)

            # Plotting
            canvas = ROOT.TCanvas("canvas", f"{target_nbins} bins SF hist", 800, 800)
            canvas.Divide(1, 2)
            canvas.cd(1)
            hist_SF.SetTitle(f"{order} order poly, njet {njet}, {target_nbins} bins SF")
            hist_SF.SetLineColor(ROOT.kBlue)
            hist_SF.SetMinimum(0.25)
            hist_SF.SetMaximum(4)
            hist_SF.Draw()
            fit_func.SetLineColor(ROOT.kRed)
            fit_func.Draw("SAME")
            horizontal_line.SetLineColor(ROOT.kGreen)
            horizontal_line.Draw("SAME")

            # Add a legend
            legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
            legend.AddEntry(hist_SF, "hist_SF", "l")
            legend.AddEntry(fit_func, "poly fit", "l")
            legend.AddEntry(horizontal_line, "horizontal line fit", "l")
            legend.Draw()

            text_box = ROOT.TPaveText(0.0, 0.7, 0.4, 0.9, "NDC")
            text_box.SetFillColor(0)
            text_box.SetBorderSize(1)
            text_box.AddText("Fit Results:")
            text_box.AddText(f"chi2 / DOF = {float('%.4g' % chi2_dof)}")
            text_box.AddText(f"P value = {float('%.4g' % p_value)}")
            text_box.Draw()

            # Pull distribution
            canvas.cd(2)
            ROOT.gPad.SetPad(0, 0, 1, 0.4)
            ROOT.gPad.SetGrid()
            pull_hist = ROOT.TH1D("pull_hist", "Pull Distribution", hist_SF.GetNbinsX(), hist_SF.GetXaxis().GetXmin(), hist_SF.GetXaxis().GetXmax())
            for i in range(1, hist_SF.GetNbinsX() + 1):
                data = hist_SF.GetBinContent(i)
                error = hist_SF.GetBinError(i)
                x_value = hist_SF.GetBinCenter(i)
                if x_value <= fit_xmax:
                    fit_value = fit_func.Eval(x_value)
                else:
                    fit_value = horizontal_line.Eval(x_value)
                if error > 0:
                    pull = (data - fit_value) / error
                    pull_hist.SetBinContent(i, pull)

            pull_hist.SetMarkerStyle(20)
            pull_hist.SetTitle("Pull Distribution;X-axis;Pull")
            pull_hist.Draw("P")
            canvas.SaveAs(f"{save_path}/{year}_njet{njet}_{target_nbins}_order{order}_goodnessOfFit.pdf")

            # ---------------------------------------------------
            # save the fit coeffs
            # ---------------------------------------------------

            max_order = 5
            param_dict = {f"p{i}": 0.0 for i in range(max_order + 1)}
            param_dict.update({f"p{i}_err": 0.0 for i in range(max_order + 1)})
            for i in range(fit_func.GetNpar()):
                param_dict[f"p{i}"] = fit_func.GetParameter(i)
                param_dict[f"p{i}_err"] = fit_func.GetParError(i)
            param_dict["horizontal_c0"] = horizontal_line.GetParameter(0)
            param_dict["polynomial_range"] = {"x_min": fit_xmin, "x_max": fit_xmax}

            out_dict_by_nbin[target_nbins] = param_dict
        out_dict_by_year[f"njet_{njet}"] = out_dict_by_nbin

    save_dict[year] = out_dict_by_year
    yaml_path = "./zpt_rewgt_params_minnlo.yaml"
    if os.path.isfile(yaml_path):
        config = OmegaConf.load(yaml_path)
        config = OmegaConf.merge(config, save_dict)
    else:
        config = OmegaConf.create(save_dict)
    OmegaConf.save(config=config, f=yaml_path)
    print(f"Saved to {yaml_path}")
