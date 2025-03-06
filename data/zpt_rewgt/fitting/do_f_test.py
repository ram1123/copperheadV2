"""
F-test
"""
import ROOT
from scipy.stats import f

# define cut ranges to do polynomial fits. pt ranges beyond that point we fit with a constant
poly_fit_ranges = {
    "2018" : {
        "njet0" : [0, 85],
        "njet1" : [0, 50],
        "njet2" : [0, 50],
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

# year = "2016preVFP"
# year = "2016postVFP"
year = "2017"
# njet = 0
run_label = "WithPurdueZptWgt_DYWithoutLHECut_16Feb_AllYear" # shar1172

for njet in [0,1,2]:
    # text file to dump the results
    outTextFile = open(f"results_{year}_njet{njet}.txt", "w")
    inDirectory = f"./plots_WS_{run_label}"
    file = ROOT.TFile(f"{inDirectory}/{year}_njet{njet}.root", "READ")
    save_path = "./plots_fTest"

    workspace = file.Get("zpt_Workspace")
    # target_nbins = 50
    # for target_nbins in [25, 50, 100, 250]:
    print(f"{year} njet{njet}------------------------------------------------------------------------------------------------------")
    for target_nbins in [50, 100]:
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



        # Draw the histogram and fit
        canvas = ROOT.TCanvas("canvas", f"{target_nbins} bins Data and DY", 800, 600)
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
        # canvas.SaveAs(f"{save_path}/{year}_njet{njet}_{target_nbins}Bins_DataDy_Hist.png")
        # canvas.SaveAs(f"{save_path}/{year}_njet{njet}_{target_nbins}Bins_DataDy_Hist.pdf")

        canvas = ROOT.TCanvas("canvas", f"{target_nbins} bins SF hist", 800, 600)
        hist_SF.SetTitle(f"njet {njet} {target_nbins} bins SF")
        hist_SF.SetMinimum(0.5)  # Set the lower bound of the Y-axis
        hist_SF.SetMaximum(4)  # Set the upper bound of the Y-axis
        hist_SF.Draw()

        canvas.Update()
        # canvas.SaveAs(f"{save_path}/{year}_njet{njet}_{target_nbins}Bins_SF_Hist.png")
        # canvas.SaveAs(f"{save_path}/{year}_njet{njet}_{target_nbins}Bins_SF_Hist.pdf")


        dimuon_pt = ROOT.RooRealVar("dimuon_pt", "Dimuon pT", 0, 200)

        # Convert the TH1F histogram to a RooDataHist
        roo_hist_SF = ROOT.RooDataHist("roo_hist", "RooFit Histogram", ROOT.RooArgList(dimuon_pt), hist_SF)

        # Print information about the RooDataHist
        roo_hist_SF.Print()



        for order in range(2,8):
            # Define two polynomial orders
            order_low = order
            order_high = order + 1

            # Fit with the lower-order polynomial
            polynomial_expr = " + ".join([f"[{i}]*x**{i}" for i in range(order_low + 1)])
            polynomial_func = ROOT.TF1(f"poly{order}", polynomial_expr, -5, 5)
            # Define the TF1 function with the generated expression
            fit_func_low = polynomial_func
            # _ = hist_SF.Fit(fit_func_low, "S")
            # _ = hist_SF.Fit(fit_func_low, "S")
            # fit_low = hist_SF.Fit(fit_func_low, "S")
            _ = hist_SF.Fit(fit_func_low, "L ", xmin=fit_xmin, xmax=fit_xmax)
            _ = hist_SF.Fit(fit_func_low, "L ", xmin=fit_xmin, xmax=fit_xmax)
            fit_low = hist_SF.Fit(fit_func_low, "", xmin=fit_xmin, xmax=fit_xmax)


            chi2_low = fit_func_low.GetChisquare()
            ndf_low = fit_func_low.GetNDF()
            # log_likelihood_low = fit_func_low.GetLogLikelihood()
            # print(f"log_likelihood_low: {log_likelihood_low}")

            # Fit with the higher-order polynomial
            polynomial_expr = " + ".join([f"[{i}]*x**{i}" for i in range(order_high + 1)])
            # Define the TF1 function with the generated expression
            polynomial_func = ROOT.TF1(f"poly{order}", polynomial_expr, -5, 5)
            # Define the TF1 function with the generated expression
            fit_func_high = polynomial_func
            _ = hist_SF.Fit(fit_func_high, "L ", xmin=fit_xmin, xmax=fit_xmax)
            _ = hist_SF.Fit(fit_func_high, "L ", xmin=fit_xmin, xmax=fit_xmax)
            fit_high = hist_SF.Fit(fit_func_high, "", xmin=fit_xmin, xmax=fit_xmax)

            chi2_high = fit_func_high.GetChisquare()
            ndf_high = fit_func_high.GetNDF()


            # Calculate F-statistic
            delta_chi2 = chi2_low - chi2_high
            delta_dof = -(ndf_high - ndf_low)
            f_statistic = delta_chi2 / chi2_high * (ndf_high) / delta_dof
            print(f"(target_nbins - order_high): {(target_nbins - order_high)}")
            print(f"ndf_high: {ndf_high}")
            print(f"delta_dof: {delta_dof}")
            print(f"f_statistic: {f_statistic}")
            # Calculate the p-value (use scipy.stats.f for F-distribution)
            # p_value = 1 - f.cdf(f_statistic, delta_dof, ndf_high)
            p_value = 1 - f.cdf(f_statistic, delta_dof, ndf_high)
            # p_value = ROOT.TMath.Prob(f_statistic, delta_dof)
            # delta_nll = 2*(low_nll-high_nll) # line 1552 if AN-19-124



            # Print results
            print(f"Lower-order {target_nbins} bins polynomial (pol{order_low}): chi2 = {chi2_low}, ndf = {ndf_low}, chi2_dof = {chi2_low/ndf_low}")
            print(f"Higher-order {target_nbins} bins polynomial (pol{order_high}): chi2 = {chi2_high}, ndf = {ndf_high}, chi2_dof = {chi2_high/ndf_high}")
            print(f"F-statistic {target_nbins} bins: {f_statistic}")
            print(f"P-value {target_nbins} bins: {p_value}")

            if p_value < 0.05:  # Typically, p-value < 0.05 indicates significant improvement
                print(f"Higher-order {order_high} polynomial significantly improves the fit versus {order_low}. chi2_low: {chi2_low/ndf_low} vs chi2_high: {chi2_high/ndf_high}")
                outTextFile.write(f"{year} njet{njet} {target_nbins} bins: Higher-order {order_high} polynomial significantly improves the fit versus {order_low}. chi2_low: {chi2_low/ndf_low} vs chi2_high: {chi2_high/ndf_high}\n")
            else:
                # print(f"Higher-order {order_high} polynomial does not significantly improve the fit.")
                pass

        # hist_data.Delete()
        # hist_dy.Delete()
        # hist_SF.Delete()
        # roo_hist_SF.Delete()
        # canvas.Close()
    file.Close()
    outTextFile.close()
