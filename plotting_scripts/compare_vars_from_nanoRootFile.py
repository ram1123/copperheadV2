"""
This scripts reads two nanoAOD root files and compares the variables defined in the variables.yaml file.
For comparison it plots two pads, one with the two histograms and the other with the difference between the two histograms.

INFO:
1. Hardcoded info: "v9" and "v12" strings.

"""
import ROOT as rt
import yaml
import os

with open("variables.yaml", "r") as f:
    config = yaml.safe_load(f)

file1_path = config["input_files"]["v9"]
file2_path = config["input_files"]["v12"]

rt.EnableImplicitMT()  # Enable ROOT's implicit multi-threading
rdf1 = rt.RDataFrame("Events", file1_path)  # 'Events' is the tree name
rdf2 = rt.RDataFrame("Events", file2_path)  # 'Events' is the tree name

plots_path = config["outputDir"]
if not os.path.exists(plots_path):
    os.makedirs(plots_path)

for var, var_info in config["variables"].items():
    print(f"Variable: {var}, Info: {var_info}")

    # Loop through the variables defined in YAML file
    var1 = rdf1.Redefine(var, f"{var}")
    h_var1 = var1.Histo1D((f"h_{var}_1", var_info["Title"], var_info["Range"][0], var_info["Range"][1], var_info["Range"][2]), var)
    h_var1.GetXaxis().SetTitle(f"{var_info['Title']}")
    h_var1.GetYaxis().SetTitle("Entries")
    h_var1.SetLineColor(rt.kRed)
    h_var1 = h_var1.GetPtr()

    var2 = rdf2.Redefine(var, var)
    h_var2 = var2.Histo1D((f"h_{var}_2", var_info["Title"], var_info["Range"][0], var_info["Range"][1], var_info["Range"][2]), var)
    h_var2.GetXaxis().SetTitle(f"{var_info['Title']}")
    h_var2.GetYaxis().SetTitle("Entries")
    h_var2.SetLineColor(rt.kBlue)
    h_var2 = h_var2.GetPtr()

    rt.gStyle.SetOptStat(0)

    canvas = rt.TCanvas("canvas", "Muon pT Distribution", 800, 600)

    # Create upper and lower pads
    upper_pad = rt.TPad("upper_pad", "Upper Pad", 0.0, 0.3, 1.0, 1.0)  # y-range from 0.3 to 1
    lower_pad = rt.TPad("lower_pad", "Lower Pad", 0.0, 0.0, 1.0, 0.3)  # y-range from 0 to 0.3

    # Draw the pads
    upper_pad.Draw()
    lower_pad.Draw()

    # Set the upper pad
    upper_pad.cd()
    h_var1.Draw("")
    h_var2.Draw("SAME")
    h_var1.GetXaxis().SetTitle(f"{var_info['Title']}")
    h_var1.GetYaxis().SetTitle("Entries")
    h_var1.GetYaxis().SetRangeUser(0, 1.5 * max(h_var1.GetMaximum(), h_var2.GetMaximum()))

    legend = rt.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(h_var1, "v9", "l")
    legend.AddEntry(h_var2, "v12", "l")
    legend.Draw()
    canvas.Update()

    lower_pad.cd()
    hist_diff = h_var1.Clone("hist_diff")
    hist_diff.Add(h_var2, -1)
    hist_diff.SetTitle("Difference")
    hist_diff.GetXaxis().SetTitle(f"{var_info['Title']}")
    hist_diff.GetYaxis().SetTitle("Difference")
    hist_diff.GetYaxis().SetRangeUser(var_info["RatioPlot"][0], var_info["RatioPlot"][1])
    hist_diff.Draw("")
    # also draw a horizontal line at 0
    line = rt.TLine(hist_diff.GetXaxis().GetXmin(), 0, hist_diff.GetXaxis().GetXmax(), 0)
    line.SetLineColor(rt.kBlack)
    line.SetLineWidth(2)
    line.SetLineStyle(2)  # Dashed line
    line.Draw("SAME")

    # Save the canvas as a PDF
    canvas.SaveAs(f"{plots_path}/{var}_comparison.pdf")
    # Clean up
    del h_var1
    del h_var2
    del hist_diff
    del canvas
    del upper_pad
    del lower_pad
    del legend
