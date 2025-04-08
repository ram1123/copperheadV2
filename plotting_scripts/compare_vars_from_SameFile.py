"""
This scripts reads two nanoAOD root files and compares the variables defined in the variables.yaml file.
For comparison it plots two pads, one with the two histograms and the other with the difference between the two histograms.

INFO:
1. Hardcoded info: "v9" and "v12" strings.

"""
import ROOT as rt
import yaml
import os

file1_path = "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/SingleMuon_Run2018C/*.root"

rt.EnableImplicitMT()  # Enable ROOT's implicit multi-threading
rdf1 = rt.RDataFrame("Events", file1_path)  # 'Events' is the tree name

plots_path = "plots/nanoAODv12/SingleMuon_Run2018C/"
if not os.path.exists(plots_path):
    os.makedirs(plots_path)

variables_to_compare = ["Muon_pt", "Muon_bsConstrainedPt"]
# variables_to_compare = ["Muon_ptErr", "Muon_bsConstrainedPtErr"]

# add cuts of different rapidity regions
# B = 0.9
# O = 0.9 < abs(Muon_eta) < 1.8
# E = 1.8 < abs(Muon_eta) < 2.4

# cuts = {
#     "B": "abs(Muon_eta) < 0.9",
#     "O": "abs(Muon_eta) > 0.9 && abs(Muon_eta) < 1.8",
#     "E": "abs(Muon_eta) > 1.8 && abs(Muon_eta) < 2.4"
# }

# Loop through the cuts and apply them to the RDataFrame
# for cut_name, cut_condition in cuts.items():
    # rdf1 = rdf1.Filter(cut_condition, cut_name)
    # print(f"Applied cut: {cut_name} with condition: {cut_condition}")

# rdf1 = rdf1.Filter("abs(Muon_eta) < 0.9", "B")  # Example: Apply the cut for B region
# rdf1 = rdf1.Filter("abs(Muon_eta) > 0.9 && abs(Muon_eta) < 1.8", "O")  # Example: Apply the cut for O region
# rdf1 = rdf1.Filter("abs(Muon_eta) > 1.8 && abs(Muon_eta) < 2.4", "E")  # Example: Apply the cut for E region


# Loop through the variables defined in YAML file
var1 = rdf1.Redefine(variables_to_compare[0], f"{variables_to_compare[0]}")
h_var1 = var1.Histo1D((f"h_{variables_to_compare[0]}_1", "Muon pT", 100, 0, 200), variables_to_compare[0])
h_var1.GetXaxis().SetTitle("Muon pT [GeV]")
h_var1.GetYaxis().SetTitle("Entries")
h_var1.SetLineColor(rt.kRed)
h_var1 = h_var1.GetPtr()

var2 = rdf1.Redefine(variables_to_compare[1], variables_to_compare[1])
h_var2 = var2.Histo1D((f"h_{variables_to_compare[1]}_2", "BSC Muon pT", 100, 0, 200), variables_to_compare[1])
h_var2.GetXaxis().SetTitle("BSC Muon pT [GeV]")
h_var2.GetYaxis().SetTitle("Entries")
h_var2.SetLineColor(rt.kBlue)
h_var2 = h_var2.GetPtr()

rt.gStyle.SetOptStat(0)

canvas = rt.TCanvas("canvas", "Muon pT Distribution", 800, 600)

ratio_plot = rt.TRatioPlot(h_var1, h_var2)
ratio_plot.Draw()
ratio_plot.GetLowerRefYaxis().SetTitle("Ratio")
ratio_plot.GetLowerRefYaxis().SetRangeUser(0.65, 1.3)
ratio_plot.GetLowerRefGraph().SetMinimum(0.65)
ratio_plot.GetLowerRefGraph().SetMaximum(1.3)

ratio_plot.GetUpperPad().cd()
legend = rt.TLegend(0.7, 0.7, 0.95, 0.9)
legend.AddEntry(h_var1, "Muon pT", "l")
legend.AddEntry(h_var2, "BSC Muon pT", "l")
legend.Draw()

# Save the canvas as a PDF
canvas.SaveAs(f"{plots_path}/{variables_to_compare[0]}_comparison.pdf")
# Clean up
del h_var1
del h_var2
del canvas
