import ROOT as rt
import sys
from os import listdir
from os.path import isfile, join

# don't show sstats box
rt.gStyle.SetOptStat(0000)

file_path = "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/SingleMuon_Run2018C/"
onlyfiles = [f for f in listdir(file_path) if isfile(join(file_path, f))]
# print(onlyfiles)



# file_path = "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/SingleMuon_Run2018C/61FAB949-40CE-B14D-82A1-1AA0955735CB_NanoAOD.root"
# Create a TChain to load all ROOT files
chain = rt.TChain("Events")
# chain.Add(file_path)

for i in range(108):
    # print(file_name)
    chain.Add(file_path + onlyfiles[i])

# get entries
print("Number of entries in the chain: ", chain.GetEntries())

# chain.Show()  # Show the structure of the chain to understand the available branches

# Define the pT bin filters
# filters = {
#     "B": "abs(Muon_eta) < 0.9",
#     "O": "abs(Muon_eta) > 0.9 && abs(Muon_eta) < 1.8",
#     "E": "abs(Muon_eta) > 1.8 && abs(Muon_eta) < 2.4"
# }

# pT filters
filters = {
    "pT0_20_B": "((Muon_pt > 0 && Muon_pt <= 20) && (abs(Muon_eta) < 0.9))",
    "pT20_50_B": "((Muon_pt > 20 && Muon_pt <= 50) && (abs(Muon_eta) < 0.9))",
    "pT50_100_B": "((Muon_pt > 50 && Muon_pt <= 100) && (abs(Muon_eta) < 0.9))",
    "pT100_200_B": "((Muon_pt > 100 && Muon_pt <= 200) && (abs(Muon_eta) < 0.9))",
    "pT200_5000_B": "((Muon_pt > 200 && Muon_pt <= 5000) && (abs(Muon_eta) < 0.9))",

    "pT0_20_O": "((Muon_pt > 0 && Muon_pt <= 20) && (abs(Muon_eta) > 0.9 && abs(Muon_eta) < 1.8))",
    "pT20_50_O": "((Muon_pt > 20 && Muon_pt <= 50) && (abs(Muon_eta) > 0.9 && abs(Muon_eta) < 1.8))",
    "pT50_100_O": "((Muon_pt > 50 && Muon_pt <= 100) && (abs(Muon_eta) > 0.9 && abs(Muon_eta) < 1.8))",
    "pT100_200_O": "((Muon_pt > 100 && Muon_pt <= 200) && (abs(Muon_eta) > 0.9 && abs(Muon_eta) < 1.8))",
    "pT200_5000_O": "((Muon_pt > 200 && Muon_pt <= 5000) && (abs(Muon_eta) > 0.9 && abs(Muon_eta) < 1.8))",

    "pT0_20_E": "((Muon_pt > 0 && Muon_pt <= 20) && (abs(Muon_eta) > 1.8 && abs(Muon_eta) < 2.4))",
    "pT20_50_E": "((Muon_pt > 20 && Muon_pt <= 50) && (abs(Muon_eta) > 1.8 && abs(Muon_eta) < 2.4))",
    "pT50_100_E": "((Muon_pt > 50 && Muon_pt <= 100) && (abs(Muon_eta) > 1.8 && abs(Muon_eta) < 2.4))",
    "pT100_200_E": "((Muon_pt > 100 && Muon_pt <= 200) && (abs(Muon_eta) > 1.8 && abs(Muon_eta) < 2.4))",
    "pT200_5000_E": "((Muon_pt > 200 && Muon_pt <= 5000) && (abs(Muon_eta) > 1.8 && abs(Muon_eta) < 2.4))"
}

range = {
    "pT0_20_B": [50,0, 20],
    "pT20_50_B": [50,20, 50],
    "pT50_100_B": [50,50, 100],
    "pT100_200_B": [50,100, 200],
    "pT200_5000_B": [50,200, 5000],

    "pT0_20_O": [50,0, 20],
    "pT20_50_O": [50,20, 50],
    "pT50_100_O": [50,50, 100],
    "pT100_200_O": [50,100, 200],
    "pT200_5000_O": [50,200, 5000],

    "pT0_20_E": [50,0, 20],
    "pT20_50_E": [50,20, 50],
    "pT50_100_E": [50,50, 100],
    "pT100_200_E": [50,100, 200],
    "pT200_5000_E": [50,200, 5000],
}

# Prepare histograms
histograms = {}

# Loop over pT bins and create histograms
for bin_name, cut in filters.items():
    print(f"===> Processing bin: {bin_name} with cut: {cut}")
    # Create histograms for Muon_pt and Muon_bsConstrainedPt
    hist_muon_pt = rt.TH1D("Muon_pt_" + bin_name, "Muon pT (" + bin_name + ")", range[bin_name][0], range[bin_name][1], range[bin_name][2])
    hist_muon_bs_pt = rt.TH1D("Muon_bsConstrainedPt_" + bin_name, "BSC Muon pT (" + bin_name + ")", range[bin_name][0], range[bin_name][1], range[bin_name][2])

    # Apply the filter for each bin and fill histograms
    chain.Draw("Muon_pt >> Muon_pt_" + bin_name, cut)
    chain.Draw("Muon_bsConstrainedPt >> Muon_bsConstrainedPt_" + bin_name, cut.replace("Muon_pt", "Muon_bsConstrainedPt"))

    # Store histograms
    histograms[bin_name] = {"Muon_pt": hist_muon_pt, "Muon_bsConstrainedPt": hist_muon_bs_pt}

# Now create a canvas to draw the histograms for comparison
canvas = rt.TCanvas("canvas", "Muon pT Comparison", 800, 600)
# canvas.Divide(2, 2)  # Divide the canvas into sub-pads for each bin

print(histograms)

# Draw histograms for each bin
for i, bin_name in enumerate(filters.keys(), 1):
    hist_muon_pt = histograms[bin_name]["Muon_pt"]
    hist_muon_bs_pt = histograms[bin_name]["Muon_bsConstrainedPt"]

    # Draw the histograms
    hist_muon_pt.SetLineColor(rt.kBlue)
    hist_muon_bs_pt.SetLineColor(rt.kRed)

    hist_muon_pt.GetXaxis().SetTitle("Muon pT [GeV]")
    hist_muon_pt.GetYaxis().SetTitle("Entries")
    hist_muon_pt.GetYaxis().SetRangeUser(0, 1.5 * max(hist_muon_pt.GetMaximum(), hist_muon_bs_pt.GetMaximum()))


    ratio_plot = rt.TRatioPlot(hist_muon_pt, hist_muon_bs_pt)
    ratio_plot.Draw()
    ratio_plot.GetLowerRefYaxis().SetTitle("Ratio")
    ratio_plot.GetLowerRefYaxis().SetRangeUser(0.0, 2.)
    ratio_plot.GetLowerRefGraph().SetMinimum(0.0)
    ratio_plot.GetLowerRefGraph().SetMaximum(2.)

    ratio_plot.GetUpperPad().cd()
    legend = rt.TLegend(0.7, 0.7, 0.95, 0.9)
    legend.AddEntry(hist_muon_pt, "Muon pT", "l")
    legend.AddEntry(hist_muon_bs_pt, "BSC Muon pT", "l")
    legend.Draw()

    # Save the canvas to a file
    canvas.SaveAs(f"MuonComparison_pTBins_{bin_name}.pdf")
    canvas.Clear()

# Clean up
for bin_name in filters.keys():
    histograms[bin_name]["Muon_pt"].Delete()
    histograms[bin_name]["Muon_bsConstrainedPt"].Delete()

canvas.Clear()
