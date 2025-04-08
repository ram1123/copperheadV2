import ROOT as rt
import sys

rt.EnableImplicitMT()  # Enable ROOT's implicit multi-threading
rdf = rt.RDataFrame("Events", "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/SingleMuon_Run2018C/*.root")

# cuts = {
#     "B": "abs(Muon_eta) < 0.9",
#     "O": "abs(Muon_eta) > 0.9 && abs(Muon_eta) < 1.8",
#     "E": "abs(Muon_eta) > 1.8 && abs(Muon_eta) < 2.4"
# }

rdf_b = rdf
# rdf_b = rdf.Filter("abs(Muon_eta) < 0.9")  # Apply the cut to the RDataFrame



canvas = rt.TCanvas("canvas", "Comparison", 800, 600)
rt.gStyle.SetOptStat(0)  # Hide stats

histA = rdf_b.Histo1D(("Muon_pt", "Muon pT", 100, 0, 200), "Muon_pt")
histB = rdf_b.Histo1D(("Muon_bsConstrainedPt", "BSC Muon pT", 100, 0, 200), "Muon_bsConstrainedPt")
histA.SetLineColor(rt.kRed)
histB.SetLineColor(rt.kBlue)
histA.SetTitle("Muon pT")

histA = histA.GetPtr()
histB = histB.GetPtr()

ratio_plot = rt.TRatioPlot(histA, histB)
ratio_plot.Draw()
ratio_plot.GetLowerRefYaxis().SetTitle("Ratio")
ratio_plot.GetLowerRefYaxis().SetRangeUser(0.65, 1.3)
ratio_plot.GetLowerRefGraph().SetMinimum(0.65)
ratio_plot.GetLowerRefGraph().SetMaximum(1.3)

ratio_plot.GetUpperPad().cd()
legend = rt.TLegend(0.7, 0.7, 0.95, 0.9)
legend.AddEntry(histA, "Muon pT", "l")
legend.AddEntry(histB, "BSC Muon pT", "l")
legend.Draw()

canvas.SaveAs("comparison_plot_MuonpT_Barrel.pdf")
histA.Delete()  # Clean up the histogram
histB.Delete()  # Clean up the histogram
canvas.Clear()  # Clear the canvas

histA = rdf_b.Histo1D(("Muon_ptErr", "Muon pTErr", 100, 0, 5), "Muon_ptErr")
histB = rdf_b.Histo1D(("Muon_bsConstrainedPtErr", "BSC Muon pTErr", 100, 0, 5), "Muon_bsConstrainedPtErr")
histA.SetLineColor(rt.kRed)
histB.SetLineColor(rt.kBlue)
histA.SetTitle("Muon pT")

histA = histA.GetPtr()
histB = histB.GetPtr()


ratio_plot = rt.TRatioPlot(histA, histB)
ratio_plot.Draw()
ratio_plot.GetLowerRefYaxis().SetTitle("Ratio")
ratio_plot.GetLowerRefYaxis().SetRangeUser(0.5, 2.0)
ratio_plot.GetLowerRefGraph().SetMinimum(0.5)
ratio_plot.GetLowerRefGraph().SetMaximum(2.0)

ratio_plot.GetUpperPad().cd()
legend = rt.TLegend(0.7, 0.7, 0.95, 0.9)
legend.AddEntry(histA, "Muon pTErr", "l")
legend.AddEntry(histB, "BSC Muon pTErr", "l")
legend.Draw()

canvas.SaveAs("comparison_plot_MuonpTErr_Barrel.pdf")
histA.Delete()  # Clean up the histogram
histB.Delete()  # Clean up the histogram
canvas.Clear()  # Clear the canvas
