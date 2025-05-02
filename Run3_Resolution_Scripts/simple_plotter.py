import ROOT
ROOT.EnableImplicitMT()  # Enable ROOT's implicit multi-threading

inFile = "/eos/purdue/store/user/rasharma/customNanoAOD_GT36/UL2018/SingleMuon_Run2018B/C3625AD3-1AB3-B048-9059-0807E6F51812_NanoAOD.root"
vars2plot = ["Jet_jetId"]

rdf = ROOT.RDataFrame("Events", inFile)

n_entries = rdf.Count().GetValue()
print(f"Number of entries in: {n_entries}")

jetID = rdf.Define("jetID", "Jet_jetId")
h_jetID = jetID.Histo1D(("h_jetID", "Jet ID", 10, 0, 10), "jetID")

h_jetID.GetXaxis().SetTitle("Jet ID")
h_jetID.GetYaxis().SetTitle("Entries")

#  THe histograms returned by the RDataFrame are RResultPtr<TH1D> objects.
# To get the actual histogram, we need to dereference it.
hist = h_jetID.GetPtr()

canvas = ROOT.TCanvas("canvas", "Jet ID Distribution", 800, 600)
canvas.cd()
hist.Draw()
canvas.Update()

canvas.SaveAs("jetID_distribution.png")
canvas.SaveAs("jetID_distribution.pdf")

inFile2 = "root://xcache.cms.rcac.purdue.edu//store/data/Run2018B/SingleMuon/NANOAOD/UL2018_MiniAODv2_NanoAODv9_GT36-v1/2430000/7B48C521-1C6E-1D46-ADD0-3FF2281827E7.root"

rdf2 = ROOT.RDataFrame("Events", inFile2)
n_entries2 = rdf2.Count().GetValue()
print(f"Number of entries in: {n_entries2}")
jetID2 = rdf2.Define("jetID", "Jet_jetId")
h_jetID2 = jetID2.Histo1D(("h_jetID2", "Jet ID", 10, 0, 10), "jetID")
h_jetID2.GetXaxis().SetTitle("Jet ID")
h_jetID2.GetYaxis().SetTitle("Entries")
hist2 = h_jetID2.GetPtr()

canvas.reset()
hist2.Draw("same")
canvas.SaveAs("jetID_distribution_v9.pdf")
