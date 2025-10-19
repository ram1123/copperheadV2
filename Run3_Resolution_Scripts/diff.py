import ROOT as rt
import yaml
import os

def load_and_plot(path, fields_to_load):
    """
    Load ROOT file using ROOT's RDataFrame and plot the muon pT distribution.
    """
    rt.EnableImplicitMT(); # Enable ROOT's implicit multi-threading
    rdf = rt.RDataFrame("Events", path)  # 'Events' is the tree name

    # check entries
    n_entries = rdf.Count().GetValue()
    print(f"Number of entries in {path}: {n_entries}")

    # Extracting muon pT from the ROOT file
    muon_pt = rdf.Define("muon_pt", "Muon_pt")
    h_muon_pt = muon_pt.Histo1D(("h_muon_pt", "Muon pT Distribution", 100, 0, 500), "muon_pt")

    h_muon_pt.GetXaxis().SetTitle("Muon pT (GeV)")
    h_muon_pt.GetYaxis().SetTitle("Entries")

    # Dereference the RResultPtr to get the TH1D histogram
    hist = h_muon_pt.GetPtr()

    return hist

def main():
    # Load the configuration from the YAML file
    with open("config/plot_config_nanoV12vsV9.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Read configuration parameters
    input_paths_labels = config["input_paths_labels"]
    fields_to_load = config["fields_to_load"]

    # Store histograms for v9 and v12
    hist = {}
    for label, path in input_paths_labels.items():
        print(f"Loading data for {label} from {path}")
        hist[label] = load_and_plot(path, fields_to_load)

    rt.gStyle.SetOptStat(0)
    # Create the canvas for comparison
    canvas = rt.TCanvas("canvas", "Muon pT Distribution", 800, 600)

    # Create upper and lower pads
    upper_pad = rt.TPad("upper_pad", "Upper Pad", 0.0, 0.3, 1.0, 1.0)  # y-range from 0.3 to 1
    lower_pad = rt.TPad("lower_pad", "Lower Pad", 0.0, 0.0, 1.0, 0.3)  # y-range from 0 to 0.3

    # Draw the pads
    upper_pad.Draw()
    lower_pad.Draw()

    # Set the upper pad
    upper_pad.cd()
    hist["v9"].SetLineColor(rt.kRed)
    hist["v12"].SetLineColor(rt.kBlue)
    hist["v9"].SetTitle("Muon pT Distribution")
    hist["v9"].Scale(1.0 / hist["v9"].Integral())
    hist["v12"].Scale(1.0 / hist["v12"].Integral())

    # Draw the histograms on the upper pad
    hist["v9"].Draw("HIST")
    hist["v12"].Draw("HIST SAME")
    hist["v9"].GetXaxis().SetTitle("Muon pT (GeV)")
    hist["v9"].GetYaxis().SetTitle("Normalized Entries")
    legend = rt.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(hist["v9"], "NanoAOD v9", "l")
    legend.AddEntry(hist["v12"], "NanoAOD v12", "l")
    legend.Draw()

    # Set the lower pad
    lower_pad.cd()
    hist_diff = hist["v9"].Clone("hist_diff")
    hist_diff.Add(hist["v12"], -1)  # Subtract v12 from v9 (v9 - v12)
    hist_diff.SetLineColor(rt.kGreen)
    hist_diff.SetLineWidth(2)
    hist_diff.SetTitle("Difference (v9 - v12)")

    # Draw the difference plot in the lower pad
    hist_diff.Draw("HIST")
    hist_diff.GetXaxis().SetTitle("Muon pT (GeV)")
    hist_diff.GetYaxis().SetTitle("Difference")
    hist_diff.GetYaxis().SetRangeUser(-0.01, 0.01)  # Adjust Y-axis range for difference

    # Save the canvas with both histograms and difference plot
    canvas.SaveAs("muon_pt_distribution_comparison_diff.pdf")
    print("Saved comparison plot as muon_pt_distribution_comparison_diff.pdf")

if __name__ == "__main__":
    main()
