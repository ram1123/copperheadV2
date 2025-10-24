import ROOT as rt
import yaml
import os

def load_and_plot(path, fields_to_load):
    """
    Load ROOT file using ROOT's RDataFrame and plot the muon pT distribution.
    """
    rdf = rt.RDataFrame("Events", path)  # 'Events' is the tree name

    # check entries
    n_entries = rdf.Count().GetValue()
    print(f"Number of entries in {path}: {n_entries}")

    muon_pt = rdf.Define("muon_pt", "Muon_pt")
    h_muon_pt = muon_pt.Histo1D(("h_muon_pt", "Muon pT Distribution", 100, 0, 500), "muon_pt")

    h_muon_pt.GetXaxis().SetTitle("Muon pT (GeV)")
    h_muon_pt.GetYaxis().SetTitle("Entries")

    #  THe histograms returned by the RDataFrame are RResultPtr<TH1D> objects.
    # To get the actual histogram, we need to dereference it.
    hist = h_muon_pt.GetPtr()

    return hist

def main():
    # Load the configuration from the YAML file
    with open("config/plot_config_nanoV12vsV9.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Read configuration parameters
    input_paths_labels = config["input_paths_labels"]
    fields_to_load = config["fields_to_load"]

    hist = {}
    for label, path in input_paths_labels.items():
        print(f"Loading data for {label} from {path}")
        hist[label] = load_and_plot(path, fields_to_load)

    canvas = rt.TCanvas("canvas", "Muon pT Distribution", 800, 600)
    canvas.cd()
    legend = rt.TLegend(0.7, 0.7, 0.9, 0.9)

    hist["v9"].SetLineColor(rt.kRed)
    hist["v12"].SetLineColor(rt.kBlue)

    # Normalize the histograms
    hist["v9"].Scale(1.0 / hist["v9"].Integral())
    hist["v12"].Scale(1.0 / hist["v12"].Integral())

    print(type(hist["v9"]))
    print(type(hist["v12"]))

    ratio_plot = rt.TRatioPlot(hist["v9"], hist["v12"])
    ratio_plot.Draw()
    ratio_plot.GetLowerRefYaxis().SetTitle("Ratio")
    ratio_plot.GetLowerRefYaxis().SetRangeUser(0.5, 1.5)  # Adjust the ratio range as needed
    ratio_plot.GetLowerRefGraph().SetMinimum(0.5)
    ratio_plot.GetLowerRefGraph().SetMaximum(1.5)

    ratio_plot.GetUpperPad().cd()
    # First draw the histograms explicitly (required for TRatioPlot to work)
    # hist["v9"].Draw("HIST")
    # hist["v12"].Draw("HIST SAME")

    # Add the legend
    legend.AddEntry(hist["v9"], "v9", "l")
    legend.AddEntry(hist["v12"], "v12", "l")
    legend.Draw()

    # Save the comparison plot
    canvas.SaveAs("muon_pt_distribution_comparison.pdf")

if __name__ == "__main__":
    main()
