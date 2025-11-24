import ROOT as rt
import yaml
import os

rt.gROOT.SetBatch(True)
rt.ROOT.EnableImplicitMT()   # Multi-threading ON
rt.gStyle.SetOptStat(0)

BASE = os.path.dirname(os.path.abspath(__file__))


def load_and_plot(path, fields_to_load):
    """
    Load ROOT file using ROOT's RDataFrame and plot the muon pT distribution.
    """
    # check if path is a text file containing list of files
    if path.endswith(".txt"):
        path = os.path.join(BASE, path)
        with open(path, "r") as f:
            files = [line.strip() for line in f if line.strip()]
        path = files
    # print(f"Loading files: {path}")
    rdf = rt.RDataFrame("Events", path, fields_to_load)

    # Number of entries
    n_entries = rdf.Count().GetValue()
    print(f"Number of entries in {path}: {n_entries}")

    # Directly histogram Muon_pt (Define not needed)
    h = rdf.Histo1D(
        ("h_muon_pt", "Muon pT Distribution", 100, 0, 500),
        "Muon_pt"
    )

    hist = h.GetPtr()   # Extract TH1D*
    hist.GetXaxis().SetTitle("Muon pT (GeV)")
    hist.GetYaxis().SetTitle("Entries")

    return hist


def main():
    config_full_path = os.path.join(BASE, "config/plot_config_nanoV12vsV9.yaml")

    with open(config_full_path, "r") as f:
        config = yaml.safe_load(f)

    input_paths_labels = config["input_paths_labels"]
    fields_to_load = config["fields_to_load"]
    directoryTag = config["directoryTag"]

    output_dir = f"plots/NanoAODv9vsV12/{directoryTag}"
    os.makedirs(output_dir, exist_ok=True)

    # Load histograms
    hist = {}
    for label, path in input_paths_labels.items():
        print(f"\nLoading data for {label}: {path}")
        hist[label] = load_and_plot(path, fields_to_load)

    print(f"\nCreating comparison plot...")
    print(f"histograms loaded: {list(hist.keys())}")

    canvas = rt.TCanvas("canvas", "Muon pT Distribution", 800, 600)

    # Colors
    hist["v9"].SetLineColor(rt.kRed)
    hist["v12"].SetLineColor(rt.kBlue)
    if "v15" in hist:
        hist["v15"].SetLineColor(rt.kGreen+2)

    # Normalize
    for key in ["v9", "v12", "v15"]:
        if key in hist:
            I = hist[key].Integral()
            if I > 0:
                hist[key].Scale(1.0 / I)

    print(type(hist["v9"]))
    print(type(hist["v12"]))

    # Main ratio plot (v9 / v12)
    # --- main ratio plot: v9 / v12 ---
    ratio_plot = rt.TRatioPlot(hist["v9"], hist["v12"])
    ratio_plot.Draw()
    ratio_plot.GetLowerRefYaxis().SetTitle("Ratio")
    ratio_plot.GetLowerRefYaxis().SetRangeUser(0.85, 1.25)
    ratio_plot.GetLowerRefGraph().SetMinimum(0.85)
    ratio_plot.GetLowerRefGraph().SetMaximum(1.25)

    # --- add v15 / v12 ratio as an extra curve, if present ---
    if "v15" in hist:
        # Clone v15 and divide by v12 to get the ratio histogram
        h_ratio_15 = hist["v15"].Clone("h_ratio_15_over_v12")
        h_ratio_15.Divide(hist["v12"])
        h_ratio_15.SetLineColor(rt.kGreen + 2)
        h_ratio_15.SetLineWidth(2)

        # Draw on the lower pad on top of the existing ratio
        ratio_plot.GetLowerPad().cd()
        h_ratio_15.Draw("HIST SAME")

    # back to upper pad for legend
    ratio_plot.GetUpperPad().cd()
    legend = rt.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(hist["v9"], "v9", "l")
    legend.AddEntry(hist["v12"], "v12", "l")
    if "v15" in hist:
        legend.AddEntry(hist["v15"], "v15", "l")
    legend.Draw()
    # set logy
    canvas.SetLogy()

    canvas.SaveAs(f"{output_dir}/muon_pt_distribution_comparison.pdf")


if __name__ == "__main__":
    main()
