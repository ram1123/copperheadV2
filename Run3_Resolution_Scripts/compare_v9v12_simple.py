import ROOT as rt
import yaml
import os

rt.gROOT.SetBatch(True)
rt.ROOT.EnableImplicitMT()   # Multi-threading ON
rt.gStyle.SetOptStat(0)
# rt.gErrorIgnoreLevel = rt.kWarning
rt.gErrorIgnoreLevel = rt.kError

BASE = os.path.dirname(os.path.abspath(__file__))


def load_and_plot(path, label, var_name, binning):
    """
    Load ROOT file using ROOT's RDataFrame and plot the muon pT distribution.
    """
    # check if path is a text file containing list of files
    if path.endswith(".txt"):
        path = os.path.join(BASE, path)
        with open(path, "r") as f:
            files = [line.strip() for line in f if line.strip()]
        path = files

    rdf = rt.RDataFrame("Events", path, [var_name])

    n_entries = rdf.Count().GetValue()
    print(f"Number of entries for {label}: {n_entries}")

    nb, xmin, xmax = binning
    h = rdf.Histo1D(
        (f"h_{var_name}_{label}", f"{var_name}", nb, xmin, xmax),
        var_name
    )

    hist = h.GetPtr()
    hist.GetXaxis().SetTitle(var_name)
    hist.GetYaxis().SetTitle("Entries")

    return hist


def main():
    config_full_path = os.path.join(BASE, "config/plot_config_nanoV12vsV9.yaml")

    with open(config_full_path, "r") as f:
        config = yaml.safe_load(f)

    input_paths_labels = config["input_paths_labels"]

    electrons_var = config["variables"]["electron"]
    muons_var = config["variables"]["muon"]
    jets_var = config["variables"]["jet"]
    variables_to_plot = electrons_var + muons_var + jets_var


    directoryTag = config["directoryTag"]

    print(f"variables to plot: {variables_to_plot}")

    output_dir = f"plots/NanoAODv9vsV12/{directoryTag}"
    os.makedirs(output_dir, exist_ok=True)

    # -------- Loop over variables to plot --------
    for vblock in variables_to_plot:
        var_name = list(vblock.keys())[0]
        cfg = vblock[var_name]

        # unpack YAML
        binning = cfg[0]["Range"]
        title = cfg[1]["Title"]
        rmin, rmax = cfg[2]["RatioPlot"]

        print(f"\n === Plotting variable: {var_name} ===\n")

        # Load histograms
        hist = {}
        for label, path in input_paths_labels.items():
            # print(f"Loading {label}: {path}")
            hist[label] = load_and_plot(path, label, var_name, binning)

        canvas = rt.TCanvas("canvas", var_name, 800, 600)

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

        # Ratio plot
        ratio_plot = rt.TRatioPlot(hist["v9"], hist["v12"])
        ratio_plot.Draw()
        ratio_plot.GetLowerRefYaxis().SetTitle("Ratio")
        ratio_plot.GetLowerRefYaxis().SetRangeUser(rmin, rmax)

        # log-scale upper pad
        ratio_plot.GetUpperPad().cd()
        # ratio_plot.GetUpperPad().SetLogy()

        # v15 overlay
        if "v15" in hist:
            h15 = hist["v15"].Clone(f"h_ratio15_{var_name}")
            h15.Divide(hist["v12"])
            h15.SetLineColor(rt.kGreen+2)
            ratio_plot.GetLowerPad().cd()
            h15.Draw("HIST SAME")

        # Legend
        ratio_plot.GetUpperPad().cd()
        legend = rt.TLegend(0.7, 0.7, 0.9, 0.9)
        legend.AddEntry(hist["v9"], "v9", "l")
        legend.AddEntry(hist["v12"], "v12", "l")
        if "v15" in hist:
            legend.AddEntry(hist["v15"], "v15", "l")
        legend.Draw()

        canvas.SaveAs(f"{output_dir}/{var_name}.pdf")
        ratio_plot.GetUpperPad().SetLogy(True)
        canvas.SaveAs(f"{output_dir}/{var_name}_log.pdf")
        ratio_plot.GetUpperPad().SetLogy(False)

if __name__ == "__main__":
    main()
