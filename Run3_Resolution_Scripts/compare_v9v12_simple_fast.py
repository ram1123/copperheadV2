import ROOT as rt
import yaml
import os

rt.gROOT.SetBatch(True)
rt.ROOT.EnableImplicitMT()   # Multi-threading ON
rt.gStyle.SetOptStat(0)
rt.gErrorIgnoreLevel = rt.kError

BASE = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path):
    """Return list of ROOT files from either a .txt list or a direct path/pattern."""
    if path.endswith(".txt"):
        path = os.path.join(BASE, path)
        with open(path, "r") as f:
            files = [line.strip() for line in f if line.strip()]
        return files
    else:
        return path  # string or pattern; ROOT can handle list or string


def main():
    config_full_path = os.path.join(BASE, "config/plot_config_nanoV12vsV9.yaml")
    with open(config_full_path, "r") as f:
        config = yaml.safe_load(f)

    input_paths_labels = config["input_paths_labels"]

    electrons_var = config["variables"]["electron"]
    muons_var     = config["variables"]["muon"]
    jets_var      = config["variables"]["jet"]
    variables_to_plot = electrons_var + muons_var + jets_var
    # variables_to_plot = jets_var

    directoryTag = config["directoryTag"]
    NormalizeToUnity = config.get("NormalizeToUnity", False)

    print(f"variables to plot: {variables_to_plot}")

    output_dir = f"plots/NanoAODv9vsV12/{directoryTag}"
    if NormalizeToUnity:
        output_dir += "_Normalized"
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------
    # 1) Build one RDataFrame per sample, reuse later
    # ------------------------------------------------
    # we can load all branches we might need; ROOT only reads what is used
    # but if you want, you can also build the union of var names automatically
    fields_to_load = config["fields_to_load"]  # or build from variables_to_plot

    rdf_map = {}
    for label, path in input_paths_labels.items():
        files = resolve_path(path)
        # print(f"\nBuilding RDataFrame for {label} from: {files}")
        rdf = rt.RDataFrame("Events", files, fields_to_load)
        # optional: count once here
        n_entries = rdf.Count()
        rdf_map[label] = (rdf, n_entries)

    # trigger all Counts in one go (optional but nice)
    print("\nComputing event counts...")
    for label, (rdf, n_entries) in rdf_map.items():
        print(f"{label}: {n_entries.GetValue()} events")

    # ------------------------------------------------
    # 2) Loop over variables; reuse RDFs
    # ------------------------------------------------
    for vblock in variables_to_plot:
        var_name = list(vblock.keys())[0]
        cfg = vblock[var_name]

        binning = cfg[0]["Range"]
        title   = cfg[1]["Title"]
        rmin, rmax = cfg[2]["RatioPlot"]

        nb, xmin, xmax = binning

        print(f"\n === Plotting variable: {var_name} ===\n")

        # Build histograms from existing RDFs
        hist = {}
        for label, (rdf, _) in rdf_map.items():
            print(f"  -> {label}")
            h = rdf.Histo1D(
                (f"h_{var_name}_{label}", title, nb, xmin, xmax),
                var_name
            )
            hist[label] = h  # keep the RResultPtr for now

        # Force computation once we actually need the TH1
        # (ROOT will run one event loop per sample and reuse branches internally)
        for label in hist:
            hist[label] = hist[label].GetPtr()
            hist[label].GetXaxis().SetTitle(title)
            hist[label].GetYaxis().SetTitle("Entries")

        # Canvas and colors
        canvas = rt.TCanvas("canvas", var_name, 800, 600)
        hist["v9"].SetLineColor(rt.kRed)
        hist["v12"].SetLineColor(rt.kBlue)
        if "v15" in hist:
            hist["v15"].SetLineColor(rt.kGreen+2)

        # Normalize
        if NormalizeToUnity:
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

        # Linear and log-y versions
        canvas.SaveAs(f"{output_dir}/{var_name}.pdf")
        ratio_plot.GetUpperPad().SetLogy(True)
        canvas.SaveAs(f"{output_dir}/{var_name}_log.pdf")
        ratio_plot.GetUpperPad().SetLogy(False)

if __name__ == "__main__":
    main()
