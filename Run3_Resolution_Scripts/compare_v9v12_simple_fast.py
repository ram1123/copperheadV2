import ROOT as rt
import yaml
import os

rt.gROOT.SetBatch(True)
rt.ROOT.EnableImplicitMT()   # Multi-threading ON
rt.gStyle.SetOptStat(0)
rt.gErrorIgnoreLevel = rt.kError

BASE = os.path.dirname(os.path.abspath(__file__))

xrd_prefix = "root://xcache.cms.rcac.purdue.edu/"

def resolve_path(path):
    """Return list of ROOT files from either a .txt list or a direct path/pattern."""
    if path.endswith(".txt"):
        path = os.path.join(BASE, path)
        with open(path, "r") as f:
            files = [xrd_prefix + line.strip() for line in f if line.strip()]
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
    # variables_to_plot = muons_var

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

        # Optionally normalize to unity
        if NormalizeToUnity:
            for h in hist.values():
                integral = h.Integral()
                if integral > 0:
                    h.Scale(1.0 / integral)
                h.GetYaxis().SetTitle("Normalized Entries")

        # Set Maximum
        max_y = max(h.GetMaximum() for h in hist.values())
        for h in hist.values():
            h.SetMaximum(1.3 * max_y)

        # ------------------------------
        #  Build custom two-pad canvas
        # ------------------------------
        canvas = rt.TCanvas("canvas", var_name, 800, 700)
        canvas.Divide(1, 2)

        # -------- Upper pad: main hist --------
        pad1 = canvas.cd(1)
        pad1.SetPad(0.0, 0.30, 1.0, 1.0)  # (x1, y1, x2, y2)
        pad1.SetBottomMargin(0.02)

        if "v9" in hist:
            hist["v9"].SetLineColor(rt.kRed)
        if "v12" in hist:
            hist["v12"].SetLineColor(rt.kBlue)
            hist["v12"].SetLineStyle(9)
        if "v15" in hist:
            hist["v15"].SetLineColor(rt.kGreen + 2)
            hist["v15"].SetLineStyle(10)

        if "v9" in hist:
            hist["v9"].Draw("HIST")
        if "v12" in hist:
            hist["v12"].Draw("HIST SAME")
        if "v15" in hist:
            hist["v15"].Draw("HIST SAME")

        legend = rt.TLegend(0.7, 0.7, 0.9, 0.9)
        if "v9" in hist:
            legend.AddEntry(hist["v9"], "v9", "l")
        if "v12" in hist:
            legend.AddEntry(hist["v12"], "v12", "l")
        if "v15" in hist:
            legend.AddEntry(hist["v15"], "v15", "l")
        legend.Draw()


        # -------- Lower pad: difference --------
        pad2 = canvas.cd(2)
        pad2.SetPad(0.0, 0.0, 1.0, 0.30)
        pad2.SetTopMargin(0.02)
        pad2.SetBottomMargin(0.25)

        # Difference histogram: v9 - v12
        if "v12" in hist and "v9" in hist:
            h_diff = hist["v9"].Clone(f"h_diff_{var_name}")
            h_diff.Add(hist["v12"], -1)     # v9 - v12
            h_diff.SetLineColor(rt.kBlue)
            h_diff.SetLineWidth(2)

            # Axis formatting
            h_diff.GetYaxis().SetTitle("v9 - v12")
            h_diff.GetYaxis().SetTitleSize(0.12)
            h_diff.GetYaxis().SetTitleOffset(0.4)
            h_diff.GetYaxis().SetLabelSize(0.10)

            # h_diff.GetXaxis().SetTitle(title)
            h_diff.GetXaxis().SetTitleSize(0.12)
            h_diff.GetXaxis().SetLabelSize(0.10)

            # range
            max_diff = max(abs(h_diff.GetMinimum()), abs(h_diff.GetMaximum()))
            # if max_diff == 0: then set it to -0.1 to 0.1
            # if max_diff == 0:
            #     max_diff = 0.05
            # h_diff.SetMinimum(-1.1 * max_diff)
            # h_diff.SetMaximum(1.1 * max_diff)

            # Draw it
            # h_diff.Draw("HIST")

        # Optionally add v15 difference
        if "v15" in hist:
            if "v9" in hist:
                h_diff15 = hist["v9"].Clone(f"h_diff15_{var_name}")
            else:
                h_diff15 = hist["v12"].Clone(f"h_diff15_{var_name}")
            h_diff15.Add(hist["v15"], -1)   # v9 - v15
            h_diff15.SetLineColor(rt.kGreen+2)
            h_diff15.SetLineWidth(2)
            h_diff15.SetLineStyle(10)

            max_diff15 = max(abs(h_diff15.GetMinimum()), abs(h_diff15.GetMaximum()))


            # h_diff15.Draw("HIST SAME")

        # set maximum of diff pad to largest of both, if available then draw
        if "v12" in hist and "v15" in hist:
            if "v9" in hist:
                overall_max = max(max_diff, max_diff15)
            else:
                overall_max = max_diff15
            if overall_max == 0:
                overall_max = 0.05

            if "v9" in hist:
                h_diff.SetMinimum(-1.1 * overall_max)
                h_diff.SetMaximum(1.1 * overall_max)
                h_diff.Draw("HIST")
                h_diff15.Draw("HIST SAME")
            else:
                h_diff15.SetMinimum(-1.1 * overall_max)
                h_diff15.SetMaximum(1.1 * overall_max)
                h_diff15.Draw("HIST")
        elif "v12" in hist:
            if max_diff == 0:
                max_diff = 0.05
            h_diff.SetMinimum(-1.1 * max_diff)
            h_diff.SetMaximum(1.1 * max_diff)
            h_diff.Draw("HIST")
        elif "v15" in hist:
            if max_diff15 == 0:
                max_diff15 = 0.05
            h_diff15.SetMinimum(-1.1 * max_diff15)
            h_diff15.SetMaximum(1.1 * max_diff15)
            h_diff15.Draw("HIST")

        # Save
        canvas.SaveAs(f"{output_dir}/{var_name}_diff.pdf")

        # Save as log scale too
        pad1.SetLogy()
        canvas.SaveAs(f"{output_dir}/{var_name}_diff_log.pdf")
        pad1.SetLogy(0)  # reset

if __name__ == "__main__":
    main()
