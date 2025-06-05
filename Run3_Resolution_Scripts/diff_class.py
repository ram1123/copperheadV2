import ROOT as rt
import yaml
import os
import awkward as ak
import numpy as np

class DistributionCompare:
    def __init__(self, year, input_paths_labels, fields, control_region=None, directoryTag="test", config_file="config/plot_config_nanoV12vsV9.yaml"):
        self.year = year
        self.input_paths_labels = input_paths_labels
        self.fields = fields
        self.control_region = control_region
        self.directoryTag = directoryTag
        self.events = {}

        # Load the config file to get the variables and other configurations
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        self.variables = config["variables"]
        self.mass_fit_range = {  # Add mean with range to be used in RooFit
            "h-peak": (125, 115, 135),
            "h-sidebands": (125, 110, 150),
            "signal": (125, 110, 150),
            "z-peak": (90, 70, 110)
        }

    def filter_region(self, events, region="h-peak"):
        dimuon_mass = events.dimuon_mass
        if region == "h-peak":
            region_filter = (dimuon_mass > 115.03) & (dimuon_mass < 135.03)
        elif region == "h-sidebands":
            region_filter = ((dimuon_mass > 110) & (dimuon_mass < 115.03)) | ((dimuon_mass > 135.03) & (dimuon_mass < 150))
        elif region == "signal":
            region_filter = (dimuon_mass >= 110) & (dimuon_mass <= 150.0)
        elif region == "z-peak":
            region_filter = (dimuon_mass >= 70) & (dimuon_mass <= 110.0)
        return events[region_filter]

    def filter_eta(self, events, region="BB"):
        if region == "BB":
            region_filter = (abs(events.mu1_eta) <= 0.9) & (abs(events.mu2_eta) <= 0.9)
        elif region == "BO":
            region_filter = (abs(events.mu1_eta) <= 0.9) & (abs(events.mu2_eta) > 0.9) & (abs(events.mu2_eta) <= 1.8)
        elif region == "BE":
            region_filter = (abs(events.mu1_eta) <= 0.9) & (abs(events.mu2_eta) > 1.8) & (abs(events.mu2_eta) <= 2.4)
        return events[region_filter]

    def load_data(self):
        """
        Load ROOT file using ROOT's RDataFrame and extract the variables once.
        """
        print(f"Loading data for {self.year} in {self.control_region} region with directory tag {self.directoryTag}")
        rt.EnableImplicitMT()  # Enable ROOT's implicit multi-threading

        for label, path in self.input_paths_labels.items():
            rdf = rt.RDataFrame("Events", path)  # 'Events' is the tree name
            print(f"Loading data for {label} from {path}")

            # Loop through the variables defined in YAML file
            print(f"Loading variables for {label}")
            print(f"variables to load: {self.variables}")
            print(f"variables to load: {self.variables['muon']}")
            print(type(self.variables['muon'][0]))
            print((self.variables['muon'][0].keys()))
            for variable, params in self.variables['muon'][0].items():
                print(f"Loading variable {variable} and params {params}")
                hist_name = f"h_{variable}"
                hist_title = f"{variable} Distribution"
                bins, xmin, xmax = params[0]['Range']
                self.events[label] = self.events.get(label, {})
                hist = rdf.Define(variable, f"{variable}").Histo1D((hist_name, hist_title, bins, xmin, xmax), variable)
                hist.GetXaxis().SetTitle(f"{params[0]['Title']}")
                hist.GetYaxis().SetTitle("Entries")

                self.events[label][variable] = hist.GetPtr()  # Store the histogram pointer

    def compare(self, var, xlabel=None, filename="comparison.pdf", events_dict=None):
        rt.gStyle.SetOptStat(0)
        bins, xmin, xmax, xtitle, ratio_range_min, ratio_range_max = self.get_hist_params(var)
        xlabel = xlabel or xtitle

        # Define Canvas
        canvas = rt.TCanvas("canvas", "canvas", 800, 800)

        # Define histograms
        histograms = []
        colors = [rt.kBlue, rt.kRed, rt.kGreen+2, rt.kBlack]
        legend = rt.TLegend(0.7, 0.7, 0.9, 0.9)

        if events_dict is None:
            events_dict = self.events

        for idx, (label, data) in enumerate(events_dict.items()):
            values = ak.to_numpy(data[var])  # Here, 'var' is a string, not a dictionary.
            hist = rt.TH1D(label, xlabel, bins, xmin, xmax)
            for v in values:
                hist.Fill(v)
            # Add overflow bin to the last bin
            if ("eta" not in var) and ("phi" not in var):
                hist.SetBinContent(bins, hist.GetBinContent(bins) + hist.GetBinContent(bins + 1))
            hist.Scale(1.0 / hist.Integral())
            hist.SetLineColor(colors[idx % len(colors)])
            hist.SetLineWidth(2)
            histograms.append(hist)
            legend.AddEntry(hist, label, "l")

        # First explicitly draw the histograms otherwise TRatioPlot will not work
        histograms[0].Draw("HIST")
        histograms[0].GetXaxis().SetTitle(xlabel)
        histograms[0].GetYaxis().SetTitle("Normalized Entries")
        for hist in histograms[1:]:
            hist.Draw("HIST SAME")

        canvas.Update()  # To properly initialize histograms for TRatioPlot

        if len(histograms) >= 2:
            ratio_plot = rt.TRatioPlot(histograms[0], histograms[1])
            ratio_plot.Draw()
            ratio_plot.GetLowerRefYaxis().SetTitle("Ratio")
            ratio_plot.GetLowerRefYaxis().SetRangeUser(ratio_range_min, ratio_range_max)
            ratio_plot.GetLowerRefGraph().SetMinimum(ratio_range_min)
            ratio_plot.GetLowerRefGraph().SetMaximum(ratio_range_max)

            ratio_plot.GetUpperPad().cd()
            legend.Draw()

            canvas.Update()

        canvas.SaveAs(filename)

        # Save the log version of the plot
        ratio_plot.GetUpperPad().SetLogy()
        # reset the y-axis range for upper pad
        histograms[0].SetMaximum(max(histograms[0].GetMaximum(), histograms[1].GetMaximum())*100)

        canvas.SaveAs(filename.replace(".pdf", "_log.pdf"))

        # clear memory
        for hist in histograms:
            hist.Delete()
        canvas.Clear()

    def compare_all(self, variables, outdir="plots/1D", events_dict=None, suffix=None):
        outdir = f"{outdir}/{self.year}/{self.directoryTag}/{self.control_region}"
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if suffix:
            suffix = f"{self.control_region}_{suffix}"
        else:
            suffix = self.control_region

        for var in variables:
            filename = f"{outdir}/{var}_{suffix}.pdf"
            self.compare(var, filename=filename, events_dict=events_dict)

    def get_hist_params(self, var):
        params = self.variables['muon'].get(var, {})
        return params['Range'] + (params['Title'], params['RatioPlot'][0], params['RatioPlot'][1])

# Main function to process all years
def main():
    # Read configuration parameters
    config_file = "config/plot_config_nanoV12vsV9.yaml"

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    input_paths_labels = config["input_paths_labels"]
    plot_types = config["plot_types"]
    fields_to_load = config["fields_to_load"]
    variables = config["variables"]["muon"]

    # Initialize the DistributionCompare class
    compare = DistributionCompare(
        year=config["year"],
        input_paths_labels=input_paths_labels,
        fields=fields_to_load,
        control_region=config["control_region"],
        directoryTag=config["directoryTag"]
    )

    # Load the data
    compare.load_data()

    # Compare the variables
    compare.compare_all(variables)

if __name__ == "__main__":
    main()
