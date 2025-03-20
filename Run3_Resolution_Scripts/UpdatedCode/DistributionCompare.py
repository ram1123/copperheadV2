import dask_awkward as dak
import awkward as ak
import yaml
import ROOT as rt
import array
import os

class DistributionCompare:
    def __init__(self, year, input_paths_labels, fields, control_region=None, directoryTag="test", varlist_file="varlist.yaml"):
        self.year = year
        self.input_paths_labels = input_paths_labels
        self.fields = fields
        self.control_region = control_region
        self.directoryTag = directoryTag
        self.events = {}
        with open(varlist_file, 'r') as f:
            self.varlist = yaml.safe_load(f)

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

    def load_data(self):
        def load(path):
            events_data = dak.from_parquet(path)
            # # Add new variable: ptErr/pT for both leading and sub-leading muons
            # events_data = self.add_new_variable(events_data)

            # Load only the required fields
            events_data = ak.zip({field: events_data[field] for field in self.fields}).compute()
            print(f"Loaded {len(events_data)} events from {path}")
            print(f"control_region: {self.control_region}")
            if self.control_region:
                events_data = self.filter_region(events_data, region=self.control_region)
            return events_data

        for label, path in self.input_paths_labels.items():
            print(f"Loading {label} : {path}")
            self.events[label] = load(path)
            print(f"{label} data loaded: {len(self.events[label])} events")

    def add_new_variable(self):
        # Add variable: ptErr/pT for both leading and sub-leading muons
        for label, data in self.events.items():
            data = ak.with_field(data, data.mu1_ptErr / data.mu1_pt, "ratio_pTErr_pt_mu1")
            data = ak.with_field(data, data.mu2_ptErr / data.mu2_pt, "ratio_pTErr_pt_mu2")
            self.events[label] = data
        print("New variable added: ptErr/pT for both leading and sub-leading muons")

    def get_hist_params(self, var):
        params = self.varlist.get(var, self.varlist["default"])
        return params

    def compare(self, var, xlabel=None, filename="comparison.pdf"):
        rt.gStyle.SetOptStat(0)
        bins, xmin, xmax, xtitle, ratio_range_min, ratio_range_max = self.get_hist_params(var)
        xlabel = xlabel or xtitle

        # Define Canvas
        canvas = rt.TCanvas("canvas", "canvas", 800, 800)

        # Define histograms
        histograms = []
        colors = [rt.kBlue, rt.kRed, rt.kGreen+2, rt.kBlack]
        legend = rt.TLegend(0.7, 0.7, 0.9, 0.9)

        for idx, (label, data) in enumerate(self.events.items()):
            values = ak.to_numpy(data[var])
            hist = rt.TH1D(label, xlabel, bins, xmin, xmax)
            for v in values:
                hist.Fill(v)
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

        # clear memory
        for hist in histograms:
            hist.Delete()
        canvas.Clear()

    def compare_all(self, variables, outdir="plots"):
        outdir = f"{outdir}/{self.year}/{self.directoryTag}"
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for var in variables:
            filename = f"{outdir}/{var}_{self.control_region}.pdf"
            self.compare(var, filename=filename)

    def compare_2D(self, var1, var2, xlabel=None, ylabel=None, filename_prefix="comparison_2D", outdir="plots_2D"):
        rt.gStyle.SetOptStat(0)

        # Set color palette
        NRGBs = 5
        NCont = 255
        stops = array.array('d', [0.00, 0.34, 0.61, 0.84, 1.00])
        red   = array.array('d', [0.00, 0.00, 0.87, 1.00, 0.51])
        green = array.array('d', [0.00, 0.81, 1.00, 0.20, 0.00])
        blue  = array.array('d', [0.51, 1.00, 0.12, 0.00, 0.00])

        rt.TColor.CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont)
        rt.gStyle.SetNumberContours(NCont)

        bins_x, xmin, xmax, xtitle, _, _ = self.get_hist_params(var1)
        bins_y, ymin, ymax, ytitle, _, _ = self.get_hist_params(var2)

        xlabel = xlabel or xtitle
        ylabel = ylabel or ytitle

        for label, data in self.events.items():
            canvas = rt.TCanvas(f"canvas_{label}", f"canvas_{label}", 800, 600)
            hist = rt.TH2D(label, f"{xlabel} vs {ylabel} - {label}", bins_x, xmin, xmax, bins_y, ymin, ymax)

            values_x = ak.to_numpy(data[var1])
            values_y = ak.to_numpy(data[var2])

            for x, y in zip(values_x, values_y):
                hist.Fill(x, y)

            hist.GetXaxis().SetTitle(xlabel)
            hist.GetYaxis().SetTitle(ylabel)
            hist.Draw("COLZ")

            outdir = f"{outdir}/{self.year}/{self.directoryTag}"
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            filename = f"{outdir}/{filename_prefix}_{var1}_vs_{var2}_{self.control_region}_{label}.pdf"
            canvas.SaveAs(filename)
