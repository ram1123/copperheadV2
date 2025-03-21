import dask_awkward as dak
import awkward as ak
import yaml
import ROOT as rt
import array
import os
import numpy as np

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

        self.mass_fit_range = { # add mean with range to be used in RooFit
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

    # Function to filter events based on leading and subleading muon rapidity
    # Eta bins:
    #   B: |eta| <= 0.9
    #   O: 0.9 < |eta| <= 1.8
    #   E: 1.8 < |eta| <= 2.4
    def filter_eta1(self, events, region="B"):
        if region == "B":
            region_filter = (abs(events.mu1_eta) <= 0.9)
        elif region == "O":
            region_filter = (abs(events.mu1_eta) > 0.9) & (abs(events.mu1_eta) <= 1.8)
        elif region == "E":
            region_filter = (abs(events.mu1_eta) > 1.8) & (abs(events.mu1_eta) <= 2.4)

        return events[region_filter]
    def filter_eta2(self, events, region="B"):
        if region == "B":
            region_filter = (abs(events.mu2_eta) <= 0.9)
        elif region == "O":
            region_filter = (abs(events.mu2_eta) > 0.9) & (abs(events.mu2_eta) <= 1.8)
        elif region == "E":
            region_filter = (abs(events.mu2_eta) > 1.8) & (abs(events.mu2_eta) <= 2.4)
        return events[region_filter]

    def filter_eta(self, events, region="BB"):
        if region == "BB":
            region_filter = (abs(events.mu1_eta) <= 0.9) & (abs(events.mu2_eta) <= 0.9)
        elif region == "BO":
            region_filter = (abs(events.mu1_eta) <= 0.9) & (abs(events.mu2_eta) > 0.9) & (abs(events.mu2_eta) <= 1.8)
        elif region == "BE":
            region_filter = (abs(events.mu1_eta) <= 0.9) & (abs(events.mu2_eta) > 1.8) & (abs(events.mu2_eta) <= 2.4)
        elif region == "OB":
            region_filter = (abs(events.mu1_eta) > 0.9) & (abs(events.mu1_eta) <= 1.8) & (abs(events.mu2_eta) <= 0.9)
        elif region == "OO":
            region_filter = (abs(events.mu1_eta) > 0.9) & (abs(events.mu1_eta) <= 1.8) & (abs(events.mu2_eta) > 0.9) & (abs(events.mu2_eta) <= 1.8)
        elif region == "OE":
            region_filter = (abs(events.mu1_eta) > 0.9) & (abs(events.mu1_eta) <= 1.8) & (abs(events.mu2_eta) > 1.8) & (abs(events.mu2_eta) <= 2.4)
        elif region == "EB":
            region_filter = (abs(events.mu1_eta) > 1.8) & (abs(events.mu1_eta) <= 2.4) & (abs(events.mu2_eta) <= 0.9)
        elif region == "EO":
            region_filter = (abs(events.mu1_eta) > 1.8) & (abs(events.mu1_eta) <= 2.4) & (abs(events.mu2_eta) > 0.9) & (abs(events.mu2_eta) <= 1.8)
        elif region == "EE":
            region_filter = (abs(events.mu1_eta) > 1.8) & (abs(events.mu1_eta) <= 2.4) & (abs(events.mu2_eta) > 1.8) & (abs(events.mu2_eta) <= 2.4)
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
            values = ak.to_numpy(data[var])
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

    # def compare_all(self, variables, events = self.events, region = "inclusive" outdir="plots"):
    def compare_all(self, variables, outdir="plots", events_dict=None, suffix=None):
        outdir = f"{outdir}/{self.year}/{self.directoryTag}"
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if suffix:
            suffix = f"{self.control_region}_{suffix}"
        else:
            suffix = self.control_region

        for var in variables:
            filename = f"{outdir}/{var}_{suffix}.pdf"
            self.compare(var, filename=filename, events_dict=events_dict)

    def compare_2D(self, var1, var2, xlabel=None, ylabel=None, filename_prefix="comparison_2D", outdir="plots_2D", events_dict=None, suffix=None):
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

        outdir = f"{outdir}/{self.year}/{self.directoryTag}"
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if events_dict is None:
            events_dict = self.events

        if suffix:
            suffix = f"{self.control_region}_{suffix}"
        else:
            suffix = self.control_region

        for label, data in events_dict.items():
            canvas = rt.TCanvas(f"canvas_{label}", f"canvas_{label}", 800, 600)
            hist = rt.TH2D(label, f"{xlabel} vs {ylabel} - {label}", bins_x, xmin, xmax, bins_y, ymin, ymax)

            values_x = ak.to_numpy(data[var1])
            values_y = ak.to_numpy(data[var2])

            for x, y in zip(values_x, values_y):
                hist.Fill(x, y)

            hist.GetXaxis().SetTitle(xlabel)
            hist.GetYaxis().SetTitle(ylabel)
            hist.Draw("COLZ")

            # remove space or special charcters from the label
            label_modified = label.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
            filename = f"{outdir}/{filename_prefix}_{var1}_vs_{var2}_{suffix}_{label_modified}.pdf"
            canvas.SaveAs(filename)

    def generateRooHist(self, x, dimuon_mass, wgts, name=""):
        print("generateRooHist version 2")
        dimuon_mass = np.asarray(ak.to_numpy(dimuon_mass)).astype(np.float64) # explicit float64 format is required
        wgts = np.asarray(ak.to_numpy(wgts)).astype(np.float64) # explicit float64 format is required
        nbins = x.getBins()
        TH = rt.TH1D("TH", "TH", nbins, x.getMin(), x.getMax())
        TH.FillN(len(dimuon_mass), dimuon_mass, wgts) # fill the histograms with mass and weights
        DEBUG = True
        if DEBUG:
            print(f"dimuon_mass: {dimuon_mass}")
            print(f"wgts: {wgts}")
            print(f"nbins: {nbins}")
            print(f"TH.Integral(): {TH.Integral()}")
            # plot the TH histogram
            c = rt.TCanvas("c","c",800,800)
            TH.Draw()
            c.SaveAs("TH.pdf")

        roohist = rt.RooDataHist(name, name, rt.RooArgSet(x), TH)
        return roohist

    def normalizeRooHist(self, x: rt.RooRealVar,rooHist: rt.RooDataHist) -> rt.RooDataHist :
        """
        Takes rootHistogram and returns a new copy with histogram values normalized to sum to one
        """
        x_name = x.GetName()
        THist = rooHist.createHistogram(x_name).Clone("clone") # clone it just in case
        THist.Scale(1/THist.Integral())
        print(f"THist.Integral(): {THist.Integral()}")
        normalizedHist_name = rooHist.GetName() + "_normalized"
        roo_hist_normalized = rt.RooDataHist(normalizedHist_name, normalizedHist_name, rt.RooArgSet(x), THist)
        return roo_hist_normalized

    def fit_dimuonInvariantMass(self, events_dict=None, outdir = "plots", suffix=None):
        """
        generate histogram from dimuon mass and wgt, fit DCB
        aftwards, plot the histogram and return the fit params
        as fit DCB sigma and chi2_dof
        """
        if events_dict is None:
            events_dict = self.events

        counter = 0
        for label, events in events_dict.items():
            if counter == 0:
                events_bsOn = events
                events_bsOn_label = label
            if counter == 1:
                events_bsOff = events
                events_bsOff_label = label
            counter += 1

        print("==================================================")
        print(f"events_bsOn: {events_bsOn}")
        print(f"events_bsOff: {events_bsOff}")
        print("==================================================")

        mass_name = "mh_ggh"
        if self.control_region == "z-peak" or self.control_region == "z_peak":
            mass = rt.RooRealVar(mass_name, mass_name, 90, 70, 110) # Z-peak
        elif self.control_region == "signal":
            mass = rt.RooRealVar(mass_name, mass_name, 120, 115, 135) # signal region
        nbins = 80
        mass.setBins(nbins)
        dimuon_mass = ak.to_numpy(events_bsOn.dimuon_mass)
        wgt = ak.to_numpy(events_bsOn.wgt_nominal)
        hist_bsOn = self.generateRooHist(mass, dimuon_mass, wgt, name=events_bsOn_label)
        hist_bsOn = self.normalizeRooHist(mass, hist_bsOn)
        print(f"fitPlot_ggh hist_bsOn: {hist_bsOn}")

        dimuon_mass = ak.to_numpy(events_bsOff.dimuon_mass)
        wgt = ak.to_numpy(events_bsOff.wgt_nominal)
        hist_bsOff = self.generateRooHist(mass, dimuon_mass, wgt, name=events_bsOff_label)
        hist_bsOff = self.normalizeRooHist(mass, hist_bsOff)
        print(f"fitPlot_ggh hist_bsOff: {hist_bsOff}")

        # --------------------------------------------------
        # Fitting
        # --------------------------------------------------

        mass_fit_range = self.mass_fit_range[self.control_region]

        MH_bsOn = rt.RooRealVar("MH" , "MH", mass_fit_range[0], mass_fit_range[1], mass_fit_range[2])
        sigma_bsOn = rt.RooRealVar("sigma" , "sigma", 1.8228, .1, 4.0)
        alpha1_bsOn = rt.RooRealVar("alpha1" , "alpha1", 1.12842, 0.01, 65)
        n1_bsOn = rt.RooRealVar("n1" , "n1", 4.019960, 0.01, 100)
        alpha2_bsOn = rt.RooRealVar("alpha2" , "alpha2", 1.3132, 0.01, 65)
        n2_bsOn = rt.RooRealVar("n2" , "n2", 9.97411, 0.01, 100)
        name = f"BSC fit"
        model_bsOn = rt.RooDoubleCBFast(name,name,mass, MH_bsOn, sigma_bsOn, alpha1_bsOn, n1_bsOn, alpha2_bsOn, n2_bsOn)

        device = "cpu"
        _ = model_bsOn.fitTo(hist_bsOn,  EvalBackend=device, Save=True, SumW2Error=True)
        fit_result = model_bsOn.fitTo(hist_bsOn,  EvalBackend=device, Save=True, SumW2Error=True)
        fit_result.Print()

        MH_bsOff = rt.RooRealVar("MH" , "MH", mass_fit_range[0], mass_fit_range[1], mass_fit_range[2])
        sigma_bsOff = rt.RooRealVar("sigma" , "sigma", 1.8228, .1, 4.0)
        alpha1_bsOff = rt.RooRealVar("alpha1" , "alpha1", 1.12842, 0.01, 65)
        n1_bsOff = rt.RooRealVar("n1" , "n1", 4.019960, 0.01, 100)
        alpha2_bsOff = rt.RooRealVar("alpha2" , "alpha2", 1.3132, 0.01, 65)
        n2_bsOff = rt.RooRealVar("n2" , "n2", 9.97411, 0.01, 100)
        name = f"BSC fit"
        model_bsOff = rt.RooDoubleCBFast(name,name,mass, MH_bsOff, sigma_bsOff, alpha1_bsOff, n1_bsOff, alpha2_bsOff, n2_bsOff)

        device = "cpu"
        _ = model_bsOff.fitTo(hist_bsOff,  EvalBackend=device, Save=True, SumW2Error=True)
        fit_result = model_bsOff.fitTo(hist_bsOff,  EvalBackend=device, Save=True, SumW2Error=True)
        fit_result.Print()

        # ------------------------------------
        # Plotting
        # ------------------------------------
        name = "Canvas"
        canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
        canvas.cd()
        legend = rt.TLegend(0.6,0.60,0.9,0.9)


        frame = mass.frame()
        hist_bsOn.plotOn(frame, rt.RooFit.MarkerColor(rt.kGreen), rt.RooFit.LineColor(rt.kGreen), Invisible=False )
        model_bsOn.plotOn(frame, Name=name, LineColor=rt.kGreen)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1), events_bsOn_label, "L")
        sigma_val = round(sigma_bsOn.getVal(), 3)
        sigma_err = round(sigma_bsOn.getError(), 3)
        mean_val = round(MH_bsOn.getVal(), 3)
        mean_err = round(MH_bsOn.getError(), 3)
        legend.AddEntry("", f"   mean: {mean_val} +- {mean_err}", "")
        legend.AddEntry("", f"   sigma: {sigma_val} +- {sigma_err}", "")

        hist_bsOff.plotOn(frame, rt.RooFit.MarkerColor(rt.kBlue), rt.RooFit.LineColor(rt.kBlue), Invisible=False )
        model_bsOff.plotOn(frame, Name=name, LineColor=rt.kBlue)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1), events_bsOff_label, "L")
        sigma_val = round(sigma_bsOff.getVal(), 3)
        sigma_err = round(sigma_bsOff.getError(), 3)
        mean_val = round(MH_bsOff.getVal(), 3)
        mean_err = round(MH_bsOff.getError(), 3)

        legend.AddEntry("", f"   mean: {mean_val} +- {mean_err}", "")
        legend.AddEntry("", f"   sigma: {sigma_val} +- {sigma_err}", "")

        # append results in a csv file, with format: category, sigma BSOn +/- error, sigma BSOFF +/- error, % diff
        with open(f"fit_results_{self.control_region}.csv", "a") as f:
            suffix = "Inclusive" if suffix == None else suffix
            f.write(f"{suffix},{round(sigma_bsOn.getVal(),3)},{round(sigma_bsOn.getError(),3)},{round(sigma_bsOff.getVal(),3)},{round(sigma_bsOff.getError(),3)},{round(100*(abs(sigma_bsOff.getVal()-sigma_bsOn.getVal()))/sigma_bsOff.getVal(),3)}\n")

        # get a latex style table
        with open(f"fit_results_{self.control_region}.txt", "a") as f:
            suffix = "Inclusive" if suffix == None else suffix
            f.write(f"{suffix} & {round(sigma_bsOn.getVal(),3)} $\\pm$ {round(sigma_bsOn.getError(),3)} & {round(sigma_bsOff.getVal(),3)} $\\pm$ {round(sigma_bsOff.getError(),3)} & {round(100*(abs(sigma_bsOff.getVal()-sigma_bsOn.getVal()))/sigma_bsOff.getVal(),3)} \\\\ \n")



        frame.SetYTitle(f"A.U.")
        frame.SetXTitle(f"Dimuon Mass (GeV)")
        frame.SetTitle(f"")

        frame.Draw()
        legend.Draw()
        canvas.Update()
        canvas.Draw()
        save_filename = f"{outdir}/{self.year}/{self.directoryTag}/fitPlot_{self.control_region}_{suffix}.pdf"
        if not os.path.exists(os.path.dirname(save_filename)):
            os.makedirs(os.path.dirname(save_filename))
        canvas.SaveAs(save_filename)
