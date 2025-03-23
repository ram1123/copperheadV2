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

        # ------------------------------------
        # Plotting
        # ------------------------------------
        name = "Canvas"
        canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
        canvas.cd()
        legend = rt.TLegend(0.6,0.60,0.9,0.9)

        # --------------------------------------------------------------------------------------------
        # ----                 Get First Histogram and Normalize it                 ----------------------
        # --------------------------------------------------------------------------------------------
        mass_name = "mh_ggh"
        if self.control_region == "z-peak" or self.control_region == "z_peak":
            mass = rt.RooRealVar(mass_name, mass_name, 90, 70, 110) # Z-peak
        elif self.control_region == "signal":
            mass = rt.RooRealVar(mass_name, mass_name, 120, 115, 135) # signal region

        frame = mass.frame()
        mass_fit_range = self.mass_fit_range[self.control_region]

        nbins = 200
        mass.setBins(nbins)

        dimuon_mass = ak.to_numpy(events_bsOn.dimuon_mass)
        wgt = ak.to_numpy(events_bsOn.wgt_nominal)
        hist_bsOn = self.generateRooHist(mass, dimuon_mass, wgt, name=events_bsOn_label)
        hist_bsOn = self.normalizeRooHist(mass, hist_bsOn)
        print(f"fitPlot_ggh hist_bsOn: {hist_bsOn}")

        # --------------------------------------------------
        # Fitting
        # --------------------------------------------------
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

        # hist_bsOn.plotOn(frame, rt.RooFit.MarkerColor(rt.kGreen), rt.RooFit.LineColor(rt.kGreen), Invisible=False )
        hist_bsOn.plotOn(frame, rt.RooFit.Name("hist_bsOn"), rt.RooFit.MarkerColor(rt.kGreen), rt.RooFit.LineColor(rt.kGreen), Invisible=False)
        model_plot_name = "model_bsOn"
        model_bsOn.plotOn(frame, rt.RooFit.Name(model_plot_name), LineColor=rt.kGreen)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1), events_bsOn_label, "L")
        sigma_val = round(sigma_bsOn.getVal(), 3)
        sigma_err = round(sigma_bsOn.getError(), 3)
        mean_val = round(MH_bsOn.getVal(), 3)
        mean_err = round(MH_bsOn.getError(), 3)

        chi2_o_ndf_on = model_bsOn.createChi2(hist_bsOn, rt.RooFit.Extended(True), rt.RooFit.DataError(rt.RooAbsData.SumW2))
        chiSquare_dof_on = round(chi2_o_ndf_on.getVal(), 3)

        legend.AddEntry("", f"   mean: {mean_val} +- {mean_err}", "")
        legend.AddEntry("", f"   sigma: {sigma_val} +- {sigma_err}", "")
        legend.AddEntry("", f"   chi2/dof: {chiSquare_dof_on}", "")



        chi2_o_ndf_On_fromFrame = frame.chiSquare()
        chiSquare_dof_on = round(chi2_o_ndf_On_fromFrame, 3)
        print(f"\n\n====> chiSquare_dof_on: {chiSquare_dof_on}")
        n_free_params = fit_result.floatParsFinal().getSize()
        print(f"\n\n====> n_free_params: {n_free_params}")
        # chi2 = frame.chiSquare(model_bsOn, hist_bsOn.getName(), n_free_params)
        # print(f"\n\n====> chi2: {chi2}")
        # print(f"\n\n====> chiSquare_dof_on: {chi2/n_free_params}")
        # Extract chi2 / ndf
        chi2_ndf = frame.chiSquare(model_plot_name, "hist_bsOn", n_free_params)
        # chi2_ndf = frame.chiSquare("model_bsOn", "hist_bsOn", n_free_params)
        print(f"Chi2/NDF = {chi2_ndf:.3f}")
        print("==================================================")
        ndf = model_bsOn.getParameters(rt.RooArgSet(mass)).getSize()
        print(f"ndf: {ndf}")
        new_chi2 = frame.chiSquare(model_plot_name, "hist_bsOn", ndf)
        print(f"new_chi2: {new_chi2}")

        # --------------------------------------------------


        # # --------------------------------------------------------------------------------------------
        # # ----                 Get Second Histogram and Normalize it                 -------------------
        # # --------------------------------------------------------------------------------------------
        # dimuon_mass = ak.to_numpy(events_bsOff.dimuon_mass)
        # wgt = ak.to_numpy(events_bsOff.wgt_nominal)
        # hist_bsOff = self.generateRooHist(mass, dimuon_mass, wgt, name=events_bsOff_label)
        # hist_bsOff = self.normalizeRooHist(mass, hist_bsOff)
        # print(f"fitPlot_ggh hist_bsOff: {hist_bsOff}")



        # MH_bsOff = rt.RooRealVar("MH" , "MH", mass_fit_range[0], mass_fit_range[1], mass_fit_range[2])
        # sigma_bsOff = rt.RooRealVar("sigma" , "sigma", 1.8228, .1, 4.0)
        # alpha1_bsOff = rt.RooRealVar("alpha1" , "alpha1", 1.12842, 0.01, 65)
        # n1_bsOff = rt.RooRealVar("n1" , "n1", 4.019960, 0.01, 100)
        # alpha2_bsOff = rt.RooRealVar("alpha2" , "alpha2", 1.3132, 0.01, 65)
        # n2_bsOff = rt.RooRealVar("n2" , "n2", 9.97411, 0.01, 100)
        # name = f"BSC fit"
        # model_bsOff = rt.RooDoubleCBFast(name,name,mass, MH_bsOff, sigma_bsOff, alpha1_bsOff, n1_bsOff, alpha2_bsOff, n2_bsOff)

        # device = "cpu"
        # _ = model_bsOff.fitTo(hist_bsOff,  EvalBackend=device, Save=True, SumW2Error=True)
        # fit_result = model_bsOff.fitTo(hist_bsOff,  EvalBackend=device, Save=True, SumW2Error=True)
        # fit_result.Print()


        # hist_bsOff.plotOn(frame, rt.RooFit.MarkerColor(rt.kBlue), rt.RooFit.LineColor(rt.kBlue), Invisible=False )
        # model_bsOff.plotOn(frame, Name=name, LineColor=rt.kBlue)
        # legend.AddEntry(frame.getObject(int(frame.numItems())-1), events_bsOff_label, "L")
        # sigma_val = round(sigma_bsOff.getVal(), 3)
        # sigma_err = round(sigma_bsOff.getError(), 3)
        # mean_val = round(MH_bsOff.getVal(), 3)
        # mean_err = round(MH_bsOff.getError(), 3)
        # chi2_o_ndf_off = model_bsOff.createChi2(hist_bsOff)
        # chiSquare_dof_off = round(chi2_o_ndf_off.getVal(), 3)

        # legend.AddEntry("", f"   mean: {mean_val} +- {mean_err}", "")
        # legend.AddEntry("", f"   sigma: {sigma_val} +- {sigma_err}", "")
        # legend.AddEntry("", f"   chi2/dof: {chiSquare_dof_off}", "")

        # append results in a csv file, with format: category, sigma BSOn +/- error, sigma BSOFF +/- error, % diff
        # with open(f"fit_results_{self.control_region}.csv", "a") as f:
        #     suffix = "Inclusive" if suffix == None else suffix
        #     f.write(f"{suffix},{round(sigma_bsOn.getVal(),3)},{round(sigma_bsOn.getError(),3)},{round(sigma_bsOff.getVal(),3)},{round(sigma_bsOff.getError(),3)},{round(100*(abs(sigma_bsOff.getVal()-sigma_bsOn.getVal()))/sigma_bsOff.getVal(),3)}\n")

        # # get a latex style table
        # with open(f"fit_results_{self.control_region}.txt", "a") as f:
        #     suffix = "Inclusive" if suffix == None else suffix
        #     f.write(f"{suffix} & {round(sigma_bsOn.getVal(),3)} $\\pm$ {round(sigma_bsOn.getError(),3)} & {round(sigma_bsOff.getVal(),3)} $\\pm$ {round(sigma_bsOff.getError(),3)} & {round(100*(abs(sigma_bsOff.getVal()-sigma_bsOn.getVal()))/sigma_bsOff.getVal(),3)} \\\\ \n")



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

    def fit_dimuonInvariantMass_DCBXBW(self, events_dict=None, outdir="plots", suffix=None):
        """
        Generate a histogram from dimuon mass and weight, fit with DCB × BW (Double Crystal Ball × Breit-Wigner),
        and return fit parameters: sigma and chi2/dof.
        """
        if events_dict is None:
            events_dict = self.events

        print("==================================================")
        print(f"Fitting {len(events_dict)} datasets...")
        print("==================================================")

        # -----------------------------
        #  Setup Canvas & Legend
        # -----------------------------
        canvas = rt.TCanvas(f"Canvas_{self.control_region}", f"Canvas_{self.control_region}", 800, 800)
        canvas.cd()
        legend = rt.TLegend(0.6, 0.60, 0.9, 0.9)

        # -----------------------------
        #  Define Mass Variable
        # -----------------------------
        mass_name = "mh_ggh"
        if self.control_region in ["z-peak", "z_peak"]:
            mass = rt.RooRealVar(mass_name, mass_name, 91.2, 70, 110)  # Z-peak
        elif self.control_region == "signal":
            mass = rt.RooRealVar(mass_name, mass_name, 125, 115, 135)  # Higgs peak

        frame = mass.frame()
        mass_fit_range = self.mass_fit_range[self.control_region]
        nbins = 250  # Optimal bin count to reduce statistical noise
        mass.setBins(nbins)

        # -----------------------------
        #  Define Breit-Wigner Model
        # -----------------------------
        mean = rt.RooRealVar("mean", "mean", 91.1880, 91, 92)  # PDG value for Z
        width = rt.RooRealVar("width", "width", 2.4955, 1.0, 3.0)  # PDG Z width (fixed)
        bw = rt.RooBreitWigner("bw", "Breit-Wigner", mass, mean, width)
        # width.setConstant(True)
        mean.setConstant(True)

        # -----------------------------
        #  Define DCB Model
        # -----------------------------
        mean_bsOn = rt.RooRealVar("mean_bsOn", "mean_bsOn", 0, -10, 10)  # Offset relative to BW
        sigma_bsOn = rt.RooRealVar("sigma", "sigma", 2.0, 0.1, 4.0)  # Gaussian resolution
        alpha1_bsOn = rt.RooRealVar("alpha1", "alpha1", 1.5, 0.01, 65.0)
        n1_bsOn = rt.RooRealVar("n1", "n1", 10.0, 0.01, 185.0)
        alpha2_bsOn = rt.RooRealVar("alpha2", "alpha2", 1.5, 0.01, 65.0)
        n2_bsOn = rt.RooRealVar("n2", "n2", 25.0, 0.01, 385.0)

        model_DCB = rt.RooCrystalBall("DCB", "DCB Fit", mass, mean_bsOn, sigma_bsOn, alpha1_bsOn, n1_bsOn, alpha2_bsOn, n2_bsOn)

        # -----------------------------
        #  Convolution (BW × DCB)
        # -----------------------------
        model = rt.RooFFTConvPdf("DCB_BW", "DCB × BW Fit", mass, bw, model_DCB)
        mass.setBins(10000, "cache")  # FFT Convolution bins for accuracy

        colors = [rt.kBlue, rt.kRed, rt.kGreen+2, rt.kMagenta, rt.kCyan, rt.kOrange, rt.kViolet]

        # -----------------------------
        #  Fit All Event Keys
        # -----------------------------
        fit_results = {}
        for idx, (label, events) in enumerate(events_dict.items()):
            print(f"Processing dataset: {label}")

            dimuon_mass = ak.to_numpy(events.dimuon_mass)
            wgt = ak.to_numpy(events.wgt_nominal)

            hist = self.generateRooHist(mass, dimuon_mass, wgt, name=label.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", ""))
            hist = self.normalizeRooHist(mass, hist)

            # Fit
            rt.EnableImplicitMT()
            _ = model.fitTo(hist, EvalBackend="cpu", Save=True, SumW2Error=True)
            _ = model.fitTo(hist, EvalBackend="cpu", Save=True, SumW2Error=True)
            fit_result = model.fitTo(hist, EvalBackend="cpu", Save=True, SumW2Error=True)
            fit_result.Print()
            fit_results[label] = fit_result

            # Plot
            color = colors[idx % len(colors)]  # Assign different color to each dataset
            hist.plotOn(frame, rt.RooFit.Name(f"hist_{label.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")}"), rt.RooFit.MarkerColor(color), rt.RooFit.LineColor(color))
            model.plotOn(frame, rt.RooFit.Name(f"model_{label.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")}"), rt.RooFit.LineColor(color))

            # Compute Fit Metrics
            sigma_val = round(sigma_bsOn.getVal(), 3)
            sigma_err = round(sigma_bsOn.getError(), 3)
            mean_val = round(mean_bsOn.getVal(), 3)
            mean_err = round(mean_bsOn.getError(), 3)

            # print("Frame items:", frame.numItems())
            # for i in range(frame.numItems()):
                # print(f"Item {i}: {frame.getObject(i).GetName()}")


            chi2_o_ndf = model.createChi2(hist, rt.RooFit.Extended(True), rt.RooFit.DataError(rt.RooAbsData.SumW2))
            chiSquare_dof = round(chi2_o_ndf.getVal(), 3)

            new_nfree_params = fit_result.floatParsFinal().getSize()
            chi2 = frame.chiSquare(f"model_{label.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")}", f"hist_{label.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")}", new_nfree_params)

            # Add Legend Entries
            legend.AddEntry(frame.getObject(int(frame.numItems()) - 1), f"{label} (Fit)", "L")
            # legend.AddEntry("", f"   mean: {mean_val} #pm {mean_err}", "")
            legend.AddEntry("", f"   sigma: {sigma_val} #pm {sigma_err}", "")
            legend.AddEntry("", f" #chi^2 : {round(chi2, 3)}", "")
            legend.AddEntry("", f" #chi^2 / ndf: {round(chi2/new_nfree_params,3)}", "")

            print(f"\n🔹 {label} → Chi²: {chi2:.3f} | Free Params: {new_nfree_params}")
            print(f"   Chi²/NDF: {chi2/new_nfree_params:.3f}")
            # break

        # -----------------------------
        #  Save Results
        # -----------------------------
        frame.SetYTitle(f"A.U.")
        frame.SetXTitle(f"Dimuon Mass (GeV)")
        frame.SetTitle("")

        frame.Draw()
        legend.Draw()
        canvas.Update()
        canvas.Draw()

        save_filename = f"{outdir}/{self.year}/{self.directoryTag}/fitPlot_{self.control_region}_{suffix}_nbins{nbins}.pdf"
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        canvas.SaveAs(save_filename)
        canvas.SaveAs(save_filename.replace(".pdf", ".png"))

        # return fit_results


    def fit_dimuonInvariantMass_DCBXBW_OLD(self, events_dict=None, outdir="plots", suffix=None):
        """
        Generate a histogram from dimuon mass and weight, fit with DCB × BW (Double Crystal Ball × Breit-Wigner),
        and return fit parameters: sigma and chi2/dof.
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

        # -----------------------------
        #  Setup Canvas & Legend
        # -----------------------------
        canvas = rt.TCanvas(f"Canvas_{self.control_region}", f"Canvas_{self.control_region}", 800, 800)
        canvas.cd()
        legend = rt.TLegend(0.6, 0.60, 0.9, 0.9)

        # -----------------------------
        #  Define Mass Variable
        # -----------------------------
        mass_name = "mh_ggh"
        if self.control_region in ["z-peak", "z_peak"]:
            mass = rt.RooRealVar(mass_name, mass_name, 91.2, 70, 110)  # Z-peak
        elif self.control_region == "signal":
            mass = rt.RooRealVar(mass_name, mass_name, 125, 115, 135)  # Higgs peak

        frame = mass.frame()
        mass_fit_range = self.mass_fit_range[self.control_region]
        nbins = 250
        mass.setBins(nbins)

        # -----------------------------
        #  Create Histogram from Data
        # -----------------------------
        dimuon_mass = ak.to_numpy(events_bsOn.dimuon_mass)
        wgt = ak.to_numpy(events_bsOn.wgt_nominal)
        hist_bsOn = self.generateRooHist(mass, dimuon_mass, wgt, name=events_bsOn_label)
        hist_bsOn = self.normalizeRooHist(mass, hist_bsOn)

        print(f"fitPlot_ggh hist_bsOn: {hist_bsOn}")

        # -----------------------------
        #  Define Breit-Wigner Model
        # -----------------------------
        mean = rt.RooRealVar("mean", "mean", 91.1880, 91, 92)  # Z mass: 91.1880 GeV from PDG
        width = rt.RooRealVar("width", "width", 2.4955, 1.0, 3.0)  # Natural Z width 2.4955 GeV from PDG
        bw = rt.RooBreitWigner("bw", "Breit-Wigner", mass, mean, width)
        width.setConstant(True)  # Fix the width parameter
        mean.setConstant(True)  # Fix the mean parameter

        # -----------------------------
        #  Define DCB Model
        # -----------------------------
        # MH_bsOn = rt.RooRealVar("MH", "MH", 91.2, 85, 95)
        mean_bsOn = rt.RooRealVar("mean_bsOn", "mean_bsOn", 0, -10, 10)  # mean is mean relative to BW
        sigma_bsOn = rt.RooRealVar("sigma", "sigma", 2.0, 0.1, 4.0)  # Gaussian resolution
        alpha1_bsOn = rt.RooRealVar("alpha1", "alpha1", 1.5, 0.01, 65.0)
        n1_bsOn = rt.RooRealVar("n1", "n1", 10.0, 0.01, 185.0)
        alpha2_bsOn = rt.RooRealVar("alpha2", "alpha2", 1.5, 0.01, 65.0)
        n2_bsOn = rt.RooRealVar("n2", "n2", 25.0, 0.01, 385.0)

        model_DCB = rt.RooCrystalBall("DCB", "DCB Fit", mass, mean_bsOn, sigma_bsOn, alpha1_bsOn, n1_bsOn, alpha2_bsOn, n2_bsOn)

        # -----------------------------
        #  Convolution (BW × DCB)
        # -----------------------------
        model = rt.RooFFTConvPdf("DCB_BW", "DCB × BW Fit", mass, bw, model_DCB)
        mass.setBins(10000, "cache")  # FFT Convolution bins for accuracy

        # -----------------------------
        #  Perform Fit
        # -----------------------------
        rt.EnableImplicitMT()
        _ = model.fitTo(hist_bsOn, EvalBackend="cpu", Save=True, SumW2Error=True)
        _ = model.fitTo(hist_bsOn, EvalBackend="cpu", Save=True, SumW2Error=True)
        fit_result = model.fitTo(hist_bsOn, EvalBackend="cpu", Save=True, SumW2Error=True)
        fit_result.Print()

        # -----------------------------
        #  Plot Fit
        # -----------------------------
        hist_bsOn.plotOn(frame, rt.RooFit.Name("hist_bsOn"), rt.RooFit.MarkerColor(rt.kGreen), rt.RooFit.LineColor(rt.kGreen), Invisible=False)
        model.plotOn(frame, rt.RooFit.Name("model_bsOn"), rt.RooFit.LineColor(rt.kBlue))

        # -----------------------------
        #  Compute Fit Metrics
        # -----------------------------
        sigma_val = round(sigma_bsOn.getVal(), 3)
        sigma_err = round(sigma_bsOn.getError(), 3)
        mean_val = round(mean_bsOn.getVal(), 3)
        mean_err = round(mean_bsOn.getError(), 3)

        chi2_o_ndf_on = model.createChi2(hist_bsOn, rt.RooFit.Extended(True), rt.RooFit.DataError(rt.RooAbsData.SumW2))
        chiSquare_dof_on = round(chi2_o_ndf_on.getVal(), 3)

        legend.AddEntry(frame.getObject(int(frame.numItems()) - 1), "Fit Model (DCB x BW)", "L")
        legend.AddEntry("", f"   mean: {mean_val} #pm {mean_err}", "")
        legend.AddEntry("", f"   sigma: {sigma_val} #pm {sigma_err}", "")

        new_nfree_params = fit_result.floatParsFinal().getSize()
        print(f"\n\n====> n_free_params: {new_nfree_params}")
        chi2_ndf = frame.chiSquare("model_bsOn", "hist_bsOn", new_nfree_params)
        print(f"Chi2 = {chi2_ndf:.3f}")
        print(f"Chi2/NDF = {chi2_ndf/new_nfree_params:.3f}")
        print("==================================================")

        # legend.AddEntry("", f"   chi2/dof: {chiSquare_dof_on}", "")
        legend.AddEntry("", f" #chi^{{2}} : {round(chi2_ndf,3)}", "")
        legend.AddEntry("", f" #chi^{{2}} / ndf: {round(chi2_ndf/new_nfree_params,3)}", "")

        # -----------------------------
        #  Save Results
        # -----------------------------
        frame.SetYTitle(f"A.U.")
        frame.SetXTitle(f"Dimuon Mass (GeV)")
        frame.SetTitle(f"")

        frame.Draw()
        legend.Draw()
        canvas.Update()
        canvas.Draw()

        save_filename = f"{outdir}/{self.year}/{self.directoryTag}/fitPlot_{self.control_region}_{suffix}_nbins{nbins}.pdf"
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        canvas.SaveAs(save_filename)
        canvas.SaveAs(save_filename.replace(".pdf", ".png"))

    def fit_dimuonInvariantMass_DCBXBW_Unbinned(self, events_dict=None, outdir="plots", suffix=None):
        """
        Perform an unbinned fit to the dimuon mass using a DCB × BW model.
        Returns fit result and plots.

        Note: Uses RooDataSet (not RooDataHist) for unbinned maximum likelihood fit.
        """
        if events_dict is None:
            events_dict = self.events

        # Setup plotting
        canvas = rt.TCanvas("canvas", "canvas", 800, 800)
        legend = rt.TLegend(0.6, 0.6, 0.9, 0.9)

        colors = [rt.kBlue, rt.kRed, rt.kGreen+2, rt.kMagenta, rt.kCyan, rt.kOrange, rt.kViolet]

        # Mass observable
        if self.control_region in ["z-peak", "z_peak"]:
            mass = rt.RooRealVar("mh_ggh", "Dimuon Mass", 91.2, 70, 110)
            mean_bw_value = 91.1880
            width_bw_value = 2.4955
        elif self.control_region == "signal":
            mass = rt.RooRealVar("mh_ggh", "Dimuon Mass", 125.2, 115, 135)
            mean_bw_value = 125.2
            width_bw_value = 0.0037
        else:
            raise ValueError(f"Unknown control region: {self.control_region}")

        frame = mass.frame()

        # BW parameters (fixed)

        mean_bw = rt.RooRealVar("mean_bw", "mean_bw", mean_bw_value, mean_bw_value - width_bw_value, mean_bw_value + width_bw_value)
        width_bw = rt.RooRealVar("width_bw", "width_bw", width_bw_value, width_bw_value - width_bw_value*0.25, width_bw_value + width_bw_value*0.25)
        mean_bw.setConstant(True)
        width_bw.setConstant(True)
        bw = rt.RooBreitWigner("bw", "Breit-Wigner", mass, mean_bw, width_bw)

        # DCB parameters (floating)
        mean_dcb = rt.RooRealVar("mean_dcb", "mean_dcb", 0.0, -10, 10)
        sigma = rt.RooRealVar("sigma", "sigma", 2.0, 0.5, 5.0)
        alpha1 = rt.RooRealVar("alpha1", "alpha1", 1.5, 0.1, 10.0)
        n1 = rt.RooRealVar("n1", "n1", 10.0, 1.0, 100.0)
        alpha2 = rt.RooRealVar("alpha2", "alpha2", 1.5, 0.1, 10.0)
        n2 = rt.RooRealVar("n2", "n2", 10.0, 1.0, 100.0)

        dcb = rt.RooCrystalBall("DCB", "DCB", mass, mean_dcb, sigma, alpha1, n1, alpha2, n2)

        # Convolution: BW × DCB
        model = rt.RooFFTConvPdf("DCB_BW", "DCB x BW", mass, bw, dcb)
        mass.setBins(10000, "cache")

        # Choose one dataset for now (unbinned fit only works per-dataset)
        for idx, (label, events) in enumerate(events_dict.items()):
            print(f"Performing unbinned fit on: {label}")

            # Extract mass and weight as numpy arrays
            dimuon_mass = ak.to_numpy(events.dimuon_mass)
            weights = ak.to_numpy(events.wgt_nominal)

            # Create RooDataSet (unbinned)
            data = rt.RooDataSet(label, label, rt.RooArgSet(mass), rt.RooFit.WeightVar("wgt"))
            wgt_var = rt.RooRealVar("wgt", "wgt", 1.0)

            for mval, w in zip(dimuon_mass, weights):
                mass.setVal(mval)
                wgt_var.setVal(w)
                data.add(rt.RooArgSet(mass), w)

            # Fit
            fit_result = model.fitTo(data, EvalBackend="cpu", Save=True, SumW2Error=True)
            fit_result.Print()

            # Plot
            color = colors[idx % len(colors)]  # Assign different color to each dataset
            data.plotOn(frame, rt.RooFit.Name("data"), rt.RooFit.MarkerColor(color))
            model.plotOn(frame, rt.RooFit.Name("DCB_BW"), rt.RooFit.LineColor(color))

            # Summary
            sigma_val = round(sigma.getVal(), 3)
            sigma_err = round(sigma.getError(), 3)
            mean_val = round(mean_dcb.getVal(), 3)
            mean_err = round(mean_dcb.getError(), 3)

            n_params = fit_result.floatParsFinal().getSize()
            chi2 = frame.chiSquare("DCB_BW", "data", n_params)
            legend.AddEntry(frame.getObject(int(frame.numItems()) - 1), f"{label} (DCB x BW)", "L")
            # legend.AddEntry("", f"mean: {mean_val} #pm {mean_err}", "")
            legend.AddEntry("", f"sigma: {sigma_val} #pm {sigma_err}", "")
            legend.AddEntry("", f"#chi^{2}/NDF: {round(chi2/n_params, 3)}", "")

            # break  # only one fit for now

        frame.SetXTitle("Dimuon Mass (GeV)")
        frame.SetYTitle("Events")
        frame.Draw()
        legend.Draw()
        canvas.Update()
        canvas.Draw()

        save_path = f"{outdir}/{self.year}/{self.directoryTag}/fitPlot_Unbinned_{self.control_region}_{suffix}.pdf"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        canvas.SaveAs(save_path)
        canvas.SaveAs(save_path.replace(".pdf", ".png"))
