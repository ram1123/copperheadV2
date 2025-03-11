import ROOT as rt
import awkward as ak

# define function to return nbins, xmin, xmax, xtitle for a given variable
def get_hist_params(var):
    if "mu1_pt" in var:      return 50, 0, 250, "p_{T} [GeV]"
    elif "mu2_pt" in var:   return 50, 0, 100, "p_{T} [GeV]"
    elif "dimuon_pt" in var: return 50, 0, 150, "p_{T} [GeV]"
    elif "dimuon_mass" in var: return 50, 70, 110, "m_{#mu#mu} [GeV]" # z-peak
    # elif "dimuon_mass" in var: return 50, 100, 160, "m_{#mu#mu} [GeV]" # signal
    elif "pt" in var:           return 50, 0, 300, "p_{T} [GeV]"
    elif "eta" in var:          return 50, -3, 3, "#eta"
    elif "phi" in var:          return 25, -3.15, 3.15, "#phi"
    elif "mass" in var:       return 50, 100, 150, "m_{#mu#mu} [GeV]"
    else:                           return 50, 0, 300, var

def plot_muon_dimuon_kinematics(events, save_prefix="kinematics"):
    # Individual muon kinematics
    mu1_pt = ak.to_numpy(events.mu1_pt)
    mu1_eta = ak.to_numpy(events.mu1_eta)
    mu1_phi = ak.to_numpy(events.mu1_phi)

    mu2_pt = ak.to_numpy(events.mu2_pt)
    mu2_eta = ak.to_numpy(events.mu2_eta)
    mu2_phi = ak.to_numpy(events.mu2_phi)

    # Dimuon kinematics
    dimuon_pt = ak.to_numpy(events.dimuon_pt)
    dimuon_eta = ak.to_numpy(events.dimuon_eta)
    dimuon_phi = ak.to_numpy(events.dimuon_phi)

    # Define histograms
    hist_mu1_pt = rt.TH1D("hist_mu1_pt", "Muon 1 pT; pT [GeV]; Events", 50, 0, max(mu1_pt)*1.1)
    hist_mu2_pt = rt.TH1D("hist_mu2_pt", "Muon 2 pT; pT [GeV]; Events", 50, 0, max(mu2_pt)*1.1)
    hist_dimuon_pt = rt.TH1D("hist_dimuon_pt", "Dimuon pT; pT [GeV]; Events", 50, 0, max(dimuon_pt)*1.1)

    hist_mu1_eta = rt.TH1D("hist_mu1_eta", "Muon 1 eta; eta; Events", 50, -2.5, 2.5)
    hist_mu2_eta = rt.TH1D("hist_mu2_eta", "Muon 2 eta; eta; Events", 50, -2.5, 2.5)
    hist_dimuon_eta = rt.TH1D("hist_dimuon_eta", "Dimuon eta; eta; Events", 50, -2.5, 2.5)

    hist_mu1_phi = rt.TH1D("hist_mu1_phi", "Muon 1 phi; phi; Events", 50, -3.14, 3.14)
    hist_mu2_phi = rt.TH1D("hist_mu2_phi", "Muon 2 phi; phi; Events", 50, -3.14, 3.14)
    hist_dimuon_phi = rt.TH1D("hist_dimuon_phi", "Dimuon phi; phi; Events", 50, -3.14, 3.14)

    # Fill histograms
    for pt in mu1_pt:
        hist_mu1_pt.Fill(pt)
    for pt in mu2_pt:
        hist_mu2_pt.Fill(pt)
    for pt in dimuon_pt:
        hist_dimuon_pt.Fill(pt)

    for eta in mu1_eta:
        hist_mu1_eta.Fill(eta)
    for eta in mu2_eta:
        hist_mu2_eta.Fill(eta)
    for eta in dimuon_eta:
        hist_dimuon_eta.Fill(eta)

    for phi in mu1_phi:
        hist_mu1_phi.Fill(phi)
    for phi in mu2_phi:
        hist_mu2_phi.Fill(phi)
    for phi in dimuon_phi:
        hist_dimuon_phi.Fill(phi)

    # Plot and save histograms
    c = rt.TCanvas("c", "c", 800, 600)

    hist_mu1_pt.Draw("HIST")
    hist_mu2_pt.SetLineColor(rt.kRed)
    hist_mu2_pt.Draw("HIST SAME")
    hist_dimuon_pt.SetLineColor(rt.kGreen)
    hist_dimuon_pt.Draw("HIST SAME")
    c.BuildLegend()
    c.SaveAs(f"{save_prefix}_pT.pdf")

    hist_mu1_eta.Draw("HIST")
    hist_mu2_eta.SetLineColor(rt.kRed)
    hist_mu2_eta.Draw("HIST SAME")
    # hist_dimuon_eta.SetLineColor(rt.kGreen)
    # hist_dimuon_eta.Draw("HIST SAME")
    c.BuildLegend()
    c.SaveAs(f"{save_prefix}_eta.pdf")

    hist_mu1_phi.Draw("HIST")
    hist_mu2_phi.SetLineColor(rt.kRed)
    hist_mu2_phi.Draw("HIST SAME")
    hist_dimuon_phi.SetLineColor(rt.kGreen)
    hist_dimuon_phi.Draw("HIST SAME")
    c.BuildLegend()
    c.SaveAs(f"{save_prefix}_phi.pdf")

# Example usage after loading your data:
# plot_muon_dimuon_kinematics(events_bs_on, save_prefix="bs_on")
# plot_muon_dimuon_kinematics(events_bs_off, save_prefix="bs_off")


import ROOT as rt
import awkward as ak

def plot_muon_dimuon_kinematics(events, save_prefix="kinematics"):
    # Individual muon kinematics
    mu1_pt = ak.to_numpy(events.mu1_pt)
    mu1_eta = ak.to_numpy(events.mu1_eta)
    mu1_phi = ak.to_numpy(events.mu1_phi)

    mu2_pt = ak.to_numpy(events.mu2_pt)
    mu2_eta = ak.to_numpy(events.mu2_eta)
    mu2_phi = ak.to_numpy(events.mu2_phi)

    # Dimuon kinematics
    dimuon_pt = ak.to_numpy(events.dimuon_pt)
    dimuon_eta = ak.to_numpy(events.dimuon_eta)
    dimuon_phi = ak.to_numpy(events.dimuon_phi)

    def create_hist(name, title, bins, range_min, range_max, data, color):
        hist = rt.TH1D(name, title, bins, range_min, range_max)
        for value in data:
            hist.Fill(value)
        hist.SetLineColor(color=color)
        hist.SetLineWidth(2)
        return hist

    # Plot and save histograms
    def draw_and_save(histograms, xlabel, filename):
        c = rt.TCanvas("c", "c", 800, 600)
        legend = rt.TLegend(0.7, 0.7, 0.9, 0.9)

        for hist, label in histograms:
            hist.Draw("HIST SAME")
            legend.AddEntry(hist, label, "L")

        histograms[0].GetXaxis().SetTitle(xlabel)
        histograms[0].Draw("HIST")
        legend.Draw()
        c.SaveAs(filename)

    histograms_pt = [
        create_hist(mu1_pt, rt.kBlack, "Muon 1 pT"),
        create_hist(mu2_pt, rt.kRed, "Muon 2 pT"),
        create_hist(dimuon_pt, rt.kBlue, "Dimuon pT")
    ]
    draw_and_save(histograms_pt, "p_{T} [GeV]", f"{save_prefix}_pT.pdf")

    histograms_eta = [
        create_hist(mu1_eta, rt.kBlack, "Muon 1 eta"),
        create_hist(mu2_eta, rt.kRed, "Muon 2 eta"),
        create_hist(dimuon_eta, rt.kGreen, "Dimuon eta")
    ]
    draw_and_save(histograms_eta, "#eta", f"{save_prefix}_eta.pdf")

    histograms_phi = [
        create_hist(mu1_phi, rt.kBlack, "Muon 1 phi"),
        create_hist(mu2_phi, rt.kRed, "Muon 2 phi"),
        create_hist(dimuon_phi, rt.kGreen, "Dimuon phi"),
    ]
    draw_and_save(histograms_phi, "phi", f"{save_prefix}_phi.pdf")

# Function to compare kinematics between two sets of events
def compare_kinematics(events_bs_on, events_bs_off, variable, xlabel_new, save_filename):
    rt.gStyle.SetOptStat(0000)
    data_on = ak.to_numpy(events_bs_on[variable])
    data_off = ak.to_numpy(events_bs_off[variable])

    # get_hist_params
    bins, range_min, range_max, xlabel = get_hist_params(variable)
    if xlabel_new:
        xlabel = xlabel_new

    hist_on = rt.TH1D("hist_on", f"{variable} BSC On", bins, range_min, range_max)
    hist_off = rt.TH1D("hist_off", f"{variable} BSC Off", bins, range_min, range_max)

    for val in data_on:
        hist_on.Fill(val)
    for val in data_off:
        hist_off.Fill(val)

    # Normalize histograms to unity
    hist_on.Scale(1.0/hist_on.Integral())
    hist_off.Scale(1.0/hist_off.Integral())

    hist_on.SetLineColor(rt.kGreen)
    hist_off.SetLineColor(rt.kBlue)

    hist_on.SetMaximum(max(hist_on.GetMaximum(), hist_off.GetMaximum())*1.1)

    c = rt.TCanvas("c", "c", 800, 600)
    hist_on.SetTitle(xlabel)
    hist_on.GetXaxis().SetTitle(xlabel)
    hist_on.GetYaxis().SetTitle("Events")
    # hist_on.Draw("")
    # hist_off.Draw("SAME")

    # Use TRatioPlot to plot ratio of two histograms
    rp = rt.TRatioPlot(hist_on, hist_off)
    c.SetTicks(0, 1)
    rp.Draw()

    rp.GetLowerRefYaxis().SetTitle("Ratio")
    rp.GetLowerRefYaxis().SetRangeUser(0.8, 1.2)
    rp.GetLowerRefGraph().SetMinimum(0.8)
    rp.GetLowerRefGraph().SetMaximum(1.2)

    rp.GetUpperPad().cd()
    legend = rt.TLegend(0.6, 0.7, 0.9, 0.9)
    legend.AddEntry(hist_on, "BSC On", "L")
    legend.AddEntry(hist_off, "BSC Off", "L")
    legend.Draw()

    c.SaveAs(f"{save_filename}_{variable}.pdf")

    hist_on.Delete()
    hist_off.Delete()
    c.Close()
