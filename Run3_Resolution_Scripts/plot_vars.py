import ROOT as rt
import awkward as ak
import numpy as np

# define function to return nbins, xmin, xmax, xtitle for a given variable
def get_hist_params(var):
    if "mu1_Ratio_pTErr_pt" == var: return 50, 0, 0.035, "p_{T} Error / p_{T}", 0.0, 2.0
    elif "mu2_Ratio_pTErr_pt" == var: return 50, 0, 0.035, "p_{T} Error / p_{T}", 0.0, 2.0
    elif "dimuon_ebe_mass_res_rel" == var: return 25, 0, 0.2, "m_{#mu#mu} Resolution", 0.0, 2.0
    elif "mu1_ptErr" == var: return 50, 0, 2.0, "p_{T} Error [GeV]", 0.0, 2.0
    elif "mu2_ptErr" == var: return 50, 0, 2.0, "p_{T} Error [GeV]", 0.0, 2.0
    elif "mu1_pt" == var:      return 50, 0, 180, "p_{T} [GeV]", 0.8, 1.2
    elif "mu2_pt" == var:   return 50, 0, 100, "p_{T} [GeV]", 0.8, 1.2
    elif "dimuon_pt" == var: return 50, 0, 150, "p_{T} [GeV]", 0.8, 1.2
    elif "dimuon_eta" == var: return 50, -4.7, 4.7, "#eta_{#mu#mu}", 0.8, 1.2
    # elif "dimuon_mass" == var: return 50, 70, 110, "m_{#mu#mu} [GeV]", 0.8, 1.2 # z-peak
    elif "dimuon_mass" == var: return 50, 100, 160, "m_{#mu#mu} [GeV]", 0.8, 1.2 # signal
    elif "dimuon_rapidity" == var: return 50, -2.4, 2.4, "#eta_{#mu#mu}", 0.8, 1.2
    elif "pt" in var:           return 50, 0, 300, "p_{T} [GeV]", 0.8, 1.2
    elif "eta" in var:          return 50, -2.4, 2.4, "#eta", 0.8, 1.2
    elif "phi" in var:          return 25, -3.15, 3.15, "#phi", 0.8, 1.2
    elif "mass" in var:       return 50, 100, 150, "m_{#mu#mu} [GeV]", 0.8, 1.2
    else:                           return 50, 0, 300, var, 0.0, 2.0

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

def compute_dimuon_rapidity(events):
    mass_mu = 0.10566  # GeV (muon mass) PDG value

    mu1_pt = events.mu1_pt
    mu1_eta = events.mu1_eta
    mu1_phi = events.mu1_phi
    # Formula from PDG: page 779
    # mT = sqrt(m^2 + pT^2)
    # E = mT * cosh(eta)
    # p_z = mT * sinh(eta)
    mu1_mT = np.sqrt(mass_mu**2 + mu1_pt**2)
    mu1_energy = mu1_mT * np.cosh(mu1_eta)
    mu1_pz = mu1_mT * np.sinh(mu1_eta)

    mu2_pt = events.mu2_pt
    mu2_eta = events.mu2_eta
    mu2_phi = events.mu2_phi
    mu2_mT = np.sqrt(mass_mu**2 + mu2_pt**2)
    mu2_energy = mu2_mT * np.cosh(mu2_eta)
    mu2_pz = mu2_mT * np.sinh(mu2_eta)

    dimuon_energy = mu1_energy + mu2_energy
    dimuon_pz = mu1_pz + mu2_pz

    dimuon_rapidity = 0.5 * np.log((dimuon_energy + dimuon_pz) / (dimuon_energy - dimuon_pz))

    return dimuon_rapidity

# Function to compare kinematics between two sets of events
def compare_kinematics(events_bs_on, events_bs_off, variable, xlabel_new, save_filename):
    rt.gStyle.SetOptStat(0000)

    # For bug checking: START: -----------------------------------
    # If variable is dimuon_eta, then use the mu1 and mu2 info to calculate dimuon_eta
    if variable == "dimuon_etaaaa":
        # To get di-muon eta use function: compute_dimuon_rapidity
        data_on = compute_dimuon_rapidity(events_bs_on)
        data_off = compute_dimuon_rapidity(events_bs_off)
    else:
        data_on = ak.to_numpy(events_bs_on[variable])
        data_off = ak.to_numpy(events_bs_off[variable])
    # For bug checking: END: -----------------------------------


    # data_on = ak.to_numpy(events_bs_on[variable])
    # data_off = ak.to_numpy(events_bs_off[variable])

    # get_hist_params
    bins, range_min, range_max, xlabel, ratio_range_min, ratio_range_max = get_hist_params(variable)
    print(f"variable: {variable}, bins: {bins}, range_min: {range_min}, range_max: {range_max}, xlabel: {xlabel}")
    if xlabel_new:
        xlabel = xlabel_new
    print(f"bins: {bins}, range_min: {range_min}, range_max: {range_max}, xlabel: {xlabel}")

    hist_on = rt.TH1D("hist_on", f"{variable} BSC", bins, range_min, range_max)
    hist_off = rt.TH1D("hist_off", f"{variable} GeoFit", bins, range_min, range_max)

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
    hist_on.GetYaxis().SetTitle("A.U.")
    # hist_on.Draw("")
    # hist_off.Draw("SAME")

    # Use TRatioPlot to plot ratio of two histograms
    rp = rt.TRatioPlot(hist_on, hist_off)
    c.SetTicks(0, 1)
    rp.Draw()

    rp.GetLowerRefYaxis().SetTitle("Ratio")
    rp.GetLowerRefYaxis().SetRangeUser(ratio_range_min, ratio_range_max)
    rp.GetLowerRefGraph().SetMinimum(ratio_range_min)
    rp.GetLowerRefGraph().SetMaximum(ratio_range_max)

    rp.GetUpperPad().cd()
    legend = rt.TLegend(0.6, 0.7, 0.9, 0.9)
    legend.AddEntry(hist_on, "BSC", "L")
    legend.AddEntry(hist_off, "GeoFit", "L")
    legend.Draw()

    c.SaveAs(f"{save_filename}_{variable}.pdf")

    # Save the log version of the plot
    rp.GetUpperPad().SetLogy()
    # reset the y-axis range for upper pad
    hist_on.SetMaximum(max(hist_on.GetMaximum(), hist_off.GetMaximum())*100)

    c.SaveAs(f"log_plots/{save_filename}_{variable}_log.pdf")

    hist_on.Delete()
    hist_off.Delete()
    c.Close()

# compare_kinematics_2D(events_bs_on, events_bs_off, "mu1_pt", "mu1_ptErr", "Leading Muon p_{T} [GeV]", "Leading Muon p_{T} Error [GeV]", save_filename="kinematics_comparison"+"_"+control_region)
def compare_kinematics_2D(events_bs_on, events_bs_off, variable1, variable2, xlabel_new1, xlabel_new2, save_filename):
    rt.gStyle.SetOptStat(0000)

    data_on_x = ak.to_numpy(events_bs_on[variable1])
    data_off_x = ak.to_numpy(events_bs_off[variable1])

    data_on_y = ak.to_numpy(events_bs_on[variable2])
    data_off_y = ak.to_numpy(events_bs_off[variable2])

    # get_hist_params
    bins_x, range_min_x, range_max_x, xlabel1, ratio_range_min_x, ratio_range_max_x = get_hist_params(variable1)
    bins_y, range_min_y, range_max_y, xlabel2, ratio_range_min_y, ratio_range_max_y = get_hist_params(variable2)
    print(f"variable1: {variable1}, bins: {bins_x}, range_min: {range_min_x}, range_max: {range_max_x}, xlabel: {xlabel1}")
    print(f"variable2: {variable2}, bins: {bins_y}, range_min: {range_min_y}, range_max: {range_max_y}, xlabel: {xlabel2}")
    if xlabel_new1:
        xlabel1 = xlabel_new1
    if xlabel_new2:
        xlabel2 = xlabel_new2
    print(f"bins: {bins_x}, range_min: {range_min_x}, range_max: {range_max_x}, xlabel: {xlabel1}")
    print(f"bins: {bins_y}, range_min: {range_min_y}, range_max: {range_max_y}, xlabel: {xlabel2}")

    hist_on = rt.TH2D("hist_on", f"{variable1} vs {variable2} BSC", bins_x, range_min_x, range_max_x, bins_y, range_min_y, range_max_y)
    hist_off = rt.TH2D("hist_off", f"{variable1} vs {variable2} GeoFit", bins_x, range_min_x, range_max_x, bins_y, range_min_y, range_max_y)

    for i in range(len(data_on_x)):
        hist_on.Fill(data_on_x[i], data_on_y[i])
    for i in range(len(data_off_x)):
        hist_off.Fill(data_off_x[i], data_off_y[i])

    # Normalize histograms to unity
    # hist_on.Scale(1.0/hist_on.Integral())
    # hist_off.Scale(1.0/hist_off.Integral())

    # hist_on.SetLineColor(rt.kGreen)
    # hist_off.SetLineColor(rt.kBlue)

    # hist_on.SetMaximum(max(hist_on.GetMaximum(), hist_off.GetMaximum())*1.1)

    c = rt.TCanvas("c", "c", 800, 600)
    hist_on.SetTitle(f"{xlabel1} vs {xlabel2}")
    hist_on.GetXaxis().SetTitle(xlabel1)
    hist_on.GetYaxis().SetTitle(xlabel2)
    hist_on.Draw("COLZ")
    c.SaveAs(f"{save_filename}_{variable1}_vs_{variable2}_BSC.pdf")

    c.Clear()
    hist_off.SetTitle(f"{xlabel1} vs {xlabel2}")
    hist_off.GetXaxis().SetTitle(xlabel1)
    hist_off.GetYaxis().SetTitle(xlabel2)
    hist_off.Draw("COLZ")
    c.SaveAs(f"{save_filename}_{variable1}_vs_{variable2}_GeoFit.pdf")
