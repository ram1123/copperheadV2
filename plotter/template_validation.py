import pickle
import hist
import numpy as np
import os
import re
import ROOT

from modules import selection

# No stat box
ROOT.gStyle.SetOptStat(000)

# --- User Input ---
# pkl_file = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_17July/stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_17July_July31_Rebinned/2018/vbf_powheg_dipole_hist.pkl"
pkl_file = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_Latest/2018/vbf_powheg_dipole_hist.pkl"
output_dir = "variation_validation_plots/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/2018/"
variable_axis_name = "score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt"

# Your custom binning
bins = selection.binning

os.makedirs(output_dir, exist_ok=True)

with open(pkl_file, "rb") as f:
    h = pickle.load(f)

axis_names = [ax.name for ax in h.axes]
var_idx = axis_names.index("variation")
x_idx = axis_names.index(variable_axis_name)

variation_labels = list(h.axes[var_idx])

# Find systematics with up/down
sys_set = set()
sys_pattern = re.compile(r"^(.*)_up$")
for var in variation_labels:
    m = sys_pattern.match(var)
    if m and f"{m.group(1)}_down" in variation_labels:
        sys_set.add(m.group(1))
sys_list = sorted(sys_set)

def get_var_idx(varstr):
    return variation_labels.index(varstr)

def project_to_1d(histogram, var_idx, x_idx, var_val, region_idx):
    sel = {}
    for i, ax in enumerate(histogram.axes):
        if i == var_idx:
            sel[ax.name] = get_var_idx(var_val)
        elif i == x_idx:
            continue  # keep variable axis for projection
        elif ax.name == "region":
            sel[ax.name] = region_idx  # select only 'h-sidebands'
        else:
            sel[ax.name] = slice(None)  # sum over other axes
    arr = histogram[sel].values()
    while arr.ndim > 1:
        arr = arr.sum(axis=0)
    return arr

region_labels = list(h.axes[axis_names.index("region")])
region_idx = region_labels.index("h-sidebands") # use h-peak or h-sidebands

for sys in sys_list:
    print(f"Plotting: {sys}_up and {sys}_down")
    try:
        nom = project_to_1d(h, var_idx, x_idx, "nominal", region_idx)
        up  = project_to_1d(h, var_idx, x_idx, f"{sys}_up", region_idx)
        dn  = project_to_1d(h, var_idx, x_idx, f"{sys}_down", region_idx)
    except Exception as e:
        print(f"  Skipping {sys} due to error: {e}")
        continue

    # --- PyROOT plotting ---
    ROOT.gROOT.SetBatch(True)
    c1 = ROOT.TCanvas(f"c_{sys}", f"Template Validation: {sys}", 800, 600)
    c1.SetLeftMargin(0.13)
    c1.SetBottomMargin(0.12)

    nbins = len(bins)-1
    h_nom = ROOT.TH1F("h_nom", "", nbins, bins)
    h_up  = ROOT.TH1F("h_up", "", nbins, bins)
    h_dn  = ROOT.TH1F("h_dn", "", nbins, bins)

    for i in range(nbins):
        h_nom.SetBinContent(i+1, nom[i])
        h_up.SetBinContent(i+1, up[i])
        h_dn.SetBinContent(i+1, dn[i])

    h_nom.SetLineColor(16)
    h_nom.SetLineWidth(2)
    h_nom.SetTitle(f"{sys} (region: h-sidebands);{variable_axis_name};Entries")
    h_up.SetLineColor(3)
    h_up.SetLineStyle(2)
    h_dn.SetLineColor(6)
    h_dn.SetLineStyle(3)

    # h_nom.SetMinimum(0.001)
    # h_nom.SetMaximum(10e9)
    h_nom.Draw("hist")
    h_up.Draw("hist same")
    h_dn.Draw("hist same")

    leg = ROOT.TLegend(0.7, 0.7, 0.99, 0.89)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(h_nom, "Nominal", "l")
    leg.AddEntry(h_up, f"up", "l")
    leg.AddEntry(h_dn, f"down", "l")
    leg.Draw()

    c1.RedrawAxis()
    c1.SaveAs(os.path.join(output_dir, f"{sys}_template_validation.pdf"))
    # To also save log scale:
    c1.SetLogy(1)
    c1.SaveAs(os.path.join(output_dir, f"{sys}_template_validation_log.pdf"))
    del h_nom, h_up, h_dn, c1

print(f"Done! plots are in: {output_dir}")
