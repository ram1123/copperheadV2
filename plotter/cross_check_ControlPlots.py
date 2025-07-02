#!/usr/bin/env python3

from dask.distributed import Client
import dask.dataframe as dd
import dask.array as da
import numpy as np
import matplotlib.pyplot as plt


use_gateway = False
if use_gateway:
    from dask_gateway import Gateway
    gateway = Gateway(
        "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
        proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
    )
    cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
    client = gateway.connect(cluster_info.name).get_client()
    print("Gateway Client created")
else: # use local cluster
    client = Client(n_workers=15,  threads_per_worker=1, processes=True, memory_limit='30 GiB')
    print("Local scale Client created")


years = ["2016", "2017", "2018"]
LOAD_PATH = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/" \
            "stage1_output/{year}/f1_0/{sample}/*/*.parquet"

bkg_MC_order = ["OTHER", "EWK", "VV", "TOP", "DY", "DYVBF", "GGH", "VBF"]
group_dict = {
    "DATA": ["data_A", "data_B", "data_C", "data_D", "data_E", "data_F", "data_G", "data_H"],
    "DY": ["dy_M-100To200_MiNNLO", "dy_M-50_MiNNLO"],
    "DYVBF": ["dy_VBF_filter_NewZWgt"],
    "TOP": ["ttjets_dl", "ttjets_sl", "st_tw_top", "st_tw_antitop", "st_t_top", "st_t_antitop"],
    "EWK": ["ewk_lljj_mll50_mjj120"],
    "VV": ["ww_2l2nu", "wz_3lnu", "wz_2l2q", "wz_1l1nu2q", "zz"],
    "OTHER": ["www", "wwz", "wzz", "zzz"],
    # "GGH": ["ggh_powhegPS"],
    # "VBF": ["vbf_powheg_dipole"]
}

# histogram edges
# mass_bins = np.linspace(110, 150, 51)  # 50 bins from 110 to 150 GeV
mass_bins = np.linspace(0, 300, 31)  # 30 bins from 0 to 300 GeV
bin_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])

def apply_vbf_h_sidebands(df):
    m = df.dimuon_mass
    sideband = ((m > 110) & (m < 115.03)) | ((m > 135.03) & (m < 150))

    btag_cut = (df.nBtagLoose_nominal >= 2) | (df.nBtagMedium_nominal >= 1)

    vbf = (df.jj_mass_nominal > 400) & (df.jj_dEta_nominal > 2.5) & (df.jet1_pt_nominal > 35)

    return df[sideband & vbf & ~btag_cut]

# 1) Data: no weights
print("Processing Data...")
data_paths = [LOAD_PATH.format(year=yr, sample=s)
              for yr in years for s in group_dict["DATA"]]
df_data = dd.read_parquet(data_paths)
df_data = apply_vbf_h_sidebands(df_data)

# simple unweighted histogram
hist_data, _ = np.histogram(df_data.dimuon_pt.compute(), bins=mass_bins)

# 2) MC groups: use wgt_nominal
print("Processing MC groups...")
hist_mc = {}
for grp in bkg_MC_order:
    print(f"Processing MC group: {grp}")
    samples = group_dict.get(grp, [])
    if not samples:
        continue

    paths = [LOAD_PATH.format(year=yr, sample=s)
             for yr in years for s in samples]
    df_mc = dd.read_parquet(paths)
    df_mc = apply_vbf_h_sidebands(df_mc)

    # if both key, DY and DYVBF, is there in the bkg_MC_order then add an additional cut
    # of gjj_mass < 350 GeV for DY and gjj_mass > 350 GeV for DYVBF
    if "DY" in bkg_MC_order and "DYVBF" in bkg_MC_order:
        if grp == "DY":
            df_mc = df_mc[df_mc.gjj_mass <= 350]
        elif grp == "DYVBF":
            df_mc = df_mc[df_mc.gjj_mass > 350]

    # pull out two aligned Dask arrays: one for mass, one for weight
    arr_mass = df_mc["dimuon_pt"].to_dask_array(lengths=True)
    arr_wt   = df_mc["wgt_nominal"].to_dask_array(lengths=True)

    # ensure mass and weight array have same chunk structure
    arr_wt = arr_wt.rechunk(arr_mass.chunks)

    # build a weighted histogram
    hist, _ = da.histogram(
        arr_mass,
        bins=mass_bins,
        weights=arr_wt
    )
    hist_mc[grp] = hist.compute()


# 3) Plot
print("Plotting...")
import ROOT

# ─── build ROOT TH1s from your numpy histograms ────────────────────────────────

nbins = len(mass_bins) - 1
xlow  = mass_bins[0]
xhigh = mass_bins[-1]

# 1) data histogram
h_data = ROOT.TH1F("h_data","Data / MC in VBF H-sidebands", nbins, xlow, xhigh)
h_data.Sumw2()
legends = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)
for i, c in enumerate(hist_data):
    h_data.SetBinContent(i+1, c)
    h_data.SetBinError(  i+1, np.sqrt(c) )  # if you want Poisson errors
legends.AddEntry(h_data, "Data", "lep")

# 2) sum all MC into one total
h_mc_tot = ROOT.TH1F("h_mc_tot","MC total", nbins, xlow, xhigh)
h_mc_tot.Sumw2()
# for the stacked display we'll also keep each component
stack = ROOT.THStack("stack","")

# pick a few ROOT colors for your groups (you can adjust)
root_colors = [
    ROOT.kOrange-3,
    ROOT.kGreen+1,
    ROOT.kAzure-9,
    ROOT.kRed-7,
    ROOT.kMagenta-7,
    ROOT.kCyan+1,
    ROOT.kBlue-3,
    ROOT.kGray+2,
]

    # petroff10 = ListedColormap(["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"])

for idx, grp in enumerate(bkg_MC_order):
    vals = hist_mc.get(grp)
    if vals is None:
        continue
    h = ROOT.TH1F(f"h_{grp}", grp, nbins, xlow, xhigh)
    h.Sumw2()
    for i, c in enumerate(vals):
        h.SetBinContent(i+1, c)
    col = root_colors[idx % len(root_colors)]
    h.SetFillColor(col)
    h.SetLineColor(col)
    h_mc_tot.Add(h)
    stack.Add(h)
    legends.AddEntry(h, grp, "f")

# ─── now make a TRatioPlot ────────────────────────────────────────────────────

ROOT.gROOT.SetBatch(True)
c = ROOT.TCanvas("c","Data/MC VBF H-sidebands",800,800)

# Define the ratio plot

# set the upper range of upper plot to 10^9
stack.SetMaximum(1e9)
stack.SetMinimum(1e-2)

ratio_plot = ROOT.TRatioPlot(stack, h_data)
ratio_plot.Draw()

ratio_plot.GetUpperPad().SetLogy()
ratio_plot.GetUpperPad().SetGridx()
ratio_plot.GetUpperPad().SetGridy()
ratio_plot.GetLowerRefYaxis().SetRangeUser(0.5, 1.5)
ratio_plot.GetLowerRefYaxis().SetTitle("Data / MC")


legends.Draw("same")

c.SaveAs("vbf_h_sidebands_data_mc_VBFFilter_pt.root")
c.SaveAs("vbf_h_sidebands_data_mc_VBFFilter_pt.pdf")


# Close the Dask client
client.close()
