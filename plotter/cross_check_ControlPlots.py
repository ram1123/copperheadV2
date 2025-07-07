#!/usr/bin/env python3

from dask.distributed import Client
from dask.distributed import performance_report
import dask.dataframe as dd
import dask.array as da
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json
import argparse
import cmsstyle as CMS

use_gateway = True
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


years = ["2018"]
LOAD_PATH = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/" \
            "stage1_output/{year}/f1_0/{sample}/*/*.parquet"
COMPACTED_PATH = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/compacted/{year}/{sample}/"
plotpath = "./plots/vbf_h_sidebands_data_mc_VBFFilter/"
os.makedirs(plotpath, exist_ok=True)


bkg_MC_order = ["OTHER", "EWK", "VV", "TOP", "DY", "DYVBF", "GGH", "VBF"]
group_dict = {
    "DATA": ["data_A", "data_B", "data_C", "data_D"], #, "data_E", "data_F", "data_G", "data_H"],
    # "DY": ["dy_M-100To200_MiNNLO", "dy_M-50_MiNNLO"],
    "DY": ["dy_M-100To200_aMCatNLO", "dy_M-50_aMCatNLO"],
    # "DYVBF": ["dy_VBF_filter_NewZWgt"],
    "TOP": ["ttjets_dl", "ttjets_sl", "st_tw_top", "st_tw_antitop", "st_t_top", "st_t_antitop"],
    "EWK": ["ewk_lljj_mll50_mjj120"],
    "VV": ["ww_2l2nu", "wz_3lnu", "wz_2l2q", "wz_1l1nu2q", "zz"],
    "OTHER": ["www", "wwz", "wzz", "zzz"],
    # "GGH": ["ggh_powhegPS"],
    # "VBF": ["vbf_powheg_dipole"]
}


# --- Argument parsing and variable selection ---
kinematic_vars = ["pt", "eta", "phi", "mass"]

parser = argparse.ArgumentParser()
parser.add_argument("--variables", nargs="+", default=["dimuon"], help="Physics object(s) to plot (dimuon, dijet, mu, jet, ...)")
parser.add_argument("--minimum_set", action="store_true", help="Use minimum set of variables")
parser.add_argument("--ifStichTwoDYs", action="store_true", help="Stitch two DY samples together")
args = parser.parse_args()

json_path = "/depot/cms/users/shar1172/copperheadV2_main/src/lib/histogram/plot_settings_vbfCat_MVA_input.json"
with open(json_path, "r") as f:
    plot_vars_config = json.load(f)

# Build variables2plot based on user selection
variables2plot = []

for particle in args.variables:
    if "dimuon" in particle:
        variables2plot.append(f"{particle}_mass")
        variables2plot.append(f"{particle}_pt")
        variables2plot.append(f"{particle}_eta")
        if args.minimum_set:
            continue
        variables2plot.append(f"{particle}_phi")
        variables2plot.append(f"{particle}_cos_theta_cs")
        variables2plot.append(f"{particle}_phi_cs")
        variables2plot.append(f"{particle}_cos_theta_eta")
        variables2plot.append(f"{particle}_phi_eta")
        variables2plot.append(f"mmj_min_dPhi_nominal")
        variables2plot.append(f"mmj_min_dEta_nominal")
        variables2plot.append(f"ll_zstar_log_nominal")
        variables2plot.append(f"dimuon_ebe_mass_res")
        variables2plot.append(f"dimuon_ebe_mass_res_rel")
        variables2plot.append(f"{particle}_rapidity")
        variables2plot.append("MET_pt")
        variables2plot.append("MET_phi")
        variables2plot.append("MET_sumEt")
        variables2plot.append("acoplanarity")
        variables2plot.append("PV_npvs")
        variables2plot.append("PV_npvsGood")
    if "dijet" in particle:
        variables2plot.append(f"jj_dEta_nominal")
        variables2plot.append(f"jj_mass_nominal")
        variables2plot.append(f"jj_pt_nominal")
        variables2plot.append(f"jj_dPhi_nominal")
        variables2plot.append(f"zeppenfeld_nominal")
        variables2plot.append(f"rpt_nominal")
        variables2plot.append(f"pt_centrality_nominal")
        variables2plot.append(f"nsoftjets2_nominal")
        variables2plot.append(f"htsoft2_nominal")
        variables2plot.append(f"nsoftjets5_nominal")
        variables2plot.append(f"htsoft5_nominal")
    if ("mu" in particle):
        for kinematic in kinematic_vars:
            variables2plot.append(f"{particle}1_{kinematic}")
            variables2plot.append(f"{particle}2_{kinematic}")
        if not args.minimum_set:
            variables2plot.append(f"{particle}1_pt_over_mass")
            variables2plot.append(f"{particle}2_pt_over_mass")
    if ("jet" in particle):
        variables2plot.append(f"njets_nominal")
        for kinematic in kinematic_vars:
            variables2plot.append(f"{particle}1_{kinematic}_nominal")
            variables2plot.append(f"{particle}2_{kinematic}_nominal")
        variables2plot.append(f"jet1_qgl_nominal")
        variables2plot.append(f"jet2_qgl_nominal")

print(f"Variables to plot: {variables2plot}")

# Build variables_to_plot from JSON info
variables_to_plot = []
for v in variables2plot:
    conf = plot_vars_config.get(v)
    if conf and "binning_linspace" in conf:
        start, stop, nbins = conf["binning_linspace"]
        variables_to_plot.append({
            "name": v,
            "bins": np.linspace(start, stop, nbins),
            "xlabel": conf.get("xlabel", v)
        })
    else:
        print(f"Skipping {v} (not found in JSON or missing binning).")

print(f"Total variables to plot: {len(variables_to_plot)}")

def apply_vbf_h_sidebands(df):
    m = df.dimuon_mass
    sideband = ((m > 110) & (m < 115.03)) | ((m > 135.03) & (m < 150))

    btag_cut = (df.nBtagLoose_nominal >= 2) | (df.nBtagMedium_nominal >= 1)

    vbf = (df.jj_mass_nominal > 400) & (df.jj_dEta_nominal > 2.5) & (df.jet1_pt_nominal > 35)

    return df[sideband & vbf & ~btag_cut]

def apply_mjj_stitching(df_mc, grp):
    """
    # To stich the two DY samples together we will use the gjj_mass
    # variable which is di-jet invariant mass at generator level
    # with the DYVBF sample use gjj_mass > 350 GeV
    # and with the other DY sample use gjj_mass <= 350 GeV
    """
    if grp == "DY":
        df_mc = df_mc.assign(
            gjj_mass=df_mc.gjj_mass.where(df_mc.gjj_mass > 350, np.nan)
        )
    elif grp == "DYVBF":
        df_mc = df_mc.assign(
            gjj_mass=df_mc.gjj_mass.where(df_mc.gjj_mass <= 350, np.nan)
        )
    else:
        print(f"WARNING: apply_mjj_stitching called with unexpected group {grp}. No stitching applied.")
    return df_mc

def ensure_compacted(year, sample):
    compacted_dir = COMPACTED_PATH.format(year=year, sample=sample)
    if not os.path.exists(compacted_dir):
        print(f"Compacted path for {year}/{sample} not found. Creating compacted dataset...")
        # Read original data
        orig_path = LOAD_PATH.format(year=year, sample=sample)
        df = dd.read_parquet(orig_path)
        print(f"{sample}: Original rows = {len(df)}")
        if len(df) == 0:
            print(f"WARNING: {orig_path} is empty!")
        # Repartition and write to compacted path
        df_repart = df.repartition(partition_size="200MB")
        df_repart.to_parquet(compacted_dir)
        print(f"Compacted dataset created at {compacted_dir}")
    else:
        # Optionally, could check if compacted_dir is empty or incomplete, but skipping for now
        pass


with performance_report(filename="dask_profile_report.html"):
    # 1) Data: no weights
    print("Processing Data...")
    # Ensure compacted datasets exist for data samples
    for yr in years:
        for s in group_dict["DATA"]:
            ensure_compacted(yr, s)
    valid_data_dirs = []
    for s in group_dict["DATA"]:
        compacted_dir = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/compacted/2018/{s}/"
        if os.path.exists(compacted_dir):
            parquet_files = [f for f in os.listdir(compacted_dir) if f.endswith(".parquet")]
            if parquet_files:
                valid_data_dirs.append(compacted_dir)
            else:
                print(f"WARNING: {compacted_dir} contains no parquet files.")
        else:
            print(f"WARNING: {compacted_dir} does not exist.")

    print("Using data dirs:", valid_data_dirs)
    all_parquet_files = []
    for d in valid_data_dirs:
        all_parquet_files.extend(glob.glob(os.path.join(d, "*.parquet")))

    print("Total parquet files found:", len(all_parquet_files))
    df_data = dd.read_parquet(all_parquet_files)
    print("Columns found:", df_data.columns)

    df_data = apply_vbf_h_sidebands(df_data)

    # Build all histogram compute tasks in a dictionary
    hist_tasks = {}

    for var in variables_to_plot:
        # Data histogram task (unweighted)
        arr_data = df_data[var["name"]].to_dask_array(lengths=True)
        hist_tasks[f"data_{var['name']}"] = da.histogram(arr_data, bins=var["bins"])

    print("Processing MC groups and building histogram tasks...")
    for grp in bkg_MC_order:
        print(f"Processing MC group: {grp}")
        samples = group_dict.get(grp, [])
        if not samples:
            continue

        # Ensure compacted datasets exist for MC samples
        for yr in years:
            for s in samples:
                ensure_compacted(yr, s)

        mc_globs = [f"/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/compacted/2018/{s}/" for s in samples]
        all_parquet_files = []
        for d in mc_globs:
            all_parquet_files.extend(glob.glob(os.path.join(d, "*.parquet")))

        print("Total parquet files found:", len(all_parquet_files))
        df_mc = dd.read_parquet(all_parquet_files)
        df_mc = apply_vbf_h_sidebands(df_mc)
        if args.ifStichTwoDYs: df_mc = apply_mjj_stitching(df_mc, grp)

        for var in variables_to_plot:
            arr_mass = df_mc[var["name"]].to_dask_array(lengths=True)
            arr_wt   = df_mc["wgt_nominal"].to_dask_array(lengths=True)
            arr_wt = arr_wt.rechunk(arr_mass.chunks)

            hist_tasks[f"{grp}_{var['name']}"] = da.histogram(
                arr_mass,
                bins=var["bins"],
                weights=arr_wt
            )

    # Compute all histograms in parallel
    print("Computing all histograms in parallel...")
    computed = client.compute(list(hist_tasks.values()))
    computed = client.gather(computed)
    results = dict(zip(hist_tasks.keys(), computed))

    # 3) Plot
    print("Plotting...")
    import ROOT

    # ─── build ROOT TH1s from your numpy histograms ────────────────────────────────

    for var in variables_to_plot:
        nbins = len(var["bins"]) - 1
        xlow  = var["bins"][0]
        xhigh = var["bins"][-1]

        # 1) data histogram
        hist_data = results[f"data_{var['name']}"][0]
        h_data = ROOT.TH1F(f"h_data_{var['name']}","Data / MC in VBF H-sidebands", nbins, xlow, xhigh)
        h_data.Sumw2()
        legends = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)
        for i, c in enumerate(hist_data):
            h_data.SetBinContent(i+1, c)
            h_data.SetBinError(  i+1, np.sqrt(c) )  # if you want Poisson errors
        legends.AddEntry(h_data, "Data", "lep")

        # 2) sum all MC into one total
        h_mc_tot = ROOT.TH1F(f"h_mc_tot_{var['name']}","MC total", nbins, xlow, xhigh)
        h_mc_tot.Sumw2()
        # for the stacked display we'll also keep each component
        stack = ROOT.THStack(f"stack_{var['name']}","")

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
            vals = results.get(f"{grp}_{var['name']}")
            if vals is None:
                continue
            hist_vals = vals[0]
            h = ROOT.TH1F(f"h_{grp}_{var['name']}", grp, nbins, xlow, xhigh)
            h.Sumw2()
            for i, c in enumerate(hist_vals):
                h.SetBinContent(i+1, c)
            col = root_colors[idx % len(root_colors)]
            h.SetFillColor(col)
            h.SetLineColor(col)
            h_mc_tot.Add(h)
            stack.Add(h)
            legends.AddEntry(h, grp, "f")

        # ─── now make a TRatioPlot ────────────────────────────────────────────────────

        CMS.SetExtraText("Simulation Preliminary")
        CMS.SetLumi("")
        canv = CMS.cmsCanvas('', 0, 1, 0, 1, '', '', square = CMS.kSquare, extraSpace=0.01, iPos=0)
        canv.Divide(1,2)
        pad1 = canv.cd(1)
        pad1.SetPad(0,0.3,1,1)
        pad1.SetBottomMargin(0.02)
        pad1.SetLogy()
        pad2 = canv.cd(2)
        pad2.SetPad(0,0,1,0.3)
        pad2.SetTopMargin(0.05)
        pad2.SetBottomMargin(0.35)

        # ----- Upper Pad: Stack + Data
        pad1.cd()
        stack.SetMaximum(1e8)
        stack.SetMinimum(1e-1)
        stack.Draw("HIST")
        h_data.SetMarkerStyle(20)
        h_data.SetMarkerSize(0.9)
        h_data.Draw("E SAME")
        legends.Draw("same")

        # ----- Lower Pad: Ratio plot
        pad2.cd()
        h_ratio = h_data.Clone("h_ratio")
        h_ratio.Divide(h_mc_tot)
        h_ratio.SetMarkerStyle(20)
        h_ratio.SetMarkerSize(0.8)
        h_ratio.SetTitle("")
        h_ratio.GetYaxis().SetTitle("Data/MC")
        h_ratio.GetYaxis().SetTitleSize(0.12)
        h_ratio.GetYaxis().SetTitleOffset(0.4)
        h_ratio.GetYaxis().SetLabelSize(0.10)
        h_ratio.GetYaxis().SetNdivisions(4)
        h_ratio.GetYaxis().SetRangeUser(0.5,1.5)
        h_ratio.GetXaxis().SetTitle(var["xlabel"])
        h_ratio.GetXaxis().SetTitleSize(0.12)
        h_ratio.GetXaxis().SetLabelSize(0.10)
        h_ratio.GetXaxis().SetTitleOffset(1.0)
        h_ratio.Draw("E")

        # Draw a horizontal line at ratio=1
        line = ROOT.TLine(h_ratio.GetXaxis().GetXmin(),1.0,h_ratio.GetXaxis().GetXmax(),1.0)
        line.SetLineStyle(2)
        line.SetLineColor(ROOT.kBlack)
        line.Draw()

        canv.SaveAs(f"{plotpath}/{var['name']}.root")
        canv.SaveAs(f"{plotpath}/{var['name']}.pdf")

# Close the Dask client
client.close()
