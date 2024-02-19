import hist
import awkward as ak
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import glob
import os
import mplhep as hep
from histogram.variable import variables_lookup, Entry
from hist.intervals import poisson_interval

# print(f"variables_lookup: {variables_lookup}")
style = hep.style.CMS
style["mathtext.fontset"] = "cm"
style["mathtext.default"] = "rm"
plt.style.use(style)

stat_err_opts = {
    "step": "post",
    "label": "Stat. unc.",
    "hatch": "//////",
    "facecolor": "none",
    "edgecolor": (0, 0, 0, 0.5),
    "linewidth": 0,
}
ratio_err_opts = {"step": "post", "facecolor": (0, 0, 0, 0.3), "linewidth": 0}


# data_samples = [
#     "data_A",
#     "data_B",
#     "data_C",
#     "data_D",
# ]
# bkg_samples = [
#     "dy_M-50",
#     "dy_M-100To200",
#     "ttjets_dl",
#     "ttjets_sl",
#     # "st_tw_top",
#     # "st_tw_antitop", 
#     # "ww_2l2nu",
#     # "wz_3lnu",
#     # "wz_2l2q",
#     # "wz_1l1nu2q",
#     # "zz",
#     # "ewk_lljj_mll50_mjj120",
# ]
# sig_samples = [
#     "ggh_powheg",
#     "vbf_powheg"
# ]
# mc_samples = [
#     *bkg_samples,
#     *sig_samples
# ]

variables = [
    # 'mu1_pt',
    # 'mu2_pt',
    # 'mu1_eta',
    # 'mu2_eta',
    # # 'mu1_phi',
    # # 'mu2_phi',
    # # # 'mu1_iso',
    # # # 'mu2_iso',
    # 'mu1_pt_over_mass',
    # 'mu2_pt_over_mass',
    "dimuon_mass",
    # "dimuon_ebe_mass_res",
    # "dimuon_ebe_mass_res_rel",
    # "dimuon_pt",
    # "dimuon_pt_log",
    # "dimuon_eta",
    # "dimuon_phi",
    # "dimuon_dEta",
    # "dimuon_dPhi",
    # "dimuon_dR",
    # "dimuon_cos_theta_cs",
    # "dimuon_phi_cs",
    "jet1_pt",
    # "jet1_eta",
    # "jet1_rap",
    # "jet1_phi",
    # "jet1_qgl",
    # "jet1_jetId",
    # "jet1_puId",
    "jet2_pt",
    # "jet2_eta",
    # "jet2_rap",
    # "jet2_phi",
    # "jet2_qgl",
    # "jet2_jetId",
    # # "jet2_puId",
    # "jj_mass",
    # # "jj_mass_log",
    # # "jj_pt",
    # # "jj_eta",
    # # "jj_phi",
    # "jj_dEta",
    # "jj_dPhi",
    # # "mmj1_dEta",
    # # "mmj1_dPhi",
    # # "mmj1_dR",
    # # "mmj2_dEta",
    # # "mmj2_dPhi",
    # # "mmj2_dR",
    # "mmj_min_dEta",
    # "mmj_min_dPhi",
    # # "mmjj_pt",
    # # "mmjj_eta",
    # "mmjj_phi",
    # "mmjj_mass",
    # "rpt",
    # "zeppenfeld",
    # "njets",
]
regions = [
    # "z_peak",
    # "h_sidebands",
    # "h_peak",
    "signal_region",
]
load_path = "/depot/cms/users/yun79/results/stage1/test"
save_path = "/depot/cms/users/yun79/valerie/fork/copperheadV2/validation/figs"

# channels = ["vbf", "ggh", "DY", "TT+ST", "Data"]
channels =  ["Data", "MC", "Bkg", "Sig"]
# plot_group_config= {
#     # 'stack': ['DY', 'EWK', 'TT+ST', 'VV', 'VVV'],
#     # 'step': ['VBF', 'ggH'], 
#     # 'errorbar': ['Data']
#     'stack': ['Bkg'],
#     'step': ['Sig'], 
#     'errorbar': ['Data']
# }

parameters = {}
parameters["grouping"] = {
    "data_A": "Data",
    "data_B": "Data",
    "data_C": "Data",
    "data_D": "Data",
    "data_E": "Data",
    "data_F": "Data",
    "data_G": "Data",
    "data_H": "Data",
    "dy_M-50": "DY",
    "dy_M-100To200": "DY",
    "ttjets_dl": "TT+ST",
    "ttjets_sl": "TT+ST",
    "ggh_powheg": "ggH",
    "vbf_powheg": "VBF",
}
parameters["plot_group"] = {
    "stack": ["DY", "EWK", "TT+ST", "VV", "VVV"],
    "step": ["VBF", "ggH"],
    "errorbar": ["Data"],
}
# parameters["plot_group"] = {
#     "Data" : "errorbar",
#     "ggH" : "step",
#     "VBF" : "step",
#     "DY" : "stack",
#     "TT+ST" : "stack",
# }

def get_plottable(
    variable: str,
    file_list: list,
    entry,
    region: str,
    # grouping: str,
    ):
    # print(f"mc_samples: {mc_samples}")
    # if len(samples)==0:
    #     print(f"empty samples for {channel}")
    #     return histogram
    # loop over data samples
    
    h_var = variables_lookup[variable]
    h_dict = {}
    for group in entry.groups:
        samples = [e for e, g in entry.entry_dict.items() if (group == g)]
        print(f"samples: {samples}")
        histogram = (
            hist.Hist.new
            .Reg(h_var.nbins, h_var.xmin, h_var.xmax, name=h_var.name, label=h_var.caption)
            .Double()
        )
        histogram_w2 = ( # weight sq histogram for stat err calculation
            hist.Hist.new
            .Reg(h_var.nbins, h_var.xmin, h_var.xmax, name=h_var.name, label=h_var.caption)
            .Double()
        )
        for sample in samples:
            df = None
            for file in file_list:
                if sample in file:
                    print(f"{sample} matched with {file}")
                    df = pd.read_csv(file)
                    break
            if df is None:
                print(f"{sample} not in stage1 results")
                continue
            if var not in df.columns:
                print(f"{variable} not found in df columns")
                continue
            # get region array
            if region == "z_peak":
                region_filter = df["z_peak"]
            elif region == "h_sidebands":
                region_filter = df["h_sidebands"]
            elif region == "h_peak":
                region_filter = df["h_peak"]
            else:
                region_filter = df["z_peak"] | df["h_sidebands"] | df["h_peak"]
    
            # refine df
            df = df[region_filter]
            # print(f'df["fraction"][0]: {df["fraction"][0]}')
            # print(f"df[variable][region_filter]: {df[variable][region_filter]}")
            # print(f"df[variable]: {df[variable]}")
            #fill histogram
            if "Data" in entry.groups: # data
                # weight = 1/df["fraction"]
                weight=1
            else: # MC
                weight = df["weight_nominal"]
                # weight = df["weight_nominal"] / df["weight_nominal"].sum()
            
            to_fill = {
                h_var.name: df[variable],
                # "grouping": grouping,
            }
            histogram.fill(**to_fill, weight=weight)
            histogram_w2.fill(**to_fill, weight=weight*weight)
        print(f"histogram.sum(): {histogram.sum()}")
        # h_list.append(histogram)
        # need to sort by histogram sum later
        h_dict[histogram.sum()] = (histogram, histogram_w2, group) 
    
    print(f"h_dict b4: {h_dict}")
    h_dict  = dict(sorted(h_dict.items()))
    print(f"h_dict after: {h_dict}")
    h_list = []
    h_w2_list = []
    labels = []
    for histogram, histogram_w2, group in h_dict.values():
        h_list.append(histogram)
        h_w2_list.append(histogram_w2)
        labels.append(group)
    # return h_list
    return h_list, h_w2_list, labels

if __name__ == "__main__":
    # dataset_fraction = 0.01
    # fraction_str = str(dataset_fraction).replace('.', '_')
    # load_path = load_path + f"/f{fraction_str}"
    print(f"load_path: {load_path}")
    file_list = glob.glob(load_path+"/*.csv")
    # print(f"file_list: {file_list}")
    # loop over variables

    plotsize = 8
    ratio_plot_size = 0.25
    integrated_lumi = 59970.0 /1000 # get this from config in the future
    fontsize=20
    year = "2018"
    
    # # temporary
    # variation = "nominal"

    # slicer = {"region": region, "channel": channel, "variation": variation, "category": category}
    

    # label: ['TT+ST', 'DY']
    # entry.stack: True
    # entry.histtype: fill
    # entry.plot_opts: {'alpha': 0.8, 'edgecolor': (0, 0, 0)}
    
    for region in regions:
        for var in variables:
            print(f"plotting: {var}")
            fig = plt.figure()
            plot_ratio = False
            if plot_ratio:
                fig.set_size_inches(plotsize * 1.2, plotsize * (1 + ratio_plot_size))
                gs = fig.add_gridspec(
                    2, 1, height_ratios=[(1 - ratio_plot_size), ratio_plot_size], hspace=0.07
                )
                # Top panel: Data/MC
                ax1 = fig.add_subplot(gs[0])
            else:
                fig, ax1 = plt.subplots()
                fig.set_size_inches(plotsize, plotsize)
                
            
            if plot_ratio:
                ax1.set_xlabel("")
                ax1.tick_params(axis="x", labelbottom=False)
            else:
                ax1.set_xlabel(variables_lookup[var].caption, loc="right")

            entries = {entry_type: Entry(entry_type, parameters) for entry_type in parameters["plot_group"].keys()}
            for entry in entries.values():
                print(f"entry.histtype: {entry.histtype}")
                print(f"entry.stack: {entry.stack}")
                print(f"entry.labels: {entry.labels}")
                print(f"entry.groups: {entry.groups}")
                # print(f"entry.entry_dict: {entry.entry_dict}")
                # print(f"entry.entry_list: {entry.entry_list}")
                # grouping = parameters["grouping"][data_samples[0]]
                dists, dists_w2, labels = get_plottable(var, file_list, entry, region)
                # print(f"dists: {dists}")
                hep.histplot(
                    dists,
                    # label=entry.groups,
                    label=labels,
                    ax=ax1,
                    # yerr=yerr,
                    stack=entry.stack,
                    histtype=entry.histtype,
                    **entry.plot_opts,
                )
                # Bkg MC errors
                if entry.entry_type == "stack":    
                    total_bkg = sum(dists).values()
                    total_sumw2 = sum(dists_w2).values()
                    if sum(total_bkg) > 0:
                        err = poisson_interval(total_bkg, total_sumw2)
                        ax1.fill_between(
                            x=dists[0].axes[0].edges,
                            y1=np.r_[err[0, :], err[0, -1]],
                            y2=np.r_[err[1, :], err[1, -1]],
                            **stat_err_opts,
                        )
            # # Bottom panel: Data/MC ratio plot
            # if plot_ratio:
                
            #---------------------------------------------------
            # for samples in [data_samples, bkg_samples, sig_samples]:
            #     grouping = parameters["grouping"][sample[0]]
            #     dists = get_plottable(var, file_list, samples, region, grouping)
            #     # print(f"dists: {dists}")
            #     hep.histplot(
            #         dists,
            #         label=[grouping],
            #         ax=ax1,
            #         # yerr=yerr,
            #         stack=entry.stack,
            #         histtype=entry.histtype,
            #         **entry.plot_opts,
            #     )
            #--------------------------------------------------
            # histtype= "errorbar"
            # hep.histplot(
            #     dists,
            #     label=["Data"],
            #     ax=ax1,
            #     # yerr=yerr,
            #     stack=True,
            #     histtype=histtype,
            #     **entries[histtype].plot_opts,
            # )
            # # dists = get_plottable(dists, var, file_list, mc_samples, region, "MC")
            # # hep.histplot(
            # #     dists,
            # #     label="MC",
            # #     ax=ax1,
            # #     # yerr=yerr,
            # #     stack=True,
            # #     histtype="errorbar",
            # #     # **entry.plot_opts,
            # # )
            # dists = get_plottable(var, file_list, bkg_samples, region, "Bkg")
            # hep.histplot(
            #     dists,
            #     label=["Bkg"],
            #     ax=ax1,
            #     # yerr=yerr,
            #     stack=True,
            #     histtype="fill",
            #     # **entry.plot_opts,
            # )
            # dists = get_plottable(var, file_list, sig_samples, region, "Sig")
            # hep.histplot(
            #     dists,
            #     label=["Sig"],
            #     ax=ax1,
            #     # yerr=yerr,
            #     stack=True,
            #     histtype="step",
            #     # **entry.plot_opts,
            # )
            
    
            
            hep.cms.label(ax=ax1, data=True, label="Work in progress", year=year, lumi=integrated_lumi, fontsize=fontsize)
            ax1.set_yscale("log")
            ax1.set_ylim(0.001, 1e9)
            ax1.legend(prop={"size": "x-small"})
            # save figure
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig.savefig(save_path+f"/V{var}_{region}.png")
    