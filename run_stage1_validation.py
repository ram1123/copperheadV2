import hist
import awkward as ak
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import mplhep
# mplhep.style.use("CMS")
plt.style.use(mplhep.style.CMS)
import glob
import os

data_samples = [
    # "data_A",
    # "data_B",
    # "data_C",
    # "data_D",
]
bkg_samples = [
    "dy_M-50",
    "dy_M-100To200",
    # "ttjets_dl",
    # "ttjets_sl",
    # "st_tw_top",
    # "st_tw_antitop", 
    # "ww_2l2nu",
    # "wz_3lnu",
    # "wz_2l2q",
    # "wz_1l1nu2q",
    # "zz",
    # "ewk_lljj_mll50_mjj120",
]
sig_samples = [
    "ggh_powheg",
    # "vbf_powheg"
]
mc_samples = [
    # *bkg_samples,
    # *sig_samples
]
variables = [
    # 'mu1_pt',
    # 'mu2_pt',
    # 'mu1_eta',
    # 'mu2_eta',
    # 'mu1_phi',
    # 'mu2_phi',
    # # 'mu1_iso',
    # # 'mu2_iso',
    'mu1_pt_over_mass',
    'mu2_pt_over_mass',
    # "dimuon_mass",
    # "dimuon_ebe_mass_res",
    # "dimuon_ebe_mass_res_rel",
    "dimuon_pt",
    # "dimuon_pt_log",
    # "dimuon_eta",
    # "dimuon_phi",
    # "dimuon_dEta",
    # "dimuon_dPhi",
    # "dimuon_dR",
    "dimuon_cos_theta_cs",
    "dimuon_phi_cs",
    # "jet1_pt",
    # "jet1_eta",
    # "jet1_rap",
    # "jet1_phi",
    # "jet1_qgl",
    # "jet1_jetId",
    # "jet1_puId",
    # "jet2_pt",
    # "jet2_eta",
    # "jet2_rap",
    # "jet2_phi",
    # "jet2_qgl",
    # "jet2_jetId",
    # "jet2_puId",
    # "jj_mass",
    # "jj_mass_log",
    # "jj_pt",
    # "jj_eta",
    # "jj_phi",
    # "jj_dEta",
    # "jj_dPhi",
    # "mmj1_dEta",
    # "mmj1_dPhi",
    # "mmj1_dR",
    # "mmj2_dEta",
    # "mmj2_dPhi",
    # "mmj2_dR",
    # "mmj_min_dEta",
    # "mmj_min_dPhi",
    # "mmjj_pt",
    # "mmjj_eta",
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
range_dict = { # order of the variables matter since many share the same word(s)
    "cos_theta" : [-1,1],
    "rpt" : [0, 1],
    "mass_res_rel" : [0,0.1],
    "mass_res" : [0,8],
    "mass_log" : [0,10],
    "mass" : [0,250],
    "pt_over_mass" : [0, 2],
    "pt_log" : [-5,10],
    "pt" : [20,250],
    "eta" : [-2.5, 2.5],
    "phi_cs" : [-np.pi,np.pi],
    "phi" : [-np.pi, np.pi],
    "rap" : [-np.pi, np.pi], # rapidity
    "dEta" : [0, 1.2*np.pi],
    "dPhi" : [0, np.pi],
    "dR" : [0, 2*np.pi],
    
    "puId" : [0, 8],
    "puId" : [0, 10],
    "zeppenfeld" : [-10, 10],
    "njets" : [0, 6],
    "qgl" : [-2, 2],
    "default" : [-100,500]
}
def get_variable(
    histogram, 
    variable: str,
    file_list: list,
    samples: list,
    region: str,
    sample_type: str,
    ):
    # print(f"mc_samples: {mc_samples}")
    # if len(samples)==0:
    #     print(f"empty samples for {sample_type}")
    #     return histogram
    # loop over data samples
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
        print(f"sample_type: {sample_type}")
        # print(f"df[variable][region_filter]: {df[variable][region_filter]}")
        # print(f"df[variable]: {df[variable]}")
        #fill histogram
        if sample_type =="Data":
            weight = 1/df["fraction"]
        else:
            weight = df["weight_nominal"] / df["weight_nominal"].sum()
        histogram.fill(dataset=sample_type,
            var=df[variable],
           weight = weight,
        )
    return histogram

if __name__ == "__main__":
    # dataset_fraction = 0.01
    # fraction_str = str(dataset_fraction).replace('.', '_')
    # load_path = load_path + f"/f{fraction_str}"
    print(f"load_path: {load_path}")
    file_list = glob.glob(load_path+"/*.csv")
    # print(f"file_list: {file_list}")
    # print(f"range_dict.keys(): {range_dict.keys()}")
    # loop over variables
    for region in regions:
        for var in variables:
            range = []
            for keyword in range_dict.keys():
                # print(f"keyword: {keyword}")
                if keyword in var:
                    range = range_dict[keyword]
                    break
            if len(range) == 0:
                range = range_dict["default"]
            min,max = range[0],range[1]
            print(f"range max: {max}")
            dists = (
                hist.Hist.new
                .StrCat(["Data", "MC", "Bkg", "Sig"], name="dataset")
                # .StrCat(["Data", "MC"], name="dataset")
                # .StrCat(["Bkg", "Sig"], name="dataset")
                .Reg(80, min, max, name="var")
                .Weight()
            )
            print(f"plotting: {var}")
            dists = get_variable(dists, var, file_list, data_samples, region, "Data")
            dists = get_variable(dists, var, file_list, mc_samples, region, "MC")
            dists = get_variable(dists, var, file_list, bkg_samples, region, "Bkg")
            dists = get_variable(dists, var, file_list, sig_samples, region, "Sig")
            
    
            fig, ax = plt.subplots()
            dists.plot1d(ax=ax)
            ax.legend(title=f"{var} validation")
            plt.xlabel(var)
            # save figure
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(save_path+f"/V{var}_{region}.png")
    