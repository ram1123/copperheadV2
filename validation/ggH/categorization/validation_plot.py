import awkward as ak
import dask_awkward as dak
import argparse
import sys
import os
import numpy as np
import json
from collections import OrderedDict


# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# Add it to sys.path
sys.path.insert(0, parent_dir)

# Now you can import your module
from src.lib.histogram.plotting import plotDataMC_compare

def filterRegion(events, region="h-peak"):
    dimuon_mass = events.dimuon_mass
    if region =="h-peak":
        region = (dimuon_mass > 115.03) & (dimuon_mass < 135.03)
    elif region =="h-sidebands":
        region = ((dimuon_mass > 110) & (dimuon_mass < 115.03)) | ((dimuon_mass > 135.03) & (dimuon_mass < 150))
    elif region =="signal":
        region = (dimuon_mass >= 110) & (dimuon_mass <= 150.0)
    elif region =="z-peak":
        region = (dimuon_mass >= 70) & (dimuon_mass <= 110.0)

    events = events[region]
    return events

def fillSampleValues(events, sample_dict, sample_groups, sample: str):
    sample_name = sample.lower()
    # find which sample group sample_name belongs to
    sample_group = next((key for key, values in sample_groups.items() if sample_name in values), None)
    print(f"sample_group: {sample_group}")
    if sample_group in sample_dict.keys():
        sample_info = sample_dict[sample_group]
        fields2load = sample_info.keys() # dimuon_mass, wgt_nominal
        
        # compute in parallel fields to load
        computed_zip = ak.zip({
            field : events[field] for field in fields2load
        }).compute()
        
        # add the computed fields to sample_dict 
        for field in fields2load:
            sample_dict[sample_group][field].append(
                ak.to_numpy(computed_zip[field])
            )
    else:
        print(f"sample {sample_group} not present in sample_dict!")

    return sample_dict
        
    # if sample.lower() == "data":
    #     full_load_path = load_path+f"*data.parquet" 
    # elif sample.lower() == "ggh":
    #     full_load_path = load_path+f"*sigMC_ggh.parquet" 
    # elif sample.lower() == "vbf":
    #     full_load_path = load_path+f"*sigMC_vbf.parquet" 
    # elif sample.lower() == "dy":
    #     full_load_path = load_path+f"*bkgMC_dy.parquet" 
    # elif sample.lower() == "ewk":
    #     full_load_path = load_path+f"*bkgMC_ewk.parquet" 
    # elif sample.lower() == "tt":
    #     full_load_path = load_path+f"*bkgMC_tt.parquet" 
    # elif sample.lower() == "st":
    #     full_load_path = load_path+f"*bkgMC_st.parquet" 
    # elif sample.lower() == "ww":
    #     full_load_path = load_path+f"*bkgMC_ww.parquet" 
    # elif sample.lower() == "wz":
    #     full_load_path = load_path+f"*bkgMC_wz.parquet" 
    # elif sample.lower() == "zz":
    #     full_load_path = load_path+f"*bkgMC_zz.parquet" 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-label",
    "--label",
    dest="label",
    default="",
    action="store",
    help="label",
    )
    parser.add_argument(
    "-cat",
    "--category",
    dest="category",
    default="ggH",
    action="store",
    help="string value production category we're working on",
    )
    parser.add_argument(
    "-save",
    "--save_path",
    dest="save_path",
    default="plots",
    action="store",
    help="string value production category we're working on",
    )
    parser.add_argument(
    "-samp",
    "--samples",
    dest="samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of samples to process for stage2. Current valid inputs are data, signal and DY",
    )
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="all",
    action="store",
    help="label",
    )
    parser.add_argument(
    "-reg",
    "--region",
    dest="region",
    default="signal",
    action="store",
    help="region value to plot, available regions are: h_peak, h_sidebands, z_peak and signal (h_peak OR h_sidebands)",
    )
    args = parser.parse_args()
    if len(args.samples) == 0:
        print("samples list is zero!")
        raise ValueError
    # load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean/{args.label}/{args.category}/stage2_output/*/"
    year = args.year
    if year == "all":
        year = "*"
    load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean/{args.label}/{args.category}/stage2_output/{year}/"
    # events = dak.from_parquet(f"{load_path}/*data.parquet")
    # print(events.fields)
    print(f"load_path : {load_path}")
    print(f"args.samples: {args.samples}")

    lumi_dict = {
        "2018" : 59.97,
        "2017" : 41.5,
        "2016postVFP": 19.5,
        "2016preVFP": 16.8,
        "all" : 137,
    }
    lumi_val = lumi_dict[year]

    possible_samples = ["data", "ggh", "vbf", "dy", "ewk", "tt", "st", "ww", "wz", "zz",]
    sample_groups = {
        "data": ["data"],
        "ggh": ["ggh"],
        "vbf": ["vbf"],
        "dy": ["dy"],
        "top": ["tt", "st"],
        "ewk": ["ewk"],
        "diboson": ["ww", "wz", "zz"]
    }
    
    
    sub_cats = ["all", 0,1,2,3,4]
    # sub_cats = range(5)
    plot_vars = ["BDT_score", "dimuon_mass"]

    for plot_var in plot_vars:
        for sub_cat in sub_cats:
            # initialize empty dictionaries that will contain the values
            sample_dict = {
                sample_name: {
                    "dimuon_mass": [],
                    "wgt_nominal" : [],
                    "BDT_score": [],
                } for sample_name in sample_groups.keys()
            }
            # data_dict = {}
            # bkg_MC_dict = {}
            for sample in args.samples:
                if sample.lower() == "data":
                    full_load_path = load_path+f"*data.parquet" 
                    # full_load_path = glob.glob(full_load_path)
                elif sample.lower() == "ggh":
                    full_load_path = load_path+f"*sigMC_ggh.parquet" 
                # elif sample.lower() == "ggh_amcps":
                    # full_load_path = load_path+f"/ggh_amcPS/*/*.parquet"
                elif sample.lower() == "vbf":
                    full_load_path = load_path+f"*sigMC_vbf.parquet" 
                elif sample.lower() == "dy":
                    full_load_path = load_path+f"*bkgMC_dy.parquet" 
                elif sample.lower() == "ewk":
                    full_load_path = load_path+f"*bkgMC_ewk.parquet" 
                elif sample.lower() == "tt":
                    full_load_path = load_path+f"*bkgMC_tt.parquet" 
                elif sample.lower() == "st":
                    full_load_path = load_path+f"*bkgMC_st.parquet" 
                elif sample.lower() == "ww":
                    full_load_path = load_path+f"*bkgMC_ww.parquet" 
                elif sample.lower() == "wz":
                    full_load_path = load_path+f"*bkgMC_wz.parquet" 
                elif sample.lower() == "zz":
                    full_load_path = load_path+f"*bkgMC_zz.parquet" 
                else:
                    print(f"unsupported sample!")
                    raise ValueError
                print(f"full_load_path: {full_load_path}")
    
                # -----------------------------------------------
                # Load events and filter to subcat
                # -----------------------------------------------
                
                events = dak.from_parquet(full_load_path)
                events = filterRegion(events, region=args.region)
                if sub_cat != "all":
                    events = events[events.subCategory_idx == sub_cat] # filter subcat
                print(f"events field from {sample}:", events.fields)
                sample_dict = fillSampleValues(events, sample_dict, sample_groups, sample)
                
    
            # ----------------------------------
            # begin plotting
            # ----------------------------------
            # plot_var = "dimuon_mass"
            # plot_var = "BDT_score"
            # define data dict
            data_dict = {
                "values" :np.concatenate(sample_dict["data"][plot_var], axis=0),
                "weights":np.concatenate(sample_dict["data"]["wgt_nominal"], axis=0)
            }
            
            # define Bkg MC dict
            bkg_MC_dict = OrderedDict()
            # start from lowest yield to highest yield
            if len(sample_dict["diboson"]["wgt_nominal"]) > 0:
                group_name = "diboson"
                bkg_MC_dict["VV"] = {
                    "values" :np.concatenate(sample_dict[group_name][plot_var], axis=0),
                    "weights":np.concatenate(sample_dict[group_name]["wgt_nominal"], axis=0)
                }
            if len(sample_dict["ewk"]["wgt_nominal"]) > 0:
                group_name = "ewk"
                bkg_MC_dict["Ewk"] = {
                    "values" :np.concatenate(sample_dict[group_name][plot_var], axis=0),
                    "weights":np.concatenate(sample_dict[group_name]["wgt_nominal"], axis=0)
                }
            if len(sample_dict["top"]["wgt_nominal"]) > 0:
                group_name = "top"
                bkg_MC_dict["Top"] = {
                    "values" :np.concatenate(sample_dict[group_name][plot_var], axis=0),
                    "weights":np.concatenate(sample_dict[group_name]["wgt_nominal"], axis=0)
                }
            if len(sample_dict["dy"]["wgt_nominal"]) > 0:
                group_name = "dy"
                bkg_MC_dict["DY"] = {
                    "values" :np.concatenate(sample_dict[group_name][plot_var], axis=0),
                    "weights":np.concatenate(sample_dict[group_name]["wgt_nominal"], axis=0)
                }
        
            
        
            # define Sig MC dict
            sig_MC_dict = OrderedDict()
            # start from lowest yield to highest yield
            if len(sample_dict["vbf"]["wgt_nominal"]) > 0:
                group_name = "vbf"
                sig_MC_dict["VBF"] = {
                    "values" :np.concatenate(sample_dict[group_name][plot_var], axis=0),
                    "weights":np.concatenate(sample_dict[group_name]["wgt_nominal"], axis=0)
                }
            if len(sample_dict["ggh"]["wgt_nominal"]) > 0:
                group_name = "ggh"
                sig_MC_dict["ggH"] = {
                    "values" :np.concatenate(sample_dict[group_name][plot_var], axis=0),
                    "weights":np.concatenate(sample_dict[group_name]["wgt_nominal"], axis=0)
                }
            # if len(group_ggH_vals) > 0:
            #     sig_MC_dict["ggH"] = {
            #         "values" :np.concatenate(group_ggH_vals, axis=0),
            #         "weights":np.concatenate(group_ggH_weights, axis=0)
            #     }
            # if len(group_VBF_vals) > 0:
            #     sig_MC_dict["VBF"] = {
            #         "values" :np.concatenate(group_VBF_vals, axis=0),
            #         "weights":np.concatenate(group_VBF_weights, axis=0)
            #     }
            
            # print(f"sig_MC_dict: {sig_MC_dict}")
        
            # -------------------------------------------------------
            # All data are prepped, now plot Data/MC histogram
            # -------------------------------------------------------
            # if args.vbf_cat_mode:
            #     production_cat = "vbf"
            # else:
            #     production_cat = "ggh"
            # full_save_path = args.save_path+f"/{args.year}/mplhep/Reg_{args.region}/Cat_{production_cat}"
            # full_save_path = args.save_path+f"/Reg_{args.region}/Cat_{args.category}/{args.label}"
            full_save_path = f"{args.save_path}/{args.label}_x_{args.category}/{args.year}_{args.region}"
        
            
            if not os.path.exists(full_save_path):
                os.makedirs(full_save_path)
            # full_save_fname = f"{full_save_path}/dimuon_mass_cat{sub_cat}.pdf"
            full_save_fname = f"{full_save_path}/{plot_var}_cat{sub_cat}.pdf"
        
        
            plot_setting_fname = "../../../src/lib/histogram/plot_settings_vbfCat_MVA_input.json"
            plot_setting_fname = "plot_settings_vbfCat_MVA_input.json"
            with open(plot_setting_fname, "r") as file:
                plot_settings = json.load(file)
            
            
            binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
            status = "Private"
            
            do_logscale = True
            plotDataMC_compare(
                binning, 
                data_dict, 
                bkg_MC_dict, 
                full_save_fname,
                sig_MC_dict=sig_MC_dict,
                title = "", 
                x_title = plot_settings[plot_var].get("xlabel"), 
                y_title = plot_settings[plot_var].get("ylabel"),
                # lumi = args.lumi,
                lumi = lumi_val,
                status = status,
                log_scale = do_logscale,
            )