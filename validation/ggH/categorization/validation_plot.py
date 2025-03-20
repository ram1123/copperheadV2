import awkward as ak
import dask_awkward as dak
import argparse
import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# Add it to sys.path
sys.path.insert(0, parent_dir)

# Now you can import your module
from src.lib.histogram.plotting import plotDataMC_compare



def fillSampleValues(events, sample_dict, sample: str):
    print(f"sample_dict b4 : {sample_dict}")
    sample_name = sample.lower()
    if sample_name in sample_dict.keys():
        sample_info = sample_dict[sample_name]
        fields2load = sample_info.keys() # dimuon_mass, wgt_nominal
        
        # compute in parallel fields to load
        computed_zip = ak.zip({
            field : events[field] for field in fields2load
        }).compute()
        
        # add the computed fields to sample_dict 
        for field in fields2load:
            sample_dict[sample_name][field].append(computed_zip[field])
    else:
        print(f"sample {sample_name} not present in sample_dict!")

    print(f"sample_dict after : {sample_dict}")
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
    "-samp",
    "--samples",
    dest="samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of samples to process for stage2. Current valid inputs are data, signal and DY",
    )
    args = parser.parse_args()
    if len(args.samples) == 0:
        print("samples list is zero!")
        raise ValueError
    load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean/{args.label}/{args.category}/stage2_output/*/"
    # events = dak.from_parquet(f"{load_path}/*data.parquet")
    # print(events.fields)
    print(f"load_path : {load_path}")
    print(f"args.samples: {args.samples}")

    possible_samples = ["data", "ggh", "vbf", "dy", "ewk", "tt", "st", "ww", "wz", "zz",]
    
    
    sub_cats = [0]
    for sub_cat in sub_cats:
        # initialize empty dictionaries that will contain the values
        sample_dict = {
            sample_name: {
                "dimuon_mass": [],
                "wgt_nominal" : [],
            } for sample_name in possible_samples
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
            events = events[events.subCategory_idx == sub_cat] # filter subcat
            print(f"events field from {sample}:", events.fields)
            sample_dict = fillSampleValues(events, sample_dict, sample)
            
            
            
    # # define data dict
    # data_dict = {
    #     "values" :np.concatenate(group_data_vals, axis=0),
    #     "weights":np.concatenate(group_data_weights, axis=0)
    # }
    
    # # define Bkg MC dict
    # bkg_MC_dict = OrderedDict()
    # # start from lowest yield to highest yield
    # if len(group_other_vals) > 0:
    #     bkg_MC_dict["other"] = {
    #         "values" :np.concatenate(group_other_vals, axis=0),
    #         "weights":np.concatenate(group_other_weights, axis=0)
    #     }
    # if len(group_VV_vals) > 0:
    #     bkg_MC_dict["VV"] = {
    #         "values" :np.concatenate(group_VV_vals, axis=0),
    #         "weights":np.concatenate(group_VV_weights, axis=0)
    #     }
    # if len(group_Ewk_vals) > 0:
    #     bkg_MC_dict["Ewk"] = {
    #         "values" :np.concatenate(group_Ewk_vals, axis=0),
    #         "weights":np.concatenate(group_Ewk_weights, axis=0)
    #     }
    # if len(group_Top_vals) > 0:
    #     bkg_MC_dict["Top"] = {
    #         "values" :np.concatenate(group_Top_vals, axis=0),
    #         "weights":np.concatenate(group_Top_weights, axis=0)
    #     }
    # if len(group_DY_vals) > 0:
    #     bkg_MC_dict["DY"] = {
    #         "values" :np.concatenate(group_DY_vals, axis=0),
    #         "weights":np.concatenate(group_DY_weights, axis=0)
    #     }

    
    # # bkg_MC_dict = {
    # #     "Top" :{
    # #         "values" :np.concatenate(group_Top_vals, axis=0),
    # #         "weights":np.concatenate(group_Top_weights, axis=0)
    # #     },
    # #     "DY" :{
    # #         "values" :np.concatenate(group_DY_vals, axis=0),
    # #         "weights":np.concatenate(group_DY_weights, axis=0)
    # #     },     
    # # }

    # # define Sig MC dict
    
    # # sig_MC_dict = {
    # #     "ggH" :{
    # #         "values" :np.concatenate(group_ggH_vals, axis=0),
    # #         "weights":np.concatenate(group_ggH_weights, axis=0)
    # #     },  
    # #     "VBF" :{
    # #         "values" :np.concatenate(group_VBF_vals, axis=0),
    # #         "weights":np.concatenate(group_VBF_weights, axis=0)
    # #     },  
    # # }
    # sig_MC_dict = OrderedDict()
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
    


    # # -------------------------------------------------------
    # # All data are prepped, now plot Data/MC histogram
    # # -------------------------------------------------------
    # # if args.vbf_cat_mode:
    # #     production_cat = "vbf"
    # # else:
    # #     production_cat = "ggh"
    # # full_save_path = args.save_path+f"/{args.year}/mplhep/Reg_{args.region}/Cat_{production_cat}"
    # full_save_path = args.save_path+f"/{args.year}/mplhep/Reg_{args.region}/Cat_{args.category}/{args.label}"

    
    # if not os.path.exists(full_save_path):
    #     os.makedirs(full_save_path)
    # full_save_fname = f"{full_save_path}/{var}.pdf"

   
    # plotDataMC_compare(
    #     binning, 
    #     data_dict, 
    #     bkg_MC_dict, 
    #     full_save_fname,
    #     sig_MC_dict=sig_MC_dict,
    #     title = "", 
    #     x_title = plot_settings[plot_var].get("xlabel"), 
    #     y_title = plot_settings[plot_var].get("ylabel"),
    #     lumi = args.lumi,
    #     status = status,
    #     log_scale = do_logscale,
    # )