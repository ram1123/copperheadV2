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
    args = parser.parse_args()
    load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean/{args.label}/{args.category}/stage2_output/*"
    events = dak.from_parquet(f"{load_path}/*data.parquet")
    # print(events.fields)

    
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