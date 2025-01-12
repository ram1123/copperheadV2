import awkward as ak
import dask_awkward as dak
import numpy as np
import json
import argparse
import sys
import os
from distributed import Client
import time    
import tqdm
import cmsstyle as CMS
from collections import OrderedDict
import glob
import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib

# Add the parent directory to the system path
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")) # in order to import plotDataMC_compare
sys.path.append(main_dir)
from src.lib.histogram.plotting import plotDataMC_compare


# real process arrangement
group_data_processes = ["data_A", "data_B", "data_C", "data_D", "data_E",  "data_F", "data_G", "data_H"]
# group_DY_processes = ["dy_M-100To200", "dy_M-50"] # dy_M-50 is not used in ggH BDT training input
# group_DY_processes = ["dy_M-100To200"]
# group_DY_processes = ["dy_M-100To200","dy_VBF_filter"]
group_DY_processes = [
    "dy_M-50",
    "dy_M-100To200",
    "dy_m105_160_amc",
    "dy_m105_160_vbf_amc",
    "dy_VBF_filter_customJMEoff",
    "dy_VBF_filter_fromGridpack",
]
# group_DY_processes = ["dy_M-100To200","dy_VBF_filter_customJMEoff"]
# group_DY_processes = [] # just VBf filter

group_Top_processes = ["ttjets_dl", "ttjets_sl", "st_tw_top", "st_tw_antitop"]
group_Ewk_processes = ["ewk_lljj_mll105_160_ptj0"]
group_VV_processes = ["ww_2l2nu", "wz_3lnu", "wz_2l2q", "wz_1l1nu2q", "zz"]# diboson
# group_ggH_processes = ["ggh_amcPS"]
group_ggH_processes = ["ggh_powhegPS"]
group_VBF_processes = ["vbf_powheg_dipole"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="2018",
    action="store",
    help="string value of year we are calculating",
    )
    parser.add_argument(
    "-data",
    "--data",
    dest="data_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of data samples represented by alphabetical letters A-H",
    )
    parser.add_argument(
    "-bkg",
    "--background",
    dest="bkg_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of bkg samples represented by shorthands: DY, TT, ST, DB (diboson), EWK",
    )
    parser.add_argument(
    "-sig",
    "--signal",
    dest="sig_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of sig samples represented by shorthands: ggH, VBF",
    )
    parser.add_argument(
    "-var",
    "--variables",
    dest="variables",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of variables to plot (ie: jet, mu, dimuon)",
    )
    parser.add_argument(
    "-load",
    "--load_path",
    dest="load_path",
    default="/depot/cms/users/yun79/results/stage1/test_full/f0_1",
    action="store",
    help="load path",
    )
    parser.add_argument(
    "-label",
    "--label",
    dest="label",
    default="",
    action="store",
    help="label",
    )
    parser.add_argument(
    "-save",
    "--save_path",
    dest="save_path",
    default="./plots/",
    action="store",
    help="save path",
    )
    parser.add_argument(
    "-lumi",
    "--lumi",
    dest="lumi",
    default="",
    action="store",
    help="string value of integrated luminosity to label",
    )
    parser.add_argument(
    "--status",
    dest="status",
    default="",
    action="store",
    help="Status of results ie Private, Preliminary, In Progress",
    )
    parser.add_argument(
    "--no_ratio",
    dest="no_ratio",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="doesn't plot Data/MC ratio",
    )
    parser.add_argument(
    "--linear_scale",
    dest="linear_scale",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, provide plots in linear scale",
    )
    parser.add_argument(
    "-reg",
    "--region",
    dest="region",
    default="signal",
    action="store",
    help="region value to plot, available regions are: h_peak, h_sidebands, z_peak and signal (h_peak OR h_sidebands)",
    )
    parser.add_argument(
    "--use_gateway",
    dest="use_gateway",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, uses dask gateway client instead of local",
    )
    parser.add_argument(
    "-cat",
    "--category",
    dest="category",
    default="nocat",
    action="store",
    help="define production mode category. optionsare ggh, vbf and nocat (no category cut)",
    )
    parser.add_argument(
    "-plotset",
    "--plot_setting",
    dest="plot_setting",
    default="",
    action="store",
    help="path to load json file with plott binning and xlabels",
    )
    parser.add_argument(
    "--zpt_on",
    dest="zpt_on",
    type=lambda x: x.lower() == 'true',  # Convert input string to a boolean
    default=False,  # Default value if the argument is not provided
    help="If false, divide zpt separate weight",
    )
    parser.add_argument(
    "-jetm",
    "--jet_multiplicity",
    dest="jet_multiplicity",
    default="0",
    action="store",
    help="Integer values of jet multiplicity to filter. Available options are: 0, 1 and 2",
    )
    #---------------------------------------------------------
    # gather arguments
    args = parser.parse_args()
    available_processes = []
    # if doing VBF filter study, add the vbf filter sample to the DY group
    
    # take data
    data_samples = args.data_samples
    if len(data_samples) >0:
        for data_letter in data_samples:
            available_processes.append(f"data_{data_letter.upper()}")
        
        # # take data as one group to save load time 
        # available_processes.append(f"data_*")
    # take bkg
    bkg_samples = args.bkg_samples
    if len(bkg_samples) >0:
        for bkg_sample in bkg_samples:
            if bkg_sample.upper() == "DY": # enforce upper case to prevent confusion
                # available_processes.append("dy_M-50")
                available_processes.append("dy_M-100To200")
                available_processes.append("dy_m105_160_amc")
            elif bkg_sample.upper() == "TT": # enforce upper case to prevent confusion
                available_processes.append("ttjets_dl")
                available_processes.append("ttjets_sl")
            elif bkg_sample.upper() == "ST": # enforce upper case to prevent confusion
                available_processes.append("st_tw_top")
                available_processes.append("st_tw_antitop")
            elif bkg_sample.upper() == "VV": # enforce upper case to prevent confusion
                available_processes.append("ww_2l2nu")
                available_processes.append("wz_3lnu")
                available_processes.append("wz_2l2q")
                # available_processes.append("wz_1l1nu2q")
                available_processes.append("zz")
            elif bkg_sample.upper() == "EWK": # enforce upper case to prevent confusion
                available_processes.append("ewk_lljj_mll105_160_ptj0")
            else:
                print(f"unknown background {bkg_sample} was given!")
        
    # take sig
    sig_samples = args.sig_samples
    if len(sig_samples) >0:
        for sig_sample in sig_samples:
            if sig_sample.upper() == "GGH": # enforce upper case to prevent confusion
                # available_processes.append("ggh_amcPS")
                available_processes.append("ggh_powhegPS")
            elif sig_sample.upper() == "VBF": # enforce upper case to prevent confusion
                available_processes.append("vbf_powheg_dipole")
            else:
                print(f"unknown signal {sig_sample} was given!")
    # gather variables to plot:
    kinematic_vars = ['pt', 'eta', 'phi']
    # kinematic_vars = ['pt']
    variables2plot = []
    if len(args.variables) == 0:
        print("no variables to plot!")
        raise ValueError
    for particle in args.variables:
        if "dimuon" in particle:
            variables2plot.append(f"{particle}_mass")
            variables2plot.append(f"{particle}_pt")
            # variables2plot.append(f"{particle}_eta")
            # variables2plot.append(f"{particle}_phi")
            # variables2plot.append(f"{particle}_cos_theta_cs")
            # variables2plot.append(f"{particle}_phi_cs")
            # variables2plot.append(f"mmj_min_dPhi_nominal")
            # variables2plot.append(f"mmj_min_dEta_nominal")
            # variables2plot.append(f"rpt_nominal")
            # variables2plot.append(f"ll_zstar_log_nominal")
            
            # --------------------------------------------------
            # variables2plot.append(f"rpt")
            # variables2plot.append(f"ll_zstar_log")
            # variables2plot.append(f"dimuon_ebe_mass_res")
            # variables2plot.append(f"{particle}_rapidity")
        elif "dijet" in particle:
            # variables2plot.append(f"gjj_mass")
            # variables2plot.append(f"jj_mass_nominal")
            # variables2plot.append(f"jj_pt_nominal")
            # variables2plot.append(f"jj_dEta_nominal")
            # variables2plot.append(f"jj_dPhi_nominal")
            # variables2plot.append(f"zeppenfeld_nominal")
            # variables2plot.append(f"rpt_nominal")
            variables2plot.append(f"pt_centrality_nominal")
            variables2plot.append(f"nsoftjets2_nominal")
            variables2plot.append(f"htsoft2_nominal")
            variables2plot.append(f"nsoftjets5_nominal")
            variables2plot.append(f"htsoft5_nominal")
            
        elif ("mu" in particle) :
            for kinematic in kinematic_vars:
                # plot both leading and subleading muons/jets
                variables2plot.append(f"{particle}1_{kinematic}")
                variables2plot.append(f"{particle}2_{kinematic}")
        elif ("jet" in particle):
            variables2plot.append(f"njets_nominal")
            for kinematic in kinematic_vars:
                # plot both leading and subleading muons/jets
                variables2plot.append(f"{particle}1_{kinematic}_nominal")
                variables2plot.append(f"{particle}2_{kinematic}_nominal")
            variables2plot.append(f"jet1_qgl_nominal")
            variables2plot.append(f"jet2_qgl_nominal")
       
        else:
            print(f"Unsupported variable: {particle} is given!")
    print(f"variables2plot: {variables2plot}")
    # plot_setting_fname = "../../src/lib/histogram/plot_settings_vbfCat_MVA_input.json"
    plot_setting_fname = args.plot_setting
    if plot_setting_fname == "":
        print("ERROR, valid plotting setting json file needs to be given")
        raise ValueError

    with open(plot_setting_fname, "r") as file:
        plot_settings = json.load(file)
    status = args.status.replace("_", " ")
    
    # define client for parallelization 
    if args.use_gateway:
            from dask_gateway import Gateway
            gateway = Gateway(
                "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
                proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
            )
            cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
            client = gateway.connect(cluster_info.name).get_client()
            print("Gateway Client created")
    else:
        client =  Client(n_workers=31,  threads_per_worker=1, processes=True, memory_limit='4 GiB') 
        print("Local scale Client created")
    # record time
    time_step = time.time()
    zpt_on = args.zpt_on
    jet_multiplicity = int(args.jet_multiplicity)
    # load saved parquet files. This increases memory use, but increases runtime significantly
    print(f"available_processes: {available_processes}")
    loaded_events = {} # intialize dictionary containing all the arrays
    for process in tqdm.tqdm(available_processes):
        print(f"loading process {process}..")
        # full_load_path = args.load_path+f"/{process}/*.parquet"
        full_load_path = args.load_path+f"/{process}/*/*.parquet"
        if len(glob.glob(full_load_path)) ==0: # check if there's files in the load path
            full_load_path = args.load_path+f"/{process}/*.parquet" # try coppperheadV1 path, if this also is empty, then skip
        print(f"full_load_path: {full_load_path}")
        try:
            events = dak.from_parquet(full_load_path)
        except:
            print(f"full_load_path: {full_load_path} Not available. Skipping")
            continue
        # print(f"events.fields: {events.fields}")

        # ------------------------------------------------------
        # select only needed variables to load to save run time
        # ------------------------------------------------------
        
        fields2load = variables2plot + [
            "wgt_nominal",
            # "fraction", 
            # "h_sidebands", 
            # "h_peak", 
            # "z_peak", 
            # "vbf_cut",
            "nBtagLoose_nominal", 
            "nBtagMedium_nominal", 
            "dimuon_mass",
            "zeppenfeld_nominal", 
            "jj_mass_nominal", 
            "jet1_pt_nominal", 
            "jj_dEta_nominal", 
            "dimuon_pt", 
            "njets_nominal",
        ]

            
        # # add in weights
        # for field in events.fields:
        #     if "wgt_nominal" in field:
        #         fields2load.append(field)
                
        is_data = "data" in process.lower()
        if not is_data: # MC sample
            if not zpt_on:
                print("Adding seperate zpt wgt!")
                if "separate_wgt_zpt_wgt" in events.fields:
                    fields2load.append("separate_wgt_zpt_wgt")
            

        # filter out redundant fields by using the set object
        fields2load = list(set(fields2load))
        
        events = events[fields2load]
        # load data to memory using compute()
        events = ak.zip({
            field : events[field] for field in events.fields
        }).compute()
        loaded_events[process] = events
    print("finished loading parquet files!")

    
    # hep.style.use("CMS")
    # Load CMS style including color-scheme (it's an editable dict)
    plt.style.use(hep.style.CMS)
    # this mplhep implementation assumes non-empty data; otherwise, it will crash
    # Dictionary for histograms and binnings

    for var in tqdm.tqdm(variables2plot):
        var_step = time.time()
        # for process in available_processes:
        if "_nominal" in var:
            plot_var = var.replace("_nominal", "")
        else:
            plot_var = var
        if plot_var not in plot_settings.keys():
            print(f"variable {var} not configured in plot settings!")
            continue
        #-----------------------------------------------
        # intialize variables for filling histograms
        binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
        if args.linear_scale:
            do_logscale = False
        else:
            do_logscale = True
        # also check if logscale config is mentioned in plot_settings, if yes, that config takes priority
        # if "logscale" in plot_settings[plot_var].keys():
        #     do_logscale = plot_settings[plot_var]["logscale"]
        print(f"do_logscale: {do_logscale} ")

        group_data_vals = []
        group_DY_vals = []
        group_Top_vals = []
        group_Ewk_vals = []
        group_VV_vals = []
        group_other_vals = []  # histograms not belonging to any other mc bkg group
        group_ggH_vals = [] # there should only be one ggH histogram, but making a list for consistency
        group_VBF_vals = [] # there should only be one VBF histogram, but making a list for consistency
        
        group_data_weights = []
        group_DY_weights = []
        group_Top_weights = []
        group_Ewk_weights = []
        group_VV_weights = []
        group_other_weights = []
        group_ggH_weights = []
        group_VBF_weights = []

        
        for process in available_processes:    
            print(f"process: {process}")
            try:
                events = loaded_events[process]
            except:
                print(f"skipping {process}")
                continue
            is_data = "data" in process.lower()
            print(f"is_data: {is_data}")
            if is_data:
                weights = ak.to_numpy(ak.fill_none(events["wgt_nominal"], value=0.0))
            else: # MC
                weights = ak.fill_none(events["wgt_nominal"], value=0.0)
                
                # weights = weights/events.wgt_nominal_muID/ events.wgt_nominal_muIso / events.wgt_nominal_muTrig #  quick test
                # temporary over write
                # print(f"events.fields: {events.fields}")
                if (not zpt_on) and ("separate_wgt_zpt_wgt" in events.fields):
                    print("removing Zpt rewgt!")
                    weights = weights/events["separate_wgt_zpt_wgt"]

                
                # print(f"weights {process} b4 numpy: {weights}")
                weights = ak.to_numpy(weights) # MC are already normalized by xsec*lumi
                # for some reason, some nan weights are still passes ak.fill_none() bc they're "nan", not None, this used to be not a problem
                # could be an issue of copying bunching of parquet files from one directory to another, but not exactly sure
                weights = np.nan_to_num(weights, nan=0.0) 
            #-----------------------------------------------    
            # obtain the category selection

            

            # ------------------------------------------------
            # take the mass region and category cuts 
            # ------------------------------------------------

            # do mass region cut
            mass = events.dimuon_mass
            z_peak = ((mass > 76) & (mass < 106))
            h_sidebands =  ((mass > 110) & (mass < 115.03)) | ((mass > 135.03) & (mass < 150))
            h_peak = ((mass > 115.03) & (mass < 135.03))
            if args.region == "signal":
                region = h_sidebands | h_peak
            elif args.region == "h_peak":
                region = h_peak 
            elif args.region == "h_sidebands":
                print("h_sidebands region chosen!")
                region = h_sidebands 
            elif args.region == "z_peak":
                region = z_peak 
            else: 
                print("ERROR: acceptable region!")
                raise ValueError

            # do category cut
            btag_cut =ak.fill_none((events.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((events.nBtagMedium_nominal >= 1), value=False)
            # vbf_cut = ak.fill_none(events.vbf_cut, value=False) # in the future none values will be replaced with False
            vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) 
            vbf_cut = ak.fill_none(vbf_cut, value=False)
            # if args.vbf_cat_mode:
            if args.category == "vbf":
                print("vbf mode!")
                prod_cat_cut =  vbf_cut & ak.fill_none(events.jet1_pt_nominal > 35, value=False) 
                prod_cat_cut = prod_cat_cut & ~btag_cut # btag cut is for VH and ttH categories
                print("applying jet1 pt 35 Gev cut!")
            # else: # we're interested in ggH category
            elif args.category == "ggh":
                print("ggH mode!")
                prod_cat_cut =  ~vbf_cut 
                prod_cat_cut = prod_cat_cut & ~btag_cut # btag cut is for VH and ttH categories
            elif args.category == "nocat":
                print("nocat mode!")
                prod_cat_cut =  ak.ones_like(vbf_cut, dtype="bool")
            else:
                print("Error: invalid category option!")
                raise ValueError

            # do jet multiplicty cut
            if jet_multiplicity != 2:
                jet_multiplicity_cut = events.njets_nominal == jet_multiplicity
            else :
                jet_multiplicity_cut = events.njets_nominal >= jet_multiplicity
            
            category_selection = (
                prod_cat_cut & 
                region &
                jet_multiplicity_cut
            )
            print(f"category_selection length: {len(category_selection)}")
            print(f"category_selection {process} sum : {ak.sum(ak.values_astype(category_selection, np.int32))}")
            # print(f"category_selection {process} : {category_selection}")

            # filter events fro selected category
            category_selection = ak.to_numpy(category_selection) # this will be multiplied with weights
            # weights = weights*category_selection
            weights = weights[category_selection]
            
            # 
            events = events[category_selection]
            fraction_weight = ak.ones_like(events.wgt_nominal) # TBF, all fractions should be same
            print(f"var: {var}")
            values = ak.to_numpy(ak.fill_none(events[var], value=-999.0))
            # print(f"weights.shape: {weights[weights>0].shape}")
            
            # temporary overwrite start -------------------------
            # we have bad ll_zstar_log caluclation, so we re-calculate on the spot
            # if var == "ll_zstar_log":
            #     print("ll_zstar_log overwrite!")
            #     values = ak.to_numpy(np.log(np.abs(events["zeppenfeld"])))
            # elif var == "rpt":
            #     print("rpt overwrite!")
            #     numerator = np.abs(events["jj_pt"] + events["dimuon_pt"])
            #     denominator = np.abs(events["jet1_pt"]) + np.abs(events["jet2_pt"]) +  np.abs(events["dimuon_pt"])
            #     values = ak.to_numpy(numerator/denominator)
            #     # debug
            #     print(f"events.jj_pt is nan: {np.any(np.isnan(events.jj_pt))}")
            #     print(f"events.dimuon_pt is nan: {np.any(np.isnan(events.dimuon_pt))}")
            #     print(f"events.jet1_pt is nan: {np.any(np.isnan(events.jet1_pt))}")
            #     print(f"events.jet2_pt is nan: {np.any(np.isnan(events.jet2_pt))}")
            #     print(f"events.jj_pt is none: {np.any(ak.is_none(events.jj_pt))}")
            #     print(f"events.dimuon_pt is none: {np.any(ak.is_none(events.dimuon_pt))}")
            #     print(f"events.jet1_pt is none: {np.any(ak.is_none(events.jet1_pt))}")
            #     print(f"events.jet2_pt is none: {np.any(ak.is_none(events.jet2_pt))}")
                
            print(f"values is nan: {np.any(np.isnan(values))}")
            print(f"values is none: {np.any(ak.is_none(values))}")
            # temporary overwrite end -------------------------
            # print(f"values[0]: {values[0]}")
            values_filter = values!=-999.0
            values = values[values_filter]
            weights = weights[values_filter]
            # MC samples are already normalized by their xsec*lumi, but data is not
            if process in group_data_processes:
                fraction_weight = fraction_weight[values_filter]
                # print(f"fraction_weight: {fraction_weight}")
                weights = weights*fraction_weight
            # print(f"weights.shape: {weights[weights>0].shape}")

            if process in group_data_processes:
                print("data activated")
                group_data_vals.append(values)
                group_data_weights.append(weights)
            #-------------------------------------------------------
            elif process in group_DY_processes:
                print("DY activated")
                group_DY_vals.append(values)
                group_DY_weights.append(weights)
            #-------------------------------------------------------
            elif process in group_Top_processes:
                print("top activated")
                group_Top_vals.append(values)
                group_Top_weights.append(weights)
            #-------------------------------------------------------
            elif process in group_Ewk_processes:
                print("Ewk activated")
                group_Ewk_vals.append(values)
                group_Ewk_weights.append(weights)
            #-------------------------------------------------------
            elif process in group_VV_processes:
                print("VV activated")
                group_VV_vals.append(values)
                group_VV_weights.append(weights)
            #-------------------------------------------------------
            elif process in group_ggH_processes:
                print("ggH activated")
                group_ggH_vals.append(values)
                group_ggH_weights.append(weights)
            #-------------------------------------------------------
            elif process in group_VBF_processes:
                print("VBF activated")
                group_VBF_vals.append(values)
                group_VBF_weights.append(weights)
            #-------------------------------------------------------
            else: # put into "other" bkg group
                print("other activated")
                group_other_vals.append(values)
                group_other_weights.append(weights)
        
        
        # -------------------------------------------------------
        # Aggregate the data into Sample types b4 plotting
        # -------------------------------------------------------

        # define data dict
        data_dict = {
            "values" :np.concatenate(group_data_vals, axis=0),
            "weights":np.concatenate(group_data_weights, axis=0)
        }
        
        # define Bkg MC dict
        bkg_MC_dict = OrderedDict()
        # start from lowest yield to highest yield
        if len(group_other_vals) > 0:
            bkg_MC_dict["other"] = {
                "values" :np.concatenate(group_other_vals, axis=0),
                "weights":np.concatenate(group_other_weights, axis=0)
            }
        if len(group_VV_vals) > 0:
            bkg_MC_dict["VV"] = {
                "values" :np.concatenate(group_VV_vals, axis=0),
                "weights":np.concatenate(group_VV_weights, axis=0)
            }
        if len(group_Ewk_vals) > 0:
            bkg_MC_dict["Ewk"] = {
                "values" :np.concatenate(group_Ewk_vals, axis=0),
                "weights":np.concatenate(group_Ewk_weights, axis=0)
            }
        if len(group_Top_vals) > 0:
            bkg_MC_dict["Top"] = {
                "values" :np.concatenate(group_Top_vals, axis=0),
                "weights":np.concatenate(group_Top_weights, axis=0)
            }
        if len(group_DY_vals) > 0:
            bkg_MC_dict["DY"] = {
                "values" :np.concatenate(group_DY_vals, axis=0),
                "weights":np.concatenate(group_DY_weights, axis=0)
            }

        
        # bkg_MC_dict = {
        #     "Top" :{
        #         "values" :np.concatenate(group_Top_vals, axis=0),
        #         "weights":np.concatenate(group_Top_weights, axis=0)
        #     },
        #     "DY" :{
        #         "values" :np.concatenate(group_DY_vals, axis=0),
        #         "weights":np.concatenate(group_DY_weights, axis=0)
        #     },     
        # }

        # define Sig MC dict
        
        # sig_MC_dict = {
        #     "ggH" :{
        #         "values" :np.concatenate(group_ggH_vals, axis=0),
        #         "weights":np.concatenate(group_ggH_weights, axis=0)
        #     },  
        #     "VBF" :{
        #         "values" :np.concatenate(group_VBF_vals, axis=0),
        #         "weights":np.concatenate(group_VBF_weights, axis=0)
        #     },  
        # }
        sig_MC_dict = OrderedDict()
        if len(group_ggH_vals) > 0:
            sig_MC_dict["ggH"] = {
                "values" :np.concatenate(group_ggH_vals, axis=0),
                "weights":np.concatenate(group_ggH_weights, axis=0)
            }
        if len(group_VBF_vals) > 0:
            sig_MC_dict["VBF"] = {
                "values" :np.concatenate(group_VBF_vals, axis=0),
                "weights":np.concatenate(group_VBF_weights, axis=0)
            }
        


        # -------------------------------------------------------
        # All data are prepped, now plot Data/MC histogram
        # -------------------------------------------------------
        # if args.vbf_cat_mode:
        #     production_cat = "vbf"
        # else:
        #     production_cat = "ggh"
        # full_save_path = args.save_path+f"/{args.year}/mplhep/Reg_{args.region}/Cat_{production_cat}"
        if zpt_on:
            full_save_path = args.save_path+f"/{args.year}/Reg_{args.region}/Cat_{args.category}/{args.label}/Njet_{jet_multiplicity}/ZptReWgt_On"
        else:
            full_save_path = args.save_path+f"/{args.year}/Reg_{args.region}/Cat_{args.category}/{args.label}/Njet_{jet_multiplicity}/ZptReWgt_Off"
        
        if not os.path.exists(full_save_path):
            os.makedirs(full_save_path)
            
        full_save_fname = f"{full_save_path}/{var}.pdf"
        plotDataMC_compare(
            binning, 
            data_dict, 
            bkg_MC_dict, 
            full_save_fname,
            sig_MC_dict=sig_MC_dict,
            title = "", 
            x_title = plot_settings[plot_var].get("xlabel"), 
            y_title = plot_settings[plot_var].get("ylabel"),
            lumi = args.lumi,
            status = status,
            log_scale = do_logscale,
        )
        full_save_fname = f"{full_save_path}/{var}.png"
        plotDataMC_compare(
            binning, 
            data_dict, 
            bkg_MC_dict, 
            full_save_fname,
            sig_MC_dict=sig_MC_dict,
            title = "", 
            x_title = plot_settings[plot_var].get("xlabel"), 
            y_title = plot_settings[plot_var].get("ylabel"),
            lumi = args.lumi,
            status = status,
            log_scale = do_logscale,
        )
        


        

        var_elapsed = round(time.time() - var_step, 3)
        print(f"Finished processing {var} in {var_elapsed} s.")
    
    time_elapsed = round(time.time() - time_step, 3)
    print(f"Finished in {time_elapsed} s.")