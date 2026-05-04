import time
import numpy as np
import pickle
import awkward as ak
import dask_awkward as dak
from distributed import Client
from omegaconf import OmegaConf
from typing import Tuple, List, Dict
import ROOT
import argparse
import os
import copy
import pandas as pd
import glob

# get parent directory of current file
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(parent_dir)
from modules.fit_functions import MakeFEWZxBernDof3, plot_6_23, plot_6_26, getSigBkgPdf
from modules.fit_functions import getBWZ_gamma, getBWZxBern, getLandxBern, getFEWZxBern, getPowerLaw
from modules.GoF_utils import getGOF_KS

FUNC_ORDER = ['BWZRedux', 'BwzGamma', 'BWZxBern', 'sumExp', 'PowerLaw', 'FEWZxBern', 'LandauxBern', 'Polynomial']


def create_core_pdf(pdf_type, subcat_index, x, init_vals):
    """
    Create a RooAbsPdf of a given type with per-subcategory RooRealVars,
    initializing them with provided values and ranges.

    Args:
        pdf_type (str): 'BWZRedux' or 'sumExp'.
        subcat_index (int): Index for naming the PDF.
        x (RooRealVar): Observable (formerly 'mass').
        init_vals (dict): Dict with float values keyed by param name.

    Returns:
        tuple: (RooAbsPdf, [list of RooRealVar parameters])
    """
    prefix = f"subCat{subcat_index}_{pdf_type}"

    if pdf_type == "BWZRedux":
        a = ROOT.RooRealVar(f"{prefix}_a_coeff", f"{prefix}_a_coeff", 
                          init_vals["a_coeff"], -0.0,0.2)
        b = ROOT.RooRealVar(f"{prefix}_b_coeff", f"{prefix}_b_coeff", 
                          init_vals["b_coeff"], -0.01, 0.01)
        c = ROOT.RooRealVar(f"{prefix}_c_coeff", f"{prefix}_c_coeff", 
                          init_vals["c_coeff"], 0,5) 
        pdf = ROOT.RooModZPdf(prefix, prefix, x, a, b, c)
        return pdf, [a, b, c]
    elif pdf_type == "sumExp":
        # a1 = ROOT.RooRealVar(f"{prefix}_a1_coeff", f"{prefix}_a1_coeff", 
        #                    init_vals["a1_coeff"], -1.0, 0)
        # a2 = ROOT.RooRealVar(f"{prefix}_a2_coeff", f"{prefix}_a2_coeff", 
        #                    init_vals["a2_coeff"], -1.0, 0.0)
        a1 = ROOT.RooRealVar(f"{prefix}_a1_coeff", f"{prefix}_a1_coeff", 
                           init_vals["a1_coeff"], -0.3, 0)
        a2 = ROOT.RooRealVar(f"{prefix}_a2_coeff", f"{prefix}_a2_coeff", 
                           init_vals["a2_coeff"], -0.3, 0.0)
        f  = ROOT.RooRealVar(f"{prefix}_f_coeff", f"{prefix}_f_coeff", 
                           init_vals["f_coeff"], 0.0, 1.0)
        pdf = ROOT.RooSumTwoExpPdf(prefix, prefix, x, a1, a2, f)

        return pdf, [a1, a2, f]
        
    elif pdf_type == "PowerLaw":
        pdf, param_l = getPowerLaw(x, init_vals)
        return pdf, param_l
        
    elif pdf_type == "FEWZxBern":
        pdf, param_l = getFEWZxBern(x, init_vals, fewz_workspace_path="./modules/ucsd_workspace/")
        return pdf, param_l
        
    elif pdf_type == "LandauxBern":
        pdf, param_l = getLandxBern(x, init_vals)
        return pdf, param_l
        
    elif pdf_type == "BWZxBern":
        pdf, param_l = getBWZxBern(x, init_vals)
        return pdf, param_l
        
    elif pdf_type == "BwzGamma":
        pdf, param_l = getBWZ_gamma(x, init_vals)
        return pdf, param_l
        
    elif pdf_type == "Polynomial":
        a0 = ROOT.RooRealVar("polynomial_a0", "polynomial_a0", init_vals["a0"], -5, 5)
        a1 = ROOT.RooRealVar("polynomial_a1", "polynomial_a1", init_vals["a1"], -5, 5)
        a2 = ROOT.RooRealVar("polynomial_a2", "polynomial_a2", init_vals["a2"], -5, 5)

        name = "polynomial_dof3"
        pdf = ROOT.RooChebychev(name, name, x, [a0, a1, a2])
        param_l = [a0, a1, a2]
        return pdf, param_l
        
    else:
        raise ValueError(f"Unsupported PDF type: {pdf_type}")


def getEnvelope(coreFunction_dict, cat_ix : int, func_order : list):
    # intialize RooCategory for each subCat with a same name so that combine changes index over all cats
    cat_index = ROOT.RooCategory(f"pdf_index_ggh","Index of Pdf which is active") # if I have one rooCategory over all categories, then I get "ERROR:InputArguments -- RooAbsCategory::defineState(pdf_index_ggh): index X (0-4) already assigned"
    
    pdf_list = []
    for func_name in func_order:
        pdf = coreFunction_dict[func_name][cat_ix]
        pdf_list.append(pdf)

    print([pdf.Print() for pdf in pdf_list])
    pdf_list = ROOT.RooArgList(*pdf_list)

    env_pdf = ROOT.RooMultiPdf(f"EnvPdf_cat{cat_ix}", f"EnvPdf_cat{cat_ix}", cat_index, pdf_list)
    penalty = 0.5
    env_pdf.setCorrectionFactor(penalty) 
    # [pdf.Print() for pdf in pdf_list]
    print(f"cat{cat_ix} cat_index: {cat_index}")
    
    return env_pdf, [pdf_list, cat_index]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-load",
    "--load_path",
    dest="load_path",
    default=None,
    action="store",
    help="save path to store stage1 output files",
    )
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="all",
    action="store",
    help="string value of year we are calculating",
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
    "-l",
    "--label",
    dest="label",
    default="",
    action="store",
    help="MVA model name to load",
    )
    parser.add_argument(
    "-save",
    "--save_path",
    dest="save_path",
    default=".",
    action="store",
    help="root output path for stage3-style products",
    )
    parser.add_argument(
    "--max-chi2ndf",
    dest="max_chi2ndf",
    default=2.0,
    type=float,
    help="maximum allowed sideband chi2/ndf for a truth-function candidate",
    )
    args = parser.parse_args()
    # check for valid arguments
    if args.load_path == None:
        print("load path to load stage1 output is not specified!")
        raise ValueError

    category = args.category.lower()
    # load_path = "/work/users/yun79/stage2_output/ggH/test/processed_events_data.parquet"
    # load_path = f"{args.load_path}/{category}/{args.year}/processed_events_data.parquet"
    # if args.year=="all":
    #     load_path = f"{args.load_path}/{category}/*/processed_events_data.parquet"
    # elif args.year=="2016only":
    #     load_path = f"{args.load_path}/{category}/2016*/processed_events_data.parquet"
    # else:
    #     load_path = f"{args.load_path}/{category}/{args.year}/processed_events_data.parquet"

    # remove category we assume that the load_path already has category specified
    if args.year=="all":
        load_path = f"{args.load_path}/*/processed_events_data_*.parquet"
    elif args.year=="2016only":
        load_path = f"{args.load_path}/2016*/processed_events_data_*.parquet"
    else:
        load_path = f"{args.load_path}/{args.year}/processed_events_data_*.parquet"
    print(f"load_path: {load_path}")
    processed_eventsData = dak.from_parquet(load_path).compute()
    print(f"processed_eventsData length: {ak.num(processed_eventsData.dimuon_mass, axis=0)}")
    print("events loaded!")

    # Keep bias-study outputs inside the same stage3 tree as the rest of the workflow,
    # but isolate them in a dedicated subdirectory so they do not collide with stage3 workspaces.
    stage3_base_path = f"{args.save_path}/stage3/{args.year}/{args.label}"
    base_path = f"{stage3_base_path}/bias_test"
    plot_save_path = base_path
    os.makedirs(plot_save_path, exist_ok=True)

    # Define your list of column names
    column_list = ["year", "category", "dataset", "yield"]
    yield_df = pd.DataFrame(columns=column_list)

    
    device = "cpu"
    # device = "cuda"
    # ROOT.RooAbsReal.setCudaMode(True)
    # Create model for physics sample
    # -------------------------------------------------------------
    # Create observables
    mass_name = "mh_ggh"
    mass = ROOT.RooRealVar(mass_name, mass_name, 120, 110, 150)
    nbins = 800
    mass.setBins(nbins)
    mass.setRange("hiSB", 135, 150 )
    mass.setRange("loSB", 110, 115 )
    mass.setRange("h_peak", 115, 135 )
    mass.setRange("full", 110, 150 )
    # fit_range = "loSB,hiSB" # we're fitting bkg only
    fit_range = "hiSB,loSB" # we're fitting bkg only
    
    subCatIdx_name = "subCategory_idx"
    # subCatIdx_name = "subCategory_idx_val"

    # Initialize BWZ Redux
    # --------------------------------------------------------------


    # # # trying bigger range do that I don't get warning message from combine like: [WARNING] Found parameter BWZ_Redux_a_coeff at boundary (within ~1sigma)
    # name = f"BWZ_Redux_a_coeff"
    # a_coeff = ROOT.RooRealVar(name,name, 5.1288e-02,-0.5,0.5)
    # name = f"BWZ_Redux_b_coeff"
    # b_coeff = ROOT.RooRealVar(name,name, -1.3658e-04,-0.02,0.02)
    # name = f"BWZ_Redux_c_coeff"
    # c_coeff = ROOT.RooRealVar(name,name, 2.0602e+00,-10.0,10.0)
    # # # old end --------------------------------------------------

    # # sumexp subcat
    # name = f"RooSumTwoExpPdf_a1_coeff"
    # a1_coeff = ROOT.RooRealVar(name,name, -1.4756e-01,-2.0,1)
    # name = f"RooSumTwoExpPdf_a2_coeff"
    # a2_coeff = ROOT.RooRealVar(name,name, -3.4552e-02,-2.0,1)
    # name = f"RooSumTwoExpPdf_f_coeff"
    # f_coeff = ROOT.RooRealVar(name,name,  2.4864e-01,0.0,1.0)


    nSubCats = 5
    # nSubCats = 1 #FIXME
    coreFunction_dict = {
        "BWZRedux" : [],
        "BwzGamma" : [],
        "BWZxBern" : [],
        "sumExp" : [],
        "PowerLaw" : [],
        "FEWZxBern" : [],
        "LandauxBern" : [],
        "Polynomial" : [],
    }
    
    # subCat 0
    # for ix in range(nSubCats):
    #     # BWZRedux
    #     core_func_name = "BWZRedux" # match with one of the keys of coreFunction_dict
    #     # name = f"subCat{ix}_{core_func_name}"
    #     core_func, _ = create_core_pdf(core_func_name, ix, mass, init_vals)
        
    #     # core_func = ROOT.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff)
    #     coreFunction_dict[core_func_name].append(core_func)

        
    #     # sumExp
    #     core_func_name = "sumExp" # match with one of the keys of coreFunction_dict
    #     name = f"subCat{ix}_{core_func_name}"
    #     core_func = ROOT.RooSumTwoExpPdf(name, name, mass, a1_coeff, a2_coeff, f_coeff) 
    #     coreFunction_dict[core_func_name].append(core_func)

    # bwz_init_vals = {
    #     "a_coeff": 5.1288e-02,
    #     "b_coeff": -1.3658e-04,
    #     "c_coeff": 2.0602e+00,
    # }
    bwz_init_vals = { # Aug17 2025
        "a_coeff": 3.9611e-02,
        "b_coeff": -9.9358e-05,
        "c_coeff": 1.9978e+00,
    }
    sumexp_init_vals = {
        "a1_coeff": -1.4756e-01,
        "a2_coeff": -3.4552e-02,
        "f_coeff": 2.4864e-01,
    }
    # powerlaw_init_vals = {
    #     "RooSumTwoPowerLawPdf_a1_coeff": 0.00001,
    #     "PowerLaw_a2_cRooSumTwoPowerLawPdf_a2_coeffoeff": 0.1,
    #     "RooSumTwoPowerLawPdf_f_coeff": 0.9,
    # }
    # power law is very sensitive for each cat, so we need separate starting values
    powerLawStarValDict = {
        0: {
            'RooSumTwoPowerLawPdf_a1_coeff': -1.9743280291018659, 'RooSumTwoPowerLawPdf_a2_coeff': -5.070611059366495, 'RooSumTwoPowerLawPdf_f_coeff': 0.9169613224723027
           }, 
        1: {
            'RooSumTwoPowerLawPdf_a1_coeff': -2.6281519073027844, 'RooSumTwoPowerLawPdf_a2_coeff': -4.460879015664051, 'RooSumTwoPowerLawPdf_f_coeff': 0.44823019493423466
           }, 
        2: {
            'RooSumTwoPowerLawPdf_a1_coeff': -2.3033162245329164, 'RooSumTwoPowerLawPdf_a2_coeff': -4.183898141656655, 'RooSumTwoPowerLawPdf_f_coeff': 0.40115183986375974
           }, 
        3: {
            'RooSumTwoPowerLawPdf_a1_coeff': -2.4924512293219796, 'RooSumTwoPowerLawPdf_a2_coeff': -3.681762231096663, 'RooSumTwoPowerLawPdf_f_coeff': 0.2902132176929161
           }, 
        4: {
            'RooSumTwoPowerLawPdf_a1_coeff': -1.3892075987583783, 'RooSumTwoPowerLawPdf_a2_coeff': -84.29141727097351, 'RooSumTwoPowerLawPdf_f_coeff': 0.9948389705318498
           }, 
        'all': {'RooSumTwoPowerLawPdf_a1_coeff': -2.1969979862085047, 'RooSumTwoPowerLawPdf_a2_coeff': -4.18629756373829, 'RooSumTwoPowerLawPdf_f_coeff': 0.6569304312671497}
    }

    fewzxbern_init_vals = {
        "fewz_bernstein_a1": 1.5,
        "fewz_bernstein_a2": 0.75,
        "fewz_bernstein_a3": 0.75,
    }
    landauxbern_init_vals = {
        "landau_a_coeff": 0.258087,
        "landau_bernstein_a1": 1.5,
        "landau_bernstein_a2": 0.75,
    }
    bwzxbern_init_vals = {
        "BWZxBern_a_coeff": -0.02,
        "bwz_bernstein_a0": 0.3,
        "bwz_bernstein_a1": 0.3,
    }
    bwzgamma_init_vals = {
        "bwzgamma_BWZ_a_coeff": -0.02,
        "bwzgamma_Gamma_a_coeff": -0.00005,
        "bwzgamma_frac": 0.5,
    }
    poly_init_vals = {
        "a0": -0.02,
        "a1": -0.02,
        "a2": -0.02,
    }
    
    all_params = []
    
    for ix in range(nSubCats):
        # BWZRedux
        pdf_bwz, params_bwz = create_core_pdf("BWZRedux", ix, mass, bwz_init_vals)
        coreFunction_dict["BWZRedux"].append(pdf_bwz)
        all_params.extend(params_bwz)

        # sumExp
        pdf_sumexp, params_sumexp = create_core_pdf("sumExp", ix, mass, sumexp_init_vals)
        coreFunction_dict["sumExp"].append(pdf_sumexp)
        all_params.extend(params_sumexp)

        # PowerLaw
        powerlaw_init_vals = powerLawStarValDict[ix]
        pdf, params = create_core_pdf("PowerLaw", ix, mass, powerlaw_init_vals)
        coreFunction_dict["PowerLaw"].append(pdf)
        all_params.extend(params)

        # FEWZxBern
        pdf, params = create_core_pdf("FEWZxBern", ix, mass, fewzxbern_init_vals)
        coreFunction_dict["FEWZxBern"].append(pdf)
        all_params.extend(params)

        # LandauxBern
        pdf, params = create_core_pdf("LandauxBern", ix, mass, landauxbern_init_vals)
        coreFunction_dict["LandauxBern"].append(pdf)
        all_params.extend(params)

        # BWZxBern
        pdf, params = create_core_pdf("BWZxBern", ix, mass, bwzxbern_init_vals)
        coreFunction_dict["BWZxBern"].append(pdf)
        all_params.extend(params)
        
        # BwzGamma
        pdf, params = create_core_pdf("BwzGamma", ix, mass, bwzgamma_init_vals)
        coreFunction_dict["BwzGamma"].append(pdf)
        all_params.extend(params)

        # Polynomial
        pdf, params = create_core_pdf("Polynomial", ix, mass, poly_init_vals)
        coreFunction_dict["Polynomial"].append(pdf)
        all_params.extend(params)


    # print(f"all_params b4 fitting: {[param.Print() for param in all_params]}")
    # print(f"coreFunction_dict: {coreFunction_dict}")
    # raise ValueError
    # ---------------------------------------------------------------
    # Extract Data over all sub cats
    # ---------------------------------------------------------------

    # also do for all subcats for later use
    allCat_mass_arr = processed_eventsData.dimuon_mass
    allCat_mass_arr  = ak.to_numpy(allCat_mass_arr) # convert to numpy for ROOT.RooDataSet
    roo_datasetData = ROOT.RooDataSet.from_numpy({mass_name: allCat_mass_arr}, [mass])
    roo_histData_allCat = ROOT.RooDataHist("allCat_rooHist","allCat_rooHist", ROOT.RooArgSet(mass), roo_datasetData)
    
    # ---------------------------------------------------------------
    # Initialize Data for Bkg models to fit to
    # ---------------------------------------------------------------
     
    # do for cat idx 0
    subCat_filter = (processed_eventsData[subCatIdx_name] == 0)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for ROOT.RooDataSet
    roo_datasetData_subCat0 = ROOT.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat0 = ROOT.RooDataHist("subCat0_rooHist","subCat0_rooHist", ROOT.RooArgSet(mass), roo_datasetData_subCat0)

    # do for cat idx 1
    subCat_filter = (processed_eventsData[subCatIdx_name] == 1)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for ROOT.RooDataSet
    roo_datasetData_subCat1 = ROOT.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat1 = ROOT.RooDataHist("subCat1_rooHist","subCat1_rooHist", ROOT.RooArgSet(mass), roo_datasetData_subCat1)

    # do for cat idx 2
    subCat_filter = (processed_eventsData[subCatIdx_name] == 2)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for ROOT.RooDataSet
    roo_datasetData_subCat2 = ROOT.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat2 = ROOT.RooDataHist("subCat2_rooHist","subCat2_rooHist", ROOT.RooArgSet(mass), roo_datasetData_subCat2)

    # do for cat idx 3
    subCat_filter = (processed_eventsData[subCatIdx_name] == 3)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for ROOT.RooDataSet
    roo_datasetData_subCat3 = ROOT.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat3 = ROOT.RooDataHist("subCat3_rooHist","subCat3_rooHist", ROOT.RooArgSet(mass), roo_datasetData_subCat3)

    # do for cat idx 4
    subCat_filter = (processed_eventsData[subCatIdx_name] == 4)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for ROOT.RooDataSet
    roo_datasetData_subCat4 = ROOT.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat4 = ROOT.RooDataHist("subCat4_rooHist","subCat4_rooHist", ROOT.RooArgSet(mass), roo_datasetData_subCat4)



    
    
    #----------------------------------------------------------------------------
    # Do fit to the core function
    # ---------------------------------------------------------------------------
    data_histSubCat_l = [
        roo_histData_subCat0,
        roo_histData_subCat1,
        roo_histData_subCat2,
        roo_histData_subCat3,
        roo_histData_subCat4,
    ]

    df_rows = []
    nFit_params = 3
    # fit FEWZxBern separately
    for ix in range(nSubCats):
        data_histSubCat = data_histSubCat_l[ix]
        frame = mass.frame()
        hist_name = data_histSubCat.GetName()
        # ploton for chi2
        data_histSubCat.plotOn(frame, Name=hist_name,)
        

        
        for core_func_name, core_func_l in coreFunction_dict.items():
            print(f"core_func_name: {core_func_name}")
            print(f"core_func_l: {core_func_l}")
            
            core_func = core_func_l[ix]
            _ = core_func.fitTo(data_histSubCat, ROOT.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,SumW2Error=True)
            fitResult = core_func.fitTo(data_histSubCat, ROOT.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,SumW2Error=True)

            # ploton for chi2
            core_func.plotOn(frame, DataError="SumW2", Name=core_func_name)
            chi2ndf = frame.chiSquare(core_func_name, hist_name, nFit_params)
            print(f"chi^2 {core_func_name} = ", chi2ndf)
            df_rows.append({
                "BDT cat": ix,
                "fit function": core_func_name,
                "chi2ndf": chi2ndf
            })

            
            print(f"subCat index: {ix}")
            fitResult.Print("v")

    df = pd.DataFrame(df_rows, columns=["BDT cat", "fit function", "chi2ndf"])
    df.to_csv(f"{base_path}/test_chi2ndf.csv", index=False)
    summary_df = (
        df.groupby("fit function", as_index=False)
        .agg(
            mean_chi2ndf=("chi2ndf", "mean"),
            max_chi2ndf=("chi2ndf", "max"),
            min_chi2ndf=("chi2ndf", "min"),
        )
    )
    summary_df["func_index"] = summary_df["fit function"].map({name: idx for idx, name in enumerate(FUNC_ORDER)})
    summary_df = summary_df.sort_values("func_index").reset_index(drop=True)
    summary_df.to_csv(f"{base_path}/truth_function_summary.csv", index=False)

    selected_summary = summary_df[summary_df["max_chi2ndf"] <= args.max_chi2ndf].copy()
    if selected_summary.empty:
        selected_summary = summary_df.nsmallest(3, "mean_chi2ndf").copy()
        print(
            f"[bias_test] No truth functions passed max chi2/ndf <= {args.max_chi2ndf:.3f}. "
            "Falling back to the three best average sideband fits."
        )

    selected_indices = selected_summary["func_index"].astype(int).tolist()
    selected_names = selected_summary["fit function"].tolist()
    with open(f"{base_path}/selected_truth_function_indices.txt", "w") as handle:
        handle.write(" ".join(str(idx) for idx in selected_indices))
        handle.write("\n")
    with open(f"{base_path}/selected_truth_function_names.txt", "w") as handle:
        handle.write("\n".join(selected_names))
        handle.write("\n")
    print(f"[bias_test] Selected truth-function candidates: {selected_names}")
    # raise ValueError
    # print(f"all_params after fitting: {[param.Print() for param in all_params]}")
    print(f"coreFunction_dict: {coreFunction_dict}")
    
    # print("Success!")
    # raise ValueError
    
    # ---------------------------------------------------
    # Group the functions into one Envelope
    # ---------------------------------------------------


    
    # // Make a RooMultiPdf object. The order of the pdfs will be the order of their index, ie for below
    # // 0 == BWZRedux
    # // 1 == BwzGamma
    # // 2 == BWZxBern
    # // 3 == sumExp
    # // 4 == PowerLaw
    # // 5 == FEWZxBern
    # // 6 == LandauxBern
    # // 7 == Polynomial
    
    # FEWZxBern Sumexp is less dependent to dimuon mass as stated in line 1585 of RERECO AN
    # I suppose BWZredux is there bc it's the one function with overall least bias (which is why BWZredux is used if CORE-PDF is not used)
    func_order = FUNC_ORDER
    env_pdfs = []
    norm_l = []
    # cat_index = ROOT.RooCategory(f"pdf_index_ggh_cat{cat_ix}","Index of Pdf which is active");
    for cat_ix in range(nSubCats):
        env_pdf, params = getEnvelope(coreFunction_dict, cat_ix, func_order)
        env_pdfs.append(env_pdf)
        all_params.extend(params)

        nevents = data_histSubCat_l[cat_ix].sumEntries()
        env_norm = ROOT.RooRealVar(env_pdf.GetName()+"_norm","Background normalization value",nevents,0,3*nevents) # free floating value
        norm_l.append(env_norm)
        

    print(f"env_pdfs: {env_pdfs}")
    print(f"env_pdfs: {len(env_pdfs)}")

    print("Success!")
    # raise ValueError


    # ---------------------------------------------------
    # Obtain signal MC events
    # ---------------------------------------------------

    # load_path = f"{args.load_path}/{category}/{args.year}/processed_events_signalMC.parquet"
    if args.year=="all":
        load_path = f"{args.load_path}/{category}/*/processed_events_sigMC_ggh_*.parquet"
        # load_path = f"{args.load_path}/{category}/*/processed_events_sigMC_ggh_amcPS.parquet"
    elif args.year=="2016only":
        load_path = f"{args.load_path}/{category}/2016*/processed_events_sigMC_ggh_*.parquet"
    else:
        load_path = f"{args.load_path}/{category}/{args.year}/processed_events_sigMC_ggh_*.parquet" # Fig 6.15 was only with ggH process, though with all 2016, 2017 and 2018
    # load_path = f"{args.load_path}/{category}/{args.year}/processed_events_sigMC*.parquet"
    if args.year=="all":
        load_path = f"{args.load_path}/*/processed_events_sigMC_ggh_*.parquet"
        # load_path = f"{args.load_path}/{category}/*/processed_events_sigMC_ggh_amcPS.parquet"
    elif args.year=="2016only":
        load_path = f"{args.load_path}/2016*/processed_events_sigMC_ggh_*.parquet"
    else:
        load_path = f"{args.load_path}/{args.year}/processed_events_sigMC_ggh_*.parquet"
    processed_eventsSignalMC = dak.from_parquet(load_path).compute()
    print(f"ggH yield: {np.sum(processed_eventsSignalMC.wgt_nominal)}")
    print("signal events loaded")
    
    # ---------------------------------------------------
    # Define signal model's Doubcl Crystal Ball PDF
    # ---------------------------------------------------
    
    # subCat 0
    # original start ------------------------------------------------------
    # MH_subCat0 = ROOT.RooRealVar("MH" , "MH", 125, 115,135)
    # MH_subCat0.setConstant(True) # this shouldn't change, I think
    # original end ------------------------------------------------------
    # MH_subCat0 = ROOT.RooRealVar("MH" , "MH", 124.805, 120,130) # matching AN
    # MH_subCat0 = ROOT.RooRealVar("MH" , "MH", 124.805, 124,126)
    MH_subCat0 = ROOT.RooRealVar("MH" , "MH", 125) # make this frozen
    
    # sigma_subCat0 = ROOT.RooRealVar("sigma_subCat0" , "sigma_subCat0", 2, .1, 4.0)
    # alpha1_subCat0 = ROOT.RooRealVar("alpha1_subCat0" , "alpha1_subCat0", 2, 0.01, 65)
    # n1_subCat0 = ROOT.RooRealVar("n1_subCat0" , "n1_subCat0", 10, 0.01, 100)
    # alpha2_subCat0 = ROOT.RooRealVar("alpha2_subCat0" , "alpha2_subCat0", 2.0, 0.01, 65)
    # n2_subCat0 = ROOT.RooRealVar("n2_subCat0" , "n2_subCat0", 25, 0.01, 100)

    # copying parameters from official AN workspace as starting params
    sigma_subCat0 = ROOT.RooRealVar("sigma_subCat0" , "sigma_subCat0", 1.8228, .1, 4.0)
    alpha1_subCat0 = ROOT.RooRealVar("alpha1_subCat0" , "alpha1_subCat0", 1.12842, 0.01, 65)
    n1_subCat0 = ROOT.RooRealVar("n1_subCat0" , "n1_subCat0", 4.019960, 0.01, 100)
    alpha2_subCat0 = ROOT.RooRealVar("alpha2_subCat0" , "alpha2_subCat0", 1.3132, 0.01, 65)
    n2_subCat0 = ROOT.RooRealVar("n2_subCat0" , "n2_subCat0", 9.97411, 0.01, 100)

    # # temporary test
    # sigma_subCat0.setConstant(True)
    # alpha1_subCat0.setConstant(True)
    # n1_subCat0.setConstant(True)
    # alpha2_subCat0.setConstant(True)
    # n2_subCat0.setConstant(True)
    
    
    CMS_hmm_sigma_cat0_ggh = ROOT.RooRealVar("CMS_hmm_sigma_cat0_ggh" , "CMS_hmm_sigma_cat0_ggh", 0, -5 , 5 )
    CMS_hmm_sigma_cat0_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat0_ggh_fsigma = ROOT.RooFormulaVar("ggH_cat0_ggh_fsigma", "ggH_cat0_ggh_fsigma",'@0*(1+@1)',[sigma_subCat0, CMS_hmm_sigma_cat0_ggh])
    CMS_hmm_peak_cat0_ggh = ROOT.RooRealVar("CMS_hmm_peak_cat0_ggh" , "CMS_hmm_peak_cat0_ggh", 0, -5 , 5 )
    CMS_hmm_peak_cat0_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat0_ggh_fpeak = ROOT.RooFormulaVar("ggH_cat0_ggh_fpeak", "ggH_cat0_ggh_fpeak",'@0*(1+@1)',[MH_subCat0, CMS_hmm_peak_cat0_ggh])
    
    # n1_subCat0.setConstant(True) # freeze for stability
    # n2_subCat0.setConstant(True) # freeze for stability
    name = "signal_subCat0"
    signal_subCat0 = ROOT.RooCrystalBall(name,name,mass, ggH_cat0_ggh_fpeak, ggH_cat0_ggh_fsigma, alpha1_subCat0, n1_subCat0, alpha2_subCat0, n2_subCat0)

    # subCat 1
    # original start ------------------------------------------------------
    # MH_subCat1 = ROOT.RooRealVar("MH" , "MH", 125, 115,135)
    # MH_subCat1.setConstant(True) # this shouldn't change, I think
    # original end ------------------------------------------------------
    MH_subCat1 = MH_subCat0 
    
    # sigma_subCat1 = ROOT.RooRealVar("sigma_subCat1" , "sigma_subCat1", 2, .1, 4.0)
    # alpha1_subCat1 = ROOT.RooRealVar("alpha1_subCat1" , "alpha1_subCat1", 2, 0.01, 65)
    # n1_subCat1 = ROOT.RooRealVar("n1_subCat1" , "n1_subCat1", 10, 0.01, 100)
    # alpha2_subCat1 = ROOT.RooRealVar("alpha2_subCat1" , "alpha2_subCat1", 2.0, 0.01, 65)
    # n2_subCat1 = ROOT.RooRealVar("n2_subCat1" , "n2_subCat1", 25, 0.01, 100)

    # copying parameters from official AN workspace as starting params
    sigma_subCat1 = ROOT.RooRealVar("sigma_subCat1" , "sigma_subCat1", 1.503280, .1, 4.0)
    alpha1_subCat1 = ROOT.RooRealVar("alpha1_subCat1" , "alpha1_subCat1", 1.3364, 0.01, 65)
    n1_subCat1 = ROOT.RooRealVar("n1_subCat1" , "n1_subCat1", 2.815022, 0.01, 100)
    alpha2_subCat1 = ROOT.RooRealVar("alpha2_subCat1" , "alpha2_subCat1", 1.57127749, 0.01, 65)
    n2_subCat1 = ROOT.RooRealVar("n2_subCat1" , "n2_subCat1", 9.99687, 0.01, 100)

    # # temporary test
    # sigma_subCat1.setConstant(True)
    # alpha1_subCat1.setConstant(True)
    # n1_subCat1.setConstant(True)
    # alpha2_subCat1.setConstant(True)
    # n2_subCat1.setConstant(True)
    
    CMS_hmm_sigma_cat1_ggh = ROOT.RooRealVar("CMS_hmm_sigma_cat1_ggh" , "CMS_hmm_sigma_cat1_ggh", 0, -5 , 5 )
    CMS_hmm_sigma_cat1_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat1_ggh_fsigma = ROOT.RooFormulaVar("ggH_cat1_ggh_fsigma", "ggH_cat1_ggh_fsigma",'@0*(1+@1)',[sigma_subCat1, CMS_hmm_sigma_cat1_ggh])
    CMS_hmm_peak_cat1_ggh = ROOT.RooRealVar("CMS_hmm_peak_cat1_ggh" , "CMS_hmm_peak_cat1_ggh", 0, -5 , 5 )
    CMS_hmm_peak_cat1_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat1_ggh_fpeak = ROOT.RooFormulaVar("ggH_cat1_ggh_fpeak", "ggH_cat1_ggh_fpeak",'@0*(1+@1)',[MH_subCat1, CMS_hmm_peak_cat1_ggh])
    
    # n1_subCat1.setConstant(True) # freeze for stability
    # n2_subCat1.setConstant(True) # freeze for stability
    name = "signal_subCat1"
    signal_subCat1 = ROOT.RooCrystalBall(name,name,mass, ggH_cat1_ggh_fpeak, ggH_cat1_ggh_fsigma, alpha1_subCat1, n1_subCat1, alpha2_subCat1, n2_subCat1)

    # subCat 2
    # original start ------------------------------------------------------
    # MH_subCat2 = ROOT.RooRealVar("MH" , "MH", 125, 115,135)
    # MH_subCat2.setConstant(True) # this shouldn't change, I think
    # original end ------------------------------------------------------
    MH_subCat2 = MH_subCat0 
    
    # sigma_subCat2 = ROOT.RooRealVar("sigma_subCat2" , "sigma_subCat2", 2, .1, 4.0)
    # alpha1_subCat2 = ROOT.RooRealVar("alpha1_subCat2" , "alpha1_subCat2", 2, 0.01, 65)
    # n1_subCat2 = ROOT.RooRealVar("n1_subCat2" , "n1_subCat2", 10, 0.01, 100)
    # alpha2_subCat2 = ROOT.RooRealVar("alpha2_subCat2" , "alpha2_subCat2", 2.0, 0.01, 65)
    # n2_subCat2 = ROOT.RooRealVar("n2_subCat2" , "n2_subCat2", 25, 0.01, 100)

    # copying parameters from official AN workspace as starting params
    sigma_subCat2 = ROOT.RooRealVar("sigma_subCat2" , "sigma_subCat2", 1.36025, .1, 4.0)
    alpha1_subCat2 = ROOT.RooRealVar("alpha1_subCat2" , "alpha1_subCat2", 1.4173626, 0.01, 65)
    n1_subCat2 = ROOT.RooRealVar("n1_subCat2" , "n1_subCat2", 2.42748, 0.01, 100)
    alpha2_subCat2 = ROOT.RooRealVar("alpha2_subCat2" , "alpha2_subCat2", 1.629120, 0.01, 65)
    n2_subCat2 = ROOT.RooRealVar("n2_subCat2" , "n2_subCat2", 9.983334, 0.01, 100)

    # # temporary test
    # sigma_subCat2.setConstant(True)
    # alpha1_subCat2.setConstant(True)
    # n1_subCat2.setConstant(True)
    # alpha2_subCat2.setConstant(True)
    # n2_subCat2.setConstant(True)

    CMS_hmm_sigma_cat2_ggh = ROOT.RooRealVar("CMS_hmm_sigma_cat2_ggh" , "CMS_hmm_sigma_cat2_ggh", 0, -5 , 5 )
    CMS_hmm_sigma_cat2_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat2_ggh_fsigma = ROOT.RooFormulaVar("ggH_cat2_ggh_fsigma", "ggH_cat2_ggh_fsigma",'@0*(1+@1)',[sigma_subCat2, CMS_hmm_sigma_cat2_ggh])
    CMS_hmm_peak_cat2_ggh = ROOT.RooRealVar("CMS_hmm_peak_cat2_ggh" , "CMS_hmm_peak_cat2_ggh", 0, -5 , 5 )
    CMS_hmm_peak_cat2_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat2_ggh_fpeak = ROOT.RooFormulaVar("ggH_cat2_ggh_fpeak", "ggH_cat2_ggh_fpeak",'@0*(1+@1)',[MH_subCat2, CMS_hmm_peak_cat2_ggh])
    
    # n1_subCat2.setConstant(True) # freeze for stability
    # n2_subCat2.setConstant(True) # freeze for stability
    name = "signal_subCat2"
    signal_subCat2 = ROOT.RooCrystalBall(name,name,mass, ggH_cat2_ggh_fpeak, ggH_cat2_ggh_fsigma, alpha1_subCat2, n1_subCat2, alpha2_subCat2, n2_subCat2)

    # subCat 3
    # original start ------------------------------------------------------
    # MH_subCat3 = ROOT.RooRealVar("MH" , "MH", 125, 115,135)
    # MH_subCat3.setConstant(True) # this shouldn't change, I think
    # original end ------------------------------------------------------
    MH_subCat3 = MH_subCat0
    

    sigma_subCat3 = ROOT.RooRealVar("sigma_subCat3" , "sigma_subCat3", 0.1, .1, 10.0)
    alpha1_subCat3 = ROOT.RooRealVar("alpha1_subCat3" , "alpha1_subCat3", 2, 0.01, 200)
    n1_subCat3 = ROOT.RooRealVar("n1_subCat3" , "n1_subCat3", 25, 0.01, 200)
    alpha2_subCat3 = ROOT.RooRealVar("alpha2_subCat3" , "alpha2_subCat3", 2, 0.01, 65)
    n2_subCat3 = ROOT.RooRealVar("n2_subCat3" , "n2_subCat3", 25, 0.01, 200)

    # # copying parameters from official AN workspace as starting params
    # sigma_subCat3 = ROOT.RooRealVar("sigma_subCat3" , "sigma_subCat3", 1.25359, .1, 10.0)
    # alpha1_subCat3 = ROOT.RooRealVar("alpha1_subCat3" , "alpha1_subCat3", 1.4199, 0.01, 200)
    # n1_subCat3 = ROOT.RooRealVar("n1_subCat3" , "n1_subCat3", 2.409953, 0.01, 200)
    # alpha2_subCat3 = ROOT.RooRealVar("alpha2_subCat3" , "alpha2_subCat3", 1.64675, 0.01, 65)
    # n2_subCat3 = ROOT.RooRealVar("n2_subCat3" , "n2_subCat3", 9.670221, 0.01, 200)

    # # temporary test
    # sigma_subCat3.setConstant(True)
    # alpha1_subCat3.setConstant(True)
    # n1_subCat3.setConstant(True)
    # alpha2_subCat3.setConstant(True)
    # n2_subCat3.setConstant(True)

    CMS_hmm_sigma_cat3_ggh = ROOT.RooRealVar("CMS_hmm_sigma_cat3_ggh" , "CMS_hmm_sigma_cat3_ggh", 0, -5 , 5 )
    CMS_hmm_sigma_cat3_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat3_ggh_fsigma = ROOT.RooFormulaVar("ggH_cat3_ggh_fsigma", "ggH_cat3_ggh_fsigma",'@0*(1+@1)',[sigma_subCat3, CMS_hmm_sigma_cat3_ggh])
    CMS_hmm_peak_cat3_ggh = ROOT.RooRealVar("CMS_hmm_peak_cat3_ggh" , "CMS_hmm_peak_cat3_ggh", 0, -5 , 5 )
    CMS_hmm_peak_cat3_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat3_ggh_fpeak = ROOT.RooFormulaVar("ggH_cat3_ggh_fpeak", "ggH_cat3_ggh_fpeak",'@0*(1+@1)',[MH_subCat3, CMS_hmm_peak_cat3_ggh])
    
    # n1_subCat3.setConstant(True) # freeze for stability
    # n2_subCat3.setConstant(True) # freeze for stability
    name = "signal_subCat3"
    signal_subCat3 = ROOT.RooCrystalBall(name,name,mass, ggH_cat3_ggh_fpeak, ggH_cat3_ggh_fsigma, alpha1_subCat3, n1_subCat3, alpha2_subCat3, n2_subCat3)

    # subCat 4
    # original start ------------------------------------------------------
    # MH_subCat4 = ROOT.RooRealVar("MH" , "MH", 125, 115,135)
    # MH_subCat4.setConstant(True) # this shouldn't change, I think
    # original end ------------------------------------------------------
    MH_subCat4 = MH_subCat0
    
    # sigma_subCat4 = ROOT.RooRealVar("sigma_subCat4" , "sigma_subCat4", 2, .1, 4.0)
    # alpha1_subCat4 = ROOT.RooRealVar("alpha1_subCat4" , "alpha1_subCat4", 2, 0.01, 65)
    # n1_subCat4 = ROOT.RooRealVar("n1_subCat4" , "n1_subCat4", 10, 0.01, 100)
    # alpha2_subCat4 = ROOT.RooRealVar("alpha2_subCat4" , "alpha2_subCat4", 2.0, 0.01, 65)
    # n2_subCat4 = ROOT.RooRealVar("n2_subCat4" , "n2_subCat4", 25, 0.01, 100)

    # copying parameters from official AN workspace as starting params
    sigma_subCat4 = ROOT.RooRealVar("sigma_subCat4" , "sigma_subCat4", 1.28250, .1, 4.0)
    alpha1_subCat4 = ROOT.RooRealVar("alpha1_subCat4" , "alpha1_subCat4", 1.47936, 0.01, 65)
    n1_subCat4 = ROOT.RooRealVar("n1_subCat4" , "n1_subCat4", 2.24104, 0.01, 100)
    alpha2_subCat4 = ROOT.RooRealVar("alpha2_subCat4" , "alpha2_subCat4", 1.67898, 0.01, 65)
    n2_subCat4 = ROOT.RooRealVar("n2_subCat4" , "n2_subCat4", 8.8719, 0.01, 100)

    # # temporary test
    # sigma_subCat4.setConstant(True)
    # alpha1_subCat4.setConstant(True)
    # n1_subCat4.setConstant(True)
    # alpha2_subCat4.setConstant(True)
    # n2_subCat4.setConstant(True)

    CMS_hmm_sigma_cat4_ggh = ROOT.RooRealVar("CMS_hmm_sigma_cat4_ggh" , "CMS_hmm_sigma_cat4_ggh", 0, -5 , 5 )
    CMS_hmm_sigma_cat4_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat4_ggh_fsigma = ROOT.RooFormulaVar("ggH_cat4_ggh_fsigma", "ggH_cat4_ggh_fsigma",'@0*(1+@1)',[sigma_subCat4, CMS_hmm_sigma_cat4_ggh])
    CMS_hmm_peak_cat4_ggh = ROOT.RooRealVar("CMS_hmm_peak_cat4_ggh" , "CMS_hmm_peak_cat4_ggh", 0, -5 , 5 )
    CMS_hmm_peak_cat4_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat4_ggh_fpeak = ROOT.RooFormulaVar("ggH_cat4_ggh_fpeak", "ggH_cat4_ggh_fpeak",'@0*(1+@1)',[MH_subCat4, CMS_hmm_peak_cat4_ggh])
    
    # n1_subCat4.setConstant(True) # freeze for stability
    # n2_subCat4.setConstant(True) # freeze for stability
    name = "signal_subCat4"
    signal_subCat4 = ROOT.RooCrystalBall(name,name,mass, ggH_cat4_ggh_fpeak, ggH_cat4_ggh_fsigma, alpha1_subCat4, n1_subCat4, alpha2_subCat4, n2_subCat4)
    
    
    # ---------------------------------------------------
    # Define signal MC samples to fit to for ggH
    # ---------------------------------------------------

    # subCat 0
    subCat_filter = (processed_eventsSignalMC[subCatIdx_name] == 0)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat0_SigMC = ak.to_numpy(
        processed_eventsSignalMC.wgt_nominal[subCat_filter]
    ) # weights

    # generate a weighted histogram 
    roo_histData_subCat0_signal = ROOT.TH1F("subCat0_rooHist_signal", "subCat0_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat0_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat0_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat0_signal = ROOT.RooDataHist("subCat0_rooHist_signal", "subCat0_rooHist_signal", ROOT.RooArgSet(mass), roo_histData_subCat0_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat0_signal = roo_histData_subCat0_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat0"],
        "dataset": ["ggH"], 
        "yield": [data_subCat0_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)

    # define normalization value from signal MC event weights 
    flat_MC_SF = 1.00
    # flat_MC_SF = 0.92 # temporary flat SF to match my Data/MC agreement to that of AN's
    norm_val = np.sum(wgt_subCat0_SigMC)* flat_MC_SF 
    # norm_val = 254.528077 # quick test
    sig_norm_subCat0 = ROOT.RooRealVar(signal_subCat0.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat0 norm_val: {norm_val}")
    sig_norm_subCat0.setConstant(True)

    # subCat 1
    subCat_filter = (processed_eventsSignalMC[subCatIdx_name] == 1)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat1_SigMC = ak.to_numpy(
        processed_eventsSignalMC.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat1_signal = ROOT.TH1F("subCat1_rooHist_signal", "subCat1_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat1_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat1_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat1_signal = ROOT.RooDataHist("subCat1_rooHist_signal", "subCat1_rooHist_signal", ROOT.RooArgSet(mass), roo_histData_subCat1_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat1_signal = roo_histData_subCat1_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat1"],
        "dataset": ["ggH"], 
        "yield": [data_subCat1_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)
    

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat1_SigMC)* flat_MC_SF
    # norm_val = 295.214 # quick test
    sig_norm_subCat1 = ROOT.RooRealVar(signal_subCat1.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat1 norm_val: {norm_val}")
    sig_norm_subCat1.setConstant(True)

    # subCat 2
    subCat_filter = (processed_eventsSignalMC[subCatIdx_name] == 2)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat2_SigMC = ak.to_numpy(
        processed_eventsSignalMC.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat2_signal = ROOT.TH1F("subCat2_rooHist_signal", "subCat2_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat2_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat2_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat2_signal = ROOT.RooDataHist("subCat2_rooHist_signal", "subCat2_rooHist_signal", ROOT.RooArgSet(mass), roo_histData_subCat2_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat2_signal = roo_histData_subCat2_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat2"],
        "dataset": ["ggH"], 
        "yield": [data_subCat2_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat2_SigMC) * flat_MC_SF
    # norm_val = 124.0364 # quick test
    sig_norm_subCat2 = ROOT.RooRealVar(signal_subCat2.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat2 norm_val: {norm_val}")
    sig_norm_subCat2.setConstant(True)

    # subCat 3
    subCat_filter = (processed_eventsSignalMC[subCatIdx_name] == 3)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat3_SigMC = ak.to_numpy(
        processed_eventsSignalMC.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat3_signal = ROOT.TH1F("subCat3_rooHist_signal", "subCat3_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat3_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat3_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat3_signal = ROOT.RooDataHist("subCat3_rooHist_signal", "subCat3_rooHist_signal", ROOT.RooArgSet(mass), roo_histData_subCat3_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat3_signal = roo_histData_subCat3_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat3"],
        "dataset": ["ggH"], 
        "yield": [data_subCat3_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat3_SigMC)* flat_MC_SF
    # norm_val = 116.4918 # quick test
    sig_norm_subCat3 = ROOT.RooRealVar(signal_subCat3.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat3 norm_val: {norm_val}")
    sig_norm_subCat3.setConstant(True)
    
    # subCat 4
    subCat_filter = (processed_eventsSignalMC[subCatIdx_name] == 4)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat4_SigMC = ak.to_numpy(
        processed_eventsSignalMC.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat4_signal = ROOT.TH1F("subCat4_rooHist_signal", "subCat4_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat4_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat4_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat4_signal = ROOT.RooDataHist("subCat4_rooHist_signal", "subCat4_rooHist_signal", ROOT.RooArgSet(mass), roo_histData_subCat4_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat4_signal = roo_histData_subCat4_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat4"],
        "dataset": ["ggH"], 
        "yield": [data_subCat4_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)
    print(f"yield_df after ggH: {yield_df}")

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat4_SigMC)* flat_MC_SF
    # norm_val = 45.423052 # quick test
    sig_norm_subCat4 = ROOT.RooRealVar(signal_subCat4.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat4 norm_val: {norm_val}")
    sig_norm_subCat4.setConstant(True)
    
    # ---------------------------------------------------
    # Fit signal model simultaneously. Sigma, and left and right tails are different for each category
    # ---------------------------------------------------

    

    # subCat 0
    _ = signal_subCat0.fitTo(data_subCat0_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat0.fitTo(data_subCat0_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat0.setConstant(True)
    alpha1_subCat0.setConstant(True)
    n1_subCat0.setConstant(True)
    alpha2_subCat0.setConstant(True)
    n2_subCat0.setConstant(True)

    

    # subCat 1
    _ = signal_subCat1.fitTo(data_subCat1_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat1.fitTo(data_subCat1_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat1.setConstant(True)
    alpha1_subCat1.setConstant(True)
    n1_subCat1.setConstant(True)
    alpha2_subCat1.setConstant(True)
    n2_subCat1.setConstant(True)

    

    # subCat 2
    _ = signal_subCat2.fitTo(data_subCat2_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat2.fitTo(data_subCat2_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat2.setConstant(True)
    alpha1_subCat2.setConstant(True)
    n1_subCat2.setConstant(True)
    alpha2_subCat2.setConstant(True)
    n2_subCat2.setConstant(True)

    
    
    # subCat 3
    _ = signal_subCat3.fitTo(data_subCat3_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat3.fitTo(data_subCat3_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat3.setConstant(True)
    alpha1_subCat3.setConstant(True)
    n1_subCat3.setConstant(True)
    alpha2_subCat3.setConstant(True)
    n2_subCat3.setConstant(True)
    # sigma_subCat3.setConstant(False)
    # alpha1_subCat3.setConstant(False)
    # n1_subCat3.setConstant(False)
    # alpha2_subCat3.setConstant(False)
    # n2_subCat3.setConstant(False)


    # subCat 4
    _ = signal_subCat4.fitTo(data_subCat4_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat4.fitTo(data_subCat4_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat4.setConstant(True)
    alpha1_subCat4.setConstant(True)
    n1_subCat4.setConstant(True)
    alpha2_subCat4.setConstant(True)
    n2_subCat4.setConstant(True)

    # ---------------------------------------------------
    # Obtain signal MC events for VBF
    # ---------------------------------------------------

    # load_path = f"{args.load_path}/{category}/{args.year}/processed_events_signalMC.parquet"
    if args.year=="all":
        load_path = f"{args.load_path}/*/processed_events_sigMC_vbf_*.parquet"
        # load_path = f"{args.load_path}/{category}/*/processed_events_sigMC_qqh_amcPS.parquet"
    elif args.year=="2016only":
        load_path = f"{args.load_path}/2016*/processed_events_sigMC_vbf_*.parquet"
    else:
        load_path = f"{args.load_path}/{args.year}/processed_events_sigMC_vbf_*.parquet" # Fig 6.15 was only with qqH process, though with all 2016, 2017 and 2018

    print(load_path)
    load_path = glob.glob(load_path)
    
    load_path = [s for s in load_path if "/2024" not in s] # FIXME : temporarily remove 2024 vbf parquet bc it's corrupt
    print(load_path)
    processed_eventsSignalMC_vbf = dak.from_parquet(load_path).compute()
    print(f"qqH yield: {np.sum(processed_eventsSignalMC_vbf.wgt_nominal)}")
    print("signal events loaded")
    
    # ---------------------------------------------------
    # Define vbf signal model's Doubcl Crystal Ball PDF
    # ---------------------------------------------------
    
    # subCat 0
    
    sigma_subCat0_vbf = ROOT.RooRealVar("sigma_subCat0_vbf" , "sigma_subCat0_vbf", 2, .1, 4.0)
    alpha1_subCat0_vbf = ROOT.RooRealVar("alpha1_subCat0_vbf" , "alpha1_subCat0_vbf", 2, 0.01, 65)
    n1_subCat0_vbf = ROOT.RooRealVar("n1_subCat0_vbf" , "n1_subCat0_vbf", 10, 0.01, 100)
    alpha2_subCat0_vbf = ROOT.RooRealVar("alpha2_subCat0_vbf" , "alpha2_subCat0_vbf", 2.0, 0.01, 65)
    n2_subCat0_vbf = ROOT.RooRealVar("n2_subCat0_vbf" , "n2_subCat0_vbf", 25, 0.01, 100)

    # # temporary test
    # sigma_subCat0_vbf.setConstant(True)
    # alpha1_subCat0_vbf.setConstant(True)
    # n1_subCat0_vbf.setConstant(True)
    # alpha2_subCat0_vbf.setConstant(True)
    # n2_subCat0_vbf.setConstant(True)
    

    qqH_cat0_ggh_fsigma = ROOT.RooFormulaVar("qqH_cat0_ggh_fsigma", "qqH_cat0_ggh_fsigma",'@0*(1+@1)',[sigma_subCat0_vbf, CMS_hmm_sigma_cat0_ggh])
    qqH_cat0_ggh_fpeak = ROOT.RooFormulaVar("qqH_cat0_qqh_fpeak", "qqH_cat0_ggh_fpeak",'@0*(1+@1)',[MH_subCat0, CMS_hmm_peak_cat0_ggh])
    
    # n1_subCat0_vbf.setConstant(True) # freeze for stability
    # n2_subCat0_vbf.setConstant(True) # freeze for stability
    name = "signal_subCat0_vbf"
    signal_subCat0_vbf = ROOT.RooCrystalBall(name,name,mass, qqH_cat0_ggh_fpeak, qqH_cat0_ggh_fsigma, alpha1_subCat0_vbf, n1_subCat0_vbf, alpha2_subCat0_vbf, n2_subCat0_vbf)

    # subCat 1

    
    sigma_subCat1_vbf = ROOT.RooRealVar("sigma_subCat1_vbf" , "sigma_subCat1_vbf", 2, .1, 4.0)
    alpha1_subCat1_vbf = ROOT.RooRealVar("alpha1_subCat1_vbf" , "alpha1_subCat1_vbf", 2, 0.01, 65)
    n1_subCat1_vbf = ROOT.RooRealVar("n1_subCat1_vbf" , "n1_subCat1_vbf", 10, 0.01, 100)
    alpha2_subCat1_vbf = ROOT.RooRealVar("alpha2_subCat1_vbf" , "alpha2_subCat1_vbf", 2.0, 0.01, 65)
    n2_subCat1_vbf = ROOT.RooRealVar("n2_subCat1_vbf" , "n2_subCat1_vbf", 25, 0.01, 100)

    # # temporary test
    # sigma_subCat1_vbf.setConstant(True)
    # alpha1_subCat1_vbf.setConstant(True)
    # n1_subCat1_vbf.setConstant(True)
    # alpha2_subCat1_vbf.setConstant(True)
    # n2_subCat1_vbf.setConstant(True)
    
    qqH_cat1_ggh_fsigma = ROOT.RooFormulaVar("qqH_cat1_ggh_fsigma", "qqH_cat1_ggh_fsigma",'@0*(1+@1)',[sigma_subCat1_vbf, CMS_hmm_sigma_cat1_ggh])
    qqH_cat1_ggh_fpeak = ROOT.RooFormulaVar("qqH_cat1_ggh_fpeak", "qqH_cat1_ggh_fpeak",'@0*(1+@1)',[MH_subCat1, CMS_hmm_peak_cat1_ggh])
    
    # n1_subCat1_vbf.setConstant(True) # freeze for stability
    # n2_subCat1_vbf.setConstant(True) # freeze for stability
    name = "signal_subCat1_vbf"
    signal_subCat1_vbf = ROOT.RooCrystalBall(name,name,mass, qqH_cat1_ggh_fpeak, qqH_cat1_ggh_fsigma, alpha1_subCat1_vbf, n1_subCat1_vbf, alpha2_subCat1_vbf, n2_subCat1_vbf)

    # subCat 2
   
    sigma_subCat2_vbf = ROOT.RooRealVar("sigma_subCat2_vbf" , "sigma_subCat2_vbf", 2, .1, 4.0)
    alpha1_subCat2_vbf = ROOT.RooRealVar("alpha1_subCat2_vbf" , "alpha1_subCat2_vbf", 2, 0.01, 65)
    n1_subCat2_vbf = ROOT.RooRealVar("n1_subCat2_vbf" , "n1_subCat2_vbf", 10, 0.01, 100)
    alpha2_subCat2_vbf = ROOT.RooRealVar("alpha2_subCat2_vbf" , "alpha2_subCat2_vbf", 2.0, 0.01, 65)
    n2_subCat2_vbf = ROOT.RooRealVar("n2_subCat2_vbf" , "n2_subCat2_vbf", 25, 0.01, 100)

    # # temporary test
    # sigma_subCat2_vbf.setConstant(True)
    # alpha1_subCat2_vbf.setConstant(True)
    # n1_subCat2_vbf.setConstant(True)
    # alpha2_subCat2_vbf.setConstant(True)
    # n2_subCat2_vbf.setConstant(True)

    qqH_cat2_ggh_fsigma = ROOT.RooFormulaVar("qqH_cat2_ggh_fsigma", "qqH_cat2_ggh_fsigma",'@0*(1+@1)',[sigma_subCat2_vbf, CMS_hmm_sigma_cat2_ggh])
    qqH_cat2_ggh_fpeak = ROOT.RooFormulaVar("qqH_cat2_ggh_fpeak", "qqH_cat2_ggh_fpeak",'@0*(1+@1)',[MH_subCat2, CMS_hmm_peak_cat2_ggh])
    
    # n1_subCat2_vbf.setConstant(True) # freeze for stability
    # n2_subCat2_vbf.setConstant(True) # freeze for stability
    name = "signal_subCat2_vbf"
    signal_subCat2_vbf = ROOT.RooCrystalBall(name,name,mass, qqH_cat2_ggh_fpeak, qqH_cat2_ggh_fsigma, alpha1_subCat2_vbf, n1_subCat2_vbf, alpha2_subCat2_vbf, n2_subCat2_vbf)

    # subCat 3

    sigma_subCat3_vbf = ROOT.RooRealVar("sigma_subCat3_vbf" , "sigma_subCat3_vbf", 0.1, .1, 10.0)
    alpha1_subCat3_vbf = ROOT.RooRealVar("alpha1_subCat3_vbf" , "alpha1_subCat3_vbf", 2, 0.01, 200)
    n1_subCat3_vbf = ROOT.RooRealVar("n1_subCat3_vbf" , "n1_subCat3_vbf", 25, 0.01, 200)
    alpha2_subCat3_vbf = ROOT.RooRealVar("alpha2_subCat3_vbf" , "alpha2_subCat3_vbf", 2, 0.01, 65)
    n2_subCat3_vbf = ROOT.RooRealVar("n2_subCat3_vbf" , "n2_subCat3_vbf", 25, 0.01, 200)


    # # temporary test
    # sigma_subCat3_vbf.setConstant(True)
    # alpha1_subCat3_vbf.setConstant(True)
    # n1_subCat3_vbf.setConstant(True)
    # alpha2_subCat3_vbf.setConstant(True)
    # n2_subCat3_vbf.setConstant(True)

    qqH_cat3_ggh_fsigma = ROOT.RooFormulaVar("qqH_cat3_ggh_fsigma", "qqH_cat3_ggh_fsigma",'@0*(1+@1)',[sigma_subCat3_vbf, CMS_hmm_sigma_cat3_ggh])
    qqH_cat3_ggh_fpeak = ROOT.RooFormulaVar("qqH_cat3_ggh_fpeak", "qqH_cat3_ggh_fpeak",'@0*(1+@1)',[MH_subCat3, CMS_hmm_peak_cat3_ggh])
    
    # n1_subCat3_vbf.setConstant(True) # freeze for stability
    # n2_subCat3_vbf.setConstant(True) # freeze for stability
    name = "signal_subCat3_vbf"
    signal_subCat3_vbf = ROOT.RooCrystalBall(name,name,mass, qqH_cat3_ggh_fpeak, qqH_cat3_ggh_fsigma, alpha1_subCat3_vbf, n1_subCat3_vbf, alpha2_subCat3_vbf, n2_subCat3_vbf)

    # subCat 4
    
    sigma_subCat4_vbf = ROOT.RooRealVar("sigma_subCat4_vbf" , "sigma_subCat4_vbf", 2, .1, 4.0)
    alpha1_subCat4_vbf = ROOT.RooRealVar("alpha1_subCat4_vbf" , "alpha1_subCat4_vbf", 2, 0.01, 65)
    n1_subCat4_vbf = ROOT.RooRealVar("n1_subCat4_vbf" , "n1_subCat4_vbf", 10, 0.01, 100)
    alpha2_subCat4_vbf = ROOT.RooRealVar("alpha2_subCat4_vbf" , "alpha2_subCat4_vbf", 2.0, 0.01, 65)
    n2_subCat4_vbf = ROOT.RooRealVar("n2_subCat4_vbf" , "n2_subCat4_vbf", 25, 0.01, 100)


    # # temporary test
    # sigma_subCat4_vbf.setConstant(True)
    # alpha1_subCat4_vbf.setConstant(True)
    # n1_subCat4_vbf.setConstant(True)
    # alpha2_subCat4_vbf.setConstant(True)
    # n2_subCat4_vbf.setConstant(True)

    qqH_cat4_ggh_fsigma = ROOT.RooFormulaVar("qqH_cat4_ggh_fsigma", "qqH_cat4_ggh_fsigma",'@0*(1+@1)',[sigma_subCat4_vbf, CMS_hmm_sigma_cat4_ggh])
    qqH_cat4_ggh_fpeak = ROOT.RooFormulaVar("qqH_cat4_ggh_fpeak", "qqH_cat4_ggh_fpeak",'@0*(1+@1)',[MH_subCat4, CMS_hmm_peak_cat4_ggh])
    
    # n1_subCat4_vbf.setConstant(True) # freeze for stability
    # n2_subCat4_vbf.setConstant(True) # freeze for stability
    name = "signal_subCat4_vbf"
    signal_subCat4_vbf = ROOT.RooCrystalBall(name,name,mass, qqH_cat4_ggh_fpeak, qqH_cat4_ggh_fsigma, alpha1_subCat4_vbf, n1_subCat4_vbf, alpha2_subCat4_vbf, n2_subCat4_vbf)
    
    
    # ---------------------------------------------------
    # Define signal MC samples to fit to for qqH
    # ---------------------------------------------------

    # subCat 0
    subCat_filter = (processed_eventsSignalMC_vbf[subCatIdx_name] == 0)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC_vbf.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat0_vbf_SigMC = ak.to_numpy(
        processed_eventsSignalMC_vbf.wgt_nominal[subCat_filter]
    ) # weights

    # generate a weighted histogram 
    roo_histData_subCat0_vbf_signal = ROOT.TH1F("subCat0_vbf_rooHist_signal", "subCat0_vbf_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat0_vbf_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat0_vbf_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat0_vbf_signal = ROOT.RooDataHist("subCat0_vbf_rooHist_signal", "subCat0_vbf_rooHist_signal", ROOT.RooArgSet(mass), roo_histData_subCat0_vbf_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat0_vbf_signal = roo_histData_subCat0_vbf_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat0"],
        "dataset": ["VBF"], 
        "yield": [data_subCat0_vbf_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)

    # define normalization value from signal MC event weights 
    flat_MC_SF = 1.00
    # flat_MC_SF = 0.92 # temporary flat SF to match my Data/MC agreement to that of AN's
    norm_val = np.sum(wgt_subCat0_vbf_SigMC)* flat_MC_SF 
    # norm_val = 254.528077 # quick test
    sig_norm_subCat0_vbf = ROOT.RooRealVar(signal_subCat0_vbf.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat0_vbf norm_val: {norm_val}")
    sig_norm_subCat0_vbf.setConstant(True)

    # subCat 1
    subCat_filter = (processed_eventsSignalMC_vbf[subCatIdx_name] == 1)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC_vbf.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat1_vbf_SigMC = ak.to_numpy(
        processed_eventsSignalMC_vbf.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat1_vbf_signal = ROOT.TH1F("subCat1_vbf_rooHist_signal", "subCat1_vbf_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat1_vbf_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat1_vbf_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat1_vbf_signal = ROOT.RooDataHist("subCat1_vbf_rooHist_signal", "subCat1_vbf_rooHist_signal", ROOT.RooArgSet(mass), roo_histData_subCat1_vbf_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat1_vbf_signal = roo_histData_subCat1_vbf_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat1"],
        "dataset": ["VBF"], 
        "yield": [data_subCat1_vbf_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat1_vbf_SigMC)* flat_MC_SF
    # norm_val = 295.214 # quick test
    sig_norm_subCat1_vbf = ROOT.RooRealVar(signal_subCat1_vbf.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat1_vbf norm_val: {norm_val}")
    sig_norm_subCat1_vbf.setConstant(True)

    # subCat 2
    subCat_filter = (processed_eventsSignalMC_vbf[subCatIdx_name] == 2)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC_vbf.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat2_vbf_SigMC = ak.to_numpy(
        processed_eventsSignalMC_vbf.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat2_vbf_signal = ROOT.TH1F("subCat2_vbf_rooHist_signal", "subCat2_vbf_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat2_vbf_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat2_vbf_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat2_vbf_signal = ROOT.RooDataHist("subCat2_vbf_rooHist_signal", "subCat2_vbf_rooHist_signal", ROOT.RooArgSet(mass), roo_histData_subCat2_vbf_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat2_vbf_signal = roo_histData_subCat2_vbf_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat2"],
        "dataset": ["VBF"], 
        "yield": [data_subCat2_vbf_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat2_vbf_SigMC) * flat_MC_SF
    # norm_val = 124.0364 # quick test
    sig_norm_subCat2_vbf = ROOT.RooRealVar(signal_subCat2_vbf.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat2_vbf norm_val: {norm_val}")
    sig_norm_subCat2_vbf.setConstant(True)

    # subCat 3
    subCat_filter = (processed_eventsSignalMC_vbf[subCatIdx_name] == 3)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC_vbf.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat3_vbf_SigMC = ak.to_numpy(
        processed_eventsSignalMC_vbf.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat3_vbf_signal = ROOT.TH1F("subCat3_vbf_rooHist_signal", "subCat3_vbf_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat3_vbf_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat3_vbf_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat3_vbf_signal = ROOT.RooDataHist("subCat3_vbf_rooHist_signal", "subCat3_vbf_rooHist_signal", ROOT.RooArgSet(mass), roo_histData_subCat3_vbf_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat3_vbf_signal = roo_histData_subCat3_vbf_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat3"],
        "dataset": ["VBF"], 
        "yield": [data_subCat3_vbf_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat3_vbf_SigMC)* flat_MC_SF
    # norm_val = 116.4918 # quick test
    sig_norm_subCat3_vbf = ROOT.RooRealVar(signal_subCat3_vbf.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat3_vbf norm_val: {norm_val}")
    sig_norm_subCat3_vbf.setConstant(True)
    
    # subCat 4
    subCat_filter = (processed_eventsSignalMC_vbf[subCatIdx_name] == 4)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC_vbf.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat4_vbf_SigMC = ak.to_numpy(
        processed_eventsSignalMC_vbf.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat4_vbf_signal = ROOT.TH1F("subCat4_vbf_rooHist_signal", "subCat4_vbf_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat4_vbf_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat4_vbf_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat4_vbf_signal = ROOT.RooDataHist("subCat4_vbf_rooHist_signal", "subCat4_vbf_rooHist_signal", ROOT.RooArgSet(mass), roo_histData_subCat4_vbf_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat4_vbf_signal = roo_histData_subCat4_vbf_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat4"],
        "dataset": ["VBF"], 
        "yield": [data_subCat4_vbf_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)
    print(f"yield_df after VBF: \n {yield_df}")

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat4_vbf_SigMC)* flat_MC_SF
    sig_norm_subCat4_vbf = ROOT.RooRealVar(signal_subCat4_vbf.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat4_vbf norm_val: {norm_val}")
    sig_norm_subCat4_vbf.setConstant(True)
    
    # ---------------------------------------------------
    # Fit signal model individually, not simultaneous. Sigma, and left and right tails are different for each category
    # ---------------------------------------------------

    # subCat 0
    _ = signal_subCat0_vbf.fitTo(data_subCat0_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat0_vbf.fitTo(data_subCat0_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()


    # Freeze the MH parameters. Source: "Crucially, we need to freeze the fit parameters of the signal mode" https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/tutorial2023/parametric_exercise/#signal-modelling
    MH_subCat0.setConstant(True)
    

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat0_vbf.setConstant(True)
    alpha1_subCat0_vbf.setConstant(True)
    n1_subCat0_vbf.setConstant(True)
    alpha2_subCat0_vbf.setConstant(True)
    n2_subCat0_vbf.setConstant(True)


    # subCat 1
    _ = signal_subCat1_vbf.fitTo(data_subCat1_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat1_vbf.fitTo(data_subCat1_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat1_vbf.setConstant(True)
    alpha1_subCat1_vbf.setConstant(True)
    n1_subCat1_vbf.setConstant(True)
    alpha2_subCat1_vbf.setConstant(True)
    n2_subCat1_vbf.setConstant(True)



    # subCat 2
    _ = signal_subCat2_vbf.fitTo(data_subCat2_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat2_vbf.fitTo(data_subCat2_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat2_vbf.setConstant(True)
    alpha1_subCat2_vbf.setConstant(True)
    n1_subCat2_vbf.setConstant(True)
    alpha2_subCat2_vbf.setConstant(True)
    n2_subCat2_vbf.setConstant(True)


    
    # subCat 3
    _ = signal_subCat3_vbf.fitTo(data_subCat3_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat3_vbf.fitTo(data_subCat3_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat3_vbf.setConstant(True)
    alpha1_subCat3_vbf.setConstant(True)
    n1_subCat3_vbf.setConstant(True)
    alpha2_subCat3_vbf.setConstant(True)
    n2_subCat3_vbf.setConstant(True)



    # subCat 4
    # _ = signal_subCat4_vbf.fitTo(data_subCat4_vbf_signal,  EvalBackend=device, Save=True, )
    # fit_result = signal_subCat4_vbf.fitTo(data_subCat4_vbf_signal,  EvalBackend=device, Save=True, )
    _ = signal_subCat4_vbf.fitTo(data_subCat4_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat4_vbf.fitTo(data_subCat4_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat4_vbf.setConstant(True)
    alpha1_subCat4_vbf.setConstant(True)
    n1_subCat4_vbf.setConstant(True)
    alpha2_subCat4_vbf.setConstant(True)
    n2_subCat4_vbf.setConstant(True)


    
        
    # # -------------------------------------------------------------------------
    # # Plotting
    # # -------------------------------------------------------------------------
    
    # # -------------------------------------------------------------------------
    # # do signal ggH plotting with fit and data
    # # -------------------------------------------------------------------------
    
    # # subCat 0
    # print(f"data_subCat0_signal.sumEntries(): {data_subCat0_signal.sumEntries()}")
    # name = "Canvas"
    # canvas = ROOT.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    # canvas.cd()
    # frame = mass.frame()
    # legend = ROOT.TLegend(0.65,0.55,0.9,0.7)
    # name = data_subCat0_signal.GetName()
    # data_subCat0_signal.plotOn(frame, DataError="SumW2", Name=name)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    # name = signal_subCat0.GetName()
    # signal_subCat0.plotOn(frame, Name=name, LineColor=ROOT.kGreen)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    # frame.Draw()
    # legend.Draw()
        
    
    # canvas.Update()
    # canvas.Draw()
    # canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat0.pdf")

    # # subCat 1
    # print(f"data_subCat1_signal.sumEntries(): {data_subCat1_signal.sumEntries()}")
    # name = "Canvas"
    # canvas = ROOT.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    # canvas.cd()
    # frame = mass.frame()
    # legend = ROOT.TLegend(0.65,0.55,0.9,0.7)
    # name = data_subCat1_signal.GetName()
    # data_subCat1_signal.plotOn(frame, DataError="SumW2", Name=name)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    # name = signal_subCat1.GetName()
    # signal_subCat1.plotOn(frame, Name=name, LineColor=ROOT.kGreen)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    # frame.Draw()
    # legend.Draw()
    
    # canvas.Update()
    # canvas.Draw()
    # canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat1.pdf")

    # # subCat 2
    # print(f"data_subCat2_signal.sumEntries(): {data_subCat2_signal.sumEntries()}")
    # name = "Canvas"
    # canvas = ROOT.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    # canvas.cd()
    # frame = mass.frame()
    # legend = ROOT.TLegend(0.65,0.55,0.9,0.7)
    # name = data_subCat2_signal.GetName()
    # data_subCat2_signal.plotOn(frame, DataError="SumW2", Name=name)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    # name = signal_subCat2.GetName()
    # signal_subCat2.plotOn(frame, Name=name, LineColor=ROOT.kGreen)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    # frame.Draw()
    # legend.Draw()
    
    # canvas.Update()
    # canvas.Draw()
    # canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat2.pdf")

    # # subCat 3
    # print(f"data_subCat3_signal.sumEntries(): {data_subCat3_signal.sumEntries()}")
    # name = "Canvas"
    # canvas = ROOT.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    # canvas.cd()
    # frame = mass.frame()
    # legend = ROOT.TLegend(0.65,0.55,0.9,0.7)
    # name = data_subCat3_signal.GetName()
    # data_subCat3_signal.plotOn(frame, DataError="SumW2", Name=name)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    # name = signal_subCat3.GetName()
    # signal_subCat3.plotOn(frame, Name=name, LineColor=ROOT.kGreen)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    # frame.Draw()
    # legend.Draw()
    
    # canvas.Update()
    # canvas.Draw()
    # canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat3.pdf")

    # # subCat 4
    # print(f"data_subCat4_signal.sumEntries(): {data_subCat4_signal.sumEntries()}")
    # name = "Canvas"
    # canvas = ROOT.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    # canvas.cd()
    # frame = mass.frame()
    # legend = ROOT.TLegend(0.65,0.55,0.9,0.7)
    # name = data_subCat4_signal.GetName()
    # data_subCat4_signal.plotOn(frame, DataError="SumW2", Name=name)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    # name = signal_subCat4.GetName()
    # signal_subCat4.plotOn(frame, Name=name, LineColor=ROOT.kGreen)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    # frame.Draw()
    # legend.Draw()
    
    # canvas.Update()
    # canvas.Draw()
    # canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat4.pdf")

    # # -------------------------------------------------------------------------
    # # do signal VBF plotting with fit and data
    # # -------------------------------------------------------------------------
    
    # # subCat 0
    # print(f"data_subCat0_vbf_signal.sumEntries(): {data_subCat0_vbf_signal.sumEntries()}")
    # name = "Canvas"
    # canvas = ROOT.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    # canvas.cd()
    # frame = mass.frame()
    # legend = ROOT.TLegend(0.65,0.55,0.9,0.7)
    # name = data_subCat0_vbf_signal.GetName()
    # data_subCat0_vbf_signal.plotOn(frame, DataError="SumW2", Name=name)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    # name = signal_subCat0_vbf.GetName()
    # signal_subCat0_vbf.plotOn(frame, Name=name, LineColor=ROOT.kGreen)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    # frame.Draw()
    # legend.Draw()
    
    # canvas.Update()
    # canvas.Draw()
    # canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat0_vbf.pdf")

    # # subCat 1
    # print(f"data_subCat1_vbf_signal.sumEntries(): {data_subCat1_vbf_signal.sumEntries()}")
    # name = "Canvas"
    # canvas = ROOT.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    # canvas.cd()
    # frame = mass.frame()
    # legend = ROOT.TLegend(0.65,0.55,0.9,0.7)
    # name = data_subCat1_vbf_signal.GetName()
    # data_subCat1_vbf_signal.plotOn(frame, DataError="SumW2", Name=name)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    # name = signal_subCat1_vbf.GetName()
    # signal_subCat1_vbf.plotOn(frame, Name=name, LineColor=ROOT.kGreen)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    # frame.Draw()
    # legend.Draw()
    
    # canvas.Update()
    # canvas.Draw()
    # canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat1_vbf.pdf")

    # # subCat 2
    # print(f"data_subCat2_vbf_signal.sumEntries(): {data_subCat2_vbf_signal.sumEntries()}")
    # name = "Canvas"
    # canvas = ROOT.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    # canvas.cd()
    # frame = mass.frame()
    # legend = ROOT.TLegend(0.65,0.55,0.9,0.7)
    # name = data_subCat2_vbf_signal.GetName()
    # data_subCat2_vbf_signal.plotOn(frame, DataError="SumW2", Name=name)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    # name = signal_subCat2_vbf.GetName()
    # signal_subCat2_vbf.plotOn(frame, Name=name, LineColor=ROOT.kGreen)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    # frame.Draw()
    # legend.Draw()
    
    # canvas.Update()
    # canvas.Draw()
    # canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat2_vbf.pdf")

    # # subCat 3
    # print(f"data_subCat3_vbf_signal.sumEntries(): {data_subCat3_vbf_signal.sumEntries()}")
    # name = "Canvas"
    # canvas = ROOT.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    # canvas.cd()
    # frame = mass.frame()
    # legend = ROOT.TLegend(0.65,0.55,0.9,0.7)
    # name = data_subCat3_vbf_signal.GetName()
    # data_subCat3_vbf_signal.plotOn(frame, DataError="SumW2", Name=name)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    # name = signal_subCat3_vbf.GetName()
    # signal_subCat3_vbf.plotOn(frame, Name=name, LineColor=ROOT.kGreen)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    # frame.Draw()
    # legend.Draw()
    
    # canvas.Update()
    # canvas.Draw()
    # canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat3_vbf.pdf")

    # # subCat 4
    # print(f"data_subCat4_vbf_signal.sumEntries(): {data_subCat4_vbf_signal.sumEntries()}")
    # name = "Canvas"
    # canvas = ROOT.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    # canvas.cd()
    # frame = mass.frame()
    # legend = ROOT.TLegend(0.65,0.55,0.9,0.7)
    # name = data_subCat4_vbf_signal.GetName()
    # data_subCat4_vbf_signal.plotOn(frame, DataError="SumW2", Name=name)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    # name = signal_subCat4_vbf.GetName()
    # signal_subCat4_vbf.plotOn(frame, Name=name, LineColor=ROOT.kGreen)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    # frame.Draw()
    # legend.Draw()
    
    # canvas.Update()
    # canvas.Draw()
    # canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat4_vbf.pdf")

    # ---------------------------------------------------
    # Save to Signal, Background and Data to Workspace
    # ---------------------------------------------------
    # workspace_path = "./workspaces"
    workspace_path = f"{base_path}/workspaces"
    os.makedirs(workspace_path, exist_ok=True)


    # unfreeze the hmm sigma and peak b4 saving
    CMS_hmm_sigma_cat0_ggh.setConstant(False)
    CMS_hmm_peak_cat0_ggh.setConstant(False)

    CMS_hmm_sigma_cat1_ggh.setConstant(False)
    CMS_hmm_peak_cat1_ggh.setConstant(False)
    
    CMS_hmm_sigma_cat2_ggh.setConstant(False)
    CMS_hmm_peak_cat2_ggh.setConstant(False)
    
    CMS_hmm_sigma_cat3_ggh.setConstant(False)
    CMS_hmm_peak_cat3_ggh.setConstant(False)
    
    CMS_hmm_sigma_cat4_ggh.setConstant(False)
    CMS_hmm_peak_cat4_ggh.setConstant(False)




    # conver nameing schemes
    corePdf_subCat0 = env_pdfs[0]
    corePdf_subCat1 = env_pdfs[1]
    corePdf_subCat2 = env_pdfs[2]
    corePdf_subCat3 = env_pdfs[3]
    corePdf_subCat4 = env_pdfs[4]


    nevents = roo_datasetData_subCat0.sumEntries()
    bkg_subCat0_norm = ROOT.RooRealVar(corePdf_subCat0.GetName()+"_norm","Background normalization value",nevents,0,3*nevents) # free floating value
    
    nevents = roo_datasetData_subCat1.sumEntries()
    bkg_subCat1_norm = ROOT.RooRealVar(corePdf_subCat1.GetName()+"_norm","Background normalization value",nevents,0,3*nevents) # free floating value
    
    nevents = roo_datasetData_subCat2.sumEntries()
    bkg_subCat2_norm = ROOT.RooRealVar(corePdf_subCat2.GetName()+"_norm","Background normalization value",nevents,0,3*nevents) # free floating value
    
    nevents = roo_datasetData_subCat3.sumEntries()
    bkg_subCat3_norm = ROOT.RooRealVar(corePdf_subCat3.GetName()+"_norm","Background normalization value",nevents,0,3*nevents) # free floating value
    
    nevents = roo_datasetData_subCat4.sumEntries()
    bkg_subCat4_norm = ROOT.RooRealVar(corePdf_subCat4.GetName()+"_norm","Background normalization value",nevents,0,3*nevents) # free floating value
    
    
    # subCat 0 
    fout = ROOT.TFile(f"{workspace_path}/workspace_bkg_cat0_{category}.root","RECREATE")
    wout = ROOT.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    roo_histData_subCat0.SetName("data_cat0_ggh");
    corePdf_subCat0.SetName("bkg_cat0_ggh_pdf");
    bkg_subCat0_norm.SetName(corePdf_subCat0.GetName()+"_norm"); 
    # make norm for data
    nevents = roo_histData_subCat0.sumEntries()
    roo_histData_subCat0_norm = ROOT.RooRealVar(roo_histData_subCat0.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    wout.Import(roo_histData_subCat0_norm);
    wout.Import(roo_histData_subCat0);
    # wout.Import(cat_subCat0);
    wout.Import(bkg_subCat0_norm);
    wout.Import(corePdf_subCat0);
    # wout.Print();
    wout.Write();

    fout = ROOT.TFile(f"{workspace_path}/workspace_sig_cat0_{category}.root","RECREATE")
    wout = ROOT.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    signal_subCat0.SetName("ggH_cat0_ggh_pdf");
    roo_histData_subCat0_signal.SetName("data_ggH_cat0_ggh");
    sig_norm_subCat0.SetName(signal_subCat0.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat0);
    wout.Import(signal_subCat0); 
    wout.Import(roo_histData_subCat0_signal); 
    
    signal_subCat0_vbf.SetName("qqH_cat0_ggh_pdf");
    roo_histData_subCat0_vbf_signal.SetName("data_qqH_cat0_ggh");
    sig_norm_subCat0_vbf.SetName(signal_subCat0_vbf.GetName()+"_norm"); 
    wout.Import(signal_subCat0_vbf);
    wout.Import(roo_histData_subCat0_vbf_signal); 
    wout.Import(sig_norm_subCat0_vbf); 
    
    # wout.Print();
    wout.Write();
    

    # subCat 1 
    fout = ROOT.TFile(f"{workspace_path}/workspace_bkg_cat1_{category}.root","RECREATE")
    wout = ROOT.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    roo_histData_subCat1.SetName("data_cat1_ggh");
    corePdf_subCat1.SetName("bkg_cat1_ggh_pdf");
    bkg_subCat1_norm.SetName(corePdf_subCat1.GetName()+"_norm");
    # make norm for data
    nevents = roo_histData_subCat1.sumEntries()
    roo_histData_subCat1_norm = ROOT.RooRealVar(roo_histData_subCat1.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    wout.Import(roo_histData_subCat1_norm);
    wout.Import(roo_histData_subCat1);
    # wout.Import(cat_subCat1);
    wout.Import(bkg_subCat1_norm);
    wout.Import(corePdf_subCat1);
    # wout.Print();
    wout.Write();

    fout = ROOT.TFile(f"{workspace_path}/workspace_sig_cat1_{category}.root","RECREATE")
    wout = ROOT.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    signal_subCat1.SetName("ggH_cat1_ggh_pdf"); 
    roo_histData_subCat1_signal.SetName("data_ggH_cat1_ggh");
    sig_norm_subCat1.SetName(signal_subCat1.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat1);
    wout.Import(signal_subCat1); 
    wout.Import(roo_histData_subCat1_signal); 

    signal_subCat1_vbf.SetName("qqH_cat1_ggh_pdf"); 
    roo_histData_subCat1_vbf_signal.SetName("data_qqH_cat1_ggh");
    sig_norm_subCat1_vbf.SetName(signal_subCat1_vbf.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat1_vbf);
    wout.Import(signal_subCat1_vbf); 
    wout.Import(roo_histData_subCat1_vbf_signal); 
    # wout.Print();
    wout.Write();

    # subCat 2
    fout = ROOT.TFile(f"{workspace_path}/workspace_bkg_cat2_{category}.root","RECREATE")
    wout = ROOT.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    roo_histData_subCat2.SetName("data_cat2_ggh");
    corePdf_subCat2.SetName("bkg_cat2_ggh_pdf");
    bkg_subCat2_norm.SetName(corePdf_subCat2.GetName()+"_norm");
    # make norm for data
    nevents = roo_histData_subCat2.sumEntries()
    roo_histData_subCat2_norm = ROOT.RooRealVar(roo_histData_subCat2.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    wout.Import(roo_histData_subCat2_norm);
    wout.Import(roo_histData_subCat2);
    # wout.Import(cat_subCat2);
    wout.Import(bkg_subCat2_norm);
    wout.Import(corePdf_subCat2);
    # wout.Print();
    wout.Write();

    fout = ROOT.TFile(f"{workspace_path}/workspace_sig_cat2_{category}.root","RECREATE")
    wout = ROOT.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    signal_subCat2.SetName("ggH_cat2_ggh_pdf"); 
    roo_histData_subCat2_signal.SetName("data_ggH_cat2_ggh");
    sig_norm_subCat2.SetName(signal_subCat2.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat2);
    wout.Import(signal_subCat2); 
    wout.Import(roo_histData_subCat2_signal); 

    signal_subCat2_vbf.SetName("qqH_cat2_ggh_pdf"); 
    roo_histData_subCat2_vbf_signal.SetName("data_qqH_cat2_ggh");
    sig_norm_subCat2_vbf.SetName(signal_subCat2_vbf.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat2_vbf);
    wout.Import(signal_subCat2_vbf); 
    wout.Import(roo_histData_subCat2_vbf_signal); 
    # wout.Print();
    wout.Write();


    # subCat 3
    fout = ROOT.TFile(f"{workspace_path}/workspace_bkg_cat3_{category}.root","RECREATE")
    wout = ROOT.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    roo_histData_subCat3.SetName("data_cat3_ggh");
    corePdf_subCat3.SetName("bkg_cat3_ggh_pdf");
    bkg_subCat3_norm.SetName(corePdf_subCat3.GetName()+"_norm");
    # make norm for data
    nevents = roo_histData_subCat3.sumEntries()
    roo_histData_subCat3_norm = ROOT.RooRealVar(roo_histData_subCat3.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    wout.Import(roo_histData_subCat3_norm);
    wout.Import(roo_histData_subCat3);
    # wout.Import(cat_subCat3);
    wout.Import(bkg_subCat3_norm);
    wout.Import(corePdf_subCat3);
    # wout.Print();
    wout.Write();

    fout = ROOT.TFile(f"{workspace_path}/workspace_sig_cat3_{category}.root","RECREATE")
    wout = ROOT.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    signal_subCat3.SetName("ggH_cat3_ggh_pdf"); 
    roo_histData_subCat3_signal.SetName("data_ggH_cat3_ggh");
    sig_norm_subCat3.SetName(signal_subCat3.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat3);
    wout.Import(signal_subCat3); 
    wout.Import(roo_histData_subCat3_signal); 

    signal_subCat3_vbf.SetName("qqH_cat3_ggh_pdf"); 
    roo_histData_subCat3_vbf_signal.SetName("data_qqH_cat3_ggh");
    sig_norm_subCat3_vbf.SetName(signal_subCat3_vbf.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat3_vbf);
    wout.Import(signal_subCat3_vbf); 
    wout.Import(roo_histData_subCat3_vbf_signal); 
    # wout.Print();
    wout.Write();

    # subCat 4
    fout = ROOT.TFile(f"{workspace_path}/workspace_bkg_cat4_{category}.root","RECREATE")
    wout = ROOT.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    roo_histData_subCat4.SetName("data_cat4_ggh");
    corePdf_subCat4.SetName("bkg_cat4_ggh_pdf");
    bkg_subCat4_norm.SetName(corePdf_subCat4.GetName()+"_norm");
    # make norm for data
    nevents = roo_histData_subCat4.sumEntries()
    roo_histData_subCat4_norm = ROOT.RooRealVar(roo_histData_subCat4.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    wout.Import(roo_histData_subCat4_norm);
    wout.Import(roo_histData_subCat4);
    # wout.Import(cat_subCat4);
    wout.Import(bkg_subCat4_norm);
    wout.Import(corePdf_subCat4);
    # wout.Print();
    wout.Write();

    fout = ROOT.TFile(f"{workspace_path}/workspace_sig_cat4_{category}.root","RECREATE")
    wout = ROOT.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    signal_subCat4.SetName("ggH_cat4_ggh_pdf"); 
    roo_histData_subCat4_signal.SetName("data_ggH_cat4_ggh");
    sig_norm_subCat4.SetName(signal_subCat4.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat4);
    wout.Import(signal_subCat4); 
    wout.Import(roo_histData_subCat4_signal); 

    signal_subCat4_vbf.SetName("qqH_cat4_ggh_pdf"); 
    roo_histData_subCat4_vbf_signal.SetName("data_qqH_cat4_ggh");
    sig_norm_subCat4_vbf.SetName(signal_subCat4_vbf.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat4_vbf);
    wout.Import(signal_subCat4_vbf); 
    wout.Import(roo_histData_subCat4_vbf_signal);
    # wout.Print();
    wout.Write();


    print(f"workspace_path: {workspace_path}")
