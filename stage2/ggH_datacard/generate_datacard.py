import time
import numpy as np
import pickle
import awkward as ak
import dask_awkward as dak
from distributed import Client
from omegaconf import OmegaConf
from typing import Tuple, List, Dict
import ROOT as rt
import ROOT
# from modules.fit_functions import MakeFEWZxBernDof3, plot_6_23, plot_6_26, getSigBkgPdf
# from modules.fit_functions import getBWZ_gamma, getBWZxBern, getLandxBern, getFEWZxBern
import argparse
import os
import copy
import pandas as pd
# from modules.utils import getGOF_KS
from src.corrections.jet import applyUpDown, getJecJerUncertainties

def fillWgtVarations(df : pd.DataFrame, events, nSubCats : int):
    # print(f"fillWgtVarations b4: \n {df}")
    wgt_fields = [col for col in df.columns if "wgt" in col]
    for subCat_ix in range(nSubCats):
        subcat_filter = events.subCategory_idx == subCat_ix
        row_data = {}
        for wgt_name in wgt_fields:
            wgt_values = events[wgt_name]
            wgt_values = wgt_values[subcat_filter]
            wgt_yield = ak.sum(wgt_values)
            
            row_data[wgt_name] = wgt_yield
        df.loc[f"subCat{subCat_ix}"] = row_data
    # print(f"fillWgtVarations after: \n {df}")
    # raise ValueError
    
    return df


def getRelativeYield2Nominal(df):
    df_rel = df.div(df["wgt_nominal"], axis=0)
    return df_rel


# def fillJecJerVarations(df : pd.DataFrame, events, nSubCats : int, jec_unc_fields : list):
#     wgt_name = "wgt_nominal"
#     for jec_unc_name in jec_unc_fields:
#         col_data = []
#         for subCat_ix in range(nSubCats):
#             subCat_field = f"subCategory_idx_{jec_unc_name}"
#             subcat_filter = events[subCat_field] == subCat_ix
#             wgt_values = events[wgt_name]
#             wgt_values = wgt_values[subcat_filter]
#             wgt_yield = ak.sum(wgt_values)
#             col_data.append(wgt_yield)
#             # print(f"{jec_unc_name} subcat {subCat_ix} yield: {wgt_yield}")
#         df[jec_unc_name] = col_data
#     return df


def fillJecJerVarationsByYear(df : pd.DataFrame, load_path, years : list, nSubCats : int, jec_unc_fields : list):
    wgt_name = "wgt_nominal"
    jec_unc_fields = jec_unc_fields + ["nominal"] # add nominal
    for subCat_ix in range(nSubCats):
        for year in years:
            year_load_path = load_path.replace("year_value", year)
            print(f"year_load_path: {year_load_path}")
            events = ak.from_parquet(year_load_path)
            print(f"events.fields: {events.fields}")
            row_data = {}
            # row_data["year"] =  year
            for jec_unc_name in jec_unc_fields:
                if jec_unc_name == "nominal":
                    subCat_field = f"subCategory_idx"
                else:
                    subCat_field = f"subCategory_idx_{jec_unc_name}"
                if not subCat_field in events.fields: 
                    continue # ie. Absolute_2017 in events from 2018 stage2
                subcat_filter = events[subCat_field] == subCat_ix
                wgt_values = events[wgt_name]
                wgt_values = wgt_values[subcat_filter]
                wgt_yield = ak.sum(wgt_values)
                row_data[jec_unc_name] = wgt_yield
                
                # print(f"{jec_unc_name} subcat {subCat_ix} yield: {wgt_yield}")
            df.loc[f"subCat{subCat_ix}_{year}"] = row_data
    return df

def combine_dfByYear(df_JecByYear, jec_unc_fields : list, years : list, nSubCats : int):
    # jec_unc_fields = df_JecByYear.columns
    # print(f"combineDfByYear jec_unc_fields: {jec_unc_fields}")

    # define out df
    row_labels = [f"subCat{i}" for i in range(nSubCats)]
    out_df = pd.DataFrame(index=row_labels, columns=jec_unc_fields)
    # print(f"combineDfByYear out_df b4: {out_df}")
    
    
    for subCat_ix in range(nSubCats):
        row_data = {
            jec_unc_field : 0 for jec_unc_field in jec_unc_fields
        }
        print(row_data)
        for year in years:
            row_idx = f"subCat{subCat_ix}_{year}"
            for jec_unc_field in jec_unc_fields:
                value = df_JecByYear.loc[row_idx, jec_unc_field]
                # print(f"combineDfByYear {row_idx} {jec_unc_field} value: {value}")
                
                if pd.isna(value):
                    value = df_JecByYear.loc[row_idx, "nominal"]
                    row_data[jec_unc_field] += value
                else:
                    row_data[jec_unc_field] += value
        # print(f"subCat{subCat_ix} row_data: {row_data}")
        combined_row_idx = f"subCat{subCat_ix}"
        out_df.loc[combined_row_idx] = row_data
    # print(f"combineDfByYear out_df after: {out_df}")

    return out_df
    

def getProcessedEvents(events, fields2load, jec_unc_fields):
    bdt_fields = [
        "BDT_score",
        "subCategory_idx",
    ]
    bdt_fields_variation = []
    for jec_unc_field in jec_unc_fields:
        for bdt_field in bdt_fields:
            name = f"{bdt_field}_{jec_unc_field}"
            if name in events.fields:
                bdt_fields_variation.append(name)

    wgt_fields = [field for field in events.fields if "wgt" in field]

    # keep only fields that exist
    requested = fields2load + bdt_fields + bdt_fields_variation + wgt_fields
    fields_total = [field for field in requested if field in events.fields]

    missing = [field for field in requested if field not in events.fields]
    if missing:
        print(f"[INFO] Missing fields skipped: {missing}")

    print("fields_total:", fields_total)

    processed_events = ak.zip({field: events[field] for field in fields_total}).compute()
    return processed_events


def flipDfAddSample(df, sample : str):
    """
    flips the rows and columns of the given df and adds sample string value to each column
    """
    # print(f"pre flip: {df}")
    df_flipped = df.T  # Transpose (flip rows and columns)
    df_flipped.columns = df_flipped.columns.astype(str) + f"_{sample}"
    # print(df_flipped)
    return df_flipped
    
def getDataCardLikeDf(samples, base_path):
    # collect dfs
    # df_dict = {}
    df_dict_l = []
    for sample in samples:
        df_path = f"{base_path}/{sample}_total_relYield.csv"
        df = pd.read_csv(df_path, index_col=0)
        df = flipDfAddSample(df, sample)
        df_dict_l.append(df)

    # assuming that the row indices are identical, we stitch the dfs
    df_stitched = pd.concat(df_dict_l, axis=1)
    return df_stitched


def factor_pair(df, nuis, proc):
    """
    Return 'up/down' factors for nuisance `nuis` and process `proc` (ggh or vbf),
    or '-' if missing/invalid.
    """
    try:
        up = float(df.loc[f"{nuis}_up", proc])
        dn = float(df.loc[f"{nuis}_down", proc])
        if np.isnan(up) or np.isnan(dn):
            return "-"        
        # if up > 0 and dn > 0:
        if round(up, 3) == round(dn, 3):
            return f"{up:.4g}"
        else:
            return f"{up:.4g}/{dn:.4g}"
    except Exception:
        pass
    return "-"



def extract_nuisances(df):
    """
    Take the DataFrame row index, strip '_up' and '_down' suffixes,
    and return a sorted list of unique nuisance names.
    """
    nuis = {str(idx).strip().removesuffix("_up").removesuffix("_down")
            for idx in df.index}
    nuis = sorted(nuis)
    nuis.remove("wgt_nominal")
    return nuis

def buildDataCard(df, samples, subCat_ix, year):
    nuisances = extract_nuisances(df)
    lines = []
    # print(nuisances)
    lines.append(f"""
bin                cat{subCat_ix}_ggh     cat{subCat_ix}_ggh     cat{subCat_ix}_ggh               
process            ggH_hmm      qqH_hmm      bkg  
process            -2           -1           1
rate               1            1            1
------------
BR_mm            lnN       1.012        1.012        - 
------------
QCDscale_ggH     lnN     0.933/1.046  -            -          
QCDscale_qqH     lnN     -            0.997/1.004  -          
pdf_Higgs_gg     lnN     1.032        -            -          
pdf_Higgs_qq     lnN     -            1.021        -      
------------
""")# QCD and pdf Source table 2.1 from AN-19-124. Lumi source: https://twiki.cern.ch/twiki/bin/viewauth/CMS/LumiRecommendationsRun2
        
    if year == "2016":
        lines.append("lumi_13TeV_2016       lnN     1.012       1.012       -")
    elif year == "2017":
        lines.append("lumi_13TeV_2017       lnN     1.023       1.023       -")
    elif year == "2018":
        lines.append("lumi_13TeV_2018       lnN     1.025       1.025       -")
    elif "2022" in year:
        lines.append("lumi_13p6TeV_Corr     lnN     1.0138      1.0138      -")
    elif "2023" in year:
        lines.append("lumi_13p6TeV_Corr     lnN     1.0017      1.0017      -")
        lines.append("lumi_13p6TeV_23_24    lnN     1.0127      1.0127      -")
    elif "2024" in year:
        lines.append("lumi_13p6TeV_Corr     lnN     1.0020      1.0020      -")
        lines.append("lumi_13p6TeV_23_24    lnN     1.0068      1.0068      -")
        lines.append("lumi_13p6TeV_uncorr   lnN     1.0144      1.0144      -")
    elif year == "all":
        lines.append("lumi_13p6TeV_Corr     lnN     1.0020      1.0020      -")
        lines.append("lumi_13p6TeV_23_24    lnN     1.0068      1.0068      -")
        lines.append("lumi_13p6TeV_uncorr   lnN     1.0144      1.0144      -")
    
    for u in nuisances:
        # for sample i
        ggh_val = factor_pair(df, u, f"subCat{subCat_ix}_ggh")
        vbf_val = factor_pair(df, u, f"subCat{subCat_ix}_vbf")
        lines.append(f"{u}   lnN   {ggh_val}   {vbf_val}   -")

    datacard_str = "\n".join(lines)
    print(datacard_str)
    # raise ValueError
    return datacard_str
    
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
    default="plots",
    action="store",
    help="Output files will be saved here",
    )
    args = parser.parse_args()
    # check for valid arguments
    if args.load_path == None:
        print("load path to load stage1 output is not specified!")
        raise ValueError

    category = args.category.lower()
    nSubCats = 5
    samples = [
        "ggh",
        "vbf"
    ]
    # sample = "ggh" #FIXME
    # year = "2017" #FIXME

    # make save directory
    base_path = f"{args.save_path}/stage3/{args.year}/{args.label}/datacards/"
    plot_save_path = base_path
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    # make pd df ------------------------------------------------------------------
    for sample in samples:
        year = args.year
        fname = f"processed_events_sigMC_{sample}*.parquet"
        if year=="all":
            load_path = f"{args.load_path}/*/{fname}"
        elif year=="2016only":
            load_path = f"{args.load_path}/2016*/{fname}"
        else:
            load_path = f"{args.load_path}/{year}/{fname}"
        print(f"load_path: {load_path}")
        # processed_events = ak.from_parquet(load_path)
        events = dak.from_parquet(load_path)
        # print(f"events.fields: {events.fields}")
        fields2load  = [
            "dimuon_mass",
        ]
        jec_unc_fields = ["Absolute", "FlavorQCD"]
        # jec_unc_fields = []
        jec_unc_fields = applyUpDown(jec_unc_fields)
        processed_events = getProcessedEvents(events, fields2load, jec_unc_fields)
        print(f"processed_events.wgt_nominal: {processed_events.wgt_nominal}")
        # print(f"processed_events.wgt_nominal len: {ak.num(processed_events.wgt_nominal, axis=0)}")
        # print(f"processed_events.wgt_l1prefiring_up: {processed_events.wgt_l1prefiring_up}")
        
        print(f"processed_events length: {ak.num(processed_events.dimuon_mass, axis=0)}")
        print("events loaded!")
    
        events = processed_events
        wgt_fields = [field for field in events.fields if "wgt" in field]

        fields2process = wgt_fields + jec_unc_fields
        # fields2process = fields2process 
        print(fields2process)
        # raise ValueError
        print(events.fields)
        
    

        # --------------------------------------------------------------
        # fill in wgt unc
        # --------------------------------------------------------------

        # Define your row labels
        row_labels = [f"subCat{i}" for i in range(nSubCats)]
        
        # Create the empty DataFrame
        df_wgts = pd.DataFrame(index=row_labels, columns=wgt_fields)
        df_wgts = fillWgtVarations(df_wgts, events, nSubCats)
        # df_wgts = fillJecJerVarations(df_wgts, events, nSubCats, jec_unc_fields) # we don't use this function any more
        df_wgts.to_csv(f"{base_path}/{sample}_abs_yield.csv")
        df_wgts_rel = getRelativeYield2Nominal(df_wgts)
        df_wgts_rel.to_csv(f"{base_path}/{sample}_relative2nominal.csv")



        # --------------------------------------------------------------
        # fill in jec/jer unc
        # --------------------------------------------------------------

        if args.year == "all":
            # years = ["2018", "2017", "2016postVFP", "2016preVFP"]
            years = ["2022preEE", "2022postEE", "2023", "2023BPix", "2024"]
        else:
            years = [args.year]
        row_labels = []
        for year in years:
            row_labels = row_labels + [f"subCat{i}_{year}" for i in range(nSubCats)]
            
        # df = pd.DataFrame(index=row_labels, columns=(jec_unc_fields+["year"))
        # jec_unc_fields = ["Absolute", "FlavorQCD", "Absolute_2018", "Absolute_2017"]
        jec_yml_path = "configs/parameters/jec.yaml"

        if args.year == "all":
            # years_for_jec = ["2018", "2017", "2016postVFP", "2016preVFP"]
            years_for_jec = ["2022preEE", "2022postEE", "2023", "2023BPix", "2024"]
            jec_unc_base = []
            for y in years_for_jec:
                jec_unc_base.extend(getJecJerUncertainties(jec_yml_path, year=y))
            jec_unc_base = sorted(set(jec_unc_base))
        else:
            jec_unc_base = getJecJerUncertainties(jec_yml_path, year=args.year)

        jec_unc_fields = applyUpDown(jec_unc_base)
        print(f"jec_unc_fields: {jec_unc_fields}")
        
        # raise ValueError
        df_JecByYear = pd.DataFrame(index=row_labels, columns=(jec_unc_fields + ["nominal"]))
        fname = f"processed_events_sigMC_{sample}*.parquet"
        load_path = f"{args.load_path}/year_value/{fname}"
        print(f"load_path: {load_path}")
        df_JecByYear = fillJecJerVarationsByYear(df_JecByYear, load_path, years, nSubCats, jec_unc_fields)
        print(f"df_JecByYear: {df_JecByYear}")
        df_JecCombined = combine_dfByYear(df_JecByYear, jec_unc_fields, years, nSubCats)
        df_JecCombined.to_csv(f"{base_path}/{sample}_jecUnc_absYield.csv")

        df_JecCombined_rel = df_JecCombined.div(df_wgts["wgt_nominal"], axis=0)
        df_JecCombined_rel.to_csv(f"{base_path}/{sample}_jecUnc_relYield.csv")

        
        df_total = pd.concat([df_wgts_rel, df_JecCombined_rel], axis=1)
        # df_total = pd.concat([df_wgts_rel], axis=1)

        df_total.to_csv(f"{base_path}/{sample}_total_relYield.csv")

    # make pd df ------------------------------------------------------------------
    
    # --------------------------------------------------------------
    # convert df to something more datacard-like
    # --------------------------------------------------------------
    df_datacardLike = getDataCardLikeDf(samples, base_path)
    df_datacardLike.to_csv(f"{base_path}/datacardLikeDf.csv")
        
    datacard_start = """
imax *                                                                                                                        
jmax *                                                                                                                        
kmax *                                                                                                                        
------------                                                                                                                  
shapes ggH_hmm     catCAT_INDEX_ggh           my_workspace/workspace_sig_catCAT_INDEX_ggh.root     w:ggH_catCAT_INDEX_ggh_pdf 
shapes qqH_hmm     catCAT_INDEX_ggh           my_workspace/workspace_sig_catCAT_INDEX_ggh.root     w:qqH_catCAT_INDEX_ggh_pdf            
shapes bkg         catCAT_INDEX_ggh           my_workspace/workspace_bkg_catCAT_INDEX_ggh.root     w:bkg_catCAT_INDEX_ggh_pdf            
shapes data_obs    catCAT_INDEX_ggh           my_workspace/workspace_bkg_catCAT_INDEX_ggh.root     w:data_catCAT_INDEX_ggh

------------                                                                                                                  
bin                catCAT_INDEX_ggh      
observation        -1                                                                                                         
------------     

"""

    datacard_end = """
------------
CMS_hmm_peak_catCAT_INDEX_ggh   param  0  0.001
CMS_hmm_sigma_catCAT_INDEX_ggh  param  0  0.1
------------
pdf_index_ggh discrete
""" # Source: CMS_hmm_peak_catCAT_INDEX_ggh and CMS_hmm_sigma_catCAT_INDEX_ggh are the shape uncertainties from line 1385 of AN-19-124 
    # nSubCats = 1
    for subCat_ix in range(nSubCats):
        datacard_subCat_str = datacard_start.replace("CAT_INDEX", str(subCat_ix))
        datacard_subCat_str += buildDataCard(df_datacardLike, samples, subCat_ix, args.year)
        datacard_subCat_str += datacard_end.replace("CAT_INDEX", str(subCat_ix))
        
        # Write it to a file
        datacard_fname = f"{base_path}/datacard_cat{subCat_ix}_ggh.txt"
        with open(datacard_fname, "w") as f:
            f.write(datacard_subCat_str)
        # Write it to a file
        datacard_fname = f"{base_path}/datacard_cat{subCat_ix}_ggh_test.txt"
        datacard_subCat_str = datacard_subCat_str.replace("pdf_index_ggh discrete", "")
        with open(datacard_fname, "w") as f:
            f.write(datacard_subCat_str)

    print(f"Datacards saved at {base_path}")

    print("Success!")

