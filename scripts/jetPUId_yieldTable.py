import os
import glob
import pandas as pd
import dask_awkward as dak
import awkward as ak
from distributed import Client



def filterRegion(events, region="h-peak"):
    if isinstance(events, pd.DataFrame):
        fields = events.columns
    else: # awkward zip
        fields = events.fields  
    if "dimuon_mass" not in fields:
        raise ValueError("dimuon_mass not found in events fields for region selection.")
    dimuon_mass = events["dimuon_mass"]
    z_peak = (dimuon_mass >= 70.0) & (dimuon_mass < 110.0)
    h_peak = (dimuon_mass >= 115.0) & (dimuon_mass < 135.0)
    h_sidebands = ((dimuon_mass >= 110.0) & (dimuon_mass < 115.0)) | (
        (dimuon_mass >= 135.0) & (dimuon_mass < 150.0)
    )
    if region == "z-peak":
        mask = z_peak
    elif region == "h-peak":
        mask = h_peak
    elif region == "h-sidebands":
        mask = h_sidebands
    elif region == "signal":
        mask = h_sidebands | h_peak
    elif region == "full":
        mask = z_peak | h_sidebands | h_peak
    else:
        raise ValueError(
            f"Invalid region selection: {region}. Valid options are: 'z-peak', 'h-peak', 'h-sidebands', 'signal', 'full'."
        )

    return mask, events[mask]


def applyRegionCatCuts(
    events,
    category: str,
    region_name: str,
    process: str,
    variation: str,
    do_vbf_filter_study: bool = False,
    do_VH_veto: bool = False,
):
    use_var = (
        "nominal"
        if (isinstance(variation, str) and variation.startswith("wgt"))
        else variation
    )

    # Helper to fetch the right column, falling back to _nominal or base if needed
    def varcol(base):
        """
        Fetch the appropriate column from the events object, handling variations.

        Attempts to retrieve the column named '{base}_{use_var}', falling back to '{base}_nominal' and then '{base}'.
        Raises a KeyError if none of these columns are present in events.fields.

        Parameters
        ----------
        base : str
            The base name of the column to retrieve.

        Returns
        -------
        awkward.Array
            The selected column from the events object.

        Raises
        ------
        KeyError
            If none of the candidate columns are found in events.fields.
        """
        # print(f"Fetching variable column for: {base}")
        # print(f"Using variation: {use_var}")
        for cand in (f"{base}_{use_var}", f"{base}_nominal", base):
            if cand in events.fields:
                return events[cand]
        raise KeyError(
            f"[selection] Missing required field for selection: tried {base}_{use_var}, {base}_nominal, {base}"
        )

    # do mass region cut
    region, _ = filterRegion(events, region=region_name)

    # --- category cuts: USE varcol(...) for JES/JER-affected columns ---
    nbt_loose = varcol("nBtagLoose")
    nbt_medium = varcol("nBtagMedium")
    jj_mass = varcol("jj_mass")
    jj_dEta = varcol("jj_dEta")
    jet1_pt = varcol("jet1_pt")
    njets = varcol("njets")  # if you cut on it anywhere

    prod_cat_cut = ak.ones_like(region, dtype="bool")

    # do category cut
    if category == "nocat":
        prod_cat_cut = prod_cat_cut  # no additional cut
    else:  # VBF or ggH
        if do_VH_veto:
            print("Applying VH veto!")
            # NOTE: fatjet and MET veto for VH: nfatJets_drmuon == 0 and MET_pt < 150 GeV
            fatjet_veto = ak.fill_none((events.nfatJets_drmuon == 0), value=False)
            met_veto = ak.fill_none((events.MET_pt < 150), value=False)

            # INFO: Apply both fatjet and MET vetoes together
            prod_cat_cut = prod_cat_cut & fatjet_veto & met_veto

        # NOTE: btag cut for VH and ttH categories
        btagLoose_filter = ak.fill_none((nbt_loose >= 2), value=False)
        btagMedium_filter = ak.fill_none((nbt_medium >= 1), value=False) & ak.fill_none(
            (njets >= 2), value=False
        )
        btag_cut = btagLoose_filter | btagMedium_filter

        # vbf_cut = ak.fill_none(events.vbf_cut, value=False) # in the future none values will be replaced with False
        vbf_cut = (jj_mass > 400) & (jj_dEta > 2.5) & (jet1_pt > 35)
        vbf_cut = ak.fill_none(vbf_cut, value=False)
        if category == "vbf":
            # print("vbf mode!")
            prod_cat_cut = prod_cat_cut & vbf_cut
            prod_cat_cut = prod_cat_cut & (
                ~btag_cut
            )  # btag cut is for VH and ttH categories
        elif category == "ggh":
            # print("ggH mode!")
            prod_cat_cut = prod_cat_cut & ~vbf_cut
            prod_cat_cut = prod_cat_cut & (
                ~btag_cut
            )  # btag cut is for VH and ttH categories
        else:
            print("Error: invalid category option!")
            print(
                "Error: invalid category option! Valid options are: 'vbf', 'ggh', 'nocat'."
            )
            raise ValueError(
                "Invalid category option! Valid options are: 'vbf', 'ggh', 'nocat'."
            )

    if do_vbf_filter_study:
        if "dy_" in process:
            vbf_filter = ak.fill_none((events.gjj_mass > 350), value=False)
            is_vbf_filter = ("dy_VBF_filter" in process) or (
                process == "dy_m105_160_vbf_amc"
            )
            if is_vbf_filter:
                # print(f"applying VBF filter cut on: {process}")

                prod_cat_cut = prod_cat_cut & vbf_filter
            else:
                # print(f"cutting off inclusive dy: {process}")
                prod_cat_cut = prod_cat_cut & ~vbf_filter
        else:
            # print(f"no extra processing for {process}")
            pass

    category_selection = prod_cat_cut & region
    # filter events for selected category

    # print(f"len(events) {process} b4 selection: {len(events)}")
    events = events[category_selection]
    return events

def compute_weighted_yields(samples: dict[str, str], region: str, year: str, weight_field: str = "wgt_nominal") -> pd.DataFrame:
    """
    samples: dict of {sample_name: directory_with_parquet_files}
    weight_field: field to sum for weighted yield
    """
    rows = []

    for sample, d in samples.items():
        parquet_glob = os.path.join(d, "*/*.parquet")
        files = glob.glob(parquet_glob)
        # print(files)
        # if not files:
        #     rows.append({"sample": sample, "path": d, "n_files": 0, "yield_wgt": 0.0, "status": "no_files"})
        #     continue

        # Read Parquet via dask_awkward
        events = dak.from_parquet(files)
        # print(events.fields)
        # _, events = filterRegion(events, region=region) # eventwise cut over region
        events = applyRegionCatCuts(
                    events,
                    "vbf",
                    region,
                    sample,
                    "nominal",
                    False,
                )

        # Sum weights (lazy) -> then compute to get a scalar
        # if weight_field not in events.fields:
        #     rows.append({"sample": sample, "path": d, "n_files": len(files), "yield_wgt": None, "status": f"missing_field:{weight_field}"})
        #     continue

        # y = dak.sum(events[weight_field]).compute() 
        print(f"weight_field: {weight_field}")
        if "separate_wgt_zpt_wgt" in events.fields:
            y = dak.sum(events[weight_field]/events["separate_wgt_zpt_wgt"]).compute()
        else:
            y = dak.sum(events[weight_field]).compute() 
            
        raw_num = dak.num(events[weight_field], axis=0).compute() 
        # Ensure plain Python float (sometimes comes back numpy scalar)
        # rows.append({"sample": sample, "path": d, "n_files": len(files), "yield_wgt": float(y), "status": "ok"})
        rows.append({"year": year, "sample": sample, "region": region, "yield": float(y), "raw_num": raw_num})

    df = pd.DataFrame(rows).sort_values("sample").reset_index(drop=True)
    return df

def compare_yields(df_a: pd.DataFrame,
                   df_b: pd.DataFrame,
                   label_a="runA",
                   label_b="runB") -> pd.DataFrame:
    """
    Compare yield_wgt between two yield dataframes.
    Adds absolute and percent differences.
    """
    # keep only needed cols
    cols2keep = ["sample", "yield", "region", "year", "raw_num"]
    cols2compare = ["yield", "raw_num"]
    # a = df_a[cols2keep].rename(columns={"yield": f"yield_{label_a}", "yield": f"raw_num_{label_a}"})
    a = df_a[cols2keep].rename(columns={col : f"{col}_{label_a}" for col in cols2compare})
    b = df_b[cols2keep].rename(columns={col : f"{col}_{label_b}" for col in cols2compare})
    # b = df_b[cols2keep].rename(columns={"yield": f"yield_{label_b}"})

    # merge on sample
    df = pd.merge(a, b, on="sample", how="outer").fillna(0)

    # differences
    diff_col_name = "yield diff"
    df[diff_col_name] = df[f"yield_{label_b}"] - df[f"yield_{label_a}"]
    df[f"percent {diff_col_name}"] = 100 * df[diff_col_name] / df[f"yield_{label_a}"].replace(0, pd.NA)

    return df.sort_values("sample").reset_index(drop=True)


if __name__ == "__main__":
    client = Client(n_workers=30,  threads_per_worker=1, processes=True, memory_limit='10 GiB')
    # --------------------------------------
    # Signal region yield
    # --------------------------------------
    years = [
        # "2024",
        "2023BPix",
        "2023",
        "2022preEE",
        "2022postEE",
    ]
    df_compare_l = []
    for year in years:
        # jet horn 30 GeV cut
        label="Run3_nanoAODv12_02Feb_FilterJetsHorn30GeV"
        
        # base_path = f"/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/{label}/stage1_output/{year}/compacted"
        base_path = f"/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/{label}/stage1_output/{year}/f1_0"
    
        samples = {
            "DY":  f"{base_path}/dyTo2L_M-50_incl",
            # "TT":  f"{base_path}/ttjets_*",
            "ggH": f"{base_path}/ggh_powhegPS",
            "VBF": f"{base_path}/vbf_powheg*",
        }
        region="signal"
        # region="h-sidebands"
        df_yields30GeVCut = compute_weighted_yields(samples, region, year, weight_field="wgt_nominal")
        print(df_yields30GeVCut)
    
        # no jet horn 30 GeV cut
        label="Run3_nanoAODv12_02Feb_FilterJets"
        
        # base_path = f"/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/{label}/stage1_output/{year}/compacted"
        base_path = f"/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/{label}/stage1_output/{year}/f1_0"
    
        if year=="2024":
            samples = {
                "DY":  f"{base_path}/dyTo2Mu_M-50_aMCatNLO",
                # "TT":  f"{base_path}/ttjets_*",
                "ggH": f"{base_path}/ggh_powhegPS",
                "VBF": f"{base_path}/vbf_powheg*",
            }
        # elif "2023" in year:
        #     samples = {
        #         "DY":  f"{base_path}/dyTo2L_M-50_incl",
        #         # "TT":  f"{base_path}/ttjets_*",
        #         "ggH": f"{base_path}/ggh_powhegPS",
        #         "VBF": f"{base_path}/vbf_powheg*",
        #     }
        else:
            samples = {
                "DY":  f"{base_path}/dyTo2L_M-50_incl",
                # "TT":  f"{base_path}/ttjets_*",
                "ggH": f"{base_path}/ggh_powhegPS",
                "VBF": f"{base_path}/vbf_powheg*",
            }
        df_yields = compute_weighted_yields(samples, region, year, weight_field="wgt_nominal")
        print(df_yields)
    
        df_compare = compare_yields(df_yields, df_yields30GeVCut, "baseline", "30GeVCut")
        df_compare.to_csv(f"df_compare_{year}_reg_{region}.csv")
        print(df_compare)
        df_compare_l.append(df_compare)
    
    # df_total = pd.concat(df_compare_l)
    df_total = pd.concat(df_compare_l, ignore_index=True)
    print(df_total)
    df_total.to_csv(f"df_total_{region}.csv")
    
    # --------------------------------------
    # H-sidebands region yield with data included
    # --------------------------------------
    
    years = [
        # "2024",
        "2023BPix",
        "2023",
        "2022preEE",
        "2022postEE",
    ]
    df_compare_l = []
    for year in years:
        # jet horn 30 GeV cut
        label="Run3_nanoAODv12_02Feb_FilterJetsHorn30GeV"
        
        # base_path = f"/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/{label}/stage1_output/{year}/compacted"
        base_path = f"/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/{label}/stage1_output/{year}/f1_0"
    
        samples = {
            "DY":  f"{base_path}/dyTo2L_M-50_incl",
            "Data":  f"{base_path}/data*",
            # "TT":  f"{base_path}/ttjets_*",
            "ggH": f"{base_path}/ggh_powhegPS",
            "VBF": f"{base_path}/vbf_powheg*",
        }
        region="h-sidebands"
        df_yields30GeVCut = compute_weighted_yields(samples, region, year, weight_field="wgt_nominal")
        print(df_yields30GeVCut)
    
        # no jet horn 30 GeV cut
        # label="Run3_nanoAODv12_01Feb_JecJer"
        label="Run3_nanoAODv12_02Feb_FilterJets"
        
        # base_path = f"/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/{label}/stage1_output/{year}/compacted"
        base_path = f"/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/{label}/stage1_output/{year}/f1_0"
    
        if year=="2024":
            samples = {
                "DY":  f"{base_path}/dyTo2Mu_M-50_aMCatNLO",
                "Data":  f"{base_path}/data*",
                # "TT":  f"{base_path}/ttjets_*",
                "ggH": f"{base_path}/ggh_powhegPS",
                "VBF": f"{base_path}/vbf_powheg*",
            }
        # elif "2023" in year:
        #     samples = {
        #         "DY":  f"{base_path}/dyTo2L_M-50_incl",
        #         # "TT":  f"{base_path}/ttjets_*",
        #         "ggH": f"{base_path}/ggh_powhegPS",
        #         "VBF": f"{base_path}/vbf_powheg*",
        #     }
        else:
            samples = {
                "DY":  f"{base_path}/dyTo2L_M-50_incl",
                "Data":  f"{base_path}/data*",
                # "TT":  f"{base_path}/ttjets_*",
                "ggH": f"{base_path}/ggh_powhegPS",
                "VBF": f"{base_path}/vbf_powheg*",
            }
        df_yields = compute_weighted_yields(samples, region, year, weight_field="wgt_nominal")
        print(df_yields)
    
        df_compare = compare_yields(df_yields, df_yields30GeVCut, "baseline", "30GeVCut")
        df_compare.to_csv(f"df_compare_{year}_reg_{region}.csv")
        print(df_compare)
        df_compare_l.append(df_compare)
    
    df_total = pd.concat(df_compare_l)
    print(df_total)
    df_total.to_csv(f"df_total_{region}.csv")