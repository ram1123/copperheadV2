import dask_awkward as dak 
import awkward as ak
import numpy as np
import argparse
from omegaconf import OmegaConf
import time


def calculate_AMS(sig_yields, bkg_yields):
    """
    calculate the combined AMS of multiple bins. 
    We assume the shape of sig_yields and bkg_yields are same and share the same bins
    (ie signal yield in bin1 is from the same bin as background yield in bin2)
    both sig_yields and bkg_yields are np like 1-D array
    """
    assert len(sig_yields) == len(bkg_yields)
    
    ams_sum = 0

    for ix in range(len(sig_yields)):
        S = sig_yields[ix]
        B = bkg_yields[ix]
        AMS = (S + B) * np.log(1 + S/B) - S
        AMS = np.sqrt(2*AMS)
        ams_sum += AMS
        
    combined_ams = np.sqrt(ams_sum) # add by quadrature
    return combined_ams

def obtain_BDT_edges(target_yields_cum_sum, years, load_path):
    score_edge_dict = {}
    for year in years:
    

        # full_load_path = f"{sysargs.load_path}/{sysargs.year}/processed_events_sig*.parquet"
        full_load_path = f"{load_path}/{year}/processed_events_sigMC_ggh.parquet" # ignore VBF signal sample
        events = dak.from_parquet(full_load_path)
        
        signal_score = ak.to_numpy(events.BDT_score.compute())
        signal_wgt = ak.to_numpy(events.wgt_nominal.compute())
        signal_wgt = signal_wgt /np.sum(signal_wgt) # normalize wgt
        
        
        # sort data
        sorted_indices = np.argsort(signal_score)
        signal_score = signal_score[sorted_indices]
        signal_wgt =signal_wgt[sorted_indices]
        print(f"target_yields_cum_sum: {target_yields_cum_sum}")
    
        # Compute cumulative weights
        cumulative_weights = np.cumsum(signal_wgt)
        # Find bin edges
        sorted_data = signal_score
        bin_edges = [0.0]  # Start with the minimum value
        current_yield = 0
        target_index = 0
        
        for i, cum_weight in enumerate(cumulative_weights):
            current_yield = cum_weight 
            # print(f"current_yield : {current_yield}")
            if current_yield >= target_yields_cum_sum[target_index]:
                bin_edges.append(sorted_data[i])
                target_index += 1
                if target_index >= len(target_yields_cum_sum):
                    break
        
        
        if len(bin_edges) < (len(target_yields_cum_sum) +1 ):
            bin_edges.append(1.1) # add the last bin edge to maximum value and a bit
        else:
            bin_edges[-1] = 1.1 # switch the last bin edge to maximum value and a bit
        print("Bin edges:", bin_edges)
        # raise ValueError
        hist, _ = np.histogram(signal_score, bins=bin_edges, weights=signal_wgt)

        
        # sanity check
        target_yields = np.diff(target_yields_cum_sum, prepend=0)
        print(f"target_yields: {target_yields}")
        print(f"binnning histogram distribution: {hist}")
        print(f"np.sum(hist): {np.sum(hist)}")
        signal_subcat_idx = np.digitize(signal_score, bin_edges) -1 # digitize starts at one, not zero
        print(f"signal_subcat_idx: {signal_subcat_idx}")
        print(f"np.max(signal_score): {np.max(signal_score)}")

        score_edge_dict[year] = bin_edges 
    
    # print(f"score_edge_dict: {score_edge_dict}")
    return score_edge_dict


def get_signal_yields(bdt_score_edges, year:str, load_path:str):
    """

    return: out_arr of size len(bdt_score_edges) -1, value in each bin represnting signal yield in that category
    """
    full_load_path = f"{load_path}/{year}/processed_events_sigMC*.parquet"  # include all signal
    events = dak.from_parquet(full_load_path)
    signal_score = ak.to_numpy(events.BDT_score.compute())
    signal_wgt = ak.to_numpy(events.wgt_nominal.compute())

    subCat_idx = np.digitize(signal_score, bdt_score_edges) -1 # idx starts with 0
    print(f"np.max(subCat_idx): {np.max(subCat_idx)}")
    print(f"np.min(subCat_idx): {np.min(subCat_idx)}")

    out_arr = np.zeros(len(bdt_score_edges)-1)

    for ix in range(len(out_arr)):
        cat_filter = (ix == subCat_idx)
        cat_wgt_sum = np.sum(signal_wgt[cat_filter])
        out_arr[ix] = cat_wgt_sum
        
    print(f"{year} np.sum(out_arr): {np.sum(out_arr)}")
    return out_arr

def get_background_yields(bdt_score_edges, year:str, load_path:str):
    """
    return: out_arr of size len(bdt_score_edges) -1, value in each bin represnting signal yield in that category
    """
    full_load_path = f"{load_path}/{year}/processed_events_data.parquet"  # use data for bkg
    events = dak.from_parquet(full_load_path)
    background_score = ak.to_numpy(events.BDT_score.compute())
    background_wgt = ak.to_numpy(events.wgt_nominal.compute())

    subCat_idx = np.digitize(background_score, bdt_score_edges) -1 # idx starts with 0
    print(f"np.max(subCat_idx): {np.max(subCat_idx)}")
    print(f"np.min(subCat_idx): {np.min(subCat_idx)}")

    out_arr = np.zeros(len(bdt_score_edges)-1)

    for ix in range(len(out_arr)):
        cat_filter = (ix == subCat_idx)
        cat_wgt_sum = np.sum(background_wgt[cat_filter])
        out_arr[ix] = cat_wgt_sum
        
    print(f"{year} np.sum(out_arr): {np.sum(out_arr)}")
    return out_arr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="2018",
    action="store",
    help="Year to process (2016preVFP, 2016postVFP, 2017 or 2018)",
    )
    parser.add_argument(
    "-load",
    "--load_path",
    dest="load_path",
    default="",
    action="store",
    help="path were stage2 output is saved",
    )

    start_time = time.time()
    """
    pseudocode:

    obtain a list of ggH signal effs

    obtain BDT edges are the target signal effs for each era
    for each era: 
    get BDT score edges
    
    obtain the signal yields (ggH + VBF) and bkg yields (data) for each bin
    obtain combined AMS for said bin
    repeat

    plot the resulting AMS

    
    """




    
    sysargs = parser.parse_args()

    load_path = sysargs.load_path

    # target_yields = [0.30457106, 0.35325641, 0.14842342, 0.13939539, 0.05435372]
    # target_yields_cum_sum = np.cumsum(np.array(target_yields))
    # print(f"target_yields len: {len(target_yields)}")
    # print(f"target_yields_cum_sum len: {len(target_yields_cum_sum)}")

    target_yields_cum_sum = np.array([0.5,1.0])
    print(f"target_yields_cum_sum: {target_yields_cum_sum}")
    # years = ["2018"]
    years = ["2016preVFP", "2016postVFP", "2017", "2018"]

    BDT_score_edge_dict = obtain_BDT_edges(target_yields_cum_sum, years, load_path)
    print(f"BDT_score_edge_dict: {BDT_score_edge_dict}")

    signal_arrs = []
    background_arrs = []
    for year, bdt_score_edges in BDT_score_edge_dict.items():
        signal_arr = get_signal_yields(bdt_score_edges, year, load_path)
        signal_arrs.append(signal_arr)
        background_arr = get_background_yields(bdt_score_edges, year, load_path)
        background_arrs.append(background_arr)

    print(f"signal_arrs: {signal_arrs}")
    signal_yield = sum(signal_arrs)
    print(f"signal_yield: {signal_yield}")
    print(f"signal_yield sum: {np.sum(signal_yield)}")


    print(f"background_arrs: {background_arrs}")
    background_yield = sum(background_arrs)
    print(f"background_yield: {background_yield}")
    print(f"background_yield sum: {np.sum(background_yield)}")
    
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")

    # # save the new bin edges
    # config_path = f"/work/users/yun79/valerie/fork/copperheadV2/configs/MVA/ggH/BDT_edges.yaml"
    # # Load the config file
    # config = OmegaConf.load(config_path)
    # print(f"old config: {config}")
    # bin_edges = [float(value) for value in bin_edges] # need to convert to 32-bit b4 writing to omegaconf
    # config[sysargs.year] = bin_edges 
    # print(f"new config: {config}")
    # # Overwrite the yaml file
    # OmegaConf.save(config, config_path)
    