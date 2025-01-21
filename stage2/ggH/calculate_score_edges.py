import dask_awkward as dak 
import awkward as ak
import numpy as np
import argparse
from omegaconf import OmegaConf




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
    sysargs = parser.parse_args()


    # full_load_path = "/depot/cms/users/yun79/hmm/copperheadV1clean/V2_Jan17_JecDefault_valerieZpt/ggh/stage2_output/ggh/2018/processed_events_sig*.parquet"
    # extract only the signal samples (VBF and ggH)
    # full_load_path = f"{sysargs.load_path}/{sysargs.year}/processed_events_sig*.parquet"
    full_load_path = f"{sysargs.load_path}/{sysargs.year}/processed_events_sigMC_ggh.parquet" # ignore VBF signal sample
    # full_load_path = f"{sysargs.load_path}/ggh/{sysargs.year}/processed_events_sig*.parquet"
    events = dak.from_parquet(full_load_path)
    
    signal_score = ak.to_numpy(events.BDT_score.compute())
    signal_wgt = ak.to_numpy(events.wgt_nominal.compute())
    signal_wgt = signal_wgt /np.sum(signal_wgt) # normalize wgt
    target_yields = [0.3, 0.35, 0.15, 0.15, 0.05]
    print(sum(target_yields))
    target_yields_cum_sum = np.cumsum(np.array(target_yields))
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
            # print(f"target_yields_cum_sum[target_index] : {target_yields_cum_sum[target_index]}")
            # print(f"current_yield : {current_yield}")
            bin_edges.append(sorted_data[i])
            target_index += 1
            if target_index >= len(target_yields_cum_sum):
                break
    
    
    if len(bin_edges) < (len(target_yields) +1 ):
        bin_edges.append(1.1) # add the last bin edge to maximum value and a bit
    else:
        bin_edges[-1] = 1.1 # switch the last bin edge to maximum value and a bit
    print("Bin edges:", bin_edges)
    hist, _ = np.histogram(signal_score, bins=bin_edges, weights=signal_wgt)
    # sanity check
    print(f"original target_yields: {target_yields}")
    print(f"binnning histogram distribution: {hist}")
    print(f"np.sum(hist): {np.sum(hist)}")
    signal_subcat_idx = np.digitize(signal_score, bin_edges) -1 # digitize starts at one, not zero
    print(f"signal_subcat_idx: {signal_subcat_idx}")
    print(f"np.max(signal_score): {np.max(signal_score)}")


    # save the new bin edges
    config_path = f"/work/users/yun79/valerie/fork/copperheadV2/configs/MVA/ggH/BDT_edges.yaml"
    # Load the config file
    config = OmegaConf.load(config_path)
    print(f"old config: {config}")
    bin_edges = [float(value) for value in bin_edges] # need to convert to 32-bit b4 writing to omegaconf
    config[sysargs.year] = bin_edges 
    print(f"new config: {config}")
    # Overwrite the yaml file
    OmegaConf.save(config, config_path)
    