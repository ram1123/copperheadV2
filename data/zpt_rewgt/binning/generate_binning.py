import dask_awkward as dak
import numpy as np
import awkward as ak
import argparse
import sys
from distributed import LocalCluster, Client, progress
np.set_printoptions(threshold=sys.maxsize)
import os
from omegaconf import OmegaConf
import copy

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

    # mu1_pt = events.mu1_pt
    # mu1ptOfInterest = (mu1_pt > 75) & (mu1_pt < 150.0)
    # events = events[region&mu1ptOfInterest]
    events = events[region]
    return events

def ranges_overlap(range1, range2):
    """
    Check if two ranges overlap.

    Parameters:
    range1 (tuple): A tuple representing the first range (start1, end1).
    range2 (tuple): A tuple representing the second range (start2, end2).

    Returns:
    bool: True if the ranges overlap, False otherwise.
    """
    start1, end1 = range1
    start2, end2 = range2
        
    # Check for overlap
    return max(start1, start2) < min(end1, end2)

def getMidpoint(a: float, b: float) -> float:
    """
    Calculate the midpoint of two float values.

    :param a: The first float value.
    :param b: The second float value.
    :return: The midpoint of the two values.
    """
    return (a + b) / 2

def checkNewCharacteristic(new_hist, new_hist_err):
    assert len(new_hist) == 2 # we assume new_hist has size of two
    # obtain up and down histogram ranges
    lower_bin_down_up = (
        new_hist[0] - new_hist_err[0],
        new_hist[0] + new_hist_err[0] 
    )
    upper_bin_down_up = (
        new_hist[1] - new_hist_err[1],
        new_hist[1] + new_hist_err[1] 
    )
    # check if either lower bin edge range or upper one has zero difference. Then return False
    if abs(lower_bin_down_up[0]-lower_bin_down_up[1]) ==0:
        return False
    elif abs(upper_bin_down_up[0]-upper_bin_down_up[1]) ==0:
        return False
    
    no_range_overlap = not ranges_overlap(lower_bin_down_up, upper_bin_down_up) # no range overlap means there was a characateristic we missed
    new_characteristic = no_range_overlap

    # print(f"lower_bin_down_up: {lower_bin_down_up}")
    # print(f"upper_bin_down_up: {upper_bin_down_up}")
    # print(f"no_range_overlap: {no_range_overlap}")
    return new_characteristic

def getSF_hist(data_event, dy_event, binning):
    """
    return SF (Data/DY ratio) histogram with its error
    """
    # obtain histogram for data
    values = data_event["dimuon_pt"]
    weights = data_event["wgt_nominal"]
    data_hist, _ = np.histogram(values, bins=binning, weights = weights)
    data_hist_w2, _ = np.histogram(values, bins=binning, weights = weights*weights)
    data_hist = ak.to_numpy(data_hist)
    data_hist_err = np.sqrt(ak.to_numpy(data_hist_w2))
    # obtain histogram for dy
    values = dy_event["dimuon_pt"]
    weights = dy_event["wgt_nominal"]
    dy_hist, _ = np.histogram(values, bins=binning, weights = weights)
    dy_hist_w2, _ = np.histogram(values, bins=binning, weights = weights*weights)
    dy_hist = ak.to_numpy(dy_hist)
    dy_hist_err = np.sqrt(ak.to_numpy(dy_hist_w2))
    
    # initialize ratio histogram and fill in values
    ratio_hist = np.zeros_like(data_hist)
    inf_filter = dy_hist>0
    ratio_hist[inf_filter] = data_hist[inf_filter]/  dy_hist[inf_filter]
    # add relative uncertainty of data and bkg_mc by adding by quadrature
    rel_unc_ratio = np.sqrt((dy_hist_err/dy_hist)**2 + (data_hist_err/data_hist)**2)
    ratio_err = rel_unc_ratio*ratio_hist
    return ratio_hist, ratio_err

if __name__ == "__main__":
    """
    This file is meant to define the Zpt histogram binning for zpt fitting
    """
    cluster = LocalCluster(processes=True)
    cluster.adapt(minimum=8, maximum=31) #min: 8 max: 32
    client = Client(cluster)
    print("Local scale Client created")

    run_label = "V2_Jan09_ForZptReWgt"
    # run_label = args.label
    # year = "2018"
    # year = "2017"
    # year = "2016postVFP"
    year = "2016preVFP"
    base_path = f"/depot/cms/users/yun79/hmm/copperheadV1clean/{run_label}/stage1_output/{year}/f1_0" # define the save path of stage1 outputs
    
    
    # load the data and dy samples
    data_events = dak.from_parquet(f"{base_path}/data_*/*/*.parquet")
    dy_events = dak.from_parquet(f"{base_path}/dy_M-50/*/*.parquet")
    
    # apply z-peak region filter and nothing else
    data_events = filterRegion(data_events, region="z-peak")
    dy_events = filterRegion(dy_events, region="z-peak")
    
    njet_field = "njets_nominal"
    for njet in [0,1,2]:
    # for njet in [0]:
        if njet != 2:
            data_events_loop = data_events[data_events[njet_field] ==njet]
            dy_events_loop = dy_events[dy_events[njet_field] ==njet]
        else:
            data_events_loop = data_events[data_events[njet_field] >=njet]
            dy_events_loop = dy_events[dy_events[njet_field] >=njet]

        xmax = 200
        xmin = 0
        initial_bins = np.linspace(xmin, xmax, 5)
        
        old_bins = initial_bins
        current_bins = copy.deepcopy(old_bins)
        new_bins = copy.deepcopy(current_bins)
        # loop over old bins and divide them into two equal bins
        print(f"current_bins: {current_bins}")
        
        bin_has_changed = True
        fields2load = ["wgt_nominal", "dimuon_pt"]
        data_dict = {field: data_events_loop[field].compute() for field in fields2load}
        dy_dict = {field: dy_events_loop[field].compute() for field in fields2load}

        bin_values_already_tested = []

        while True:
            bin_has_changed = False # make this false until flipped True
            print(f"njet {njet} loop start ---------------------------------------------------------------------")
            print(f"current_bins length: {len(current_bins)}")
            print(f"current_bins: {', '.join(map(str, current_bins.tolist()))}")
            
            for bin_idx in range(len(current_bins)-1):
                bin_low_edge = current_bins[bin_idx]
                bin_high_edge = current_bins[bin_idx+1]
                
                bin_mid = getMidpoint(bin_low_edge, bin_high_edge)
                # check if this bin_mid value has been already tested and if so, skip
                if bin_mid in bin_values_already_tested:
                    print(f"{bin_mid} has been already tested. Skipping!")
                    continue
                
                # Make new Binning and plot histogram
                new_binning = np.array([bin_low_edge, bin_mid, bin_high_edge])
                # new_hist, edges = np.histogram(data, bins=new_binning)
                # new_hist_err = np.sqrt(new_hist)
                new_hist, new_hist_err = getSF_hist(data_dict, dy_dict, new_binning)
                new_charaacteristic = checkNewCharacteristic(new_hist, new_hist_err)
                print(f"bin_low_edge: {bin_low_edge}")
                # print(f"bin_high_edge: {bin_high_edge}")
                # print(f"bin_mid: {bin_mid}")
                # print(f"new_hist: {new_hist}")
                # print(f"new_hist_err: {new_hist_err}")
                # print(f"edges: {edges}")
                # if new binning leads to new characateristic, keep new binning
                if new_charaacteristic:
                    # add new bin edge and sort
                    new_bins = list(new_bins) + [bin_mid]
                    new_bins = list(set(new_bins)) # remove any redundant values as sanity check
                    new_bins = np.array(sorted(new_bins)) 
                    print(f"adding edge {bin_mid}")
                    # print(f"new_bins: {new_bins}")
                    bin_has_changed = True 
                else:
                    print(f"NOT adding edge {bin_mid}")
                    bin_values_already_tested.append(bin_mid)
            
            # repeat until no new bin edge has been added, then end loop
            if bin_has_changed:
                current_bins = new_bins
            else: 
                print("No new bins were found. Ending Loop!")
                print(f"njet {njet} final binning: {current_bins}")
                break # end loop of no bin has changed

        """
        Test one more time to make sure no characteristic was lost
        """
        print(f"current_bins: {', '.join(map(str, current_bins.tolist()))}")
        
        for bin_idx in range(len(current_bins)-1):
            bin_low_edge = current_bins[bin_idx]
            bin_high_edge = current_bins[bin_idx+1]
            
            bin_mid = getMidpoint(bin_low_edge, bin_high_edge)
            # check if this bin_mid value has been already tested and if so, skip
            
            # Make new Binning and plot histogram
            new_binning = np.array([bin_low_edge, bin_mid, bin_high_edge])
            # new_hist, edges = np.histogram(data, bins=new_binning)
            # new_hist_err = np.sqrt(new_hist)
            new_hist, new_hist_err = getSF_hist(data_dict, dy_dict, new_binning)
            new_charaacteristic = checkNewCharacteristic(new_hist, new_hist_err)
            if new_charaacteristic:
                print(f"Validation test failed at bin mid value {bin_mid}")
                raise ValueError

        """
        If passed the test, save the binning
        """
        # # Specify the directory path
        # save_path = "binning"
        # # Create the directory if it doesn't exist
        # os.makedirs(save_path, exist_ok=True)
        
        binning_path = f"{year}_njet{njet}.yml"
        data = {"rewgt_binning": current_bins.tolist()}
        # Save the data to a YAML file
        OmegaConf.save(config=data, f=binning_path)

