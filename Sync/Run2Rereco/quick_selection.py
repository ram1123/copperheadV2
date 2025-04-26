import pandas as pd
import dask.dataframe as dd
import numpy as np
import numpy as np
import awkward as ak
import glob
import dask_awkward as dak
from distributed import LocalCluster, Client, progress
import matplotlib.pyplot as plt

def filter_df(df, lumi, run):
    lumi_filter = df["lumi"] == lumi
    run_filter = df["run"] == run
    return df[lumi_filter & run_filter]
if __name__ == "__main__":

    client =  Client(n_workers=60,  threads_per_worker=1, processes=True, memory_limit='8 GiB') 
    """
    Load event dump df from Adish and compare
    """
    Adish_df = pd.read_csv("EventDump_Adish_v2.csv")
    df_unique_pairs = Adish_df[["run", "lumi"]].drop_duplicates()
    
    
    
    year = "2017"
    label="Run2Rereco_synch_Apr23_2025"
    load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean/{label}/stage1_output/{year}"
    
    my_df = dd.read_parquet(f'{load_path}/data_B/*.parquet')
    fields2compute = [
        "run",
        "luminosityBlock",
        "event",
        "mu1_pt",
        "mu1_eta",
        "mu1_phi",
        "mu2_pt",
        "mu2_eta",
        "mu2_phi",
        "dimuon_mass",
        "jet1_pt_nominal",
        "jet1_eta_nominal",
        "jet1_phi_nominal",
        "jet2_pt_nominal",
        "jet2_eta_nominal",
        "jet2_phi_nominal",
        "jj_mass_nominal",
        "nBtagMedium_nominal",
    ]
    print(my_df.columns)
    my_df = my_df[fields2compute]
    # my_df = my_df[my_df["event"]==296503858].compute()
    # my_df
    my_df = my_df.dropna(subset=["mu1_pt"])
    
    my_df = my_df.rename(columns={"luminosityBlock": "lumi"})
    my_df = my_df.compute()


    # filter lumi and run values you want and reset index
    lumi_target = 160
    run_target = 297292
    my_df = filter_df(my_df, lumi_target, run_target)
    Adish_df = filter_df(Adish_df, lumi_target, run_target)
    my_df = my_df.reset_index(drop=True)
    Adish_df = Adish_df.reset_index(drop=True)
    print(f'my_df["mu1_pt"]: {my_df["mu1_pt"]}')
    print(f'Adish_df["m1pt"]: {Adish_df["m1pt"]}')
    diff = my_df["mu1_pt"] - Adish_df["m1pt"]
    
    print(f'diff: {diff}')
    
    plt.hist(diff, bins=50, histtype='step', label='Data')
    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Entries')
    plt.title('mu1_pt')
    plt.legend()
    plt.savefig(f"test.png")
    

    # relative diff
    plt.clf()
    diff = diff/my_df["mu1_pt"] *100 # multiply 100 to get percentage
    plt.hist(diff, bins=50, histtype='step', label='Data')
    plt.xlabel('percentage diff %')
    plt.ylabel('Entries')
    plt.title('mu1_pt')
    plt.legend()
    plt.savefig(f"test_relative.png")
    