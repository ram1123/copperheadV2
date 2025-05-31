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


def renameColumns(df):
    df.columns = [col.replace('m1', 'mu1_') for col in df.columns]
    df.columns = [col.replace('m2', 'mu2_') for col in df.columns]
    df.columns = [col.replace('j1', 'jet1_') for col in df.columns]
    df.columns = [col.replace('j2', 'jet2_') for col in df.columns]
    df.columns = [col.replace('_nominal', '') for col in df.columns]
    return df


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
    # rename Adish columns to match my name scheme
    
    my_df = renameColumns(my_df)
    Adish_df = renameColumns(Adish_df)
    print(f'Adish_df: {Adish_df.columns}')
    print(f'my_df: {my_df.columns}')
    
    fields2plot = [
        "pt",
        "eta",
        "phi"
    ]
    particles = [
        "mu1",
        "mu2",
        "jet1",
        "jet2",
        
    ]
    plot_path = "plots"
    for field in fields2plot:
        for particle in particles:
            var_name = f"{particle}_{field}"
            print(f'my_df: {my_df[var_name]}')
            print(f'Adish_df: {Adish_df[var_name]}')
            diff = my_df[var_name] - Adish_df[var_name]
            
            print(f'diff: {diff}')
            plt.clf()
            plt.hist(diff, bins=50, histtype='step', label='Data')
            # Add labels and title
            plt.xlabel('Value')
            plt.ylabel('Entries')
            plt.title(var_name+" diff")
            plt.legend()
            plt.savefig(f"{plot_path}/diff_{var_name}.png")
            
        
            # relative diff
            plt.clf()
            diff = diff/my_df[var_name] *100 # multiply 100 to get percentage
            plt.hist(diff, bins=50, histtype='step', label='Data')
            plt.xlabel('percentage diff %')
            plt.ylabel('Entries')
            plt.title(var_name+" diff")
            plt.legend()
            plt.savefig(f"{plot_path}/diff_{var_name}_relative.png")
    