import os
import pandas as pd
import uproot
import sys
import ROOT
import logging
import time
import argparse
from dask import delayed, compute
from dask.distributed import Client
import dask.dataframe as dd
import hepconvert
import uuid

# ROOT Error Handling: Suppress non-critical warnings
ROOT.gErrorIgnoreLevel = ROOT.kError  # Only show errors, not warnings

# Set up logging for better tracking and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Argument parsing for making Dask optional
parser = argparse.ArgumentParser(description="Process ROOT files with or without Dask.")
parser.add_argument('--use-dask', action='store_true', help="Use Dask for parallel processing")
parser.add_argument('--use-gateway', action='store_true', help="Use Dask Gateway for cluster mode")
args = parser.parse_args()

if args.use_dask and args.use_gateway:
    print("Error: You cannot use both --use-dask and --use-gateway at the same time.")
    sys.exit(1)

def create_dask_client():
    """Creates and returns a Dask client based on input arguments."""
    if args.use_gateway:
        from dask_gateway import Gateway
        gateway = Gateway(
            "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
            proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
        )
        cluster_info = gateway.list_clusters()[0]  # Get the first cluster by default
        client = gateway.connect(cluster_info.name).get_client()
        print(f"client: {client}")
        print("Gateway Client created")
    else:
        client = Client(n_workers=11, threads_per_worker=1, processes=True, memory_limit='90 GiB')
        print("Local scale Client created")
    return client


def get_num_entries_in_nanoAOD(file_path):
    """
    Get the number of entries in the NanoAOD file.
    This function assumes that the file contains an 'Events' tree.

    Parameters:
        file_path (str): Path to the NanoAOD ROOT file.

    Returns:
        int: Number of entries in the NanoAOD file.
    """
    # check if the file starts with root:// if not add the prefix root://xcache.cms.rcac.purdue.edu/
    # if not file_path.startswith("root://"):
        # file_path = "root://xcache.cms.rcac.purdue.edu/" + file_path
        # file_path = "root://cms-xrd-global.cern.ch//" + file_path
        # pass
    try:
        file = ROOT.TFile.Open(file_path, "READ")
        if file and file.IsOpen():
            tree = file.Get("Events")
            if tree:
                num_entries = tree.GetEntries()
                file.Close()
                return num_entries
            else:
                logger.warning(f"Tree 'Events' not found in {file_path}.")
                file.Close()
                return 0
        else:
            logger.warning(f"ROOT file '{file_path}' failed to open.")
            return 0
    except Exception as e:
        logger.error(f"Error opening ROOT file '{file_path}': {e}")
        return 0

def get_num_entries_in_nanoAOD_uproot(file_path, ifMiniAOD=False):
    """
    Get the number of entries in the NanoAOD file using uproot.
    This function assumes that the file contains an 'Events' tree.

    Parameters:
        file_path (str): Path to the NanoAOD ROOT file.
        ifMiniAOD (bool): Flag to indicate if the file is MiniAOD.

    Returns:
        int: Number of entries in the NanoAOD file.
    """
    retries = 5 if ifMiniAOD else 1  # Retry up to 5 times if ifMiniAOD is True
    for attempt in range(retries):
        if attempt == 2:
            file_path = file_path.replace("root://xcache.cms.rcac.purdue.edu/", "root://cms-xrd-global.cern.ch/")
        elif attempt > 2:
            file_path = file_path.replace("root://cms-xrd-global.cern.ch/", "root://xrootd-cms.infn.it/")
        try:
            with uproot.open(file_path) as file:
                tree = file["Events"]
                num_entries = tree.num_entries
                if num_entries > 0 or not ifMiniAOD:
                    return num_entries
                else:
                    logger.warning(f"Attempt {attempt + 1}/{retries}: No entries found in '{file_path}'. Retrying...")
        except Exception as e:
            logger.error(f"Error opening ROOT file '{file_path}' on attempt {attempt + 1}/{retries}: {e}")
        time.sleep(1)  # Optional: Add a delay between retries
    logger.error(f"Failed to retrieve entries from '{file_path}' after {retries} attempts.")
    return 0


def check_root_to_parquet(file_path):
    """
    Check if the ROOT file can be converted to Parquet format using hepconvert.

    Parameters:
        file_path (str): Path to the ROOT file.

    Returns:
        bool: True if conversion is successful, False otherwise.
    """
    try:
        tmp_path = f"/tmp/{uuid.uuid4().hex}.parquet"
        hepconvert.root_to_parquet(
            in_file=str(file_path),
            out_file=tmp_path,
            tree="Events",
            keep_branches=["dimuon_pt"],
            force=True,
        )
        os.remove(tmp_path)  # Clean up temp file
        return True
    except Exception as e:
        logger.error(f"Error converting ROOT file '{file_path}' to Parquet: {e}")
        return False


def check_missing_files(input_file, output_dir, year, additional_string, client):
    """Process files to check for missing or corrupt files."""
    # Load the input file into a pandas DataFrame
    df = pd.read_csv(input_file, sep=' ', header=None, names=['configFile', 'inputMiniAOD', 'outputDirectory', 'nEvents', 'CondorLogPath'])

    # Generate the expected output file names: mini.root -> mini_NanoAOD.root
    df['outputNanoAODFile'] = df['inputMiniAOD'].apply(lambda x: x.split('/')[-1].replace(".root", "_NanoAOD.root"))
    df['expected_output_file'] = df['outputDirectory'] + "/" + df['outputNanoAODFile']

    # Add "root://xcache.cms.rcac.purdue.edu/" or "root://cms-xrd-global.cern.ch/" to the inputMiniAOD files
    df['inputMiniAOD'] = df['inputMiniAOD'].apply(lambda x: "root://xcache.cms.rcac.purdue.edu/" + x)
    # df['inputMiniAOD'] = df['inputMiniAOD'].apply(lambda x: "root://cms-xrd-global.cern.ch/" + x)


    # input_files = df['inputMiniAOD'].tolist()
    # output_files = df['expected_output_file'].tolist()

    # Use Dask delayed tasks to process files concurrently
    # results = []
    # for file in input_files:
    #     results.append(delayed(get_num_entries_in_nanoAOD)(file))
    # for file in output_files:
    #     results.append(delayed(get_num_entries_in_nanoAOD)(file))

    # with pd.option_context('display.max_colwidth', None):
    #     display(df)
    # print(df.head(1))
    # ddf = dd.from_pandas(df, npartitions=1)
    # print(ddf.head(1))

    # Compute the results
    print("Goint to compute the results")
    # results = compute(*results)
    task1 = [delayed(get_num_entries_in_nanoAOD_uproot)(file, ifMiniAOD=True) for file in df['inputMiniAOD']]
    results_mini = compute(*task1)

    task2 = [delayed(get_num_entries_in_nanoAOD_uproot)(file) for file in df['expected_output_file']]
    results_nano = compute(*task2)


    # Split results into the number of events (from MiniAOD) and entries (from NanoAOD)
    # num_events_results_mini = results[:len(input_files)]
    # num_events_results_nano = results[len(input_files):]

    # Store the results in the DataFrame
    df['nEvents_from_inputMiniAOD'] = results_mini
    df['nEvents_from_nanoAOD'] = results_nano

    # task3 : use check_root_to_parquet for only those files for which nEvents_from_nanoAOD is not 0. Then add the output to the original dataframe

    # Initialize the 'parquet_conversion_success' column with False
    df['parquet_conversion_success'] = False

    # # Filter files where nEvents_from_nanoAOD is not 0
    # files_to_convert = df[df['nEvents_from_nanoAOD'] != 0]['expected_output_file']

    # # Use Dask delayed tasks to process these files concurrently
    # conversion_tasks = [delayed(check_root_to_parquet)(file) for file in files_to_convert]
    # conversion_results = compute(*conversion_tasks)

    # # Update the DataFrame with the conversion results
    # df.loc[df['nEvents_from_nanoAOD'] != 0, 'parquet_conversion_success'] = conversion_results


    # Get CSV files
    csv_file = f"AllFiles_{year}_{additional_string}.csv"
    df.to_csv(csv_file, index=False)

    # Get CSV files with miniAOD and NanoAOD events done't match or NanoAOD is 0
    df_mismatch = df[(df['nEvents_from_inputMiniAOD'] != df['nEvents_from_nanoAOD']) | (df['nEvents_from_nanoAOD'] == 0)]
    df_mismatch.to_csv(f"Mismatch_MiniNano_nNano0_{year}_{additional_string}.csv", index=False)


    df_mismatch = df_mismatch[['configFile', 'inputMiniAOD', 'outputDirectory', 'nEvents', 'CondorLogPath']]
    df_mismatch.to_csv(f"Mismatch_MiniNano_nNano0_{year}_{additional_string}_configFile.txt", sep=' ', header=False, index=False)

    # Get CSV files with miniAOD and NanoAOD events done't match or NanoAOD is 0 or miniAOD is 0
    df_mismatch = df[(df['nEvents_from_inputMiniAOD'] != df['nEvents_from_nanoAOD']) | (df['nEvents_from_nanoAOD'] == 0) | (df['nEvents_from_inputMiniAOD'] == 0)]
    df_mismatch.to_csv(f"Mismatch_MiniNano_nNano0_nMini0_{year}_{additional_string}.csv", index=False)

    df_mismatch = df_mismatch[['configFile', 'inputMiniAOD', 'outputDirectory', 'nEvents', 'CondorLogPath']]
    df_mismatch.to_csv(f"Mismatch_MiniNano_nNano0_nMini0_{year}_{additional_string}_configFile.txt", sep=' ', header=False, index=False)


def main():
    """Main processing function."""
    years_and_input_files = {
        '2018GT36': 'OriginalTxtFilesForNanoAODv12Production/UL2018-GT36.txt',
        '2018GT36_debug': 'OriginalTxtFilesForNanoAODv12Production/UL2018-GT36_debug.txt',
        '2018Re': 'OriginalTxtFilesForNanoAODv12Production/HMuMu_UL2018_NanoAODv12_06March_Data_Run2018A.txt',
        '2018': 'OriginalTxtFilesForNanoAODv12Production/HMuMu_UL2018_06March_AllJobs_grepDATA.txt',
        '2017': 'OriginalTxtFilesForNanoAODv12Production/HMuMu_UL2017_3Feb_AllJobs.txt',
        '2016APV': 'OriginalTxtFilesForNanoAODv12Production/HMuMu_UL2017_8Feb_2016APV.txt',
        '2016': 'OriginalTxtFilesForNanoAODv12Production/HMuMu_UL2017_8Feb_2016.txt',
    }

    years_and_output_dirs = {
        '2018GT36': '/eos/purdue/store/user/rasharma/customNanoAOD_GT36/UL2018/',
        '2018GT36_debug': '/eos/purdue/store/user/rasharma/customNanoAOD_GT36/UL2018/',
        '2018Re': '/eos/purdue/store/user/rasharma/CustomNanoAODv12_v2/UL2018/',
        '2018': '/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/',
        '2017': '/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/',
        '2016APV': '/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/',
        '2016': '/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/',
    }

    # years = ['2018']
    # year = ['2018Re', '2018', '2017', '2016APV', '2016']
    # years = ['2018', '2017', '2016APV', '2016']
    years = ['2018GT36_debug']
    # years = ['2018GT36']
    # additional_string = "4April_GlobalRedirector"
    # additional_string = "4April_Xcache"
    # additional_string = "4April_local_retries"
    additional_string = "4April_debug"
    # additional_string = "4April_AllYears"

    client = create_dask_client()  # Initialize Dask client

    for year in years:
        input_file = years_and_input_files[year]
        output_dir = years_and_output_dirs[year]
        check_missing_files(input_file, output_dir, year, additional_string, client)


if __name__ == "__main__":
    main()
