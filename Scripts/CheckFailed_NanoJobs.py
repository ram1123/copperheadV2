import os
import pandas as pd
import uproot
import dask
from dask_gateway import Gateway
from dask.distributed import Client, progress, as_completed
import sys
import ROOT
import logging
import time

# ROOT Error Handling: Suppress non-critical warnings
ROOT.gErrorIgnoreLevel = ROOT.kError  # Only show errors, not warnings

# Set up logging for better tracking and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def get_num_events(file_path, tree_name="Events", max_attempts=10, sleep_interval=5.0):
    """
    Retrieve the number of events in a ROOT file using uproot.

    This function attempts to open the ROOT file and read the specified tree,
    returning the number of entries. As the file we are reading is a MiniAOD file,
    we expect it to have > 0 entries. If the number of entries is zero, the function
    will retry the operation up to 'max_attempts' times, waiting 'sleep_interval' seconds
    between attempts.

    Parameters:
        file_path (str): Path to the ROOT file.
        tree_name (str): Name of the tree to retrieve (default is "Events").
        max_attempts (int): Maximum number of attempts to retrieve a non-zero entry count (default is 10).
        sleep_interval (float): Time in seconds to wait between attempts (default is 0.1 seconds).

    Returns:
        int: Number of events in the ROOT file, or 0 if the file/tree is inaccessible or empty.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            with uproot.open(file_path) as f:
                if tree_name in f:
                    tree = f[tree_name]
                    num_entries = tree.num_entries
                    if num_entries != 0:
                        return num_entries
                    else:
                        logger.warning(f"Attempt {attempt}: Tree '{tree_name}' found but has 0 entries in {file_path}")
                else:
                    logger.warning(f"Tree '{tree_name}' not found in {file_path}")
                    return 0
        except Exception as e:
            logger.error(f"Attempt {attempt}/{max_attempts}: Error opening {file_path} with uproot: {e}")
        # Sleep between attempts if not the last try
        if attempt < max_attempts:
            time.sleep(sleep_interval)
    return 0


def get_num_entries_in_nanoAOD(file_path):
    """
    Get the number of entries in the NanoAOD file.
    This function assumes that the file contains an 'Events' tree.

    Parameters:
        file_path (str): Path to the NanoAOD ROOT file.

    Returns:
        int: Number of entries in the NanoAOD file.
    """
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


# Function to process missing & corrupt files using Dask for parallelization
def check_missing_files(input_file, output_dir, year, additional_string):
    """Process files to check for missing or corrupt files."""
    # Load the input file into a pandas DataFrame
    df = pd.read_csv(input_file, sep=' ', header=None, names=['configFile', 'inputMiniAOD', 'outputDirectory', 'nEvents', 'CondorLogPath'])

    # Generate the expected output file names
    df['outputNanoAODFile'] = df['inputMiniAOD'].apply(lambda x: x.split('/')[-1].replace(".root", "_NanoAOD.root"))
    df['expected_output_file'] = df['outputDirectory'] + "/" + df['outputNanoAODFile']

    # Set up Dask client (with improved resource management)
    gateway = Gateway(
        "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
        proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
    )
    cluster_info = gateway.list_clusters()[0]  # Get the first cluster by default. There should be only one cluster
    client = gateway.connect(cluster_info.name).get_client()
    logger.info(f"Dask client: {client}")

    # Add "root://xcache.cms.rcac.purdue.edu/" to the inputMiniAOD files
    df['inputMiniAOD'] = df['inputMiniAOD'].apply(lambda x: "root://xcache.cms.rcac.purdue.edu/" + x)

    # Scatter the files across workers before computation to avoid slow graph building
    input_files = df['inputMiniAOD'].tolist()
    output_files = df['expected_output_file'].tolist()

    # # Scatter input files and output files to Dask workers to reduce overhead
    # scattered_input_files = client.scatter(input_files)
    # scattered_output_files = client.scatter(output_files)

    # Create Dask delayed tasks for both fetching events from MiniAOD and entries from NanoAOD
    tasks = [
        dask.delayed(get_num_events)(file) for file in input_files
    ] + [
        dask.delayed(get_num_entries_in_nanoAOD)(file) for file in output_files
    ]

    # Compute the results in parallel using Dask
    results = dask.compute(*tasks)

    # Split results into the number of events (from MiniAOD) and entries (from NanoAOD)
    num_events_results_mini = results[:len(df['inputMiniAOD'])]
    num_events_results_nano = results[len(df['inputMiniAOD']):]

    # Store the results in the DataFrame
    df['nEvents_from_inputMiniAOD'] = num_events_results_mini
    df['nEvents_from_nanoAOD'] = num_events_results_nano

    # Filter out missing or corrupt files (where nEvents_from_nanoAOD is 0)
    missing_files_df = df[df['nEvents_from_nanoAOD'] == 0]  # If NanoAOD has 0 events, consider it corrupt
    missing_files_df = missing_files_df[['configFile', 'inputMiniAOD', 'outputDirectory', 'nEvents', 'CondorLogPath']]

    # Dynamically generate the output filename based on year and additional string
    output_filename = f'missing_or_corrupt_files_{year}_{additional_string}.txt'

    # Save the missing or corrupt files list to a file
    missing_files_df.to_csv(output_filename, sep=' ', header=False, index=False)

    # Save full df to CSV file
    df.to_csv(f'full_df_{year}_{additional_string}.csv', index=False)

    # Save another dataframe where nEntries_in_nanoAOD and nEvents_from_inputMiniAOD are not equal
    df_not_equal = df[df['nEvents_from_nanoAOD'] != df['nEvents_from_inputMiniAOD']]
    df_not_equal.to_csv(f'full_df_DifferentNEvents_{year}_{additional_string}.csv', sep=' ', header=False, index=False)

    df_not_equal = df_not_equal[['configFile', 'inputMiniAOD', 'outputDirectory', 'nEvents', 'CondorLogPath']]
    df_not_equal.to_csv(f'skim_df_DifferentNEvents_{year}_{additional_string}.csv', sep=' ', header=False, index=False)

    logger.info(f"{output_filename}: Missing or corrupt files: {len(missing_files_df)}")


# Main function to process all years
def main():
    """Main processing function."""
    years_and_input_files = {
        '2018Re': 'OriginalTxtFilesForNanoAODv12Production/HMuMu_UL2018_NanoAODv12_06March_Data_Run2018A.txt',
        # '2018': 'OriginalTxtFilesForNanoAODv12Production/HMuMu_UL2018_06March_AllJobs.txt',
        '2018': 'OriginalTxtFilesForNanoAODv12Production/HMuMu_UL2018_06March_AllJobs_grepDATA.txt',
        '2017': 'OriginalTxtFilesForNanoAODv12Production/HMuMu_UL2017_3Feb_AllJobs.txt',
        '2016APV': 'OriginalTxtFilesForNanoAODv12Production/HMuMu_UL2017_8Feb_2016APV.txt',
        '2016': 'OriginalTxtFilesForNanoAODv12Production/HMuMu_UL2017_8Feb_2016.txt',
        'TestGautschi': 'skim_df_DifferentNEvents_2018_14March_Gautschi.csv',
        'TestHammer': 'skim_df_DifferentNEvents_2018_14March_hammer.csv',
        'TestLxplus': 'skim_df_DifferentNEvents_2018_14March_Lxplus.csv',
    }

    years_and_output_dirs = {
        '2018Re': '/eos/purdue/store/user/rasharma/CustomNanoAODv12_v2/UL2018/',
        '2018': '/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/',
        '2017': '/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/',
        '2016APV': '/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/',
        '2016': '/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/',
        'TestGautschi': '/eos/purdue/store/user/rasharma/Test_Gautschi/UL2018/',
        'TestHammer': '/eos/purdue/store/user/rasharma/Test_Hammer/UL2018/',
        'TestLxplus': '/eos/purdue/store/user/rasharma/Test_Lxplus/UL2018/',
    }

    # years = ['2018', '2017', '2016APV', '2016']
    years = ['TestGautschi', 'TestHammer', 'TestLxplus']
    additional_string = "20March"

    for year in years:
        input_file = years_and_input_files[year]
        output_dir = years_and_output_dirs[year]
        check_missing_files(input_file, output_dir, year, additional_string)


if __name__ == "__main__":
    main()
