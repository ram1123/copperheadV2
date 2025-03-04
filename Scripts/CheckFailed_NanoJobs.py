import os
import pandas as pd
import uproot
import dask
from dask_gateway import Gateway
from dask.distributed import Client, progress
import sys
import ROOT
import logging

ROOT.gErrorIgnoreLevel = ROOT.kError  # Only show errors, not warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Function to check if a ROOT file is corrupt using uproot
def is_root_file_corrupt_uproot(file_path, tree_name="Events"):
    """Check if a ROOT file is corrupt using uproot and if it has entries in the TTree."""
    try:
        with uproot.open(file_path) as f:
            if tree_name in f:
                tree = f[tree_name]
                num_entries = tree.num_entries
                return num_entries == 0  # True if file is empty (invalid), False if valid
            else:
                logger.warning(f"Tree '{tree_name}' not found in {file_path}")
                return True  # Tree is missing
    except Exception as e:
        logger.error(f"Error with uproot: {e}")
        return True  # File is corrupted or inaccessible

# Function to check if a ROOT file is corrupt using ROOT library
def is_root_file_corrupt_ROOT(file_path):
    """Check if ROOT file can be opened, is not in a zombie state, and contains a valid 'Events' tree."""
    print(f"Checking file: {file_path}")
    try:
        file = ROOT.TFile.Open(file_path, "READ")
        if not file or not file.IsOpen():
            logger.warning(f"ROOT file '{file_path}' failed to open.")
            return True  # ROOT file failed to open

        if file.IsZombie():
            logger.warning(f"ROOT file '{file_path}' is in a zombie state.")
            file.Close()
            return True  # ROOT file is in a zombie state

        # Check if the 'Events' tree exists and has valid entries
        tree = file.Get("Events")
        if not tree:
            logger.warning(f"Tree 'Events' not found in {file_path}.")
            file.Close()
            return True  # Tree 'Events' not found

        # Check if the tree has entries
        if tree.GetEntries() == 0:
            logger.warning(f"Tree 'Events' in '{file_path}' has 0 entries.")
            file.Close()
            return True  # ROOT file is empty

        # Close the file properly if all checks pass
        file.Close()
        return False  # ROOT file is valid

    except Exception as e:
        logger.error(f"Error opening ROOT file '{file_path}': {e}")

    # Ensure the file is closed
    if 'file' in locals() and file.IsOpen():
        file.Close()

    return True  # File is corrupted or inaccessible


# Function to process missing & corrupt files using Dask for parallelization
def check_missing_files(input_file, output_dir, year, additional_string):
    """Process files to check for missing or corrupt files."""
    # Load the input file into a pandas DataFrame
    df = pd.read_csv(input_file, sep=' ', header=None, names=['configFile', 'inputMiniAOD', 'outputDirectory', 'nEvents', 'CondorLogPath'])

    # Generate the expected output file names
    df['outputNanoAODFile'] = df['inputMiniAOD'].apply(lambda x: x.split('/')[-1].replace(".root", "_NanoAOD.root"))
    df['expected_output_file'] = df['outputDirectory'] + "/" + df['outputNanoAODFile']

    # Set up Dask client (with improved resource management)
    # client = Client(n_workers=4, threads_per_worker=2, memory_limit='4GB')  # Customize resources based on your setup

    gateway = Gateway(
        "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
        proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
    )
    cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
    client = gateway.connect(cluster_info.name).get_client()
    # logger.info(f"Dask client: {client}")

    # Create Dask delayed tasks for ROOT file corruption check
    tasks = [dask.delayed(is_root_file_corrupt_ROOT)(file) for file in df['expected_output_file']]

    # Compute the results in parallel using Dask
    results = dask.compute(*tasks)

    # Store corruption status in DataFrame
    df['file_corrupt'] = results

    # Filter out missing or corrupt files
    missing_files_df = df[df['file_corrupt'] == True]
    missing_files_df = missing_files_df[['configFile', 'inputMiniAOD', 'outputDirectory', 'nEvents', 'CondorLogPath']]

    # Dynamically generate the output filename based on year and additional string
    output_filename = f'missing_or_corrupt_files_{year}_{additional_string}.txt'

    # Save the missing or corrupt files list to a file
    missing_files_df.to_csv(output_filename, sep=' ', header=False, index=False)

    # Print results
    # logger.info(f"Missing or corrupt files: {len(missing_files_df)}")
    # logger.info(f"The list of missing or corrupt files is saved to '{output_filename}'")

    logger.info(f"{output_filename}: Missing or corrupt files: {len(missing_files_df)}")

# Main function to process all years
def main():
    """Main processing function."""
    # Define years and corresponding input/output files for each year
    years_and_input_files = {
        '2018v1': 'OriginalTxtFilesForNanoAODv12Production/HMuMu_UL2018_3Feb.txt',
        '2018': 'OriginalTxtFilesForNanoAODv12Production/HMuMu_UL2018_3Feb_AllJobs.txt',
        '2017': 'OriginalTxtFilesForNanoAODv12Production/HMuMu_UL2017_3Feb_AllJobs.txt',
        '2016APV': 'OriginalTxtFilesForNanoAODv12Production/HMuMu_UL2017_8Feb_2016APV.txt',
        '2016': 'OriginalTxtFilesForNanoAODv12Production/HMuMu_UL2017_8Feb_2016.txt',
    }

    years_and_output_dirs = {
        '2018v1': '/eos/purdue/store/user/rasharma/customNanoAOD/UL2018/',
        '2018': '/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/',
        '2017': '/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/',
        '2016APV': '/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/',
        '2016': '/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/',
    }

    # List of years to process
    # years = ['2018v1', '2018', '2017', '2016APV', '2016']
    years = ['2018', '2017', '2016APV', '2016']
    # years = ['2017', '2016APV', '2016']
    # years = ['2018']
    additional_string = "03March"

    # Process files for each year
    for year in years:
        input_file = years_and_input_files[year]
        output_dir = years_and_output_dirs[year]
        check_missing_files(input_file, output_dir, year, additional_string)

# Run the main function
if __name__ == "__main__":
    main()
