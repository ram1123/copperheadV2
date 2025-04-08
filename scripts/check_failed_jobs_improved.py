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
from tqdm import tqdm
from dask.diagnostics import ProgressBar
import subprocess
import dask.dataframe as dd
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed

# ROOT Error Handling: Suppress non-critical warnings
ROOT.gErrorIgnoreLevel = ROOT.kError  # Only show errors, not warnings

# Set up logging for better tracking and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Argument parsing for making Dask optional
parser = argparse.ArgumentParser(description="Process ROOT files with or without Dask.")
parser.add_argument('--use-dask', action='store_true', help="Use Dask for parallel processing")
parser.add_argument('--use-gateway', action='store_true', help="Use Dask Gateway for cluster mode")
parser.add_argument('--use-multiprocessing', action='store_true', help="Use multiprocessing for parallel processing")
# add option to run miniAOD files too: option name --use-miniAOD
parser.add_argument('--use-miniAOD', action='store_true', help="Use MiniAOD files for processing")

args = parser.parse_args()


def check_failed_jobs(input_file, output_dir, year, additional_string):
    # Step-1: Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} does not exist.")
        return

    # Step-2: Convert the input file to a DataFrame
    try:
        df = pd.read_csv(input_file, sep=' ', header=None, names=['configFile', 'inputMiniAOD', 'outputDirectory', 'nEvents', 'CondorLogPath'])
        logger.info(f"Successfully read input file {input_file}.")
    except Exception as e:
        logger.error(f"Error reading input file {input_file}: {e}")
        return

    print(df.head())

    # Step-3: Check if the output nanoAOD file exists
    # Generate the expected output file names: mini.root -> mini_NanoAOD.root
    # Repalce the redirector "root://eos.cms.rcac.purdue.edu/" with "/eos/purdue" to search files locally (for speed)
    df['outputNanoAODFile'] = df['inputMiniAOD'].apply(lambda x: x.split('/')[-1].replace(".root", "_NanoAOD.root"))
    df['expected_output_file'] = df['outputDirectory'] + "/" + df['outputNanoAODFile']
    df['expected_output_file'] = df['expected_output_file'].str.replace("root://eos.cms.rcac.purdue.edu/", "/eos/purdue")

    # Step-4: Check if the output nanoAOD file "expected_output_file" exists
    if args.use_dask:
        ddf = dd.from_pandas(df, npartitions=8)
        ddf['output_file_exists'] = ddf['expected_output_file'].map(os.path.exists, meta=('output_file_exists', 'bool'))
        df = ddf.compute()
    else:
        df['output_file_exists'] = df['expected_output_file'].apply(lambda x: os.path.exists(x))

    print(df.head())

    # Step-5: Check the entries in the output nanoAOD file
    df['nEntries_FromNanoAOD'] = 0
    if args.use_multiprocessing:
        with Pool() as pool:
            # df['nEntries_FromNanoAOD'] = list(tqdm(pool.imap(get_entries, df['expected_output_file']), total=len(df), desc="Processing files"))
            # check entries only if output file exists
            df['nEntries_FromNanoAOD'] = list(tqdm(pool.imap(get_entries, df.loc[df['output_file_exists'], 'expected_output_file']), total=len(df), desc="Processing files"))
    else:
        df['nEntries_FromNanoAOD'] = df.apply(lambda row: get_entries(row['expected_output_file']) if row['output_file_exists'] else 0, axis=1)

    print(df.head())
    # Save the DataFrame to a CSV file
    output_csv = f"UL{year}_{additional_string}_all_jobs.csv"
    df.to_csv(output_csv, index=False)
    logger.info(f"Output saved to {output_csv}")

    if 'inputMiniAOD' in df.columns and args.use_miniAOD:
        # Process the files in parallel and get the number of entries
        file_paths = df['inputMiniAOD'].tolist()
        num_entries_list = process_files_in_parallel(file_paths, max_workers=1) # FIXME: If max_workers > 1 then the number of entries are placed in wrong order

        # Add the new column to the DataFrame
        df['numEntriesDAS'] = num_entries_list

        # Save the updated DataFrame to a new CSV file
        df.to_csv(output_csv.replace(".csv","_WithDASEntry.csv"), index=False)
        print(f"Updated DataFrame saved to '{output_csv.replace(".csv","_WithDASEntry.csv")}'")

        df_mismatched = df[df['numEntriesDAS'] != df['nEvents_from_nanoAOD']]
        if not df_mismatched.empty:
            output_csv = output_csv.replace(".csv","_mismatch.csv")
            df_mismatched.to_csv(f"{output_csv}", index=False)
            print(f"Mismatched entries saved to '{outfile}_skim.csv', having entries {len(df_mismatched)}")
        else:
            print("No mismatched entries found.")
    else:
        print("The column 'inputMiniAOD' is not found in the CSV file.")



def get_entries(file_path):
    """Get the number of entries in a ROOT file."""
    try:
        with uproot.open(file_path) as f:
            tree = f['Events']
            n_entries = tree.num_entries
        return n_entries
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return 0

def run_command(command):
    try:
        result = subprocess.check_output(command, shell=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        return ""

def get_num_entries_from_das(file_path):
    """
    Get the number of entries in the MiniAOD file using dasgoclient.
    This function assumes that the file contains an 'Events' tree.

    Parameters:
        file_path (str): Path to the MiniAOD ROOT file.

    Returns:
        int: Number of entries in the MiniAOD file, or 0 if an error occurs.
    """
    try:
        # Construct the dasgoclient command
        command = f'dasgoclient --query="file={file_path}" --json'
        result = run_command(command)

        # Parse the JSON result
        data = json.loads(result)

        # Validate the JSON structure and extract the number of events
        if data and 'file' in data[0] and 'nevents' in data[0]['file'][0]:
            num_entries = data[0]['file'][0]['nevents']
            return num_entries
        else:
            print(f"Unexpected JSON structure: {data}")
            return 0
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from DAS for '{file_path}': {e}")
        return 0
    except KeyError as e:
        print(f"Missing key in DAS response for '{file_path}': {e}")
        return 0
    except Exception as e:
        print(f"Error retrieving number of entries from DAS for '{file_path}': {e}")
        return 0

def process_files_in_parallel(file_paths, max_workers=8):
    """
    Process a list of file paths in parallel to retrieve the number of entries.

    Parameters:
        file_paths (list): List of file paths to process.
        max_workers (int): Maximum number of threads to use.

    Returns:
        list: List of number of entries corresponding to each file path.
    """
    num_entries_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor
        future_to_file = {executor.submit(get_num_entries_from_das, file_path): file_path for file_path in file_paths}

        # Use tqdm to display progress
        for future in tqdm(as_completed(future_to_file), total=len(file_paths), desc="Processing files"):
            file_path = future_to_file[future]
            try:
                num_entries = future.result()
                num_entries_list.append(num_entries)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                num_entries_list.append(0)

    return num_entries_list


def main():
    """Main processing function."""
    years_and_input_files = {
        '2018GT36': 'OriginalTxtFilesForNanoAODv12Production/UL2018-GT36.txt',
        '2018GT36_debug': 'OriginalTxtFilesForNanoAODv12Production/UL2018-GT36_debug.txt',
        '2018Re': 'OriginalTxtFilesForNanoAODv12Production/HMuMu_UL2018_NanoAODv12_06March_Data_Run2018A.txt',
        # '2018': 'OriginalTxtFilesForNanoAODv12Production/HMuMu_UL2018_06March_AllJobs_grepDATA.txt',
        '2018MC': 'OriginalTxtFilesForNanoAODv12Production/HMuMu_UL2018_06March_AllJobs_grepMC.txt',
        '2017': 'OriginalTxtFilesForNanoAODv12Production/HMuMu_UL2017_3Feb_AllJobs.txt',
        '2016APV': 'OriginalTxtFilesForNanoAODv12Production/HMuMu_UL2017_8Feb_2016APV.txt',
        '2016': 'OriginalTxtFilesForNanoAODv12Production/HMuMu_UL2017_8Feb_2016.txt',
    }

    years_and_output_dirs = {
        '2018GT36': '/eos/purdue/store/user/rasharma/customNanoAOD_GT36/UL2018/',
        '2018GT36_debug': '/eos/purdue/store/user/rasharma/customNanoAOD_GT36/UL2018/',
        '2018Re': '/eos/purdue/store/user/rasharma/CustomNanoAODv12_v2/UL2018/',
        '2018': '/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/',
        '2018MC': '/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/',
        '2017': '/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/',
        '2016APV': '/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/',
        '2016': '/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/',
    }

    # years = ['2017', '2016APV', '2016', '2018']
    # years = ['2018GT36_debug']
    years = ['2018MC','2018GT36']
    additional_string = "7April_ImprovedScript"
    # additional_string = "2018GT36_debug"

    for year in years:
        print(f"Processing for year: {year}")
        input_file = years_and_input_files[year]
        output_dir = years_and_output_dirs[year]
        check_failed_jobs(input_file, output_dir, year, additional_string)


if __name__ == "__main__":
    main()
