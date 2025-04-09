import os
import pandas as pd
import uproot
import ROOT
import logging
import argparse
import subprocess
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import time
import random

# ROOT Error Handling: Suppress non-critical warnings
ROOT.gErrorIgnoreLevel = ROOT.kError  # Only show errors, not warnings

# Set up logging for better tracking and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Argument parsing for making Dask optional
parser = argparse.ArgumentParser(description="Process ROOT files with or without Dask.")
parser.add_argument('--use-multiprocessing', action='store_true', help="Use multiprocessing for parallel processing")
parser.add_argument('--use-miniAOD', action='store_true', help="Use miniAOD files for entry count")
args = parser.parse_args()


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

    # Step-3: Check if the output nanoAOD file "expected_output_file" exists
def check_output_file_exists(df, running_location="purdue"):
    # Step-3: Check if the output nanoAOD file "expected_output_file" exists
    # Generate the expectedoutput file names: mini.root -> mini_NanoAOD.root
    df['outputNanoAODFile'] = df['inputMiniAOD'].apply(lambda x: x.split('/')[-1].replace(".root", "_NanoAOD.root"))
    df['expected_output_file'] = df['outputDirectory'] + "/" + df['outputNanoAODFile']
    if running_location == "purdue":
        # Replace the EOS path with the local path
        df['expected_output_file'] = df['expected_output_file'].str.replace("root://eos.cms.rcac.purdue.edu/", "/eos/purdue")
        df['output_file_exists'] = df['expected_output_file'].apply(lambda x: os.path.exists(x))
    else:
        # Use the gfal-ls command to check for file existence
        pass
    return df

def check_entries_from_nanoAOD_files(df):
    # Step-4: Check the entries in the output nanoAOD file
    logger.info("==> Computing number of entries in the output files...")
    df['nEntries_FromNanoAOD'] = 0

    # Check entries only if output file exists
    existing_files = df.loc[df['output_file_exists'], 'expected_output_file'].tolist()

    if args.use_multiprocessing:
        with Pool() as pool:
            # Use tqdm to show progress
            entries = list(tqdm(pool.imap(get_entries, existing_files), total=len(existing_files), desc="Processing files"))
            # Ensure the length of the result matches the number of rows in the DataFrame
            df.loc[df['output_file_exists'], 'nEntries_FromNanoAOD'] = entries
    else:
        # Use a regular approach if not using multiprocessing
        df['nEntries_FromNanoAOD'] = df.apply(lambda row: get_entries(row['expected_output_file']) if row['output_file_exists'] else 0, axis=1)

    return df


def check_entries_from_miniAOD_files(df, output_csv="output.csv"):
    if 'inputMiniAOD' in df.columns and args.use_miniAOD:
        # Process the files in parallel and get the number of entries from DAS
        file_paths = df['inputMiniAOD'].tolist()

        # Get number of entries from DAS
        num_entries_list = process_files_in_parallel(file_paths, max_workers=126)  # Use 126 workers or adjust as needed

        # Add the new column to the DataFrame
        df['numEntriesDAS'] = num_entries_list

        # Save the updated DataFrame to a new CSV file
        df.to_csv(output_csv.replace(".csv", "_WithDASEntry.csv"), index=False)
        logger.info(f"Updated DataFrame saved to '{output_csv.replace('.csv','_WithDASEntry.csv')}'")

        # Now, check for mismatches
        df_mismatched = df[df['numEntriesDAS'] != df['nEntries_FromNanoAOD']]
        if not df_mismatched.empty:
            mismatch_csv = output_csv.replace(".csv", "_mismatch.csv")
            df_mismatched.to_csv(f"{mismatch_csv}", index=False)
            logger.info(f"Mismatched entries saved to '{mismatch_csv}', having entries {len(df_mismatched)}")
        else:
            logger.info("No mismatched entries found.")
    else:
        logger.info("The column 'inputMiniAOD' is not found in the CSV file.")


def process_files_in_parallel(file_paths, max_workers=8):
    """
    Process a list of file paths in parallel to retrieve the number of entries.

    Parameters:
        file_paths (list): List of file paths to process.
        max_workers (int): Maximum number of threads to use.

    Returns:
        list: List of number of entries corresponding to each file path, in the same order as input.
    """
    num_entries_list = [None] * len(file_paths)  # Preallocate a list to store results in order
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor with their index
        future_to_index = {
            executor.submit(get_num_entries_from_das, file_path): idx
            for idx, file_path in enumerate(file_paths)
        }

        # Use tqdm to display progress
        for future in tqdm(as_completed(future_to_index), total=len(file_paths), desc="Processing files"):
            idx = future_to_index[future]
            try:
                num_entries = future.result()
                num_entries_list[idx] = num_entries  # Store result at the correct index
            except Exception as e:
                print(f"Error processing file {file_paths[idx]}: {e}")
                num_entries_list[idx] = 0  # Default to 0 on error

    return num_entries_list

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
        # sleep randomly between 0 to 10 seconds
        time.sleep(random.uniform(0, 10))
        # Construct the dasgoclient command
        command = f'dasgoclient --query="file={file_path}" --json'
        # print(f"Running command: {command}")
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

    # Step-3: Check if the output nanoAOD file "expected_output_file" exists
    df = check_output_file_exists(df, running_location="purdue")

    # Step-4: Check the entries in the output nanoAOD file
    df = check_entries_from_nanoAOD_files(df)

    # Save the DataFrame to a CSV file
    output_csv = f"UL{year}_{additional_string}_all_jobs.csv"
    df.to_csv(output_csv, index=False)
    logger.info(f"Output saved to {output_csv}")

    # Step-5: Check the entries in the input miniAOD file
    #              Two options:
    #              1. Use the dasgoclient command with option --json to fetch the number of events
    #              2. Use uproot to get the number of entries in the input file
    df = check_entries_from_miniAOD_files(df, output_csv)


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
    # years = ['2018MC','2018GT36']
    years = ['2018GT36']
    additional_string = "8April_ImprovedScript"
    # additional_string = "2018GT36_debug"

    for year in years:
        print(f"Processing for year: {year}")
        input_file = years_and_input_files[year]
        output_dir = years_and_output_dirs[year]
        check_failed_jobs(input_file, output_dir, year, additional_string)


if __name__ == "__main__":
    main()
