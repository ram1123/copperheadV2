import os
import pandas as pd
import dask.dataframe as dd
import uproot
import sys
import ROOT
import logging
import time
import argparse


# ROOT Error Handling: Suppress non-critical warnings
ROOT.gErrorIgnoreLevel = ROOT.kError  # Only show errors, not warnings

# Set up logging for better tracking and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()


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
    if not file_path.startswith("root://"):
        # file_path = "root://xcache.cms.rcac.purdue.edu/" + file_path
        file_path = "root://cms-xrd-global.cern.ch//" + file_path
    try:
        # print("input file: ",file_path)
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


import dask.dataframe as dd

def check_missing_files(input_file, output_dir, year, additional_string):
    """Process files to check for missing or corrupt files."""
    # Load the input file into a pandas DataFrame
    df = pd.read_csv(input_file, sep=' ', header=None, names=['configFile', 'inputMiniAOD', 'outputDirectory', 'nEvents', 'CondorLogPath'])

    # Convert the Pandas DataFrame to a Dask DataFrame
    ddf = dd.from_pandas(df, npartitions=4)  # You can adjust npartitions based on your system

    # Handle missing values and apply the split operation within map_partitions
    def process_partition(partition):
        partition['outputNanoAODFile'] = partition['inputMiniAOD'].fillna('').apply(lambda x: x.split('/')[-1].replace(".root", "_NanoAOD.root") if isinstance(x, str) else "")
        return partition

    # Use map_partitions to process each partition in the Dask DataFrame
    ddf = ddf.map_partitions(process_partition)

    # Compute the result and bring it back to Pandas (if necessary)
    df = ddf.compute()

    df['expected_output_file'] = df['outputDirectory'] + "/" + df['outputNanoAODFile']

    input_files = df['inputMiniAOD'].tolist()
    output_files = df['expected_output_file'].tolist()

    # Use regular Python loop for processing if Dask is not available
    results = []
    for file in input_files:
        results.append(get_num_entries_in_nanoAOD(file))
    for file in output_files:
        results.append(get_num_entries_in_nanoAOD(file))

    # Split results into the number of events (from MiniAOD) and entries (from NanoAOD)
    num_events_results_mini = results[:len(input_files)]
    num_events_results_nano = results[len(input_files):]

    # Store the results in the DataFrame
    df['nEvents_from_inputMiniAOD'] = num_events_results_mini
    df['nEvents_from_nanoAOD'] = num_events_results_nano

    # Filter out missing or corrupt files (where nEvents_from_nanoAOD is 0)
    missing_files_df = df[df['nEvents_from_nanoAOD'] == 0]  # If NanoAOD has 0 events, consider it corrupt
    missing_files_df = missing_files_df[['configFile', 'inputMiniAOD', 'outputDirectory', 'nEvents', 'CondorLogPath']]

    output_filename = f'missing_or_corrupt_files_{year}_{additional_string}.txt'
    missing_files_df.to_csv(output_filename, sep=' ', header=False, index=False)

    df.to_csv(f'full_df_{year}_{additional_string}.csv', index=False)

    df_not_equal = df[df['nEvents_from_nanoAOD'] != df['nEvents_from_inputMiniAOD']]
    df_not_equal.to_csv(f'full_df_DifferentNEvents_{year}_{additional_string}.csv', sep=' ', header=False, index=False)

    df_not_equal = df_not_equal[['configFile', 'inputMiniAOD', 'outputDirectory', 'nEvents', 'CondorLogPath']]
    df_not_equal.to_csv(f'skim_df_DifferentNEvents_{year}_{additional_string}.csv', sep=' ', header=False, index=False)

    logger.info(f"{output_filename}: Missing or corrupt files: {len(missing_files_df)}")
    logger.info(f"full_df_{year}_{additional_string}.csv: Full DataFrame saved.")
    logger.info(f"skim_df_DifferentNEvents_{year}_{additional_string}.csv: {len(df_not_equal)} entries with different nEvents saved.")



def main():
    """Main processing function."""
    years_and_input_files = {
        '2018GT36': 'OriginalTxtFilesForNanoAODv12Production/UL2018-GT36.txt',
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
        '2018GT36': '/eos/purdue/store/user/rasharma/customNanoAOD_GT36/UL2018/',
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
    # years = ['TestGautschi', 'TestHammer', 'TestLxplus']
    years = ['2018GT36']
    additional_string = "25March"

    for year in years:
        input_file = years_and_input_files[year]
        output_dir = years_and_output_dirs[year]
        check_missing_files(input_file, output_dir, year, additional_string)

if __name__ == "__main__":
    main()
