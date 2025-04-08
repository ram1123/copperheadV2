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
    elif args.use_dask:
        client = Client(n_workers=11, threads_per_worker=1, processes=True, memory_limit='10 GiB')
        print("Local scale Client created")
    return client


def file_exists_with_gfal(file_path):
    """
    Check if a file exists using the `gfal-ls` command.

    Parameters:
        file_path (str): Path to the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    try:
        # Run the gfal-ls command
        command = ["gfal-ls", file_path]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check the return code to determine if the file exists
        if result.returncode == 0:
            return True
        else:
            logger.warning(f"File not found: {file_path}. gfal-ls output: {result.stderr.strip()}")
            return False
    except Exception as e:
        logger.error(f"Error checking file existence with gfal-ls for '{file_path}': {e}")
        return False

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
    if not ifMiniAOD:
        # # Check if the file exists
        temp_file  = file_path.replace("root://eos.cms.rcac.purdue.edu/", "/eos/purdue/")
        if not os.path.exists(temp_file):
            logger.warning(f"File '{temp_file}' does not exist. ")
            return 0
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

# fetch the number of entries from dasgoclient command for miniAOD
# [rasharma@lxplus979 ~]$ dasgoclient --query="file=/store/data/Run2016G/SingleMuon/MINIAOD/UL2016_MiniAODv2-v2/120000/817DB9CF-3017-8247-B2AD-5AF4BCF07CEC.root" --json
# [
# {"das":{"expire":1744035502,"instance":"prod/global","primary_key":"file.name","record":1,"services":["dbs3:files"]},"file":[{"adler32":"85bc6977","auto_cross_section":0,"block.name":"/SingleMuon/Run2016G-UL2016_MiniAODv2-v2/MINIAOD#7d06c1b2-ddf8-4e4f-b414-31edc184ea7f","block_id":24151964,"block_name":"/SingleMuon/Run2016G-UL2016_MiniAODv2-v2/MINIAOD#7d06c1b2-ddf8-4e4f-b414-31edc184ea7f","branch_hash_id":null,"check_sum":"3566413521","create_by":null,"created_by":null,"creation_date":null,"creation_time":null,"dataset":"/SingleMuon/Run2016G-UL2016_MiniAODv2-v2/MINIAOD","dataset_id":14218939,"file_id":704099266,"file_type_id":1,"is_file_valid":1,"last_modification_date":1624277465,"last_modified_by":"wmagent@vocms0280.cern.ch","md5":null,"modification_time":1624277465,"modified_by":"wmagent@vocms0280.cern.ch","name":"/store/data/Run2016G/SingleMuon/MINIAOD/UL2016_MiniAODv2-v2/120000/817DB9CF-3017-8247-B2AD-5AF4BCF07CEC.root","nevents":162072,"size":3776716506,"type":"EDM"}],"qhash":"60f0caf6313aa766d4917ef60cc543c6"}
# ]
def get_num_entries_from_das(file_path, ifMiniAOD=False):
    """
    Get the number of entries in the MiniAOD file using dasgoclient.
    This function assumes that the file contains an 'Events' tree.

    Parameters:
        file_path (str): Path to the MiniAOD ROOT file.
        ifMiniAOD (bool): Flag to indicate if the file is MiniAOD.

    Returns:
        int: Number of entries in the MiniAOD file.
    """
    # Check if the file starts with root://, if not add the prefix
    if not file_path.startswith("root://"):
        file_path = "root://xcache.cms.rcac.purdue.edu/" + file_path

    # Use dasgoclient to get the number of events
    try:
        command = f"dasgoclient --query='file={file_path}' --json"
        result = os.popen(command).read()
        data = eval(result)
        num_entries = data[0]['file'][0]['nevents']
        return num_entries
    except Exception as e:
        logger.error(f"Error retrieving number of entries from DAS for '{file_path}': {e}")
        return 0



def check_missing_files(input_file, output_dir, year, additional_string, client=None):
    """Process files to check for missing or corrupt files."""
    # Load the input file into a pandas DataFrame
    df = pd.read_csv(input_file, sep=' ', header=None, names=['configFile', 'inputMiniAOD', 'outputDirectory', 'nEvents', 'CondorLogPath'])

    # Generate the expected output file names: mini.root -> mini_NanoAOD.root
    df['outputNanoAODFile'] = df['inputMiniAOD'].apply(lambda x: x.split('/')[-1].replace(".root", "_NanoAOD.root"))
    df['expected_output_file'] = df['outputDirectory'] + "/" + df['outputNanoAODFile']

    # Add "root://xcache.cms.rcac.purdue.edu/" or "root://cms-xrd-global.cern.ch/" to the inputMiniAOD files
    # df['inputMiniAOD'] = df['inputMiniAOD'].apply(lambda x: "root://xcache.cms.rcac.purdue.edu/" + x)
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

    # Compute the results with a progress bar
    print("Going to compute the results")

    # Wrap the Dask computation with a progress bar
    with ProgressBar():
        # task1 = [delayed(get_num_entries_in_nanoAOD_uproot)(file, ifMiniAOD=True) for file in tqdm(df['inputMiniAOD'], desc="Processing MiniAOD files")]
        # results_mini = compute(*task1)

        task2 = [delayed(get_num_entries_in_nanoAOD_uproot)(file) for file in tqdm(df['expected_output_file'], desc="Processing NanoAOD files")]
        results_nano = compute(*task2)

    # Store the results in the DataFrame
    # df['nEvents_from_inputMiniAOD'] = results_mini
    df['nEvents_from_nanoAOD'] = results_nano

    # task3 : use check_root_to_parquet for only those files for which nEvents_from_nanoAOD is not 0. Then add the output to the original dataframe

    # Initialize the 'parquet_conversion_success' column with False
    # df['parquet_conversion_success'] = False

    # # Filter files where nEvents_from_nanoAOD is not 0
    # files_to_convert = df[df['nEvents_from_nanoAOD'] != 0]['expected_output_file']

    # # Use Dask delayed tasks to process these files concurrently
    # conversion_tasks = [delayed(check_root_to_parquet)(file) for file in files_to_convert]
    # conversion_results = compute(*conversion_tasks)

    # Update the DataFrame with the conversion results
    # df.loc[df['nEvents_from_nanoAOD'] != 0, 'parquet_conversion_success'] = conversion_results


    # Get CSV files
    csv_file = f"AllFiles_{year}_{additional_string}.csv"
    df.to_csv(csv_file, index=False)
    print(f"CSV file created: {csv_file}, with {len(df)} entries.")

    # # Get CSV files with miniAOD and NanoAOD events done't match or NanoAOD is 0
    # df_mismatch = df[(df['nEvents_from_inputMiniAOD'] != df['nEvents_from_nanoAOD']) | (df['nEvents_from_nanoAOD'] == 0)]
    # df_mismatch = df[(df['nEvents_from_inputMiniAOD'] != df['nEvents_from_nanoAOD']) | (df['nEvents_from_nanoAOD'] == 0)]
    # print(df_mismatch.head())
    # df_mismatch.to_csv(f"Mismatch_MiniNano_nNano0_{year}_{additional_string}.csv", index=False)
    # print(f"CSV file created: Mismatch_MiniNano_nNano0_{year}_{additional_string}.csv, with {len(df_mismatch)} entries.")

    # df_mismatch = df_mismatch[['configFile', 'inputMiniAOD', 'outputDirectory', 'nEvents', 'CondorLogPath']]
    # df_mismatch.to_csv(f"Mismatch_MiniNano_nNano0_{year}_{additional_string}_configFile.txt", sep=' ', header=False, index=False)
    # # print(f"CSV file created: Mismatch_MiniNano_nNano0_{year}_{additional_string}_configFile.txt, with {len(df_mismatch)} entries.")

    # # Get CSV files with miniAOD and NanoAOD events done't match or NanoAOD is 0 or miniAOD is 0
    # df_mismatch = df[(df['nEvents_from_inputMiniAOD'] != df['nEvents_from_nanoAOD']) | (df['nEvents_from_nanoAOD'] == 0) | (df['nEvents_from_inputMiniAOD'] == 0)]
    # df_mismatch.to_csv(f"Mismatch_MiniNano_nNano0_nMini0_{year}_{additional_string}.csv", index=False)
    # print(f"CSV file created: Mismatch_MiniNano_nNano0_nMini0_{year}_{additional_string}.csv, with {len(df_mismatch)} entries.")

    # df_mismatch = df_mismatch[['configFile', 'inputMiniAOD', 'outputDirectory', 'nEvents', 'CondorLogPath']]
    # df_mismatch.to_csv(f"Mismatch_MiniNano_nNano0_nMini0_{year}_{additional_string}_configFile.txt", sep=' ', header=False, index=False)
    # # print(f"CSV file created: Mismatch_MiniNano_nNano0_nMini0_{year}_{additional_string}_configFile.txt, with {len(df_mismatch)} entries.")

    # # Get CSV file having parquet_conversion_success as False
    # # df_mismatch = df[df['parquet_conversion_success'] == False]
    # # df_mismatch.to_csv(f"Mismatch_ParquetConversion_{year}_{additional_string}.csv", index=False)
    # # print(f"CSV file created: Mismatch_ParquetConversion_{year}_{additional_string}.csv, with {len(df_mismatch)} entries.")
    # df_mismatch = df_mismatch[['configFile', 'inputMiniAOD', 'outputDirectory', 'nEvents', 'CondorLogPath']]
    # df_mismatch.to_csv(f"Mismatch_ParquetConversion_{year}_{additional_string}_configFile.txt", sep=' ', header=False, index=False)
    # # print(f"CSV file created: Mismatch_ParquetConversion_{year}_{additional_string}_configFile.txt, with {len(df_mismatch)} entries.")


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

    # years = ['2018']
    # year = ['2018Re', '2018', '2017', '2016APV', '2016']
    years = ['2017', '2016APV', '2016', '2018']
    # years = ['2018GT36_debug']
    # years = ['2018GT36']
    years = ['2018MC']
    # additional_string = "4April_GlobalRedirector"
    # additional_string = "4April_Xcache"
    # additional_string = "4April_local_retries"
    # additional_string = "4April_debug"
    additional_string = "6April_All"
    # additional_string = "4April_AllYears"

    if args.use_dask or args.use_gateway:
        print("Using Dask for parallel processing.")
        client = create_dask_client()  # Initialize Dask client

    for year in years:
        print(f"\n\n===> Processing year: {year}")
        input_file = years_and_input_files[year]
        output_dir = years_and_output_dirs[year]
        if args.use_dask or args.use_gateway:
            check_missing_files(input_file, output_dir, year, additional_string, client)
        else:
            check_missing_files(input_file, output_dir, year, additional_string)

if __name__ == "__main__":
    main()
