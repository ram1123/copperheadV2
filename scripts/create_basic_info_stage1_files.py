"""
Steps:
1. Read the output files from the main path of stage1 (/path/year/f1_0/sample/)
2. For each file save following info in a text file:
    - year
    - sample
    - xrdadler32 checksum
    - size in bytes
    - full path
    - file creation time (unix timestamp)

# --------
- Create two functions:
    - create file list for all files in stage1
    - create basic info tsv file for all files in stage1
# --------
How to run:
python scripts/create_basic_info_stage1_files.py -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -o stage1_basic_info

NOTE: Assume the path structure is : /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/{label}/stage1_output/{year}/f1_0/{sample}/file.parquet
"""

import os
import sys
import subprocess
from pathlib import Path
import time
import csv
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


def find_parquet_files(base_path: str):
    """Recursively find all .parquet files under base_path."""
    root_files = []
    year = []
    sample = []
    for dirpath, test, filenames in os.walk(base_path):
        # print(f"Scanning {dirpath}, {test}, {len(filenames)} files")
        for filename in filenames:
            if filename.lower().endswith(".parquet"):
                full_path = os.path.join(dirpath, filename)
                root_files.append(full_path)
                # year.append(dirpath.split("/")[-5])  # assuming structure /path/year/f1_0/sample/file.parquet
                # sample.append(dirpath.split("/")[-3])

    return root_files


def get_file_info(file_path: str):
    """Get basic info of the file: size, checksum, creation time."""
    try:
        size = os.path.getsize(file_path)
        ctime = time.ctime(os.path.getctime(file_path))
        result = subprocess.run(['xrdadler32', file_path], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"xrdadler32 command failed: {result.stderr}")
        checksum = result.stdout.strip().split()[0]
        return size, checksum, ctime
    except Exception as e:
        sys.stderr.write(f"Error getting info for {file_path}: {e}\n")
        return None, None, None


def process_file(file_path, base_path):
    parts = Path(file_path).parts
    try:
        rel = Path(file_path).relative_to(base_path)
        year, _, sample, *_ = rel.parts
        size, checksum, ctime = get_file_info(file_path)
        if None in (size, checksum, ctime):
            return None
        return [year, sample, checksum, size, file_path, ctime]
    except Exception as e:
        sys.stderr.write(f"Failed to process {file_path}: {e}\n")
        return None


def main(label: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    base_path = f"/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/{label}/stage1_output/"
    output_file = f"{output_dir}/{label}.csv"

    root_files = find_parquet_files(base_path)
    results = []

    with ThreadPoolExecutor(max_workers=128) as executor:
        futures = [executor.submit(process_file, f, base_path) for f in root_files]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing files"
        ):
            res = future.result()
            if res:
                results.append(res)

    with open(output_file, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["year", "sample", "checksum", "size", "full_path", "ctime"])
        writer.writerows(results)

    print(f"Saved info for {len(results)} files to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create basic info files for stage1 parquet datasets.")
    parser.add_argument("-l", "--label", required=False, default="Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar", help="Label for the dataset")
    parser.add_argument("-o", "--output", required=False, default="stage1_basic_info", help="Output directory for the info files")
    args = parser.parse_args()
    main(label=args.label, output_dir=args.output)
