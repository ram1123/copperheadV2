import pandas as pd
import pyarrow.parquet as pq
import glob

input_parquet_glob = "/depot/cms/users/shar1172/hmm/copperheadV1clean/May31_2l2nu/stage1_output/2018/f1_0/data_B/0/*.parquet"
output_txt = "sync_output_file.txt"

columns = {
    "run": "run",
    "luminosityBlock": "luminosityBlock",
    "event": "event",
    "pTL1": "mu1_pt",
    "pTL2": "mu2_pt",
    "MET_pt": "MET_pt",
    "massZ1": "dimuon_mass",
    "pTZ1": "dimuon_pt",
}

# Find all parquet files matching the pattern
file_list = glob.glob(input_parquet_glob)
if not file_list:
    raise FileNotFoundError(f"No parquet files found matching: {input_parquet_glob}")

# Read all files into a single DataFrame
dfs = []
for file in file_list:
    table = pq.read_table(file, columns=list(columns.values()))
    dfs.append(table.to_pandas())

# Concatenate all DataFrames
df = pd.concat(dfs, ignore_index=True)

# Rename columns
df.columns = columns.keys()

# Save with tab separation
df.to_csv(output_txt, sep="\t", index=False, header=True)
