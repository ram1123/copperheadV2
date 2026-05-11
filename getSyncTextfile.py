import pandas as pd
import pyarrow.parquet as pq
import glob

input_parquet_glob = "/depot/cms/users/shar1172/hmm/copperheadV1clean/2l2nu_sync/stage1_output/2018/f1_0/data_B/0/*.parquet"
output_txt = "sync_output_file.txt"

columns = {
    "run": "run",
    "luminosityBlock": "luminosityBlock",
    "event": "event",
    "pTL1": "mu1_pt",
    "etaL1": "mu1_eta",
    "phiL1": "mu1_phi",
    "pTL2": "mu2_pt",
    "etaL2": "mu2_eta",
    "phiL2": "mu2_phi",
    "MET_pt": "MET_pt",
    "MET_phi": "MET_phi",
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

# Apply selection
# df = df[
#     (df["mu1_pt"] > 25) & (abs(df["mu1_eta"]) < 2.4) &
#     (df["mu2_pt"] > 25) & (abs(df["mu2_eta"]) < 2.4) &
#     (df["dimuon_pt"] > 55) & (abs(df["dimuon_mass"] - 91) < 15) &
#     (df["MET_pt"] > 100)
# ]

# df = df.sort_values(by="run")
# short by run, then by luminosityBlock, then by event
df = df.sort_values(by=["run", "luminosityBlock", "event"])

# Rename columns
df.columns = columns.keys()

# Round all numeric columns to 2 decimal places
df = df.round(2)

header_line = (
    # "run        lumi       event           pTL1       etaL1      phiL1      "
    # "pTL2       etaL2      phiL2      MET_pt     MET_phi    massZ1     pTZ1"
    "run        lumi       event           pTL1             "
    "pTL2            MET_pt         massZ1     pTZ1"
)
separator_line = "=" * len(header_line)

with open(output_txt, "w") as f:
    f.write(header_line + "\n")
    f.write(separator_line + "\n")
    for _, row in df.iterrows():
        # f.write(f"{int(row['run']):<10}{int(row['luminosityBlock']):>6}  {int(row['event']):>13}  "
        #         f"{row['pTL1']:>9.2f}  {row['etaL1']:>8.2f}  {row['phiL1']:>8.2f}  "
        #         f"{row['pTL2']:>9.2f}  {row['etaL2']:>8.2f}  {row['phiL2']:>8.2f}  "
        #         f"{row['MET_pt']:>9.2f}  {row['MET_phi']:>8.2f}  {row['massZ1']:>8.2f}  {row['pTZ1']:>8.2f}\n")
        f.write(f"{int(row['run']):<10}{int(row['luminosityBlock']):>6}  {int(row['event']):>13}  "
                f"{row['pTL1']:>9.2f}   "
                f"{row['pTL2']:>9.2f}   "
                f"{row['MET_pt']:>9.2f}   {row['massZ1']:>8.2f}  {row['pTZ1']:>8.2f}\n")
