"""
check_HLT_branch_dask.py

Summary:
    This script checks whether the branch `HLT_IsoMu27` exists in each ROOT file
    located in a specified EOS directory. It uses uproot for fast ROOT I/O and
    Dask Gateway to parallelize the check across a distributed cluster.

    Files that are either missing the branch or are unreadable will be reported.

How to Run:
    1. Ensure your environment has access to:
        - uproot
        - dask
        - dask_gateway

    2. Ensure Dask Gateway access from your environment (e.g., RCAC at Purdue).

    3. Run the script in a Python environment (e.g., cmsenv, or container with uproot + dask):
        python check_HLT_branch_dask.py

Output:
    Prints a list of files missing the specified HLT branch.

Author: Ram Krishna Sharma
Date: 2025-06-20
"""

import glob
import uproot
from dask import delayed
from dask.distributed import Client
from dask_gateway import Gateway

# ----------------------------------------------------------
# Connect to the Dask Gateway cluster at Purdue
# ----------------------------------------------------------
gateway = Gateway(
    "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
    proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
)
cluster_info = gateway.list_clusters()[0]
client = gateway.connect(cluster_info.name).get_client()

# ----------------------------------------------------------
# Collect input ROOT files from the specified EOS directory
# ----------------------------------------------------------
input_files = glob.glob(
    "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/SingleMuon_Run2017F/*.root"
)

# ----------------------------------------------------------
# Function to check if a specific branch is missing in a ROOT file
# Returns (file_path, "YES") if the branch is missing or file is unreadable
# Returns (file_path, "NO") if the branch is found
# ----------------------------------------------------------
@delayed
def check_missing_branch(file_path, branch="HLT_IsoMu27"):
    """
    Check whether the given ROOT file contains the specified branch.

    Parameters:
    - file_path (str): Path to the ROOT file
    - branch (str): Name of the branch to check (default: "HLT_IsoMu27")

    Returns:
    - tuple: (file_path, "YES") if branch is missing or unreadable
             (file_path, "NO") if branch exists
    """
    try:
        with uproot.open(file_path) as f:
            tree = f["Events"]
            if branch in tree.keys():
                return (file_path, "NO")  # Branch exists
            else:
                return (file_path, "YES")  # Branch missing
    except Exception:
        return (file_path, "YES")  # Treat as missing if file can't be opened

# ----------------------------------------------------------
# Run the branch check in parallel across the cluster
# ----------------------------------------------------------
results = client.gather(client.compute([check_missing_branch(f) for f in input_files]))

# ----------------------------------------------------------
# Print the files where the branch is missing or file failed to open
# ----------------------------------------------------------
print("Files missing branch:")
for path, status in results:
    if status == "YES":
        print(f"{path} - MISSING")

print(f"\nTotal files checked: {len(input_files)}")
print(f"Files with missing branch: {sum(1 for _, s in results if s == 'YES')}")
