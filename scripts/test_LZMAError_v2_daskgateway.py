from dask_gateway import Gateway
from dask.distributed import as_completed
import hepconvert
from pathlib import Path
import uproot
import os
import traceback
import uuid
from datetime import datetime

# Setup Dask Gateway
gateway = Gateway(
    address="http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
    proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
)
cluster_info = gateway.list_clusters()[0]  # get the first (and only) running cluster
client = gateway.connect(cluster_info.name).get_client()
print(f"Dask client dashboard: {client.dashboard_link}")

# Directory of files
inDir = Path("/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9")
files = list(inDir.glob("*.root"))

# Distributed task function
def test_file(file):
    import uproot
    import hepconvert
    import os, uuid, traceback

    try:
        tree = uproot.open(f"{file}:Events")
        _ = tree.arrays(["Muon_pt"], entry_stop=10)

        tmp_path = f"/tmp/{uuid.uuid4().hex}.parquet"
        try:
            hepconvert.root_to_parquet(
                in_file=str(file),
                out_file=tmp_path,
                tree="Events",
                keep_branches=["dimuon_pt"],
                force=True,
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return (str(file), True)

    except Exception as e:
        if "lzma data error" in str(e).lower():
            return (str(file), False)
        else:
            return (str(file), traceback.format_exc())

# Submit to Dask
futures = client.map(test_file, files)

# Collect results
bad_files = []
other_errors = []

for future, result in as_completed(futures, with_results=True):
    f, status = result
    if status is False:
        bad_files.append(f)
    elif status is not True:
        other_errors.append((f, status))

# Save logs
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
bad_files_path = f"lzma_corrupted_files_{timestamp}.txt"
other_errors_path = f"other_errors_{timestamp}.log"

with open(bad_files_path, "w") as f:
    f.write(f"# LZMA-corrupted files found on {datetime.now()}\n\n")
    f.write("\n".join(bad_files))

with open(other_errors_path, "w") as f:
    f.write(f"# Other errors encountered on {datetime.now()}\n\n")
    for fpath, err_msg in other_errors:
        f.write(f"--- {fpath} ---\n")
        f.write(err_msg)
        f.write("\n\n")

# Done
print(f"\n✅ Scan completed using Dask Gateway.")
print(f"🧨 {len(bad_files)} LZMA-corrupted files logged to: {bad_files_path}")
print(f"⚠️  {len(other_errors)} other error(s) logged to: {other_errors_path}")