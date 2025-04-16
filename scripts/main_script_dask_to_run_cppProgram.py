"""
Motivation for this script:
   - When we generated the custom NanoAOD files, then for some of the files, we
      obtained the error "LZMA decompression error", as described in the root
      forum link: https://root-forum.cern.ch/t/how-to-skip-fix-lzma-error/63362?u=ramkrishna

    - For this reason we need to run the script to find the files which are
      causing the error, and then we need to remove those files from the
      directory.

    - The script is designed to run on the Dask cluster, and it tries to find
      the files which are causing the error, and then it gives the list of
      files which are causing the error.

Workflow:
    1. Connect to an existing Dask Gateway cluster.
    2. Retrieve the list of input ROOT files from a specified directory.
    3. Submit tasks to process each file using the `process_file` function.
    4. Collect results as tasks complete, logging successes and failures.
    5. Print a summary of the processing results.
"""
from dask_gateway import Gateway
import dask
from dask.distributed import as_completed
import glob

# Connect to the existing Dask Gateway cluster
gateway = Gateway(
    "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
    proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
)
cluster_info = gateway.list_clusters()[0]  # assumes at least one cluster exists
client = gateway.connect(cluster_info.name).get_client()
print(f"Connected to Dask Gateway Cluster: {client}")

# "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/"
# "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_v2/UL2017/"
# "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016APV/UL2016APV/"
# "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/"
input_files = glob.glob(
    "/eos/purdue/store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/"
    "*/*.root"
)

log_file = "UL2016_LZMA_errors_cpp_NEW.txt"

@dask.delayed
def process_file(remote_file):
    import subprocess

    try:
        result = subprocess.run(
            ["root", "-l", "-b", "-q", f"/depot/cms/private/users/shar1172/copperheadV2_CheckSetup/scripts/muon_pt_reader.C(\"{remote_file}\")"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        output = result.stdout + result.stderr
        if "lzma" in output.lower():
            return (remote_file, False, "LZMA error detected")
        elif "ERRORERROR:" in output:
            return (remote_file, False, "ERRORERROR detected")
        return (remote_file, True, output.strip())

    except subprocess.CalledProcessError as e:
        return (remote_file, False, e.stderr.strip())

# Submit all tasks to Dask futures
futures = [client.compute(process_file(file.replace("/eos/purdue", "root://eos.cms.rcac.purdue.edu:1094/")))
           for file in input_files]

success_count = 0
fail_count = 0

for future in as_completed(futures):
    remote_file, success, message = future.result()
    if success:
        print(f"Success: {message}")
        success_count += 1
    else:
        print(f"Failure processing {remote_file}: {message}")
        with open(log_file, "a") as log:
            log.write(f"{remote_file},{message}\n")
        fail_count += 1

print(f"\nSummary:\nSuccessfully processed files: {success_count}\nFailed files: {fail_count}\nTotal files: {success_count + fail_count}")
