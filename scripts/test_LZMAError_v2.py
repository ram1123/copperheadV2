import hepconvert
from pathlib import Path
import uproot
import os
import traceback
from tqdm import tqdm
from multiprocessing import Pool
import tempfile
from datetime import datetime
import uuid

# Input directory
# inDir = Path("/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8")
inDir = Path("/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9")
# inDir = Path("/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8")
files = list(inDir.glob("*.root"))

def test_file(file):
    try:
        tree = uproot.open(f"{file}:Events")

        # 1. Try to read Muon_pt
        _ = tree.arrays(["Muon_pt"], entry_stop=10)

        # fd, tmp_path = tempfile.mkstemp(suffix=".parquet", dir="/tmp")
        # os.close(fd)  # Close the file descriptor so hepconvert can write

        # print(f"temp_path: {tmp_path}")
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
            os.remove(tmp_path)  # Clean up temp file

        return (file, True)

    except Exception as e:
        if "lzma data error" in str(e).lower():
            return (file, False)
        else:
            return (file, traceback.format_exc())  # Capture full traceback

# Output files
bad_files_path = "lzma_corrupted_files.txt"
other_errors_path = "other_errors.log"

bad_files = []
other_errors = []

# Parallel scan
with Pool(processes=16) as pool:
    for f, status in tqdm(pool.imap_unordered(test_file, files), total=len(files)):
        if status is False:
            bad_files.append(str(f))
        elif status != True:
            other_errors.append((str(f), status))

# Save LZMA errors
with open(bad_files_path, "w") as f:
    f.write(f"# LZMA-corrupted files found on {datetime.now()}\n\n")
    f.write("\n".join(bad_files))

# Save other errors with tracebacks
with open(other_errors_path, "w") as f:
    f.write(f"# Other errors encountered on {datetime.now()}\n\n")
    for fpath, err_msg in other_errors:
        f.write(f"--- {fpath} ---\n")
        f.write(err_msg)
        f.write("\n\n")

# Summary
print(f"\nScan completed.")
print(f"{len(bad_files)} LZMA-corrupted files logged to: {bad_files_path}")
print(f"{len(other_errors)} other error(s) logged to: {other_errors_path}")
