import pandas as pd
from pathlib import Path
import subprocess
import time

# Input CSV file and output directory
inFile = "Mismatch_MiniNano_nNano0_nMini0_2018GT36_5April_2018GT36.csv"
outDir = "/depot/cms/hmm/shar1172/FailedJobsMiniAODRootFiles"

# Read the CSV file and extract file paths (assuming second column holds paths)
df = pd.read_csv(inFile, header=None)
miniAOD_files = df.iloc[:, 1].tolist()

# List of fallback redirectors
fallbacks = [
    "root://xcache.cms.rcac.purdue.edu/",
    "root://cmsxrootd.fnal.gov/",
    "root://cms-xrd-global.cern.ch/",
    "root://xrootd-cms.infn.it/",
    "root://dcache-cms-xrootd.desy.de:1094/"
]

# Settings for xrdcp options (retries handled internally by xrdcp)
retries = 3  # used for --retry option
sleep_time = 3  # seconds to wait between fallback attempts

def build_command(src, dest):
    return [
        "xrdcp",
        "--retry", str(retries),
        "--retry-policy", "force",
        "--continue",
        "--streams", "2",
        "--cksum", "adler32:source",
        "--rm-bad-cksum",
        "--nopbar",
        "--silent",
        # "-f",
        src,
        dest,
    ]

# Loop over each file from the CSV
for i, file in enumerate(miniAOD_files):
    if i == 0:  # Skip header if present
        continue

    print(f"\nProcessing file {i}: {file}")
    file_path = Path(file)
    fileName = file_path.name
    # Recreate directory structure based on the file path components
    dirStructure = "/".join(file_path.parts[2:-1])  # skip protocol and base parts
    outDirPath = Path(outDir) / dirStructure
    outDirPath.mkdir(parents=True, exist_ok=True)

    # Build the primary xrdcp source URL by doing needed replacements
    xrd_src = str(file_path).replace("root:/xcache", "root://xcache").replace("edu/store", "edu//store")
    xrd_dest = str(outDirPath / fileName)

    primary_command = build_command(xrd_src, xrd_dest)
    # print(f"  → Copying to: {xrd_dest} from {xrd_src}")
    print(f"  -> copy command: {' '.join(primary_command)}")

    success = False
    try:
        subprocess.run(primary_command, capture_output=True, text=True, check=True)
        print("  ✅ Copied successfully with primary redirector.")
        success = True
    except subprocess.CalledProcessError as e:
        print("  ⚠️  Primary redirector failed.")
        print(f"      STDOUT: {e.stdout.strip()}")
        print(f"      STDERR: {e.stderr.strip()}")

    # If the primary attempt fails, try fallback redirectors
    if not success:
        for fallback in fallbacks:
            # Replace the protocol part of the source with the fallback.
            # Assuming the original xrd_src starts with "root://", split it to get the path.
            prefix = "root://"
            if xrd_src.startswith(prefix):
                parts = xrd_src[len(prefix):].split("/", 1)
                if len(parts) == 2:
                    new_src = fallback.rstrip("/") + "/" + parts[1]
                else:
                    new_src = fallback.rstrip("/") + "/"
            else:
                new_src = fallback.rstrip("/") + "/" + xrd_src

            print(f"  → Trying fallback: {new_src}")
            fallback_command = build_command(new_src, xrd_dest)
            print(f"  -> copy command: {' '.join(fallback_command)}")
            try:
                subprocess.run(fallback_command, capture_output=True, text=True, check=True)
                print(f"  ✅ Copied successfully with fallback: {fallback}")
                success = True
                break
            except subprocess.CalledProcessError as fe:
                print(f"  ⚠️  Fallback {fallback} failed.")
                print(f"      STDOUT: {fe.stdout.strip()}")
                print(f"      STDERR: {fe.stderr.strip()}")
                time.sleep(sleep_time)
        if not success:
            print("  ❌ Failed to copy file with all redirectors.")
            with open("error_files.txt", "a") as error_log:
                error_log.write(f"{file}\n")

    if i > 1: exit()

print("\n📦 All files processed.")
