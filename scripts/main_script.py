import subprocess
import glob

input_files = glob.glob(
    "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/"
    "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/*.root"
)

log_file = "TTTo2L2Nu_TuneCP5_13TeV_errors.txt"
success_count = 0
fail_count = 0

for file in input_files:
    remote_file = file.replace("/eos/purdue", "root://eos.cms.rcac.purdue.edu:1094/")

    try:
        result = subprocess.run(
            ["python3", "process_root_file.py", remote_file],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout.strip())
        success_count += 1

    except subprocess.CalledProcessError as e:
        print(f"ROOT subprocess failed for file: {remote_file}")
        print("Error message:", e.stderr.strip())
        with open(log_file, "a") as log:
            log.write(f"{remote_file}, Error: {e.stderr.strip()}\n")
        fail_count += 1

print(f"\nSummary:\nSuccessfully processed files: {success_count}\nFailed files: {fail_count}\nTotal files: {success_count + fail_count}")
