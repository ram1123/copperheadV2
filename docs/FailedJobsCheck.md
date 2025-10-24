# Check files for LZMA compression errors

- Script is located here: [scripts/main_script_dask_to_run_cppProgram.py](../scripts/main_script_dask_to_run_cppProgram.py)
    - This script needs only the path of the folder with the files to check
    - It returns a text file with the names of the files that have errors

## Delete the corrupted files
- The script is located here: [scripts/delete_corrupted_files.py](../scripts/delete_corrupted_files.py)
    - This script needs the path of the text file with the names of the corrupted files
    - It will delete the corrupted files from the folder where they are located
    - **NOTE:** Run this script from the fresh terminal and only setup the proxy nothing else, otherwise it will not work properly.

# Check for the failed jobs

- The script is located here: [scripts/tag_corrupted_file_from_nanoProduction.py](../scripts/tag_corrupted_file_from_nanoProduction.py)
    - Need to update two hardcoded informations:
        1. The text file that we used to submit the jobs having columns: ['configFile', 'inputMiniAOD', 'outputDirectory', 'nEvents', 'CondorLogPath']
        2. The path where one can find the output nanoAOD root files
