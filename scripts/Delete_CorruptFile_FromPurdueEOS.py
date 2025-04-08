import os
import sys
import pandas as pd

import os

def delete_file(file_path, log_file="delete_commands.sh"):
    """Delete a file and log the gfal-rm command."""
    # Replace root://eos.cms.rcac.purdue.edu/ with davs://eos.cms.rcac.purdue.edu:9000
    file_path = file_path.replace("root://eos.cms.rcac.purdue.edu/", "davs://eos.cms.rcac.purdue.edu:9000")

    # the gfal-rm command
    command = f"gfal-rm {file_path}"

    # Append the command
    with open(log_file, 'a') as f:
        f.write(command + '\n')

    # Execute the gfal-rm command to delete the file
    # os.system(command)

# Function to get the expected output file name then delete it
def get_expected_output_file(input_file):
    """Get the expected output file name from the input file."""
    # Load the input file into a pandas DataFrame
    df = pd.read_csv(input_file, sep=' ', header=None, names=['configFile', 'inputMiniAOD', 'outputDirectory', 'nEvents', 'CondorLogPath'])

    # Generate the expected output file names
    df['outputNanoAODFile'] = df['inputMiniAOD'].apply(lambda x: x.split('/')[-1].replace(".root", "_NanoAOD.root"))
    df['expected_output_file'] = df['outputDirectory'] + "/" + df['outputNanoAODFile']

    # Delete the expected output files
    for file_path in df['expected_output_file']:
        delete_file(file_path)
        # sys.exit(0)


def main():
    # Specify the path to the input text file
    input_file = "missing_or_corrupt_files_2018_24Feb.txt"  # Change this to your actual file path

    # Get expected output files and delete them
    get_expected_output_file(input_file)

if __name__ == "__main__":
    main()
