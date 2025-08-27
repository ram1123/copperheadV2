#!/usr/bin/env python3
import os
import glob
import ROOT
import subprocess
import sys
import argparse

"""
# Set env and run the script
voms-proxy-init -verify --rfc --voms cms -valid 192:00
source /cvmfs/oasis.opensciencegrid.org/osg-software/osg-wn-client/current/el8-x86_64/setup.sh
"""

import logging
# from modules.utils import logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info("Logger initialized")


# logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)

def system_with_terminal_display(command, show=False):
    if show:
        logger.info("Executing command: {}".format(command))
    return subprocess.call(command, shell=True)


def isValidRootFile(fname):
    if not os.path.exists(fname):
        return False
    f = ROOT.TFile.Open(fname)
    if not f:
        return False
    try:
        isValid = not (
            f.IsZombie() or f.TestBit(ROOT.TFile.kRecovered) or f.GetListOfKeys().IsEmpty()
        )
    finally:
        f.Close()
    if not isValid:
        logger.warning("Zombie or invalid ROOT file: {}".format(fname))
    return isValid


def checkfaulty(fname, ref=None):
    # If no reference file is provided, compare the file to itself (best-effort)
    close_ref = False
    if not ref:
        ref = ROOT.TFile.Open(fname)
        close_ref = True
    faultyfiles = []
    probe = ROOT.TFile.Open(fname)

    if not probe:
        logger.error("Could not open file {}".format(fname))
        if close_ref and ref:
            ref.Close()
        return False

    for e in ref.GetListOfKeys():
        name = e.GetName()
        try:
            k = probe.GetListOfKeys().FindObject(name)
            if not k:
                raise RuntimeError("Missing key: {}".format(name))
            _ = k.ReadObj()
        except Exception:
            faultyfiles.append(probe.GetName())
            break

    probe.Close()
    if close_ref and ref:
        ref.Close()

    if faultyfiles:
        logger.warning("Faulty files found: {}".format(", ".join(faultyfiles)))
        return False

    return True


def isValidAndFaultFree(fname, ref=None):
    # First check if it's a valid ROOT file
    if not isValidRootFile(fname):
        return False

    # Then check for faulty keys
    return checkfaulty(fname, ref)

def checksum(filename):
    """
    check the xrdadler32 checksum for input file and return the value
    """
    # davs://eos.cms.rcac.purdue.edu:9000
    filenametemp = filename.replace(
        "davs://eos.cms.rcac.purdue.edu:9000", "root://eos.cms.rcac.purdue.edu/"
    )
    logger.info("Calculating checksum for file manually: {}".format(filenametemp))
    os.system("xrdadler32 {}".format(filenametemp))
    logger.info("Calculating checksum for file using subprocess: {}".format(filenametemp))
    result = subprocess.run(
        ["xrdadler32", filenametemp], capture_output=True, text=True
    )
    if result.returncode != 0:
        logger.error("Failed to compute checksum for {}: {}".format(filenametemp, result.stderr))
        return None
    value = (result.stdout.strip()).split()[0]
    logger.info("Checksum value: {}".format(value))
    return value

def searchListFilesWithMemory(inputDir, recursive=False):
    """
    This function does following task:
    1. Search for root files recursively in the `inputDir` (if recursive=True)
    2. Then for each root file create dict with its size in MB
    3. Split that list of root files into the smaller lists such that in each list total sum of file size should be less then 20 GB
    """
    pattern = os.path.join(inputDir, "**", "*.root") if recursive else os.path.join(inputDir, "*.root")
    root_files = glob.glob(pattern, recursive=recursive)
    file_sizes = {f: os.path.getsize(f) / (1024 * 1024) for f in root_files}  # Size in MB
    logger.info("Found {} root files.".format(len(root_files)))

    # Split files into chunks of < 20 GB (20480 MB)
    chunks = []
    current_chunk = []
    current_size = 0.0
    threshold_mb = 20480.0

    for f, size in file_sizes.items():
        if current_size + size < threshold_mb:
            current_chunk.append(f)
            current_size += size
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [f]
            current_size = size

    if current_chunk:
        chunks.append(current_chunk)

    logger.info("Split into {} chunks.".format(len(chunks)))
    return chunks

def merge_files(targetFile, filesToMerge, year):
    logger.info("Merging {} files into: {}".format(len(filesToMerge), targetFile))
    name_of_targetFile = targetFile.split("/")[-1]
    if len(filesToMerge) > 200:
        logger.info("A lot of files to merge; this might take some time...")
        tempTargets = []
        tempFilesToMerge = [
            filesToMerge[x : x + 200] for x in range(0, len(filesToMerge), 200)
        ]

        temp_directory = os.path.join(
            "/depot/cms/hmm/shar1172/HaddingEOS/tmp",
            os.getenv("USER", "default_user"),
            year,
        )
        os.makedirs(temp_directory, exist_ok=True)
        logger.info("Using temporary directory: {}".format(temp_directory))
        for i, batch in enumerate(tempFilesToMerge):
            tempTargetFile = os.path.join(
                temp_directory,
                os.path.basename(targetFile).replace(".root", "-temp{}.root".format(i)),
            )
            logger.info(
                "Merging batch {0} into temp file {1}".format(i, tempTargetFile)
            )
            tempTargets.append(tempTargetFile)
            # Check if temporary target file already exists and is valid
            if os.path.exists(tempTargetFile):
                system_with_terminal_display(
                    "rm {tempTargetFile}".format(tempTargetFile=tempTargetFile),
                    True
                )
                # if isValidRootFile(tempTargetFile):
                #     continue
                # else:
                #     logger.info(
                #         "Removing temp hadd file {tempTargetFile}".format(
                #             tempTargetFile=tempTargetFile
                #         )
                #     )
                #     system_with_terminal_display(
                #         "rm {tempTargetFile}".format(tempTargetFile=tempTargetFile)
                #     )

            system_with_terminal_display(
                "python3 /depot/cms/users/shar1172/copperheadV2_main/haddnano.py  {0} {1}".format(tempTargetFile, " ".join(batch))
                ,False
            )

        # Final merge
        system_with_terminal_display(
            "python3 /depot/cms/users/shar1172/copperheadV2_main/haddnano.py  {0} {1}".format("/depot/cms/hmm/shar1172/HaddingEOS/"+ year + "/" + name_of_targetFile, " ".join(tempTargets))
        )
        system_with_terminal_display(
            "xrdcp -f {0} {1}".format(
                "/depot/cms/hmm/shar1172/HaddingEOS/"+ year + "/" + name_of_targetFile, targetFile
            ),
            False
        )

        # if move success (check also the xrdadler32 checksum?) then remove the local file
        # step-1 check the xrdadler32 checksum for both files and compare
        local_checksum = checksum(targetFile)
        remote_checksum = checksum(
            "/depot/cms/hmm/shar1172/HaddingEOS/" + year + "/" + name_of_targetFile
        )
        logger.info("Local checksum: {}".format(local_checksum))
        logger.info("Remote checksum: {}".format(remote_checksum))

        if local_checksum == remote_checksum:
            logger.info("Checksum verified successfully; removing local file.")
            os.remove(
                "/depot/cms/hmm/shar1172/HaddingEOS/" + year + "/" + name_of_targetFile
            )
        else:
            logger.error("Checksum verification failed; keeping local file.")
            sys.exit()

        # Cleanup
        for tempTarget in tempTargets:
            logger.debug(
                "Removing temp hadd file {tempTarget}".format(tempTarget=tempTarget)
            )
            os.remove(tempTarget)
    else:
        logger.info("Files are < 200; merging directly.")
        logger.debug("python3 /depot/cms/users/shar1172/copperheadV2_main/haddnano.py  {0} {1}".format(targetFile, " ".join(filesToMerge)))
        system_with_terminal_display(
            "python3 /depot/cms/users/shar1172/copperheadV2_main/haddnano.py  {0} {1}".format("/depot/cms/hmm/shar1172/HaddingEOS/"+ year + "/" + name_of_targetFile, " ".join(filesToMerge))
        )
        # move file to final destination
        system_with_terminal_display(
            "xrdcp -f {0} {1}".format(
                "/depot/cms/hmm/shar1172/HaddingEOS/"+ year + "/" + name_of_targetFile, targetFile
            ),
            False
        )
        # if move success (check also the xrdadler32 checksum?) then remove the local file
        # step-1 check the xrdadler32 checksum for both files and compare
        local_checksum = checksum(targetFile)
        remote_checksum = checksum("/depot/cms/hmm/shar1172/HaddingEOS/"+ year + "/" + name_of_targetFile)
        logger.info("Local checksum: {}".format(local_checksum))
        logger.info("Remote checksum: {}".format(remote_checksum))

        if local_checksum == remote_checksum:
            logger.info("Checksum verified successfully; removing local file.")
            os.remove("/depot/cms/hmm/shar1172/HaddingEOS/"+ year + "/" + name_of_targetFile)
        else:
            logger.error("Checksum verification failed; keeping local file.")
            sys.exit()

def main():
    parser = argparse.ArgumentParser(description="Merge ROOT files using haddnano.py.")
    parser.add_argument(
        "-i",
        "--inputDir",
        type=str,
        required=True,
        help="Path of the input directory that contains ROOT files to be merged.",
    )
    parser.add_argument(
        "-o",
        "--outputDir",
        type=str,
        required=True,
        help="Path of the output directory where the merged ROOT file will be saved.",
    )
    parser.add_argument(
        "-f",
        "--outputFile",
        type=str,
        required=True,
        help="Name of the hadd-ed output ROOT file.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search for ROOT files recursively in the input directory.",
    )
    parser.add_argument(
        "-y",
        "--year",
        type=str,
        required=True,
        help="Year of the data (e.g. 2016, 2017, 2018)."
    )

    args = parser.parse_args()

    inputDir = args.inputDir
    outputDir = args.outputDir
    outputFile = args.outputFile

    # Create output directory, NOTE the redirector
    system_with_terminal_display(
        "gfal-mkdir {}".format("davs://eos.cms.rcac.purdue.edu:9000" + outputDir)
    )

    # NOTE: different redirector for the output path and creation of directory in above command.
    outputDir = "root://eos.cms.rcac.purdue.edu/" + outputDir

    logger.info("Input directory: {}".format(inputDir))
    logger.info("Output directory: {}".format(outputDir))
    logger.info("Output file: {}".format(outputFile))

    if not os.path.isdir(inputDir):
        logger.error(
            "The specified input directory does not exist: {}".format(inputDir)
        )
        sys.exit(1)

    filelist = searchListFilesWithMemory(inputDir, recursive=args.recursive)

    logger.debug("File list (chunked): {}".format(filelist))

    if not filelist:
        logger.error("No ROOT files found to merge in {} (recursive={}).".format(inputDir, args.recursive))
        sys.exit(1)

    targetFile = os.path.join(outputDir, outputFile)

    logger.info("Merging {} chunks into parts under {}".format(len(filelist), outputDir))
    logger.debug("Files to merge: {}".format(filelist))

    for count, filesToMerge in enumerate(filelist):
        logger.debug("Merging file {}/{}: {}".format(count + 1, len(filelist), filesToMerge))
        merge_files(targetFile.replace(".root", "_part{}.root".format(count + 1)), filesToMerge, args.year)

if __name__ == "__main__":
    main()
