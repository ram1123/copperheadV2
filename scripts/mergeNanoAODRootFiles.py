#!/usr/bin/env python3
import os
import glob
import ROOT
import subprocess
import sys
import argparse
from pathlib import Path

"""
Set up the environment before running:

    voms-proxy-init -verify --rfc --voms cms -valid 192:00
    source /cvmfs/oasis.opensciencegrid.org/osg-software/osg-wn-client/current/el8-x86_64/setup.sh

Example:

    python scripts/mergeNanoAODRootFiles.py \
      -i /eos/purdue/store/user/.../Sample \
      -o /store/user/<user>/merged/UL2018/Sample \
      -f Sample.root \
      -y UL2018

time python scripts/mergeNanoAODRootFiles.py \
    -i /eos/purdue/store/user/rasharma/customNanoAODv15/UL2018/DYJets_NanoAOD \
    -o /eos/purdue/store/user/rasharma/customNanoAODv15/hadded/UL2018/DYJets_NanoAOD \
    -f DY_VBF_filter.root \
    -y UL2018


time python scripts/mergeNanoAODRootFiles.py \
    -i /eos/purdue/store/user/rasharma/customNanoAODv15/UL2017/DYJets_NanoAOD \
    -o /eos/purdue/store/user/rasharma/customNanoAODv15/hadded/UL2017/DYJets_NanoAOD \
    -f DY_VBF_filter.root \
    -y UL2017


time python scripts/mergeNanoAODRootFiles.py \
    -i /eos/purdue/store/user/rasharma/customNanoAODv15/UL2016/DYJets_NanoAOD \
    -o /eos/purdue/store/user/rasharma/customNanoAODv15/hadded/UL2016/DYJets_NanoAOD \
    -f DY_VBF_filter.root \
    -y UL2016


time python scripts/mergeNanoAODRootFiles.py \
    -i /eos/purdue/store/user/rasharma/customNanoAODv15/UL2016APV/DYJets_NanoAOD \
    -o /eos/purdue/store/user/rasharma/customNanoAODv15/hadded/UL2016APV/DYJets_NanoAOD \
    -f DY_VBF_filter.root \
    -y UL2016APV
"""

from modules.utils import logger


DEFAULT_HADDNANO_PATH = Path(__file__).resolve().parents[1] / "haddnano.py"
DEFAULT_LOCAL_TMP_ROOT = Path("/tmp") / "mergeNanoAODRootFiles"
DEFAULT_EOS_DAVS_PREFIX = "davs://eos.cms.rcac.purdue.edu:9000"
DEFAULT_EOS_ROOT_PREFIX = "root://eos.cms.rcac.purdue.edu/"


def normalize_eos_dir(path, eos_davs_prefix, eos_root_prefix):
    path = path.rstrip("/")

    if path.startswith(eos_davs_prefix):
        return path, path.replace(eos_davs_prefix, eos_root_prefix, 1)
    if path.startswith(eos_root_prefix):
        return path.replace(eos_root_prefix, eos_davs_prefix, 1), path
    if path.startswith("/eos/purdue/store/"):
        store_path = path.replace("/eos/purdue", "", 1)
        return eos_davs_prefix + store_path, eos_root_prefix + store_path
    if path.startswith("/store/"):
        return eos_davs_prefix + path, eos_root_prefix + path

    raise ValueError(
        "Unsupported outputDir '{}'. Use /store/..., /eos/purdue/store/..., davs://..., or root://...".format(path)
    )


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

def checksum(filename, eos_davs_prefix, eos_root_prefix):
    """
    check the xrdadler32 checksum for input file and return the value
    """
    if filename.startswith("/eos/purdue/store/"):
        filenametemp = filename.replace("/eos/purdue", eos_root_prefix.rstrip("/"), 1)
    else:
        filenametemp = filename.replace(eos_davs_prefix, eos_root_prefix)
    logger.info("Calculating checksum for file manually: {}".format(filenametemp))
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

    # Split files into chunks of < 4 GB (4096 MB)
    chunks = []
    current_chunk = []
    current_size = 0.0
    threshold_mb = 4096.0 #  4 GB in MB

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

def merge_files(targetFile, filesToMerge, year, haddnano_path, local_tmp_root, eos_davs_prefix, eos_root_prefix):
    logger.info("Merging {} files into: {}".format(len(filesToMerge), targetFile))
    name_of_targetFile = targetFile.split("/")[-1]
    local_year_dir = Path(local_tmp_root) / os.getenv("USER", "default_user") / year
    local_year_dir.mkdir(parents=True, exist_ok=True)
    local_target = str(local_year_dir / name_of_targetFile)

    if len(filesToMerge) > 200:
        logger.info("A lot of files to merge; this might take some time...")
        tempTargets = []
        tempFilesToMerge = [
            filesToMerge[x : x + 200] for x in range(0, len(filesToMerge), 200)
        ]

        logger.info("Using temporary directory: {}".format(local_year_dir))
        for i, batch in enumerate(tempFilesToMerge):
            tempTargetFile = str(
                local_year_dir / os.path.basename(targetFile).replace(".root", "-temp{}.root".format(i))
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

            system_with_terminal_display(
                "python3 {0} {1} {2}".format(haddnano_path, tempTargetFile, " ".join(batch)),
                False,
            )

        system_with_terminal_display(
            "python3 {0} {1} {2}".format(haddnano_path, local_target, " ".join(tempTargets))
        )
        system_with_terminal_display(
            "xrdcp -f {0} {1}".format(local_target, targetFile),
            False
        )

        remote_checksum = checksum(targetFile, eos_davs_prefix, eos_root_prefix)
        local_checksum = checksum(local_target, eos_davs_prefix, eos_root_prefix)
        logger.info("Local checksum: {}".format(local_checksum))
        logger.info("Remote checksum: {}".format(remote_checksum))

        if local_checksum == remote_checksum:
            logger.info("Checksum verified successfully; removing local file.")
            os.remove(local_target)
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
        logger.debug("python3 {0} {1} {2}".format(haddnano_path, local_target, " ".join(filesToMerge)))
        system_with_terminal_display(
            "python3 {0} {1} {2}".format(haddnano_path, local_target, " ".join(filesToMerge))
        )
        system_with_terminal_display(
            "xrdcp -f {0} {1}".format(local_target, targetFile),
            False
        )
        # if move success (check also the xrdadler32 checksum?) then remove the local file
        # step-1 check the xrdadler32 checksum for both files and compare
        remote_checksum = checksum(targetFile, eos_davs_prefix, eos_root_prefix)
        local_checksum = checksum(local_target, eos_davs_prefix, eos_root_prefix)
        logger.info("Local checksum: {}".format(local_checksum))
        logger.info("Remote checksum: {}".format(remote_checksum))

        if local_checksum == remote_checksum:
            logger.info("Checksum verified successfully; removing local file.")
            os.remove(local_target)
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
        help="Year of the data (e.g. UL2016postVFP, UL2017, UL2018)."
    )
    parser.add_argument(
        "--haddnano-path",
        type=str,
        default=str(DEFAULT_HADDNANO_PATH),
        help="Path to haddnano.py. Default: repo-local haddnano.py",
    )
    parser.add_argument(
        "--local-tmp-root",
        type=str,
        default=str(DEFAULT_LOCAL_TMP_ROOT),
        help="Local staging root for intermediate merged files.",
    )
    parser.add_argument(
        "--eos-davs-prefix",
        type=str,
        default=DEFAULT_EOS_DAVS_PREFIX,
        help="DAVS prefix used when creating EOS directories.",
    )
    parser.add_argument(
        "--eos-root-prefix",
        type=str,
        default=DEFAULT_EOS_ROOT_PREFIX,
        help="root:// prefix used for xrdcp and xrdadler32.",
    )

    args = parser.parse_args()

    inputDir = args.inputDir
    outputDir = args.outputDir
    outputFile = args.outputFile

    # Create output directory, NOTE the redirector
    outputDir_davs, outputDir_root = normalize_eos_dir(
        outputDir,
        args.eos_davs_prefix,
        args.eos_root_prefix,
    )

    system_with_terminal_display(
        "gfal-mkdir -p {}".format(outputDir_davs)
    )

    logger.info("Input directory: {}".format(inputDir))
    logger.info("Output directory (DAVS): {}".format(outputDir_davs))
    logger.info("Output directory (ROOT): {}".format(outputDir_root))
    logger.info("Output file: {}".format(outputFile))
    logger.info("haddnano.py path: {}".format(args.haddnano_path))
    logger.info("Local tmp root: {}".format(args.local_tmp_root))

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

    targetFile = os.path.join(outputDir_root, outputFile)

    logger.info("Merging {} chunks into parts under {}".format(len(filelist), outputDir_root))
    logger.debug("Files to merge: {}".format(filelist))

    for count, filesToMerge in enumerate(filelist):
        logger.debug("Merging file {}/{}: {}".format(count + 1, len(filelist), filesToMerge))
        merge_files(
            targetFile.replace(".root", "_part{}.root".format(count + 1)),
            filesToMerge,
            args.year,
            args.haddnano_path,
            args.local_tmp_root,
            args.eos_davs_prefix,
            args.eos_root_prefix,
        )


if __name__ == "__main__":
    main()
