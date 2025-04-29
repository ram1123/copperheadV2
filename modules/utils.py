import logging
from rich.logging import RichHandler
from rich.console import Console
from typing import Optional
import os
import sys

LOGGER_NAME = "CopperHead"
NO_GIT_INFO_AVAILABLE = "No git info available"

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ColorLogFormatter(logging.Formatter):
     """A class for formatting colored logs.
     Reference: https://stackoverflow.com/a/70796089/2302094
     """

     # FORMAT = "%(prefix)s%(msg)s"
    #  FORMAT = "\n[%(levelname)s] - [%(filename)s:#%(lineno)d] - %(prefix)s%(levelname)s - %(message)s %(suffix)s\n"
     FORMAT = "\n{}[%(levelname)5s] - [%(filename)s:#%(lineno)d] - [%(funcName)s; %(module)s]{} - %(prefix)s%(message)s %(suffix)s\n".format(
         bcolors.HEADER, bcolors.ENDC
     )
    #  FORMAT = "\n%(asctime)s - [%(filename)s:#%(lineno)d] - %(prefix)s%(levelname)s - %(message)s %(suffix)s\n"

     LOG_LEVEL_COLOR = {
         "DEBUG": {'prefix': bcolors.OKBLUE, 'suffix': bcolors.ENDC},
         "INFO": {'prefix': bcolors.OKGREEN, 'suffix': bcolors.ENDC},
         "WARNING": {'prefix': bcolors.WARNING, 'suffix': bcolors.ENDC},
         "CRITICAL": {'prefix': bcolors.FAIL, 'suffix': bcolors.ENDC},
         "ERROR": {'prefix': bcolors.FAIL+bcolors.BOLD, 'suffix': bcolors.ENDC+bcolors.ENDC},
     }

     def format(self, record):
         """Format log records with a default prefix and suffix to terminal color codes that corresponds to the log level name."""
         if not hasattr(record, 'prefix'):
             record.prefix = self.LOG_LEVEL_COLOR.get(record.levelname.upper()).get('prefix')

         if not hasattr(record, 'suffix'):
             record.suffix = self.LOG_LEVEL_COLOR.get(record.levelname.upper()).get('suffix')

         formatter = logging.Formatter(self.FORMAT, datefmt='%m/%d/%Y %I:%M:%S %p' )
         return formatter.format(record)

logger = logging.getLogger(LOGGER_NAME) # need to give it a name, otherwise *way* too much info gets printed out from e.g. numba
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(ColorLogFormatter())

# Set up stream handler (for stdout)
formatter = logging.Formatter("%(message)s")
stream_handler = RichHandler(show_time=False, rich_tracebacks=True,tracebacks_word_wrap=False)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)

def ifPathExists(load_path):
    if not os.path.exists(load_path):
        logger.error(f"Path: {load_path} does not exists")
        sys.exit()
    else:
        logger.info(f"Path exists: {load_path}")

def get_git_info():
    """Get the current git commit hash, branch name, and the difference between the current version of the code and the last commit.
    Returns:
        tuple: A tuple containing the commit hash, branch name, and the difference.
    """
    try:
        import subprocess
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "diff"], text=True).strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Subprocess error while getting git info: {e}")
        commit_hash, branch_name, diff = None, None, None
    except Exception as e:
        logger.error(f"Unexpected error while getting git info: {e}")
        commit_hash, branch_name, diff = None, None, None

    return commit_hash, branch_name, diff

def get_git_info_str():
    """Get the current git commit hash, branch name, and the difference as a string.
    Returns:
        str: A string containing the commit hash, branch name, and the difference.
    """
    commit_hash, branch_name, diff = get_git_info()
    if commit_hash is None or branch_name is None:
        return NO_GIT_INFO_AVAILABLE
    else:
        return f"Commit: {commit_hash}, Branch: {branch_name}, Diff: {diff}"
