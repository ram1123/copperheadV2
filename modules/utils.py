import logging
from rich.logging import RichHandler
import os
import sys
import awkward as ak
from pathlib import Path

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
# logger.setLevel(logging.INFO)
# logger.setLevel(logging.WARNING)
logger.setLevel(logging.ERROR)

def ifPathExists(load_path):
    if not os.path.exists(load_path):
        logger.error(f"Path: {load_path} does not exists")
        sys.exit()
    else:
        logger.info(f"Path exists: {load_path}")


def get_compacted_path(stage1_path):
    """
    Prefer a sibling `compacted` directory when the provided stage1 path points
    at `.../f1_0`, otherwise fall back to the original path.

    This is intentionally a light existence check only; it does not validate
    that every expected sample exists under the chosen directory.
    """
    stage1_path = Path(stage1_path)
    compacted_stage1_path = Path(str(stage1_path).replace("/f1_0", "/compacted"))
    logger.debug(f"compacted_stage1_path: {compacted_stage1_path}")
    if os.path.isdir(compacted_stage1_path):
        return compacted_stage1_path
    if os.path.isdir(stage1_path):
        return stage1_path

    logger.critical(
        f"Neither {compacted_stage1_path} nor {stage1_path} exists! Exiting!"
    )
    raise FileNotFoundError(
        f"Neither {compacted_stage1_path} nor {stage1_path} exists! Exiting!"
    )

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


def fillEventNans(events, category="vbf"):
    """
    checked that this function is unnecssary for vbf category, but have it for robustness
    """
    if category == "vbf":
        for field in events.fields:
            if "phi" in field:
                events[field] = ak.fill_none(events[field], value=-10) # we're working on a DNN, so significant deviation may be warranted
            else: # for all other fields (this may need to be changed)
                events[field] = ak.fill_none(events[field], value=0)
    else:
        logger.info("ERROR: unsupported category!")
        raise ValueError
    return events



def fillSampleValues(events, sample_dict, sample: str, fields2load=None):
    """
    inputs:
    sample_dict: dictionary with sample name as keys, lazy dask awkward zip as values
    return
    computed sample_dict: dictionary with sample name as keys, eager dask awkward zip as values
    Description:
    takes lazy dask awkward zips from sample_dict and computes only fields specified in fields2load. This is then returned
    """
    # find which sample group sample_name belongs to
    if fields2load is None:
        fields2load = ["wgt_nominal", "BDT_score", "dimuon_mass", "subCategory_idx"]
    if sample in sample_dict.keys():
        # compute in parallel fields to load
        computed_zip = ak.zip({
            field : events[field] for field in fields2load
        }).compute()
        for field in fields2load:
            sample_dict[sample][field] = ak.to_numpy(computed_zip[field])

    else:
        print(f"sample {sample} not present in sample_dict!")

    return sample_dict


def getDimuMassBySubCat(sample_dict, sample="", nSubCats=5):
    dimuon_mass = sample_dict[sample]["dimuon_mass"]
    wgt_nominal = sample_dict[sample]["wgt_nominal"]
    subCat_ixs = sample_dict[sample]["subCategory_idx"]
    dict_by_subCat = {}
    for target_subCat in range(nSubCats):
        subCat_filter = target_subCat == subCat_ixs
        dimuon_mass_subCat = dimuon_mass[subCat_filter]
        wgt_nominal_subCat = wgt_nominal[subCat_filter]
        subCat_dict = {
            "dimuon_mass" : dimuon_mass_subCat,
            "wgt_nominal" : wgt_nominal_subCat,
        }
        dict_by_subCat[target_subCat] = subCat_dict
    return dict_by_subCat


def pair_and_remove(df, cols=("mu1_eta","mu2_eta"), wgt_col="wgt_nominal"):
    """
    Pair each negative-weight row with at most one positive-weight row
    using Hungarian algorithm (minimizing L1 distance).
    Remove the paired rows from the original df.
    Returns:
        matches_df: DataFrame with match info (neg_idx, pos_idx, dist, ...)
        remaining_df: df with matched rows removed
    """    
    # Split
    neg = df[df[wgt_col] < 0].copy()
    pos = df[df[wgt_col] > 0].copy()

    if len(neg) == 0 or len(pos) == 0:
        return pd.DataFrame(), df.copy()

    # Arrays
    X = neg.loc[:, cols].to_numpy(dtype=float)
    Y = pos.loc[:, cols].to_numpy(dtype=float)
    print(f"X: {X}")
    print(f"y: {Y}")
    # Cost matrix (L1 distance)
    cost = np.abs(X[:, None, :] - Y[None, :, :]).sum(axis=2)
    print(f"cost: {cost}")

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    print(f"row_ind: {row_ind}")
    print(f"col_ind: {col_ind}")

    # Map back to indices
    neg_idx = neg.index.to_numpy()[row_ind]
    pos_idx = pos.index.to_numpy()[col_ind]
    dists   = cost[row_ind, col_ind]

    # Matches dataframe
    data = {
        "neg_idx": neg_idx,
        "pos_idx": pos_idx,
        "dist": dists,
        "neg_wgt": df.loc[neg_idx, wgt_col].to_numpy(),
        "pos_wgt": df.loc[pos_idx, wgt_col].to_numpy(),
    }
    for c in cols:
        data[f"neg_{c}"] = df.loc[neg_idx, c].to_numpy()
        data[f"pos_{c}"] = df.loc[pos_idx, c].to_numpy()

    matches_df = pd.DataFrame(data).sort_values("dist").reset_index(drop=True)

    # Drop matched rows from original df
    matched_indices = np.concatenate([neg_idx, pos_idx])
    remaining_df = df.drop(index=matched_indices).reset_index(drop=True)

    return matches_df, remaining_df

def getSqrtSOverB(bin_edges, sig_counts, bkg_counts, save_path, fname):
    # Example: signal and background histograms
    # bin_edges = np.array([0, 1, 2, 3, 4, 5])
    # sig_counts = np.array([5, 10, 20, 15, 5])
    # bkg_counts = np.array([50, 40, 30, 20, 10])

    # Cumulative yields above each bin edge
    S_cum = np.cumsum(sig_counts[::-1])[::-1]
    B_cum = np.cumsum(bkg_counts[::-1])[::-1]

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        significance = np.sqrt(S_cum) / B_cum
        significance[B_cum <= 0] = np.nan  # mask bins with no background

    # Compute bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Plot vs bin centers
    plt.plot(bin_centers, significance, marker='o', label=r'$\sqrt{S}/B$')
    plt.xlabel("Variable (bin center)")
    plt.ylabel(r"$\sqrt{S}/B$")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(f"{save_path}/{fname}.pdf")
