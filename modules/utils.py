import logging
from rich.logging import RichHandler
from rich.console import Console
from typing import Optional
import os
import sys
import numpy as np
import awkward as ak
import ROOT
import ROOT as rt
import matplotlib.pyplot as plt

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

def filterRegion(events, region="h-peak"):
    dimuon_mass = events.dimuon_mass
    if region =="h-peak":
        region = (dimuon_mass > 115) & (dimuon_mass < 135)
    elif region =="h-sidebands":
        region = ((dimuon_mass > 110) & (dimuon_mass < 115)) | ((dimuon_mass > 135) & (dimuon_mass < 150))
    elif region =="signal":
        region = (dimuon_mass >= 110) & (dimuon_mass <= 150.0)
    elif region =="z-peak":
        region = (dimuon_mass >= 70) & (dimuon_mass <= 110.0)

    events = events[region]
    return events

def convertVectorType4D(vector, vector_name):
    new_vector = ak.zip(
        {
            "pt": vector.pt,
            "eta": vector.eta,
            "phi": vector.phi,
            "mass": vector.mass,
            "charge": vector.charge,
        },
        with_name=vector_name,
        behavior=vector.behavior,
    )

def print_workspace_vars(ws):
    for var in ws.allVars():
        name = var.GetName()
        val  = var.getVal()
        vmin = var.getMin()
        vmax = var.getMax()
        is_const = var.isConstant()
        print(
            f"{name:20s} = {val:10.5f} "
            f"range = [{vmin:10.5f}, {vmax:10.5f}] "
            f"{'(const)' if is_const else ''}"
        )


def freeze_all_vars(w, make_exception=[]):
    for v in w.allVars():
        do_freeze = True
        name = v.GetName()
        # print(f"name: {name}")
        for exception_name in make_exception:
            if (exception_name in name) or (exception_name == name): # skip
                do_freeze = False
                continue
        if do_freeze:
            v.setConstant(True)

def getNewRangeHist(hist2copy: rt.TH1D, new_hist_name: str, xlow_new: float, xhigh_new: float):
    """
    """
    # get binwise content
    # Initialize result dictionary
    bin_dict = {"xlow": [], "xhigh": [], "content": []}
    
    # Fill dictionary with bin info
    hist = hist2copy
    for i in range(1, hist.GetNbinsX() + 1):
        xlow = hist.GetBinLowEdge(i)
        xhigh = xlow + hist.GetBinWidth(i)
        content = hist.GetBinContent(i)
        if (xlow >= xlow_new) and (xhigh <= xhigh_new):
            bin_dict["xlow"].append(xlow)
            bin_dict["xhigh"].append(xhigh)
            bin_dict["content"].append(content)
        
    # print(bin_dict)
    new_nbins = len(bin_dict["content"])

    new_hist = rt.TH1D(new_hist_name, new_hist_name, new_nbins, xlow_new, xhigh_new)
    for i in range(1, new_hist.GetNbinsX() + 1):
        content = bin_dict["content"][i-1]
        new_hist.SetBinContent(i, content)

    # # sanity check
    # for i in range(1, new_hist.GetNbinsX() + 1):
    #     xlow = new_hist.GetBinLowEdge(i)
    #     xhigh = xlow + new_hist.GetBinWidth(i)
    #     content = new_hist.GetBinContent(i)
    #     print(f"{new_hist_name} bin{i} {(xlow, xhigh, content)}")
    # print(f"new_nbins: {new_nbins}")
    # raise ValueError
    return new_hist

def getGOF_KS(x: rt.RooRealVar, data: rt.RooDataHist, pdf: rt.RooAbsPdf, cat_name:str, save_path:str):
    """
    Get KS value for specific value
    """
    # raise ValueError
    nbins = x.getBins()
    var_name = x.GetName()
    # # Generate toy dataset
    # data = pdf.generate(rt.RooArgSet(x), 1000)
    hist_data_orig = data.createHistogram(var_name).Clone("clone")# clone it just in case
    # Create a histogram of the PDF
    hist_pdf_orig = pdf.createHistogram(var_name, nbins)


    plot_line_width = 0.5
    # -------------------------------------------
    # Sanity check: plot the pdf and cdf of signal region PDF
    # -------------------------------------------
    # data_counts = np.array([hist_data_orig.GetBinContent(i) for i in range(1, hist_data_orig.GetNbinsX()+1)])
    pdf_counts = np.array([hist_pdf_orig.GetBinContent(i) for i in range(1, hist_pdf_orig.GetNbinsX()+1)])
    # print(f"data_counts: {data_counts}")
    # print(f"pdf_counts: {pdf_counts}")
    pdf_cdf = np.cumsum(pdf_counts) / np.sum(pdf_counts)
    
    bin_centers = []
    hist = hist_pdf_orig
    for i in range(1, hist.GetNbinsX() + 1):  # 1-based bin index
        center = hist.GetBinCenter(i)
        bin_centers.append(center)
    bin_centers = np.array(bin_centers)
    plt.plot(bin_centers, pdf_cdf, label='PDF CDF')
    plt.xlabel('mass')
    plt.ylabel('')
    plt.legend()
    plt.grid(True)
    if save_path != "":
        plt.savefig(f"{save_path}/GoF_cdfs_SignalFitRange_{cat_name}.pdf")
    plt.clf()
    
    # Draw the normalized pdf histogram:
    plt.plot(bin_centers, pdf_counts/np.sum(pdf_counts), label='PDF PDF')
    plt.xlabel('mass')
    plt.ylabel('')
    plt.legend()
    plt.grid(True)
    if save_path != "":
        plt.savefig(f"{save_path}/GoF_pdfs_SignalFitRange_{cat_name}.pdf")
    plt.clf()


    # -------------------------------------------
    # Do KS test
    # -------------------------------------------
    return_dict = {}
    for test_range_name in ["loSB", "hiSB"]:
        xlow, xhigh = list(x.getRange(test_range_name))
        data_hist_new = getNewRangeHist(hist_data_orig, f"data_hist_{test_range_name}", xlow, xhigh)
        pdf_hist_new = getNewRangeHist(hist_pdf_orig, f"pdf_hist_{test_range_name}", xlow, xhigh)
        data_counts = np.array([data_hist_new.GetBinContent(i) for i in range(1, data_hist_new.GetNbinsX()+1)])
        pdf_counts = np.array([pdf_hist_new.GetBinContent(i) for i in range(1, pdf_hist_new.GetNbinsX()+1)])
        # print(f"data_counts: {data_counts}")
        # print(f"pdf_counts: {pdf_counts}")
        # nevents=np.sum(data_counts)
        data_cdf = np.cumsum(data_counts) / np.sum(data_counts)
        pdf_cdf = np.cumsum(pdf_counts) / np.sum(pdf_counts)
        ks_statistic = np.max(np.abs(data_cdf - pdf_cdf))
        print(f"ks_statistic {cat_name} {test_range_name}: {ks_statistic}")
        nevents = data_hist_new.Integral()
        
        return_dict[test_range_name] = {
            "ks_statistic": ks_statistic,
            "nevents" : nevents,
                                       }
        
    
        # Draw the cdf histogram
        # Extract bin centers (excluding underflow/overflow)
        bin_centers = []
        hist = data_hist_new
        for i in range(1, hist.GetNbinsX() + 1):  # 1-based bin index
            center = hist.GetBinCenter(i)
            bin_centers.append(center)
        bin_centers = np.array(bin_centers)
        plt.plot(bin_centers, data_cdf, label='Data CDF', linewidth=plot_line_width)
        plt.plot(bin_centers, pdf_cdf, label='PDF CDF', linewidth=plot_line_width)
        plt.xlabel('mass')
        plt.ylabel('')
        # plt.title('Simple NumPy Plot')
        plt.legend()
        plt.grid(True)
        if save_path != "":
            plt.savefig(f"{save_path}/GoF_cdfs_{test_range_name}_{cat_name}.pdf")
        plt.clf()

        # Draw the normalized pdf histogram:
        plt.plot(bin_centers, data_counts/np.sum(data_counts), label='Data PDF', linewidth=plot_line_width)
        plt.plot(bin_centers, pdf_counts/np.sum(pdf_counts), label='PDF PDF', linewidth=plot_line_width)
        plt.xlabel('mass')
        plt.ylabel('')
        # plt.title('Simple NumPy Plot')
        plt.legend()
        plt.grid(True)
        if save_path != "":
            plt.savefig(f"{save_path}/GoF_pdfs_{test_range_name}_{cat_name}.pdf")
        plt.clf()
        
    return return_dict