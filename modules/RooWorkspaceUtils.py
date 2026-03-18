import ROOT
import numpy as np

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

def rebinRooDataHist(x, rooDataHist, rebin_factor):
    """
    convert to TH1, rebin, then convert back to rooDataHist
    """
    # Step 3: Convert original RooDataHist into a TH1
    h_original = rooDataHist.createHistogram("h_original", x)

    # Step 4: Rebin the TH1
    new_name = f"{rooDataHist.GetName()}_rebinned"
    h_rebinned = h_original.Rebin(rebin_factor, f"{rooDataHist.GetName()}_rebinned")

    # Step 5: Build a new RooDataHist from the rebinned TH1
    rebinned_dh = ROOT.RooDataHist(new_name, new_name, ROOT.RooArgList(x), h_rebinned)
    return rebinned_dh

def hist_stddev_with_unc(counts: np.ndarray, edges: np.ndarray):
    """
    Compute std deviation and its uncertainty using ROOT's TH1D.
    Parameters
    ----------
    counts : np.ndarray
        Histogram bin contents (like from np.histogram).
    edges : np.ndarray
        Histogram bin edges (len = len(counts)+1).
    Returns
    -------
    std : float
        Standard deviation of the histogram (RMS).
    std_err : float
        Uncertainty on the standard deviation.
    """
    # Ensure histograms store sum of squares of weights for error calculation
    ROOT.TH1.SetDefaultSumw2(True)
    nbins = len(counts)
    h = ROOT.TH1D("h_tmp", "temporary hist", nbins, edges)

    for i, c in enumerate(counts, start=1):  # ROOT bins start at 1
        h.SetBinContent(i, float(c))

    std     = h.GetStdDev()
    std_err = h.GetStdDevError()
    return std, std_err