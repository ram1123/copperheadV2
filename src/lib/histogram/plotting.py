import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
import cmsstyle as CMS
import mplhep as hep
from hist.intervals import poisson_interval

stat_err_opts = {
    "step": "post",
    "label": "Stat. unc.",
    "hatch": "//////",
    "facecolor": "none",
    "edgecolor": (0, 0, 0, 0.5),
    "linewidth": 0,
}
ratio_err_opts = {"step": "post", "facecolor": (0, 0, 0, 0.3), "linewidth": 0}

def getHistAndErrs(
    binning: np.array,
    values: np.array,
    weights: np.array
    ) -> Tuple[np.array, np.array] :
    np_hist, _ = np.histogram(values, bins=binning, weights = weights)
    np_hist_w2, _ = np.histogram(values, bins=binning, weights = weights*weights)
    np_hist_err = np.sqrt(np_hist_w2)
    return np_hist, np_hist_err


def plotDataMC_compare(
    binning: np.array,
    data: Dict[str, np.array],
    bkg_MC_dict: Dict[str, Dict[str, np.array]],
    save_full_path: str,
    sig_MC_dict = {},
    title="default title",
    x_title="Mass (GeV)",
    y_title="Events",
    plot_ratio=True,
    log_scale=True,
    lumi = "",
    status = "Private Work",
    CenterOfMass = 13,
    ):
    """
    Takes in
    Params:
    binning : np array of bin edges compatible to np.histogram
    data: Dictionary with "values" and "weights" as keys and relevant np array for values
    bkg_MC_dict: Ordered dictionary with the bkg_MC sample names as keys and its respective dictionary to histogram as values
        the keys are ordered such that bkg_MC sample with the least yield iterate first
    save_full_path: full path INCLUDING the filename to save the plot at
    sig_MC_dict: dictionary with same structure as bkg_MC_dict. if an empty dictionary, plot only Data and MC
    """
    plt.style.use(hep.style.CMS)

    if plot_ratio:
        fig, (ax_main, ax_ratio) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    else: # skip ratio plot
        fig, ax_main = plt.subplots()
    fig.subplots_adjust(hspace=0.1)

    # -----------------------------------------
    # plot data
    # -----------------------------------------
    values = data["values"]
    weights = data["weights"]
    data_hist, data_hist_err = getHistAndErrs(binning, values, weights)
    hep.histplot(
        data_hist,
        xerr=True,
        yerr=data_hist_err,
        bins=binning,
        stack=False,
        histtype='errorbar',
        color='black',
        label='Data',
        ax=ax_main,
    )


    # -----------------------------------------
    # plot bkg_MC
    # -----------------------------------------
    # collect bkg_MC samples
    bkg_MC_hist_l = []
    bkg_MC_histW2_l = []
    bkg_mc_sample_names = []
    for bkg_mc_sample, bkg_mc_sample_arrs in bkg_MC_dict.items():
        values = bkg_mc_sample_arrs["values"]
        weights = bkg_mc_sample_arrs["weights"]
        np_hist, _ = np.histogram(values, bins=binning, weights = weights)
        np_hist_w2, _ = np.histogram(values, bins=binning, weights = weights*weights)
        bkg_mc_sample_names.append(bkg_mc_sample)
        bkg_MC_hist_l.append(np_hist)
        bkg_MC_histW2_l.append(np_hist_w2)
    # plot bkg_MC in one go
    hep.histplot(
        bkg_MC_hist_l,
        bins=binning,
        stack=True,
        histtype='fill',
        label=bkg_mc_sample_names,
        sort='label_r',
        ax=ax_main,
    )
    ax_main.set_ylabel(y_title)

    if log_scale:
        ax_main.set_yscale('log')
        ax_main.set_ylim(0.01, 1e9)
        # temporary overwrite to match the range of AN plots
        if x_title == "ll_zstar_log":
            ax_main.set_ylim(0.1,  599.48425032)
        elif x_title == "$R_{p_T}$":
            ax_main.set_ylim(0.35938137,  774.26368268)



    # -----------------------------------------
    # plot signal MC
    # -----------------------------------------
    if len(sig_MC_dict.keys()) > 0:
        for sig_mc_sample,  sig_mc_sample_arrs in sig_MC_dict.items():
            values = sig_mc_sample_arrs["values"]
            weights = sig_mc_sample_arrs["weights"]
            sig_MC_hist, _ = getHistAndErrs(binning, values, weights)
            # print(f"{sig_mc_sample} hist: {sig_MC_hist}")
            hep.histplot(
                sig_MC_hist,
                bins=binning,
                histtype='step',
                label=sig_mc_sample,
                # color =  "black",
                ax=ax_main,
            )


    # -----------------------------------------
    # Data/MC ratio
    # -----------------------------------------
    data_hist = ak.to_numpy(data_hist)
    if plot_ratio:
        # compute Data/MC ratio
        # get bkg_MC errors
        bkg_mc_w2_sum = np.sum(np.asarray(bkg_MC_histW2_l), axis=0)
        bkg_mc_err = np.sqrt(bkg_mc_w2_sum)
        # initialize ratio histogram and fill in values
        data_hist = ak.to_numpy(data_hist) # make sure data hist is numpy array, not awkward. it seems like np.zeros_like() propagates ak arrays
        ratio_hist = np.zeros_like(data_hist)
        bkg_mc_sum = np.sum(np.asarray(bkg_MC_hist_l), axis=0)
        bkg_mc_sum = ak.to_numpy(bkg_mc_sum)
        inf_filter = bkg_mc_sum>0
        ratio_hist[inf_filter] = data_hist[inf_filter]/  bkg_mc_sum[inf_filter]
        # add relative uncertainty of data and bkg_mc by adding by quadrature
        # Protect against division by zero by adding a small epsilon to denominators
        epsilon = 1e-10
        bkg_mc_sum_safe = bkg_mc_sum + epsilon
        data_hist_safe = data_hist + epsilon

        # Calculate relative uncertainty ratio
        term1 = (bkg_mc_err / bkg_mc_sum_safe) ** 2
        term2 = (data_hist_err / data_hist_safe) ** 2
        rel_unc_ratio = np.sqrt(term1 + term2)

        # Compute ratio error
        ratio_err = rel_unc_ratio * ratio_hist
        # print(f"plotDataMC_compare ratio_err: {ratio_err}")


        hep.histplot(ratio_hist,
                     bins=binning, histtype='errorbar', yerr=ratio_err,
                     color='black', label='Ratio', ax=ax_ratio)

        # compute MC uncertainty
        # source: https://github.com/kondratyevd/hmumu-coffea/blob/master/python/plotter.py#L228
        # den = bkg_mc_sum[inf_filter]
        den = bkg_mc_sum
        den_sumw2 = bkg_mc_w2_sum
        if sum(den) > 0:
            unity = np.ones_like(den)
            w2 = np.zeros_like(den)
            w2[den > 0] = den_sumw2[den > 0] / den[den > 0] ** 2
            den_unc = poisson_interval(unity, w2)
            ax_ratio.fill_between(
                binning,
                np.r_[den_unc[0], den_unc[0, -1]],
                np.r_[den_unc[1], den_unc[1, -1]],
                label="Stat. unc.",
                **ratio_err_opts,
            )


        ax_ratio.axhline(1, color='gray', linestyle='--')
        ax_ratio.axhline(1.2, color='gray', linestyle='--')
        ax_ratio.axhline(0.8, color='gray', linestyle='--')
        ax_ratio.axhline(1.4, color='gray', linestyle='--')
        ax_ratio.axhline(0.6, color='gray', linestyle='--')
        ax_ratio.set_xlabel(x_title)
        ax_ratio.set_ylabel('Data / MC')
        ax_ratio.set_xlim(binning[0], binning[-1])
        ax_ratio.set_ylim(0.5,1.5)
        ax_ratio.set_yticks([0.6, 0.8, 1.0, 1.2, 1.4]) # explicitly ask for 1.4 and 0.6
    else:
        ax_main.set_xlabel(x_title)




    # -----------------------------------------
    # Legend, title, etc +  save figure
    # -----------------------------------------
    ax_main.legend(loc="upper right")
    if title != "":
        ax_main.set_title(title)
    else: # if title is not specified, add the typicaly CMS label in its place
        hep.cms.label(data=True, loc=0, label=status, com=CenterOfMass, lumi=lumi, ax=ax_main)
     # save figure, we assume that the directory exists
    plt.savefig(save_full_path)
    fig.clf()
    plt.close(fig)
