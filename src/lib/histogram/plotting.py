import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
import cmsstyle as CMS
import mplhep as hep
from hist.intervals import poisson_interval
from matplotlib.colors import ListedColormap
import pandas as pd
from modules.utils import logger

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


def plotDataMC_compare_hda(
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
    raise ValueError

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
    plot_ratio_range="auto", # available options "fixed" or "auto"
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
    petroff10 = ListedColormap(["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"])
    colors = petroff10.colors  # specify colors

    if plot_ratio:
        fig, (ax_main, ax_ratio) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    else: # skip ratio plot
        fig, ax_main = plt.subplots()
    fig.subplots_adjust(hspace=0.1)

    # -----------------------------------------
    # plot data
    # -----------------------------------------
    data_hist = data["hist_arr"]
    data_hist_err = np.sqrt(data["hist_w2_arr"])

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
        hist_arr = bkg_mc_sample_arrs["hist_arr"]
        hist_w2_arr = bkg_mc_sample_arrs["hist_w2_arr"]
        bkg_mc_sample_names.append(bkg_mc_sample)
        bkg_MC_hist_l.append(hist_arr)
        bkg_MC_histW2_l.append(hist_w2_arr)
    # plot bkg_MC in one go
    color_idx = len(bkg_MC_hist_l)
    hep.histplot(
        bkg_MC_hist_l,
        bins=binning,
        stack=True,
        histtype='fill',
        label=bkg_mc_sample_names,
        sort='label_r',
        ax=ax_main,
        color=colors[:color_idx],
    )
    ax_main.set_ylabel(y_title)

    if log_scale:
        ax_main.set_yscale('log')
        ax_main.set_ylim(0.001, 1e9)
        # temporary overwrite to match the range of AN plots
        # if x_title == "ll_zstar_log":
        #     ax_main.set_ylim(0.1,  599.48425032)
        # elif x_title == "$R_{p_T}$":
        #     ax_main.set_ylim(0.35938137,  774.26368268)

    # -----------------------------------------
    # plot signal MC
    # -----------------------------------------
    if len(sig_MC_dict.keys()) > 0:
        for sig_mc_sample,  sig_mc_sample_arrs in sig_MC_dict.items():
            sig_MC_hist = sig_mc_sample_arrs["hist_arr"]
            hep.histplot(
                sig_MC_hist,
                bins=binning,
                histtype='step',
                label=sig_mc_sample,
                # color =  "black",
                ax=ax_main,
                color=colors[color_idx],
            )
            color_idx += 1

    # -----------------------------------------
    # Data/MC ratio
    # -----------------------------------------
    if plot_ratio:
        # compute Data/MC ratio
        # get bkg_MC errors
        bkg_mc_w2_sum = np.sum(np.asarray(bkg_MC_histW2_l), axis=0)
        bkg_mc_err = np.sqrt(bkg_mc_w2_sum)
        # initialize ratio histogram and fill in values
        data_hist = ak.to_numpy(data_hist)
        bkg_mc_sum = np.sum(np.asarray(bkg_MC_hist_l), axis=0)
        # instead of zero like we should fill it with NaNs to avoid misleading points at zero. NaNs will not be plotted
        ratio_hist = np.full_like(data_hist, np.nan, dtype=float)

        mc_pos = bkg_mc_sum > 0
        data_pos = data_hist > 0
        both_pos = mc_pos & data_pos

        ratio_hist[both_pos] = data_hist[both_pos] / bkg_mc_sum[both_pos]

        rel_unc_ratio = np.zeros_like(bkg_mc_sum, dtype=float)
        rel_unc_ratio[both_pos] = np.sqrt(
            (bkg_mc_err[both_pos] / bkg_mc_sum[both_pos]) ** 2
            + (data_hist_err[both_pos] / data_hist[both_pos]) ** 2
        )

        ratio_err = np.zeros_like(rel_unc_ratio)
        ratio_err[both_pos] = rel_unc_ratio[both_pos] * ratio_hist[both_pos]
        # logger.debug(f"plotDataMC compare ratio_err: {ratio_err}")

        hep.histplot(ratio_hist,
                     bins=binning, histtype='errorbar', yerr=ratio_err,
                     color='black', label='Ratio', ax=ax_ratio)

        # compute MC uncertainty
        # source: https://github.com/kondratyevd/hmumu-coffea/blob/master/python/plotter.py#L228
        # den = bkg_mc_sum[inf_filter]
        den = bkg_mc_sum
        den_sumw2 = bkg_mc_w2_sum

        if np.sum(den) > 0:
            unity = np.ones_like(den, dtype=float)
            w2 = np.zeros_like(den, dtype=float)

            den_pos = den > 0
            w2[den_pos] = den_sumw2[den_pos] / (den[den_pos] ** 2)

            # --- avoid poisson_interval with w2==0 (causes divide-by-zero inside hist) ---
            ok = den_pos & (w2 > 0)

            den_unc_full = np.full((2, den.size), np.nan, dtype=float)

            if np.any(ok):
                den_unc_ok = poisson_interval(unity[ok], w2[ok])
                den_unc_full[:, ok] = den_unc_ok

            # if den>0 but w2==0, the uncertainty is exactly 0 -> band is exactly 1
            zero_unc = den_pos & ~(w2 > 0)
            den_unc_full[:, zero_unc] = 1.0

            ax_ratio.fill_between(
                binning,
                np.r_[den_unc_full[0], den_unc_full[0, -1]],
                np.r_[den_unc_full[1], den_unc_full[1, -1]],
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

        finite = np.isfinite(ratio_hist)
        if np.any(finite) and plot_ratio_range == "auto":
            rmin = float(np.nanmin(ratio_hist[finite]))
            rmax = float(np.nanmax(ratio_hist[finite]))

            # ensure unity is visible
            rmin = min(rmin, 1.0)
            rmax = max(rmax, 1.0)

            # small padding (10% of span, or 0.05 minimum)
            span = max(rmax - rmin, 1e-6)
            pad = max(0.10 * span, 0.05)
            ylo = rmin - pad
            yhi = rmax + pad

            # optional safety clamp (avoid insane autoscale)
            ylo = max(ylo, 0.0)
            yhi = min(yhi, 5.0)
            full_range = yhi - ylo

            # Percentile levels requested
            percentiles = np.array([0.10, 0.25, 0.50, 0.75, 0.90])

            # Calculate ticks: ylo + (percentile * range)
            set_yticks = ylo + (percentiles * full_range)

            set_yticks = np.round(set_yticks, 2).tolist()            
        elif len(plot_ratio_range) == 2:
            # fallback if ratio_hist is all-NaN or plot_ratio_range is not "auto"
            ylo, yhi = plot_ratio_range[0], plot_ratio_range[1]
            full_range = yhi - ylo

            # Percentile levels requested
            percentiles = np.array([0.10, 0.25, 0.50, 0.75, 0.90])

            # Calculate ticks: ylo + (percentile * range)
            set_yticks = ylo + (percentiles * full_range)

            set_yticks = np.round(set_yticks, 2).tolist()
        else:
            # fallback if ratio_hist is all-NaN or plot_ratio_range is not "auto"
            ylo, yhi = 0.5, 1.5
            set_yticks = [0.6, 0.8, 1.0, 1.2, 1.4]
        ax_ratio.set_ylim(ylo, yhi)
        ax_ratio.set_yticks(set_yticks)
    else:
        ax_main.set_xlabel(x_title)

    # -----------------------------------------
    # compute and display separation power
    # -----------------------------------------
    # Separation power, d=\frac{1}{2}\int{|s(x)~-~b(x)|}$, where $s(x)$ and $b(x)$ are normalized signal and background distributions.
    logger.debug("Computing separation power for signal samples...")
    if len(sig_MC_dict) > 0 and len(bkg_MC_hist_l) > 0:
        # bin widths for integral
        widths = np.diff(binning)

        # sum background histograms and normalize to density
        bkg_sum = np.sum(np.asarray(bkg_MC_hist_l), axis=0)
        bkg_norm = np.sum(bkg_sum * widths)

        if bkg_norm > 0:
            bkg_density = bkg_sum / bkg_norm
        else:
            bkg_density = None  # cannot compute separation

        # loop over each signal sample and place text in upper-right, offset per sample
        for idx, sig_name in enumerate(sig_MC_dict.keys()):
            sig_arr = sig_MC_dict[sig_name]["hist_arr"]
            sig_norm = np.sum(sig_arr * widths)

            # separation power
            d_val = np.nan

            if bkg_density is not None and sig_norm > 0:
                sig_density = sig_arr / sig_norm
                d_val = 0.5 * np.sum(np.abs(sig_density - bkg_density) * widths)

            # Text placement
            if log_scale:
                y_pos = 0.93 - idx * 0.07
                x_pos = 0.35
            else:
                y_pos = 0.45 - idx * 0.07
                x_pos = 0.95

            text_val = (
                f"{sig_name}: d = {d_val:.2f}"
                if np.isfinite(d_val)
                else f"{sig_name}: d = N/A"
            )

            ax_main.text(
                x_pos,
                y_pos,
                text_val,
                ha="right",
                va="center",
                transform=ax_main.transAxes,
            )
    logger.debug("Finished computing separation power for signal samples.")

    # -----------------------------------------
    # Legend, title, etc +  save figure
    # -----------------------------------------
    ax_main.legend(loc="best", ncol=2)
    if title != "":
        ax_main.set_title(title)
    # save figure, we assume that the directory exists
    hep.cms.label(data=True, loc=0, text=status, com=CenterOfMass, lumi=lumi, ax=ax_main)
    plt.savefig(save_full_path)
    plt.close(fig)

    # Save the raw event number, yield  for each MC and data to the text file along with the data/mc ratio value
    sig_hist_by_name = {
        sig_mc_sample: sig_mc_sample_arrs["hist_arr"]
        for sig_mc_sample, sig_mc_sample_arrs in sig_MC_dict.items()
    }
    bin_edges = np.asarray(binning, dtype=float)
    with open(save_full_path.replace(".pdf", ".txt"), "w") as f:
        # record the exact bin edges used, so the yields below can be traced back
        # to (and the binning re-used from) the histogram that produced them
        f.write(f"Binning ({len(bin_edges) - 1} bins): ")
        f.write("[" + ", ".join(f"{edge:.6g}" for edge in bin_edges) + "]\n")
        f.write(f"Data: {np.sum(data_hist)}\n")
        for bkg_mc_sample, bkg_mc_hist in zip(bkg_mc_sample_names, bkg_MC_hist_l):
            f.write(f"{bkg_mc_sample}: {np.sum(bkg_mc_hist):.2f}\n")
        for sig_mc_sample, sig_hist in sig_hist_by_name.items():
            f.write(f"{sig_mc_sample}: {np.sum(sig_hist):.2f}\n")
        if plot_ratio:
            f.write(
                f"Data/MC ratio (Sum ratio_hist): {np.sum(ratio_hist)}\n"
            )
            f.write(
                f"Data/MC ratio (Sum ratio_hist then divide by number of bins): {np.sum(ratio_hist) / (len(bin_edges) - 1):.2f}\n"
            )

        # -----------------------------------------
        # Bin-by-bin values, in their own section for readability
        # -----------------------------------------
        col_width = 16

        def _fmt_row(cells):
            return "".join(f"{str(c):>{col_width}}" for c in cells)

        f.write("\n" + "=" * 80 + "\n")
        f.write("Bin-by-bin values: Data, MC sample groups, Data/MC ratio\n")
        f.write("=" * 80 + "\n")
        header = ["bin_low", "bin_high", "Data"] + bkg_mc_sample_names + list(sig_hist_by_name.keys())
        if plot_ratio:
            header += ["Data/MC"]
        f.write(_fmt_row(header) + "\n")
        for i in range(len(binning) - 1):
            row = [f"{binning[i]:.4g}", f"{binning[i+1]:.4g}", f"{data_hist[i]:.4g}"]
            row += [f"{bkg_mc_hist[i]:.4g}" for bkg_mc_hist in bkg_MC_hist_l]
            row += [f"{sig_hist[i]:.4g}" for sig_hist in sig_hist_by_name.values()]
            if plot_ratio:
                ratio_val = ratio_hist[i]
                row.append(f"{ratio_val:.4g}" if np.isfinite(ratio_val) else "nan")
            f.write(_fmt_row(row) + "\n")
    # logger.debug(f"Plot saved to {save_full_path} and raw event numbers saved to {save_full_path.replace('.pdf', '.txt')}")


def plotDataMC_compare_normalized(
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
    This function divides into data, bkg and sig samples and normalizes them such that each histogram sums to one
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
    weights = weights/ np.sum(weights) # temp overwrite
    logger.debug(f"data weights sum: {np.sum(weights)}")
    data_hist, data_hist_err = getHistAndErrs(binning, values, weights)
    data_hist = data_hist /np.sum(data_hist)
    hep.histplot(
        data_hist,
        xerr=True,
        yerr=np.zeros_like(data_hist_err),
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
        # weights = weights/np.sum(weights) # temp overwrite
        logger.debug(f"bkg weights sum: {np.sum(weights)}")
        np_hist, _ = np.histogram(values, bins=binning, weights = weights)
        np_hist = np_hist / np.sum(np_hist)
        np_hist_w2, _ = np.histogram(values, bins=binning, weights = weights*weights)
        bkg_mc_sample_names.append(bkg_mc_sample)
        bkg_MC_hist_l.append(np_hist)
        bkg_MC_histW2_l.append(np_hist_w2)
    # plot bkg_MC in one go
    # hep.histplot(
    #     bkg_MC_hist_l,
    #     bins=binning,
    #     stack=True,
    #     histtype='fill',
    #     label=bkg_mc_sample_names,
    #     sort='label_r',
    #     ax=ax_main,
    # )

    # plot bkg_MC in one go # tempoverwrite
    val_l = []
    wgt_l = []
    for bkg_mc_sample, bkg_mc_sample_arrs in bkg_MC_dict.items():
        values = bkg_mc_sample_arrs["values"]
        weights = bkg_mc_sample_arrs["weights"]
        val_l.append(values)
        wgt_l.append(weights)

    values = np.concatenate(val_l, axis=0)
    weights = np.concatenate(wgt_l, axis=0)
    bkg_hist, _ = np.histogram(values, bins=binning, weights = weights)
    bkg_hist = bkg_hist / np.sum(bkg_hist)
    hep.histplot(
        bkg_hist,
        bins=binning,
        stack=True,
        histtype='fill',
        label="bkg",
        sort='label_r',
        ax=ax_main,
    )
    ax_main.set_ylabel(y_title)

    if log_scale:
        ax_main.set_yscale('log')
        ax_main.set_ylim(0.01, 1e9)
        # temporary overwrite to match the range of AN plots
        # if x_title == "ll_zstar_log":
        #     ax_main.set_ylim(0.1,  599.48425032)
        # elif x_title == "$R_{p_T}$":
        #     ax_main.set_ylim(0.35938137,  774.26368268)



    # # -----------------------------------------
    # # plot signal MC
    # # -----------------------------------------
    # if len(sig_MC_dict.keys()) > 0:
    #     for sig_mc_sample,  sig_mc_sample_arrs in sig_MC_dict.items():
    #         values = sig_mc_sample_arrs["values"]
    #         weights = sig_mc_sample_arrs["weights"]
    #         sig_MC_hist, _ = getHistAndErrs(binning, values, weights)
    #         hep.histplot(
    #             sig_MC_hist,
    #             bins=binning,
    #             histtype='step',
    #             label=sig_mc_sample,
    #             # color =  "black",
    #             ax=ax_main,
    #         )

    # temp overwrite
    if len(sig_MC_dict.keys()) > 0:
        val_l = []
        wgt_l = []
        for sig_mc_sample,  sig_mc_sample_arrs in sig_MC_dict.items():
            val_l.append(sig_mc_sample_arrs["values"])
            wgt_l.append(sig_mc_sample_arrs["weights"])
        values = np.concatenate(val_l, axis=0)
        weights = np.concatenate(wgt_l, axis=0)
        # weights = weights/np.sum(weights)
        logger.debug(f"sig values: {values}")
        logger.debug(f"sig weights sum : {np.sum(weights)}")

        sig_MC_hist, _ = getHistAndErrs(binning, values, weights)
        sig_MC_hist = sig_MC_hist / np.sum(sig_MC_hist)
        hep.histplot(
            sig_MC_hist,
            bins=binning,
            histtype='step',
            label="signal",
            # color =  "black",
            ax=ax_main,
        )
    # temp overwrite

    # -----------------------------------------
    # Data/MC ratio
    # -----------------------------------------
    if plot_ratio:
        # compute Data/MC ratio
        # get bkg_MC errors
        bkg_mc_w2_sum = np.sum(np.asarray(bkg_MC_histW2_l), axis=0)
        bkg_mc_err = np.sqrt(bkg_mc_w2_sum)
        # initialize ratio histogram and fill in values
        data_hist = ak.to_numpy(data_hist)
        ratio_hist = np.zeros_like(data_hist)
        bkg_mc_sum = np.sum(np.asarray(bkg_MC_hist_l), axis=0)
        inf_filter = bkg_mc_sum>0
        ratio_hist[inf_filter] = data_hist[inf_filter]/  bkg_mc_sum[inf_filter]
        # add relative uncertainty of data and bkg_mc by adding by quadrature
        rel_unc_ratio = np.sqrt((bkg_mc_err/bkg_mc_sum)**2 + (data_hist_err/data_hist)**2)
        ratio_err = rel_unc_ratio*ratio_hist
        # logger.debug(f"plotDataMC_compare_normalized ratio_err: {ratio_err}")


        hep.histplot(ratio_hist,
                     bins=binning, histtype='errorbar',
                     yerr=np.zeros_like(ratio_err),
                     color='black', label='Ratio', ax=ax_ratio)

        # compute MC uncertainty
        # source: https://github.com/kondratyevd/hmumu-coffea/blob/master/python/plotter.py#L228
        # den = bkg_mc_sum[inf_filter]
        den = bkg_mc_sum
        den_sumw2 = bkg_mc_w2_sum
        # den_sumw2 = bkg_mc_w2_sum[inf_filter]
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
    ax_main.legend(loc="best", ncol=2)

    if title != "":
        ax_main.set_title(title)
    # save figure, we assume that the directory exists
    hep.cms.label(data=True, loc=0, text=status, com=CenterOfMass, lumi=lumi, ax=ax_main)
    plt.savefig(save_full_path)
    plt.close(fig)

    # Save the raw event number, yield  for each MC and data to the text file along with the data/mc ratio value
    with open(save_full_path.replace(".pdf", ".txt"), "w") as f:
        f.write(f"Data: {np.sum(data_hist)}\n")
        for bkg_mc_sample, bkg_mc_hist in zip(bkg_mc_sample_names, bkg_MC_hist_l):
            f.write(f"{bkg_mc_sample}: {np.sum(bkg_mc_hist)}\n")
        if len(sig_MC_dict.keys()) > 0:
            for sig_mc_sample, sig_mc_hist in sig_MC_dict.items():
                f.write(f"{sig_mc_sample}: {np.sum(sig_mc_hist)}\n")
        if plot_ratio:
            f.write(f"Data/MC ratio: {ratio_hist}\n")
    # logger.debug(f"Plot saved to {save_full_path} and raw event numbers saved to {save_full_path.replace('.pdf', '.txt')}")

def plotDataMC_compare_eager(
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
    petroff10 = ListedColormap(["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"])
    colors = petroff10.colors  # specify colors


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
    color_idx = len(bkg_MC_hist_l)
    hep.histplot(
        bkg_MC_hist_l, 
        bins=binning, 
        stack=True, 
        histtype='fill', 
        label=bkg_mc_sample_names, 
        sort='label_r',
        ax=ax_main,
        color=colors[:color_idx],
    )
    ax_main.set_ylabel(y_title)

    if log_scale:
        ax_main.set_yscale('log')
        ax_main.set_ylim(0.01, 1e9)
        # temporary overwrite to match the range of AN plots
        # if x_title == "ll_zstar_log":
        #     ax_main.set_ylim(0.1,  599.48425032)
        # elif x_title == "$R_{p_T}$":
        #     ax_main.set_ylim(0.35938137,  774.26368268)



    # -----------------------------------------
    # plot signal MC
    # -----------------------------------------
    if len(sig_MC_dict.keys()) > 0:
        for sig_mc_sample,  sig_mc_sample_arrs in sig_MC_dict.items():
            values = sig_mc_sample_arrs["values"]
            weights = sig_mc_sample_arrs["weights"]
            sig_MC_hist, _ = getHistAndErrs(binning, values, weights)
            hep.histplot(
                sig_MC_hist, 
                bins=binning, 
                histtype='step', 
                label=sig_mc_sample, 
                # color =  "black",
                ax=ax_main,
                color=colors[color_idx],
            )
            color_idx += 1

    # -----------------------------------------
    # Data/MC ratio
    # -----------------------------------------
    if plot_ratio: 
        # compute Data/MC ratio
        # get bkg_MC errors
        bkg_mc_w2_sum = np.sum(np.asarray(bkg_MC_histW2_l), axis=0)
        bkg_mc_err = np.sqrt(bkg_mc_w2_sum)
        # initialize ratio histogram and fill in values
        data_hist = ak.to_numpy(data_hist)
        ratio_hist = np.zeros_like(data_hist)
        bkg_mc_sum = np.sum(np.asarray(bkg_MC_hist_l), axis=0)
        inf_filter = bkg_mc_sum>0
        ratio_hist[inf_filter] = data_hist[inf_filter]/  bkg_mc_sum[inf_filter]
        # add relative uncertainty of data and bkg_mc by adding by quadrature
        rel_unc_ratio = np.sqrt((bkg_mc_err/bkg_mc_sum)**2 + (data_hist_err/data_hist)**2)
        ratio_err = rel_unc_ratio*ratio_hist
        # print(f"plotDataMC_compare ratio_err: {ratio_err}")


        hep.histplot(ratio_hist, 
                     bins=binning, histtype='errorbar', yerr=ratio_err, 
                     color='black', label='Ratio', ax=ax_ratio)

        # compute MC uncertainty 
        # source: https://github.com/kondratyevd/hmumu-coffea/blob/master/python/plotter.py#L228
        # den = bkg_mc_sum[inf_filter]
        den = bkg_mc_sum
        den_sumw2 = bkg_mc_w2_sum
        # den_sumw2 = bkg_mc_w2_sum[inf_filter]
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
    ax_main.legend(loc="best", ncol=2)
    if title != "":
        ax_main.set_title(title)
    # save figure, we assume that the directory exists
    hep.cms.label(data=True, loc=0, label=status, com=CenterOfMass, lumi=lumi, ax=ax_main)
    plt.savefig(save_full_path)
    print(f"plots saved: {save_full_path}")


def plotFig_6_13(
    binning: np.array, 
    bkg_MC: Dict[str, np.array], 
    sig_MC: Dict[str, np.array], 
    save_full_path: str,
    title="default title", 
    x_title="Mass (GeV)", 
    y_title="A.U.",
    significance_tuple=None,
    log_scale=False,
    lumi = "",
    status = "Private Work",
    CenterOfMass = 13,
    bdtCat_boundaries = [],
    ):
    """
    """
    plt.style.use(hep.style.CMS)


    if significance_tuple is not None:
        fig, (ax_main, ax_ratio) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    else: # skip ratio plot
        fig, ax_main = plt.subplots()
    fig.subplots_adjust(hspace=0.1)

    # -----------------------------------------
    # plot signal and background
    # -----------------------------------------
    values = bkg_MC["BDT_score"]
    weights = bkg_MC["wgt_nominal"]
    # print(f"values: {values}")
    # print(f"weights: {weights}")
    bkg_hist_orig, _ = getHistAndErrs(binning, values, weights)
    bkg_hist = bkg_hist_orig / np.sum(bkg_hist_orig) # normalize
    hep.histplot(
        bkg_hist, 
        xerr=True, 
        bins=binning, 
        stack=False, 
        histtype='step', 
        color='red', 
        label='Background', 
        ax=ax_main,
    )

    values = sig_MC["BDT_score"]
    weights = sig_MC["wgt_nominal"]
    sig_hist_orig, _ = getHistAndErrs(binning, values, weights)
    sig_hist = sig_hist_orig / np.sum(sig_hist_orig) # normalize
    hep.histplot(
        sig_hist, 
        xerr=True, 
        bins=binning, 
        stack=False, 
        histtype='step', 
        color='blue', 
        label='Signal $m_H$=125 GeV', 
        ax=ax_main,
    )


    ax_main.set_ylabel(y_title)

    if log_scale:
        ax_main.set_yscale('log')
        # ax_main.set_ylim(0.01, 1e9)
    else:
        # ax_main.set_ylim(0.00, 0.055)
        ax_main.set_ylim(0.00, 0.07)

    # -----------------------------------------
    # save hist as csv
    # -----------------------------------------
    df = pd.DataFrame({
        "sig_hist": sig_hist,
        "bkg_hist": bkg_hist,
        # "binning":  [binning],
    })
    df.to_csv(save_full_path.replace(".pdf", ".csv"))
    df = pd.DataFrame({
        # "sig_hist": sig_hist,
        # "bkg_hist": bkg_hist,
        "binning":  binning,
    })
    df.to_csv(save_full_path.replace(".pdf", "Binning.csv"))

    # -----------------------------------------
    # add boundaries
    # -----------------------------------------
    for boundary in bdtCat_boundaries:
        ax_main.axvline(
            x=boundary,
            color='grey',
            linestyle=':',
            linewidth=1.5,
            alpha=0.9  
        )
    # axvline(x=0, ymin=0, ymax=1, **kwargs)

    # -----------------------------------------
    # plot significance
    # -----------------------------------------
    if significance_tuple is not None: 
        # compute Data/MC ratio
        subCatSignificance_hist, binning = significance_tuple



        hep.histplot(subCatSignificance_hist, 
                     bins=binning, 
                     histtype='step',
                     color='black', label='Ratio', ax=ax_ratio)



        ax_ratio.axhline(1, color='gray', linestyle='--')
        ax_ratio.axhline(0.8, color='gray', linestyle='--')
        ax_ratio.axhline(0.6, color='gray', linestyle='--')
        ax_ratio.axhline(0.4, color='gray', linestyle='--')
        ax_ratio.axhline(0.2, color='gray', linestyle='--')
        ax_ratio.set_ylim(0.1,1) 
        ax_ratio.set_xlabel(x_title)
        ax_ratio.set_ylabel('Data / MC')
        ax_ratio.set_xlim(binning[0], binning[-1])
        # ax_ratio.set_yticks([0.6, 0.8, 1.0, 1.2, 1.4]) # explicitly ask for 1.4 and 0.6
        ax_ratio.set_yticks([0.2, 0.4, 0.6, 0.8, ]) # explicitly ask for 1.4 and 0.6
    else:
        ax_main.set_xlabel(x_title)
        ax_main.set_xlim(binning[0], binning[-1])

    # -----------------------------------------
    # Legend, title, etc +  save figure
    # -----------------------------------------
    ax_main.legend(loc="best", ncol=1)
    if title != "":
        ax_main.set_title(title)
    # save figure, we assume that the directory exists
    hep.cms.label(data=True, loc=0, label=status, com=CenterOfMass, lumi=lumi, ax=ax_main)
    plt.savefig(save_full_path)


def plotScatter(df, variables, x_var, save_path):
    # Scatter plot
    bdt_edges = np.array([ # 2018 UL subcat edges
        0.8443986773490906,
        1.1
    ])
    bdt_edges = bdt_edges*2 -1
    score_name =  "BDT_score"

    # plt.ylim(0, 10)
    for y_var in variables:
        if y_var == x_var:
            continue
        for (lo, hi) in zip(bdt_edges[:-1], bdt_edges[1:]):
            mask = (df[score_name] > lo) & (df[score_name] <= hi)
            plt.scatter(df.loc[mask, x_var], df.loc[mask, y_var], alpha=0.002, label=f"{lo:.2f} < BDT < {hi:.2f}",)
        plt.legend()
        plt.xlabel(x_var)
        plt.xlim(0,200)
        plt.ylabel(y_var)
        plt.title(f"Scatter plot: {x_var} vs {y_var}")
        plt.grid(True)
        plt.savefig(f"{save_path}/{x_var}_{y_var}.png")
        plt.clf()


def plot2D(df, variables, x_var, plot_settings, save_path, inclusive=False):
    # get binning
    x_bins = np.linspace(*plot_settings[getPlotVar(x_var)]["binning_linspace"])

    # Scatter plot
    if inclusive:
        bdt_edges = np.array([ # 2018 UL subcat edges
            0.0,
            1.1
        ])
    else:
        bdt_edges = np.array([ # 2018 UL subcat edges
            0.8443986773490906,
            1.1
        ])
    bdt_edges = bdt_edges*2 -1
    score_name =  "BDT_score"
    for y_var in variables:
        if y_var == x_var:
            continue
        y_bins = np.linspace(*plot_settings[getPlotVar(y_var)]["binning_linspace"])

        for (lo, hi) in zip(bdt_edges[:-1], bdt_edges[1:]):
            mask = (df[score_name] > lo) & (df[score_name] <= hi)
            plt.hist2d(
                df.loc[mask, x_var], 
                df.loc[mask, y_var],
                bins=[x_bins, y_bins],   # pass lists/arrays of edges
                cmap="viridis",       # colormap
                label=f"{lo:.2f} < BDT < {hi:.2f}"
            )
        # Add color scale (mapping counts → colors)
        cbar = plt.colorbar()
        cbar.set_label("Counts")   # Label for the colorbar

        plt.legend()
        plt.xlabel(x_var)
        if x_var == "dimuon_pt":
            plt.xlim(0,200)
        plt.ylabel(y_var)
        plt.title(f"{x_var} vs {y_var}, {lo:.2f} < BDT < {hi:.2f}")
        plt.grid(True)
        if inclusive:
            plt.savefig(f"{save_path}/hist2D{x_var}_{y_var}Incl.png")
            plt.savefig(f"{save_path}/hist2D{x_var}_{y_var}Incl.pdf")
        else:
            plt.savefig(f"{save_path}/hist2D{x_var}_{y_var}.png")
            plt.savefig(f"{save_path}/hist2D{x_var}_{y_var}.pdf")
        plt.clf()