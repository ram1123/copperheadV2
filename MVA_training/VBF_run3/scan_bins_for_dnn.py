import numpy as np

# from rich import print
# from scipy.special import logit

import logging

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)


# ------------------------------
# Asimov per-bin Z^2 (additive)
# ------------------------------
def z2_asimov(S, B, eps=1e-9):
    """
    Return Z^2 per bin using the Asimov formula (additive across bins).
    Uses a conservative Poisson limit (Z^2 ≈ 2S) when B ~ 0.
    """
    S = np.asarray(S, dtype=float)
    B = np.asarray(B, dtype=float)
    # guard against tiny negative weights from MC/systematics
    S = np.maximum(S, 0.0)
    B = np.maximum(B, 0.0)

    out = np.zeros_like(S, dtype=float)
    mask = B > eps
    # out[mask] = np.sqrt(2.0 *((S[mask] + B[mask]) * np.log1p(S[mask] / B[mask]) - S[mask]))
    out[mask] = 2.0 * ((S[mask] + B[mask]) * np.log1p(S[mask] / B[mask]) - S[mask])
    out[~mask] = 2.0 * S[~mask]
    return out if out.ndim else out.item()


# -----------------------------------------------------
# Significance-optimized using uniform binning
# -----------------------------------------------------
def make_significance_binning_uniform(
    sig_score,
    bkg_score,
    sig_w=None,
    bkg_w=None,
    fine_bins=10,
    score_min=None,
    score_max=None,
    min_total_events_per_bin=0.0,  # stability: (S+B) >= this
    min_signal_per_bin=0.3,  # CMS H→μμ-style guard: S >= this
    clamp_edges=True,  # clamp first/last edges to [0,1]
):
    """
    Create simple uniform binning with fine_bins with the condition min_total_events_per_bin
    and min_signal_per_bin. Then compute the significance for each bin then the combined significance.
    Return the bin edges, per-bin (S, B, Z), and total Z.
    If no valid binning can be found, return (None, None, None, None, None, None).
    Parameters:
    - sig_score, bkg_score: arrays of signal and background scores
    - sig_w, bkg_w: optional arrays of weights (same length as scores)
    - fine_bins: desired number of bins
    - score_min, score_max: optional min/max score values to consider
    - min_total_events_per_bin: minimum (S+B) per bin for stability
    - min_signal_per_bin: minimum S per bin (CMS H→μμ-style guard)
    - clamp_edges: if True, clamp the first/last edges to [0,1
    Returns:
    - edges: array of bin edges (length fine_bins+1)
    - S_bins: array of signal counts per bin (length fine_bins)
    - B_bins: array of background counts per bin (length fine_bins)
    - S_NoWgt_bins: array of unweighted signal counts per bin (length fine_bins)
    - B_NoWgt_bins: array of unweighted background counts per bin (length fine_bins)
    - Z_bins: array of per-bin Asimov Z (length fine_bins)
    - Z_tot: total Asimov Z (scalar)
    """
    sig_score = np.asarray(sig_score, dtype=float)
    bkg_score = np.asarray(bkg_score, dtype=float)
    if sig_w is not None:
        sig_w = np.asarray(sig_w, dtype=float)
        if sig_w.shape != sig_score.shape:
            raise ValueError("sig_w must have the same shape as sig_score")
    else:
        sig_w = np.ones_like(sig_score, dtype=float)
    if bkg_w is not None:
        bkg_w = np.asarray(bkg_w, dtype=float)
        if bkg_w.shape != bkg_score.shape:
            raise ValueError("bkg_w must have the same shape as bkg_score")
    else:
        bkg_w = np.ones_like(bkg_score, dtype=float)

    if score_min is None:
        score_min = float(min(sig_score.min(), bkg_score.min()))
    if score_max is None:
        score_max = float(max(sig_score.max(), bkg_score.max()))
    if score_min >= score_max:
        raise ValueError("score_min must be less than score_max")

    # create fine bins across the requested score range
    fine_edges = np.linspace(score_min, score_max, fine_bins + 1, dtype=float)

    # histogram signal and background in fine bins
    S_NoWgt_hist, _ = np.histogram(sig_score, bins=fine_edges)
    B_NoWgt_hist, _ = np.histogram(bkg_score, bins=fine_edges)
    S_hist, _ = np.histogram(sig_score, bins=fine_edges, weights=sig_w)
    B_hist, _ = np.histogram(bkg_score, bins=fine_edges, weights=bkg_w)

    # print histograms for debugging
    logger.debug(f"S_NoWgt_hist: {S_NoWgt_hist}")
    logger.debug(f"B_NoWgt_hist: {B_NoWgt_hist}")
    # Compute the significance for each fine bin, then obtain the total significance
    total_Z2 = 0.0
    Z_hist = []
    for bin_i in range(fine_bins):
        S_bin = S_hist[bin_i]
        B_bin = B_hist[bin_i]
        S_bin_nw = S_NoWgt_hist[bin_i]
        B_bin_nw = B_NoWgt_hist[bin_i]
        if (S_bin_nw + B_bin_nw) < min_total_events_per_bin:
            logger.debug(
                "Rejecting uniform bin %s: raw events=%s below threshold=%s",
                bin_i,
                S_bin_nw + B_bin_nw,
                min_total_events_per_bin,
            )
            return None, None, None, None, None, None, None
        if S_bin < min_signal_per_bin:
            logger.debug(
                "Rejecting uniform bin %s: weighted signal=%s below threshold=%s",
                bin_i,
                S_bin,
                min_signal_per_bin,
            )
            return None, None, None, None, None, None, None
        significance = z2_asimov(S_bin, B_bin)
        Z_hist.append(np.sqrt(significance) if significance > 0.0 else 0.0)
        total_Z2 += significance
    total_Z = np.sqrt(total_Z2) if total_Z2 > 0.0 else 0.0

    if total_Z <= 0.0:
        logger.debug(
            f"No valid binning found for fine_bins={fine_bins} with the given constraints."
        )
        return None, None, None, None, None, None, None

    return (
        fine_edges,
        S_hist,
        B_hist,
        S_NoWgt_hist,
        B_NoWgt_hist,
        Z_hist,
        total_Z,
    )


# -----------------------------------------------------
# Significance-optimized binning via dynamic programming
# -----------------------------------------------------
def make_significance_binning(
    sig_score,
    bkg_score,
    sig_w=None,
    bkg_w=None,
    fine_bins=500,
    score_min=None,
    score_max=None,
    min_total_events_per_bin=0.0,  # require (S+B) >= this
    min_signal_per_bin=0.3,  # require S >= this
    tol_frac=0.01,  # allow merge if local Z² drop <= 5%
    clamp_edges=True,
):
    """
    Greedy R->L merging with a 5% loss veto:
      - Merge neighboring fine bins from the right as long as merging causes
        at most `tol_frac` fractional loss in local Z².
      - If the accumulator fails S or (S+B) guards, keep merging regardless of loss
        until guards are satisfied.
      - No cap on the final number of bins.
    Returns:
        edges, S_bins, B_bins, S_bins_nw, B_bins_nw, Z_bins, Z_tot
    """
    # -------- clean inputs --------
    sig_score = np.asarray(sig_score, float)
    bkg_score = np.asarray(bkg_score, float)
    sig_w = np.ones_like(sig_score) if sig_w is None else np.asarray(sig_w, float)
    bkg_w = np.ones_like(bkg_score) if bkg_w is None else np.asarray(bkg_w, float)

    def _clean(x, w):
        m = np.isfinite(x) & np.isfinite(w)
        return x[m], w[m]

    sig_score, sig_w = _clean(sig_score, sig_w)
    bkg_score, bkg_w = _clean(bkg_score, bkg_w)
    if sig_score.size == 0 or bkg_score.size == 0:
        raise ValueError("Empty signal or background arrays after cleaning.")

    if score_min is None:
        score_min = min(sig_score.min(), bkg_score.min())
    if score_max is None:
        score_max = max(sig_score.max(), bkg_score.max())

    logger.info(f"score_min = {score_min}, score_max = {score_max}")

    if (
        not np.isfinite(score_min)
        or not np.isfinite(score_max)
        or score_max <= score_min
    ):
        raise ValueError("Invalid score range.")
    sig_score = np.clip(sig_score, score_min, score_max)
    bkg_score = np.clip(bkg_score, score_min, score_max)

    # -------- fine prebinning --------
    fine_edges = np.linspace(score_min, score_max, int(fine_bins) + 1)
    print(f"fine_edges: {fine_edges}")
    
    S_hist, _ = np.histogram(sig_score, bins=fine_edges, weights=sig_w)
    B_hist, _ = np.histogram(bkg_score, bins=fine_edges, weights=bkg_w)
    S_hist_nw, _ = np.histogram(sig_score, bins=fine_edges)
    B_hist_nw, _ = np.histogram(bkg_score, bins=fine_edges)
    nF = len(S_hist)

    def SB(a, b):
        return S_hist[a:b].sum(), B_hist[a:b].sum()

    def SBnw(a, b):
        return S_hist_nw[a:b].sum(), B_hist_nw[a:b].sum()

    # -------- greedy merge from right --------
    bins_idx = []  # list of (i_start, i_end) for final bins
    acc_l = nF - 1
    acc_r = nF  # accumulator is [acc_l, acc_r)
    S_acc, B_acc = SB(acc_l, acc_r)
    S_acc_nw, B_acc_nw = SBnw(acc_l, acc_r)

    i = acc_l - 1
    while True:
        can_merge = i >= 0
        # force-merge if guards not satisfied
        need_guard_merge = (S_acc < min_signal_per_bin) or (
            (S_acc_nw + B_acc_nw) < min_total_events_per_bin
        )

        if can_merge:
            # local separate vs merged Z²
            S_L, B_L = SB(i, i + 1)
            z2_sep = z2_asimov(S_L, B_L) + z2_asimov(S_acc, B_acc)
            S_L_nw, B_L_nw = SBnw(i, i + 1)
            S_m, B_m = (S_L + S_acc, B_L + B_acc)
            S_m_nw, B_m_nw = (S_L_nw + S_acc_nw, B_L_nw + B_acc_nw)
            z2_mrg = z2_asimov(S_m, B_m)

            # fractional drop if we merge (robust to z2_sep ~ 0)
            frac_drop = 0.0 if z2_sep <= 0.0 else max(0.0, (z2_sep - z2_mrg) / z2_sep)

            if need_guard_merge or (frac_drop <= tol_frac):
                # accept merge and continue extending left
                logger.debug(
                    f"Merging bins [{i}, {i+1}) and [{acc_l}, {acc_r}): frac_drop = {frac_drop:.4f}"
                )
                acc_l = i
                S_acc, B_acc = S_m, B_m
                S_acc_nw, B_acc_nw = S_m_nw, B_m_nw
                i -= 1
            else:
                # veto merge -> freeze current accumulator as a bin; start new acc at left bin
                bins_idx.append((acc_l, acc_r))
                acc_r = acc_l
                acc_l = i
                S_acc, B_acc = SB(acc_l, acc_r)
                S_acc_nw, B_acc_nw = SBnw(acc_l, acc_r)
                i -= 1
        else:
            # nothing left to merge: finalize accumulator
            bins_idx.append((acc_l, acc_r))
            break

    # left->right order
    bins_idx = bins_idx[::-1]

    # -------- edges & summaries --------
    edges_idx = [bins_idx[0][0]] + [b for (_, b) in bins_idx]
    edges = fine_edges[edges_idx]
    print(f"edges: {edges}")
    print(f"edges_idx: {edges_idx}")
    print(f"fine_edges: {fine_edges}")
    print(f"fine_bins: {fine_bins}")
    # raise ValueError

    if clamp_edges:
        edges[0] = max(edges[0], score_min)
        edges[-1] = min(edges[-1], score_max)
        for t in range(1, len(edges)):
            if edges[t] <= edges[t - 1]:
                edges[t] = np.nextafter(edges[t - 1], np.inf)

    S_bins, B_bins, S_bins_nw, B_bins_nw, Z_bins = [], [], [], [], []
    for a, b in bins_idx:
        S, B = SB(a, b)
        Sn, Bn = SBnw(a, b)
        S_bins.append(S)
        B_bins.append(B)
        S_bins_nw.append(Sn)
        B_bins_nw.append(Bn)
        Z_bins.append(np.sqrt(z2_asimov(S, B)))
    S_bins = np.array(S_bins)
    B_bins = np.array(B_bins)
    S_bins_nw = np.array(S_bins_nw)
    B_bins_nw = np.array(B_bins_nw)
    Z_bins = np.array(Z_bins)
    Z_tot = np.sqrt(np.sum(z2_asimov(S_bins, B_bins)))

    return edges, S_bins, B_bins, S_bins_nw, B_bins_nw, Z_bins, Z_tot


# ---------------------------------------------------
# Scan nbins and pick the highest total Asimov Z setup
# ---------------------------------------------------
def scan_nbins_for_best_edges(
    sig_score,
    bkg_score,
    sig_w=None,
    bkg_w=None,
    nbins_list=range(2, 14),
    score_min=None,
    score_max=None,
    min_total_events_per_bin=0.0,  # stability: (S+B) >= this
    min_signal_per_bin=0.3,  # CMS H→μμ-style guard: S >= this
    clamp_edges=True,  # clamp first/last edges to [0,1]
    frac_tol=0.01,  # allow merge if local Z² drop <= 1%
):
    """
    Scan multiple nbins and return the one with the highest total Asimov Z.
    See make_significance_binning() for parameter details.
    """
    best_Z = -np.inf
    best_result = (None, None, None, None, None, None, None)
    nb_list = []
    Z_tot_list = []
    S_NoWgt_bins_list = []
    B_NoWgt_bins_list = []
    # nbins_list = reversed(nbins_list)
    logger.info(f"nbins_list: {nbins_list}")
    print(f"nbins_list: {nbins_list}")
    # raise ValueError
    for nb in nbins_list:
        outfile = f"binning_scan_{nb}_vs_Ztot_{str(frac_tol).replace('.', 'p')}_log.pdf"
        edges, S_bins, B_bins, S_NoWgt_bins, B_NoWgt_bins, Z_bins, Z_tot = (
            # make_significance_binning_uniform(
            make_significance_binning(
                sig_score,
                bkg_score,
                sig_w=sig_w,
                bkg_w=bkg_w,
                fine_bins=nb,
                score_min=score_min,
                score_max=score_max,
                min_total_events_per_bin=min_total_events_per_bin,
                min_signal_per_bin=min_signal_per_bin,
                clamp_edges=clamp_edges,
                tol_frac=frac_tol,  # allow merge if local Z² drop <= 1% (only for non-uniform binning)
            )
        )
        produced_bins = len(edges) - 1
        nb_list.append(nb)
        Z_tot_list.append(Z_tot if Z_tot is not None else -np.inf)
        S_NoWgt_bins_list.append(S_NoWgt_bins[-1])
        B_NoWgt_bins_list.append(B_NoWgt_bins[-1])
        print(
            f"nbins={nb:>3}: Z_tot = {Z_tot:<2.2f}, produced bins = {produced_bins}"
        )
        print(f"         S_bins (NoWgt) = {S_NoWgt_bins}")
        print(f"         B_bins (NoWgt) = {B_NoWgt_bins}")
        print(f"         S_bins               = {S_bins}")
        print(f"         B_bins               = {B_bins}")
        print(f"         Edges = {edges}")
        if Z_tot is not None and Z_tot > best_Z:
            best_Z = Z_tot
            best_result = (
                nb,
                edges,
                S_NoWgt_bins,
                S_bins,
                B_NoWgt_bins,
                B_bins,
                Z_bins,
                Z_tot,
            )

    # plot for the nb vs Z_tot scan
    plot_binning_scan(
        nb_list, Z_tot_list, S_NoWgt_bins_list, B_NoWgt_bins_list, outfile
    )
    plot_binning_scan_root(
        nb_list,
        Z_tot_list,
        S_NoWgt_bins_list,
        B_NoWgt_bins_list,
        outfile.replace(".pdf", "_root.pdf"),
    )

    if best_Z < 0.0:
        raise RuntimeError("Failed to find a valid binning configuration.")
    return best_result


# ------------------------------------
# Plotting
# ------------------------------------
def plot_binning_scan(
    nb_list, Z_tot_list, S_NoWgt_bins_list, B_NoWgt_bins_list, outfile
):
    try:
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(8, 5))
        color1 = "tab:blue"
        color2 = "tab:green"
        color3 = "tab:red"
        ax1.set_xlabel("Number of bins")
        ax1.set_ylabel("Total Asimov Z", color=color1)
        (l1,) = ax1.plot(
            nb_list, Z_tot_list, marker="o", color=color1, label="Total Asimov Z"
        )
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_xticks(nb_list)
        ax1.grid(True)

        # Second y-axis for S_NoWgt_bins_list (log scale)
        # ax2 = ax1.twinx()
        # ax2.set_ylabel("S_NoWgt (last bin)", color=color2)
        # (l2,) = ax2.plot(
        #     nb_list,
        #     S_NoWgt_bins_list,
        #     marker="s",
        #     color=color2,
        #     label="S_NoWgt (last bin)",
        # )
        # ax2.tick_params(axis="y", labelcolor=color2)
        # ax2.set_yscale('log')

        # Third y-axis for B_NoWgt_bins_list (log scale)
        # ax3 = ax1.twinx()
        # # Offset the third axis to the right
        # ax3.spines["right"].set_position(("axes", 1.2))
        # ax3.set_frame_on(True)
        # ax3.patch.set_visible(False)
        # for sp in ax3.spines.values():
        #     sp.set_visible(True)
        # ax3.set_ylabel("B_NoWgt (last bin)", color=color3)
        # (l3,) = ax3.plot(
        #     nb_list,
        #     B_NoWgt_bins_list,
        #     marker="^",
        #     color=color3,
        #     label="B_NoWgt (last bin)",
        # )
        # ax3.tick_params(axis="y", labelcolor=color3)
        # ax3.set_yscale('log')

        # Set different y-limits for S and B if needed
        # if len(S_NoWgt_bins_list) > 0:
        #     smin, smax = min(S_NoWgt_bins_list), max(S_NoWgt_bins_list)
        #     ax2.set_ylim(smin * 0.9, smax * 1.1)
        # if len(B_NoWgt_bins_list) > 0:
        #     bmin, bmax = min(B_NoWgt_bins_list), max(B_NoWgt_bins_list)
        #     ax3.set_ylim(bmin * 0.9, bmax * 1.1)

        # Add legends
        # lines = [l1, l2, l3]
        lines = [l1]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left")

        plt.title("Scan of binning configurations")
        fig.tight_layout()
        plt.savefig(f"{outfile}")
        plt.close(fig)
        logger.info(f"Saved binning scan plot to {outfile}")
    except ImportError:
        logger.info("matplotlib not available")
        pass  # matplotlib not available


# plotting with ROOT
def plot_binning_scan_root(
    nb_list, Z_tot_list, S_NoWgt_bins_list, B_NoWgt_bins_list, outfile
):
    import ROOT
    from ROOT import gStyle

    gStyle.SetOptStat(0)
    c = ROOT.TCanvas("c", "Binning Scan", 800, 600)
    c.SetGrid()

    n = len(nb_list)
    if n == 0:
        print("No data to plot.")
        return
    x = np.array(nb_list, dtype=float)
    y1 = np.array(Z_tot_list, dtype=float)
    y2 = np.array(S_NoWgt_bins_list, dtype=float)
    y3 = np.array(B_NoWgt_bins_list, dtype=float)

    # Make separate plots for each y-axis
    mg = ROOT.TMultiGraph()
    mg.Add(ROOT.TGraph(n, x, y1), "AP")
    # mg.Add(ROOT.TGraph(n, x, y2), "AP")
    # mg.Add(ROOT.TGraph(n, x, y3), "AP")
    # mg.Draw("A")
    mg.SetTitle("Number of bins")
    mg.GetYaxis().SetTitle("Total Asimov Z")
    # mg.GetYaxis().SetTitle("Total Asimov Z / S_NoWgt / B_NoWgt")
    # mg.GetXaxis().SetLimits(min(x) - 1, max(x) + 1)
    # mg.GetYaxis().SetRangeUser(0, max(max(y1), max(y2), max(y3)) * 1.2)
    # mg.GetXaxis().SetNdivisions(n, False)
    # mg.GetYaxis().SetNdivisions(510, False)
    # mg.GetYaxis().SetTitleOffset(1.2)
    # mg.GetYaxis().SetTitleSize(0.05)
    # mg.GetXaxis().SetTitleSize(0.05)
    # mg.GetXaxis().SetLabelSize(0.04)

    # increase the size of the markers
    for i in range(mg.GetListOfGraphs().GetEntries()):
        gr = mg.GetListOfGraphs().At(i)
        gr.SetMarkerStyle(20 + i)  # different marker style for each graph
        gr.SetMarkerSize(1.5)  # increase marker size
        gr.SetLineWidth(2)  # increase line width

    # draw mg with color options
    mg.Draw("AP")

    # Add legend
    legend = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)
    legend.AddEntry(mg.GetListOfGraphs().At(0), "Total Asimov Z", "lp")
    # legend.AddEntry(mg.GetListOfGraphs().At(1), "S_NoWgt (last bin)", "lp")
    # legend.AddEntry(mg.GetListOfGraphs().At(2), "B_NoWgt (last bin)", "lp")
    # legend.Draw()

    c.SaveAs(outfile)
    c.SaveAs(outfile.replace(".pdf", ".root"))
    print(f"Saved binning scan plot to {outfile}")


# ------------------------------------
# Background collection (example usage)
# ------------------------------------
def collect_scores(process_globs, selection, category="vbf", region_name="h-peak"):
    """
    Build score and weight arrays by concatenating any set of processes.
    - process_globs: mapping or iterable of (process_name, parquet_glob)
    - selection: your modules.selection (needs applyRegionCatCuts)
    """
    import dask_awkward as dak

    if hasattr(process_globs, "items"):
        items = process_globs.items()
    else:
        try:
            items = dict(process_globs).items()
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "process_globs must be a mapping or iterable of (name, glob) pairs."
            ) from exc

    scores, weights = [], []
    # do_vbf_filter_study = False
    do_vbf_filter_study = True
    for name, globpath in items:
        # if "dy_" in name:
        #     do_vbf_filter_study = True
        # else:
        #     do_vbf_filter_study = False
        print(
            f"Processing {name} from {globpath} (do_vbf_filter_study={do_vbf_filter_study})"
        )
        ev = dak.from_parquet(globpath)
        ev = selection.applyRegionCatCuts(
            ev,
            category=category,
            region_name=region_name,
            process=name,
            variation="nominal",
            do_vbf_filter_study=do_vbf_filter_study,
        )
        # print(f"ev.fields: {ev.fields}")
        # print(f"ev.fields: {dak.max(ev.dnn_vbf_score).compute()}")
        # print(f"ev.fields: {dak.min(ev.dnn_vbf_score).compute()}")
        # raise ValueError 
        # dnn_score = ev["dnn_vbf_score"].compute().to_numpy()
        # dnn_score = np.atanh(logit(dnn_score))
        # print(dnn_score.max())
        # print(dnn_score.min())
        # raise ValueError 
        # scores.append(dnn_score)
        scores.append(ev["dnn_vbf_score_atanh"].compute().to_numpy())
        weights.append(ev["wgt_nominal"].compute().to_numpy())
    return np.concatenate(scores), np.concatenate(weights)


def collect_bkg(process_globs, selection, category="vbf", region_name="h-peak"):
    """Backward-compatible alias for background collection."""
    return collect_scores(
        process_globs, selection, category=category, region_name=region_name
    )


# -----------------------
# Example driver snippet
# -----------------------
if __name__ == "__main__":
    from modules import selection

    from distributed import Client

    client = Client(
        n_workers=64, threads_per_worker=1, processes=True, memory_limit="2 GiB"
    )
    print("Local scale Client created")
    # stage1_path="/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc"
    # compacted_tag="May08_2026_FixDimuonMass"
    # stage1_path="/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc"
    # stage1_path="/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc"
    # compacted_tag="Jun01_2026_FixDimuonMass"
    stage1_path="/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc"
    # compacted_tag="May08_2026_FixDimuonMass"
    # compacted_tag="Jun05_2026_RamMay2025Binning_FixDimuonMass"
    # compacted_tag="Jun07_2026_50nTrialsFoldsAll_FixDimuonMass"
    compacted_tag="Jun08_2026_20nTrialsFoldsAll_FixDimuonMass"
    
    year="*"
    sig_globs = {
        "vbf_powheg_dipole": f"{stage1_path}/stage1_output/{year}/compacted_{compacted_tag}/vbf_powheg_dipole/**/*.parquet",
        # "ggh_powhegPS": f"{stage1_path}/stage1_output/{year}/compacted_{compacted_tag}/ggh_powhegPS/**/*.parquet",
    }
    # sig_globs = {
    #     "vbf_powheg_dipole": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/*/compacted_19September_FixDimuonMass/vbf_powheg_dipole/**/*.parquet",
    #     # "ggh_powhegPS": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/*/compacted_19September_FixDimuonMass/ggh_powhegPS/**/*.parquet",
    # }
    sig_score, sig_w = collect_scores(sig_globs, selection)

    bkg_globs = {
        "dy_VBF_filter": f"{stage1_path}/stage1_output/{year}/compacted_{compacted_tag}/dy_VBF_filter/**/*.parquet",
        # "dy_M-100To200_MiNNLO": f"{stage1_path}/stage1_output/{year}/compacted_{compacted_tag}/dy_M-100To200_MiNNLO/**/*.parquet",
        "dy_M-Incl_MiNNLO": f"{stage1_path}/stage1_output/{year}/compacted_{compacted_tag}/dy*MiNNLO/**/*.parquet",
        # "dyInclM-50_aMCatNLO": f"{stage1_path}/stage1_output/{year}/compacted_{compacted_tag}/dy*M-50_aMCatNLO/**/*.parquet",
        # "dy_M-50_MiNNLO": f"{stage1_path}/stage1_output/{year}/compacted_{compacted_tag}/dy_M-50_MiNNLO/**/*.parquet",
        "ewk_incl": f"{stage1_path}/stage1_output/{year}/compacted_{compacted_tag}/ewk*/**/*.parquet",
        # "ewk_lljj_mll50_mjj120": f"{stage1_path}/stage1_output/{year}/compacted_{compacted_tag}/ewk_lljj_mll50_mjj120/**/*.parquet",
        # "ewk_zlljj": f"{stage1_path}/stage1_output/{year}/compacted_{compacted_tag}/ewk_zlljj/**/*.parquet",
        "ttjets_dl": f"{stage1_path}/stage1_output/{year}/compacted_{compacted_tag}/ttjets_dl/**/*.parquet",
        "ttjets_sl": f"{stage1_path}/stage1_output/{year}/compacted_{compacted_tag}/ttjets_sl/**/*.parquet",
        "zz": f"{stage1_path}/stage1_output/{year}/compacted_{compacted_tag}/zz*/**/*.parquet",
        # "ww": f"{stage1_path}/stage1_output/{year}/compacted_{compacted_tag}/ww_*/**/*.parquet",
        # "wz": f"{stage1_path}/stage1_output/{year}/compacted_{compacted_tag}/wz_*/**/*.parquet",
    }

    # bkg_globs = {
    #     "dy_VBF_filter": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/*/compacted_19September_FixDimuonMass/dy_VBF_filter/**/*.parquet",
    #     "dy_M-50_aMCatNLO": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/*/compacted_19September_FixDimuonMass/dy_M-50_aMCatNLO/**/*.parquet",
    #     "dy_M-100To200_aMCatNLO": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/*/compacted_19September_FixDimuonMass/dy_M-100To200_aMCatNLO/**/*.parquet",
    #     "ewk_lljj_mll50_mjj120": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/*/compacted_19September_FixDimuonMass/ewk_lljj_mll50_mjj120/**/*.parquet",
    #     "ttjets_dl": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/*/compacted_19September_FixDimuonMass/ttjets_dl/**/*.parquet",
    #     "ttjets_sl": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/*/compacted_19September_FixDimuonMass/ttjets_sl/**/*.parquet",
    #     "zz": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/*/compacted_19September_FixDimuonMass/zz/**/*.parquet",
    # }
    
    bkg_score, bkg_w = collect_scores(bkg_globs, selection)

    score_min = float(min(sig_score.min(), bkg_score.min()))
    score_upper = float(max(sig_score.max(), bkg_score.max()))
    print(f"compacted path: {stage1_path}/stage1_output/{year}/compacted_{compacted_tag}")
    print(f"score_min: {score_min}")
    print(f"score_upper: {score_upper}")
    print(
        f"Derived dnn_vbf_score_atanh range: [{score_min:.6f}, {score_upper:.6f}]"
    )

    # frac_tol = 0.005  # allow merge if local Z² drop <= 1%
    frac_tol = 0.01  # recommended
    # max_nbins = 15  # max number of bins to try
    # max_nbins = 50  # max number of bins to try
    # max_nbins = 25  # max number of bins to try
    # max_nbins = 30  # max number of bins to try
    # max_nbins = 65  # max number of bins to try
    # max_nbins = 70  # max number of bins to try
    # max_nbins = 57  # max number of bins to try
    # max_nbins = 25  # max number of bins to try
    # max_nbins = 25  # max number of bins to try
    max_nbins = 40  # max number of bins to try
    # max_nbins = 24  # max number of bins to try
    # max_nbins = 23  # max number of bins to try
    #------------------------------
    # min_total_events_per_bin=5.0
    min_total_events_per_bin= 15
    nb, edges, S_NoWgt_bins, S_bins, B_NoWgt_bins, B_bins, Z_bins, Ztot = (
        scan_nbins_for_best_edges(
            sig_score,
            bkg_score,
            sig_w,
            bkg_w,
            nbins_list=range(3, max_nbins + 1),
            score_min=score_min,
            score_max=score_upper,
            min_total_events_per_bin=min_total_events_per_bin,
            min_signal_per_bin=0.05,
            clamp_edges=True,
            frac_tol=frac_tol,  # allow merge if local Z² drop <= 1%
        )
    )

    print(f"max xbins: {max_nbins}")
    print(f"Best nbins = {nb}, total Asimov Z = {Ztot:.3f}")
    print("edges = np.array([")
    for e in edges:
        print(f"  {e:.6f},")
    print("])")
    print("Per-bin (S, B, Z):")
    for i, (S_norm, S, B_norm, B, Z) in enumerate(
        zip(S_bins, S_NoWgt_bins, B_bins, B_NoWgt_bins, Z_bins)
    ):
        print(
            f"#   bin {i:2}: ({edges[i]:.3f},{edges[i+1]:.3f}): "
            f"S={S_norm:>4f}({S:>7})  "
            f"B={B_norm:>4f}({B:>7})  "
            f"Z={Z:.2f}\n"
        )
    # ------------------------------------
    # Save best binning to a text file
    # ------------------------------------
    with open(
        f"best_binning_{max_nbins}bins_{str(frac_tol).replace('.', 'p')}.txt", "w"
    ) as f:
        f.write(f"# Best binning with {nb} bins, total Asimov Z = {Ztot:.3f}\n")
        f.write("edges = np.array([\n")
        for e in edges:
            f.write(f"  {e:.6f},\n")
        f.write("])\n")
        f.write("# Per-bin (S, B, Z):\n")
        for i, (S_norm, S, B_norm, B, Z) in enumerate(
            zip(S_bins, S_NoWgt_bins, B_bins, B_NoWgt_bins, Z_bins)
        ):
            f.write(
                f"#   bin {i:2}: "
                f"({edges[i]:6.3f},{edges[i+1]:6.3f})  "
                f"S={S_norm:8.3f} ({S:9.0f})  "
                f"B={B_norm:10.3f} ({B:9.0f})  "
                f"Z={Z:5.3f}\n"
            )
    logger.info("Saved best binning to file.")
