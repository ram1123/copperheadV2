"""
This script helps in generating non-uniform bin edges for the dnn_vbf_score
such that the distribution is approximately straight.
"""
import dask_awkward as dak
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

from plotter.validation_plotter_unified import applyRegionCatCuts

# ----------------------------
# Utilities
# ----------------------------
def weighted_quantile(x, q, w=None):
    """
    Weighted quantiles of 1D array x at probabilities q (0..1).
    If w is None, reduces to np.quantile.
    """
    x = np.asarray(x, dtype=float)
    q = np.asarray(q, dtype=float)
    if w is None:
        return np.quantile(x, q, method="linear")
    w = np.asarray(w, dtype=float)
    assert x.ndim == 1 and x.size == w.size
    # sort by x
    s = np.argsort(x)
    x, w = x[s], w[s]
    # cumulative normalized weights (CDF at the *right* edge of each sample)
    cum_w = np.cumsum(w)
    cum_w /= cum_w[-1]
    return np.interp(q, cum_w, x)

def make_quantile_binning(signal_scores, signal_weights=None, nbins=13,
                          min_bin_width=None, decimals=3):
    """
    Build bin edges so each bin has ~equal signal yield (by weight).
    - signal_scores: 1D np.array of VBF scores (your DNN output)
    - signal_weights: optional per-event weights
    - nbins: number of bins wanted
    - min_bin_width: enforce a minimum bin width (same units as score). None to disable.
    - decimals: round edges for readability/reproducibility
    Returns: np.array of shape (nbins+1,)
    """
    # raw quantile edges
    qs = np.linspace(0.0, 1.0, nbins + 1)
    edges = weighted_quantile(signal_scores, qs, signal_weights)

    # ensure strictly increasing edges (handles duplicates when the score is discrete/flat)
    # If duplicates appear, we jitter slightly within local neighborhood.
    diffs = np.diff(edges)
    if np.any(diffs <= 0):
        eps = 1e-9 * (np.nanmax(signal_scores) - np.nanmin(signal_scores) + 1.0)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i-1]:
                edges[i] = edges[i-1] + eps

    # enforce a minimum bin width if requested
    if min_bin_width is not None:
        e = [edges[0]]
        for x in edges[1:]:
            if x - e[-1] < min_bin_width:
                x = e[-1] + min_bin_width
            e.append(x)
        # clamp last to max if we overshoot due to min_bin_width
        e[-1] = max(e[-1], edges[-1])
        edges = np.array(e)

    # pretty rounding (without breaking monotonicity)
    edges = np.round(edges, decimals=decimals)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = np.nextafter(edges[i-1], np.inf)

    return edges

def check_flatness(signal_scores, signal_weights, edges):
    """
    Quick flatness diagnostic: returns (bin_yields, yield_std/mean, chi2/ndf for flatness).
    """
    # histogram the signal in the proposed bins
    y, _ = np.histogram(signal_scores, bins=edges, weights=signal_weights)
    mean = y.mean()
    # Poisson-ish uncertainty per bin (using weights): sqrt(sum(w^2))
    if signal_weights is None:
        sig = np.sqrt(y)
    else:
        # compute sum of weights^2 in each bin
        y2, _ = np.histogram(signal_scores, bins=edges, weights=np.asarray(signal_weights)**2)
        sig = np.sqrt(y2)
    # chi2 against a flat expectation = mean in each bin
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.sum(((y - mean) ** 2) / np.where(sig > 0, sig**2, np.inf))
    ndf = (len(y) - 1) if len(y) > 1 else 0
    spread = (np.std(y) / mean) if mean > 0 else np.nan
    chi2_ndf = chi2 / ndf if ndf > 0 else np.nan
    return y, spread, chi2_ndf

# ----------------------------
# Main code
# ----------------------------

if __name__ == "__main__":
    parquet_file_path = "/depot/cms/private/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2018/compacted_11August_FixDimuonMass/vbf_powheg_dipole/0/*.parquet"
    events = dak.from_parquet(parquet_file_path)

    # apply the region and category cuts
    events = applyRegionCatCuts(
        events,
        category="vbf",
        region_name="h-peak",
        njets="inclusive",
        process="vbf_powheg_dipole",
        do_vbf_filter_study=False
    )

    # Compute the DNN VBF score and weights
    dnn_vbf_score = events["dnn_vbf_score_atanh"].compute().to_numpy()
    wgt_nominal = events["wgt_nominal"].compute().to_numpy()

    vbf_score = dnn_vbf_score
    vbf_w = wgt_nominal

    # choose number of bins (same count as your current scheme, or whatever you want)
    nbins = 11

    # build quantile-based edges
    edges = make_quantile_binning(
        signal_scores=vbf_score,
        signal_weights=vbf_w,
        nbins=nbins,
        min_bin_width=None,   # e.g. 0.01 to prevent ultra-narrow bins near the peak
        decimals=3            # tidy printing
    )

    # edges should start from 0.0
    edges[0] = 0.0

    print("Proposed quantile binning:")
    print("binning_vbf = np.array([")
    for e in edges:
        print(f"    {e},")
    print("])")

    # plot the dnn score
    plt.figure(figsize=(10, 6))
    plt.hist((vbf_score), bins=edges, weights=vbf_w, alpha=0.7)
    plt.xlabel("DNN VBF Score")
    plt.ylabel("Weighted Events")
    plt.title("Weighted Distribution of DNN VBF Score")
    plt.grid()
    plt.savefig("dnn_vbf_score_distribution.pdf")

    # sanity check: are VBF yields flat across bins?
    y, spread, chi2_ndf = check_flatness(vbf_score, vbf_w, edges)
    print(f"\nVBF yield per bin: {y.astype(int)}")
    print(f"Relative spread (std/mean): {spread:.3f}")
    print(f"Flatness chi2/ndf: {chi2_ndf:.2f}")
