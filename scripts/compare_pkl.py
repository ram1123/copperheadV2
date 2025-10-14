import pickle
import numpy as np


def _reduce_to_1d(arr: np.ndarray) -> np.ndarray:
    """Sum over all axes except the last (bin axis), returning 1D array."""
    if arr.ndim == 1:
        return arr
    flat = arr.reshape((-1, arr.shape[-1]))
    return flat.sum(axis=0)


def _get_last_axis_edges(hobj) -> np.ndarray:
    """Return bin edges of the last (score) axis from a Hist view/object."""
    ax = hobj.axes[-1]
    # hist.axis.Variable exposes .edges
    try:
        edges = np.asarray(ax.edges)
    except AttributeError:
        # Fallback for boost-histogram style
        edges = np.asarray(ax.edges)
    return edges


def load_nominal_from_pkl(pkl_path, region=None, channel=None):
    """
    Load histogram from .pkl and return (values_1d, edges) for the nominal, value-only slice.

    If region/channel are provided, take that slice; otherwise sum over them.
    """
    with open(pkl_path, "rb") as f:
        hist = pickle.load(f)

    try:
        h = hist[{"variation": "nominal", "val_sumw2": "value"}]
    except Exception as e:
        raise RuntimeError(f"Could not extract nominal values from {pkl_path}: {e}")

    sel = {}
    if region is not None:
        sel["region"] = region
    if channel is not None:
        sel["channel"] = channel
    if sel:
        try:
            h = h[sel]
        except Exception as e:
            raise RuntimeError(f"Failed to select {sel} in {pkl_path}: {e}")

    edges = _get_last_axis_edges(h)
    arr = np.asarray(h.view(flow=False))
    arr = _reduce_to_1d(arr)
    return arr, edges


def _edges_aligned(fine: np.ndarray, coarse: np.ndarray, tol: float = 1e-12) -> bool:
    """Return True if all coarse edges are present in fine (within tol)."""
    if coarse[0] < fine[0] - tol or coarse[-1] > fine[-1] + tol:
        return False
    # Build fast lookup using rounding within tolerance
    # Use a set of rounded strings to avoid float fuzz
    def key(x):
        return round(float(x) / max(tol, 1e-15))
    fine_keys = {key(x) for x in fine}
    return all(key(x) in fine_keys for x in coarse)


def _rebin_to_edges(values: np.ndarray, from_edges: np.ndarray, to_edges: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Rebin histogram counts from `from_edges` (fine) to `to_edges` (coarse) by exact edge aggregation.
    Requires that every `to_edges` value is present in `from_edges` within `tol`.
    """
    if not _edges_aligned(from_edges, to_edges, tol=tol):
        raise ValueError("Edges are not aligned (cannot rebin exactly)")

    # Map edge -> index in fine
    def key(x):
        return round(float(x) / max(tol, 1e-15))
    idx = {key(e): i for i, e in enumerate(from_edges)}

    out = np.zeros(len(to_edges) - 1, dtype=float)
    for j in range(len(to_edges) - 1):
        i0 = idx[key(to_edges[j])]
        i1 = idx[key(to_edges[j + 1])]
        # sum bins covering [to_edges[j], to_edges[j+1])
        out[j] = float(np.sum(values[i0:i1]))
    return out


def compare_nominals(pkl1, pkl2, region=None, channel=None, tol_edges=1e-12):
    """
    Compare nominal hist values from two .pkl files, optionally by region/channel.
    If bin edges differ but are *aligned* (one is a refinement of the other),
    rebin the finer histogram onto the coarser edges before comparing.
    """
    nom1, edges1 = load_nominal_from_pkl(pkl1, region=region, channel=channel)
    nom2, edges2 = load_nominal_from_pkl(pkl2, region=region, channel=channel)

    # Handle possible edge differences
    same_edges = np.array_equal(edges1, edges2)
    if not same_edges:
        # Try rebinning fine -> coarse if aligned
        len1, len2 = len(edges1), len(edges2)
        try:
            if _edges_aligned(edges1, edges2, tol=tol_edges):
                nom1 = _rebin_to_edges(nom1, edges1, edges2, tol=tol_edges)
                edges1 = edges2
                same_edges = True
                print("Note: Re-binned file1 to match file2 edges.")
            elif _edges_aligned(edges2, edges1, tol=tol_edges):
                nom2 = _rebin_to_edges(nom2, edges2, edges1, tol=tol_edges)
                edges2 = edges1
                same_edges = True
                print("Note: Re-binned file2 to match file1 edges.")
        except Exception as e:
            print(f"Warning: Could not rebin due to edge mismatch ({e}). Will compare totals only.")

    if not same_edges:
        # Edges incompatible: compare only totals, then exit.
        tot1 = float(np.sum(nom1))
        tot2 = float(np.sum(nom2))
        print("Comparing nominal hist *totals* (bin edges incompatible):")
        print(f"  {pkl1}\n  {pkl2}")
        if region is not None:
            print(f"  region={region}")
        else:
            print("  region=ALL (summed)")
        if channel is not None:
            print(f"  channel={channel}")
        else:
            print("  channel=ALL (summed)")
        diff_tot = tot1 - tot2
        rel_tot = diff_tot / tot1 if tot1 != 0 else 0.0
        print(f"Totals -> NEW: {tot1:.8f}  OLD: {tot2:.8f}  AbsDiff: {diff_tot:.8e}  RelDiff(%): {rel_tot*100:.6f}")
        return

    # Now edges match: safe bin-by-bin comparison
    if nom1.shape != nom2.shape:
        raise ValueError(f"Shape mismatch: {nom1.shape} vs {nom2.shape}")

    diff = nom1 - nom2
    rel_diff = np.divide(diff, nom1, out=np.zeros_like(diff), where=nom1 != 0)

    print("Comparing nominal hist values between:")
    print(f"  {pkl1}")
    print(f"  {pkl2}")
    if region is not None:
        print(f"  region={region}")
    else:
        print("  region=ALL (summed)")
    if channel is not None:
        print(f"  channel={channel}")
    else:
        print("  channel=ALL (summed)")

    header = f"{'Bin':<5} {'EdgeLow':>12} {'EdgeHigh':>12} {'NEW':>15} {'OLD':>15} {'AbsDiff':>15} {'RelDiff(%)':>15}"
    print(header)
    for i, (el, eh, v1, v2, d, rd) in enumerate(zip(edges1[:-1], edges1[1:], nom1, nom2, diff, rel_diff)):
        v1f = float(v1)
        v2f = float(v2)
        df = float(d)
        rdf = float(rd) * 100.0
        print(f"{i:<5} {el:12.6g} {eh:12.6g} {v1f:15.8f} {v2f:15.8f} {df:15.8e} {rdf:15.6f}")


if __name__ == "__main__":
    # Example usage (current paths). Adjust region/channel as needed, e.g. region='h-peak', channel='vbf'.
    compare_nominals(
        "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar//stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_Aug14_OLDBR_NewSel_NoSyst/2018/vbf_powheg_dipole_hist.pkl",
        "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar//stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_Aug14_OLDBR_NewSelv2_NoSyst/2018/vbf_powheg_dipole_hist.pkl",
        region=None,
        channel="vbf",
    )
    compare_nominals(
        "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar//stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_Aug14_OLDBR_NewSel_NoSyst/2018/vbf_powheg_dipole_hist.pkl",
        "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar//stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_Aug14_OLDBR_NewSelv2_NoSyst/2018/vbf_powheg_dipole_hist.pkl",
        region="h-peak",
        channel="vbf",
    )

    compare_nominals(
        "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar//stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_Aug14_OLDBR_NewSel_NoSyst/2018/vbf_powheg_dipole_hist.pkl",
        "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar//stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_Aug14_OLDBR_NewSelv2_NoSyst/2018/vbf_powheg_dipole_hist.pkl",
        region="h-sidebands",
        channel="vbf",
    )
    # compare_nominals(
    #     "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar//stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_WithJES_13August_NoSyst/2018/vbf_powheg_dipole_hist.pkl",
    #     "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_FastRead_NoSyst/2018/vbf_powheg_dipole_hist.pkl",
    #     region="h-peak",
    #     channel="vbf",
    # )

    # compare_nominals(
    #     "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_FastRead/2018/vbf_powheg_dipole_hist.pkl",
    #     "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_FastRead_NoSyst/2018/vbf_powheg_dipole_hist.pkl",
    #     region="h-sidebands",
    #     channel="vbf",
    # )
