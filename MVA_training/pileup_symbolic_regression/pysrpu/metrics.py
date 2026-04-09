import numpy as np

def compute_eff_rej(scores, y, threshold, direction):
    hs = scores[y > 0.5]
    pu = scores[y <= 0.5]

    if direction == "keep_high":
        hs_eff = (hs >= threshold).mean()
        pu_rej = (pu < threshold).mean()
    else:
        hs_eff = (hs <= threshold).mean()
        pu_rej = (pu > threshold).mean()

    return float(hs_eff), float(pu_rej)


def compute_wp_vs_pt(df, pass_mask, pt_bins):
    centers = []
    hs_eff = []
    pu_rej = []

    for lo, hi in zip(pt_bins[:-1], pt_bins[1:]):
        sel = (df["pt"] >= lo) & (df["pt"] < hi)
        if sel.sum() == 0:
            hs_eff.append(np.nan)
            pu_rej.append(np.nan)
            continue

        hs = sel & (df["y_hs"] > 0.5)
        pu = sel & (df["y_hs"] <= 0.5)

        eff = pass_mask[hs].mean() if hs.sum() else np.nan
        rej = 1.0 - pass_mask[pu].mean() if pu.sum() else np.nan

        hs_eff.append(eff)
        pu_rej.append(rej)

    return np.array(hs_eff), np.array(pu_rej)


def pt_sculpt_metrics(df: "pd.DataFrame", pass_mask: "np.ndarray") -> dict:
    """Compute quick pT-correlation/slope diagnostics for sculpting detection."""
    pass