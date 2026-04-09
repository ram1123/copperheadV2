import numpy as np

def threshold_and_direction(hs_scores: "np.ndarray", pu_scores: "np.ndarray", hs_eff_target: float) -> tuple[float,str,float] | tuple[None,None,None]:
    """Pick keep_low vs keep_high threshold maximizing PU rejection at HS-eff target.""" 
    thr_hi = np.quantile(hs_scores, 1.0 - hs_eff_target)  # keep_high
    rej_hi = float((pu_scores < thr_hi).mean())
    thr_lo = np.quantile(hs_scores, hs_eff_target)        # keep_low
    rej_lo = float((pu_scores > thr_lo).mean())
    return (float(thr_lo),"keep_low",rej_lo) if rej_lo>rej_hi else (float(thr_hi),"keep_high",rej_hi)


def derive_thresholds(df: "pd.DataFrame", score: "np.ndarray", regions: list[str], pt_min: float, pt_max: float, hs_eff_target: float) -> dict:
    """Compute per-region threshold metadata for selected model score."""
    pass
