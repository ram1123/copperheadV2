import numpy as np

def region_mask_eta(eta: np.ndarray, region: str) -> np.ndarray:
    if region == "HEpos": return (eta >= 2.5) & (eta < 3.0)
    if region == "HEneg": return (eta <= -2.5) & (eta > -3.0)
    if region == "HFpos": return eta >= 3.0
    if region == "HFneg": return eta <= -3.0
    if region == "HE":    return (np.abs(eta) >= 2.5) & (np.abs(eta) < 3.0)
    if region == "HF":    return np.abs(eta) >= 3.0
    raise ValueError(region)


def available_regions(split_signed=True):
    if split_signed:
        return ["HEpos", "HEneg", "HFpos", "HFneg"]
    return ["HE", "HF"]
