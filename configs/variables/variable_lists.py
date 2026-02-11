"""
Central definition of variable lists used for:
- DNN training
- Validation plots
- Control plots

Current contents:
- VBF training variables
- VBF additional (pair-based) variables (as produced by your jet-loop code)

Usage:
    from variable_lists import VBF_TRAINING_VARS, VBF_ADDITIONAL_VARS, VAR_SETS

    vars_to_plot = VAR_SETS["vbf_training"]
"""

from typing import Dict, List
from rich import print

# ----------------------------------------------------------------------
# ggH: training variables
# ----------------------------------------------------------------------
GGH_TRAINING_VARS: List[str] = [
    "dimuon_cos_theta_cs",
    "dimuon_phi_cs",
    "dimuon_rapidity",
    "dimuon_pt",

    "mu1_eta",
    "mu1_pt_over_mass",
    "mu2_eta",
    "mu2_pt_over_mass",

    "njets_nominal",
    "jet1_pt_nominal",
    "jet1_eta_nominal",
    "jet2_pt_nominal",
    "jet2_eta_nominal",

    "jj_mass_nominal",
    "jj_dEta_nominal",
    "jj_dPhi_nominal",

    "mmj_min_dEta_nominal",
    "mmj_min_dPhi_nominal",

    "zeppenfeld_nominal",
]

# ----------------------------------------------------------------------
# VBF: training variables
# ----------------------------------------------------------------------
VBF_TRAINING_VARS: List[str] = [
    "dimuon_mass",
    "dimuon_pt",
    "dimuon_pt_log",
    "dimuon_rapidity",
    "dimuon_ebe_mass_res",
    "dimuon_ebe_mass_res_rel",
    "dimuon_cos_theta_cs",
    "dimuon_phi_cs",
    "jet1_pt_nominal",
    "jet1_eta_nominal",
    "jet1_phi_nominal",
    "jet2_pt_nominal",
    "jet2_eta_nominal",
    "jet2_phi_nominal",
    "jj_mass_nominal",
    # "jj_mass_log_nominal",
    "jj_dEta_nominal",
    "nsoftjets5_nominal",
    "htsoft2_nominal",
    "rpt_nominal",
    "ll_zstar_log_nominal",
    "mmj_min_dEta_nominal",
    "pt_centrality_nominal",
]

# ----------------------------------------------------------------------
# Muons: basic kinematics (plots)
# ----------------------------------------------------------------------
MUON_PLOT_VARS: List[str] = [
    "mu1_pt",
    "mu2_pt",
    "mu1_eta",
    "mu2_eta",
    "mu1_phi",
    "mu2_phi",
    "mu1_pt_over_mass",
    "mu2_pt_over_mass",
]

DIMUON_PLOT_VARS: List[str] = [
    "dimuon_eta",
    "dimuon_pt",
    "dimuon_phi",
    "dimuon_mass",
]

# ----------------------------------------------------------------------
# Jets: additional variables
# -----------------------------------------------------------------------
JETS_PLOT_VARS: List[str] = [
    "njets_nominal",
    "jet1_pt_nominal",
    "jet1_eta_nominal",
    "jet1_phi_nominal",
    "jet2_pt_nominal",
    "jet2_eta_nominal",
    "jet2_phi_nominal",
]

# ----------------------------------------------------------------------
# VBF: additional variables (pair-based, nominal only)
# Produced by your snippet for tag in: lead, maxmjj, maxdeta, maxmjj_deta25
#
# Note: your producer names them like:
#   vbf_{tag}_jet1_pt_{variation}  (variation = "nominal")
# ----------------------------------------------------------------------
_VBF_PAIR_TAGS: List[str] = [
    "lead",
    "maxmjj",
    "maxdeta",
    "maxmjj_deta25",
]

_VBF_PAIR_BASE_VARS: List[str] = [
    "jet1_pt",
    "jet1_eta",
    "jet1_phi",
    "jet2_pt",
    "jet2_eta",
    "jet2_phi",
    "mjj",
    "deta",
]

VBF_ADDITIONAL_VARS: List[str] = (
    [
        f"vbf_{tag}_{v}_nominal"
        for tag in _VBF_PAIR_TAGS
        for v in _VBF_PAIR_BASE_VARS
    ]
    + [
        "vbf_maxmjj_deta25_hasPair_nominal",
    ]
)


# ----------------------------------------------------------------------
# Convenience dict
# ----------------------------------------------------------------------
VAR_SETS: Dict[str, List[str]] = {
    "vbf_training": VBF_TRAINING_VARS,
    "ggh_training": GGH_TRAINING_VARS,
    "muon_plots": MUON_PLOT_VARS,
    "dimuon_plots": DIMUON_PLOT_VARS,
    # "vbf_additional": VBF_ADDITIONAL_VARS,
}


def unique_preserve_order(seq: List[str]) -> List[str]:
    """Return list of unique items, preserving original order."""
    unique_vars = list(dict.fromkeys(seq))
    return unique_vars

def get_all_vars(test = False) -> List[str]:
    """Get full list of unique vars from all sets."""
    all_vars = []
    for var_list in VAR_SETS.values():
        all_vars.extend(var_list)

    if test:  # just fetch the first 5 vars for testing
        all_vars = all_vars[:5]

    return unique_preserve_order(all_vars)

if __name__ == "__main__":
    full_list_of_vars = []
    for set_name in VAR_SETS:
        vars_ = unique_preserve_order(VAR_SETS[set_name])
        print("# ----------------------------------------------------")
        print(f"Variable set '{set_name}' ({len(vars_)} vars):")
        print(vars_)
        full_list_of_vars.extend(vars_)

    print("=====")
    print(f"Full list of unique vars ({len(unique_preserve_order(full_list_of_vars))} vars):")
    print(unique_preserve_order(full_list_of_vars))
