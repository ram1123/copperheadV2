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
    # "dimuon_pt_log",
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
# General variables: PV, MET, muon ID/iso/detector/jet-assoc/sv vars, and muon pair-based vars (pt, eta, iso, ip, detector)
# ----------------------------------------------------------------------
PV_VARS: List[str] = [
    "PV_npvs",
    "PV_npvsGood",
]

MET_VARS: List[str] = [
    "PuppiMET_pt",
    "PuppiMET_phi",
    "PuppiMET_sumEt",
]

muon_ip_vars = [
    "mu1_dxy",
    "mu2_dxy",
    "mu1_dxyErr",
    "mu2_dxyErr",
    "mu1_dxybs",
    "mu2_dxybs",
    "mu1_dz",
    "mu2_dz",
    "mu1_dzErr",
    "mu2_dzErr",
    "mu1_ip3d",
    "mu2_ip3d",
    "mu1_sip3d",
    "mu2_sip3d",
]

muon_id_vars = [
    "mu1_highPurity",
    "mu2_highPurity",
    "mu1_inTimeMuon",
    "mu2_inTimeMuon",
    "mu1_isGlobal",
    "mu2_isGlobal",
    "mu1_isPFcand",
    "mu2_isPFcand",
    "mu1_isStandalone",
    "mu2_isStandalone",
    "mu1_isTracker",
    "mu2_isTracker",
    "mu1_looseId",
    "mu2_looseId",
    "mu1_mediumId",
    "mu2_mediumId",
    "mu1_mediumPromptId",
    "mu2_mediumPromptId",
    "mu1_tightCharge",
    "mu2_tightCharge",
    "mu1_pdgId",
    "mu2_pdgId",
]

muon_iso_vars = [
    "mu1_miniIsoId",
    "mu2_miniIsoId",
    "mu1_miniPFRelIso_all",
    "mu2_miniPFRelIso_all",
    "mu1_miniPFRelIso_chg",
    "mu2_miniPFRelIso_chg",
    "mu1_multiIsoId",
    "mu2_multiIsoId",
    "mu1_pfIsoId",
    "mu2_pfIsoId",
    "mu1_pfRelIso03_all",
    "mu2_pfRelIso03_all",
    "mu1_pfRelIso03_chg",
    "mu2_pfRelIso03_chg",
    "mu1_pfRelIso04_all",
    "mu2_pfRelIso04_all",
    "mu1_puppiIsoId",
    "mu2_puppiIsoId",
    "mu1_tkIsoId",
    "mu2_tkIsoId",
    "mu1_tkRelIso",
    "mu2_tkRelIso",
]

muon_detector_vars = [
    "mu1_nStations",
    "mu2_nStations",
    "mu1_nTrackerLayers",
    "mu2_nTrackerLayers",
    "mu1_segmentComp",
    "mu2_segmentComp",
]

muon_jet_assoc_vars = [
    "mu1_jetIdx",
    "mu2_jetIdx",
    "mu1_jetNDauCharged",
    "mu2_jetNDauCharged",
    "mu1_jetPtRelv2",
    "mu2_jetPtRelv2",
    "mu1_jetRelIso",
    "mu2_jetRelIso",
]

muon_sv_vars = [
    "mu1_svIdx",
    "mu2_svIdx",
]

mu12_pt_comb_vars = [
    "mu12_pt_sum",
    "mu12_pt_diff",
    "mu12_pt_absdiff",
    "mu12_pt_prod",
    "mu12_pt_ratio12",
    "mu12_pt_ratio21",
    "mu12_pt_min",
    "mu12_pt_max",
    "mu12_pt_asym",
]

mu12_eta_comb_vars = [
    "mu12_eta_sum",
    "mu12_eta_diff",
    "mu12_eta_absdiff",
    "mu12_eta_prod",
    "mu12_absEta_sum",
    "mu12_absEta_diff",
    "mu12_absEta_min",
    "mu12_absEta_max",
]

mu12_iso_comb_vars = [
    "mu12_iso04_sum",
    "mu12_iso04_diff",
    "mu12_iso04_absdiff",
    "mu12_iso04_prod",
    "mu12_iso04_min",
    "mu12_iso04_max",
    "mu12_iso04_asym",
]

mu12_ip_comb_vars = [
    "mu12_dxy_sum",
    "mu12_dxy_diff",
    "mu12_dxy_absdiff",
    "mu12_dz_sum",
    "mu12_dz_diff",
    "mu12_dz_absdiff",
    "mu12_sip3d_sum",
    "mu12_sip3d_diff",
    "mu12_sip3d_absdiff",
    "mu12_sip3d_prod",
    "mu12_sip3d_min",
    "mu12_sip3d_max",
]

mu12_detector_comb_vars = [
    "mu12_nStations_min",
    "mu12_nStations_max",
    "mu12_nStations_sum",
    "mu12_nTrackerLayers_min",
    "mu12_nTrackerLayers_max",
    "mu12_nTrackerLayers_sum",
    "mu12_q1q2",
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
    "dimuon_cos_theta_cs",
    "dimuon_phi_cs",
    "dimuon_rapidity",
]

DIMUON_PLOT_VARS_EXTENDED: List[str] = [
    "dimuon_ebe_mass_res",
    "dimuon_ebe_mass_res_rel",
    "uncalibrated_dimuon_ebe_mass_res",
    "dimuon_cos_theta_cs",
    "dimuon_phi_cs",
    "dimuon_phi_eta",
    "dimuon_dEta",
    "dimuon_dPhi",
    "dimuon_dR",
    "acoplanarity",
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

DIJET_VARS: List[str] = [
    "jj_dEta_nominal",
    "jj_mass_nominal",
    "nsoftjets5_nominal",
    "htsoft2_nominal",
    "pt_centrality_nominal",
    "rpt_nominal",
    "nsoftjets2_nominal",
    "htsoft5_nominal",
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
    "vbf_additional": VBF_ADDITIONAL_VARS,
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
