
FEATURES = {
    # -----------------------
    # Dimuon system
    # -----------------------
    "dimuon_mass": {
        "column": "dimuon_mass",
        "range": (50, 200),
        "bins": 60,
        "title": r"$m_{\mu\mu}$ [GeV]",
    },
    "dimuon_ebe_mass_res": {
        "column": "dimuon_ebe_mass_res",
        "range": (0, 15),
        "bins": 50,
        "title": r"$\sigma_{m_{\mu\mu}}$ [GeV]",
    },
    "dimuon_ebe_mass_res_rel": {
        "column": "dimuon_ebe_mass_res_rel",
        "range": (0, 0.1),
        "bins": 50,
        "title": r"$\sigma_{m_{\mu\mu}} / m_{\mu\mu}$",
    },
    "dimuon_pt": {
        "column": "dimuon_pt",
        "range": (0, 500),
        "bins": 60,
        "title": r"$p_T^{\mu\mu}$ [GeV]",
    },
    "dimuon_pt_log": {
        "column": "dimuon_pt_log",
        "range": (0, 500),
        "bins": 60,
        "title": r"$\log(p_T^{\mu\mu})$",
    },
    "dimuon_rapidity": {
        "column": "dimuon_rapidity",
        "range": (-2.5, 2.5),
        "bins": 50,
        "title": r"$y_{\mu\mu}$",
    },
    "dimuon_cos_theta_cs": {
        "column": "dimuon_cos_theta_cs",
        "range": (-1, 1),
        "bins": 50,
        "title": r"$\cos\theta_{\mathrm{CS}}$",
    },
    "dimuon_phi_cs": {
        "column": "dimuon_phi_cs",
        "range": (-3.2, 3.2),
        "bins": 50,
        "title": r"$\phi_{\mathrm{CS}}$",
    },

    # -----------------------
    # Dijet system
    # -----------------------
    "jj_mass_nominal": {
        "column": "jj_mass_nominal",
        "range": (0, 3000),
        "bins": 60,
        "title": r"$m_{jj}$ [GeV]",
    },
    "jj_mass_log_nominal": {
        "column": "jj_mass_log_nominal",
        "range": (0, 3000),
        "bins": 60,
        "title": r"$\log(m_{jj})$",
    },
    "jj_dEta_nominal": {
        "column": "jj_dEta_nominal",
        "range": (0, 8),
        "bins": 50,
        "title": r"$|\Delta\eta_{jj}|$",
    },

    # -----------------------
    # Jets
    # -----------------------
    "jet1_pt_nominal": {
        "column": "jet1_pt_nominal",
        "range": (0, 300),
        "bins": 60,
        "title": r"$p_T^{j1}$ [GeV]",
    },
    "jet1_eta_nominal": {
        "column": "jet1_eta_nominal",
        "range": (-5, 5),
        "bins": 50,
        "title": r"$\eta_{j1}$",
    },
    "jet1_phi_nominal": {
        "column": "jet1_phi_nominal",
        "range": (-3.2, 3.2),
        "bins": 50,
        "title": r"$\phi_{j1}$",
    },
    "jet1_qgl_nominal": {
        "column": "jet1_qgl_nominal",
        "range": (0, 1),
        "bins": 50,
        "title": r"QGL$_{j1}$",
    },
    "jet2_pt_nominal": {
        "column": "jet2_pt_nominal",
        "range": (0, 300),
        "bins": 60,
        "title": r"$p_T^{j2}$ [GeV]",
    },
    "jet2_eta_nominal": {
        "column": "jet2_eta_nominal",
        "range": (-5, 5),
        "bins": 50,
        "title": r"$\eta_{j2}$",
    },
    "jet2_phi_nominal": {
        "column": "jet2_phi_nominal",
        "range": (-3.2, 3.2),
        "bins": 50,
        "title": r"$\phi_{j2}$",
    },
    "jet2_qgl_nominal": {
        "column": "jet2_qgl_nominal",
        "range": (0, 1),
        "bins": 50,
        "title": r"QGL$_{j2}$",
    },
    # -----------------------
    # Event topology
    # -----------------------
    "rpt_nominal": {
        "column": "rpt_nominal",
        "range": (0, 1),
        "bins": 50,
        "title": r"$R_{p_T}$",
    },
    "ll_zstar_log_nominal": {
        "column": "ll_zstar_log_nominal",
        "range": (-10, 5),
        "bins": 60,
        "title": r"$\log(z^*)$",
    },
    "mmj_min_dEta_nominal": {
        "column": "mmj_min_dEta_nominal",
        "range": (0, 6),
        "bins": 50,
        "title": r"$\min|\Delta\eta(\mu\mu, j)|$",
    },
    "nsoftjets5_nominal": {
        "column": "nsoftjets5_nominal",
        "range": (0, 10),
        "bins": 11,
        "title": r"$N_{\mathrm{soft\ jets}}$",
    },
    "htsoft2_nominal": {
        "column": "htsoft2_nominal",
        "range": (0, 200),
        "bins": 50,
        "title": r"$H_T^{\mathrm{soft}}$ [GeV]",
    },
    "pt_centrality_nominal": {
        "column": "pt_centrality_nominal",
        "range": (-20, 20),
        "bins": 50,
        "title": r"$p_T$ centrality",
    },

    # -----------------------
    # Meta
    # -----------------------
    "year": {
        "column": "year",
        "range": (2015.0, 2026.0),
        "bins": 12,
        "title": "Year",
    },
}
