import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load feature names (must match the order used in training)
feature_names = [
    "dimuon_mass",
    "dimuon_ebe_mass_res",
    "dimuon_ebe_mass_res_rel",
    "jj_mass_nominal",
    "jj_mass_log_nominal",
    "rpt_nominal",
    "ll_zstar_log_nominal",
    "jj_dEta_nominal",
    "nsoftjets5_nominal",
    "mmj_min_dEta_nominal",
    "dimuon_pt",
    "dimuon_pt_log",
    "dimuon_rapidity",
    "jet1_pt_nominal",
    "jet1_eta_nominal",
    "jet1_phi_nominal",
    "jet2_pt_nominal",
    "jet2_eta_nominal",
    "jet2_phi_nominal",
    "jet1_qgl_nominal",
    "jet2_qgl_nominal",
    "dimuon_cos_theta_cs",
    "dimuon_phi_cs",
    "htsoft2_nominal",
    "pt_centrality_nominal",
    "year",
]

# Load SHAP values
shap_values = np.load("shap_values.npy")  # shape: (n_samples, n_features)

# Compute mean absolute SHAP value for each feature
shap_values_df = pd.DataFrame(shap_values, columns=feature_names)
shap_values_mean = shap_values_df.abs().mean().sort_values(ascending=False)

# Plot top 10 features
plt.figure(figsize=(10, 6))
plt.barh(shap_values_mean.index[:10][::-1], shap_values_mean.values[:10][::-1])
plt.xlabel("Mean |SHAP value|")
plt.title("Top 10 SHAP Feature Importances")
plt.tight_layout()
plt.savefig("shap_feature_importance_top10.png")
plt.savefig("shap_feature_importance_top10.pdf")

# Plot top 13 features
plt.figure(figsize=(10, 6))
plt.barh(shap_values_mean.index[:13][::-1], shap_values_mean.values[:13][::-1])
plt.xlabel("Mean |SHAP value|")
plt.title("Top 13 SHAP Feature Importances")
plt.tight_layout()
plt.savefig("shap_feature_importance_top13.png")
plt.savefig("shap_feature_importance_top13.pdf")

# Plot next 13 feature importance
plt.figure(figsize=(10, 6))
plt.barh(shap_values_mean.index[13:26][::-1], shap_values_mean.values[13:26][::-1])
plt.xlabel("Mean |SHAP value|")
plt.title("Next 13 SHAP Feature Importances")
plt.tight_layout()
plt.savefig("shap_feature_importance_next13.png")
plt.savefig("shap_feature_importance_next13.pdf")
