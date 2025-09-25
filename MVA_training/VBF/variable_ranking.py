import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dnn_train import Net  # our model definition

# —— CONFIG —————————————————————————————————————————————————————————————————
FOLD = 3
LABEL = "Run2_nanoAODv12_08June"
LABEL = "Run2_nanoAODv12_UpdatedQGL_17July"
TRAINED_MODEL_DIR = (
    f"/depot/cms/users/shar1172/"
    f"copperheadV2_main/dnn/trained_models/"
    # f"{LABEL}/2018_signal_vbf_16July2025/fold{FOLD}"
    f"{LABEL}/2018_h-peak_vbf_2018_UpdatedQGL_17July_Test/fold{FOLD}"
)
DATA_PATH = (
    f"/depot/cms/users/shar1172/"
    # f"copperheadV2_main/dnn/trained_models/{LABEL}/2018_signal_vbf_16July2025"
    f"copperheadV2_main/dnn/trained_models/{LABEL}/2018_h-peak_vbf_2018_UpdatedQGL_17July_Test"
)
# CHECKPOINT = f"{TRAINED_MODEL_DIR}/best_model_weights.pt"
CHECKPOINT = f"{TRAINED_MODEL_DIR}/final_model_weights.pt"
FEATURES_PKL = f"{DATA_PATH}/training_features.pkl"
# ————————————————————————————————————————————————————————————————————————

# 1) load feature names
# with open(FEATURES_PKL, "rb") as f:
#     feature_names = pickle.load(f)
feature_names = [
        'dimuon_mass',
        "dimuon_ebe_mass_res", "dimuon_ebe_mass_res_rel",
         'jj_mass_nominal', 'jj_mass_log_nominal',
         'rpt_nominal',
         'll_zstar_log_nominal',
         'jj_dEta_nominal',
         'nsoftjets5_nominal',
         'mmj_min_dEta_nominal',
         'dimuon_pt', 'dimuon_pt_log', 'dimuon_rapidity',
         'jet1_pt_nominal', 'jet1_eta_nominal', 'jet1_phi_nominal',  'jet2_pt_nominal', 'jet2_eta_nominal', 'jet2_phi_nominal',
         'jet1_qgl_nominal', 'jet2_qgl_nominal',
         'dimuon_cos_theta_cs', 'dimuon_phi_cs',
         'htsoft2_nominal',
         'pt_centrality_nominal',
         'year'
]
n_inputs = len(feature_names)

# 2) build our model
model = Net(n_inputs)
model.eval()

# 3) load checkpoint (yours is actually the state_dict itself)
ckpt = torch.load(CHECKPOINT, map_location="cpu")
print(">>> checkpoint keys:", ckpt.keys())
# since ckpt.keys() are the layer parameter names, we load it directly:
model.load_state_dict(ckpt)

# 4) grab the first‐layer weight matrix
#    assume our Net has an attribute `fc1` for the very first Linear layer
W = model.fc1.weight.detach().abs().cpu().numpy()  # shape (n_hidden, n_inputs)

# 5) compute a simple importance per input: sum of abs‐weights over the hidden units
importances = W.sum(axis=0)  # shape (n_inputs,)

# 6) make a DataFrame and sort
df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})
df = df.sort_values("importance", ascending=False).reset_index(drop=True)

# 7) print out the top 30
print(df.head(30))

# 8) quick bar‐plot of the ranking
plt.figure(figsize=(11,21))
plt.barh(df["feature"].iloc[:30][::-1], df["importance"].iloc[:30][::-1])
# plt.xlabel("sum |w| from input → first hidden layer")
plt.title("Feature ranking")

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)


plt.tight_layout()
plt.savefig("variable_ranking.pdf")
plt.close()

# Use shap for more advanced feature importance analysis
import shap

Valid_parquet = f"{DATA_PATH}/data_df_validation_{FOLD}.parquet"
df_valid = pd.read_parquet(Valid_parquet)
X_valid = df_valid[feature_names].values

# Prepare a background sample (e.g., 100 random validation samples)
background = torch.tensor(X_valid[np.random.choice(X_valid.shape[0], 100, replace=False)], dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
background = background.to(device)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
X_valid_tensor = X_valid_tensor.to(device)

# 9) create a SHAP DeepExplainer
explainer = shap.DeepExplainer(model, background)

# 10) compute SHAP values for the validation set
shap_values = explainer.shap_values(X_valid_tensor, check_additivity=False)

# If DeepExplainer returned a list (one array per output class) pick the one we care about.
# e.g. for a single-output net it'll already be one array, not a list
if isinstance(shap_values, list):
    # say we want the shap for class 1:
    shap_values = shap_values[1]

# Now shap_values.shape is (n_samples, n_features, 1); we need to drop that last axis
shap_values = np.squeeze(shap_values, axis=2)   # result is (n_samples, n_features)

# 11) plot SHAP values
shap.summary_plot(shap_values, X_valid, feature_names=feature_names, max_display=30, show=False)
plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.savefig("shap_feature_importance.pdf")
plt.close()

# 12) now we can build our DataFrame
shap_values_df = pd.DataFrame(shap_values, columns=feature_names)
shap_values_mean = shap_values_df.abs().mean().sort_values(ascending=False)
plt.figure(figsize=(11,21))
plt.barh(shap_values_mean.index[:30][::-1],
         shap_values_mean.values[:30][::-1])
plt.title("SHAP Feature Importance (mean absolute value)")
plt.tight_layout()
plt.savefig("shap_feature_importance_mean.pdf")
plt.close()

