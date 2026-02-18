import pickle
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn

import ROOT
from array import array

from dnn_train import Net, customROC_curve_AN, plotSigVsBkg, prepare_features
import mplhep as hep
plt.style.use(hep.style.CMS)

# Setup
FOLD = 3
# LABEL = "Run2_nanoAODv12_08June" # With Jet QGL bug
LABEL = "Run2_nanoAODv12_UpdatedQGL_17July"  # With Jet QGL bug fixed
LABEL = "Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt"  # With Jet QGL bug fixed
# TRAINED_MODEL_DIR = f"/depot/cms/users/shar1172/copperheadV2_main/dnn/trained_models/Run2_nanoAODv12_08June_MiNNLO/fold{FOLD}"
# DATA_PATH = f"/depot/cms/users/shar1172/copperheadV2_main/dnn/trained_models/{LABEL}"
# TRAINED_MODEL_DIR = f"/depot/cms/users/shar1172/copperheadV2_main/MVA_training/VBF/dnn/trained_models/Run2_nanoAODv12_08June_signal_vbf/fold{FOLD}"
# DATA_PATH = f"/depot/cms/users/shar1172/copperheadV2_main/MVA_training/VBF/dnn/trained_models/Run2_nanoAODv12_08June_signal_vbf"
# TRAINED_MODEL_DIR = f"/depot/cms/users/shar1172/copperheadV2_main/dnn/trained_models/Run2_nanoAODv12_08June/2018_h-peak_vbf/fold{FOLD}"
# DATA_PATH = f"/depot/cms/users/shar1172/copperheadV2_main/dnn/trained_models/Run2_nanoAODv12_08June/2018_h-peak_vbf"
# TRAINED_MODEL_DIR = f"/depot/cms/users/shar1172/copperheadV2_main/dnn/trained_models/Run2_nanoAODv12_08June/2018_h-peak_vbf/fold{FOLD}"
# DATA_PATH = f"/depot/cms/users/shar1172/copperheadV2_main/dnn/trained_models/Run2_nanoAODv12_08June/2018_h-peak_vbf"
# TRAINED_MODEL_DIR = f"/depot/cms/users/shar1172/copperheadV2_main/dnn/trained_models/Run2_nanoAODv12_08June/2018_signal_vbf_15July2025/fold{FOLD}"
# DATA_PATH = f"/depot/cms/users/shar1172/copperheadV2_main/dnn/trained_models/Run2_nanoAODv12_08June/2018_signal_vbf_15July2025"

TRAINED_MODEL_DIR = (
    f"/depot/cms/users/shar1172/"
    f"copperheadV2_main/dnn/trained_models/"
    # f"{LABEL}/2018_h-peak_vbf_AllYear_16July/fold{FOLD}"
    # f"{LABEL}/2018_h-peak_vbf_2018_UpdatedQGL_17July_Test/fold{FOLD}"
    # f"{LABEL}/2018_h-peak_vbf_ScanHyperParamV2/fold{FOLD}"
    f"{LABEL}/run2_h-peak_vbf_ScanHyperParamV1/fold{FOLD}"
)
DATA_PATH = (
    f"/depot/cms/users/shar1172/"
    # f"copperheadV2_main/dnn/trained_models/{LABEL}/2018_h-peak_vbf_AllYear_16July"
    # f"copperheadV2_main/dnn/trained_models/{LABEL}/2018_h-peak_vbf_2018_UpdatedQGL_17July_Test"
    # f"copperheadV2_main/dnn/trained_models/{LABEL}/2018_h-peak_vbf_ScanHyperParamV2"
    f"copperheadV2_main/dnn/trained_models/{LABEL}/run2_h-peak_vbf_ScanHyperParamV1"
)
FEATURES_PKL = f"{DATA_PATH}/training_features.pkl"

training_features = [
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

# 1) load feature names
with open(FEATURES_PKL, "rb") as f:
    training_features_test = pickle.load(f)
n_inputs = len(training_features)

print(f"Input features: {training_features}")
print(f"Total number of input features: {n_inputs}")

print(f"From PKL: Input features: {training_features_test}")
print(f"From PKL: Total number of input features: {len(training_features_test)}")

# ----- Load checkpoint on CPU and rebuild the trained arch -----
# sd = torch.load(f"{TRAINED_MODEL_DIR}/best_model_weights.pt", map_location="cpu")
sd = torch.load(f"{TRAINED_MODEL_DIR}/final_model_weights.pt", map_location="cpu")
# Infer architecture from weights
in_dim  = sd["fc1.weight"].shape[1]
h1      = sd["fc1.weight"].shape[0]
h2      = sd["fc2.weight"].shape[0]
h3      = sd["fc3.weight"].shape[0]
hidden  = (h1, h2, h3)
print(f"Loaded model with input dim: {in_dim}, hidden: {hidden}")
assert in_dim == n_inputs, f"Input dim mismatch: {in_dim} vs {n_inputs}"


model = Net(
    n_feat=in_dim,
    hidden=hidden,
    dropout=(0.0, 0.0, 0.0),
    activation="selu",
    use_batchnorm=True,
)
model.load_state_dict(sd, strict=True)
model.eval()  # "cpu" or "cuda"

# 3) torchsummary: force CPU
from torchsummary import summary

summary(model, input_size=(len(training_features),), device="cpu")

# Load data
df_train = pd.read_parquet(f"{DATA_PATH}/data_df_train_{FOLD}.parquet")
df_valid = pd.read_parquet(f"{DATA_PATH}/data_df_validation_{FOLD}.parquet")
df_eval = pd.read_parquet(f"{DATA_PATH}/data_df_evaluation_{FOLD}.parquet")

# print feature names in the df_eval
# print("Features in df_eval:", df_eval.columns.tolist())
# Ensure the training features are in the dataframes
for feature in training_features:
    if feature not in df_eval.columns:
        raise ValueError(f"Feature '{feature}' not found in df_eval")

feats_tr = prepare_features(df_train, training_features_test)
feats_va = prepare_features(df_valid, training_features_test)
feats_ev = prepare_features(df_eval,  training_features_test)

# Sanity: input dim must match inferred in_dim
assert len(feats_tr) == in_dim, f"Feature count mismatch: {len(feats_tr)} vs {in_dim}"

# ----- Recompute predictions per split (CPU) -----
with torch.no_grad():
    p_train = model(torch.tensor(df_train[feats_tr].values, dtype=torch.float32)).squeeze(1).numpy()
    p_valid = model(torch.tensor(df_valid[feats_va].values, dtype=torch.float32)).squeeze(1).numpy()
    p_eval  = model(torch.tensor(df_eval [feats_ev].values, dtype=torch.float32)).squeeze(1).numpy()

# ----- Build score_dict like in training -----
score_dict = {
    "train": {
        "prediction": p_train,
        "label":      df_train.label.values,
        "weight":     df_train.wgt_nominal.values,
    },
    "valid+eval": {
        "prediction": np.concatenate([p_valid, p_eval], axis=0),
        "label":      np.concatenate([df_valid.label.values, df_eval.label.values], axis=0),
        "weight":     np.concatenate([df_valid.wgt_nominal.values, df_eval.wgt_nominal.values], axis=0),
    },
}

# ----- Plot Sig vs Bkg (matplotlib) -----
bins = np.linspace(0, 1, 30)
plotSigVsBkg(
    score_dict,
    bins=bins,
    plt_save_path=f"{TRAINED_MODEL_DIR}/sig_vs_bkg_best_sameAsTrained_UpdatedScale_New.pdf",
    transformPrediction=False,
    normalize=True,
    log_scale=False,
)

# ----- Rebuild ROC (ROOT) from valid+eval (matches your AN-style helpers) -----
pred_ve = score_dict["valid+eval"]["prediction"]
label_ve = score_dict["valid+eval"]["label"]
wgt_ve   = score_dict["valid+eval"]["weight"]

eff_bkg, eff_sig, _ = customROC_curve_AN(label_ve, pred_ve, wgt_ve, ucsd_mode=True)
eff_bkg_plot = np.clip(eff_bkg, 1e-6, 1)
# sort to integrate AUC
idx = np.argsort(eff_sig)
auc = np.trapz(1 - eff_bkg_plot[idx], x=eff_sig[idx])


# Optional: Debug score distribution using ROOT
ROOT.gStyle.SetOptStat(0)
c1 = ROOT.TCanvas("c1", "", 800, 600)
h_sig = ROOT.TH1F("h_sig", "Model Score;DNN output;Normalized Events", 60, 0, 1)
h_bkg = ROOT.TH1F("h_bkg", "Model Score;DNN output;Normalized Events", 60, 0, 1)
for val, weight in zip(pred_ve[label_ve == 1], wgt_ve[label_ve == 1]):
    h_sig.Fill(val, weight)
for val, weight in zip(pred_ve[label_ve == 0], wgt_ve[label_ve == 0]):
    h_bkg.Fill(val, weight)
h_sig.Scale(1.0 / h_sig.Integral())
h_bkg.Scale(1.0 / h_bkg.Integral())
h_sig.SetLineColor(ROOT.kRed)
h_bkg.SetLineColor(ROOT.kBlue)
h_sig.SetMaximum(1.2 * max(h_sig.GetMaximum(), h_bkg.GetMaximum()))
h_sig.Draw("HIST")
h_bkg.Draw("HIST SAME")
leg = ROOT.TLegend(0.4, 0.75, 0.6, 0.9)
leg.AddEntry(h_sig, "Signal", "l")
leg.AddEntry(h_bkg, "Background", "l")
leg.Draw()
c1.SaveAs(f"{TRAINED_MODEL_DIR}/score_distribution_best.pdf")
c1.SetLogy()
c1.SaveAs(f"{TRAINED_MODEL_DIR}/score_distribution_best_log.pdf")


# Plot ROC using ROOT
roc = ROOT.TGraph(len(eff_sig), array('f', eff_sig), array('f', eff_bkg_plot))
roc.SetTitle(f"ROC Curve;Signal Efficiency;Background Efficiency")
roc.SetLineColor(ROOT.kBlack)
roc.SetLineWidth(2)
# range 0 to 1 for both axes
c2 = ROOT.TCanvas("c2", "", 800, 600)
roc.Draw("AL")
roc.GetXaxis().SetLimits(0, 1.0)
roc.GetYaxis().SetRangeUser(0.001, 1.0)
# add text for AUC
auc_text = ROOT.TLatex(0.6, 0.2, f"AUC = {auc:.3f}")
auc_text.SetNDC()
auc_text.SetTextSize(0.04)
auc_text.Draw()
c2.SetGrid()
c2.SaveAs(f"{TRAINED_MODEL_DIR}/reconstructed_ROC_best_UCSD.pdf")
c2.SetLogy()
roc.Draw("AL")
roc.GetXaxis().SetLimits(0, 1.1)
roc.GetYaxis().SetRangeUser(0.001, 1.)
auc_text.Draw()
c2.SaveAs(f"{TRAINED_MODEL_DIR}/reconstructed_ROC_best_UCSD_log.pdf")
