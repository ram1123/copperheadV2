import pickle
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import ROOT
from array import array

from dnn_train import Net, customROC_curve_AN
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
    f"{LABEL}/2018_h-peak_vbf_ScanHyperParamV2/fold{FOLD}"
)
DATA_PATH = (
    f"/depot/cms/users/shar1172/"
    # f"copperheadV2_main/dnn/trained_models/{LABEL}/2018_h-peak_vbf_AllYear_16July"
    # f"copperheadV2_main/dnn/trained_models/{LABEL}/2018_h-peak_vbf_2018_UpdatedQGL_17July_Test"
    f"copperheadV2_main/dnn/trained_models/{LABEL}/2018_h-peak_vbf_ScanHyperParamV2"
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

# Load model
model = Net(n_feat=len(training_features))
# model.load_state_dict(torch.load(f"{TRAINED_MODEL_DIR}/final_model_weights.pt", map_location=torch.device("cpu"), weights_only=True))
model.load_state_dict(torch.load(f"{TRAINED_MODEL_DIR}/best_model_weights.pt", map_location=torch.device("cuda")))
model.eval()

print("Model Architecture:\n", model)

print("====== torch summary.  ============")
from torchsummary import summary
summary(model, input_size=(len(training_features),))
print("==================================\n\n")

print("====== Print model parameter  ============")
for name, param in model.named_parameters():
    print(f"\t==> {name}: {param.shape}")
print("=========================================\n\n")

# save model architecture as pdf file
## Create a dummy input to trace the model
dummy_input = torch.randn(1, len(training_features))

## Generate graph
from torchviz import make_dot
y = model(dummy_input)
graph = make_dot(y, params=dict(model.named_parameters()))

# Save as PDF
graph.render(f"{TRAINED_MODEL_DIR}/model_architecture", format="pdf")

print(f"Model architecture graph saved to {TRAINED_MODEL_DIR}/model_architecture.pdf")

# Load data
df_valid = pd.read_parquet(f"{DATA_PATH}/data_df_validation_{FOLD}.parquet")
df_eval = pd.read_parquet(f"{DATA_PATH}/data_df_evaluation_{FOLD}.parquet")

# print feature names in the df_eval
# print("Features in df_eval:", df_eval.columns.tolist())
# Ensure the training features are in the dataframes
for feature in training_features:
    if feature not in df_eval.columns:
        raise ValueError(f"Feature '{feature}' not found in df_eval")

X = np.concatenate([df_valid[training_features].values, df_eval[training_features].values])
y = np.concatenate([df_valid.label.values, df_eval.label.values])
w = np.concatenate([df_valid.wgt_nominal.values, df_eval.wgt_nominal.values])

# Predict
with torch.no_grad():
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_pred = model(X_tensor).numpy().flatten()

# Optional: Debug score distribution using ROOT
ROOT.gStyle.SetOptStat(0)
c1 = ROOT.TCanvas("c1", "", 800, 600)
h_sig = ROOT.TH1F("h_sig", "Model Score;Score;Normalized Events", 21, 0, 1)
h_bkg = ROOT.TH1F("h_bkg", "Model Score;Score;Normalized Events", 21, 0, 1)
for val, weight in zip(y_pred[y == 1], w[y == 1]):
    h_sig.Fill(val, weight)
for val, weight in zip(y_pred[y == 0], w[y == 0]):
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
c1.SaveAs(f"{TRAINED_MODEL_DIR}/score_distribution_log_best.pdf")

# Rebuild ROC
eff_bkg, eff_sig, thresholds = customROC_curve_AN(y, y_pred, w)

# Compute AUC using numpy.trapz before plotting
eff_bkg_plot = np.clip(eff_bkg, 1e-6, 1)
# auc = np.trapz(1 - eff_bkg_plot, x=eff_sig)
# Sort by eff_sig
sorted_indices = np.argsort(eff_sig)
eff_sig_sorted = eff_sig[sorted_indices]
eff_bkg_sorted = eff_bkg_plot[sorted_indices]

auc = np.trapz(1 - eff_bkg_sorted, x=eff_sig_sorted)

# Plot ROC using ROOT
roc = ROOT.TGraph(len(eff_sig), array('f', eff_sig), array('f', eff_bkg_plot))
roc.SetTitle(f"ROC Curve;#epsilon_{{sig}};#epsilon_{{bkg}} ")
roc.SetLineColor(ROOT.kBlack)
roc.SetLineWidth(2)
# range 0 to 1 for both axes
c2 = ROOT.TCanvas("c2", "", 800, 600)
roc.Draw("AL")
roc.GetXaxis().SetLimits(0, 1)
roc.GetYaxis().SetRangeUser(0.001, 1)
# add text for AUC
auc_text = ROOT.TLatex(0.6, 0.2, f"AUC = {auc:.3f}")
auc_text.SetNDC()
auc_text.SetTextSize(0.04)
auc_text.Draw()
c2.SetGrid()
c2.SaveAs(f"{TRAINED_MODEL_DIR}/reconstructed_ROC_best.pdf")
c2.SetLogy()
roc.Draw("AL")
roc.GetXaxis().SetLimits(0, 1.1)
roc.GetYaxis().SetRangeUser(0.001, 1.)
auc_text.Draw()
c2.SaveAs(f"{TRAINED_MODEL_DIR}/reconstructed_ROC_log_best.pdf")
