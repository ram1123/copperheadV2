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
LABEL = "May28_NanoV12"
TRAINED_MODEL_DIR = f"/depot/cms/users/shar1172/copperheadV2_MergeFW/MVA_training/VBF/dnn/trained_models/{LABEL}/fold{FOLD}"
DATA_PATH = f"/depot/cms/users/shar1172/copperheadV2_MergeFW/MVA_training/VBF/dnn/trained_models/{LABEL}"

training_features = [
    'dimuon_mass', 'dimuon_pt', 'dimuon_pt_log', 'dimuon_rapidity',
    'dimuon_cos_theta_cs', 'dimuon_phi_cs',
    'jet1_pt_nominal', 'jet1_eta_nominal', 'jet1_phi_nominal', 'jet1_qgl_nominal',
    'jet2_pt_nominal', 'jet2_eta_nominal', 'jet2_phi_nominal', 'jet2_qgl_nominal',
    'jj_mass_nominal', 'jj_mass_log_nominal', 'jj_dEta_nominal', 'rpt_nominal',
    'll_zstar_log_nominal', 'mmj_min_dEta_nominal', 'nsoftjets5_nominal', 'htsoft2_nominal'
]

# Load model
model = Net(n_feat=len(training_features))
model.load_state_dict(torch.load(f"{TRAINED_MODEL_DIR}/best_model_weights.pt", map_location=torch.device("cpu"), weights_only=True))
model.eval()

# Load data
df_valid = pd.read_parquet(f"{DATA_PATH}/data_df_validation_{FOLD}")
df_eval = pd.read_parquet(f"{DATA_PATH}/data_df_evaluation_{FOLD}")

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
h_sig = ROOT.TH1F("h_sig", "Model Score;Score;Normalized Events", 50, 0, 1)
h_bkg = ROOT.TH1F("h_bkg", "Model Score;Score;Normalized Events", 50, 0, 1)
for val, weight in zip(y_pred[y == 1], w[y == 1]):
    h_sig.Fill(val, weight)
for val, weight in zip(y_pred[y == 0], w[y == 0]):
    h_bkg.Fill(val, weight)
h_sig.Scale(1.0 / h_sig.Integral())
h_bkg.Scale(1.0 / h_bkg.Integral())
h_sig.SetLineColor(ROOT.kRed)
h_bkg.SetLineColor(ROOT.kBlue)
h_sig.Draw("HIST")
h_bkg.Draw("HIST SAME")
leg = ROOT.TLegend(0.4, 0.75, 0.6, 0.9)
leg.AddEntry(h_sig, "Signal", "l")
leg.AddEntry(h_bkg, "Background", "l")
leg.Draw()
c1.SaveAs(f"{TRAINED_MODEL_DIR}/score_distribution.pdf")
c1.SetLogy()
c1.SaveAs(f"{TRAINED_MODEL_DIR}/score_distribution_log.pdf")

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
c2.SaveAs(f"{TRAINED_MODEL_DIR}/reconstructed_ROC.pdf")
c2.SetLogy()
roc.Draw("AL")
roc.GetXaxis().SetLimits(0, 1.1)
roc.GetYaxis().SetRangeUser(0.001, 1.)
auc_text.Draw()
c2.SaveAs(f"{TRAINED_MODEL_DIR}/reconstructed_ROC_log.pdf")
