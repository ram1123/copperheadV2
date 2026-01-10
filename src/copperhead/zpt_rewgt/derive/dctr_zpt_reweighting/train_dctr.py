# Env to load before run:
# conda activate /depot/cms/conda_envs/shar1172/pfn_env 
# python train_dctr.py
#
import numpy as np
import matplotlib.pyplot as plt
from energyflow.archs import PFN
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import dask_awkward as dak
import dask
import awkward as ak
from tensorflow.keras.utils import to_categorical
import os
from pathlib import Path
import argparse

def filterRegion(events, region="h-peak"):
    dimuon_mass = events.dimuon_mass
    if region =="h-peak":
        region = (dimuon_mass > 115.03) & (dimuon_mass < 135.03)
    elif region =="h-sidebands":
        region = ((dimuon_mass > 110) & (dimuon_mass < 115.03)) | ((dimuon_mass > 135.03) & (dimuon_mass < 150))
    elif region =="signal":
        region = (dimuon_mass >= 110) & (dimuon_mass <= 150.0)
    elif region =="z-peak":
        region = (dimuon_mass >= 70) & (dimuon_mass <= 110.0)

    # mu1_pt = events.mu1_pt
    # mu1ptOfInterest = (mu1_pt > 75) & (mu1_pt < 150.0)
    # events = events[region&mu1ptOfInterest]
    events = events[region]
    return events

parser = argparse.ArgumentParser()
parser.add_argument("--load", action="store_true", help="Load saved model instead of training")
parser.add_argument("--prefix", default="pfn_model", help="add prefix to output model and plots")
args = parser.parse_args()
LOAD_EXISTING_MODEL = args.load

# Config
run_label = "April09_NanoV12"
year = "2018"
features = ["mu1_pt", "mu2_pt", "dimuon_pt", "njets_nominal", "dimuon_mass"]
pfn_features = ["mu1_pt", "mu2_pt", "dimuon_pt", "njets_nominal"]

# Dask paths
base = Path(f"/depot/cms/users/shar1172/hmm/copperheadV1clean/{run_label}/stage1_output/{year}/f1_0")
mc_pattern = str(base / "dy*" / "*" / "*.parquet")
data_pattern = str(base / "data_*" / "*" / "*.parquet")
# mc_pattern = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/{run_label}/stage1_output/{year}/f1_0/dy*/*/*.parquet"
# data_pattern = f"/depot/cms/users/shar1172/hmm/copperheadV1clean/{run_label}/stage1_output/{year}/f1_0/data_*/*/*.parquet"

# Load with dask_awkward
df_mc = dak.from_parquet(mc_pattern)[features]
df_data = dak.from_parquet(data_pattern)[features]

# apply z-peak region filter and nothing else
df_data = filterRegion(df_data, region="z-peak")
df_mc = filterRegion(df_mc, region="z-peak")



# Add labels
df_mc["label"] = dak.full_like(df_mc["mu1_pt"], 0)
df_data["label"] = dak.full_like(df_data["mu1_pt"], 1)


# Combine and convert to NumPy
# Concatenate and convert
# Compute separately
df_mc_awk = df_mc[pfn_features + ["label"]].compute()
df_data_awk = df_data[pfn_features + ["label"]].compute()

# Concatenate after materialization
df_all_awk = ak.concatenate([df_mc_awk, df_data_awk], axis=0)

# df_all_np = ak.to_numpy(df_all_awk)             # convert to NumPy

# Separate features and labels
# X = df_all_np[:, :-1].astype(np.float32)
# y = df_all_np[:, -1].astype(np.int64)
X = np.stack([ak.to_numpy(df_all_awk[f]) for f in ["mu1_pt", "mu2_pt", "dimuon_pt", "njets_nominal"]], axis=1)
y = ak.to_numpy(df_all_awk["label"])
# y = ak.to_numpy(df_all_awk["label"]).astype("f4")
y = to_categorical(y, num_classes=2)

# Reshape for PFN (samples, particles=1, features=4)
X = np.expand_dims(X, axis=1)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define and train PFN model
# # pfn = PFN(input_dim=4, Phi_sizes=(64, 64), F_sizes=(64, 64, 1), activations='relu')
# pfn = PFN(input_dim=4, Phi_sizes=(64, 64), F_sizes=(64, 64, 1))
# # pfn = PFN(input_dim=4, Phi_sizes=(64, 64), F_sizes=(64, 64, 1), 
#           # phi_activation='relu', F_activation='relu')
# history = pfn.fit(X_train, y_train, epochs=10, batch_size=256, validation_data=(X_val, y_val))

# # Save model
# pfn.save("pfn_model.h5")

if LOAD_EXISTING_MODEL and os.path.exists("pfn_model.h5"):
    print("🔁 Loading pre-trained model...")
    pfn = PFN(input_dim=4, Phi_sizes=(64, 64), F_sizes=(64, 64, 1))
    pfn.model.load_weights("pfn_model.h5")
else:
    print("🧠 Training new PFN model...")
    pfn = PFN(input_dim=4, Phi_sizes=(64, 64), F_sizes=(64, 64, 1))
    history = pfn.fit(X_train, y_train, epochs=10, batch_size=256, validation_data=(X_val, y_val))
    pfn.save("pfn_model.h5")

# Predict
# y_pred = pfn.predict(X_val).flatten()
# Predict (keep the (n_samples, 2) shape)
y_pred = pfn.predict(X_val)

# ROC Curve
# fpr, tpr, _ = roc_curve(y_val, y_pred)
# Use only the signal class probability (e.g., class 1)
y_val_scalar = np.argmax(y_val, axis=1)        # convert one-hot back to 0/1
y_pred_score = y_pred[:, 1]                    # class 1 probability

fpr, tpr, _ = roc_curve(y_val_scalar, y_pred_score)
roc_auc = auc(fpr, tpr)

# Plot ROC
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("PFN ROC Curve")
plt.legend()
plt.grid(True)
plt.savefig("pfn_roc_curve.png")

# Plot classifier output
plt.figure()
plt.hist(y_pred_score[y_val_scalar == 0], bins=50, alpha=0.6, label="DY MC")
plt.hist(y_pred_score[y_val_scalar == 1], bins=50, alpha=0.6, label="Data")
plt.xlabel("Classifier Output")
plt.ylabel("Events")
plt.title("PFN Classifier Score")
plt.legend()
plt.grid(True)
plt.savefig("pfn_output_dist.png")
