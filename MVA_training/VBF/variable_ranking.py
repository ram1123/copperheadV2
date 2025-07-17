import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dnn_train import Net  # your model definition

# —— CONFIG —————————————————————————————————————————————————————————————————
FOLD = 3
LABEL = "Run2_nanoAODv12_08June"
TRAINED_MODEL_DIR = (
    f"/depot/cms/users/shar1172/"
    f"copperheadV2_main/dnn/trained_models/"
    f"Run2_nanoAODv12_08June_MiNNLO/fold{FOLD}"
)
DATA_PATH = (
    f"/depot/cms/users/shar1172/"
    f"copperheadV2_main/dnn/trained_models/{LABEL}"
)
CHECKPOINT = f"{TRAINED_MODEL_DIR}/best_model_weights.pt"
FEATURES_PKL = f"{DATA_PATH}/training_features.pkl"
# ————————————————————————————————————————————————————————————————————————

# 1) load feature names
with open(FEATURES_PKL, "rb") as f:
    feature_names = pickle.load(f)
n_inputs = len(feature_names)

# 2) build your model
model = Net(n_inputs)
model.eval()

# 3) load checkpoint (yours is actually the state_dict itself)
ckpt = torch.load(CHECKPOINT, map_location="cpu")
print(">>> checkpoint keys:", ckpt.keys())
# since ckpt.keys() are the layer parameter names, we load it directly:
model.load_state_dict(ckpt)

# 4) grab the first‐layer weight matrix
#    assume your Net has an attribute `fc1` for the very first Linear layer
W = model.fc1.weight.detach().abs().cpu().numpy()  # shape (n_hidden, n_inputs)

# 5) compute a simple importance per input: sum of abs‐weights over the hidden units
importances = W.sum(axis=0)  # shape (n_inputs,)

# 6) make a DataFrame and sort
df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})
df = df.sort_values("importance", ascending=False).reset_index(drop=True)

# 7) print out the top 20
print(df.head(20))

# 8) quick bar‐plot of the ranking
plt.figure(figsize=(8,6))
plt.barh(df["feature"].iloc[:20][::-1], df["importance"].iloc[:20][::-1])
plt.xlabel("sum |w| from input → first hidden layer")
plt.title("Feature ranking")
plt.tight_layout()
plt.savefig("variable_ranking.pdf")
plt.show()
