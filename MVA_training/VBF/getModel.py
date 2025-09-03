import torch
import torch.nn as nn
from collections import OrderedDict
from torchviz import make_dot

# Setup
FOLD = 3
# LABEL = "Run2_nanoAODv12_08June" # With Jet QGL bug
LABEL = "Run2_nanoAODv12_UpdatedQGL_17July"  # With Jet QGL bug fixed
# LABEL = "Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt"  # With Jet QGL bug fixed

TRAINED_MODEL_DIR = (
    f"/depot/cms/users/shar1172/"
    f"copperheadV2_main/dnn/trained_models/"
    # f"{LABEL}/2018_h-peak_vbf_AllYear_16July/fold{FOLD}"
    f"{LABEL}/2018_h-peak_vbf_2018_UpdatedQGL_17July_Test/fold{FOLD}"
    # f"{LABEL}/2018_h-peak_vbf_ScanHyperParamV2/fold{FOLD}"
)

training_features = [
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

# --- 1) Load checkpoint state_dict (no model yet) ---
ckpt_path = f"{TRAINED_MODEL_DIR}/best_model_weights.pt"
state = torch.load(ckpt_path, map_location="cpu")
state_dict = state if isinstance(state, dict) and "state_dict" not in state else state["state_dict"]

# --- 2) Infer layer sizes from checkpoint ---
def infer_mlp_sizes(sd, n_feat):
    # Expect keys like fc1.weight [h1, n_feat], fc2.weight [h2, h1], fc3.weight [h3, h2], output.weight [1, h3]
    sizes = []
    # Sort to keep fc1, fc2, fc3 order
    for i in [1, 2, 3]:
        wkey = f"fc{i}.weight"
        if wkey in sd:
            out_dim, in_dim = sd[wkey].shape
            sizes.append(out_dim)
        else:
            break
    # Final layer (optional name "output")
    if "output.weight" in sd:
        out_dim, last_hidden = sd["output.weight"].shape
        assert out_dim == 1, "Expected scalar output."
    return sizes  # e.g. [512, 204, 202]

hidden_sizes = infer_mlp_sizes(state_dict, len(training_features))
print(f"[INFO] Inferred hidden sizes from checkpoint: {hidden_sizes}")

# --- 3) Define a compatible MLP for visualization/loading ---
class CompatibleMLP(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        layers = []
        prev = in_dim
        for i, h in enumerate(hidden, start=1):
            layers.append((f"fc{i}", nn.Linear(prev, h)))
            layers.append((f"bn{i}", nn.BatchNorm1d(h)))
            layers.append((f"relu{i}", nn.ReLU(inplace=True)))
            layers.append((f"drop{i}", nn.Dropout(p=0.1)))
            prev = h
        layers.append(("output", nn.Linear(prev, 1)))
        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.model(x)

viz_model = CompatibleMLP(in_dim=len(training_features), hidden=hidden_sizes)

# Try strict load; if BN buffers names differ, allow non-strict
try:
    viz_model.load_state_dict(state_dict, strict=True)
except Exception as e:
    print(f"[WARN] Strict load failed ({e}); trying strict=False.")
    viz_model.load_state_dict(state_dict, strict=False)

viz_model.eval()

# --- 4) Print a concise architecture & parameter counts to stdout and a text file ---
def summarize_model(m):
    lines = []
    total = 0
    for name, p in m.named_parameters():
        num = p.numel()
        total += num
        lines.append(f"{name:25s} {tuple(p.shape)}  params={num}")
    lines.append(f"\nTotal trainable params: {total}")
    return "\n".join(lines)

arch_text = str(viz_model) + "\n\n" + summarize_model(viz_model)
print("\n[MODEL ARCHITECTURE]\n" + arch_text)

with open(f"{TRAINED_MODEL_DIR}/model_architecture.txt", "w") as f:
    f.write(arch_text)

# --- 5) Save computation-graph as PDF (torchviz) ---
# Use a tiny dummy batch with your feature count
dummy_input = torch.randn(1, len(training_features))
y = viz_model(dummy_input)
graph = make_dot(y, params=dict(viz_model.named_parameters()))
pdf_path = f"{TRAINED_MODEL_DIR}/model_architecture.pdf"
graph.render(pdf_path.replace(".pdf", ""), format="pdf")
print(f"[OK] Saved model architecture PDF to: {pdf_path}")
