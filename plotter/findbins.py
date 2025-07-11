"""
This script helps in generating non-uniform bin edges for the dnn_vbf_score
such that the distribution is approximately straight.
"""
import dask_awkward as dak
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

from plotter.validation_plotter_unified import applyRegionCatCuts

# Path to your Parquet file
parquet_file_path = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June//stage1_output/2018/compacted_OLD_WithDNNScore_MassSetTo125/vbf_powheg_dipole/0/*.parquet"

# 1. Read the Parquet file using dask_awkward
events = dak.from_parquet(parquet_file_path)

# 2. Apply the region and category cuts
events = applyRegionCatCuts(
    events,
    category="vbf",
    region_name="h-peak",
    njets="inclusive",
    process="vbf_powheg_dipole",
    do_vbf_filter_study=False
)

# 3. Compute the dnn_vbf_score column as a NumPy array
dnn_vbf_score = events["dnn_vbf_score"].compute().to_numpy()
dnn_vbf_score = np.arctanh((dnn_vbf_score+1)/2.0)
wgt_nominal = events["wgt_nominal"].compute().to_numpy()

sort_idx = np.argsort(dnn_vbf_score)
score_sorted = dnn_vbf_score[sort_idx]
weights_sorted = wgt_nominal[sort_idx]

# reverse the order to get descending scores
score_sorted = score_sorted[::-1]
weights_sorted = weights_sorted[::-1]

edges = [score_sorted[0]]
cumsum = 0
target = 1.0

for score, we in zip(score_sorted, weights_sorted):
    cumsum += we
    if cumsum >= target:
        edges.append(score)
        target += 1.0

if edges[-1] < score_sorted[-1]:
    edges.append(score_sorted[-1])

bin_edges = np.array(edges)
print(bin_edges)
