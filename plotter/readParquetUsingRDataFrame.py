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
# get atanh of the dnn_vbf_score
atanh_dnn_vbf_score = np.arctanh((dnn_vbf_score+1)/2.0)
wgt_nominal = events["wgt_nominal"].compute().to_numpy()

# 4. Plot with matplotlib, including error bars
# bins = np.linspace(0, 1, 52)
# bins = [0.0, 0.04397368, 0.20310317, 0.31207183, 0.35124612, 0.39282253, 0.43408683,
#  0.46635309, 0.51985765, 0.56432748, 0.63402843, 0.67790931, 1.0]
bins = [1.0, 0.65993273, 0.61075753, 0.56419611, 0.50403064, 0.44700417,
 0.41756606, 0.37281281, 0.33280191, 0.27831182, 0.13914499,0.0]
# reverse the order to get descending scores
bins = bins[::-1]
bins = np.array(bins)
hist, bin_edges = np.histogram(dnn_vbf_score, bins=bins, weights=wgt_nominal)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# 5. Calculated weighted Poisson errors
# For weighted histograms, the variance per bin is sum of squared weights
sumw2, _ = np.histogram(dnn_vbf_score, bins=bins, weights=wgt_nominal**2)
errors = np.sqrt(sumw2)

plt.errorbar(bin_centers, hist, yerr=errors, fmt='o', label='dnn_vbf_score', alpha=0.7)
plt.xlabel('dnn_vbf_score')
plt.ylabel('Frequency')
plt.title('Histogram of dnn_vbf_score with error bars')
plt.legend()
plt.grid(True)
plt.savefig('dnn_vbf_score_histogram_DYVBFFilter_all_AfterVBFSelection.pdf')

# log scale for better visibility
plt.yscale('log')
plt.savefig('dnn_vbf_score_histogram_DYVBFFilter_all_log_AfterVBFSelection.pdf')
plt.close()
print("Histogram saved as 'dnn_vbf_score_histogram_DYVBFFilter_all_AfterVBFSelection.pdf'")
print("Histogram saved as 'dnn_vbf_score_histogram_DYVBFFilter_all_log_AfterVBFSelection.pdf'")
