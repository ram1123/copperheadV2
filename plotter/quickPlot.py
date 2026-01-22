import dask_awkward as dak
import awkward as ak
import matplotlib.pyplot as plt

inPath = "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_20Jan_JVMFilterJets/stage1_output/2022postEE/compacted/dyTo2L_M-50_incl/0/*.parquet"

# Select the field so it's not a record/structured array
arr_dask = dak.from_parquet(inPath, columns=["dimuon_pt"])["dimuon_pt"]

# Compute as Awkward, drop None, then convert to NumPy
arr = arr_dask.compute()
arr = arr[~ak.is_none(arr)]
htsoft2 = ak.to_numpy(arr)


print(f"Total entries: {htsoft2.size}")

# plt.hist(htsoft2, bins=100, range=(100, 130), histtype="step", lw=2)
plt.hist(htsoft2, bins=100, range=(0, 500), histtype="step", lw=2, label="OLD")
plt.xlabel("Dimuon pT (GeV)")
plt.ylabel("Events")
plt.title("Dimuon pT Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("dimuon_pt_nominal.pdf")

# log y plot
plt.yscale("log")
plt.savefig("dimuon_pt_nominal_log.pdf")
plt.close()
