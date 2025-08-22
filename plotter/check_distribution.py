import dask_awkward as dak
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

inPath = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2018/compacted/vbf_powheg_dipole/0/*.parquet"
inPath2 = "/depot/cms/users/shar1172/hmm/copperheadV1clean/debug_softjet/stage1_output/2018/compacted/vbf_powheg_dipole/0/*.parquet"

# Select the field so it's not a record/structured array
arr_dask = dak.from_parquet(inPath, columns=["nsoftjets5_nominal"])[
    "nsoftjets5_nominal"
]
arr_dask2 = dak.from_parquet(inPath2, columns=["nsoftjets5_nominal"])[
    "nsoftjets5_nominal"
]

# Compute as Awkward, drop None, then convert to NumPy
arr = arr_dask.compute()
arr = arr[~ak.is_none(arr)]
htsoft2 = ak.to_numpy(arr)

arr2 = arr_dask2.compute()
arr2 = arr2[~ak.is_none(arr2)]
htsoft22 = ak.to_numpy(arr2)

print(f"Total entries: {htsoft2.size}")
print(f"Total entries: {htsoft22.size}")

# plt.hist(htsoft2, bins=100, range=(100, 130), histtype="step", lw=2)
plt.hist(htsoft2, bins=21, range=(0, 20), histtype="step", lw=2, label="OLD")
plt.hist(htsoft22, bins=21, range=(0, 20), histtype="step", lw=2, label="NEW")
plt.xlabel("Dimuon Mass (GeV)")
plt.ylabel("Events")
plt.title("Dimuon Mass Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("htsoft2_nominal_JESVar.pdf")
plt.close()
