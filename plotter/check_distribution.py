import dask_awkward as dak

# inPath = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2018/compacted/vbf_powheg_dipole/0/*.parquet"
inPath = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2018/compacted/vbf_powheg_dipole/0/*.parquet"

# input = dak.from_parquet(inPath)["dimuon_mass"]
# input = dak.from_parquet(inPath, columns=["dimuon_mass"])
# input = dak.from_parquet(inPath, columns=["dimuon_mass"], split_row_groups="adaptive")
input = dak.from_parquet(inPath, columns=["dimuon_mass"], split_row_groups=True)
# input = dak.from_parquet(inPath, split_row_groups="adaptive")["dimuon_mass"]

dimuon_mass = input.compute().to_numpy()
# # print total entries
# print(f"Total entries: {len(dimuon_mass)}")

# # plot the dimuon mass distribution
# import matplotlib.pyplot as plt

# plt.hist(dimuon_mass, bins=100)
# plt.xlabel("Dimuon Mass (GeV)")
# plt.ylabel("Events")
# plt.title("Dimuon Mass Distribution")
# plt.savefig("dimuon_mass_distribution.pdf")
# plt.close()
