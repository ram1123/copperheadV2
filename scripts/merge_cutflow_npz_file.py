import os
import numpy as np
import glob


from pathlib import Path

import ROOT

ROOT.gROOT.SetBatch(False)  
ROOT.gStyle.SetOptStat(0)

# Enable JSROOT for Jupyter inline plots
# ROOT.enableJSVis()


def print_cutflow(npz_path):
    cf = np.load(npz_path, allow_pickle=True)
    labels     = cf["labels"]
    nevonecut  = cf["nevonecut"].astype(np.int64)   # cumulative
    nevcutflow = cf["nevcutflow"].astype(np.int64)  # per-cut

    total = int(nevonecut[0])  # TotalEntries

    print("Cutflow stats:")
    for name, cum, per in zip(labels, nevonecut, nevcutflow):
        eff      = 100.0 * per / total
        cum_eff  = 100.0 * cum / total
        print(
            f"Cut {name:20s} :cumulative pass = {per:<8d} "
            f"pass = {cum:<8d} all = {total:<8d} "
            f"-- cumulative eff = {eff:4.1f} %                    "
            f"-- eff = {cum_eff:4.1f} %"
        )

# example:
npz_path0 = "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_16Dec_NoJVM/stage1_output/2022preEE/f1_0/data_*/0/"
npz_path1 = "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_16Dec_NoJVM/stage1_output/2022preEE/f1_0/data_C/0/"
print_cutflow(f"{npz_path1}/cutflow_data_C_0.npz")
print("-"*51)
npz_path2 = "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_16Dec_NoJVM/stage1_output/2022preEE/f1_0/data_D/0/"
print_cutflow(f"{npz_path2}/cutflow_data_D_0.npz")



def merge_cutflows(pattern, out_path):
    files = sorted(glob.glob(pattern))
    print(files)
    if not files:
        raise RuntimeError(f"No files match pattern {pattern}")

    labels = None
    sum_nevonecut  = None
    sum_nevcutflow = None

    for i, path in enumerate(files):
        print(f"==> {i},  {path}")
        cf = np.load(path, allow_pickle=True)

        this_labels     = cf["labels"]
        this_nevonecut  = cf["nevonecut"].astype(np.int64)
        this_nevcutflow = cf["nevcutflow"].astype(np.int64)

        if labels is None:
            labels         = this_labels
            sum_nevonecut  = this_nevonecut.copy()
            sum_nevcutflow = this_nevcutflow.copy()
        else:
            # sanity check: same cut ordering
            if not np.all(this_labels == labels):
                raise RuntimeError(f"Label mismatch in file {path}")
            sum_nevonecut  += this_nevonecut
            sum_nevcutflow += this_nevcutflow

    # save merged file
    np.savez(out_path,
             labels=labels,
             nevonecut=sum_nevonecut,
             nevcutflow=sum_nevcutflow)

    print(f"Merged {len(files)} files → {out_path}")

# example: merge all chunks for data_C in 2017
npz_path = "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_16Dec_NoJVM/stage1_output/2022preEE/f1_0"

os.system(f"rm -f {npz_path}/cutflow_data_merged.npz")
os.system(f"ls {npz_path}/cutflow_data_merged.npz")

merge_cutflows(
    "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_16Dec_NoJVM/stage1_output/2022preEE/f1_0/data_*/0/*.npz",
    f"{npz_path}/cutflow_data_merged.npz"
)

# then print merged result:
print_cutflow(f"{npz_path}/cutflow_data_merged.npz")