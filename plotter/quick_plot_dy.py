import os
import glob
import dask_awkward as dak
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

from rich import print

base_dir = (
    "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/"
    "Run3_nanoAODv12_21Jan_JVMFilterJets/stage1_output"
)

years = [
    "2022preEE",
    "2022postEE",
    "2023",
    "2023BPix"
]

dy_samples = [
    "dy_M-50_aMCatNLO",
    "dyTo2L_M-50_0j",
    "dyTo2L_M-50_1j",
    "dyTo2L_M-50_2j",
    "dyTo2L_M-50_incl",
    "dyTo2Mu_M-105To160",
    "dyTo2Mu_MLL_10To50",
    "dyTo2Mu_MLL_50To120",
    "dyTo2Mu_MLL_120To200",
]

for year in years:
    for dy_sample in dy_samples:

        inPath = f"{base_dir}/{year}/compacted/{dy_sample}/0/*.parquet"

        files = glob.glob(inPath)
        if len(files) == 0:
            print(f"[SKIP] {year} / {dy_sample} (no files)")
            continue

        print(f"[INFO] Processing {year} / {dy_sample}")

        arrays = dak.from_parquet(files, columns=["dimuon_pt", "wgt_nominal"]).compute()

        pt = arrays["dimuon_pt"]
        wgt = arrays["wgt_nominal"]

        mask = (~ak.is_none(pt)) & (~ak.is_none(wgt))
        pt = ak.to_numpy(pt[mask])
        wgt = ak.to_numpy(wgt[mask])

        if pt.size == 0:
            print(f"[SKIP] {year} / {dy_sample} (empty after mask)")
            continue

        print(f"  Entries: {pt.size}, SumW: {wgt.sum():.3f}")

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.hist(
            pt,
            bins=100,
            range=(0, 500),
            weights=wgt,
            histtype="step",
            linewidth=1.5,
        )
        ax.set_xlabel(r"p$_T^{\mu\mu}$ [GeV]", fontsize=14)
        ax.set_ylabel("Weighted events", fontsize=14)
        ax.set_title(f"{dy_sample} ({year})", fontsize=15)
        ax.set_yscale("log")
        ax.grid(True)

        outname = f"dimuon_pt_{dy_sample}_{year}.pdf"
        fig.savefig(outname)
        plt.close(fig)
